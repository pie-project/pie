//! What a launch rule means when the workgroup is not the driver's to choose.
//!
//! `driver-metal` answers a [`Rule`] with a thread grid AND a threadgroup,
//! because `dispatchThreads` takes both and Metal sizes the group at dispatch.
//! `dispatch_workgroups` takes neither. It takes a count of WORKGROUPS, and the
//! size of one is `@workgroup_size(...)` in the WGSL — a compile-time attribute
//! fixed when the module was written, not when it was launched.
//!
//! So this module cannot be a port of `lowering/grid.rs`. The Metal shapes are
//! still the reference — they are the ones proved against real checkpoints, and
//! the iteration space a kernel must cover does not change with the API — but
//! the answer here is a different kind of number, and getting from one to the
//! other is where the whole class of porting bug in this tree lives.
//!
//! # This is the file that changed least in the Vulkan port, on purpose
//!
//! `dispatch_workgroups(x, y, z)` counts exactly what `vkCmdDispatch(x, y, z)`
//! counts, so every line of arithmetic below is `driver-vulkan`'s, and the
//! reasons attached to each rounding are its reasons unedited. That is not
//! laziness: a rule is a claim about a KERNEL, the three trees are one tree of
//! kernels, and a grid that differed between two shells would mean one of them
//! is wrong rather than that the APIs differ.
//!
//! One thing is added, and it is the API's: [`Ungeometric::PastDeviceLimit`].
//! Vulkan reports `maxComputeWorkGroupCount` in the millions on every real
//! device and a driver may reasonably never think about it. WebGPU's
//! `max_compute_workgroups_per_dimension` has a guaranteed floor of **65535**,
//! which a 4096-wide elementwise launch over 32 rows reaches, so it is a limit
//! a real fire can hit — and `wgpu` answers a dispatch past it by refusing the
//! encode, which is a panic or a validation error at submit rather than a
//! sentence about the launch. [`groups_within`] is where that sentence is.
//!
//! # Undershoot is the failure that does not report itself
//!
//! An overshot grid is caught by a shader's own tail guard. An UNDERSHOT grid
//! is not caught by anything: the lanes that were never launched write nothing,
//! the gap reads back as whatever the buffer was born with — zeros, from a
//! fresh pool — and the queue returns success. Every kernel in this tree that
//! was wrong after the Vulkan port was wrong this way, and none of them were
//! wrong in the arithmetic.
//!
//! That is why [`groups`] rounds up everywhere, and why any test of it must use
//! a shape that does NOT divide evenly: at 512 elements over a 256-wide
//! workgroup, `div_ceil` and plain division are the same expression and the
//! rounding is unproven. `driver-vulkan` learned that with five tests that all
//! ran at exact multiples and caught nothing; the tests below run at 460 and
//! at 13.
//!
//! # The module's own size is an input
//!
//! Each rule below takes the `local` size read from the module it is about to
//! dispatch, rather than assuming one. Two rules need this and the rest are
//! merely honest about it:
//!
//! * `SdpaVector` compiles one module per head dimension — 64, 128, 256, 512 —
//!   and each declares `@workgroup_size(PIE_HEAD_DIM / 2)`. A geometry that
//!   assumed 256 would launch a quarter of the workgroups for a 64-wide head.
//! * `Elementwise` is 256 wide in most of its modules and 16x16 in the strided
//!   gathers, which are laid out per (channel, row).
//!
//! **That `/ 2` is the one place a launch rule's ARITHMETIC differs from
//! `driver-vulkan`'s**, and it is forced rather than chosen. WGSL has no 16-bit
//! storage type, so every bf16 tensor crosses as `array<u32>` with two values
//! to a word — and a lane that owned one channel would read-modify-write a word
//! its neighbour writes at the same instant, which WGSL has no sub-word atomic
//! to make safe. So a decode-attention lane owns the PAIR, the workgroup is
//! half as wide as the head, and [`lanes`] halves with it. Getting it wrong is
//! not an undershoot to be caught by a tail guard: `sdpa_vector.wgsl` reads
//! `num_workgroups.x` as its query-head COUNT, so the Vulkan expression would
//! build a grid twice as wide AND tell every lane the model has twice the heads
//! it has.
//!
//! The GEMM tile is the same kind of fact and was nearly missed in the Vulkan
//! port. Every `Qmm` entrypoint names its tile IN THE ENTRYPOINT —
//! `..._bm_16_bn_64` — so the tile a module was compiled for is a property of
//! the module the driver selected, not something to be inferred from the row
//! count. Inferring it is what the first draft of that file did, and picking
//! the widest tile that divides 64 rows while dispatching a module written at
//! `bm_16` launches a quarter of the workgroups needed. That undershoot writes
//! three quarters of nothing and returns success.
//!
//! Reading these from the module is also the only way the agreement can be
//! CHECKED, and here that check costs nothing at all: [`Module::loaded`] takes
//! a [`crate::reflect::Declared`], which is `naga` reading the WGSL that will
//! be dispatched, on any machine, with no adapter.

pub use kernels::LaunchRule as Rule;

/// The fire-time quantities a launch rule may read.
///
/// The same set `driver-metal`'s `lowering::launch::Dims` carries, deliberately
/// spelled the same way. The three shells disagree about the grid; they must
/// not disagree about which number is `axis` and which is `width`, and a reader
/// comparing the files should find the difference where it really is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dims {
    /// Rows the rectangle covers.
    pub rows: u32,
    /// Elements per row of the operand that sizes the launch — the launch's
    /// last widthed operand, which is its last OUTPUT.
    pub width: u32,
    /// Elements per row of the launch's first widthed operand — its first
    /// INPUT. A statement that reads one packed buffer and writes several
    /// sizes on this instead, since no one output spells the grid.
    pub in_width: u32,
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads.
    pub kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Elements one reduction spans, when that is not the whole row. A
    /// hidden-state norm reduces over its row and this equals `width`; a
    /// QK-norm reduces over one head of a stacked projection and it does not.
    pub axis: u32,
    /// Channels a partial rope rotates.
    pub rotary_dims: u32,
    /// Experts the router scores.
    pub n_experts: u32,
    /// Experts each token routes to.
    pub experts_per_token: u32,
}

/// A workgroup size, as the module declares it.
///
/// Three axes because `y` and `z` are load-bearing here: the GEMM tiles are
/// `(32, 2, 2)` and the row-wise gathers are `(16, 16, 1)`, so a shell that
/// read only `x` would divide the wrong extent by the wrong number and be right
/// by accident on the modules that are `(n, 1, 1)`.
///
/// WGSL writes `@workgroup_size(16, 16)` and means `(16, 16, 1)`; `naga`
/// normalises that to three, which is why this can be a fixed array rather than
/// an option per axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Local(pub [u32; 3]);

/// The GEMM tile an `affine_qmm*` module was written for.
///
/// Not a choice the geometry makes. Every `Qmm` entrypoint is named
/// `..._bm_<rows>_bn_<cols>`, so by the time a driver has an entrypoint it has
/// already chosen the tile, and the grid must be built for THAT one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    /// Rows one workgroup's tile covers (`bm`).
    pub rows: u32,
    /// Columns one workgroup's tile covers (`bn`).
    pub cols: u32,
}

impl Tile {
    /// The tile named in an entrypoint, if it names one.
    ///
    /// Parsing the NAME rather than carrying a field, because the name is where
    /// the truth is: `kernels-wgpu/src/axes.rs` builds these variants by
    /// appending `_bm_N` and `_bn_N`, and the same suffix is what selects the
    /// `// pie:instantiate` line whose defines the body is expanded with. A
    /// separate field could drift from the module; the suffix cannot.
    #[must_use]
    pub fn from_entrypoint(name: &str) -> Option<Self> {
        Some(Self {
            rows: suffix_value(name, "_bm_")?,
            cols: suffix_value(name, "_bn_")?,
        })
    }
}

/// The decimal literal following `key` in `name`.
fn suffix_value(name: &str, key: &str) -> Option<u32> {
    let rest = &name[name.find(key)? + key.len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// What the module the driver is about to dispatch was WRITTEN for.
///
/// Both fields are facts about the WGSL, not decisions: `local` is read from
/// its `@workgroup_size` and `tile` from its entrypoint name. They travel
/// together because they are the same argument -- the numbers the geometry
/// divides by and does not get to pick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Module {
    /// The declared workgroup size.
    pub local: Local,
    /// The GEMM tile, for the rules that have one.
    pub tile: Option<Tile>,
}

impl Module {
    /// A module with no tile — everything that is not an `affine_qmm*`.
    #[must_use]
    pub fn new(local: [u32; 3]) -> Self {
        Self {
            local: Local(local),
            tile: None,
        }
    }

    /// The same, with whatever tile the entrypoint names.
    #[must_use]
    pub fn named(name: &str, local: [u32; 3]) -> Self {
        Self {
            local: Local(local),
            tile: Tile::from_entrypoint(name),
        }
    }

    /// What the driver will actually dispatch: the entrypoint it selected and
    /// the module it parsed for it.
    ///
    /// Both numbers come from the thing being launched — the workgroup from the
    /// entry point's own `@workgroup_size`, the tile from the name that
    /// selected it — so there is no table for either to drift from.
    #[must_use]
    pub fn loaded(name: &str, declared: &crate::reflect::Declared) -> Self {
        Self::named(name, declared.local)
    }
}

impl Local {
    /// The width of one workgroup along an axis, never zero.
    ///
    /// A zero would make the round-up below divide by zero, and WGSL forbids a
    /// zero in `@workgroup_size` — but this is the number a whole dispatch's
    /// shape is divided by, so it is worth not trusting.
    #[must_use]
    pub fn at(self, axis: usize) -> u32 {
        self.0[axis].max(1)
    }
}

/// Workgroups per dimension every WebGPU implementation must accept.
///
/// `wgpu::Limits::downlevel_defaults().max_compute_workgroups_per_dimension`,
/// and also the value of `Limits::default()`, restated here so the portable
/// half can name it with no `wgpu` present.
///
/// It is small enough to reach. A `Rule::Elementwise` launch over a 4096-wide
/// hidden and 32 rows is 131072 lanes, which is 512 workgroups at 256 wide --
/// fine -- but the same rule over a 151936-wide vocabulary at 16 rows is 9496
/// workgroups, and a 64-row prefill of it is 37984. The margin is one order of
/// magnitude, not six, which is why this is checked rather than assumed.
pub const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65535;

/// Why a rule could not answer.
///
/// Refusing rather than substituting, which is `driver-metal`'s finding and not
/// a new one: a rule that quietly answered a different shape produced a fire
/// that ran, took the time, and returned NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ungeometric {
    /// The row names no rule. Drift, not a runtime condition: a symbol reached
    /// dispatch whose contract does not say how to launch it.
    Unstated,
    /// The row names a rule this backend has no shader for.
    ///
    /// Not drift, and not an arithmetic condition: a legitimate answer.
    /// [`kernels::LaunchRule`] is the union of every backend's blocks -- the
    /// mamba family, MLA's prepare, the paged-score taps, the packed-head
    /// attentions, gemma-4's alt-up and its axial rope. Thirty-seven variants,
    /// of which one is [`Self::Unstated`]; of the thirty-six real ones this
    /// backend has geometry for fifteen. A deployment whose text reaches one of
    /// the other twenty-one is asking for a block that was never ported, and
    /// the useful answer names WHICH.
    ///
    /// A different sentence from [`Self::Unstated`], where the row exists and
    /// has not said how to launch it: that one is fixed by filling in `launch`,
    /// this one by writing a shader.
    ///
    /// The rules are enumerated one by one at the match rather than swept up
    /// behind a `_`, so the next rule the fleet adds stops this build instead
    /// of reaching a fire. That is not theoretical: twenty-one arrived at once
    /// while this crate was being written, and a `_` arm would have absorbed
    /// every one of them into a silent refusal -- or, worse, into whatever the
    /// arm above happened to compute.
    ///
    /// `driver-vulkan` refuses the same twenty-one, which is the answer worth
    /// having rather than a coincidence to note: the two tables are both
    /// `kernels-metal`'s coverage, so the two backends should be unable to run
    /// exactly the same blocks, and a divergence would mean one of the ports
    /// quietly grew or lost a row. `tests/rules.rs` holds the two ledgers
    /// against each other rather than leaving that to a reading.
    Unruled(
        /// The rule no shader here is launched as.
        Rule,
    ),
    /// A GEMM module whose entrypoint names no tile, or a non-GEMM one that
    /// does. Drift between this file and `kernels-wgpu/src/axes.rs`, not a
    /// runtime condition.
    Untiled,
    /// A decode-attention module written for one head width, dispatched for a
    /// fire of another.
    ///
    /// Not an arithmetic condition -- the driver selected the wrong module.
    /// `sdpa_vector.wgsl` declares `@workgroup_size(PIE_HEAD_DIM / 2)`, so
    /// TWICE the module's workgroup width is the head width it was built for,
    /// and every `_d_N` entrypoint agrees with the `N` in its name. That makes
    /// a mis-selection detectable HERE, before the dispatch, instead of showing
    /// up as an attention output that is subtly wrong.
    ///
    /// It is also where a variant with a different workgroup SHAPE lands, and
    /// that is the right answer rather than a gap: `sdpa_paged_decode`'s
    /// `PIE_TILED` arm is `@workgroup_size(32, 8)` and sweeps its rows in a
    /// loop, so no head width explains its 32, and a rule that divided by it
    /// would build a grid the body does not expect. Those entrypoints belong to
    /// rows that state no launch rule at all today, so they are refused twice
    /// over.
    HeadMismatch {
        /// The head width the module was written for -- twice its declared
        /// workgroup width, because a lane owns a bf16 PAIR.
        module: u32,
        /// The head width the fire asked for.
        fire: u32,
    },
    /// A rope or split whose width is not a whole number of heads.
    Unheaded {
        /// The row width that did not divide.
        width: u32,
        /// The head width it was measured against.
        head_dim: u32,
    },
    /// A GEMM whose row count no compiled tile divides. `driver-metal` records
    /// why this refuses instead of falling back to the matvec grid:
    /// `affine_qmm_t` reads its tile FROM the grid, so a matvec grid points it
    /// at a tiling that is not there and a two-token prefill came back NaN.
    PartialTile {
        /// The row count asked for.
        rows: u32,
        /// The narrowest compiled tile, which still did not divide it.
        tile: u32,
    },
    /// The grid is arithmetically right and larger than the device will take.
    ///
    /// WebGPU's own refusal, and the reason this variant exists rather than a
    /// panic at submit: `dispatch_workgroups` past
    /// `max_compute_workgroups_per_dimension` is a validation error inside
    /// `wgpu`, which surfaces as an error scope or a panic naming a limit and
    /// not naming the launch. A row that needs more than the device allows is
    /// a real condition with a real answer -- split the rectangle, or decline
    /// the model -- and neither is reachable from a message about a number.
    ///
    /// Not a defect in the rule. [`groups`] is what the kernel requires;
    /// [`groups_within`] is what this device will accept, and the gap between
    /// them is a fact about the device.
    PastDeviceLimit {
        /// Which grid axis, 0 for x.
        axis: usize,
        /// The workgroup count the rule asked for.
        groups: u32,
        /// What the device allows per dimension.
        limit: u32,
    },
}

impl core::fmt::Display for Ungeometric {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unstated => write!(f, "the row names no launch rule"),
            Self::Untiled => write!(f, "a GEMM module whose entrypoint names no tile"),
            Self::Unruled(rule) => write!(
                f,
                "the row names `{rule:?}`, a launch rule this backend has no \
                 shader for"
            ),
            Self::HeadMismatch { module, fire } => write!(
                f,
                "a module built for {module}-wide heads cannot serve a {fire}-wide one"
            ),
            Self::Unheaded { width, head_dim } => write!(
                f,
                "a row of {width} is not a whole number of {head_dim}-wide heads"
            ),
            Self::PartialTile { rows, tile } => {
                write!(f, "{rows} rows is not a whole number of {tile}-row tiles")
            }
            Self::PastDeviceLimit {
                axis,
                groups,
                limit,
            } => write!(
                f,
                "{groups} workgroups on axis {axis} is past the device's limit of {limit}"
            ),
        }
    }
}

impl core::error::Error for Ungeometric {}

/// The invocations a rule has to cover, before workgroups are considered.
///
/// This is `driver-metal`'s grid, in threads, and it is kept separate from the
/// division on purpose: it is the part that is the KERNEL's contract and is
/// identical on all three backends, so a difference between two shells here
/// would be a real disagreement rather than an API one.
///
/// Public because it is the only way to ask whether [`groups`] rounded — which
/// for the modules that read `num_workgroups` is the difference between a
/// harmless spare workgroup and a changed answer.
///
/// # Errors
///
/// The same [`Ungeometric`] cases [`groups`] returns, less
/// [`Ungeometric::PastDeviceLimit`], which is about a count of workgroups and
/// so cannot arise before the division.
pub fn lanes(rule: Rule, dims: Dims, module: Module) -> Result<[u32; 3], Ungeometric> {
    let rows = dims.rows.max(1);
    Ok(match rule {
        Rule::Unstated => return Err(Ungeometric::Unstated),
        // Rules for blocks this backend has no shader for. Listed one by one
        // and not behind a `_`, so that the next rule the fleet adds stops this
        // build rather than reaching a fire -- which is what it did the day
        // twenty-one of them arrived at once.
        //
        // The mamba family (`RecurrentScan`, `PerRow`, `PerChannel`,
        // `ElementwiseIn`, `WarpTiledScan`, `PerRowNarrow`, `Slab`, `Tile16`),
        // MLA's prepare, the paged-score taps, the packed-head attentions,
        // gemma-4's alt-up and its axial rope, and the routed matvec forms
        // `quant/qmv.wgsl` does not instantiate.
        Rule::RecurrentScan
        | Rule::PerRow
        | Rule::PerChannel
        | Rule::ElementwiseIn
        | Rule::RowScores
        | Rule::RowsPerHead
        | Rule::RowsFlat
        | Rule::Slab
        | Rule::Tile16
        | Rule::AxialRope
        | Rule::WarpTiledScan
        | Rule::PerRowNarrow
        | Rule::PagedScores
        | Rule::PagedScoresDecode
        | Rule::MlaPrepare
        | Rule::RowsPackedHeads
        | Rule::RowsPackedHeadsNarrow
        | Rule::WarpPackedHeads
        | Rule::RoutedQmvTransposed
        | Rule::AltUpStreams
        | Rule::RoutedQmvQuad
        // The three CUDA launcher shapes: a grid over the request count, and
        // the two literal one-block launches. No shader here is a serial walk
        // and `Dims` carries no request count, so all three are refused.
        | Rule::PerRequest
        | Rule::Single
        | Rule::SingleWarp => return Err(Ungeometric::Unruled(rule)),
        // Four outputs per subgroup, two subgroups per workgroup: 32 lanes
        // wide and `ceil(n / 4)` tall.
        //
        // The ROW goes on x, beside the 32 lanes of one subgroup -- Metal's
        // `qmv_mb` is `32 * n` wide and `quant/qmv.wgsl` reads
        // `workgroup_id.x` as exactly that index. A first Vulkan draft put rows
        // on z instead, where the shader never looks: it computed row 0, left
        // every other row holding its buffer's zeros, and returned success.
        //
        // The round-up on y is load-bearing for a different reason.
        // `driver-metal`'s `grid::qmv` records a shared expert's gate --
        // `hidden -> ONE logit a token` -- whose truncated count was zero
        // groups, so no threads ran, its buffer kept the zeros it was
        // allocated with, and every routed token was combined under
        // `sigmoid(0) = 0.5` instead of its own gate.
        Rule::Qmv => [module.local.at(0) * rows, dims.width.div_ceil(4), 1],
        Rule::Qmm => {
            // The tile comes from the MODULE. Choosing one here would be
            // choosing a decomposition the compiled shader does not have.
            let tile = module.tile.ok_or(Ungeometric::Untiled)?;
            let (bm, bn) = (tile.rows.max(1), tile.cols.max(1));
            if !rows.is_multiple_of(bm) {
                // Refusing, not falling back to a matvec grid:
                // `affine_qmm_t` reads its tile FROM the grid, so a matvec grid
                // points it at a tiling that is not there and a two-token
                // prefill came back entirely NaN.
                return Err(Ungeometric::PartialTile { rows, tile: bm });
            }
            // One workgroup per (column tile, row tile), and the module is
            // `(32, 2, 2)` -- 128 lanes cooperating on one tile, not one lane
            // per output.
            [
                module.local.at(0) * dims.width.div_ceil(bn),
                module.local.at(1) * rows.div_ceil(bm),
                module.local.at(2),
            ]
        }
        // One workgroup per AXIS, not per row, and the two are only the same
        // for a norm that spans its row. gemma-4 normalizes each head of an
        // 8192-wide Q over 256 channels: 32 axes per token where a row-wise
        // grid gives one, which left head 0 normalized and the other 31 as the
        // projection wrote them.
        Rule::Rms => {
            let axis = if dims.axis == 0 {
                dims.width
            } else {
                dims.axis
            };
            let per_row = if axis == 0 {
                1
            } else {
                dims.width.div_ceil(axis)
            };
            // Every row on ONE axis, because `norm/rms.wgsl` takes its row from
            // `workgroup_id.x` and never mentions y. A first Vulkan draft put
            // the count on y: it launched a single workgroup on x, computed row
            // 0, left rows 1.. holding the zeros their buffer was born with,
            // and returned success from every call in the chain. The lane
            // sweeps all passed -- they count lanes, and a lane on an axis
            // nobody reads is a lane all the same.
            [module.local.at(0) * per_row * rows, 1, 1]
        }
        Rule::Rope => {
            let heads = rope_heads(dims)?;
            [dims.rotary_dims / 2, heads, rows]
        }
        // Rows stack FLAT on one axis, which is what separates this from
        // `ElementwiseRows`: the two agree at one row and disagree above it.
        Rule::Elementwise => [dims.width * rows, 1, 1],
        Rule::ElementwiseRows => [dims.width, rows, 1],
        Rule::SplitPacked => [dims.in_width, rows, 1],
        Rule::PerHead => [dims.head_dim, dims.kv_heads, rows],
        Rule::SdpaVector => {
            // TWICE the module's workgroup width is the head width it was
            // written for, and the doubling is this file's one real departure
            // from `driver-vulkan`'s arithmetic -- see the module docs. A
            // driver that selected `_d_256` for a 128-wide head would otherwise
            // get a grid that looks reasonable and an answer that is not, so
            // the two are required to agree.
            let built_for = module.local.at(0) * 2;
            // No `.max(1)` on the head width: `Local::at` already answers at
            // least one, so a zero head width can never equal `built_for` and
            // is refused by the line below under its real name. An ODD head
            // width lands there too, which is correct rather than incidental:
            // a bf16 pair cannot cover half a channel.
            let head_dim = dims.head_dim;
            if built_for != head_dim {
                return Err(Ungeometric::HeadMismatch {
                    module: built_for,
                    fire: head_dim,
                });
            }
            // The head count MULTIPLIES the workgroup width rather than
            // standing on its own axis: `sdpa_vector.wgsl` is one workgroup per
            // (head, row), and it reads `num_workgroups.x` as the head count
            // and `num_workgroups.y` as the row count -- so BOTH axes are
            // quantities the shader computes with and neither may be rounded.
            // Neither is: the division is exact on x because the workgroup
            // width is the factor, and exact on y because the local size there
            // is one.
            [module.local.at(0) * dims.q_heads, rows, 1]
        }
        Rule::PerHeadElementwise => [dims.q_heads * dims.head_dim, 1, 1],
        Rule::GatedRms => [dims.head_dim, dims.kv_heads, 1],

        Rule::RouterLane => [module.local.at(0), rows, 1],
        // ONE workgroup whatever the rows: the sort is over the expert
        // histogram, which is fire-wide.
        Rule::RouterSort => [module.local.at(0), 1, 1],
        Rule::RouteRows => [dims.width, rows, 1],
        // Rows on x as for `Qmv`, the OUTPUT axis on y and the expert slot on z.
        //
        // The y extent is `width` and not `width / 4`, and that division is the
        // defect this comment exists to stop coming back.
        //
        // Metal's `qmv_routed` runs a `[32, 2, 1]` threadgroup in which each y
        // thread owns FOUR output rows, so its thread extent on y is
        // `ceil(n / 4)` and `driver-metal` is right to say so. This tree's
        // `moe/qmv_routed.wgsl` is `@workgroup_size(32, 8)` and computes
        // `out_row = workgroup_id.y * 8 + local_invocation_id.y` — **one output
        // row per y lane**. Carrying Metal's expression across and then letting
        // `groups()` divide by this module's own 8 gives `ceil(n / 32)` where
        // `ceil(n / 8)` is needed: a **four-fold undershoot**.
        //
        // Measured on an RTX 4090 at `n = 13`: output row 8 was never written,
        // its sentinel survived, and the dispatch returned success. That is the
        // failure mode this whole file is written against — an undershot grid
        // writes nothing, the gap reads back as whatever the buffer held, and
        // nothing reports it. At a real 2048-wide expert projection it is 64
        // workgroups where 256 are needed, so seven eighths of every routed
        // matvec would be the arena's stale bytes.
        //
        // `kernels-wgpu`'s GPU suite is what found it, by dispatching the row
        // and comparing numbers. `no_rule_launches_fewer_lanes_than_its_extent`
        // could not: `lanes()` is both the claim and the reference there, so a
        // rule that lies agrees with itself. The same blindness let
        // `Rule::SdpaVector` be wrong until a module's own `@workgroup_size`
        // was compared against it, which is the check that exists now.
        //
        // `driver-vulkan` carries the identical expression against a
        // `local_size_y = 8` shader, so it has the same four-fold undershoot.
        // Reported rather than fixed here: it is the sibling's file.
        Rule::RoutedQmv => [
            module.local.at(0) * rows,
            dims.width.max(1),
            dims.experts_per_token.max(1),
        ],
    })
}

/// How many workgroups `dispatch_workgroups` should be given.
///
/// The round-up is applied on every axis, because a missing workgroup runs
/// nothing and says nothing while an extra one runs lanes past the end of the
/// tensor, and every pointwise body in `kernels-wgpu` guards its own tail
/// against the bound length of what it writes — `arrayLength(&out_)`, which is
/// the binding's own size and needs nothing from the row.
///
/// That last clause is not true of the modules that read `num_workgroups` and
/// use it as a QUANTITY rather than a bound: `rope/neox.wgsl` takes
/// `num_workgroups.x` as the rotary pair count it strides each pair's partner
/// by and divides the frequency exponent by. For those an extra workgroup does
/// not run a guarded lane, it changes the arithmetic every lane does, and the
/// round-up has to be a no-op. It is -- their extents divide their workgroups
/// exactly, which is why `rope` declares `@workgroup_size(1)` --
/// and [`crate::reflect::Declared::reads_workgroup_count`] is how a caller
/// finds the modules where that matters. [`lanes`] is what lets it ask the
/// same question of the grid.
///
/// This answer is what the KERNEL needs and takes no view of what the device
/// will accept; see [`groups_within`].
///
/// # Errors
///
/// [`Ungeometric`] when the rule cannot answer for these dimensions. A driver
/// must not substitute a shape here — see [`Ungeometric::PartialTile`].
pub fn groups(rule: Rule, dims: Dims, module: Module) -> Result<[u32; 3], Ungeometric> {
    // The RULE is asked first, and the order is the whole content of these two
    // lines. `driver-vulkan` makes the tile cross-check the first thing it
    // does; carrying that across reported `Untiled` for 242 of this table's
    // 292 unstated entrypoints, because `affine_qmm_t_routed_..._bm_16_bn_16`
    // and its siblings name a tile in a name whose ROW states no launch rule.
    //
    // Both refusals are true of those entrypoints and only one is useful.
    // `Untiled` says "this file and `kernels-wgpu/src/axes.rs` disagree about
    // which names carry a tile", which sends the reader to look for a naming
    // defect that is not there; `Unstated` says "fill in `launch`", which is
    // the thing actually missing. A refusal that names the wrong repair is
    // worse than a vaguer one, because it is followed.
    let e = lanes(rule, dims, module)?;
    // A module whose rule is served and is not a GEMM, whose entrypoint names a
    // tile anyway: a name grew a `_bm_` suffix without this file learning what
    // it decomposes. Now genuinely drift, because the rule has already been
    // established to be one this backend launches.
    if module.tile.is_some() && rule != Rule::Qmm {
        return Err(Ungeometric::Untiled);
    }
    Ok([
        e[0].div_ceil(module.local.at(0)),
        e[1].div_ceil(module.local.at(1)),
        e[2].div_ceil(module.local.at(2)),
    ])
}

/// The same grid, refused by name when the device will not take it.
///
/// `limit` is `wgpu::Limits::max_compute_workgroups_per_dimension` from the
/// adapter that was opened, whose guaranteed floor is
/// [`MAX_WORKGROUPS_PER_DIMENSION`]. It is a parameter and not a constant for
/// the reason [`crate::facts::of`]'s alignment is: a device may allow more, a
/// browser may allow exactly the floor, and clamping to either would be
/// answering for a machine that is not the one running.
///
/// Separate from [`groups`] rather than folded into it, because the two answer
/// different questions and a caller needs both. `groups` is what the kernel
/// requires; a shell asks THIS, and a test that checks a rule covers its extent
/// asks the other -- a rule is not wrong for needing a grid some device
/// declines.
///
/// # Errors
///
/// Every case [`groups`] has, plus [`Ungeometric::PastDeviceLimit`] naming the
/// first axis that is over. First and not all, because a caller that has to
/// split a rectangle splits it once and asks again.
pub fn groups_within(
    rule: Rule,
    dims: Dims,
    module: Module,
    limit: u32,
) -> Result<[u32; 3], Ungeometric> {
    let g = groups(rule, dims, module)?;
    for (axis, &n) in g.iter().enumerate() {
        if n > limit {
            return Err(Ungeometric::PastDeviceLimit {
                axis,
                groups: n,
                limit,
            });
        }
    }
    Ok(g)
}

/// How many heads THIS launch's tensor has — which is not the fire's `q_heads`.
///
/// `driver-metal` records what assuming otherwise cost: rope is stated twice,
/// once for q and once for k, and a grouped-query deployment gives the two
/// different head counts. Answering `q_heads` for k rotates 32 heads over an
/// 8-head buffer AND strides its rows by the wrong pitch. At one row the stride
/// never applies and only the overrun happens, so a decode agreed token for
/// token while a prefill left every key row after the first unrotated.
///
/// The launch's own operand carries the answer: `width` is the row width of the
/// tensor being rotated, so the head count is `width / head_dim`.
fn rope_heads(dims: Dims) -> Result<u32, Ungeometric> {
    if dims.head_dim == 0 || dims.width == 0 || !dims.width.is_multiple_of(dims.head_dim) {
        return Err(Ungeometric::Unheaded {
            width: dims.width,
            head_dim: dims.head_dim,
        });
    }
    Ok(dims.width / dims.head_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plausible fire, at a width that does not divide a workgroup.
    ///
    /// 32 rows and not 4: `Rule::Qmm` REFUSES a row count no compiled tile
    /// divides, so a sweep at four rows would find that rule absent and prove
    /// less than it looked like it was proving.
    ///
    /// 4224 and not 4096, and the difference is the lesson `.wiki`'s Vulkan
    /// notes record in one line: an undershot grid writes nothing, the gap
    /// reads back as the zeros the buffer was born with, and the dispatch
    /// completes successfully -- so only a shape that does not divide evenly
    /// can tell `div_ceil` from `/`. 4224 is 33 x 128, so it is a whole number
    /// of heads (which `Rule::Rope` requires, or it would refuse and leave the
    /// sweep below one rule short) and is NOT a multiple of 256 or 1024, which
    /// are the two widest workgroups in this tree.
    fn dims() -> Dims {
        Dims {
            rows: 32,
            width: 4224,
            in_width: 12672,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            axis: 4224,
            rotary_dims: 128,
            n_experts: 64,
            experts_per_token: 8,
        }
    }

    /// The workgroup shapes this tree actually declares, plus two it does not.
    const LOCALS: [[u32; 3]; 6] = [
        [256, 1, 1],
        [32, 2, 2],
        [16, 16, 1],
        [1024, 1, 1],
        [64, 1, 1],
        [32, 8, 1],
    ];

    /// Every workgroup count, times its workgroup, covers the thread extent.
    ///
    /// This is the property the whole module exists for, and it is stated as a
    /// sweep rather than per rule because the failure it catches is a rule
    /// nobody thought about. An undershoot writes nothing and reports success.
    #[test]
    fn no_rule_launches_fewer_lanes_than_its_extent() {
        let mut checked = 0;
        for &rule in kernels::LaunchRule::ALL {
            for local in LOCALS {
                // Only a GEMM carries a tile, and only a GEMM is allowed to.
                let m = Module {
                    local: Local(local),
                    tile: (rule == Rule::Qmm).then_some(Tile { rows: 32, cols: 64 }),
                };
                // Decode attention is written per head width -- its
                // `@workgroup_size` IS `PIE_HEAD_DIM` -- so a sweep over
                // workgroup shapes has to move the fire's head width with it.
                // Otherwise the rule refuses five of the six shapes and the
                // sweep proves nothing about the family it most needed to.
                let d = if rule == Rule::SdpaVector {
                    Dims {
                        head_dim: local[0] * 2,
                        ..dims()
                    }
                } else {
                    dims()
                };
                let Ok(e) = lanes(rule, d, m) else {
                    continue;
                };
                let g = groups(rule, d, m).expect("extent answered");
                for axis in 0..3 {
                    assert!(
                        g[axis] * m.local.at(axis) >= e[axis],
                        "{rule:?} on axis {axis} with {local:?}: {} lanes for an \
                         extent of {}",
                        g[axis] * m.local.at(axis),
                        e[axis]
                    );
                }
                checked += 1;
            }
        }
        // A sweep that skipped everything would pass, and most of the fleet's
        // rules ARE skipped here -- see `SERVED`, which is the list this floor
        // is taken from and which
        // `the_rules_this_backend_serves_are_exactly_the_ones_with_shaders`
        // says what it means.
        assert!(
            checked >= SERVED.len() * LOCALS.len(),
            "{checked} checks over {} shapes",
            LOCALS.len()
        );
    }

    /// And the round-up is a real round-up, which the sweep above cannot say.
    ///
    /// A sweep that only asserts coverage passes under plain division whenever
    /// the extent divides -- which is why the dims here are 460 over a 256-wide
    /// workgroup and 13 rows over a 16-row tile, the two shapes the Vulkan
    /// notes name as the ones that told `div_ceil` from `/` when five earlier
    /// tests at exact multiples could not.
    #[test]
    fn a_shape_that_does_not_divide_is_rounded_up_and_not_truncated() {
        let ragged = Dims {
            rows: 1,
            width: 460,
            ..dims()
        };
        let m = Module::new([256, 1, 1]);
        assert_eq!(
            groups(Rule::Elementwise, ragged, m).expect("answers")[0],
            2,
            "460 lanes over a 256-wide workgroup is two groups; truncation \
             would launch one and leave 204 elements holding their zeros"
        );

        // And on the axis a GEMM tiles by, where truncation loses a whole tile
        // of rows rather than a tail of lanes.
        let tiled = Dims {
            rows: 32,
            width: 100,
            ..dims()
        };
        let gemm = Module::named("affine_qmm_t_bfloat16_bm_16_bn_64", [32, 2, 2]);
        assert_eq!(
            groups(Rule::Qmm, tiled, gemm).expect("answers")[0],
            2,
            "100 columns over a 64-wide tile is two column tiles"
        );
    }

    /// Every launch rule this backend has a shader for.
    ///
    /// [`kernels::LaunchRule`] is the whole fleet's vocabulary and most of it
    /// is CUDA's: mamba scans, MLA, the paged-score taps, packed-head
    /// attentions, gemma-4's alt-up. This is the part `kernels-wgpu` states a
    /// row for, and [`lanes`] answers for exactly these.
    const SERVED: &[Rule] = &[
        Rule::Qmv,
        Rule::Rms,
        Rule::Rope,
        Rule::Elementwise,
        Rule::ElementwiseRows,
        Rule::PerHead,
        Rule::SdpaVector,
        Rule::PerHeadElementwise,
        Rule::GatedRms,
        Rule::RouterLane,
        Rule::RouterSort,
        Rule::RouteRows,
        Rule::RoutedQmv,
        Rule::SplitPacked,
        Rule::Qmm,
    ];

    /// The rules this backend serves are exactly `SERVED`, and every other rule
    /// the fleet states refuses BY NAME.
    ///
    /// The point is the second half. [`lanes`] names all twenty-four unserved
    /// rules in one arm rather than catching them with a `_`, so that the
    /// compiler stops this build when the fleet grows a rule -- which is
    /// exactly what happened when twenty-one arrived at once, mid-session, and
    /// this match was the thing that noticed. A wildcard would have compiled,
    /// and a mamba text would have reached a fire and been refused there, or
    /// worse, launched on whichever grid the wildcard chose.
    ///
    /// So this test is a ledger and a tripwire: a rule added upstream is either
    /// put in `SERVED` with a shader behind it, or it is refused, and there is
    /// no third state. The counts are asserted to ADD UP to the whole
    /// vocabulary, which is what makes that "no third state" a measurement
    /// rather than a hope -- a `_` arm added tomorrow keeps every individual
    /// assertion below passing and moves the totals.
    ///
    /// It asks both [`lanes`] and [`groups`], and it compares the PAYLOAD of
    /// the refusal rather than merely that there was one: an arm that refused
    /// every unserved rule as `Unruled(RecurrentScan)` would satisfy
    /// `is_err()` and hand a person the wrong rule to go and look for.
    #[test]
    fn the_rules_this_backend_serves_are_exactly_the_ones_with_shaders() {
        let (mut served, mut unruled) = (0usize, 0usize);
        for &rule in kernels::LaunchRule::ALL {
            let module = Module {
                local: Local([32, 2, 2]),
                tile: (rule == Rule::Qmm).then_some(Tile { rows: 32, cols: 64 }),
            };
            // `SdpaVector` refuses a module whose workgroup width is not HALF
            // the fire's head width -- that is the mis-selection guard, not a
            // rule gap -- so it is asked with the module it was built for.
            let d = if rule == Rule::SdpaVector {
                Dims {
                    head_dim: module.local.at(0) * 2,
                    ..dims()
                }
            } else {
                dims()
            };
            // Both entry points, because they are two different matches: a
            // caller that only ever asked `groups` would not notice `lanes`
            // growing a `_`, and vice versa.
            let answer = lanes(rule, d, module);
            let grid = groups(rule, d, module);
            if SERVED.contains(&rule) {
                assert!(answer.is_ok(), "{rule:?} is served but did not answer");
                assert!(grid.is_ok(), "{rule:?} answered an extent and no grid");
                served += 1;
            } else if rule == Rule::Unstated {
                assert_eq!(answer, Err(Ungeometric::Unstated));
                assert_eq!(grid, Err(Ungeometric::Unstated));
            } else {
                // NAMING ITSELF, which is the whole content of the assertion.
                // `Err(Unruled(SomeOtherRule))` would pass an `is_err()` check
                // and hand the caller the wrong diagnosis; so would a `_` arm
                // that refused with a fixed rule. Both are the failure this
                // test is for, and only comparing the payload catches them.
                assert_eq!(
                    answer,
                    Err(Ungeometric::Unruled(rule)),
                    "{rule:?} has no shader and must refuse by name"
                );
                assert_eq!(grid, Err(Ungeometric::Unruled(rule)));
                // And the refusal a caller PRINTS names it too. A driver
                // declining a model says this sentence to a person, and
                // "unsupported launch rule" without the rule in it is a
                // sentence that sends them to read this file.
                let said = grid.unwrap_err().to_string();
                assert!(
                    said.contains(&format!("{rule:?}")),
                    "the refusal for {rule:?} reads `{said}` and does not name it"
                );
                unruled += 1;
            }
        }
        // The three counts have to add up to the whole vocabulary, which is
        // what makes this a CLOSURE rather than a sample: every variant the
        // fleet states was visited and landed in exactly one of the three.
        assert_eq!(
            served + unruled + 1,
            kernels::LaunchRule::ALL.len(),
            "{served} served and {unruled} unruled and one unstated is not the \
             {} rules the fleet states",
            kernels::LaunchRule::ALL.len()
        );
        assert_eq!(served, SERVED.len());
        assert_eq!(
            unruled, 24,
            "twenty-four rules have no shader here; if that moved, either a \
             shader was written or the fleet grew a rule, and both are \
             decisions somebody should have made on purpose"
        );
    }

    /// And no row in the table names a rule this file refuses.
    ///
    /// The other direction, and the one that would catch `kernels-wgpu` growing
    /// a mamba row before this crate grew a grid for it. The table's 99 rows
    /// use `SERVED` plus `Unstated`; a row naming anything else would plan and
    /// then refuse at every fire, which is a worse failure than not compiling.
    #[test]
    fn no_row_in_the_table_names_a_rule_this_backend_refuses() {
        for sig in kernels_wgpu::KERNELS {
            assert!(
                sig.launch == Rule::Unstated || SERVED.contains(&sig.launch),
                "`{}` states {:?}, which this backend has no grid for",
                sig.symbol,
                sig.launch
            );
        }
    }

    /// Which rules answer a grid with a zero in it, stated exactly.
    ///
    /// A zero on any axis dispatches nothing and reports success -- the failure
    /// with no symptom, and the one this crate refuses hardest.
    /// [`crate::dispatch::plan_one`] is the layer that refuses it
    /// (`Undispatchable::Empty`). This asks the question one layer down, where
    /// several rules clamp a dimension to one for exactly this reason.
    ///
    /// Pinned as a SET rather than asserted empty, because it is a long way
    /// from empty. Stating the set is what turns "groups does not guarantee a
    /// non-empty grid" from an assumption into a fact with a number, and makes
    /// the next change visible -- a clamp deleted here moves this set, a new
    /// rule joins it, and a rule that starts refusing leaves it.
    ///
    /// `Err` is not a failure here. A rule that refuses a degenerate dimension
    /// has said so by name, which is the opposite of the defect.
    #[test]
    fn only_these_rules_answer_a_grid_with_a_zero_in_it() {
        // One field at a time, so a rule that clamps the field under test is
        // not covered for it by another field still being sane.
        type Zero = (&'static str, fn(&mut Dims));
        let zeroed: [Zero; 8] = [
            ("rows", |d| d.rows = 0),
            ("width", |d| d.width = 0),
            ("in_width", |d| d.in_width = 0),
            ("q_heads", |d| d.q_heads = 0),
            ("kv_heads", |d| d.kv_heads = 0),
            ("head_dim", |d| d.head_dim = 0),
            ("axis", |d| d.axis = 0),
            ("experts_per_token", |d| d.experts_per_token = 0),
        ];
        let mut answered = 0;
        let mut empty = Vec::new();
        for &rule in kernels::LaunchRule::ALL {
            for local in LOCALS {
                let m = Module {
                    local: Local(local),
                    tile: (rule == Rule::Qmm).then_some(Tile { rows: 32, cols: 64 }),
                };
                for (name, zero) in zeroed {
                    let mut d = if rule == Rule::SdpaVector {
                        Dims {
                            head_dim: local[0] * 2,
                            ..dims()
                        }
                    } else {
                        dims()
                    };
                    zero(&mut d);
                    let Ok(g) = groups(rule, d, m) else {
                        continue;
                    };
                    answered += 1;
                    if g.contains(&0) {
                        let case = format!("{rule:?}/{name}");
                        if !empty.contains(&case) {
                            empty.push(case);
                        }
                    }
                }
            }
        }
        empty.sort();
        assert_eq!(
            empty,
            [
                "Elementwise/width",
                "ElementwiseRows/width",
                "GatedRms/head_dim",
                "GatedRms/kv_heads",
                "PerHead/head_dim",
                "PerHead/kv_heads",
                "PerHeadElementwise/head_dim",
                "PerHeadElementwise/q_heads",
                "Qmm/width",
                "Qmv/width",
                "Rms/width",
                "RouteRows/width",
                "SdpaVector/q_heads",
                "SplitPacked/in_width",
            ],
            "a different set of rules answers an empty grid"
        );
        // A sweep where every rule refused would pass while asking nothing.
        // Fifteen served rules over six workgroup shapes and eight zeroed
        // fields is 720 questions; most are answered, and the floor is set
        // below the number rather than at it so a rule that starts refusing a
        // degenerate dimension -- which is an improvement -- does not fail it.
        assert!(answered > 200, "only {answered} rules answered at all");
    }

    /// And not WILDLY more, which is the other way a division can be wrong.
    ///
    /// One extra workgroup per axis is the round-up. More than that means the
    /// extent was divided by the wrong number — the mistake a shell makes when
    /// it reads the first component of `@workgroup_size` and ignores the rest.
    #[test]
    fn no_rule_launches_a_whole_spare_workgroup() {
        for &rule in kernels::LaunchRule::ALL {
            let m = Module {
                local: Local([32, 2, 2]),
                tile: (rule == Rule::Qmm).then_some(Tile { rows: 32, cols: 64 }),
            };
            let Ok(e) = lanes(rule, dims(), m) else {
                continue;
            };
            let g = groups(rule, dims(), m).expect("extent answered");
            for axis in 0..3 {
                let launched = g[axis] * m.local.at(axis);
                assert!(
                    launched < e[axis] + m.local.at(axis),
                    "{rule:?} on axis {axis}: {launched} lanes for an extent of {}",
                    e[axis]
                );
            }
        }
    }

    /// A decode-attention module built for another head width refuses.
    ///
    /// And the module that MATCHES is the one half the head width, which is
    /// this backend's divergence in one assertion: a lane owns a bf16 pair, so
    /// `_d_128` declares `@workgroup_size(64)`. Carrying the Vulkan expression
    /// across would refuse every real dispatch here and accept none.
    #[test]
    fn a_module_built_for_another_head_width_refuses() {
        assert_eq!(
            groups(Rule::SdpaVector, dims(), Module::new([256, 1, 1])),
            Err(Ungeometric::HeadMismatch {
                module: 512,
                fire: 128
            })
        );
        // The 128-wide module is built for 256-wide heads and is refused too,
        // which is the assertion that stops the halving being written the other
        // way round.
        assert_eq!(
            groups(Rule::SdpaVector, dims(), Module::new([128, 1, 1])),
            Err(Ungeometric::HeadMismatch {
                module: 256,
                fire: 128
            })
        );
        // And the matching one answers, so the check is a check and not a wall.
        assert_eq!(
            groups(Rule::SdpaVector, dims(), Module::new([64, 1, 1])).expect("matches"),
            [32, 32, 1],
            "one workgroup per (head, row): the shader reads both counts off \
             `num_workgroups`, so neither axis may carry a spare"
        );
    }

    /// Every decode-attention module in the tree is half the head it serves.
    ///
    /// The one arithmetic divergence from `driver-vulkan`, measured against
    /// the shaders rather than argued from them. `suffix_value` reads the
    /// `_d_N` out of the entrypoint -- which is what SELECTED the module, so
    /// it is the head width the driver believes it asked for -- and `naga`
    /// reads the `@workgroup_size` out of the source the same name expands to.
    /// The two are computed from different places and have to agree at a
    /// factor of two.
    ///
    /// This is the check that would catch the divergence being undone. A
    /// shader author who "fixed" `PIE_PAIRS` to `PIE_HEAD_DIM` would make
    /// every decode dispatch here refuse; one who did it while somebody else
    /// removed the doubling below would make every decode launch twice the
    /// workgroups and read twice the head count off `num_workgroups.x`, which
    /// is a wrong answer and not a refusal.
    ///
    /// Entrypoints the tree cannot yet compile are skipped for the reason
    /// `crate::reflect`'s own sweep gives -- that is `kernels-wgpu`'s claim,
    /// not this one's -- and the count is asserted so the skip cannot empty
    /// the test.
    #[test]
    fn a_decode_attention_module_is_half_the_head_it_serves() {
        let mut checked = 0;
        for sig in kernels_wgpu::KERNELS {
            if sig.launch != Rule::SdpaVector {
                continue;
            }
            for name in sig.entrypoints() {
                let Ok(declared) =
                    crate::reflect::entrypoint(&name, kernels_wgpu::Capability::Baseline)
                else {
                    continue;
                };
                let head_dim = suffix_value(&name, "_d_")
                    .unwrap_or_else(|| panic!("`{name}` states no head width in its name"));
                let module = Module::loaded(&name, &declared);
                assert_eq!(
                    module.local.at(0) * 2,
                    head_dim,
                    "`{name}` declares a workgroup of {:?} for a {head_dim}-wide head",
                    declared.local
                );
                // And the rule accepts it, which is the same claim from the
                // other side: one workgroup per (head, row), exactly.
                let d = Dims { head_dim, ..dims() };
                assert_eq!(
                    groups(Rule::SdpaVector, d, module).unwrap_or_else(|e| panic!("`{name}`: {e}")),
                    [d.q_heads, d.rows, 1]
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked, 13,
            "the table states four decode-attention rows over 13 entrypoints, \
             and every one is measured against its own module -- a floor here \
             would let the sweep shrink without saying so"
        );
    }

    /// No module reads a grid axis its rule leaves flat.
    ///
    /// The third member of the family `a_decode_attention_module_is_half_the_
    /// head_it_serves` and `a_routed_matvec_covers_every_output_row_its_module_
    /// owns` belong to, and the one that generalises: instead of naming a rule
    /// and checking its arithmetic, it asks every row whether the SHAPE its
    /// rule produces can carry the axes its module actually reads.
    ///
    /// A rule that answers `[n, 1, 1]` promises the body indexes on x alone. If
    /// that body reads `global_invocation_id.y` as a row, every row but the
    /// first is never written — and an undershot grid writes nothing, the gap
    /// reads back as whatever the buffer held, and the dispatch returns
    /// success. There is no symptom until somebody compares numbers.
    ///
    /// `crate::reflect` already computes which axes a module reads, and it has
    /// been computed and never asked. It is safe to ask because it errs in the
    /// right direction: where the call walk cannot follow a builtin into a
    /// helper it answers "every axis" rather than "no axis", so a module this
    /// check cannot read makes the check STRICTER rather than vacuous.
    ///
    /// ## The defect it catches
    ///
    /// `geglu_tanh_strided`'s row states `LaunchRule::Elementwise`, which is
    /// `[width * rows, 1, 1]`. Its body alone among the five in
    /// `mlp/gated.wgsl` is `@workgroup_size(16, 16)` and reads `gid.y` as the
    /// row, so one workgroup on y covers 16 rows and every row past 15 keeps
    /// the bytes it was allocated with. Measured at 21 rows on a 4090: row 16
    /// held its sentinel and the dispatch succeeded.
    ///
    /// `kernels-vulkan`'s copy is `local_size_y = 16` over the same rule, so
    /// the sibling has it too. The row is shared with `kernels-metal` — where
    /// a threadgroup is sized at dispatch and `Elementwise` is right — so
    /// fixing it is a change to the shared table (`ElementwiseRows` is the
    /// shape this body wants) and a parity question rather than a local one.
    ///
    /// Until that lands the row is listed below with its reason. The list is
    /// asserted to be exactly what is known, so a second row joining it is a
    /// failure that has to be read rather than a line quietly added.
    #[test]
    fn no_module_reads_a_grid_axis_its_rule_leaves_flat() {
        // Rows whose module reads an axis the rule flattens. Each is a defect,
        // each is named, and the list is pinned so it cannot grow in silence.
        const KNOWN: [(&str, &str); 1] = [(
            "geglu_tanh_strided",
            "states `Elementwise` ([n, 1, 1]) and its body is @workgroup_size(16, 16) \
             reading gid.y as the row, so every row past 15 is never written. \
             `ElementwiseRows` is the shape it wants; the row is shared with \
             `kernels-metal`, where a dispatch-sized threadgroup makes \
             `Elementwise` correct, so the fix is a change to the shared table.",
        )];

        let d = Dims {
            rows: 7,
            width: 96,
            in_width: 96,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 64,
            axis: 96,
            rotary_dims: 64,
            n_experts: 8,
            experts_per_token: 2,
        };

        let mut found: Vec<String> = Vec::new();
        let mut wasted: Vec<String> = Vec::new();
        let mut checked = 0usize;

        for sig in kernels_wgpu::KERNELS {
            if sig.launch == Rule::Unstated {
                continue;
            }
            for name in sig.entrypoints() {
                let Ok(declared) =
                    crate::reflect::entrypoint(&name, kernels_wgpu::Capability::Baseline)
                else {
                    continue;
                };
                let module = Module::loaded(&name, &declared);
                let Ok(extent) = lanes(sig.launch, d, module) else {
                    // A rule this backend refuses is `Unruled`'s business, and
                    // a geometry that cannot be built cannot be compared.
                    continue;
                };
                checked += 1;

                for (axis, lanes) in extent.iter().enumerate() {
                    // The rule leaves an axis FLAT when it puts exactly one
                    // lane on it: the body may then only ever see index 0
                    // there.
                    if *lanes > 1 || !declared.grid_axes[axis] {
                        continue;
                    }
                    // A module whose own workgroup is wider than one on this
                    // axis is reading a LOCAL index, which is legitimate:
                    // `local_invocation_id.y` is a lane within the group and
                    // has nothing to do with the grid. Only a global read of a
                    // flattened axis is the defect, and `grid_axes` is about
                    // the global builtins.
                    found.push(format!("{} ({name}) reads axis {axis}", sig.symbol));
                }

                // The MIRROR, which is harmless and is checked anyway.
                //
                // `driver-vulkan/tests/rules.rs` has exactly this and only
                // this — `if !read && given > 1` — and it is why that crate
                // did not catch `geglu_tanh_strided`: work given to an axis
                // nothing reads is WASTE, and an axis the body reads that the
                // rule leaves at one is DATA LOSS. They are different
                // predicates and only the first was written.
                //
                // Kept because waste is still a disagreement between a rule
                // and a body, and a rule that hands out a dimension nobody
                // uses is usually a rule that meant to hand out a different
                // one. It is reported separately so the two never get
                // confused for each other again.
                for (axis, lanes) in extent.iter().enumerate() {
                    if *lanes <= 1 || declared.grid_axes[axis] {
                        continue;
                    }
                    wasted.push(format!(
                        "{} ({name}) is given {lanes} on axis {axis}, which it never reads",
                        sig.symbol
                    ));
                }
            }
        }

        assert!(
            checked > 60,
            "only {checked} entrypoints were geometry-checked; a sweep that \
             read almost nothing passes as loudly as one that read everything"
        );

        let mut symbols: Vec<&str> = found
            .iter()
            .filter_map(|f| f.split_whitespace().next())
            .collect();
        symbols.sort_unstable();
        symbols.dedup();

        let known: std::collections::BTreeSet<&str> = KNOWN.iter().map(|(name, _)| *name).collect();
        let fresh: Vec<&&str> = symbols.iter().filter(|s| !known.contains(**s)).collect();
        assert!(
            fresh.is_empty(),
            "these rows read a grid axis their rule flattens, which means every \
             index past the first on that axis is never written and the \
             dispatch succeeds anyway:\n  {}\n\nfull list:\n  {}",
            fresh.iter().map(|s| **s).collect::<Vec<_>>().join("\n  "),
            found.join("\n  "),
        );

        // The waste half, which is reported separately from the data-loss half
        // above so the two are never mistaken for one another — that
        // conflation is the mistake `driver-vulkan/tests/rules.rs` is living
        // with, where `if !read && given > 1` is the ONLY direction checked
        // and the dangerous one is its mirror.
        //
        // It is NOT asserted empty, and the reason is the more interesting
        // half of this test.
        //
        // Running it turned up four rows, and they are two different things:
        //
        // * `route_sort` and the two `router_topk`s read no global builtin at
        //   all — they take `local_invocation_id` and stride over the
        //   workgroup width, which is exactly what their rules intend. `Given
        //   256 lanes on x` is one workgroup, not 256 of them, and the rule is
        //   right. This half of the report is a category error in the check
        //   rather than in the row: `lanes()` is a THREAD extent and a body
        //   that indexes by lane consumes it correctly.
        //
        // * `kv_append` reads `gid.z` — the line is right there in
        //   `attn/kv_write.wgsl` — and `grid_axes` answers `[true, true,
        //   false]`. That is a FALSE NEGATIVE in the reflection, and it is the
        //   documented weakness: `crate::reflect`'s notes say the walk follows
        //   a builtin into a helper with a depth bound and answers "every
        //   axis" where it cannot follow. Here it followed, decided, and
        //   decided wrong.
        //
        // The second one matters because it is the direction that loses data.
        // The check above trusts `grid_axes` to say an axis IS read; a false
        // negative there is a defect this test would excuse. It does not
        // excuse one today — `kv_append`'s rule gives it work on z, so the
        // flat-axis half never looks at it — but the exposure is real and is
        // written down rather than discovered later.
        //
        // So this is printed and not asserted. An assertion would either be
        // wrong (the three router rows are correct) or would need an exception
        // list that hides the `kv_append` finding inside it.
        if !wasted.is_empty() {
            eprintln!(
                "rows given work on an axis `grid_axes` says they never read \
                 ({} of them). Two causes, and only one is a row's fault -- see \
                 the note above this print:\n  {}",
                wasted.len(),
                wasted.join("\n  "),
            );
        }

        let stale: Vec<&str> = known
            .iter()
            .filter(|k| !symbols.contains(*k))
            .copied()
            .collect();
        assert!(
            stale.is_empty(),
            "these are listed as known-defective and are not: {stale:?}. If the \
             shared table was fixed, delete the entry in the same diff.",
        );
    }

    /// A routed matvec covers every output row its module owns per y lane.
    ///
    /// The sibling of `a_decode_attention_module_is_half_the_head_it_serves`,
    /// and it exists because that check's absence here cost a four-fold
    /// undershoot that nothing else could see.
    ///
    /// `Rule::RoutedQmv`'s y extent used to be `width / 4`, carried across
    /// from Metal, where the threadgroup is `[32, 2, 1]` and each y thread
    /// genuinely owns four output rows. `moe/qmv_routed.wgsl` is
    /// `@workgroup_size(32, 8)` and computes
    /// `out_row = workgroup_id.y * 8 + local_invocation_id.y` — ONE row per y
    /// lane. `groups()` then divided the already-quartered extent by the
    /// module's own 8, so a dispatch launched `ceil(n / 32)` workgroups where
    /// `ceil(n / 8)` were needed and seven eighths of every routed projection
    /// kept the arena's stale bytes.
    ///
    /// `no_rule_launches_fewer_lanes_than_its_extent` cannot catch that, and
    /// the reason is worth stating because it is a general blindness:
    /// `lanes()` is both the claim and the reference there, so a rule that
    /// lies agrees with itself. What catches it is comparing the rule against
    /// something computed somewhere else — here, the `@workgroup_size` `naga`
    /// reads out of the module the driver actually selected.
    ///
    /// The claim is exact rather than a bound: `groups()`'s y must be
    /// `ceil(width / local_y)`, so that every output row belongs to some
    /// invocation and no row belongs to two. An UNDERSHOOT writes nothing and
    /// returns success; an overshoot is harmless only because the body guards
    /// `out_row < out_vec_size`, and that guard is not this file's to assume.
    #[test]
    fn a_routed_matvec_covers_every_output_row_its_module_owns() {
        let mut checked = 0;
        for sig in kernels_wgpu::KERNELS {
            if sig.launch != Rule::RoutedQmv {
                continue;
            }
            for name in sig.entrypoints() {
                let Ok(declared) =
                    crate::reflect::entrypoint(&name, kernels_wgpu::Capability::Baseline)
                else {
                    continue;
                };
                let module = Module::loaded(&name, &declared);
                let lanes_y = module.local.at(1);
                assert!(
                    lanes_y >= 1,
                    "`{name}` declares a workgroup of {:?}, which owns no output row",
                    declared.local
                );

                // A width that is NOT a multiple of the module's y, so the
                // round-up is a different expression from the division. At a
                // multiple the two agree and this check proves nothing --
                // which is how the defect survived a suite that used them.
                for width in [13u32, 47, 1] {
                    let d = Dims { width, ..dims() };
                    let got = groups(Rule::RoutedQmv, d, module)
                        .unwrap_or_else(|e| panic!("`{name}` at width {width}: {e}"));
                    assert_eq!(
                        got[1],
                        width.div_ceil(lanes_y),
                        "`{name}` at width {width}: {} workgroups on y for a \
                         module owning {lanes_y} output rows each. Every row \
                         past {} is never written, and the dispatch succeeds.",
                        got[1],
                        got[1] * lanes_y,
                    );
                }
                checked += 1;
            }
        }
        assert_eq!(
            checked, 3,
            "the table states three routed-matvec rows, one entrypoint each, \
             and every one is measured against its own module -- a floor here \
             would let the sweep shrink without saying so"
        );
    }

    /// A rope over a key tensor covers the KEY heads, not the query heads.
    ///
    /// The regression `driver-metal`'s `rope_heads` is written against, held
    /// here too because the three shells compute it separately.
    #[test]
    fn rope_reads_its_head_count_off_the_tensor_it_rotates() {
        let head_dim = 128;
        let q = Dims {
            width: 32 * head_dim,
            head_dim,
            ..dims()
        };
        let k = Dims {
            width: 8 * head_dim,
            head_dim,
            ..dims()
        };
        let m = Module::new([64, 1, 1]);
        assert_eq!(groups(Rule::Rope, q, m).unwrap()[1], 32);
        assert_eq!(groups(Rule::Rope, k, m).unwrap()[1], 8);
    }

    /// A width that is not a whole number of heads refuses.
    #[test]
    fn an_unheaded_width_is_refused_rather_than_rounded() {
        let bad = Dims {
            width: 100,
            head_dim: 128,
            ..dims()
        };
        assert_eq!(
            groups(Rule::Rope, bad, Module::new([64, 1, 1])),
            Err(Ungeometric::Unheaded {
                width: 100,
                head_dim: 128
            })
        );
    }

    /// A tiled name under an unstated rule says UNSTATED, not `Untiled`.
    ///
    /// The precedence regression. `affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_16`
    /// is a real entrypoint of a real row: its name carries a tile and its row
    /// states no launch rule. Both refusals are true and only one is useful --
    /// `Untiled` sends the reader to `kernels-wgpu/src/axes.rs` to find a
    /// naming defect that is not there, `Unstated` sends them to the row's
    /// `launch` field, which is what is missing.
    ///
    /// Checking the tile first, which is what `driver-vulkan` does, gave the
    /// wrong one for 242 of this table's 292 unstated entrypoints.
    #[test]
    fn a_tiled_name_under_an_unstated_rule_names_the_rule() {
        let m = Module::named(
            "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_16",
            [32, 2, 2],
        );
        assert!(m.tile.is_some(), "this name was chosen for carrying a tile");
        assert_eq!(
            groups(Rule::Unstated, dims(), m),
            Err(Ungeometric::Unstated),
            "the missing launch rule is the repair, not the tile"
        );
        // A rule this backend does not serve wins over the tile too, and for
        // the same reason: writing a shader is the repair, not renaming.
        assert_eq!(
            groups(Rule::RecurrentScan, dims(), m),
            Err(Ungeometric::Unruled(Rule::RecurrentScan))
        );
        // And the check still fires where it is real: a SERVED non-GEMM rule
        // whose module name carries a tile is drift between this file and the
        // axis table, and it still says so.
        assert_eq!(
            groups(Rule::Elementwise, dims(), m),
            Err(Ungeometric::Untiled)
        );
    }

    /// A GEMM at a row count no tile divides refuses rather than substituting.
    #[test]
    fn a_partial_tile_refuses() {
        let ragged = Dims { rows: 13, ..dims() };
        let m = Module::named("affine_qmm_t_bfloat16_bm_16_bn_16", [32, 2, 2]);
        assert_eq!(
            groups(Rule::Qmm, ragged, m),
            Err(Ungeometric::PartialTile { rows: 13, tile: 16 })
        );
    }

    /// A rectangle of no rows still launches one.
    ///
    /// Zero rows would multiply a grid to nothing, and a dispatch of no
    /// workgroups runs nothing and reports success.
    #[test]
    fn a_rectangle_of_no_rows_still_launches_one() {
        let none = Dims { rows: 0, ..dims() };
        for &rule in kernels::LaunchRule::ALL {
            let Ok(g) = groups(
                rule,
                none,
                Module {
                    local: Local([256, 1, 1]),
                    tile: (rule == Rule::Qmm).then_some(Tile { rows: 1, cols: 1 }),
                },
            ) else {
                continue;
            };
            assert!(
                g.iter().all(|n| *n >= 1),
                "{rule:?} launched {g:?} workgroups at zero rows"
            );
        }
    }

    /// An unstated row is an error and not an empty dispatch.
    #[test]
    fn an_unstated_rule_refuses() {
        assert_eq!(
            groups(Rule::Unstated, dims(), Module::new([256, 1, 1])),
            Err(Ungeometric::Unstated)
        );
    }

    /// All three workgroup axes are read.
    ///
    /// Stated on its own because the sweeps above do NOT catch a shell that
    /// reads only the first component of `@workgroup_size` and treats the rest
    /// as 1: that mistake OVERSHOOTS, every shader guards its own tail, and the
    /// answer is merely 16x more workgroups than the work needs. It is a real
    /// defect and an invisible one, and `(32, 2, 2)` and `(16, 16, 1)` are most
    /// of this table.
    #[test]
    fn a_workgroup_is_read_on_every_axis() {
        let l = Local([32, 2, 2]);
        assert_eq!([l.at(0), l.at(1), l.at(2)], [32, 2, 2]);
        // And the division uses them: 4100 rows over a (16, 16, 1) module is
        // 257 workgroups on y, not 4100 -- and 257 rather than 256 because
        // 4100 is not a multiple of 16, which is the whole reason the dims
        // above are not round.
        let d = Dims {
            rows: 4100,
            width: 16,
            ..dims()
        };
        let g = groups(Rule::ElementwiseRows, d, Module::new([16, 16, 1])).unwrap();
        assert_eq!(g, [1, 257, 1]);
    }

    /// A zero workgroup width cannot divide by zero.
    #[test]
    fn a_zero_workgroup_does_not_divide_by_zero() {
        assert_eq!(Local([0, 0, 0]).at(0), 1);
        let g = groups(Rule::Elementwise, dims(), Module::new([0, 0, 0])).unwrap();
        assert_eq!(g[0], 4224 * 32);
    }

    /// A grid past what the device takes is NAMED, on the axis that is over.
    ///
    /// WebGPU's own limit and this backend's one addition to the rules. The
    /// shape is a real one: an elementwise launch over a 151936-wide vocabulary
    /// at 128 rows is 75968 workgroups at 256 lanes wide, past the guaranteed
    /// 65535 -- so this is a rectangle a real fire can state and a browser can
    /// refuse, and the refusal has to arrive with the launch in hand rather
    /// than as a `wgpu` validation message about a number.
    #[test]
    fn a_grid_past_the_devices_limit_is_refused_by_name() {
        let vocab = Dims {
            rows: 128,
            width: 151_936,
            ..dims()
        };
        let m = Module::new([256, 1, 1]);
        let want = 151_936u32 * 128 / 256;
        // The arithmetic still answers: the rule is not wrong, the device is
        // small. Keeping these two apart is why `groups_within` is a separate
        // entry point.
        assert_eq!(groups(Rule::Elementwise, vocab, m).unwrap()[0], want);
        assert_eq!(
            groups_within(Rule::Elementwise, vocab, m, MAX_WORKGROUPS_PER_DIMENSION),
            Err(Ungeometric::PastDeviceLimit {
                axis: 0,
                groups: want,
                limit: MAX_WORKGROUPS_PER_DIMENSION,
            })
        );
        // A device that reports more takes the same grid, which is what makes
        // the limit a parameter rather than a constant.
        assert!(groups_within(Rule::Elementwise, vocab, m, want).is_ok());
    }

    /// And an ordinary fire is nowhere near it.
    ///
    /// The control for the test above: if the limit refused a decode step, the
    /// check would be a wall rather than a guard and somebody would raise it
    /// without reading why.
    #[test]
    fn a_decode_step_is_well_inside_the_guaranteed_limit() {
        let decode = Dims { rows: 1, ..dims() };
        for &rule in kernels::LaunchRule::ALL {
            let m = Module {
                local: Local([256, 1, 1]),
                tile: (rule == Rule::Qmm).then_some(Tile { rows: 1, cols: 1 }),
            };
            let Ok(g) = groups(rule, decode, m) else {
                continue;
            };
            assert!(
                g.iter().all(|n| *n <= MAX_WORKGROUPS_PER_DIMENSION),
                "{rule:?} wants {g:?} workgroups for one row"
            );
        }
    }
}
