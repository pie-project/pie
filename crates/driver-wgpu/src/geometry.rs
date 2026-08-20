//! What a launch rule means when the workgroup is not the driver's to choose.
//!
//! `dispatch_workgroups(x, y, z)` takes a count of WORKGROUPS; the size of one
//! is `@workgroup_size(...)` in the WGSL, fixed when the module was written.
//! The Metal shapes in `lowering/grid.rs` remain the reference for the
//! iteration space a kernel must cover, but the answer here is a different kind
//! of number. The arithmetic below is `driver-vulkan`'s, because `vkCmdDispatch`
//! counts the same things and a grid that differed between two shells would
//! mean one of them is wrong.
//!
//! One thing is this API's: [`Ungeometric::PastDeviceLimit`]. WebGPU's
//! `max_compute_workgroups_per_dimension` has a guaranteed floor of **65535**,
//! which a 4096-wide elementwise launch over 32 rows reaches, and `wgpu`
//! answers a dispatch past it by refusing the encode. [`groups_within`] turns
//! that into a sentence instead of a panic at submit.
//!
//! [`groups`] rounds up everywhere, because an undershot grid reports nothing:
//! the lanes never launched write nothing, the gap reads back as whatever the
//! buffer was born with, and the queue returns success. Any test of it must use
//! a shape that does NOT divide evenly — at 512 elements over a 256-wide
//! workgroup, `div_ceil` and plain division are the same expression. The tests
//! below run at 460 and at 13.
//!
//! Each rule takes the `local` size read from the module it is about to
//! dispatch rather than assuming one. `SdpaVector` compiles one module per head
//! dimension, each declaring `@workgroup_size(PIE_HEAD_DIM / 2)`; `Elementwise`
//! is 256 wide in most of its modules and 16x16 in the strided gathers.
//!
//! **That `/ 2` is the one place a launch rule's ARITHMETIC differs from
//! `driver-vulkan`'s**: WGSL has no 16-bit storage type, so every bf16 tensor
//! crosses as `array<u32>` with two values to a word, and a decode-attention
//! lane owns the PAIR because there is no sub-word atomic to make sharing one
//! safe. [`lanes`] halves with it. `sdpa_vector.wgsl` also reads
//! `num_workgroups.x` as its query-head COUNT, so the Vulkan expression would
//! both build a grid twice as wide and tell every lane the model has twice the
//! heads it has.
//!
//! Every `Qmm` entrypoint names its tile IN THE ENTRYPOINT — `..._bm_16_bn_64`
//! — so the tile is a property of the module the driver selected, never
//! inferred from the row count. [`Module::loaded`] checks the agreement against
//! a [`crate::reflect::Declared`], which is `naga` reading the WGSL that will be
//! dispatched, on any machine, with no adapter.

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
    /// attentions, gemma-4's alt-up and its axial rope. Forty-two variants, of
    /// which one is [`Self::Unstated`]; of the forty-one real ones this backend
    /// has geometry for seventeen and refuses twenty-four. A deployment whose
    /// text reaches one of those twenty-four is asking for a block that was
    /// never ported, and the useful answer names WHICH.
    ///
    /// Both counts are asserted in `tests/rules.rs` and neither is asserted
    /// here, which is why this sentence said "thirty-seven ... fifteen ...
    /// twenty-one" long after the arm had grown: the fleet added rules and the
    /// test moved with them while the prose did not.
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
    /// `driver-vulkan` refuses the same set, which is the answer worth having
    /// rather than a coincidence to note: the two tables are both
    /// `kernels-metal`'s coverage, so the two backends should be unable to run
    /// exactly the same blocks, and a divergence would mean one of the ports
    /// quietly grew or lost a row. `tests/rules.rs` holds the two ledgers
    /// against each other rather than leaving that to a reading — and it is
    /// where the COUNT lives, too. It was written here as well, as
    /// "twenty-one", and stayed that way while the arm grew to twenty-four:
    /// a number repeated away from the assertion that owns it is a number
    /// nobody updates.
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
    ///
    /// # Where the shape came from, and who fixed it
    ///
    /// Not the driver, and not by choosing a different kernel here — that is
    /// the choice this refusal exists to push back to the text, and the text
    /// makes it: `llama_like`'s projection is a `guarded_value` whose arms are
    /// the GEMM and the GEMV.
    ///
    /// Its predicate used to be `GuardPred::TokensGT(tile - 1)`, which asks
    /// whether there are ENOUGH rows where the kernel needs a WHOLE NUMBER OF
    /// TILES, so every count above the tile that the tile did not divide —
    /// fifteen in sixteen — was admitted to an arm that could not run, and a
    /// 35-token prompt reached this variant on Metal, Vulkan and wgpu alike.
    /// `GuardPred::TokensMultipleOf(tile)` is the predicate that guard wanted,
    /// and it is what the text states now.
    ///
    /// So this refusal should be UNREACHABLE from a stated text today. It is
    /// kept, and kept refusing, because it is the only thing standing between
    /// a mis-stated guard and a fire that runs on a tiling that is not there.
    /// `driver-wgpu/tests/arena.rs::every_row_count_the_guard_sends_to_the_gemm_has_a_grid`
    /// sweeps every row count up to four tiles and requires the text's arm and
    /// this file's answer to agree on each.
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
                write!(
                    f,
                    "{rows} rows is not a whole number of {tile}-row tiles \
                     (a stated text guards this arm with \
                     `GuardPred::TokensMultipleOf({tile})`, so reaching this \
                     means some text does not)"
                )
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
            // width is the factor, and exact on y because the row count is
            // multiplied by the local size there before being divided by it.
            //
            // That y factor is not always one. `sdpa_paged.wgsl`'s decode arm
            // spends its second local axis on KEYS -- the block of the history
            // one reduction serves -- so a workgroup is `PIE_PAIRS x PIE_KB`
            // and the row still gets exactly one of them.
            [
                module.local.at(0) * dims.q_heads,
                rows * module.local.at(1),
                1,
            ]
        }
        // The two rules Metal distinguishes and this backend cannot.
        //
        // On Metal they are genuinely different launches: the tiled kernel
        // gives a query row to a SIMDGROUP and walks the scores as scalar dot
        // products, so a 32-row tile is 32 simdgroups and 1024 threads, while
        // the MMA kernel multiplies 8x8 fragments on the matrix unit where one
        // simdgroup owns EIGHT rows -- the same tile in 128 threads, declared
        // on the shader with `max_total_threads_per_threadgroup(128)`. That is
        // why `kernels::LaunchRule` has two variants and not a parameter.
        //
        // WGSL has no matrix unit to distinguish them WITH.
        // `wgpu::Features::SUBGROUP_MATRIX` exists and this crate has no tier
        // for it, deliberately -- `capability.rs` offers Baseline, Fp16 and
        // Subgroup, and a tier is a promise about a BODY. So
        // `attn/sdpa_paged_mma.wgsl` is a scalar body wearing Metal's
        // entrypoint names, `@workgroup_size(32, 8)` over a 32-row tile, which
        // is `attn/sdpa_paged.wgsl`'s tiled arm exactly.
        //
        // The arm is therefore shared, and nothing here asserts the two
        // shaders agree: the extents below are LANES, read from each module's
        // own `local`, so if the MMA body were ever rewritten at another width
        // this keeps returning that width's grid. What the two rules share is
        // the DECOMPOSITION -- tiles of 32 query rows on y, heads on x -- and
        // that is what is written here.
        Rule::SdpaTiled | Rule::SdpaMma => {
            // TILES of query rows on y, where `SdpaVector` puts rows.
            //
            // The tile height is the SHADER's, and this backend's is not
            // Metal's. `attn/sdpa_paged.wgsl`'s tiled arm is
            // `@workgroup_size(32, 8)` and its body sweeps `for (var rr =
            // lid.y; rr < 32u; rr = rr + 8u)`, so a GROUP covers 32 rows with 8
            // y lanes -- four rows per lane -- while Metal runs one 1024-thread
            // threadgroup per tile. Reading the height off `local.at(1)` would
            // give 8 and overshoot the grid four-fold; reading it off the
            // module at all would be reading a number the shader does not
            // publish.
            //
            // So it is stated here, once, next to the reason: `PIE_QT` in the
            // shader. A `//#define` would let the two drift with nothing to
            // notice, and the shader's own `rr < 32u` is the literal this must
            // agree with.
            const TILE: u32 = 32;
            // Rounded UP, which is the one place in this file that rounding a
            // grid is correct rather than a defect. `sdpa_paged.wgsl` takes the
            // true row count as `params.n_rows` and its rows loop skips
            // `row >= n_rows`, so a partial last tile knows it is past the end
            // -- and the row states `n_rows <- Source::Named(<keys::Rows as keys::Fact>::KEY)` so that it gets
            // told. Every other rounded axis in this file is refused because
            // the shader has no such scalar.
            let tiles = rows.div_ceil(TILE).max(1);
            // LANES, not workgroups. `groups` divides what this returns by the
            // module's own `@workgroup_size`, so a rule that returned the
            // group count would be divided a second time -- and the tiled arm
            // is `(32, 8)`, so `[q_heads, tiles, 1]` becomes
            // `[ceil(q_heads/32), ceil(tiles/8), 1]`: one group on each axis
            // for any fire this model produces, and every head past the first
            // and every tile past the eighth never written.
            //
            // That is the four-fold undershoot this file's `RoutedQmv` comment
            // describes, in a new rule, and it is what
            // `no_module_reads_a_grid_axis_its_rule_leaves_flat` reported
            // within a minute of the rule being written -- by name, on axis 1.
            //
            // So both axes are multiplied back up by the module's own local
            // size. Both divisions are then exact, which is the property every
            // other served rule here has.
            [
                module.local.at(0) * dims.q_heads,
                module.local.at(1) * tiles,
                1,
            ]
        }
        // THE ROW AXIS, which this stated as a literal 1 until `kernels-metal`
        // crossed `attn` and found it.
        //
        // `gate.metal`'s header says `grid = (n_q * head_dim, rows, 1)` and
        // its body indexes `tgpos.y`; the rule built `(q_heads * head_dim, 1,
        // 1)`, so a 512-token prefill gated token zero and left the other 511
        // with an ungated attention output. A decode is one row and looked
        // correct throughout -- the same reason `RouterLane`'s mixture prefill
        // and `GatedRms`'s hid for as long as they did. Third of the three.
        //
        // **No row in this tree names this rule today**: `attn/gate.wgsl`'s
        // two entrypoints sit behind `kernel!(gate ...)` and
        // `kernel!(q_gate_split ...)`, both of which state no `launch`, so
        // `Rule::Unstated` refuses them BY NAME rather than launching them
        // short. That is why wgpu never computed the wrong answer here and
        // also why nothing here would have caught it: the arm is a trap laid
        // for whoever fills those rows, and this is the trap removed rather
        // than a defect repaired.
        //
        // Stated as metal's corrected header does. wgpu's own shader reads
        // three axes for `gate_bfloat16` (`gid.x` the channel pair, `gid.y`
        // the head, `gid.z` the row) and two for `q_gate_split_bfloat16`, so
        // whoever fills those rows owes this arm another look -- one rule
        // cannot serve both shapes, exactly as `RouterSort` had to split from
        // `RouterLane`.
        Rule::PerHeadElementwise => [dims.q_heads * dims.head_dim, rows, 1],
        // THE ROW AXIS IS ON Z, and it was a literal 1.
        //
        // `norm/gated_rms.wgsl` indexes it both ways it can be built: the
        // strided arm takes `wg.z * strided.row_pitch + wg.y * vd` with `.z`
        // the TOKEN, and the dense arm takes `(wg.z * grid.y + wg.y) * vd`,
        // which is a row-major fold whose outer term is the row. With `z = 1`
        // every workgroup reads `wg.z == 0`, so a fire normalizes its FIRST
        // row and leaves every other one exactly as the projection wrote it --
        // fully written and only partly normalized, so nothing downstream can
        // report it.
        //
        // A decode is one row, which is why nothing here saw it.
        // `kernels-vulkan` found the same missing axis while crossing `norm`,
        // and `LaunchRule::RouterLane`'s own doc records the identical finding
        // about the identical mistake -- "with `grid.y = 1` a mixture prefill
        // routed row 0 only" -- which is already fixed one line below this.
        // Third time for this shape in this file's vocabulary.
        Rule::GatedRms => [dims.head_dim, dims.kv_heads, rows],

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
        //
        // The y extent is `axis` and not `width`, which is the ROW's own
        // statement rather than this file's reading of the rectangle. A routed
        // projection writes a whole token's `k` results end to end, so the
        // output is `k` times as wide as one result and `width` is `k *
        // out_vec_size`; `grid_param = Some(1)` on all three routed matvecs
        // names the second word, which is `out_vec_size` itself. `axis` falls
        // back to `width` for a row that states nothing, so this is the same
        // answer wherever the two agree. `driver-metal` reads `dims.axis`
        // here too.
        Rule::RoutedQmv => [
            module.local.at(0) * rows,
            dims.axis.max(1),
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
    use crate::lowering::hold::Facts;
    use crate::lowering::routine::Stated;

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

    /// One statement, wide enough that no arm in the tree runs out of operands.
    ///
    /// The three sweeps below drive the ROUTINE path, because that is where a
    /// grid comes from now: `kernels_wgpu::KERNELS` is empty, no symbol
    /// carries a `LaunchRule` any more, and the extent a launch actually gets
    /// is the one a body computed and handed to `ctx.dispatch`. To read that
    /// extent something has to feed the arm, and this is the feed.
    ///
    /// Nothing here is measured. No sweep below looks at an operand's address,
    /// its width or its bytes -- what is measured is the extent a body
    /// computes and the module it names -- so the statement only has to be
    /// long enough and generous enough that an arm refuses for a reason that
    /// belongs to the BODY: twelve arena operands where the widest arm binds
    /// five, six weights where the widest names four.
    ///
    /// That is what lets a sweep skip a refusal and still mean something. A
    /// refusal here is "this body is not compiled for that head width" or
    /// "that scalar was not stated", never "the fixture was too short".
    fn statement() -> Vec<model_compiler::lower::Arg> {
        statement_wide(128)
    }

    /// [`statement`] whose operands are `width` wide.
    ///
    /// THE RESULT'S WIDTH IS THE STATEMENT'S NOW. A routed matvec reads its
    /// output width off `y.width` -- the rectangle the text gave the operand
    /// -- where the driver used to hand it `out_vec_size` as a fire number. A
    /// sweep over output widths therefore has to vary the RECTANGLE, not just
    /// the scalar run, or the body answers 128 whatever the case says.
    fn statement_wide(width: u32) -> Vec<model_compiler::lower::Arg> {
        let mut args: Vec<model_compiler::lower::Arg> = (0..12usize)
            .map(|n| model_compiler::lower::Arg::Arena {
                at: n * 256,
                width,
                bytes: 2,
            })
            .collect();
        args.extend((0..6).map(|n| model_compiler::lower::Arg::Weight(format!("w.{n}"))));
        args
    }

    /// The fire a sweep states, as the `Env` values a body reads.
    fn firing(symbol: &str, rows: u32, width: u32, fire: crate::dispatch::Geometry) -> Facts {
        crate::lowering::hold::facts(symbol, rows, fire, 3, width, width)
    }

    /// What one body asks for, run through the arm that feeds it.
    ///
    /// [`crate::lowering::routine::plan`] minus the binding: the arm finds the
    /// operands a symbol's routine wants, the body states its module, its
    /// entrypoint and its LANES. Only the last is geometry, and it is the half
    /// a row's `LaunchRule` used to state -- so this is the seam a claim about
    /// a grid has to be made at now, and every rule this file names is checked
    /// against a body that reached the same shape by its own arithmetic.
    ///
    /// A body may state MORE than one dispatch, so this answers all of them.
    fn fired(symbol: &str, facts: Facts, scalars: &[u32]) -> Result<Vec<Stated>, String> {
        fired_wide(symbol, facts, scalars, 128)
    }

    /// [`fired`], over a statement whose operands are `width` wide.
    fn fired_wide(
        symbol: &str,
        facts: Facts,
        scalars: &[u32],
        width: u32,
    ) -> Result<Vec<Stated>, String> {
        let routine = crate::lowering::routine::armed(symbol)
            .ok_or_else(|| format!("`{symbol}` is claimed by no armed stem"))?;
        let args = statement_wide(width);
        let mut handles = crate::lowering::hold::Handles::with_scalars(
            &args,
            crate::lowering::routine::results(routine),
            scalars,
        );
        let taken =
            crate::lowering::bind::bind(routine.args, routine.sources, &mut handles, facts)
                .map_err(|why| format!("the binder refused: {why:?}"))?;
        // ON A PLANNER THAT CAN ANSWER, because these bodies ask. A fact only
        // the fire can answer is no longer bound into `args` before the body
        // runs -- that is the whole of the marks migration -- so `state`'s
        // fire-less planner refuses every routine that reaches for one, and a
        // geometry sweep built on it measured nothing at all.
        let handles = core::cell::RefCell::new(handles);
        crate::lowering::routine::stating(routine, &taken, &handles, facts)
            .map_err(|why| why.to_string())
    }

    /// The widest extent any fire asks of each module, and how many census
    /// symbols answered no fire at all.
    ///
    /// THREE fires, because "flat" has to be a fact about a body's ARITHMETIC
    /// and not about one fixture's numbers: an axis is left flat only when
    /// every fire that reached the module left it at a single lane, so a body
    /// that flattens an axis because this run happens to make it one is not a
    /// finding. The scalar runs differ per WORD as well as per fire, since a
    /// body that divides one stated number by another would otherwise be free
    /// to answer one twice.
    ///
    /// Two of them were not enough, and the way they failed is the argument
    /// for the third. Both stated a word 1 larger than half their width, and
    /// `rms_strided_head_row` reads `width / stated(1)` as its head count and
    /// puts it on y -- so both answered ONE head, agreed with each other, and
    /// made a rule that flattens nothing look flat. Two fires that share an
    /// accident are one fire. The numbers here are chosen so the quotients are
    /// wide, and the merge keeps the WIDEST per axis, so adding a fire can
    /// only ever REMOVE a finding and never introduce one.
    ///
    /// Keyed by the entrypoint the BODY named, which is not always the symbol
    /// that was fired: several bodies pick their module by head width, so two
    /// symbols reach one module and a symbol can reach a module its own name
    /// does not spell. The symbol kept beside the extent is the first that
    /// reached it, for the failure message.
    ///
    /// A body that refuses a fire -- a head width it is not compiled for, a
    /// tiling it does not carry -- is skipped, and the count of the symbols
    /// that refused EVERY fire is returned rather than asserted: it measures
    /// the generosity of the fixture, which is not the caller's subject.
    fn widest() -> (
        std::collections::BTreeMap<String, ([u32; 3], String)>,
        usize,
    ) {
        let fires = [
            (
                7u32,
                128u32,
                crate::dispatch::Geometry {
                    q_heads: 4,
                    kv_heads: 2,
                    head_dim: 64,
                    rotary_dims: 64,
                    n_experts: 8,
                    experts_per_token: 2,
                    ..Default::default()
                },
                [128u32, 96, 64, 48, 32, 16, 8, 4, 2, 3, 5, 7],
            ),
            (
                13,
                256,
                crate::dispatch::Geometry {
                    q_heads: 6,
                    kv_heads: 3,
                    head_dim: 128,
                    rotary_dims: 64,
                    n_experts: 16,
                    experts_per_token: 4,
                    ..Default::default()
                },
                [256u32, 160, 96, 64, 4, 48, 24, 12, 6, 9, 15, 21],
            ),
            // THE WIDE FIRE, and its first two words are the whole point of
            // it. Several arms read a count as a QUOTIENT of two stated
            // numbers -- `rms_strided_head_row` takes `heads = width /
            // stated(1)`, and `norm::per_head_row` puts that head count on y
            // and nothing else -- so a run whose word 1 is more than half its
            // width answers ONE head by integer division, on a rule that
            // flattens nothing. Both fires above do exactly that (128/96 and
            // 256/160), which read as a flat y axis and was a fact about the
            // fixture.
            //
            // 512 over 32 is sixteen heads, and the axis states its true
            // extent. The merge below keeps the WIDEST answer per axis, so
            // adding a fire can only ever remove a finding: this cannot mask
            // an axis that is flat under the other two and is why the accident
            // is repaired by adding a fire rather than by editing one.
            (
                5,
                512,
                crate::dispatch::Geometry {
                    q_heads: 8,
                    kv_heads: 4,
                    head_dim: 128,
                    rotary_dims: 128,
                    n_experts: 32,
                    experts_per_token: 8,
                    ..Default::default()
                },
                [512u32, 32, 128, 96, 8, 64, 48, 24, 12, 10, 18, 30],
            ),
        ];

        // The accident above is a property of these numbers, so it is pinned
        // here rather than left to a reader to rediscover. Word 1 is read as a
        // divisor of the width by the whole norm family, and an axis derived
        // from a quotient that truncates to one is indistinguishable from an
        // axis a rule genuinely flattens. At least one fire must divide its
        // width more than once, or this sweep is blind to every per-head extent
        // in the fleet.
        assert!(
            fires
                .iter()
                .any(|(_, width, _, run)| run[1] > 0 && *width / run[1] > 1),
            "no fire states a word 1 that divides its width more than once, \
             so every per-head axis reads as flat by accident"
        );

        let mut seen: std::collections::BTreeMap<String, ([u32; 3], String)> =
            std::collections::BTreeMap::new();
        let mut unreached = 0usize;
        for symbol in kernels_wgpu::entrypoints() {
            let mut answered = false;
            for (rows, width, fire, scalars) in &fires {
                let Ok(stated) = fired(&symbol, firing(&symbol, *rows, *width, *fire), scalars)
                else {
                    continue;
                };
                answered = true;
                for one in stated {
                    let extent = one.lanes;
                    let at = seen
                        .entry(one.entrypoint)
                        .or_insert_with(|| ([0; 3], symbol.clone()));
                    for (widest, lane) in at.0.iter_mut().zip(extent) {
                        *widest = (*widest).max(lane);
                    }
                }
            }
            if !answered {
                unreached += 1;
            }
        }
        (seen, unreached)
    }

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
        Rule::SdpaTiled,
        Rule::SdpaMma,
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

    // RETIRED: THE TABLE IS EMPTY, so the walk has no row whose rule to read.
    //
    // It was `no_row_in_the_table_names_a_rule_this_backend_refuses`, and it
    // asserted the direction `the_rules_this_backend_serves_are_exactly_the_
    // ones_with_shaders` above cannot: that walk starts from the VOCABULARY
    // and proves every `kernels::LaunchRule` variant is either served with a
    // grid or refused by name, which says nothing about which of them anybody
    // actually states. This one started from the FLEET -- every row of
    // `kernels_wgpu::KERNELS`, all 99 of them -- and required each row's
    // `launch` to be `Rule::Unstated` or a member of `SERVED`. A row naming
    // any of the twenty-four unruled variants would compile, would plan, and
    // would then answer `Ungeometric::Unruled` at every fire it ever reached,
    // which is a worse failure than a build that stops: it arrives per launch,
    // at runtime, on a machine that already loaded the weights. What it was
    // written to catch was `kernels-wgpu` growing a mamba row -- `PerRow`,
    // `RecurrentScan`, `Slab` -- before this crate grew a grid for it.
    //
    // It BECAME BLIND, and it went blind QUIETLY, which is the part worth
    // recording. `for sig in KERNELS` over an empty table runs its body zero
    // times and the test passes; there was no floor here, no `assert!(checked
    // >= n)` of the kind `binding.rs`'s two retired walks carried, so nothing
    // in this file's output changed on the day its subject was deleted. It
    // did not start agreeing that no row names an unruled rule. There are no
    // rows to name one. A green run reported that and a reader would have
    // believed it, which is why it is written down here rather than left to
    // pass.
    //
    // Nothing inherits the claim WHOLE, because nothing states a rule any
    // more: a routine computes its own lanes and hands them to `ctx.dispatch`,
    // so there is no field left for a body to fill in wrongly and no
    // `Ungeometric::Unruled` on the path a plan takes. The half that survives
    // is the one about grids that cannot run, and it is now measured on real
    // launches instead of declarations -- `kernels-wgpu`'s
    // `tests/routines.rs::no_routine_dispatches_an_empty_grid` fires every
    // body and refuses a zero on any axis, and this file's
    // `no_module_reads_a_grid_axis_its_rule_leaves_flat` drives the census
    // through its arms and checks the extent each body computes against the
    // axes its module reads. Both catch a body that cannot be launched
    // correctly; neither can catch, or needs to, a body that names a rule
    // nobody serves.

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
                "SdpaMma/q_heads",
                "SdpaTiled/q_heads",
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
    /// the shaders rather than argued from them. Three numbers computed in
    /// three different places have to agree at a factor of two: the head width
    /// the fire states, the `_d_N` in the entrypoint the BODY named -- which
    /// is what SELECTS the module, so it is the head width the driver believes
    /// it asked for -- and the `@workgroup_size` `naga` reads out of the
    /// source that name expands to.
    ///
    /// This used to walk the rows whose `launch` was `Rule::SdpaVector`.
    /// `kernels_wgpu::KERNELS` is empty, so the population is now the five
    /// crossed bodies that dispatch through `attn`'s `vector_grid`, fired at
    /// every head width the tree compiles. A body picks its entrypoint by
    /// indexing a literal spelling table with the fire's head width and
    /// REFUSES a width it was not compiled for, so the eleven entrypoints
    /// pinned at the bottom are the whole decode fleet as the bodies can reach
    /// it -- and a body that stops naming one fails that list rather than
    /// quietly shrinking the sweep.
    ///
    /// This is the check that would catch the divergence being undone. A
    /// shader author who "fixed" `PIE_PAIRS` to `PIE_HEAD_DIM` would make
    /// every decode dispatch here fail on the workgroup; one who did it while
    /// somebody else removed the halving in `vector_grid` would make every
    /// decode launch twice the workgroups and read twice the head count off
    /// `num_workgroups.x`, which is a wrong answer and not a refusal.
    ///
    /// The extent is compared BOTH ways for that reason. `Rule::SdpaVector`
    /// and `vector_grid` compute the same three-axis shape from the same three
    /// facts in two crates that never call each other, so the day they stop
    /// agreeing one of them is a grid nothing else in this tree can tell from
    /// a right one.
    #[test]
    fn a_decode_attention_module_is_half_the_head_it_serves() {
        /// The bodies that dispatch a decode attention through `vector_grid`.
        const DECODE: [&str; 5] = [
            "sdpa_paged_decode",
            "sdpa_paged_decode_sink",
            "sdpa_vector_decode",
            "sdpa_vector_decode_swa",
            "sdpa_vector_decode_sink",
        ];
        /// Every head width any of them is compiled for, offered to all five.
        const WIDTHS: [u32; 4] = [64, 128, 256, 512];

        let (rows, q_heads) = (7u32, 4u32);
        let mut reached: Vec<String> = Vec::new();
        for stem in DECODE {
            for head_dim in WIDTHS {
                let symbol = format!("{stem}_bfloat16_d_{head_dim}");
                let fire = crate::dispatch::Geometry {
                    q_heads,
                    kv_heads: 2,
                    head_dim,
                    rotary_dims: head_dim,
                    n_experts: 0,
                    experts_per_token: 0,
                    ..Default::default()
                };
                // A width this body is not compiled for is a refusal and not a
                // gap: the pinned list below is what says which pairs were
                // supposed to answer.
                // THE WIDTH THIS CASE IS ABOUT, at the slot the routines read
                // it from. `head_dim` is `Const<i32>` at params[3] now -- it
                // was a fact the driver recovered from the SYMBOL -- so a run
                // of eights of 128 named the 128-wide module whatever width
                // the case swept.
                // `[n_kv_heads, scale, window, head_dim, q_heads]`, which is
                // the run these routines' `Const` marks claim, in order. It
                // was a flat run of 128s -- a head width of 128 whatever the
                // case swept, and a q-head count of 128 against the fire's
                // four.
                // THE TWO FAMILIES DECLARE DIFFERENT RUNS. `sdpa_paged_*`
                // takes `[n_kv_heads, scale, window, head_dim, q_heads]`;
                // `sdpa_vector_*` has no paging and takes `[scale, head_dim,
                // q_heads]`, so its width is at slot 1 and not 3. A flat run
                // of 128s named the 128-wide module whatever the case swept.
                let mut scalars = [128u32; 8];
                if stem.starts_with("sdpa_paged") {
                    scalars[0] = 2;
                    scalars[3] = head_dim;
                    scalars[4] = q_heads;
                } else if stem.contains("swa") || stem.contains("sink") {
                    // `[scale, window, head_dim, q_heads]`: the window stands
                    // between the scale and the width. `sinks` is a WEIGHT and
                    // takes no slot in the scalar run.
                    scalars[2] = head_dim;
                    scalars[3] = q_heads;
                } else {
                    scalars[1] = head_dim;
                    scalars[2] = q_heads;
                }
                let Ok(stated) = fired(&symbol, firing(&symbol, rows, 128, fire), &scalars) else {
                    continue;
                };
                assert_eq!(
                    stated.len(),
                    1,
                    "`{symbol}` is one decode dispatch, and it stated {}",
                    stated.len()
                );
                let name = stated[0].entrypoint.clone();
                let declared =
                    crate::reflect::entrypoint(&name, kernels_wgpu::Capability::Baseline)
                        .unwrap_or_else(|e| panic!("`{name}`, which `{symbol}`'s body named: {e}"));
                let module = Module::loaded(&name, &declared);
                assert_eq!(
                    suffix_value(&name, "_d_"),
                    Some(head_dim),
                    "`{symbol}` fired at a {head_dim}-wide head and its body \
                     named `{name}`, which selects another module"
                );
                assert_eq!(
                    module.local.at(0) * 2,
                    head_dim,
                    "`{name}` declares a workgroup of {:?} for a {head_dim}-wide head",
                    declared.local
                );

                // And the rule says the same thing from the other side: the
                // body's own extent, and `Rule::SdpaVector`'s, computed apart.
                let d = Dims {
                    rows,
                    q_heads,
                    head_dim,
                    ..dims()
                };
                assert_eq!(
                    stated[0].lanes,
                    lanes(Rule::SdpaVector, d, module).unwrap_or_else(|e| panic!("`{name}`: {e}")),
                    "`{name}`: the body's extent and `Rule::SdpaVector`'s are \
                     the same shape reached twice, and they have parted"
                );
                assert_eq!(
                    groups(Rule::SdpaVector, d, module).unwrap_or_else(|e| panic!("`{name}`: {e}")),
                    [q_heads, rows, 1],
                    "`{name}`: one workgroup per (head, row), exactly"
                );
                reached.push(name);
            }
        }

        reached.sort();
        assert_eq!(
            reached,
            [
                "sdpa_paged_decode_bfloat16_d_128",
                "sdpa_paged_decode_bfloat16_d_256",
                "sdpa_paged_decode_bfloat16_d_512",
                "sdpa_paged_decode_bfloat16_d_64",
                "sdpa_paged_decode_sink_bfloat16_d_64",
                "sdpa_vector_decode_bfloat16_d_128",
                "sdpa_vector_decode_bfloat16_d_256",
                "sdpa_vector_decode_bfloat16_d_64",
                "sdpa_vector_decode_sink_bfloat16_d_64",
                "sdpa_vector_decode_swa_bfloat16_d_256",
                "sdpa_vector_decode_swa_bfloat16_d_512",
            ],
            "the decode fleet as its bodies can name it -- a width that stops \
             answering is a module that stopped being reachable, and a width \
             that starts is one nothing has measured"
        );
    }

    /// No module reads a grid axis its body leaves flat.
    ///
    /// The third member of the family `a_decode_attention_module_is_half_the_
    /// head_it_serves` and `a_routed_matvec_covers_every_output_row_its_module_
    /// owns` belong to, and the one that generalises: instead of naming a rule
    /// and checking its arithmetic, it asks every SYMBOL in the census whether
    /// the shape its launch actually gets can carry the axes its module really
    /// reads.
    ///
    /// A launch that answers `[n, 1, 1]` promises the body indexes on x alone.
    /// If that body reads `global_invocation_id.y` as a row, every row but the
    /// first is never written — and an undershot grid writes nothing, the gap
    /// reads back as whatever the buffer held, and the dispatch returns
    /// success. There is no symptom until somebody compares numbers.
    ///
    /// That is the claim; the units it is made in are the subtle part, and the
    /// body below spells them out. A lane is a THREAD, `dispatch_workgroups`
    /// takes workgroups, and one lane on an axis whose workgroup is two wide
    /// is a global extent of two rather than of one. Reading `lanes[a] == 1`
    /// as "the shader sees index 0 there" is a statement about a one-wide
    /// workgroup and about nothing else.
    ///
    /// ## Where the shape comes from now
    ///
    /// It used to come from the row: `lanes(sig.launch, ..)` for every row in
    /// `kernels_wgpu::KERNELS`. The table is empty, so the extent is taken
    /// where a real launch takes it — the arm feeds the routine, the BODY
    /// computes its own lanes and names its own entrypoint, and this compares
    /// those two answers against each other. That is a stronger reading than
    /// the row was: the row said what the driver would ask for, the body says
    /// what it asks for, and the entrypoint keyed below is the module that
    /// will actually run rather than the one the symbol suggests.
    ///
    /// It is also a NARROWER population, and the narrowing is the honest part.
    /// A module no body names at any of [`widest`]'s three fires has no grid
    /// to be wrong about, so it is not measured here; that an entrypoint is
    /// reachable at all is `hold.rs`'s
    /// `every_entrypoint_is_claimed_by_the_stem_that_owns_it` and `reflect`'s
    /// own census sweep, not this one's.
    ///
    /// `crate::reflect` already computes which axes a module reads, and it has
    /// been computed and never asked. It is safe to ask because it errs in the
    /// right direction: where the call walk cannot follow a builtin into a
    /// helper it answers "every axis", so a module this check cannot read makes
    /// it STRICTER rather than vacuous.
    ///
    /// `kernels-wgpu`'s `a_flat_rows_shader_does_not_read_its_row_off_the_y_
    /// axis` is the same claim from the shader side, on the bodies that call
    /// `elementwise` and on the y axis only. This one is over every body, on
    /// all three axes, and against the module the body chose.
    ///
    /// ## The defect it caught, and how it was closed
    ///
    /// `geglu_tanh_strided`'s row states `LaunchRule::Elementwise`, which is
    /// `[width * rows, 1, 1]`. Its body alone among the five in
    /// `mlp/gated.wgsl` was `@workgroup_size(16, 16)` and read `gid.y` as the
    /// row, so one workgroup on y covered 16 rows and every row past 15 kept
    /// the bytes it was allocated with. Measured at 21 rows on a 4090: row 16
    /// held its sentinel and the dispatch succeeded. gemma's PLE reaches this
    /// with `rows` = the fire's token count, so any prefill past sixteen
    /// tokens was dropping most of its per-layer embeddings, silently.
    ///
    /// The obvious fix was a new rule -- `ElementwiseRows` is the shape that
    /// body wanted -- but the row is shared with `kernels-metal`, where a
    /// threadgroup is sized at dispatch and `Elementwise` is already right.
    /// Changing the row would have been three tables and three drivers to fix
    /// one body. Changing the BODY to match the rule it already states was
    /// local, so that is what happened: it is `@workgroup_size(256)` and
    /// `gid.x` is a flat element index. `kernels-vulkan`'s copy is still
    /// `local_size_y = 16` over the same rule and still has this.
    ///
    /// That defect is also the reason the exemption below is written with two
    /// conjuncts rather than one. `@workgroup_size(16, 16)` makes the global y
    /// extent SIXTEEN, so an exemption phrased as "the axis is not really
    /// flat" would have passed `geglu_tanh_strided` — the axis was never flat,
    /// and sixteen rows out of twenty-one is still a silent loss. Only a body
    /// that reads `@builtin(num_workgroups)`, and so builds its index out of
    /// the grid it was actually given, is excused.
    ///
    /// ## The defect the routine path caught, which the row path could not
    ///
    /// Re-anchored on real launches, this sweep reaches every entrypoint a
    /// body can name rather than the rows that stated a rule, and it named
    /// three: `gdn_prep_bfloat16`, `gdn_prep_prefill_bfloat16` and
    /// `gdn_prep_slotted_bfloat16`, all reading y against a grid stating one
    /// lane there. None of them reads `gid` at all — they index off `wid.z`
    /// and `lid` — and the report was still correct, because
    /// `ssm/gdn_prep.wgsl` declared `@builtin(global_invocation_id)` for every
    /// variant while only the recurrent arm read it. naga materialises an
    /// `Expression::FunctionArgument` for a declared argument whether or not
    /// anything uses it, so `reflect::axes_read` found a grid position it
    /// could not account for and returned its `[true; 3]` fallback: the
    /// entrypoints CLAIMED to read all three axes. The declaration was the
    /// lie, and the repair was to declare the builtin only where it is read.
    /// An unread builtin costs nothing at run time and is therefore invisible
    /// there; it is not invisible to reflection, which reads the signature and
    /// not the body.
    ///
    /// Recorded because the temptation was to exempt it here. An
    /// entrypoint whose `grid_axes` is `[true; 3]` is exactly the shape of
    /// `reflect`'s fallback, and skipping those would have made this quiet
    /// again — and would have blinded it to a real class: a module that reads
    /// three axes against a grid that fills one. The check was right to ask.
    ///
    /// ## And the accident it reported, which was the fixture's
    ///
    /// `rms_strided_head_row_bfloat16` came back on the same run and was not a
    /// defect of any kind. Its y extent is `norm::per_head_row`'s head count,
    /// which the arm reads as `width / stated(1)`, and both fires then in use
    /// stated a word 1 past half their width — so integer division answered
    /// one head twice and a rule that flattens nothing looked flat. That is
    /// the structural weakness of reading a real launch instead of a rule: a
    /// rule literals a `1` on the axes it flattens and cannot be talked out of
    /// it, while a launch's `1` may be arithmetic or may be an accident, and
    /// the two are not distinguishable from one sample. [`widest`]'s third
    /// fire is the repair, and its doc says why adding one is sound where
    /// editing one would not have been.
    ///
    /// The list below is consequently EMPTY, and the emptiness is asserted
    /// from both ends: an entry that stops being defective fails just as
    /// loudly as a module that starts being one. That is what said the fix had
    /// landed -- `these are listed as known-defective and are not` -- in the
    /// same run that proved it.
    #[test]
    fn no_module_reads_a_grid_axis_its_rule_leaves_flat() {
        // ENTRYPOINTS whose module reads an axis its body flattens -- the
        // entrypoint and not the symbol, because the entrypoint is what the
        // body named and what `naga` was read for. Each would be a defect,
        // each would be named, and the list is pinned so it cannot grow in
        // silence.
        const KNOWN: [(&str, &str); 0] = [];

        let (seen, unreached) = widest();

        let mut found: Vec<String> = Vec::new();
        let mut wasted: Vec<String> = Vec::new();
        let mut checked = 0usize;

        for (name, (extent, symbol)) in &seen {
            let declared = crate::reflect::entrypoint(name, kernels_wgpu::Capability::Baseline)
                .unwrap_or_else(|e| panic!("`{name}`, which `{symbol}`'s body named: {e}"));
            checked += 1;

            for (axis, lanes) in extent.iter().enumerate() {
                if *lanes > 1 || !declared.grid_axes[axis] {
                    continue;
                }
                // LANES ARE THREADS, and one lane is not one index. The global
                // extent on an axis is `lanes.div_ceil(local) * local` -- what
                // `routine::bind` computes and hands to `dispatch_workgroups`
                // -- so a single lane against a workgroup of two is an extent
                // of TWO, and a body reading `global_invocation_id.y` there
                // sees both values. Reading `lanes == 1` as "the shader only
                // ever sees index 0" is true of a one-wide workgroup and false
                // of every other.
                //
                // But a wider workgroup is not a defence either, and that
                // distinction is the whole of `geglu_tanh_strided` below:
                // `@workgroup_size(16, 16)` over `[width * rows, 1, 1]` is an
                // extent of SIXTEEN on y, never flat, and rows 16 and up were
                // still never written -- because the body read `gid.y` as a
                // row of a rectangle nothing had sized. Exempting on workgroup
                // width alone retires the defect this test exists for.
                //
                // What separates the two is whether the body reconstructs its
                // index from THE GRID IT WAS GIVEN.
                // `cast_qmm_input_bfloat16_to_float16` is `@workgroup_size(32,
                // 2, 2)` over `elementwise`'s `[count, 1, 1]`, reads `gid.y`,
                // and is correct: `idx = gid.x + gid.y * groups.x * 32u` makes
                // y the high digit of a flat index whose span is the
                // launcher's own, and `quant/qmm_t.wgsl` says why in as many
                // words -- *"the alternative is a flat index that disagrees
                // with the launcher's grid, which is a wrong answer"*. It
                // takes `@builtin(num_workgroups)` in order to agree.
                // `geglu_tanh_strided` did not, and could not have.
                //
                // So the exemption takes BOTH, and neither alone: a module
                // whose workgroup is one wide on the axis sees index 0 there
                // whatever arithmetic it does, and a module that never asks
                // how big the grid is cannot be folding it into an index.
                //
                // `reads_workgroup_count` is DECLARATION-shaped -- `reflect`
                // sets it from the entry point's `@builtin` arguments, not
                // from a use -- so a body that declares the builtin and
                // ignores it takes an exemption it has not earned. A real
                // hole, and the narrow direction to err in: the alternative is
                // failing the one body in this tree that does this correctly.
                //
                // The waste half below already recorded this units confusion
                // from the other side -- *"`lanes` is a THREAD extent and a
                // body that indexes by lane consumes it correctly"* -- and
                // that note was written while the data-loss half above still
                // read a lane as an index. The same category error, found
                // twice, fixed once.
                let local = declared.local[axis].max(1);
                if local > 1 && declared.reads_workgroup_count {
                    continue;
                }
                found.push(format!(
                    "{name} reads axis {axis}, given {lanes} lane over a \
                     workgroup of {local}, fired as `{symbol}`"
                ));
            }

            // The MIRROR, which is harmless and is checked anyway.
            //
            // `driver-vulkan/tests/rules.rs` has exactly this and only this —
            // `if !read && given > 1` — and it is why that crate did not catch
            // `geglu_tanh_strided`: work given to an axis nothing reads is
            // WASTE, and an axis the body reads that the launch leaves at one
            // is DATA LOSS. They are different predicates and only the first
            // was written.
            //
            // Kept because waste is still a disagreement between a grid and a
            // body, and a grid that hands out a dimension nobody uses is
            // usually a grid that meant to hand out a different one. It is
            // reported separately so the two never get confused for each other
            // again.
            //
            // LANES here too, and this arm has the units problem the loss arm
            // above just had -- but in the harmless direction, which is why it
            // is left as it is and not quietly changed. In WORKGROUP units the
            // predicate would be `lanes.div_ceil(local) > 1`, and that is the
            // form that would be assertable: `route_sort` and the two
            // `router_topk`s are `256` lanes over a `256`-wide workgroup, so
            // they are ONE workgroup and would stop being reported, while
            // `residual_add`'s two-dimensional grid against a `gid.x`-only
            // body is 64 workgroups on an axis nothing reads and would
            // survive. That is the whole of the note below, arithmetic rather
            // than an allow-list. It is not made here because promoting this
            // half from reported to asserted is a decision to take with a run
            // in hand, and nothing has run it in these units.
            for (axis, lanes) in extent.iter().enumerate() {
                if *lanes <= 1 || declared.grid_axes[axis] {
                    continue;
                }
                wasted.push(format!(
                    "{name} is given {lanes} on axis {axis}, which it never \
                     reads, fired as `{symbol}`"
                ));
            }
        }

        assert!(
            // A FLOOR and not a count. What it can be is bounded by the
            // fixture rather than by the tree: a body that refuses every one
            // of [`widest`]'s fires is not measured here, and tightening this
            // into a census would make it a test of how plausible those fires
            // are. The coverage is printed instead, so a sweep that quietly
            // stops reaching half the fleet is visible in the output.
            checked > 10,
            "only {checked} entrypoints were geometry-checked; a sweep that \
             read almost nothing passes as loudly as one that read everything"
        );
        eprintln!(
            "{checked} modules measured through their bodies; {unreached} \
             census symbols refused every fire"
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
            "these modules read a grid axis their body flattens, which means \
             every index past the first on that axis is never written and the \
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
        //   workgroup width, which is exactly what their grids intend. `Given
        //   256 lanes on x` is one workgroup, not 256 of them, and the grid is
        //   right. This half of the report is a category error in the check
        //   rather than in the body: `lanes` is a THREAD extent and a body
        //   that indexes by lane consumes it correctly.
        //
        // * `kv_append` reads only `gid.x` and `gid.y`, and `grid_axes` says
        //   so. This note used to record a FALSE NEGATIVE here, on the
        //   strength of a `gid.z` read straight out of `attn/kv_write.wgsl`.
        //   That `gid.z` belongs to the OTHER body in the same file: the two
        //   `kv_append` variants are one `//#if defined(PIE_PAGED)` apart, and
        //   the page index is the paged one's. The reflection was reading the
        //   compiled module and was right; the note was reading the file and
        //   was wrong.
        //
        //   `reflect::tests::two_entrypoints_of_one_file_are_told_apart_by_the_axis_they_read`
        //   now pins the pair, because a mistake made by reading a file is one
        //   the next reader of that file will make again.
        //
        // So what is printed below is the first cause only, and the check
        // above keeps trusting `grid_axes` to say an axis IS read -- a trust
        // that is now measured on the one pair in the tree built to test it.
        //
        // It is printed and not asserted because the modules it names are
        // RIGHT: asserting emptiness would fail on three that consume their
        // lanes correctly, and asserting an exception list would be a list of
        // three correct modules. What the print buys is that a FOURTH joining
        // them is visible in the output rather than silent.
        if !wasted.is_empty() {
            eprintln!(
                "modules given work on an axis `grid_axes` says they never \
                 read ({} of them). Expected for a body that indexes by lane \
                 over a workgroup -- see the note above this print:\n  {}",
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
             body or the shader was fixed, delete the entry in the same diff.",
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
    /// something computed somewhere else -- here, the `@workgroup_size` `naga`
    /// reads out of the module the driver actually selected, and the extent
    /// `moe`'s `routed_qmv_grid` computes for the same fire.
    ///
    /// The three rows this used to walk are gone with the rest of
    /// `kernels_wgpu::KERNELS`, and the three symbols below are what replaced
    /// them: the same three entrypoints, reached the way a plan reaches them.
    /// The row's `grid_param = Some(1)` -- "the y extent is the second scalar,
    /// which is `out_vec_size`, not the rectangle's width" -- is now the
    /// BODY's `routed_qmv_grid(rows, out_vec_size, slots)`, and it is read
    /// here off the same scalar run the shader's uniform block gets.
    ///
    /// The claim is exact rather than a bound: `groups()`'s y must be
    /// `ceil(out_vec_size / local_y)`, so that every output row belongs to
    /// some invocation and no row belongs to two. An UNDERSHOOT writes nothing
    /// and returns success; an overshoot is harmless only because the body
    /// guards `out_row < out_vec_size`, and that guard is not this file's to
    /// assume.
    #[test]
    fn a_routed_matvec_covers_every_output_row_its_module_owns() {
        /// Every symbol whose body dispatches through `moe::routed_qmv_grid`.
        const ROUTED: [&str; 3] = [
            "affine_qmv_routed_bfloat16_gs_64_b_4",
            "affine_qmv_routed_bias_bfloat16_gs_64_b_4",
            "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
        ];

        let (rows, slots) = (7u32, 2u32);
        let mut checked = 0;
        for symbol in ROUTED {
            // Output widths that are NOT multiples of the module's y, so the
            // round-up is a different expression from the division. At a
            // multiple the two agree and this check proves nothing -- which is
            // how the defect survived a suite that used them.
            for out_vec in [13u32, 47, 1] {
                // Word 1 is `out_vec_size` and word 4 is the slot count: the
                // same run the shader's uniform block gets, read by the arm
                // and handed to the body, which is the only path a y extent
                // takes now.
                // `[x_slot_stride, x_row_stride, slots_per_row]`, which is
                // the run this routine's three `Const` marks claim, in order.
                // It was `[128, out_vec, 128, 128, slots, ..]` -- the shader's
                // uniform block as the ARM once packed it, with the slot count
                // at word 4. The body reads slot 2 now, and `out_vec` is not a
                // scalar at all: it is `y.width`, the rectangle above.
                let scalars = [128, 128, slots, 0, 0, 0, 0, 0];
                let fire = crate::dispatch::Geometry {
                    q_heads: 4,
                    kv_heads: 2,
                    head_dim: 64,
                    rotary_dims: 64,
                    n_experts: 8,
                    experts_per_token: slots,
                    ..Default::default()
                };
                let stated =
                    fired_wide(symbol, firing(symbol, rows, out_vec, fire), &scalars, out_vec)
                        .unwrap_or_else(|e| {
                            panic!("`{symbol}` is a matvec this tree plans: {e}")
                        });
                assert_eq!(stated.len(), 1, "`{symbol}` is one dispatch");
                let name = stated[0].entrypoint.clone();
                let declared =
                    crate::reflect::entrypoint(&name, kernels_wgpu::Capability::Baseline)
                        .unwrap_or_else(|e| panic!("`{name}`, which `{symbol}`'s body named: {e}"));
                let module = Module::loaded(&name, &declared);
                let lanes_y = module.local.at(1);
                assert!(
                    // Not a formality: `Local::at` clamps to one, so a module
                    // that stopped tiling on y would make the round-up below
                    // the identity and this whole sweep would prove nothing
                    // while still passing. The premise is asserted so it
                    // cannot rot into an assumption.
                    lanes_y > 1,
                    "`{name}` declares a workgroup of {:?}, which owns ONE \
                     output row -- the round-up below is then an identity and \
                     measures nothing",
                    declared.local
                );

                // The ROW's own statement rather than the rectangle's width: a
                // routed projection writes a whole token's `k` results end to
                // end, so `width` is `k * out_vec_size` and a grid that read
                // the rectangle would launch `k` times over. `axis` is what
                // `grid_param = Some(1)` used to name and what the body reads
                // out of word 1.
                let d = Dims {
                    rows,
                    axis: out_vec,
                    width: out_vec * 4,
                    experts_per_token: slots,
                    ..dims()
                };
                assert_eq!(
                    stated[0].lanes,
                    lanes(Rule::RoutedQmv, d, module).unwrap_or_else(|e| panic!("`{name}`: {e}")),
                    "`{name}` at an output of {out_vec}: the body's extent and \
                     `Rule::RoutedQmv`'s are the same shape reached twice"
                );
                assert_eq!(
                    stated[0].lanes[1],
                    out_vec,
                    "`{name}`: the y extent is the OUTPUT WIDTH, and a grid \
                     that read the rectangle instead launches {} times over",
                    stated[0].lanes[1] / out_vec.max(1)
                );

                let got = groups(Rule::RoutedQmv, d, module)
                    .unwrap_or_else(|e| panic!("`{name}` at an output of {out_vec}: {e}"));
                assert_eq!(
                    got[1],
                    out_vec.div_ceil(lanes_y),
                    "`{name}` at an output of {out_vec}: {} workgroups on y for \
                     a module owning {lanes_y} output rows each. Every row past \
                     {} is never written, and the dispatch succeeds.",
                    got[1],
                    got[1] * lanes_y,
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked, 9,
            "three routed matvecs at three output widths each, every one \
             measured against its own module -- a floor here would let the \
             sweep shrink without saying so"
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
    /// Which served rules carry a ROW AXIS, stated once so that a missing one
    /// is a failing test rather than a silent third of a prefill.
    ///
    /// # The shape this exists for
    ///
    /// A grid axis the shader body indexes and the launch does not state.
    /// The output is FULLY WRITTEN — every row gets bytes — and only partly
    /// computed, so nothing downstream sees a hole and no golden gate over a
    /// one-row decode can notice. It has now happened three times in this
    /// vocabulary:
    ///
    /// * [`Rule::RouterLane`], whose own doc in `kernels` records it: "with
    ///   `grid.y = 1` a mixture prefill routed row 0 only, and every other
    ///   row's expert ids were whatever the last layer left there".
    /// * [`Rule::GatedRms`] here, found by `kernels-vulkan` crossing `norm`.
    ///   `norm/gated_rms.wgsl` reads `wg.z` as the token in both arms it can
    ///   be built as, and this file passed a literal `1`.
    /// * `rms_single_row` on vulkan, whose grid counts AXES rather than rows.
    ///
    /// # Why a ledger and not one assertion
    ///
    /// Row-independence is a real answer for some rules —
    /// [`Rule::RouterSort`] is one workgroup over a fire-wide histogram, and
    /// says so — so "every rule scales with rows" would be false. What can be
    /// checked is that each answer is DELIBERATE: a rule that stops scaling,
    /// or a new rule that never started, has to be written down here.
    #[test]
    fn every_served_rule_states_whether_it_has_a_row_axis() {
        // `true` where the grid must grow with the fire's rows.
        const ROW_AXIS: &[(Rule, bool)] = &[
            (Rule::Qmv, true),
            (Rule::Rms, true),
            (Rule::Rope, true),
            (Rule::Elementwise, true),
            (Rule::ElementwiseRows, true),
            (Rule::PerHead, true),
            (Rule::SdpaVector, true),
            (Rule::SdpaTiled, true),
            (Rule::SdpaMma, true),
            // WAS `false` here, on the assumption that the caller launched
            // it per row. `kernels-metal`'s attn crossing proved otherwise --
            // `gate.metal` indexes `tgpos.y` -- and the ledger is what made
            // that assumption a written claim somebody could refute. See the
            // arm in `lanes`.
            (Rule::PerHeadElementwise, true),
            (Rule::GatedRms, true),
            (Rule::RouterLane, true),
            // One workgroup whatever the rows -- the counting sort reduces
            // over the whole fire's histogram. Its own arm says so.
            (Rule::RouterSort, false),
            (Rule::RouteRows, true),
            (Rule::RoutedQmv, true),
            (Rule::SplitPacked, true),
            (Rule::Qmm, true),
        ];

        let named: Vec<Rule> = ROW_AXIS.iter().map(|(r, _)| *r).collect();
        assert_eq!(
            named,
            SERVED.to_vec(),
            "this ledger and `SERVED` have parted; a rule with no entry has \
             no stated answer about its row axis, which is the whole point"
        );

        for &(rule, wants_rows) in ROW_AXIS {
            let m = Module {
                local: Local(if rule == Rule::SdpaVector {
                    [64, 1, 1]
                } else {
                    [256, 1, 1]
                }),
                tile: (rule == Rule::Qmm).then_some(Tile { rows: 32, cols: 64 }),
            };
            let base = if rule == Rule::SdpaVector {
                Dims {
                    head_dim: 128,
                    ..dims()
                }
            } else {
                dims()
            };
            // THIRTY-TWO against SIXTY-FOUR, and both numbers are chosen.
            //
            // Not 1 against 8: the tiled rules count 16-row tiles, so both are
            // a single tile and `SdpaTiled` came back looking row-independent
            // -- a probe too coarse to see the thing it was built for, which
            // is the failure mode a ledger like this is most likely to have.
            //
            // Not 1 against 64 either: a GEMM refuses a partial tile, so
            // `Qmm` at one row is `PartialTile { rows: 1, tile: 32 }` and the
            // sweep panicked before it compared anything. Both counts are
            // whole tiles for every tile size in this file.
            let one = lanes(rule, Dims { rows: 32, ..base }, m).expect("thirty-two rows");
            let many = lanes(rule, Dims { rows: 64, ..base }, m).expect("sixty-four rows");
            let grew = one != many;
            assert_eq!(
                grew,
                wants_rows,
                "`{rule:?}` at 32 rows is {one:?} and at 64 is {many:?}; \
                 this ledger says its grid {} scale with rows. A rule whose \
                 SHADER indexes a row axis and whose launch states 1 writes \
                 every row and computes the first.",
                if wants_rows { "must" } else { "must not" }
            );
        }
    }

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
