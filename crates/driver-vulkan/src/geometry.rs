//! What a launch rule means when the workgroup is not the driver's to choose.
//!
//! `driver-metal` answers a [`Rule`] with a thread grid AND a threadgroup,
//! because `dispatchThreads` takes both and Metal sizes the group at dispatch.
//! `vkCmdDispatch` takes neither. It takes a count of WORKGROUPS, and the size
//! of one is `[numthreads(...)]` on the entrypoint — fixed when `slangc` ran,
//! months before any fire.
//!
//! So this module cannot be a port of `lowering/grid.rs`. The Metal shapes are
//! still the reference — they are the ones proved against real checkpoints, and
//! the iteration space a kernel must cover does not change with the API — but
//! the answer here is a different kind of number, and getting from one to the
//! other is where the whole class of porting bug in this tree lives.
//!
//! # Undershoot is the failure that does not report itself
//!
//! An overshot grid is caught by a shader's own tail guard. An UNDERSHOT grid
//! is not caught by anything: the lanes that were never launched write nothing,
//! the gap reads back as whatever the buffer was born with — zeros, from a
//! fresh pool — and `vkQueueSubmit` returns success. Every kernel in this tree
//! that was wrong after the Vulkan port was wrong this way, and none of them
//! were wrong in the arithmetic.
//!
//! That is why [`groups`] rounds up everywhere, and why the check that it does
//! is a sweep over the whole 480-entrypoint table in `tests/rules.rs` rather
//! than a comment.
//!
//! # The module's own size is an input
//!
//! Each rule below takes the `local` size read from the module it is about to
//! dispatch, rather than assuming one. Two rules need this and the rest are
//! merely honest about it:
//!
//! * `SdpaVector` compiles one module per head dimension — 64, 128, 256, 512 —
//!   and each declares `[numthreads(PIE_HEAD_DIM, 1, 1)]`. A geometry that assumed
//!   256 would launch a quarter of the workgroups for a 64-wide head.
//! * `Elementwise` is 256 wide in nineteen of its twenty modules and 16x16 in
//!   `geglu_tanh_strided`, which is laid out per (channel, row).
//!
//! The GEMM tile is the same kind of fact and was nearly missed. All 108 `Qmm`
//! entrypoints name their tile IN THE ENTRYPOINT -- `..._bm_16_bn_64` -- so the
//! tile a module was compiled for is a property of the module the driver
//! selected, not something to be inferred from the row count. Inferring it is
//! what the first draft of this file did, and picking the widest tile that
//! divides 64 rows while dispatching a module compiled at `bm_16` launches a
//! quarter of the workgroups needed. That undershoot writes three quarters of
//! nothing and returns success.
//!
//! Reading these from the module is also the only way the agreement can be
//! CHECKED, and `tests/rules.rs` checks it for all 480 entrypoints.

pub use kernels::LaunchRule as Rule;

/// The fire-time quantities a launch rule may read.
///
/// The same set `driver-metal`'s `lowering::launch::Dims` carries, deliberately
/// spelled the same way. The two shells disagree about the grid; they must not
/// disagree about which number is `axis` and which is `width`, and a reader
/// comparing the two files should find the difference where it really is.
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
/// Three axes because `local_size_y` and `local_size_z` are load-bearing here:
/// the GEMM tiles are `(32, 2, 2)` and the row-wise gathers are `(16, 16, 1)`,
/// so a shell that read only `x` would divide the wrong extent by the wrong
/// number and be right by accident on the eleven kernels that are `(n, 1, 1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Local(pub [u32; 3]);

/// The GEMM tile an `affine_qmm*` module was compiled for.
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
    /// the truth is: `kernels-vulkan/src/axes.rs` builds these variants by
    /// appending `_bm_N` and `_bn_N`, and the same suffix is what selects the
    /// `-D` that `slangc` compiled the tile from. A separate field could drift
    /// from the module; the suffix cannot.
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

/// What the module the driver is about to dispatch was COMPILED for.
///
/// Both fields are facts about the `.spv`, not decisions: `local` is read from
/// its `OpExecutionMode LocalSize` and `tile` from its entrypoint name. They
/// travel together because they are the same argument -- the numbers the
/// geometry divides by and does not get to pick.
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
    /// the module it loaded for it.
    ///
    /// Both numbers come from the thing being launched — the workgroup from
    /// the module's own `OpExecutionMode`, the tile from the name that
    /// selected it — so there is no table for either to drift from.
    #[must_use]
    pub fn loaded(name: &str, declared: &crate::spirv::Declared) -> Self {
        Self::named(name, declared.local)
    }
}

impl Local {
    /// The width of one workgroup along an axis, never zero.
    ///
    /// A zero would make the round-up below divide by zero, and a module that
    /// declared one could not be created in the first place — but this is the
    /// number a whole dispatch's shape is divided by, so it is worth not
    /// trusting.
    #[must_use]
    pub fn at(self, axis: usize) -> u32 {
        self.0[axis].max(1)
    }
}

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
    /// A GEMM module whose entrypoint names no tile, or a non-GEMM one that
    /// does. Drift between this file and `kernels-vulkan/src/axes.rs`, not a
    /// runtime condition.
    Untiled,
    /// A decode-attention module compiled for one head width, dispatched for
    /// a fire of another.
    ///
    /// Not an arithmetic condition -- the driver selected the wrong module.
    /// `sdpa_vector.slang` declares `[numthreads(PIE_HEAD_DIM, 1, 1)]`, so the
    /// module's own workgroup width IS the head width it was built for, and
    /// all 13 entrypoints agree with the `_d_N` in their name. That makes a
    /// mis-selection detectable HERE, before the dispatch, instead of showing
    /// up as an attention output that is subtly wrong.
    HeadMismatch {
        /// The head width the module was compiled for.
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
    /// A rule this backend compiles no module for.
    ///
    /// `kernels::LaunchRule` is the whole fleet's vocabulary and CUDA states
    /// rules for blocks nothing here implements -- mamba's scans, MLA's
    /// prepare, gemma-4's alt-up streams, the packed-head attentions. A text
    /// naming one is a model this backend does not serve, and the refusal is
    /// carried by the rule rather than by a shape.
    ///
    /// Named individually in [`lanes`]'s match rather than caught by a `_`
    /// arm: the compiler is what tells this file that the fleet grew a rule,
    /// and a wildcard would turn that into a silent `Unruled` at run time.
    /// It has already paid for itself once, when twenty-one rules arrived at
    /// once.
    Unruled(Rule),
}

impl core::fmt::Display for Ungeometric {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unstated => write!(f, "the row names no launch rule"),
            Self::Untiled => write!(f, "a GEMM module whose entrypoint names no tile"),
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
            Self::Unruled(rule) => {
                write!(f, "this backend compiles no module launched as {rule:?}")
            }
        }
    }
}

impl core::error::Error for Ungeometric {}

/// The invocations a rule has to cover, before workgroups are considered.
///
/// This is `driver-metal`'s grid, in threads, and it is kept separate from the
/// division on purpose: it is the part that is the KERNEL's contract and is
/// identical on both backends, so a difference between the two shells here
/// would be a real disagreement rather than an API one.
///
/// Public because it is the only way to ask whether [`groups`] rounded — which
/// for the 34 modules that read `gl_NumWorkGroups` is the difference between a
/// harmless spare workgroup and a changed answer.
///
/// # Errors
///
/// The same [`Ungeometric`] cases [`groups`] returns.
pub fn lanes(rule: Rule, dims: Dims, module: Module) -> Result<[u32; 3], Ungeometric> {
    let rows = dims.rows.max(1);
    Ok(match rule {
        Rule::Unstated => return Err(Ungeometric::Unstated),
        // Rules for blocks this backend has no shaders for. Listed one by one
        // and not behind a `_`, so that the next rule the fleet adds stops
        // this build rather than reaching a fire.
        //
        // Every name below is accounted for here, and
        // `every_refused_rule_is_a_rule_this_comment_accounts_for` is what
        // keeps that true -- an earlier draft said "the mamba four" and then
        // named six, and left six more with no mention at all.
        //
        // The mamba/recurrent block: `RecurrentScan`, `PerRow`, `PerChannel`,
        // `ElementwiseIn`, `WarpTiledScan` and `PerRowNarrow`.
        //
        // The six CUDA launcher shapes that arrived with the sparse-attention
        // and rope work, none of which has a shader here: `RowScores` (one
        // block a row with a float of shared scratch PER ROW of the causal
        // rectangle, which is not a shape a rounded-up `Rms` may stand in
        // for), `RowsPerHead` (`Rms`' grid with the per-head reading folded
        // in), `RowsFlat` (one THREAD a row, not one block), `Slab` (a CAPPED
        // grid-stride walk, so a launch shape and not a cover), `Tile16` (the
        // only 2-D block in the vocabulary, whose kernels read
        // `threadIdx.y`), and `AxialRope` (one warp per (head, row) with
        // `grid.x` literally one).
        //
        // Then MLA's prepare (`MlaPrepare`), the paged-score taps
        // `PagedScores` and `PagedScoresDecode` that `driver-vulkan`
        // advertises `has_attn_score: false` for, the three packed-head
        // attentions `RowsPackedHeads`, `RowsPackedHeadsNarrow` and
        // `WarpPackedHeads`, gemma-4's alt-up (`AltUpStreams`), and
        // `RoutedQmvTransposed` and `RoutedQmvQuad`, whose transposed and quad
        // forms `quant/qmv.slang` does not compile.
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
        // The three CUDA launcher shapes. `PerRequest` is a grid over the
        // batch's REQUEST count and `Dims` here carries no request field;
        // `Single` and `SingleWarp` are one block of a fixed width, which is
        // a statement about a serial walk no shader in this crate performs.
        | Rule::PerRequest
        | Rule::Single
        | Rule::SingleWarp => return Err(Ungeometric::Unruled(rule)),
        // Four outputs per subgroup, two subgroups per workgroup: 32 lanes
        // wide and `ceil(n / 4)` tall. The round-up is load-bearing --
        // `driver-metal`'s `grid::qmv` records a shared expert's gate, one
        // logit a token, whose truncated count was zero groups and whose
        // buffer therefore kept its zeros while every routed token was
        // combined under `sigmoid(0)`.
        // The ROW goes on x, beside the 32 lanes of one subgroup -- Metal's
        // `qmv_mb` is `32 * n` wide and `quant/qmv.slang` reads
        // `gl_WorkGroupID.x` as exactly that index. A first draft put rows on
        // z instead, where the shader never looks: it computed row 0, left
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
            // Every row on ONE axis, because `norm/rms.slang` takes its row
            // from `gl_WorkGroupID.x` and never mentions y. A first draft put
            // the count on y: it launched a single workgroup on x, computed
            // row 0, left rows 1.. holding the zeros their buffer was born
            // with, and returned success from every call in the chain. The
            // lane sweeps all passed -- they count lanes, and a lane on an
            // axis nobody reads is a lane all the same -- so it took a real
            // dispatch to see it once, and the axis check in `tests/rules.rs`
            // to see it for all 480 entrypoints thereafter.
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
            // The module's workgroup width is the head width it was compiled
            // for. A driver that selected `_d_256` for a 128-wide head would
            // otherwise get a grid that looks reasonable and an answer that is
            // not, so the two are required to agree.
            let built_for = module.local.at(0);
            // No `.max(1)` on the head width. It was here, and it was dead:
            // `Local::at` already answers at least one, so a zero head width
            // can never equal `built_for` and is refused by the line below
            // under its real name. Deleting the clamp changed no test, which
            // is how it was found.
            let head_dim = dims.head_dim;
            if built_for != head_dim {
                return Err(Ungeometric::HeadMismatch {
                    module: built_for,
                    fire: head_dim,
                });
            }
            // The head count MULTIPLIES the workgroup width rather than
            // standing on its own axis: `sdpa_vector.slang` is one workgroup
            // per head, and `gl_NumWorkGroups.y` is the row count it reads.
            [head_dim * dims.q_heads, rows, 1]
        }
        // One workgroup per (query head, TILE of query rows), which is the
        // shape `attn/sdpa_paged.slang`'s `PIE_TILED` body reads: `group.x` is
        // the head, `group.y * 32 + local.y` is the row, and `gl_NumWorkGroups
        // .x` is the head count. The tile height is the module's own `local.y`
        // -- 32 -- and not a number stated here, for `SdpaVector`'s reason: a
        // driver that picked a tile the shader was not built for would get a
        // grid that looks reasonable and an answer that is not.
        //
        // The rows round UP, which `groups` does by dividing by `local.y`, and
        // the rows of the last tile that are past the end are exactly why the
        // row hands the kernel `Source::Rows`: they stay in the loop, reach
        // every barrier, and contribute nothing. `driver-metal`'s `sdpa_tiles`
        // is the same grid with the multiply written out, because a Metal
        // dispatch takes threads and quantises at the threadgroup.
        // The matrix-unit tier is the SAME grid here, and that is a real
        // difference from `driver-metal` rather than a copy. Metal's mma
        // threadgroup is 128 threads -- four simdgroups of eight query rows --
        // because a Metal simdgroup owns eight rows on the matrix unit.
        // `attn/sdpa_paged_mma.slang` is `[numthreads(32, 32, 1)]` like the
        // scalar tile, with `local.y` the query row and `group.y * 32` its
        // base, so the workgroup count is identical and only what happens
        // inside one differs. Taking Metal's 128 would dispatch a quarter of
        // the tiles this shader needs and leave three rows in four unwritten.
        Rule::SdpaTiled | Rule::SdpaMma => [dims.q_heads * module.local.at(0), rows, 1],
        Rule::PerHeadElementwise => [dims.q_heads * dims.head_dim, 1, 1],
        Rule::GatedRms => [dims.head_dim, dims.kv_heads, 1],

        Rule::RouterLane => [module.local.at(0), rows, 1],
        // ONE workgroup whatever the rows: the sort is over the expert
        // histogram, which is fire-wide.
        Rule::RouterSort => [module.local.at(0), 1, 1],
        Rule::RouteRows => [dims.width, rows, 1],
        // Rows on x as for `Qmv`, and the expert slot on z.
        //
        // The y axis is the row's OWN statement of its output length, which is
        // what `grid_param = Some(1)` names: `out_vec_size`, the second word.
        // It used to be `width.div_ceil(4)`, and that division was a guess --
        // a routed projection writes a whole token's `k` results end to end,
        // so the rectangle is `k` times as wide as one result, and dividing by
        // four is right only when four experts are routed to. gpt-oss routes
        // to four, which is why it went unseen; qwen3's MoE routes to eight,
        // where the same expression asks for twice the workgroups it needs
        // (harmless, `active_out` guards the tail), and anything routing to
        // fewer than four UNDERSHOOTS -- and an undershot grid writes nothing,
        // reads back as whatever the arena held, and reports success.
        //
        // Found by the `driver-wgpu` agent, who carried the identical
        // expression, hit it against their own shader, and left a note saying
        // this file had it too. Fixed here from that report.
        Rule::RoutedQmv => [
            module.local.at(0) * rows,
            dims.axis.max(1),
            dims.experts_per_token.max(1),
        ],
    })
}

/// How many workgroups `vkCmdDispatch` should be given.
///
/// The round-up is applied on every axis, because a missing workgroup runs
/// nothing and says nothing while an extra one runs lanes past the end of the
/// tensor, and every body in `kernels-vulkan` guards its own tail against the
/// bound length of what it writes.
///
/// That last clause is only true of 631 of the 665 modules, and the exception
/// is the more interesting half. Thirty-four of them read `gl_NumWorkGroups`
/// and use it as a QUANTITY rather than a bound: `rope/neox.slang` takes
/// `gl_NumWorkGroups.x` as the rotary pair count it strides each pair's partner
/// by and divides the frequency exponent by. For those an extra workgroup does
/// not run a guarded lane, it changes the arithmetic every lane does, and the
/// round-up has to be a no-op. It is -- their extents divide their workgroups
/// exactly, which is why `rope` is compiled at `local_size (1, 1, 1)` -- and
/// `tests/rules.rs` checks that rather than assuming it. [`lanes`] is what lets
/// a caller ask the same question.
///
/// # Errors
///
/// [`Ungeometric`] when the rule cannot answer for these dimensions. A driver
/// must not substitute a shape here — see [`Ungeometric::PartialTile`].
pub fn groups(rule: Rule, dims: Dims, module: Module) -> Result<[u32; 3], Ungeometric> {
    // A non-GEMM module that names a tile means a name grew a `_bm_` suffix
    // without this file learning what it decomposes.
    if module.tile.is_some() && rule != Rule::Qmm {
        return Err(Ungeometric::Untiled);
    }
    let e = lanes(rule, dims, module)?;
    Ok([
        e[0].div_ceil(module.local.at(0)),
        e[1].div_ceil(module.local.at(1)),
        e[2].div_ceil(module.local.at(2)),
    ])
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

    /// A plausible fire, at a row count a GEMM tile divides.
    ///
    /// 32 and not 4: `Rule::Qmm` REFUSES a row count no compiled tile divides,
    /// so a sweep at four rows would find that rule absent and prove less than
    /// it looked like it was proving.
    fn dims() -> Dims {
        Dims {
            rows: 32,
            width: 4096,
            in_width: 12288,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            axis: 4096,
            rotary_dims: 128,
            n_experts: 64,
            experts_per_token: 8,
        }
    }

    /// Every workgroup count, times its workgroup, covers the thread extent.
    ///
    /// This is the property the whole module exists for, and it is stated as a
    /// sweep rather than per rule because the failure it catches is a rule
    /// nobody thought about. An undershoot writes nothing and reports success.
    #[test]
    fn no_rule_launches_fewer_lanes_than_its_extent() {
        let locals = [
            [256, 1, 1],
            [32, 2, 2],
            [16, 16, 1],
            [1024, 1, 1],
            [64, 1, 1],
            [32, 8, 1],
        ];
        let mut checked = 0;
        for &rule in kernels::LaunchRule::ALL {
            for local in locals {
                // Only a GEMM carries a tile, and only a GEMM is allowed to.
                let m = Module {
                    local: Local(local),
                    tile: (rule == Rule::Qmm).then_some(Tile { rows: 32, cols: 64 }),
                };
                // Decode attention is compiled per head width -- its
                // the x extent IS `PIE_HEAD_DIM` -- so a sweep over workgroup
                // shapes has to move the fire's head width with it. Otherwise
                // the rule refuses five of the six shapes and the sweep proves
                // nothing about the family it most needed to.
                let d = if rule == Rule::SdpaVector {
                    Dims {
                        head_dim: local[0],
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
        // rules ARE skipped here -- see `SERVED`, which is the list this
        // floor is taken from and the one test that says what it means.
        assert!(
            checked >= SERVED.len() * locals.len(),
            "{checked} checks over {} shapes",
            locals.len()
        );
    }

    /// Every launch rule this backend lays a grid out for.
    ///
    /// `kernels::LaunchRule` is the whole fleet's vocabulary, and most of it
    /// is CUDA's: mamba scans, MLA, the paged-score taps, packed-head
    /// attentions, gemma-4's alt-up. `lanes` answers for exactly the rules
    /// below and refuses the rest by name.
    ///
    /// This list is NOT the same as "what `kernels-vulkan` has a shader for",
    /// which is what it used to claim. It is two rules longer. `GatedRms` and
    /// `PerHeadElementwise` are laid out here and named by no Vulkan row --
    /// `kernels-cuda` states them on `attn_sink_correction`,
    /// `per_head_rmsnorm` and three SSM rows, and this backend has no shader
    /// for any of those. The grids are right; they are simply written ahead
    /// of the table rather than behind it, and a reader counting coverage off
    /// this list would have counted two kernels that do not exist here.
    ///
    /// `rules.rs`'s `every_rule_the_table_names_is_one_this_driver_can_lay_out`
    /// is what keeps that gap counted: it reads the table, asserts all
    /// fifteen rules it names are served, and names those two as the
    /// difference -- in both directions, so a row that starts naming either
    /// one fails until the list is corrected.
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

    /// The rules this backend serves are exactly `SERVED`, and every other
    /// rule the fleet states refuses BY NAME.
    ///
    /// The point is the second half. `lanes` names all twenty-four unserved
    /// rules in one arm rather than catching them with a `_`, so that the
    /// compiler stops this build when the fleet grows a rule -- which is
    /// exactly what happened when twenty-one arrived at once and this match
    /// was the thing that noticed. A wildcard would have compiled, and a
    /// mamba text would have reached a fire and been refused there, or worse,
    /// launched on whichever grid the wildcard chose.
    ///
    /// So this test is a ledger and a tripwire: a rule added upstream is
    /// either put in `SERVED` with a module behind it, or it is refused, and
    /// there is no third state.
    #[test]
    fn the_rules_this_backend_serves_are_exactly_the_ones_with_shaders() {
        let m = Module {
            local: Local([32, 2, 2]),
            tile: Some(Tile { rows: 32, cols: 64 }),
        };
        for &rule in kernels::LaunchRule::ALL {
            let module = Module {
                tile: (rule == Rule::Qmm).then_some(Tile { rows: 32, cols: 64 }),
                ..m
            };
            // `SdpaVector` refuses a module whose workgroup width is not the
            // fire's head width -- that is the mis-selection guard, not a
            // rule gap -- so it is asked with the module it was built for.
            let d = if rule == Rule::SdpaVector {
                Dims {
                    head_dim: module.local.at(0),
                    ..dims()
                }
            } else {
                dims()
            };
            let answer = lanes(rule, d, module);
            if SERVED.contains(&rule) {
                assert!(answer.is_ok(), "{rule:?} is served but did not answer");
            } else if rule == Rule::Unstated {
                assert_eq!(answer, Err(Ungeometric::Unstated));
            } else {
                assert_eq!(
                    answer,
                    Err(Ungeometric::Unruled(rule)),
                    "{rule:?} has no shader and must refuse by name"
                );
            }
        }
    }

    /// Every rule `lanes` refuses by name is a rule its comment explains.
    ///
    /// `the_rules_this_backend_serves_are_exactly_the_ones_with_shaders`
    /// already keeps the SET honest: a rule the fleet adds is either served or
    /// refused, and the compiler stops the build until someone chooses. What
    /// nothing kept honest was the PROSE beside the arm, and it had gone bad
    /// in both available ways at once -- it opened with "the mamba four" and
    /// then named six, and six further rules (`RowScores`, `RowsPerHead`,
    /// `RowsFlat`, `Slab`, `Tile16`, `AxialRope`) appeared in the arm with no
    /// mention anywhere above it.
    ///
    /// That is the same failure `SERVED`'s doc had, and it matters for the
    /// same reason: the arm says a rule is refused, and only the comment says
    /// WHY -- whether this backend lacks the shader, lacks the grid, or
    /// deliberately advertises the capability off. A reader deciding whether
    /// to implement one reads the comment, and an unnamed rule reads as an
    /// oversight when it is a decision.
    ///
    /// So this reads the comment lines between the arm's opening line and its
    /// `Unruled` return, and requires each refused rule to be named in
    /// backticks among them. Adding a rule to the arm without a word about it
    /// now fails here rather than merely compiling.
    #[test]
    fn every_refused_rule_is_a_rule_this_comment_accounts_for() {
        const SRC: &str = include_str!("geometry.rs");

        let open = SRC
            .find("// Rules for blocks this backend has no shaders for.")
            .expect("the arm's opening comment line moved");
        let close = SRC[open..]
            .find("=> return Err(Ungeometric::Unruled(rule)),")
            .expect("the arm's return moved")
            + open;
        let prose: String = SRC[open..close]
            .lines()
            .filter(|l| l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");

        let mut unnamed = Vec::new();
        for &rule in kernels::LaunchRule::ALL {
            if SERVED.contains(&rule) || rule == Rule::Unstated {
                continue;
            }
            if !prose.contains(&format!("`{rule:?}`")) {
                unnamed.push(format!("{rule:?}"));
            }
        }
        assert!(
            unnamed.is_empty(),
            "`lanes` refuses {unnamed:?} and its comment says nothing about \
             them. Say which of the three it is -- no shader, no grid, or a \
             capability this driver advertises off -- rather than leaving the \
             next reader to guess from the name."
        );
    }

    /// Which rules answer a grid with a zero in it, stated exactly.
    ///
    /// A zero on any axis dispatches nothing and reports success -- the
    /// failure with no symptom, and the one this crate refuses hardest.
    /// `plan_one` is the layer that refuses it (`Undispatchable::Empty`, and
    /// deleting that line fails the suite). This asks the question one layer
    /// down, where several rules clamp a dimension to one for exactly this
    /// reason and FIVE of those clamps could be deleted with everything
    /// staying green -- the real plans never state a zero, so the clamps only
    /// ever mattered for the input nothing was feeding them.
    ///
    /// Pinned as a SET rather than asserted empty, because it is a long way
    /// from empty: FOURTEEN rule-and-field pairs answer a grid with a zero in
    /// it. Almost every rule that multiplies a dimension does, and the ones
    /// that clamp clamp only the dimension they happened to be bitten by.
    ///
    /// So `groups` does not guarantee a non-empty grid and was never the
    /// layer that did. `plan_one` is, and singly: delete its check and the
    /// suite fails. Stating the set is what turns that from an assumption
    /// into a fact with a number, and makes the next change visible -- a
    /// clamp deleted here moves this set, a new rule joins it, and a rule
    /// that starts refusing leaves it.
    ///
    /// `Err` is not a failure here. A rule that refuses a degenerate
    /// dimension has said so by name, which is the opposite of the defect.
    #[test]
    fn only_these_rules_answer_a_grid_with_a_zero_in_it() {
        // The same six workgroup shapes the sibling sweep uses. One shape
        // left most rules refusing on a mismatch that had nothing to do with
        // the zero under test, which made the sweep look thorough and ask
        // almost nothing -- seven answers out of a hundred and twelve.
        let locals = [
            [256, 1, 1],
            [32, 2, 2],
            [16, 16, 1],
            [1024, 1, 1],
            [64, 1, 1],
            [32, 8, 1],
        ];
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
            for local in locals {
                let m = Module {
                    local: Local(local),
                    tile: (rule == Rule::Qmm).then_some(Tile { rows: 32, cols: 64 }),
                };
                for (name, zero) in zeroed {
                    // Decode attention is compiled per head width, so a
                    // sweep over workgroup shapes has to move the head width
                    // with it -- except when the head width IS the zero
                    // under test.
                    let mut d = if rule == Rule::SdpaVector {
                        Dims {
                            head_dim: local[0],
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
        assert!(answered > 200, "only {answered} rules answered at all");
    }

    /// And not WILDLY more, which is the other way a division can be wrong.
    ///
    /// One extra workgroup per axis is the round-up. More than that means the
    /// extent was divided by the wrong number — the mistake a shell makes when
    /// it reads `local_size_x` and ignores `y`.
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
    #[test]
    fn a_module_built_for_another_head_width_refuses() {
        assert_eq!(
            groups(Rule::SdpaVector, dims(), Module::new([256, 1, 1])),
            Err(Ungeometric::HeadMismatch {
                module: 256,
                fire: 128
            })
        );
        // And the matching one answers, so the check is a check and not a wall.
        assert!(groups(Rule::SdpaVector, dims(), Module::new([128, 1, 1])).is_ok());
    }

    /// A rope over a key tensor covers the KEY heads, not the query heads.
    ///
    /// The regression `driver-metal`'s `rope_heads` is written against, held
    /// here too because the two shells compute it separately.
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

    /// A GEMM at a row count no tile divides refuses rather than substituting.
    #[test]
    fn a_partial_tile_refuses() {
        let ragged = Dims { rows: 3, ..dims() };
        let m = Module::named("affine_qmm_t_bfloat16_bm_16_bn_16", [32, 2, 2]);
        assert_eq!(
            groups(Rule::Qmm, ragged, m),
            Err(Ungeometric::PartialTile { rows: 3, tile: 16 })
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
    /// reads `local_size_x` and treats y and z as 1: that mistake OVERSHOOTS,
    /// every shader guards its own tail, and the answer is merely 16x more
    /// workgroups than the work needs. It is a real defect and an invisible
    /// one, and `(32, 2, 2)` and `(16, 16, 1)` are most of this table.
    #[test]
    fn a_workgroup_is_read_on_every_axis() {
        let l = Local([32, 2, 2]);
        assert_eq!([l.at(0), l.at(1), l.at(2)], [32, 2, 2]);
        // And the division uses them: 4096 rows over a (16, 16, 1) module is
        // 256 workgroups on y, not 4096.
        let d = Dims {
            rows: 4096,
            width: 16,
            ..dims()
        };
        let g = groups(Rule::ElementwiseRows, d, Module::new([16, 16, 1])).unwrap();
        assert_eq!(g, [1, 256, 1]);
    }

    /// A zero workgroup width cannot divide by zero.
    #[test]
    fn a_zero_workgroup_does_not_divide_by_zero() {
        assert_eq!(Local([0, 0, 0]).at(0), 1);
        let g = groups(Rule::Elementwise, dims(), Module::new([0, 0, 0])).unwrap();
        assert_eq!(g[0], 4096 * 32);
    }
}
