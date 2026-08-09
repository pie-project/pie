//! How a rectangle becomes a CUDA launch.
//!
//! The Metal driver has had this file for as long as it has had a generic
//! executor -- `driver-metal/src/lowering/launch.rs`, which turns a
//! [`kernels::LaunchRule`] into a thread grid. CUDA has not, because on CUDA
//! the grid was computed in C++, inside the launcher, one `(H + BLOCK - 1) /
//! BLOCK` per kernel. The two backends were not disagreeing about
//! arithmetic; only one of them had written the arithmetic down anywhere a
//! table could point at.
//!
//! So the rule vocabulary is the shared crate's and the arithmetic is this
//! backend's, which is the split [`kernels::LaunchRule`]'s own doc describes:
//! *"This is data. The arithmetic each variant names stays in the driver,
//! beside the doc comment that explains it."*
//!
//! # A port, not a rewrite
//!
//! This is `driver-cuda/src/bind/launch.rs` moved to the crate that now does
//! the launching, with the numbers untouched. Untouched is the requirement:
//! the migration is an A/B against the deleted C++ launchers, and a rule that
//! changed the arithmetic on the way across cannot be compared with the
//! `<<<>>>` it replaced. Every improvement this file's authors could see is
//! named in the function that declines to make it.
//!
//! The port carried nothing but the arithmetic. [`eval`] reads
//! [`kernels::LaunchRule`] and the constants below and nothing else, so there
//! was no `driver-cuda` type to find a local equivalent for -- which is also
//! why the move is a copy rather than a translation.
//!
//! # One axis Metal does not have
//!
//! [`Launch`] carries `smem`, and Metal's does not. Dynamic shared memory is
//! a launch parameter on CUDA and a threadgroup-memory binding on Metal, so
//! the rule has to produce it here or the reduction kernels cannot run. It is
//! the only structural difference the port found, and it is why [`Launch`] is
//! this crate's type rather than one lifted from `kernels`.
//!
//! Six rules now size it from [`Dims`] rather than from a constant, and each
//! one is a kernel that reads `extern __shared__` and would otherwise read
//! whatever the last launch left there: [`Rule::SdpaVector`] wants
//! `(rows + 256) * sizeof(float)`, [`Rule::RouterSort`] wants
//! `(3 * n_experts + 34) * sizeof(int)`, [`Rule::Rope`] wants a cached
//! sin/cos pair per rotated channel, [`Rule::RecurrentScan`] wants
//! `2 * head_dim * sizeof(float)` of q/k staging, [`Rule::RowScores`] wants
//! one float per row of the rectangle, and [`Rule::Rms`] keeps its fixed warp
//! scratch. Too LITTLE dynamic shared memory is not a launch failure — the
//! kernel reads past its allocation into another block's, answers, and
//! reports success — which is why these are computed rather than defaulted.
//!
//! Two of those six landed after the rest, and the count of what they unblock
//! is why the vocabulary grew rather than the rows bending to it: fifty-odd
//! kernels across `ssm`, `attn` and `norm` were extracted, proved NVRTC-clean
//! and left rowless, and the two dominant causes were a shared allocation no
//! rule computed and a grid no rule opened. Neither is a backend behind its
//! vocabulary. Both were the vocabulary being short.
//!
//! # The axes are not in the same order twice, and that is the kernels'
//!
//! Metal puts the row on `grid.z` for the per-head shapes and on `grid.y` for
//! the rest. CUDA does neither consistently, because a CUDA kernel reads the
//! axis it was written to read: `split_qkv` takes its token from `blockIdx.y`
//! and `attn_naive` takes its token from `blockIdx.y` while its HEAD is
//! `blockIdx.x`; `attn_sink_rescale` is the other way round; `pad_head_dim`
//! puts the head on `blockIdx.x` and the token on `blockIdx.y` while
//! `zamba_rmsnorm_gated` puts the token on `blockIdx.x` and the group on
//! `blockIdx.y`. Two of those four are transposes of each other and both are
//! LEGAL launches of the wrong shape — a transposed grid does not fail, it
//! runs `heads * rows` blocks that each address the wrong cell.
//!
//! So every rule below states the axis order of the launcher it came from and
//! nothing else. Reading one rule's order off another's is the mistake this
//! module is arranged to make visible.
//!
//! # Why layer 3, when there is no CUDA in it
//!
//! Nothing here links, loads or calls anything: a rule is arithmetic over a
//! row's stated shape, and this module would compile on a machine with no
//! driver, no toolkit and no device -- as layers 1 and 2 do. It sits behind
//! the feature anyway because its only reader is [`crate::runtime::fire`],
//! and a module published to consumers that cannot use it is a surface that
//! has to be kept for nobody. If a second reader ever appears -- an offline
//! planner sizing a grid without a GPU -- moving this down a layer costs one
//! line and breaks nothing, because there is nothing to break.
//!
//! # What is ported, and what the source of truth for it is
//!
//! Thirty-three of the thirty-six. Every one of them is derived from a host
//! launcher still standing in `kernels-cuda/csrc/src/**/*.cu` — those files
//! hold host code only now, the device text having been extracted into `.cuh`
//! headers this crate compiles, so what remains in them is exactly the
//! `<<<grid, block, smem, stream>>>` a rule has to reproduce. Each function
//! below names its file, its kernel and its launch expression. **A rule with
//! no cited launcher is a guess**, and the two that have none —
//! [`Rule::Qmv`] and [`Rule::Qmm`] — stay [`Ungeometric::Unported`] for
//! exactly that reason: this backend has no affine-quantized matvec or matmul
//! `<<<>>>` for a DENSE projection at all. Those go through cuBLAS, which is
//! a library call with no grid, and the quantized ones through Marlin, whose
//! grid comes from a host heuristic over an SM count and a tuning cache and
//! is therefore not a function of [`Dims`]. `mxfp4_marlin.cu` holds three
//! launchers and all three are repacks, which are [`Rule::Elementwise`].
//!
//! [`Rule::RoutedQmv`] was on that list and has come off it. The reading that
//! put it there was about dense projections and was never checked against the
//! MoE decode path, where `quant/dequant_wna16.cu:74` launched
//! `dim3(routes, ceil(intermediate / 8))` at 256 — a grid that is a function
//! of the rectangle and nothing else. That launcher has since been deleted as
//! unreached (§43.9) and the rule outlived it, which is the ordinary way
//! round here: the row is what fires now, and `quant/dequant_wna16.cuh:295`
//! is where the axis assignment is read back. A refusal citing the launcher it could
//! not find is cheap to overturn the day someone finds one, which is the
//! whole of why they are written that way.
//!
//! Five of the thirty-three are this file's oldest new ones and every one of
//! them was written against a `<<<>>>` first and a kernel second:
//! [`Rule::RecurrentScan`] from `ssm/gated_delta_net.cu`,
//! [`Rule::PerRow`] from `attn/kv_paged.cu`, [`Rule::PerChannel`] from
//! `ssm/causal_conv1d.cu`, [`Rule::ElementwiseIn`] from `norm/dsv4_hc.cu`
//! and [`Rule::RowScores`] from `attn/dsa_indexer.cu`. What each one does
//! NOT serve is stated beside it, because the shapes they were nearly
//! confused with are the ones a reader will reach for next.
//!
//! Eight more landed in one pass after those, asked for by name by two
//! migration agents who had audited every unrowed kernel and could say what
//! each missing shape cost: [`Rule::RowsPerHead`] from `norm/rmsnorm.cu`,
//! [`Rule::RowsFlat`] from `moe/dsv4_routing.cu`, [`Rule::Slab`] from
//! `quant/dequant_wna16.cu`, [`Rule::RoutedQmv`] from the same file,
//! [`Rule::Tile16`] from the three vision towers, [`Rule::AxialRope`] from
//! `vision/gemma4_vision.cu`, [`Rule::WarpTiledScan`] from
//! `ssm/gated_delta_net.cu` and [`Rule::PerRowNarrow`] from
//! `vision/gemma4_audio.cu`. **Two of them broke ground the module header
//! had recorded as closed**: [`Rule::Tile16`]'s block is the first here that
//! is not `[n, 1, 1]`, and [`Rule::AxialRope`]'s and
//! [`Rule::WarpTiledScan`]'s grids are the first with a `grid.z` above one.
//! Neither needed a field: [`Launch::block`] was already three wide and
//! [`Launch::grid`] always has been, so what was missing was a launcher
//! anyone had written down and not a quantity anyone could measure.
//!
//! # What is still refused, and why widening a rule would not reach it
//!
//! **Eight more landed in the round after those**, each against a launcher a
//! previous agent had read, refused and written down: [`Rule::PagedScores`]
//! and [`Rule::PagedScoresDecode`] from `attn/attention_naive_paged.cu`,
//! [`Rule::MlaPrepare`] from `attn/mla_paged.cu`,
//! [`Rule::RowsPackedHeads`], [`Rule::RowsPackedHeadsNarrow`] and
//! [`Rule::WarpPackedHeads`] from `attn/qkv_fused.cu`,
//! [`Rule::RoutedQmvTransposed`] from `quant/dequant_wna16.cu` and
//! [`Rule::AltUpStreams`] from `norm/altup.cu` — **both of those source
//! launchers have since been deleted as unreached (§43), which is what a
//! routed row is FOR; the rules and the kernels are untouched.** **Two of them needed a
//! [`Dims`] field and both fields are DISTINCT QUANTITIES rather than second
//! readings** — [`Dims::requests`] and [`Dims::altup_streams`], each of which
//! says in its own doc which existing field it would otherwise have
//! overloaded and what that would have launched. A third,
//! [`Rule::MlaPrepare`], gained the first reader [`Dims::rotary_dims`] has
//! ever had, and did not need a field: MLA's `qk_rope_head_dim` is the
//! channels that rotate, which is what that field already counted.
//!
//! [`Rule::PagedScores`] is also the first rule here whose shared memory is a
//! function of a head width — `(head_dim + 128) * sizeof(float)`, never a
//! literal — which is the shape [`Rule::SdpaVector`] has and gets from the
//! wrong extent.
//!
//! **Three more landed in the round after that**, and two of them are the
//! same finding read twice: [`Rule::Single`] and [`Rule::SingleWarp`] from
//! `layout/slot_ops.cu:61` and `attn/kv_paged.cu:516`/`:533`, whose grid is a
//! LITERAL `1` that no quotient reproduces, and [`Rule::PerRequest`] from
//! `attn/attention_naive.cu:174`. `PerRequest` is the one of the three whose
//! launcher text three kernels share and whose VERDICT they do not:
//! `attn/page_compact.cu:45`/`:48` open the same `<<<num_requests, kBlock>>>`
//! and keep [`Rule::PerRow`], because their statement's rectangle is already
//! the request count. Neither needed a [`Dims`] field — [`Dims::requests`]
//! was already there and is now filled at the production call site — and
//! none of the three could be got by widening a rule, because the number
//! they differ on is which axis the host counted.
//!
//! Of the shapes read off their launchers and left refused, each fails on a
//! number [`Dims`] does not carry rather than on arithmetic nobody wrote.
//! **A third grid axis is no longer one of the reasons** — [`Rule::AxialRope`],
//! [`Rule::WarpTiledScan`], [`Rule::PagedScores`] and [`Rule::AltUpStreams`]
//! open one — so what stops each of these is now stated as the axis's VALUE:
//!
//! * **A third grid axis over a channel count.** `vision/gemma4_audio.cu`
//!   launches `dim3((F + 15) / 16, (T + 15) / 16, C)` for `k_conv2d_s2`,
//!   `k_chlast` and `k_chfirst` at `186-197` — [`Rule::Tile16`]'s grid with
//!   a convolution's channels on `grid.z`. Three independent extents, and
//!   [`Dims`] carries two plus head counts. The same objection the stream
//!   count draws applies to spelling `C` as a head, with one addition: these
//!   rectangles are `[C, T, F]` and the tiled pair are TRANSPOSES of each
//!   other, so a rule that took `rows` and `width` off a statement would
//!   have to be told which of the three axes each one is.
//! * **A page window on `grid.x`.** `layout/envelope.cu`'s `dot<128>` is
//!   `dim3(p_max, num_kv_heads)` and `attn/attention_naive.cu`'s
//!   `attn_mtp_paged_history` sizes its shared scores on
//!   `max_global_tokens + history_steps`. A page bound and a draft-history
//!   depth are host plan numbers, not extents of the fire's rectangle —
//!   `Dims::rows` is the token count and answering with it would launch a
//!   window the size of the batch.
//! * **A layer count on `grid.z`.** `layout/gather_tokens.cu` launches
//!   `dim3(num_ops, 1, num_layers)`. A layer count is not a rectangle's
//!   extent, and the same launcher picks between an `int4` and a `u16`
//!   kernel on a stride's alignment, which is a kernel choice rather than a
//!   rectangle.
//! * **A second head width.** `ssm/gated_delta_net.cu`'s
//!   `chunk_gated_delta_prefill_batched_cached` is [`Rule::RecurrentScan`]'s
//!   grid and block to the digit and wants `K_d * V_d * sizeof(float)` of
//!   shared memory — plus a `cudaFuncSetAttribute` raising the 48 KB cap.
//!   [`Dims::head_dim`] is one number and this launcher needs two, so the
//!   shapes that look alike differ in a quantity the rule cannot see. Two
//!   rules or none, and none until a `Dims` carries `V_d`.
//!   [`Rule::WarpTiledScan`] is not the exception it looks like: it needs
//!   only the VALUE width, takes it as `width / kv_heads` — a quotient of
//!   the output rectangle — and allocates nothing, so it never has to tell
//!   the two head widths apart. The launcher that wants both still cannot be
//!   stated.
//! * **A block width a rule fixes.** `vision/qwen3_vl_tower.cu:249` launches
//!   `k_split_rope_qkv<<<dim3(NH, N), HEAD / 2>>>` where [`Rule::PerHead`]'s
//!   grid matches to the digit and its `PAD_BLOCK` does not. Widening it is
//!   not a spelling change — it quadruples a launch — and the honest shape is
//!   a rule per block width, which is what [`Rule::PerRowNarrow`] is: the
//!   audio tower's `<<<rows, 128>>>` layernorm, stated rather than rounded up
//!   into [`Rule::PerRow`]'s 256. A second such rule costs one function and
//!   nothing else, and no row has asked for this one.
//! * **A 32-by-8 tile.** `quant/dtype_cast.cu:140` and
//!   `quant/quant_bf16_to_fp8.cu:128` launch `dim3(BX, BY)` with
//!   `BX = 32, BY = 8` over `dim3((n + 31) / 32, (m + 7) / 8)` —
//!   [`Rule::Tile16`]'s idea at different constants in both dimensions.
//!   Nothing about it is unstatable and no row has asked; it is recorded so
//!   that the next reader does not widen `tile16` into it, which would
//!   change eleven vision launches to fix three quantizer ones.
//!
//! Every other variant answers [`Ungeometric::Unported`] rather than a guess:
//! a rule this backend has not written the arithmetic for is not a rule with
//! a default, and the whole reason the table can be trusted is that a driver
//! refuses what it cannot state.

use kernels::LaunchRule as Rule;

/// The fire-time quantities a CUDA launch rule may read.
///
/// Nine of the twelve are the nine `driver-metal`'s `Dims` carries, spelled
/// the same way, minus its `axis` — so the two structures diff against each
/// other and a reader can check a rule against its Metal twin without
/// translating a vocabulary on the way. `axis` is absent because no CUDA
/// launcher has a reduction span that is not the row: `rmsnorm.cu`'s every
/// entry point is `<<<rows, 256>>>` and the per-head reading of those symbols
/// is a SECOND kernel here, not a second grid, so a field for it would be a
/// field whose meaning nothing checks. That is the one place the two
/// backends' rectangles genuinely differ and it is recorded rather than
/// papered over.
///
/// The tenth is [`Dims::stated_head_dim`], which Metal has no field for and
/// wants one: `lowering/dispatch.rs:301` folds the same question into its
/// head width the moment it is asked — `head_dim: stated_head.unwrap_or(
/// geometry.head_dim)` — so on that side "the statement named none" and "the
/// statement named the fire's number" are one value and no rule can tell them
/// apart. This side keeps them apart. See that field for what the fold costs
/// and why the fold itself is still right for [`Dims::head_dim`].
///
/// The eleventh and twelfth are [`Dims::requests`] and
/// [`Dims::altup_streams`], and each is here for the same reason
/// `stated_head_dim` is: a DISTINCT QUANTITY rather than a second reading of
/// a field that already has one. Each field's own doc says which existing
/// field it would otherwise have had to overload and what that would have
/// launched.
///
/// One field is supplied and read by exactly one rule that nothing fills
/// today — [`Dims::requests`]; [`Dims::rotary_dims`] gained its first reader
/// in [`Rule::MlaPrepare`]. Both are here because a caller that has the
/// number should not have to know which backend wants it: a `Dims` assembled
/// from a statement is the same value on both drivers, and a field that
/// appears when a rule lands is a field every existing call site has to be
/// revisited to fill.
///
/// **A zero is refused, never floored — with exactly one exception, and it is
/// labelled.** Every field below says what supplies it and what its zero
/// means, and every rule that reads one checks it: a head count of zero makes
/// `grid.y` zero, and a grid with a zero axis launches no blocks, returns
/// `CUDA_SUCCESS`, and leaves the output exactly as the last kernel wrote it.
/// That is indistinguishable from a fire that ran, which is what
/// [`Ungeometric::Empty`] exists to make impossible.
///
/// [`Dims::stated_head_dim`] is the exception: its zero is ABSENCE and is a
/// legal, meaningful, common value — `Source::IfPresent`'s false arm, reaching
/// a rule instead of a binder. It is the one field for which a zero must not
/// be refused, and it is spelled as its own field rather than as a second
/// reading of [`Dims::head_dim`] precisely so that "refuse a zero" can stay
/// the rule for the other eleven.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dims {
    /// Rows the rectangle covers — tokens, requests or routed slots,
    /// whichever the statement's lowering counted. Zero is a rectangle that
    /// collapsed.
    pub rows: u32,
    /// Elements per row of the launch's last widthed operand — its output.
    pub width: u32,
    /// Elements per row of its first widthed operand — its input. Read by
    /// the rules that size on what a launch READS, which is what a statement
    /// that unpacks one buffer into a wider one needs.
    pub in_width: u32,
    /// Query heads — the head count of the tensor a per-head launch reads
    /// ROW-MAJOR, with the head on `grid.y`. `attn_sink_rescale`'s
    /// `num_q_heads` and `attn_naive`'s, and half of the rope's total.
    /// Zero is an attention with no query heads, which is not a shape.
    pub q_heads: u32,
    /// Key/value heads — and, for the rules whose operand is not q, THE HEAD
    /// COUNT OF THE TENSOR THIS LAUNCH ADDRESSES.
    ///
    /// The same reading `driver-metal` gives it, and it is deliberate there:
    /// `PerHead` is stated once per tensor and a grouped-query fire has two
    /// head counts, so the rule takes the one the operand names rather than
    /// the fire's q. A launch that read `q_heads` for a kv-shaped tensor
    /// covers heads the buffer does not have — thirty-two over an eight-head
    /// operand — and the overrun lands on whatever the arena put next.
    ///
    /// It is also the axis a grouped norm counts: `zamba_rmsnorm_gated`'s
    /// groups are `hidden / group_size`, and the GDN recurrence's are its
    /// value heads. Zero means the tensor has no heads, which no head-shaped
    /// rule may launch over.
    pub kv_heads: u32,
    /// Elements per head — `head_dim`, `V_d`, `K_d`, or a norm group's width,
    /// which are the same quantity under four names **on every shape a rule
    /// here states**.
    ///
    /// Read as a BLOCK width by [`Rule::PerHeadElementwise`] and as a shared
    /// allocation by [`Rule::Rope`] and [`Rule::RecurrentScan`]. Zero is a
    /// head of no channels: the launcher's own `if (head_dim <= 0) return;`
    /// guard, moved to where a rule can answer it rather than to where a
    /// kernel silently does nothing.
    ///
    /// The four names are not always one number and the recurrence is where
    /// that shows. `ssm/gated_delta_net.cu` carries `K_d` and `V_d`
    /// separately — a key head and a value head — and every launcher whose
    /// shared memory is `2 * K_d * sizeof(float)` reads the KEY width, which
    /// is what [`Rule::RecurrentScan`] states this field to be. A launcher
    /// that wants BOTH is refused rather than served with one of them:
    /// `chunk_gated_delta_prefill_batched_cached`'s `K_d * V_d * sizeof(float)`
    /// is the same grid and the same block as the rule and a different
    /// allocation, and a field carrying one head width cannot tell them
    /// apart. See the module header's refusal list.
    ///
    /// **It is the FIRE's head width, and it is filled from the fire when the
    /// statement names none.** `driver-cuda/src/bind/mod.rs:1321` spells that
    /// — `spec.per_head_dim.unwrap_or_else(|| extent(ctx.head_dim))` — and it
    /// is right for every reader above, each of which is asking "how wide is
    /// a head here" and would rather have the model's answer than nothing. It
    /// is not an answer to "did the statement name a head width", which is a
    /// different question and is [`Dims::stated_head_dim`]'s.
    pub head_dim: u32,
    /// The per-head width THE STATEMENT NAMED, and **zero means it named
    /// none**.
    ///
    /// The one field here whose zero is a value rather than a refusal, and
    /// the whole reason it exists. `table/norm.rs:36` states what
    /// `rmsnorm.cu`'s four launchers are handed:
    ///
    /// ```text
    /// num_rows <- IfPresent(PerHeadDim,
    ///                       Mul(Rows, Div(Width(In(0)), PerHeadDim)),
    ///                       Rows)
    /// hidden   <- IfPresent(PerHeadDim, PerHeadDim, Width(In(0)))
    /// ```
    ///
    /// `Source::IfPresent` reads `spec.per_head_dim` — an `Option` — so the
    /// ahead-of-time operand side has always been able to see the ABSENCE.
    /// The geometry side could not, and [`Rule::RowsPerHead`] is that same
    /// conditional in the only two terms a launch has, so this field is the
    /// `Option` arriving where a rule can read it: `Some(d)` is `d`, `None`
    /// is `0`.
    ///
    /// # Why this is not [`Dims::head_dim`] under another name
    ///
    /// Three fixes were available and two of them are wrong in ways worth
    /// recording, because both look like one-line improvements.
    ///
    /// **Zeroing `head_dim` when the statement names none.** It is read by
    /// [`Rule::PerHead`], [`Rule::PerHeadElementwise`], [`Rule::GatedRms`],
    /// [`Rule::Rope`], [`Rule::RecurrentScan`], [`Rule::AxialRope`] and
    /// [`Rule::WarpTiledScan`], and not ONE of their statements sets
    /// `per_head_dim` — only `OpKind::RmsnormPerHead` does. Six of the seven
    /// reach [`headed`], which refuses a zero head width outright, so every
    /// rope, every per-head pad, every gated norm and every recurrence in the
    /// tree would answer [`Ungeometric::Empty`] and no fire would launch.
    /// [`Rule::PerHeadElementwise`] does not even get that far: it clamps the
    /// value into a block width, and `0.clamp(32, 256)` is 32, so it would
    /// launch a quarter-warp-wide block over a 256-channel head and stride
    /// `d += blockDim.x` past seven eighths of it — a fire that runs and
    /// writes a fraction of its output.
    ///
    /// **Filling `head_dim` with `width` when the statement names none.**
    /// The arithmetic is right — `width / width` is 1, so the grid is `rows`
    /// blocks and the multiple check passes trivially, and "one head spanning
    /// the row" is even a true sentence about a plain RMSNorm. It fails for a
    /// reason that has nothing to do with this rule: the binder fills one
    /// `Dims` per launch and cannot know which rule will read it, so every
    /// other reader in the list above would be handed a ROW width where it
    /// expects a HEAD width. `recurrent_scan` would allocate
    /// `2 * hidden * sizeof(float)` of shared memory and be refused for
    /// exceeding the cap; `rope` would compute `half = width / 2` and launch
    /// `BLOCK / half` — zero — pairs per block.
    ///
    /// **A second field.** Which is this, and it costs one `u32` per launch
    /// and one line per construction site.
    ///
    /// # What it is NOT: a change of mind about `head_dim`
    ///
    /// `bind/mod.rs:1312-1321` explains why the STATEMENT's head width wins
    /// over the fire's in [`Dims::head_dim`], and cites the defect
    /// `driver-metal`'s `stated_head.unwrap_or(geometry.head_dim)` records
    /// having had: gemma-4's two layer kinds disagree about the head width,
    /// so a grid that took the count from the fire and the width from nowhere
    /// describes neither layer. That stands, untouched. This field is a
    /// SECOND quantity — not "which head width", but "was one stated at all"
    /// — and the two answers differ exactly where `unwrap_or` erases the
    /// difference: a statement naming nothing and a statement naming the
    /// fire's own number produce the same `head_dim` and different
    /// `stated_head_dim`.
    ///
    /// # What fills it
    ///
    /// `driver-cuda`'s `jit_dims`, from `spec.per_head_dim`, **with no
    /// fallback at all** — `unwrap_or(0)` and nothing else. A fallback here
    /// would be the defect this field exists to fix, wearing a new name: a
    /// plain `Rmsnorm` of 2048 channels under 128-wide heads takes the
    /// present arm, opens `rows · 16` blocks instead of `rows`, and each one
    /// norms a whole row's `hidden` channels from a sixteenth of a row's
    /// offset. `width % head_dim == 0` holds, so nothing refuses; the launch
    /// runs, the tower answers, and the answer is fifteen sixteenths of it
    /// written over itself.
    ///
    /// A caller that is not `driver-cuda` states it the same way: the number
    /// its statement named, or zero. `model-loader`'s transforms name none.
    pub stated_head_dim: u32,
    /// Channels a partial rope rotates.
    ///
    /// Supplied by the statement. Its FIRST reader is [`Rule::MlaPrepare`],
    /// which computes MLA's head packing from `qk_rope_head_dim / 2` — and
    /// that is the same quantity under a second name, not a second meaning:
    /// MLA's `k_pe` is exactly the channels that rotate, and `rotate_partial`
    /// rotates exactly the channels this field counts. Nothing else here
    /// reads it, and the reason is worth knowing: CUDA's partial rotation is
    /// a DIFFERENT KERNEL — `rope/rope.cu`'s `rotate_partial`, launched
    /// `<<<num_tokens, 256>>>` over a flat per-token grid — so the rotary
    /// extent reaches it as an operand and never as a grid.
    /// [`Rule::Rope`]'s head-pair arithmetic is `rotate`'s, which rotates
    /// the whole head and therefore reads [`Dims::head_dim`].
    ///
    /// A zero is refused by its reader, not floored: `heads_per_block` is
    /// `BLOCK / (rotary / 2)` and a zero rotary divides by zero. An ODD one
    /// is refused too — a rotation turns PAIRS, and `rotary == 1` has no
    /// pair to turn.
    pub rotary_dims: u32,
    /// Experts the router scores. Read by [`Rule::RouterSort`], which sizes
    /// the counting sort's shared counters on it — three `int` arrays plus a
    /// scan's warp partials. Zero is a mixture with no experts.
    pub n_experts: u32,
    /// Experts each token routes to. Read by [`Rule::RoutedQmv`],
    /// [`Rule::RoutedQmvTransposed`] and [`Rule::RoutedQmvQuad`], which
    /// multiply it by the rows to get the routed slot count their grid
    /// counts. Zero is a mixture that routes nowhere, and it is ABSENCE
    /// rather than a value: `driver-cuda`'s `jit_dims` fills it from
    /// `DispatchCtx::experts_per_token` (`bind/mod.rs:1431`) and leaves it
    /// zero for a fire with no mixture, so the rules refuse there rather
    /// than launching one expert's worth of one.
    pub experts_per_token: u32,
    /// Requests the fire covers — the CSR's `R`, the extent
    /// `qo_indptr` / `kv_page_indptr` / `kv_last_page_lens` are indexed by.
    ///
    /// **Why it cannot be [`Dims::rows`].** On a PREFILL those are two
    /// different numbers and the launcher this field exists for spells both
    /// in one `dim3`: `attention_naive_paged.cu:108` is
    /// `dim3 grid(num_requests, total_tokens, num_q_heads)`. `rows` is
    /// `total_tokens` for that statement — `table::attn`'s
    /// `attention_naive_paged` row says `total_tokens: I32 <- Source::Rows`
    /// on the line above `num_requests: I32 <- Source::Attn("num_requests")`
    /// — so a rule that read `rows` for `grid.x` would launch a square grid
    /// of `total_tokens` by `total_tokens` where the launcher launches
    /// `R` by `total_tokens`, and on a 4-request 512-token fire that is 128x
    /// the blocks with every extra one indexing `qo_indptr` past its end.
    /// The decode form genuinely has one token per request and is
    /// [`Rule::PagedScoresDecode`], which reads `rows` and says so.
    ///
    /// **Nothing fills it today** — no longer true, and the correction is
    /// worth more than the sentence it replaces. It was a REACHABILITY fact:
    /// the request count lives on `driver-cuda`'s `AttnCtx`, and `jit_dims`
    /// is a nested `fn` inside `dispatch_generated` whose call sites are
    /// emitted by `crate::emit` with a fixed argument list that carried no
    /// attention context. `dispatch_generated` HAD the context all along —
    /// it takes `attn: Option<&AttnCtx>` for the operand side — so closing
    /// the gap was widening the emitted list by one argument, which
    /// `abi::emit_rust_dispatch` now does: `bind/mod.rs:1461` is
    /// `requests: attn.map_or(0, |a| extent(a.num_requests))`.
    ///
    /// A fire with NO attention context still leaves it zero, and a zero is
    /// ABSENCE — every reader refuses on it, which is
    /// [`Dims::experts_per_token`]'s position exactly. What was NOT adopted
    /// is filling it from `rows`: a field that is present and wrong is worse
    /// than one that is absent, and the paragraph above counts what the
    /// substitution launches.
    pub requests: u32,
    /// AltUp residual streams — `K`, the rank of the parallel residual the
    /// gemma-3n block predicts and corrects.
    ///
    /// **Why it cannot be [`Dims::kv_heads`].** `kv_heads` is filled from
    /// `ctx.num_kv_heads` and means "the head count of the tensor this
    /// launch addresses". A stream is not a head: it has no query partner,
    /// no page, and its count comes from the model's AltUp rank rather than
    /// from its attention. [`Rule::WarpTiledScan`] already reads `kv_heads`
    /// for `grid.y`, so a fire that borrowed that field for a stream count
    /// would launch `num_kv_heads` streams — eight where the model has four,
    /// or four where it has eight — and the kernel's own `k >= K` guard
    /// turns the first into silently skipped blocks and the second into a
    /// correct-looking answer over half the streams. §22's rule applies
    /// directly: `Dims` may not carry a number that means one thing to the
    /// fire and another to the rule.
    ///
    /// **Why it cannot be [`Dims::n_experts`].** That is the router's
    /// vocabulary, read by [`Rule::RouterSort`] to size shared counters, and
    /// the two numbers coexist in a gemma-3n fire. Reusing it would make
    /// "the AltUp rank sizes the counting sort" spellable.
    ///
    /// Unlike [`Dims::requests`] this one IS filled: `DispatchCtx` carries
    /// `altup_streams` and `table::norm`'s AltUp rows already read it as
    /// `Source::Ctx("altup_streams")`, so `jit_dims` can hand it over
    /// without reaching for anything it does not have. Zero is absence and
    /// [`Rule::AltUpStreams`] refuses it.
    pub altup_streams: u32,
}

/// A launch, in CUDA's spelling: blocks, threads per block, dynamic shared
/// bytes.
///
/// `grid` is BLOCKS and `block` is THREADS, which is the one place a reader
/// coming from the Metal side has to stop: `dispatchThreads` takes a total
/// thread count and `cuLaunchKernel` takes a block count, so the same rule
/// produces numbers that differ by a factor of the block width. Writing one
/// where the other is meant launches `grid.x` threads instead of `grid.x`
/// blocks — a real fire, a real result, and every row past the first
/// untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// Blocks per axis.
    pub grid: [u32; 3],
    /// Threads per block per axis.
    pub block: [u32; 3],
    /// Dynamic shared memory, in bytes.
    pub smem: u32,
}

/// Why a rule could not produce a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ungeometric {
    /// The row states no rule, so nothing can be dispatched from it. Drift,
    /// not a runtime condition — the same meaning `Source::Unbound` has for
    /// an operand.
    Unstated,
    /// The rule is real and this backend has not ported its arithmetic.
    ///
    /// Distinct from [`Ungeometric::Unstated`] because they are different
    /// bugs: an unstated row is a table that has not been filled in, and an
    /// unported rule is a driver that has not caught up with one. Only the
    /// second is fixed here.
    Unported(Rule),
    /// A launch over an empty extent.
    ///
    /// Refused rather than clamped. A zero grid launches nothing and returns
    /// success, so a fire whose rectangle collapsed would look exactly like a
    /// fire that ran — which is the failure `program::run::launch` already
    /// refuses for the PTIR lane, for the same reason.
    Empty,
}

/// Threads per block for the pointwise passes.
///
/// 256 because that is what every launcher in the deleted `norm/altup_aux.cu`
/// used, and the port's first duty is to reproduce those launches rather than
/// to improve on them. It is a tuning constant with one reader, which is the
/// shape a tuning constant should have.
const BLOCK: u32 = 256;

/// The widest block CUDA will launch.
const MAX_BLOCK: u32 = 1024;

/// Threads per warp — the unit `block_sum`'s shared scratch is counted in.
const WARP: u32 = 32;

/// One `float`, in bytes. Spelled once because six of the rules below turn a
/// count of them into a shared-memory size, and `sizeof(float)` written out
/// six times is six places for a `2` to become a `4`.
const FLOAT: u32 = 4;

/// The pad/strip block, `attn/head_dim_pad.cuh`'s `kPadBlock`.
///
/// A separate constant from [`BLOCK`] because it is not a taste: those two
/// kernels stride `d += kPadBlock` — the COMPILE-TIME constant, not
/// `blockDim.x` — so a block narrower than 128 leaves every column from
/// `blockDim.x` to 127 of each head unwritten, which for `pad_head_dim` is
/// padding that was never zeroed and for `strip_head_dim` is a head whose
/// tail keeps whatever the destination held. Neither fails; both answer.
const PAD_BLOCK: u32 = 128;

/// The narrowest and widest block `attn_sink_rescale`'s launcher will build
/// from a head width: `(head_dim < 32) ? 32 : (head_dim > 128 ? 128 : head_dim)`.
///
/// The floor is a whole warp, because a block of fewer lanes than a warp
/// wastes the scheduler slot it was issued in. The cap is the launcher's
/// measured choice and it is safe to exceed a head with it only because the
/// kernels stride by `blockDim.x` — see [`per_head_elementwise`].
const SINK_BLOCK_MIN: u32 = WARP;
/// See [`SINK_BLOCK_MIN`].
const SINK_BLOCK_MAX: u32 = 128;

/// The counting sort's block, `moe/moe_dispatch.cu`'s `BS`. One block, and
/// as wide as CUDA will let it be, because the scan is block-wide.
const SORT_BLOCK: u32 = MAX_BLOCK;

/// The softmax router's block, `moe/topk_softmax.cuh`'s `kSoftmaxBlock`.
///
/// Sixty-four, and the header does not merely prefer it —
/// `static_assert(kSoftmaxBlock == 64, "block_argmax folds exactly one upper
/// warp")`. A rule that widened this to [`BLOCK`] would compile, launch, and
/// fold a warp that was never written.
const ROUTER_BLOCK: u32 = 64;

/// The largest half-head `rope/rope.cu` will cache sin/cos pairs for —
/// `kMaxCachedPairs`, whose comment reads *"32 KB caps the table at head_dim
/// 8192; past that the pairs are recomputed."*
const ROPE_MAX_CACHED_PAIRS: u32 = 4096;

/// The recurrence's block, `ssm/gated_delta_net.cu`'s `constexpr int BLOCK`.
///
/// A separate constant from [`PAD_BLOCK`], which is also 128, because the two
/// are the same number for opposite reasons and a shared name would invite
/// one to be tuned for the other. `kPadBlock` is a CONTRACT — the pad strides
/// by the compile-time constant, so a narrower block leaves columns
/// unwritten. This is a MEASUREMENT: `recurrent_step` strides every loop by
/// `blockDim.x` and sizes its shared arrays on `K_d`, so a different width
/// computes the same bytes at a different speed. It is reproduced because a
/// port that changes the arithmetic cannot be A/B'd against the arithmetic it
/// replaced — the standard [`rms`] already holds for its 256.
const SCAN_BLOCK: u32 = 128;

/// The prefill convolution's block, `ssm/causal_conv1d.cu`'s
/// `constexpr int BLOCK = 64` in `prefill_dispatch`.
///
/// Sixty-four threads striding a channel's tokens, and the width is the
/// launcher's alone: the kernel's every loop is `t += blockDim.x` and its
/// only cross-thread step is a `__syncthreads()` before thread zero writes
/// the state tail. Narrower is legal and slower; wider is legal and idle
/// past `N`. Reproduced rather than improved, for [`SCAN_BLOCK`]'s reason.
const CONV_BLOCK: u32 = 64;

/// Elements one `bf16_to_narrow` load moves — `quant/dequant_wna16.cu`'s
/// `const long long n_vec8 = n / 8;`.
///
/// A device-side vector width reaching a grid, which is unusual enough to
/// name: the kernel's fast path reads `__nv_bfloat162` four at a time, so the
/// launcher's unit of work is eight elements and its grid is stated in those.
/// [`slab`] is the only rule that divides by it.
const SLAB_VEC: u32 = 8;

/// The grid cap [`slab`] launches under — `std::min<long long>(..., 1024)`.
///
/// Spelled apart from [`MAX_BLOCK`], which is the same number about a
/// different axis: that one is the widest BLOCK a launch may ask for and this
/// one is the most BLOCKS a stride loop is given. They are equal today and
/// nothing keeps them so, and a rule that read the block cap for a grid would
/// follow the next person who tunes a block width.
const SLAB_GRID_MAX: u32 = 1024;

/// The square tile [`tile16`] walks a rectangle in — `dim3 B2(16,16)`, which
/// all three vision towers declare on one line above their launches.
///
/// 256 threads like every other block here and arranged in two dimensions,
/// which is the whole distinction: the kernels index a matrix with
/// `threadIdx.y` for the row and `threadIdx.x` for the column, so the same
/// thread count spelled `[256, 1, 1]` gives every one of them row zero.
const TILE: u32 = 16;

/// Warps a [`warp_tiled_scan`] block splits the value channels over —
/// `ssm/gated_delta_net.cu`'s `constexpr int WARPS = 4;`.
///
/// The block is `WARPS * 32` and each warp owns its own slice of `V_d`, so
/// this number divides the third grid axis and multiplies the block. Changing
/// one without the other launches a grid that covers the value width with
/// blocks that do not.
const SCAN_WARPS: u32 = 4;

/// The block [`per_row_narrow`] launches — `vision/gemma4_audio.cu`'s literal
/// `128` at both SSCP layernorms.
///
/// A fourth 128 in this file, spelled apart from [`PAD_BLOCK`] and
/// [`SCAN_BLOCK`] for the reason all three are named at all: they are the same
/// number about three different contracts. The pad's is a compile-time stride,
/// the scan's is a measurement, and this one is a FOLD ORDER — the kernel sums
/// `(blockDim.x + 31) / 32` warp partials serially in thread zero, so a rule
/// that borrowed either of the others would change an answer's last bit
/// wherever the widths differ.
const LAYERNORM_BLOCK: u32 = 128;

/// The reference paged attention's block — `attn/attention_naive_paged.cu`'s
/// `constexpr int BLOCK = 128;` at `:35`.
///
/// **That file is DELETED and the 128 is not.** It is cited here in the past
/// tense the way [`crate::families::attn`] and `kernels-cuda/csrc/
/// CMakeLists.txt` cite it: the value was read off the ahead-of-time launcher
/// while the launcher existed, which is what §17.6 asks, and a measurement
/// does not expire because its witness was retired. What is lost is the
/// CHECK — nothing compares this constant to a `.cu` line any more — so the
/// oracle moved to the kernel, which is where it can still bite:
/// `kernels-cuda-new/tests/launch_rules.rs` quotes `template <int BLOCK>` and
/// the `reduce = smem + head_dim` partition out of
/// `attn/attention_naive_paged.cuh`, and the instantiation string carries the
/// 128 into NVRTC, which rejects a disagreement. Provenance is `git log
/// --follow crates/kernels-cuda/csrc/src/attn/attention_naive_paged.cu`.
///
/// A FIFTH 128 in this file, and it is named apart from [`PAD_BLOCK`],
/// [`SCAN_BLOCK`] and [`LAYERNORM_BLOCK`] for the reason all of them are: the
/// same number under four contracts. This one is a SHARED-MEMORY CONTRACT and
/// the only one of the four that reaches a byte count — `smem = (head_dim +
/// BLOCK) * sizeof(float)` and the kernel cuts the tail of that allocation
/// into exactly `BLOCK` reduction slots. Change the block without changing
/// the allocation and the reduction reads past what was asked for; change the
/// allocation without the block and it reads slots nothing wrote. It is also
/// the template argument both kernels are instantiated at, so the number is
/// pinned in three places at once.
const PAGED_BLOCK: u32 = 128;

/// AltUp's block — the deleted `csrc/src/norm/altup.cu`'s
/// `constexpr int BLOCK = 128;`, which was spelled once above both launchers
/// and is now spelled only here, because §43 took the file.
///
/// A SIXTH 128, and the one that is purely a TILE: it divides the third grid
/// axis and nothing else. The kernels stride nothing and share nothing — one
/// thread owns one `(t, k, h)` cell — so a different width computes the same
/// values in the same order over a differently shaped grid. It is named
/// rather than borrowed from [`SCAN_BLOCK`] because a tile that divides a
/// grid axis and a block that sizes a register slab are two decisions, and
/// the one thing a shared constant would guarantee is that tuning either
/// silently retiles the other.
const ALTUP_BLOCK: u32 = 128;

/// One block per row, [`BLOCK`] wide, with scratch for the warp combine.
///
/// The width is fixed rather than sized on the row because the reduction is
/// ORDER-SENSITIVE: `block_sum` folds warp by warp, so a different block
/// width sums the same values in a different order and answers with a
/// different last bit. Sizing it on `width` is the obvious improvement and it
/// is deliberately not taken here — a port that changes the arithmetic cannot
/// be checked against the arithmetic it replaced.
fn rms(rows: u32) -> Launch {
    Launch {
        grid: [rows, 1, 1],
        block: [BLOCK, 1, 1],
        // `block_sum` writes one float per warp and reads them back from
        // lane 0 of the first. Sizing this on anything but the block width
        // is a race the hardware does not report.
        smem: (BLOCK / WARP) * 4,
    }
}

/// One block per row PER HEAD — [`rms`]' grid with the per-head reading of
/// the same launcher folded in, [`BLOCK`] wide, nothing shared.
///
/// `norm/rmsnorm.cu`, `rmsnorm_gated_bf16`:
///
/// ```text
/// constexpr int BLOCK = 256;
/// dim3 grid(num_rows);
/// dim3 block(BLOCK);
/// device::rmsnorm_gated<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
///     ..., num_rows, hidden, eps);
/// ```
///
/// and `rmsnorm_strided_bf16` at `85-98` — which `rmsnorm_bf16` at `38-44`
/// forwards to — `rmsnorm_gemma_bf16` at `259-271`, `rmsnorm_no_scale_bf16`
/// at `283-285` and `rmsnorm_gated_fp32_in_bf16` at `296-298`: five
/// launchers, one grid.
///
/// **`num_rows` is not `dims.rows` for these five, and that is the whole
/// rule.** The C++ takes the row count as an ARGUMENT, and `table/norm.rs:36`
/// spells what the caller passes:
///
/// ```text
/// num_rows <- IfPresent(PerHeadDim,
///                       Mul(Rows, Div(Width(In(0)), PerHeadDim)),
///                       Rows)
/// hidden   <- IfPresent(PerHeadDim, PerHeadDim, Width(In(0)))
/// ```
///
/// Two ops lower to one symbol — `OpKind::Rmsnorm` norms `rows` rows of
/// `width`, `OpKind::RmsnormPerHead` norms `rows * (width / head_dim)` rows of
/// `head_dim` — so a backend that computes its own grid has to compute the
/// conditional the argument used to carry. This is that conditional, in the
/// only two terms a launch has. [`rms`] states the second arm alone and norms
/// a whole q projection as one row.
///
/// **The head width read here is [`Dims::stated_head_dim`] and never
/// [`Dims::head_dim`]**, which is why this is the one head-shaped rule that
/// does not call [`headed`]: zero is `Source::IfPresent`'s false arm reaching
/// a rule instead of a binder, and refusing it would refuse every plain
/// RMSNorm in the tree. [`Dims::head_dim`] cannot answer this question at
/// all — a binder with no statement to read fills it from the fire's
/// attention configuration, so its zero never arrives and its non-zero says
/// nothing about what was stated. See that field for the two one-line fixes
/// that look like they would work and what each of them breaks.
///
/// A `width` that is not a multiple of the stated head width is refused
/// rather than rounded. The kernel norms `hidden = head_dim` channels from
/// `row * head_dim`, so a rounded-up grid runs its last block off the end of
/// the row and a rounded-down one leaves a head unnormalised — and neither is
/// a shape a statement can have meant. This is the refusal the constraint
/// *"a statement that names a per-head width the row's operands contradict
/// must decline by name"* asks for: the statement said `head_dim`, the
/// operands said `width`, the two do not divide, and `Error::Geometry`
/// carries the symbol that could not be launched.
///
/// **Both arms are the launcher's, at the launcher's numbers.**
/// `tests/launch_rules.rs` holds them against `norm/rmsnorm.cu` and
/// `tests/rows_per_head.rs` fires them: at hidden 2048 over 128-wide heads a
/// stated 128 gives `rows · 16` blocks of a 128-channel norm and an absent
/// one gives `rows` blocks of a 2048-channel norm, and the second is what
/// this rule used to be unable to say. Under the old reading — the fire's
/// `head_dim` standing in for a statement that named none — the plain norm
/// took the per-head arm and opened SIXTEEN TIMES the blocks, each norming a
/// whole row's width from a sixteenth of a row's offset;
/// `the_absent_arm_is_not_the_fires_head_width` reproduces that grid on
/// demand so the defect stays measurable rather than remembered.
fn rows_per_head(rows: u32, width: u32, stated_head_dim: u32) -> Result<Launch, Ungeometric> {
    let blocks = if stated_head_dim == 0 {
        rows
    } else {
        if width == 0 || !width.is_multiple_of(stated_head_dim) {
            return Err(Ungeometric::Empty);
        }
        rows.checked_mul(width / stated_head_dim).ok_or(Ungeometric::Empty)?
    };
    Ok(Launch { grid: [blocks, 1, 1], block: [BLOCK, 1, 1], smem: 0 })
}

/// Flat pointwise: `n` elements, [`BLOCK`] per block, rounded up.
fn elementwise(n: u32) -> Launch {
    Launch {
        grid: [n.div_ceil(BLOCK), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// Flat pointwise over what the launch READS — `rows * in_width` elements,
/// the row folded into the index.
///
/// `norm/dsv4_hc.cu`, `hc_post_bf16` and `hc_expand_bf16`, which carry the
/// same three lines:
///
/// ```text
/// constexpr int BLOCK = 256;
/// const long long total = static_cast<long long>(N) * hidden_size;
/// if (total <= 0 || hc_mult > device::MAX_HC_MULT) return;
/// const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
/// device::hc_post<device::bf16><<<grid, BLOCK, 0, stream>>>(...);
/// ```
///
/// [`elementwise`]'s arithmetic at the other width. Both kernels read
/// `[N, H]` and write `[N, M, H]` — a hyper-connection expand and the
/// collapse that scatters a layer's output back across `M` residual streams —
/// and both give a thread one INPUT element, looping `M` writes over it. So
/// `total` is the input rectangle, the guard is `if (idx >= N * H) return;`,
/// and the extent the rule states is the extent the kernels bound themselves
/// by.
///
/// **[`Rule::Elementwise`] is not merely a coarser answer here.** Sized on
/// [`Dims::width`] — the last output's row width, `M * H` — it launches `M`
/// times the blocks, and every thread past `N * H` hits the guard and
/// returns. Correct, and up to eightfold: `MAX_HC_MULT` is 8, so a 16 × 2048
/// expand issues 1 024 blocks to do 128 blocks of work. `norm/dsv4_hc.cuh`
/// refused it in those words before this rule existed — *"a rule that has to
/// be wrong to be right is not the rule"* — and the file's discipline says
/// the same thing from the other end: a rule whose blocks are mostly thrown
/// away by a bounds check cannot be checked against the `<<<>>>` it claims to
/// reproduce, because the two agree on the answer and differ on the launch.
///
/// The direction that does not announce itself is the other one. Were this
/// stated on `width` for a pass whose output is NARROWER than its input, the
/// grid would cover a fraction of the elements the kernel indexes and the
/// tail would keep whatever the previous layer left there — a residual stream
/// that silently stops being updated. Reading `in_width` is what makes the
/// rule's extent the kernel's own rather than a coincidence of which operand
/// happens to be wider.
///
/// [`split_packed`] is the near neighbour and the reason this is a separate
/// variant rather than that one: it also sizes on `in_width`, and it puts the
/// row on `grid.y` and gives `grid.x` one row's worth of blocks. Stated here
/// that grid covers `1 / N` of the elements — one row's worth — and runs `N`
/// duplicate copies of it.
///
/// `ssm/nemotron_h.cu`'s `mamba_split` is this launch to the digit —
/// `total = N * projection_dim`, `<<<(total + 255) / 256, 256>>>` — and stays
/// refused one level up: `nemotron_mamba_split_bf16` fires
/// `mamba_split_conv_dt` over `N * (conv_dim + num_heads)` instead whenever
/// `gate == nullptr`, which is neither width. A row would state one arm of a
/// decision the host makes on a pointer, and the geometry landing does not
/// change that.
fn elementwise_in(n: u32) -> Launch {
    Launch {
        grid: [n.div_ceil(BLOCK), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// `ceil(rows / 256)` blocks of [`BLOCK`] — ONE THREAD per row.
///
/// `moe/dsv4_routing.cu`, `hash_route_lookup_bf16`, at `39-40` of the file as
/// it stood when DELETED (the citation used to read `57-60`, before §43.9
/// took the sibling router's launcher out from above it):
///
/// ```text
/// // One thread per token, not one block: the kernel's whole body is a table
/// // read and a K-long gather.
/// const int grid = (tokens + kDsv4Block - 1) / kDsv4Block;
/// device::hash_route_lookup<device::bf16><<<grid, kDsv4Block, 0, stream>>>(...)
/// ```
///
/// with `kDsv4Block = 256`. The comment above the launch is the launcher's
/// own and it is the rule: a body with no reduction in it costs a thread, and
/// a block each would launch 256 times the blocks and idle 255 lanes of every
/// one.
///
/// **Every other flat rule here multiplies the rows by a width before
/// dividing, and that is the confusion this exists to prevent.**
/// [`elementwise`] over the same rectangle launches `rows * width / 256`
/// blocks — `width` times too many — and each of them indexes a token id at
/// `blockIdx.x * 256 + threadIdx.x`, so the overshoot is not idle work but
/// `width - 1` blocks per row reading a routing table off the end of itself.
/// [`per_row`] is the other neighbour and errs the other way: one block per
/// row is 256 times the blocks a thread-per-row grid asks for, correct only
/// because these kernels bounds-check their own index.
///
/// Three more launchers were this shape to the digit, and one pair has since
/// gone: `layout/geometry.cu`'s `derive_kv_len` and `resolve_slot_to_block`,
/// both `<<<(n + kThreads - 1) / kThreads, kThreads>>>` with
/// `kThreads = 256`, were deleted with their file (§43) — they had no row, so
/// no shim entry, and no C++ caller either. They stayed rowless for a reason
/// that was never geometry: they are composed by the driver from no
/// statement, so there was nothing to state a rule ON, and that is also why
/// deleting them cost no row. `layout/geometry.cuh` still holds both
/// `__global__`s, so the shape above is still readable there.
/// The pair that remains is `quant/quant_bf16_to_fp8.cu`'s
/// `absmax_to_scale_inv` at `94-96` and `149-151`,
/// `<<<(rows + BLOCK - 1) / BLOCK, BLOCK>>>`.
///
/// # The one row this rule has, and where its host went
///
/// `moe::hash_route_lookup_dev` in `families::moe` was the only real member,
/// and BOTH are gone now: §5 step 5 took the family into fn-world, where
/// `x::moe::hash_route_lookup` is the host program and states these three
/// numbers itself from the same `<<<>>>` `fire::dsv4_routing.rs` did. The
/// `_dev` suffix went with them — it existed because a symbol may not be
/// both walked and unit-hosted, and a `unit!` kernel with a host `fn` beside
/// it is neither. The rule STAYS in the vocabulary rather than being retired
/// with its last row. A rule and a Rust launcher stating one
/// rectangle is a check and not a duplication — either can be read against
/// the other, which is exactly what nothing could do while the transcription
/// lived only in a `.cu` nobody diffed against it.
fn rows_flat(rows: u32) -> Launch {
    Launch {
        grid: [rows.div_ceil(BLOCK), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// A capped grid-stride slab — `min(ceil(units / 256), 1024)` blocks of
/// [`BLOCK`], where `units` is the count of [`SLAB_VEC`]-wide loads.
///
/// `quant/dequant_wna16.cu`, `bf16_to_fp16`, at `63-75`:
///
/// ```text
/// constexpr int BS = 256;
/// const long long n = static_cast<long long>(count);
/// const long long n_vec8 = n / 8;
/// const long long units = n_vec8 > 0 ? n_vec8 : n;
/// const int blocks = static_cast<int>(
///     std::min<long long>((units + BS - 1) / BS, 1024));
/// device::bf16_to_narrow<__half><<<std::max(blocks, 1), BS, 0, stream>>>(
///     ..., n);
/// ```
///
/// The grid does not cover the extent and is not meant to. `bf16_to_narrow`
/// walks `for (i = tid; i < n8; i += gridDim.x * blockDim.x)` with a scalar
/// tail after it, so 1024 blocks compute a 100M-element cast in as many
/// passes as they need. The cap is the CONTRACT and not a tuning number a
/// backend may raise: it is the only rule here whose grid is smaller than its
/// work, and a kernel without the stride loop launched this way computes a
/// prefix and reports success.
///
/// **`ceil((n / 8) / 256)` and not `ceil(n / 2048)`.** The launcher truncates
/// to whole vec8 loads FIRST, so the two disagree wherever `n` is not a
/// multiple of 8 — at `n = 16385` the first is 8 blocks and the second 9 —
/// and the tail those loads leave is the scalar loop's, not another block's.
/// Reassociating the division is the obvious simplification and it launches a
/// block that has nothing to do at exactly the extents a test picks last.
///
/// `n < 8` takes the `units = n` arm: below one vector there is nothing to
/// truncate to, and the floor of one block is the launcher's `max(blocks, 1)`.
///
/// What it does NOT serve is `quant/quant_bf16_to_fp8.cu`'s `absmax_bf16` at
/// `40-42`, where `blocks_full = (n + BLOCK - 1) / BLOCK` is capped by a
/// ternary rather than a `std::min` and launched at `<<<blocks, BLOCK>>>` —
/// the same cap over UNVECTORISED elements. Same shape, different divisor, and one rule
/// serving both would launch an eighth of that kernel's grid: correct only
/// while its own stride loop is, which is a property of a kernel and not of
/// the rule that launched it.
fn slab(n: u32) -> Launch {
    let units = if n >= SLAB_VEC { n / SLAB_VEC } else { n };
    Launch {
        grid: [units.div_ceil(BLOCK).clamp(1, SLAB_GRID_MAX), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// Pointwise with the row on its own grid axis.
///
/// What a pass whose rows are not contiguous needs: `mean_streams` reads
/// `[K, T, H]` and writes `[T, H]`, so a flat index over the output would
/// have to be divided back into a row and a channel by the kernel. The row
/// axis is `grid.x` and the channel axis is `grid.y`, which is the same
/// shape `LaunchRule::ElementwiseRows` names on Metal.
fn elementwise_rows(rows: u32, width: u32) -> Launch {
    Launch {
        grid: [rows, width.div_ceil(BLOCK), 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// One block per row, as wide as the row, rounded up to a warp and capped.
///
/// The cap is safe only because the kernels stride: `unpack_predict_coefs`
/// walks `kk += blockDim.x`, so a block narrower than the row computes all
/// of it in several passes. Before the stride loop this cap would have
/// silently computed a prefix — see `altup_aux.cuh`.
fn route_rows(rows: u32, width: u32) -> Launch {
    Launch {
        grid: [rows, 1, 1],
        block: [width.div_ceil(WARP).max(1).saturating_mul(WARP).min(MAX_BLOCK), 1, 1],
        smem: 0,
    }
}

/// One block per row, a fixed [`BLOCK`] wide, nothing shared — the scatter
/// that strides its own row.
///
/// `attn/kv_paged.cu`, `write_kv_to_pages_bf16`:
///
/// ```text
/// constexpr int BLOCK = 256;
/// const int launch_tokens = total_tokens - first_token;
/// if (launch_tokens <= 0) return;
/// device::write_kv<true><<<launch_tokens, BLOCK, 0, stream>>>(...);
/// ```
///
/// The row is `grid.x` and it is the only axis: `write_kv` reads
/// `const int t = blockIdx.x + first_token;` as its token and then strides
/// the destination cell, `const long long row = h_kv * d;` followed by
/// `for (int i = threadIdx.x; i < row; i += blockDim.x)`. Nothing is reduced,
/// nothing is shared, and the block width is free — which is exactly why this
/// is a rule and not a reading of an existing one.
///
/// **[`rms`] is this grid and this block and means something else.** Its 256
/// is a fold's contract and its 32 bytes are the fold's scratch, and a
/// scatter that borrowed the name would inherit both: the day a backend takes
/// the improvement [`rms`]'s own doc declines — sizing the block on the row
/// width — every KV write follows it, and a write whose block width came from
/// a norm's tuning is not a bug anything reports. `attn/kv_paged.cuh` reached
/// the same answer from the other side and said so before this rule existed:
/// *"There is no rule for that shape... Naming either would be inventing a
/// rule under an existing name."* [`route_rows`] is refused for the plainer
/// reason that it would launch a block sized on the row.
///
/// Six write forms and two more launchers are this shape to the digit:
/// `write_kv` again at `<<<n_max, 256>>>` for the device-window entry,
/// `write_kv_at_positions<HND>` at `<<<total_tokens, 256>>>`,
/// `write_kv_explicit<HND>` at `<<<B, 256>>>`,
/// `write_kv_explicit_devwin<HND>` at `<<<n_max, 256>>>`,
/// `copy_kv_cells<HND>` at `<<<N, 256>>>`, `write_kv_fp8_per_tensor` at
/// `<<<total_tokens, 256>>>`, and `attn/attention_naive.cu`'s
/// `mtp_shift_hidden` at `<<<total_tokens, 256>>>` — which states
/// [`Rule::Rms`] today and takes 32 bytes it never reads.
///
/// What it does NOT serve is the same file's `mtp_update_pending_hidden`,
/// `<<<num_requests, 256>>>`. A request count is not a row count, and a rule
/// that opened this grid over it would run one block per token against a
/// buffer with one slot per request. That launcher is [`Rule::PerRequest`]
/// now, which is this grid over [`Dims::requests`].
///
/// The two rows in `attn/page_compact.cuh` have that launcher's TEXT —
/// `<<<num_requests, kBlock>>>` at `attn/page_compact.cu:45` and `:48` — and
/// keep THIS rule, which looks like an inconsistency and is the opposite of
/// one: `dsl::cuda::compact_page_csr` records `Shape(vec![Dim::Requests])`,
/// so for that statement [`Dims::rows`] IS the request count and moving them
/// would trade a number that is always right for one that is zero without an
/// attention context. [`per_request`] carries the full argument.
///
/// **This unblocks a geometry, not yet a row, for the six KV forms.** Each of
/// them is `template <bool HND_LAYOUT>` and each launcher picks the arm with
/// `if (hnd_layout)` on a value the host holds — a page-layout scheme, not an
/// extent — so those rows stay refused where they already were, on the
/// instantiation rather than on the launch. `write_kv_fp8_per_tensor` is
/// refused twice over, on a `__nv_fp8_interpretation_t` no `Ty` spells.
/// `mtp_shift_hidden` is the one that can move today, and moving it also
/// stops it claiming a reduction's rule.
///
/// `width` is deliberately unread. The destination row's extent reaches these
/// kernels as OPERANDS — `h_kv` and `d`, which the page geometry owns and the
/// fire's rectangle does not — so a rule that sized anything on
/// [`Dims::width`] here would be reading a number the launch does not use.
fn per_row(rows: u32) -> Launch {
    Launch { grid: [rows, 1, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// One block per row, [`LAYERNORM_BLOCK`] wide, nothing shared — [`per_row`]'s
/// grid at half its block.
///
/// `vision/gemma4_audio.cu`, both SSCP layernorms:
///
/// ```text
/// vd::k_layernorm_relu<bfd><<<T1*F1,128,0,S>>>(
///     D(c0cl),D(w.sscp0_norm),D(c0cl),T1*F1,C0,EPS);
/// ```
///
/// at `189`, and the same line over `T2*F2` and `C1` at `196`. The rectangle
/// is `[T*F, C]` — `k_layernorm_relu` reads `int r = blockIdx.x;` and strides
/// `C` channels from `r * C` — so the grid is a row count and the width is
/// the channel count the block folds.
///
/// **The block width is the whole rule and it is numerics, not tuning.**
/// `gemma4_audio.cuh:141` states it against this launcher: the fold sums
/// `(blockDim.x + 31) / 32` per-warp partials SERIALLY in thread zero, so 128
/// threads and 256 threads add the same values in a different order and answer
/// with a different last bit. Stating [`Rule::PerRow`] here is a launch that
/// runs, a tower that answers, and an encoder that is no longer the
/// checkpoint's — which is the `rmsnorm_residual_add_scale_rmsnorm` precedent
/// the same header cites, where a 512-wide scalar kernel could not take a
/// 256-wide rule for the same reason read the other way.
///
/// The 32 floats of `__shared__ float wm[32]` are STATIC and do not appear
/// here. A dynamic allocation is what a rule produces; a kernel's own static
/// arrays are the kernel's, and adding them to `smem` would hand the launch
/// bytes it never binds — the mistake [`rms`]'s 32 bytes look like and are
/// not.
fn per_row_narrow(rows: u32) -> Launch {
    Launch { grid: [rows, 1, 1], block: [LAYERNORM_BLOCK, 1, 1], smem: 0 }
}

/// **One block**, [`BLOCK`] wide, nothing shared — the grid is a literal.
///
/// `layout/slot_ops.cu:60-62`:
///
/// ```text
/// :60   constexpr int kThreads = 256;
/// :61   device::copy_if_valid_slot<<<1, kThreads, 0, stream>>>(
/// :62       src, dst, bytes, slot_ids, request);
/// ```
///
/// and `attn/kv_paged.cu:515-517`:
///
/// ```text
/// :515  if (R <= 0 || keep_pages <= 0) return;
/// :516  device::build_window_page_view<<<1, 256, 0, stream>>>(
/// :517      src_indices, src_indptr, keep_pages, dst_indptr, dst_indices, R);
/// ```
///
/// **The `1` is the host's, not a quotient, and the difference is a defect
/// waiting on a fixture.** [`rows_flat`] answers `ceil(rows / 256)`, which
/// IS `1` for every rectangle of 256 rows or fewer — so a row that reached
/// for it passes every small fixture and over-launches in production, the
/// shape §22.5 and §22.7 each measured once. [`per_row`] and [`route_rows`]
/// open one block per row: `copy_if_valid_slot` would then repeat one byte
/// copy `rows` times (idempotent, so RIGHT BY ACCIDENT), and
/// `build_window_page_view` would put `rows` blocks on one output CSR, each
/// writing `dst_indptr[0..=R]` from its own prefix.
///
/// **Nothing in [`Dims`] is read**, and that is the rule's content rather
/// than an omission. The rectangle reaches both kernels as an operand —
/// `bytes` and `R` — and both stride it inside the one block
/// (`slot_ops.cuh:64`'s `for (usize i = threadIdx.x; i < bytes; i +=
/// blockDim.x)`). A rule that consulted `Dims::rows` here would be reading a
/// number the launch does not use, and `eval` would then have a refusal to
/// make that the launcher never makes.
fn single() -> Launch {
    Launch { grid: [1, 1, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// [`single`] at ONE WARP — `<<<1, 32>>>`.
///
/// `attn/kv_paged.cu:532-535`:
///
/// ```text
/// :532  if (splits <= 0 || page_size <= 0) return;
/// :533  device::build_full_split_view<<<1, 32, 0, stream>>>(
/// :534      src_indptr, src_last_page_len, splits, page_size,
/// :535      dst_indptr, dst_indices, dst_last, src_indices);
/// ```
///
/// The kernel's body is `if (threadIdx.x != 0) return;` followed by a serial
/// walk over `splits` (`kv_paged.cuh:854`), so 31 of the 32 lanes exit
/// immediately and the launcher chose the smallest block that is a whole
/// warp. [`single`] would launch 256 threads where 32 are wanted: not a
/// wrong ANSWER for this kernel — the extra lanes take the same return — and
/// still not this launcher, which is the only thing a rule is judged by. The
/// pair is [`per_row`]/[`per_row_narrow`] again: two block widths, two
/// variants, no `Dims` field carrying a width that is not a fire's to state.
fn single_warp() -> Launch {
    Launch { grid: [1, 1, 1], block: [WARP, 1, 1], smem: 0 }
}

/// One block per REQUEST, [`BLOCK`] wide, nothing shared.
///
/// `attn/attention_naive.cu:174`:
///
/// ```text
/// :174  device::mtp_update_pending_hidden<bf16><<<num_requests, BLOCK, 0, stream>>>(
/// :175      static_cast<const bf16*>(target_hidden),
/// :176      static_cast<bf16*>(pending_hidden),
/// :177      qo_indptr, slot_ids, num_requests, hidden_size);
/// ```
///
/// and `attn/page_compact.cu:44-48`, twice:
///
/// ```text
/// :44   device::count_kept<device::kBlock>
/// :45       <<<num_requests, device::kBlock, 0, stream>>>(
/// :47   device::scan_and_scatter<device::kBlock>
/// :48       <<<num_requests, device::kBlock, 0, stream>>>(
/// ```
///
/// with `page_compact.cuh:114`'s `constexpr int kBlock = 256`.
///
/// # This is [`per_row`] with the other axis, and the two are not
/// interchangeable
///
/// [`Dims::requests`]' own doc makes the argument and these three launchers
/// are what it is about: **a request count is not a row count.** On a prefill
/// of 4 requests and 512 tokens `per_row` opens 512 blocks where the launcher
/// opens 4, and blocks `4..512` read `qo_indptr[r]`, `qo_indptr[r + 1]` and
/// `slot_ids[r]` off arrays with `num_requests + 1` and `num_requests`
/// entries. `mtp_update_pending_hidden` then writes `pending_hidden` at
/// `slot * hidden_size` for whatever `slot_ids` read past its end contains —
/// a scatter into another request's state, out of a buffer indexed by SLOT.
///
/// On a pure decode `total_tokens == num_requests` and the two rules are the
/// same launch, which is why the substitution survives every single-token
/// fixture and why the fire test below states a prefill shape.
///
/// **`count_kept` and `scan_and_scatter` state [`Rule::PerRow`] and KEEP it,
/// and that is the sharpest thing in this doc.** Three launchers, identical
/// text, and the verdict splits — because a rule reproduces a launcher only
/// through a fire, and the two page-compaction rows fire under a statement
/// whose rectangle IS the request count. `dsl::cuda::compact_page_csr`
/// records its result as `Shape(vec![Dim::Requests])`, `lower.rs:716`
/// resolves that to `n_requests`, the statement is `whole`, and
/// [`crate::x::attn::page_compact`] sets the chain out in full -- it was
/// `families::attn`'s `PAGE_COMPACT` when this was written, and the unit
/// crossed into fn-world with both rows. Moving those two here would make them read
/// [`Dims::requests`], which `jit_dims` fills from the ATTENTION context and
/// leaves zero without one — turning two rules that agree into one that
/// answers [`Ungeometric::Empty`]. `mtp_update_pending_hidden` goes the other
/// way for the mirror reason: its op records a `StateRef` and no result at
/// all, so its fire's rectangle is its INPUT's, `[Tokens, hidden]`.
///
/// So this variant is not a rule for one kernel in the sense §10.5 forbids —
/// it is the axis three launchers count on — but it hosts exactly one row
/// today, and that is stated here rather than left to be inferred.
fn per_request(requests: u32) -> Launch {
    Launch { grid: [requests, 1, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// One block per row, [`BLOCK`] wide, with one float of shared scratch per
/// ROW OF THE RECTANGLE — the sparse index network's top-k.
///
/// `attn/dsa_indexer.cu`, `dsa_index_topk_mask`:
///
/// ```text
/// if (tokens <= 0) return;
/// const std::size_t smem = static_cast<std::size_t>(tokens) * sizeof(float);
/// device::index_topk_mask<bf16><<<tokens, device::kBlock, smem, stream>>>(...);
/// ```
///
/// `kBlock` is 256, so the geometry is [`rms`]'s to the digit and the shared
/// allocation is the whole of the difference. `index_topk_mask` declares
/// `extern __shared__ float logit[]` and fills `logit[0..nkeys)` where
/// `nkeys = blockIdx.x + 1` — the causal prefix — so the buffer is one float
/// per KEY and every key is a token of this same fire. The `float` is
/// literal: the kernel is `template <class T>` over its bf16 operands and
/// accumulates in fp32 regardless, so the allocation is four bytes a row at
/// every instantiation and not a function of the row's element width.
/// `attn/dsa_indexer.cuh` states the consequence of getting the size from the
/// block instead: *"a launch that under-sizes shared memory does not fail, it
/// reads another block's floats."* At [`rms`]'s 32 bytes the last row of a
/// 4 096-token prefill selects its top-k from eight floats it wrote and 4 088
/// it did not.
///
/// **The rule does not cap it and the launcher does not either.** At four
/// bytes a row the request crosses the 48 KB default a little under 12 288
/// rows — a little, because `index_topk_mask` also declares three static
/// shared scalars that come out of the same budget — and `cuLaunchKernel`
/// then refuses with `CUDA_ERROR_INVALID_VALUE`. That is the one direction of
/// a shared-memory error which reports itself, and it is the reason a rule
/// may state this at all: clamping it here would turn a loud refusal into a
/// quiet wrong mask, which is the trade this whole module is arranged to
/// decline. A fire that wants longer prefills needs
/// `cudaFuncAttributeMaxDynamicSharedMemorySize` raised, which is a host
/// call and not a grid.
fn row_scores(rows: u32) -> Result<Launch, Ungeometric> {
    let smem = rows.checked_mul(FLOAT).ok_or(Ungeometric::Empty)?;
    Ok(Launch { grid: [rows, 1, 1], block: [BLOCK, 1, 1], smem })
}

/// One block per COLUMN, [`CONV_BLOCK`] wide, the rows walked inside the
/// block — the short causal convolution's prefill.
///
/// `ssm/causal_conv1d.cu`, `prefill_dispatch`:
///
/// ```text
/// if (N <= 0 || C <= 0 || K <= 0) return;
/// constexpr int BLOCK = 64;
/// dim3 grid(C);
/// dim3 block(BLOCK);
/// device::causal_conv1d_prefill<device::bf16, SILU><<<grid, block, 0, stream>>>(...);
/// ```
///
/// `C` is the rectangle's WIDTH — `x` and `y` are both `[N, C]` — so this is
/// the transpose of every other rule in the file: the grid is the row width
/// and the row count is a loop bound inside the block, `for (t = threadIdx.x;
/// t < N; t += blockDim.x)`.
///
/// **The token axis is not a grid axis and cannot be made one.** A channel's
/// output at token `t` reads inputs back to `t - K + 1`, and for the leading
/// `K - 1` tokens it reads them out of `state_out` — the same buffer the
/// block rewrites with the sequence's tail after a `__syncthreads()`. One
/// block per channel is what makes that read-then-write ordered. Spread the
/// tokens over blocks and the tail-writer races the prefix-readers: no fault,
/// no diagnostic, a convolution answered from a history another block had
/// already overwritten.
///
/// It serves both instantiations, which are two kernels rather than one with
/// a flag: `causal_conv1d_prefill_bf16` fires `<SILU=true>` and
/// `causal_conv1d_prefill_noact_bf16` fires `<SILU=false>` for gemma-4's
/// audio lconv1d. It does NOT serve the file's batched forms —
/// `causal_conv1d_prefill_batched` opens `(channel-tile, request)` off
/// `qo_indptr`, and a request count is not an extent [`Dims`] carries.
///
/// `rows` is read as a precondition only: a rectangle of no tokens launches
/// `C` blocks whose stride loop runs zero times, writes nothing, and reports
/// success — which is [`Ungeometric::Empty`]'s whole subject.
fn per_channel(width: u32) -> Launch {
    Launch { grid: [width, 1, 1], block: [CONV_BLOCK, 1, 1], smem: 0 }
}

// ── The head-shaped rules ────────────────────────────────────────────
//
// Everything below this line is what the migration was blocked on. `Dims`
// could not state a head count at all until this commit, so seven families
// extracted their device text, proved it NVRTC-clean, and then had nowhere
// to put the row: a `(requests, heads)` grid is not an approximation of a
// `(rows)` grid, it is a different launch. Each rule names the launcher it
// reproduces, because a rule that cannot be checked against a `<<<>>>` is a
// rule nobody can falsify.

/// One block per (head, row), 128 threads — the head-dim pad and its inverse.
///
/// `attn/head_dim_pad.cu`, `pad_head_dim` and `strip_head_dim`:
///
/// ```text
/// dim3 grid(num_heads, num_tokens);
/// dim3 block(BLOCK);            // device::kPadBlock == 128
/// device::pad_head_dim<bf16><<<grid, block, 0, stream>>>(...);
/// ```
///
/// **The head is `grid.x` and the row is `grid.y`**, which is the transpose
/// of every other head-shaped rule here — the kernels read `blockIdx.y` as
/// the token and `blockIdx.x` as the head, and a grid handed to them the
/// other way round runs the same number of blocks and addresses the wrong
/// cell in each. Nothing reports that; the tensor comes out fully written.
///
/// The block is [`PAD_BLOCK`] and not the head width, and that is the
/// kernels' requirement rather than a tuning number: both stride
/// `d += kPadBlock`, the constant, so a narrower block never visits the
/// columns above it. See [`PAD_BLOCK`] for what each of the two leaves
/// behind when that happens.
///
/// `head_dim` is read only as a precondition. A head of no channels makes
/// both loops execute zero times, so the launch runs `heads * rows` blocks
/// that write nothing and reports success — the shape of failure
/// [`Ungeometric::Empty`] exists to refuse.
fn per_head(rows: u32, heads: u32) -> Launch {
    Launch { grid: [heads, rows, 1], block: [PAD_BLOCK, 1, 1], smem: 0 }
}

/// One block per (row, head), as wide as a head — the per-head pointwise
/// passes.
///
/// `attn/attn_sink.cu`, `attention_sink_rescale_bf16`:
///
/// ```text
/// const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(num_q_heads));
/// const int block = (head_dim < 32) ? 32 : (head_dim > 128 ? 128 : head_dim);
/// device::attn_sink_rescale<bf16><<<grid, block, 0, stream>>>(...);
/// ```
///
/// The row is `grid.x` and the head is `grid.y` — `attn_sink_rescale` reads
/// `blockIdx.x` as the token and `blockIdx.y` as the head, and both are
/// bounds-checked against their operands, which is why an over-wide grid is
/// merely wasted and an over-wide BLOCK is not.
///
/// The clamp is what makes one rule serve two launchers.
/// `attn/dsv4_compress.cu`'s `combine_attn_outputs_bf16` is the same grid at
/// `clamp(head_dim, 32, 256)`, so this rule hands it half the threads on a
/// 256-wide head. It computes the same output: every loop in it is
/// `for (d = threadIdx.x; d < head_dim; d += blockDim.x)`, so a narrower
/// block makes two passes where the launcher made one. Slower on one kernel,
/// never wrong on either — and the reverse substitution would not be safe,
/// because a 256-wide block on the 128-clamped kernel is 128 lanes reading
/// `sinks[h]` for a head they then do not write.
///
/// What it does NOT serve is a kernel whose block width is a TEMPLATE
/// argument. `qwen_gdn_qk_norm<bf16, BLOCK>` fixes 128 inside the
/// instantiation and reduces through scratch sized on it; this rule answers
/// 128 only while `K_d` is 128, which is every Qwen3.5 GDN config measured
/// but is not a property of the rule. Such a kernel needs its width stated,
/// not derived.
fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    Launch {
        grid: [rows, heads, 1],
        block: [head_dim.clamp(SINK_BLOCK_MIN, SINK_BLOCK_MAX), 1, 1],
        smem: 0,
    }
}

/// One block per (row, head), 256 threads — the gated and per-head norms.
///
/// `ssm/nemotron_h.cu`, `zamba_rmsnorm_gated_bf16`:
///
/// ```text
/// constexpr int BLOCK = 256;
/// const int groups = hidden / group_size;
/// dim3 grid(N, groups);
/// device::zamba_rmsnorm_gated<<<grid, BLOCK, 0, stream>>>(...);
/// ```
///
/// Two hundred and fifty-six is not negotiable here and [`Rule::Rms`]'s is
/// not either, for the same reason at a different width: the kernel declares
/// `__shared__ float buf[256]` STATICALLY and folds it with
/// `for (off = blockDim.x / 2; off > 0; off >>= 1)`. A wider block indexes
/// past a static array — which the hardware does not report — and a block
/// that is not a power of two drops the odd lane out of the fold and
/// normalises by a sum that is missing a term. Finite, plausible, wrong.
///
/// Two more launchers are this grid to the digit and are unblocked by it:
/// `norm/dsv4_hc.cu`'s `attn_sink_correction_bf16` and `per_head_rmsnorm_bf16`,
/// both `dim3 grid(N, num_heads); dim3 block(256);`. `per_head_rmsnorm` is
/// the one that proves the head axis is load-bearing rather than convenient —
/// it reads `gridDim.y` for its head count, so a rule that dropped the axis
/// would not waste blocks, it would tell the kernel there is one head and
/// stride every row by one head's width.
///
/// `head_dim` is the group width and is read as a precondition only: a group
/// of no channels reduces nothing and `rsqrtf(0/0 + eps)` is a scale the
/// kernel then multiplies a row by.
fn gated_rms(rows: u32, heads: u32) -> Launch {
    Launch { grid: [rows, heads, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// One block per (row, value head), [`SCAN_BLOCK`] threads, with two key
/// heads' worth of floats shared — the gated delta-net recurrence.
///
/// `ssm/gated_delta_net.cu`, `gdn_recurrent_step_bf16`:
///
/// ```text
/// constexpr int BLOCK = 128;
/// dim3 grid(B, V_h);
/// dim3 block(BLOCK);
/// const int shmem_bytes = 2 * K_d * sizeof(float);
/// device::recurrent_step<device::bf16, true><<<grid, block, shmem_bytes, stream>>>(...);
/// ```
///
/// The row is `grid.x` and the head is `grid.y`, which is [`gated_rms`]'s
/// order and the transpose of [`per_head_elementwise`]'s — `recurrent_step`
/// opens `const int b = blockIdx.x; const int h = blockIdx.y;`.
///
/// **The two key heads are the contract; the 128 is not.** The kernel takes
/// `extern __shared__ float smem[]` and splits it `float* sq = smem; float*
/// sk = smem + K_d;` — a staging slab for the query and the key of this step,
/// each `K_d` floats, each filled by a `i += blockDim.x` stride. Every loop
/// in the body strides the same way, so the width is free and reproduced only
/// because it is what was measured. The ALLOCATION is not free, and the near
/// miss is the reason this rule exists rather than a widened old one:
/// [`Rule::Rope`] answers `head_dim * sizeof(float)` — EXACTLY HALF — so a
/// row that stated it would give `sk` the same bytes as `sq` and stage the
/// key over the query. Both arrays are written before either is read, so the
/// step consumes a key it just clobbered, the recurrent state absorbs it, and
/// every token after that one in the same fire inherits the result. No fault,
/// no NaN, a state that has quietly stopped being the model's.
///
/// Four device templates in the file are this launch to the digit, each in
/// two state storage types and two arms of the `KLast` layout flag — sixteen
/// instantiations, one geometry. `recurrent_step` is the decode step,
/// `recurrent_step_batched` is it over a slot table,
/// `recurrent_step_batched_gqa` folds `h_kv = h / (V_h / K_h)` inside the
/// same grid — a grouped-query read that costs the geometry nothing — and
/// `chunk_gated_delta_prefill_batched` counts requests on `grid.x` where the
/// step counts tokens, with the step's block and the step's shared size.
///
/// **The allocation is `sizeof(float)` and does not move with the row's
/// element.** `template <typename StateT, bool KLast>` names the STATE's
/// storage — fp32 or bf16 — while `q_norm`, `k_norm`, `v` and `out` are
/// `float*` in every instantiation and the staging slab is declared
/// `extern __shared__ float smem[]`. So the bf16 launcher computes the same
/// `2 * K_d * sizeof(float)` the fp32 one does, and a future reader tempted
/// to scale this by a row's `elem` width would halve the slab for exactly
/// the instantiation whose name says bf16 — which is `sk` written over `sq`
/// again, from the other direction.
///
/// **What it does not serve, and none of these is a widening.** The `_cached`
/// prefill wants `K_d * V_d * sizeof(float)` behind a `cudaFuncSetAttribute`
/// raising the 48 KB cap — the same grid, the same block, an allocation in a
/// head width [`Dims`] does not carry. The `_warp_tiled_gqa` pair opens a
/// third axis, `ceil(V_d / WARPS)`, takes no shared memory, and runs a
/// `WARPS * 32` block. `_fla` puts a value tile on `grid.x` and drops to a
/// 64-wide block; `_smem` stages the bf16 state slab and asks for
/// `K_d * BV * sizeof(bf16)` on top of the two float arrays. `_fused` adds
/// one float for a broadcast scalar.
///
/// Three of those are also reached through a host test — `_fla` and `_smem`
/// behind `PIE_QWEN35_GDN_FLA_STEP` and `PIE_QWEN35_GDN_SMEM_STEP`, `_fused`
/// behind a `constexpr bool` that is `false` today — so this rule states the
/// geometry of the kernel the launcher falls back to, which is the only one
/// of the set whose `<<<>>>` is a function of [`Dims`] alone. The env-gated
/// siblings are different kernels with different grids, and choosing between
/// them is a host decision that no rule should encode.
///
/// The family module's note that all fourteen recurrence kernels want
/// `2 * K_d * sizeof(float)` at 128 was measured against the four that do.
///
/// Zero heads and zero `head_dim` are refused together by [`headed`]: a
/// grid of no value heads runs nothing and reports success, and a shared
/// request of zero bytes hands `sq` and `sk` the same null offset.
fn recurrent_scan(rows: u32, heads: u32, head_dim: u32) -> Result<Launch, Ungeometric> {
    let smem = head_dim
        .checked_mul(2)
        .and_then(|floats| floats.checked_mul(FLOAT))
        .ok_or(Ungeometric::Empty)?;
    Ok(Launch { grid: [rows, heads, 1], block: [SCAN_BLOCK, 1, 1], smem })
}

/// The recurrence tiled by warps over the VALUE width — `dim3(rows, heads,
/// ceil(value_width / 4))` at [`SCAN_WARPS`] warps of threads, nothing
/// shared.
///
/// `ssm/gated_delta_net.cu`, `chunk_gated_delta_prefill_batched_warp_tiled_gqa`:
///
/// ```text
/// constexpr int WARPS = 4;
/// constexpr int BLOCK = WARPS * 32;
/// dim3 grid(R, V_h, (V_d + WARPS - 1) / WARPS);
/// dim3 block(BLOCK);
/// device::chunk_gated_delta_prefill_batched_warp_tiled_gqa<float, false>
///     <<<grid, block, 0, stream>>>(...);
/// ```
///
/// at `759-791`, where the `<float, false>` launch is one of two arms — a
/// `k_last` branch fires `<float, true>` at `784-785` on the SAME `grid` and
/// `block`, so the geometry is one and the instantiation is two, exactly as
/// [`axial_rope`]'s launcher fires one grid over `q` and then over `k`.
/// `..._state_bf16` at `816-850` carries the same five lines over
/// `__nv_bfloat16`. `grid.x` is the REQUEST count `R` — these
/// launchers read `qo_indptr` and each block walks one request's tokens — so
/// the rectangle a row states here is counted in requests, which is one of
/// the three things [`Dims::rows`] says it may be.
///
/// **The third axis is the first in this file.** [`recurrent_scan`] is the
/// same two leading axes and stops there, giving each block the whole value
/// width; this one cuts the value channels four ways so a block can hold its
/// slice of the state in registers. The missing shared allocation is the tell
/// that they are two rules and not one with a parameter: the scan's block
/// reads `2 * K_d` floats it must be GIVEN, and a warp-tiled launch that
/// inherited that allocation would be paying for staging it does not do.
///
/// **`V_d` is `width / kv_heads` and not [`Dims::head_dim`], deliberately.**
/// The output row of both launchers is `V_h * V_d` wide, so the value width
/// is a quotient of two fields whose meanings are already fixed, and taking
/// it from `head_dim` instead would read as `V_d` the field
/// [`recurrent_scan`] reads as `K_d`. Those are equal in every Qwen3.5 GDN
/// config measured and are not the same number — see [`Dims::head_dim`] on
/// the four names — and one field standing for both is exactly what keeps
/// `chunk_gated_delta_prefill_batched_cached` refused. A rule that needs only
/// the value width can ask for it in terms that cannot be confused.
///
/// The launcher's `if (K_d > 256 || V_h % K_h != 0) throw` is a GQA
/// precondition on values the host holds and not a geometry, so it does not
/// appear here; a fire that violates it has stated a shape the kernel does
/// not implement, which is a row's business rather than a grid's.
///
/// What it does NOT serve is the `_ilp2` pair in the same two functions,
/// `dim3 grid(R, V_h, (V_d + TILE_V - 1) / TILE_V)` with `TILE_V = WARPS * 2`
/// — a second kernel behind `qwen_gdn_gqa_ilp2_enabled()`, which is
/// `constexpr bool ... { return false; }` at `48`. A host constant chooses
/// it, so stating it would be a rule for a launch no build makes.
fn warp_tiled_scan(rows: u32, heads: u32, value_width: u32) -> Launch {
    Launch {
        grid: [rows, heads, value_width.div_ceil(SCAN_WARPS)],
        block: [SCAN_WARPS * WARP, 1, 1],
        smem: 0,
    }
}

/// One block per (query head, row), 256 threads, with the KV extent in
/// shared memory — the single-pass naive attention.
///
/// `attn/attention_naive.cu`, `attention_naive_bf16`:
///
/// ```text
/// dim3 grid(num_q_heads, num_tokens);
/// dim3 block(BLOCK);            // device::BLOCK == 256
/// const std::size_t shmem_bytes =
///     sizeof(float) * (static_cast<std::size_t>(num_tokens) + BLOCK);
/// device::attn_naive<bf16><<<grid, block, shmem_bytes, stream>>>(...);
/// ```
///
/// The head is `grid.x` here and the row is `grid.y` — the transpose of
/// [`per_head_elementwise`], and the kernel reads it that way.
///
/// **The shared allocation is the whole reason this rule could not be
/// approximated.** `attn_naive` lays out `scores[num_tokens]` followed by
/// `reduce_buf[BLOCK]` in one `extern __shared__` block and takes the second
/// pointer as `smem + num_tokens`. Launch it with less and the reduction
/// scratch overlaps the scores it is reducing: the kernel runs, the softmax
/// denominator is computed from bytes the same kernel is overwriting, and
/// the answer is finite. A rule that defaulted `smem` to zero would do that
/// on every fire.
///
/// The kernel strides by the constant `BLOCK` and not by `blockDim.x`, so
/// the 256 is a precondition exactly as [`PAD_BLOCK`] is.
///
/// `rows` is both the query count and the KV extent because this kernel
/// attends over the same tokens it is launched for — it is the unpaged form.
/// The paged siblings in the same file size their scratch on a page window
/// that is not in [`Dims`] at all, and they keep their launchers.
fn sdpa_vector(rows: u32, q_heads: u32) -> Result<Launch, Ungeometric> {
    let smem = rows
        .checked_add(BLOCK)
        .and_then(|floats| floats.checked_mul(FLOAT))
        .ok_or(Ungeometric::Empty)?;
    Ok(Launch { grid: [q_heads, rows, 1], block: [BLOCK, 1, 1], smem })
}

/// Pointwise over the launch's INPUT width with the row on `grid.y` — one
/// packed buffer taken apart into several.
///
/// `attn/split_packed.cu`, `split_qkv_bf16` and `split_qkv_bf16_devwin`:
///
/// ```text
/// const int max_dim = q_dim > kv_dim ? q_dim : kv_dim;
/// const int xblocks = (max_dim + BLOCK - 1) / BLOCK;   // BLOCK == 256
/// dim3 grid(xblocks, n_tokens);
/// device::split_qkv<bf16><<<grid, BLOCK, 0, stream>>>(...);
/// ```
///
/// **This rule's `grid.x` is deliberately WIDER than the launcher's**, and
/// `attn/split_packed.cuh` is where the licence for that was written down
/// before the port existed: *"`SplitPacked`'s grid over the INPUT width
/// (`q_dim + 2 * kv_dim`) is WIDER than the launcher's over
/// `max(q_dim, kv_dim)`, and the outputs are identical either way — every
/// loop below strides by `blockDim.x * gridDim.x` and bounds itself on its
/// own output width, so extra blocks contribute nothing but a shorter loop.
/// The port does not have to reproduce `max(q_dim, kv_dim)` to be correct."*
///
/// It sizes on `in_width` and not on `width` because the three outputs are
/// each a fraction of the work and no one of them spells the grid — which is
/// the entire distinction between this rule and [`Rule::ElementwiseRows`],
/// whose arithmetic is otherwise the same shape at the other width.
///
/// It is NOT the shape of `norm/dsv4_hc.cu`'s `hc_post` and `hc_expand`,
/// which also size on their input and are flat: `grid = (N * hidden_size +
/// 255) / 256`, one axis, the row folded in. Those two want
/// [`Rule::Elementwise`] read at the input width, which is a reading the
/// rule's own doc does not have, so they keep their launchers and are
/// reported as still blocked rather than approximated here.
fn split_packed(rows: u32, in_width: u32) -> Launch {
    Launch { grid: [in_width.div_ceil(BLOCK), rows, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// The rotation: rows on `grid.x`, packed heads on `grid.y`, a cached
/// sin/cos table in shared memory.
///
/// `rope/rope.cu`, `rope_bf16`:
///
/// ```text
/// constexpr int BLOCK = 256;
/// constexpr int kMaxCachedPairs = 4096;
/// const int half = head_dim / 2;
/// const int cache_pairs = half <= kMaxCachedPairs ? half : 0;
/// const device::usize smem = static_cast<device::usize>(cache_pairs) * 2 * sizeof(float);
/// const int total_heads = num_q_heads + num_kv_heads;
/// const int heads_per_block = half >= BLOCK ? 1 : (BLOCK / half);
/// dim3 grid(num_tokens, (total_heads + heads_per_block - 1) / heads_per_block);
/// device::rotate<false, false><<<grid, block, smem, stream>>>(...);
/// ```
///
/// Q and K rotate in ONE launch here, which is the structural difference
/// from Metal's rope and the reason this rule reads both head counts rather
/// than deriving one from `width / head_dim`: `rotate` walks a packed
/// `q_heads + kv_heads` head axis and splits it internally, so there is no
/// per-tensor dispatch for a grouped-query fire to get wrong. The defect
/// `driver-metal`'s `rope_heads` exists to prevent — k rotated over q's head
/// count — is not reachable through this launcher.
///
/// `heads_per_block` is also an OPERAND of the kernel, and the rule and the
/// binder must derive it from the same `head_dim` or the grid covers a head
/// count the kernel does not agree it has. Both read [`Dims::head_dim`],
/// which is what makes that a binding question rather than a geometry one.
///
/// The shared table is the sin/cos pairs for one head, and it is dropped
/// entirely past [`ROPE_MAX_CACHED_PAIRS`] — the kernel recomputes them when
/// `cache_pairs` is zero, so a zero `smem` here is a slower rotation and not
/// a wrong one. That is the only smem in this file whose zero is legitimate.
fn rope(rows: u32, q_heads: u32, kv_heads: u32, head_dim: u32) -> Launch {
    let half = head_dim / 2;
    let heads_per_block = if half >= BLOCK { 1 } else { BLOCK / half };
    let cache_pairs = if half <= ROPE_MAX_CACHED_PAIRS { half } else { 0 };
    Launch {
        grid: [rows, (q_heads + kv_heads).div_ceil(heads_per_block), 1],
        block: [BLOCK, 1, 1],
        smem: cache_pairs * 2 * FLOAT,
    }
}

/// One block per row, [`ROUTER_BLOCK`] wide — the router's top-k.
///
/// `moe/topk_softmax.cu`, `router_topk_softmax_bf16`:
///
/// ```text
/// device::router_topk_softmax<device::bf16><<<N, device::kSoftmaxBlock, 0, stream>>>(...);
/// ```
///
/// The row axis is `grid.x` and it is the whole point of the rule.
/// `LaunchRule::RouterLane`'s own doc records what its absence cost on the
/// other backend: *"with `grid.y = 1` a mixture prefill routed row 0 only,
/// and every other row's expert ids were whatever the last layer left
/// there."* CUDA never had that defect and gets the rule anyway, because a
/// vocabulary in which one backend's rules are the ones that were once wrong
/// is not a vocabulary.
///
/// The width does NOT scale with the expert count, which is where this
/// backend's arithmetic and Metal's part company. Metal sizes the
/// threadgroup on `n_experts` because its kernel gives each expert a lane;
/// `topk_softmax.cuh` gives each expert several and folds them, and its
/// `static_assert(kSoftmaxBlock == 64, "block_argmax folds exactly one upper
/// warp")` says a different width does not merely tune worse. The expert
/// bound is enforced by the launcher's own refusal — *"a wider router would
/// overrun the kernel's static shared arrays"* — and a value precondition is
/// not something a `LaunchRule` can carry.
///
/// So this serves the 64-wide routers and only those: `topk_softmax`,
/// `router_topk_softmax` and both `topk_sigmoid_bias` instantiations.
/// `moe/topk_sigmoid.cu`'s `topk_sigmoid` was 128 and `moe/dsv4_routing.cu`'s
/// `topk_sqrtsoftplus` 256 — same shape, different static arrays, and the
/// vocabulary has no way to say "this rule at that width". Both launchers
/// have since gone (§43): the rows are routed, so the shim held no entry for
/// either and nothing else called them, and the widths now live only in
/// `moe/topk_sigmoid.cuh` and `moe/dsv4_routing.cuh` where the JIT reads
/// them. This rule still does not carry them, for the same reason.
fn router_lane(rows: u32) -> Launch {
    Launch { grid: [rows, 1, 1], block: [ROUTER_BLOCK, 1, 1], smem: 0 }
}

/// ONE block, whatever the rows, with the sort's counters in shared memory.
///
/// `moe/moe_dispatch.cu`, `moe_align_decode`, as it read before §43 deleted
/// it as unreached — the row is routed and the kernel is unchanged:
///
/// ```text
/// constexpr int BS = 1024;
/// // counts + offsets(+1) + fill, then 32 warp partials and one running base
/// // for the block-wide scan.
/// const std::size_t smem =
///     static_cast<std::size_t>(3 * num_experts + 1 + 33) * sizeof(std::int32_t);
/// device::moe_align_decode<device::i32><<<1, BS, smem, stream>>>(...);
/// ```
///
/// Two rules rather than one with [`Rule::RouterLane`], and the launcher's
/// own comment is the argument: *"ONE BLOCK, whatever the routing: the scan
/// is block-wide and the counters are in shared memory. A grid over rows
/// would run N copies of the sort, each clearing what the others are
/// reading."* The two shared a name on the other backend until the row axis
/// landed on one of them, and the grid is the contract — so two contracts
/// need two rows.
///
/// The smem is stated from [`Dims::n_experts`] and is the reason this rule
/// could not have been faked with a constant: the sort's counters, offsets
/// and fill are each `n_experts` long, and a mixture with 256 experts wants
/// four times what one with 64 does. Under-allocate and the scan's warp
/// partials land inside the offsets it is scanning.
///
/// The same file's `moe_bucket_exact` is this launch without the scan's 33
/// extra words. This rule hands it 132 bytes it will not read, which is a
/// dynamic allocation being larger than the kernel's use of it — legal, and
/// the direction of the error that is not a silent one.
///
/// **That file is now DELETED and `moe_bucket_exact`'s host is
/// `driver-cuda/src/fire/moe_dispatch.rs`, which states `(3E + 1) * 4`
/// exactly.** The rule stays on the device row
/// `moe::moe_bucket_exact_dev`, still over-allocating and still safe; the
/// 33-word gap is what made that symbol a `Control::Supplies` walk rather
/// than a row anyone could have finished sourcing.
fn router_sort(n_experts: u32) -> Result<Launch, Ungeometric> {
    let words = n_experts
        .checked_mul(3)
        .and_then(|counters| counters.checked_add(34))
        .ok_or(Ungeometric::Empty)?;
    let smem = words.checked_mul(FLOAT).ok_or(Ungeometric::Empty)?;
    Ok(Launch { grid: [1, 1, 1], block: [SORT_BLOCK, 1, 1], smem })
}

/// One block per (route, warp-tile of the output width) — `dim3(rows *
/// experts_per_token, ceil(width / 8))` at [`BLOCK`] threads, one warp per
/// output column.
///
/// `quant/dequant_wna16.cu`'s `wna16_moe_gate_up_decode_bf16` **has since been
/// deleted** (§43.9) — it was routed, so `abi::emit_c_shim` emitted no entry
/// for it, and it had no C++ caller. What it read, transcribed here before it
/// went, was:
///
/// ```text
/// const int routes = num_tokens * top_k;
/// if (routes <= 0 || hidden <= 0 || intermediate <= 0) return;
/// if (hidden % 8 != 0 || hidden % group_size != 0) return;
/// constexpr int GU_WARPS = DECODE_BLOCK / 32;
/// const dim3 grid(routes, (intermediate + GU_WARPS - 1) / GU_WARPS);
/// device::wna16_gate_up_decode<<<grid, DECODE_BLOCK, 0, stream>>>(...);
/// ```
///
/// with `DECODE_BLOCK = 256` — the one line of it that survives, at
/// `quant/dequant_wna16.cu:23` — so `GU_WARPS` is 8 and the divisor is the
/// block's warp count. The two guards are host facts and this rule keeps
/// them: the `routes <= 0` arm is [`Ungeometric::Empty`], and the
/// `hidden % 8` / `hidden % group_size` arm is the caller's, because a JIT
/// fire has no early-return to fall into. The AXIS ASSIGNMENT — routes on
/// `x`, intermediate tiles on `y` — is now witnessed by the kernel itself at
/// `quant/dequant_wna16.cuh:295` and `:298`, which is what
/// `tests/launch_rules.rs` pins.
///
/// **This was refused for having no launcher and the refusal was stale**: the
/// module header said *"this backend has no affine-quantized matvec `<<<>>>`
/// at all"*, which is true of the DENSE
/// projections — cuBLAS and Marlin, a library call and a host heuristic — and
/// was never true of the MoE decode path, whose grid is a function of the
/// rectangle and nothing else. [`Rule::Qmv`] and [`Rule::Qmm`] stay refused
/// on the reading that still holds.
///
/// `routes` is `rows * experts_per_token` because a routed decode expands
/// each token into its top-k slots before the GEMV sees it, which is the same
/// reading `driver-metal`'s `shapes::routed_qmv` takes of the same two
/// fields. [`Dims::rows`] stays the TOKEN count on both sides; a rule that
/// took `rows` as the routed count would launch `top_k` times too few blocks
/// and leave every token past the first expert holding the last layer's
/// intermediate.
///
/// **This unblocks a geometry, and as of the round below TWO rows.** It was
/// written when `wna16_gate_up_decode` and `wna16_down_decode` could not be
/// NAMED — both are plain `__global__`s and
/// `crate::device::DeviceKernel::instantiation` could only emit
/// `template_path<elem>` — and `crate::device::DeviceKernel::PLAIN` closed
/// that. `dequant_wna16.cuh` still records why they cannot be templated
/// honestly (fp16 activations, bf16 output, `const void* const*` scales); the
/// rows name them by their bare qualified paths instead. What HAS changed
/// since is `driver-cuda`'s `jit_dims`: it filled `experts_per_token: 0` and
/// now fills it from `DispatchCtx::experts_per_token` (`bind/mod.rs:1431`),
/// so this rule and [`routed_qmv_transposed`] answer a grid at the
/// production call site for a fire that carries a mixture, and
/// [`Ungeometric::Empty`] for one that does not.
///
/// Three launchers this rule was asked to carry and does not — the fourth,
/// `wna16_down_decode`, is [`Rule::RoutedQmvTransposed`] now. All three were
/// deleted with their host files in §43; the kernels they launched are still
/// in `quant/dequant_fp4.cuh`, and the refusals are re-pinned there:
///
/// * `quant/dequant_fp4.cu`'s `mxfp4_moe_gate_up_decode<4>` and
///   `mxfp4_moe_down_decode<4>`, both `<<<grid, 128>>>` over a divisor of
///   `warps * kPairs = 16`. Half the block and twice the tile, so one rule
///   serving both would halve one grid or double the other. The three
///   constants that state it survive at `dequant_fp4.cu:39`, `:42` and `:44`.
/// * `mxfp4_moe_gate_up_decode_grouped<kTok>`, whose `grid.x` was an EXPERT
///   count and whose `kTok` came from `std::getenv("PIE_MXFP4_MOE_KTOK")`
///   through a switch of four cases and a default of 4. An environment
///   variable is not an extent of the rectangle, and a rule that read one
///   would state a grid that changes with the shell. The kernel reads its
///   expert off `blockIdx.x` at `quant/dequant_fp4.cuh:471` and takes
///   `kPairs = 2` at `:469`, so both halves of the refusal outlived the
///   `getenv` that motivated it.
fn routed_qmv(routes: u32, width: u32) -> Launch {
    Launch {
        grid: [routes, width.div_ceil(BLOCK / WARP), 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// The MXFP4 decode block — `quant/dequant_fp4.cu:39`'s
/// `constexpr int kMxfp4DecodeBlock = 128;`.
///
/// A SEVENTH 128 in this file, named apart from [`PAGED_BLOCK`],
/// [`ALTUP_BLOCK`], [`PAD_BLOCK`], [`SCAN_BLOCK`] and [`LAYERNORM_BLOCK`] for
/// the reason all of them are: the same number under a different contract.
/// This one is a WARP-COUNT contract. The launcher divides it by 32 to get
/// `warps` and multiplies that by the kernel's template argument to get the
/// tile that divides `grid.y`, so the block width and the grid's second axis
/// are one decision here — halve the block and the tile halves with it.
const MXFP4_DECODE_BLOCK: u32 = 128;

/// Output rows one WARP of the MXFP4 decode GEMVs owns — the template
/// argument, `4` for both legs.
///
/// `quant/dequant_fp4.cu:41` (`kMxfp4GateUpPairs`) and `:43`
/// (`kMxfp4DownRows`), each with the note that it was swept with
/// `driver-cuda/csrc/bench/moe_bench.cu` at gpt-oss's shape. TWO constants in
/// the C++ and one here, because they are the same number under the same
/// contract — the kernel's `<N>` — and the two legs are two ROWS of one rule.
/// The day a sweep parts them is the day the rule splits, which is a row
/// stating a different variant rather than a number read off nothing.
const MXFP4_ROWS_PER_WARP: u32 = 4;

/// [`routed_qmv`]'s axes at a QUAD tile — `dim3(routes, ceil(width / 16))` at
/// [`MXFP4_DECODE_BLOCK`] threads, nothing shared.
///
/// `quant/dequant_fp4.cu:67-70`:
///
/// ```text
/// const int warps = kMxfp4DecodeBlock / 32;
/// const int pairs_per_block = warps * kMxfp4GateUpPairs;
/// dim3 grid(num_tokens * top_k,
///           (intermediate + pairs_per_block - 1) / pairs_per_block);
/// device::mxfp4_moe_gate_up_decode<kMxfp4GateUpPairs>
///     <<<grid, kMxfp4DecodeBlock, 0, stream>>>(
/// ```
///
/// and its down twin at `:152-156`, which spells `rows_per_warp` where the
/// first spells `kMxfp4GateUpPairs` and is otherwise the same five lines.
///
/// **Why it is not [`routed_qmv`].** That rule is `dim3(routes, ceil(width /
/// 8))` at 256 threads and this is `dim3(routes, ceil(width / 16))` at 128 —
/// the block is HALF as wide and the tile is TWICE as tall, so `grid.x`
/// agrees, `grid.y` does not, and the total thread count is a quarter. The
/// near miss is measured in [`kernels::LaunchRule::RoutedQmvQuad`]'s doc: at
/// gpt-oss's 2 880-wide intermediate it is 360 blocks of 256 against 180 of
/// 128, and every output row is claimed by two blocks that derive different
/// expert rows from their own `warp_id`.
///
/// **Why the tile is a product and not a constant.** `16` is
/// `(MXFP4_DECODE_BLOCK / WARP) * MXFP4_ROWS_PER_WARP`, and both factors are
/// the launcher's. Writing `16` would be a number that agreed with the C++ by
/// coincidence rather than by derivation, which is the shape of every rule in
/// this file that was later found to have been reproducing an old sweep.
///
/// **`width` here is the PER-ROUTE width and the caller divides.** The two
/// statements this serves declare `[Tokens, k, intermediate]`, so
/// [`Dims::width`] is `k * intermediate`; `eval`'s arm divides the fanout out
/// and checks two ways that it was the right divisor. This function takes the
/// number the launcher takes, which is `intermediate`.
fn routed_qmv_quad(routes: u32, width: u32) -> Launch {
    let tile = (MXFP4_DECODE_BLOCK / WARP) * MXFP4_ROWS_PER_WARP;
    Launch {
        grid: [routes, width.div_ceil(tile), 1],
        block: [MXFP4_DECODE_BLOCK, 1, 1],
        smem: 0,
    }
}

/// [`routed_qmv`]'s two axes swapped — `dim3(ceil(width / 8), routes)` at
/// [`BLOCK`] threads.
///
/// `quant/dequant_wna16.cu:101-104`, as it read before §43.9 deleted the
/// launcher as unreached — `quant/dequant_wna16.cuh:371` and `:374` are the
/// surviving witness that this leg swaps the axes:
///
/// ```text
/// constexpr int BS = 256;
/// constexpr int WARPS = BS / 32;
/// const dim3 grid((hidden + WARPS - 1) / WARPS, routes);
/// device::wna16_down_decode<<<grid, BS, 0, stream>>>(
/// ```
///
/// against its gate/up sibling nine lines earlier at `73-75`:
///
/// ```text
/// constexpr int GU_WARPS = DECODE_BLOCK / 32;
/// const dim3 grid(routes, (intermediate + GU_WARPS - 1) / GU_WARPS);
/// device::wna16_gate_up_decode<<<grid, DECODE_BLOCK, 0, stream>>>(
/// ```
///
/// **A transpose is not a tuning difference and it is not a bug in either
/// launcher.** The two kernels read `blockIdx` opposite ways round —
/// `dequant_wna16.cuh:295-298` takes `route = blockIdx.x` and
/// `row = blockIdx.y * warps + warp`, `:371-374` takes `route = blockIdx.y`
/// and `h = blockIdx.x * warps + warp` — so each launcher matches its own
/// kernel and neither can be restated as the other. This is a second rule and
/// not a parameter for the reason [`Rule::PerRowNarrow`] is a second rule:
/// what a rule NAMES has to be checkable against one launcher, and a boolean
/// that swaps two axes is a rule that agrees with everything.
///
/// **What firing one under the other's rule does.** The area is identical, so
/// no count, no occupancy figure and no launch error moves. On the shape
/// `dequant_wna16.cu` runs at decode — `routes = 8`, `hidden = 2048`,
/// `WARPS = 8` — the correct grid is `(256, 8)` and the transposed one is
/// `(8, 256)`. `wna16_down_decode` then computes `h` from `blockIdx.x` and
/// finds it in `[0, 8)` instead of `[0, 2048)`, so 8 of 2048 hidden columns
/// are written and `route = blockIdx.y` runs to 256 where 8 routes exist,
/// indexing `topk_idx` 248 entries past its end. Whether that faults depends
/// on the allocator. What it does not do is report anything.
fn routed_qmv_transposed(routes: u32, width: u32) -> Launch {
    Launch {
        grid: [width.div_ceil(BLOCK / WARP), routes, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// A [`TILE`]-square block over a rectangle — `ceil(width / 16)` by
/// `ceil(rows / 16)` blocks of `dim3(16, 16)`, nothing shared.
///
/// `vision/gemma4_vision.cu`, `k_addpos_grid2d`:
///
/// ```text
/// dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}
/// ...
/// vd::k_addpos_grid2d<bfd><<<G2(Hd,N),B2,0,S>>>(D(h),D(w.pos_table),pos,N,Hd,PT);
/// ```
///
/// — `B2` and `G2` at `117`, the launch at `144`. All three towers declare
/// that same pair on one line (`vision/gemma4_audio.cu:131`,
/// `vision/qwen3_vl_tower.cu:139`) and launch eleven kernels through it:
/// `k_matmul` (audio `165`, `203`, `242`, `289`), `k_matmul_bias` (audio
/// `283`), `k_addpos_grid2d`, `k_qk` and `k_av` (vision `151`), `k_pool`
/// (vision `165`), `k_glu` (audio `250`), `k_qkv_scale` (audio `240`),
/// `k_sscp_flatten` (audio `201`), `k_rel_pos_enc` (audio `220`) and
/// `k_merge_gather` (qwen `165`, `168`).
///
/// **X is the output WIDTH and Y is the ROW COUNT, in that order**, which is
/// the transpose every reader of `G2(X, Y)` has to get right once: the
/// kernels read `blockIdx.x * blockDim.x + threadIdx.x` as a column and
/// `blockIdx.y * blockDim.y + threadIdx.y` as a row. Swapped, a `16 x 2048`
/// rectangle launches 1 by 128 blocks instead of 128 by 1 and computes a
/// sixteenth of it.
///
/// **The block is the first here that is not `[n, 1, 1]`.** Flattening it to
/// `[256, 1, 1]` is the same thread count and a different addressing —
/// `gemma4_naive_kernels.cuh:117` reads `threadIdx.y` for its row, and with
/// `blockDim.y == 1` every thread of every block computes row zero, sixteen
/// times over. That is a full grid of legal work against a sixteenth of the
/// output, which no hardware reports.
///
/// **THREE of the eleven walk a rectangle that is not the statement's
/// output**, and a row for any of them has to say so rather than inherit it.
/// These are the same species of defect as a wrong citation — a `Dims` field
/// that is PRESENT and WRONG, which no refusal catches because nothing is
/// missing:
///
/// * `k_av` launches `G2(64, N)` where 64 is one head's width and the host
///   loops the twelve heads; the row it belongs to is 768 wide. A rule handed
///   the statement's width launches **12x the blocks**.
/// * `k_rel_pos_enc` launches `((Hd + 15) / 16, (P + 15) / 16)` over the
///   position table, whose rows are the 13 POSITIONS and not the tokens.
/// * `k_glu` launches `G2(Hd, N)` where `Hd` is the OUTPUT width: the kernel
///   reads `x[n * 2 * D + d]` and `x[n * 2 * D + D + d]`
///   (`gemma4_audio.cuh:133`), so its input row is `2 * Hd` and a rule taking
///   the width from `In(0)` launches **twice** the grid.
///
/// The rule is right about the arithmetic in all three; what would be wrong
/// is the rectangle handed to it, which is a row's business and not a grid's.
/// `tests/launch_rules.rs`'s `transcribed` module compares all three at the
/// launcher's OWN rectangle for exactly this reason — agreeing there is the
/// claim this rule makes, and it is not the claim a row would need.
///
/// What it does NOT serve is the 32-by-8 tile the quantizers use —
/// `quant/dtype_cast.cu:140`, `quant/quant_bf16_to_fp8.cu:128`, `dim3
/// block(BX, BY)` with `BX = 32, BY = 8` over `dim3((n + 31) / 32, (m + 7) /
/// 8)`. Same idea, different constants in both dimensions, and a rule whose
/// tile is in its name cannot quietly become the other one.
fn tile16(rows: u32, width: u32) -> Launch {
    Launch {
        grid: [width.div_ceil(TILE), rows.div_ceil(TILE), 1],
        block: [TILE, TILE, 1],
        smem: 0,
    }
}

/// One WARP per (head, row), heads on `grid.y` and rows on `grid.z` —
/// `dim3(1, heads, rows)` at [`WARP`] threads.
///
/// `vision/gemma4_vision.cu`, `k_rope_axial2d`, at `150` — ONE line, TWO
/// launches:
///
/// ```text
/// dim3 rg(1,NH,N);vd::k_rope_axial2d<bfd><<<rg,32,0,S>>>(D(q),pos,N,NH,THETA);vd::k_rope_axial2d<bfd><<<rg,32,0,S>>>(D(k),pos,N,NH,THETA);
/// ```
///
/// **`grid.x` is literally one and stays one.** The kernel
/// (`vision/gemma4_vision.cuh:106`) reads `blockIdx.z` as its token,
/// `blockIdx.y` as its head and `blockIdx.x * blockDim.x + threadIdx.x` as a
/// channel, and returns on `c >= 16`: the head is 64 channels read as four
/// 16-wide quarters, and one thread rotates `(c, c+16)` by the x angle and
/// `(32+c, 48+c)` by the y. Sixteen lanes of the warp work and the other
/// sixteen return, so the axis a channel tiling would have used is spent
/// before it starts and the two counts move up rather than across.
///
/// **No other rule here opens a `grid.z`**, and the reason it is this one is
/// that every axis comes off the rectangle: a head count and a row count,
/// both already in [`Dims`]. The third-axis shapes still refused fail on the
/// axis's VALUE and not on the axis — see the module header on a stream
/// count, a layer count and a channel count.
///
/// `kv_heads` and not `q_heads`: the launcher fires this over `q` and then
/// over `k` on the SAME `rg` with the same `NH` — the two launches above —
/// so the head count is the addressed tensor's, which is the reading
/// [`Dims::kv_heads`] states and [`per_head`] takes. Each launch is handed
/// one tensor; there is no `k` parameter, and a reading of the signature that
/// put both in one launch would halve the grid this rule has to produce.
///
/// The launcher guards `Hd != 768 || NH != 12` and throws at `129` before
/// reaching this line — the kernel hard-codes a 64-wide head and a 16-wide
/// quarter — so a row for it states a shape the tower already checked. That
/// is a precondition on a value and not a grid, and it is recorded here
/// because a rule that produced this grid for a 128-wide head would launch
/// half a rotation and report success.
fn axial_rope(rows: u32, heads: u32) -> Launch {
    Launch { grid: [1, heads, rows], block: [WARP, 1, 1], smem: 0 }
}

/// The reference paged attention's PREFILL launch — `dim3(requests, rows,
/// q_heads)` at [`PAGED_BLOCK`] threads, with `(head_dim + 128) * 4` bytes of
/// DYNAMIC shared memory.
///
/// `attn/attention_naive_paged.cu`, three launchers of one kernel, all four
/// lines identical bar the head width's source:
///
/// ```text
/// :108  dim3 grid(num_requests, total_tokens, num_q_heads);
/// :109  dim3 block(BLOCK);
/// :110  const std::size_t smem = (head_dim + BLOCK) * sizeof(float);
/// :111  device::naive_paged_attn<BLOCK><<<grid, block, smem, stream>>>(
/// ```
///
/// and again at `:195-198` and `:245-248`, where the head width is
/// `kv_layer.head_dim` rather than a parameter. `constexpr int BLOCK = 128;`
/// is at `:35`.
///
/// **The shared allocation is the whole of why this is a rule.** It is a sum
/// and not a product, and the addend is the BLOCK: the kernel cuts
/// `extern __shared__ float smem[]` into `q_smem = smem` of `head_dim` floats
/// and `reduce = smem + head_dim` of `BLOCK` floats
/// (`attention_naive_paged.cuh:402-404`). [`sdpa_vector`]'s `(rows + 256) * 4`
/// has the same shape and adds the block to the TOKEN count, so on a 4096-token
/// fire over 128-wide heads it asks for 17,408 bytes where this launcher asks
/// for 1,024 — a request the hardware serves, out of a budget the occupancy
/// then pays for, against a kernel that reads 256 of it.
///
/// **`grid.x` is a REQUEST count.** See [`Dims::requests`] for why that is a
/// field and not `rows`, and for the fact that nothing fills it today: this
/// rule answers [`Ungeometric::Empty`] at every production call site, which
/// is [`Rule::RoutedQmv`]'s position and the honest one.
///
/// **No row states it either**, and the second blocker is an operand rather
/// than a grid: `naive_paged_attn` takes `device::KvScheme` and
/// `device::KvDType` by value — one-byte enums declared at
/// `attention_naive_paged.cuh:187` and `:198` — and `kernels::Ty` has no
/// word for either, `crate::runtime::args` no `ArgValue` to marshal one. Both
/// are outside this change's grant, so the rule is stated, cited, pinned and
/// rowless, which is the state `attn/head_dim_pad`'s two kernels were in
/// before [`Rule::PerHead`] and the state this file prefers to a guess.
fn paged_scores(requests: u32, rows: u32, q_heads: u32, head_dim: u32) -> Launch {
    Launch {
        grid: [requests, rows, q_heads],
        block: [PAGED_BLOCK, 1, 1],
        smem: (head_dim + PAGED_BLOCK) * FLOAT,
    }
}

/// The same family's DECODE launch — `dim3(rows, q_heads)` at
/// [`PAGED_BLOCK`] threads, same `(head_dim + 128) * 4`.
///
/// `attn/attention_naive_paged.cu:147-150`:
///
/// ```text
/// dim3 grid(num_requests, num_q_heads);
/// dim3 block(BLOCK);
/// const std::size_t smem = (kv_layer.head_dim + BLOCK) * sizeof(float);
/// device::naive_paged_decode<BLOCK><<<grid, block, smem, stream>>>(
/// ```
///
/// **`rows` and not `requests`, and the difference is a decode's contract.**
/// A decode fire is one token per request, so the statement's rectangle has
/// `total_tokens == num_requests` and the row count IS the request count.
/// [`paged_scores`] cannot make that identification because a prefill's
/// `total_tokens` is a sum over requests and its launcher spells both numbers
/// in one `dim3`.
///
/// **`blockIdx.y` means something different in the two kernels**, which is
/// why they are two rules rather than one with a degenerate axis.
/// `naive_paged_decode` reads `q_head = blockIdx.y`
/// (`attention_naive_paged.cuh:541`); `naive_paged_attn` reads
/// `qo_off = blockIdx.y` and takes the head from `blockIdx.z`
/// (`:370-371`). Collapsing them would give one of the two the other's
/// addressing at exactly the shapes where the numbers agree.
///
/// Rowless for [`paged_scores`]' second reason: `naive_paged_decode` takes
/// the same two `KvScheme` / `KvDType` enums by value.
fn paged_scores_decode(rows: u32, q_heads: u32, head_dim: u32) -> Launch {
    Launch {
        grid: [rows, q_heads, 1],
        block: [PAGED_BLOCK, 1, 1],
        smem: (head_dim + PAGED_BLOCK) * FLOAT,
    }
}

/// MLA's fused prepare — `dim3(rows, 1 + ceil(q_heads / heads_per_block))` at
/// [`BLOCK`] threads, nothing shared.
///
/// `attn/mla_paged.cu`:
///
/// ```text
/// :56  constexpr int BS = 256;
/// :59  const int half = rope / 2;
/// :64  const int heads_per_block = half >= BS ? 1 : (BS / half);
/// :65  const int q_blocks = (heads + heads_per_block - 1) / heads_per_block;
/// :73  dim3 grid(total_tokens, 1 + q_blocks);
/// :74  device::mla_prepare<BS><<<grid, BS, 0, stream>>>(
/// ```
///
/// with `rope = layer.qk_rope_head_dim` at `:58`.
///
/// **What the leading `1` does, because a reader who takes it for padding
/// will delete it.** It is the KV LANE, and `mla_paged.cuh:216-221` says so:
/// *"blockIdx.y == 0 handles the KV lane (norm, k_pe rotation, page write);
/// blockIdx.y >= 1 splits the query heads. The KV lane must own both the k_pe
/// rotation and the page write, because the write consumes the rotated value
/// and a cross-block dependency inside one kernel would need a grid sync."*
/// So the axis is `1 + q_blocks` and not `max(1, q_blocks)` and not
/// `q_blocks` rounded up: one lane does a job no head does, and the head
/// blocks are numbered from one. A rule that dropped the `1` would launch the
/// right number of head blocks and never normalise the latent, never rotate
/// `k_pe`, and never write the page — and `q_nope`/`q_pe` would still fill,
/// so the fire produces a plausible query against an unwritten cache.
///
/// **The packing is [`rope`]'s, computed on the ROTARY width.** `half` is
/// `qk_rope_head_dim / 2` and not `head_dim / 2`: an MLA head is
/// `kv_lora_rank + qk_rope_head_dim` wide (576 for DeepSeek-V3) and only the
/// 64-channel `k_pe`/`q_pe` tail turns, so a rule that read
/// [`Dims::head_dim`] would compute `heads_per_block = 1` where the launcher
/// computes 8 and launch eight times the query blocks. See
/// [`Dims::rotary_dims`] for why that field is the honest reading and not an
/// overload.
///
/// `rotary_dims` must be even — a rotation turns pairs — and non-zero, or
/// `BS / half` divides by zero. Both are refused rather than floored, in
/// `eval`.
fn mla_prepare(rows: u32, q_heads: u32, rotary_dims: u32) -> Launch {
    let half = rotary_dims / 2;
    let heads_per_block = if half >= BLOCK { 1 } else { BLOCK / half };
    let q_blocks = q_heads.div_ceil(heads_per_block);
    Launch { grid: [rows, 1 + q_blocks, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// One block per (row, packed head) — `dim3(rows, q_heads + kv_heads)` at
/// [`BLOCK`] threads, nothing shared.
///
/// `attn/qkv_fused.cu:245-248`:
///
/// ```text
/// :245  constexpr int BLOCK = 256;
/// :246  dim3 grid(num_rows, num_q_heads + num_kv_heads);
/// :247  device::qkv_packed_qk_norm_rope_vnorm_write_kv<BLOCK>
/// :248      <<<grid, BLOCK, 0, stream>>>(
/// ```
///
/// **The sum is undivided and that is the point.** The kernel reads
/// `head_idx = blockIdx.y` and splits on `is_q = head_idx < num_q_heads`
/// (`qkv_fused.cuh:413-414`), so the two head banks share one axis and the
/// kernel recovers which bank a block is in. [`gated_rms`] is the nearest
/// ported shape — `[rows, kv_heads, 1]` at 256 with no shared memory, the
/// same launch in every respect but this one — and its `grid.y` is short by
/// every query head. On llama-3.2's 32 q / 8 kv that is 8 of 40 blocks, and
/// the 8 it launches compute the QUERY heads 0..8 rather than the kv bank,
/// because the kernel's own `is_q` test reads the axis it was given. A
/// quarter of q normed and rotated, no kv written, no error.
///
/// `q_heads + kv_heads` is checked against overflow in `eval` and each is
/// checked for zero: a fused epilogue with no kv bank is not this launcher.
fn rows_packed_heads(rows: u32, packed_heads: u32) -> Launch {
    Launch { grid: [rows, packed_heads, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// [`rows_packed_heads`] at [`SCAN_BLOCK`] threads — the DECODE form.
///
/// `attn/qkv_fused.cu`, inside the `QKV_DECODE_LAUNCH` macro:
///
/// ```text
/// :98   constexpr int BLOCK = 128;
/// :99   dim3 grid(num_requests, num_q_heads + num_kv_heads);
/// :101  device::qkv_decode_qk_norm_rope_write_kv<BLOCK, true>
/// :102      <<<grid, BLOCK, 0, stream>>>(
/// :126  device::qkv_decode_qk_norm_rope_write_kv<BLOCK, false>
/// :127      <<<grid, BLOCK, 0, stream>>>(
/// ```
///
/// Two launches on one grid, chosen by `rope_table != nullptr` — see
/// `crate::device::Term::NonNull`, which is how a row states that choice.
///
/// **Not a parameter of [`rows_packed_heads`], for two reasons and the second
/// is the load-bearing one.** The first is [`Rule::PerRowNarrow`]'s: the
/// kernel folds `__shared__ float buf[BLOCK]` by halving from `BLOCK / 2`, so
/// a different width sums the same values in a different order. The second is
/// that `BLOCK` is the kernel's FIRST TEMPLATE ARGUMENT and sizes that array:
/// launching the `<128>` instantiation on 256 threads has threads 128..255
/// write `buf[128..255]`, which the instantiation never allocated, and the
/// halving reduction then reads 128 slots that were never written by the
/// `<128>` code path. Undefined, unreported, and numerically plausible.
///
/// **`rows` is the request count here** for [`paged_scores_decode`]'s reason:
/// this is the decode path, one token per request, and `table::attn`'s
/// `qkv_decode_fused` row already spells `num_requests: I32 <- Source::Rows`.
fn rows_packed_heads_narrow(rows: u32, packed_heads: u32) -> Launch {
    Launch { grid: [rows, packed_heads, 1], block: [SCAN_BLOCK, 1, 1], smem: 0 }
}

/// One WARP per (row, packed head), flattened — `ceil(rows * packed_heads /
/// (BLOCK / WARP))` blocks of [`BLOCK`].
///
/// `attn/qkv_fused.cu`, the first half of the same macro:
///
/// ```text
/// :51  constexpr int WARP_BLOCK = 256;
/// :52  const int total_units = num_requests * (num_q_heads + num_kv_heads);
/// :53  dim3 warp_grid((total_units + (WARP_BLOCK / 32) - 1) / (WARP_BLOCK / 32));
/// :57  device::qkv_decode_qk_norm_rope_write_kv_warp<
/// :58      (HEAD_DIM_VALUE), true><<<warp_grid, WARP_BLOCK, 0, stream>>>(
/// :70  device::qkv_decode_qk_norm_rope_write_kv_warp<
/// :71      (HEAD_DIM_VALUE), false><<<warp_grid, WARP_BLOCK, 0, stream>>>(
/// ```
///
/// **The unit of work is a warp, and the grid is one-dimensional because the
/// reduction is.** `HEAD_DIM` is a template argument, `ELEMS_PER_THREAD =
/// HEAD_DIM / 32` is a constant, and the norm folds with `__shfl_xor_sync`
/// and no `__syncthreads` — so a warp is self-contained and the block is only
/// a container for eight of them. The kernel recovers the pair from
/// `unit = blockIdx.x * warps_per_block + warp_id` and returns on
/// `unit >= num_requests * total_qk_heads` (`qkv_fused.cuh:264-265`), which
/// is what makes the ragged last block safe.
///
/// **Two ways to get this wrong, both of which run.** Stated as
/// [`rows_packed_heads`]' 2-D grid it launches `rows * packed_heads` blocks
/// where the launcher launches an eighth of that, and every block past the
/// first computes `unit` from a `blockIdx.x` that now counts (row, head)
/// pairs rather than warp groups — so eight times the blocks covering an
/// eighth of the pairs. Stated at [`per_row`]'s one block per row it covers
/// `rows * 8` of `rows * packed_heads` units, which for 40 packed heads is
/// one fifth of them, and the fifth it covers is correct.
///
/// The divisor is `BLOCK / WARP` — warps per block — and NOT a tile width. It
/// moves with the block, which is why both are read from the same two
/// constants here rather than written as an 8.
fn warp_packed_heads(rows: u32, packed_heads: u32) -> Result<Launch, Ungeometric> {
    let units = rows.checked_mul(packed_heads).ok_or(Ungeometric::Empty)?;
    Ok(Launch {
        grid: [units.div_ceil(BLOCK / WARP), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    })
}

/// A third grid axis over an ALTUP STREAM count — `dim3(rows, streams,
/// ceil(hidden / 128))` at [`ALTUP_BLOCK`] threads, nothing shared.
///
/// `csrc/src/norm/altup.cu`, both launchers, identical — **as they read
/// before §43 deleted the file**; `norm/altup.cuh:83-85` and `:113-115` are
/// what witnesses the axis order now:
///
/// ```text
/// :18  const dim3 grid(T, K, (H + BLOCK - 1) / BLOCK);
/// :19  device::altup_predict<device::bf16><<<grid, BLOCK, 0, stream>>>(
/// :32  const dim3 grid(T, K, (H + BLOCK - 1) / BLOCK);
/// :33  device::altup_correct<device::bf16><<<grid, BLOCK, 0, stream>>>(
/// ```
///
/// with `constexpr int BLOCK = 128;` above both.
///
/// **`hidden` is PER STREAM and the rectangle's width is not.** The value is
/// `[K, tokens, H]` and a statement has one row width, so `table::norm`'s
/// AltUp rows already divide: `h: I32 <- Source::Div(&Source::Width(
/// &Source::In(0)), &Source::CtxNonZero("altup_streams"))`. This rule makes
/// the same division for the same reason, and `eval` refuses a width that is
/// not a multiple of the stream count — a `[K, T, H]` value whose width does
/// not divide by `K` is not this shape, and flooring the division would tile
/// `floor(H)` columns of a wider row and leave the tail of every stream
/// holding the last layer's activations.
///
/// **[`warp_tiled_scan`] is the near miss, and it is wrong twice.** It is the
/// only other rule with a `grid.z`, at this exact 128-wide block, and:
///
/// * its `z` is `ceil(V_d / 4)` where this is `ceil(H / 128)`. Same axis,
///   divisors 32 apart: on gemma-3n's `H = 2048` this launches 16 blocks and
///   that rule would launch 512 — or, read the other way at the value width
///   these kernels see, **1/32 of the blocks and 31/32 of hidden untouched**.
///   The kernels' own `h >= H` guard makes the shortfall silent.
/// * its `grid.y` is filled from [`Dims::kv_heads`], an attention head count.
///   See [`Dims::altup_streams`] for why a stream count cannot live there.
///
/// The `t >= T || k >= K` half of the kernels' guard is dead under exactly
/// this grid and the `h >= H` half is not — `H` is tiled by 128 and the last
/// tile is ragged. `altup.cuh` says so and keeps the whole guard.
fn altup_streams(rows: u32, streams: u32, hidden: u32) -> Launch {
    Launch {
        grid: [rows, streams, hidden.div_ceil(ALTUP_BLOCK)],
        block: [ALTUP_BLOCK, 1, 1],
        smem: 0,
    }
}

/// A head geometry, checked.
///
/// Both numbers or neither: a rule handed `heads = 0` builds a grid axis of
/// zero and launches nothing, and a rule handed `head_dim = 0` builds a full
/// grid of blocks whose every loop runs zero times. The first is CUDA
/// returning success for no work; the second is the hardware returning
/// success for work that was skipped. Neither is reported, both look like a
/// fire that ran, and this is the one place that can tell the difference.
const fn headed(heads: u32, head_dim: u32) -> Result<(), Ungeometric> {
    if heads == 0 || head_dim == 0 {
        return Err(Ungeometric::Empty);
    }
    Ok(())
}

/// The launch `rule` produces for `dims`.
///
/// A free function rather than a method on [`Rule`] because the rule is
/// `kernels`' and the arithmetic is this backend's — the same split
/// `driver-metal` makes, so that the two can disagree about numbers without
/// disagreeing about vocabulary.
///
/// # Errors
///
/// [`Ungeometric`], and every variant of it is drift rather than a condition
/// a fire can be in.
pub fn eval(rule: Rule, dims: Dims) -> Result<Launch, Ungeometric> {
    // A rectangle covers at least one row. Zero is refused rather than
    // floored: the callers that legitimately launch a single row state one,
    // and a rectangle that collapsed to nothing is a lowering bug that a
    // floor would hide behind a kernel doing one row of work.
    if dims.rows == 0 {
        return Err(Ungeometric::Empty);
    }
    Ok(match rule {
        Rule::Unstated => return Err(Ungeometric::Unstated),
        Rule::Rms => rms(dims.rows),
        Rule::Elementwise => {
            let n = dims.rows.checked_mul(dims.width).ok_or(Ungeometric::Empty)?;
            if n == 0 {
                return Err(Ungeometric::Empty);
            }
            elementwise(n)
        }
        Rule::ElementwiseRows => {
            if dims.width == 0 {
                return Err(Ungeometric::Empty);
            }
            elementwise_rows(dims.rows, dims.width)
        }
        Rule::RouteRows => {
            if dims.width == 0 {
                return Err(Ungeometric::Empty);
            }
            route_rows(dims.rows, dims.width)
        }
        // `kv_heads` and not `q_heads`: the pad addresses whichever tensor
        // the statement named, and a grouped-query fire has two head counts.
        // See [`Dims::kv_heads`] for what reading the wrong one overruns.
        Rule::PerHead => {
            headed(dims.kv_heads, dims.head_dim)?;
            per_head(dims.rows, dims.kv_heads)
        }
        Rule::PerHeadElementwise => {
            headed(dims.q_heads, dims.head_dim)?;
            per_head_elementwise(dims.rows, dims.q_heads, dims.head_dim)
        }
        Rule::GatedRms => {
            headed(dims.kv_heads, dims.head_dim)?;
            gated_rms(dims.rows, dims.kv_heads)
        }
        Rule::SdpaVector => {
            if dims.q_heads == 0 {
                return Err(Ungeometric::Empty);
            }
            sdpa_vector(dims.rows, dims.q_heads)?
        }
        Rule::SplitPacked => {
            if dims.in_width == 0 {
                return Err(Ungeometric::Empty);
            }
            split_packed(dims.rows, dims.in_width)
        }
        // `head_dim < 2` is the launcher's own `if (half <= 0) return;`,
        // answered here instead: a rotation over a head of one channel has
        // no pair to turn, and `BLOCK / half` would divide by zero.
        Rule::Rope => {
            headed(dims.q_heads + dims.kv_heads, dims.head_dim)?;
            if dims.head_dim < 2 {
                return Err(Ungeometric::Empty);
            }
            rope(dims.rows, dims.q_heads, dims.kv_heads, dims.head_dim)
        }
        Rule::RouterLane => router_lane(dims.rows),
        Rule::RouterSort => {
            if dims.n_experts == 0 {
                return Err(Ungeometric::Empty);
            }
            router_sort(dims.n_experts)?
        }
        // The value-head axis, and `head_dim` is the KEY width the shared
        // slab is cut from — see [`recurrent_scan`] for why one field
        // standing for both of the recurrence's head widths is what keeps
        // its `_cached` sibling refused.
        Rule::RecurrentScan => {
            headed(dims.kv_heads, dims.head_dim)?;
            recurrent_scan(dims.rows, dims.kv_heads, dims.head_dim)?
        }
        Rule::PerRow => per_row(dims.rows),
        Rule::PerRowNarrow => per_row_narrow(dims.rows),
        // A LITERAL one on the grid, so there is nothing to refuse: no field
        // is read, and an empty rectangle is the kernel's own early-out on
        // an operand it takes as a count. `single`'s doc argues why deriving
        // the `1` from `Dims::rows` is the wrong answer twice over.
        Rule::Single => single(),
        Rule::SingleWarp => single_warp(),
        // A REQUEST count, and zero is ABSENCE — `jit_dims` fills
        // `Dims::requests` from `AttnCtx::num_requests` and leaves it zero
        // for a fire with no attention context, which is the same reading
        // [`Rule::PagedScores`] makes of the same field.
        Rule::PerRequest => {
            if dims.requests == 0 {
                return Err(Ungeometric::Empty);
            }
            per_request(dims.requests)
        }
        // The width is a channel count here and the grid's only axis, so an
        // empty one is a launch of nothing rather than a row of nothing —
        // the same refusal [`Rule::ElementwiseRows`] makes for the same
        // number read the other way round.
        Rule::PerChannel => {
            if dims.width == 0 {
                return Err(Ungeometric::Empty);
            }
            per_channel(dims.width)
        }
        Rule::ElementwiseIn => {
            let n = dims.rows.checked_mul(dims.in_width).ok_or(Ungeometric::Empty)?;
            if n == 0 {
                return Err(Ungeometric::Empty);
            }
            elementwise_in(n)
        }
        Rule::RowScores => row_scores(dims.rows)?,
        // `stated_head_dim == 0` is the ABSENT per-head dim and the rule's own
        // second arm, so this is the one head-shaped rule that does not go
        // through `headed`. It reads the STATEMENT's head width and not the
        // fire's `head_dim`, which cannot distinguish "named none" from
        // "named the fire's own number". See [`Dims::stated_head_dim`].
        Rule::RowsPerHead => rows_per_head(dims.rows, dims.width, dims.stated_head_dim)?,
        Rule::RowsFlat => rows_flat(dims.rows),
        Rule::Slab => {
            let n = dims.rows.checked_mul(dims.width).ok_or(Ungeometric::Empty)?;
            if n == 0 {
                return Err(Ungeometric::Empty);
            }
            slab(n)
        }
        // The routed count is `rows * experts_per_token` and `driver-cuda`'s
        // `jit_dims` fills the second with zero — "absent, not
        // zero-as-a-value" — so this answers `Empty` at that call site until
        // a fire carries an expert count, which is where `Rule::RouterSort`
        // already stands.
        Rule::RoutedQmv => {
            if dims.width == 0 || dims.experts_per_token == 0 {
                return Err(Ungeometric::Empty);
            }
            let routes =
                dims.rows.checked_mul(dims.experts_per_token).ok_or(Ungeometric::Empty)?;
            routed_qmv(routes, dims.width)
        }
        // The same two refusals as [`Rule::RoutedQmv`], and for the same
        // reasons: a zero width is a grid of no columns, and a zero expert
        // count is ABSENCE rather than "routes to none".
        //
        // AND A THIRD THIS RULE MAKES ALONE. Its two statements declare their
        // outputs as `[Tokens, k, intermediate]` — the routed extent as a
        // third dim — and `lower::row_width` is the product of every dim but
        // the leading one, so `Dims::width` is `k * intermediate` where
        // `Rule::RoutedQmv`'s statements declare `[Tokens, intermediate]` and
        // it is `intermediate` outright. The launcher divides the PER-ROUTE
        // width, so this rule divides the fanout out first.
        //
        // A width that does not decompose is not a shape this rule dislikes,
        // it is a rectangle whose third dim is not the fanout — there is no
        // per-route width to slab, so there is nothing to launch. That is
        // `Empty`'s own meaning and it is why the guard is here rather than a
        // clamp.
        //
        // What is NOT here is the corroboration `in_width ==
        // experts_per_token` — true of both stacked statements, whose first
        // input is the route-index row — because `Empty` is for a rectangle
        // that collapsed and not for a shape a rule dislikes, and
        // `every_rule_in_the_vocabulary_is_answered` holds this file to that.
        // It lives in `tests/launch_rules.rs`, over the rows, which is where
        // a claim about what a row may state belongs.
        Rule::RoutedQmvQuad => {
            if dims.width == 0 || dims.experts_per_token == 0 {
                return Err(Ungeometric::Empty);
            }
            if dims.width % dims.experts_per_token != 0 {
                return Err(Ungeometric::Empty);
            }
            let routes =
                dims.rows.checked_mul(dims.experts_per_token).ok_or(Ungeometric::Empty)?;
            routed_qmv_quad(routes, dims.width / dims.experts_per_token)
        }
        // The width is the rectangle's own axis here, not a stacked extent:
        // a zero one is a grid of no columns and the same refusal
        // `ElementwiseRows` makes.
        Rule::Tile16 => {
            if dims.width == 0 {
                return Err(Ungeometric::Empty);
            }
            tile16(dims.rows, dims.width)
        }
        // `head_dim` is checked and unread: the rotation is a warp wide
        // whatever the head, and a head of no channels is still a launch of
        // blocks that turn nothing.
        Rule::AxialRope => {
            headed(dims.kv_heads, dims.head_dim)?;
            axial_rope(dims.rows, dims.kv_heads)
        }
        // `V_d` is the output row divided by the value heads — see
        // [`warp_tiled_scan`] for why it is not `head_dim`, which
        // `RecurrentScan` reads as the KEY width.
        Rule::WarpTiledScan => {
            headed(dims.kv_heads, dims.head_dim)?;
            if dims.width == 0 || !dims.width.is_multiple_of(dims.kv_heads) {
                return Err(Ungeometric::Empty);
            }
            warp_tiled_scan(dims.rows, dims.kv_heads, dims.width / dims.kv_heads)
        }
        // `requests` is ABSENT at every call site `driver-cuda` has today —
        // see [`Dims::requests`] — so this refuses in production rather than
        // launching a square grid over the token count.
        Rule::PagedScores => {
            headed(dims.q_heads, dims.head_dim)?;
            if dims.requests == 0 {
                return Err(Ungeometric::Empty);
            }
            paged_scores(dims.requests, dims.rows, dims.q_heads, dims.head_dim)
        }
        Rule::PagedScoresDecode => {
            headed(dims.q_heads, dims.head_dim)?;
            paged_scores_decode(dims.rows, dims.q_heads, dims.head_dim)
        }
        // An odd or absent rotary width is refused, not floored: `half` is
        // `rotary / 2` and divides the block, so zero divides by zero and one
        // rounds a rotation that has no pair to turn down to none.
        Rule::MlaPrepare => {
            if dims.q_heads == 0 || dims.rotary_dims < 2 || !dims.rotary_dims.is_multiple_of(2)
            {
                return Err(Ungeometric::Empty);
            }
            mla_prepare(dims.rows, dims.q_heads, dims.rotary_dims)
        }
        // BOTH head counts, and both non-zero: the axis is their sum and the
        // kernel splits on `head_idx < num_q_heads`, so a fire with no kv
        // bank is not this launcher and a fire with no q bank is a launch
        // whose every block takes the kv arm.
        Rule::RowsPackedHeads => {
            headed(dims.q_heads, dims.head_dim)?;
            rows_packed_heads(dims.rows, packed_heads(&dims)?)
        }
        Rule::RowsPackedHeadsNarrow => {
            headed(dims.q_heads, dims.head_dim)?;
            rows_packed_heads_narrow(dims.rows, packed_heads(&dims)?)
        }
        Rule::WarpPackedHeads => {
            headed(dims.q_heads, dims.head_dim)?;
            warp_packed_heads(dims.rows, packed_heads(&dims)?)?
        }
        // [`Rule::RoutedQmv`]'s refusals, unchanged: `experts_per_token` is
        // zero at the production call site and this rule answers `Empty`
        // there for the same reason its untransposed twin does.
        Rule::RoutedQmvTransposed => {
            if dims.width == 0 || dims.experts_per_token == 0 {
                return Err(Ungeometric::Empty);
            }
            let routes =
                dims.rows.checked_mul(dims.experts_per_token).ok_or(Ungeometric::Empty)?;
            routed_qmv_transposed(routes, dims.width)
        }
        // The width is `streams * hidden` and the rule tiles ONE stream, so
        // a width that does not divide by the stream count is not a
        // `[K, T, H]` value and is refused rather than floored.
        Rule::AltUpStreams => {
            if dims.altup_streams == 0
                || dims.width == 0
                || !dims.width.is_multiple_of(dims.altup_streams)
            {
                return Err(Ungeometric::Empty);
            }
            altup_streams(dims.rows, dims.altup_streams, dims.width / dims.altup_streams)
        }
        other => return Err(Ungeometric::Unported(other)),
    })
}

/// `q_heads + kv_heads`, checked — the axis the three fused QKV rules open.
///
/// Both counts have to be there. The kernels split on
/// `is_q = head_idx < num_q_heads` and address a packed `[q | k | v]` row
/// from it, so a zero kv bank makes the packed stride wrong for every block
/// and a zero q bank makes every block take the kv arm at head index zero.
/// Neither is a shape; both would launch.
fn packed_heads(dims: &Dims) -> Result<u32, Ungeometric> {
    if dims.q_heads == 0 || dims.kv_heads == 0 {
        return Err(Ungeometric::Empty);
    }
    dims.q_heads.checked_add(dims.kv_heads).ok_or(Ungeometric::Empty)
}

#[cfg(test)]
mod tests {
    use super::{Dims, Launch, Rule, Ungeometric, eval};

    /// gemma-3n's shape: four AltUp streams, 2048 hidden, sixteen tokens.
    const T: u32 = 16;
    const H: u32 = 2048;
    const K: u32 = 4;

    /// The rules this backend has written the arithmetic for.
    ///
    /// Stated so `every_rule_in_the_vocabulary_is_answered` can check the two
    /// halves against each other. A rule that evaluates and is missing here,
    /// or is listed here and refuses, is a table and a driver that disagree
    /// about what is ported — which is exactly the state a reader cannot
    /// detect by reading either one.
    const PORTED: &[Rule] = &[
        Rule::Rms,
        Rule::Elementwise,
        Rule::ElementwiseRows,
        Rule::RouteRows,
        Rule::PerHead,
        Rule::PerHeadElementwise,
        Rule::GatedRms,
        Rule::SdpaVector,
        Rule::SplitPacked,
        Rule::Rope,
        Rule::RouterLane,
        Rule::RouterSort,
        Rule::RecurrentScan,
        Rule::PerRow,
        Rule::PerChannel,
        Rule::ElementwiseIn,
        Rule::RowScores,
        Rule::RowsPerHead,
        Rule::RowsFlat,
        Rule::Slab,
        Rule::RoutedQmv,
        Rule::Tile16,
        Rule::AxialRope,
        Rule::WarpTiledScan,
        Rule::PerRowNarrow,
        Rule::PagedScores,
        Rule::PagedScoresDecode,
        Rule::MlaPrepare,
        Rule::RowsPackedHeads,
        Rule::RowsPackedHeadsNarrow,
        Rule::WarpPackedHeads,
        Rule::RoutedQmvTransposed,
        Rule::AltUpStreams,
        Rule::RoutedQmvQuad,
        Rule::Single,
        Rule::SingleWarp,
        Rule::PerRequest,
    ];

    /// A rectangle with every field filled, because a head-shaped rule
    /// refuses a zero one.
    ///
    /// This used to be `Dims { rows, width, in_width }` and could not be
    /// anything else — `Dims` had three fields. It grew because the fields
    /// grew, and it fills the new ones rather than defaulting them for a
    /// reason the tests below depend on: `Ungeometric::Empty` is a real
    /// answer now, so a fixture of zeroed head counts would make every
    /// head-shaped rule refuse and `every_rule_in_the_vocabulary_is_answered`
    /// would report a vocabulary gap that is really an unfilled fixture.
    ///
    /// The numbers are llama-3.2's grouped-query shape at gemma-3n's hidden
    /// size — 32 query heads, 8 key heads, 128 channels each — because a
    /// fixture where `q_heads == kv_heads` cannot catch a rule reading the
    /// wrong one.
    ///
    /// `stated_head_dim` is ZERO for the same reason: the fixture is a fire
    /// whose statement named no per-head width, which is what all but one
    /// `OpKind` in the tree is, and a fixture that named one could not catch
    /// a rule reading [`Dims::head_dim`] where it means to read the
    /// statement's. Against `head_dim: 128` the confusion is visible at a
    /// glance — `rows` blocks or `rows * width / 128`.
    ///
    /// `requests` is 4 and `rows` is the caller's, so a fixture fire is four
    /// requests of however many tokens: a rule reading `rows` for a request
    /// axis is visible at a glance for the same reason. `altup_streams` is 4
    /// and `kv_heads` is 8, so a rule reading the head count for a stream
    /// count is too — and `H` divides both, which is why the fixture's width
    /// has to be a multiple of 8 as well as of 4.
    fn dims(rows: u32, width: u32) -> Dims {
        Dims {
            rows,
            width,
            in_width: width,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            stated_head_dim: 0,
            rotary_dims: 64,
            n_experts: 128,
            experts_per_token: 8,
            requests: 4,
            altup_streams: K,
        }
    }

    /// Every rule the JIT's rows state evaluates. The rows and the arithmetic
    /// are two crates and nothing but this test makes them agree about which
    /// rules are live.
    #[test]
    fn every_stated_rule_is_ported() {
        for k in crate::unit::rows() {
            let d = dims(T, H);
            assert!(
                !matches!(eval(k.sig.launch, d), Err(Ungeometric::Unported(_))),
                "{} states {:?}, which this runtime has not ported",
                k.sig.symbol,
                k.sig.launch
            );
        }
    }

    /// **Every variant of the vocabulary is answered, not just the ones some
    /// row happens to state today.**
    ///
    /// `every_stated_rule_is_ported` walks the rows, so it can only see what
    /// the rows already name — and a rule absent from a list nobody must
    /// remember to extend is how `LaunchRule::RouterLane` kept its defect:
    /// it dropped the row axis, a mixture prefill routed row 0 and ran every
    /// other row through the first row's experts, and it survived review
    /// because the rule was simply not in the list that would have caught it.
    /// `kernels::LaunchRule::ALL` exists for that reason and this is the
    /// reader of it here.
    ///
    /// So the vocabulary is enumerated instead: every variant either
    /// evaluates and is on [`PORTED`], or refuses in a sentence that names
    /// itself. Adding a variant to `kernels` fails this test until someone
    /// decides which.
    #[test]
    fn every_rule_in_the_vocabulary_is_answered() {
        for &rule in Rule::ALL {
            match eval(rule, dims(T, H)) {
                Ok(_) => assert!(
                    PORTED.contains(&rule),
                    "{rule:?} evaluates and is not on PORTED — either the \
                     arithmetic was added without saying so, or it is a guess"
                ),
                Err(Ungeometric::Unported(named)) => {
                    assert_eq!(named, rule, "a refusal must name the rule it refused");
                    assert!(
                        !PORTED.contains(&rule),
                        "{rule:?} is on PORTED and refuses; the list is stale"
                    );
                }
                Err(Ungeometric::Unstated) => {
                    assert_eq!(rule, Rule::Unstated, "only the empty row is unstated");
                }
                Err(Ungeometric::Empty) => panic!(
                    "{rule:?} calls a {T}x{H} rectangle empty, which no rule may: \
                     `Empty` is for a rectangle that collapsed, not for a shape \
                     a rule dislikes"
                ),
            }
        }
    }

    /// The reduction pair reproduces `compute_rms_bf16`'s launcher exactly:
    /// `compute_rms_kernel<<<T, 256, (256 / 32) * sizeof(float), stream>>>`.
    ///
    /// Transcribed from `norm/altup_aux.cu` at the commit this pilot forked
    /// from, and it is the whole precondition of the migration — a rule that
    /// does not reproduce the launcher it replaces is a rewrite, and a
    /// rewrite cannot be A/B'd against the thing it rewrote.
    #[test]
    fn rms_reproduces_the_cpp_launcher() {
        assert_eq!(
            eval(Rule::Rms, dims(T, H)),
            Ok(Launch { grid: [T, 1, 1], block: [256, 1, 1], smem: 32 })
        );
    }

    /// `tanh_kernel<<<(numel + 255) / 256, 256, 0, stream>>>`, where the
    /// row's `numel` is `rows * width`.
    #[test]
    fn elementwise_reproduces_the_cpp_launcher() {
        let numel = T * H;
        assert_eq!(
            eval(Rule::Elementwise, dims(T, H)),
            Ok(Launch { grid: [numel.div_ceil(256), 1, 1], block: [256, 1, 1], smem: 0 })
        );
    }

    /// `mean_streams_bf16` launched `dim3(T, (H + 127) / 128)` with 128
    /// threads; the rule says 256. The kernel is a pure map guarded by
    /// `h >= H`, so the two cover the same channels and answer the same
    /// bits — what a test can hold is the COVERAGE, which is the property
    /// the guard depends on.
    #[test]
    fn elementwise_rows_covers_every_channel() {
        let l = eval(Rule::ElementwiseRows, dims(T, H)).expect("rule evaluates");
        assert_eq!(l.grid[0], T, "one block per row");
        assert!(
            l.grid[1] * l.block[0] >= H,
            "grid.y ({}) x block ({}) must cover H ({H})",
            l.grid[1],
            l.block[0]
        );
        // And not by more than one block, which is what "rounded up" means
        // and what a rule that merely over-covered would fail.
        assert!((l.grid[1] - 1) * l.block[0] < H);
    }

    /// `unpack_predict_coefs_kernel<<<T, K * K>>>` — one block per row, as
    /// wide as the row. The rule rounds up to a warp, which the C++ did not,
    /// and the stride loop is what makes the two agree: sixteen elements
    /// over thirty-two threads leaves sixteen idle rather than sixteen
    /// unwritten.
    #[test]
    fn route_rows_covers_the_row_it_is_given() {
        let predict = eval(Rule::RouteRows, dims(T, K * K)).expect("rule evaluates");
        assert_eq!(predict.grid, [T, 1, 1]);
        assert!(predict.block[0] >= K * K);
        assert_eq!(predict.block[0] % 32, 0, "a partial warp is a wasted scheduler slot");

        // A row wider than any block still gets a legal launch, because the
        // kernel strides. This is the case the pre-stride kernel could not
        // have been given at all: `<<<T, 4096>>>` is a launch failure.
        let wide = eval(Rule::RouteRows, dims(T, 4096)).expect("rule evaluates");
        assert_eq!(wide.block, [1024, 1, 1]);
    }

    /// A collapsed rectangle is refused, not floored.
    #[test]
    fn an_empty_extent_is_refused() {
        assert_eq!(eval(Rule::Rms, dims(0, H)), Err(Ungeometric::Empty));
        assert_eq!(eval(Rule::ElementwiseRows, dims(T, 0)), Err(Ungeometric::Empty));
        assert_eq!(eval(Rule::RouteRows, dims(T, 0)), Err(Ungeometric::Empty));
        assert_eq!(eval(Rule::Elementwise, dims(T, 0)), Err(Ungeometric::Empty));
    }

    /// An unported rule says which one, and an unstated row says neither.
    #[test]
    fn the_two_refusals_are_different_sentences() {
        assert_eq!(eval(Rule::Unstated, dims(T, H)), Err(Ungeometric::Unstated));
        assert_eq!(eval(Rule::Qmv, dims(T, H)), Err(Ungeometric::Unported(Rule::Qmv)));
    }

    /// **A head-shaped rule refuses a rectangle with no heads in it.**
    ///
    /// The failure this rules out is not a crash. `grid.y = 0` is a legal
    /// argument to `cuLaunchKernel`: it launches no blocks, returns
    /// `CUDA_SUCCESS`, and leaves the output holding whatever the previous
    /// layer wrote — so a fire whose head count never got filled in looks
    /// exactly like a fire that ran, for as many layers as the model has.
    /// `head_dim = 0` is worse, because the grid is full and every block's
    /// loop runs zero times: the profiler shows the kernel, the timing shows
    /// the launch, and nothing was written.
    ///
    /// [`Rule::RowsPerHead`] is deliberately not in the list and
    /// [`Dims::stated_head_dim`] is deliberately not the field being zeroed:
    /// its zero is the ABSENT statement, which is the arm every plain
    /// RMSNorm takes. That is exactly why it is a second field —
    /// `a_zero_stated_head_is_the_absent_arm_and_not_a_refusal` states the
    /// other half.
    #[test]
    fn a_rule_that_reads_a_head_count_refuses_a_rectangle_without_one() {
        let headless = Dims { q_heads: 0, kv_heads: 0, ..dims(T, H) };
        let flat = Dims { head_dim: 0, ..dims(T, H) };
        for rule in [Rule::PerHead, Rule::PerHeadElementwise, Rule::GatedRms, Rule::Rope] {
            assert_eq!(eval(rule, headless), Err(Ungeometric::Empty), "{rule:?} with no heads");
            assert_eq!(eval(rule, flat), Err(Ungeometric::Empty), "{rule:?} with no channels");
        }
        assert_eq!(eval(Rule::SdpaVector, headless), Err(Ungeometric::Empty));
        assert_eq!(
            eval(Rule::RouterSort, Dims { n_experts: 0, ..dims(T, H) }),
            Err(Ungeometric::Empty),
            "a sort over no experts allocates no counters and scans them"
        );
        assert_eq!(eval(Rule::SplitPacked, Dims { in_width: 0, ..dims(T, H) }), Err(Ungeometric::Empty));
    }

    /// **The one zero that is a value**, and the reason [`Dims`] has ten
    /// fields instead of nine.
    ///
    /// A statement that names no per-head width is the COMMON case — only
    /// `OpKind::RmsnormPerHead` sets `spec.per_head_dim` — so a
    /// [`Rule::RowsPerHead`] that refused zero would refuse every plain
    /// RMSNorm in the tree, and one that read [`Dims::head_dim`] instead
    /// would take the per-head arm on a number the statement never said.
    /// Both arms are here at one rectangle, and they differ by the ratio
    /// that made this worth a field.
    #[test]
    fn a_zero_stated_head_is_the_absent_arm_and_not_a_refusal() {
        let absent = dims(T, H);
        assert_eq!(absent.stated_head_dim, 0);
        assert_ne!(absent.head_dim, 0, "the fire still has an attention head width");
        assert_eq!(
            eval(Rule::RowsPerHead, absent).map(|l| l.grid),
            Ok([T, 1, 1]),
            "one block per row: the statement named no head, so `hidden` is the whole row"
        );
        let stated = Dims { stated_head_dim: 128, ..absent };
        assert_eq!(
            eval(Rule::RowsPerHead, stated).map(|l| l.grid),
            Ok([T * (H / 128), 1, 1]),
            "one block per (row, head) once a head was named"
        );
        assert_eq!(
            eval(Rule::RowsPerHead, Dims { stated_head_dim: 96, ..absent }),
            Err(Ungeometric::Empty),
            "a named head the row's width does not divide is refused, not rounded"
        );
    }

    /// **A rule reads only the dims it names**, which is what makes [`Dims`]
    /// safe to have grown: a field added for one rule must not move another's
    /// launch.
    ///
    /// Falsified by construction — `Rms` and `Elementwise` were ported when
    /// `Dims` had three fields, so if either moves when a head count changes,
    /// the widening changed an arithmetic the migration is A/B'ing against.
    ///
    /// The tenth field gets the check in both directions, because it is a
    /// second reading of a quantity a ninth field already carries and the
    /// whole point is that the two are independent: no head-shaped rule may
    /// move when the STATEMENT's head width changes, and
    /// [`Rule::RowsPerHead`] may not move when the FIRE's does.
    #[test]
    fn a_new_field_cannot_move_an_old_rule() {
        let d = dims(T, H);
        let other = Dims { q_heads: 7, kv_heads: 3, head_dim: 64, n_experts: 4, ..d };
        for rule in [Rule::Rms, Rule::Elementwise, Rule::ElementwiseRows, Rule::RouteRows] {
            assert!(rule_eq(rule, d, other), "{rule:?} read a field it does not name");
        }
        assert_ne!(
            eval(Rule::PerHead, d),
            eval(Rule::PerHead, other),
            "and a rule that DOES name one must move"
        );

        let stated = Dims { stated_head_dim: 64, ..d };
        for &rule in PORTED {
            if rule == Rule::RowsPerHead {
                continue;
            }
            assert!(
                rule_eq(rule, d, stated),
                "{rule:?} moved when the STATEMENT's head width changed — only RowsPerHead reads it"
            );
        }
        assert_ne!(
            eval(Rule::RowsPerHead, d),
            eval(Rule::RowsPerHead, stated),
            "and the one rule that DOES name it must move"
        );

        let refired = Dims { head_dim: 64, ..stated };
        assert_eq!(
            eval(Rule::RowsPerHead, stated),
            eval(Rule::RowsPerHead, refired),
            "RowsPerHead read the FIRE's head width — it must read only the statement's"
        );
    }

    fn rule_eq(rule: Rule, a: Dims, b: Dims) -> bool {
        eval(rule, a) == eval(rule, b)
    }
}
