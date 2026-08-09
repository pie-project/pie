//! Layer 2: FlashInfer's FA2 launch arithmetic, as Rust.
//!
//! # What this module is
//!
//! [`crate::plan`] ported `scheduler.cuh` — the part of FlashInfer that
//! decides *which CTA does which work*. This module ports the other host half:
//! the part of `decode.cuh` and `prefill.cuh` that decides *what shape a CTA
//! is*, and therefore which template instantiation must exist for it.
//!
//! Both `BatchDecodeWithPagedKVCacheDispatched` (`decode.cuh:749-834`) and
//! `BatchPrefillWithPagedKVCacheDispatched` (`prefill.cuh:4176-4356`) are host
//! functions whose body is a chain of `constexpr` derivations ending in a
//! `<<<>>>`-equivalent `cudaLaunchKernel`. Under nvcc those derivations are the
//! compiler's; the four `attention_flashinfer_hd<N>.cu` translation units exist
//! precisely so that the compiler can run them ahead of time, once per
//! head_dim, and emit every instantiation the run-time arms might select.
//!
//! Under a JIT the same derivations are ordinary integer arithmetic on facts a
//! fire already holds — head_dim, the GQA group, the KV element width, the
//! device's compute capability and shared-memory budget — so they become this
//! file, and the instantiation they select becomes one row rather than a
//! translation unit.
//!
//! # This module takes numbers and returns numbers
//!
//! [`crate::plan`]'s rule, for [`crate::plan`]'s reason: no `cudarc`, no
//! feature gate, no device query. Every device fact upstream reads inline —
//! `GetCudaComputeCapability`, `cudaDevAttrMaxSharedMemoryPerMultiprocessor`,
//! `cudaDevAttrMaxSharedMemoryPerBlockOptin`,
//! `cudaOccupancyMaxActiveBlocksPerMultiprocessor` — arrives as a parameter
//! ([`Device`], [`Occupancy`]), because a derivation that queries a device
//! cannot be tested without one and the whole argument for moving this code is
//! that it is arithmetic.
//!
//! The occupancy one is the interesting parameter and north-star §5 step 7
//! names it: upstream asks
//! `cudaOccupancyMaxActiveBlocksPerMultiprocessor` **about the kernel**, which
//! is a per-cubin fact rather than a per-device one. With the kernel JIT'd that
//! becomes `cuOccupancyMaxActiveBlocksPerMultiprocessor` on the resulting
//! `CUfunction`, in layer 3 — so this module takes the answer rather than the
//! question.
//!
//! # Faithful first
//!
//! `std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL)` is not a formula to
//! improve; it is the reason a `bdx` is 16 and not 8, and a different `bdx` is
//! a different kernel. Every derivation below is transcribed with the C++ line
//! beside it, integer division included — `num_threads / (bdx * bdy)` at
//! `bdx*bdy = 24` is **5**, not 5.33 and not 6, and the 120-thread block that
//! results is what upstream launches.
//!
//! # The two things this module does NOT claim
//!
//! * **Nothing about bit-exactness against the nvcc-built kernels.** No output
//!   has been compared. What is established is that the constants here are the
//!   ones the C++ computes, by transcription with citations; that the resulting
//!   instantiation produces the same numbers as `libpie_attn_flashinfer.a` is a
//!   device measurement nobody has taken.
//!
//!   It is at least a well-posed question, and that is worth recording,
//!   because it is not automatic. `runtime::nvrtc::options` (`nvrtc.rs:861`)
//!   passes `--fmad=false --prec-div=true --prec-sqrt=true` to every unit —
//!   contraction off, division and square root correctly rounded. Those are
//!   the tree's numerics contract, they are SHARED rather than per-unit, and
//!   FA2 does not restate them ([`crate::families::fa2`]'s `OPTIONS` is one
//!   flag). Whoever takes the measurement compares against an archive built
//!   by a `cc::Build` that passes none of the three, so a difference is as
//!   likely to be nvcc's default contraction as anything in this file.
//! * **Nothing about sm_90.** [`Device`] carries a compute capability and the
//!   arithmetic reads it, but the only architecture any of this has been
//!   compiled for is `compute_89` — and that was the hand-run probe, which
//!   asked for a VIRTUAL target. The fire asks for `--gpu-architecture=sm_XY`
//!   from the device in hand, because only `sm_XY` makes NVRTC emit SASS and a
//!   `compute_XY` would hand the driver PTX to JIT a second time at load. Same
//!   front end and the same errors, so the probe's results transfer; the flag
//!   does not. §44.7's rule holds regardless: every sm_90 claim in this
//!   migration is argued from the call graph and none from a run.
//!
//! # The measurements this port must carry, and where they still live
//!
//! `driver-cuda/csrc/attn/attention_flashinfer.cu` is unusually well
//! documented and **it is still the file that runs** — north-star §5 step 8's
//! host-program half is not written yet. Nothing below has been lost. It is
//! listed because a port that consumes a measurement is a regression even if
//! it compiles, and the next person to delete that file needs the list rather
//! than a diff to read it out of.
//!
//! * **The sm_89-only `gencode`, and its named regression**
//!   (`attention_flashinfer.cu:60-69`). The `cc::Build` states
//!   `-gencode arch=compute_89,code=sm_89` and the file says what that costs:
//!   *"On an sm_90 part the three post-kernels below would fail to launch
//!   with 'no kernel image is available for execution on the device'. That is
//!   a REGRESSION IN COVERAGE against the archive build, which reads its arch
//!   list from CMake."* Under the JIT this measurement **inverts and must be
//!   restated, not copied**: NVRTC is handed `--gpu-architecture` at compile
//!   time from the device it is about to run on, so the coverage gap closes
//!   by construction and the thing worth recording is that it closed. What
//!   does NOT close is §44.7 — only `compute_89` has ever been compiled, and
//!   an sm_90 claim is still argued from the call graph.
//!
//! * **The gemma-4 planner regression** — the reason
//!   `plan_static_nonsplit_decode` exists at all
//!   (`attention_flashinfer.cu:105-115`, `:225-250`). Re-running FlashInfer's
//!   full planner per decode batch was a hundredfold cost, and the
//!   short-circuit is legal only because `DecodeWorkEstimator` has already
//!   forced split-kv to false for the TP1 latency shapes: *"the schedule is
//!   independent of KV lengths, so avoid rerunning the full FlashInfer
//!   planner for every decode batch."* Its guard is
//!   `current_device_major() >= 8 && num_requests > 0 && num_requests <= 512`
//!   and two env knobs. **This is host arithmetic and it survives verbatim
//!   into Rust** — the JIT changes nothing about it.
//!
//! * **The roofline note that bounds the short-circuit**
//!   (`attention_flashinfer.cu:241-242`): *"the static plan is unsplit by
//!   construction, which is what leaves a sliding layer at batch\*kv_heads
//!   CTAs (8 on 148 SMs for gemma-4) and ~50x off its bandwidth roofline."*
//!   It is the reason `window_split_kv_enabled() && window_left >= 0` takes
//!   the real planner instead. A Rust port that drops the windowed branch
//!   would be 50x slower on exactly one layer type and fully correct, which
//!   is the worst shape a regression can have.
//!
//! * **`head_dim_supports_cascade_merge`'s `{64, 128, 256, 512}`**
//!   (`attention_flashinfer.cu:376`, `:985`). That set is **upstream's**, not
//!   ours, and it is the same set as [`crate::families::fa2::HEAD_DIMS`] by
//!   coincidence of origin rather than by construction. Whoever changes one
//!   must say which they changed; they are two facts that happen to agree.
//!
//! * **The `.cuh` single-includer rule** (`attention_flashinfer.cu:73-80`):
//!   *"a non-template `__global__` in a `.cuh` takes external linkage, so a
//!   second includer is a `multiple definition` at link even if it never
//!   launches anything (§21.6, measured on nvcc 13.0.88)."* Under NVRTC there
//!   is no link step across translation units — each unit is its own program —
//!   so the rule **stops applying** rather than being obeyed. That is why
//!   `csrc/src/attn/fa2.cuh` can be the root of all 56 units at once, and it
//!   is the one place where a JIT retires a constraint outright.

use core::fmt;

/// The KV cache element width FA2 is launched over, in bytes.
///
/// A named type rather than a bare `usize` because `sizeof(DTypeKV)` appears in
/// four separate derivations and selects a different arm in three of them
/// (`decode.cuh:762`, `:770`, `:772`; `prefill.cuh:4225`), and because pie
/// reaches exactly one value of it — see [`KvWidth::BF16`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvWidth(pub u32);

impl KvWidth {
    /// Two bytes. **The only width the lattice instantiates.**
    ///
    /// pie dequantises every KV scheme into `kv_layer.{k,v}_bf16_pages` before
    /// FA2 sees a page — that is what
    /// `attn::dequant_kv_cache_layer_to_bf16_active` is for — so there is no
    /// dtype axis here and `kernels.def` never had one either. The parameter
    /// exists because the arithmetic branches on it and a branch with one
    /// reachable arm should still be *written* the way upstream wrote it, so
    /// that adding an fp8 KV path later is a call-site change rather than a
    /// re-derivation.
    pub const BF16: Self = Self(2);

    /// `sizeof(DTypeKV*)` — a pointer, not the element.
    ///
    /// `decode.cuh:773` sizes one of its two smem alternatives in POINTERS
    /// (`tile_size_per_bdx * num_threads * sizeof(DTypeKV*)`), which is 8 on
    /// every 64-bit target and does not vary with [`KvWidth`]. Spelled here so
    /// the transcription of `:772-775` reads like the C++ rather than like a
    /// literal 8 in the middle of it.
    pub const POINTER: u32 = 8;
}

/// The device facts the FA2 launchers query inline.
///
/// [`crate::plan::Device`] is the sibling of this and deliberately not the same
/// struct: a planner needs the SM count and this needs the shared-memory
/// budget, and merging them would make each caller supply facts its callee
/// never reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Device {
    /// `GetCudaComputeCapability().first` — `decode.cuh:763`,
    /// `utils.cuh:349`.
    ///
    /// The ONLY thing decode reads it for is
    /// `DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM`: `>= 8` gives two smem
    /// stages, anything older gives one.
    pub cc_major: u32,
    /// `cudaDevAttrMaxSharedMemoryPerMultiprocessor` — `prefill.cuh:4213-4215`.
    pub max_smem_per_sm: u32,
    /// `cudaDevAttrMaxSharedMemoryPerBlockOptin` — `prefill.cuh:4216-4218`.
    pub max_smem_per_block_optin: u32,
}

impl Device {
    /// The L40S this tree is developed on, as measured by
    /// `driver-cuda`'s device probe: sm_89, 100 KB of shared memory per SM and
    /// 99 KB opt-in per block.
    ///
    /// A named constant so that a doc test, a no-GPU unit test and the
    /// derivation table below all quote the same numbers, and so that a box
    /// with different ones is a different `Device` rather than a silently
    /// different answer. It is **not** a default: [`DecodeGeometry::derive`]
    /// takes a `Device` by value and there is no `impl Default`, because the
    /// wrong shared-memory budget produces a valid-looking `NUM_MMA_KV` and a
    /// kernel that is quietly one CTA per SM.
    pub const L40S: Self =
        Self { cc_major: 8, max_smem_per_sm: 102_400, max_smem_per_block_optin: 101_376 };
}

/// Whether a geometry could be derived, and if not, which constraint refused
/// it.
///
/// `fire/gemv.rs`'s rule — *"it declined" cannot be spelled like "it ran"* —
/// applied one layer earlier. Every arm here is a `FLASHINFER_ERROR` or a
/// `static_assert` in the C++, i.e. a case upstream itself refuses; the
/// difference is that upstream refuses at compile time for the instantiations
/// it was given and at `throw` time for the rest, and this refuses at
/// derivation time with the numbers in hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// `decode.cuh:765` — `static_assert(bdx <= 32)`.
    ///
    /// `bdx = HEAD_DIM / vec_size` and `vec_size >= HEAD_DIM/32`, so this can
    /// only fire when `head_dim` is not a multiple of 32: the integer division
    /// at `:764` then rounds `vec_size` down and `bdx` up past a warp.
    DecodeBdxOverWarp { head_dim: u32, bdx: u32 },
    /// `utils.cuh:164-183` — `DISPATCH_GQA_GROUP_SIZE`'s `else`.
    ///
    /// The kernel-side set is `{1, 2, 3, 4, 8}` and `kernels.def`'s
    /// `PIE_ATTN_DECODE_GQA` list is exactly it. Ratios outside it (5/6/7) are
    /// routed to the prefill path by the model's `force_prefill_path` gate and
    /// never reach a decode dispatch.
    DecodeGroupSize { group_size: u32 },
    /// `decode.cuh:762` — `vec_size` came out zero, which means a `head_dim`
    /// of zero reached the derivation.
    DecodeEmptyHeadDim,
    /// `utils.cuh:135-162` — `DISPATCH_CTA_TILE_Q`'s `default`.
    ///
    /// The instantiated set is `{16, 32, 64, 128}` and the planner's
    /// `fa2_determine_cta_tile_q` returns only those, so reaching this means a
    /// `PrefillPlanInfo` was built somewhere other than [`crate::plan`].
    PrefillCtaTileQ { cta_tile_q: u32 },
    /// `prefill.cuh:4270-4278` — *"Even the smallest KV tile … exceeds this
    /// GPU's shared memory per block"*.
    ///
    /// `max_num_mma_kv_smem < 1`. The head dim and tile do not fit at all on
    /// this part; upstream raises `FLASHINFER_ERROR`, which is a `throw`.
    PrefillKvTileTooLarge { head_dim: u32, cta_tile_q: u32, fixed_smem: u32, per_mma_kv: u32 },
    /// `prefill.cuh:221-232` — `KernelTraits::IsInvalid()`, reported by
    /// `:4289-4296` as *"FlashInfer Internal Error: Invalid configuration"*.
    ///
    /// Reachable here for the pairs upstream's first clause prunes — a
    /// `CTA_TILE_Q` of 32 below head_dim 512, or anything but 16/32 at head_dim
    /// 512 — which is exactly what `fa2_determine_cta_tile_q` will not produce.
    /// Carried as a refusal rather than a `debug_assert` because a row states
    /// its instantiation and a row can be edited.
    PrefillTraitsInvalid { cta_tile_q: u32, num_mma_kv: u32, num_mma_d_vo: u32 },
    /// `prefill.cuh:4298-4306` — the exact final check: the derived
    /// `SharedStoragePaged` is larger than the part's opt-in limit.
    PrefillSmemOverBudget { smem_bytes: u32, limit: u32 },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Refusal::DecodeBdxOverWarp { head_dim, bdx } => write!(
                f,
                "fa2 decode: head_dim {head_dim} gives bdx={bdx}, and decode.cuh:765 \
                 static_asserts bdx <= 32 -- head_dim must be a multiple of 32"
            ),
            Refusal::DecodeGroupSize { group_size } => write!(
                f,
                "fa2 decode: GQA group {group_size} is outside DISPATCH_GQA_GROUP_SIZE's \
                 {{1,2,3,4,8}} (utils.cuh:164); 5/6/7 route to the prefill path"
            ),
            Refusal::DecodeEmptyHeadDim => {
                f.write_str("fa2 decode: head_dim 0 has no vec_size (decode.cuh:762)")
            }
            Refusal::PrefillCtaTileQ { cta_tile_q } => write!(
                f,
                "fa2 prefill: cta_tile_q {cta_tile_q} is outside DISPATCH_CTA_TILE_Q's \
                 {{16,32,64,128}} (utils.cuh:135)"
            ),
            Refusal::PrefillKvTileTooLarge { head_dim, cta_tile_q, fixed_smem, per_mma_kv } => {
                write!(
                    f,
                    "fa2 prefill: even NUM_MMA_KV=1 does not fit for head_dim={head_dim} \
                     cta_tile_q={cta_tile_q} (fixed {fixed_smem} B + {per_mma_kv} B per tile) \
                     -- prefill.cuh:4270"
                )
            }
            Refusal::PrefillTraitsInvalid { cta_tile_q, num_mma_kv, num_mma_d_vo } => write!(
                f,
                "fa2 prefill: KernelTraits::IsInvalid() for cta_tile_q={cta_tile_q} \
                 num_mma_kv={num_mma_kv} num_mma_d_vo={num_mma_d_vo} (prefill.cuh:221)"
            ),
            Refusal::PrefillSmemOverBudget { smem_bytes, limit } => write!(
                f,
                "fa2 prefill: SharedStoragePaged is {smem_bytes} B, over this part's \
                 {limit} B opt-in limit (prefill.cuh:4298)"
            ),
        }
    }
}

// ─── decode ─────────────────────────────────────────────────────────────────

/// The seven integers `BatchDecodeWithPagedKVCacheKernel` is instantiated on,
/// plus the launch they imply.
///
/// Template parameter list, in upstream's order (`decode.cuh:401-403`):
///
/// ```text
/// POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx,
/// vec_size, bdx, bdy, bdz, AttentionVariant, Params
/// ```
///
/// The first is always `kNone` here (`fa2.cuh`'s `POS_ENC`, and pie applies
/// RoPE before attention), the last two are types a row spells; the middle six
/// are these fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeGeometry {
    /// `decode.cuh:771` via `utils.cuh:349-356` — 2 on Ampere and newer, 1
    /// otherwise.
    pub num_stages_smem: u32,
    /// `decode.cuh:770` — `GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2 : 4) : 1`.
    ///
    /// The whole of the MQA special case: at group size 1 there is only one
    /// query head per KV head, so a CTA would be four times too small, and the
    /// launcher widens the KV tile instead of the block.
    pub tile_size_per_bdx: u32,
    /// `decode.cuh:762` — `max(16 / sizeof(DTypeKV), HEAD_DIM / 32)`.
    ///
    /// Sixteen bytes is one `cp.async.ca.shared.global` of the widest kind, so
    /// the first term is *"as many elements as fit one vectorised load"*; the
    /// second is *"enough that `bdx` fits a warp"*. The max of the two is why
    /// head_dim 512 has `vec_size` 16 while 64, 128 and 256 all have 8.
    pub vec_size: u32,
    /// `decode.cuh:764` — `HEAD_DIM / vec_size`. Threads spanning one head.
    pub bdx: u32,
    /// `decode.cuh:767` — `GROUP_SIZE`. Query heads per KV head, one per
    /// thread-y.
    pub bdy: u32,
    /// `decode.cuh:769` — `num_threads / (bdx * bdy)`, INTEGER division.
    ///
    /// At GQA 3 and head_dim 64 that is `128 / 24 = 5`, and the block is
    /// 8x3x5 = 120 threads rather than 128. Upstream launches the 120.
    pub bdz: u32,
    /// `decode.cuh:768` — `max(128, bdx * bdy)`.
    ///
    /// Not `bdx*bdy*bdz`: it is the value `bdz` is derived FROM, and it is what
    /// upstream hands `cudaOccupancyMaxActiveBlocksPerMultiprocessor` at
    /// `:715`. Kept as a field for that reason — see [`Occupancy`].
    pub num_threads: u32,
    /// `decode.cuh:772-775`, whole:
    ///
    /// ```text
    /// 2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV)
    ///   + max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*),
    ///         2 * bdy * bdz * sizeof(float))
    /// ```
    ///
    /// The leading `2 *` is K and V; the `max` is the page-offset staging array
    /// against the cross-`bdz` `(m, d)` exchange, which occupy the same bytes at
    /// different times.
    pub smem_bytes: u32,
    /// The head dim this was derived for, carried so a row and a launch cannot
    /// disagree about which lattice point they are.
    pub head_dim: u32,
}

impl DecodeGeometry {
    /// `BatchDecodeWithPagedKVCacheDispatched`'s `constexpr` prologue,
    /// `decode.cuh:762-775`.
    ///
    /// # Errors
    ///
    /// [`Refusal::DecodeEmptyHeadDim`], [`Refusal::DecodeBdxOverWarp`] and
    /// [`Refusal::DecodeGroupSize`] — the two `static_assert`s and the
    /// `DISPATCH_GQA_GROUP_SIZE` `else`, in the order the C++ reaches them.
    pub const fn derive(
        head_dim: u32,
        group_size: u32,
        kv: KvWidth,
        dev: Device,
    ) -> Result<Self, Refusal> {
        if head_dim == 0 {
            return Err(Refusal::DecodeEmptyHeadDim);
        }
        // `:762`. `16UL / sizeof(DTypeKV)` and `HEAD_DIM / 32UL` are both
        // integer divisions in `size_t`; at head_dim < 32 the second term is 0
        // and the first wins, which is upstream's behaviour and not a guard.
        let a = 16 / kv.0;
        let b = head_dim / 32;
        let vec_size = if a > b { a } else { b };
        if vec_size == 0 {
            return Err(Refusal::DecodeEmptyHeadDim);
        }
        // `:764`, `:765`.
        let bdx = head_dim / vec_size;
        if bdx > 32 {
            return Err(Refusal::DecodeBdxOverWarp { head_dim, bdx });
        }
        // `:766` — DISPATCH_GQA_GROUP_SIZE, `utils.cuh:164-183`.
        if !matches!(group_size, 1 | 2 | 3 | 4 | 8) {
            return Err(Refusal::DecodeGroupSize { group_size });
        }
        // `:767`, `:768`, `:769`.
        let bdy = group_size;
        let lanes = bdx * bdy;
        let num_threads = if lanes > 128 { lanes } else { 128 };
        let bdz = num_threads / lanes;
        // `:770`. The `sizeof(DTypeKV) == 1` arm is fp8 KV, which pie does not
        // reach; it is transcribed rather than dropped so that the day it is
        // reached the tile halves the way upstream halves it.
        let tile_size_per_bdx = if group_size == 1 {
            if kv.0 == 1 {
                2
            } else {
                4
            }
        } else {
            1
        };
        // `:771` — DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM,
        // `utils.cuh:349-356`.
        let num_stages_smem = if dev.cc_major >= 8 { 2 } else { 1 };
        // `:772-775`.
        let staged = 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim * kv.0;
        let offsets = tile_size_per_bdx * num_threads * KvWidth::POINTER;
        let exchange = 2 * bdy * bdz * 4;
        let tail = if offsets > exchange { offsets } else { exchange };
        Ok(Self {
            num_stages_smem,
            tile_size_per_bdx,
            vec_size,
            bdx,
            bdy,
            bdz,
            num_threads,
            smem_bytes: staged + tail,
            head_dim,
        })
    }

    /// `decode.cuh:783` — `dim3 nthrs(bdx, bdy, bdz)`.
    #[must_use]
    pub const fn block(&self) -> [u32; 3] {
        [self.bdx, self.bdy, self.bdz]
    }

    /// `decode.cuh:782` — `dim3 nblks(padded_batch_size, num_kv_heads)`.
    ///
    /// `padded_batch_size` is `PlanInfo::padded_batch_size`, which
    /// [`crate::plan::decode`] computes; `num_kv_heads` is
    /// `params.paged_kv.num_heads`. Both are the caller's, which is why this is
    /// a method taking them rather than a field.
    #[must_use]
    pub const fn grid(padded_batch_size: u32, num_kv_heads: u32) -> [u32; 3] {
        [padded_batch_size, num_kv_heads, 1]
    }
}

// ─── the occupancy hook ─────────────────────────────────────────────────────

/// `cudaOccupancyMaxActiveBlocksPerMultiprocessor`'s answer, as a parameter.
///
/// # Why this is a type and not a `u32`
///
/// Three call sites in the FlashInfer host code take an occupancy and each
/// multiplies it by the SM count to get a grid bound
/// (`decode.cuh:715`, `scheduler.cuh:192`, `:258`). The number is a property of
/// **a compiled kernel on a device**, not of the device — the same source at a
/// different `NUM_MMA_KV` has a different answer — so passing a bare integer
/// makes it indistinguishable from the SM count that sits beside it in every
/// one of those expressions, and the two have been transposed before.
///
/// # Where the number comes from now
///
/// North-star §5 step 7, in its own words: *"occupancy comes from
/// `cuOccupancyMaxActiveBlocksPerMultiprocessor` on the JIT'd `CUfunction`"*.
/// The query needs the entry, the block size and the dynamic shared memory, and
/// all three are downstream of [`DecodeGeometry`] — `num_threads` and
/// `smem_bytes` are the two upstream passes at `decode.cuh:715-716`.
///
/// **`num_threads`, not `bdx*bdy*bdz`.** At GQA 3 those differ (128 against
/// 120), and upstream passes the former. Passing the block size instead would
/// report a slightly higher occupancy and a slightly wrong `max_grid_size`, in
/// the direction that over-splits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Occupancy {
    /// Blocks per SM the driver says this entry achieves at this block size and
    /// dynamic shared-memory size.
    pub blocks_per_sm: u32,
    /// `cudaDevAttrMultiProcessorCount`. Carried beside the occupancy because
    /// the only thing either is ever used for is their product.
    pub num_sm: u32,
}

impl Occupancy {
    /// `decode.cuh:718` — `uint32_t(num_blocks_per_sm) * uint32_t(num_sm)`.
    ///
    /// The number `crate::plan::decode::estimate` takes as `max_grid_size`.
    #[must_use]
    pub const fn max_grid_size(self) -> u32 {
        self.blocks_per_sm * self.num_sm
    }
}

// ─── prefill ────────────────────────────────────────────────────────────────

/// `prefill.cuh:72-96` — `get_num_warps_q`.
const fn num_warps_q(cta_tile_q: u32) -> u32 {
    if cta_tile_q == 32 {
        1 // HEAD_DIM_VO >= 512
    } else if cta_tile_q > 16 {
        4
    } else {
        1
    }
}

/// `prefill.cuh:83-85` — `get_num_warps_kv`, which is `4 / get_num_warps_q`.
///
/// Note upstream's parameter name at `:83` is `cta_tile_kv` and the argument
/// every caller passes is `CTA_TILE_Q` (`:4195`, `:2408`, `:3996`). The name is
/// wrong upstream and the behaviour is what is transcribed.
const fn num_warps_kv(cta_tile_q: u32) -> u32 {
    4 / num_warps_q(cta_tile_q)
}

/// `prefill.cuh:87-96` — `get_num_mma_q`.
const fn num_mma_q(cta_tile_q: u32) -> u32 {
    if cta_tile_q == 32 {
        2 // HEAD_DIM_VO >= 512
    } else if cta_tile_q > 64 {
        2
    } else {
        1
    }
}

const fn align16(n: u32) -> u32 {
    n.div_ceil(16) * 16
}

/// The eight integers `KernelTraits` is instantiated on, plus the launch they
/// imply.
///
/// `BatchPrefillWithPagedKVCacheKernel` takes ONE template argument — a
/// `KernelTraits` — and `prefill.cuh:4282-4285` builds it from these eight
/// numbers and five types. `fa2.cuh`'s `PagedTraits` alias fills the types in,
/// so a row spells `PagedTraits<MASK, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV,
/// NUM_MMA_D_QK, NUM_MMA_D_VO, NUM_WARPS_Q, NUM_WARPS_KV, Variant>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillGeometry {
    /// The planner's choice, `PrefillPlanInfo::cta_tile_q` — one of
    /// `{16, 32, 64, 128}`, decided by
    /// [`crate::plan::arith::fa2_determine_cta_tile_q`] from the batch's
    /// average packed QO length. **A run-time axis, and `kernels.def` says so:**
    /// *"CTA_TILE_Q (from plan_info)"* is on its list of axes that must stay
    /// fully instantiated.
    pub cta_tile_q: u32,
    /// `prefill.cuh:4193` — `kBf16VOSplit ? 1 : get_num_mma_q(CTA_TILE_Q)`.
    pub num_mma_q: u32,
    /// `prefill.cuh:4280-4281` — `DISPATCH_NUM_MMA_KV(min(smem budget, register
    /// budget))`, snapped DOWN to one of `{8, 4, 2, 1}` by `utils.cuh:116-133`.
    ///
    /// This is the field that makes the prefill lattice a *derivation* rather
    /// than a table: it depends on the part's shared-memory budget, so the same
    /// head dim and tile give different instantiations on different silicon.
    pub num_mma_kv: u32,
    /// `prefill.cuh:4206` — `HEAD_DIM_QK / 16`.
    pub num_mma_d_qk: u32,
    /// `prefill.cuh:4207` — `HEAD_DIM_VO / 16`.
    pub num_mma_d_vo: u32,
    /// `prefill.cuh:4194` — `kBf16VOSplit ? 2 : get_num_warps_q(CTA_TILE_Q)`.
    pub num_warps_q: u32,
    /// `prefill.cuh:4195` — `kBf16VOSplit ? 2 : get_num_warps_kv(CTA_TILE_Q)`.
    pub num_warps_kv: u32,
    /// `prefill.cuh:198` — `NUM_MMA_KV * NUM_WARPS_KV * 16`. Not a template
    /// argument; carried because the shared-storage derivation is written in
    /// terms of it, exactly as `SharedStorageQKVO`'s parameter list is.
    pub cta_tile_kv: u32,
    /// `prefill.cuh:4297` — `sizeof(typename KTraits::SharedStoragePaged)`,
    /// re-derived field by field by [`PrefillGeometry::shared_storage_paged`].
    ///
    /// **The one number in this module that is a struct layout rather than an
    /// expression**, and therefore the one that can be wrong silently.
    /// `fa2.cuh` exports the compiler's own answer as a `__device__` variable
    /// template beside every instantiation so that layer 3 can compare rather
    /// than trust; see [`PrefillGeometry::ECHO_TEMPLATE`].
    pub smem_bytes: u32,
    /// The head dim this was derived for. `HEAD_DIM_QK == HEAD_DIM_VO` for
    /// every pie call site — `attention_flashinfer.cu:1148-1149` passes
    /// `head_dim` twice — so one field states both.
    pub head_dim: u32,
}

impl PrefillGeometry {
    /// The `__device__` variable template `fa2.cuh` exports so that
    /// [`PrefillGeometry::smem_bytes`] can be checked against the compiler's
    /// `sizeof` instead of believed.
    ///
    /// A name expression is formed by substituting the same `PagedTraits<...>`
    /// the row spells; `cuModuleGetGlobal` on the lowered name and four bytes
    /// of D2H give the number NVRTC computed.
    ///
    /// # The leading `&` is not decoration, it is the whole mechanism
    ///
    /// `nvrtcAddNameExpression` refuses a bare variable name. Measured on this
    /// box with `libnvrtc.so.13`, both the variable template and a plain
    /// `__constant__ unsigned` fail identically:
    ///
    /// ```text
    /// __nv_name_map(4): error: expression must have a constant value
    /// __nv_name_map(4): error: Name expression must form address of a
    ///   __global__ function or the address of a __device__/__constant__
    ///   variable.
    /// ```
    ///
    /// With the `&` both resolve, rc=0. So a function's name IS its address
    /// and a variable's is not — which reads as an NVRTC quirk and is actually
    /// C++: `f` decays, `v` does not. It cost two probe rounds to find because
    /// the diagnostic says "must form address of" while pointing at a
    /// `__constant__` variable that plainly has one, and because the FA2
    /// kernels — the only name expressions this crate had before — are
    /// functions and never needed it.
    ///
    /// **This constant therefore includes the `&`**, and a caller concatenates
    /// the traits spelling and a `>` onto it. A version that dropped the `&`
    /// would fail at JIT time with a message about a name, having compiled
    /// every unit correctly first.
    ///
    /// # Checked once, by hand, against the compiler
    ///
    /// The probe reported `.const .align 4 .u32 probe_smem_echo = 49232;` for
    /// hd128 / `CTA_TILE_Q` 64 / `NUM_MMA_KV` 4 / `kCausal` / `VariantFull`,
    /// and [`PrefillGeometry::shared_storage_paged`] returns **49232** for the
    /// same point: 3 x 16384 for the `q`/`k`/`v` union alternative, plus five
    /// trailing `alignas(16)` members at 16 bytes each. The five are what make
    /// this worth exporting — four of them are one-element placeholders whose
    /// `sizeof` is 1, 1, 2 and 8, and reading them as their element widths
    /// under-counts by 48 bytes and produces a launch that fails at the smem
    /// cap for reasons nothing reports. One point is not the lattice, but it
    /// is the point where the arithmetic was most likely to be wrong.
    ///
    /// Nothing reads the echo at run time yet; this constant is the
    /// specification for that gate, stated where the derivation is so the two
    /// cannot be edited apart.
    pub const ECHO_TEMPLATE: &'static str =
        "&::pie_cuda_driver::kernels::attn::fa2::smem_bytes_paged";

    /// `BatchPrefillWithPagedKVCacheDispatched`'s `constexpr` prologue and its
    /// `NUM_MMA_KV` search, `prefill.cuh:4191-4306`.
    ///
    /// `use_fp16_qk_reduction` is a parameter because it is a template argument
    /// of the launcher (`:4177`) and `attention_flashinfer_common.cuh:764`
    /// passes `true`; it has no effect on the accumulator type for bf16 Q (the
    /// `std::conditional` at `:4208-4210` also requires `is_same_v<DTypeQ,
    /// half>`), and its one live effect is the rope clause of
    /// `max_num_mma_kv_reg` at `:4263-4266`, which `kNone` makes unreachable.
    /// Carried so the transcription is complete rather than pruned.
    ///
    /// # Errors
    ///
    /// [`Refusal::PrefillCtaTileQ`], [`Refusal::PrefillKvTileTooLarge`],
    /// [`Refusal::PrefillTraitsInvalid`] and [`Refusal::PrefillSmemOverBudget`]
    /// — upstream's `DISPATCH_CTA_TILE_Q` default, its two `FLASHINFER_ERROR`
    /// bounds checks and its `IsInvalid()` guard.
    pub const fn derive(
        head_dim: u32,
        cta_tile_q: u32,
        kv: KvWidth,
        use_fp16_qk_reduction: bool,
        dev: Device,
    ) -> Result<Self, Refusal> {
        if !matches!(cta_tile_q, 16 | 32 | 64 | 128) {
            return Err(Refusal::PrefillCtaTileQ { cta_tile_q });
        }
        let q_width = 2u32; // sizeof(DTypeQ), bf16

        // `:4191-4195`. `is_fp4_type_v<DTypeKV>` is false for every element
        // type in the lattice, so `kBf16VOSplit` reduces to the 16-bit clause.
        let vo_split_layout = kv.0 == 2 && head_dim >= 512 && cta_tile_q == 32;
        let (num_mma_q, num_warps_q_, num_warps_kv_) = if vo_split_layout {
            (1, 2, 2)
        } else {
            (num_mma_q(cta_tile_q), num_warps_q(cta_tile_q), num_warps_kv(cta_tile_q))
        };

        // `:4206-4207`. HEAD_DIM_QK == HEAD_DIM_VO at every pie call site.
        let num_mma_d_qk = head_dim / 16;
        let num_mma_d_vo = head_dim / 16;

        // `:4225-4228` — fp8 repack staging. False for bf16.
        let use_repack = kv.0 == 1 && head_dim != 64 && head_dim <= 256 && cta_tile_q > 16;
        // `:4230-4233` — K and V time-share one smem buffer.
        let kv_shared = num_mma_d_vo > 16
            && num_mma_d_vo % num_warps_kv_ == 0
            && (kv.0 == 2 || cta_tile_q > 16);
        // `:4234-4235`.
        let vo_split_dispatch = num_mma_d_vo > 16 && num_mma_d_vo % num_warps_kv_ == 0;

        // `:4236-4242`.
        let per_mma_kv = (if kv_shared {
            head_dim * 16 * num_warps_kv_ * kv.0
        } else {
            (head_dim + head_dim) * 16 * num_warps_kv_ * kv.0
        }) + (if use_repack { head_dim * 16 * num_warps_kv_ * q_width } else { 0 })
            + (if vo_split_dispatch { cta_tile_q * num_warps_kv_ * 16 * q_width } else { 0 });

        // `:4243-4244`, `:4245-4248` (kNone => 0), `:4249-4250`.
        let vo_split_fixed = if vo_split_dispatch { num_warps_kv_ * cta_tile_q * 8 + 2048 } else { 0 };
        let shared_rope_freq = 0; // POS_ENC is kNone; `:4245`'s guard is false.
        let fixed_smem = cta_tile_q * head_dim * q_width + vo_split_fixed + shared_rope_freq;

        // `:4256-4257`. bf16 makes this 1.
        let min_valid_mma_kv = if kv.0 == 1 && num_warps_q_ > 2 { num_warps_q_ / 2 } else { 1 };
        // `:4258-4259` — *"we expect each sm execute two threadblocks"*.
        let ctas_per_sm =
            if dev.max_smem_per_sm >= 2 * (fixed_smem + min_valid_mma_kv * per_mma_kv) { 2 } else { 1 };
        // `:4260-4261`.
        let per_block = {
            let a = dev.max_smem_per_sm / ctas_per_sm;
            if a < dev.max_smem_per_block_optin { a } else { dev.max_smem_per_block_optin }
        };
        // `:4263-4266`. The rope clause cannot fire at kNone, so this is
        // `8 / NUM_MMA_Q`; `use_fp16_qk_reduction` is read only there.
        let _ = use_fp16_qk_reduction;
        let max_mma_kv_reg = 8 / num_mma_q;
        // `:4268-4269`, `:4270-4278`.
        if per_block <= fixed_smem || (per_block - fixed_smem) < per_mma_kv {
            return Err(Refusal::PrefillKvTileTooLarge {
                head_dim,
                cta_tile_q,
                fixed_smem,
                per_mma_kv,
            });
        }
        let max_mma_kv_smem = (per_block - fixed_smem) / per_mma_kv;
        // `:4280-4281` + `utils.cuh:116-133` — snap DOWN to {8,4,2,1}.
        let budget = if max_mma_kv_smem < max_mma_kv_reg { max_mma_kv_smem } else { max_mma_kv_reg };
        let num_mma_kv = if budget >= 8 {
            8
        } else if budget >= 4 {
            4
        } else if budget >= 2 {
            2
        } else {
            1
        };

        // `prefill.cuh:221-232` — `KernelTraits::IsInvalid()`, the clauses that
        // can fire for this lattice. `DTypeQKAccum` is `float` (4 bytes) and
        // `USE_VO_SPLIT` is `vo_split_dispatch`.
        let num_mma_d_vo_tile = if num_mma_d_vo > 16 { 16 } else { num_mma_d_vo };
        let num_mma_d_vo_per_warp =
            if vo_split_dispatch { num_mma_d_vo / num_warps_kv_ } else { num_mma_d_vo };
        let reg_frags = if vo_split_dispatch { num_mma_d_vo_per_warp } else { num_mma_d_vo_tile };
        let invalid = (if head_dim >= 512 { cta_tile_q > 32 } else { cta_tile_q == 32 })
            || num_mma_d_vo < 4
            || (num_mma_d_vo == 4 && num_mma_kv % 2 == 1)
            || num_mma_q * (8 * reg_frags + 2 * 4 * num_mma_kv) >= 256;
        if invalid {
            return Err(Refusal::PrefillTraitsInvalid { cta_tile_q, num_mma_kv, num_mma_d_vo });
        }

        let cta_tile_kv = num_mma_kv * num_warps_kv_ * 16;
        let smem_bytes = Self::shared_storage_paged(
            cta_tile_q,
            cta_tile_kv,
            head_dim,
            num_warps_kv_,
            kv,
            q_width,
        );
        // `:4298-4306` — the exact final check.
        if smem_bytes > dev.max_smem_per_block_optin {
            return Err(Refusal::PrefillSmemOverBudget {
                smem_bytes,
                limit: dev.max_smem_per_block_optin,
            });
        }
        Ok(Self {
            cta_tile_q,
            num_mma_q,
            num_mma_kv,
            num_mma_d_qk,
            num_mma_d_vo,
            num_warps_q: num_warps_q_,
            num_warps_kv: num_warps_kv_,
            cta_tile_kv,
            smem_bytes,
            head_dim,
        })
    }

    /// `sizeof(SharedStorageQKVO<..., kEnableVOSplitOpt = true>)`,
    /// `prefill.cuh:98-147`, member by member.
    ///
    /// # Why this is spelled out rather than approximated
    ///
    /// It is a union of three alternatives followed by five `alignas(16)`
    /// arrays, and a `size_of` is not a sum: every member starts at a multiple
    /// of 16 and the whole is rounded to 16, which is why the four
    /// one-element placeholder arrays (`k_sf_smem`, `v_sf_smem`,
    /// `kv_smem_repack`, and one of `p_smem`/`vosplit_md_smem`) cost 16 bytes
    /// each and not 1, 1, 2 and 8. Getting that wrong under-counts by 48 bytes,
    /// which is exactly the size of error that fits inside a rounding and
    /// surfaces as a `cudaErrorInvalidValue` on one head dim only.
    ///
    /// `SharedStorageWithRopeFreq` is NOT applied: `USE_SHARED_ROPE_FREQ`
    /// (`prefill.cuh:229-230`) requires `kRoPELlama`, and the lattice is
    /// `kNone`.
    #[must_use]
    pub const fn shared_storage_paged(
        cta_tile_q: u32,
        cta_tile_kv: u32,
        head_dim: u32,
        num_warps_kv: u32,
        kv: KvWidth,
        q_width: u32,
    ) -> u32 {
        // `:102-103`, `:104`, `:108-110`. `kEnableVOSplitOpt` is true for the
        // paged storage (`:293-295`).
        let kv_share_shape = head_dim / 16 > 16 && (head_dim / 16) % num_warps_kv == 0;
        let vo_split = kv_share_shape;
        let v_share_active = kv_share_shape && (kv.0 == 2 || cta_tile_q > 16);

        // Union alternative 1, `:112-118`.
        let mut a = 0;
        a = align16(a) + cta_tile_q * head_dim * q_width; // q_smem
        a = align16(a) + cta_tile_kv * head_dim * kv.0; // k_smem
        a = align16(a) + if v_share_active { kv.0 } else { cta_tile_kv * head_dim * kv.0 };
        let a = align16(a);

        // Union alternative 2, `:119-126`.
        let sync_o_elems = if num_warps_kv == 1 || vo_split {
            1
        } else {
            num_warps_kv * cta_tile_q * if head_dim > 256 { 256 } else { head_dim }
        };
        let sync_md_elems = if num_warps_kv == 1 { 1 } else { num_warps_kv * cta_tile_q };
        let mut b = 0;
        b = align16(b) + sync_o_elems * 4;
        b = align16(b) + sync_md_elems * 8;
        let b = align16(b);

        // Union alternative 3, `:128`.
        let c = align16(cta_tile_q * head_dim * q_width);

        let mut off = if a > b { a } else { b };
        if c > off {
            off = c;
        }

        // The five trailing members, `:132-147`. `is_fp4_type_v<DTypeKV>` and
        // `USE_KV_REPACK` are both false for this lattice, so three of them are
        // one-element placeholders that still cost their alignment.
        off = align16(off) + 1; // k_sf_smem: uint8_t[1]
        off = align16(off) + 1; // v_sf_smem: uint8_t[1]
        off = align16(off) + q_width; // kv_smem_repack: DTypeQ[1]
        off = align16(off) + if vo_split { cta_tile_q * cta_tile_kv * q_width } else { q_width };
        off = align16(off) + if vo_split { num_warps_kv * cta_tile_q * 8 } else { 8 };
        align16(off)
    }

    /// `prefill.cuh:4204` — `dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV)`.
    ///
    /// The `32` is a warp and is a literal upstream; it is not `WARP_SIZE`
    /// read from anywhere.
    #[must_use]
    pub const fn block(&self) -> [u32; 3] {
        [32, self.num_warps_q, self.num_warps_kv]
    }

    /// `prefill.cuh:4203` — `dim3 nblks(padded_batch_size, 1, num_kv_heads)`.
    ///
    /// **Three axes with the middle one 1**, and the KV heads in `z` rather
    /// than `y` — decode puts them in `y`. Two launchers in one library with
    /// different conventions is the reason this is a named function per
    /// launcher and not one shared helper.
    #[must_use]
    pub const fn grid(padded_batch_size: u32, num_kv_heads: u32) -> [u32; 3] {
        [padded_batch_size, 1, num_kv_heads]
    }
}

#[cfg(test)]
mod tests {
    use super::{Device, KvWidth, PrefillGeometry};

    /// The one point where the derivation was checked against the compiler.
    ///
    /// `libnvrtc.so.13` on this box, `compute_89`, compiling
    /// `csrc/src/attn/fa2.cuh` with a `__constant__ unsigned` initialised to
    /// `sizeof(typename KT::SharedStoragePaged)` for
    /// `PagedTraits<kCausal, 64, 1, 4, 8, 8, 4, 1, VariantFull>` — head dim
    /// 128, `CTA_TILE_Q` 64, `NUM_MMA_KV` 4 — emitted
    ///
    /// ```text
    /// .const .align 4 .u32 probe_smem_echo = 49232;
    /// ```
    ///
    /// 49152 is the `q_smem`/`k_smem`/`v_smem` union alternative, three
    /// `alignas(16)` arrays of `64 * 128` bf16. The remaining **80** is the
    /// five trailing `alignas(16)` members at 16 bytes each, four of which are
    /// one-element placeholders whose element widths are 1, 1, 2 and 8. An
    /// arithmetic that read those widths instead of the alignment returns
    /// 49184 and is wrong by exactly the 48 bytes this test exists to catch.
    ///
    /// This is one point of thirty-six and it does not make the other
    /// thirty-five right. It is the point at which the layout was most likely
    /// to be mis-transcribed, and it is the only number in this module that
    /// came from a compiler rather than from reading one.
    #[test]
    fn the_shared_storage_arithmetic_agrees_with_nvrtc_at_the_probed_point() {
        assert_eq!(
            PrefillGeometry::shared_storage_paged(64, 64, 128, 1, KvWidth::BF16, 2),
            49_232,
            "NVRTC computed 49232 for `PagedTraits<kCausal,64,1,4,8,8,4,1,VariantFull>`"
        );
    }

    /// The same point, reached through `derive` rather than by hand.
    ///
    /// `shared_storage_paged` takes `cta_tile_kv` and `num_warps_kv` as
    /// parameters, so the test above can agree with the compiler while
    /// [`PrefillGeometry::derive`] feeds it the wrong ones. This closes that:
    /// at head dim 128 and `CTA_TILE_Q` 64 the launcher's search must land on
    /// `NUM_WARPS_Q` 4, `NUM_WARPS_KV` 1 and `CTA_TILE_KV` 64
    /// (`prefill.cuh:72-96`, `:198`), which is the geometry the probe spelled.
    #[test]
    fn derive_reaches_the_probed_point() {
        let g = PrefillGeometry::derive(128, 64, KvWidth::BF16, true, &Device::L40S)
            .expect("hd128 / CTA_TILE_Q 64 is a valid point");
        assert_eq!((g.num_warps_q, g.num_warps_kv), (4, 1));
        assert_eq!(g.num_mma_q, 1);
        assert_eq!((g.num_mma_d_qk, g.num_mma_d_vo), (8, 8));
        assert_eq!(
            g.cta_tile_kv,
            g.num_mma_kv * g.num_warps_kv * 16,
            "`CTA_TILE_KV = NUM_MMA_KV * NUM_WARPS_KV * 16`, `prefill.cuh:198`"
        );
        assert_eq!(
            g.smem_bytes,
            PrefillGeometry::shared_storage_paged(
                g.cta_tile_q,
                g.cta_tile_kv,
                g.head_dim,
                g.num_warps_kv,
                KvWidth::BF16,
                2,
            ),
            "the geometry's own smem must be the layout function's answer"
        );
    }
}
