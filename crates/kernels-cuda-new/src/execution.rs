use kernels::{KernelSig, Ty};

use crate::device::{DeviceKernel, Take};
use crate::{table, unit};

/// How a stated symbol is executed.
#[derive(Clone, Copy)]
pub enum Execution {
    /// A `__global__` this tree holds, compiled by NVRTC and fired by
    Jit(&'static DeviceKernel),
    /// A host program over several of our kernels.
    Composed(&'static [Step]),
    /// A library the driver links, or the driver itself.
    Service(Service),
}

/// One step of a composition -- **one variant, because the ten measured
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    /// Fire a symbol this tree already states, over the op's own operands.
    Fire {
        /// The symbol. Whatever executes it is that symbol's business.
        symbol: &'static str,
        /// Where each of the step's operands comes from, in the STEP's order,
        take: &'static [Take],
    },
}

impl Step {
    /// The symbol this step fires.
    #[must_use]
    pub fn symbol(&self) -> &'static str {
        match self {
            Step::Fire { symbol, .. } => symbol,
        }
    }

    /// The step's argument map.
    #[must_use]
    pub fn take(&self) -> &'static [Take] {
        match self {
            Step::Fire { take, .. } => take,
        }
    }
}

/// One op: a symbol, the steps it is, and the launcher they were read off.
pub struct Composition {
    /// The stated symbol this composes -- a row of [`crate::table`].
    pub symbol: &'static str,
    /// The steps, in launch order, on one stream.
    pub steps: &'static [Step],
    /// Every operand whose name CHANGES between the op and a step, written
    pub renames: &'static [(&'static str, &'static str)],
    /// The launcher, with lines, so the sequence can be checked against the
    pub because: &'static str,
}

/// The ops this crate can state -- **two of the ten, and the eight refusals
#[rustfmt::skip]
pub static COMPOSED: &[Composition] = &[
    Composition {
        symbol: "attn::compact_page_csr",
        steps: &[
            Step::Fire {
                symbol: "attn::count_kept",
                take: &[Take::From(1), Take::From(3), Take::From(5), Take::From(6), Take::From(4)],
            },
            Step::Fire {
                symbol: "attn::scan_and_scatter",
                take: &[
                    Take::From(0), Take::From(1), Take::From(2), Take::From(3), Take::From(4),
                    Take::From(5), Take::From(6), Take::From(8), Take::From(9), Take::From(7),
                ],
            },
        ],
        renames: &[("scratch_counts", "counts")],
        because: "`attn/page_compact.cu:42-51`: `count_kept<kBlock><<<num_requests, kBlock, 0, stream>>>` \
                  then `scan_and_scatter<kBlock><<<num_requests, kBlock, 0, stream>>>`, the second reading \
                  the `counts` buffer the first fills. The launcher's `if (num_requests <= 0 || \
                  scratch_counts == nullptr) return;` is not a step: the first half is \
                  `Ungeometric::Empty` from `Dims::rows`, which every rule already answers",
    },

];

/// WHO serves a symbol. A name, and nothing else.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Service {
    /// CUTLASS, through FlashInfer's grouped-GEMM MoE runner. The kernels are
    Cutlass,
    /// NCCL. `driver-cuda/build.rs:422` links `nccl`; `kernels-cuda` neither
    Nccl,
    /// The P2P/NVLink all-reduce plane -- vLLM's custom all-reduce and
    CustomAllReduce,
    /// The driver itself. A `cudaMemcpyAsync` pair, a staged LoRA apply built
    DriverOp,
}

impl Service {
    /// Every variant, so a test can assert each has a member.
    pub const ALL: &'static [Service] =
        &[Service::Cutlass, Service::Nccl, Service::CustomAllReduce, Service::DriverOp];

    /// What to print. Not a `Display` impl, because this is a label in a
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Service::Cutlass => "CUTLASS (flashinfer MoE)",
            Service::Nccl => "NCCL",
            Service::CustomAllReduce => "custom all-reduce (vLLM/TRT-LLM)",
            Service::DriverOp => "the driver itself",
        }
    }
}

/// What a symbol IS, as opposed to who runs it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum Kind {
    /// One `<<<>>>`, one instantiation. Migrable, whether or not migrated.
    Kernel,
    /// A host program over kernels.
    Op,
    /// A library, or the driver. Never a kernel.
    Service,
}

impl Execution {
    /// Which of the three kinds this execution makes the symbol.
    #[must_use]
    pub fn kind(&self) -> Kind {
        match self {
            Execution::Jit(_) => Kind::Kernel,
            Execution::Composed(_) => Kind::Op,
            Execution::Service(_) => Kind::Service,
        }
    }
}

impl core::fmt::Debug for Execution {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Execution::Jit(row) => write!(f, "Jit({})", row.sig.symbol),
            Execution::Composed(steps) => write!(f, "Composed({} step(s))", steps.len()),
            Execution::Service(service) => write!(f, "Service({service:?})"),
        }
    }
}

/// The stated symbols a SERVICE executes, with the evidence for each.
#[rustfmt::skip]
pub static SERVED: &[(&str, Service, &str)] = &[

    ("moe::flashinfer_cutlass_moe_bf16",   Service::Cutlass,
     "THE EXEMPLAR. `csrc/third_party/flashinfer_moe/*.cu` holds 0 `__global__`; `src/moe/flashinfer_moe.cu` holds 0 and calls no kernel of ours; and `cutlass/` is in no source directory of this repo -- CPM fetches it into `target/**/_deps/flashinfer-src/3rdparty/cutlass` at configure time. The kernels are templates in headers we do not have. It returns `bool`, but a service that declines is still a service: the fallback is the CALLER's, not the row's. The 0 `__global__` was read a second time and finished the argument: a file with no device text in it is not device text, so the 817-line HOST program (workspace query, arch probe, autotuner, per-device tactic memo, on-disk tactic cache, dispatch) is `driver-cuda/src/fire/flashinfer_moe.rs` and `src/moe/flashinfer_moe.cu` is a five-function `extern \"C\"` seam over `CutlassMoeFCRunner` with two standard headers. It is NOT on `RUST_SERVED`: since `moe` crossed into fn-world this symbol is `x::moe::MOE_FUSED_CUTLASS`, a `contract!` with no `Entry` -- the driver-op shape, the third row of `x/mod.rs`'s registration table -- and `bind::service::moe_flashinfer_cutlass_moe_bf16` is the seam through which it reaches `driver-cuda/src/fire/flashinfer_moe.rs`. The `Service::Cutlass` classification is unchanged and is the only claim this entry makes"),

    ("dist::all_reduce_bf16",              Service::Nccl,
     "NCCL all-reduce, in place; no `csrc/src/dist/` exists, so there is no C++ of ours to extract"),
    ("dist::all_reduce_bf16_out",          Service::Nccl,
     "NCCL all-reduce, out of place; same absent directory, same `NcclComm` method"),
    ("dist::all_gather_bf16",              Service::Nccl,
     "NCCL all-gather; `driver-cuda/build.rs:422` links `nccl` and `kernels-cuda` does not include `nccl.h` at all"),

    ("comm::all_reduce_bf16",                   Service::CustomAllReduce,
     "`impl_->allreduce<__nv_bfloat16>`, vLLM's one/two-shot NVLink kernel; was `custom_all_reduce.cu:603-621`, now `fire/all_reduce.rs::CustomAllReduce::all_reduce_bf16`, header fetched not vendored. A null `car` is a REFUSAL, not a fallback (`Decline::NoInstance`)"),
    ("comm::all_reduce_residual_rmsnorm_bf16",  Service::CustomAllReduce,
     "`flashinfer::trtllm_allreduce_fusion`'s `kARResidualRMSNorm` pattern -- 1 of the 240 template points `kernels.def`'s `PIE_AR_FUSION_PATTERN` axis existed to prune; was `custom_all_reduce.cu:623-662`, now `fire/all_reduce.rs::CustomAllReduce::all_reduce_residual_rmsnorm_bf16`. Declines when `can_fuse_residual_rmsnorm` refuses -- the fused landing IS this kernel and there is no other way to spell it"),

    ("qwen35_verify_stash_store", Service::DriverOp,
     "a `cudaMemcpyAsync` trio moving a layer's in-proj triple into the verify stash; `executor_bind.rs::AWAITING_THE_VERIFY_STASH_POOL`. No launcher, no grid"),
    ("qwen35_verify_stash_load",  Service::DriverOp,
     "the load half of the same trio, moving the stash back into the workspace"),
    ("pie_lora_qkv_correction",   Service::DriverOp,
     "the driver's own arm: `bind/mod.rs:1895` calls `(*state).apply(ctx.cublas, ...)`, a staged LoRA apply built out of grouped GEMM calls the driver already had. With no adapters staged it does NOTHING, which is an answer a `__global__` could not give"),

    ("moe::build_moe_ptrs_aligned_bf16", Service::DriverOp,
     "the aligned MoE leg's step 3 of 8: one launch of `build_moe_ptrs_aligned<bf16>` that bakes three staging bases into six device pointer arrays. It DECLARES `gu_stage`/`act_stage`/`out_stage`, the destinations every op below it writes into, so the aligned leg cannot start without it -- which makes this symbol the gate on retiring `moe::flashinfer_cutlass_moe_bf16`, the only leg qwen3.5 decode takes and the one every aligned-leg condition already falls back FROM. Body: `driver-cuda/src/fire/moe_ptrs.rs`, whose per-fire bump arena carves the six, called from `bind/mod.rs`'s driver-op table beside `pie_lora_qkv_correction`. It had a `Walk` until this entry existed and `a_walk_is_only_a_walk` is right to refuse both -- see the note where that walk stood for where its `Control::Supplies` went"),

    ("moe::moe_grouped_gemm_bf16", Service::DriverOp,
     "the aligned leg's steps 4 and 5, and the only consumer the pointer build's six arrays have. Two implementations behind one symbol, chosen by `x::moe::supported`: the WMMA kernel inside its rectangle (3.0x the library at both of qwen3.5's shapes, `x/moe.rs`'s decode census) and `x::gemm::dense::batched_act_x_wt_bf16` outside it, which is `gemm.cpp:1145-1241` -- grouped-batched falling back to plain batched, with the stream-capture latch. qwen3.5 needs both: gate_up is `K=2048` against a `SHORT_K` of 512 and down is `K=512`. Body: `driver-cuda/src/fire/moe_grouped.rs`, which reads `ctx.moe_ptrs` and picks its triple by the bank pointer the statement names. A `contract!` with NO `Entry` in `x::moe`, third registration shape -- the bind that stood there served half the symbol and its `Refusal::Wide` was final by `bind/mod.rs`'s \"a refusal is not a fallthrough\""),

    ("gemm::act_x_wt_bf16",             Service::DriverOp,
     "`bind::quant_gemm::act_x_w` is the router the lowering reaches (it is emitted as `gemm::act_x_w`, this symbol's contract's `lowered_as`), and its bf16 arm is a direct Rust call to `x::gemm::dense::act_x_wt_bf16` — the autotuner, the cuBLASLt plan cache and the on-disk tactic cache. `beta` is `spec.beta_one`'s residual fold, which only a driver op can see"),
    ("gemm::act_x_wt_bf16_out_fp32",    Service::DriverOp,
     "one `cublasGemmEx`, bf16 in / fp32 out; `gemm.cpp:1030-1058` was the whole body and the body is now `x::gemm::act_x_wt_bf16_out_fp32`. The measurement that made it `Service::Cublas` still holds -- extracting a kernel from it extracts nothing -- and the handle it needs is the driver's"),
    ("gemm::act_x_wt_bias_bf16",        Service::DriverOp,
     "a `gemm::act_x_wt_bf16` and then a `norm::add_bias_bf16` over the result (`gemm.cpp:2395-2398`), which `COMPOSED` stated as two ops and fn-world spells as a two-call body: `x::gemm::act_x_wt_bias_bf16`. Its `beta` is a literal 0.0, so the one fact the dense sibling needs, this one never asks"),
    ("gemm::grouped_act_x_wt_bf16",     Service::DriverOp,
     "one `cublasGemmGroupedBatchedEx`; `gemm.cpp:1242-1294`. Measured, not read: it is CLASSIC cuBLAS, not the cuBLASLt the previous entry claimed. The group boundaries are fire-global and no `Source` names one, so its consumer is and always was `fire::lora`'s hand-written staged apply, calling `x::gemm::grouped_act_x_wt_bf16` directly"),

    ("attn::dispatch_attention_flashinfer_decode", Service::DriverOp,
     "the plain paged decode. RESOURCE: `DecodePlanCache` (and `decode_plan_full`, the gemma-4 second kind -- `bind::attn_plan` picks between them by the layer's window, which is a DECISION and the reason it is module-level in `bind/mod.rs` rather than nested in the generated dispatcher that used to be its only caller). Arm: `bind::fa2_decode`, over `fire::flashinfer_fa2_dispatch::attn_dispatch_attention_flashinfer_decode` -> `::decode` -> `fire::flashinfer_fa2::fire_decode` -- a KV dequant of the active pages, the three-arm variant cascade (`decode_arm`, whose ORDER is load-bearing: a windowed layer with a soft cap takes the soft-cap arm), then one `fire_raw` with a `DecodeParams` by value. Transcribed from `driver-cuda/csrc/attn/attention_flashinfer.cu:490-522, 660-692` (deleted)"),
    ("attn::dispatch_attention_flashinfer_decode_capture", Service::DriverOp,
     "the same decode with the attention scores captured. RESOURCE: the same `DecodePlanCache`, plus the score sink itself -- `AttnCtx::score_out` is arena-STABLE across a fire because the capture predicate is FOLDED, so one exec serves a fire that wants scores and one that does not, and an address recorded at bind time has to still be right when the predicate goes true. A trace value could not promise that. Arm: `bind::fa2_decode_capture`, over `fire::flashinfer_fa2_dispatch::decode_capture`; the capture arm cascade is `families::fa2`'s `DecodeArm::{CaptureFull, CaptureWindow}` and the params mirror is `fa2::params::DecodeScoreParams`. THE POST-KERNEL IS STILL A ROW: `attn::attn_score_normalize` in `families::attn`'s `ATTN_SCORE_POST`, fired by `driver-cuda/src/fire/attn_score.rs`. Transcribed from `attention_flashinfer.cu:532-594` (deleted)"),
    ("attn::dispatch_attention_flashinfer_prefill_bf16", Service::DriverOp,
     "the plain paged prefill. RESOURCE: `PrefillPlanCache`. Arm: `bind::fa2_prefill`, over `fire::flashinfer_fa2_dispatch::prefill` -> `fire::flashinfer_fa2::fire_prefill`. The `DISPATCH_NUM_MMA_KV` switch (`utils.cuh:116-133`) does NOT survive as a switch: the archive instantiated all four points because the choice came from a device query, and `fa2::PrefillGeometry::derive` makes the query once on the host so the fire names ONE unit -- the largest single saving of this migration and invisible in the row count. An SM90 plan is `Decline::Sm90Unported` and NOT a forward to `kernels-cuda`'s hopper unit, per §44.7. Transcribed from `attention_flashinfer.cu:776-836` (deleted)"),
    ("attn::dispatch_attention_flashinfer_prefill_capture_bf16", Service::DriverOp,
     "the prefill sibling of the capturing decode. RESOURCE: `PrefillPlanCache` plus the arena-stable sinks, and it takes one MORE than the decode form -- `folded_out` beside `score_out`, since a prefill's raw scores and their per-request fold are different extents. Arm: `bind::fa2_prefill_capture`, over `fire::flashinfer_fa2_dispatch::prefill_capture`; `make_prefill_params` is the one Rust function all four prefill dispatches share. TWO post-kernels, both rows: `attn::attn_prefill_score_normalize` on `dim3(nr, nh, window) x 256` and `attn::attn_prefill_score_fold` on `dim3(nr, 32u) x 256`. The `32u` and `attn_score_fold_heads`' `64u` are DIFFERENT literal grid axes in one file, which is why neither is a `LaunchRule`. Transcribed from `attention_flashinfer.cu:831-933` (deleted)"),
    ("attn::dispatch_attention_flashinfer_prefill_custom", Service::DriverOp,
     "the custom-mask prefill. RESOURCE: `PrefillPlanCache` -- raised by `Prepare::CustomPlan` and not `PrefillPlan`, because the prepare that stages the mask and its CSR is a different one, plus `AttnCtx::mask_d` / `mask_indptr_d`, which ride the ctx for the score sink's reason -- the predicate is folded. Arm: `bind::fa2_prefill_custom`. Two arms, `PrefillArm::{Custom, CustomSoftcap}`, and NO causal axis: the mask IS the causality, so a custom dispatch that also set `CAUSAL` would mask twice. `window_left` is written `-1` for the same reason and NOT taken from the cache -- the one place the params filler's cache-sourced window is overwritten. Transcribed from `attention_flashinfer.cu:1115-1224` (deleted)"),
    ("attn::attention_flashinfer_prefill", Service::DriverOp,
     "the PLANLESS prefill: it plans and fires in one call, so the planner's own refusals are its refusals too. RESOURCE: **NONE, and that is the point** -- `fire::flashinfer_fa2_dispatch::attn_attention_flashinfer_prefill` builds a `PrefillPlanCache` on its own stack, drops it, and asks `plan_device()`, a read-only capability query and not a pool. It is a driver op by the second condition alone: it walks `AttnCtx::qo_indptr_h` and `kv_page_indptr_h`, the HOST mirrors of the CSR, to learn `num_pages_in_batch` -- and no `Cx` query answers a host pointer. Arm: `bind::fa2_prefill_planless`, which is the one of the six that also needs `rows`. `causal = true`, `full_attention_variant = false`, from the C++'s own call at `attention_flashinfer.cu:935-1075` (deleted)"),

    ("gemm::mla_absorb_q_to_latent_bf16", Service::DriverOp,
     "one `cublasGemmStridedBatchedEx` over the head axis, `batchCount = heads`; `gemm.cpp:2419-2442`, whose own comment names the per-head scalar kernels it REPLACED. Body: `x::attn::mla_absorb_q_to_latent_bf16`, taking `handle: *mut c_void` as fn-world spells a resource it cannot own. Arm: `bind::mla_absorb_q`, reading the four widths off `LaunchSpec::params` because both absorbs take the WHOLE `kv_b_proj` bank and slice it themselves. The archive's `tokens <= 0 || heads <= 0` early return is a `Refusal::Empty` now: under `void` a caller could not tell it from a launch"),
    ("gemm::mla_absorb_latent_to_v_bf16", Service::DriverOp,
     "the second absorb, same single strided-batched call; `gemm.cpp:2444-2468`. It reads the SECOND half of each head's bank -- `kv_b_proj + qk_nope_dim * kv_lora_rank` bf16 elements -- which is the one pointer arithmetic step the port had to carry, and `OP_T` on the weight where the first absorb takes `OP_N`. Body: `x::attn::mla_absorb_latent_to_v_bf16`; arm: `bind::mla_absorb_v`"),
];

/// **The rows the DRIVER executes, in Rust — the consumer side of the
pub static RUST_SERVED: &[&str] = &[

    "comm::all_reduce_bf16",
    "comm::all_reduce_residual_rmsnorm_bf16",

];

/// The service that executes a symbol, if a service does.
#[must_use]
pub fn service(symbol: &str) -> Option<Service> {
    SERVED.iter().find(|(s, _, _)| *s == symbol).map(|(_, service, _)| *service)
}

/// The composition that executes a symbol, if one does.
#[must_use]
pub fn composition(symbol: &str) -> Option<&'static Composition> {
    COMPOSED.iter().find(|c| c.symbol == symbol)
}

/// The operand list a fire of `symbol` would bind.
#[must_use]
pub fn sig_of(symbol: &str) -> Option<&'static KernelSig> {
    if let Some((_, unit)) = unit::unit_of(symbol) {
        if let Some(row) = unit.row(symbol) {
            return Some(row.sig);
        }
    }
    table::sig(symbol)
}

/// The `const` view of a pointer type, for the one narrowing a composition
fn read_only(ty: Ty) -> Option<Ty> {
    Some(match ty {
        Ty::BufMut => Ty::Buf,
        Ty::U32sMut => Ty::U32s,
        Ty::I32sMut => Ty::I32s,
        Ty::F32sMut => Ty::F32s,
        Ty::U8sMut => Ty::U8s,
        Ty::U16sMut => Ty::U16s,
        Ty::I8sMut => Ty::I8s,
        _ => return None,
    })
}

impl Composition {
    /// Everything about this composition a machine can check, checked.
    pub fn agrees(&self) -> Result<(), String> {
        let op = table::sig(self.symbol)
            .ok_or_else(|| format!("`{}` is composed and is not a row of `table::KERNELS`", self.symbol))?;
        if self.steps.len() < 2 {
            return Err(format!(
                "`{}` composes {} step(s) -- a SEQUENCE of one launch is a kernel and of none is \
                 nothing, and neither is an op. This check counts LAUNCHES, and that is only the \
                 right count while every `Step` is one: a choice-shaped step would name two rows \
                 in one step, and this rule would have to be re-derived as *fewer than two ROWS* \
                 rather than repaired. It is written this way deliberately -- an op that chooses \
                 between two kernels at one launch site is NOT excluded by anything here except \
                 the absence of a variant to spell it",
                self.symbol,
                self.steps.len()
            ));
        }
        if self.because.len() < 40 {
            return Err(format!("`{}` is composed on a citation too short to check", self.symbol));
        }
        for step in self.steps {
            let symbol = step.symbol();
            let at = format!("`{}` step `{symbol}`", self.symbol);
            if symbol == self.symbol {
                return Err(format!("{at} fires itself"));
            }
            let sig = sig_of(symbol)
                .ok_or_else(|| format!("{at} names a symbol no row and no unit states"))?;
            let take = step.take();
            if take.len() != sig.operands.len() {
                return Err(format!(
                    "{at} takes {} arguments and the row that would fire declares {}",
                    take.len(),
                    sig.operands.len()
                ));
            }
            for (slot, take) in take.iter().enumerate() {
                let wants = sig.operands[slot];
                match take {
                    Take::From(index) => {
                        let Some(source) = op.operands.get(*index) else {
                            return Err(format!(
                                "{at} fills `{}` from operand {index} of an op with {}",
                                wants.name,
                                op.operands.len()
                            ));
                        };
                        if source.ty != wants.ty && read_only(source.ty) != Some(wants.ty) {
                            return Err(format!(
                                "{at} fills `{}` ({:?}) from `{}` ({:?})",
                                wants.name, wants.ty, source.name, source.ty
                            ));
                        }
                        if source.name != wants.name
                            && !self.renames.contains(&(source.name, wants.name))
                        {
                            return Err(format!(
                                "{at} fills `{}` from `{}` and the composition does not state that \
                                 rename -- an unwritten rename is how a transposition of two \
                                 same-typed operands passes every type check there is",
                                wants.name, source.name
                            ));
                        }
                    }
                    Take::Null => {
                        if !wants.nullable {
                            return Err(format!(
                                "{at} nulls `{}`, which the row does not declare nullable",
                                wants.name
                            ));
                        }
                        if read_only(wants.ty).is_none() && !matches!(wants.ty, Ty::Buf | Ty::U32s | Ty::U8s | Ty::I32s | Ty::F32s | Ty::U16s | Ty::I8s | Ty::Bf16s | Ty::F16s | Ty::I64s) {
                            return Err(format!(
                                "{at} nulls `{}`, which is {:?} and not a pointer",
                                wants.name, wants.ty
                            ));
                        }
                    }
                }
            }
        }
        for (from, to) in self.renames {
            let used = self.steps.iter().any(|step| {
                let Some(sig) = sig_of(step.symbol()) else { return false };
                step.take().iter().enumerate().any(|(slot, take)| {
                    matches!(take, Take::From(index)
                        if op.operands.get(*index).is_some_and(|o| o.name == *from)
                            && sig.operands.get(slot).is_some_and(|w| w.name == *to))
                })
            });
            if !used {
                return Err(format!(
                    "`{}` states the rename `{from}` -> `{to}` and no step performs it",
                    self.symbol
                ));
            }
        }
        Ok(())
    }

    /// Whether every step of this composition, transitively, is something this
    #[must_use]
    pub fn fireable(&self) -> bool {
        self.steps.iter().all(|step| match execution(step.symbol()) {
            Some(Execution::Jit(_)) => true,
            Some(Execution::Composed(_)) => {
                composition(step.symbol()).is_some_and(Composition::fireable)
            }
            _ => false,
        })
    }
}

/// No symbol composes itself, through any number of steps.
pub fn acyclic(table: &[Composition]) -> Result<(), String> {
    #[derive(Clone, Copy, PartialEq)]
    enum Colour {
        White,
        Grey,
        Black,
    }
    let mut colour = vec![Colour::White; table.len()];
    let index_of = |symbol: &str| table.iter().position(|c| c.symbol == symbol);

    for start in 0..table.len() {
        if colour[start] != Colour::White {
            continue;
        }
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
        colour[start] = Colour::Grey;
        while let Some((node, cursor)) = stack.pop() {
            if cursor == table[node].steps.len() {
                colour[node] = Colour::Black;
                continue;
            }
            stack.push((node, cursor + 1));
            let Some(next) = index_of(table[node].steps[cursor].symbol()) else { continue };
            match colour[next] {
                Colour::Grey => {
                    let mut path: Vec<&str> =
                        stack.iter().map(|(n, _)| table[*n].symbol).collect();
                    path.push(table[next].symbol);
                    return Err(format!(
                        "a composition cycle: {} -- a step names a symbol whose steps reach it \
                         again, so expanding this op does not terminate",
                        path.join(" -> ")
                    ));
                }
                Colour::White => {
                    colour[next] = Colour::Grey;
                    stack.push((next, 0));
                }
                Colour::Black => {}
            }
        }
    }
    Ok(())
}

/// How a symbol executes -- the join over all three tables.
#[must_use]
pub fn execution(symbol: &str) -> Option<Execution> {
    if let Some(service) = service(symbol) {
        return Some(Execution::Service(service));
    }
    if let Some(composition) = composition(symbol) {
        return Some(Execution::Composed(composition.steps));
    }
    let (_, unit) = unit::unit_of(symbol)?;
    unit.rows.iter().find(|row| row.sig.symbol == symbol).map(Execution::Jit)
}

#[cfg(test)]
mod tests {
    use super::{
        COMPOSED, Execution, Kind, SERVED, Service, Step, execution,
        service,
    };

    /// Every variant of [`Service`] has at least one member.
    #[test]
    fn no_service_name_is_unevidenced() {
        for service in Service::ALL {
            assert!(
                SERVED.iter().any(|(_, s, _)| s == service),
                "`Service::{service:?}` has no member -- a library named on nobody's evidence"
            );
        }
    }

    /// Every entry carries a citation, and no symbol is served twice.
    #[test]
    fn every_served_row_is_cited_once() {
        let mut seen: Vec<&str> = Vec::new();
        for (symbol, _, why) in SERVED {
            assert!(!seen.contains(symbol), "`{symbol}` is served twice");
            assert!(why.len() > 20, "`{symbol}` is served on a citation too short to check: {why:?}");
            seen.push(symbol);
        }
    }

    /// The lookup is the table, and a symbol nobody serves gets `None`.
    #[test]
    fn the_lookup_is_the_table() {
        for (symbol, expected, _) in SERVED {
            assert_eq!(service(symbol), Some(*expected));
            assert!(matches!(execution(symbol), Some(Execution::Service(s)) if s == *expected));
        }
        assert_eq!(service("norm::residual_add_bf16"), None);
    }

    /// The kind of an execution is the kind of its arm -- three arms, three
    #[test]
    fn the_four_arms_are_the_three_kinds() {
        const STEPS: &[Step] = &[Step::Fire { symbol: "norm::residual_add_bf16", take: &[] }];
        assert_eq!(Execution::Composed(STEPS).kind(), Kind::Op);
        assert_eq!(Execution::Service(Service::Nccl).kind(), Kind::Service);

        let jit = crate::unit::rows().next().expect("some unit hosts a row");
        assert_eq!(Execution::Jit(jit).kind(), Kind::Kernel);

        let composed = COMPOSED.first().expect("COMPOSED still has a member");
        assert_eq!(Execution::Composed(composed.steps).kind(), Kind::Op);
    }

    /// A ROW MAY NOT BE TAKEN OVER BEFORE IT IS CLASSIFIED.
    #[test]
    fn every_taken_over_row_was_classified_first() {
        for symbol in super::RUST_SERVED {
            assert!(
                super::service(symbol).is_some() || super::composition(symbol).is_some(),
                "`RUST_SERVED` names `{symbol}`, which is in neither `SERVED` nor `COMPOSED`. \
                 The list drops the row's shim entry, so the C++ body goes -- state what the \
                 body IS, with the file and line, before taking it over."
            );
        }
    }

}
