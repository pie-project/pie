//! Dense matmul: the x·Wᵀ family, batched, grouped, and the cuBLAS routes.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::operands;
use kernels::KernelSig;
use kernels::Lit;
use kernels::Source;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // The plain x·Wᵀ, which every family fires and which the table had
    // never carried -- invisible to the audit until its launcher regex
    // stopped requiring the return type to start the line (`inline void`).
    // ── COLLECTIVES ────────────────────────────────────────────────
    //
    // A collective is a launch like any other, with two things in its
    // row that no other row here says.
    //
    // `whole`, for a reason stronger than XQA's: every rank must enter
    // the same collective the same number of times, so a row window
    // that split one rank's launch and not another's would deadlock
    // rather than compute the wrong answer. The refusal is not an
    // optimisation.
    //
    // And they are SYNCHRONISATION points. The graph-capture rules have
    // to know that, which is why they are stated rather than reached
    // for through `tp->` from inside a hand-written pass.
    // THE THREE NCCL COLLECTIVES ARE UNSTATED, and not for want of a
    // signature: they are METHODS on `NcclComm`, which lives in the
    // DRIVER, and `kernels-cuda` neither includes `nccl.h` nor links
    // NCCL. A free wrapper here would have to either take a driver type
    // this crate cannot see, or reimplement the dispatch each method
    // does -- the custom-all-reduce fast path, the watchdog count, the
    // async NCCL error check -- which is a second implementation, not a
    // wrapper.
    //
    // The fused landing below was the OTHER kind and went in: its method
    // is on `kernels::comm::CustomAllReduce`, a kernels-side class, so a
    // free form taking the instance needed nothing this crate lacks.
    //
    // What would close these is a layer decision rather than a
    // signature: either `kernels-cuda` gains an NCCL dependency and the
    // collectives move down, or the symbols admit they name DRIVER
    // operations and the ABI grows a second namespace root. Both are
    // real answers; neither is this file's to pick.
    kernel!(all_reduce "dist::all_reduce_bf16", whole = true,
        in_place = &[(0, 0)]),
    // The OUT-OF-PLACE sum. Same collective, a separate destination --
    // which the two-step landing needs, because its residual add reads
    // the summed partial and writes somewhere else again. No alias
    // pair, and that absence is the whole difference from the row
    // above.
    kernel!(all_reduce_out "dist::all_reduce_bf16_out", whole = true),
    kernel!(all_gather "dist::all_gather_bf16", whole = true),
    // The FUSED landing: sum, add the residual, norm. Two results — the
    // residual stream updated in place (operand 1) and the normed
    // activation — which is why the row needs a pair list and not a
    // single alias. Whether a fire takes this or the two-step form is a
    // GUARD in the text, not a driver test: see `all_reduce_residual_rmsnorm`.
    // The P2P arm of the all-reduce, and the one a row can describe: a
    // kernels-side kernel reached through a kernels-side instance.
    //
    // WHAT THE RUST FORM OF THESE TWO NEEDS (`new-horizon.md` §50.6 in full).
    // `comm/custom_all_reduce.cu` holds zero `__global__`, zero `__device__`
    // and zero `<<<>>>` — the one `__global__` it ever had was the `_exact`
    // twin, deleted with its row. Every kernel it reaches is a template in
    // vLLM's or trtllm's headers, all four of which are CPM-fetched and none
    // vendored. §43.4 records the reachability audit calling this lifecycle
    // dead AND BEING WRONG; both rows below are live and
    // `model/tests/tp_quantized_spec.rs` fires the second.
    //
    //  1. FIRES one of vLLM's one-shot/two-shot NVLink kernels via
    //     `impl_->allreduce<__nv_bfloat16>`, or trtllm's
    //     `allreduce_fusion_kernel_launcher<Pattern, T, NRanks, fp32_acc>`.
    //     One fire per call.
    //  2. INTERMEDIATES: none between kernels — but a lifecycle underneath
    //     (`vllm::Signal` and `vllm::RankData` banks, three fusion buffers, a
    //     Lamport region, the peer pointers). That is a Rust struct with
    //     `Drop`, not a fire.
    //  3. HOST DECIDES peer-access eligibility, `can_handle` on message size,
    //     fusion-versus-plain, `fp32_acc`, and `NRanks` — a TEMPLATE ARGUMENT
    //     taken from `world_size_`. That last is the `gemv` shape exactly:
    //     one instantiation per rank count, chosen on a runtime fact, which
    //     is what `Specialisation` says. A null `car` is a REFUSAL, and a
    //     refusal stays a refusal.
    //  4. MISSING: the by-value aggregate (`AllReduceFusionParams<T>` crosses
    //     by value — see the note on `ArgValue` in `runtime/args.rs`), and a
    //     `Source` binding one device address per rank. `Ty::BufArray` is
    //     `const void* const*`, which is precisely that shape, and it already
    //     exists — NO TABLE ROW USES IT YET. What is missing is the binding,
    //     not the type.
    //
    // The lifecycle needs no vocabulary at all. `cudaIpcGetMemHandle`,
    // `cudaIpcOpenMemHandle`, `cudaDeviceEnablePeerAccess` and `cudaMalloc`
    // are all bound in `cudarc::driver::sys`, so the ctor, dtor,
    // `register_buffer`, `register_graph_buffers` and `can_handle` become
    // Rust without a single new `Source` — they are not fires and never
    // needed a row.
    //
    // **THAT HAS HAPPENED.** `driver-cuda/src/fire/all_reduce.rs` is the
    // whole lifecycle, and `custom_all_reduce.{cu,hpp}` and
    // `custom_all_reduce_stub.cpp` are DELETED. The paragraph below was
    // written when the fire and the lifecycle had to cross together; they did
    // not, and the reason is the measurement that was taken afterwards:
    // `custom_all_reduce.cu` held **zero `__global__` and zero `<<<>>>`**.
    // There was no C++ launcher needing the C++ object — there was a 664-line
    // host program and two calls into headers. So the lifecycle crossed
    // alone, and the two calls became refusals that name what would satisfy
    // them (`fire::all_reduce::Decline::NoDeviceText`, carrying the resolved
    // template point's name expression).
    //
    // Both rows are on `execution::RUST_SERVED` as well as `execution::SERVED`
    // — the first pair in the tree to be on both — so `abi::emit_c_shim` drops
    // their shim entries and `bind::service` spells them. **They remain fully
    // unsourced**, which is the honest state: no operand of either row has a
    // `Source`, so `emit_rust_dispatch` skips them whole and always did. The
    // mechanism that carries each row is therefore the shim entry, not a
    // dispatch arm.
    //
    // THE REMAINING BLOCKER IS UNCHANGED, and it is only about the two
    // launches now. NVRTC reads the vendored tree through
    // `Headers::LibraryAndVendor`; the CPM checkout (`${flashinfer_SOURCE_DIR}`)
    // is a C++ compiler include path and is on no NVRTC path.
    // `kernels-cuda-new/csrc/vendor/flashinfer/` has **no `comm/` directory at
    // all**, so both of the surviving headers —
    // `comm/trtllm_allreduce_fusion.cuh` and `comm/vllm_custom_all_reduce.cuh`
    // — are unreachable to NVRTC. There is no device text to carry, hence no
    // unit, hence no row to fire. **The vendored tree must gain `comm/`
    // first**, and that tree is `vendor-role`'s.
    //
    // So the dependency order for what is LEFT is: header, then unit, then
    // row, then the by-value aggregate. The Rust that would fire it is
    // already written and already computes every one of the launcher's
    // arguments; what it does at the launch point is decline.
    //
    // See the same finding for the sm90 prefill in `families/attn.rs`, and
    // `new-horizon.md` §50 for what the split between the vendored and the
    // CPM-only headers turns out to mean.
    //
    // ITS STUB CARRIED A LATENT DEFECT, and this is the only place it is
    // written down. `custom_all_reduce_stub.cpp` was selected on sm_100/sm_120;
    // its ctor and `register_buffer` still took `NcclComm&`, while
    // `custom_all_reduce.hpp` had replaced that parameter with
    // `HostAllgather`. **The stub could not compile if it were ever
    // selected.** It was invisible on this box — an L40S, sm_89, which took
    // the real file — and it was pre-existing, not caused by any move.
    // Whoever first built for Blackwell would have met it.
    //
    // The fix was not to repair the stub. Under §48.3 a stub goes when the
    // thing it stands in for goes, and a stub's Rust form is not a file: an
    // architecture stub exists because a real implementation cannot compile
    // for the target, and in Rust that is `#[cfg]` or a runtime capability
    // check, both of which the driver already does. A JIT unit that is never
    // selected costs nothing to not compile — which is exactly what an AOT
    // archive cannot do, since it must hold a symbol for every target it might
    // run on. The stub was an artefact of ahead-of-time linking and did not
    // survive the move to NVRTC. **It is deleted, along with the thing it
    // stood in for**, and the arch-conditional `set()` in
    // `kernels-cuda/csrc/CMakeLists.txt` went with it.
    kernel!(all_reduce_p2p "comm::all_reduce_bf16", whole = true,
        operands = operands![
            car: CustomAllReduce,
            input: Buf,
            output: BufMut,
            count: Usize,
            stream: Stream,
        ]),
    kernel!(all_reduce_residual_rmsnorm "comm::all_reduce_residual_rmsnorm_bf16",
        whole = true, in_place = &[(0, 1)],
        operands = operands![
            car: CustomAllReduce,
            input: Buf,
            residual_inout: BufMut,
            rms_gamma: Buf,
            norm_out: BufMut,
            tokens: I32,
            hidden: I32,
            eps: F32,
            stream: Stream,
        ]),
    // `beta` is a LITERAL zero and not a context field: this symbol
    // OVERWRITES its destination. The accumulating spelling is
    // `gemm::act_x_w`, whose arm reads `spec.beta_one` to choose between
    // two arities, and a row that pretended one symbol did both would be
    // stating the other one's contract.
    kernel!(gemm_xwt "gemm::act_x_wt_bf16",
        lowered_as = Some("gemm::act_x_w"),
        operands = operands![
            handle: CublasHandle <- Source::Ctx("cublas"),
            act: Buf <- Source::In(0),
            // A DENSE WEIGHT IS THE STATEMENT'S NAME, not a slot in the
            // run — the lowering does not produce it, it names it. The
            // slot spelling stays first because a trace that DID stage
            // one means it.
            w: Buf <- Source::Or(&Source::Weight(0), &Source::WeightNamed),
            y: BufMut <- Source::Out(0),
            m: I32 <- Source::Rows,
            n: I32 <- Source::OutWidth(0),
            k: I32 <- Source::InWidth(0),
            beta: F32 <- Source::Beta,
        ]),
    // ── the WEIGHT REPRESENTATION axis ─────────────────────────────
    //
    // One row per way a weight can be stored, because the statement
    // NAMES which — `MatW::gemm_symbol`. The driver used to pick between
    // these by building a `WeightView` from a per-layer descriptor the
    // statement never mentioned, and `gemm::act_x_w` routed on it; a
    // kernel chosen by the driver is the shape every defect in this
    // arc's ledger had.
    //
    // Each takes the scales (and zero-points, where the checkpoint
    // carries them) as WEIGHTS — `MatW::scale_names` derives their names
    // off the weight's own, which is how the loader already finds them.
    // A dense statement names one tensor; a quantized one names two or
    // three, and says so.
    //
    // THREE ways, not four. `gemm::act_x_wt_tensor_scaled` was the
    // fourth and is deleted: `ScaleLayout::PerTensor` had no constructor
    // outside `dsl.rs`, and `model-loader`'s `QuantGranularity` — which
    // every shipped scale and every loader-encoded scale states — spells
    // only `PerChannel` and `PerGroup`, so no checkpoint format in this
    // tree could publish the fact the fourth arm selected on. The C++
    // entry point stays in `gemm/gemm.hpp`; re-stating the row is an
    // eight-line edit the day the loader grows the granularity.
    kernel!(gemm_xwt_channel_scaled "gemm::act_x_wt_channel_scaled",
        operands = operands![
            handle: CublasHandle,
            act: Buf,
            w: Buf,
            w_dtype: Dtype,
            w_nbytes: Usize,
            scale: Buf,
            scale_dtype: Dtype,
            scale_numel: Usize,
            zero_point: Buf,
            channel_axis: I32,
            y: BufMut,
            m: I32,
            n: I32,
            k: I32,
            beta: F32,
        ]),
    kernel!(gemm_xwt_grouped_scaled "gemm::act_x_wt_grouped_scaled",
        operands = operands![
            handle: CublasHandle,
            act: Buf,
            w: Buf,
            w_dtype: Dtype,
            w_nbytes: Usize,
            scale: Buf,
            scale_dtype: Dtype,
            scale_numel: Usize,
            zero_point: Buf,
            group_size: I32,
            y: BufMut,
            m: I32,
            n: I32,
            k: I32,
            beta: F32,
        ]),
    // MXFP4 with E8M0 block scales — gpt-oss's expert banks. Its scales
    // are not a layout question, so it is its own row rather than a
    // `Scaled` variant.
    kernel!(gemm_xwt_mxfp4_marlin "gemm::act_x_wt_mxfp4_marlin",
        operands = operands![
            handle: CublasHandle,
            act: Buf,
            w: Buf,
            w_nbytes: Usize,
            scale: Buf,
            scale_numel: Usize,
            y: BufMut,
            m: I32,
            n: I32,
            k: I32,
            beta: F32,
        ]),
    // `gemm::batched_act_x_wt_bf16` WAS HERE — the batched twin, one GEMM
    // per pointer-array entry. Deleted by `new-horizon.md` §38: its whole
    // consumer set was `dsl::cuda::gemm_batched_xwt`, which no model text,
    // golden, lowering, driver fire or test ever called. The LAUNCHER stays
    // (§10.10) and so does everything it forwards into:
    // `gemm.hpp:401`'s `inline batched_act_x_wt_bf16` is a one-line
    // forwarder to `batched_act_x_w` (`gemm.hpp:303`, `gemm.cpp:2397`),
    // which is separately declared, separately callable, and the only
    // entry to `gemm_batched_bf16_impl`'s `cublasGemmGroupedBatchedEx`.
    // Deleting the row removed the shim entry, not the arithmetic.
    kernel!(gemm_out_fp32 "gemm::act_x_wt_bf16_out_fp32",
        operands = operands![
            // NO `handle: CublasHandle` — §45. `execution::RUST_SERVED`
            // names this row, so its body is `driver-cuda`'s
            // `bind::service::gemm_act_x_wt_bf16_out_fp32` and the handle
            // comes off the dispatch context the arm already holds. A
            // `Ty::CublasHandle` here was one backend's library type in a
            // vocabulary two backends share, bound from `Source::Ctx` and
            // therefore carrying nothing the statement said.
            act: Buf <- Source::In(0),
            w: Buf <- Source::Weight(0),
            y: F32sMut <- Source::Out(0),
            m: I32 <- Source::Rows,
            n: I32 <- Source::OutWidth(0),
            k: I32 <- Source::InWidth(0),
        ]),
    // The group boundaries (`M_array`) are fire-global, so a row window would
    // cut a group in half.
    kernel!(gemm_grouped "gemm::grouped_act_x_wt_bf16", whole = true,
        operands = operands![
            // The handle went the same way, and this row is the one where it
            // shows least: every operand is `Source::Unbound`, so no arm is
            // emitted at all and the only consumer is `fire::lora`'s
            // hand-written staged apply — which calls
            // `bind::service::gemm_grouped_act_x_wt_bf16` directly and holds
            // a handle of its own.
            act_ptrs_host: BufArray,
            w_ptrs_host: BufArray,
            y_ptrs_host: BufArrayMut,
            m_array_host: I32s,
            group_count: I32,
            n: I32,
            k: I32,
            beta: F32,
        ]),
    // The fused Q/K/V triple had a row here — `gemm::gemv3_bf16` — and it
    // is deleted. `cuda::gemv3` had no caller in any model text, and the
    // fusion this row names is three `matmul` statements a text already
    // knows how to write (§27). `gemv.cu` is now deleted too, and no row
    // moved to this table with it: its two `__global__` templates are
    // `csrc/src/gemm/gemv.cuh` and their four rows are JIT rows in
    // `families::gemm`, not AOT rows here. `gemv_bf16` — the launcher
    // `gemm.cpp` calls from the live `act_x_wt_bf16` path — is Rust now
    // (`driver-cuda/src/fire/gemv.rs`), which is why it never needed one:
    // a row cannot state a `dim3(32, kWarps)` block and cannot DECLINE,
    // and `gemv_bf16` does both.
    // The sink rescale, and the fp32 LSE it eats. The LSE has no row of
    // its own: it is a second OUTPUT of the decode dispatch, requested
    // by an argument, so the kernel that changes is none.
    // A projection with its bias in the EPILOGUE — one launch where a
    // matmul plus an AddBias is two, and a different accumulation order.
    // TWO weights, and the order is the statement's: the projection
    // first, then the bias it lands with. A row states that once; the arm
    // it replaces read `args[2]` and `args[3]` and said so in a comment.
    kernel!(gemm_bias "gemm::act_x_wt_bias_bf16",
        operands = operands![
            handle: CublasHandle <- Source::Ctx("cublas"),
            act: Buf <- Source::In(0),
            w: Buf <- Source::Weight(0),
            bias: Buf <- Source::Weight(1),
            y: BufMut <- Source::Out(0),
            m: I32 <- Source::Rows,
            n: I32 <- Source::OutWidth(0),
            k: I32 <- Source::InWidth(0),
            stream: Stream <- Source::Ctx("stream"),
            beta: F32 <- Source::Lit(Lit::F32(0.0)),
        ]),
];
