#include "sampling_ir/runtime.hpp"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "cuda_check.hpp"
#include "executor/persistent_inputs.hpp"

namespace pie_cuda_driver::sampling_ir {

namespace {

// Find a late-bound device-resident input by key. Returns nullptr when absent.
const LateInput* find_late_input(
    std::span<const LateInput> late_inputs, std::uint32_t key) {
    for (const auto& e : late_inputs) {
        if (e.key == key) return &e;
    }
    return nullptr;
}

// Round up to an 8-byte boundary so each staged input is aligned for f32/u32/
// u64/bf16 element reads by the kernel.
constexpr std::size_t align8(std::size_t n) { return (n + 7u) & ~static_cast<std::size_t>(7u); }

}  // namespace

void SamplingIrRuntime::stage_submit_inputs(const FireContext& ctx) {
    staged_.clear();
    if (ctx.submit_inputs.empty() && ctx.late_value_inputs.empty()) return;

    // Submit-bound + host-late values stage identically (host bytes → device);
    // the binding policy (fail-loud vs skip) is driven by the program's declared
    // class, not by how the value was staged. Keys are disjoint across the two.
    auto each = [&](auto&& fn) {
        for (const SubmitInput& in : ctx.submit_inputs) fn(in);
        for (const SubmitInput& in : ctx.late_value_inputs) fn(in);
    };

    // Pack all inputs contiguously (8-byte aligned) and size the blob. A
    // zero-length entry carries no value (e.g. a host-late key with no staged
    // bytes — `sampling_late_lens == 0`): skip it so it never shadows the
    // HostLate device-alias / skip-on-miss path.
    std::size_t total = 0;
    each([&](const SubmitInput& in) {
        if (in.len_bytes == 0 || in.data == nullptr) return;
        staged_.emplace(in.key, StagedSpan{total, in.len_bytes});
        total += align8(in.len_bytes);
    });
    if (total > host_stage_.size()) {
        // Grow with headroom to avoid reallocating every fire as programs vary.
        host_stage_ = DeviceBuffer<std::uint8_t>::alloc(align8(total * 2));
    }
    // Upload each input's bytes into its slot. Async on the fire's stream so the
    // copies are ordered before the kernel DAG that consumes them.
    std::uint8_t* base = host_stage_.data();
    each([&](const SubmitInput& in) {
        if (in.len_bytes == 0 || in.data == nullptr) return;
        const StagedSpan& s = staged_.at(in.key);
        CUDA_CHECK(cudaMemcpyAsync(base + s.offset, in.data, in.len_bytes,
                                   cudaMemcpyHostToDevice, ctx.stream));
    });
}

RunStatus SamplingIrRuntime::try_run(const FireContext& ctx) {
    // ── Mode select ─────────────────────────────────────────────────────
    // No program on the request → the executor runs its legacy sampler.
    if (ctx.program_bytecode.empty()) {
        return RunStatus::NoProgram;
    }
    // No backend registered (W1 inert state) → fall back to legacy so the
    // executor stays correct while delta's JIT lands.
    if (backend_ == nullptr) {
        return RunStatus::NoProgram;
    }

    // ── Compile-or-fetch (delta's cache, keyed by bytecode hash) ─────────
    const ProgramHandle program = backend_->get_or_compile(ctx.program_bytecode, ctx.manifest);
    if (program == kInvalidProgram) {
        if (std::getenv("PIE_SAMPLING_IR_TRACE")) {
            std::cerr << "[ir-trace] try_run FAILED: get_or_compile returned invalid "
                         "(codegen/JIT/NVRTC) bytecode_bytes="
                      << ctx.program_bytecode.size()
                      << " reason=\"" << backend_->last_error() << "\"\n";
        }
        return RunStatus::Failed;
    }
    const ProgramInterface& iface = backend_->interface(program);
    last_iface_ = &iface;
    if (std::getenv("PIE_SAMPLING_IR_TRACE")) {
        std::cerr << "[ir-trace] program compiled: inputs=" << iface.inputs.size()
                  << " outputs=" << iface.outputs.size()
                  << " submit_inputs=" << ctx.submit_inputs.size() << "\n";
        for (const InputDecl& d : iface.inputs) {
            std::cerr << "[ir-trace]   input id=" << d.input_id
                      << " cls=" << static_cast<int>(d.cls)
                      << " host_key=" << d.host_key
                      << " elem_count=" << d.elem_count << "\n";
        }
    }

    // ── Stage submit-bound host inputs (WS1a) ───────────────────────────
    // Upload each per-fire submit-bound value into the runtime's stable device
    // blob; `staged_` then maps key → device offset for the binding loop.
    stage_submit_inputs(ctx);

    // ── Bind inputs ─────────────────────────────────────────────────────
    resolved_.clear();
    resolved_.reserve(iface.inputs.size());
    for (const InputDecl& decl : iface.inputs) {
        ResolvedInput r;
        r.input_id  = decl.input_id;
        r.cls       = decl.cls;
        r.intrinsic = decl.intrinsic;
        r.elem_count = decl.elem_count;

        switch (decl.cls) {
            case BindingClass::Const:
                // Literal inlined by codegen — nothing to bind.
                r.device_ptr = nullptr;
                r.present = true;
                break;

            case BindingClass::Intrinsic: {
                // intrinsic(logits) = the LM-head output row for this fire;
                // intrinsic(mtp-logits) = the speculator DRAFT row (the MTP head's
                // next-token logits). Both read `ctx.logits` — the bf16
                // [rows, vocab] base — because the MTP draft logits live in DRAFT
                // ROWS of `ws.logits`, not a separate buffer; only the row differs.
                const auto* base = static_cast<const __nv_bfloat16*>(ctx.logits);
                int row = ctx.sample_row;
                if (decl.intrinsic == IntrinsicKind::MtpLogits)
                    row = (ctx.mtp_draft_row >= 0) ? ctx.mtp_draft_row
                                                   : ctx.sample_row;
                r.device_ptr =
                    base + static_cast<std::size_t>(row) *
                               static_cast<std::size_t>(ctx.vocab_size);
                r.elem_count = static_cast<std::size_t>(ctx.vocab_size);
                r.present = true;
                break;
            }

            case BindingClass::HostSubmit: {
                // Submit-bound: staged into `host_stage_` this fire. A missing
                // one is a host/bridge bug, not a retry condition → fail loud
                // rather than silently skipping.
                auto sit = staged_.find(decl.host_key);
                if (sit == staged_.end()) {
                    if (std::getenv("PIE_SAMPLING_IR_TRACE")) {
                        std::cerr << "[ir-trace] try_run FAILED: HostSubmit key "
                                  << decl.host_key << " not staged. staged keys: ";
                        for (const auto& kv : staged_) std::cerr << kv.first << " ";
                        std::cerr << "\n";
                    }
                    return RunStatus::Failed;
                }
                r.device_ptr = host_stage_.data() + sit->second.offset;
                if (decl.elem_count != 0) r.elem_count = decl.elem_count;
                r.present = true;
                break;
            }

            case BindingClass::HostLate: {
                // #31 self-spec greedy-v0: a `SelfSpecDraftInput` draft binding (the
                // verify's k draft tokens) is driver-internal & device-resident —
                // forward-(N-1)'s drafts the host refed as THIS forward's verify
                // INPUT, resident in `pi->tokens` at `sample_row + 1` (the anchor sits
                // at `sample_row`; the drafts start after it — charlie executor.cpp
                // :4245-4251). The marker survives the bytecode `0x40` → reader →
                // host_avail → jit_backend `InputDecl.intrinsic` projection (charlie
                // fa5b5e68); we're in the `HostLate` case, so `cls==HostLate ∧
                // intrinsic==SelfSpecDraftInput` is the distinct flag (NOT a generic
                // HostLate fall-through → that would be the #19 wrong-buffer bind).
                // Bind flag-FIRST: there is NO host upload and NO late carrier, so the
                // staged/device-alias lookups below would `SkippedLateBindMiss`.
                // Source = `pi->tokens` (the verify input), NEVER `pi->sampled`
                // (transient; never holds the [k] drafts — delta's witness-independence
                // invariant). `pi->tokens` are u32 ids, read by the kernel as the [k]
                // i32 draft operand (non-negative ids → bit-identical).
                if (decl.intrinsic == IntrinsicKind::SelfSpecDraftInput) {
                    if (ctx.pi == nullptr) return RunStatus::Failed;
                    r.device_ptr = ctx.pi->tokens.data() +
                                   static_cast<std::size_t>(ctx.sample_row + 1);
                    r.elem_count = decl.elem_count;
                    r.present = true;
                    if (std::getenv("PIE_SAMPLING_IR_TRACE")) {
                        std::cerr << "[ir-trace]   SelfSpecDraftInput RESOLVED key="
                                  << decl.host_key << " base_row="
                                  << (ctx.sample_row + 1) << " elem_count="
                                  << r.elem_count << " (pi->tokens, NOT sampled)\n";
                    }
                    break;
                }
                // Late-bound, resolved in priority order (spec §7.4):
                //   1. staged host-late VALUE bytes (WS1b correctness path);
                //   2. a device-resident value (output-ref alias / true-async);
                //   3. else SkippedLateBindMiss — discard + retry, fail loud.
                auto sit = staged_.find(decl.host_key);
                if (sit != staged_.end()) {
                    r.device_ptr = host_stage_.data() + sit->second.offset;
                    if (decl.elem_count != 0) r.elem_count = decl.elem_count;
                    r.present = true;
                    break;
                }
                const LateInput* e = find_late_input(ctx.late_inputs, decl.host_key);
                if (e == nullptr) {
                    return RunStatus::SkippedLateBindMiss;
                }
                r.device_ptr = e->device_ptr;
                // The device-alias carrier (#27 cut #2 (B)) ships only a ptr +
                // self-arm flag, no length — take the shape from the program's
                // declared InputDecl (as the staged path does) when the carrier
                // didn't supply one.
                r.elem_count = (e->elem_count != 0) ? e->elem_count : decl.elem_count;
                r.present = true;
                // Positive device-alias-resolution probe (item-1 merged-path
                // debug): a non-null ptr logged HERE proves the Late carrier was
                // consumed BEFORE the consuming kernel — distinguishing a real
                // resolve from a SkippedLateBindMiss / pre-empted abort.
                if (std::getenv("PIE_SAMPLING_IR_TRACE")) {
                    std::cerr << "[ir-trace]   HostLate RESOLVED device-alias key="
                              << decl.host_key << " ptr=" << r.device_ptr
                              << " elem_count=" << r.elem_count << "\n";
                }
                break;
            }
        }
        resolved_.push_back(r);
    }

    // ── Resolve output destinations ─────────────────────────────────────
    // A single-token Token output writes straight into `pi.sampled` at the
    // sample row, so the existing D2H + `build_token_only_dense` path picks it
    // up unchanged. A multi-element Token output (spec-verify's `-1`-sentinel
    // `Vector<k>` accept-prefix) lands in runtime-owned `out_scratch_`; the
    // executor reads it via `last_output_ptrs()` + the declared `elem_count`.
    out_ptrs_.clear();
    out_ptrs_.reserve(iface.outputs.size());
    std::size_t scratch_ints = 0;
    for (const DeclaredOutput& o : iface.outputs) {
        const bool single_token = (o.cls == OutputClass::Token && o.elem_count <= 1);
        if (!single_token) scratch_ints += std::max<std::size_t>(o.elem_count, 1);
    }
    if (scratch_ints > out_scratch_.size()) {
        out_scratch_ = DeviceBuffer<std::int32_t>::alloc(scratch_ints * 2);
    }
    std::size_t scratch_off = 0;
    for (const DeclaredOutput& o : iface.outputs) {
        void* dst = nullptr;
        const bool single_token = (o.cls == OutputClass::Token && o.elem_count <= 1);
        if (single_token) {
            auto* sampled = static_cast<std::int32_t*>(ctx.pi->sampled.data());
            dst = sampled + ctx.sample_row;
        } else {
            // Vector Token (spec-verify) or future rich outputs → scratch.
            dst = out_scratch_.data() + scratch_off;
            scratch_off += std::max<std::size_t>(o.elem_count, 1);
        }
        out_ptrs_.push_back(dst);
    }

    // ── Launch the kernel DAG ───────────────────────────────────────────
    LaunchArgs args;
    args.inputs      = std::span<const ResolvedInput>(resolved_);
    args.output_ptrs = std::span<void* const>(out_ptrs_);
    // Batched de-hardwiring (Task #4): one grid=num_rows launch over the
    // [num_rows, vocab] block. 1 = single-row (custom-program / MVP). The
    // bindings above are base pointers (Intrinsic = block base at sample_row,
    // HostSubmit/output = [num_rows] bases); the batched kernel strides per row.
    args.num_rows    = ctx.num_rows > 0 ? ctx.num_rows : 1;
    args.vocab_size  = ctx.vocab_size;
    args.prng_offset = ctx.prng_offset;
    args.row_seeds   = ctx.row_seeds;  // ambient per-row S (Model B); null = no RNG
    backend_->launch(program, args, ctx.stream);

    return RunStatus::Handled;
}

}  // namespace pie_cuda_driver::sampling_ir
