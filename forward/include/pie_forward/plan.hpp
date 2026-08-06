#pragma once

// C++ ownership layer over the forward crate's generated C ABI, modeled on
// `loader/include/pie_loader/plan.hpp` but much smaller: tracing has no
// checkpoint handle, no diagnostics list and no verification pass, so what
// remains is exactly the part C++ has to supply — RAII over storage that
// belongs to Rust, and typed accessors over the POD the driver walks.
//
// `pie_forward.h` is generated from the Rust definitions by
// `cargo run -p pie-forward-cbindgen`, so nothing here restates a struct or
// an enum; a hand-written copy is the drift the loader's header replaced
// (`loader/architecture.md` §9).
//
// One ownership difference from the loader: the plan *header* is caller
// storage (`pie_forward_trace_llama_like` takes `PieForwardPlan*`, not
// `PieForwardPlan**`), so this class holds the POD by value and Rust owns
// only the arena behind `owner`. `pie_forward_release` frees that arena and
// resets the header to empty, which is why the destructor can call it
// unconditionally.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "pie_forward.h"

namespace pie_forward {

/// An owned traced form.
///
/// Move-only: the arena behind `plan_.owner` must be released exactly once,
/// which matches how `pie_loader::LoadPlan` treats its plan pointer.
class ForwardPlan {
  public:
    /// A contiguous run of value ids (an op's inputs or outputs).
    struct IdSpan {
        const std::uint32_t* data = nullptr;
        std::size_t size = 0;
        const std::uint32_t* begin() const noexcept { return data; }
        const std::uint32_t* end() const noexcept { return data + size; }
        std::uint32_t operator[](std::size_t i) const { return data[i]; }
    };

    ForwardPlan() = default;
    ForwardPlan(const ForwardPlan&) = delete;
    ForwardPlan& operator=(const ForwardPlan&) = delete;
    ForwardPlan(ForwardPlan&& other) noexcept
        : plan_(std::exchange(other.plan_, PieForwardPlan{})) {}
    ForwardPlan& operator=(ForwardPlan&& other) noexcept {
        if (this != &other) {
            reset();
            plan_ = std::exchange(other.plan_, PieForwardPlan{});
        }
        return *this;
    }
    ~ForwardPlan() { reset(); }

    void reset() noexcept {
        // Release empties the header (owner becomes null), so this is safe
        // on a default-constructed or moved-from instance.
        pie_forward_release(&plan_);
    }

    /// Trace the llama_like family. Mirroring
    /// `pie_loader::LoadPlan::compile`: the facts are the whole request, and
    /// none of them is a model's name.
    static ForwardPlan trace_llama_like(const PieForwardLlamaLikeFacts& facts) {
        PieForwardPlan raw{};
        const PieForwardStatus status = pie_forward_trace_llama_like(&facts, &raw);
        if (status != PieForwardStatus::Ok) {
            throw std::runtime_error(
                "forward plan: trace failed (" + status_name(status) + ")");
        }
        return ForwardPlan(raw);
    }

    /// Trace the LOWERED llama_like — the same text, with the CUDA backend
    /// facts and a fire class in hand, so the class arms run and the
    /// traced form STATES its kernels as `Launch` ops (north-star-dsl.md).
    /// Call once per class the deployment fires; the semantic trace above
    /// remains the parity reference.
    static ForwardPlan trace_llama_like_cuda(
        const PieForwardLlamaLikeFacts& facts,
        const PieForwardLlamaLikeCudaFacts& cuda,
        PieForwardFireClass fire_class) {
        PieForwardPlan raw{};
        const PieForwardStatus status = pie_forward_trace_llama_like_cuda(
            &facts, &cuda, static_cast<std::uint32_t>(fire_class), &raw);
        if (status != PieForwardStatus::Ok) {
            throw std::runtime_error(
                "forward plan: lowered trace failed (" + status_name(status) + ")");
        }
        return ForwardPlan(raw);
    }

    /// Trace the LOWERED qwen3_5 hybrid — the same text as
    /// `trace_qwen3_5_hybrid`, with the CUDA backend facts and a fire
    /// class in hand (north-star-dsl.md rung 4c). Classes 2/3 (the MTP
    /// service classes) are refused until 4c-iv.
    static ForwardPlan trace_qwen3_5_hybrid_cuda(
        const PieForwardQwen35HybridFacts& facts,
        const PieForwardQwen35CudaFacts& cuda,
        PieForwardFireClass fire_class) {
        PieForwardPlan raw{};
        const PieForwardStatus status = pie_forward_trace_qwen3_5_hybrid_cuda(
            &facts, &cuda, static_cast<std::uint32_t>(fire_class), &raw);
        if (status != PieForwardStatus::Ok) {
            throw std::runtime_error(
                "forward plan: lowered qwen3_5 trace failed (" +
                status_name(status) + ")");
        }
        return ForwardPlan(raw);
    }

    /// Trace one gemma-4 CLASS. Only Decode/Prefill exist: gemma-4 has
    /// no recurrent state, so it has none of qwen3_5's service classes.
    static ForwardPlan trace_gemma4_cuda(
        const PieForwardGemma4Facts& facts,
        const PieForwardGemma4CudaFacts& cuda,
        PieForwardFireClass fire_class) {
        PieForwardPlan raw{};
        const PieForwardStatus status = pie_forward_trace_gemma4_cuda(
            &facts, &cuda, static_cast<std::uint32_t>(fire_class), &raw);
        if (status != PieForwardStatus::Ok) {
            throw std::runtime_error(
                "forward plan: gemma4 trace failed (" +
                status_name(status) + ")");
        }
        return ForwardPlan(raw);
    }

    /// Trace the qwen3_5_moe MoE MLP-block FRAGMENT — the first traced form
    /// carrying `dyn` ops (`TopK`, selector-carrying `Matmul`s,
    /// `WeightedSum`, `SigmoidGateAdd`).
    ///
    /// This wrapper exposes the plan; the declared EXECUTORS do not consume
    /// it. Feeding a dyn trace to either emitter throws rather than
    /// half-emitting: their op-kind switches end in a loud default arm
    /// ("op kind N has no emission rule" — the CUDA executor's
    /// `declared_forward.cpp`, the Metal DAG builder's `declared_dag.hpp`,
    /// which also refuses the MoE weight names before ever reaching a dyn
    /// kind; `driver/metal/tests/llama_like_declared_dag_test.cpp` pins the
    /// refusal). Emitting the grouped-GEMM lowering is a later, much larger
    /// lift.
    static ForwardPlan trace_qwen3_5_moe_mlp(const PieForwardQwen35MoeMlpFacts& facts) {
        PieForwardPlan raw{};
        const PieForwardStatus status = pie_forward_trace_qwen3_5_moe_mlp(&facts, &raw);
        if (status != PieForwardStatus::Ok) {
            throw std::runtime_error(
                "forward plan: moe mlp trace failed (" + status_name(status) + ")");
        }
        return ForwardPlan(raw);
    }

    /// Trace the qwen3_5 GDN linear-attention block FRAGMENT — the traced
    /// form carrying the GDN kinds (`SplitGdn`, `CausalConv1d`, `GdnPrep`,
    /// `GatedDelta`, `RmsnormGated`) and the first ops addressing
    /// PER-REQUEST state (the conv/recurrent slabs, implicit behind the
    /// state ops' layer param exactly as the KV cache is behind
    /// `KvAppend`'s).
    ///
    /// Like the MoE fragment above, this wrapper exposes the plan; the
    /// declared EXECUTORS do not consume it — both emitters' op-kind
    /// switches end in the loud default arm, and the Metal DAG builder
    /// refuses the GDN weight names before ever reaching a GDN kind
    /// (`driver/metal/tests/llama_like_declared_dag_test.cpp` pins both
    /// refusals). Emitting the GDN core is the driver-side rung, not this
    /// one.
    static ForwardPlan trace_qwen3_5_gdn(const PieForwardQwen35GdnFacts& facts) {
        PieForwardPlan raw{};
        const PieForwardStatus status = pie_forward_trace_qwen3_5_gdn(&facts, &raw);
        if (status != PieForwardStatus::Ok) {
            throw std::runtime_error(
                "forward plan: gdn trace failed (" + status_name(status) + ")");
        }
        return ForwardPlan(raw);
    }

    /// Trace the qwen3_5 FULL-attention block FRAGMENT — the traced form
    /// carrying the gated-attention kinds (`SplitQGate`, `SigmoidGateMul`)
    /// plus the partial `Rope` (param1 = rotary width) and the Gemma-fold
    /// `RmsnormPerHead` (param1 = variant), with `KvAppend`/`Attention`
    /// marking the layer's KV cache exactly as llama_like's do.
    ///
    /// Like the fragments above, this wrapper exposes the plan; the
    /// declared EXECUTORS do not consume it — both emitters' op-kind
    /// switches end in the loud default arm on the appended kinds
    /// (`driver/metal/tests/llama_like_declared_dag_test.cpp` pins the
    /// refusal). Emitting the gated attention is a driver-side rung.
    static ForwardPlan trace_qwen3_5_full_attn(const PieForwardQwen35FullAttnFacts& facts) {
        PieForwardPlan raw{};
        const PieForwardStatus status = pie_forward_trace_qwen3_5_full_attn(&facts, &raw);
        if (status != PieForwardStatus::Ok) {
            throw std::runtime_error(
                "forward plan: full attn trace failed (" + status_name(status) + ")");
        }
        return ForwardPlan(raw);
    }

    /// Trace the full qwen3_5 HYBRID model — embed → per-layer {GDN or
    /// full attention on the checkpoint's schedule; dense or MoE MLP} →
    /// final norm → lm_head. Composes every fragment vocabulary, so the
    /// declared executors refuse it loudly; the wrapper serves the
    /// toolchain side (planning, tests, cross-language pinning).
    static ForwardPlan trace_qwen3_5_hybrid(const PieForwardQwen35HybridFacts& facts) {
        PieForwardPlan raw{};
        const PieForwardStatus status = pie_forward_trace_qwen3_5_hybrid(&facts, &raw);
        if (status != PieForwardStatus::Ok) {
            throw std::runtime_error(
                "forward plan: hybrid trace failed (" + status_name(status) + ")");
        }
        return ForwardPlan(raw);
    }

    explicit operator bool() const noexcept { return plan_.owner != nullptr; }

    const PieForwardPlan& view() const {
        if (plan_.owner == nullptr) throw std::runtime_error("forward plan: empty");
        return plan_;
    }

    /// The flat launch list this plan lowers to for one fire's rows —
    /// the SHADOW comparison (`.wiki/tart/dsl.md` migration step 6).
    ///
    /// EXECUTES NOTHING. The result describes what would run, so that a
    /// caller can compare it against what its walk actually launched.
    ///
    /// The view points into storage this plan owns and is valid until
    /// the NEXT `lower()` on this plan (one slot). `const` because the
    /// only mutation is that slot — a cache, in the sense `mutable`
    /// exists for — and because a shadow must not need a mutable handle
    /// to a plan the body reads.
    PieForwardLowered lower(
        const PieForwardRow* rows,
        std::size_t rows_len,
        bool captures_across_splits = false) const {
        PieForwardLowered out{};
        const PieForwardStatus status = pie_forward_lower(
            const_cast<PieForwardPlan*>(&plan_), rows, rows_len,
            captures_across_splits ? 1 : 0, &out);
        if (status != PieForwardStatus::Ok) {
            throw std::runtime_error(
                "forward plan: lower failed (" + status_name(status) + ")");
        }
        return out;
    }

    /// The launcher symbol one rectangle names, as a view into the
    /// lowering's own name table (NOT the plan's — those are weights).
    static std::string_view kernel_name(
        const PieForwardLowered& lowered, const PieForwardLaunch& launch) {
        if (launch.kernel_name >= lowered.kernel_names_len) return {};
        const PieForwardName& n = lowered.kernel_names[launch.kernel_name];
        return std::string_view(
            reinterpret_cast<const char*>(lowered.kernel_name_bytes.ptr) + n.offset,
            n.len);
    }

    std::size_t op_count() const { return view().ops.len; }
    const PieForwardOp& op(std::size_t i) const {
        const PieForwardPlan& plan = view();
        if (i >= plan.ops.len) throw std::runtime_error("forward plan: op out of range");
        return plan.ops.ptr[i];
    }

    std::size_t value_count() const { return view().values.len; }
    const PieForwardValue& value(std::uint32_t id) const {
        const PieForwardPlan& plan = view();
        if (id >= plan.values.len) {
            throw std::runtime_error("forward plan: value id out of range");
        }
        return plan.values.ptr[id];
    }

    IdSpan inputs(const PieForwardOp& op) const { return ids(op.inputs); }
    IdSpan outputs(const PieForwardOp& op) const { return ids(op.outputs); }
    /// `Launch` only: the weight names the stated kernel consumes, as NAME
    /// indices (resolve each with [`name`]), in signature order.
    IdSpan aux_names(const PieForwardOp& op) const { return ids(op.aux_names); }

    /// A name-table entry as a view into the plan's blob; valid for the
    /// plan's lifetime (the strings are not NUL-terminated — see
    /// `PieForwardBytes` in the generated header).
    std::string_view name(std::uint32_t index) const {
        const PieForwardPlan& plan = view();
        if (index >= plan.names.len) {
            throw std::runtime_error("forward plan: name index out of range");
        }
        const PieForwardName entry = plan.names.ptr[index];
        return std::string_view(
            reinterpret_cast<const char*>(plan.name_bytes.ptr) + entry.offset,
            entry.len);
    }

    /// The op's weight name, or empty for the kinds that reference none.
    std::string_view weight_name(const PieForwardOp& op) const {
        if (op.weight_name == PIE_FORWARD_NO_NAME) return {};
        return name(op.weight_name);
    }

    std::string_view family() const { return name(view().family); }
    std::uint64_t compiler_version() const { return view().compiler_version; }

    static std::string status_name(PieForwardStatus status) {
        switch (status) {
        case PieForwardStatus::Ok: return "ok";
        case PieForwardStatus::InvalidArgument: return "invalid argument";
        }
        return "unknown status";
    }

  private:
    explicit ForwardPlan(const PieForwardPlan& plan) noexcept : plan_(plan) {}

    IdSpan ids(PieForwardIdRange range) const {
        const PieForwardPlan& plan = view();
        if (static_cast<std::size_t>(range.offset) + range.len > plan.value_ids.len) {
            throw std::runtime_error("forward plan: id range out of range");
        }
        return IdSpan{plan.value_ids.ptr + range.offset, range.len};
    }

    PieForwardPlan plan_ = PieForwardPlan{};
};

}  // namespace pie_forward
