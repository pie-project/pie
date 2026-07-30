#pragma once

/// The llama_like declared forward, as THIS backend's dispatch descriptors.
///
/// This is emitter #2 over `pie_forward`'s traced form: it walks the plan the
/// exact way `driver/cuda/src/model/llama_like/declared_forward.cpp` walks it
/// — same weight-name vocabulary (`forward/src/family.rs`'s), same
/// throw-on-drift discipline — but where the CUDA emitter launches kernels,
/// this one only EMITS an ordered list of dispatch descriptors. What a
/// descriptor commits to is the operation and its dataflow; dtype, PSO and
/// launch geometry are the Mac increment's, deliberately absent.
///
/// The kinds stay in this namespace rather than joining the shared
/// `pie::metal::Kernel`, per gemma4's precedent (`gemma4/decode_step.hpp`):
/// that enum is the M=1 argument-table ABI, and a family that cannot yet be
/// bound has no business appending to it.
///
/// v0 consumes the UNFUSED trace only (three projection matmuls, no
/// `SplitQkv`), with qk-norm both on and off. Fusion is the emitter's
/// decision, never the trace's (see the CUDA emitter's peephole comments), so
/// a fused-binding trace is refused loudly rather than half-understood.

#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "pie_forward/plan.hpp"

namespace pie::metal::llama_like {

enum class Kind : std::uint8_t {
    Embed,
    RmsNorm,         // attn_norm / mlp_norm — the pre-block norms
    ProjQ,
    ProjK,
    ProjV,
    RmsNormPerHead,  // q_norm / k_norm (absent when the facts had no qk_norm)
    Rope,            // one dispatch for q and k together, as the plan states it
    KvAppend,
    Attention,
    ProjO,           // carries fuse_residual (the trace's beta_one)
    MlpGateUp,
    Swiglu,
    MlpDown,         // carries fuse_residual (the trace's beta_one)
    FinalNorm,
    LmHead,
};

inline const char* kind_name(Kind k) {
    switch (k) {
        case Kind::Embed: return "Embed";
        case Kind::RmsNorm: return "RmsNorm";
        case Kind::ProjQ: return "ProjQ";
        case Kind::ProjK: return "ProjK";
        case Kind::ProjV: return "ProjV";
        case Kind::RmsNormPerHead: return "RmsNormPerHead";
        case Kind::Rope: return "Rope";
        case Kind::KvAppend: return "KvAppend";
        case Kind::Attention: return "Attention";
        case Kind::ProjO: return "ProjO";
        case Kind::MlpGateUp: return "MlpGateUp";
        case Kind::Swiglu: return "Swiglu";
        case Kind::MlpDown: return "MlpDown";
        case Kind::FinalNorm: return "FinalNorm";
        case Kind::LmHead: return "LmHead";
    }
    return "?";
}

/// The geometry the descriptors are emitted against — the same facts the
/// trace was built from, restated so the walk can cross-check the plan
/// against what this driver believes it is scheduling.
struct LlamaLikeGeometry {
    int hidden = 0;
    int layers = 0;
    int q_heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int intermediate = 0;
    int vocab = 0;
};

/// One emitted dispatch descriptor.
///
/// `fuse_residual` mirrors qwen3.5's `Dispatch::fuse_residual`: the residual
/// add rides in the projection's epilogue instead of being its own dispatch,
/// which is how the trace's `beta_one` matmuls (o_proj, down) land here.
/// `reads`/`writes` carry the plan's SSA activation value ids so the shared
/// live-range colourer (`batch/scratch_color.hpp`) can be run over the DAG
/// without re-deriving dataflow.
struct Dispatch {
    Kind kind;
    int layer = -1;    // -1 for prologue/epilogue, like the plan's ops
    int ordinal = 0;   // dense 0..N-1 — the future argument-table key
    bool fuse_residual = false;
    std::uint32_t plan_op = 0;  // index of the plan op this descriptor emits

    std::uint32_t reads[3] = {0, 0, 0};
    int n_reads = 0;
    std::uint32_t writes[3] = {0, 0, 0};
    int n_writes = 0;
};

namespace detail {

[[noreturn]] inline void throw_unknown_weight(std::string_view name) {
    throw std::runtime_error(
        "llama_like declared dag: unknown weight name '" + std::string(name) +
        "' (trace vocabulary is forward/src/family.rs's)");
}

/// A plan weight name split into its layer index and field: "layer.3.q_proj"
/// → {3, "q_proj"}; prologue/epilogue names keep layer -1. Same parse as the
/// CUDA emitter's `parse_weight_name`.
struct ParsedWeightName {
    int layer = -1;
    std::string_view field;
};

inline ParsedWeightName parse_weight_name(std::string_view name) {
    constexpr std::string_view prefix = "layer.";
    if (name.substr(0, prefix.size()) != prefix) {
        return ParsedWeightName{-1, name};
    }
    const std::size_t dot = name.find('.', prefix.size());
    if (dot == std::string_view::npos) throw_unknown_weight(name);
    int layer = -1;
    const char* first = name.data() + prefix.size();
    const char* last = name.data() + dot;
    const auto [ptr, ec] = std::from_chars(first, last, layer);
    if (ec != std::errc() || ptr != last || layer < 0) {
        throw_unknown_weight(name);
    }
    return ParsedWeightName{layer, name.substr(dot + 1)};
}

}  // namespace detail

/// Emit the ordered dispatch DAG for `plan`.
///
/// One descriptor per plan op — v0 declares the trace verbatim and leaves
/// every fusion beyond the trace's own `beta_one` accumulates to the Mac
/// increment. Throws on anything the v0 vocabulary does not cover: a fused
/// QKV binding (`SplitQkv` / `Matmul(qkv)`), a non-Plain norm variant, a
/// non-standard rope, or any name/layer that drifted from `family.rs`.
inline std::vector<Dispatch> build_llama_like_declared_dag(
    const pie_forward::ForwardPlan& plan, const LlamaLikeGeometry& g)
{
    using pie_forward::PieForwardNormVariant;
    using pie_forward::PieForwardOp;
    using pie_forward::PieForwardOpKind;
    using pie_forward::PieForwardRopeKind;

    std::vector<Dispatch> dag;
    dag.reserve(plan.op_count());

    const auto check_layer = [&](int layer, std::size_t at) {
        if (layer < -1 || layer >= g.layers) {
            throw std::runtime_error(
                "llama_like declared dag: op " + std::to_string(at) +
                " names layer " + std::to_string(layer) +
                " outside the geometry's 0.." + std::to_string(g.layers - 1));
        }
    };

    for (std::size_t i = 0; i < plan.op_count(); ++i) {
        const PieForwardOp& op = plan.op(i);

        Dispatch d{};
        d.layer = static_cast<int>(op.layer);
        d.ordinal = static_cast<int>(dag.size());
        d.plan_op = static_cast<std::uint32_t>(i);
        check_layer(d.layer, i);

        const auto ids_in = plan.inputs(op);
        const auto ids_out = plan.outputs(op);
        if (ids_in.size > 3 || ids_out.size > 3) {
            throw std::runtime_error(
                "llama_like declared dag: op " + std::to_string(i) +
                " has more operands than any v0 kind carries");
        }
        for (std::size_t j = 0; j < ids_in.size; ++j) d.reads[d.n_reads++] = ids_in[j];
        for (std::size_t j = 0; j < ids_out.size; ++j) d.writes[d.n_writes++] = ids_out[j];

        switch (op.kind) {
        case PieForwardOpKind::Embed: {
            if (plan.weight_name(op) != "embed") {
                detail::throw_unknown_weight(plan.weight_name(op));
            }
            d.kind = Kind::Embed;
            break;
        }
        case PieForwardOpKind::Rmsnorm: {
            if (op.param0 !=
                static_cast<std::uint32_t>(PieForwardNormVariant::Plain)) {
                throw std::runtime_error(
                    "llama_like declared dag: only the Plain rmsnorm variant "
                    "is emitted (Gemma folding is a different arithmetic)");
            }
            const std::string_view name = plan.weight_name(op);
            const detail::ParsedWeightName nm = detail::parse_weight_name(name);
            if (nm.field == "attn_norm" || nm.field == "mlp_norm") {
                if (nm.layer != d.layer) detail::throw_unknown_weight(name);
                d.kind = Kind::RmsNorm;
            } else if (nm.layer < 0 && nm.field == "final_norm") {
                d.kind = Kind::FinalNorm;
            } else {
                detail::throw_unknown_weight(name);
            }
            break;
        }
        case PieForwardOpKind::Matmul: {
            const std::string_view name = plan.weight_name(op);
            const detail::ParsedWeightName nm = detail::parse_weight_name(name);
            if (nm.layer != d.layer) detail::throw_unknown_weight(name);
            const bool beta_one = op.param0 != 0;
            if (nm.field == "qkv") {
                // The unfused trace is the v0 contract; see the header note.
                throw std::runtime_error(
                    "llama_like declared dag: fused QKV trace (Matmul(qkv)) "
                    "is not emitted in v0 — trace with fused_qkv=0");
            } else if (nm.field == "q_proj") {
                d.kind = Kind::ProjQ;
            } else if (nm.field == "k_proj") {
                d.kind = Kind::ProjK;
            } else if (nm.field == "v_proj") {
                d.kind = Kind::ProjV;
            } else if (nm.field == "o_proj") {
                d.kind = Kind::ProjO;
                d.fuse_residual = beta_one;
            } else if (nm.field == "gate_up") {
                d.kind = Kind::MlpGateUp;
            } else if (nm.field == "down") {
                d.kind = Kind::MlpDown;
                d.fuse_residual = beta_one;
            } else {
                detail::throw_unknown_weight(name);
            }
            if (beta_one && !d.fuse_residual) {
                throw std::runtime_error(
                    "llama_like declared dag: beta_one on '" +
                    std::string(name) +
                    "' — the trace's residual accumulates moved");
            }
            break;
        }
        case PieForwardOpKind::SplitQkv: {
            throw std::runtime_error(
                "llama_like declared dag: SplitQkv found — v0 consumes the "
                "UNFUSED trace only (trace with fused_qkv=0)");
        }
        case PieForwardOpKind::RmsnormPerHead: {
            const std::string_view name = plan.weight_name(op);
            const detail::ParsedWeightName nm = detail::parse_weight_name(name);
            if ((nm.field != "q_norm" && nm.field != "k_norm") ||
                nm.layer != d.layer) {
                detail::throw_unknown_weight(name);
            }
            if (static_cast<int>(op.param0) != g.head_dim) {
                throw std::runtime_error(
                    "llama_like declared dag: per-head norm head_dim " +
                    std::to_string(op.param0) + " != geometry's " +
                    std::to_string(g.head_dim));
            }
            d.kind = Kind::RmsNormPerHead;
            break;
        }
        case PieForwardOpKind::Rope: {
            if (op.param0 !=
                static_cast<std::uint32_t>(PieForwardRopeKind::Standard)) {
                throw std::runtime_error(
                    "llama_like declared dag: only standard rope is emitted");
            }
            d.kind = Kind::Rope;
            break;
        }
        case PieForwardOpKind::KvAppend:
        case PieForwardOpKind::Attention: {
            if (static_cast<int>(op.param0) != d.layer) {
                throw std::runtime_error(
                    "llama_like declared dag: op " + std::to_string(i) +
                    " addresses cache layer " + std::to_string(op.param0) +
                    " inside layer " + std::to_string(d.layer) +
                    "'s bracket — the trace's shape drifted");
            }
            d.kind = op.kind == PieForwardOpKind::KvAppend ? Kind::KvAppend
                                                           : Kind::Attention;
            break;
        }
        case PieForwardOpKind::Swiglu: {
            if (static_cast<int>(op.param0) != g.intermediate) {
                throw std::runtime_error(
                    "llama_like declared dag: swiglu inter " +
                    std::to_string(op.param0) + " != geometry's " +
                    std::to_string(g.intermediate));
            }
            d.kind = Kind::Swiglu;
            break;
        }
        case PieForwardOpKind::LmHead: {
            // Tied embeddings trace the lm head as "embed"; either name is
            // the binding's business, not this walk's.
            const std::string_view name = plan.weight_name(op);
            if (name != "embed" && name != "lm_head") {
                detail::throw_unknown_weight(name);
            }
            d.kind = Kind::LmHead;
            break;
        }
        default:
            throw std::runtime_error(
                "llama_like declared dag: op kind " +
                std::to_string(static_cast<std::uint32_t>(op.kind)) +
                " has no emission rule");
        }

        dag.push_back(d);
    }
    return dag;
}

}  // namespace pie::metal::llama_like
