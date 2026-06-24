// gemma4_decode_run.cpp — gemma4 integration driver: stage -> bind -> consts -> scratch ->
// PSOs -> resident -> decode loop -> (optional golden-tap dump) -> argmax.
//
// The gemma4 analog of delta's qwen3.6 decode_run. Wires the full raw-Metal M=1 gemma4
// decode step end-to-end on the real 4-bit-tied checkpoint, exercising alpha's gemma4
// encode_fn (build_gemma4_dag + load_gemma4_psos + encode_gemma4_step), the weight staging
// + const-param binder, and the no_recycle activation dataflow (one buffer per dispatch so
// every intermediate survives for the parity dump).
//
// Usage: gemma4_decode_run <checkpoint_dir> <kernels_dir> [comma_separated_prompt_ids]
//   PIE_DUMP_TAPS=<dir>  dump every golden-tapped dispatch as <layer>.<tag>.npy for
//                        charlie's cosine_bisect vs ~/parity-golden/gemma4-4bit-pos7.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gemma4_abi.hpp"
#include "gemma4_decode_step.hpp"
#include "gemma4_encode.hpp"
#include "gemma4_heap_bind.hpp"
#include "mtl4_context.hpp"
#include "safetensors_view.hpp"

using namespace pie_metal_driver::raw_metal;

namespace pie_metal_driver::raw_metal::gemma4 {
namespace {

void write_u32(const SlotHandle& s, uint32_t v) { std::memcpy(s.contents(), &v, sizeof(v)); }

inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = uint32_t(h) << 16;
    float f; std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void write_npy_f32(const std::string& path, const float* data, size_t n) {
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, " +
                      std::to_string(n) + ", ), }";
    size_t base = 10 + hdr.size() + 1;
    hdr.append((64 - (base % 64)) % 64, ' ');
    hdr.push_back('\n');
    uint16_t hlen = uint16_t(hdr.size());
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "[gemma4_decode_run] cannot open %s\n", path.c_str()); return; }
    const char magic[8] = {'\x93','N','U','M','P','Y','\x01','\x00'};
    std::fwrite(magic, 1, 8, f);
    std::fwrite(&hlen, 2, 1, f);
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    std::fwrite(data, sizeof(float), n, f);
    std::fclose(f);
}

// charlie's golden tag for a dispatch kind (<layer>.<tag>.npy), or nullptr if untapped.
const char* golden_tag(Kernel k) {
    switch (k) {
        case Kernel::EmbedGather:      return "embed";
        case Kernel::PleCombine:       return "ple_input";
        case Kernel::AttnNorm:         return "attn_norm";
        case Kernel::QmvQ:             return "q_proj";
        case Kernel::QNorm:            return "q_norm";
        case Kernel::QmvK:             return "k_proj";
        case Kernel::QmvV:             return "v_proj";
        case Kernel::KNorm:            return "k_norm";
        case Kernel::VNorm:            return "v_norm";
        case Kernel::RopeK:            return "rope_k";
        case Kernel::RopeQ:            return "rope_q";
        case Kernel::Sdpa:             return "sdpa";
        case Kernel::QmvO:             return "o_proj";
        case Kernel::PostAttnNorm:     return "post_attn_norm";
        case Kernel::AttnResidual:     return "attn_resid";
        case Kernel::FfnNorm:          return "ffn_norm";
        case Kernel::QmvGate:          return "gate_proj";
        case Kernel::QmvUp:            return "up_proj";
        case Kernel::GegluTanh:        return "geglu";
        case Kernel::QmvDown:          return "down_proj";
        case Kernel::PostFfnNorm:      return "post_ffn_norm";
        case Kernel::FfnResidual:      return "ffn_resid";
        case Kernel::PleGateGemv:      return "ple_gate";
        case Kernel::PleGeglu:         return "ple_gated";
        case Kernel::PleProjLayerGemv: return "ple_proj";
        case Kernel::PleNorm:          return "ple_norm";
        case Kernel::PleResidual:      return "ple_resid";
        case Kernel::LayerScalar:      return "layer_out";
        case Kernel::FinalRms:         return "final_norm";
        case Kernel::LmHead:           return "logits";
        case Kernel::FinalSoftcap:     return "logits_softcap";
        default:                       return nullptr;  // PleTokenGather/PleProjGemv/PleProjNorm/KvAppend/Argmax
    }
}

// Full per-kind name (every Kernel, incl. untapped) for ablation/attribution reporting.
const char* kind_name(Kernel k) {
    switch (k) {
        case Kernel::EmbedGather:      return "EmbedGather";
        case Kernel::PleTokenGather:   return "PleTokenGather";
        case Kernel::PleProjGemv:      return "PleProjGemv";
        case Kernel::PleProjNorm:      return "PleProjNorm";
        case Kernel::PleCombine:       return "PleCombine";
        case Kernel::AttnNorm:         return "AttnNorm";
        case Kernel::QmvQ:             return "QmvQ";
        case Kernel::QmvK:             return "QmvK";
        case Kernel::QmvV:             return "QmvV";
        case Kernel::QNorm:            return "QNorm";
        case Kernel::KNorm:            return "KNorm";
        case Kernel::VNorm:            return "VNorm";
        case Kernel::RopeQ:            return "RopeQ";
        case Kernel::RopeK:            return "RopeK";
        case Kernel::KvAppend:         return "KvAppend";
        case Kernel::Sdpa:             return "Sdpa";
        case Kernel::QmvO:             return "QmvO";
        case Kernel::PostAttnNorm:     return "PostAttnNorm";
        case Kernel::AttnResidual:     return "AttnResidual";
        case Kernel::FfnNorm:          return "FfnNorm";
        case Kernel::QmvGate:          return "QmvGate";
        case Kernel::QmvUp:            return "QmvUp";
        case Kernel::GegluTanh:        return "GegluTanh";
        case Kernel::QmvDown:          return "QmvDown";
        case Kernel::PostFfnNorm:      return "PostFfnNorm";
        case Kernel::FfnResidual:      return "FfnResidual";
        case Kernel::PleGateGemv:      return "PleGateGemv";
        case Kernel::PleGeglu:         return "PleGeglu";
        case Kernel::PleProjLayerGemv: return "PleProjLayerGemv";
        case Kernel::PleNorm:          return "PleNorm";
        case Kernel::PleResidual:      return "PleResidual";
        case Kernel::LayerScalar:      return "LayerScalar";
        case Kernel::FinalRms:         return "FinalRms";
        case Kernel::LmHead:           return "LmHead";
        case Kernel::FinalSoftcap:     return "FinalSoftcap";
        case Kernel::Argmax:           return "Argmax";
        default:                       return "?";
    }
}

// Valid flat element count per tapped kind (matches the golden tensor's flattened shape).
size_t golden_len(Kernel k, int L, const Gemma4Geometry& g) {
    switch (k) {
        case Kernel::EmbedGather: case Kernel::AttnNorm: case Kernel::QmvO:
        case Kernel::PostAttnNorm: case Kernel::AttnResidual: case Kernel::FfnNorm:
        case Kernel::QmvDown: case Kernel::PostFfnNorm: case Kernel::FfnResidual:
        case Kernel::PleProjLayerGemv: case Kernel::PleNorm: case Kernel::PleResidual:
        case Kernel::LayerScalar: case Kernel::FinalRms:
            return g.hidden;
        case Kernel::PleCombine:   return size_t(g.n_layers) * g.per_layer_emb_dim;
        case Kernel::QmvQ: case Kernel::QNorm: case Kernel::RopeQ: case Kernel::Sdpa:
            return g.q_dim_at(L);
        case Kernel::QmvK: case Kernel::QmvV: case Kernel::KNorm: case Kernel::VNorm:
        case Kernel::RopeK:
            return g.kv_dim_at(L);
        case Kernel::QmvGate: case Kernel::QmvUp: case Kernel::GegluTanh:
            return g.intermediate_at(L);
        case Kernel::PleGateGemv: case Kernel::PleGeglu:
            return g.per_layer_emb_dim;
        case Kernel::LmHead: case Kernel::FinalSoftcap:
            return g.vocab;
        default: return g.hidden;
    }
}

// Bind every dispatch's scratch activation X/Out (no_recycle: one pool buffer per ordinal).
// Tracks logical registers along the reference dataflow; returns outbuf[ordinal] = the
// buffer holding that dispatch's output value (for the golden dump). Bind index conventions
// mirror the .metal kernels (Qmv X=3/Out=4; Rms X=0/Out=2; residual/geglu X=0/Res=1/Out=2;
// rope/vnorm in-place or X=0; embed Out=4; sdpa Q=0/Out=3).
void wire_dataflow(RawMetalContext& ctx, const std::vector<Gemma4Dispatch>& dag,
                   const Gemma4Geometry& g, const std::vector<SlotHandle>& pool,
                   const BoundGemma4& b, std::vector<SlotHandle>& outbuf) {
    const size_t ple_dim_bytes = size_t(g.per_layer_emb_dim) * 2;  // bf16 slice stride
    auto A = [&](int ord, uint8_t idx, const SlotHandle& s, size_t off = 0) {
        ctx.arg_bind_ordinal(ord, idx, s, off);
    };

    // logical registers (SlotHandle of the buffer currently holding each value)
    SlotHandle hidden, embed, ple_token, ple_proj, ple_projn, ple_input;
    SlotHandle normed, q, qn, k, v, kn, vn, sdpa_o, o, fn, gate, up, geglu, down;
    SlotHandle plegate, plegated, pleproj, finalnorm, post_tmp;

    for (const auto& d : dag) {
        const int ord = d.ordinal;
        const int L = d.layer;
        const SlotHandle& P = pool[ord];
        outbuf[ord] = SlotHandle{};  // default invalid (untappable)

        switch (d.kind) {
            // ── PLE precompute ──
            case Kernel::EmbedGather:
                A(ord, (uint8_t)bind::EmbedScaled::Out, P); embed = hidden = P; outbuf[ord] = P; break;
            case Kernel::PleTokenGather:
                A(ord, (uint8_t)bind::EmbedScaled::Out, P); ple_token = P; break;
            case Kernel::PleProjGemv:
                A(ord, (uint8_t)bind::Qmv::X, embed); A(ord, (uint8_t)bind::Qmv::Out, P);
                ple_proj = P; break;
            case Kernel::PleProjNorm:
                A(ord, (uint8_t)bind::Rms::X, ple_proj); A(ord, (uint8_t)bind::Rms::Out, P);
                ple_projn = P; break;
            case Kernel::PleCombine:
                A(ord, (uint8_t)bind::PleCombine::Proj, ple_projn);
                A(ord, (uint8_t)bind::PleCombine::Token, ple_token);
                A(ord, (uint8_t)bind::PleCombine::Out, P);
                ple_input = P; outbuf[ord] = P; break;

            // ── attention ──
            case Kernel::AttnNorm:
                A(ord, (uint8_t)bind::Rms::X, hidden); A(ord, (uint8_t)bind::Rms::Out, P);
                normed = P; outbuf[ord] = P; break;
            case Kernel::QmvQ:
                A(ord, (uint8_t)bind::Qmv::X, normed); A(ord, (uint8_t)bind::Qmv::Out, P);
                q = P; outbuf[ord] = P; break;
            case Kernel::QNorm:
                A(ord, (uint8_t)bind::Rms::X, q); A(ord, (uint8_t)bind::Rms::Out, P);
                qn = P; outbuf[ord] = P; break;
            case Kernel::QmvK:
                A(ord, (uint8_t)bind::Qmv::X, normed); A(ord, (uint8_t)bind::Qmv::Out, P);
                k = P; outbuf[ord] = P; break;
            case Kernel::QmvV:
                A(ord, (uint8_t)bind::Qmv::X, normed); A(ord, (uint8_t)bind::Qmv::Out, P);
                v = P; outbuf[ord] = P; break;
            case Kernel::KNorm:
                A(ord, (uint8_t)bind::Rms::X, k); A(ord, (uint8_t)bind::Rms::Out, P);
                kn = P; outbuf[ord] = P; break;
            case Kernel::VNorm:
                A(ord, (uint8_t)bind::VNorm::X, v); A(ord, (uint8_t)bind::VNorm::Out, P);
                vn = P; outbuf[ord] = P; break;
            case Kernel::RopeK:  // in-place on kn
                A(ord, (uint8_t)bind::Rope::X, kn); outbuf[ord] = kn; break;
            case Kernel::RopeQ:  // in-place on qn
                A(ord, (uint8_t)bind::Rope::X, qn); outbuf[ord] = qn; break;
            case Kernel::KvAppend:
                A(ord, (uint8_t)bind::KvAppend::K, kn); A(ord, (uint8_t)bind::KvAppend::V, vn);
                break;  // pages/pos bound in bind_gemma4_weights
            case Kernel::Sdpa:
                A(ord, (uint8_t)bind::Sdpa::Q, qn); A(ord, (uint8_t)bind::Sdpa::Out, P);
                sdpa_o = P; outbuf[ord] = P; break;
            case Kernel::QmvO:
                A(ord, (uint8_t)bind::Qmv::X, sdpa_o); A(ord, (uint8_t)bind::Qmv::Out, P);
                o = P; outbuf[ord] = P; break;
            case Kernel::PostAttnNorm:
                A(ord, (uint8_t)bind::Rms::X, o); A(ord, (uint8_t)bind::Rms::Out, P);
                post_tmp = P; outbuf[ord] = P; break;
            case Kernel::AttnResidual:  // out = post_attn_norm + residual(pre)
                A(ord, (uint8_t)bind::Geglu::Gate, post_tmp);  // residual_add X=0
                A(ord, (uint8_t)bind::Geglu::Up, hidden);      // residual=1
                A(ord, (uint8_t)bind::Geglu::Out, P);          // out=2
                hidden = P; outbuf[ord] = P; break;

            // ── FFN ──
            case Kernel::FfnNorm:
                A(ord, (uint8_t)bind::Rms::X, hidden); A(ord, (uint8_t)bind::Rms::Out, P);
                fn = P; outbuf[ord] = P; break;
            case Kernel::QmvGate:
                A(ord, (uint8_t)bind::Qmv::X, fn); A(ord, (uint8_t)bind::Qmv::Out, P);
                gate = P; outbuf[ord] = P; break;
            case Kernel::QmvUp:
                A(ord, (uint8_t)bind::Qmv::X, fn); A(ord, (uint8_t)bind::Qmv::Out, P);
                up = P; outbuf[ord] = P; break;
            case Kernel::GegluTanh:
                A(ord, (uint8_t)bind::Geglu::Gate, gate); A(ord, (uint8_t)bind::Geglu::Up, up);
                A(ord, (uint8_t)bind::Geglu::Out, P);
                geglu = P; outbuf[ord] = P; break;
            case Kernel::QmvDown:
                A(ord, (uint8_t)bind::Qmv::X, geglu); A(ord, (uint8_t)bind::Qmv::Out, P);
                down = P; outbuf[ord] = P; break;
            case Kernel::PostFfnNorm:
                A(ord, (uint8_t)bind::Rms::X, down); A(ord, (uint8_t)bind::Rms::Out, P);
                post_tmp = P; outbuf[ord] = P; break;
            case Kernel::FfnResidual:
                A(ord, (uint8_t)bind::Geglu::Gate, post_tmp); A(ord, (uint8_t)bind::Geglu::Up, hidden);
                A(ord, (uint8_t)bind::Geglu::Out, P);
                hidden = P; outbuf[ord] = P; break;

            // ── PLE residual + layer scalar ──
            case Kernel::PleGateGemv:
                A(ord, (uint8_t)bind::Qmv::X, hidden); A(ord, (uint8_t)bind::Qmv::Out, P);
                plegate = P; outbuf[ord] = P; break;
            case Kernel::PleGeglu:  // gelu_tanh(gate) * ple_input[L slice]
                A(ord, (uint8_t)bind::Geglu::Gate, plegate);
                A(ord, (uint8_t)bind::Geglu::Up, ple_input, size_t(L) * ple_dim_bytes);
                A(ord, (uint8_t)bind::Geglu::Out, P);
                plegated = P; outbuf[ord] = P; break;
            case Kernel::PleProjLayerGemv:
                A(ord, (uint8_t)bind::Qmv::X, plegated); A(ord, (uint8_t)bind::Qmv::Out, P);
                pleproj = P; outbuf[ord] = P; break;
            case Kernel::PleNorm:
                A(ord, (uint8_t)bind::Rms::X, pleproj); A(ord, (uint8_t)bind::Rms::Out, P);
                post_tmp = P; outbuf[ord] = P; break;
            case Kernel::PleResidual:
                A(ord, (uint8_t)bind::Geglu::Gate, post_tmp); A(ord, (uint8_t)bind::Geglu::Up, hidden);
                A(ord, (uint8_t)bind::Geglu::Out, P);
                hidden = P; outbuf[ord] = P; break;
            case Kernel::LayerScalar:  // hidden *= layer_scalar[0]
                A(ord, (uint8_t)bind::LayerScalar::X, hidden);
                A(ord, (uint8_t)bind::LayerScalar::Out, P);
                hidden = P; outbuf[ord] = P; break;

            // ── tail ──
            case Kernel::FinalRms:
                A(ord, (uint8_t)bind::Rms::X, hidden); A(ord, (uint8_t)bind::Rms::Out, P);
                finalnorm = P; outbuf[ord] = P; break;
            case Kernel::LmHead:  // logits -> IO (Out bound in bind_gemma4_weights)
                A(ord, (uint8_t)bind::Qmv::X, finalnorm); outbuf[ord] = b.logits; break;
            case Kernel::FinalSoftcap:
                A(ord, (uint8_t)bind::Softcap::Logits, b.logits);
                A(ord, (uint8_t)bind::Softcap::Out, b.logits_capped);
                outbuf[ord] = b.logits_capped; break;
            case Kernel::Argmax: break;
        }
    }
}

}  // anonymous namespace

int run_main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    setbuf(stderr, nullptr);
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <checkpoint_dir> <kernels_dir> [comma_separated_prompt_ids]\n", argv[0]);
        return 2;
    }
    const std::string ckpt_dir = argv[1];
    const std::string kernels_dir = argv[2];
    // Default = gemma4 pos-7 golden prompt (charlie's capture, ~/parity-golden/gemma4-4bit-pos7).
    std::string ids_csv = (argc > 3) ? argv[3] : "818,5279,529,7001,563,9079,236764,506";
    std::vector<uint32_t> ids;
    for (size_t p = 0; p < ids_csv.size();) {
        size_t c = ids_csv.find(',', p);
        if (c == std::string::npos) c = ids_csv.size();
        ids.push_back(uint32_t(std::stoul(ids_csv.substr(p, c - p))));
        p = c + 1;
    }
    const int max_ctx = std::getenv("PIE_MAX_CTX") ? std::atoi(std::getenv("PIE_MAX_CTX")) : 4096;

    Gemma4Geometry g;  // E2B defaults
    std::printf("[gemma4_decode_run] checkpoint=%s\n[gemma4_decode_run] kernels=%s\n",
                ckpt_dir.c_str(), kernels_dir.c_str());
    std::printf("[gemma4_decode_run] prompt ids (%zu):", ids.size());
    for (uint32_t id : ids) std::printf(" %u", id);
    std::printf("\n");

    // ── Open checkpoint + build the DAG ──
    SafetensorsView view(ckpt_dir);
    std::vector<Gemma4Dispatch> dag = build_gemma4_dag(g);
    // Driver argmaxes host-side over logits_capped → drop the optional device-argmax
    // dispatch (its PSO is intentionally not loaded; encoding it would set_pso(nil)).
    dag.erase(std::remove_if(dag.begin(), dag.end(),
                             [](const Gemma4Dispatch& d) { return d.kind == Kernel::Argmax; }),
              dag.end());

    size_t weights_bytes = 0;
    for (const auto& name : gemma4_weight_tensors(g)) weights_bytes += view.get(name).nbytes;

    // KV bytes (non-shared layers only).
    size_t kv_bytes = 0;
    for (int L = 0; L < g.n_layers; ++L)
        if (!g.is_kv_shared(L))
            kv_bytes += 2 * size_t(g.n_kv_heads) * max_ctx * g.head_dim_at(L) * 2;

    // Scratch: no_recycle, one buffer per dispatch sized to the widest M=1 activation.
    int widest = g.intermediate_at(g.n_layers - 1);                    // 12288 (double-wide)
    widest = std::max(widest, g.n_layers * g.per_layer_emb_dim);       // 8960 (PLE tables)
    widest = std::max(widest, g.q_dim_at(g.n_layers - 1));             // full q
    const size_t scratch_slot = ((size_t(widest) * 2 + 255) / 256) * 256;
    const size_t scratch_bytes = dag.size() * scratch_slot;
    const size_t consts_bytes  = dag.size() * 8 * 256;
    const size_t io_bytes      = size_t(g.vocab) * 2 * 2 + 4096;
    const size_t heap_bytes = ((weights_bytes + 255) / 256) * 256 + kv_bytes + scratch_bytes
                            + consts_bytes + io_bytes + (64u << 20);

    std::printf("[gemma4_decode_run] dag=%zu weights=%.2f GB kv=%.1f MB scratch=%.1f MB heap=%.2f GB\n",
                dag.size(), weights_bytes / 1e9, kv_bytes / 1048576.0,
                scratch_bytes / 1048576.0, heap_bytes / 1e9);

    auto ctx = RawMetalContext::create(heap_bytes);
    if (!ctx) { std::fprintf(stderr, "[gemma4_decode_run] context create failed\n"); return 1; }

    // ── Stage + bind weights/KV/IO, consts, scratch dataflow ──
    BoundGemma4 b = stage_gemma4_weights(*ctx, view, g, max_ctx);
    bind_gemma4_weights(*ctx, b, dag, g);

    std::vector<SlotHandle> pool(dag.size());
    for (size_t i = 0; i < dag.size(); ++i) pool[i] = ctx->heap_alloc(scratch_slot);
    std::vector<SlotHandle> outbuf(dag.size());
    wire_dataflow(*ctx, dag, g, pool, b, outbuf);

    const int n_consts = bind_gemma4_consts(*ctx, dag, g, max_ctx);
    std::printf("[gemma4_decode_run] bound %d const-param slots\n", n_consts);

    Gemma4StepPsos psos;
    std::string err;
    if (!load_gemma4_psos(*ctx, kernels_dir, g, psos, /*with_argmax=*/false, &err)) {
        std::fprintf(stderr, "[gemma4_decode_run] PSO load failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("[gemma4_decode_run] PSOs compiled\n");

    ctx->make_resident();

    // ── Decode loop: feed each prompt id as an M=1 step (KV append-only across steps) ──
    // PIE_CONCUR selects the encoder concurrency policy (0=barrier-after-each default,
    // 1=Gate‖Up overlap). Set it for dump+gate runs to validate a concurrency mode bit-exact.
    const int concur = std::getenv("PIE_CONCUR") ? std::atoi(std::getenv("PIE_CONCUR")) : 0;
    StepTiming last{};
    for (size_t i = 0; i < ids.size(); ++i) {
        write_u32(b.io_token,    ids[i]);
        write_u32(b.io_position, uint32_t(i));
        write_u32(b.io_seqlen,   uint32_t(i + 1));
        last = ctx->run_step(
            [&](StepEncoder& se) { encode_gemma4_step(se, dag, psos, g, false,
                                                      BarrierVisibility::ExecutionOnly, concur); },
            int(i & 1));
        std::printf("[gemma4_decode_run] step %zu (id=%u pos=%zu): encode_ms=%.4f gpu_exec_ms=%.4f\n",
                    i, ids[i], i, last.encode_ms, last.gpu_exec_ms);
    }
    std::printf("[gemma4_decode_run] HEADLINE last-step: encode_ms=%.4f gpu_exec_ms=%.4f total_ms=%.4f\n",
                last.encode_ms, last.gpu_exec_ms, last.total_ms());

    // ── Optional CONCURRENCY A/B (PIE_CONCUR_AB=1): min-floor of barrier-after-each (concur=0)
    //    vs Gate‖Up overlap (concur=1) vs the all-drop ceiling (concur=-1). Answers manager's
    //    measure-first gate: does dropping the barrier on the 2.15ms Gate‖Up pair move gemma4's
    //    number, or is it ~0 like qwen's already-overlapped pairs? ──
    if (std::getenv("PIE_CONCUR_AB")) {
        const int reps = std::getenv("PIE_CONCUR_REPS") ? std::atoi(std::getenv("PIE_CONCUR_REPS")) : 30;
        auto timed = [&](int cc) {
            double best = 1e9;
            for (int r = 0; r < reps; ++r) {
                StepTiming t = ctx->run_step(
                    [&](StepEncoder& se) { encode_gemma4_step(se, dag, psos, g, false,
                                                              BarrierVisibility::ExecutionOnly, cc); }, 0);
                if (r > 0) best = std::min(best, t.gpu_exec_ms);
            }
            return best;
        };
        const double b0 = timed(0), b1 = timed(1), b2 = timed(2), bc = timed(-1);
        std::printf("\n==== gemma4 CONCURRENCY A/B (min-floor over %d reps) ====\n", reps);
        std::printf("  barriers: each=%d  +gate||up=%d  greedy=%d\n",
                    gemma4_plan_barrier_count(dag, g, 0), gemma4_plan_barrier_count(dag, g, 1),
                    gemma4_plan_barrier_count(dag, g, 2));
        std::printf("  concur=0  barrier-after-each   = %.4f ms\n", b0);
        std::printf("  concur=1  Gate‖Up overlap      = %.4f ms  (Δ %.4f, %.1f%%)\n",
                    b1, b0 - b1, 100.0 * (b0 - b1) / b0);
        std::printf("  concur=2  greedy RAW predicate = %.4f ms  (Δ %.4f, %.1f%%)\n",
                    b2, b0 - b2, 100.0 * (b0 - b2) / b0);
        std::printf("  concur=-1 ALL-drop ceiling     = %.4f ms  (Δ %.4f, %.1f%% — max upside)\n",
                    bc, b0 - bc, 100.0 * (b0 - bc) / b0);
        return 0;
    }

    // ── Optional per-kind ABLATION ranking (PIE_ABLATE_RANK=1): for every Kernel kind present
    //    in the DAG, time the full DAG vs the DAG with that kind's dispatches removed; Δ = the
    //    aggregate gpu-exec cost of that kind (differencing-free — kernels still execute, only
    //    downstream reads garbage, but gpu_exec is valid). Ranks the hot kernels for the perf
    //    pass (delta's qwen3.6 method). State is intact from the decode loop. ──
    if (std::getenv("PIE_ABLATE_RANK")) {
        const int reps = std::getenv("PIE_ABLATE_REPS") ? std::atoi(std::getenv("PIE_ABLATE_REPS")) : 30;
        auto timed = [&](const std::vector<Gemma4Dispatch>& d) {
            double best = 1e9;  // min-floor over reps (thermal-robust)
            for (int r = 0; r < reps; ++r) {
                StepTiming t = ctx->run_step(
                    [&](StepEncoder& se) { encode_gemma4_step(se, d, psos, g); }, 0);
                if (r > 0) best = std::min(best, t.gpu_exec_ms);
            }
            return best;
        };
        const double full = timed(dag);
        // Collect distinct kinds in DAG order, with dispatch counts.
        std::vector<Kernel> order; std::vector<int> counts;
        for (const auto& d : dag) {
            auto it = std::find(order.begin(), order.end(), d.kind);
            if (it == order.end()) { order.push_back(d.kind); counts.push_back(1); }
            else counts[it - order.begin()]++;
        }
        struct Row { const char* name; int n; double dms; };
        std::vector<Row> rows;
        for (size_t i = 0; i < order.size(); ++i) {
            std::vector<Gemma4Dispatch> ablated;
            for (const auto& d : dag) if (d.kind != order[i]) ablated.push_back(d);
            const double abl = timed(ablated);
            rows.push_back({kind_name(order[i]), counts[i], full - abl});
        }
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.dms > b.dms; });
        std::printf("\n==== gemma4 per-kind ABLATION RANK (full=%.4f ms, min-floor over %d reps) ====\n",
                    full, reps);
        std::printf("  %-18s %5s %10s %9s %12s\n", "kind", "#disp", "Δms", "%step", "Δms/disp");
        for (const auto& r : rows)
            std::printf("  %-18s %5d %10.4f %8.1f%% %12.5f\n",
                        r.name, r.n, r.dms, 100.0 * r.dms / full, r.n ? r.dms / r.n : 0.0);
        return 0;
    }

    // ── Optional targeted kind ABLATE (PIE_ABLATE=QmvGate,QmvUp,...): same as one rank row,
    //    for A/B-ing a specific fusion/coop experiment. ──
    if (const char* abl = std::getenv("PIE_ABLATE")) {
        const int reps = std::getenv("PIE_ABLATE_REPS") ? std::atoi(std::getenv("PIE_ABLATE_REPS")) : 30;
        std::string want = abl;
        auto in_set = [&](Kernel k) {
            const std::string n = kind_name(k);
            size_t p = 0;
            while (p <= want.size()) {
                size_t c = want.find(',', p);
                std::string tok = want.substr(p, c == std::string::npos ? std::string::npos : c - p);
                if (tok == n) return true;
                if (c == std::string::npos) break;
                p = c + 1;
            }
            return false;
        };
        auto timed = [&](const std::vector<Gemma4Dispatch>& d) {
            double best = 1e9;
            for (int r = 0; r < reps; ++r) {
                StepTiming t = ctx->run_step(
                    [&](StepEncoder& se) { encode_gemma4_step(se, d, psos, g); }, 0);
                if (r > 0) best = std::min(best, t.gpu_exec_ms);
            }
            return best;
        };
        std::vector<Gemma4Dispatch> ablated; int removed = 0;
        for (const auto& d : dag) { if (in_set(d.kind)) ++removed; else ablated.push_back(d); }
        const double full = timed(dag), abl_ms = timed(ablated);
        std::printf("[gemma4_decode_run] ABLATE '%s' (min-floor over %d reps)\n", abl, reps);
        std::printf("  full DAG    (%3zu disp) = %.4f ms\n", dag.size(), full);
        std::printf("  ablated DAG (%3zu disp) = %.4f ms  (removed %d)\n", ablated.size(), abl_ms, removed);
        std::printf("  Δ (cost of '%s') = %.4f ms  (%.1f%% of step, %.5f ms/disp)\n",
                    abl, full - abl_ms, 100.0 * (full - abl_ms) / full,
                    removed ? (full - abl_ms) / removed : 0.0);
        return 0;
    }

    // ── Optional golden-tap dump (cosine_bisect vs charlie's pos-7 golden) ──
    if (const char* dump_dir = std::getenv("PIE_DUMP_TAPS")) {
        int dumped = 0;
        for (const auto& d : dag) {
            const char* tag = golden_tag(d.kind);
            if (!tag || !outbuf[d.ordinal].valid()) continue;
            const size_t n = golden_len(d.kind, d.layer, g);
            const uint16_t* src = static_cast<const uint16_t*>(outbuf[d.ordinal].contents());
            std::vector<float> vf(n);
            for (size_t j = 0; j < n; ++j) vf[j] = bf16_to_f32(src[j]);
            std::string path = std::string(dump_dir) + "/";
            if (d.layer >= 0) path += std::to_string(d.layer) + ".";
            path += std::string(tag) + ".npy";
            write_npy_f32(path, vf.data(), n);
            ++dumped;
        }
        std::printf("[gemma4_decode_run] dumped %d taps -> %s\n", dumped, dump_dir);
    }

    // ── Argmax the (softcapped) logits ──
    const uint16_t* lb = static_cast<const uint16_t*>(b.logits_capped.contents());
    int best = 0; float best_v = bf16_to_f32(lb[0]);
    for (int i = 1; i < g.vocab; ++i) {
        float v = bf16_to_f32(lb[i]);
        if (v > best_v) { best_v = v; best = i; }
    }
    std::printf("[gemma4_decode_run] argmax(bf16)=%d  logit=%.4f\n", best, best_v);
    std::printf("[gemma4_decode_run] OK\n");
    return 0;
}

}  // namespace pie_metal_driver::raw_metal::gemma4

int main(int argc, char** argv) {
    return pie_metal_driver::raw_metal::gemma4::run_main(argc, argv);
}
