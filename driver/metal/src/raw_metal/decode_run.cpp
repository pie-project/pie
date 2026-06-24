// decode_run.cpp — delta's headline integration driver: stage → bind → consts → scratch →
// PSOs → resident → run_step → report encode-ms/gpu-exec-ms + argmax.
//
// Wires the complete raw-Metal M=1 decode step end-to-end on the real qwen3.5/qwen3.6
// checkpoint, exercising every lane: alpha's RawMetalContext + run_step, beta's
// build_decode_dag + scratch schedule + PSO loader + encode_decode_step, and delta's
// weight staging (heap_bind) + const-param binder (decode_consts) + IO scalars.
//
// Usage: decode_run <checkpoint_dir> <kernels_dir> [token_id] [position]
//   token_id  — the input token to decode from (default 0).
//   position  — the current sequence position / rope pos (default 0; seq_len = pos+1).
//
// Prints the per-step encode/GPU split and the argmax of the produced logits. (Per-tap
// <layer>.<tag>.npy dumps for charlie's cosine_bisect are a follow-up; this proves the
// full forward runs fault-free and returns timing + a sampled token.)

#include <cmath>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "decode_abi.hpp"
#include "decode_consts.hpp"
#include "decode_psos.hpp"
#include "decode_step.hpp"
#include "decode_timing.hpp"
#include "heap_bind.hpp"
#include "heap_bind_metal.hpp"
#include "heap_layout.hpp"
#include "mtl4_context.hpp"
#include "safetensors_view.hpp"
#include "scratch_schedule.hpp"

using namespace pie_metal_driver::raw_metal;

namespace {

void write_u32(const SlotHandle& s, uint32_t v) {
    std::memcpy(s.contents(), &v, sizeof(v));
}

// bf16 (upper 16 bits of f32) -> float.
inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Minimal numpy .npy writer (float32) for parity dumps. rows==0 → 1-D shape (n,);
// rows>0 → 2-D shape (rows, n/rows) to match charlie's golden tensors.
void write_npy_f32(const std::string& path, const float* data, size_t n, size_t rows = 0) {
    std::string shape = rows ? "(" + std::to_string(rows) + ", " + std::to_string(n / rows) + ")"
                             : "(" + std::to_string(n) + ",)";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape + ", }";
    size_t base = 10 + hdr.size() + 1;             // magic+ver+len(2) ... +newline
    size_t pad = (64 - (base % 64)) % 64;
    hdr.append(pad, ' ');
    hdr.push_back('\n');
    uint16_t hlen = uint16_t(hdr.size());
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "[decode_run] cannot open %s\n", path.c_str()); return; }
    const char magic[8] = {'\x93','N','U','M','P','Y','\x01','\x00'};
    std::fwrite(magic, 1, 8, f);
    std::fwrite(&hlen, 2, 1, f);
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    std::fwrite(data, sizeof(float), n, f);
    std::fclose(f);
}

// Find the pool buffer a dispatch writes its output activation to (beta's scratch schedule).
int out_buffer_id(const ScratchSchedule& sched, int ordinal, uint8_t out_bind_index) {
    for (const auto& sb : sched.per_dispatch[ordinal].binds)
        if (sb.bind_index == out_bind_index) return sb.buffer_id;
    return -1;
}

const char* kind_name(Kernel k) {
    static const char* n[] = {"EmbedGather","Rms","QmvIn","QmvInZ","GdnInA","GdnInB","GdnPrep","GdnCore",
        "GatedRms","QmvOut","Residual","QmvQ","QSplit","QmvK","QmvV","QNorm","KNorm","Rope",
        "RopeK","KvAppend","Sdpa","AttnGate","QmvO","FfnRms","QmvGate","QmvUp","SiluMul",
        "QmvDown","LayerOut","FinalRms","QmvLmHead","Argmax"};
    return n[int(k)];
}

// The scratch bind-index a dispatch writes its primary output activation to (-1 = output is
// not in scratch: KvAppend→KV pages, QmvLmHead→IO logits).
int output_bind_index(Kernel k) {
    switch (k) {
        case Kernel::EmbedGather: return int(bind::Embed::Out);          // 4
        case Kernel::Rms: case Kernel::FfnRms: case Kernel::QNorm:
        case Kernel::KNorm: case Kernel::FinalRms: return int(bind::Rms::Out);  // 2
        case Kernel::QmvIn: case Kernel::QmvInZ: case Kernel::QmvOut: case Kernel::QmvQ:
        case Kernel::QmvK: case Kernel::QmvV: case Kernel::QmvO: case Kernel::QmvGate:
        case Kernel::QmvUp: case Kernel::QmvDown: return int(bind::Qmv::Out);   // 4
        case Kernel::GdnInA: case Kernel::GdnInB: return int(bind::Dense::Out); // 2
        case Kernel::GdnCore:  return int(bind::GdnCore::CoreOut);   // 3
        case Kernel::GatedRms: return int(bind::GatedRms::Out);      // 3
        case Kernel::QSplit:   return int(bind::QSplit::QOut);       // 1
        case Kernel::AttnGate: return int(bind::AttnGate::Attn);     // 0 (in-place)
        case Kernel::Rope: case Kernel::RopeK: return int(bind::Rope::X);  // 0 (in-place)
        case Kernel::Sdpa:     return int(bind::Sdpa::Out);          // 3
        case Kernel::SiluMul:  return int(bind::SiluMul::Out);       // 2
        case Kernel::Residual: case Kernel::LayerOut: return int(bind::Residual::Out);  // 2
        default: return -1;  // KvAppend, QmvLmHead, Argmax
    }
}

// charlie's golden tag for a kernel kind (<layer>.<tag>.npy), or nullptr if the kind's
// output is not golden-tapped (KvAppend → KV pages; Argmax). GdnCore is the pre-gated-rms
// core (gdn_core_pre, not in the golden but harmless); GatedRms is the golden `gdn_core`.
const char* golden_tag(Kernel k) {
    switch (k) {
        case Kernel::EmbedGather: return "embed";
        case Kernel::Rms:         return "attn_norm";
        case Kernel::QmvIn:       return "gdn_in_qkv";
        case Kernel::QmvInZ:      return "gdn_in_z";
        case Kernel::GdnInA:      return "gdn_in_a";
        case Kernel::GdnInB:      return "gdn_in_b";
        case Kernel::GdnCore:     return "gdn_core_pre";
        case Kernel::GatedRms:    return "gdn_core";
        case Kernel::QmvOut:      return "gdn_out";
        case Kernel::Residual:    return "attn_resid";
        case Kernel::QmvQ:        return "q_proj";
        case Kernel::QSplit:      return "q_split";
        case Kernel::QmvK:        return "k_proj";
        case Kernel::QmvV:        return "v_proj";
        case Kernel::QNorm:       return "q_norm";
        case Kernel::KNorm:       return "k_norm";
        case Kernel::Rope:        return "rope_q";
        case Kernel::RopeK:       return "rope_k";
        case Kernel::Sdpa:        return "sdpa";
        case Kernel::AttnGate:    return "attn_gated";
        case Kernel::QmvO:        return "o_proj";
        case Kernel::FfnRms:      return "ffn_norm";
        case Kernel::QmvGate:     return "gate_proj";
        case Kernel::QmvUp:       return "up_proj";
        case Kernel::SiluMul:     return "swiglu";
        case Kernel::QmvDown:     return "down_proj";
        case Kernel::LayerOut:    return "layer_out";
        case Kernel::FinalRms:    return "final_norm";
        case Kernel::QmvLmHead:   return "logits";
        default:                  return nullptr;  // KvAppend, Argmax
    }
}

// Valid output element count (flat) per kind — must match the golden tensor shape so
// cosine_bisect compares like-for-like (the scratch slot holds 6144 bf16; kernels write
// only this prefix).
size_t golden_len(Kernel k, const DecodeGeometry& g) {
    const int q_dim = g.n_q_heads * g.head_dim;     // 2048
    const int kv    = g.n_kv_heads * g.head_dim;    // 512
    switch (k) {
        case Kernel::EmbedGather: case Kernel::Rms: case Kernel::FfnRms: case Kernel::FinalRms:
        case Kernel::QmvOut: case Kernel::QmvO: case Kernel::QmvDown:
        case Kernel::Residual: case Kernel::LayerOut: return g.hidden;          // 1024
        case Kernel::QmvIn:       return g.gdn_conv_dim;                        // 6144
        case Kernel::QmvInZ: case Kernel::GdnCore: case Kernel::GatedRms:
            return g.gdn_v_total;                                              // 2048
        case Kernel::GdnInA: case Kernel::GdnInB: return g.gdn_v_heads;         // 16
        case Kernel::QmvQ:        return 2 * q_dim;                             // 4096
        case Kernel::QSplit:      return q_dim;                                 // 2048
        case Kernel::QmvK: case Kernel::QmvV: case Kernel::KNorm: case Kernel::RopeK:
            return kv;                                                         // 512
        case Kernel::QNorm: case Kernel::Rope: case Kernel::Sdpa: case Kernel::AttnGate:
            return q_dim;                                                      // 2048
        case Kernel::QmvGate: case Kernel::QmvUp: case Kernel::SiluMul:
            return g.intermediate;                                            // 3584
        case Kernel::QmvLmHead:   return g.vocab;                               // 248320
        default:                  return g.hidden;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <checkpoint_dir> <kernels_dir> [comma_separated_prompt_ids]\n", argv[0]);
        return 2;
    }
    const std::string ckpt_dir   = argv[1];
    const std::string kernels_dir = argv[2];
    // Default = the golden's prompt (qwen36-pos7): decode-loop over these ids, accumulating
    // KV + GDN conv/recurrent state; final step (token 304 @ pos 7) should argmax to 264.
    std::string ids_csv = (argc > 3) ? argv[3] : "785,6722,315,9625,374,264,3460,304";
    std::vector<uint32_t> ids;
    for (size_t p = 0; p < ids_csv.size();) {
        size_t c = ids_csv.find(',', p);
        if (c == std::string::npos) c = ids_csv.size();
        ids.push_back(uint32_t(std::stoul(ids_csv.substr(p, c - p))));
        p = c + 1;
    }
    const int max_ctx = 4096;

    DecodeGeometry g;  // defaults match Qwen3.5-0.8B (qwen3.6)

    std::printf("[decode_run] checkpoint=%s\n[decode_run] kernels=%s\n", ckpt_dir.c_str(),
                kernels_dir.c_str());
    std::printf("[decode_run] prompt ids (%zu):", ids.size());
    for (uint32_t id : ids) std::printf(" %u", id);
    std::printf("\n");

    // ── Open the checkpoint (zero-copy mmap) + size the heap from the manifest ──
    SafetensorsView view(ckpt_dir);
    size_t weights_bytes = 0;
    for (const auto& name : decode_weight_tensors(g))
        weights_bytes += view.get(name).nbytes;
    const HeapPlan plan = plan_heap(g, weights_bytes, max_ctx);

    const bool fuse_residual = std::getenv("PIE_FUSE_RESIDUAL") != nullptr;
    const bool gdn_prep = std::getenv("PIE_GDN_PREP") != nullptr;
    const std::vector<Dispatch> dag = build_decode_dag(g, /*with_argmax=*/false, fuse_residual, gdn_prep);
    if (fuse_residual)
        std::printf("[decode_run] PIE_FUSE_RESIDUAL: residual folded into QmvO/QmvOut/QmvDown epilogue\n");
    if (gdn_prep)
        std::printf("[decode_run] PIE_GDN_PREP: GdnCore split into GdnPrep (once/head) + recurrent (128x->1x q/k)\n");

    if (std::getenv("PIE_DAG_DUMP")) {
        for (const auto& d : dag)
            std::printf("ord=%3d L=%2d %-11s grid=%u,%u,%u tg=%u,%u,%u\n",
                        d.ordinal, d.layer, kind_name(d.kind),
                        d.grid.x, d.grid.y, d.grid.z, d.tg.x, d.tg.y, d.tg.z);
        return 0;
    }
    const char* dump_dir      = std::getenv("PIE_DUMP_TAPS");
    // Dump mode preserves every intermediate to end-of-run → force no_recycle.
    const bool no_recycle     = std::getenv("PIE_NO_RECYCLE") != nullptr || dump_dir != nullptr;
    const bool force_barriers = std::getenv("PIE_FORCE_BARRIERS") != nullptr;

    // beta's scratch schedule (WAR/WAW coloring) over the same ordinals. no_recycle gives
    // every value its own buffer (colors_used == #values) for the dump/aliasing diagnostic.
    ScratchSchedule sched = build_scratch_schedule(dag, g, no_recycle);
    std::printf("[decode_run] scratch: colors_used=%d hazard_free=%d no_recycle=%d force_barriers=%d\n",
                sched.colors_used, sched.hazard_free ? 1 : 0, no_recycle ? 1 : 0, force_barriers ? 1 : 0);

    if (std::getenv("PIE_SCHED_DUMP")) {
        const int lim = std::getenv("PIE_SCHED_LIM") ? std::atoi(std::getenv("PIE_SCHED_LIM"))
                                                      : int(dag.size());
        for (int o = 0; o < lim && o < int(dag.size()); ++o) {
            std::printf("ord=%3d %-11s out_bind=%d :", o, kind_name(dag[o].kind),
                        output_bind_index(dag[o].kind));
            for (const auto& sb : sched.per_dispatch[o].binds)
                std::printf("  [bind=%d->buf%d]", sb.bind_index, sb.buffer_id);
            std::printf("\n");
        }
        return 0;
    }

    const size_t consts_budget = decode_consts_budget(dag);
    // Headroom over plan.total: the scratch pool (colors_used slots, up to 362 under
    // no_recycle), const slots + per-alloc 256-align padding across ~700 weight/KV/state/IO
    // allocs (plan.total aligns the SUM once; staging aligns each tensor).
    const size_t heap_bytes = plan.total + consts_budget
                            + size_t(sched.colors_used) * plan.scratch_slot_bytes + (32u << 20);

    std::printf("[decode_run] dag dispatches=%zu  weights=%.1f MB  heap=%.1f MB (plan %.1f + consts %.2f + pad)\n",
                dag.size(), weights_bytes / 1048576.0, heap_bytes / 1048576.0,
                plan.total / 1048576.0, consts_budget / 1048576.0);

    auto ctx = RawMetalContext::create(heap_bytes);
    if (!ctx) { std::fprintf(stderr, "[decode_run] context create failed\n"); return 1; }

    // ── Stage weights/state/KV/IO; bind weight/state/KV/IO slots by ordinal ──
    BoundDecode b = stage_decode_weights(*ctx, view, g, plan);
    bind_decode_dag(*ctx, b, dag, g, gdn_prep);

    // ── Allocate the scratch pool (colors_used slots) and hand to beta's bind pass.
    //    Under no_recycle this is one buffer per value (~362); else SCRATCH_POOL=6. ──
    std::vector<SlotHandle> pool(sched.colors_used);
    for (int i = 0; i < sched.colors_used; ++i)
        pool[i] = ctx->heap_alloc(plan.scratch_slot_bytes);
    if (const char* zs = std::getenv("PIE_ZERO_SCRATCH")) {
        const int fill = (zs[0] == 'F') ? 0x3c : 0x00;  // 'F' => 0x3c3c bf16 ~0.0114 sentinel
        for (int i = 0; i < sched.colors_used; ++i)
            std::memset(pool[i].contents(), fill, plan.scratch_slot_bytes);
    }
    bind_scratch(*ctx, dag, sched, pool.data(), int(pool.size()));

    // ── delta's geometry const params (the previously-unbound `constant&` args) ──
    const int n_consts = bind_decode_consts(*ctx, dag, g, max_ctx, gdn_prep);
    std::printf("[decode_run] bound %d const-param slots\n", n_consts);

    // ── Compile the kernel PSOs ──
    DecodeStepPsos psos;
    std::string err;
    if (!load_decode_psos(*ctx, kernels_dir, psos, /*with_argmax=*/false, &err, fuse_residual, gdn_prep)) {
        std::fprintf(stderr, "[decode_run] PSO load failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("[decode_run] PSOs compiled\n");

    // ── Residency (I2): one set, after all binds ──
    ctx->make_resident();

    // ── Optional embed tap (state-independent first-divergence check): encode ONLY the
    //    EmbedGather dispatch for the last prompt token and dump its output. ──
    if (std::getenv("PIE_DUMP_EMBED") && dag[0].kind == Kernel::EmbedGather) {
        const uint32_t tok = ids.back();
        write_u32(b.io[int(IoSlot::TokenId)], tok);
        ctx->run_step([&](StepEncoder& se) {
            se.set_pso(psos[dag[0].kind]);
            se.set_argtable_ordinal(0);
            se.dispatch(dag[0].grid, dag[0].tg);
            se.barrier();
        });
        const int bid = out_buffer_id(sched, 0, uint8_t(bind::Embed::Out));
        if (bid >= 0) {
            const uint16_t* eb = static_cast<const uint16_t*>(pool[bid].contents());
            std::vector<float> e(g.hidden);
            for (int i = 0; i < g.hidden; ++i) e[i] = bf16_to_f32(eb[i]);
            write_npy_f32("/tmp/embed_ours.npy", e.data(), e.size());
            std::printf("[decode_run] embed tap: token=%u -> /tmp/embed_ours.npy [%d] (buf %d)\n",
                        tok, g.hidden, bid);
        } else {
            std::printf("[decode_run] embed tap: could not locate Embed::Out scratch buffer\n");
        }
    }

    // ── Optional single-step tap-by-ordinal (determinism + divergence localization):
    //    encode dag[0..O] for the FIRST prompt token at pos 0 (empty KV / fresh state) and
    //    dump dispatch O's output activation. Run twice + diff to find non-deterministic
    //    kernels; compare to charlie's golden to find the first divergence. ──
    if (const char* to = std::getenv("PIE_TAP_ORDINAL")) {
        const int O = std::atoi(to);
        const std::string out_path = std::getenv("PIE_TAP_OUT") ? std::getenv("PIE_TAP_OUT")
                                                                : "/tmp/tap.npy";
        write_u32(b.io[int(IoSlot::TokenId)], ids[0]);
        write_u32(b.io[int(IoSlot::Position)], 0u);
        write_u32(b.io[int(IoSlot::SeqLen)],  1u);
        const bool perstep = std::getenv("PIE_TAP_PERSTEP") != nullptr;
        if (perstep) {
            for (int i = 0; i <= O && i < int(dag.size()); ++i) {
                ctx->run_step([&](StepEncoder& se) {
                    se.set_pso(psos[dag[i].kind]);
                    se.set_argtable_ordinal(dag[i].ordinal);
                    se.dispatch(dag[i].grid, dag[i].tg);
                });
            }
        } else {
            ctx->run_step([&](StepEncoder& se) {
                for (int i = 0; i <= O && i < int(dag.size()); ++i) {
                    se.set_pso(psos[dag[i].kind]);
                    se.set_argtable_ordinal(dag[i].ordinal);
                    se.dispatch(dag[i].grid, dag[i].tg);
                    se.barrier();
                }
            });
        }
        const int obi = output_bind_index(dag[O].kind);
        int bid = (obi >= 0) ? out_buffer_id(sched, O, uint8_t(obi)) : -1;
        if (const char* tb = std::getenv("PIE_TAP_BUF")) bid = std::atoi(tb);  // read raw pool[N]
        const size_t n = plan.scratch_slot_bytes / 2;  // bf16 elems in a scratch slot
        if (bid >= 0) {
            const uint16_t* ob = static_cast<const uint16_t*>(pool[bid].contents());
            std::vector<float> v(n);
            for (size_t i = 0; i < n; ++i) v[i] = bf16_to_f32(ob[i]);
            write_npy_f32(out_path, v.data(), n);
            std::printf("[decode_run] tap ord=%d kind=%s -> %s [%zu] (buf %d)\n",
                        O, kind_name(dag[O].kind), out_path.c_str(), n, bid);
        } else {
            std::printf("[decode_run] tap ord=%d kind=%s: output not in scratch (untappable)\n",
                        O, kind_name(dag[O].kind));
        }
        return 0;
    }

    // ── Decode loop: feed each prompt id as an M=1 step, accumulating KV + GDN conv/
    //    recurrent state in-place (I4). Mirrors the golden's PIE_PARITY_DECODE capture. ──
    const bool perstep = std::getenv("PIE_PERSTEP") != nullptr;

    // GDN conv-state cross-step ping-pong: ConvState (RO) and ConvStateOut are DISTINCT
    // buffers, so the conv history must be advanced token-to-token by swapping their bind
    // each step (step i reads what step i-1 wrote). Bound once = stale zeros every step =
    // wrong GDN output at pos>0. recurrent_state is in-place RMW (auto-accumulates).
    struct GdnDisp { int ord; int layer; Kernel kind; };
    std::vector<GdnDisp> gdn_disp;  // GdnCore (+ GdnPrep when split): both touch conv_state
    for (const auto& d : dag)
        if (d.kind == Kernel::GdnCore || d.kind == Kernel::GdnPrep)
            gdn_disp.push_back({d.ordinal, d.layer, d.kind});

    StepTiming last{};
    for (size_t i = 0; i < ids.size(); ++i) {
        write_u32(b.io[int(IoSlot::TokenId)], ids[i]);
        write_u32(b.io[int(IoSlot::Position)], uint32_t(i));
        write_u32(b.io[int(IoSlot::SeqLen)],  uint32_t(i + 1));
        for (const auto& gd : gdn_disp) {                // even step: read A→write B; odd: B→A
            const bool even = (i % 2 == 0);
            const SlotHandle& A = b.gdn[gd.layer].conv_state;
            const SlotHandle& C = b.gdn[gd.layer].conv_state_out;
            uint8_t cs_bind, cso_bind;
            if (gd.kind == Kernel::GdnPrep) {            // prep writes q/k conv_state channels
                cs_bind  = (uint8_t)bind::GdnPrep::ConvState;
                cso_bind = (uint8_t)bind::GdnPrep::ConvStateOut;
            } else if (gdn_prep) {                       // recurrent writes v conv_state channels
                cs_bind  = (uint8_t)bind::GdnCoreRecurrent::ConvState;
                cso_bind = (uint8_t)bind::GdnCoreRecurrent::ConvStateOut;
            } else {                                     // in-kernel-share GdnCore (production)
                cs_bind  = (uint8_t)bind::GdnCore::ConvState;
                cso_bind = (uint8_t)bind::GdnCore::ConvStateOut;
            }
            ctx->arg_bind_ordinal(gd.ord, cs_bind,  even ? A : C);
            ctx->arg_bind_ordinal(gd.ord, cso_bind, even ? C : A);
        }
        if (perstep) {
            for (const auto& d : dag) {
                ctx->run_step([&](StepEncoder& se) {
                    se.set_pso(psos[d.kind]);
                    se.set_argtable_ordinal(d.ordinal);
                    se.dispatch(d.grid, d.tg);
                });
            }
        } else {
            last = ctx->run_step(
                [&](StepEncoder& se) { encode_decode_step(se, dag, psos, force_barriers); }, int(i & 1));
        }
        std::printf("[decode_run] step %zu (id=%u pos=%zu): encode_ms=%.4f gpu_exec_ms=%.4f\n",
                    i, ids[i], i, last.encode_ms, last.gpu_exec_ms);
    }
    std::printf("[decode_run] HEADLINE last-step: encode_ms=%.4f gpu_exec_ms=%.4f total_ms=%.4f\n",
                last.encode_ms, last.gpu_exec_ms, last.total_ms());

    // ── Optional COARSE per-phase attribution (PIE_ATTRIB=1): re-time cumulative DAG
    //    prefixes [0..N) at phase boundaries (embed, each layer end, final_norm, lm_head),
    //    min over repeats. Marginal(boundary) = cum(N_i) - cum(N_{i-1}) localizes the real
    //    gpu-exec ms per phase (complements beta's per-dispatch timestamp tool). State is
    //    intact from the 8-step loop; truncated re-runs measure representative kernel cost. ──
    if (std::getenv("PIE_ATTRIB")) {
        std::vector<std::pair<int, std::string>> cuts;   // (N = #dispatches, label)
        cuts.emplace_back(1, "embed");
        for (size_t k = 0; k < dag.size(); ++k)
            if (dag[k].kind == Kernel::LayerOut)
                cuts.emplace_back(int(k + 1), "L" + std::to_string(dag[k].layer));
            else if (dag[k].kind == Kernel::FinalRms)
                cuts.emplace_back(int(k + 1), "final_norm");
        cuts.emplace_back(int(dag.size()), "lm_head");
        const int reps = std::getenv("PIE_ATTRIB_REPS") ? std::atoi(std::getenv("PIE_ATTRIB_REPS")) : 8;
        std::printf("[decode_run] ATTRIB (min gpu_exec_ms over %d reps; marginal = phase cost):\n", reps);
        double prev = 0.0;
        for (auto& [N, label] : cuts) {
            std::vector<Dispatch> sub(dag.begin(), dag.begin() + N);
            double best = 1e9;
            for (int r = 0; r < reps; ++r) {
                StepTiming t = ctx->run_step(
                    [&](StepEncoder& se) { encode_decode_step(se, sub, psos, force_barriers); }, 0);
                best = std::min(best, t.gpu_exec_ms);
            }
            std::printf("  %-11s N=%3d  cum=%.4f  marginal=%.4f\n", label.c_str(), N, best, best - prev);
            prev = best;
        }
        return 0;
    }


    // ── Optional ABLATION timing (PIE_ATTRIB_ABLATE=GdnInA,GdnInB,...): the decisive,
    //    differencing-free cost measurement. Time the full DAG, then time the DAG with all
    //    dispatches of the named kinds REMOVED (timing-only — downstream reads garbage, but
    //    gpu_exec is valid since the kernels still execute). delta = the true aggregate cost of
    //    those dispatches, free of prefix-diff adjacent-smearing. Answers: is a kind launch-
    //    floor (small delta) or real compute/BW (large delta)? ──
    if (const char* abl = std::getenv("PIE_ATTRIB_ABLATE")) {
        const int reps = std::getenv("PIE_ATTRIB_ABLATE_REPS")
                             ? std::atoi(std::getenv("PIE_ATTRIB_ABLATE_REPS")) : 30;
        std::string want = abl;
        auto in_set = [&](Kernel k) {
            std::string n = kind_name(k);
            size_t p = 0;
            while (p < want.size()) {
                size_t c = want.find(',', p);
                std::string tok = want.substr(p, c == std::string::npos ? c : c - p);
                if (tok == n) return true;
                if (c == std::string::npos) break;
                p = c + 1;
            }
            return false;
        };
        auto timed = [&](const std::vector<Dispatch>& d) {
            double sum = 0.0; int used = 0;
            for (int r = 0; r < reps; ++r) {
                StepTiming t = ctx->run_step(
                    [&](StepEncoder& se) { encode_decode_step(se, d, psos, force_barriers); }, 0);
                if (r > 0) { sum += t.gpu_exec_ms; ++used; }
            }
            return used ? sum / used : 0.0;
        };
        std::vector<Dispatch> ablated;
        int removed = 0;
        for (auto& d : dag) { if (in_set(d.kind)) ++removed; else ablated.push_back(d); }
        double full = timed(dag);
        double abl_ms = timed(ablated);
        std::printf("[decode_run] ABLATE '%s' (mean over %d reps)\n", abl, reps);
        std::printf("  full DAG       (%3zu disp) = %.4f ms\n", dag.size(), full);
        std::printf("  ablated DAG    (%3zu disp) = %.4f ms  (removed %d dispatches)\n",
                    ablated.size(), abl_ms, removed);
        std::printf("  Δ (cost of removed kinds) = %.4f ms  (%.1f%% of step, %.5f ms/disp)\n",
                    full - abl_ms, 100.0 * (full - abl_ms) / full,
                    removed ? (full - abl_ms) / removed : 0.0);
        return 0;
    }

    // ── Optional RELIABLE per-dispatch attribution (PIE_ATTRIB_PERDISP=1): the same
    //    prefix-truncation method as PIE_ATTRIB (which reconciles exactly to the real 7.56ms),
    //    but cut at EVERY ordinal. marginal[N] = cum[N+1] - cum[N] cancels the fixed CB
    //    submit floor; aggregating each kind's marginal across all its 24/18/6 layer-instances
    //    denoises (independent-pass min noise averages out). This is the within-layer launch-
    //    vs-compute split WITHOUT the broken in-encoder timestamps: the min marginal across all
    //    kinds ≈ the pure per-dispatch launch floor; kinds near it are fusion targets, kinds
    //    well above it are kernel-efficiency (BW) targets. ──
    if (std::getenv("PIE_ATTRIB_PERDISP")) {
        const int reps = std::getenv("PIE_ATTRIB_PERDISP_REPS")
                             ? std::atoi(std::getenv("PIE_ATTRIB_PERDISP_REPS")) : 20;
        const int nN = int(dag.size());
        std::vector<double> cum(nN + 1, 0.0);   // cum[N] = mean gpu_exec of prefix [0..N)
        for (int N = 0; N <= nN; ++N) {
            std::vector<Dispatch> sub(dag.begin(), dag.begin() + N);
            double sum = 0.0; int used = 0;
            for (int r = 0; r < reps; ++r) {
                StepTiming t = ctx->run_step(
                    [&](StepEncoder& se) { encode_decode_step(se, sub, psos, force_barriers); }, 0);
                // discard the first (warm-up) rep; mean over the rest is monotonic in prefix
                // length (longer = more work), so marginals don't go spuriously negative.
                if (r > 0) { sum += t.gpu_exec_ms; ++used; }
            }
            cum[N] = used ? sum / used : 0.0;
        }
        std::printf("[decode_run] PERDISP (reliable prefix-diff, mean over %d reps)\n", reps);
        std::printf("  CB submit floor (N=0) = %.4f ms\n", cum[0]);
        // Per-kind aggregation of marginals.
        std::vector<double> by_kind(64, 0.0);
        std::vector<int>    cnt_kind(64, 0);
        std::vector<double> marg(nN, 0.0);
        double tot = 0.0;
        for (int o = 0; o < nN; ++o) {
            double m = cum[o + 1] - cum[o];   // marginal cost of dispatch o
            marg[o] = m;
            int ki = int(dag[o].kind);
            by_kind[ki] += m;
            cnt_kind[ki] += 1;
            tot += m;
        }
        std::printf("  Σ marginals = %.4f ms (reconciles to full-DAG cum = %.4f)\n", tot, cum[nN]);
        // Sort kinds by total marginal (the fusion/kernel-opt ranking).
        std::vector<int> order;
        for (int ki = 0; ki < 64; ++ki) if (cnt_kind[ki] > 0) order.push_back(ki);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return by_kind[a] > by_kind[b]; });
        // Pure-launch-floor estimate = the smallest mean-per-dispatch across kinds.
        double floor_ms = 1e30;
        for (int ki : order) floor_ms = std::min(floor_ms, by_kind[ki] / cnt_kind[ki]);
        std::printf("  est. launch floor ≈ %.5f ms/disp (cheapest kind's mean)\n", floor_ms);
        std::printf("  %-12s %8s %4s %9s %8s %7s  %s\n",
                    "kind", "total_ms", "n", "ms/disp", "%step", "×floor", "verdict");
        for (int ki : order) {
            double per = by_kind[ki] / cnt_kind[ki];
            double xf  = per / floor_ms;
            const char* v = (xf < 1.6) ? "FUSE(launch)" : "KERNEL-OPT(BW)";
            std::printf("  %-12s %8.4f %4d %9.5f %7.1f%% %6.1f×  %s\n",
                        kind_name(Kernel(ki)), by_kind[ki], cnt_kind[ki], per,
                        100.0 * by_kind[ki] / tot, xf, v);
        }
        return 0;
    }

    // ── PER-DISPATCH attribution (PIE_ATTRIB_TS=1) — cumulative-prefix wall-clock + robust
    //    median-per-kind. Why this and not timestamps: alpha's precise MTL4 timestamp write
    //    is NOT gated by the compute barrier (verified — exec-only AND Device-flush both give
    //    a ~0.3ms issue-cadence sum, not the real 7.6ms; lm_head an impossible 11µs/127MB).
    //    So we use delta's trustworthy `gpu_exec_ms` (GPU-event domain) at per-dispatch cuts:
    //    cum(k) = min-over-reps gpu_exec of prefix [0..k); marginal(i)=cum(i+1)-cum(i). A raw
    //    per-dispatch marginal carries a per-cut "drain" artifact (the prefix's last dispatch
    //    is fully drained instead of overlapped — gave multi-ms single-cut spikes), so we
    //    aggregate each KIND by the MEDIAN of its marginals (robust to the few bad cuts) and
    //    report median ms/disp × count. This is the within-layer fuse/kernel-opt ranking. ──
    if (std::getenv("PIE_ATTRIB_TS")) {
        const int    reps = std::getenv("PIE_ATTRIB_TS_REPS")
                                ? std::atoi(std::getenv("PIE_ATTRIB_TS_REPS")) : 6;
        const size_t N    = dag.size();
        std::vector<double> cum_ms(N + 1, 0.0);
        for (size_t k = 1; k <= N; ++k) {
            std::vector<Dispatch> sub(dag.begin(), dag.begin() + k);
            double best = 1e30;
            for (int r = 0; r < reps; ++r) {
                StepTiming t = ctx->run_step(
                    [&](StepEncoder& se) { encode_decode_step(se, sub, psos, force_barriers); }, 0);
                best = std::min(best, t.gpu_exec_ms);
            }
            cum_ms[k] = best;
        }
        for (size_t k = 1; k <= N; ++k)
            if (cum_ms[k] < cum_ms[k - 1]) cum_ms[k] = cum_ms[k - 1];  // enforce monotone

        // Group marginals by kind; median ms/disp per kind is the robust per-dispatch cost.
        std::array<std::vector<double>, kKernelKindCount> by_kind_marg;
        for (size_t i = 0; i < N; ++i)
            by_kind_marg[static_cast<int>(dag[i].kind)].push_back(cum_ms[i + 1] - cum_ms[i]);
        auto median = [](std::vector<double> v) -> double {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            const size_t m = v.size() / 2;
            return (v.size() & 1) ? v[m] : 0.5 * (v[m - 1] + v[m]);
        };
        struct Row { Kernel k; int n; double med; double total; };
        std::vector<Row> rows;
        double step_total = 0.0;
        for (int ki = 0; ki < kKernelKindCount; ++ki) {
            const auto& v = by_kind_marg[ki];
            if (v.empty()) continue;
            const double med = median(v);
            const double tot = med * double(v.size());
            rows.push_back({static_cast<Kernel>(ki), int(v.size()), med, tot});
            step_total += tot;
        }
        std::sort(rows.begin(), rows.end(),
                  [](const Row& a, const Row& b) { return a.total > b.total; });
        std::printf("\n==== per-dispatch attribution (cumulative-prefix, median/kind, reps=%d) ====\n", reps);
        std::printf("full-DAG gpu_exec = %.4f ms ; sum-of-median-kind = %.4f ms\n", cum_ms[N], step_total);
        std::printf("  %-14s %4s  %9s  %10s  %6s\n", "kind", "n", "med_ms/d", "total_ms", "%step");
        for (const auto& r : rows)
            std::printf("  %-14s %4d  %9.5f  %10.4f  %5.1f%%\n",
                        kernel_name(r.k), r.n, r.med, r.total,
                        step_total > 0 ? 100.0 * r.total / step_total : 0.0);
        std::printf("==== end attribution ====\n");
        std::printf("note: unresolved = %.4f ms (full-DAG %.4f − sum-of-median %.4f) is distributed\n"
                    "      across sub-noise-floor dispatches: a single M=1 dispatch (~5-20us) is below\n"
                    "      the gpu_exec event noise, so only individually-large kernels resolve above it\n"
                    "      (gdn_core/lm_head/embed). The rest = launch-floor → FUSION targets.\n",
                    cum_ms[N] - step_total, cum_ms[N], step_total);
        return 0;
    }

    // ── Optional full 363-tap dump (PIE_DUMP_TAPS=<dir>): after the last decode step, with
    //    no_recycle every value is preserved in its own pool buffer → emit each dispatch's
    //    output as <layer>.<tag>.npy for charlie's cosine_bisect vs the pos-7 golden. ──
    if (dump_dir) {
        int dumped = 0;
        for (const auto& d : dag) {
            const char* tag = golden_tag(d.kind);
            if (!tag) continue;
            const size_t n = golden_len(d.kind, g);
            const uint16_t* src = nullptr;
            if (d.kind == Kernel::QmvLmHead) {
                src = static_cast<const uint16_t*>(b.io[int(IoSlot::Logits)].contents());
            } else {
                const int bid = out_buffer_id(sched, d.ordinal, uint8_t(output_bind_index(d.kind)));
                if (bid < 0) continue;
                src = static_cast<const uint16_t*>(pool[bid].contents());
            }
            std::string path = std::string(dump_dir) + "/";
            if (d.layer >= 0) path += std::to_string(d.layer) + ".";
            path += std::string(tag) + ".npy";
            std::vector<float> v(n);
            for (size_t i = 0; i < n; ++i) v[i] = bf16_to_f32(src[i]);
            write_npy_f32(path, v.data(), n, /*rows=*/1);
            ++dumped;
        }
        std::printf("[decode_run] dumped %d taps -> %s\n", dumped, dump_dir);
    }

    // ── Argmax the logits (IO::Logits). lm_head writes bf16 (affine_qmv_*_bfloat16). ──
    const auto& logits_slot = b.io[int(IoSlot::Logits)];
    const uint16_t* lb = static_cast<const uint16_t*>(logits_slot.contents());
    int best = 0;
    float best_v = bf16_to_f32(lb[0]);
    for (int i = 1; i < g.vocab; ++i) {
        float v = bf16_to_f32(lb[i]);
        if (v > best_v) { best_v = v; best = i; }
    }
    std::printf("[decode_run] argmax(bf16)=%d  logit=%.4f  (golden expects 264)\n", best, best_v);

    std::printf("[decode_run] OK\n");
    return 0;
}
