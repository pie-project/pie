// golden_tap.cpp — see golden_tap.hpp.

#include "golden_tap.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <algorithm>

namespace pie::metal {

namespace {

// Which scratch bind index carries each kernel's OUTPUT, what the reference calls
// that tap, and how wide one row is. Mirrors tests/mlx/model/qwen3_5.cpp's
// dump_kernel call sites exactly; a kernel absent from this table is not tapped
// (QSplit and KvAppend have no reference counterpart, Argmax is host-side).
struct Tap {
    const char* name;
    std::uint8_t out_bind;
    int width;
    /// Rows are the SORT's, not the batch's. The sorted stack is a driver-only
    /// row order, so these diff against no reference -- but they diff perfectly
    /// against ANOTHER RUN of this driver, which is what an arm-on/arm-off
    /// bisect of a routed wrong answer needs and what the note below used to
    /// rule out for the wrong reason.
    bool sorted = false;
    /// The tap is int32, not bf16. The routing tensors are indices; dumping
    /// them as floats would read every id as a denormal.
    bool i32 = false;
    /// Rows, when the tap is neither one-per-token nor one-per-sorted-row.
    /// `tile_expert` is one per TILE.
    int tile_rows = 0;
};

bool tap_for(const Dispatch& d, const DecodeGeometry& g, Tap& out) {
    const int q_dim  = g.n_q_heads * g.head_dim;
    const int kv_dim = g.n_kv_heads * g.head_dim;
    switch (d.kind) {
        case Kernel::EmbedGather:   out = {"embed",      4, g.hidden}; return true;
        case Kernel::Rms:           out = {"attn_norm",  2, g.hidden}; return true;
        case Kernel::FfnRms:        out = {"ffn_norm",   2, g.hidden}; return true;
        case Kernel::FinalRms:      out = {"final_norm", 2, g.hidden}; return true;

        case Kernel::QmvIn:         out = {"gdn_in_qkv", 4, g.gdn_conv_dim}; return true;
        case Kernel::QmvInZ:        out = {"gdn_in_z",   4, g.gdn_v_total};  return true;
        case Kernel::GdnInA:        out = {"gdn_in_a",   4, g.gdn_v_heads};  return true;
        case Kernel::GdnInB:        out = {"gdn_in_b",   4, g.gdn_v_heads};  return true;
        // The reference's `gdn_core` tap is the output of gated_delta_net, which
        // already includes the gate RMSNorm — so it lines up with GatedRms here,
        // not with the bare recurrence.
        case Kernel::GatedRms:      out = {"gdn_core",   3, g.gdn_v_heads * g.gdn_v_dim}; return true;
        case Kernel::QmvOut:        out = {"gdn_out",    4, g.hidden}; return true;

        case Kernel::QmvQ:          out = {"q_proj",     4, 2 * q_dim}; return true;
        case Kernel::QmvK:          out = {"k_proj",     4, kv_dim};    return true;
        case Kernel::QmvV:          out = {"v_proj",     4, kv_dim};    return true;
        case Kernel::QNorm:         out = {"q_norm",     2, q_dim};     return true;
        case Kernel::KNorm:         out = {"k_norm",     2, kv_dim};    return true;
        case Kernel::Rope:          out = {"rope_q",     0, q_dim};     return true;
        case Kernel::RopeK:         out = {"rope_k",     0, kv_dim};    return true;
        case Kernel::Sdpa:
        case Kernel::SdpaPaged:     out = {"sdpa",       3, q_dim};     return true;
        case Kernel::AttnGate:      out = {"attn_gated", 0, q_dim};     return true;
        case Kernel::QmvO:          out = {"o_proj",     4, g.hidden};  return true;
        case Kernel::Residual:      out = {"attn_resid", 2, g.hidden};  return true;

        case Kernel::QmvGate:       out = {"gate_proj",  4, g.intermediate}; return true;
        case Kernel::QmvUp:         out = {"up_proj",    4, g.intermediate}; return true;
        // Routed, the dense SwiGLU that remains is the SHARED expert's, at its
        // own width. Named `swiglu` at `g.intermediate` for both, this tap was
        // zero elements wide on every routed checkpoint -- present, empty, and
        // silently skipped by anything comparing it.
        case Kernel::SiluMul:
            out = g.is_moe() ? Tap{"shared_act", 2, g.shared_intermediate}
                             : Tap{"swiglu", 2, g.intermediate};
            return true;
        case Kernel::QmvDown:       out = {"down_proj",  4, g.hidden};       return true;
        case Kernel::LayerOut:      out = {"layer_out",  2, g.hidden};       return true;

        // ── the mixture, in TOKEN order ──
        // What a reference can also produce. The sorted stack below is ours
        // alone, and both are needed: these say WHETHER the mixture is wrong,
        // those say WHERE.
        case Kernel::LlRouter:      out = {"router",     4, g.n_experts}; return true;
        case Kernel::LlMoeCombine:  out = {"moe_out",    2, g.hidden};   return true;

        // ── the mixture, in SORT order ──
        // Only comparable to another run of this driver, which is exactly the
        // comparison a routed wrong answer needs: the routed GEMM and the
        // routed matvec compute the same thing over the same rows, so with the
        // router bit-identical between the two arms every one of these must be
        // too. The first one that is not is the kernel at fault.
        case Kernel::LlMoeGather:      out = {"moe_sorted_x", 1, g.hidden, true}; return true;
        case Kernel::LlExpertGate:     out = {"moe_gp",   4, g.moe_intermediate, true}; return true;
        case Kernel::LlExpertUp:       out = {"moe_up",   4, g.moe_intermediate, true}; return true;
        case Kernel::LlExpertSiluMul:  out = {"moe_hh",   2, g.moe_intermediate, true}; return true;
        case Kernel::LlExpertDown:     out = {"moe_sorted_out", 4, g.hidden, true}; return true;

        // ── the routing itself ──
        // The sort's four outputs. `tile_expert` is what the routed MATMUL
        // reads and the routed MATVEC does not, so it is the one input the two
        // arms of a batched/unbatched bisect do not share.
        case Kernel::LlMoeSort:        out = {"moe_row_expert", 2, 1, true, true}; return true;
        case Kernel::LlSharedGate:  out = {"shared_gate", 4, g.shared_intermediate}; return true;
        case Kernel::LlSharedUp:    out = {"shared_up",  4, g.shared_intermediate};  return true;
        case Kernel::LlSharedDown:  out = {"shared_down", 4, g.hidden};   return true;
        case Kernel::LlSharedGateProj: out = {"shared_g", 4, 1};          return true;
        case Kernel::LlSharedCombine:  out = {"ffn_out",  3, g.hidden};   return true;
        default: return false;
    }
}

float bf16_to_f32(std::uint16_t h) {
    const std::uint32_t bits = std::uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void write_npy(const std::string& path, const std::vector<float>& data, int rows, int width) {
    char shape[64];
    std::snprintf(shape, sizeof(shape), "(%d, %d), ", rows, width);
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': ";
    header += shape;
    header += "}";
    // The header (magic + version + length + text + '\n') must be 64-byte aligned.
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\n';

    std::ofstream out(path, std::ios::binary);
    if (!out) return;
    const char magic[] = "\x93NUMPY\x01\x00";
    out.write(magic, 8);
    const std::uint16_t len = std::uint16_t(header.size());
    out.write(reinterpret_cast<const char*>(&len), 2);
    out.write(header.data(), std::streamsize(header.size()));
    out.write(reinterpret_cast<const char*>(data.data()),
              std::streamsize(data.size() * sizeof(float)));
}

}  // namespace

const std::string& golden_tap_dir() {
    static const std::string dir = [] {
        const char* e = std::getenv("PIE_METAL_GOLDEN_DIR");
        const std::string d(e == nullptr ? "" : e);
        // Created here, once, rather than left to the caller. `write_npy` opens
        // with `std::ofstream` and RETURNS SILENTLY if the open fails, so a
        // directory that does not exist produced a run that looked like a
        // successful dump and left nothing behind -- and the diff that was
        // supposed to bisect a wrong answer then had no files to compare.
        // Failing to create is not fatal: the run is still a valid benchmark.
        if (!d.empty()) ::mkdir(d.c_str(), 0755);
        return d;
    }();
    return dir;
}

bool golden_taps_recycle() {
    // `PIE_METAL_TAPS_LAYER` implies this: it asks for exactly one layer to be
    // kept out of the recycling, which is pointless if the recycling is off
    // everywhere anyway and fatal on a checkpoint whose no-recycle pool does
    // not fit. See the pinning in `build_scratch_schedule`.
    static const bool on = std::getenv("PIE_METAL_TAPS_RECYCLE") != nullptr ||
                           std::getenv("PIE_METAL_TAPS_LAYER") != nullptr;
    return on;
}

/// The taps a dispatch publishes. Usually one -- the value it computes -- but
/// the sort has four outputs and three of them are what a routed bisect needs
/// to see, so this is a list rather than the single `Tap` it was.
int taps_for(const Dispatch& d, const DecodeGeometry& g, Tap out[4]) {
    int n = 0;
    Tap primary{};
    if (tap_for(d, g, primary)) out[n++] = primary;
    if (d.kind == Kernel::LlMoeSort) {
        out[n++] = {"moe_perm", 1, 1, true, true};  // llama::kMoeSortPermBind
        out[n++] = {"moe_inv", 5, 1, true, true};
        // One entry per TILE, not per row: the count is the batched layout's
        // `sorted / tile`, which only the caller knows.
        out[n++] = {"moe_tile_expert", 3, 1, false, true, -1};
    }
    return n;
}

void dump_golden_taps(const std::vector<Dispatch>& dag,
                      const ScratchSchedule& sched,
                      const SlotHandle* pool,
                      int pool_n,
                      const DecodeGeometry& g,
                      int n_rows,
                      std::size_t row_stride_bytes,
                      const char* prefix,
                      int sorted_rows,
                      int tile_rows) {
    const std::string& dir = golden_tap_dir();
    if (dir.empty() || n_rows <= 0) return;
    const std::size_t n = std::min(dag.size(), sched.per_dispatch.size());

    // q_norm/k_norm, both ropes and attn_gate rewrite their input in place, so
    // under no_recycle they share one buffer with the tap before them and that
    // buffer only ever holds the LAST writer's value. Dumping the earlier name
    // too would publish the later tensor under it and read as a divergence that
    // is really just the dump lying. Only the final writer of a colour is named.
    std::vector<int> last_writer(std::size_t(pool_n), -1);
    for (std::size_t di = 0; di < n; ++di) {
        Tap taps[4]{};
        const int nt = taps_for(dag[di], g, taps);
        for (int ti = 0; ti < nt; ++ti)
            for (const ScratchBind& sb : sched.per_dispatch[di].binds)
                if (sb.bind_index == taps[ti].out_bind && sb.buffer_id < pool_n) {
                    last_writer[std::size_t(sb.buffer_id)] = int(di);
                    break;
                }
    }

    for (std::size_t di = 0; di < n; ++di) {
        Tap taps[4]{};
        const int nt = taps_for(dag[di], g, taps);
        for (int ti = 0; ti < nt; ++ti) {
        const Tap& tap = taps[ti];
        int color = -1;
        for (const ScratchBind& sb : sched.per_dispatch[di].binds)
            if (sb.bind_index == tap.out_bind) { color = sb.buffer_id; break; }
        if (color < 0 || color >= pool_n || !pool[color].valid()) continue;
        if (last_writer[std::size_t(color)] != int(di)) continue;
        const auto* base = static_cast<const std::uint8_t*>(pool[color].contents());
        if (base == nullptr) continue;

        // The sorted stack is packed: one row per (token, slot) pair the sort
        // emitted, contiguous at the tap's own width. The token-major stride
        // the caller passes is the batch's and does not apply to it.
        const int rn = tap.tile_rows < 0 ? tile_rows
                     : tap.sorted        ? sorted_rows
                                         : n_rows;
        const std::size_t elem = tap.i32 ? 4 : 2;
        const std::size_t stride = (tap.sorted || tap.tile_rows < 0)
                                       ? std::size_t(tap.width) * elem
                                       : row_stride_bytes;
        if (rn <= 0) continue;

        std::vector<float> rows(std::size_t(rn) * std::size_t(tap.width));
        for (int t = 0; t < rn; ++t) {
            const auto* src = base + std::size_t(t) * stride;
            for (int i = 0; i < tap.width; ++i) {
                const std::size_t o = std::size_t(t) * std::size_t(tap.width) + std::size_t(i);
                if (tap.i32) {
                    std::int32_t v = 0;
                    std::memcpy(&v, src + std::size_t(i) * 4, 4);
                    rows[o] = float(v);
                } else {
                    std::uint16_t v = 0;
                    std::memcpy(&v, src + std::size_t(i) * 2, 2);
                    rows[o] = bf16_to_f32(v);
                }
            }
        }
        const std::string name = dag[di].layer < 0
            ? std::string(prefix) + tap.name
            : std::string(prefix) + std::to_string(dag[di].layer) + "." + tap.name;
        write_npy(dir + "/" + name + ".npy", rows, rn, tap.width);
        }
    }
}

void dump_golden_bf16(const std::string& name,
                      const void* bf16,
                      int rows,
                      int width,
                      std::size_t row_stride_elems) {
    const std::string& dir = golden_tap_dir();
    if (dir.empty() || bf16 == nullptr || rows <= 0 || width <= 0) return;
    const auto* src = static_cast<const std::uint16_t*>(bf16);
    std::vector<float> out(std::size_t(rows) * std::size_t(width));
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < width; ++i)
            out[std::size_t(r) * std::size_t(width) + std::size_t(i)] =
                bf16_to_f32(src[std::size_t(r) * row_stride_elems + std::size_t(i)]);
    write_npy(dir + "/" + name + ".npy", out, rows, width);
}

void dump_golden_bf16_sorted(const std::string& name,
                             const void* bf16,
                             const std::int32_t* perm,
                             int stored_rows,
                             int rows,
                             int slots,
                             int width) {
    const std::string& dir = golden_tap_dir();
    if (dir.empty() || bf16 == nullptr || perm == nullptr) return;
    if (rows <= 0 || slots <= 0 || width <= 0 || stored_rows <= 0) return;
    const auto* src = static_cast<const std::uint16_t*>(bf16);
    std::vector<float> out(std::size_t(rows) * std::size_t(slots) * std::size_t(width), 0.0f);
    for (int p = 0; p < stored_rows; ++p) {
        const std::int32_t sel = perm[p];
        if (sel < 0 || sel >= rows * slots) continue;
        const std::size_t dst = std::size_t(sel) * std::size_t(width);
        const std::size_t s = std::size_t(p) * std::size_t(width);
        for (int i = 0; i < width; ++i) out[dst + std::size_t(i)] = bf16_to_f32(src[s + std::size_t(i)]);
    }
    write_npy(dir + "/" + name + ".npy", out, rows, slots * width);
}

void dump_golden_tokens(const std::uint32_t* ids, int n) {
    const std::string& dir = golden_tap_dir();
    if (dir.empty() || ids == nullptr || n <= 0) return;
    std::ofstream out(dir + "/tokens.txt");
    for (int i = 0; i < n; ++i) out << (i ? "," : "") << ids[i];
    out << "\n";
}

}  // namespace pie::metal
