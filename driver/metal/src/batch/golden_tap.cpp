// golden_tap.cpp — see golden_tap.hpp.

#include "golden_tap.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
        // The sorted tensors are deliberately untapped: their row order is the
        // driver's own, so a dump of them would diff against nothing. What is
        // here is every value a reference can also produce, which until now was
        // nothing at all -- the routed FFN was the one block of this family no
        // parity run could see, and it is where Qwen3.6-35B-A3B went wrong.
        case Kernel::LlRouter:      out = {"router",     4, g.n_experts}; return true;
        case Kernel::LlMoeCombine:  out = {"moe_out",    2, g.hidden};   return true;
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
        return std::string(e == nullptr ? "" : e);
    }();
    return dir;
}

bool golden_taps_recycle() {
    static const bool on = std::getenv("PIE_METAL_TAPS_RECYCLE") != nullptr;
    return on;
}

void dump_golden_taps(const std::vector<Dispatch>& dag,
                      const ScratchSchedule& sched,
                      const SlotHandle* pool,
                      int pool_n,
                      const DecodeGeometry& g,
                      int n_rows,
                      std::size_t row_stride_bytes) {
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
        Tap tap{};
        if (!tap_for(dag[di], g, tap)) continue;
        for (const ScratchBind& sb : sched.per_dispatch[di].binds)
            if (sb.bind_index == tap.out_bind && sb.buffer_id < pool_n) {
                last_writer[std::size_t(sb.buffer_id)] = int(di);
                break;
            }
    }

    for (std::size_t di = 0; di < n; ++di) {
        Tap tap{};
        if (!tap_for(dag[di], g, tap)) continue;
        int color = -1;
        for (const ScratchBind& sb : sched.per_dispatch[di].binds)
            if (sb.bind_index == tap.out_bind) { color = sb.buffer_id; break; }
        if (color < 0 || color >= pool_n || !pool[color].valid()) continue;
        if (last_writer[std::size_t(color)] != int(di)) continue;
        const auto* base = static_cast<const std::uint8_t*>(pool[color].contents());
        if (base == nullptr) continue;

        std::vector<float> rows(std::size_t(n_rows) * std::size_t(tap.width));
        for (int t = 0; t < n_rows; ++t) {
            const auto* src = reinterpret_cast<const std::uint16_t*>(
                base + std::size_t(t) * row_stride_bytes);
            for (int i = 0; i < tap.width; ++i)
                rows[std::size_t(t) * std::size_t(tap.width) + std::size_t(i)] =
                    bf16_to_f32(src[i]);
        }
        const std::string name = dag[di].layer < 0
            ? std::string(tap.name)
            : std::to_string(dag[di].layer) + "." + tap.name;
        write_npy(dir + "/" + name + ".npy", rows, n_rows, tap.width);
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
