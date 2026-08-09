#pragma once

/// The entrypoint names, built here and nowhere else.
///
/// An MSL entrypoint is generated: a `.metal` file holds a template body and a
/// macro that stamps it over an axis product, so the name a pipeline is built
/// from is `<base><axis suffix>`. The driver used to assemble that with `+`,
/// at some forty sites, and nothing checked the result — a name no shader
/// instantiates is a PSO compile failure at model load, and `AffineFormat`
/// below records the worse case, where the name resolves and the numbers are
/// wrong.
///
/// So the grammar lives on this side of the boundary, with the shaders that
/// define it, and every name is checked against
/// `entrypoints.generated.h` — the set `scripts/metal-kernel-audit.py`
/// reads out of the shader tree. A checkpoint whose format has no
/// instantiation now fails by NAME, at the call, saying what is compiled.
///
/// `.wiki/kernel-metal-refactor.md` §6 invariant (2) is this file.

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>

#include "quant/affine_format.hpp"
#include "pie/kernels/entrypoints.generated.h"

namespace pie::kernels {

/// Whether any shader under this directory instantiates `name`.
///
/// A binary search over the generated table, which is sorted and `constexpr`.
/// Called once per PSO at setup, so the cost is nothing; the reason it is not
/// a hash is that the sorted form is also what makes `compiled_for` cheap.
inline bool is_entrypoint(std::string_view name) {
    return std::binary_search(std::begin(kEntrypoints), std::end(kEntrypoints), name);
}

/// Every instantiation of `base`, for a failure that says what IS available.
///
/// The generated table is sorted, so a base's instantiations are contiguous
/// and this is two bounds rather than a scan.
inline std::string compiled_for(std::string_view base) {
    const auto* first = std::lower_bound(std::begin(kEntrypoints),
                                         std::end(kEntrypoints), base);
    std::string out;
    for (const auto* at = first; at != std::end(kEntrypoints); ++at) {
        if (at->size() < base.size() || at->substr(0, base.size()) != base) break;
        // A longer base that shares this prefix is a different kernel
        // (`affine_qmm_t` and `affine_qmm_t_bias`), and its instantiations are
        // not answers to this question.
        const std::string_view tail = at->substr(base.size());
        if (!tail.empty() && tail.find("_bfloat16") != 0 && tail.find("_f32") != 0) {
            continue;
        }
        if (!out.empty()) out += ", ";
        out += tail.empty() ? "<no suffix>" : std::string(tail);
    }
    return out.empty() ? std::string("nothing") : out;
}

// ── the axis suffixes, spelled once each ────────────────────────────────────
//
// One function per axis rather than one per kernel: the axes are the same
// handful everywhere, and a kernel is a base plus which of them it carries.
// The table in `crates/kernels-metal/src/` states the same thing for the same
// reason.

/// The activation dtype. One point today, and `AffineFormat` explains why it
/// is still spelled rather than assumed.
inline std::string bf16() { return "_bfloat16"; }

/// The activation dtype and the affine format together, which is how every
/// quantised entrypoint spells them.
inline std::string affine(const pie::metal::AffineFormat& q) {
    return q.kernel_suffix();
}

/// The head width an attention kernel is instantiated at.
inline std::string head_dim(int d) { return "_d_" + std::to_string(d); }

/// A GEMM's row and column tile.
inline std::string tile(int bm, int bn) {
    return "_bm_" + std::to_string(bm) + "_bn_" + std::to_string(bn);
}

/// A GEMM's row tile alone, for the strided forms whose column tile is fixed.
inline std::string tile_rows(int bm) { return "_bm_" + std::to_string(bm); }

/// The chunk length a recurrent scan is unrolled for.
inline std::string chunk(int l) { return "_l_" + std::to_string(l); }

/// The rows a kernel handles per lane. Spelled `_v_` because it is the value
/// width on the wide matvec and the row count on the GDN scan, and the two
/// share the token rather than the meaning.
inline std::string rows(int v) { return "_v_" + std::to_string(v); }

/// The fixed 32-entry page table, or nothing. An axis with a point that adds
/// no text, which is what `sdpa_paged_decode` compiles: one template at
/// `<..., 0, false, 32>` and `<..., 32, true, 32>`.
inline std::string page32(bool fixed) { return fixed ? "_p32" : ""; }

/// The k-loop unroll of the wide strided matvec.
inline std::string k_unroll(int kl) { return "_kl_" + std::to_string(kl); }

/// The entrypoint `base` names at this instantiation.
///
/// Throws rather than returning an error, and that is the right shape here:
/// every caller is PSO setup at model load, none of them has a fallback, and
/// the alternative — handing an unbuildable name to the Metal compiler — is
/// what this exists to stop. The message carries the base, what was asked for,
/// and what the shaders hold, because the three together are the whole
/// diagnosis.
inline std::string entrypoint(std::string_view base,
                              std::initializer_list<std::string> axes) {
    std::string name(base);
    for (const std::string& axis : axes) name += axis;
    if (!is_entrypoint(name)) {
        throw std::runtime_error(
            "no Metal kernel `" + std::string(base) + "` is compiled for `" +
            name.substr(base.size()) + "` (asked for `" + name +
            "`). Compiled: " + compiled_for(base) +
            ". Add the instantiation to the .metal, regenerate with "
            "scripts/metal-kernel-audit.py --write, and give it an axis point "
            "in crates/kernels-metal/src/.");
    }
    return name;
}

/// A kernel with no axes at all: the name IS the entrypoint. Checked for the
/// same reason — a typo in a literal is the same failure as a bad product.
inline std::string entrypoint(std::string_view base) { return entrypoint(base, {}); }

}  // namespace pie::kernels
