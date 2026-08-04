// The expert slab: bounded slots, filled from the checkpoint on demand.
//
// This is the half of expert paging that decides whether a streamed model is
// CORRECT, as opposed to whether it is fast, and it is testable with no device
// because the slabs are pointers and strides. What it has to get right:
//
//   * the band arithmetic -- slot `s` must hold expert `e`'s bytes and not its
//     neighbour's, which is the failure that would read as a quality
//     regression rather than a crash;
//   * eviction -- a re-paged expert must come back byte-identical, because a
//     slab that returns a stale slot is a model that answers differently
//     depending on how many experts happened to fit;
//   * the layer axis -- two layers with the same expert index are two
//     different experts, and a collision there would silently serve layer 0's
//     weights to layer 7;
//   * the parallel bands -- ONE slot number has to mean the same expert for
//     `gate_proj` and for `down_proj` at once, because the kernels index all
//     of them with a single `expert_ids` buffer. That is the claim a
//     one-tensor-per-cache design cannot make, and it is checked here on
//     tensors with DIFFERENT band sizes so that a shared stride could not
//     accidentally satisfy it.
//
// The bytes are made distinguishable per (tensor, layer, expert) so that any
// of those failures produces a wrong byte rather than a plausible one.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "loader/expert_slab.hpp"

using pie::metal::ExpertSlab;
using pie::metal::SlabTensor;

namespace {
int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}

constexpr std::uint32_t kLayers = 3;
constexpr std::uint32_t kExperts = 4;
// Two kinds on purpose, and deliberately unequal: a slot has to mean one
// expert across both, which a shared stride would make untestable.
constexpr std::uint64_t kBand[2] = {64, 96};
constexpr std::size_t kKinds = 2;

/// A byte that names its own (kind, layer, expert, position).
std::uint8_t byte_of(std::size_t t, std::uint32_t layer, std::uint32_t expert, std::uint64_t i) {
    return static_cast<std::uint8_t>((t * 101u + layer * 37u + expert * 11u + i) & 0xFFu);
}

int finish(const char* name) {
    std::printf("\n==== %s: %d passed, %d failed ====\n", name, g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
}  // namespace

int main() {
    std::printf("[expert slab: bounded slots paged from a checkpoint]\n");

    // Two "files", one per tensor kind: every layer's experts back to back,
    // exactly as a routed bank sits in a converted checkpoint.
    std::vector<std::uint8_t> file[kKinds];
    std::vector<SlabTensor> tensors;
    for (std::size_t t = 0; t < kKinds; ++t) {
        file[t].resize(std::size_t(kLayers) * kExperts * kBand[t]);
        SlabTensor s;
        s.suffix = t == 0 ? "mlp.experts.gate_proj" : "mlp.experts.down_proj";
        s.band_bytes = kBand[t];
        for (std::uint32_t l = 0; l < kLayers; ++l) {
            const std::uint64_t base = std::uint64_t(l) * kExperts * kBand[t];
            for (std::uint32_t e = 0; e < kExperts; ++e) {
                for (std::uint64_t i = 0; i < kBand[t]; ++i) {
                    file[t][std::size_t(base + e * kBand[t] + i)] = byte_of(t, l, e, i);
                }
            }
            s.layer_base.push_back(file[t].data() + base);
        }
        tensors.push_back(std::move(s));
    }

    const auto holds = [&](const ExpertSlab& slab, std::uint32_t slot,
                           std::uint32_t layer, std::uint32_t expert) {
        for (std::size_t t = 0; t < kKinds; ++t) {
            const std::uint8_t* got = slab.slot_data(t, slot);
            for (std::uint64_t i = 0; i < kBand[t]; ++i) {
                if (got[i] != byte_of(t, layer, expert, i)) return false;
            }
        }
        return true;
    };

    // Two slots against twelve experts: everything interesting is an eviction.
    std::vector<std::uint8_t> mem[kKinds];
    std::vector<std::uint8_t*> bases;
    for (std::size_t t = 0; t < kKinds; ++t) {
        mem[t].assign(std::size_t(2 * kBand[t]), 0u);
        bases.push_back(mem[t].data());
    }
    ExpertSlab slab(tensors, kExperts, bases, 2);

    expect(slab.num_slots() == 2, "the slab holds the two slots it was budgeted");
    expect(!slab.fully_resident(),
           "two slots against twelve experts is not fully resident");
    expect(slab.slot_bytes() == kBand[0] + kBand[1],
           "a slot costs every routed tensor of one expert, not one of them");

    // The claim a per-tensor cache cannot make: ONE slot number, both tensors,
    // and the tensors have different band sizes so no shared stride can be
    // hiding the answer.
    slab.end_batch();
    const std::uint32_t s0 = slab.ensure_resident(1, 2);
    expect(holds(slab, s0, 1, 2),
           "one slot number means the same expert for BOTH routed tensors");

    // Same expert index in three different layers. If the layer axis collapsed,
    // these would share a slot and two of the three would read another layer.
    bool distinct = true;
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        slab.end_batch();  // a fresh batch, or the third acquire has no victim
        const std::uint32_t s = slab.ensure_resident(l, 2);
        if (!holds(slab, s, l, 2)) distinct = false;
    }
    expect(distinct, "expert 2 of each layer pages in as ITS OWN bytes");

    // A hit does not copy.
    slab.end_batch();
    const std::uint64_t before = slab.misses();
    const std::uint32_t again = slab.ensure_resident(kLayers - 1, 2);
    expect(slab.misses() == before && slab.hits() > 0,
           "an expert already in a slot is a hit, not a second copy");
    expect(holds(slab, again, kLayers - 1, 2), "and the hit's bytes are still right");

    // Eviction and return. Fill both slots, push both out, then ask for the
    // first one back and require it byte-identical.
    slab.end_batch();
    slab.ensure_resident(0, 0);
    slab.ensure_resident(0, 1);
    slab.end_batch();
    slab.ensure_resident(0, 2);
    slab.ensure_resident(0, 3);
    slab.end_batch();
    const std::uint32_t revived = slab.ensure_resident(0, 0);
    expect(holds(slab, revived, 0, 0),
           "an evicted expert comes back byte-identical, not stale");

    expect(slab.slab_bytes() == 2 * (kBand[0] + kBand[1]),
           "paging every expert through two slots did not grow the slab");

    // More experts at once than there are slots is a configuration error, not
    // a silent overwrite of a slot a kernel may still be reading.
    slab.end_batch();
    slab.ensure_resident(1, 0);
    slab.ensure_resident(1, 1);
    bool threw = false;
    try {
        slab.ensure_resident(1, 2);
    } catch (const std::exception&) {
        threw = true;
    }
    expect(threw, "wanting a third expert while two are pinned refuses rather than corrupts");

    // A budget that holds everything reports it, so the caller can skip the
    // routing readback that streaming otherwise pays for every layer.
    std::vector<std::uint8_t> big[kKinds];
    std::vector<std::uint8_t*> big_bases;
    for (std::size_t t = 0; t < kKinds; ++t) {
        big[t].assign(std::size_t(kLayers) * kExperts * kBand[t], 0u);
        big_bases.push_back(big[t].data());
    }
    ExpertSlab whole(tensors, kExperts, big_bases, kLayers * kExperts);
    expect(whole.fully_resident(), "a slab budgeted for every expert says so");

    // A tensor that does not span every mixture layer would make (layer,
    // expert) mean a different expert for it than for its siblings.
    std::vector<SlabTensor> ragged = tensors;
    ragged[1].layer_base.pop_back();
    bool rejected = false;
    try {
        ExpertSlab bad(ragged, kExperts, bases, 2);
    } catch (const std::exception&) {
        rejected = true;
    }
    expect(rejected, "a routed tensor missing a layer is refused");

    return finish("expert_slab_test");
}
