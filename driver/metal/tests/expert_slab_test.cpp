// The expert slab: bounded slots, filled from the checkpoint on demand.
//
// This is the half of expert paging that decides whether a streamed model is
// CORRECT, as opposed to whether it is fast, and it is testable with no device
// because the slab is a pointer and a stride. What it has to get right:
//
//   * the band arithmetic -- slot `s` must hold expert `e`'s bytes and not its
//     neighbour's, which is the failure that would read as a quality
//     regression rather than a crash;
//   * eviction -- a re-paged expert must come back byte-identical, because a
//     slab that returns a stale slot is a model that answers differently
//     depending on how many experts happened to fit;
//   * the flattening -- two banks with the same expert index are two different
//     experts, and a collision there would silently serve layer 0's weights to
//     layer 7.
//
// The bytes are made distinguishable per (bank, expert) so that any of those
// three failures produces a wrong byte rather than a plausible one.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "loader/expert_slab.hpp"

using pie::metal::ExpertBank;
using pie::metal::ExpertSlab;

namespace {
int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}

constexpr std::uint32_t kBanks = 3;
constexpr std::uint32_t kArity = 4;
constexpr std::uint64_t kBand = 64;

/// A byte that names its own (bank, expert, position), so a misread is visible.
std::uint8_t byte_of(std::uint32_t bank, std::uint32_t expert, std::uint64_t i) {
    return static_cast<std::uint8_t>((bank * 37u + expert * 11u + i) & 0xFFu);
}

bool slot_holds(const ExpertSlab& slab, std::uint32_t slot,
                std::uint32_t bank, std::uint32_t expert) {
    const std::uint8_t* got = slab.slot_data(slot);
    for (std::uint64_t i = 0; i < kBand; ++i) {
        if (got[i] != byte_of(bank, expert, i)) return false;
    }
    return true;
}

int finish(const char* name) {
    std::printf("\n==== %s: %d passed, %d failed ====\n", name, g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
}  // namespace

int main() {
    std::printf("[expert slab: bounded slots paged from a checkpoint]\n");

    // One "file": every bank's bands laid out back to back, each bank starting
    // at its own offset, exactly as the banks sit in a converted checkpoint.
    std::vector<std::uint8_t> file(std::size_t(kBanks) * kArity * kBand);
    std::vector<ExpertBank> banks;
    for (std::uint32_t b = 0; b < kBanks; ++b) {
        const std::uint64_t base = std::uint64_t(b) * kArity * kBand;
        for (std::uint32_t e = 0; e < kArity; ++e) {
            for (std::uint64_t i = 0; i < kBand; ++i) {
                file[std::size_t(base + e * kBand + i)] = byte_of(b, e, i);
            }
        }
        banks.push_back(ExpertBank{file.data(), base, kBand, kArity,
                                   "layer" + std::to_string(b)});
    }

    // Two slots against twelve experts: everything interesting is an eviction.
    std::vector<std::uint8_t> slab_mem(2 * kBand, 0u);
    ExpertSlab slab(banks, slab_mem.data(), 2);

    expect(slab.num_slots() == 2, "the slab holds the two slots it was budgeted");
    expect(!slab.fully_resident(),
           "two slots against twelve experts is not fully resident");
    expect(slab.slab_bytes() == 2 * kBand, "the slab is slots x band");

    // Same expert index in three different banks. If the flattening collided,
    // these would share a slot and two of the three checks would read another
    // layer's bytes.
    bool distinct = true;
    std::vector<std::uint32_t> slots;
    for (std::uint32_t b = 0; b < kBanks; ++b) {
        slab.end_batch();  // a fresh batch, or the third acquire has no victim
        const std::uint32_t s = slab.ensure_resident(b, 2);
        slots.push_back(s);
        if (!slot_holds(slab, s, b, 2)) distinct = false;
    }
    expect(distinct, "expert 2 of each bank pages in as ITS OWN bytes");

    // A hit does not copy: ask for something already there and watch the
    // page-in count stand still.
    slab.end_batch();
    const std::uint64_t before = slab.misses();
    const std::uint32_t again = slab.ensure_resident(kBanks - 1, 2);
    expect(slab.misses() == before && slab.hits() > 0,
           "an expert already in a slot is a hit, not a second copy");
    expect(slot_holds(slab, again, kBanks - 1, 2), "and the hit's bytes are still right");

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
    expect(slot_holds(slab, revived, 0, 0),
           "an evicted expert comes back byte-identical, not stale");

    // The slab is bounded in the way that matters: paging twelve experts
    // through two slots never grew it.
    expect(slab.slab_bytes() == 2 * kBand,
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
    std::vector<std::uint8_t> big(std::size_t(kBanks) * kArity * kBand, 0u);
    ExpertSlab whole(banks, big.data(), kBanks * kArity);
    expect(whole.fully_resident(),
           "a slab budgeted for every expert says so");

    // A bank whose bands are a different size cannot share a slot stride, and
    // saying so beats serving half an expert.
    std::vector<ExpertBank> mismatched = banks;
    mismatched[1].band_bytes = kBand / 2;
    bool rejected = false;
    try {
        ExpertSlab bad(mismatched, slab_mem.data(), 2);
    } catch (const std::exception&) {
        rejected = true;
    }
    expect(rejected, "banks that disagree on band size are refused");

    return finish("expert_slab_test");
}
