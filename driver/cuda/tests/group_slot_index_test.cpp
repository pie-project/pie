// What the slot index owes, on a machine with no GPU.
//
// The whole point of splitting `GroupSlotIndex` out of the cache is that the
// eviction rule is the part that decides whether streaming works, and it needs
// no device to check. A slab that cannot hold the instances one batch touches
// thrashes; a slab that can must not evict anything at all. Both of those are
// statements about this class alone.

#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

#include <pie_loader/group_slot_index.hpp>

namespace {

int failures = 0;

void check(bool ok, std::string_view what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

using pie_loader::GroupSlotIndex;

// An unfilled slot is not "least recently used at time zero" -- it is empty,
// and filling it costs nothing. Every free slot has to be spent before the
// first eviction, or a cache sized for the working set would still evict.
void test_free_slots_go_first() {
    GroupSlotIndex idx(64, 4);
    for (std::uint32_t i = 0; i < 4; ++i) {
        const auto got = idx.acquire(i);
        check(got.slot == i, "free slots are handed out in order");
        check(!got.evicted, "taking a free slot evicts nothing");
    }
    check(idx.evictions() == 0, "no evictions while slots remain free");
    idx.unpin_all();

    const auto fifth = idx.acquire(4);
    check(fifth.evicted, "the fifth instance must evict");
    check(idx.evictions() == 1, "one eviction");
}

// A hit has to age its slot, or the batch that keeps re-reading one instance
// would eventually evict it. This is the difference between LRU and FIFO and
// it only shows up when a hit is followed by a miss.
void test_a_hit_is_not_a_victim() {
    GroupSlotIndex idx(64, 2);
    idx.acquire(0);
    idx.acquire(1);
    idx.unpin_all();

    // Touch 0 so 1 becomes the oldest.
    const auto hit = idx.find(0);
    check(hit == 0, "instance 0 is still where it was put");
    idx.touch_and_pin(static_cast<std::uint32_t>(hit));
    idx.unpin_all();

    idx.acquire(2);
    check(idx.find(0) >= 0, "the touched instance survived");
    check(idx.find(1) == GroupSlotIndex::kAbsent, "the untouched one was evicted");
}

// Eviction has to unmap the instance that left, not just overwrite the slot.
// Otherwise `find` keeps returning a slot that now holds something else, which
// is the one failure mode that produces silently wrong weights rather than an
// error.
void test_eviction_unmaps_the_instance_that_left() {
    GroupSlotIndex idx(8, 1);
    idx.acquire(3);
    idx.unpin_all();
    check(idx.find(3) == 0, "3 is in the only slot");

    idx.acquire(5);
    check(idx.find(3) == GroupSlotIndex::kAbsent, "3 is gone");
    check(idx.find(5) == 0, "5 took the slot");
}

// Within one batch nothing already used may be evicted, because a kernel may
// still be reading it. A batch that wants more than the cache holds is a
// configuration error, and saying so beats corrupting a live slot.
void test_a_batch_cannot_evict_itself() {
    GroupSlotIndex idx(64, 2);
    idx.acquire(0);
    idx.acquire(1);
    bool threw = false;
    try {
        idx.acquire(2);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "a batch wider than the cache is refused, not served wrong");

    idx.unpin_all();
    bool ok = true;
    try {
        idx.acquire(2);
    } catch (const std::exception&) {
        ok = false;
    }
    check(ok, "the next batch may evict what the last one pinned");
}

// The residency case: a slab with a slot per instance never evicts, whatever
// the access order. This is the property that lets the same code path serve a
// driver that has never heard of streaming.
void test_a_slab_that_fits_never_evicts() {
    const std::uint32_t arity = 16;
    GroupSlotIndex idx(arity, arity);
    // A deliberately awkward order, repeated.
    const std::vector<std::uint32_t> order = {7, 0, 15, 7, 3, 0, 11, 15, 1, 2};
    for (int round = 0; round < 8; ++round) {
        for (const auto instance : order) {
            const auto slot = idx.find(instance);
            if (slot == GroupSlotIndex::kAbsent) {
                idx.acquire(instance);
            } else {
                idx.touch_and_pin(static_cast<std::uint32_t>(slot));
            }
        }
        idx.unpin_all();
    }
    check(idx.evictions() == 0, "a slab sized to the arity never evicts");
}

// An instance outside the arity is a contract violation, not a wrap-around.
void test_an_instance_outside_the_arity_is_refused() {
    GroupSlotIndex idx(4, 2);
    bool threw = false;
    try {
        idx.find(4);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    check(threw, "instance == arity is out of range");

    threw = false;
    try {
        GroupSlotIndex(0, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a group with no instances is refused");
}

}  // namespace

int main() {
    test_free_slots_go_first();
    test_a_hit_is_not_a_victim();
    test_eviction_unmaps_the_instance_that_left();
    test_a_batch_cannot_evict_itself();
    test_a_slab_that_fits_never_evicts();
    test_an_instance_outside_the_arity_is_refused();

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "group_slot_index_test: OK\n";
    return 0;
}
