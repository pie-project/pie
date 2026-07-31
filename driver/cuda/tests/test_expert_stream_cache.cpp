// Host-only unit tests for the SSD expert-streaming bookkeeping:
// ExpertSlotIndex (LRU slot map). Extent tables now come from the Rust
// stream plan (see weight_loader storage_compiler / stream tests); the
// GPU cache path is covered by test_expert_stream_cache_gpu.

#include "expert_stream_cache.hpp"

#include <iostream>
#include <stdexcept>

namespace {

using pie_cuda_driver::ExpertSlotIndex;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::cerr << "FAILED at " << __FILE__ << ":" << __LINE__     \
                      << ": " #cond "\n";                                \
            std::exit(1);                                                \
        }                                                                \
    } while (0)

template <typename Fn>
bool throws(Fn&& fn)
{
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void test_slot_index_fills_free_slots_first()
{
    ExpertSlotIndex idx(/*num_layers=*/2, /*num_experts=*/4, /*num_slots=*/3);
    CHECK(idx.find(0, 0) == -1);

    const auto a = idx.acquire(0, 0);
    const auto b = idx.acquire(0, 1);
    const auto c = idx.acquire(1, 2);
    CHECK(!a.evicted && !b.evicted && !c.evicted);
    CHECK(a.slot != b.slot && b.slot != c.slot && a.slot != c.slot);
    CHECK(idx.find(0, 0) == a.slot);
    CHECK(idx.find(0, 1) == b.slot);
    CHECK(idx.find(1, 2) == c.slot);
    CHECK(idx.evictions() == 0);
}

void test_slot_index_evicts_lru()
{
    ExpertSlotIndex idx(1, 8, 2);
    const auto a = idx.acquire(0, 0);  // age 1
    const auto b = idx.acquire(0, 1);  // age 2
    idx.unpin_all();

    idx.touch_and_pin(a.slot);  // expert 0 now newest; expert 1 is LRU
    idx.unpin_all();

    const auto c = idx.acquire(0, 2);
    CHECK(c.evicted);
    CHECK(c.slot == b.slot);      // expert 1's slot reused
    CHECK(idx.find(0, 1) == -1);  // old mapping gone
    CHECK(idx.find(0, 0) == a.slot);
    CHECK(idx.find(0, 2) == c.slot);
    CHECK(idx.evictions() == 1);
}

void test_slot_index_pins_protect_current_batch()
{
    ExpertSlotIndex idx(1, 8, 2);
    const auto a = idx.acquire(0, 0);
    const auto b = idx.acquire(0, 1);
    // Both pinned: nothing can be evicted.
    CHECK(throws([&] { idx.acquire(0, 2); }));

    idx.unpin_all();
    idx.touch_and_pin(b.slot);  // pin only b; a is the sole candidate
    const auto c = idx.acquire(0, 2);
    CHECK(c.slot == a.slot);
    CHECK(idx.find(0, 0) == -1);
    CHECK(idx.find(0, 1) == b.slot);
}

void test_slot_index_rejects_out_of_range()
{
    ExpertSlotIndex idx(2, 4, 2);
    CHECK(throws([&] { (void)idx.find(2, 0); }));
    CHECK(throws([&] { (void)idx.find(0, 4); }));
    CHECK(throws([&] { (void)idx.acquire(-1, 0); }));
}

}  // namespace

int main()
{
    test_slot_index_fills_free_slots_first();
    test_slot_index_evicts_lru();
    test_slot_index_pins_protect_current_batch();
    test_slot_index_rejects_out_of_range();
    std::cout << "expert_stream_cache: all tests passed\n";
    return 0;
}
