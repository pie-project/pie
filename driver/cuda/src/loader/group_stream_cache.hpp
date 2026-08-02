#pragma once

// A bounded slab of group instances, paged in on demand.
//
// It holds *several* groups, not one, and that is forced by the contract
// vocabulary rather than chosen: a name template carries one `{}`, so a
// checkpoint that names its experts `layers.<L>.experts.<E>.w1` is L groups of
// arity E, not one group of arity L*E. The slab has to span them anyway --
// budgeting per layer would size every layer for its worst step -- so the key
// here is (group, instance) flattened, and every group must agree on the slot
// stride, which for a real architecture they do because every layer's experts
// have the same shape.
//
// The loader compiles a group to one plan plus `arity` source bindings and
// says nothing about residency, because running the plan `arity` times into
// `arity` destinations and running it on demand into a few slots are the same
// program. This is the second reading. `GroupSlotIndex` decides which instance
// goes where; this moves the bytes and hands back the tensors.
//
// Two facts from the loader's uniformity proof do all the work here:
//
//   * every instance's plan has the same `memory.persistent_bytes`, so one
//     slot stride serves all of them, and the slab is just `slots × stride`;
//   * every instance's buffers land at the same offsets within that arena, so
//     a slot's tensor pointers are fixed at slot creation and a page-in
//     changes only the bytes behind them. A caller that captured a pointer
//     while its instance was resident is safe for exactly as long as the pin
//     lasts, and no longer -- which is what `end_batch` marks.
//
// Because a slot's plan is a whole `LoadPlanView`, the executor that runs the
// resident load runs it verbatim: transforms, tile maps and casts all work on
// the page-in path, and nothing here needs to know what the group contains.
//
// Single-threaded, like the forward pass that calls it. `ensure_resident`
// blocks until the instance is there.

#include <chrono>
#include <ostream>
#include <iostream>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Unconditionally CUDA, unlike the executor it wraps: a slab is device memory
// and there is no host-side reading of this component to preserve. The part
// that is worth testing without a device is `GroupSlotIndex`, which is why it
// is a separate header, and why that header is the loader's rather than this
// driver's -- Metal pages the same groups by the same rule.
#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "pie_loader/plan.hpp"
#include "pie_loader/checkpoint_source.hpp"

#include <pie_loader/group_slot_index.hpp>
#include "loader/load_plan_executor.hpp"
#include "loader/loader_config.hpp"
#include "loader/weight_copy_engine.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

// Which instance is in which slot is not a CUDA question, and Metal asks it
// too. The rule lives in the loader so both backends evict the same way; only
// the half that moves bytes is below.
using pie_loader::GroupSlotIndex;

class GroupStreamCache {
public:
    /// `budget_bytes` bounds the slab; the slot count is clamped to
    /// `[1, arity]`, so a budget large enough for the whole group degenerates
    /// to residency with no page-ins after the first sweep. Throws if the
    /// budget cannot hold one slot.
    GroupStreamCache(
        pie_loader::CheckpointSource& loader,
        pie_loader::PieLoaderGroupSlice groups,
        std::uint64_t budget_bytes,
        std::uint64_t host_budget_bytes = 0,
        bool verbose = false)
        : loader_(loader), groups_(groups), verbose_(verbose)
    {
        const std::uint64_t total_instances = scan_groups();
        if (budget_bytes < slot_bytes_) {
            throw std::runtime_error(
                "group stream cache: one instance needs " +
                std::to_string(slot_bytes_) + " bytes but the budget is " +
                std::to_string(budget_bytes));
        }
        std::uint64_t slots = budget_bytes / slot_bytes_;
        if (slots > total_instances) slots = total_instances;
        index_ = GroupSlotIndex(
            static_cast<std::uint32_t>(total_instances),
            static_cast<std::uint32_t>(slots));
        slot_stores_.resize(slots);
        slot_filled_.assign(slots, false);
        // A page-in is one slot, and how big that is decides which engine
        // is the right one. The default engine spreads a copy over a pool of
        // streams and stages it through host reader lanes -- machinery that
        // buys throughput on a whole model and is pure setup on a few hundred
        // kilobytes, which is where the 175-to-18-microsecond result came
        // from. But a GPT-OSS slot is 12 MiB, not 192 KiB, and there that same
        // machinery is what the transfer needs. So pick by measurement rather
        // than by family: below a lane's staging buffer there is nothing for a
        // lane to do.
        constexpr std::uint64_t kSmallSlot = 2u << 20;
        if (slot_bytes_ < kSmallSlot) {
            copy_engine_.prefer_small_transfers();
        }
        // A knob because prefetching is only visibly worth anything against a
        // cold page cache, and telling whether it helped means running the
        // same thing without it.
        prefetch_enabled_ =
            !loader_config::env_present("PIE_CUDA_STREAM_NO_PREFETCH");
        // Read once, not per access: `verify_fill` is called on every hit as
        // well as every miss, and a `getenv` there is a scan of the
        // environment inside the forward pass.
        verify_fills_ =
            loader_config::env_truthy("PIE_CUDA_STREAM_VERIFY");
        CUDA_CHECK(cudaEventCreateWithFlags(&transform_done_,
                                            cudaEventDisableTiming));
        allocate_slab();
        // Everything past the allocation can throw -- a missing binding, a
        // layout that moved, any CUDA_CHECK. The destructor does not run for
        // an object whose constructor did not finish, so without this a failed
        // load leaks the whole slab and the pinned tier.
        try {
            if (fully_resident()) {
                // The budget can hold the group outright, so there is nothing
                // to decide at runtime and no reason to pay for deciding. Fill
                // every slot now and the forward never calls back into the
                // host: no routed-set readback, no pointer rebuild, no
                // eviction. A host tier would only be a second copy of what
                // the slab already holds, so it is neither allocated nor
                // warmed.
                place_all();
            } else {
                allocate_host_tier(host_budget_bytes, total_instances);
                warm_host_tier();
            }
        } catch (...) {
            free_slab();
            free_host_tier();
            if (transform_done_ != nullptr) cudaEventDestroy(transform_done_);
            throw;
        }
    }

    ~GroupStreamCache() {
        // Also on a bare env var, not only under the boot config's verbose:
        // measuring paging is a thing one does to an otherwise ordinary run,
        // and the rest of verbose is a wall of load-time noise.
        if (verbose_ || loader_config::env_truthy("PIE_CUDA_STREAM_STATS")) {
            report(std::cerr);
        }
        free_slab();
        free_host_tier();
        if (transform_done_ != nullptr) cudaEventDestroy(transform_done_);
    }

    /// What paging actually cost, in the terms the next decision needs.
    ///
    /// Hit rate says whether the slab is big enough; ns/miss and MiB/s say
    /// whether the source is fast enough. They point at different fixes -- a
    /// larger slab versus a faster tier or a prefetch -- and a run that is
    /// slow for the second reason will not improve by any amount of the first.
    void report(std::ostream& out) const {
        const Stats s = stats();
        const std::uint64_t accesses = s.hits + s.misses;
        if (accesses == 0) return;
        // Warm-up held out of both the rate and the per-miss cost.
        const std::uint64_t steady_ns =
            s.page_in_ns > s.first_page_in_ns ? s.page_in_ns - s.first_page_in_ns : 0;
        const std::uint64_t steady_misses = s.misses > 1 ? s.misses - 1 : 0;
        const std::uint64_t steady_bytes =
            s.bytes_paged_in > s.first_page_in_bytes
                ? s.bytes_paged_in - s.first_page_in_bytes
                : 0;
        const double page_in_ms = static_cast<double>(steady_ns) / 1e6;
        const double secs = static_cast<double>(steady_ns) / 1e9;
        out << "[pie-driver-cuda] group stream cache: " << accesses
            << " accesses, " << (100.0 * static_cast<double>(s.hits) /
                                 static_cast<double>(accesses))
            << "% hit, " << s.misses << " page-ins ("
            << (static_cast<double>(s.first_page_in_ns) / 1e6)
            << " ms of warm-up held out), "
            << (steady_misses == 0
                    ? 0.0
                    : static_cast<double>(steady_ns) / 1e3 /
                          static_cast<double>(steady_misses))
            << " us each, of "
            << (steady_misses == 0 ? 0.0
                                   : static_cast<double>(steady_bytes) /
                                         static_cast<double>(steady_misses) / 1048576.0)
            << " MiB in " << page_in_ms << " ms ("
            << (secs == 0.0 ? 0.0
                            : static_cast<double>(steady_bytes) / 1048576.0 / secs)
            << " MiB/s), " << (static_cast<double>(s.evict_wait_ns) / 1e6)
            << " ms waiting to evict, " << s.prefetches
            << " prefetched, "
            << s.host_tier_hits << " from host tier of "
            << host_slots_ << " slots; of the page-in, alloc " << s.alloc_ms
            << " ms, transfer " << s.transfer_ms << " ms, transform "
            << s.transform_ms << " ms\n";
    }

    GroupStreamCache(const GroupStreamCache&) = delete;
    GroupStreamCache& operator=(const GroupStreamCache&) = delete;

    /// The group named `name`, or `kNoGroup`. A bind path holds names, not
    /// indices, and resolving once at bind keeps the forward pass indexing.
    static constexpr std::size_t kNoGroup = static_cast<std::size_t>(-1);
    std::size_t find_group(std::string_view name) const noexcept {
        for (std::size_t g = 0; g < groups_.len; ++g) {
            const auto& group = groups_.ptr[g];
            if (std::string_view(
                    reinterpret_cast<const char*>(group.name.ptr),
                    group.name.len) == name) {
                return g;
            }
        }
        return kNoGroup;
    }
    std::uint32_t arity(std::size_t group) const {
        if (group >= groups_.len) {
            throw std::out_of_range("group stream cache: no such group");
        }
        return groups_.ptr[group].arity;
    }
    /// Every instance of every group, which is what the slab is sized against.
    std::uint32_t total_instances() const noexcept { return index_.arity(); }
    std::uint32_t num_slots() const noexcept { return index_.num_slots(); }
    std::uint64_t slot_bytes() const noexcept { return slot_bytes_; }
    std::uint64_t slab_bytes() const noexcept {
        return slot_bytes_ * num_slots();
    }
    /// True when the slab holds the whole group, so no page-in can ever miss
    /// after the first sweep.
    ///
    /// Not a capture claim. Every family that can stream turns CUDA graph
    /// capture off on the cache existing at all, because a streamed contract
    /// publishes groups instead of stacks and the forward then takes the
    /// per-expert path, which reads the routing table back and synchronizes
    /// whether or not the slab can miss. What sizing does buy is `all_placed`
    /// below, and what that buys is a pointer table written once.
    bool fully_resident() const noexcept {
        return num_slots() == total_instances();
    }

    /// True once the slab both *can* hold the whole group and actually does:
    /// every instance has been placed, so every one has a permanent slot and
    /// no eviction can ever follow.
    ///
    /// This is the state worth special-casing. The per-layer D2H that reads
    /// the routed set exists only so the host can decide which slots to page
    /// in and where the kernel should look; once nothing can move, both
    /// answers are fixed and the whole barrier is dead weight. Measured on
    /// gpt-oss-20b, that barrier is the entire steady-state cost of streaming
    /// -- a warm slab decodes in 4.64 ms against 3.93 ms resident, and the
    /// difference is the host waiting for a routing it no longer needs.
    bool all_placed() const noexcept {
        return fully_resident() && placed_ == total_instances();
    }

    /// The tensors of an instance that is already resident, or null.
    ///
    /// No pin, no page-in, no statistics: this is for a caller that has
    /// established by other means -- `all_placed()` -- that residency cannot
    /// change under it, and only wants the pointers.
    const WeightStore* store_of(
        std::size_t group, std::uint32_t instance) const
    {
        const auto found = index_.find(flatten(group, instance));
        if (found == GroupSlotIndex::kAbsent) return nullptr;
        return &slot_stores_[static_cast<std::uint32_t>(found)];
    }

    /// Make `instance` resident and return its tensors, keyed by the runtime
    /// names its plan finalizes.
    ///
    /// A miss synchronizes `compute_stream` before it writes, because the
    /// victim slot may still be under a kernel that was launched while its old
    /// instance was pinned. The instance stays pinned -- and its pointers stay
    /// valid -- until `end_batch`.
    const WeightStore& ensure_resident(
        std::size_t group,
        std::uint32_t instance,
        cudaStream_t compute_stream)
    {
        const std::uint32_t key = flatten(group, instance);
        const auto found = index_.find(key);
        if (found != GroupSlotIndex::kAbsent) {
            const auto slot = static_cast<std::uint32_t>(found);
            index_.touch_and_pin(slot);
            ++stats_.hits;
            verify_fill(slot, key, compute_stream);
            return slot_stores_[slot];
        }

        const auto acquired = index_.acquire(key);
        ++stats_.misses;
        if (!acquired.evicted) ++placed_;
        if (acquired.evicted) {
            const auto started = Clock::now();
            sync_stream(compute_stream);
            stats_.evict_wait_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now() - started).count());
        }
        fill(acquired.slot, group, instance, key, compute_stream);
        verify_fill(acquired.slot, key, compute_stream);
        return slot_stores_[acquired.slot];
    }

    /// Page `instance` in now, against a guess, on a stream of its own.
    ///
    /// Say that `instance` will be wanted, without waiting for it.
    ///
    /// Only the host half: it asks the kernel to start faulting in the bytes
    /// this instance reads. That is the half worth doing cheaply, because it
    /// is the half the OS cannot do for itself -- a router picks experts at
    /// runtime and no readahead heuristic will guess the order -- and because
    /// it needs no stream, no event and no second slab. The device half still
    /// happens inside `ensure_resident`, against pages that are already there.
    ///
    /// Advisory throughout. Calling it is never required and never wrong;
    /// calling it for an instance that is already resident just wastes a few
    /// `madvise` calls, which is why it does not check.
    void prefetch(std::size_t group, std::uint32_t instance) noexcept
    {
        if (!prefetch_enabled_ || group >= groups_.len) return;
        const auto& g = groups_.ptr[group];
        const std::size_t per_instance = g.bindings_per_instance;
        if (per_instance == 0) return;
        const std::size_t start =
            static_cast<std::size_t>(instance) * per_instance;
        if (start + per_instance > g.bindings.len) return;
        for (std::size_t i = 0; i < per_instance; ++i) {
            const auto& binding = g.bindings.ptr[start + i];
            const auto span = span_by_instr_[group].find(binding.instr_id);
            if (span == span_by_instr_[group].end()) continue;
            loader_.advise_will_need(
                binding.file_id, binding.file_offset + span->second.base_offset,
                span->second.bytes);
        }
        ++stats_.prefetches;
    }

    /// End the batch: every slot becomes evictable again. Pointers handed out
    /// since the last call must not be used past this point.
    void end_batch() noexcept { index_.unpin_all(); }

    struct Stats {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t bytes_paged_in = 0;
        /// Wall time inside `fill`, which is the whole of a miss: reading the
        /// checkpoint, any transform, and the copy. Blocking, so it is time
        /// the forward pass did not spend computing.
        std::uint64_t page_in_ns = 0;
        /// Wall time spent waiting on the compute stream before overwriting a
        /// slot. Separated because it is the cost of the slab being too small
        /// rather than of the bytes moving, and the two want different fixes.
        std::uint64_t evict_wait_ns = 0;
        /// The page-in split the way a fix would be: allocating the plan's
        /// buffers, moving the bytes, running the transform. A miss of a few
        /// megabytes is not bandwidth-bound, so which of these dominates is
        /// the whole question.
        double alloc_ms = 0;
        double transfer_ms = 0;
        double transform_ms = 0;
        /// The first page-in, held out of the rest. It pays for the copy
        /// engine's streams and staging pool, and at a few hundred
        /// microseconds against a steady-state miss of a few tens it would
        /// otherwise dominate the average and hide what a miss really costs.
        std::uint64_t first_page_in_ns = 0;
        /// Its bytes, held out for the same reason. Reporting MiB/s over the
        /// unadjusted byte total against a time total that excludes warm-up
        /// inflates the rate by exactly the warm-up transfer, and MiB/s is the
        /// number that decides whether the source is fast enough.
        std::uint64_t first_page_in_bytes = 0;
        std::uint64_t prefetches = 0;
        /// Misses served from the host tier rather than the checkpoint. They
        /// are still misses -- the slab did not hold the instance -- but they
        /// cost one H2D instead of a read, a plan and a transform.
        std::uint64_t host_tier_hits = 0;
    };

    Stats stats() const noexcept { return stats_; }
    std::uint64_t evictions() const noexcept { return index_.evictions(); }

private:
    /// Validate the groups and lay out the one flat key space over them.
    ///
    /// Returns the total instance count, and on the way establishes the
    /// single slot stride the whole slab is sized in.
    std::uint64_t scan_groups()
    {
        if (groups_.len == 0) {
            throw std::runtime_error(
                "group stream cache: nothing to page (no groups)");
        }
        std::uint64_t total_instances = 0;
        for (std::size_t g = 0; g < groups_.len; ++g) {
            const auto& group = groups_.ptr[g];
            const std::string name = pie_loader::bytes_to_string(group.name);
            if (group.plan == nullptr) {
                throw std::runtime_error(
                    "group stream cache: group \"" + name + "\" has no plan");
            }
            const std::uint64_t bytes = group.plan->memory.persistent_bytes;
            if (bytes == 0) {
                throw std::runtime_error(
                    "group stream cache: group \"" + name + "\" occupies no "
                    "persistent memory, so there is nothing to page");
            }
            // One stride for the whole slab, so a slot can hold any instance
            // of any group. Every layer's experts have the same shape, so a
            // disagreement here means the groups were not meant to share a
            // cache and sizing one slab for them would be a guess.
            if (slot_bytes_ == 0) {
                slot_bytes_ = bytes;
            } else if (bytes != slot_bytes_) {
                throw std::runtime_error(
                    "group stream cache: group \"" + name + "\" wants " +
                    std::to_string(bytes) + " bytes per instance but \"" +
                    pie_loader::bytes_to_string(groups_.ptr[0].name) +
                    "\" wants " + std::to_string(slot_bytes_) +
                    ", so they cannot share a slab");
            }
            group_base_.push_back(total_instances);
            total_instances += group.arity;
            span_by_instr_.push_back(read_spans(*group.plan));
        }
        return total_instances;
    }

    /// How much each rebindable instruction reads, by instruction id.
    ///
    /// A binding says where an instance reads and not how much, because the
    /// span is one of the fields the group proved index-independent -- so it
    /// is on the instruction, once, and this is where the two are put back
    /// together. The set of instruction kinds carrying a source is exactly
    /// what the loader binds; a kind missing here prefetches nothing, which
    /// is slower and not wrong.
    struct Span {
        std::uint64_t base_offset = 0;
        std::uint64_t bytes = 0;
    };
    static std::unordered_map<std::uint32_t, Span> read_spans(
        const pie_loader::LoadPlanView& plan)
    {
        std::unordered_map<std::uint32_t, Span> out;
        using Tag = pie_loader::PieLoaderStorageOp::Tag;
        for (std::size_t i = 0; i < plan.instrs.len; ++i) {
            const auto& instr = plan.instrs.ptr[i];
            const pie_loader::PieLoaderSourceExtentView* source = nullptr;
            switch (instr.op.tag) {
            case Tag::ExtentWrite:
                source = &instr.op.extent_write.source;
                break;
            case Tag::BulkExtentWrite:
                source = &instr.op.bulk_extent_write.source;
                break;
            case Tag::TileMap:
                if (instr.op.tile_map.has_source) {
                    source = &instr.op.tile_map.source;
                }
                break;
            default:
                break;
            }
            if (source == nullptr) continue;
            out.emplace(instr.id,
                        Span{source->stride.base_offset, source->span_bytes});
        }
        return out;
    }

    std::uint8_t* slot_base(std::uint32_t slot) const noexcept {
        return slab_ + static_cast<std::uint64_t>(slot) * slot_bytes_;
    }

    /// One flat key space over every instance of every group, which is what
    /// the slot index is addressed by.
    std::uint32_t flatten(std::size_t group, std::uint32_t instance) const
    {
        if (group >= groups_.len) {
            throw std::out_of_range(
                "group stream cache: group " + std::to_string(group) +
                " outside " + std::to_string(groups_.len));
        }
        return static_cast<std::uint32_t>(group_base_[group]) + instance;
    }

    /// Serve a slot from the host tier, if this instance is in it.
    ///
    /// A whole-slot copy, which is exactly right and is the point of the
    /// tier: every instance lays its buffers out identically -- that is what
    /// the group proved -- so a slot is a flat blob and one instance's is
    /// interchangeable with another's byte for byte. Nothing has to be
    /// re-planned, re-allocated or re-transformed; a tier hit is one H2D of
    /// bytes that are already in the form the kernels read.
    bool fill_from_host(std::uint32_t slot, std::uint32_t key, cudaStream_t stream)
    {
        if (host_tier_ == nullptr || !slot_filled_[slot]) return false;
        const auto found = host_of_.find(key);
        if (found == host_of_.end()) return false;
        // Async, on the stream that is about to read the slot. A blocking
        // `cudaMemcpy` would be correct too, but it synchronizes the whole
        // device, and at a slot of any size that costs more than the copy: on
        // GPT-OSS it turned a 1.5 ms page-in into a 3.2 ms one. Stream order
        // is all the ordering this needs, because the kernels that read the
        // slot are queued on the same stream and the eviction that made the
        // slot free already synchronized it.
        CUDA_CHECK(cudaMemcpyAsync(
            slot_base(slot), host_slot(found->second), slot_bytes_,
            cudaMemcpyHostToDevice, stream));
        ++stats_.host_tier_hits;
        stats_.bytes_paged_in += slot_bytes_;
        return true;
    }

    /// Keep this instance in host memory, if the tier has room.
    ///
    /// Fill-once and never evict. A tier that thrashes is worse than no tier
    /// -- every eviction paid a D2H that bought nothing -- and with the tier
    /// sized in instances the operator asked for, refusing to overcommit is
    /// the whole policy. Which instances win is arrival order, which for a
    /// routed model is the first step's routing, and no worse a guess than
    /// any other before a run has been observed.
    void keep_in_host(std::uint32_t slot, std::uint32_t key, cudaStream_t stream)
    {
        if (host_tier_ == nullptr || host_of_.count(key) != 0) return;
        if (host_of_.size() >= host_slots_) return;
        const std::uint32_t at = static_cast<std::uint32_t>(host_of_.size());
        // Async for the same reason the hit is: a blocking copy here
        // synchronizes the device once per fill, which on a slot of any size
        // costs more than the copy itself. Nobody reads this host slot until
        // a later `fill_from_host` on this same stream, so stream order is
        // the whole of the ordering it needs.
        CUDA_CHECK(cudaMemcpyAsync(
            host_slot(at), slot_base(slot), slot_bytes_,
            cudaMemcpyDeviceToHost, stream));
        host_of_.emplace(key, at);
    }

    /// Diagnostic: a page-in of the same instance must produce the same bytes.
    ///
    /// Env-gated because it costs a device sync and a D2H per miss. It answers
    /// exactly one question -- whether a slot's contents are a function of the
    /// instance, as the group's uniformity claim says they are -- and that is
    /// the question any divergence between a streamed and a resident run
    /// reduces to.
    void verify_fill(std::uint32_t slot, std::uint32_t key, cudaStream_t stream)
    {
        if (!verify_fills_) return;
        // Over the named tensors rather than the raw slot, and windowed
        // across each rather than sampled at its ends. The gaps a plan leaves
        // between its buffers are nobody's bytes and nothing reads them, so
        // hashing them would only report that the slot was previously
        // somebody else; and a page-in is many buffers, so a wrong one in the
        // middle is exactly what the ends cannot see.
        constexpr std::size_t kWindows = 16;
        constexpr std::size_t kWindow = 4096;
        std::vector<std::uint8_t> buf(kWindow);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::uint64_t h = 1469598103934665603ull;
        std::vector<std::string> names;
        for (const auto& entry : slot_stores_[slot]) names.push_back(entry.first);
        std::sort(names.begin(), names.end());
        for (const auto& name : names) {
            const auto& t = slot_stores_[slot].get(name);
            const std::size_t bytes = t.nbytes();
            if (bytes < kWindow) continue;
            const std::size_t stride =
                std::max<std::size_t>(kWindow, bytes / kWindows);
            for (std::size_t off = 0; off + kWindow <= bytes; off += stride) {
                CUDA_CHECK(cudaMemcpy(
                    buf.data(),
                    static_cast<const std::uint8_t*>(t.data()) + off, kWindow,
                    cudaMemcpyDeviceToHost));
                for (const std::uint8_t b : buf) {
                    h = (h ^ b) * 1099511628211ull;
                }
            }
        }
        const auto seen = fill_digest_.find(key);
        if (seen == fill_digest_.end()) {
            fill_digest_.emplace(key, h);
        } else if (seen->second != h) {
            std::cerr << "[group stream cache] VERIFY instance " << key
                      << " paged in differently: " << seen->second << " then "
                      << h << "\n";
            seen->second = h;
        }
    }

    std::uint8_t* host_slot(std::uint32_t at) const noexcept {
        return host_tier_ + static_cast<std::uint64_t>(at) * slot_bytes_;
    }

    /// Run instance `instance`'s plan into `slot`.
    ///
    /// The executor is the one the resident load uses, told two things: the
    /// destination arena is this slot, and the sources are this instance's.
    /// Its output goes to a scratch store; on a slot's first fill that store
    /// becomes the slot's, and afterwards it is checked against it and thrown
    /// away, because the layout is what the loader proved constant, not the
    /// bytes.
    void fill(std::uint32_t slot, std::size_t group_index,
              std::uint32_t instance, std::uint32_t key, cudaStream_t stream)
    {
        const auto started_total = Clock::now();
        if (fill_from_host(slot, key, stream)) {
            stats_.page_in_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now() - started_total).count());
            return;
        }
        const auto& group = groups_.ptr[group_index];
        WeightStore scratch;
        WeightStoreBuilder builder(scratch);
        // One copy engine for the life of the cache, not one per page-in: it
        // creates copy streams and a pinned pool on first use and drops them
        // with itself, and at a few megabytes a miss that setup is most of
        // the cost.
        LoadPlanExecutor executor(loader_, builder, {}, &copy_engine_);

        LoadPlanExecution how;
        how.persistent_arena = slot_base(slot);
        how.persistent_arena_bytes = slot_bytes_;
        const std::size_t per_instance = group.bindings_per_instance;
        if (per_instance != 0) {
            const std::size_t start =
                static_cast<std::size_t>(instance) * per_instance;
            if (start + per_instance > group.bindings.len) {
                throw std::runtime_error(
                    "group stream cache: group \"" +
                    pie_loader::bytes_to_string(group.name) + "\" instance " +
                    std::to_string(instance) + " has no bindings");
            }
            how.source_bindings = group.bindings.ptr + start;
            how.source_binding_count = per_instance;
        }

        const auto started = Clock::now();
        const auto stats = executor.execute(*group.plan, how);
        // The executor's `flush` synchronizes its copy streams, but a plan's
        // transforms -- tile maps, MXFP4 and WNA16 dequants, casts -- launch on
        // the legacy default stream and nothing has waited for those. The
        // compute stream is created `cudaStreamNonBlocking`, so it carries no
        // implicit ordering against stream 0, and without this the caller can
        // launch a GEMM over a slot whose dequant is still running.
        //
        // Groups whose plan is pure `ExtentWrite` -- GPT-OSS's packed experts,
        // which is what this path was measured on -- never expose it. One that
        // carries a `scale_per_block`, like DeepSeek-V4's, always would.
        //
        // An event rather than a synchronize: the point is to order the two
        // streams, not to stop the host, and `keep_in_host` below reads the
        // slot on `stream` and needs the same ordering.
        CUDA_CHECK(cudaEventRecord(transform_done_, nullptr));
        CUDA_CHECK(cudaStreamWaitEvent(stream, transform_done_, 0));
        stats_.bytes_paged_in += stats.h2d_copy_bytes;
        const std::uint64_t elapsed_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - started).count());
        if (stats_.misses == 1) {
            stats_.first_page_in_ns = elapsed_ns;
            stats_.first_page_in_bytes = stats_.bytes_paged_in;
        }
        stats_.alloc_ms += stats.phase_alloc_ms;
        stats_.transfer_ms += stats.phase_transfer_ms;
        stats_.transform_ms += stats.phase_transform_ms;
        stats_.page_in_ns += elapsed_ns;

        if (!slot_filled_[slot]) {
            slot_stores_[slot] = std::move(scratch);
            slot_filled_[slot] = true;
        } else {
            check_layout_held(slot, scratch, flatten(group_index, instance));
        }
        keep_in_host(slot, key, stream);
    }

    /// The invariant the whole design rests on: a page-in replaces bytes, not
    /// addresses. If this ever fires, a caller holding a pointer from an
    /// earlier instance is reading the wrong memory, and that is the one bug
    /// this component could produce silently.
    void check_layout_held(
        std::uint32_t slot,
        const WeightStore& fresh,
        std::uint32_t instance) const
    {
        const auto& held = slot_stores_[slot];
        for (const auto& [name, record] : fresh) {
            const auto it = held.find(name);
            if (it == held.end() || it->second.data() != record.data()) {
                throw std::runtime_error(
                    "group stream cache: instance " +
                    std::to_string(instance) + " laid \"" + name +
                    "\" out differently from the instance already in slot " +
                    std::to_string(slot) +
                    ", so the group's instances are not interchangeable");
            }
        }
    }

    void allocate_slab()
    {
        const cudaError_t err = cudaMalloc(
            reinterpret_cast<void**>(&slab_), slab_bytes());
        if (err != cudaSuccess) {
            slab_ = nullptr;
            throw std::runtime_error(
                "group stream cache: could not allocate " +
                std::to_string(slab_bytes()) + " bytes: " +
                cudaGetErrorString(err));
        }
    }

    /// Pin `host_budget_bytes` of host memory as a second tier.
    ///
    /// Pinned, because unpinned host memory would make every tier hit a staged
    /// copy through a bounce buffer and give back most of what the tier is
    /// for. Sized in whole slots and never more than the whole group: a tier
    /// larger than the group is memory that can never be used.
    void allocate_host_tier(std::uint64_t host_budget_bytes,
                            std::uint64_t total_instances)
    {
        if (host_budget_bytes < slot_bytes_) return;
        std::uint64_t slots = host_budget_bytes / slot_bytes_;
        if (slots > total_instances) slots = total_instances;
        const cudaError_t err = cudaHostAlloc(
            reinterpret_cast<void**>(&host_tier_),
            static_cast<std::size_t>(slots * slot_bytes_), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            // Not fatal: without a tier every miss reads the checkpoint, which
            // is the behaviour this is an optimisation over.
            host_tier_ = nullptr;
            if (verbose_) {
                std::cerr << "[pie-driver-cuda] group stream cache: no host "
                             "tier (" << cudaGetErrorString(err) << ")\n";
            }
            return;
        }
        host_slots_ = static_cast<std::uint32_t>(slots);
        host_of_.reserve(host_slots_);
    }

    /// Give every instance its permanent slot, at load.
    ///
    /// Only called when `fully_resident()`, where the index hands out a
    /// distinct slot per key and never evicts, so after this every `find`
    /// hits and `all_placed()` is true for the rest of the process. The bytes
    /// cost the same as loading the group resident would have -- which is what
    /// a budget this large asked for -- and buy back the per-layer barrier.
    void place_all()
    {
        if (slot_stores_.empty()) return;
        const auto started = Clock::now();
        for (std::size_t g = 0; g < groups_.len; ++g) {
            const std::uint32_t arity = groups_.ptr[g].arity;
            for (std::uint32_t i = 0; i < arity; ++i) {
                const std::uint32_t key = flatten(g, i);
                if (index_.find(key) != GroupSlotIndex::kAbsent) continue;
                const auto acquired = index_.acquire(key);
                if (!acquired.evicted) ++placed_;
                fill(acquired.slot, g, i, key, /*stream=*/nullptr);
                index_.release(acquired.slot);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        index_.unpin_all();
        // Placement is not a miss: it is the load the budget asked for, and
        // counting it would make every later hit rate meaningless.
        stats_ = Stats{};
        if (verbose_) {
            std::cerr << "[pie-driver-cuda] group stream cache: placed "
                      << placed_ << " instances in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                             Clock::now() - started).count()
                      << " ms; the slab holds the whole group, so the forward "
                         "will not call back into the host\n";
        }
    }

    /// Read every instance the tier has room for, once, before the first token.
    ///
    /// Filling the tier on demand cannot pay for the misses that dominate a
    /// routed model. Those are compulsory -- the first time a token routes to
    /// an expert, nothing has seen it yet -- and a cache populated by misses
    /// only ever serves the *second* miss on a key. Measured on gpt-oss-20b at
    /// a 6 GB slab, 754 page-ins produced 141 tier hits, because at an 88% hit
    /// rate almost nothing is fetched twice.
    ///
    /// So this is not a cache warm-up but a placement decision, made where the
    /// resident path makes the same one: at load. It costs a full read of the
    /// group here in exchange for every later page-in being a PCIe transfer
    /// rather than a disk read, which the numbers price at roughly 4 GB/s
    /// against 15 GB/s.
    void warm_host_tier()
    {
        if (host_tier_ == nullptr || host_slots_ == 0 || slot_stores_.empty()) {
            return;
        }
        const auto started = Clock::now();
        std::uint32_t staged = 0;
        for (std::size_t g = 0; g < groups_.len && host_of_.size() < host_slots_;
             ++g) {
            const std::uint32_t arity = groups_.ptr[g].arity;
            for (std::uint32_t i = 0; i < arity && host_of_.size() < host_slots_;
                 ++i) {
                const std::uint32_t key = flatten(g, i);
                // Stage through a different slot each time until every slot has
                // been through one fill, then reuse the first.
                //
                // This is not load balancing: a slot serves reads from the host
                // tier only once its store exists, and the store is built by a
                // fill. Staging through slot 0 alone would leave every other
                // slot owing one disk read the first time it is used, which at
                // a large slab is nearly every page-in the run performs -- 486
                // of 760, measured. Since the bytes are being read here anyway,
                // spreading them builds all the stores for free.
                const std::uint32_t scratch =
                    staged < num_slots() ? staged : 0;
                // `fill` stages into the tier itself; this loop only has to
                // choose where.
                fill(scratch, g, i, key, nullptr);
                // The staging slot is about to be rewritten by the next
                // iteration, whose copies run on the engine's own non-blocking
                // streams -- unordered against the D2H `fill` just queued into
                // the tier. Without this, instance i's tier entry can be
                // overwritten by instance i+1's page-in while it is still being
                // read out, and the corruption is invisible: the layout check
                // passes because only the bytes are wrong.
                //
                // In the steady state the same reuse is safe only because
                // eviction synchronizes first. This loop evicts nothing, so it
                // has to say so itself.
                CUDA_CHECK(cudaStreamSynchronize(nullptr));
                ++staged;
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        // The scratch fills are not misses the operator asked for, and counting
        // them would report a page-in rate that no token paid for.
        stats_ = Stats{};
        if (verbose_ || loader_config::env_truthy("PIE_CUDA_STREAM_STATS")) {
            std::cerr << "[pie-driver-cuda] group stream cache: warmed "
                      << host_of_.size() << " host tier slots in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                             Clock::now() - started).count()
                      << " ms\n";
        }
    }

    void free_host_tier() noexcept
    {
        if (host_tier_ != nullptr) {
            cudaFreeHost(host_tier_);
            host_tier_ = nullptr;
        }
    }

    void free_slab() noexcept
    {
        if (slab_ != nullptr) {
            cudaFree(slab_);
            slab_ = nullptr;
        }
    }

    static void sync_stream(cudaStream_t stream)
    {
        if (stream != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    using Clock = std::chrono::steady_clock;

    pie_loader::CheckpointSource& loader_;
    WeightCopyEngine copy_engine_{loader_};
    pie_loader::PieLoaderGroupSlice groups_;
    /// Where each group's instances start in the one flat key space.
    std::vector<std::uint64_t> group_base_;
    /// Per group, what each of its plan's rebindable instructions reads.
    std::vector<std::unordered_map<std::uint32_t, Span>> span_by_instr_;
    bool verbose_ = false;
    bool prefetch_enabled_ = true;
    bool verify_fills_ = false;

    std::uint64_t slot_bytes_ = 0;
    std::uint8_t* slab_ = nullptr;
    /// The host tier: pinned, slot-shaped, fill-once.
    std::uint8_t* host_tier_ = nullptr;
    std::uint32_t host_slots_ = 0;
    std::unordered_map<std::uint32_t, std::uint32_t> host_of_;

    /// Orders a page-in's transforms, which run on the legacy default stream,
    /// against whichever stream the reader is on.
    cudaEvent_t transform_done_ = nullptr;

    /// Distinct instances that have ever been given a slot. Only meaningful
    /// alongside `fully_resident()`, where a slot is never taken back.
    std::uint32_t placed_ = 0;
    GroupSlotIndex index_;
    std::vector<WeightStore> slot_stores_;
    std::vector<bool> slot_filled_;
    Stats stats_;
    /// `verify_fill`'s record of what each instance hashed to last time.
    std::unordered_map<std::uint32_t, std::uint64_t> fill_digest_;
};

}  // namespace pie_cuda_driver
