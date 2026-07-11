#pragma once

// PTIR GLOBAL device channel registry (plan W0.1). Replaces the per-instance
// `ChannelArena` OWNERSHIP model with a process-wide table keyed by the
// runtime-minted GLOBAL channel id. A channel bound into many forward passes /
// instances (multi-pass channels, W3) resolves ONE shared device cell ring, so
// draft→verify chaining and cross-pass sharing observe the same full/empty bits
// and cells.
//
// WHAT MOVES vs WHAT STAYS (owner constraint: "the ring/bits/readiness kernel
// machinery moves, it doesn't change"): the ring layout, the full/empty bits,
// and the readiness / predicated-commit / index-bump kernels (channels.hpp) are
// UNCHANGED. Only the storage ownership moves here. Channel references indirect
// through a per-instance dense-index → global-slot view (`ChannelView`): the
// runner remaps its dense channel lists to slots host-side before each kernel
// launch, so the list-taking kernels (`k_stage_readiness`, `k_commit_bump`) see
// slots and need no signature change.
//
// The pass-ephemeral commit flag (`pass_commit`) is NOT shared — it lives in the
// runner (`Tier0Runner`). Only the durable ring state (cells, full[], head/tail,
// cap1) is registry-owned and shared.

#include <algorithm>
#include <cstddef>
#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <pie_driver_abi.h>

#include "ptir/channels.hpp"  // kMaxRing + the (unchanged) ring kernels
#include "ptir/trace.hpp"
#include "cuda_check.hpp"

namespace pie_cuda_driver::ptir {

// Registry device-array capacity in SLOTS (grows on demand). One inferlet's
// live channel count is small; this is a generous initial reservation.
inline constexpr std::uint32_t kInitialChannelSlots = 1024;

static __global__ void pack_bool_channel_cell(
    const std::uint8_t* source,
    std::uint8_t* destination,
    std::size_t numel) {
    const std::size_t byte =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t packed_bytes = (numel + 7) / 8;
    if (byte >= packed_bytes) return;
    std::uint8_t packed = 0;
    const std::size_t first = byte * 8;
    for (std::size_t bit = 0; bit < 8 && first + bit < numel; ++bit) {
        if (source[first + bit] != 0) {
            packed |= static_cast<std::uint8_t>(1u << bit);
        }
    }
    destination[byte] = packed;
}

// The shared, global device channel table. One per driver (owned by
// `PtirDispatch::Impl`). Slots are compact indices into the shared device
// arrays; `slot_of_` maps a global channel id → slot. Freed slots are recycled.
class DeviceChannelRegistry {
  public:
    DeviceChannelRegistry() { grow(kInitialChannelSlots); }
    ~DeviceChannelRegistry() { free_all(); }
    DeviceChannelRegistry(const DeviceChannelRegistry&) = delete;
    DeviceChannelRegistry& operator=(const DeviceChannelRegistry&) = delete;

    bool register_endpoint(const PieChannelDesc& desc,
                           PieChannelEndpointBinding* binding,
                           std::string* err) {
        if (contains(desc.channel_id)) {
            if (err) *err = "ptir: duplicate channel id";
            return false;
        }
        const std::uint64_t wire_bytes = wire_cell_bytes(desc);
        const std::uint64_t native_bytes = native_cell_bytes(desc);
        if (wire_bytes == 0 || native_bytes == 0 ||
            static_cast<std::uint64_t>(desc.capacity) + 1 > kMaxRing ||
            wire_bytes > std::numeric_limits<std::size_t>::max() /
                             (static_cast<std::uint64_t>(desc.capacity) + 1)) {
            if (err) *err = "ptir: unsupported channel geometry";
            return false;
        }
        const std::uint32_t slot = alloc_slot();
        if (slot == kBadSlot) {
            if (err) *err = "ptir: channel registry out of slots";
            return false;
        }
        host_mirror_bytes_[slot] =
            wire_bytes * (static_cast<std::uint64_t>(desc.capacity) + 1);
        try {
            init_slot(slot, desc);
            if (desc.dtype == PIE_CHANNEL_DTYPE_BOOL) {
                CUDA_CHECK(cudaMalloc(
                    &wire_staging_[slot], host_mirror_bytes_[slot]));
            }
            CUDA_CHECK(cudaHostAlloc(
                &host_mirror_[slot], host_mirror_bytes_[slot], cudaHostAllocPortable));
            CUDA_CHECK(cudaHostAlloc(
                reinterpret_cast<void**>(&host_words_[slot]),
                4 * sizeof(std::uint64_t), cudaHostAllocPortable));
        } catch (...) {
            free_slot(slot);
            throw;
        }
        slot_of_.emplace(desc.channel_id, slot);
        id_of_[slot] = desc.channel_id;
        refcounts_[slot] = 0;
        dtype_[slot] = desc.dtype;
        host_role_[slot] = desc.host_role;
        reader_wait_ids_[slot] = desc.reader_wait_id;
        writer_wait_ids_[slot] = desc.writer_wait_id;
        seeded_[slot] = desc.seeded != 0;
        extern_direction_[slot] = desc.extern_dir;
        extern_names_[slot].clear();
        if (desc.extern_name.len != 0) {
            extern_names_[slot].assign(
                reinterpret_cast<const char*>(desc.extern_name.ptr),
                desc.extern_name.len);
        }
        attachment_direction_masks_[slot] = 0;
        shapes_[slot].assign(desc.shape.ptr, desc.shape.ptr + desc.shape.len);
        pulled_tail_[slot] = 0;
        seed_credit_[slot] = static_cast<std::uint8_t>(
            desc.seeded != 0 &&
            desc.host_role == PIE_CHANNEL_HOST_ROLE_WRITER);
        std::memset(host_mirror_[slot], 0, host_mirror_bytes_[slot]);
        std::memset(host_words_[slot], 0, 4 * sizeof(std::uint64_t));
        *binding = PieChannelEndpointBinding{
            .channel_id = desc.channel_id,
            .mirror_base = reinterpret_cast<std::uint64_t>(host_mirror_[slot]),
            .word_base = reinterpret_cast<std::uint64_t>(host_words_[slot]),
            .mirror_bytes = host_mirror_bytes_[slot],
            .word_bytes = 4 * sizeof(std::uint64_t),
            .cell_bytes = static_cast<std::uint32_t>(wire_bytes),
            .capacity = desc.capacity,
            .head_word_index = 0,
            .tail_word_index = 1,
            .poison_word_index = 2,
            .closed_word_index = 3,
        };
        return true;
    }

    bool close_endpoint(std::uint64_t id, std::string* err) {
        auto it = slot_of_.find(id);
        if (it == slot_of_.end()) return false;
        const std::uint32_t slot = it->second;
        if (refcounts_[slot] != 0) {
            if (err) *err = "ptir: channel still has instance attachments";
            return false;
        }
        std::atomic_ref<std::uint64_t>(host_words_[slot][3]).store(
            1, std::memory_order_release);
        free_slot(slot);
        slot_of_.erase(it);
        return true;
    }

    // Resolve an explicitly registered endpoint for an instance attachment.
    std::uint32_t get_or_create(std::uint64_t id, const Channel& decl, std::string* err) {
        auto it = slot_of_.find(id);
        if (it != slot_of_.end()) {
            const std::uint32_t slot = it->second;
            if (!decl_matches(slot, decl)) {
                if (err)
                    *err = "ptir: channel " + std::to_string(id) +
                           " re-bound with a conflicting shape/dtype/capacity decl";
                return kBadSlot;
            }
            if (decl.extern_dir < 0) {
                if (extern_direction_[slot] != PIE_CHANNEL_EXTERN_NONE ||
                    refcounts_[slot] != 0) {
                    if (err) *err = "ptir: private channel is already attached";
                    return kBadSlot;
                }
            } else {
                const std::uint8_t direction_bit =
                    static_cast<std::uint8_t>(1u << decl.extern_dir);
                if (extern_direction_[slot] == PIE_CHANNEL_EXTERN_NONE ||
                    extern_names_[slot] != decl.extern_name ||
                    (attachment_direction_masks_[slot] & direction_bit) != 0) {
                    if (err) *err = "ptir: incompatible extern attachment";
                    return kBadSlot;
                }
                attachment_direction_masks_[slot] |= direction_bit;
            }
            if (refcounts_[slot] >= (decl.extern_dir < 0 ? 1u : 2u)) {
                if (err) {
                    *err = "ptir: channel " + std::to_string(id) +
                           " has too many instance attachments";
                }
                return kBadSlot;
            }
            ++refcounts_[slot];
            return slot;
        }
        if (err) *err = "ptir: channel " + std::to_string(id) +
                        " was not registered";
        return kBadSlot;
    }

    bool contains(std::uint64_t id) const { return slot_of_.find(id) != slot_of_.end(); }

    // The compact device slot backing global channel `id`, or `kBadSlot` if the
    // channel is not registered (host-put / seed for an unbound global id).
    std::uint32_t slot_for(std::uint64_t id) const {
        auto it = slot_of_.find(id);
        return it == slot_of_.end() ? kBadSlot : it->second;
    }

    // Release one bound-instance attachment. Endpoint storage remains owned by
    // the explicit channel registration until close_endpoint.
    void release(std::uint64_t id, std::int8_t extern_dir) {
        auto it = slot_of_.find(id);
        if (it == slot_of_.end()) return;
        const std::uint32_t slot = it->second;
        if (refcounts_[slot] != 0) --refcounts_[slot];
        if (extern_dir >= 0) {
            attachment_direction_masks_[slot] &= static_cast<std::uint8_t>(
                ~(1u << static_cast<std::uint8_t>(extern_dir)));
        }
    }

    std::size_t size() const { return slot_of_.size(); }

    // ── shared device arrays (indexed by SLOT) — the kernels' inputs ──
    std::uint8_t*  d_full() { return d_full_; }
    std::uint32_t* d_head() { return d_head_; }
    std::uint32_t* d_tail() { return d_tail_; }
    std::uint32_t* d_cap1() { return d_cap1_; }
    std::uint32_t  slot_capacity() const { return cap_slots_; }

    // ── per-slot cell resolution (host-computed device pointers) ──
    std::size_t cell_bytes(std::uint32_t slot) const { return cell_bytes_[slot]; }
    void* committed_base(std::uint32_t slot) {
        return static_cast<std::uint8_t*>(cell_base_[slot]);
    }
    void* committed_cell(std::uint32_t slot) {
        return static_cast<std::uint8_t*>(cell_base_[slot]) +
               (std::size_t)host_head_[slot] * cell_bytes_[slot];
    }
    void* pending_cell(std::uint32_t slot) {
        return static_cast<std::uint8_t*>(cell_base_[slot]) +
               (std::size_t)host_tail_[slot] * cell_bytes_[slot];
    }
    std::uint32_t host_head(std::uint32_t slot) const { return host_head_[slot]; }
    std::uint32_t host_tail(std::uint32_t slot) const { return host_tail_[slot]; }
    std::uint64_t channel_id(std::uint32_t slot) const { return id_of_[slot]; }
    std::uint64_t reader_wait_id(std::uint32_t slot) const {
        return reader_wait_ids_[slot];
    }
    std::uint64_t writer_wait_id(std::uint32_t slot) const {
        return writer_wait_ids_[slot];
    }
    std::uint64_t host_wait_id(std::uint32_t slot) const {
        return host_role_[slot] == PIE_CHANNEL_HOST_ROLE_READER
            ? reader_wait_ids_[slot]
            : (host_role_[slot] == PIE_CHANNEL_HOST_ROLE_WRITER
                   ? writer_wait_ids_[slot]
                   : 0);
    }
    std::uint64_t poison_target(std::uint32_t slot) const {
        return std::max(reserved_head_[slot], reserved_tail_[slot]) + 1;
    }
    void* host_mirror(std::uint32_t slot) const { return host_mirror_[slot]; }
    std::uint64_t* host_words(std::uint32_t slot) const { return host_words_[slot]; }
    std::size_t wire_bytes(std::uint32_t slot) const {
        return host_mirror_bytes_[slot] / host_cap1_[slot];
    }
    bool can_publish(std::uint32_t slot) const { return can_publish_n(slot, 1); }

    // Whether `count` MORE publishes fit the host-reader capacity — used by
    // batch validation, which must aggregate every member's planned publish
    // to a shared slot before any of them schedules.
    bool can_publish_n(std::uint32_t slot, std::uint64_t count) const {
        const std::uint64_t head =
            std::atomic_ref<const std::uint64_t>(host_words_[slot][0]).load(
                std::memory_order_acquire);
        return head <= reserved_tail_[slot] &&
               reserved_tail_[slot] + count - 1 - head <
                   static_cast<std::uint64_t>(host_cap1_[slot] - 1);
    }

    std::uint64_t schedule_host_publish(
        std::uint32_t slot,
        const void* device_ptr,
        cudaStream_t stream) {
        const std::uint64_t tail = reserved_tail_[slot]++;
        const std::size_t bytes = wire_bytes(slot);
        auto* destination = static_cast<std::uint8_t*>(host_mirror_[slot]) +
            (tail % host_cap1_[slot]) * bytes;
        const void* copy_source = device_ptr;
        if (dtype_[slot] == PIE_CHANNEL_DTYPE_BOOL) {
            auto* packed = static_cast<std::uint8_t*>(wire_staging_[slot]) +
                (tail % host_cap1_[slot]) * bytes;
            constexpr std::uint32_t kThreads = 256;
            const std::uint32_t blocks = static_cast<std::uint32_t>(
                (bytes + kThreads - 1) / kThreads);
            pack_bool_channel_cell<<<blocks, kThreads, 0, stream>>>(
                static_cast<const std::uint8_t*>(device_ptr),
                packed,
                cell_bytes_[slot]);
            CUDA_CHECK(cudaGetLastError());
            copy_source = packed;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            destination, copy_source, bytes, cudaMemcpyDeviceToHost, stream));
        return tail + 1;
    }

    void finalize_host_publish(
        std::uint32_t slot,
        std::uint64_t target_tail,
        bool failed) {
        if (failed) {
            std::atomic_ref<std::uint64_t>(host_words_[slot][2]).store(
                target_tail == 0 ? 1 : target_tail, std::memory_order_release);
            return;
        }
        std::atomic_ref<std::uint64_t>(host_words_[slot][1]).store(
            target_tail, std::memory_order_release);
        std::atomic_ref<std::uint64_t>(host_words_[slot][2]).store(
            0, std::memory_order_release);
    }

    // ── §4.3 writer-ring pull (host produces, device consumes) ──

    // The host-published producer cursor (release-stored by the runtime put).
    std::uint64_t host_ring_tail(std::uint32_t slot) const {
        return std::atomic_ref<const std::uint64_t>(host_words_[slot][1]).load(
            std::memory_order_acquire);
    }

    // Host inputs available to the next fire: published-but-unconsumed ring
    // entries plus the one-shot seed credit of a seeded Writer channel.
    std::uint64_t writer_available(std::uint32_t slot) const {
        return host_ring_tail(slot) - reserved_head_[slot] +
               (seed_credit_[slot] != 0 ? 1 : 0);
    }

    // Schedule one fire's consume of this writer channel: the seed credit is
    // spent first (no ring entry moves — returns 0, no head word publishes),
    // otherwise one ring entry is reserved and the returned target head is
    // release-published by the completion callback on success.
    std::uint64_t schedule_writer_consume(std::uint32_t slot) {
        if (seed_credit_[slot] != 0) {
            seed_credit_[slot] = 0;
            return 0;
        }
        return ++reserved_head_[slot];
    }

    // Pull host-published ring entries (mirror cells up to the released tail
    // word) into the device cell ring + full bits, stream-ordered before the
    // pass that consumes them. Bool cells unpack on the CPU into `staging`,
    // which must outlive the async copies (it rides the fire's finalize
    // entry). A seeded ring holds its bind-time seed at device index 0, so
    // host entries land one index later.
    void pull_writer_ring(std::uint32_t slot,
                          cudaStream_t stream,
                          std::vector<std::vector<std::uint8_t>>& staging) {
        const std::uint64_t tail = host_ring_tail(slot);
        while (pulled_tail_[slot] < tail) {
            const std::uint64_t sequence = pulled_tail_[slot]++;
            const std::size_t bytes = wire_bytes(slot);
            const auto* source =
                static_cast<const std::uint8_t*>(host_mirror_[slot]) +
                (sequence % host_cap1_[slot]) * bytes;
            const std::uint64_t index =
                (sequence + (seeded_[slot] ? 1 : 0)) % host_cap1_[slot];
            const void* copy_source = source;
            std::size_t copy_bytes = bytes;
            if (dtype_[slot] == PIE_CHANNEL_DTYPE_BOOL) {
                staging.emplace_back(cell_bytes_[slot], 0);
                auto& native = staging.back();
                for (std::size_t i = 0; i < native.size(); ++i) {
                    native[i] = static_cast<std::uint8_t>(
                        (source[i / 8] >> (i % 8)) & 1u);
                }
                copy_source = native.data();
                copy_bytes = native.size();
            }
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<std::uint8_t*>(cell_base_[slot]) +
                    index * cell_bytes_[slot],
                copy_source, copy_bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(
                d_full_ + static_cast<std::size_t>(slot) * kMaxRing + index,
                1, 1, stream));
        }
    }

    // Seed a slot's committed cell contents (host bytes → cell 0), pre-loop.
    void seed_cell(std::uint32_t slot, const void* data, std::size_t bytes) {
        CUDA_CHECK(cudaMemcpy(
            committed_base(slot), data, bytes, cudaMemcpyHostToDevice));
    }

    void publish_host_seed(std::uint32_t slot, const void* data, std::size_t bytes) {
        auto* destination = static_cast<std::uint8_t*>(host_mirror_[slot]);
        if (dtype_[slot] == PIE_CHANNEL_DTYPE_BOOL) {
            std::memset(destination, 0, wire_bytes(slot));
            const auto* native = static_cast<const std::uint8_t*>(data);
            for (std::size_t i = 0; i < bytes; ++i) {
                if (native[i] != 0) {
                    destination[i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));
                }
            }
        } else {
            std::memcpy(destination, data, bytes);
        }
        reserved_tail_[slot] = 1;
        std::atomic_ref<std::uint64_t>(host_words_[slot][1]).store(
            1, std::memory_order_release);
    }

    // Standalone/test surface: write NATIVE bytes into the committed (head)
    // device cell and mark it full — the tier-0 runner tests drive channels
    // directly, bypassing endpoints. The production host-input path is
    // `pull_writer_ring` (the shared pinned ring, §4.3).
    void host_feed(std::uint32_t slot, const void* data, std::size_t bytes) {
        CUDA_CHECK(cudaMemcpy(
            committed_cell(slot), data, bytes, cudaMemcpyHostToDevice));
        std::uint8_t one = 1;
        CUDA_CHECK(cudaMemcpy(
            d_full_ + (std::size_t)slot * kMaxRing + host_head_[slot],
            &one, 1, cudaMemcpyHostToDevice));
    }

    // Host `take`: read the committed (head) cell, mark it empty, advance head.
    void host_take(std::uint32_t slot, void* out, std::size_t bytes) {
        cudaMemcpy(out, committed_cell(slot), bytes, cudaMemcpyDeviceToHost);
        std::uint8_t zero = 0;
        cudaMemcpy(d_full_ + (std::size_t)slot * kMaxRing + host_head_[slot], &zero, 1,
                   cudaMemcpyHostToDevice);
        const std::uint32_t nh = (host_head_[slot] + 1) % host_cap1_[slot];
        host_head_[slot] = nh;
        cudaMemcpy(d_head_ + slot, &nh, sizeof(nh), cudaMemcpyHostToDevice);
    }

    // Host `consume`: the consume half of `host_take` WITHOUT the D2H read — mark
    // the committed (head) cell empty + advance head. Phase-3 device value path:
    // the cell already left by DMA straight from `committed_cell`, so no host read.
    void host_consume(std::uint32_t slot) {
        std::uint8_t zero = 0;
        cudaMemcpy(d_full_ + (std::size_t)slot * kMaxRing + host_head_[slot], &zero, 1,
                   cudaMemcpyHostToDevice);
        const std::uint32_t nh = (host_head_[slot] + 1) % host_cap1_[slot];
        host_head_[slot] = nh;
        cudaMemcpy(d_head_ + slot, &nh, sizeof(nh), cudaMemcpyHostToDevice);
    }

    bool committed_full(std::uint32_t slot) {
        std::uint8_t f = 0;
        cudaMemcpy(&f, d_full_ + (std::size_t)slot * kMaxRing + host_head_[slot], 1,
                   cudaMemcpyDeviceToHost);
        return f != 0;
    }

    void read_committed(std::uint32_t slot, void* out, std::size_t bytes) {
        cudaMemcpy(out, committed_cell(slot), bytes, cudaMemcpyDeviceToHost);
    }

    void apply_host_commit(const std::vector<std::uint32_t>& taken_slots,
                           const std::vector<std::uint32_t>& put_slots) {
        for (std::uint32_t slot : put_slots) {
            host_tail_[slot] = (host_tail_[slot] + 1) % host_cap1_[slot];
        }
        for (std::uint32_t slot : taken_slots) {
            host_head_[slot] = (host_head_[slot] + 1) % host_cap1_[slot];
        }
    }

    void apply_host_consume(const std::vector<std::uint32_t>& slots) {
        for (std::uint32_t slot : slots) {
            host_head_[slot] = (host_head_[slot] + 1) % host_cap1_[slot];
        }
    }

    // Refresh the host head/tail mirrors of the given slots from the device
    // after a commit-bump (the runner passes the pass's touched slots).
    void sync_host_rings(const std::vector<std::uint32_t>& slots) {
        for (std::uint32_t slot : slots) {
            cudaMemcpy(&host_head_[slot], d_head_ + slot, sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(&host_tail_[slot], d_tail_ + slot, sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost);
        }
    }

    static constexpr std::uint32_t kBadSlot = 0xFFFFFFFFu;

  private:
    bool decl_matches(std::uint32_t slot, const Channel& decl) const {
        const std::size_t cb = decl_cell_bytes(decl);
        return cell_bytes_[slot] == cb &&
               host_cap1_[slot] == cap1_of(decl) &&
               dtype_[slot] == static_cast<std::uint8_t>(decl.type.dtype) &&
               shapes_[slot] == decl.type.shape.dims &&
               seeded_[slot] == decl.has_seed &&
               host_role_[slot] ==
                   (decl.host_reader
                        ? PIE_CHANNEL_HOST_ROLE_READER
                        : (decl.host_visible ? PIE_CHANNEL_HOST_ROLE_WRITER
                                             : PIE_CHANNEL_HOST_ROLE_NONE));
    }
    static std::uint32_t cap1_of(const Channel& decl) {
        std::uint32_t cap1 = decl.capacity + 1;
        return cap1 > kMaxRing ? kMaxRing : cap1;
    }
    static std::size_t decl_cell_bytes(const Channel& decl) {
        std::size_t cb = decl.type.shape.numel() * dtype_size(decl.type.dtype);
        return cb == 0 ? dtype_size(decl.type.dtype) : cb;
    }

    static std::uint64_t wire_cell_bytes(const PieChannelDesc& desc) {
        std::uint64_t numel = 1;
        for (std::size_t i = 0; i < desc.shape.len; ++i) {
            if (numel > std::numeric_limits<std::uint64_t>::max() /
                             desc.shape.ptr[i]) {
                return 0;
            }
            numel *= desc.shape.ptr[i];
        }
        if (desc.dtype == PIE_CHANNEL_DTYPE_BOOL) {
            return numel > std::numeric_limits<std::uint64_t>::max() - 7
                ? 0
                : (numel + 7) / 8;
        }
        return numel > std::numeric_limits<std::uint64_t>::max() / 4
            ? 0
            : numel * 4;
    }

    static std::uint64_t native_cell_bytes(const PieChannelDesc& desc) {
        std::uint64_t numel = 1;
        for (std::size_t i = 0; i < desc.shape.len; ++i) {
            if (numel > std::numeric_limits<std::uint64_t>::max() /
                             desc.shape.ptr[i]) {
                return 0;
            }
            numel *= desc.shape.ptr[i];
        }
        const std::uint64_t element_bytes =
            desc.dtype == PIE_CHANNEL_DTYPE_BOOL ? 1 : 4;
        return numel > std::numeric_limits<std::size_t>::max() / element_bytes
            ? 0
            : numel * element_bytes;
    }

    std::uint32_t alloc_slot() {
        if (!free_slots_.empty()) {
            std::uint32_t s = free_slots_.back();
            free_slots_.pop_back();
            return s;
        }
        if (next_slot_ >= cap_slots_) grow(cap_slots_ * 2);
        if (next_slot_ >= cap_slots_) return kBadSlot;
        return next_slot_++;
    }

    void init_slot(std::uint32_t slot, const PieChannelDesc& desc) {
        const std::uint32_t cap1 = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(desc.capacity) + 1),
            kMaxRing);
        std::uint64_t numel = 1;
        for (std::size_t i = 0; i < desc.shape.len; ++i) {
            numel *= desc.shape.ptr[i];
        }
        const std::size_t cb = static_cast<std::size_t>(
            numel * (desc.dtype == PIE_CHANNEL_DTYPE_BOOL ? 1 : 4));
        cell_bytes_[slot] = cb;
        host_cap1_[slot] = cap1;
        CUDA_CHECK(cudaMalloc(&cell_base_[slot], cb * cap1));
        CUDA_CHECK(cudaMemset(cell_base_[slot], 0, cb * cap1));
        std::uint32_t head0 = 0;
        std::uint32_t tail0 = desc.seeded != 0 ? 1 % cap1 : 0;
        std::uint8_t full0[kMaxRing]{};
        if (desc.seeded != 0) full0[0] = 1;
        host_head_[slot] = head0;
        host_tail_[slot] = tail0;
        reserved_tail_[slot] = desc.seeded != 0 ? 1 : 0;
        reserved_head_[slot] = 0;
        CUDA_CHECK(cudaMemcpy(
            d_cap1_ + slot, &cap1, sizeof(cap1), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_head_ + slot, &head0, sizeof(head0), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_tail_ + slot, &tail0, sizeof(tail0), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_full_ + static_cast<std::size_t>(slot) * kMaxRing,
            full0, kMaxRing, cudaMemcpyHostToDevice));
    }

    void free_slot(std::uint32_t slot) {
        if (cell_base_[slot]) {
            cudaFree(cell_base_[slot]);
            cell_base_[slot] = nullptr;
        }
        if (wire_staging_[slot] != nullptr) {
            cudaFree(wire_staging_[slot]);
            wire_staging_[slot] = nullptr;
        }
        cell_bytes_[slot] = 0;
        if (host_mirror_[slot] != nullptr) {
            cudaFreeHost(host_mirror_[slot]);
            host_mirror_[slot] = nullptr;
        }
        if (host_words_[slot] != nullptr) {
            cudaFreeHost(host_words_[slot]);
            host_words_[slot] = nullptr;
        }
        host_mirror_bytes_[slot] = 0;
        reserved_tail_[slot] = 0;
        reserved_head_[slot] = 0;
        pulled_tail_[slot] = 0;
        seed_credit_[slot] = 0;
        shapes_[slot].clear();
        extern_names_[slot].clear();
        attachment_direction_masks_[slot] = 0;
        refcounts_[slot] = 0;
        reader_wait_ids_[slot] = 0;
        writer_wait_ids_[slot] = 0;
        id_of_[slot] = 0;
        free_slots_.push_back(slot);
    }

    void grow(std::uint32_t new_cap) {
        // Reallocate the shared device arrays, preserving live slot state.
        std::uint8_t*  nf = nullptr;
        std::uint32_t *nh = nullptr, *nt = nullptr, *nc = nullptr;
        cudaMalloc(&nf, (std::size_t)new_cap * kMaxRing);
        cudaMalloc(&nh, new_cap * sizeof(std::uint32_t));
        cudaMalloc(&nt, new_cap * sizeof(std::uint32_t));
        cudaMalloc(&nc, new_cap * sizeof(std::uint32_t));
        cudaMemset(nf, 0, (std::size_t)new_cap * kMaxRing);
        cudaMemset(nh, 0, new_cap * sizeof(std::uint32_t));
        cudaMemset(nt, 0, new_cap * sizeof(std::uint32_t));
        cudaMemset(nc, 0, new_cap * sizeof(std::uint32_t));
        if (cap_slots_ > 0) {
            cudaMemcpy(nf, d_full_, (std::size_t)cap_slots_ * kMaxRing, cudaMemcpyDeviceToDevice);
            cudaMemcpy(nh, d_head_, cap_slots_ * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(nt, d_tail_, cap_slots_ * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(nc, d_cap1_, cap_slots_ * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice);
            cudaFree(d_full_);
            cudaFree(d_head_);
            cudaFree(d_tail_);
            cudaFree(d_cap1_);
        }
        d_full_ = nf; d_head_ = nh; d_tail_ = nt; d_cap1_ = nc;
        cap_slots_ = new_cap;
        cell_base_.resize(new_cap, nullptr);
        wire_staging_.resize(new_cap, nullptr);
        cell_bytes_.resize(new_cap, 0);
        host_mirror_.resize(new_cap, nullptr);
        host_words_.resize(new_cap, nullptr);
        host_mirror_bytes_.resize(new_cap, 0);
        reserved_tail_.resize(new_cap, 0);
        reserved_head_.resize(new_cap, 0);
        pulled_tail_.resize(new_cap, 0);
        seed_credit_.resize(new_cap, 0);
        shapes_.resize(new_cap);
        dtype_.resize(new_cap, 0);
        host_role_.resize(new_cap, 0);
        reader_wait_ids_.resize(new_cap, 0);
        writer_wait_ids_.resize(new_cap, 0);
        seeded_.resize(new_cap, false);
        extern_direction_.resize(new_cap, 0);
        extern_names_.resize(new_cap);
        attachment_direction_masks_.resize(new_cap, 0);
        refcounts_.resize(new_cap, 0);
        host_head_.resize(new_cap, 0);
        host_tail_.resize(new_cap, 0);
        host_cap1_.resize(new_cap, 1);
        id_of_.resize(new_cap, 0);
    }

    void free_all() {
        for (void* p : cell_base_) if (p) cudaFree(p);
        for (void* p : wire_staging_) if (p) cudaFree(p);
        for (void* p : host_mirror_) if (p) cudaFreeHost(p);
        for (std::uint64_t* p : host_words_) if (p) cudaFreeHost(p);
        if (d_full_) cudaFree(d_full_);
        if (d_head_) cudaFree(d_head_);
        if (d_tail_) cudaFree(d_tail_);
        if (d_cap1_) cudaFree(d_cap1_);
        d_full_ = nullptr; d_head_ = d_tail_ = d_cap1_ = nullptr;
    }

    std::unordered_map<std::uint64_t, std::uint32_t> slot_of_;
    std::vector<std::uint64_t> id_of_;
    std::vector<void*> cell_base_;
    std::vector<void*> wire_staging_;
    std::vector<std::size_t> cell_bytes_;
    std::vector<void*> host_mirror_;
    std::vector<std::uint64_t*> host_words_;
    std::vector<std::uint64_t> host_mirror_bytes_;
    std::vector<std::uint64_t> reserved_tail_;
    std::vector<std::uint64_t> reserved_head_;
    std::vector<std::uint64_t> pulled_tail_;
    std::vector<std::uint8_t> seed_credit_;
    std::vector<std::vector<std::uint32_t>> shapes_;
    std::vector<std::uint8_t> dtype_, host_role_, extern_direction_;
    std::vector<std::uint64_t> reader_wait_ids_, writer_wait_ids_;
    std::vector<std::string> extern_names_;
    std::vector<std::uint8_t> attachment_direction_masks_;
    std::vector<bool> seeded_;
    std::vector<std::uint32_t> refcounts_;
    std::vector<std::uint32_t> host_head_, host_tail_, host_cap1_;
    std::vector<std::uint32_t> free_slots_;
    std::uint32_t next_slot_ = 0;
    std::uint32_t cap_slots_ = 0;

    std::uint8_t*  d_full_ = nullptr;
    std::uint32_t* d_head_ = nullptr;
    std::uint32_t* d_tail_ = nullptr;
    std::uint32_t* d_cap1_ = nullptr;
};

// Per-instance channel view: a dense-declaration-index → global-slot map. The
// instance OWNS this (not storage); the runner resolves each dense channel op to
// a shared registry slot through it. Built from the submission's
// `channel_ids` (dense idx → global id) against the registry.
class ChannelView {
  public:
    ChannelView() = default;
    ~ChannelView() { release_all(); }
    ChannelView(const ChannelView&) = delete;
    ChannelView& operator=(const ChannelView&) = delete;

    // Bind this view's dense channels to the registry. `decls` are the trace's
    // channel declarations (dense order); `channel_ids[i]` is dense channel i's
    // global id (from the wire's `ptir_program_channel_ids`). Seeds are applied
    // by the caller (only on the instance's first fire). Returns false + `*err`
    // on a decl conflict / OOM.
    bool bind(DeviceChannelRegistry* reg, const std::vector<Channel>& decls,
              const std::vector<std::uint64_t>& channel_ids, std::string* err) {
        reg_ = reg;
        dense_to_slot_.assign(decls.size(), DeviceChannelRegistry::kBadSlot);
        global_ids_.clear();
        global_ids_.reserve(decls.size());
        extern_directions_.clear();
        extern_directions_.reserve(decls.size());
        for (std::size_t i = 0; i < decls.size(); ++i) {
            const std::uint64_t gid =
                i < channel_ids.size() ? channel_ids[i] : (std::uint64_t)i;
            const std::uint32_t slot = reg_->get_or_create(gid, decls[i], err);
            if (slot == DeviceChannelRegistry::kBadSlot) {
                release_all();
                return false;
            }
            dense_to_slot_[i] = slot;
            global_ids_.push_back(gid);
            extern_directions_.push_back(decls[i].extern_dir);
        }
        return true;
    }

    std::uint32_t num_channels() const { return (std::uint32_t)dense_to_slot_.size(); }
    std::uint32_t slot(ChannelId dense) const { return dense_to_slot_[dense]; }
    std::uint64_t global_id(ChannelId dense) const {
        return dense < global_ids_.size() ? global_ids_[dense] : (std::uint64_t)dense;
    }
    const std::vector<std::uint32_t>& slots() const { return dense_to_slot_; }

    // Map a dense-channel-id list to registry slots (the runner remaps its
    // readiness / commit lists before uploading them to the kernels).
    std::vector<std::uint32_t> to_slots(const std::vector<std::uint32_t>& dense) const {
        std::vector<std::uint32_t> out;
        out.reserve(dense.size());
        for (std::uint32_t c : dense) out.push_back(dense_to_slot_[c]);
        return out;
    }

    // ── shared arrays (registry-owned; indexed by slot) ──
    std::uint8_t*  d_full() { return reg_->d_full(); }
    std::uint32_t* d_head() { return reg_->d_head(); }
    std::uint32_t* d_tail() { return reg_->d_tail(); }
    std::uint32_t* d_cap1() { return reg_->d_cap1(); }

    // ── dense cell resolution (delegates to the registry via slot) ──
    std::size_t cell_bytes(ChannelId c) const { return reg_->cell_bytes(slot(c)); }
    std::size_t wire_bytes(ChannelId c) const { return reg_->wire_bytes(slot(c)); }
    void* committed_cell(ChannelId c) { return reg_->committed_cell(slot(c)); }
    void* pending_cell(ChannelId c) { return reg_->pending_cell(slot(c)); }
    void* committed_base(ChannelId c) { return reg_->committed_base(slot(c)); }
    void seed_cell(ChannelId c, const void* data, std::size_t bytes) {
        reg_->seed_cell(slot(c), data, bytes);
    }
    void publish_host_seed(ChannelId c, const void* data, std::size_t bytes) {
        reg_->publish_host_seed(slot(c), data, bytes);
    }
    void pull_writer_ring(
        ChannelId c,
        cudaStream_t stream,
        std::vector<std::vector<std::uint8_t>>& staging) {
        reg_->pull_writer_ring(slot(c), stream, staging);
    }
    void host_feed(ChannelId c, const void* data, std::size_t bytes) {
        reg_->host_feed(slot(c), data, bytes);
    }
    void host_take(ChannelId c, void* out, std::size_t bytes) {
        reg_->host_take(slot(c), out, bytes);
    }
    void host_consume(ChannelId c) { reg_->host_consume(slot(c)); }
    bool committed_full(ChannelId c) { return reg_->committed_full(slot(c)); }
    void read_committed(ChannelId c, void* out, std::size_t bytes) {
        reg_->read_committed(slot(c), out, bytes);
    }
    // Refresh host mirrors of THIS view's slots after a commit-bump.
    void sync_host_rings() { reg_->sync_host_rings(dense_to_slot_); }
    void apply_host_commit(const std::vector<std::uint32_t>& taken_slots,
                           const std::vector<std::uint32_t>& put_slots) {
        reg_->apply_host_commit(taken_slots, put_slots);
    }
    void apply_host_consume(const std::vector<std::uint32_t>& slots) {
        reg_->apply_host_consume(slots);
    }

    DeviceChannelRegistry* registry() { return reg_; }

  private:
    void release_all() {
        if (reg_ != nullptr) {
            for (std::size_t i = 0; i < global_ids_.size(); ++i) {
                reg_->release(global_ids_[i], extern_directions_[i]);
            }
        }
        reg_ = nullptr;
        dense_to_slot_.clear();
        global_ids_.clear();
        extern_directions_.clear();
    }

    DeviceChannelRegistry* reg_ = nullptr;
    std::vector<std::uint32_t> dense_to_slot_;
    std::vector<std::uint64_t> global_ids_;
    std::vector<std::int8_t> extern_directions_;
};

}  // namespace pie_cuda_driver::ptir
