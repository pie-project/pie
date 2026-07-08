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
// slots and need no signature change; `k_channel_bits` (which iterates by index)
// takes an extra `slot_map` so bit `c` reflects the channel at dense index `c`.
//
// The pass-ephemeral scalars (`pass_commit`, the derived bits word) are NOT
// shared — they live in the runner (`Tier0Runner`). Only the durable ring state
// (cells, full[], head/tail, cap1) is registry-owned and shared.

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "ptir/channels.hpp"  // kMaxRing + the (unchanged) ring kernels
#include "ptir/trace.hpp"

namespace pie_cuda_driver::ptir {

// Registry device-array capacity in SLOTS (grows on demand). One inferlet's
// live channel count is small; this is a generous initial reservation.
inline constexpr std::uint32_t kInitialChannelSlots = 1024;

// The shared, global device channel table. One per driver (owned by
// `PtirDispatch::Impl`). Slots are compact indices into the shared device
// arrays; `slot_of_` maps a global channel id → slot. Freed slots are recycled.
class DeviceChannelRegistry {
  public:
    DeviceChannelRegistry() { grow(kInitialChannelSlots); }
    ~DeviceChannelRegistry() { free_all(); }
    DeviceChannelRegistry(const DeviceChannelRegistry&) = delete;
    DeviceChannelRegistry& operator=(const DeviceChannelRegistry&) = delete;

    // Resolve (creating on first sight) the compact device slot backing global
    // channel `id`, using `decl` for shape/dtype/capacity/seed. On a later
    // reference the decl MUST match the first registration (the multi-pass
    // sharing invariant, §6 "Decl conflicts on shared channels"); a mismatch is
    // a loud error. Returns `kBadSlot` and sets `*err` on conflict/OOM.
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
            return slot;
        }
        const std::uint32_t slot = alloc_slot();
        if (slot == kBadSlot) {
            if (err) *err = "ptir: channel registry out of slots";
            return kBadSlot;
        }
        init_slot(slot, decl);
        slot_of_.emplace(id, slot);
        id_of_[slot] = id;
        return slot;
    }

    bool contains(std::uint64_t id) const { return slot_of_.find(id) != slot_of_.end(); }

    // The compact device slot backing global channel `id`, or `kBadSlot` if the
    // channel is not registered (host-put / seed for an unbound global id).
    std::uint32_t slot_for(std::uint64_t id) const {
        auto it = slot_of_.find(id);
        return it == slot_of_.end() ? kBadSlot : it->second;
    }

    // Free a global channel's device storage (W0.3 release marker). Device
    // lifetime follows the WIT resource drop (plan §6): the guest sends exactly
    // one release marker when it drops the channel handle; a channel shared by
    // several passes is kept alive by the guest for those passes' lifetimes.
    void release(std::uint64_t id) {
        auto it = slot_of_.find(id);
        if (it == slot_of_.end()) return;
        free_slot(it->second);
        slot_of_.erase(it);
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

    // Seed a slot's committed cell contents (host bytes → cell 0), pre-loop.
    void seed_cell(std::uint32_t slot, const void* data, std::size_t bytes) {
        cudaMemcpy(committed_base(slot), data, bytes, cudaMemcpyHostToDevice);
    }

    // Host `put`: fill the committed (head) cell + mark it full (host-fed
    // channel arriving — the direct host→driver path).
    void host_feed(std::uint32_t slot, const void* data, std::size_t bytes) {
        cudaMemcpy(committed_cell(slot), data, bytes, cudaMemcpyHostToDevice);
        std::uint8_t one = 1;
        cudaMemcpy(d_full_ + (std::size_t)slot * kMaxRing + host_head_[slot], &one, 1,
                   cudaMemcpyHostToDevice);
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

    bool committed_full(std::uint32_t slot) {
        std::uint8_t f = 0;
        cudaMemcpy(&f, d_full_ + (std::size_t)slot * kMaxRing + host_head_[slot], 1,
                   cudaMemcpyDeviceToHost);
        return f != 0;
    }

    void read_committed(std::uint32_t slot, void* out, std::size_t bytes) {
        cudaMemcpy(out, committed_cell(slot), bytes, cudaMemcpyDeviceToHost);
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
        return cell_bytes_[slot] == cb && host_cap1_[slot] == cap1_of(decl);
    }
    static std::uint32_t cap1_of(const Channel& decl) {
        std::uint32_t cap1 = decl.capacity + 1;
        return cap1 > kMaxRing ? kMaxRing : cap1;
    }
    static std::size_t decl_cell_bytes(const Channel& decl) {
        std::size_t cb = decl.type.shape.numel() * dtype_size(decl.type.dtype);
        return cb == 0 ? dtype_size(decl.type.dtype) : cb;
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

    void init_slot(std::uint32_t slot, const Channel& decl) {
        const std::uint32_t cap1 = cap1_of(decl);
        const std::size_t cb = decl_cell_bytes(decl);
        cell_bytes_[slot] = cb;
        host_cap1_[slot] = cap1;
        cudaMalloc(&cell_base_[slot], cb * cap1);
        cudaMemset(cell_base_[slot], 0, cb * cap1);

        std::uint32_t head0 = 0, tail0 = 0;
        std::uint8_t full0[kMaxRing];
        std::memset(full0, 0, kMaxRing);
        if (decl.has_seed) {
            // Channel::from(seed): cell 0 committed-full, tail past it.
            full0[0] = 1;
            tail0 = 1 % cap1;
        }
        host_head_[slot] = head0;
        host_tail_[slot] = tail0;
        cudaMemcpy(d_cap1_ + slot, &cap1, sizeof(cap1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_head_ + slot, &head0, sizeof(head0), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tail_ + slot, &tail0, sizeof(tail0), cudaMemcpyHostToDevice);
        cudaMemcpy(d_full_ + (std::size_t)slot * kMaxRing, full0, kMaxRing,
                   cudaMemcpyHostToDevice);
    }

    void free_slot(std::uint32_t slot) {
        if (cell_base_[slot]) {
            cudaFree(cell_base_[slot]);
            cell_base_[slot] = nullptr;
        }
        cell_bytes_[slot] = 0;
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
        cell_bytes_.resize(new_cap, 0);
        host_head_.resize(new_cap, 0);
        host_tail_.resize(new_cap, 0);
        host_cap1_.resize(new_cap, 1);
        id_of_.resize(new_cap, 0);
    }

    void free_all() {
        for (void* p : cell_base_) if (p) cudaFree(p);
        if (d_full_) cudaFree(d_full_);
        if (d_head_) cudaFree(d_head_);
        if (d_tail_) cudaFree(d_tail_);
        if (d_cap1_) cudaFree(d_cap1_);
        d_full_ = nullptr; d_head_ = d_tail_ = d_cap1_ = nullptr;
    }

    std::unordered_map<std::uint64_t, std::uint32_t> slot_of_;
    std::vector<std::uint64_t> id_of_;
    std::vector<void*> cell_base_;
    std::vector<std::size_t> cell_bytes_;
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

    // Bind this view's dense channels to the registry. `decls` are the trace's
    // channel declarations (dense order); `channel_ids[i]` is dense channel i's
    // global id (from the wire's `ptir_program_channel_ids`). Seeds are applied
    // by the caller (only on the instance's first fire). Returns false + `*err`
    // on a decl conflict / OOM.
    bool bind(DeviceChannelRegistry* reg, const std::vector<Channel>& decls,
              const std::vector<std::uint64_t>& channel_ids, std::string* err) {
        reg_ = reg;
        dense_to_slot_.assign(decls.size(), DeviceChannelRegistry::kBadSlot);
        global_ids_ = channel_ids;
        for (std::size_t i = 0; i < decls.size(); ++i) {
            const std::uint64_t gid =
                i < channel_ids.size() ? channel_ids[i] : (std::uint64_t)i;
            const std::uint32_t slot = reg_->get_or_create(gid, decls[i], err);
            if (slot == DeviceChannelRegistry::kBadSlot) return false;
            dense_to_slot_[i] = slot;
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
    void* committed_cell(ChannelId c) { return reg_->committed_cell(slot(c)); }
    void* pending_cell(ChannelId c) { return reg_->pending_cell(slot(c)); }
    void* committed_base(ChannelId c) { return reg_->committed_base(slot(c)); }
    void seed_cell(ChannelId c, const void* data, std::size_t bytes) {
        reg_->seed_cell(slot(c), data, bytes);
    }
    void host_feed(ChannelId c, const void* data, std::size_t bytes) {
        reg_->host_feed(slot(c), data, bytes);
    }
    void host_take(ChannelId c, void* out, std::size_t bytes) {
        reg_->host_take(slot(c), out, bytes);
    }
    bool committed_full(ChannelId c) { return reg_->committed_full(slot(c)); }
    void read_committed(ChannelId c, void* out, std::size_t bytes) {
        reg_->read_committed(slot(c), out, bytes);
    }
    // Refresh host mirrors of THIS view's slots after a commit-bump.
    void sync_host_rings() { reg_->sync_host_rings(dense_to_slot_); }

    DeviceChannelRegistry* registry() { return reg_; }

  private:
    DeviceChannelRegistry* reg_ = nullptr;
    std::vector<std::uint32_t> dense_to_slot_;
    std::vector<std::uint64_t> global_ids_;
};

}  // namespace pie_cuda_driver::ptir
