#pragma once

// PTIR channel device arena + epoch-ring commit (docs/ptir/overview.md §7.1).
// Channels are the only stateful construct (§1): bounded
// queues of cells with full/empty bits, capacity trace-known. This is the tier-0
// device layout + the readiness / predicated-commit / index-bump kernels the
// stage-runner drives.
//
// LAYOUT (§7.3 "Channels lower to addresses"): per-instance arena — a channel's
// cell lives at `instance_base + channel_offset + ring_index * cell_bytes`. Each
// channel is a ring of `capacity + 1` cells (capacity-1 = double buffer). Ring
// indices head/tail track the committed-read / pending-write positions; a packed
// bit per (channel, cell) records full/empty. Stage readiness reads the full/head
// bits directly (§7.1, C2): a channel is ready iff full[channel @ head].
//
// COMMIT (§7.1, T4): reads take the committed (head) cell, puts write the pending
// (tail) cell; at pass end a predicated index bump publishes puts and consumes
// takes — but ONLY if the pass-wide commit flag is set (all stages ready). On a
// readiness miss the bump is skipped, so pending writes are discarded next pass
// (dummy-run, pass-atomic — T3/T4). Within a pass a channel is a register: a
// double put resolves last-wins because both target the same tail cell.
//
// Tier-0 drives this with a synchronous submit loop (degenerate depth 0, §5):
// channels still order everything; the host just blocks. Tier 1 fuses the
// readiness check into the kernel prologue and the bump into the last kernel's
// epilogue.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

#include <cuda_runtime.h>
#include <cuda/atomic>

#include "cuda_check.hpp"
#include "pie_native/ptir/trace.hpp"

namespace pie_cuda_driver::pipeline {

// Shared pure-host PTIR decode model (trace/op-table/container/bound/
// fire-geometry) now lives in pie_native::ptir (driver/common); bring it into
// scope so the CUDA-side tier-0/1 code below can use it unqualified.
using namespace pie_native::ptir;

// Maximum physical ring slots per channel (`capacity + 1`). Run-ahead guests
// use capacities above the original depth-7 prototype, so keep enough room
// for the supported scheduler depths without truncating declarations.
inline constexpr std::uint32_t kMaxRing = 64;

// ─────────────────────────── device-side kernels ─────────────────────────

// Stage readiness: AND this stage's requirement into the pass commit flag.
// need_full_ch: channels whose first op is take/read (committed cell must be
// full). need_empty_ch: channels whose first op is put (the ring must have room
// — standard ring-not-full `(tail+1)%cap1 != head`, reserving one sentinel cell
// so a capacity-N channel holds ≤ N unconsumed items). A miss clears pass_commit
// → dummy-run, no publish.
__global__ void k_stage_readiness(const std::uint8_t* __restrict__ full,
                                  const std::uint32_t* __restrict__ head,
                                  const std::uint32_t* __restrict__ tail,
                                  const std::uint32_t* __restrict__ cap1,
                                  const std::uint32_t* __restrict__ need_full_ch, std::uint32_t n_full,
                                  const std::uint32_t* __restrict__ need_empty_ch, std::uint32_t n_empty,
                                  std::uint32_t* __restrict__ pass_commit) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    std::uint32_t ok = 1;
    for (std::uint32_t i = 0; i < n_full; ++i) {
        std::uint32_t c = need_full_ch[i];
        if (!full[c * kMaxRing + head[c]]) ok = 0;
    }
    for (std::uint32_t i = 0; i < n_empty; ++i) {
        std::uint32_t c = need_empty_ch[i];
        if (((tail[c] + 1) % cap1[c]) == head[c]) ok = 0;   // ring full → no room
    }
    *pass_commit &= ok;
}

// End-of-pass predicated commit bump (§7.1). Iff *pass_commit: for each taken
// channel advance head (consume), for each put channel advance tail (publish).
// A channel both taken and put (loop-carried ping-pong) advances both.
__device__ inline void commit_bump(
    std::uint8_t* full,
    std::uint32_t* head,
    std::uint32_t* tail,
    const std::uint32_t* cap1,
    const std::uint32_t* taken_ch,
    std::uint32_t n_taken,
    const std::uint32_t* put_ch,
    std::uint32_t n_put,
    const std::uint32_t* pass_commit) {
    if (!*pass_commit) return;
    for (std::uint32_t i = 0; i < n_put; ++i) {
        std::uint32_t c = put_ch[i];
        full[c * kMaxRing + tail[c]] = 1;        // publish the pending cell
        tail[c] = (tail[c] + 1) % cap1[c];
    }
    for (std::uint32_t i = 0; i < n_taken; ++i) {
        std::uint32_t c = taken_ch[i];
        full[c * kMaxRing + head[c]] = 0;        // consume the committed cell
        head[c] = (head[c] + 1) % cap1[c];
    }
}

__global__ void k_commit_bump(std::uint8_t* __restrict__ full,
                              std::uint32_t* __restrict__ head, std::uint32_t* __restrict__ tail,
                              const std::uint32_t* __restrict__ cap1,
                              const std::uint32_t* __restrict__ taken_ch, std::uint32_t n_taken,
                              const std::uint32_t* __restrict__ put_ch, std::uint32_t n_put,
                              const std::uint32_t* __restrict__ pass_commit) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    commit_bump(
        full, head, tail, cap1, taken_ch, n_taken,
        put_ch, n_put, pass_commit);
}

struct CommitBumpLane {
    std::uint8_t* full = nullptr;
    std::uint32_t* head = nullptr;
    std::uint32_t* tail = nullptr;
    const std::uint32_t* cap1 = nullptr;
    const std::uint32_t* taken = nullptr;
    std::uint32_t taken_count = 0;
    const std::uint32_t* put = nullptr;
    std::uint32_t put_count = 0;
    const std::uint32_t* commit = nullptr;
};

__global__ void k_commit_bump_batch(
    const CommitBumpLane* lanes,
    std::uint32_t count) {
    const std::uint32_t lane = blockIdx.x;
    if (lane >= count || threadIdx.x != 0) return;
    const CommitBumpLane value = lanes[lane];
    commit_bump(
        value.full,
        value.head,
        value.tail,
        value.cap1,
        value.taken,
        value.taken_count,
        value.put,
        value.put_count,
        value.commit);
}

inline void launch_commit_bump_batch(
    std::span<const CommitBumpLane> lanes,
    cudaStream_t stream) {
    if (lanes.empty()) return;
    CommitBumpLane* device = nullptr;
    CUDA_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&device),
        lanes.size() * sizeof(CommitBumpLane),
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device,
        lanes.data(),
        lanes.size() * sizeof(CommitBumpLane),
        cudaMemcpyHostToDevice,
        stream));
    k_commit_bump_batch<<<lanes.size(), 1, 0, stream>>>(
        device, static_cast<std::uint32_t>(lanes.size()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(device, stream));
}

inline constexpr std::uint32_t kMaxConditionalConsumeChannels = 64;

struct ConditionalConsumeSlots {
    std::uint32_t n = 0;
    std::uint32_t slots[kMaxConditionalConsumeChannels] = {};
};

inline constexpr std::uint64_t kNoChannelTicket = ~std::uint64_t{0};
inline constexpr std::uint32_t kTicketConsume = 1u << 0;
inline constexpr std::uint32_t kTicketPublish = 1u << 1;
inline constexpr std::uint32_t kTicketHostWriter = 1u << 2;
inline constexpr std::uint32_t kTicketPackedBool = 1u << 3;
inline constexpr std::uint32_t kTicketRequireInput = 1u << 4;

    struct DeviceHostChannelTicket {
        std::uint32_t slot = 0;
        std::uint32_t flags = 0;
        std::uint64_t expected_head = kNoChannelTicket;
        std::uint64_t expected_tail = kNoChannelTicket;
        std::uint64_t* words = nullptr;
        const std::uint8_t* mirror = nullptr;
        std::uint8_t* cells = nullptr;
        std::uint32_t cap1 = 0;
        std::uint32_t wire_bytes = 0;
        std::uint32_t native_bytes = 0;
    };

    struct PullValidateHostChannelLane {
        std::uint8_t* full = nullptr;
        std::uint32_t* pass_commit = nullptr;
        std::uint32_t ticket_offset = 0;
        std::uint32_t ticket_count = 0;
        std::uint32_t initial_commit = 0;
    };

    __device__ inline std::uint64_t load_system_acquire(std::uint64_t* word) {
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> ref(*word);
        return ref.load(cuda::memory_order_acquire);
    }

    __device__ inline void store_system_release(std::uint64_t* word, std::uint64_t value) {
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> ref(*word);
        ref.store(value, cuda::memory_order_release);
    }

    __device__ inline void store_system_release(
        std::uint32_t* word,
        std::uint32_t value) {
        cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> ref(*word);
        ref.store(value, cuda::memory_order_release);
    }

    __global__ void k_pull_validate_host_channels_batch(
        const DeviceHostChannelTicket* tickets,
        const PullValidateHostChannelLane* lanes,
        std::uint32_t lane_count) {
        const std::uint32_t lane_index = blockIdx.x;
        if (lane_index >= lane_count) return;
        const PullValidateHostChannelLane lane = lanes[lane_index];
        __shared__ std::uint32_t valid;
        if (threadIdx.x == 0) {
            // Snapshot words: [0] pass_commit seed, [1] kill — reset both;
            // the ringed snapshot may carry a stale kill from a previous
            // occurrence of this ring slot.
            lane.pass_commit[0] = lane.initial_commit;
            lane.pass_commit[1] = 0;
        }
        __syncthreads();

        for (std::uint32_t index = 0; index < lane.ticket_count; ++index) {
            const DeviceHostChannelTicket ticket =
                tickets[lane.ticket_offset + index];
            if (threadIdx.x == 0) {
                const std::uint64_t head =
                    load_system_acquire(ticket.words + 0);
                const std::uint64_t tail =
                    load_system_acquire(ticket.words + 1);
                bool ok = true;
                if ((ticket.flags & kTicketConsume) != 0) {
                    ok = head == ticket.expected_head;
                }
                if ((ticket.flags & kTicketRequireInput) != 0) {
                    ok = ok && tail > head;
                }
                if ((ticket.flags & kTicketPublish) != 0) {
                    const std::uint64_t same_fire_consume =
                        (ticket.flags & kTicketConsume) != 0 ? 1u : 0u;
                    ok = ok && tail == ticket.expected_tail &&
                         tail - head <
                             static_cast<std::uint64_t>(ticket.cap1 - 1) +
                                 same_fire_consume;
                }
                valid = ok ? 1u : 0u;
                if (!ok) atomicAnd(lane.pass_commit, 0u);
            }
            __syncthreads();

            const bool pull =
                valid != 0 &&
                (ticket.flags & kTicketHostWriter) != 0 &&
                (ticket.flags & kTicketConsume) != 0;
            std::uint32_t ring = 0;
            if (pull) {
                ring = static_cast<std::uint32_t>(
                    ticket.expected_head % ticket.cap1);
                const std::uint8_t* source =
                    ticket.mirror +
                    static_cast<std::size_t>(ring) * ticket.wire_bytes;
                std::uint8_t* destination =
                    ticket.cells +
                    static_cast<std::size_t>(ring) * ticket.native_bytes;
                if ((ticket.flags & kTicketPackedBool) != 0) {
                    for (std::uint32_t i = threadIdx.x;
                         i < ticket.native_bytes;
                         i += blockDim.x) {
                        destination[i] = static_cast<std::uint8_t>(
                            (source[i / 8] >> (i % 8)) & 1u);
                    }
                } else {
                    for (std::uint32_t i = threadIdx.x;
                         i < ticket.native_bytes;
                         i += blockDim.x) {
                        destination[i] = source[i];
                    }
                }
            }
            __syncthreads();
            if (pull && threadIdx.x == 0) {
                lane.full[
                    static_cast<std::size_t>(ticket.slot) * kMaxRing +
                    ring] = 1;
            }
            __syncthreads();
        }
    }

    struct HostChannelSettlementLane {
        std::uint8_t* full = nullptr;
        std::uint32_t* head = nullptr;
        const std::uint32_t* cap1 = nullptr;
        const std::uint32_t* commit = nullptr;
        std::uint32_t* host_commit = nullptr;
        ConditionalConsumeSlots consume{};
        const DeviceHostChannelTicket* tickets = nullptr;
        std::uint32_t ticket_count = 0;
    };

    __global__ void k_settle_host_channels_batch(
        const HostChannelSettlementLane* lanes,
        std::uint32_t count) {
        const std::uint32_t lane_index = blockIdx.x;
        if (lane_index >= count) return;
        const HostChannelSettlementLane lane = lanes[lane_index];
        const std::uint32_t committed = *lane.commit;
        if (threadIdx.x == 0) {
            if (lane.host_commit != nullptr) {
                // Mapped snapshot: mirror both words ([0] commit, [1] kill)
                // so the completion callback classifies the lane without a
                // D2H copy.
                store_system_release(lane.host_commit + 1, lane.commit[1]);
                store_system_release(lane.host_commit, committed);
            }
            if (committed != 0) {
                for (std::uint32_t i = 0; i < lane.consume.n; ++i) {
                    const std::uint32_t slot = lane.consume.slots[i];
                    lane.full[
                        static_cast<std::size_t>(slot) * kMaxRing +
                        lane.head[slot]] = 0;
                    lane.head[slot] =
                        (lane.head[slot] + 1) % lane.cap1[slot];
                }
            }
        }
        __syncthreads();
        if (committed == 0) return;
        for (std::uint32_t ticket_index = threadIdx.x;
             ticket_index < lane.ticket_count;
             ticket_index += blockDim.x) {
            const DeviceHostChannelTicket& ticket =
                lane.tickets[ticket_index];
            if ((ticket.flags & kTicketConsume) != 0) {
                store_system_release(
                    ticket.words + 0, ticket.expected_head + 1);
            }
            if ((ticket.flags & kTicketPublish) != 0) {
                store_system_release(
                    ticket.words + 1, ticket.expected_tail + 1);
                store_system_release(ticket.words + 2, 0);
            }
        }
    }

    inline void launch_settle_host_channels_batch(
        std::span<const HostChannelSettlementLane> lanes,
        cudaStream_t stream) {
        if (lanes.empty()) return;
        HostChannelSettlementLane* device = nullptr;
        CUDA_CHECK(cudaMallocAsync(
            reinterpret_cast<void**>(&device),
            lanes.size() * sizeof(HostChannelSettlementLane),
            stream));
        CUDA_CHECK(cudaMemcpyAsync(
            device,
            lanes.data(),
            lanes.size() * sizeof(HostChannelSettlementLane),
            cudaMemcpyHostToDevice,
            stream));
        k_settle_host_channels_batch<<<lanes.size(), 128, 0, stream>>>(
            device, static_cast<std::uint32_t>(lanes.size()));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFreeAsync(device, stream));
    }

    inline DeviceHostChannelTicket* launch_pull_validate_host_channels_batch(
        const std::vector<DeviceHostChannelTicket>& tickets,
        const std::vector<PullValidateHostChannelLane>& lanes,
        cudaStream_t stream) {
        if (lanes.empty()) return nullptr;
        DeviceHostChannelTicket* device_tickets = nullptr;
        PullValidateHostChannelLane* device_lanes = nullptr;
        try {
            if (!tickets.empty()) {
                CUDA_CHECK(cudaMallocAsync(
                    reinterpret_cast<void**>(&device_tickets),
                    tickets.size() * sizeof(DeviceHostChannelTicket),
                    stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    device_tickets,
                    tickets.data(),
                    tickets.size() * sizeof(DeviceHostChannelTicket),
                    cudaMemcpyHostToDevice,
                    stream));
            }
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&device_lanes),
                lanes.size() * sizeof(PullValidateHostChannelLane),
                stream));
            CUDA_CHECK(cudaMemcpyAsync(
                device_lanes,
                lanes.data(),
                lanes.size() * sizeof(PullValidateHostChannelLane),
                cudaMemcpyHostToDevice,
                stream));
            k_pull_validate_host_channels_batch<<<lanes.size(), 256, 0, stream>>>(
                device_tickets,
                device_lanes,
                static_cast<std::uint32_t>(lanes.size()));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaFreeAsync(device_lanes, stream));
        } catch (...) {
            auto release = [stream](void* allocation) {
                if (allocation == nullptr) return;
                if (cudaFreeAsync(allocation, stream) != cudaSuccess) {
                    cudaStreamSynchronize(stream);
                    cudaFree(allocation);
                }
            };
            release(device_lanes);
            release(device_tickets);
            throw;
        }
        return device_tickets;
    }

// ──────────────────────────── host-side arena ────────────────────────────

// One instance's channel arena. Owns device memory for the cell blob + ring
// bookkeeping (head/tail/full) + the derived bits word and the pass commit flag.
// Cell pointers are stable; the stage-runner resolves take/read/put targets
// through committed_cell() / pending_cell().
class ChannelArena {
  public:
    ChannelArena() = default;
    ~ChannelArena() { free(); }
    ChannelArena(const ChannelArena&) = delete;
    ChannelArena& operator=(const ChannelArena&) = delete;

    // Allocate from a trace's channel declarations. Seeds (`Channel::from`) put
    // the first cell full at instantiation.
    void init(const std::vector<Channel>& channels) {
        free();
        num_ = static_cast<std::uint32_t>(channels.size());
        std::vector<std::uint32_t> h_cap1(num_), h_head(num_, 0), h_tail(num_, 0), h_off(num_);
        std::vector<std::uint8_t> h_full((std::size_t)num_ * kMaxRing, 0);
        cell_bytes_.resize(num_);
        std::size_t blob = 0;
        for (std::uint32_t c = 0; c < num_; ++c) {
            std::uint32_t cap1 = channels[c].capacity + 1;
            if (cap1 > kMaxRing) cap1 = kMaxRing;
            h_cap1[c] = cap1;
            std::size_t cb = channels[c].type.shape.numel() * dtype_size(channels[c].type.dtype);
            if (cb == 0) cb = dtype_size(channels[c].type.dtype);
            cell_bytes_[c] = cb;
            h_off[c] = static_cast<std::uint32_t>(blob);
            blob += cb * cap1;
            if (channels[c].has_seed) {
                // Channel::from(seed): first cell (head 0) full; tail advances past it.
                h_full[(std::size_t)c * kMaxRing + 0] = 1;
                h_tail[c] = 1 % cap1;
            }
        }
        blob_bytes_ = blob == 0 ? 1 : blob;
        cudaMalloc(&d_blob_, blob_bytes_);
        cudaMemset(d_blob_, 0, blob_bytes_);
        cudaMalloc(&d_cap1_, num_ * sizeof(std::uint32_t));
        cudaMalloc(&d_head_, num_ * sizeof(std::uint32_t));
        cudaMalloc(&d_tail_, num_ * sizeof(std::uint32_t));
        cudaMalloc(&d_off_, num_ * sizeof(std::uint32_t));
        cudaMalloc(&d_full_, (std::size_t)num_ * kMaxRing);
        cudaMalloc(&d_commit_, sizeof(std::uint32_t));
        cudaMemcpy(d_cap1_, h_cap1.data(), num_ * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_head_, h_head.data(), num_ * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tail_, h_tail.data(), num_ * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_off_, h_off.data(), num_ * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_full_, h_full.data(), (std::size_t)num_ * kMaxRing, cudaMemcpyHostToDevice);
        host_off_ = h_off;
        host_cap1_ = h_cap1;
        host_head_ = h_head;   // host mirrors of the ring indices (tier-0 sync loop)
        host_tail_ = h_tail;
    }

    // Seed a channel's initial committed cell contents (host bytes → cell 0).
    void seed_cell(ChannelId c, const void* data, std::size_t bytes) {
        cudaMemcpy(committed_base(c), data, bytes, cudaMemcpyHostToDevice);
    }

    // Simulate a host `put`: write the committed (head) cell and mark it full, so
    // the next pass's readiness check sees the channel ready. Models a host-fed
    // channel arriving (the direct host→driver path).
    void host_feed(ChannelId c, const void* data, std::size_t bytes) {
        cudaMemcpy(committed_cell(c), data, bytes, cudaMemcpyHostToDevice);
        std::uint8_t one = 1;
        cudaMemcpy(d_full_ + (std::size_t)c * kMaxRing + host_head(c), &one, 1, cudaMemcpyHostToDevice);
    }

    // Simulate a host `take`: read the committed (head) cell, mark it empty, and
    // advance head — freeing a ring slot for the producer (the host-harvest side
    // of a produce channel, e.g. `out.take().await`). Keeps host + device mirrors
    // in lock-step.
    void host_take(ChannelId c, void* out, std::size_t bytes) {
        cudaMemcpy(out, committed_cell(c), bytes, cudaMemcpyDeviceToHost);
        std::uint8_t zero = 0;
        cudaMemcpy(d_full_ + (std::size_t)c * kMaxRing + host_head(c), &zero, 1, cudaMemcpyHostToDevice);
        std::uint32_t nh = (host_head(c) + 1) % host_cap1_[c];
        host_head_[c] = nh;
        cudaMemcpy(d_head_ + c, &nh, sizeof(nh), cudaMemcpyHostToDevice);
    }
    // The head index lives on device; for the synchronous tier-0 loop the host
    // mirrors it, so committed_cell() resolves without a D2H.
    void* committed_cell(ChannelId c) {
        return static_cast<std::uint8_t*>(d_blob_) + host_off_[c] + (std::size_t)host_head(c) * cell_bytes_[c];
    }
    // Device pointer to channel c's pending (tail) cell — where put writes.
    void* pending_cell(ChannelId c) {
        return static_cast<std::uint8_t*>(d_blob_) + host_off_[c] + (std::size_t)host_tail(c) * cell_bytes_[c];
    }
    // Committed cell at ring index 0 (used to seed contents before the loop).
    void* committed_base(ChannelId c) {
        return static_cast<std::uint8_t*>(d_blob_) + host_off_[c];
    }

    std::uint32_t num_channels() const { return num_; }
    std::size_t cell_bytes(ChannelId c) const { return cell_bytes_[c]; }

    std::uint8_t* d_full() { return d_full_; }
    std::uint32_t* d_head() { return d_head_; }
    std::uint32_t* d_tail() { return d_tail_; }
    std::uint32_t* d_cap1() { return d_cap1_; }
    std::uint32_t* d_commit() { return d_commit_; }

    // Host mirrors of head/tail for pointer resolution (tier-0 sync loop). Kept
    // in lock-step with the device by advance_host_rings() after a committed bump.
    std::uint32_t host_head(ChannelId c) const { return host_head_.empty() ? 0 : host_head_[c]; }
    std::uint32_t host_tail(ChannelId c) const { return host_tail_.empty() ? seed_tail(c) : host_tail_[c]; }

    // After the device commit-bump, refresh the host head/tail mirrors from the
    // device so the next pass resolves cell pointers correctly.
    void sync_host_rings() {
        host_head_.resize(num_);
        host_tail_.resize(num_);
        cudaMemcpy(host_head_.data(), d_head_, num_ * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_tail_.data(), d_tail_, num_ * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    }

    // Read a committed cell back to host (debug / test harness).
    void read_committed(ChannelId c, void* out, std::size_t bytes) {
        cudaMemcpy(out, committed_cell(c), bytes, cudaMemcpyDeviceToHost);
    }

    // Is channel c's committed (head) cell full? (host take would block if not.)
    bool committed_full(ChannelId c) {
        std::uint8_t f = 0;
        cudaMemcpy(&f, d_full_ + (std::size_t)c * kMaxRing + host_head(c), 1, cudaMemcpyDeviceToHost);
        return f != 0;
    }

    // Debug: print head/tail/full-word per channel.
    void dump() {
        std::vector<std::uint32_t> hh(num_), tt(num_);
        std::vector<std::uint8_t> ff((std::size_t)num_ * kMaxRing);
        cudaMemcpy(hh.data(), d_head_, num_ * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(tt.data(), d_tail_, num_ * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(ff.data(), d_full_, ff.size(), cudaMemcpyDeviceToHost);
        for (std::uint32_t c = 0; c < num_; ++c)
            std::printf("        ch%u head=%u tail=%u full[head]=%u cap1=%u\n", c, hh[c], tt[c],
                        ff[(std::size_t)c * kMaxRing + hh[c]], host_cap1_.empty()?0:host_cap1_[c]);
    }

  private:
    std::uint32_t seed_tail(ChannelId c) const { return host_cap1_.empty() ? 0 : (1 % host_cap1_[c]); }

    void free() {
        if (d_blob_) cudaFree(d_blob_);
        if (d_cap1_) cudaFree(d_cap1_);
        if (d_head_) cudaFree(d_head_);
        if (d_tail_) cudaFree(d_tail_);
        if (d_off_) cudaFree(d_off_);
        if (d_full_) cudaFree(d_full_);
        if (d_commit_) cudaFree(d_commit_);
        d_blob_ = nullptr; d_cap1_ = d_head_ = d_tail_ = d_off_ = d_commit_ = nullptr;
        d_full_ = nullptr;
        num_ = 0;
    }

    std::uint32_t num_ = 0;
    std::size_t blob_bytes_ = 0;
    void* d_blob_ = nullptr;
    std::uint32_t *d_cap1_ = nullptr, *d_head_ = nullptr, *d_tail_ = nullptr, *d_off_ = nullptr;
    std::uint8_t* d_full_ = nullptr;
    std::uint32_t *d_commit_ = nullptr;
    std::vector<std::size_t> cell_bytes_;
    std::vector<std::uint32_t> host_off_, host_cap1_, host_head_, host_tail_;
};

}  // namespace pie_cuda_driver::pipeline
