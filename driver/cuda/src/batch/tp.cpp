#include "batch/tp.hpp"

#include "batch/forward.hpp"
#include "batch/graph_variant.hpp"
#include "batch/tp_gate.hpp"
#include "pipeline/batch_compose.hpp"
#include "store/recurrent_state_cache.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "distributed.hpp"

namespace pie_cuda_driver {

namespace {

std::mutex g_tp_cpu_gates_mu;
std::unordered_map<std::string, std::shared_ptr<TpSequenceGate>>
    g_tp_cpu_gates;

std::shared_ptr<TpSequenceGate> tp_cpu_gate_for(const std::string& key) {
    std::lock_guard<std::mutex> lk(g_tp_cpu_gates_mu);
    auto& gate = g_tp_cpu_gates[key];
    if (!gate) gate = std::make_shared<TpSequenceGate>();
    return gate;
}

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

// Returns true when a fire epoch was consumed (or the gate is disabled, in
// which case the follower falls through to the NCCL header broadcast).
// Returns false when shutdown was requested — the caller must NOT post the
// header broadcast in that case, because no peer will ever match it.
bool tp_cpu_gate_wait(const std::string& key,
                      std::uint64_t& seen,
                      std::atomic<bool>& stop) {
    if (key.empty()) return true;
    auto gate = tp_cpu_gate_for(key);
    constexpr auto spin_budget = std::chrono::microseconds(2000);
    const auto start = std::chrono::steady_clock::now();
    while (!stop.load(std::memory_order_relaxed)) {
        if (gate->stopped()) return false;
        const std::uint64_t seq = gate->published();
        if (tp_cpu_gate_consume_one(seq, seen)) return true;
        if (std::chrono::steady_clock::now() - start >= spin_budget) break;
        cpu_relax();
    }
    if (stop.load(std::memory_order_relaxed)) return false;

    return gate->wait_one(seen, stop);
}

// ---------------------------------------------------------------------------
// TP stage profiler (PIE_TP_PROFILE=1)
//
// The TP fan-out sits on the decode critical path, so knowing which stage of
// it costs what is the difference between optimising and guessing. Host-side
// wall clock per stage, accumulated per rank and dumped at process exit.
// ---------------------------------------------------------------------------
struct TpProfile {
    enum Stage {
        kGateWait = 0,
        kWakeLatency,     // rank-0 notify -> follower observes the epoch
        kStartSkew,       // rank-0 notify -> follower has launched its forward
        kHeaderRecv,      // follower: header broadcast + D2H + stream sync
        kPayloadRecv,     // follower: grouped payload broadcasts
        kHostViews,       // follower: D2H of qo/KV layout + stream sync
        kAttnPlan,        // follower: attention planner (host side)
        kForwardBody,     // follower: forward launch (graph replay or eager)
        kRootBroadcast,   // rank 0: header + payload broadcasts
        kStageCount,
    };
    std::array<std::uint64_t, kStageCount> ns{};
    std::array<std::uint64_t, kStageCount> hits{};

    static constexpr bool enabled() { return false; }

    static TpProfile& instance() {
        static TpProfile p;
        return p;
    }

    ~TpProfile() {
        if (!enabled()) return;
        static const char* names[kStageCount] = {
            "gate_wait", "wake_latency", "start_skew", "header_recv",
            "payload_recv", "host_views", "attn_plan", "forward_body",
            "root_broadcast"};
        for (int i = 0; i < kStageCount; ++i) {
            if (hits[i] == 0) continue;
            std::fprintf(stderr,
                         "[tp-profile] %-15s calls=%-8llu total=%8.1f ms  "
                         "avg=%8.1f us\n",
                         names[i],
                         static_cast<unsigned long long>(hits[i]),
                         static_cast<double>(ns[i]) / 1e6,
                         static_cast<double>(ns[i]) / 1e3 / hits[i]);
        }
    }
};

class TpStageTimer {
  public:
    explicit TpStageTimer(TpProfile::Stage stage)
        : stage_(stage), on_(TpProfile::enabled()) {
        if (on_) start_ = std::chrono::steady_clock::now();
    }
    ~TpStageTimer() { stop(); }

    void stop() {
        if (!on_ || stopped_) return;
        stopped_ = true;
        const auto dt = std::chrono::steady_clock::now() - start_;
        auto& p = TpProfile::instance();
        p.ns[stage_] += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count());
        ++p.hits[stage_];
    }

  private:
    TpProfile::Stage stage_;
    bool on_;
    bool stopped_ = false;
    std::chrono::steady_clock::time_point start_;
};

// Broadcast header sent from rank 0 → followers before each fire's
// per-fire payload. Followers parse it to size the subsequent broadcasts
// + the forward call. Two magic values:
//
//   * TP_FIRE_MAGIC: a regular fire is incoming; payload broadcasts follow.
//   * TP_STOP_MAGIC: shutdown sentinel; follower exits its serve loop.
//
// Trivially copyable so the exact header bytes can be broadcast to followers.
struct TpFireHeader {
    std::int32_t magic;
    std::int32_t total_tokens;
    std::int32_t num_requests;
    std::int32_t is_pure_decode;
    std::int32_t kv_indices_count;
    std::int32_t required_kv_pages;
    std::int32_t mask_bytes;
    std::int32_t mask_indptr_count;
    // 1 = slot_ids[R] (int32) and is_fresh[R] (uint8) follow the
    // existing payload broadcasts. Inert (0) for archs that don't use
    // rs_cache — followers skip those broadcasts.
    std::int32_t has_slot_ids;
    // 1 = w_page[N] and w_off[N] explicit write descriptors follow.
    std::int32_t has_write_desc;
    // Number of compact logit rows in pi.sample_idx.
    std::int32_t logit_rows;
    std::int32_t structured_window_left;
    std::int32_t rs_mode;
    std::int32_t rs_fold_lens_count;
    std::int32_t rs_buffer_ids_count;
    // Length of the buffer READ id array. The other three read-side arrays
    // are fixed by R (indptr R+1, lens R, heads R), so this is the only one
    // the follower cannot derive.
    std::int32_t rs_buffer_read_ids_count;
    // Chosen by rank 0 so both ranks always agree on how this fire's payload
    // travels; the follower never decides for itself.
    std::int32_t transport;
    // Index into the cross-device event ring for this fire.
    std::int32_t payload_slot;
};
static_assert(std::is_trivially_copyable_v<TpFireHeader>);

// Identifiers for every persistent-input buffer that can travel in a fire's
// payload. Both transports below are driven from ONE description keyed by
// these ids, so an NCCL broadcast and a peer push can never disagree about
// what a fire carries.
enum class TpBuf : int {
    kTokens = 0, kPositions, kRowValid, kWPage, kWOff,
    kQoIndptr, kKvPageIndptr, kKvLastPageLens, kKvPageIndices,
    kCustomMask, kCustomMaskIndptr,
    kSlotIds, kIsFresh, kRsSlotFlags,
    kRsFoldLens, kRsBufferSlotIndptr, kRsBufferSlotIds,
    kRsBufferReadIndptr, kRsBufferReadIds, kRsBufferReadLens, kRsBufferHeads,
    kSampleIdx,
    kCount,
};
constexpr int kTpBufCount = static_cast<int>(TpBuf::kCount);

enum : std::int32_t {
    TP_TRANSPORT_NCCL = 0,
};

struct TpPayloadEntry {
    TpBuf id;
    void* local;         // this rank's pointer
    std::size_t bytes;
};

void* tp_buffer_ptr(PersistentInputs& pi, TpBuf id) {
    switch (id) {
        case TpBuf::kTokens:             return pi.tokens.data();
        case TpBuf::kPositions:          return pi.positions.data();
        case TpBuf::kRowValid:           return pi.row_valid.data();
        case TpBuf::kWPage:              return pi.w_page.data();
        case TpBuf::kWOff:               return pi.w_off.data();
        case TpBuf::kQoIndptr:           return pi.qo_indptr.data();
        case TpBuf::kKvPageIndptr:       return pi.kv_page_indptr.data();
        case TpBuf::kKvLastPageLens:     return pi.kv_last_page_lens.data();
        case TpBuf::kKvPageIndices:      return pi.kv_page_indices.data();
        case TpBuf::kCustomMask:         return pi.custom_mask.data();
        case TpBuf::kCustomMaskIndptr:   return pi.custom_mask_indptr.data();
        case TpBuf::kSlotIds:            return pi.slot_ids.data();
        case TpBuf::kIsFresh:            return pi.is_fresh.data();
        case TpBuf::kRsSlotFlags:        return pi.rs_slot_flags.data();
        case TpBuf::kRsFoldLens:         return pi.rs_fold_lens.data();
        case TpBuf::kRsBufferSlotIndptr: return pi.rs_buffer_slot_indptr.data();
        case TpBuf::kRsBufferSlotIds:    return pi.rs_buffer_slot_ids.data();
        case TpBuf::kRsBufferReadIndptr: return pi.rs_buffer_read_indptr.data();
        case TpBuf::kRsBufferReadIds:    return pi.rs_buffer_read_slot_ids.data();
        case TpBuf::kRsBufferReadLens:   return pi.rs_buffer_read_lens.data();
        case TpBuf::kRsBufferHeads:      return pi.rs_buffer_heads.data();
        case TpBuf::kSampleIdx:          return pi.sample_idx.data();
        case TpBuf::kCount:              break;
    }
    return nullptr;
}


// ---------------------------------------------------------------------------
// Process-shared fire mailbox
//
// TP ranks are threads in one process (the CPU gate already depends on that:
// it is a process-global registry, so a cross-process follower could never be
// notified). The fire header and the host-side CSR views the attention planner
// needs are all host-known on rank 0, yet the follower used to obtain them by
// broadcasting to device and copying back with a stream synchronize. That is a
// full host-device round trip whose completion depends on rank 0's matching
// kernel, which sits behind rank 0's entire previous decode step — so the
// follower could never run ahead. Measured at ~2.8 ms per fire and flat across
// batch widths.
//
// Passing the same bytes through shared host memory removes the round trip
// altogether. The gate's publish/wait pair provides the ordering: `publish()`
// stores the sequence with release semantics under its mutex and every waiter
// loads it with acquire, so mailbox writes that precede the notify are visible
// to the follower that observes the epoch.
//
// It also removes a real divergence risk: the follower used to plan attention
// from the DEVICE-written CSRs while rank 0 planned from its host-side upper
// bounds, so the two ranks could pick different attention kernels for the same
// fire. Both now plan from identical values.
// One published fire. The mailbox holds a RING of these: rank 0 runs ahead by
// up to `[model.scheduler] frame_dispatch_depth` frames, so it can be
// publishing fire N+1
// while the follower is still reading fire N. A single shared slot let the two
// overlap and handed the follower a half-overwritten `qo_indptr`, which
// FlashInfer's scheduler rejected ("should be non-negative") and killed the
// follower thread — the process then stalled waiting for a rank that was gone.
struct TpFireSlot {
    TpFireHeader header{};
    std::vector<std::uint32_t> qo_indptr;
    std::vector<std::uint32_t> kv_page_indptr;
    std::vector<std::uint32_t> kv_last_page_lens;
    std::vector<std::uint32_t> kv_page_indices;
};

// Depth is a throughput buffer, NOT a correctness bound: the dispatch depth
// counts FRAMES while this ring is indexed per FIRE, and one frame can carry
// an unbounded number of steps (and one step an unbounded number of MTP
// drafts), so rank 0 can lap any fixed depth inside a single `launch()`.
// `tp_mailbox_reserve_slot` below enforces the invariant; this only sets how
// far rank 0 may run ahead before it has to wait.
constexpr std::uint64_t kTpMailboxRing = 8;

struct TpFireMailbox {
    std::array<TpFireSlot, kTpMailboxRing> slots{};

    // Epochs the follower has finished READING out of the ring (it copies the
    // header and planner views into locals before posting any collective, so
    // it always releases a slot without needing rank 0 to progress first —
    // that is what makes the reserve below deadlock-free).
    std::atomic<std::uint64_t> consumed_fires{0};
    // Set by rank 0 immediately before the gate notify; the follower diffs it
    // on wake to separate "how long the wake took" from "how long my own work
    // takes". Plain store/load: the gate's release/acquire pair orders it.
    std::chrono::steady_clock::time_point notify_time{};
    std::uint64_t fire_seq = 0;   // rank 0 only
};

// Set `PIE_TP_WATCHDOG=1` to have rank 0 report the publish/consume cursors
// whenever they stop advancing. A TP hang is otherwise silent: both ranks are
// parked and neither reports anything, and this says immediately whether the
// follower fell behind, ran ahead, or never woke.
struct TpStallWatchdog {
    std::atomic<std::uint64_t> published{0};
    std::atomic<std::uint64_t> consumed{0};
    std::atomic<int> rank0_phase{-1};   // 0=publish 1=broadcast 2=settle
    std::atomic<std::uint64_t> follower_forwards{0};
    std::array<std::atomic<std::uint64_t>, 8> collectives{};
    std::atomic<std::uint64_t> rank0_phase_seq{0};
    std::atomic<bool> running{false};
    std::thread thread;
    static constexpr bool enabled() { return false; }
    static TpStallWatchdog& instance() {
        static TpStallWatchdog w;
        return w;
    }
    void start() {
        if (!enabled() || running.exchange(true)) return;
        thread = std::thread([this] {
            std::uint64_t last_pub = ~0ull, last_con = ~0ull;
            int stuck = 0;
            static const char* kPhase[] = {"publish", "broadcast",
                                           "enqueue_done", "in_settle",
                                           "settle_done"};
            while (running.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                const auto p = published.load(std::memory_order_acquire);
                const auto c = consumed.load(std::memory_order_acquire);
                if (p == last_pub && c == last_con) {
                    const int ph = rank0_phase.load(std::memory_order_acquire);
                    // Every rank, not just the first two: at tp>2 the whole
                    // question is *which* rank stopped arriving, and a pair of
                    // counters cannot name it.
                    char coll[256];
                    int off = 0;
                    for (std::size_t r = 0; r < collectives.size(); ++r) {
                        const auto n = collectives[r].load();
                        if (n == 0) continue;
                        off += std::snprintf(coll + off, sizeof(coll) - off,
                                             "%sr%zu=%llu", off ? " " : "", r,
                                             (unsigned long long)n);
                    }
                    std::fprintf(stderr,
                        "[tp-watchdog] STALLED %ds: published=%llu consumed=%llu "
                        "(delta=%lld) rank0_last_phase=%s seq=%llu "
                        "follower_forwards=%llu collectives=[%s]\n",
                        (++stuck) * 5,
                        (unsigned long long)p, (unsigned long long)c,
                        (long long)(p - c),
                        (ph >= 0 && ph < 5) ? kPhase[ph] : "none",
                        (unsigned long long)rank0_phase_seq.load(),
                        (unsigned long long)follower_forwards.load(),
                        coll);
                } else {
                    stuck = 0;
                }
                last_pub = p; last_con = c;
            }
        });
        thread.detach();
    }
};

std::mutex g_tp_mailboxes_mu;
std::unordered_map<std::string, std::shared_ptr<TpFireMailbox>> g_tp_mailboxes;


std::shared_ptr<TpFireMailbox> tp_mailbox_for(const std::string& key) {
    std::lock_guard<std::mutex> lk(g_tp_mailboxes_mu);
    auto& box = g_tp_mailboxes[key];
    if (!box) box = std::make_shared<TpFireMailbox>();
    return box;
}

template <typename T>
void assign_span(std::vector<T>& dst, const T* src, std::size_t count) {
    dst.resize(count);
    if (count != 0 && src != nullptr) {
        std::memcpy(dst.data(), src, count * sizeof(T));
    }
}

constexpr std::int32_t TP_FIRE_MAGIC = 0x55504954;  // 'TPIU' tag
constexpr std::int32_t TP_MTP_MAGIC  = 0x50544D54;  // 'TMTP' tag
constexpr std::int32_t TP_STOP_MAGIC = 0x504F5453;  // 'STOP' tag

// Lazily allocated device buffer holding the broadcast header.
// Both rank 0 and followers reuse it across fires; no need to plumb it
// through BatchEngine.
std::int32_t* tp_hdr_dev_buf() {
    thread_local std::int32_t* buf = nullptr;
    if (buf == nullptr) {
        CUDA_CHECK(cudaMalloc(&buf, sizeof(TpFireHeader)));
    }
    return buf;
}

// Pinned staging for the fire header. `cudaMemcpyAsync` against PAGEABLE host
// memory is synchronous: the driver stages it through its own pinned buffer
// and waits, which drains everything already queued on the stream. With a
// wide decode step in flight that turned each header hop into a full pipeline
// drain — measured at 2.2 ms per fire on rank 0 and 3.1 ms on the follower.
// Page-locking the staging buffer keeps the copy asynchronous.
TpFireHeader* tp_hdr_host_buf() {
    thread_local TpFireHeader* buf = nullptr;
    if (buf == nullptr) {
        void* raw = nullptr;
        CUDA_CHECK(cudaMallocHost(&raw, sizeof(TpFireHeader)));
        buf = static_cast<TpFireHeader*>(raw);
    }
    return buf;
}

// Rank 0 may not overwrite a ring slot the follower has not finished reading.
// Without this the ring is only a heuristic: `launch()` runs `enqueue_step` for
// every step of a frame back to back, and `enqueue_mtp_draft_logits` publishes
// once per (work item x draft), both in tight host loops with no rank
// synchronisation, so rank 0 laps the ring and `assign_span`'s `resize` can
// free the very vector the follower is memcpy-ing out of.
//
// Deadlock-free by construction: the follower copies the header and the
// planner views into locals and releases the slot BEFORE it posts any
// collective, so it never needs rank 0 to make progress first.
bool tp_cpu_gate_stopped(const std::string& key) {
    if (key.empty()) return false;
    return tp_cpu_gate_for(key)->stopped();
}

void tp_mailbox_reserve_slot(TpFireMailbox& box, const std::string& key) {
    if (box.fire_seq < kTpMailboxRing) return;  // ring not full yet
    const std::uint64_t oldest_live = box.fire_seq - kTpMailboxRing + 1;
    if (box.consumed_fires.load(std::memory_order_acquire) >= oldest_live) {
        return;
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (int spin = 0;; ++spin) {
        if (box.consumed_fires.load(std::memory_order_acquire) >= oldest_live) {
            return;
        }
        // A stopped gate means the follower is gone (teardown, or a failed
        // frame fail-stopped the group). Returning lets the caller proceed to
        // its collective, which then fails loudly instead of hanging here.
        if (tp_cpu_gate_stopped(key)) return;
        if (spin < 64) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
        if (std::chrono::steady_clock::now() > deadline) {
            throw std::runtime_error(
                "tp: mailbox ring stalled - the follower stopped consuming "
                "fires (published " + std::to_string(box.fire_seq) +
                ", consumed " +
                std::to_string(box.consumed_fires.load()) + ")");
        }
    }
}

}  // namespace

void tp_publish_fire(const std::string& cpu_gate_key,
                     const TpFirePlanViews& views,
                     int N, int R, bool is_pure_decode,
                     int kv_indices_count,
                     int required_kv_pages,
                     int mask_bytes, int mask_indptr_count,
                     bool has_slot_ids,
                     bool has_write_desc,
                     int logit_rows,
                     int structured_window_left,
                     RsExecutionMode rs_mode,
                     int rs_fold_lens_count,
                     int rs_buffer_ids_count,
                     int rs_buffer_read_ids_count) {
    if (cpu_gate_key.empty()) return;
    // Before anything is written into the mailbox: a fire published to a group
    // that has lost a rank can never complete.
    tp_check_ranks_alive();
    auto box = tp_mailbox_for(cpu_gate_key);
    TpStallWatchdog::instance().start();
    tp_mailbox_reserve_slot(*box, cpu_gate_key);
    TpFireSlot& fire = box->slots[box->fire_seq % kTpMailboxRing];
    fire.header = TpFireHeader{
        TP_FIRE_MAGIC, N, R, is_pure_decode ? 1 : 0,
        kv_indices_count, required_kv_pages, mask_bytes, mask_indptr_count,
        has_slot_ids ? 1 : 0,
        has_write_desc ? 1 : 0,
        logit_rows,
        structured_window_left,
        static_cast<std::int32_t>(rs_mode),
        rs_fold_lens_count,
        rs_buffer_ids_count,
        rs_buffer_read_ids_count,
        TP_TRANSPORT_NCCL,
        0,
    };
    const std::size_t requests = static_cast<std::size_t>(std::max(R, 0));
    assign_span(fire.qo_indptr, views.qo_indptr, requests + 1);
    assign_span(fire.kv_page_indptr, views.kv_page_indptr, requests + 1);
    assign_span(fire.kv_last_page_lens, views.kv_last_page_lens, requests);
    assign_span(fire.kv_page_indices, views.kv_page_indices,
                views.kv_page_indices_len);
    ++box->fire_seq;
    {
        auto& wd = TpStallWatchdog::instance();
        wd.published.store(box->fire_seq, std::memory_order_release);
        wd.rank0_phase.store(0, std::memory_order_release);
        wd.rank0_phase_seq.store(box->fire_seq, std::memory_order_release);
    }
    if (TpStallWatchdog::enabled()) {
        std::fprintf(stderr,
            "[tp-hdr] R0 seq=%llu N=%d R=%d pure=%d kvidx=%d mask=%d "
            "mask_indptr=%d slots=%d logit_rows=%d rs=%d\n",
            (unsigned long long)box->fire_seq, N, R, is_pure_decode ? 1 : 0,
            kv_indices_count, mask_bytes, mask_indptr_count,
            has_slot_ids ? 1 : 0, logit_rows, static_cast<int>(rs_mode));
    }
    box->notify_time = std::chrono::steady_clock::now();
    // Release/acquire on the gate publishes everything written above.
    tp_cpu_gate_notify(cpu_gate_key);
}

void tp_broadcast_inputs(NcclComm& comm, PersistentInputs& pi,
                         const std::string& cpu_gate_key,
                         const TpFirePlanViews& views,
                         int N, int R, bool is_pure_decode,
                         int kv_indices_count,
                         int required_kv_pages,
                         int mask_bytes, int mask_indptr_count,
                         bool has_slot_ids,
                         bool has_write_desc,
                         int logit_rows,
                         int structured_window_left,
                         RsExecutionMode rs_mode,
                         int rs_fold_lens_count,
                         int rs_buffer_ids_count,
                         int rs_buffer_read_ids_count,
                         cudaStream_t stream)
{
    if (mask_bytes < 0 || mask_indptr_count < 0 ||
        (mask_indptr_count != 0 && mask_indptr_count != R + 1) ||
        static_cast<std::size_t>(mask_bytes) > pi.custom_mask.size() ||
        static_cast<std::size_t>(mask_indptr_count) >
            pi.custom_mask_indptr.size()) {
        throw std::runtime_error(
            "TP root custom mask metadata exceeds persistent capacity");
    }
    TpStageTimer root_timer(TpProfile::kRootBroadcast);
    tp_check_ranks_alive();
    const TpFireHeader header{
        TP_FIRE_MAGIC, N, R, is_pure_decode ? 1 : 0,
        kv_indices_count, required_kv_pages, mask_bytes, mask_indptr_count,
        has_slot_ids ? 1 : 0,
        has_write_desc ? 1 : 0,
        logit_rows,
        structured_window_left,
        static_cast<std::int32_t>(rs_mode),
        rs_fold_lens_count,
        rs_buffer_ids_count,
        rs_buffer_read_ids_count,
    };
    const bool via_mailbox = !cpu_gate_key.empty();
    if (!via_mailbox) {
        // Gate off: the follower is parked inside `ncclBroadcast`, so the
        // header still has to travel through the device.
        auto* d_hdr = tp_hdr_dev_buf();
        TpFireHeader* hdr_host = tp_hdr_host_buf();
        *hdr_host = header;
        CUDA_CHECK(cudaMemcpyAsync(d_hdr, hdr_host, sizeof(TpFireHeader),
                                   cudaMemcpyHostToDevice, stream));
        NCCL_CHECK_ASYNC(
            ncclBroadcast(d_hdr, d_hdr, sizeof(TpFireHeader), ncclChar, 0,
                          comm.comm(), stream),
            comm.comm());
    }
    // Group the payload broadcasts so NCCL submits them as a single batch
    // — tens of microseconds of host-side launch overhead saved per fire,
    // most visible at small batch sizes (decode where each broadcast is
    // sub-KB but the fixed per-op cost dominates).
    const bool state_only_fold =
        rs_mode == RsExecutionMode::BufferFold;
    NCCL_CHECK(ncclGroupStart());
    if (!state_only_fold) {
        NCCL_CHECK(ncclBroadcast(pi.tokens.data(), pi.tokens.data(),
                                 static_cast<std::size_t>(N) * 4, ncclChar, 0,
                                 comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.positions.data(), pi.positions.data(),
                                 static_cast<std::size_t>(N) * 4, ncclChar, 0,
                                 comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(
            pi.row_valid.data(), pi.row_valid.data(),
            static_cast<std::size_t>(N), ncclChar, 0,
            comm.comm(), stream));
    }
    if (!state_only_fold && has_write_desc && N > 0) {
        NCCL_CHECK(ncclBroadcast(
            pi.w_page.data(), pi.w_page.data(),
            static_cast<std::size_t>(N) * 4, ncclChar, 0,
            comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(
            pi.w_off.data(), pi.w_off.data(),
            static_cast<std::size_t>(N) * 4, ncclChar, 0,
            comm.comm(), stream));
    }
    NCCL_CHECK(ncclBroadcast(pi.qo_indptr.data(), pi.qo_indptr.data(),
                             static_cast<std::size_t>(R + 1) * 4, ncclChar, 0,
                             comm.comm(), stream));
    if (!state_only_fold) {
        NCCL_CHECK(ncclBroadcast(
            pi.kv_page_indptr.data(), pi.kv_page_indptr.data(),
            static_cast<std::size_t>(R + 1) * 4, ncclChar, 0,
            comm.comm(), stream));
    }
    if (!state_only_fold && R > 0) {
        NCCL_CHECK(ncclBroadcast(pi.kv_last_page_lens.data(),
                                 pi.kv_last_page_lens.data(),
                                 static_cast<std::size_t>(R) * 4, ncclChar, 0,
                                 comm.comm(), stream));
    }
    if (!state_only_fold && kv_indices_count > 0) {
        if (static_cast<std::size_t>(kv_indices_count) >
            pi.kv_page_indices.size()) {
            throw std::runtime_error(
                "tp: kv_page_indices broadcast count " +
                std::to_string(kv_indices_count) + " exceeds device capacity " +
                std::to_string(pi.kv_page_indices.size()));
        }
        NCCL_CHECK(ncclBroadcast(pi.kv_page_indices.data(),
                                 pi.kv_page_indices.data(),
                                 static_cast<std::size_t>(kv_indices_count) * 4,
                                 ncclChar, 0, comm.comm(), stream));
    }
    if (!state_only_fold && mask_indptr_count > 0) {
        if (mask_bytes > 0) {
            NCCL_CHECK(ncclBroadcast(
                pi.custom_mask.data(), pi.custom_mask.data(),
                static_cast<std::size_t>(mask_bytes), ncclChar, 0,
                comm.comm(), stream));
        }
        NCCL_CHECK(ncclBroadcast(pi.custom_mask_indptr.data(),
                                 pi.custom_mask_indptr.data(),
                                 static_cast<std::size_t>(mask_indptr_count) * 4,
                                 ncclChar, 0, comm.comm(), stream));
    }
    if (has_slot_ids && R > 0) {
        NCCL_CHECK(ncclBroadcast(pi.slot_ids.data(), pi.slot_ids.data(),
                                 static_cast<std::size_t>(R) * 4, ncclChar, 0,
                                 comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.is_fresh.data(), pi.is_fresh.data(),
                                 static_cast<std::size_t>(R), ncclChar, 0,
                                 comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(
            pi.rs_slot_flags.data(), pi.rs_slot_flags.data(),
            static_cast<std::size_t>(R), ncclChar, 0, comm.comm(), stream));
    }
    if (rs_fold_lens_count > 0) {
        NCCL_CHECK(ncclBroadcast(
            pi.rs_fold_lens.data(), pi.rs_fold_lens.data(),
            static_cast<std::size_t>(rs_fold_lens_count) *
                sizeof(std::uint32_t),
            ncclChar, 0, comm.comm(), stream));
    }
    if ((rs_mode == RsExecutionMode::BufferWrite ||
         rs_mode == RsExecutionMode::BufferFold) &&
        R >= 0) {
        NCCL_CHECK(ncclBroadcast(
            pi.rs_buffer_slot_indptr.data(),
            pi.rs_buffer_slot_indptr.data(),
            static_cast<std::size_t>(R + 1) * sizeof(std::uint32_t),
            ncclChar, 0, comm.comm(), stream));
        if (rs_buffer_ids_count > 0) {
            NCCL_CHECK(ncclBroadcast(
                pi.rs_buffer_slot_ids.data(),
                pi.rs_buffer_slot_ids.data(),
                static_cast<std::size_t>(rs_buffer_ids_count) *
                    sizeof(std::uint32_t),
                ncclChar, 0, comm.comm(), stream));
        }
        // The buffer READ side. A follower that does not get it skips the
        // replay of the buffered prefix entirely and folds a DIFFERENT
        // recurrent state than rank 0 -- silently, since the all-reduce
        // happily mixes the two.
        NCCL_CHECK(ncclBroadcast(
            pi.rs_buffer_read_indptr.data(), pi.rs_buffer_read_indptr.data(),
            static_cast<std::size_t>(R + 1) * sizeof(std::uint32_t),
            ncclChar, 0, comm.comm(), stream));
        if (R > 0) {
            NCCL_CHECK(ncclBroadcast(
                pi.rs_buffer_read_lens.data(), pi.rs_buffer_read_lens.data(),
                static_cast<std::size_t>(R) * sizeof(std::uint32_t),
                ncclChar, 0, comm.comm(), stream));
            NCCL_CHECK(ncclBroadcast(
                pi.rs_buffer_heads.data(), pi.rs_buffer_heads.data(),
                static_cast<std::size_t>(R) * sizeof(std::uint32_t),
                ncclChar, 0, comm.comm(), stream));
        }
        if (rs_buffer_read_ids_count > 0) {
            NCCL_CHECK(ncclBroadcast(
                pi.rs_buffer_read_slot_ids.data(),
                pi.rs_buffer_read_slot_ids.data(),
                static_cast<std::size_t>(rs_buffer_read_ids_count) *
                    sizeof(std::uint32_t),
                ncclChar, 0, comm.comm(), stream));
        }
    }
    if (!state_only_fold && logit_rows > 0) {
        NCCL_CHECK(ncclBroadcast(pi.sample_idx.data(), pi.sample_idx.data(),
                                 static_cast<std::size_t>(logit_rows) * 4,
                                 ncclChar, 0, comm.comm(), stream));
    }
    NCCL_CHECK_ASYNC(ncclGroupEnd(), comm.comm());
}

// MTP draft steps ALSO have to go through the mailbox. The follower derives
// its slot from the number of gate epochs it has consumed, so a bare
// `tp_cpu_gate_notify` that publishes nothing would leave rank 0's `fire_seq`
// permanently behind the follower's epoch count and every later fire would
// read the wrong slot — a stale header, or a zeroed one that trips the
// "unexpected header magic" abort and kills the follower thread.
void tp_publish_mtp(const std::string& cpu_gate_key,
                    int rows,
                    int draft_step,
                    int max_global_tokens) {
    if (cpu_gate_key.empty()) return;
    tp_check_ranks_alive();
    auto box = tp_mailbox_for(cpu_gate_key);
    tp_mailbox_reserve_slot(*box, cpu_gate_key);
    TpFireSlot& fire = box->slots[box->fire_seq % kTpMailboxRing];
    fire.header = TpFireHeader{};
    fire.header.magic = TP_MTP_MAGIC;
    fire.header.total_tokens = rows;
    fire.header.num_requests = draft_step;
    fire.header.is_pure_decode = max_global_tokens;
    fire.header.structured_window_left = -2;
    fire.header.transport = TP_TRANSPORT_NCCL;
    ++box->fire_seq;
    auto& wd = TpStallWatchdog::instance();
    wd.published.store(box->fire_seq, std::memory_order_release);
    wd.rank0_phase.store(0, std::memory_order_release);
    wd.rank0_phase_seq.store(box->fire_seq, std::memory_order_release);
    box->notify_time = std::chrono::steady_clock::now();
    tp_cpu_gate_notify(cpu_gate_key);
}

void tp_broadcast_mtp_step(
    NcclComm& comm,
    PersistentInputs& pi,
    const std::string& cpu_gate_key,
    int rows,
    int draft_step,
    int max_global_tokens,
    cudaStream_t stream) {
    // Gate-off configurations never touch the mailbox, so this is their only
    // liveness check before the collectives below.
    tp_check_ranks_alive();
    if (cpu_gate_key.empty()) {
        // Gate off: the follower is parked inside the header broadcast.
        auto* device_header = tp_hdr_dev_buf();
        TpFireHeader* header = tp_hdr_host_buf();
        *header = TpFireHeader{};
        header->magic = TP_MTP_MAGIC;
        header->total_tokens = rows;
        header->num_requests = draft_step;
        header->is_pure_decode = max_global_tokens;
        header->structured_window_left = -2;
        CUDA_CHECK(cudaMemcpyAsync(
            device_header, header, sizeof(TpFireHeader),
            cudaMemcpyHostToDevice, stream));
        NCCL_CHECK_ASYNC(
            ncclBroadcast(
                device_header, device_header, sizeof(TpFireHeader), ncclChar,
                0, comm.comm(), stream),
            comm.comm());
    }
    NCCL_CHECK(ncclGroupStart());
    for (void* buffer : {
             static_cast<void*>(pi.tokens.data()),
             static_cast<void*>(pi.positions.data()),
             static_cast<void*>(pi.sample_idx.data()),
             static_cast<void*>(pi.mtp_request_ids.data())}) {
        NCCL_CHECK(ncclBroadcast(
            buffer, buffer, static_cast<std::size_t>(rows) * 4,
            ncclChar, 0, comm.comm(), stream));
    }
    NCCL_CHECK_ASYNC(ncclGroupEnd(), comm.comm());
}

void tp_watchdog_count_collective(int rank) {
    if (!TpStallWatchdog::enabled()) return;
    if (rank < 0 || rank >= 8) return;
    TpStallWatchdog::instance().collectives[
        static_cast<std::size_t>(rank)].fetch_add(1,
            std::memory_order_relaxed);
}

void tp_watchdog_mark_phase(int phase) {
    TpStallWatchdog::instance().rank0_phase.store(
        phase, std::memory_order_release);
}

void tp_cpu_gate_notify(const std::string& key) {
    if (key.empty()) return;
    auto gate = tp_cpu_gate_for(key);
    gate->publish();
}

void tp_cpu_gate_request_stop(const std::string& key) {
    if (key.empty()) return;
    // Keep the registry entry: `tp_cpu_gate_wait` looks the gate up by key on
    // every iteration and would otherwise default-construct a fresh, un-stopped
    // gate that nothing can ever notify — a follower looping back after its
    // last fire would park on it forever.
    auto gate = tp_cpu_gate_for(key);
    gate->request_stop();
}

namespace detail {
std::atomic<bool> g_tp_rank_failed{false};
}  // namespace detail

namespace {
std::mutex g_tp_rank_failure_mu;
std::string g_tp_rank_failure_msg;
}  // namespace

void tp_throw_rank_failure() {
    std::string msg;
    {
        std::lock_guard<std::mutex> lk(g_tp_rank_failure_mu);
        msg = g_tp_rank_failure_msg;
    }
    throw std::runtime_error(
        msg.empty() ? std::string("tp: a rank left the group") : msg);
}

void tp_report_rank_failure(const std::string& cpu_gate_key,
                            int rank,
                            const std::string& what) {
    bool first = false;
    {
        std::lock_guard<std::mutex> lk(g_tp_rank_failure_mu);
        if (g_tp_rank_failure_msg.empty()) {
            g_tp_rank_failure_msg =
                "tp rank " + std::to_string(rank) +
                " left the group and every later fire would hang: " + what;
            first = true;
        }
    }
    // Release: the message above must be visible to whoever observes the flag.
    detail::g_tp_rank_failed.store(true, std::memory_order_release);
    if (!first) return;
    std::cerr << "[pie-driver-cuda] tp rank " << rank
              << " left the group: " << what
              << " — failing the TP group\n";
    // Peers parked on the gate wake and see a stopped gate.
    tp_cpu_gate_request_stop(cpu_gate_key);
    // Peers already inside a collective this rank will never join are waiting
    // on the device, where no flag can reach them. Aborting the communicators
    // makes those waits return an error, which is the difference between a
    // reported failure and a hang.
    nccl_abort_all_comms();
}

// ============================================================================
// TP follower service loop
// ============================================================================
//
// Symmetric counterpart of the frame step pipeline for ranks > 0:
//
//   * Inputs arrive via NCCL broadcast from rank 0.
//   * No sampling — only rank 0 owns the direct PTIR publish path.
//   * Graph capture/replay mirrors rank 0 for graph-safe pure decode so
//     NCCL collectives inside the body enter capture or replay on every
//     rank in the same order.
//
// The loop blocks on `ncclBroadcast` for the header. NCCL serialises ops
// per-comm, so a follower naturally idles until rank 0 issues the
// matching broadcast in `tp_broadcast_inputs`.
void tp_follower_serve(BatchEngine& engine, std::atomic<bool>& stop) {
    if (engine.tp_comm == nullptr) {
        throw std::runtime_error("tp_follower_serve: no tp_comm");
    }
    auto& pi      = engine.inputs;
    auto& comm    = *engine.tp_comm;
    auto* d_hdr   = tp_hdr_dev_buf();
    cudaStream_t stream = nullptr;
    std::uint64_t cpu_gate_seq = 0;
    if (engine.runtime_quant_context == nullptr) {
        throw std::runtime_error(
            "TP follower has no runtime-quant context");
    }
    ops::ScopedRuntimeQuantContext quant_scope(
        *engine.runtime_quant_context);

    // Sized lazily; R is at most max_workspace_tokens (one request per token).
    std::vector<std::uint32_t> h_qo, h_kvpp;
    // Non-null whenever the CPU gate is on, which is also the only
    // configuration in which TP ranks share this process.
    const std::shared_ptr<TpFireMailbox> mailbox =
        engine.tp_cpu_gate_key.empty()
            ? nullptr
            : tp_mailbox_for(engine.tp_cpu_gate_key);

    while (!stop.load()) {
        // A false return means shutdown, not a fire. Posting the header
        // broadcast anyway would block forever on a collective rank 0 will
        // never match, which is what previously wedged teardown.
        {
            TpStageTimer gate_timer(TpProfile::kGateWait);
            if (!tp_cpu_gate_wait(engine.tp_cpu_gate_key, cpu_gate_seq, stop)) {
                break;
            }
        }
        if (mailbox != nullptr && TpProfile::enabled()) {
            const auto now = std::chrono::steady_clock::now();
            auto& prof = TpProfile::instance();
            prof.ns[TpProfile::kWakeLatency] += static_cast<std::uint64_t>(
                std::max<std::int64_t>(0,
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        now - mailbox->notify_time).count()));
            ++prof.hits[TpProfile::kWakeLatency];
        }
        // 1. Receive header. This is a full host-device round trip on the
        //    critical path: the follower cannot size any payload broadcast
        //    until it has the header's contents host-side.
        TpFireHeader hdr{};
        std::uint64_t fire_slot = 0;
        {
            TpStageTimer header_timer(TpProfile::kHeaderRecv);
            if (mailbox != nullptr) {
                // Straight out of host memory: no collective, no readback and
                // no stream synchronize, so the follower is free to enqueue
                // its receives without waiting on rank 0's GPU queue.
                // `cpu_gate_seq` counts consumed epochs starting at 1, and
                // rank 0 published epoch k into slot (k-1) % ring.
                fire_slot = (cpu_gate_seq + kTpMailboxRing - 1) %
                            kTpMailboxRing;
                hdr = mailbox->slots[fire_slot].header;
            } else {
                TpFireHeader* hdr_host = tp_hdr_host_buf();
                NCCL_CHECK_ASYNC(
                    ncclBroadcast(d_hdr, d_hdr, sizeof(TpFireHeader),
                                  ncclChar, 0, comm.comm(), stream),
                    comm.comm());
                CUDA_CHECK(cudaMemcpyAsync(hdr_host, d_hdr,
                                           sizeof(TpFireHeader),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                hdr = *hdr_host;
            }
        }

        if (TpStallWatchdog::enabled() && hdr.magic == TP_FIRE_MAGIC) {
            std::fprintf(stderr,
                "[tp-hdr] R1 seq=%llu N=%d R=%d pure=%d kvidx=%d mask=%d "
                "mask_indptr=%d slots=%d logit_rows=%d rs=%d slot=%llu\n",
                (unsigned long long)cpu_gate_seq, hdr.total_tokens,
                hdr.num_requests, hdr.is_pure_decode, hdr.kv_indices_count,
                hdr.mask_bytes, hdr.mask_indptr_count, hdr.has_slot_ids,
                hdr.logit_rows, hdr.rs_mode,
                (unsigned long long)fire_slot);
        }
        {
            static const std::uint64_t kill_at = [] {
                const char* v = std::getenv("PIE_TP_TEST_KILL_FOLLOWER_AT_FIRE");
                return v ? std::strtoull(v, nullptr, 10) : 0ull;
            }();
            if (kill_at != 0 && cpu_gate_seq == kill_at) {
                throw std::runtime_error("injected follower fault");
            }
        }
        if (hdr.magic == TP_STOP_MAGIC) break;
        if (hdr.magic == TP_MTP_MAGIC) {
            // Header is already copied out, and an MTP fire carries no planner
            // views, so the slot is free here. Release it BEFORE the
            // collectives below, or rank 0 (which is what unblocks them) could
            // be waiting on this very slot.
            if (mailbox != nullptr) {
                mailbox->consumed_fires.store(cpu_gate_seq,
                                              std::memory_order_release);
                TpStallWatchdog::instance().consumed.store(
                    cpu_gate_seq, std::memory_order_release);
            }
            const int rows = hdr.total_tokens;
            NCCL_CHECK(ncclGroupStart());
            for (void* buffer : {
                     static_cast<void*>(pi.tokens.data()),
                     static_cast<void*>(pi.positions.data()),
                     static_cast<void*>(pi.sample_idx.data()),
                     static_cast<void*>(pi.mtp_request_ids.data())}) {
                NCCL_CHECK(ncclBroadcast(
                    buffer, buffer, static_cast<std::size_t>(rows) * 4,
                    ncclChar, 0, comm.comm(), stream));
            }
            NCCL_CHECK_ASYNC(ncclGroupEnd(), comm.comm());
            if (!engine.system_drafter.draft_step) {
                throw std::runtime_error(
                    "TP follower received MTP work without a native draft head");
            }
            engine.system_drafter.draft_step(
                engine.ws,
                engine.kv_cache,
                engine.cublas,
                reinterpret_cast<const std::int32_t*>(pi.tokens.data()),
                reinterpret_cast<const std::int32_t*>(pi.positions.data()),
                pi.sample_idx.data(),
                pi.mtp_request_ids.data(),
                pi.kv_page_indices.data(),
                pi.kv_page_indptr.data(),
                pi.kv_last_page_lens.data(),
                nullptr,
                rows,
                hdr.num_requests,
                hdr.is_pure_decode);
            continue;
        }
        if (hdr.magic != TP_FIRE_MAGIC) {
            std::cerr << "[pie-driver-cuda] tp follower: unexpected header "
                      << "magic 0x" << std::hex << hdr.magic << std::dec
                      << "; aborting\n";
            break;
        }

        const int N = hdr.total_tokens;
        const int R = hdr.num_requests;
        if (hdr.required_kv_pages < 0 ||
            hdr.required_kv_pages > engine.kv_cache.num_pages()) {
            throw std::runtime_error(
                "TP follower received invalid required KV page high-water");
        }
        engine.kv_cache.ensure_pages(hdr.required_kv_pages);
        const bool is_pure_decode = (hdr.is_pure_decode != 0);
        const bool has_write_desc = hdr.has_write_desc != 0;
        const int logit_rows = hdr.logit_rows;
        const int structured_window_left =
            hdr.structured_window_left;
        if (!valid_rs_execution_mode(hdr.rs_mode) ||
            hdr.rs_fold_lens_count < 0 ||
            hdr.rs_buffer_ids_count < 0 ||
            hdr.rs_buffer_read_ids_count < 0) {
            throw std::runtime_error(
                "TP follower received invalid RS metadata header");
        }
        const RsExecutionMode rs_mode =
            static_cast<RsExecutionMode>(hdr.rs_mode);
        const bool state_only_fold =
            rs_mode == RsExecutionMode::BufferFold;
        const int rs_fold_lens_count = hdr.rs_fold_lens_count;
        const int rs_buffer_ids_count = hdr.rs_buffer_ids_count;
        const int rs_buffer_read_ids_count = hdr.rs_buffer_read_ids_count;
        const std::size_t rs_rows =
            static_cast<std::size_t>(std::max(R, 0));
        const bool header_has_slots = hdr.has_slot_ids != 0;
        const bool header_buffered =
            rs_mode == RsExecutionMode::BufferWrite ||
            rs_mode == RsExecutionMode::BufferFold;
        if (!tp_rs_metadata_shape_valid(
                rs_mode,
                rs_rows,
                header_has_slots ? rs_rows : 0,
                header_has_slots ? rs_rows : 0,
                static_cast<std::size_t>(rs_fold_lens_count),
                static_cast<std::size_t>(rs_buffer_ids_count),
                header_buffered ? rs_rows + 1 : 0)) {
            throw std::runtime_error(
                "TP follower received inconsistent RS metadata header");
        }
        if (static_cast<std::size_t>(std::max(R, 0)) >
                pi.rs_fold_lens.size() ||
            static_cast<std::size_t>(rs_fold_lens_count) >
                pi.rs_fold_lens.size() ||
            static_cast<std::size_t>(rs_buffer_ids_count) >
                pi.rs_buffer_slot_ids.size() ||
            static_cast<std::size_t>(rs_buffer_read_ids_count) >
                pi.rs_buffer_read_slot_ids.size()) {
            throw std::runtime_error(
                "TP follower RS metadata exceeds persistent capacity");
        }

        // 2. Receive payloads. Mirror order in `tp_broadcast_inputs`,
        //    grouped so NCCL submits the batch as a single op.
        if (hdr.mask_bytes < 0 || hdr.mask_indptr_count < 0 ||
            (hdr.mask_indptr_count != 0 &&
             hdr.mask_indptr_count != R + 1) ||
            static_cast<std::size_t>(hdr.mask_bytes) >
                pi.custom_mask.size() ||
            static_cast<std::size_t>(hdr.mask_indptr_count) >
                pi.custom_mask_indptr.size()) {
            throw std::runtime_error(
                "TP follower custom mask exceeds persistent capacity");
        }
        const bool have_custom_mask =
            !state_only_fold && hdr.mask_indptr_count > 0;
        const bool have_slot_ids = (hdr.has_slot_ids != 0) && R > 0;
        TpStageTimer payload_timer(TpProfile::kPayloadRecv);
        NCCL_CHECK(ncclGroupStart());
        if (!state_only_fold) {
            NCCL_CHECK(ncclBroadcast(pi.tokens.data(), pi.tokens.data(),
                                     static_cast<std::size_t>(N) * 4,
                                     ncclChar, 0, comm.comm(), stream));
            NCCL_CHECK(ncclBroadcast(pi.positions.data(), pi.positions.data(),
                                     static_cast<std::size_t>(N) * 4,
                                     ncclChar, 0, comm.comm(), stream));
            NCCL_CHECK(ncclBroadcast(
                pi.row_valid.data(), pi.row_valid.data(),
                static_cast<std::size_t>(N), ncclChar, 0,
                comm.comm(), stream));
        }
        if (!state_only_fold && has_write_desc && N > 0) {
            NCCL_CHECK(ncclBroadcast(
                pi.w_page.data(), pi.w_page.data(),
                static_cast<std::size_t>(N) * 4, ncclChar, 0,
                comm.comm(), stream));
            NCCL_CHECK(ncclBroadcast(
                pi.w_off.data(), pi.w_off.data(),
                static_cast<std::size_t>(N) * 4, ncclChar, 0,
                comm.comm(), stream));
        }
        NCCL_CHECK(ncclBroadcast(pi.qo_indptr.data(), pi.qo_indptr.data(),
                                 static_cast<std::size_t>(R + 1) * 4,
                                 ncclChar, 0, comm.comm(), stream));
        if (!state_only_fold) {
            NCCL_CHECK(ncclBroadcast(
                pi.kv_page_indptr.data(), pi.kv_page_indptr.data(),
                static_cast<std::size_t>(R + 1) * 4,
                ncclChar, 0, comm.comm(), stream));
        }
        if (!state_only_fold && R > 0) {
            NCCL_CHECK(ncclBroadcast(pi.kv_last_page_lens.data(),
                                     pi.kv_last_page_lens.data(),
                                     static_cast<std::size_t>(R) * 4,
                                     ncclChar, 0, comm.comm(), stream));
        }
        if (!state_only_fold && hdr.kv_indices_count > 0) {
            NCCL_CHECK(ncclBroadcast(pi.kv_page_indices.data(),
                                     pi.kv_page_indices.data(),
                                     static_cast<std::size_t>(hdr.kv_indices_count) * 4,
                                     ncclChar, 0, comm.comm(), stream));
        }
        if (!state_only_fold && have_custom_mask) {
            if (hdr.mask_bytes > 0) {
                NCCL_CHECK(ncclBroadcast(
                    pi.custom_mask.data(), pi.custom_mask.data(),
                    static_cast<std::size_t>(hdr.mask_bytes),
                    ncclChar, 0, comm.comm(), stream));
            }
            NCCL_CHECK(ncclBroadcast(pi.custom_mask_indptr.data(),
                                     pi.custom_mask_indptr.data(),
                                     static_cast<std::size_t>(hdr.mask_indptr_count) * 4,
                                     ncclChar, 0, comm.comm(), stream));
        }
        if (have_slot_ids) {
            NCCL_CHECK(ncclBroadcast(pi.slot_ids.data(), pi.slot_ids.data(),
                                     static_cast<std::size_t>(R) * 4,
                                     ncclChar, 0, comm.comm(), stream));
            NCCL_CHECK(ncclBroadcast(pi.is_fresh.data(), pi.is_fresh.data(),
                                     static_cast<std::size_t>(R),
                                     ncclChar, 0, comm.comm(), stream));
            NCCL_CHECK(ncclBroadcast(
                pi.rs_slot_flags.data(), pi.rs_slot_flags.data(),
                static_cast<std::size_t>(R), ncclChar, 0,
                comm.comm(), stream));
        }
        if (rs_fold_lens_count > 0) {
            NCCL_CHECK(ncclBroadcast(
                pi.rs_fold_lens.data(), pi.rs_fold_lens.data(),
                static_cast<std::size_t>(rs_fold_lens_count) *
                    sizeof(std::uint32_t),
                ncclChar, 0, comm.comm(), stream));
        }
        if (rs_mode == RsExecutionMode::BufferWrite ||
            rs_mode == RsExecutionMode::BufferFold) {
            NCCL_CHECK(ncclBroadcast(
                pi.rs_buffer_slot_indptr.data(),
                pi.rs_buffer_slot_indptr.data(),
                static_cast<std::size_t>(R + 1) *
                    sizeof(std::uint32_t),
                ncclChar, 0, comm.comm(), stream));
            if (rs_buffer_ids_count > 0) {
                NCCL_CHECK(ncclBroadcast(
                    pi.rs_buffer_slot_ids.data(),
                    pi.rs_buffer_slot_ids.data(),
                    static_cast<std::size_t>(rs_buffer_ids_count) *
                        sizeof(std::uint32_t),
                    ncclChar, 0, comm.comm(), stream));
            }
            NCCL_CHECK(ncclBroadcast(
                pi.rs_buffer_read_indptr.data(),
                pi.rs_buffer_read_indptr.data(),
                static_cast<std::size_t>(R + 1) * sizeof(std::uint32_t),
                ncclChar, 0, comm.comm(), stream));
            if (R > 0) {
                NCCL_CHECK(ncclBroadcast(
                    pi.rs_buffer_read_lens.data(),
                    pi.rs_buffer_read_lens.data(),
                    static_cast<std::size_t>(R) * sizeof(std::uint32_t),
                    ncclChar, 0, comm.comm(), stream));
                NCCL_CHECK(ncclBroadcast(
                    pi.rs_buffer_heads.data(), pi.rs_buffer_heads.data(),
                    static_cast<std::size_t>(R) * sizeof(std::uint32_t),
                    ncclChar, 0, comm.comm(), stream));
            }
            if (rs_buffer_read_ids_count > 0) {
                NCCL_CHECK(ncclBroadcast(
                    pi.rs_buffer_read_slot_ids.data(),
                    pi.rs_buffer_read_slot_ids.data(),
                    static_cast<std::size_t>(rs_buffer_read_ids_count) *
                        sizeof(std::uint32_t),
                    ncclChar, 0, comm.comm(), stream));
            }
        }
        if (!state_only_fold && logit_rows > 0) {
            NCCL_CHECK(ncclBroadcast(pi.sample_idx.data(), pi.sample_idx.data(),
                                     static_cast<std::size_t>(logit_rows) * 4,
                                     ncclChar, 0, comm.comm(), stream));
        }
        NCCL_CHECK_ASYNC(ncclGroupEnd(), comm.comm());
        payload_timer.stop();

        // 3. Host views of the qo/KV layout for the per-arch attention planner
        // (it runs outside the captured kernel sequence). Rank 0 already holds
        // exactly these, and plans from them itself, so take them from the
        // mailbox: no readback, no synchronize, and both ranks provably plan
        // from the same numbers.
        TpStageTimer host_views_timer(TpProfile::kHostViews);
        h_qo.resize(R + 1);
        h_kvpp.resize(R + 1);
        std::vector<std::uint32_t> h_kvpi(
            state_only_fold
                ? 0
                : static_cast<std::size_t>(
                      std::max(0, hdr.kv_indices_count)));
        std::vector<std::uint32_t> h_kvlpl(
            state_only_fold ? 0 : static_cast<std::size_t>(R));
        std::vector<std::int32_t> h_slot_ids;
        std::vector<std::uint8_t> h_is_fresh;
        // Only synchronize if something was actually read back. On the common
        // decode path the mailbox supplies everything, so the follower issues
        // no D2H at all and never blocks on rank 0's GPU queue here.
        bool host_readback = false;
        if (mailbox != nullptr) {
            const auto copy_from = [](auto& dst, const auto& src) {
                const std::size_t n = std::min(dst.size(), src.size());
                if (n != 0) std::memcpy(dst.data(), src.data(),
                                        n * sizeof(dst[0]));
            };
            const TpFireSlot& fire = mailbox->slots[fire_slot];
            copy_from(h_qo, fire.qo_indptr);
            if (!state_only_fold) {
                copy_from(h_kvpp, fire.kv_page_indptr);
                copy_from(h_kvlpl, fire.kv_last_page_lens);
                copy_from(h_kvpi, fire.kv_page_indices);
            }
            // Everything this fire's slot carries now lives in locals, so rank
            // 0 may reuse it. Released before the payload collectives below for
            // the same reason as the MTP branch.
            mailbox->consumed_fires.store(cpu_gate_seq,
                                          std::memory_order_release);
            auto& wd_ = TpStallWatchdog::instance();
            wd_.consumed.store(cpu_gate_seq, std::memory_order_release);
            wd_.follower_forwards.fetch_add(1, std::memory_order_release);
            // Recurrent-state metadata still comes back from device; those
            // paths are cold and not worth widening the mailbox for.
            if (have_slot_ids) {
                h_slot_ids.resize(R);
                h_is_fresh.resize(R);
                host_readback = true;
                CUDA_CHECK(cudaMemcpyAsync(
                    h_slot_ids.data(), pi.slot_ids.data(),
                    static_cast<std::size_t>(R) * 4,
                    cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    h_is_fresh.data(), pi.is_fresh.data(),
                    static_cast<std::size_t>(R),
                    cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    pi.rs_slot_flags_host.data(), pi.rs_slot_flags.data(),
                    static_cast<std::size_t>(R),
                    cudaMemcpyDeviceToHost, stream));
            }
            if (rs_fold_lens_count > 0) {
                host_readback = true;
                CUDA_CHECK(cudaMemcpyAsync(
                    pi.rs_fold_lens_host.data(), pi.rs_fold_lens.data(),
                    static_cast<std::size_t>(rs_fold_lens_count) *
                        sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost, stream));
            }
            if (rs_mode == RsExecutionMode::BufferWrite ||
                rs_mode == RsExecutionMode::BufferFold) {
                host_readback = true;
                CUDA_CHECK(cudaMemcpyAsync(
                    pi.rs_buffer_slot_indptr_host.data(),
                    pi.rs_buffer_slot_indptr.data(),
                    static_cast<std::size_t>(R + 1) * sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost, stream));
                if (rs_buffer_ids_count > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        pi.rs_buffer_slot_ids_host.data(),
                        pi.rs_buffer_slot_ids.data(),
                        static_cast<std::size_t>(rs_buffer_ids_count) *
                            sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream));
                }
                CUDA_CHECK(cudaMemcpyAsync(
                    pi.rs_buffer_read_indptr_host.data(),
                    pi.rs_buffer_read_indptr.data(),
                    static_cast<std::size_t>(R + 1) * sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost, stream));
                if (R > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        pi.rs_buffer_read_lens_host.data(),
                        pi.rs_buffer_read_lens.data(),
                        static_cast<std::size_t>(R) * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaMemcpyAsync(
                        pi.rs_buffer_heads_host.data(),
                        pi.rs_buffer_heads.data(),
                        static_cast<std::size_t>(R) * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream));
                }
                if (rs_buffer_read_ids_count > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        pi.rs_buffer_read_slot_ids_host.data(),
                        pi.rs_buffer_read_slot_ids.data(),
                        static_cast<std::size_t>(rs_buffer_read_ids_count) *
                            sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream));
                }
            }
        } else {
            host_readback = true;
            CUDA_CHECK(cudaMemcpyAsync(h_qo.data(), pi.qo_indptr.data(),
                                       static_cast<std::size_t>(R + 1) * 4,
                                       cudaMemcpyDeviceToHost, stream));
            if (!state_only_fold) {
                CUDA_CHECK(cudaMemcpyAsync(
                    h_kvpp.data(), pi.kv_page_indptr.data(),
                    static_cast<std::size_t>(R + 1) * 4,
                    cudaMemcpyDeviceToHost, stream));
            }
            if (!state_only_fold && R > 0) {
                CUDA_CHECK(cudaMemcpyAsync(h_kvlpl.data(), pi.kv_last_page_lens.data(),
                                           static_cast<std::size_t>(R) * 4,
                                           cudaMemcpyDeviceToHost, stream));
            }
            if (!h_kvpi.empty()) {
                CUDA_CHECK(cudaMemcpyAsync(
                    h_kvpi.data(), pi.kv_page_indices.data(),
                    h_kvpi.size() * sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost, stream));
            }
            if (have_slot_ids) {
                h_slot_ids.resize(R);
                h_is_fresh.resize(R);
                CUDA_CHECK(cudaMemcpyAsync(h_slot_ids.data(), pi.slot_ids.data(),
                                           static_cast<std::size_t>(R) * 4,
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(h_is_fresh.data(), pi.is_fresh.data(),
                                           static_cast<std::size_t>(R),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    pi.rs_slot_flags_host.data(), pi.rs_slot_flags.data(),
                    static_cast<std::size_t>(R), cudaMemcpyDeviceToHost, stream));
            }
            if (rs_fold_lens_count > 0) {
                CUDA_CHECK(cudaMemcpyAsync(
                    pi.rs_fold_lens_host.data(), pi.rs_fold_lens.data(),
                    static_cast<std::size_t>(rs_fold_lens_count) *
                        sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost, stream));
            }
            if (rs_mode == RsExecutionMode::BufferWrite ||
                rs_mode == RsExecutionMode::BufferFold) {
                CUDA_CHECK(cudaMemcpyAsync(
                    pi.rs_buffer_slot_indptr_host.data(),
                    pi.rs_buffer_slot_indptr.data(),
                    static_cast<std::size_t>(R + 1) *
                        sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost, stream));
                if (rs_buffer_ids_count > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        pi.rs_buffer_slot_ids_host.data(),
                        pi.rs_buffer_slot_ids.data(),
                        static_cast<std::size_t>(rs_buffer_ids_count) *
                            sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream));
                }
                CUDA_CHECK(cudaMemcpyAsync(
                    pi.rs_buffer_read_indptr_host.data(),
                    pi.rs_buffer_read_indptr.data(),
                    static_cast<std::size_t>(R + 1) * sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost, stream));
                if (R > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        pi.rs_buffer_read_lens_host.data(),
                        pi.rs_buffer_read_lens.data(),
                        static_cast<std::size_t>(R) * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaMemcpyAsync(
                        pi.rs_buffer_heads_host.data(),
                        pi.rs_buffer_heads.data(),
                        static_cast<std::size_t>(R) * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream));
                }
                if (rs_buffer_read_ids_count > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        pi.rs_buffer_read_slot_ids_host.data(),
                        pi.rs_buffer_read_slot_ids.data(),
                        static_cast<std::size_t>(rs_buffer_read_ids_count) *
                            sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream));
                }
            }
        }
        if (host_readback) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        host_views_timer.stop();
        for (int request = 0; request < R && have_slot_ids; ++request) {
            const std::uint8_t flags =
                pi.rs_slot_flags_host[static_cast<std::size_t>(request)];
            const std::uint8_t expected_fresh =
                (flags & PIE_RS_FLAG_RESET) != 0 ? 1u : 0u;
            if (h_is_fresh[static_cast<std::size_t>(request)] !=
                expected_fresh) {
                throw std::runtime_error(
                    "TP follower RS flags/reset metadata mismatch");
            }
        }

        std::vector<std::uint32_t> h_rs_slots(
            h_slot_ids.begin(), h_slot_ids.end());
        const std::span<const std::uint32_t> rs_slots(h_rs_slots);
        const std::span<const std::uint8_t> rs_flags(
            have_slot_ids ? pi.rs_slot_flags_host.data() : nullptr,
            have_slot_ids ? static_cast<std::size_t>(R) : 0);
        const std::span<const std::uint32_t> rs_fold_lens(
            rs_fold_lens_count > 0
                ? pi.rs_fold_lens_host.data()
                : nullptr,
            static_cast<std::size_t>(rs_fold_lens_count));
        const bool buffered =
            rs_mode == RsExecutionMode::BufferWrite ||
            rs_mode == RsExecutionMode::BufferFold;
        const std::span<const std::uint32_t> rs_buffer_ids(
            buffered ? pi.rs_buffer_slot_ids_host.data() : nullptr,
            buffered ? static_cast<std::size_t>(rs_buffer_ids_count) : 0);
        const std::span<const std::uint32_t> rs_buffer_indptr(
            buffered ? pi.rs_buffer_slot_indptr_host.data() : nullptr,
            buffered ? static_cast<std::size_t>(R + 1) : 0);
        pipeline::RsExecutionPlan follower_rs_plan;
        std::string rs_error;
        if (!pipeline::plan_rs_execution(
                rs_slots, rs_flags, rs_fold_lens,
                rs_buffer_ids, rs_buffer_indptr, h_qo,
                engine.rs_cache != nullptr,
                engine.rs_cache != nullptr &&
                    engine.rs_cache->rs_buffer_pool_enabled(),
                engine.rs_cache != nullptr
                    ? static_cast<std::uint32_t>(
                          engine.rs_cache->rs_buffer_page_tokens())
                    : 0,
                follower_rs_plan, &rs_error) ||
            follower_rs_plan.mode != rs_mode) {
            throw std::runtime_error(
                "TP follower RS metadata mismatch: " + rs_error);
        }

        // 4. Run the same forward function as rank 0. Channel publication is
        // rank-0-only after the collectives complete.
        if (rs_mode != RsExecutionMode::BufferFold) {
            TpStageTimer plan_timer(TpProfile::kAttnPlan);
            // Claim a plan-staging slot exactly as rank 0 does. Without this
            // the follower rewrites slot 0 on every fire and never records an
            // upload event, so fire N+1's plan lands on top of the plan fire
            // N's attention kernel is still reading — and that kernel then
            // never retires. `AttentionWorkspace` sizes this ring for the
            // in-flight STEP count precisely so ranks can run ahead.
            engine.attn_ws.begin_plan_update();
            engine.forward_fn.invoke_prepare(
                engine.attn_ws,
                ForwardFn::PrepareInputs{
                    .qo_indptr_h = h_qo.data(),
                    .kv_page_indices_h = h_kvpi.data(),
                    .kv_page_indices_d =
                        reinterpret_cast<const std::uint32_t*>(
                            pi.kv_page_indices.data()),
                    .kv_page_indptr_h = h_kvpp.data(),
                    .kv_page_indptr_d =
                        reinterpret_cast<const std::uint32_t*>(
                            pi.kv_page_indptr.data()),
                    .kv_last_page_lens_h = h_kvlpl.data(),
                    .kv_last_page_lens_d =
                        reinterpret_cast<const std::uint32_t*>(
                            pi.kv_last_page_lens.data()),
                    .total_tokens = N,
                    .num_requests = R,
                    .is_pure_decode = is_pure_decode,
                    .have_custom_mask = have_custom_mask,
                    .runtime_window_left = structured_window_left,
                });
            engine.attn_ws.end_plan_update(stream);
        }
        // Mirror rank 0's graph capture/replay decision so NCCL ops
        // inside the body record on both ranks simultaneously (otherwise
        // rank 0 would record while rank 1 executes, deadlocking the
        // first capture). The same `(R)` shape key keeps the per-rank
        // graph caches in lockstep; the captured graph on rank 1 has no
        // PTIR publication, just the forward kernels + NCCL.
        TpStageTimer forward_timer(TpProfile::kForwardBody);
        const bool try_graphs = forward_graph_replay_eligible(
            engine,
            is_pure_decode,
            have_custom_mask,
            rs_mode == RsExecutionMode::BufferWrite,
            rs_mode == RsExecutionMode::BufferFold,
            has_write_desc,
            structured_window_left,
            have_slot_ids,
            h_is_fresh.data(),
            R,
            /*num_images=*/0,
            /*num_clips=*/0,
            /*has_stage_hooks=*/false,
            // lora is capability-gated to tp_size == 1 (context.cpp), so a
            // TP follower can never see a lora fire.
            /*has_lora=*/false);
        const std::uint32_t graph_layout =
            engine.forward_fn.invoke_graph_layout();
        const std::uint32_t graph_variant =
            make_graph_variant(/*small_spec=*/false,
                               /*rs_verify=*/false,
                               have_custom_mask,
                               /*fused_argmax=*/false,
                               graph_layout);
        if (try_graphs) {
            const ForwardGraphKey key{R, N, graph_variant};
            cudaGraphExec_t exec = engine.graph_cache->get(key);
            if (exec == nullptr) {
                exec = capture_forward_graph_exec(
                    engine, h_qo.data(), h_kvpi.data(), h_kvpp.data(),
                    h_kvlpl.data(),
                    N, R, is_pure_decode,
                    have_custom_mask,
                    have_slot_ids ? h_slot_ids.data() : nullptr,
                    have_slot_ids ? h_is_fresh.data() : nullptr,
                    have_slot_ids ? pi.slot_ids.data() : nullptr,
                    logit_rows > 0 ? pi.sample_idx.data() : nullptr,
                    logit_rows,
                    pi.w_page.data(),
                    pi.w_off.data(),
                    has_write_desc,
                    structured_window_left,
                    // The LM head is sharded across ranks and the followers
                    // never sample, so the fused reduction has no meaning here.
                    /*logits_argmax_chunk_tokens=*/0);
                engine.graph_cache->put(key, exec);
            }
            CUDA_CHECK(cudaGraphLaunch(exec, /*stream=*/nullptr));
        } else {
            pie_cuda_driver::ForwardFn::ForwardInputs fwd_in;
            fwd_in.token_ids = reinterpret_cast<const std::int32_t*>(pi.tokens.data());
            fwd_in.positions = reinterpret_cast<const std::int32_t*>(pi.positions.data());
            fwd_in.qo_indptr_d         = pi.qo_indptr.data();
            fwd_in.kv_page_indices_d   = pi.kv_page_indices.data();
            fwd_in.kv_page_indptr_d    = pi.kv_page_indptr.data();
            fwd_in.kv_last_page_lens_d = pi.kv_last_page_lens.data();
            fwd_in.qo_indptr_h         = h_qo.data();
            fwd_in.kv_page_indices_h   = h_kvpi.data();
            fwd_in.kv_page_indptr_h    = h_kvpp.data();
            fwd_in.kv_last_page_lens_h = h_kvlpl.data();
            fwd_in.total_tokens        = N;
            fwd_in.num_requests        = R;
            fwd_in.is_pure_decode      = is_pure_decode;
            fwd_in.custom_mask_d        = have_custom_mask ? pi.custom_mask.data()        : nullptr;
            fwd_in.custom_mask_indptr_d = have_custom_mask ? pi.custom_mask_indptr.data() : nullptr;
            fwd_in.slot_ids_h          = have_slot_ids ? h_slot_ids.data() : nullptr;
            fwd_in.is_fresh_h          = have_slot_ids ? h_is_fresh.data() : nullptr;
            fwd_in.slot_ids_d          = have_slot_ids ? pi.slot_ids.data() : nullptr;
            fwd_in.is_fresh_d          = have_slot_ids ? pi.is_fresh.data() : nullptr;
            fwd_in.rs_slot_flags_h     =
                have_slot_ids ? pi.rs_slot_flags_host.data() : nullptr;
            fwd_in.rs_buffer_slot_ids_h =
                buffered ? pi.rs_buffer_slot_ids_host.data() : nullptr;
            fwd_in.rs_buffer_slot_indptr_h =
                buffered ? pi.rs_buffer_slot_indptr_host.data() : nullptr;
            // Rank 0 decides whether a row replays; the follower must reach
            // the SAME decision or its recurrence starts from a different
            // state, and the all-reduce hides the disagreement.
            const bool follower_has_read =
                buffered && R > 0 &&
                std::any_of(
                    pi.rs_buffer_read_lens_host.data(),
                    pi.rs_buffer_read_lens_host.data() + R,
                    [](std::uint32_t n) { return n != 0; });
            if (follower_has_read &&
                pi.rs_buffer_read_indptr_host[static_cast<std::size_t>(R)] !=
                    static_cast<std::uint32_t>(rs_buffer_read_ids_count)) {
                throw std::runtime_error(
                    "TP follower RS buffer read CSR does not match its id "
                    "count");
            }
            fwd_in.rs_buffer_read_slot_ids_h =
                follower_has_read ? pi.rs_buffer_read_slot_ids_host.data()
                                  : nullptr;
            fwd_in.rs_buffer_read_indptr_h =
                follower_has_read ? pi.rs_buffer_read_indptr_host.data()
                                  : nullptr;
            fwd_in.rs_buffer_read_lens_h =
                follower_has_read ? pi.rs_buffer_read_lens_host.data()
                                  : nullptr;
            fwd_in.rs_buffer_heads_h =
                buffered && R > 0 ? pi.rs_buffer_heads_host.data() : nullptr;
            fwd_in.rs_fold_lens_h =
                rs_fold_lens_count > 0
                    ? pi.rs_fold_lens_host.data()
                    : nullptr;
            fwd_in.rs_fold_lens_d =
                rs_fold_lens_count > 0
                    ? reinterpret_cast<const std::int32_t*>(
                          pi.rs_fold_lens.data())
                    : nullptr;
            fwd_in.rs_buffer_write =
                rs_mode == RsExecutionMode::BufferWrite;
            fwd_in.rs_buffer_fold =
                rs_mode == RsExecutionMode::BufferFold;
            fwd_in.logit_row_indices_d = logit_rows > 0 ? pi.sample_idx.data() : nullptr;
            fwd_in.num_logit_rows      = logit_rows;
            fwd_in.runtime_window_left = structured_window_left;
            fwd_in.w_page_d = has_write_desc ? pi.w_page.data() : nullptr;
            fwd_in.w_off_d = has_write_desc ? pi.w_off.data() : nullptr;
            fwd_in.row_valid_d = pi.row_valid.data();
            fwd_in.has_write_desc = has_write_desc;
            engine.forward_fn.invoke_body(
                engine.ws, engine.kv_cache, engine.attn_ws, engine.cublas,
                fwd_in);
        }
        forward_timer.stop();
        if (mailbox != nullptr && TpProfile::enabled()) {
            const auto now = std::chrono::steady_clock::now();
            auto& prof = TpProfile::instance();
            prof.ns[TpProfile::kStartSkew] += static_cast<std::uint64_t>(
                std::max<std::int64_t>(0,
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        now - mailbox->notify_time).count()));
            ++prof.hits[TpProfile::kStartSkew];
        }
    }
}

void tp_send_shutdown(NcclComm& comm, const std::string& cpu_gate_key) {
    tp_cpu_gate_notify(cpu_gate_key);
    auto* d_hdr = tp_hdr_dev_buf();
    cudaStream_t stream = nullptr;
    TpFireHeader* hdr_host = tp_hdr_host_buf();
    *hdr_host = TpFireHeader{
        .magic = TP_STOP_MAGIC,
        .structured_window_left = -2,
    };
    CUDA_CHECK(cudaMemcpyAsync(d_hdr, hdr_host, sizeof(TpFireHeader),
                               cudaMemcpyHostToDevice, stream));
    NCCL_CHECK_ASYNC(ncclBroadcast(d_hdr, d_hdr, sizeof(TpFireHeader), ncclChar, 0,
                                   comm.comm(), stream),
                     comm.comm());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace pie_cuda_driver
