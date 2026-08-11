#pragma once

#include "batch/tp_gate.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace pie_cuda_driver {

// Logical identity shared by rank 0's publication and every follower's
// consumption receipt. A mismatch in any field means the ranks are not
// executing the same fire, even if they happen to enter the same collective.
struct TpLogicalFireIdentity {
    std::uint64_t sequence = 0;
    TpFireKind kind = TpFireKind::Forward;
    int total_tokens = 0;
    int num_requests = 0;
    int required_kv_pages = 0;
    int detail = 0;

    bool operator==(const TpLogicalFireIdentity&) const = default;
};

// A separate budget is used for fire and launch receipts, so a burst of one
// kind cannot suppress the other. The environment value is a PER-RANK limit;
// the hard ceiling prevents a malformed diagnostic topology from creating an
// unbounded log on a paid pod.
class TpReceiptBudget {
  public:
    explicit TpReceiptBudget(std::uint64_t limit) : limit_(limit) {}

    bool take(int rank) {
        if (limit_ == 0) return false;
        std::lock_guard<std::mutex> lock(mu_);
        auto& used = used_by_rank_[rank];
        if (used >= limit_) return false;
        ++used;
        return true;
    }

  private:
    std::uint64_t limit_;
    std::mutex mu_;
    std::unordered_map<int, std::uint64_t> used_by_rank_;
};

inline std::uint64_t tp_fire_receipt_limit() {
    constexpr std::uint64_t kMaxReceiptsPerRank = 4096;
    const char* raw = std::getenv("PIE_TP_FIRE_RECEIPTS");
    if (raw == nullptr || *raw == '\0') return 0;
    std::uint64_t parsed = 0;
    for (const char* cursor = raw; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') return 0;
        const auto digit = static_cast<std::uint64_t>(*cursor - '0');
        if (parsed > (kMaxReceiptsPerRank - digit) / 10) {
            return kMaxReceiptsPerRank;
        }
        parsed = parsed * 10 + digit;
    }
    return parsed;
}

inline TpReceiptBudget& tp_fire_receipt_budget() {
    static TpReceiptBudget budget(tp_fire_receipt_limit());
    return budget;
}

inline TpReceiptBudget& tp_launch_receipt_budget() {
    static TpReceiptBudget budget(tp_fire_receipt_limit());
    return budget;
}

inline const char* tp_fire_kind_name(TpFireKind kind) {
    switch (kind) {
        case TpFireKind::Forward: return "forward";
        case TpFireKind::MtpDraft: return "mtp_draft";
    }
    return "unknown";
}

inline void emit_tp_fire_receipt(
    int rank,
    const char* phase,
    const TpLogicalFireIdentity& identity) {
    if (!tp_fire_receipt_budget().take(rank)) return;
    static std::mutex output_mu;
    std::lock_guard<std::mutex> lock(output_mu);
    std::cerr << "[pie-driver-cuda] tp_fire_receipt"
              << " phase=" << phase
              << " rank=" << rank
              << " seq=" << identity.sequence
              << " kind=" << tp_fire_kind_name(identity.kind)
              << " tokens=" << identity.total_tokens
              << " requests=" << identity.num_requests
              << " required_kv_pages=" << identity.required_kv_pages
              << " detail=" << identity.detail << '\n';
}

inline void emit_tp_launch_receipt(
    int rank,
    std::uint64_t sequence,
    const char* phase,
    std::size_t step_count,
    int status) {
    if (!tp_launch_receipt_budget().take(rank)) return;
    static std::mutex output_mu;
    std::lock_guard<std::mutex> lock(output_mu);
    std::cerr << "[pie-driver-cuda] context_launch_receipt"
              << " phase=" << phase
              << " rank=" << rank
              << " seq=" << sequence
              << " steps=" << step_count;
    if (status != std::numeric_limits<int>::min()) {
        std::cerr << " status=" << status;
    }
    std::cerr << '\n';
}

}  // namespace pie_cuda_driver
