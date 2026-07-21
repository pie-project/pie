#pragma once

// Composed multi-program forward batch (W1.1 follow-up): merge WIRE program
// geometry (engine-assembled launch slices) with DEVICE-GEOMETRY program
// geometry (channel-resolved `FireGeometry`, see descriptor_resolve.hpp) into
// ONE flat forward batch. CUDA-free, host-only.
//
// Wire layout contract (runtime `build_batch_request`): every program owns the
// wire request rows `[row_indptr[p], row_indptr[p+1])`. A device-geometry
// program's span is an EMPTY placeholder (zero query rows, zero sampling rows,
// zero mask rows — its wire kv pages are capacity-accounting only and are
// DROPPED here); a wire program's span carries its real geometry.
//
// Composed order: wire programs first (in program order — their token rows
// keep their wire positions, so token-row side-channels like image/audio
// anchor rows stay valid verbatim), then device-geometry programs (in program
// order). `prog_sample_offsets[p]` records where program `p`'s sampled rows
// land in the GATHERED logits buffer (the executor gathers sampled rows in
// composed request order); `Dispatch::finish` slices each program's logits
// base from it.
//
// Out of scope (v1, enforced by the runtime scheduler + defended here):
//   * dense device masks (`fg.has_mask`) in a MULTI-program batch
//   * wire custom (non-causal) BRLE masks co-batched with device geometry

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "pie_native/launch_view.hpp"

#include "batch/rs_metadata.hpp"
#include "pie_native/ptir/fire_geometry.hpp"

namespace pie_cuda_driver::pipeline {

// Shared pure-host PTIR decode model (trace/op-table/container/bound/
// fire-geometry) now lives in pie_native::ptir (driver/common); bring it into
// scope so the CUDA-side tier-0/1 code below can use it unqualified.
using namespace pie_native::ptir;

inline int mtp_global_history_tokens(
    int input_position,
    std::uint32_t draft_step,
    bool prefix_global) {
    return std::max(
        0,
        input_position -
            (prefix_global ? static_cast<int>(draft_step) : 0));
}

inline bool plan_mtp_draft_rows(
    std::span<const std::uint32_t> counts,
    std::uint32_t base,
    std::uint32_t capacity,
    std::vector<std::uint32_t>& starts,
    std::string* error = nullptr,
    std::uint32_t per_program_limit = 32) {
    starts.assign(counts.size(), UINT32_MAX);
    std::uint32_t used = 0;
    for (std::size_t program = 0; program < counts.size(); ++program) {
        const std::uint32_t count = counts[program];
        if (count == 0) continue;
        if (count > per_program_limit) {
            if (error != nullptr) {
                *error = "MtpLogits program exceeds the per-program draft clamp";
            }
            return false;
        }
        if (used > capacity || count > capacity - used ||
            count > UINT32_MAX - used ||
            base > UINT32_MAX - (used + count)) {
            if (error != nullptr) {
                *error = "aggregate MtpLogits rows exceed the batch reserve";
            }
            return false;
        }
        starts[program] = base + used;
        used += count;
    }
    return true;
}

struct ComposedBatch {
    std::vector<std::uint32_t> token_ids;
    std::vector<std::uint32_t> position_ids;
    std::vector<std::uint32_t> qo_indptr;
    std::vector<std::uint32_t> kv_page_indices;
    std::vector<std::uint32_t> kv_page_indptr;
    std::vector<std::uint32_t> kv_last_page_lens;
    std::vector<std::uint32_t> sampling_indices;  // request-relative rows
    std::vector<std::uint32_t> sampling_indptr;   // per composed REQUEST
    // Folded recurrent state and buffered-page CSR in the same request order as
    // qo_indptr. These remain distinct id spaces throughout composition.
    std::vector<std::uint32_t> rs_slot_ids;
    std::vector<std::uint8_t> rs_slot_flags;
    std::vector<std::uint32_t> rs_fold_lens;
    std::vector<std::uint32_t> rs_buffer_slot_ids;
    std::vector<std::uint32_t> rs_buffer_slot_indptr;
    std::vector<StructuredMaskDescriptor> structured_masks;
    // Explicit KV-write descriptor covering EVERY composed token row
    // (`has_write_desc` routes the whole forward's KV append through it, so
    // wire rows get their standard append target synthesized).
    std::vector<std::uint32_t> w_page;
    std::vector<std::uint32_t> w_off;
    // Per-PROGRAM offset of each program's sampled rows within the gathered
    // logits buffer (`n_prog + 1` entries, last = total sampled rows).
    std::vector<std::uint32_t> prog_sample_offsets;
    std::vector<std::uint32_t> prog_sample_starts;
    std::vector<std::uint32_t> prog_sample_counts;
    std::vector<std::uint32_t> prog_request_starts;
    std::vector<std::uint32_t> prog_request_counts;
    std::vector<std::uint32_t> prog_row_counts;
    std::vector<std::uint32_t> prog_token_counts;
    std::vector<std::uint32_t> prog_kv_lens;
    std::vector<std::uint32_t> prog_page_counts;
    std::vector<std::uint32_t> prog_query_lens;
    std::vector<std::uint32_t> prog_key_lens;
    bool has_write_desc = false;
};

namespace detail {

// Synthesize the standard paged-append write target for one wire request's
// token rows: row `i` of the request writes KV position `klen - qo_len + i`.
inline bool synthesize_wire_write_desc(const ComposedBatch& out,
                                       std::size_t request,
                                       std::uint32_t page_size,
                                       std::vector<std::uint32_t>& w_page,
                                       std::vector<std::uint32_t>& w_off,
                                       std::string* err) {
    const std::uint32_t qo_len =
        out.qo_indptr[request + 1] - out.qo_indptr[request];
    const std::uint32_t page_begin = out.kv_page_indptr[request];
    const std::uint32_t page_count = out.kv_page_indptr[request + 1] - page_begin;
    const std::uint32_t last_len = out.kv_last_page_lens[request];
    const std::uint64_t klen = page_count == 0
        ? 0
        : static_cast<std::uint64_t>(page_count - 1) * page_size + last_len;
    if (klen < qo_len) {
        if (err) *err = "ptir compose: wire KV extent shorter than query span";
        return false;
    }
    for (std::uint32_t i = 0; i < qo_len; ++i) {
        const std::uint64_t pos = klen - qo_len + i;
        const std::uint64_t page_slot = pos / page_size;
        if (page_slot >= page_count) {
            if (err) *err = "ptir compose: wire write target beyond page span";
            return false;
        }
        w_page.push_back(out.kv_page_indices[page_begin + page_slot]);
        w_off.push_back(static_cast<std::uint32_t>(pos % page_size));
    }
    return true;
}

}  // namespace detail

inline bool global_sampling_rows(
    std::span<const std::uint32_t> qo_indptr,
    std::span<const std::uint32_t> sampling_indptr,
    std::span<const std::uint32_t> sampling_indices,
    std::vector<std::int32_t>& out,
    std::string* error) {
    if (qo_indptr.empty() ||
        sampling_indptr.size() != qo_indptr.size() ||
        sampling_indptr.back() != sampling_indices.size()) {
        if (error != nullptr) {
            *error = "sampling CSR does not match query CSR";
        }
        return false;
    }
    out.clear();
    out.reserve(sampling_indices.size());
    for (std::size_t request = 0;
         request + 1 < qo_indptr.size();
         ++request) {
        const std::uint32_t begin = qo_indptr[request];
        const std::uint32_t length =
            qo_indptr[request + 1] - begin;
        for (std::uint32_t index = sampling_indptr[request];
             index < sampling_indptr[request + 1];
             ++index) {
            if (sampling_indices[index] >= length) {
                if (error != nullptr) {
                    *error = "sampling index exceeds request token span";
                }
                return false;
            }
            out.push_back(static_cast<std::int32_t>(
                begin + sampling_indices[index]));
        }
    }
    return true;
}

inline bool validate_folded_rs_bindings(
    std::span<const std::uint32_t> slot_ids,
    std::span<const std::uint8_t> slot_flags,
    std::span<const std::uint32_t> fold_lens,
    std::size_t resolved_requests,
    bool required,
    std::string* error = nullptr) {
    if (slot_ids.size() != slot_flags.size()) {
        if (error != nullptr) {
            *error = "folded RS slot ids/flags length mismatch";
        }
        return false;
    }
    if (!fold_lens.empty() && fold_lens.size() != slot_ids.size()) {
        if (error != nullptr) {
            *error = "folded RS fold lengths do not match slot ids";
        }
        return false;
    }
    for (std::size_t request = 0; request < slot_flags.size(); ++request) {
        const std::uint8_t flags = slot_flags[request];
        if ((flags & ~(PIE_RS_FLAG_RESET | PIE_RS_FLAG_FOLD)) != 0) {
            if (error != nullptr) {
                *error = "folded RS flags contain unknown bits";
            }
            return false;
        }
        const bool folds = (flags & PIE_RS_FLAG_FOLD) != 0;
        if (folds != (!fold_lens.empty() && fold_lens[request] != 0)) {
            if (error != nullptr) {
                *error = "folded RS flags/fold lengths are inconsistent";
            }
            return false;
        }
        if (!fold_lens.empty() &&
            fold_lens[request] >
                static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
            if (error != nullptr) {
                *error = "folded RS fold length exceeds model ABI";
            }
            return false;
        }
    }
    if (!slot_ids.empty() && slot_ids.size() != resolved_requests) {
        if (error != nullptr) {
            *error = "folded RS slots do not match resolved request count";
        }
        return false;
    }
    if (required && resolved_requests != 0 && slot_ids.empty()) {
        if (error != nullptr) {
            *error = "RS forward is missing folded slot ids/flags";
        }
        return false;
    }
    return true;
}

struct RsExecutionPlan {
    pie_cuda_driver::RsExecutionMode mode =
        pie_cuda_driver::RsExecutionMode::None;
    std::vector<std::uint32_t> fold_qo_indptr;
    std::uint32_t fold_tokens = 0;
};

enum class StructuredMaskCoverage : std::uint8_t {
    None,
    Complete,
    Mixed,
};

inline StructuredMaskCoverage structured_mask_coverage(
    std::span<const StructuredMaskDescriptor> masks) noexcept {
    const std::size_t explicit_count = static_cast<std::size_t>(
        std::count_if(masks.begin(), masks.end(), [](const auto& mask) {
            return static_cast<bool>(mask);
        }));
    if (explicit_count == 0) return StructuredMaskCoverage::None;
    return explicit_count == masks.size()
        ? StructuredMaskCoverage::Complete
        : StructuredMaskCoverage::Mixed;
}

inline bool plan_rs_execution(
    std::span<const std::uint32_t> slot_ids,
    std::span<const std::uint8_t> slot_flags,
    std::span<const std::uint32_t> fold_lens,
    std::span<const std::uint32_t> buffer_slot_ids,
    std::span<const std::uint32_t> buffer_slot_indptr,
    std::span<const std::uint32_t> resolved_qo_indptr,
    bool rs_cache_present,
    bool buffer_pool_enabled,
    std::uint32_t buffer_page_tokens,
    RsExecutionPlan& plan,
    std::string* error = nullptr) {
    plan = RsExecutionPlan{};
    if (resolved_qo_indptr.empty()) {
        if (error != nullptr) *error = "resolved RS query CSR is empty";
        return false;
    }
    const std::size_t requests = resolved_qo_indptr.size() - 1;
    if (!validate_folded_rs_bindings(
            slot_ids, slot_flags, fold_lens, requests, rs_cache_present,
            error)) {
        return false;
    }
    if (slot_ids.empty()) {
        const bool empty_buffer_csr =
            buffer_slot_ids.empty() &&
            (buffer_slot_indptr.empty() ||
             (buffer_slot_indptr.size() == requests + 1 &&
              buffer_slot_indptr.front() == 0 &&
              buffer_slot_indptr.back() == 0 &&
              std::is_sorted(
                  buffer_slot_indptr.begin(),
                  buffer_slot_indptr.end())));
        if (!empty_buffer_csr) {
            if (error != nullptr) {
                *error = "buffered RS metadata has no folded slot bindings";
            }
            return false;
        }
        return true;
    }

    const bool has_buffer_csr =
        !buffer_slot_ids.empty() || !buffer_slot_indptr.empty();
    if (has_buffer_csr) {
        if (buffer_slot_indptr.size() != requests + 1 ||
            buffer_slot_indptr.front() != 0 ||
            buffer_slot_indptr.back() != buffer_slot_ids.size()) {
            if (error != nullptr) {
                *error = "buffered RS CSR does not match resolved requests";
            }
            return false;
        }
        for (std::size_t i = 1; i < buffer_slot_indptr.size(); ++i) {
            if (buffer_slot_indptr[i] < buffer_slot_indptr[i - 1]) {
                if (error != nullptr) {
                    *error = "buffered RS CSR is not monotonic";
                }
                return false;
            }
        }
    }

    const bool any_fold = std::any_of(
        slot_flags.begin(), slot_flags.end(), [](std::uint8_t flags) {
            return (flags & PIE_RS_FLAG_FOLD) != 0;
        });
    const bool all_fold = std::all_of(
        slot_flags.begin(), slot_flags.end(), [](std::uint8_t flags) {
            return (flags & PIE_RS_FLAG_FOLD) != 0;
        });
    if (any_fold != all_fold) {
        if (error != nullptr) {
            *error = "mixed folded and forward RS rows are unsupported";
        }
        return false;
    }
    if (!any_fold && buffer_slot_ids.empty()) {
        plan.mode = pie_cuda_driver::RsExecutionMode::Forward;
        return true;
    }
    if (!buffer_pool_enabled || buffer_page_tokens == 0) {
        if (error != nullptr) {
            *error = "buffered RS execution requires a configured buffer pool";
        }
        return false;
    }

    if (any_fold) {
        if (!has_buffer_csr || fold_lens.size() != requests) {
            if (error != nullptr) {
                *error = "folded RS replay is missing buffer metadata";
            }
            return false;
        }
        plan.mode = pie_cuda_driver::RsExecutionMode::BufferFold;
        plan.fold_qo_indptr.reserve(requests + 1);
        plan.fold_qo_indptr.push_back(0);
        for (std::size_t request = 0; request < requests; ++request) {
            const std::uint32_t tokens = fold_lens[request];
            const std::uint64_t required_slabs =
                (static_cast<std::uint64_t>(tokens) + buffer_page_tokens - 1) /
                buffer_page_tokens;
            const std::uint32_t available_slabs =
                buffer_slot_indptr[request + 1] -
                buffer_slot_indptr[request];
            if (tokens == 0 || required_slabs > available_slabs ||
                tokens > static_cast<std::uint32_t>(
                             std::numeric_limits<int>::max()) -
                    plan.fold_tokens) {
                if (error != nullptr) {
                    *error = "folded RS replay exceeds its buffered slabs";
                }
                return false;
            }
            plan.fold_tokens += tokens;
            plan.fold_qo_indptr.push_back(plan.fold_tokens);
        }
        return true;
    }

    if (!has_buffer_csr) {
        if (error != nullptr) *error = "buffered RS write is missing its CSR";
        return false;
    }
    plan.mode = pie_cuda_driver::RsExecutionMode::BufferWrite;
    for (std::size_t request = 0; request < requests; ++request) {
        const std::uint32_t begin = resolved_qo_indptr[request];
        const std::uint32_t end = resolved_qo_indptr[request + 1];
        if (end < begin) {
            if (error != nullptr) *error = "resolved RS query CSR is not monotonic";
            return false;
        }
        const std::uint64_t required_slabs =
            (static_cast<std::uint64_t>(end - begin) + buffer_page_tokens - 1) /
            buffer_page_tokens;
        const std::uint32_t available_slabs =
            buffer_slot_indptr[request + 1] -
            buffer_slot_indptr[request];
        if (required_slabs > available_slabs) {
            if (error != nullptr) {
                *error = "buffered RS write exceeds its assigned slabs";
            }
            return false;
        }
    }
    return true;
}

inline std::optional<int> runtime_window_for_tail_aligned(
    std::span<const StructuredMaskDescriptor> masks,
    std::span<const std::uint32_t> positions,
    std::span<const std::uint32_t> qo_indptr,
    std::span<const std::uint32_t> kv_page_indptr,
    std::span<const std::uint32_t> kv_last_page_lens,
    std::uint32_t page_size) {
    if (masks.empty() || qo_indptr.size() != masks.size() + 1 ||
        kv_page_indptr.size() != masks.size() + 1 ||
        kv_last_page_lens.size() != masks.size() ||
        qo_indptr.back() != positions.size() || page_size == 0) {
        return std::nullopt;
    }
    const auto& first = masks.front();
    int window_left = -2;
    if (first.kind == StructuredMaskKind::Causal) {
        window_left = -1;
    } else if (
        first.kind == StructuredMaskKind::SlidingWindow &&
        first.window > 0 &&
        first.window - 1 <=
            static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        window_left = static_cast<int>(first.window - 1);
    } else {
        return std::nullopt;
    }
    for (std::size_t request = 0; request < masks.size(); ++request) {
        const auto& mask = masks[request];
        if (mask.kind != first.kind || mask.window != first.window ||
            mask.sink != first.sink) {
            return std::nullopt;
        }
        const std::uint32_t query_begin = qo_indptr[request];
        const std::uint32_t query_end = qo_indptr[request + 1];
        const std::uint32_t query_count = query_end - query_begin;
        const std::uint32_t page_count =
            kv_page_indptr[request + 1] - kv_page_indptr[request];
        const std::uint64_t key_count = page_count == 0
            ? 0
            : static_cast<std::uint64_t>(page_count - 1) * page_size +
                kv_last_page_lens[request];
        if (key_count < query_count ||
            key_count > std::numeric_limits<std::uint32_t>::max()) {
            return std::nullopt;
        }
        const std::uint32_t first_position =
            static_cast<std::uint32_t>(key_count) - query_count;
        for (std::uint32_t query = 0; query < query_count; ++query) {
            if (positions[query_begin + query] != first_position + query) {
                return std::nullopt;
            }
        }
    }
    return window_left;
}

// Compose the forward batch from the wire launch view + the per-program
// descriptor resolution. Requires `view.ptir_program_row_indptr` (the runtime
// ships it for every direct launch). Returns false + `*err` on a violated
// contract; the executor must fail the fire.
inline bool compose_forward_batch(const pie_native::LaunchView& view,
                                  const ResolvedPrograms& resolved,
                                  std::uint32_t page_size,
                                  ComposedBatch& out,
                                  std::string* err) {
    const std::size_t n_prog = view.ptir_program_hashes.size();
    if (resolved.per_program.size() != n_prog ||
        resolved.is_device_geometry.size() != n_prog) {
        if (err) *err = "ptir compose: resolution/program count mismatch";
        return false;
    }
    const auto row_indptr = view.ptir_program_row_indptr;
    if (row_indptr.size() != n_prog + 1) {
        if (err) *err = "ptir compose: missing program row attribution CSR";
        return false;
    }
    const auto qo = view.qo_indptr;
    const auto tok = view.token_ids;
    const auto pos = view.position_ids;
    const auto kvpi = view.kv_page_indices;
    const auto kvpp = view.kv_page_indptr;
    const auto kvlpl = view.kv_last_page_lens;
    const auto sidx = view.sampling_indices;
    const auto sptr = view.sampling_indptr;
    const std::size_t wire_rows = qo.size() > 0 ? qo.size() - 1 : 0;
    if (row_indptr.data()[n_prog] != wire_rows) {
        if (err) *err = "ptir compose: row attribution does not cover the wire";
        return false;
    }
    std::vector<std::uint32_t> input_request_starts(n_prog);
    std::vector<std::uint32_t> input_request_counts(n_prog);
    std::uint32_t input_request_count = 0;
    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint32_t row_begin = row_indptr.data()[p];
        const std::uint32_t row_end = row_indptr.data()[p + 1];
        if (row_end < row_begin || row_end > wire_rows) {
            if (err) *err = "ptir compose: malformed program row attribution";
            return false;
        }
        std::uint32_t count = row_end - row_begin;
        if (resolved.is_device_geometry[p]) {
            const auto& fg = resolved.per_program[p];
            if (fg.qo_indptr.empty()) {
                if (err) *err = "ptir compose: resolved query CSR is empty";
                return false;
            }
            count = static_cast<std::uint32_t>(fg.qo_indptr.size() - 1);
        }
        input_request_starts[p] = input_request_count;
        input_request_counts[p] = count;
        if (count > UINT32_MAX - input_request_count) {
            if (err) *err = "ptir compose: request count overflow";
            return false;
        }
        input_request_count += count;
    }
    const auto input_rs_slots = view.rs_slot_ids.as<std::uint32_t>();
    const auto input_rs_flags = view.rs_slot_flags.as<std::uint8_t>();
    const auto input_rs_fold_lens =
        view.rs_fold_lens.as<std::uint32_t>();
    const bool has_folded_rs =
        !input_rs_slots.empty() || !input_rs_flags.empty() ||
        !input_rs_fold_lens.empty();
    std::string rs_error;
    if (has_folded_rs &&
        !validate_folded_rs_bindings(
            input_rs_slots,
            input_rs_flags,
            input_rs_fold_lens,
            input_request_count,
            false,
            &rs_error)) {
        if (err) *err = "ptir compose: " + rs_error;
        return false;
    }
    const auto input_rs_buffer_ids =
        view.rs_buffer_slot_ids.as<std::uint32_t>();
    const auto input_rs_buffer_indptr =
        view.rs_buffer_slot_indptr.as<std::uint32_t>();
    const bool has_rs_buffer = !input_rs_buffer_indptr.empty();
    if (has_rs_buffer) {
        if (input_rs_buffer_indptr.size() != input_request_count + 1 ||
            input_rs_buffer_indptr.front() != 0 ||
            input_rs_buffer_indptr.back() != input_rs_buffer_ids.size()) {
            if (err) *err =
                "ptir compose: buffered RS CSR does not cover resolved requests";
            return false;
        }
        for (std::size_t request = 1;
             request < input_rs_buffer_indptr.size();
             ++request) {
            if (input_rs_buffer_indptr[request] <
                input_rs_buffer_indptr[request - 1]) {
                if (err) *err = "ptir compose: buffered RS CSR is not monotonic";
                return false;
            }
        }
    } else if (!input_rs_buffer_ids.empty()) {
        if (err) *err = "ptir compose: buffered RS ids have no CSR";
        return false;
    }

    out = ComposedBatch{};
    out.qo_indptr.push_back(0);
    out.kv_page_indptr.push_back(0);
    out.sampling_indptr.push_back(0);
    if (has_rs_buffer) out.rs_buffer_slot_indptr.push_back(0);
    out.prog_sample_offsets.assign(n_prog + 1, 0);
    out.prog_sample_starts.assign(n_prog, 0);
    out.prog_sample_counts.assign(n_prog, 0);
    out.prog_request_starts.assign(n_prog, 0);
    out.prog_request_counts.assign(n_prog, 0);
    constexpr std::uint32_t unavailable =
        std::numeric_limits<std::uint32_t>::max();
    out.prog_row_counts.assign(n_prog, unavailable);
    out.prog_token_counts.assign(n_prog, unavailable);
    out.prog_kv_lens.assign(n_prog, unavailable);
    out.prog_page_counts.assign(n_prog, unavailable);
    out.prog_query_lens.assign(n_prog, unavailable);
    out.prog_key_lens.assign(n_prog, unavailable);
    auto append_rs = [&](std::size_t program) {
        const std::uint32_t begin = input_request_starts[program];
        const std::uint32_t count = input_request_counts[program];
        if (has_folded_rs) {
            out.rs_slot_ids.insert(
                out.rs_slot_ids.end(),
                input_rs_slots.begin() + begin,
                input_rs_slots.begin() + begin + count);
            out.rs_slot_flags.insert(
                out.rs_slot_flags.end(),
                input_rs_flags.begin() + begin,
                input_rs_flags.begin() + begin + count);
            if (!input_rs_fold_lens.empty()) {
                out.rs_fold_lens.insert(
                    out.rs_fold_lens.end(),
                    input_rs_fold_lens.begin() + begin,
                    input_rs_fold_lens.begin() + begin + count);
            }
        }
        if (has_rs_buffer) {
            for (std::uint32_t request = begin;
                 request < begin + count;
                 ++request) {
                const std::uint32_t lo = input_rs_buffer_indptr[request];
                const std::uint32_t hi = input_rs_buffer_indptr[request + 1];
                if (hi != lo) {
                    out.rs_buffer_slot_ids.insert(
                        out.rs_buffer_slot_ids.end(),
                        input_rs_buffer_ids.begin() + lo,
                        input_rs_buffer_ids.begin() + hi);
                }
                out.rs_buffer_slot_indptr.push_back(
                    static_cast<std::uint32_t>(
                        out.rs_buffer_slot_ids.size()));
            }
        }
    };

    bool any_write_desc = false;
    for (std::size_t p = 0; p < n_prog; ++p) {
        if (resolved.is_device_geometry[p] &&
            resolved.per_program[p].has_write_desc) {
            any_write_desc = true;
        }
    }

    // Pass 1 — wire programs, in program order. Placeholder rows of
    // device-geometry programs are skipped wholesale (their spans are empty
    // of tokens/sampling; their wire kv pages are capacity padding).
    for (std::size_t p = 0; p < n_prog; ++p) {
        if (resolved.is_device_geometry[p]) continue;
        const std::uint32_t program_row_begin = row_indptr.data()[p];
        const std::uint32_t program_row_end = row_indptr.data()[p + 1];
        if (program_row_end >= qo.size()) {
            if (err) *err =
                "ptir compose: wire query CSR shorter than attribution";
            return false;
        }
        if (qo.data()[program_row_end] <
            qo.data()[program_row_begin]) {
            if (err) *err = "ptir compose: wire query CSR is not monotonic";
            return false;
        }
        out.prog_row_counts[p] = program_row_end - program_row_begin;
        out.prog_request_starts[p] =
            static_cast<std::uint32_t>(out.qo_indptr.size() - 1);
        out.prog_request_counts[p] =
            program_row_end - program_row_begin;
        out.prog_token_counts[p] =
            qo.data()[program_row_end] - qo.data()[program_row_begin];
        out.prog_query_lens[p] = out.prog_token_counts[p];
        if (program_row_end >= kvpp.size()) {
            if (err) *err =
                "ptir compose: wire page CSR shorter than attribution";
            return false;
        }
        if (kvpp.data()[program_row_end] <
            kvpp.data()[program_row_begin]) {
            if (err) *err = "ptir compose: wire page CSR is not monotonic";
            return false;
        }
        out.prog_page_counts[p] =
            kvpp.data()[program_row_end] -
            kvpp.data()[program_row_begin];
        if (program_row_end - program_row_begin == 1) {
            const std::uint32_t request = program_row_begin;
            if (request + 1 >= kvpp.size() || request >= kvlpl.size()) {
                if (err) *err =
                    "ptir compose: wire KV CSR shorter than attribution";
                return false;
            }
            const std::uint32_t query_len =
                qo.data()[request + 1] - qo.data()[request];
            const std::uint32_t page_count =
                kvpp.data()[request + 1] - kvpp.data()[request];
            const std::uint32_t last_len = kvlpl.data()[request];
            const std::uint64_t kv_len = page_count == 0
                ? 0
                : static_cast<std::uint64_t>(page_count - 1) * page_size +
                    last_len;
            if (kv_len > std::numeric_limits<std::uint32_t>::max()) {
                if (err) *err = "ptir compose: KV extent exceeds lane ABI";
                return false;
            }
            out.prog_query_lens[p] = query_len;
            out.prog_kv_lens[p] = static_cast<std::uint32_t>(kv_len);
            out.prog_key_lens[p] = static_cast<std::uint32_t>(kv_len);
        }
        out.prog_sample_offsets[p] =
            static_cast<std::uint32_t>(out.sampling_indices.size());
        out.prog_sample_starts[p] = out.prog_sample_offsets[p];
        for (std::uint32_t r = row_indptr.data()[p];
             r < row_indptr.data()[p + 1]; ++r) {
            if (r + 1 >= qo.size() || r + 1 >= kvpp.size() ||
                r >= kvlpl.size() || r + 1 >= sptr.size()) {
                if (err) *err = "ptir compose: wire CSR shorter than attribution";
                return false;
            }
            const std::uint32_t t0 = qo.data()[r];
            const std::uint32_t t1 = qo.data()[r + 1];
            if (t1 < t0 || t1 > tok.size() || t1 > pos.size()) {
                if (err) *err = "ptir compose: wire token span out of range";
                return false;
            }
            out.token_ids.insert(out.token_ids.end(),
                                 tok.data() + t0, tok.data() + t1);
            out.position_ids.insert(out.position_ids.end(),
                                    pos.data() + t0, pos.data() + t1);
            out.qo_indptr.push_back(
                static_cast<std::uint32_t>(out.token_ids.size()));
            const std::uint32_t k0 = kvpp.data()[r];
            const std::uint32_t k1 = kvpp.data()[r + 1];
            if (k1 < k0 || k1 > kvpi.size()) {
                if (err) *err = "ptir compose: wire kv span out of range";
                return false;
            }
            const bool has_translation =
                view.kv_translation_indptr.size() == n_prog + 1;
            const std::uint32_t tr_lo = has_translation
                ? view.kv_translation_indptr.data()[p]
                : 0;
            const std::uint32_t tr_hi = has_translation
                ? view.kv_translation_indptr.data()[p + 1]
                : 0;
            if (tr_hi < tr_lo || tr_hi > view.kv_translation.size()) {
                if (err) *err = "ptir compose: malformed KV translation segment";
                return false;
            }
            for (std::uint32_t k = k0; k < k1; ++k) {
                const std::uint32_t relative = kvpi.data()[k];
                if (tr_hi > tr_lo) {
                    if (relative >= tr_hi - tr_lo) {
                        if (err) *err = "ptir compose: KV translation is unready";
                        return false;
                    }
                    out.kv_page_indices.push_back(
                        view.kv_translation.data()[tr_lo + relative]);
                } else {
                    out.kv_page_indices.push_back(relative);
                }
            }
            out.kv_page_indptr.push_back(
                static_cast<std::uint32_t>(out.kv_page_indices.size()));
            out.kv_last_page_lens.push_back(kvlpl.data()[r]);
            const std::uint64_t request_kv_len = k1 == k0
                ? 0
                : static_cast<std::uint64_t>(k1 - k0 - 1) * page_size +
                    kvlpl.data()[r];
            if (request_kv_len >
                std::numeric_limits<std::uint32_t>::max()) {
                if (err) *err =
                    "ptir compose: wire request KV extent exceeds mask ABI";
                return false;
            }
            const std::uint32_t s0 = sptr.data()[r];
            const std::uint32_t s1 = sptr.data()[r + 1];
            if (s1 < s0 || s1 > sidx.size()) {
                if (err) *err = "ptir compose: wire sampling span out of range";
                return false;
            }
            out.sampling_indices.insert(out.sampling_indices.end(),
                                        sidx.data() + s0, sidx.data() + s1);
            out.sampling_indptr.push_back(
                static_cast<std::uint32_t>(out.sampling_indices.size()));
            out.structured_masks.push_back(
                resolved.per_program[p].structured_mask);
            if (any_write_desc &&
                !detail::synthesize_wire_write_desc(
                    out, out.qo_indptr.size() - 2, page_size,
                    out.w_page, out.w_off, err)) {
                return false;
            }
        }
        out.prog_sample_counts[p] =
            static_cast<std::uint32_t>(out.sampling_indices.size()) -
            out.prog_sample_starts[p];
        append_rs(p);
    }

    // Pass 2 — device-geometry programs, appended in program order.
    for (std::size_t p = 0; p < n_prog; ++p) {
        if (!resolved.is_device_geometry[p]) continue;
        const FireGeometry& fg = resolved.per_program[p];
        const std::uint32_t request_count =
            static_cast<std::uint32_t>(fg.qo_indptr.size() - 1);
        out.prog_request_starts[p] =
            static_cast<std::uint32_t>(out.qo_indptr.size() - 1);
        out.prog_request_counts[p] = request_count;
        out.prog_row_counts[p] = request_count;
        out.prog_token_counts[p] =
            static_cast<std::uint32_t>(fg.token_ids.size());
        out.prog_query_lens[p] = out.prog_token_counts[p];
        out.prog_page_counts[p] =
            static_cast<std::uint32_t>(fg.kv_page_indices.size());
        if (request_count == 1) {
            const std::uint32_t query_len =
                fg.qo_indptr[1] - fg.qo_indptr[0];
            const std::uint32_t page_count =
                fg.kv_page_indptr.size() >= 2
                    ? fg.kv_page_indptr[1] - fg.kv_page_indptr[0]
                    : 0;
            const std::uint32_t last_len =
                fg.kv_last_page_lens.empty()
                    ? 0
                    : fg.kv_last_page_lens[0];
            const std::uint64_t kv_len = page_count == 0
                ? 0
                : static_cast<std::uint64_t>(page_count - 1) * page_size +
                    last_len;
            if (kv_len > std::numeric_limits<std::uint32_t>::max()) {
                if (err) *err = "ptir compose: resolved KV extent exceeds lane ABI";
                return false;
            }
            out.prog_query_lens[p] = query_len;
            out.prog_kv_lens[p] = static_cast<std::uint32_t>(kv_len);
            out.prog_key_lens[p] = static_cast<std::uint32_t>(kv_len);
        }
        out.prog_sample_offsets[p] =
            static_cast<std::uint32_t>(out.sampling_indices.size());
        out.prog_sample_starts[p] = out.prog_sample_offsets[p];
        const std::uint32_t tok_base =
            static_cast<std::uint32_t>(out.token_ids.size());
        const std::uint32_t page_base =
            static_cast<std::uint32_t>(out.kv_page_indices.size());
        out.token_ids.insert(out.token_ids.end(),
                             fg.token_ids.begin(), fg.token_ids.end());
        out.position_ids.insert(out.position_ids.end(),
                                fg.position_ids.begin(), fg.position_ids.end());
        for (std::size_t lane = 0; lane + 1 < fg.qo_indptr.size(); ++lane) {
            out.qo_indptr.push_back(tok_base + fg.qo_indptr[lane + 1]);
            out.kv_page_indptr.push_back(
                page_base + (lane + 1 < fg.kv_page_indptr.size()
                                 ? fg.kv_page_indptr[lane + 1]
                                 : fg.kv_page_indptr.back()));
            const std::uint32_t s1 = lane + 1 < fg.sampling_indptr.size()
                                         ? fg.sampling_indptr[lane + 1]
                                         : fg.sampling_indptr.back();
            const std::uint32_t s0 = fg.sampling_indptr[lane];
            out.sampling_indices.insert(
                out.sampling_indices.end(),
                fg.sampling_indices.begin() + s0,
                fg.sampling_indices.begin() + s1);
            out.sampling_indptr.push_back(
                static_cast<std::uint32_t>(out.sampling_indices.size()));
            out.structured_masks.push_back(fg.structured_mask);
        }
        out.prog_sample_counts[p] =
            static_cast<std::uint32_t>(out.sampling_indices.size()) -
            out.prog_sample_starts[p];
        out.kv_page_indices.insert(out.kv_page_indices.end(),
                                   fg.kv_page_indices.begin(),
                                   fg.kv_page_indices.end());
        out.kv_last_page_lens.insert(out.kv_last_page_lens.end(),
                                     fg.kv_last_page_lens.begin(),
                                     fg.kv_last_page_lens.end());
        if (any_write_desc) {
            if (fg.has_write_desc) {
                if (fg.w_page.size() != fg.token_ids.size() ||
                    fg.w_off.size() != fg.token_ids.size()) {
                    if (err) *err = "ptir compose: resolved write descriptor shape";
                    return false;
                }
                out.w_page.insert(out.w_page.end(),
                                  fg.w_page.begin(), fg.w_page.end());
                out.w_off.insert(out.w_off.end(),
                                 fg.w_off.begin(), fg.w_off.end());
            } else {
                // A mask-free device-geometry program without WSlot/WOff has
                // no defined explicit target; synthesize its standard append
                // like a wire request, lane by lane.
                for (std::size_t lane = 0; lane + 1 < fg.qo_indptr.size();
                     ++lane) {
                    const std::size_t request =
                        out.qo_indptr.size() - fg.qo_indptr.size() + lane;
                    if (!detail::synthesize_wire_write_desc(
                            out, request, page_size, out.w_page, out.w_off,
                            err)) {
                        return false;
                    }
                }
            }
        }
        append_rs(p);
    }
    out.prog_sample_offsets[n_prog] =
        static_cast<std::uint32_t>(out.sampling_indices.size());
    out.has_write_desc = any_write_desc;
    if (any_write_desc &&
        (out.w_page.size() != out.token_ids.size() ||
         out.w_off.size() != out.token_ids.size())) {
        if (err) *err = "ptir compose: composed write descriptor shape";
        return false;
    }
    const std::size_t composed_requests = out.qo_indptr.size() - 1;
    if ((has_folded_rs &&
         (out.rs_slot_ids.size() != composed_requests ||
          out.rs_slot_flags.size() != composed_requests ||
          (!input_rs_fold_lens.empty() &&
           out.rs_fold_lens.size() != composed_requests))) ||
        (has_rs_buffer &&
         out.rs_buffer_slot_indptr.size() != composed_requests + 1)) {
        if (err) *err = "ptir compose: composed RS request attribution mismatch";
        return false;
    }
    return true;
}

// Per-program logits row offsets for a PURE-WIRE launch (no device-geometry
// program resolved): the gathered order is the wire request order, so program
// `p`'s sampled rows start at `sampling_indptr[row_indptr[p]]`.
inline bool wire_program_sample_offsets(const pie_native::LaunchView& view,
                                        std::vector<std::uint32_t>& out,
                                        std::string* err) {
    const std::size_t n_prog = view.ptir_program_hashes.size();
    const auto row_indptr = view.ptir_program_row_indptr;
    const auto sptr = view.sampling_indptr;
    if (row_indptr.size() != n_prog + 1) {
        if (err) *err = "ptir compose: missing program row attribution CSR";
        return false;
    }
    out.assign(n_prog + 1, 0);
    for (std::size_t p = 0; p <= n_prog; ++p) {
        const std::uint32_t r = row_indptr.data()[p];
        if (r >= sptr.size()) {
            if (err) *err = "ptir compose: attribution beyond sampling CSR";
            return false;
        }
        out[p] = sptr.data()[r];
    }
    return true;
}

}  // namespace pie_cuda_driver::pipeline
