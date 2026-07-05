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
// composed request order); `PtirDispatch::run` slices each program's logits
// base from it.
//
// Out of scope (v1, enforced by the runtime scheduler + defended here):
//   * dense device masks (`fg.has_mask`) in a MULTI-program batch
//   * wire custom (non-causal) BRLE masks co-batched with device geometry
//   * recurrent-state (rs_cache) models with device-geometry fires

#include <cstdint>
#include <string>
#include <vector>

#include "pie_native/launch_view.hpp"

#include "ptir/fire_geometry.hpp"

namespace pie_cuda_driver::ptir {

struct ComposedBatch {
    std::vector<std::uint32_t> token_ids;
    std::vector<std::uint32_t> position_ids;
    std::vector<std::uint32_t> qo_indptr;
    std::vector<std::uint32_t> kv_page_indices;
    std::vector<std::uint32_t> kv_page_indptr;
    std::vector<std::uint32_t> kv_last_page_lens;
    std::vector<std::uint32_t> sampling_indices;  // request-relative rows
    std::vector<std::uint32_t> sampling_indptr;   // per composed REQUEST
    // Explicit KV-write descriptor covering EVERY composed token row
    // (`has_write_desc` routes the whole forward's KV append through it, so
    // wire rows get their standard append target synthesized).
    std::vector<std::uint32_t> w_page;
    std::vector<std::uint32_t> w_off;
    // Per-PROGRAM offset of each program's sampled rows within the gathered
    // logits buffer (`n_prog + 1` entries, last = total sampled rows).
    std::vector<std::uint32_t> prog_sample_offsets;
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

    out = ComposedBatch{};
    out.qo_indptr.push_back(0);
    out.kv_page_indptr.push_back(0);
    out.sampling_indptr.push_back(0);
    out.prog_sample_offsets.assign(n_prog + 1, 0);

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
        out.prog_sample_offsets[p] =
            static_cast<std::uint32_t>(out.sampling_indices.size());
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
            out.kv_page_indices.insert(out.kv_page_indices.end(),
                                       kvpi.data() + k0, kvpi.data() + k1);
            out.kv_page_indptr.push_back(
                static_cast<std::uint32_t>(out.kv_page_indices.size()));
            out.kv_last_page_lens.push_back(kvlpl.data()[r]);
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
            if (any_write_desc &&
                !detail::synthesize_wire_write_desc(
                    out, out.qo_indptr.size() - 2, page_size,
                    out.w_page, out.w_off, err)) {
                return false;
            }
        }
    }

    // Pass 2 — device-geometry programs, appended in program order.
    for (std::size_t p = 0; p < n_prog; ++p) {
        if (!resolved.is_device_geometry[p]) continue;
        const FireGeometry& fg = resolved.per_program[p];
        out.prog_sample_offsets[p] =
            static_cast<std::uint32_t>(out.sampling_indices.size());
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
        }
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

}  // namespace pie_cuda_driver::ptir
