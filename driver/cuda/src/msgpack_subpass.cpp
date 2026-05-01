#include "msgpack_subpass.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "device_buffer.hpp"
#include "kernels/dist.hpp"
#include "kernels/entropy.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/logprobs.hpp"
#include "model/qwen3_forward.hpp"
#include "sampler_type.hpp"
#include "shmem_schema.hpp"

namespace pie_cuda_driver {

namespace {

constexpr std::uint32_t TYPE_DIST     = static_cast<std::uint32_t>(SamplerType::Dist);
constexpr std::uint32_t TYPE_RAWLOG   = static_cast<std::uint32_t>(SamplerType::RawLogits);
constexpr std::uint32_t TYPE_LOGPROB  = static_cast<std::uint32_t>(SamplerType::Logprob);
constexpr std::uint32_t TYPE_LOGPROBS = static_cast<std::uint32_t>(SamplerType::Logprobs);
constexpr std::uint32_t TYPE_ENTROPY  = static_cast<std::uint32_t>(SamplerType::Entropy);

}  // namespace

void gather_raw_logits(
    const MsgpackSubpassContext& ctx,
    std::vector<response::PerRequestMsgpack>& per_req)
{
    const int V = ctx.vocab_size;
    const auto* h_qo   = ctx.qo_indptr.data();
    const auto* h_sptr = ctx.sampling_indptr.data();
    const auto* h_sidx = ctx.sampling_indices.data();

    // Pass 1: collect (req, source_row) for every RawLogits slot, in
    // slot-iteration order. `req_for_slot[i]` = the response request
    // index where the i-th gathered row's payload belongs.
    std::vector<std::int32_t> rows;
    std::vector<int> req_for_slot;
    rows.reserve(ctx.num_sampling);
    req_for_slot.reserve(ctx.num_sampling);
    for (int r = 0; r < ctx.R; ++r) {
        const std::uint32_t lo = h_sptr[r];
        const std::uint32_t hi = h_sptr[r + 1];
        const std::uint32_t qo_lo = h_qo[r];
        for (std::uint32_t k = lo; k < hi; ++k) {
            if (ctx.per_slot_type[k] != TYPE_RAWLOG) continue;
            rows.push_back(static_cast<std::int32_t>(qo_lo + h_sidx[k]));
            req_for_slot.push_back(r);
        }
    }
    if (rows.empty()) return;

    // Pass 2: one kernel launch + one D2H. Replaces the previous
    // per-slot `cudaMemcpy` loop, which serialized on the default
    // stream and incurred a launch overhead per slot.
    const std::size_t n = rows.size();
    auto d_rows   = DeviceBuffer<std::int32_t>::from_host(
        std::span<const std::int32_t>(rows));
    auto d_packed = DeviceBuffer<std::uint16_t>::alloc(n * V);
    kernels::launch_gather_bf16_rows(
        static_cast<const std::uint16_t*>(ctx.ws.logits.data()),
        d_rows.data(), d_packed.data(),
        static_cast<int>(n), V, /*stream=*/nullptr);
    const std::vector<std::uint16_t> h_packed = d_packed.to_host();

    // Pass 3: per-slot bf16 → f32 widening on host. Cheap (host-only,
    // V ≈ 150K → ~600 KB per slot) and amortizes one host alloc per
    // slot. Convention: place bf16 bits in the high 16 bits of the
    // f32 — matches `_process_raw_logits` in pie_driver.
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint16_t* src = h_packed.data() + i * V;
        std::vector<std::uint8_t> payload(V * sizeof(float));
        auto* out = reinterpret_cast<std::uint32_t*>(payload.data());
        for (int j = 0; j < V; ++j) {
            out[j] = static_cast<std::uint32_t>(src[j]) << 16;
        }
        per_req[req_for_slot[i]].logits.push_back(std::move(payload));
    }
}

void compute_entropy_slots(
    const MsgpackSubpassContext& ctx,
    std::vector<response::PerRequestMsgpack>& per_req)
{
    const auto* h_qo   = ctx.qo_indptr.data();
    const auto* h_sptr = ctx.sampling_indptr.data();
    const auto* h_sidx = ctx.sampling_indices.data();

    std::vector<std::int32_t> ent_rows;
    std::vector<int> ent_req_idx;
    ent_rows.reserve(ctx.num_sampling);
    ent_req_idx.reserve(ctx.num_sampling);
    for (int r = 0; r < ctx.R; ++r) {
        const std::uint32_t qo_lo = h_qo[r];
        for (std::uint32_t k = h_sptr[r]; k < h_sptr[r + 1]; ++k) {
            if (ctx.per_slot_type[k] != TYPE_ENTROPY) continue;
            ent_rows.push_back(static_cast<std::int32_t>(qo_lo + h_sidx[k]));
            ent_req_idx.push_back(r);
        }
    }
    if (ent_rows.empty()) return;

    auto d_ent_rows = DeviceBuffer<std::int32_t>::from_host(
        std::span<const std::int32_t>(ent_rows));
    auto d_ent_out  = DeviceBuffer<float>::alloc(ent_rows.size());
    kernels::launch_entropy_bf16(
        ctx.ws.logits.data(), d_ent_rows.data(), d_ent_out.data(),
        static_cast<int>(ent_rows.size()),
        ctx.vocab_size, /*stream=*/nullptr);
    const auto h_ent = d_ent_out.to_host();
    for (std::size_t i = 0; i < ent_req_idx.size(); ++i) {
        per_req[ent_req_idx[i]].entropies.push_back(h_ent[i]);
    }
}

void compute_logprob_slots(
    const MsgpackSubpassContext& ctx,
    const schema::DecodedRequest& dec,
    std::vector<response::PerRequestMsgpack>& per_req)
{
    namespace S = pie_cuda_driver::schema;
    const auto label_ids_view    = dec.as<std::uint32_t>(S::A_SAMPLER_LABEL_IDS);
    const auto label_indptr_view = dec.as<std::uint32_t>(S::A_SAMPLER_LABEL_INDPTR);

    const auto* h_qo   = ctx.qo_indptr.data();
    const auto* h_sptr = ctx.sampling_indptr.data();
    const auto* h_sidx = ctx.sampling_indices.data();
    const auto* h_rns  = ctx.request_num_samplers.data();

    std::vector<std::int32_t> lp_rows;
    std::vector<std::int32_t> lp_label_indptr = {0};
    std::vector<std::int32_t> lp_label_ids;
    std::vector<int> lp_req_idx;

    std::uint32_t s_off = 0;
    for (int r = 0; r < ctx.R; ++r) {
        const std::uint32_t ns =
            (ctx.request_num_samplers.size() > static_cast<std::size_t>(r)) ? h_rns[r] : 0u;
        const std::uint32_t qo_lo = h_qo[r];
        const std::uint32_t lo = h_sptr[r];
        const std::uint32_t hi = h_sptr[r + 1];
        for (std::uint32_t k = lo; k < hi; ++k) {
            const std::uint32_t type = ctx.per_slot_type[k];
            if (type != TYPE_LOGPROB && type != TYPE_LOGPROBS) continue;
            const std::uint32_t s_idx = s_off + (k - lo);
            // sampler_label_indptr is CSR with length num_samplers+1.
            const std::uint32_t li_lo =
                (s_idx < label_indptr_view.size()) ? label_indptr_view[s_idx] : 0u;
            const std::uint32_t li_hi =
                (s_idx + 1 < label_indptr_view.size()) ? label_indptr_view[s_idx + 1] : li_lo;
            const int n_labels = static_cast<int>(li_hi) - static_cast<int>(li_lo);
            lp_rows.push_back(static_cast<std::int32_t>(qo_lo + h_sidx[k]));
            for (int t = 0; t < n_labels; ++t) {
                lp_label_ids.push_back(
                    static_cast<std::int32_t>(label_ids_view[li_lo + t]));
            }
            lp_label_indptr.push_back(
                static_cast<std::int32_t>(lp_label_ids.size()));
            lp_req_idx.push_back(r);
        }
        s_off += ns;
    }
    if (lp_rows.empty()) return;

    auto d_lp_rows    = DeviceBuffer<std::int32_t>::from_host(
        std::span<const std::int32_t>(lp_rows));
    auto d_lp_lindptr = DeviceBuffer<std::int32_t>::from_host(
        std::span<const std::int32_t>(lp_label_indptr));
    // Always allocate at least 1 element so the kernel gets a valid
    // pointer for the "labels for every slot are empty" edge case.
    auto d_lp_lids = lp_label_ids.empty()
        ? DeviceBuffer<std::int32_t>::alloc(1)
        : DeviceBuffer<std::int32_t>::from_host(
              std::span<const std::int32_t>(lp_label_ids));
    auto d_lp_out = DeviceBuffer<float>::alloc(
        std::max<std::size_t>(lp_label_ids.size(), 1));

    kernels::launch_logprobs_bf16(
        ctx.ws.logits.data(), d_lp_rows.data(), d_lp_lindptr.data(),
        d_lp_lids.data(), d_lp_out.data(),
        static_cast<int>(lp_rows.size()),
        ctx.vocab_size, /*stream=*/nullptr);

    std::vector<float> h_lp(lp_label_ids.size());
    if (!lp_label_ids.empty()) {
        CUDA_CHECK(cudaMemcpy(h_lp.data(), d_lp_out.data(),
                              sizeof(float) * h_lp.size(),
                              cudaMemcpyDeviceToHost));
    }

    for (std::size_t i = 0; i < lp_req_idx.size(); ++i) {
        const std::int32_t lo = lp_label_indptr[i];
        const std::int32_t hi = lp_label_indptr[i + 1];
        per_req[lp_req_idx[i]].logprobs.emplace_back(
            h_lp.data() + lo, h_lp.data() + hi);
    }
}

void compute_dist_slots(
    const MsgpackSubpassContext& ctx,
    std::vector<response::PerRequestMsgpack>& per_req)
{
    const int V = ctx.vocab_size;
    const auto* h_qo   = ctx.qo_indptr.data();
    const auto* h_sptr = ctx.sampling_indptr.data();
    const auto* h_sidx = ctx.sampling_indices.data();

    std::vector<std::int32_t> dist_rows;
    std::vector<float> dist_temps;
    std::vector<std::int32_t> dist_topk;
    std::vector<int> dist_req_idx;
    dist_rows.reserve(ctx.num_sampling);
    dist_temps.reserve(ctx.num_sampling);
    dist_topk.reserve(ctx.num_sampling);
    dist_req_idx.reserve(ctx.num_sampling);

    for (int r = 0; r < ctx.R; ++r) {
        const std::uint32_t lo = h_sptr[r];
        const std::uint32_t hi = h_sptr[r + 1];
        const std::uint32_t qo_lo = h_qo[r];
        for (std::uint32_t k = lo; k < hi; ++k) {
            if (ctx.per_slot_type[k] != TYPE_DIST) continue;
            // per_slot_top_k was already mapped (0 → V) upstream; clamp
            // once more for safety.
            const std::int32_t Tk =
                (ctx.per_slot_top_k[k] <= 0) ? V : ctx.per_slot_top_k[k];
            dist_rows.push_back(static_cast<std::int32_t>(qo_lo + h_sidx[k]));
            dist_temps.push_back(ctx.per_slot_temp[k]);
            dist_topk.push_back(Tk);
            dist_req_idx.push_back(r);
        }
    }
    if (dist_rows.empty()) return;

    const std::size_t nd = dist_rows.size();
    auto d_dist_rows  = DeviceBuffer<std::int32_t>::from_host(
        std::span<const std::int32_t>(dist_rows));
    auto d_dist_temps = DeviceBuffer<float>::from_host(
        std::span<const float>(dist_temps));
    auto d_dist_probs = DeviceBuffer<float>::alloc(nd * static_cast<std::size_t>(V));

    kernels::launch_softmax_temp_bf16(
        ctx.ws.logits.data(), d_dist_rows.data(), d_dist_temps.data(),
        d_dist_probs.data(), static_cast<int>(nd), V, /*stream=*/nullptr);

    const std::vector<float> h_dist_probs = d_dist_probs.to_host();

    std::vector<std::pair<float, std::uint32_t>> scratch(V);
    for (std::size_t i = 0; i < nd; ++i) {
        const auto* row = h_dist_probs.data() + i * V;
        for (int j = 0; j < V; ++j) {
            scratch[j] = {row[j], static_cast<std::uint32_t>(j)};
        }
        const int K = dist_topk[i] < V ? dist_topk[i] : V;
        // Partial sort: top-K by prob descending; tie-break by lower
        // id (matches torch.topk's stable behavior).
        std::partial_sort(
            scratch.begin(), scratch.begin() + K, scratch.end(),
            [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });
        std::vector<std::uint32_t> ids(K);
        std::vector<float> probs(K);
        for (int kk = 0; kk < K; ++kk) {
            ids[kk]   = scratch[kk].second;
            probs[kk] = scratch[kk].first;
        }
        per_req[dist_req_idx[i]].dists.emplace_back(
            std::move(ids), std::move(probs));
    }
}

}  // namespace pie_cuda_driver
