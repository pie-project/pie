#include "executor/executor.hpp"

#include <mlx/mlx.h>

namespace pie_metal_driver {

namespace mx = mlx::core;

namespace {

// Copy a wire u32 slice into a host int32 buffer.
template <typename Slice>
void copy_u32(const Slice& s, std::vector<int>& dst) {
    dst.resize(s.size());
    const auto* p = s.data();
    for (std::size_t i = 0; i < s.size(); ++i) dst[i] = static_cast<int>(p[i]);
}

// Wrap a host int32 buffer as a 1-D MLX array. Copies the (small) index data
// into MLX, so the staging buffer need not outlive the array. Portable across
// MLX 0.29/0.32 (the zero-copy void* ctor is version-specific; these arrays are
// tiny so the copy is negligible).
Tensor i32_view(std::vector<int>& buf) {
    return mx::array(buf.data(), mx::Shape{static_cast<int>(buf.size())},
                     mx::int32);
}

}  // namespace

Executor::Executor(model::ModelGraph& graph, KvCacheView& kv)
    : graph_(graph), kv_(kv) {}

void Executor::compute_write_indices(
    const pie_driver::PieForwardRequestView& req) {
    const int page_size = kv_.page_size();
    const int n_req = static_cast<int>(req.qo_indptr.size()) - 1;

    stg_.kv_write_indices.assign(req.token_ids.size(), 0);
    const auto* qo  = req.qo_indptr.data();
    const auto* kpp = req.kv_page_indptr.data();
    const auto* kpi = req.kv_page_indices.data();
    const auto* pos = req.position_ids.data();

    for (int r = 0; r < n_req; ++r) {
        const int t0 = static_cast<int>(qo[r]);
        const int t1 = static_cast<int>(qo[r + 1]);
        const int page_base = static_cast<int>(kpp[r]);
        for (int i = t0; i < t1; ++i) {
            const int p = static_cast<int>(pos[i]);
            const int slot_page = p / page_size;
            const int within = p % page_size;
            const int phys_page = static_cast<int>(kpi[page_base + slot_page]);
            stg_.kv_write_indices[i] = phys_page * page_size + within;
        }
    }
}

std::vector<sampling::SamplerParams> Executor::build_sampler_params(
    const pie_driver::PieForwardRequestView& req) const {
    const std::size_t n = req.sampling_indices.size();
    std::vector<sampling::SamplerParams> params(n);

    const auto* types = req.sampler_types.data();
    const auto* temps = req.sampler_temperatures.data();
    const auto* topk  = req.sampler_top_k.data();
    const auto* topp  = req.sampler_top_p.data();
    const auto* minp  = req.sampler_min_p.data();
    const auto* seeds = req.sampler_seeds.data();

    for (std::size_t j = 0; j < n; ++j) {
        auto& sp = params[j];
        if (types) sp.type = static_cast<sampling::SamplerType>(types[j]);
        if (temps) sp.temperature = temps[j];
        if (topk)  sp.top_k = topk[j];
        if (topp)  sp.top_p = topp[j];
        if (minp)  sp.min_p = minp[j];
        if (seeds) sp.seed = seeds[j];
    }
    return params;
}

void Executor::run_forward(const pie_driver::PieForwardRequestView& req,
                           pie_driver::ResponseBuilder& builder,
                           pie_driver::PieForwardResponseView& out) {
    const int n_total = static_cast<int>(req.token_ids.size());
    const int n_req   = static_cast<int>(req.qo_indptr.size()) - 1;
    const int n_slots = static_cast<int>(req.sampling_indices.size());

    // ── plan: stage host index arrays ──
    copy_u32(req.token_ids, stg_.token_ids);
    copy_u32(req.position_ids, stg_.positions);
    copy_u32(req.sampling_indices, stg_.logit_rows);
    copy_u32(req.kv_page_indices, stg_.kv_page_indices);
    copy_u32(req.kv_page_indptr, stg_.kv_page_indptr);
    copy_u32(req.kv_last_page_lens, stg_.kv_last_page_lens);
    copy_u32(req.qo_indptr, stg_.qo_indptr);
    compute_write_indices(req);

    // Per-request linear-attention state slots (qwen3.6). Prefer the wire
    // `rs_slot_ids`; for paths that don't carry them (e.g. single-request
    // parity) fall back to identity slots 0..n_req-1.
    if (req.rs_slot_ids.size() == static_cast<std::size_t>(n_req) && n_req > 0) {
        copy_u32(req.rs_slot_ids, stg_.slot_ids);
    } else {
        stg_.slot_ids.resize(n_req > 0 ? n_req : 0);
        for (int r = 0; r < n_req; ++r) stg_.slot_ids[r] = r;
    }

    // ── stage: build the ForwardBatch (aggregate-init; Tensor has no
    // default ctor so every field must be provided up front) ──
    model::ForwardBatch batch{
        /*token_ids=*/        i32_view(stg_.token_ids),
        /*positions=*/        i32_view(stg_.positions),
        /*logit_rows=*/       i32_view(stg_.logit_rows),
        /*kv_page_indices=*/  i32_view(stg_.kv_page_indices),
        /*kv_page_indptr=*/   i32_view(stg_.kv_page_indptr),
        /*kv_last_page_lens=*/i32_view(stg_.kv_last_page_lens),
        /*qo_indptr=*/        i32_view(stg_.qo_indptr),
        /*kv_write_indices=*/ i32_view(stg_.kv_write_indices),
        /*n_total=*/          n_total,
        /*n_requests=*/       n_req,
        /*n_slots=*/          n_slots,
        /*pure_decode=*/      req.single_token_mode != 0,
    };

    // ── hybrid linear-attention seam (qwen3.6) — null/empty for other archs ──
    batch.lin_cache = lin_cache_;
    if (n_req > 0) batch.slot_ids = i32_view(stg_.slot_ids);
    batch.qo_indptr_host = stg_.qo_indptr;  // host CSR for the varlen path

    // ── forward + sample ──
    per_req_.assign(n_req > 0 ? n_req : 0, pie_driver::PerRequestOutput{});

    if (n_slots > 0) {
        Tensor logits = graph_.forward(batch, kv_);  // [n_slots, vocab]
        std::vector<sampling::SamplerParams> params = build_sampler_params(req);
        std::vector<std::uint32_t> tokens =
            sampling::sample_tokens(logits, params, fire_counter_);

        // ── pack: group per request via the sampling CSR ──
        const auto* sp_indptr = req.sampling_indptr.data();
        for (int r = 0; r < n_req; ++r) {
            const int s0 = static_cast<int>(sp_indptr[r]);
            const int s1 = static_cast<int>(sp_indptr[r + 1]);
            per_req_[r].tokens.assign(tokens.begin() + s0, tokens.begin() + s1);
        }
    } else {
        // KV-fill / prefill-only pass: still advance the cache via the graph,
        // but produce no sampled tokens.
        Tensor dummy = graph_.forward(batch, kv_);
        mx::eval(dummy);
    }

    ++fire_counter_;
    builder.build(per_req_, out);
    out.num_requests = static_cast<std::uint32_t>(n_req);
}

}  // namespace pie_metal_driver
