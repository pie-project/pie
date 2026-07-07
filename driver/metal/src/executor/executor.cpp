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

std::unique_ptr<IForwardExecutor::Inflight> Executor::submit(
    const pie_driver::PieForwardRequestView& req) {
    const int n_total = static_cast<int>(req.token_ids.size());
    const int n_req   = static_cast<int>(req.qo_indptr.size()) - 1;
    const int n_slots = static_cast<int>(req.sampling_indices.size());

    // ── plan: stage host index arrays (i32_view copies them into MLX, so stg_
    //    is free to be reused by the next submit while this forward is in flight) ──
    copy_u32(req.token_ids, stg_.token_ids);
    copy_u32(req.position_ids, stg_.positions);
    copy_u32(req.sampling_indices, stg_.logit_rows);
    copy_u32(req.kv_page_indices, stg_.kv_page_indices);
    copy_u32(req.kv_page_indptr, stg_.kv_page_indptr);
    copy_u32(req.kv_last_page_lens, stg_.kv_last_page_lens);
    copy_u32(req.qo_indptr, stg_.qo_indptr);
    compute_write_indices(req);

    if (req.rs_slot_ids.size() == static_cast<std::size_t>(n_req) && n_req > 0) {
        copy_u32(req.rs_slot_ids, stg_.slot_ids);
    } else {
        stg_.slot_ids.resize(n_req > 0 ? n_req : 0);
        for (int r = 0; r < n_req; ++r) stg_.slot_ids[r] = r;
    }

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
    batch.lin_cache = lin_cache_;
    if (n_req > 0) batch.slot_ids = i32_view(stg_.slot_ids);
    batch.qo_indptr_host         = stg_.qo_indptr;
    batch.kv_page_indptr_host    = stg_.kv_page_indptr;
    batch.kv_last_page_lens_host = stg_.kv_last_page_lens;

    // ── forward + sample to a DEVICE token (lazy), async_eval WITHOUT waiting ──
    Tensor result = graph_.forward(batch, kv_);  // [n_slots, vocab] (or KV barrier)
    if (n_slots > 0) {
        std::vector<sampling::SamplerParams> params = build_sampler_params(req);
        // sample_token_device shares the exact sampling graph as sample_tokens
        // (host), so tokens are bit-identical; it returns the FINAL [n_slots] u32
        // device array — collect() items it directly (no new op, §D3.2).
        result = sampling::sample_token_device(result, params, fire_counter_);
    }
    mx::async_eval(std::vector<mx::array>{result});  // non-blocking command-buffer submit
    ++fire_counter_;

    // Copy the per-request response grouping (the wire view may not outlive the
    // deferred collect).
    const auto* sp = req.sampling_indptr.data();
    std::vector<std::uint32_t> indptr(sp, sp + req.sampling_indptr.size());
    return std::make_unique<InflightForward>(std::move(result), std::move(indptr),
                                             n_req, n_slots);
}

void Executor::collect(IForwardExecutor::Inflight& handle,
                       pie_driver::ResponseBuilder& builder,
                       pie_driver::PieForwardResponseView& out) {
    auto& h = static_cast<InflightForward&>(handle);
    per_req_.assign(h.n_req > 0 ? h.n_req : 0, pie_driver::PerRequestOutput{});
    // Block-read the ALREADY-async_eval'd result — eval() is the sync point on the
    // in-flight array (no freshly-created op, or the pipeline serializes, §D3.2).
    h.result.eval();
    if (h.n_slots > 0) {
        const std::uint32_t* toks = h.result.data<std::uint32_t>();
        const auto& sp = h.sampling_indptr;
        for (std::int32_t r = 0; r < h.n_req; ++r) {
            const std::uint32_t s0 = sp[r], s1 = sp[r + 1];
            per_req_[r].tokens.assign(toks + s0, toks + s1);
        }
    }
    builder.build(per_req_, out);
    out.num_requests = static_cast<std::uint32_t>(h.n_req);
}

void Executor::run_forward(const pie_driver::PieForwardRequestView& req,
                           pie_driver::ResponseBuilder& builder,
                           pie_driver::PieForwardResponseView& out) {
    // Synchronous path == submit + immediate collect (the sync fallback; the
    // deferred-response serve loop calls submit/collect with N+1 in between).
    std::unique_ptr<IForwardExecutor::Inflight> h = submit(req);
    collect(*h, builder, out);
}

}  // namespace pie_metal_driver
