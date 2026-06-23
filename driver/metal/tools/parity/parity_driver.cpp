// parity_driver — dump the Metal driver's last-position logits for an exact
// token-id prompt, so an external reference (mlx-lm) can be diffed against it.
//
// Loads a real HF checkpoint via delta's loader, stages a single-request
// prefill ForwardBatch exactly as the executor does, runs charlie's ModelGraph
// over delta's PagedKvCache on the Metal GPU, and writes the `[vocab]` logits
// row of the final prompt token to a float32 .npy. Also prints argmax + top-k
// and (when an Executor is wired) cross-checks that the InProcService greedy
// sample agrees with the raw argmax — i.e. the *driver Forward path* and the
// raw graph logits pick the same token.
//
// Usage: parity_driver <hf_path> <comma_sep_token_ids> <out_logits.npy>
// Pairs with parity_check.py (the mlx-lm reference + comparison driver).

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include <pie_schema/response_builder.hpp>
#include <pie_schema/view.hpp>

#include "executor/executor.hpp"
#include "loader/model_loader.hpp"
#include "model/model_graph.hpp"
#include "service/inproc_service.hpp"

namespace mx = mlx::core;
using namespace pie_metal_driver;

static std::vector<int> parse_ids(const std::string& csv) {
    std::vector<int> ids;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) ids.push_back(std::stoi(tok));
    }
    return ids;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: parity_driver <hf_path> <comma_token_ids> "
                     "<out_logits.npy>\n";
        return 2;
    }
    const std::string hf_path = argv[1];
    const std::vector<int> ids = parse_ids(argv[2]);
    const std::string out_npy = argv[3];
    if (ids.empty()) {
        std::cerr << "[parity] no token ids parsed\n";
        return 2;
    }

    mx::set_default_device(mx::Device::gpu);
    std::cerr << "[parity] default_device = "
              << (mx::default_device() == mx::Device::gpu ? "gpu" : "cpu")
              << ", n_tokens = " << ids.size() << "\n";

    BatchingConfig batching;  // defaults: page=32, total_pages=1024
    loader::LoadedModel model = loader::load_model(hf_path, batching);
    const auto& c = model.caps;
    std::cerr << "[parity] LOADED arch=" << c.arch_name
              << " layers=" << c.num_hidden_layers
              << " heads=" << c.num_attention_heads
              << " kv_heads=" << c.num_key_value_heads
              << " head_dim=" << c.head_dim << " hidden=" << c.hidden_size
              << " vocab=" << c.vocab_size << " act=" << c.activation_dtype
              << "\n";

    const int n = static_cast<int>(ids.size());
    const int page_size = model.kv->page_size();
    if (n > page_size) {
        std::cerr << "[parity] prompt (" << n << ") exceeds one page ("
                  << page_size << "); use a shorter prompt\n";
        return 2;
    }

    std::vector<int> positions(n), write_idx(n);
    for (int i = 0; i < n; ++i) {
        positions[i] = i;
        write_idx[i] = i;  // single page 0, contiguous
    }

    // Stage the prefill ForwardBatch exactly like Executor::run_forward does
    // (single request, sample the last token row).
    model::ForwardBatch batch{
        /*token_ids=*/        mx::array(ids.data(), {n}, mx::int32),
        /*positions=*/        mx::array(positions.data(), {n}, mx::int32),
        /*logit_rows=*/       mx::array({n - 1}, {1}, mx::int32),
        /*kv_page_indices=*/  mx::array({0}, {1}, mx::int32),
        /*kv_page_indptr=*/   mx::array({0, 1}, {2}, mx::int32),
        /*kv_last_page_lens=*/mx::array({n}, {1}, mx::int32),
        /*qo_indptr=*/        mx::array({0, n}, {2}, mx::int32),
        /*kv_write_indices=*/ mx::array(write_idx.data(), {n}, mx::int32),
        /*n_total=*/    n,
        /*n_requests=*/ 1,
        /*n_slots=*/    1,
        /*pure_decode=*/false,
    };

    // Hybrid linear-attention seam (qwen3.6): single-request prefill → slot 0,
    // varlen path (qo_indptr_host={0,n}). Null/empty for non-hybrid archs.
    batch.lin_cache = model.lin_cache.get();
    batch.slot_ids  = mx::array({0}, {1}, mx::int32);
    batch.qo_indptr_host = {0, n};

    mx::array logits = model.graph->forward(batch, *model.kv);  // [1, vocab]
    mx::array row = mx::astype(mx::reshape(logits, {c.vocab_size}), mx::float32);
    mx::eval(row);

    const int argmax = mx::argmax(row, /*axis=*/0).item<int>();
    std::cerr << "[parity] raw-graph argmax token = " << argmax << "\n";

    // Top-k (descending) for a quick console glance.
    const int k = std::min(5, c.vocab_size);
    mx::array topk_idx = mx::argpartition(mx::negative(row), k - 1, 0);
    mx::eval(topk_idx);
    std::cerr << "[parity] top-" << k << " (unordered): ";
    for (int i = 0; i < k; ++i)
        std::cerr << topk_idx.data<int32_t>()[i] << " ";
    std::cerr << "\n";

    mx::save(out_npy, row);
    std::cerr << "[parity] wrote logits row -> " << out_npy << "\n";

    // Cross-check: the real driver Forward path (InProcService -> Executor ->
    // sampler) greedy-samples the same token as the raw-graph argmax.
    {
        Executor exec(*model.graph, *model.kv);
        exec.set_linear_state_cache(model.lin_cache.get());  // qwen3.6 hybrid
        // The raw-graph forward above already wrote recurrent/conv state into
        // lin_cache slot 0; reset it so this independent re-run starts clean
        // (delta's recycle contract).
        if (model.lin_cache) model.lin_cache->reset_slot(0);
        service::InProcService service(
            static_cast<std::uint32_t>(c.vocab_size));
        service.set_executor(&exec);

        std::vector<std::uint32_t> uids(ids.begin(), ids.end());
        std::vector<std::uint32_t> upos(positions.begin(), positions.end());
        std::vector<std::uint32_t> qo = {0, (std::uint32_t)n};
        std::vector<std::uint32_t> kpi = {0};
        std::vector<std::uint32_t> kpp = {0, 1};
        std::vector<std::uint32_t> lpl = {(std::uint32_t)n};
        std::vector<std::uint32_t> si = {(std::uint32_t)(n - 1)};
        std::vector<std::uint32_t> sip = {0, 1};
        std::vector<std::uint32_t> st = {pie_driver::SAMPLER_MULTINOMIAL};
        std::vector<float> stemp = {0.0f};  // temp 0 -> greedy argmax
        std::vector<std::uint32_t> stk = {0};
        std::vector<float> stp = {1.0f}, smp = {0.0f};
        std::vector<std::uint32_t> sseed = {0};

        pie_driver::PieForwardRequestView fwd{};
        fwd.token_ids         = pie_driver::slice_from_u32(uids.data(), uids.size());
        fwd.position_ids      = pie_driver::slice_from_u32(upos.data(), upos.size());
        fwd.qo_indptr         = pie_driver::slice_from_u32(qo.data(), qo.size());
        fwd.kv_page_indices   = pie_driver::slice_from_u32(kpi.data(), kpi.size());
        fwd.kv_page_indptr    = pie_driver::slice_from_u32(kpp.data(), kpp.size());
        fwd.kv_last_page_lens = pie_driver::slice_from_u32(lpl.data(), lpl.size());
        fwd.sampling_indices  = pie_driver::slice_from_u32(si.data(), si.size());
        fwd.sampling_indptr   = pie_driver::slice_from_u32(sip.data(), sip.size());
        fwd.sampler_types        = pie_driver::slice_from_u32(st.data(), st.size());
        fwd.sampler_temperatures = pie_driver::slice_from_f32(stemp.data(), stemp.size());
        fwd.sampler_top_k        = pie_driver::slice_from_u32(stk.data(), stk.size());
        fwd.sampler_top_p        = pie_driver::slice_from_f32(stp.data(), stp.size());
        fwd.sampler_min_p        = pie_driver::slice_from_f32(smp.data(), smp.size());
        fwd.sampler_seeds        = pie_driver::slice_from_u32(sseed.data(), sseed.size());
        fwd.single_token_mode = 0;

        pie_driver::PieInProcRequestView req{};
        req.method = pie_driver::PIE_METHOD_FORWARD;
        req.forward = fwd;
        pie_driver::ResponseBuilder builder;
        pie_driver::PieInProcResponseView resp{};
        service.handle_request(0, req, resp, builder);

        const auto toks = resp.forward.tokens.as<std::uint32_t>();
        const int svc_tok = toks.empty() ? -1 : static_cast<int>(toks[0]);
        std::cerr << "[parity] InProcService greedy token = " << svc_tok
                  << (svc_tok == argmax ? "  (== raw argmax OK)"
                                        : "  (!= raw argmax MISMATCH)")
                  << "\n";
    }

    std::cout << argmax << "\n";  // stdout = the driver's greedy next token
    return 0;
}
