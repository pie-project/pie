// parity_driver — dump the Metal driver's last-position logits for an exact
// token-id prompt, so an external reference (mlx-lm) can be diffed against it.
//
// Loads a real HF checkpoint via delta's loader, stages a single-request
// prefill ForwardBatch exactly as the executor does, runs charlie's ModelGraph
// over delta's PagedKvCache on the Metal GPU, and writes the `[vocab]` logits
// row of the final prompt token to a float32 .npy. Also prints argmax + top-k.
//
// Usage: parity_driver <hf_path> <comma_sep_token_ids> <out_logits.npy>
// Pairs with parity_check.py (the mlx-lm reference + comparison driver).

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "mlx/loader/model_loader.hpp"
#include "mlx/model/model_graph.hpp"

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

// Token-id chunks for the perplexity sweep. If `arg` begins with '@', it is a
// path to a file with one comma-separated chunk per line (so the model loads
// once and we accumulate NLL over the whole wikitext stream). Otherwise it is a
// single comma-separated chunk (the original single-prompt behaviour).
static std::vector<std::vector<int>> read_chunks(const std::string& arg) {
    std::vector<std::vector<int>> chunks;
    if (!arg.empty() && arg[0] == '@') {
        std::ifstream in(arg.substr(1));
        std::string line;
        while (std::getline(in, line)) {
            std::vector<int> ids = parse_ids(line);
            if (!ids.empty()) chunks.push_back(std::move(ids));
        }
    } else {
        std::vector<int> ids = parse_ids(arg);
        if (!ids.empty()) chunks.push_back(std::move(ids));
    }
    return chunks;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: parity_driver <hf_path> <comma_token_ids> "
                     "<out_logits.npy>\n";
        return 2;
    }
    const std::string hf_path = argv[1];
    const std::vector<std::vector<int>> chunks = read_chunks(argv[2]);
    const std::string out_npy = argv[3];
    if (chunks.empty() || chunks[0].empty()) {
        std::cerr << "[parity] no token ids parsed\n";
        return 2;
    }
    const std::vector<int>& ids = chunks[0];

    mx::set_default_device(mx::Device::gpu);
    std::cerr << "[parity] default_device = "
              << (mx::default_device() == mx::Device::gpu ? "gpu" : "cpu")
              << ", n_tokens = " << ids.size() << "\n";

    BatchingConfig batching;  // defaults: page=32, total_pages=1024
    // Long-context (perplexity) windows: size one KV page to the longest chunk
    // (the prefill/decode paths here use a single page) and shrink the pool so
    // the larger page doesn't blow up KV memory. Only grows past the 32 default.
    int max_len = 0;
    for (const auto& ch : chunks) max_len = std::max(max_len, static_cast<int>(ch.size()));
    if (max_len > static_cast<int>(batching.kv_page_size)) {
        batching.kv_page_size = static_cast<std::uint32_t>(max_len);
        batching.total_pages  = 8;
        std::cerr << "[parity] long-ctx: kv_page_size=" << batching.kv_page_size
                  << " total_pages=" << batching.total_pages << "\n";
    }
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

    // Isolation mode (PIE_PARITY_DECODE=1): feed the prompt one token at a time
    // through the *decode* path (linear-attn T=1 recurrence carried in
    // lin_cache, full-attn KV grown per step) instead of the single varlen
    // prefill. Decisive for localizing the qwen3.6 length-dependent drift: if
    // decode-loop matches mlx-lm but varlen-prefill doesn't, the bug is in
    // gated_delta_net_varlen; if both drift equally, it's a formula diff vs the
    // reference (decode and varlen share identical per-step prep + recurrence).
    const char* dec = std::getenv("PIE_PARITY_DECODE");
    const bool decode_mode = dec != nullptr && dec[0] != '\0' && dec[0] != '0';

    // Perplexity mode (PIE_PARITY_PPL=1): teacher-forced NLL over the sequence.
    // Emits all-position logits [n, vocab] in ONE forward, then computes
    // NLL = Σ_{t<n-1} −log_softmax(logits[t])[ids[t+1]] on-GPU and prints
    // "PPL_NLL <total_nll> <n_predictions>" (avoids dumping a 512×vocab npy).
    // A host script slides wikitext at fixed ctx-stride and aggregates ->
    // ppl = exp(ΣNLL / Σtok). Default uses the varlen prefill (one pass, fast);
    // PIE_PARITY_DECODE=1 forces the token-by-token decode loop instead (the
    // arch-safe cross-check for GDN/qwen3.6 — validate prefill≈decode once).
    const char* pplenv = std::getenv("PIE_PARITY_PPL");
    const bool ppl_mode = pplenv != nullptr && pplenv[0] != '\0' && pplenv[0] != '0';
    if (ppl_mode) {
        // Per-chunk teacher-forced NLL (load-once; accumulate across chunks).
        auto chunk_nll = [&](const std::vector<int>& cids) -> std::pair<double, int> {
            const int cn = static_cast<int>(cids.size());
            if (cn < 2) return {0.0, 0};
            std::vector<int> pos(cn), widx(cn), rows(cn);
            for (int i = 0; i < cn; ++i) { pos[i] = i; widx[i] = i; rows[i] = i; }
            mx::array all = [&] {
                if (!decode_mode) {
                    model::ForwardBatch batch{
                        /*token_ids=*/        mx::array(cids.data(), {cn}, mx::int32),
                        /*positions=*/        mx::array(pos.data(), {cn}, mx::int32),
                        /*logit_rows=*/       mx::array(rows.data(), {cn}, mx::int32),
                        /*kv_page_indices=*/  mx::array({0}, {1}, mx::int32),
                        /*kv_page_indptr=*/   mx::array({0, 1}, {2}, mx::int32),
                        /*kv_last_page_lens=*/mx::array({cn}, {1}, mx::int32),
                        /*qo_indptr=*/        mx::array({0, cn}, {2}, mx::int32),
                        /*kv_write_indices=*/ mx::array(widx.data(), {cn}, mx::int32),
                        /*n_total=*/    cn,
                        /*n_requests=*/ 1,
                        /*n_slots=*/    1,
                        /*pure_decode=*/false,
                    };
                    if (model.lin_cache) model.lin_cache->reset_slot(0);
                    batch.lin_cache = model.lin_cache.get();
                    batch.slot_ids  = mx::array({0}, {1}, mx::int32);
                    batch.qo_indptr_host = {0, cn};
                    return mx::astype(model.graph->forward(batch, *model.kv),
                                      mx::float32);  // [cn, vocab]
                }
                if (model.lin_cache) model.lin_cache->reset_slot(0);
                std::vector<mx::array> rws;
                rws.reserve(cn);
                for (int t = 0; t < cn; ++t) {
                    model::ForwardBatch step{
                        /*token_ids=*/        mx::array({cids[t]}, {1}, mx::int32),
                        /*positions=*/        mx::array({t}, {1}, mx::int32),
                        /*logit_rows=*/       mx::array({0}, {1}, mx::int32),
                        /*kv_page_indices=*/  mx::array({0}, {1}, mx::int32),
                        /*kv_page_indptr=*/   mx::array({0, 1}, {2}, mx::int32),
                        /*kv_last_page_lens=*/mx::array({t + 1}, {1}, mx::int32),
                        /*qo_indptr=*/        mx::array({0, 1}, {2}, mx::int32),
                        /*kv_write_indices=*/ mx::array({t}, {1}, mx::int32),
                        /*n_total=*/    1,
                        /*n_requests=*/ 1,
                        /*n_slots=*/    1,
                        /*pure_decode=*/true,
                    };
                    step.lin_cache = model.lin_cache.get();
                    step.slot_ids  = mx::array({0}, {1}, mx::int32);
                    step.qo_indptr_host = {0, 1};
                    rws.push_back(mx::astype(
                        mx::reshape(model.graph->forward(step, *model.kv),
                                    {1, c.vocab_size}), mx::float32));
                    mx::eval(rws.back());
                }
                return mx::concatenate(rws, /*axis=*/0);  // [cn, vocab]
            }();
            const int np = cn - 1;  // score all-but-first: t<n-1 predicts t+1
            mx::array pred = mx::slice(all, {0, 0}, {np, c.vocab_size});
            std::vector<int> tgt(np);
            for (int t = 0; t < np; ++t) tgt[t] = cids[t + 1];
            mx::array tcol = mx::array(tgt.data(), {np, 1}, mx::int32);
            mx::array lse  = mx::logsumexp(pred, /*axis=*/1, /*keepdims=*/true);
            mx::array got  = mx::take_along_axis(pred, tcol, /*axis=*/1);
            mx::array nll  = mx::sum(mx::subtract(lse, got));
            mx::eval(nll);
            return {static_cast<double>(nll.item<float>()), np};
        };

        std::cerr << "[parity] PPL mode (" << (decode_mode ? "decode-loop" : "prefill")
                  << ") chunks=" << chunks.size() << " vocab=" << c.vocab_size << "\n";
        double tot_nll = 0.0;
        long   tot_np  = 0;
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            auto [nll, np] = chunk_nll(chunks[ci]);
            tot_nll += nll;
            tot_np  += np;
            if (chunks.size() > 1 && (ci % 20 == 0 || ci + 1 == chunks.size())) {
                std::cerr << "[parity] chunk " << (ci + 1) << "/" << chunks.size()
                          << "  cum_nll=" << tot_nll << " cum_tok=" << tot_np
                          << " ppl=" << std::exp(tot_nll / std::max(1L, tot_np)) << "\n";
            }
        }
        std::cout << "PPL_NLL " << tot_nll << " " << tot_np << "\n";
        return 0;
    }

    mx::array logits = [&] {
        if (!decode_mode) {
            // Single varlen prefill (single request, sample the last row).
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
            batch.lin_cache = model.lin_cache.get();
            batch.slot_ids  = mx::array({0}, {1}, mx::int32);
            batch.qo_indptr_host = {0, n};
            return model.graph->forward(batch, *model.kv);  // [1, vocab]
        }

        // Decode loop: one token per step, state carried across steps.
        std::cerr << "[parity] DECODE-loop isolation (token-by-token)\n";
        if (model.lin_cache) model.lin_cache->reset_slot(0);
        mx::array last = mx::zeros({1, c.vocab_size}, mx::float32);
        for (int t = 0; t < n; ++t) {
            model::ForwardBatch step{
                /*token_ids=*/        mx::array({ids[t]}, {1}, mx::int32),
                /*positions=*/        mx::array({t}, {1}, mx::int32),
                /*logit_rows=*/       mx::array({0}, {1}, mx::int32),
                /*kv_page_indices=*/  mx::array({0}, {1}, mx::int32),
                /*kv_page_indptr=*/   mx::array({0, 1}, {2}, mx::int32),
                /*kv_last_page_lens=*/mx::array({t + 1}, {1}, mx::int32),
                /*qo_indptr=*/        mx::array({0, 1}, {2}, mx::int32),
                /*kv_write_indices=*/ mx::array({t}, {1}, mx::int32),
                /*n_total=*/    1,
                /*n_requests=*/ 1,
                /*n_slots=*/    1,
                /*pure_decode=*/true,
            };
            step.lin_cache = model.lin_cache.get();
            step.slot_ids  = mx::array({0}, {1}, mx::int32);
            step.qo_indptr_host = {0, 1};
            last = model.graph->forward(step, *model.kv);  // [1, vocab]
            mx::eval(last);  // force state writes before the next step
        }
        return last;
    }();


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

    std::cout << argmax << "\n";  // stdout = the driver's greedy next token
    return 0;
}
