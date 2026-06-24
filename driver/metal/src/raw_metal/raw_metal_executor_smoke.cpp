// raw_metal_executor_smoke — correctness gate for alpha's RawMetalExecutor: the
// runtime-ABI MARSHALING half of the e2e seam (delta's decoder_smoke gates the
// decoder body; this gates token/position/sampling-CSR extraction + response
// packing through the real PieForwardRequestView/ResponseBuilder types).
//
// Builds the golden qwen3.6 pos-7 prompt as a PieForwardRequestView exactly the
// way the runtime does, drives RawMetalExecutor::run_forward, and asserts the
// returned greedy token == 264. Two modes:
//   A) one prefill forward of all 8 tokens (sampling the last row);
//   B) prefill the first 7 tokens in one call, then decode the 8th in a SECOND
//      run_forward call — validates the in-heap state persists across calls
//      (the seamless prefill→decode the design hinges on).
//
// Usage: raw_metal_executor_smoke <checkpoint_dir> <kernels_dir>
//        [comma_prompt_ids] [expect_argmax]

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <pie_schema/response_builder.hpp>
#include <pie_schema/view.hpp>

#include "raw_metal/raw_metal_executor.hpp"

using namespace pie_metal_driver;

namespace {

// Build + run one FORWARD over the executor; returns the first sampled token.
int run_forward_once(raw_metal::RawMetalExecutor& exec,
                     const std::vector<std::uint32_t>& ids,
                     const std::vector<std::uint32_t>& pos,
                     std::uint32_t sample_row) {
    const std::uint32_t n = static_cast<std::uint32_t>(ids.size());
    std::vector<std::uint32_t> qo  = {0, n};
    std::vector<std::uint32_t> kpi = {0};
    std::vector<std::uint32_t> kpp = {0, 1};
    std::vector<std::uint32_t> lpl = {n};
    std::vector<std::uint32_t> si  = {sample_row};
    std::vector<std::uint32_t> sip = {0, 1};
    std::vector<std::uint32_t> st  = {pie_driver::SAMPLER_MULTINOMIAL};
    std::vector<float>         stemp = {0.0f};  // temp 0 → greedy
    std::vector<std::uint32_t> stk = {0};
    std::vector<float>         stp = {1.0f}, smp = {0.0f};
    std::vector<std::uint32_t> sseed = {0};

    pie_driver::PieForwardRequestView fwd{};
    fwd.token_ids         = pie_driver::slice_from_u32(ids.data(), ids.size());
    fwd.position_ids      = pie_driver::slice_from_u32(pos.data(), pos.size());
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
    fwd.single_token_mode = (n == 1) ? 1 : 0;

    pie_driver::ResponseBuilder builder;
    pie_driver::PieForwardResponseView out{};
    exec.run_forward(fwd, builder, out);
    const auto toks = out.tokens.as<std::uint32_t>();
    return toks.empty() ? -1 : static_cast<int>(toks[0]);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <checkpoint_dir> <kernels_dir> [comma_prompt_ids] "
            "[expect_argmax]\n", argv[0]);
        return 2;
    }
    const std::string ckpt_dir    = argv[1];
    const std::string kernels_dir = argv[2];
    const std::string ids_csv =
        (argc > 3) ? argv[3] : "785,6722,315,9625,374,264,3460,304";
    const std::uint32_t expect =
        (argc > 4) ? std::uint32_t(std::stoul(argv[4])) : 264u;

    std::vector<std::uint32_t> ids;
    for (std::size_t p = 0; p < ids_csv.size();) {
        std::size_t c = ids_csv.find(',', p);
        if (c == std::string::npos) c = ids_csv.size();
        ids.push_back(std::uint32_t(std::stoul(ids_csv.substr(p, c - p))));
        p = c + 1;
    }
    const std::uint32_t n = static_cast<std::uint32_t>(ids.size());
    std::vector<std::uint32_t> pos(n);
    for (std::uint32_t i = 0; i < n; ++i) pos[i] = i;

    auto exec = raw_metal::make_raw_metal_executor(ckpt_dir, kernels_dir, 0);
    if (!exec) {
        std::fprintf(stderr, "[exec_smoke] executor init failed\n");
        return 1;
    }

    // Mode A: single prefill forward of all tokens, sample the last row.
    const int tok_a = run_forward_once(*exec, ids, pos, n - 1);
    const bool ok_a = (tok_a == static_cast<int>(expect));
    std::printf("[exec_smoke] A (single prefill, %u toks): token=%d expect=%u → %s\n",
                n, tok_a, expect, ok_a ? "PASS" : "FAIL");

    // Mode B: prefill first n-1 (pos 0 resets state), then decode the last token
    // in a SECOND call — validates state persistence across run_forward calls.
    bool ok_b = true;
    if (n >= 2) {
        std::vector<std::uint32_t> head(ids.begin(), ids.end() - 1);
        std::vector<std::uint32_t> hpos(pos.begin(), pos.end() - 1);
        run_forward_once(*exec, head, hpos, n - 2);  // prefill (resets on pos 0)
        std::vector<std::uint32_t> tail = {ids[n - 1]};
        std::vector<std::uint32_t> tpos = {n - 1};
        const int tok_b = run_forward_once(*exec, tail, tpos, 0);  // decode step
        ok_b = (tok_b == static_cast<int>(expect));
        std::printf("[exec_smoke] B (prefill %u + decode 1, cross-call state): "
                    "token=%d expect=%u → %s\n",
                    n - 1, tok_b, expect, ok_b ? "PASS" : "FAIL");
    }

    const bool ok = ok_a && ok_b;
    std::printf("[exec_smoke] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
