#include "planner_calibration.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../cuda_check.hpp"
#include "../model/loaded_model.hpp"
#include "../store/kv_cache.hpp"
#include "../store/planner_profile_cache.hpp"
#include "forward.hpp"
#include "persistent_inputs.hpp"

namespace pie_cuda_driver {
namespace {

struct Timing {
    double mean_ms = 0.0;
    double stddev_ms = 0.0;
};

Timing summarize(std::vector<double>& samples) {
    Timing t;
    if (samples.empty()) return t;
    // The host is shared and the GPU clocks drift; a single slow sample would
    // otherwise decide the ladder. The median-of-samples mean is not worth the
    // complexity here, but discarding the worst quarter is: it removes the
    // preemption tail without assuming a distribution.
    std::sort(samples.begin(), samples.end());
    const std::size_t keep = std::max<std::size_t>(1, samples.size() * 3 / 4);
    const double sum = std::accumulate(samples.begin(),
                                       samples.begin() + keep, 0.0);
    t.mean_ms = sum / static_cast<double>(keep);
    double var = 0.0;
    for (std::size_t i = 0; i < keep; ++i) {
        const double d = samples[i] - t.mean_ms;
        var += d * d;
    }
    t.stddev_ms = keep > 1
                      ? std::sqrt(var / static_cast<double>(keep - 1))
                      : 0.0;
    return t;
}

// One synthetic forward of `R` requests carrying `tokens_per_request` each.
// Prompt length is held FIXED while the token budget varies, because that is
// how a real wave behaves: the workload sets the prompt length and the budget
// only decides how many prompts fit in one step. Coupling the two (L = N/R)
// would make attention's O(L^2) term masquerade as a budget effect.
bool run_shape(BatchEngine& engine,
               int R,
               int tokens_per_request,
               int iters,
               int warmup,
               Timing* out,
               std::string* error) {
    const int L = tokens_per_request;
    const int N = R * L;
    if (N <= 0 || N > engine.max_workspace_tokens ||
        R > engine.max_forward_requests) {
        if (error != nullptr) *error = "shape exceeds the built arena";
        return false;
    }
    const int page_size = std::max(1, engine.kv_cache.page_size());
    const int pages_per_request = (L + page_size - 1) / page_size;
    const int total_pages = R * pages_per_request;

    auto& pi = engine.inputs;
    std::vector<std::uint32_t> tokens(static_cast<std::size_t>(N), 0u);
    std::vector<std::uint32_t> positions(static_cast<std::size_t>(N), 0u);
    std::vector<std::uint32_t> qo(static_cast<std::size_t>(R) + 1);
    std::vector<std::uint32_t> kvpp(static_cast<std::size_t>(R) + 1);
    std::vector<std::uint32_t> kvlpl(static_cast<std::size_t>(R));
    std::vector<std::uint32_t> kvpi(static_cast<std::size_t>(total_pages), 0u);
    std::vector<std::uint32_t> write_page(static_cast<std::size_t>(N), 0u);
    std::vector<std::uint32_t> write_offset(static_cast<std::size_t>(N), 0u);
    std::vector<std::int32_t> sample_rows(static_cast<std::size_t>(R));

    for (int r = 0; r < R; ++r) {
        qo[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(r * L);
        kvpp[static_cast<std::size_t>(r)] =
            static_cast<std::uint32_t>(r * pages_per_request);
        kvlpl[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            L - (pages_per_request - 1) * page_size);
        sample_rows[static_cast<std::size_t>(r)] =
            static_cast<std::int32_t>(r * L + L - 1);
        for (int i = 0; i < L; ++i) {
            positions[static_cast<std::size_t>(r * L + i)] =
                static_cast<std::uint32_t>(i);
        }
    }
    qo[static_cast<std::size_t>(R)] = static_cast<std::uint32_t>(N);
    kvpp[static_cast<std::size_t>(R)] =
        static_cast<std::uint32_t>(total_pages);

    pi.tokens.copy_from_host(std::span<const std::uint32_t>(tokens));
    pi.positions.copy_from_host(std::span<const std::uint32_t>(positions));
    pi.qo_indptr.copy_from_host(std::span<const std::uint32_t>(qo));
    pi.kv_page_indices.copy_from_host(std::span<const std::uint32_t>(kvpi));
    pi.kv_page_indptr.copy_from_host(std::span<const std::uint32_t>(kvpp));
    pi.kv_last_page_lens.copy_from_host(std::span<const std::uint32_t>(kvlpl));
    pi.w_page.copy_from_host(std::span<const std::uint32_t>(write_page));
    pi.w_off.copy_from_host(std::span<const std::uint32_t>(write_offset));
    pi.sample_idx.copy_from_host(std::span<const std::int32_t>(sample_rows));
    CUDA_CHECK(cudaMemsetAsync(pi.row_valid.data(), 1,
                               static_cast<std::size_t>(N), nullptr));

    const bool is_pure_decode = (L == 1);
    engine.forward_fn.invoke_prepare(
        engine.attn_ws,
        ForwardFn::PrepareInputs{
            .qo_indptr_h = qo.data(),
            .kv_page_indices_h = kvpi.data(),
            .kv_page_indices_d =
                reinterpret_cast<const std::uint32_t*>(pi.kv_page_indices.data()),
            .kv_page_indptr_h = kvpp.data(),
            .kv_page_indptr_d =
                reinterpret_cast<const std::uint32_t*>(pi.kv_page_indptr.data()),
            .kv_last_page_lens_h = kvlpl.data(),
            .kv_last_page_lens_d =
                reinterpret_cast<const std::uint32_t*>(pi.kv_last_page_lens.data()),
            .total_tokens = N,
            .num_requests = R,
            .is_pure_decode = is_pure_decode,
        });

    ForwardFn::ForwardInputs fwd_in;
    fwd_in.token_ids = reinterpret_cast<const std::int32_t*>(pi.tokens.data());
    fwd_in.positions =
        reinterpret_cast<const std::int32_t*>(pi.positions.data());
    fwd_in.qo_indptr_d = pi.qo_indptr.data();
    fwd_in.kv_page_indices_d = pi.kv_page_indices.data();
    fwd_in.kv_page_indptr_d = pi.kv_page_indptr.data();
    fwd_in.kv_last_page_lens_d = pi.kv_last_page_lens.data();
    fwd_in.qo_indptr_h = qo.data();
    fwd_in.kv_page_indices_h = kvpi.data();
    fwd_in.kv_page_indptr_h = kvpp.data();
    fwd_in.kv_last_page_lens_h = kvlpl.data();
    fwd_in.total_tokens = N;
    fwd_in.num_requests = R;
    fwd_in.is_pure_decode = is_pure_decode;
    fwd_in.row_valid_d = pi.row_valid.data();
    fwd_in.w_page_d = pi.w_page.data();
    fwd_in.w_off_d = pi.w_off.data();
    fwd_in.has_write_desc = true;
    fwd_in.logit_row_indices_d = pi.sample_idx.data();
    fwd_in.num_logit_rows = R;
    fwd_in.emit_logits = true;
    fwd_in.runtime_window_left = -2;

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iters));
    for (int i = 0; i < warmup + iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        engine.forward_fn.invoke_body(engine.ws, engine.kv_cache,
                                      engine.attn_ws, engine.cublas, fwd_in);
        const cudaError_t sync = cudaStreamSynchronize(nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        if (sync != cudaSuccess) {
            if (error != nullptr) *error = cudaGetErrorString(sync);
            return false;
        }
        if (i < warmup) continue;
        samples.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    *out = summarize(samples);
    return true;
}

}  // namespace

namespace {
// Set once by `load_config`, read by the planner and by the boot path. A plain
// bool rather than an atomic: the write happens during config load, before any
// thread that reads it exists.
bool g_calibration_requested = false;
}  // namespace

bool planner_calibration_requested() { return g_calibration_requested; }

void set_planner_calibration_requested(bool requested) {
    g_calibration_requested = requested;
}

std::size_t calibrate_memory_planner(BatchEngine& engine,
                                     int tp_size,
                                     int kv_page_size) {
    if (!planner_calibration_requested()) return 0;

    const auto refuse = [](const char* why) -> std::size_t {
        std::cerr << "[pie-driver-cuda] planner calibration skipped: " << why
                  << "\n";
        return 0;
    };
    if (tp_size > 1) {
        // Every rank would have to run the identical sweep in lockstep or the
        // collectives inside the forward body desync and wedge the group.
        return refuse("tensor parallelism is not supported by the sweep");
    }
    if (engine.rs_cache != nullptr) {
        // Recurrent state makes a synthetic step non-idempotent: the sweep
        // would fold garbage into slots the serving run then reads.
        return refuse("recurrent-state models cannot be swept synthetically");
    }
    if (engine.forward_fn.model == nullptr) return refuse("no forward body");

    const int R = engine.max_forward_requests;
    const int max_tokens = engine.max_workspace_tokens;
    if (R <= 0 || max_tokens < R) {
        return refuse("request cap exceeds the token budget");
    }

    int dev_id = 0;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));

    engine.kv_cache.ensure_pages(1);

    const int iters = 12;
    const int warmup = 3;
    std::vector<PlannerShapeSample> measured;
    // Two axes, on purpose. The budget `N` is what we are calibrating; the
    // prompt length `L` is the workload variable we do NOT get to choose. If
    // the budget's knee sits at the same N for every L, the knee is a property
    // of the device and the model and can be cached. If it moves with L, then
    // no startup sweep can pick a budget without knowing the workload, and
    // saying so is more useful than picking one anyway.
    // The ladder of budgets we are scoring. Kept separate from `measured`
    // because a budget does NOT get its own measurement once it saturates —
    // see `effective_requests` below.
    std::vector<int> budgets;
    for (int N = R; N <= max_tokens; N *= 2) budgets.push_back(N);
    if (budgets.empty()) return refuse("no budget rung fits the built arena");

    // At prompt length L a budget can only fund `min(R, N/L)` requests. Once
    // `N/L >= R` the REQUEST cap binds, not the budget, and every larger budget
    // executes a byte-identical shape. Measuring those rungs again does not
    // measure the budget — it measures the host's noise, and then the maximin
    // below compares noise against noise. So each distinct shape is measured
    // once and every saturated budget reads that same sample.
    const auto effective_requests = [R](int budget, int L) {
        return std::min(R, budget / L);
    };
    for (int L : {1, 4, 16, 64, 256}) {
        int last_requests = 0;
        for (int N : budgets) {
            const int requests = effective_requests(N, L);
            if (requests < 1) continue;
            const int tokens = requests * L;
            if (tokens > max_tokens) continue;
            if (requests == last_requests) continue;  // saturated: same shape
            last_requests = requests;
            Timing timing;
            std::string error;
            bool ok = false;
            try {
                ok = run_shape(engine, requests, L, iters, warmup, &timing,
                               &error);
            } catch (const std::exception& e) {
                // A CUDA error is sticky, so one bad rung poisons the rest of
                // the sweep AND the process. Calibration is an opt-in
                // measurement; it must not be able to fail `load_model`.
                std::cerr << "[pie-driver-cuda] planner calibration: N=" << N
                          << " L=" << L << " threw (" << e.what()
                          << "); abandoning the sweep\n";
                return refuse("a rung faulted");
            }
            if (!ok) {
                std::cerr << "[pie-driver-cuda] planner calibration: N=" << N
                          << " L=" << L << " failed (" << error << ")\n";
                continue;
            }
            PlannerShapeSample sample;
            sample.max_forward_tokens = tokens;
            sample.max_forward_requests = requests;
            sample.tokens_per_request = L;
            sample.step_ms = timing.mean_ms;
            sample.step_ms_stddev = timing.stddev_ms;
            sample.tokens_per_s =
                timing.mean_ms > 0.0
                    ? static_cast<double>(tokens) / (timing.mean_ms / 1000.0)
                    : 0.0;
            measured.push_back(sample);
            if (engine.verbose) {
                std::cerr << "[pie-driver-cuda] planner calibration: N=" << N
                          << " L=" << L << " req=" << requests
                          << " tok=" << tokens
                          << " step=" << sample.step_ms << " ms +/- "
                          << sample.step_ms_stddev << " -> "
                          << static_cast<long long>(sample.tokens_per_s)
                          << " tok/s\n";
            }
        }
    }
    if (measured.empty()) return refuse("no shape could be measured");

    // Aggregate across prompt lengths WITHOUT weighting them: normalise each
    // budget's rate by the best rate seen at that same prompt length, then
    // score the budget by its WORST normalised rate. A budget only wins if it
    // is near-best for every prompt length, so no workload assumption enters.
    std::vector<int> prompt_lens;
    for (const auto& s : measured) prompt_lens.push_back(s.tokens_per_request);
    std::sort(prompt_lens.begin(), prompt_lens.end());
    prompt_lens.erase(std::unique(prompt_lens.begin(), prompt_lens.end()),
                      prompt_lens.end());

    // A budget resolves to the shape it can actually fund at this prompt
    // length, so saturated budgets all read the one sample that was taken
    // there instead of each carrying an independent noise draw.
    const auto rate_at = [&measured, &effective_requests](
                             int budget, int L) -> const PlannerShapeSample* {
        const int requests = effective_requests(budget, L);
        if (requests < 1) return nullptr;
        for (const auto& s : measured) {
            if (s.tokens_per_request == L &&
                s.max_forward_requests == requests) {
                return &s;
            }
        }
        return nullptr;
    };

    // Each budget carries the noise of the sample that decided its score, so
    // the step-down below can weigh the two budgets it is actually comparing.
    struct BudgetScore {
        int budget;
        double score;
        // Relative sigma of THIS budget's worst-case sample. Deliberately not
        // combined with the sigma of the best-rate sample that normalised it:
        // at the shortest prompt length every budget resolves to the same
        // shape and therefore reads the SAME sample, so folding the normaliser
        // in combined a measurement's noise with its own and inflated the
        // tolerance by sqrt(2) exactly where it did the most damage.
        double sigma;
    };
    // A prompt length where every budget funds the same shape reads the SAME
    // sample for all of them, so its ratio is identically 1.0 and it can never
    // separate two budgets. It can still set the tolerance, though: a budget
    // that is best everywhere has no argmin, so the seeded `max()` records the
    // FIRST prompt length, and on this data that is exactly one of these. The
    // result was that a rung carrying no information decided how much slower a
    // smaller budget was allowed to be -- 14% sigma there stepped down to a
    // budget measuring 18.5% slower where it mattered.
    //
    // Drop them from the scoring pass. They are not bad measurements; they just
    // answer a question nobody asked.
    const auto discriminates = [&](int L) {
        const PlannerShapeSample* first = nullptr;
        for (int budget : budgets) {
            const PlannerShapeSample* s = rate_at(budget, L);
            if (s == nullptr) continue;
            if (first == nullptr) {
                first = s;
            } else if (s != first) {
                return true;
            }
        }
        return false;
    };
    std::vector<int> scoring_lens;
    for (int L : prompt_lens) {
        if (discriminates(L)) scoring_lens.push_back(L);
    }
    if (scoring_lens.empty()) {
        return refuse("no prompt length separates the budgets");
    }
    if (scoring_lens.size() != prompt_lens.size() && engine.verbose) {
        std::cerr << "[pie-driver-cuda] planner calibration: "
                  << (prompt_lens.size() - scoring_lens.size())
                  << " of " << prompt_lens.size()
                  << " prompt lengths fund one shape for every budget and "
                  << "cannot separate them; scoring on the rest\n";
    }

    int chosen_budget = 0;
    double chosen_score = -1.0;
    double chosen_sigma = 0.0;
    std::vector<BudgetScore> budget_scores;
    for (int budget : budgets) {
        // Seeded above 1.0 so the argmin is recorded even when the budget is
        // best at EVERY prompt length. Leaving it at 1.0 meant a dominant
        // budget never entered the branch below, so its tolerance stayed 0 and
        // the arena-saving step-down could never fire — exactly the case where
        // it matters most.
        double worst = std::numeric_limits<double>::max();
        double worst_rel_sigma = 0.0;
        bool complete = true;
        for (int L : scoring_lens) {
            const PlannerShapeSample* here = rate_at(budget, L);
            if (here == nullptr || here->tokens_per_s <= 0.0) {
                complete = false;
                break;
            }
            double best_rate = 0.0;
            for (int other : budgets) {
                const PlannerShapeSample* s = rate_at(other, L);
                if (s != nullptr && s->tokens_per_s > best_rate) {
                    best_rate = s->tokens_per_s;
                }
            }
            if (best_rate <= 0.0) {
                complete = false;
                break;
            }
            const double rel = here->tokens_per_s / best_rate;
            if (rel < worst) {
                worst = rel;
                worst_rel_sigma =
                    here->step_ms > 0.0 ? here->step_ms_stddev / here->step_ms
                                        : 0.0;
            }
        }
        if (!complete) continue;
        budget_scores.push_back({budget, worst, worst_rel_sigma});
        if (worst > chosen_score) {
            chosen_score = worst;
            chosen_budget = budget;
            chosen_sigma = worst_rel_sigma;
        }
    }
    if (chosen_budget == 0) return refuse("no budget was measured at every prompt length");

    // Prefer the smallest budget that is indistinguishable from the winner:
    // extra budget that buys no rate costs arena bytes that would hold KV.
    //
    // "Indistinguishable" is decided by the noise of the TWO budgets being
    // compared, in quadrature. It used to be a single tolerance captured from
    // the winner alone, which made the whole selection turn on one rung: a
    // budget that is best at every prompt length has its argmin recorded at the
    // shortest one (nothing beats a ratio of 1.0), so the winner's tolerance
    // was whatever noise the L=1 measurement happened to carry. Two runs of the
    // same sweep on the same machine and model then disagreed -- 14% sigma on
    // that rung stepped down to a budget measuring 18.5% slower at L=256, 1%
    // sigma did not.
    double chosen_reported_score = chosen_score;
    for (const auto& candidate : budget_scores) {
        if (candidate.budget >= chosen_budget) break;
        const double tolerance = std::sqrt(candidate.sigma * candidate.sigma +
                                           chosen_sigma * chosen_sigma);
        if (candidate.score + tolerance >= chosen_score) {
            chosen_budget = candidate.budget;
            chosen_reported_score = candidate.score;
            break;
        }
    }

    PlannerProfileShape shape;
    // ONLY the token budget is pinned. `max_forward_requests` varied across the
    // sweep as a consequence of (budget, prompt length) — it was never the
    // subject of the measurement, and writing it here would silently clamp the
    // planner's request cap to whichever row happened to win.
    shape.max_forward_tokens = chosen_budget;
    // What the sweep ran inside, so a later boot can tell that it no longer
    // does. The key pins the device and the model and would happily match a
    // machine with 9 GB less to give -- which is exactly the state a leaked
    // process leaves, and exactly the state that changed the planner's pick.
    shape.budget_bytes = planner_budget_bytes();

    const auto key = make_planner_profile_key(
        prop, engine.loaded_model.hf_config(), tp_size, engine.kv_cache.format());
    std::string error;
    if (!planner_profile_cache_store(key, shape, measured, &error)) {
        std::cerr << "[pie-driver-cuda] planner calibration: could not write "
                  << planner_profile_cache_path().string() << ": " << error
                  << "\n";
        return measured.size();
    }
    std::cerr << "[pie-driver-cuda] planner calibration: measured "
              << measured.size() << " shapes across " << prompt_lens.size()
              << " prompt lengths, selected max_forward_tokens="
              << shape.max_forward_tokens
              // The SELECTED budget's score. Printing `chosen_score` here named
              // the pre-step-down winner, so a line reading "selected 1024
              // (worst-case rate 1 of best)" was quoting 2048's number.
              << " (worst-case rate " << chosen_reported_score
              << " of best; request cap and page size left to the planner, "
              << "which built this process at page_size " << kv_page_size
              << "), wrote " << planner_profile_cache_path().string() << "\n";
    return measured.size();
}

}  // namespace pie_cuda_driver
