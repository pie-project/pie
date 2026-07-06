// qwen3_forward — real-weight Metal Qwen3-0.6B forward + golden export.
//
// Loads the actual Qwen/Qwen3-0.6B model.safetensors (BF16, HF [out,in] layout —
// exactly matmul_xwt's W layout), runs a real token sequence through the full
// Metal forward (embedding → 28× decoder_layer → final RMSNorm → LM head), and:
//   • self-validates the Metal logits against the CPU f32 reference (same math);
//   • prints the argmax + top-k next-token logits;
//   • writes a golden vector (tokens, positions, weight source/layout, Metal
//     logits summary) for the CUDA cross-check on the 4090.
//
// Usage: qwen3_forward <model.safetensors> [kernels_dir] [out_golden.txt]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "arch.hpp"
#include "chain.hpp"
#include "metal_harness.hpp"
#include "reference.hpp"
#include "weights.hpp"

#ifndef QWEN3_KERNELS_DIR
#define QWEN3_KERNELS_DIR "."
#endif

using namespace ptir_metal;
namespace A = qwen3::arch;
using qwen3::SafeTensors;

namespace {

std::vector<int> topk_indices(const float* row, int vocab, int k) {
    std::vector<int> idx(vocab);
    for (int i = 0; i < vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return row[a] > row[b]; });
    idx.resize(k);
    return idx;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.safetensors> [kernels_dir] [golden.txt]\n", argv[0]); return 2; }
    std::string model = argv[1];
    std::string kernels_dir = argc > 2 ? argv[2] : QWEN3_KERNELS_DIR;
    std::string golden_path = argc > 3 ? argv[3] : "qwen3_golden.txt";

    SafeTensors st;
    if (!st.open(model)) { std::fprintf(stderr, "cannot open %s\n", model.c_str()); return 2; }
    std::printf("loaded safetensors: %s (%.2f GiB)\n", model.c_str(), st.file_size / (1024.0 * 1024 * 1024));

    qwen3::ref::LayerDims D;
    D.hidden = A::HIDDEN; D.n_q_heads = A::N_Q_HEADS; D.n_kv_heads = A::N_KV_HEADS;
    D.head_dim = A::HEAD_DIM; D.intermediate = A::INTERMEDIATE;
    D.rms_eps = A::RMS_EPS; D.rope_theta = A::ROPE_THETA;
    D.attn_scale = 1.0f / std::sqrt((float)A::HEAD_DIM);

    // Real token sequence (raw ids; the golden records them so CUDA uses the same).
    std::vector<std::int32_t> tokens = {9707, 3838, 374, 279, 6722, 315, 9625, 30};  // "Hello ... capital of France?"
    std::vector<std::int32_t> positions;
    for (std::size_t i = 0; i < tokens.size(); ++i) positions.push_back((int)i);
    D.N = (int)tokens.size();

    std::printf("loading weights (28 layers + embed + lm_head)...\n");
    std::vector<float> embed, lm_head, final_norm;
    if (!st.load_f32("model.embed_tokens.weight", embed) ||
        !st.load_f32("model.norm.weight", final_norm)) return 2;
    if (!st.load_f32("lm_head.weight", lm_head)) lm_head = embed;  // tied fallback
    std::vector<qwen3::ref::LayerWeights> layers(A::N_LAYERS);
    for (int l = 0; l < A::N_LAYERS; ++l)
        if (!qwen3::load_layer(st, l, layers[l])) return 2;
    std::printf("weights loaded. running forward (N=%d tokens)...\n", D.N);

    // CPU f32 reference forward (self-validation oracle).
    auto ref_logits = qwen3::ref::full_forward(tokens, positions, embed, layers, final_norm, lm_head, D, A::VOCAB);

    // Metal forward.
    MetalHarness h;
    if (!h.ok()) { std::fprintf(stderr, "no Metal device: %s\n", h.error().c_str()); return 2; }
    if (!h.load_library(kernels_dir + "/layer.metal")) { std::fprintf(stderr, "%s\n", h.error().c_str()); return 2; }
    qwen3::Chain ch{h};
    ch.use_mps = !(std::getenv("QWEN3_GEMM") && std::string(std::getenv("QWEN3_GEMM")) == "seq");
    std::printf("GEMM backend: %s\n", ch.use_mps ? "MPS (MPSMatrixMultiplication)" : "sequential kernel");
    auto t0 = std::chrono::steady_clock::now();
    auto x = ch.embedding(embed, tokens, D.N, D.hidden);
    for (int l = 0; l < A::N_LAYERS; ++l) x = ch.layer(x, positions, layers[l], D);
    x = ch.rmsnorm(x, final_norm, D.N, D.hidden, D.rms_eps);
    auto metal_logits = ch.matmul(x, lm_head, D.N, A::VOCAB, D.hidden);
    auto t1 = std::chrono::steady_clock::now();
    if (!ch.ok) { std::fprintf(stderr, "metal forward failed: %s\n", h.error().c_str()); return 2; }
    double metal_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Metal forward: %.1f ms (28 layers + LM head, N=%d)\n", metal_ms, D.N);

    // Self-validation: Metal vs CPU ref.
    double max_abs = 0.0;
    for (std::size_t i = 0; i < ref_logits.size(); ++i)
        max_abs = std::max(max_abs, (double)std::fabs(metal_logits[i] - ref_logits[i]));
    std::printf("Metal-vs-CPUref logits max_abs = %.3e (%zu logits)\n", max_abs, ref_logits.size());

    // Last-token next-token prediction (the useful output).
    const float* last = metal_logits.data() + (std::size_t)(D.N - 1) * A::VOCAB;
    auto top = topk_indices(last, A::VOCAB, 10);
    std::printf("next-token argmax = %d (logit %.5f)\n", top[0], last[top[0]]);
    std::printf("top-10:");
    for (int t : top) std::printf(" %d(%.3f)", t, last[t]);
    std::printf("\n");

    // Golden export for the CUDA cross-check.
    std::ofstream g(golden_path);
    g << "# Qwen3-0.6B Metal forward golden — cross-check on CUDA (4090)\n";
    g << "model: Qwen/Qwen3-0.6B model.safetensors (BF16, HF [out,in] layout)\n";
    g << "backend: Metal (Apple " << h.device_name() << "), f32 compute, fast-math OFF\n";
    g << "dims: hidden=" << A::HIDDEN << " head_dim=" << A::HEAD_DIM << " n_q=" << A::N_Q_HEADS
      << " n_kv=" << A::N_KV_HEADS << " I=" << A::INTERMEDIATE << " layers=" << A::N_LAYERS
      << " rope_theta=" << A::ROPE_THETA << " rms_eps=" << A::RMS_EPS << " vocab=" << A::VOCAB << "\n";
    g << "tokens:";
    for (int t : tokens) g << " " << t;
    g << "\npositions:";
    for (int p : positions) g << " " << p;
    g << "\nself_validation_max_abs_vs_cpuref: " << max_abs << "\n";
    g << "next_token_argmax: " << top[0] << "\n";
    auto top20 = topk_indices(last, A::VOCAB, 20);
    g << "last_token_top20_idx_logit:";
    for (int t : top20) g << " " << t << ":" << last[t];
    g << "\n";
    // Compact integrity summary over the full last-token logit vector (a tight
    // cross-check target without committing 151936 values).
    double lsum = 0.0, lsq = 0.0, lmin = last[0], lmax = last[0];
    for (int i = 0; i < A::VOCAB; ++i) {
        double v = last[i];
        lsum += v; lsq += v * v;
        lmin = std::min(lmin, (double)last[i]); lmax = std::max(lmax, (double)last[i]);
    }
    g << "last_token_logits_sum: " << lsum << "\n";
    g << "last_token_logits_l2: " << std::sqrt(lsq) << "\n";
    g << "last_token_logits_min: " << lmin << "\n";
    g << "last_token_logits_max: " << lmax << "\n";
    g << "note: full 151936-logit vector reproducible via qwen3_forward (deterministic)\n";
    g.close();
    std::printf("golden written: %s\n", golden_path.c_str());

    bool ok = max_abs < 5e-2;  // full-depth f32 accumulation over 28 layers + vocab head
    std::printf(ok ? "QWEN3_FORWARD_OK\n" : "QWEN3_FORWARD_FAIL\n");
    return ok ? 0 : 1;
}
