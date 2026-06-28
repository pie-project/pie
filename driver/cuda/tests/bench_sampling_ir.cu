// WS6 benchmark harness (lane L3 / delta): batched (M>1) IR-JIT samplers vs the
// hardwired path. Measures, per sampler × batch:
//
//   * cold      — first get_or_compile: NVRTC compile + module load (ms, one-shot)
//   * warm      — cached batched program, ONE grid=num_rows launch (us/fire)
//   * graph     — that single launch captured into a CUDA graph + replayed
//   * hardwired — sample_temp (argmax/temp/min-p) or FlashInfer (top-k/top-p),
//                 one batched launch over [B, vocab]
//
// With charlie's M>1 batched codegen the IR path is now a SINGLE grid=num_rows
// launch per fire (every value carries a per-row batch dim; my JIT supplies the
// dynamic grid + capacity-sized per-row buffers) + the 4-pass radix top-k — so
// it scales flat in B like the hardwired path instead of the B per-row launches
// of the MVP. This is the IR-vs-FlashInfer / IR-vs-sample_temp head-to-head the
// Oracle asked for.
//
// Host-input scalars (temp/p/seed) bind to small per-row [B] device buffers with
// perf-neutral values — the kernels do identical work regardless (full-vocab
// scans + fixed pivot-iteration counts); numeric parity is hotel's lane.
//
// Output: CSV to stdout (and argv[1] if given) for the programmable-sampler-bench
// wiki.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "sampling_ir/jit_backend.hpp"
#include "sampling_ir/pie_standard_samplers.h"
#include "sampling_ir/runtime.hpp"
#include "kernels/sample_temp.hpp"
#include "kernels/sample_flashinfer.hpp"
#include "kernels/argmax.hpp"

#include "bench_programs.h"

using namespace pie_cuda_driver::sampling_ir;
using pie_cuda_driver::kernels::launch_sample_temp_bf16;
using pie_cuda_driver::kernels::launch_sample_topk_topp_bf16;
using pie_cuda_driver::kernels::launch_argmax_bf16;

namespace {

void cu_check(CUresult res, const char* expr, int line) {
    if (res != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(res, &name);
        std::fprintf(stderr, "FATAL cu: %s:%d: %s -> %s\n", __FILE__, line, expr,
                     name ? name : "?");
        std::exit(2);
    }
}
void rt_check(cudaError_t e, const char* expr, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL rt: %s:%d: %s -> %s\n", __FILE__, line, expr,
                     cudaGetErrorString(e));
        std::exit(2);
    }
}
#define CU(expr) cu_check((expr), #expr, __LINE__)
#define RT(expr) rt_check((expr), #expr, __LINE__)

const int kBatches[] = {1, 8, 32, 128, 256};
constexpr int kWarmupIters = 10;
constexpr int kTimedIters = 50;

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    return static_cast<std::uint16_t>(b >> 16);
}

double ms_event(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.f;
    RT(cudaEventElapsedTime(&ms, a, b));
    return static_cast<double>(ms);
}

enum class HwKind { SampleTemp, FlashInfer, Argmax, None };

struct Sampler {
    const char* name;
    const unsigned char* bytecode;
    std::size_t len;
    HwKind hw;
    float hw_temp;
    float hw_minp;
    int hw_topk;     // FlashInfer top-k (0 = disabled)
    float hw_topp;   // FlashInfer top-p (1.0 = disabled)
    bool baked = false;                                          // v4 canonical (pie_standard_samplers.h)
    StandardSamplerKind std_kind = StandardSamplerKind::Argmax;  // baked: which program
};

}  // namespace

int main(int argc, char** argv) {
    CU(cuInit(0));
    CUdevice dev = 0;
    CU(cuDeviceGet(&dev, 0));
    CUcontext ctx = nullptr;
    CU(cuCtxCreate(&ctx, nullptr, 0, dev));

    char gpu_name[256] = {0};
    CU(cuDeviceGetName(gpu_name, sizeof(gpu_name), dev));

    const int vocab = static_cast<int>(BENCH_VOCAB);
    const int maxB = kBatches[sizeof(kBatches) / sizeof(kBatches[0]) - 1];

    std::vector<Sampler> samplers = {
        {"argmax", BENCH_ARGMAX, sizeof(BENCH_ARGMAX), HwKind::Argmax,     -1.0f, 0.0f, 0, 1.0f},
        {"temp",   BENCH_TEMP,   sizeof(BENCH_TEMP),   HwKind::SampleTemp,  1.0f, 0.0f, 0, 1.0f},
        {"min_p",  BENCH_MINP,   sizeof(BENCH_MINP),   HwKind::SampleTemp,  1.0f, 0.1f, 0, 1.0f},
        // v4 canonical (production bytecode + ambient row_seeds) — bench≡production.
        {"temp_v4",  nullptr, 0, HwKind::SampleTemp, 1.0f, 0.0f, 0, 1.0f, true, StandardSamplerKind::Temperature},
        {"min_p_v4", nullptr, 0, HwKind::SampleTemp, 1.0f, 0.1f, 0, 1.0f, true, StandardSamplerKind::MinP},
        {"top_k",  BENCH_TOPK,   sizeof(BENCH_TOPK),   HwKind::FlashInfer,  1.0f, 0.0f, 50, 1.0f},
        {"top_p",  BENCH_TOPP,   sizeof(BENCH_TOPP),   HwKind::FlashInfer,  1.0f, 0.0f, 0, 0.9f},
    };

    // ── Shared device buffers (sized for maxB) ───────────────────────────────
    // bf16 logits [maxB, vocab] with a distinct max per row.
    std::vector<std::uint16_t> h_logits(static_cast<std::size_t>(maxB) * vocab);
    for (int r = 0; r < maxB; ++r)
        for (int j = 0; j < vocab; ++j)
            h_logits[static_cast<std::size_t>(r) * vocab + j] =
                f32_to_bf16(static_cast<float>((j * 131 + r * 7) % 997) * 0.01f);

    CUdeviceptr d_logits = 0, d_out = 0;
    CU(cuMemAlloc(&d_logits, h_logits.size() * sizeof(std::uint16_t)));
    CU(cuMemAlloc(&d_out, static_cast<std::size_t>(maxB) * sizeof(std::int32_t)));
    CU(cuMemcpyHtoD(d_logits, h_logits.data(),
                    h_logits.size() * sizeof(std::uint16_t)));

    // sample_temp params (per row).
    float* d_temps = nullptr;
    float* d_minps = nullptr;
    std::uint32_t* d_seeds32 = nullptr;
    RT(cudaMalloc(&d_temps, maxB * sizeof(float)));
    RT(cudaMalloc(&d_minps, maxB * sizeof(float)));
    RT(cudaMalloc(&d_seeds32, maxB * sizeof(std::uint32_t)));
    // Ambient per-row seeds for the v4 baked RNG samplers (LaunchArgs.row_seeds).
    std::uint32_t* d_rowseeds = nullptr;
    RT(cudaMalloc(&d_rowseeds, maxB * sizeof(std::uint32_t)));
    {
        std::vector<std::uint32_t> rs(maxB);
        for (int r = 0; r < maxB; ++r) rs[r] = 1234u + static_cast<std::uint32_t>(r);
        RT(cudaMemcpy(d_rowseeds, rs.data(), maxB * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice));
    }

    // FlashInfer params + scratch.
    float* d_probs = nullptr;          // [maxB, vocab] fp32 softmax scratch
    bool* d_valid = nullptr;           // [maxB]
    std::int32_t* d_rowidx = nullptr;  // [maxB] = 0..B-1
    std::int32_t* d_topk = nullptr;    // [maxB]
    float* d_topp = nullptr;           // [maxB]
    std::uint64_t* d_seed64 = nullptr; // [maxB]
    RT(cudaMalloc(&d_probs, static_cast<std::size_t>(maxB) * vocab * sizeof(float)));
    RT(cudaMalloc(&d_valid, maxB * sizeof(bool)));
    RT(cudaMalloc(&d_rowidx, maxB * sizeof(std::int32_t)));
    RT(cudaMalloc(&d_topk, maxB * sizeof(std::int32_t)));
    RT(cudaMalloc(&d_topp, maxB * sizeof(float)));
    RT(cudaMalloc(&d_seed64, maxB * sizeof(std::uint64_t)));

    cudaEvent_t ev0, ev1;
    RT(cudaEventCreate(&ev0));
    RT(cudaEventCreate(&ev1));
    cudaStream_t stream = nullptr;
    RT(cudaStreamCreate(&stream));

    std::string csv =
        "gpu,vocab,sampler,batch,cold_compile_ms,ir_warm_us_per_fire,"
        "ir_graph_us_per_fire,hw_impl,hardwired_us_per_fire,ir_graph_vs_hw\n";

    // Batched (M>1) IR backend: one grid=num_rows launch per fire.
    SamplingIrBackend backend(/*batched_lowering=*/true);
    std::fprintf(stderr, "GPU=%s arch=%s vocab=%d (batched M>1 IR + radix top-k)\n",
                 gpu_name, backend.arch().c_str(), vocab);

    for (const Sampler& s : samplers) {
        // ── cold compile (one-shot) ──────────────────────────────────────
        cudaEvent_t c0, c1;
        RT(cudaEventCreate(&c0));
        RT(cudaEventCreate(&c1));
        RT(cudaEventRecord(c0, stream));
        // Program source: v3 self-binding bytecode, or v4 baked (binding-free)
        // canonical program + its attach manifest (pie_standard_samplers.h).
        const std::uint8_t* bc = reinterpret_cast<const std::uint8_t*>(s.bytecode);
        std::size_t blen = s.len;
        ProgramManifest manifest;
        if (s.baked) {
            StandardSamplerProgram bp = standard_sampler_program(s.std_kind, vocab);
            bc = bp.bytecode;
            blen = bp.len;
            manifest = bp.manifest;
        }
        ProgramHandle h = backend.get_or_compile(
            std::span<const std::uint8_t>(bc, blen), manifest);
        RT(cudaEventRecord(c1, stream));
        RT(cudaEventSynchronize(c1));
        const double cold_ms = ms_event(c0, c1);
        RT(cudaEventDestroy(c0));
        RT(cudaEventDestroy(c1));
        if (h == kInvalidProgram) {
            std::fprintf(stderr, "  %s: compile FAILED: %s\n", s.name,
                         backend.last_error().c_str());
            continue;
        }

        const ProgramInterface& iface = backend.interface(h);

        // One per-row [maxB] device buffer per host input (value 1.0f bits).
        std::vector<CUdeviceptr> hostbufs;
        for (const InputDecl& in : iface.inputs) {
            if (in.cls == BindingClass::Intrinsic) continue;
            CUdeviceptr b = 0;
            CU(cuMemAlloc(&b, static_cast<std::size_t>(maxB) * 4));
            // Baked min_p (host_key 1) wants a realistic 0.1 threshold; T and all
            // other host params use 1.0.
            const float val = (s.baked && in.host_key == 1) ? 0.1f : 1.0f;
            std::vector<float> vals(maxB, val);
            CU(cuMemcpyHtoD(b, vals.data(), static_cast<std::size_t>(maxB) * 4));
            hostbufs.push_back(b);
        }

        // One batched launch: logits base + host bases bound, num_rows = B.
        auto ir_launch = [&](int B, cudaStream_t st) {
            std::vector<ResolvedInput> ri;
            std::size_t hb = 0;
            for (const InputDecl& in : iface.inputs) {
                ResolvedInput r;
                r.input_id = in.input_id;
                r.cls = in.cls;
                r.intrinsic = in.intrinsic;
                r.elem_count = in.elem_count;
                r.present = true;
                r.device_ptr = (in.cls == BindingClass::Intrinsic)
                                   ? reinterpret_cast<const void*>(d_logits)
                                   : reinterpret_cast<const void*>(hostbufs[hb++]);
                ri.push_back(r);
            }
            void* outp = reinterpret_cast<void*>(d_out);
            LaunchArgs a;
            a.inputs = std::span<const ResolvedInput>(ri.data(), ri.size());
            a.output_ptrs = std::span<void* const>(&outp, 1);
            a.num_rows = B;
            a.vocab_size = vocab;
            a.prng_offset = 0;
            // v4 baked RNG samplers read the ambient per-row seed via row_seeds.
            a.row_seeds = s.baked ? reinterpret_cast<const void*>(d_rowseeds) : nullptr;
            backend.launch(h, a, st);
        };

        for (int B : kBatches) {
            // ── IR warm: single batched launch, cached program ───────────
            for (int i = 0; i < kWarmupIters; ++i) ir_launch(B, stream);
            RT(cudaStreamSynchronize(stream));
            RT(cudaEventRecord(ev0, stream));
            for (int i = 0; i < kTimedIters; ++i) ir_launch(B, stream);
            RT(cudaEventRecord(ev1, stream));
            RT(cudaEventSynchronize(ev1));
            const double warm_us = ms_event(ev0, ev1) * 1000.0 / kTimedIters;

            // ── IR graph: capture the single launch, replay ──────────────
            CUgraph graph = nullptr;
            CUgraphExec gexec = nullptr;
            CU(cuStreamBeginCapture(reinterpret_cast<CUstream>(stream),
                                    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
            ir_launch(B, stream);
            CU(cuStreamEndCapture(reinterpret_cast<CUstream>(stream), &graph));
            CU(cuGraphInstantiate(&gexec, graph, 0));
            for (int i = 0; i < kWarmupIters; ++i)
                CU(cuGraphLaunch(gexec, reinterpret_cast<CUstream>(stream)));
            RT(cudaStreamSynchronize(stream));
            RT(cudaEventRecord(ev0, stream));
            for (int i = 0; i < kTimedIters; ++i)
                CU(cuGraphLaunch(gexec, reinterpret_cast<CUstream>(stream)));
            RT(cudaEventRecord(ev1, stream));
            RT(cudaEventSynchronize(ev1));
            const double graph_us = ms_event(ev0, ev1) * 1000.0 / kTimedIters;
            CU(cuGraphExecDestroy(gexec));
            CU(cuGraphDestroy(graph));

            // ── hardwired baseline ───────────────────────────────────────
            double hw_us = 0.0;
            const char* hw_impl = "none";
            if (s.hw == HwKind::SampleTemp) {
                hw_impl = "sample_temp";
                std::vector<float> ht(B, s.hw_temp), hm(B, s.hw_minp);
                std::vector<std::uint32_t> hs(B);
                for (int r = 0; r < B; ++r) hs[r] = 1234u + r;
                RT(cudaMemcpy(d_temps, ht.data(), B * sizeof(float), cudaMemcpyHostToDevice));
                RT(cudaMemcpy(d_minps, hm.data(), B * sizeof(float), cudaMemcpyHostToDevice));
                RT(cudaMemcpy(d_seeds32, hs.data(), B * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
                auto run = [&] {
                    launch_sample_temp_bf16(
                        reinterpret_cast<const void*>(d_logits), d_temps, d_minps,
                        d_seeds32, reinterpret_cast<std::int32_t*>(d_out), B, vocab, stream);
                };
                for (int i = 0; i < kWarmupIters; ++i) run();
                RT(cudaStreamSynchronize(stream));
                RT(cudaEventRecord(ev0, stream));
                for (int i = 0; i < kTimedIters; ++i) run();
                RT(cudaEventRecord(ev1, stream));
                RT(cudaEventSynchronize(ev1));
                hw_us = ms_event(ev0, ev1) * 1000.0 / kTimedIters;
            } else if (s.hw == HwKind::Argmax) {
                // Dedicated greedy kernel — the actual de-hardwiring target for
                // argmax (the §2f baseline was sample_temp(T<0); this gates the IR
                // argmax vs the real launch_argmax_bf16 it replaces).
                hw_impl = "argmax_bf16";
                auto run = [&] {
                    launch_argmax_bf16(
                        reinterpret_cast<const void*>(d_logits),
                        reinterpret_cast<std::int32_t*>(d_out), B, vocab, stream);
                };
                for (int i = 0; i < kWarmupIters; ++i) run();
                RT(cudaStreamSynchronize(stream));
                RT(cudaEventRecord(ev0, stream));
                for (int i = 0; i < kTimedIters; ++i) run();
                RT(cudaEventRecord(ev1, stream));
                RT(cudaEventSynchronize(ev1));
                hw_us = ms_event(ev0, ev1) * 1000.0 / kTimedIters;
            } else if (s.hw == HwKind::FlashInfer) {
                hw_impl = "flashinfer";
                std::vector<float> ht(B, s.hw_temp), htp(B, s.hw_topp);
                std::vector<std::int32_t> hk(B, s.hw_topk), hidx(B);
                std::vector<std::uint64_t> hseed(B);
                for (int r = 0; r < B; ++r) { hidx[r] = r; hseed[r] = 1234u + r; }
                RT(cudaMemcpy(d_temps, ht.data(), B * sizeof(float), cudaMemcpyHostToDevice));
                RT(cudaMemcpy(d_topp, htp.data(), B * sizeof(float), cudaMemcpyHostToDevice));
                RT(cudaMemcpy(d_topk, hk.data(), B * sizeof(std::int32_t), cudaMemcpyHostToDevice));
                RT(cudaMemcpy(d_rowidx, hidx.data(), B * sizeof(std::int32_t), cudaMemcpyHostToDevice));
                RT(cudaMemcpy(d_seed64, hseed.data(), B * sizeof(std::uint64_t), cudaMemcpyHostToDevice));
                auto run = [&] {
                    launch_sample_topk_topp_bf16(
                        reinterpret_cast<const void*>(d_logits), d_probs, d_temps,
                        d_rowidx, d_topk, d_topp, d_seed64, d_valid,
                        reinterpret_cast<std::int32_t*>(d_out), B, B, vocab,
                        /*prng_offset=*/0, stream);
                };
                for (int i = 0; i < kWarmupIters; ++i) run();
                RT(cudaStreamSynchronize(stream));
                RT(cudaEventRecord(ev0, stream));
                for (int i = 0; i < kTimedIters; ++i) run();
                RT(cudaEventRecord(ev1, stream));
                RT(cudaEventSynchronize(ev1));
                hw_us = ms_event(ev0, ev1) * 1000.0 / kTimedIters;
            }

            char line[640];
            auto ratio = [](double a, double b) { return b > 0 ? a / b : 0.0; };
            std::snprintf(line, sizeof(line),
                          "%s,%d,%s,%d,%.3f,%.3f,%.3f,%s,%.3f,%.2f\n",
                          gpu_name, vocab, s.name, B,
                          (B == kBatches[0]) ? cold_ms : 0.0, warm_us, graph_us,
                          hw_impl, hw_us, ratio(graph_us, hw_us));
            csv += line;
            std::fprintf(stderr,
                         "  %-7s B=%-4d cold=%6.2fms ir_warm=%8.2fus ir_graph=%8.2fus "
                         "%s=%8.2fus  ir/hw=%.2fx\n",
                         s.name, B, (B == kBatches[0]) ? cold_ms : 0.0, warm_us,
                         graph_us, hw_impl, hw_us, ratio(graph_us, hw_us));
        }

        for (CUdeviceptr b : hostbufs) CU(cuMemFree(b));
    }

    std::fputs(csv.c_str(), stdout);
    if (argc > 1) {
        FILE* f = std::fopen(argv[1], "w");
        if (f) {
            std::fputs(csv.c_str(), f);
            std::fclose(f);
            std::fprintf(stderr, "wrote %s\n", argv[1]);
        }
    }

    RT(cudaFree(d_temps));
    RT(cudaFree(d_minps));
    RT(cudaFree(d_seeds32));
    RT(cudaFree(d_probs));
    RT(cudaFree(d_valid));
    RT(cudaFree(d_rowidx));
    RT(cudaFree(d_topk));
    RT(cudaFree(d_topp));
    RT(cudaFree(d_seed64));
    CU(cuMemFree(d_logits));
    CU(cuMemFree(d_out));
    RT(cudaEventDestroy(ev0));
    RT(cudaEventDestroy(ev1));
    RT(cudaStreamDestroy(stream));
    CU(cuCtxDestroy(ctx));
    return 0;
}
