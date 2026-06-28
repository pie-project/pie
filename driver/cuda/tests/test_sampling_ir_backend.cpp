// End-to-end test for SamplingIrBackend: feed a real PSIR v2 argmax program
// through decode -> lower (charlie) -> NVRTC JIT (delta) -> driver-API launch,
// driven entirely through echo's IProgramBackend surface. Verifies the token
// matches the host argmax and that the program cache returns the same handle.
//
// Self-contained: pure driver API + the backend (which pulls codegen/reader/
// jit). Failures abort with a message + non-zero exit so CTest catches them.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda.h>

#include "sampling_ir/jit_backend.hpp"
#include "sampling_ir/pie_standard_samplers.h"
#include "sampling_ir/runtime.hpp"
#include "kernels/sample_temp.hpp"

#include "bench_programs.h"

using namespace pie_cuda_driver::sampling_ir;
using pie_cuda_driver::kernels::launch_sample_temp_bf16;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, \
                         #cond);                                          \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

void cu_check(CUresult res, const char* expr, int line) {
    if (res != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(res, &name);
        std::fprintf(stderr, "FATAL: %s:%d: %s -> %s\n", __FILE__, line, expr,
                     name ? name : "?");
        std::exit(2);
    }
}
#define CU(expr) cu_check((expr), #expr, __LINE__)

// Little-endian byte emitters for building a PSIR v2 buffer by hand.
void put_u8(std::vector<std::uint8_t>& b, std::uint8_t v) { b.push_back(v); }
void put_u16(std::vector<std::uint8_t>& b, std::uint16_t v) {
    b.push_back(v & 0xff);
    b.push_back((v >> 8) & 0xff);
}
void put_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xff);
}

// PSIR v2 program: greedy argmax over intrinsic logits Vector{vocab}.
//   inputs: [ Input{id:0, F32 Vector{vocab}, Intrinsic(Logits)} ]
//   slot 0: ops:[ ReduceArgmax(0) ], outputs:[ (value=1, kind=Token) ]
std::vector<std::uint8_t> build_argmax_v2(std::uint32_t vocab) {
    std::vector<std::uint8_t> b;
    b.insert(b.end(), {'P', 'S', 'I', 'R'});
    put_u16(b, 2);   // version = 2
    put_u16(b, 0);   // flags
    put_u32(b, 1);   // n_inputs
    put_u32(b, 1);   // n_slots
    // Input[0]
    put_u32(b, 0);   // id = 0
    put_u8(b, 0);    // dtype = F32
    put_u8(b, 1);    // shape tag = Vector
    put_u32(b, vocab);  // a = len
    put_u32(b, 0);      // b
    put_u8(b, 1);    // binding tag = Intrinsic
    put_u8(b, 0);    // intrinsic = Logits
    // Slot[0]
    put_u32(b, 1);   // n_ops
    put_u8(b, 0x33); // ReduceArgmax
    put_u32(b, 0);   // operand v = id 0
    put_u32(b, 1);   // n_outputs
    put_u32(b, 1);   // output value id = 1 (the argmax result)
    put_u8(b, 0);    // output kind = Token (v2)
    return b;
}

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return static_cast<std::uint16_t>(bits >> 16);  // truncate (matches kernel cast)
}

}  // namespace

int main() {
    CU(cuInit(0));
    CUdevice dev = 0;
    CU(cuDeviceGet(&dev, 0));
    CUcontext ctx = nullptr;
    CU(cuCtxCreate(&ctx, nullptr, 0, dev));

    {
        SamplingIrBackend backend;
        std::fprintf(stderr, "backend arch: %s\n", backend.arch().c_str());

        const std::uint32_t vocab = 256;
        std::vector<std::uint8_t> bytecode = build_argmax_v2(vocab);

        ProgramHandle h = backend.get_or_compile(bytecode);
        if (h == kInvalidProgram) {
            std::fprintf(stderr, "get_or_compile failed: %s\n",
                         backend.last_error().c_str());
        }
        CHECK(h != kInvalidProgram);

        if (h != kInvalidProgram) {
            const ProgramInterface& iface = backend.interface(h);
            CHECK(iface.inputs.size() == 1);
            CHECK(iface.outputs.size() == 1);
            if (iface.inputs.size() == 1) {
                CHECK(iface.inputs[0].cls == BindingClass::Intrinsic);
                CHECK(iface.inputs[0].input_id == 0);
            }
            if (iface.outputs.size() == 1) {
                CHECK(iface.outputs[0].cls == OutputClass::Token);
            }

            // Host bf16 logits with a distinct maximum at a known index.
            const int expect_token = 200;
            std::vector<std::uint16_t> logits(vocab);
            for (std::uint32_t i = 0; i < vocab; ++i) {
                float v = static_cast<float>(i % 50);          // ramps, ties off-peak
                if (i == static_cast<std::uint32_t>(expect_token)) v = 1000.0f;
                logits[i] = f32_to_bf16(v);
            }

            CUdeviceptr d_logits = 0, d_token = 0;
            CU(cuMemAlloc(&d_logits, vocab * sizeof(std::uint16_t)));
            CU(cuMemAlloc(&d_token, sizeof(std::int32_t)));
            CU(cuMemcpyHtoD(d_logits, logits.data(), vocab * sizeof(std::uint16_t)));

            ResolvedInput ri;
            ri.input_id = 0;
            ri.cls = BindingClass::Intrinsic;
            ri.intrinsic = IntrinsicKind::Logits;
            ri.device_ptr = reinterpret_cast<const void*>(d_logits);
            ri.elem_count = vocab;
            ri.present = true;

            void* outs[1] = {reinterpret_cast<void*>(d_token)};

            LaunchArgs args;
            args.inputs = std::span<const ResolvedInput>(&ri, 1);
            args.output_ptrs = std::span<void* const>(outs, 1);
            args.num_rows = 1;
            args.vocab_size = static_cast<int>(vocab);
            args.prng_offset = 0;

            backend.launch(h, args, /*stream=*/nullptr);
            CU(cuCtxSynchronize());

            std::int32_t token = -1;
            CU(cuMemcpyDtoH(&token, d_token, sizeof(token)));
            std::fprintf(stderr, "argmax token = %d (expect %d)\n", token, expect_token);
            CHECK(token == expect_token);

            // Program cache: same bytecode -> same handle, no recompile.
            ProgramHandle h2 = backend.get_or_compile(bytecode);
            CHECK(h2 == h);

            CU(cuMemFree(d_logits));
            CU(cuMemFree(d_token));
        }
    }  // backend destroyed before ctx teardown

    // ── Fallback: a batched backend keeps Gather/Scatter/SortDesc programs
    //    runnable by lowering them M=1 (so "always-batched" production is safe
    //    for the open programmable surface — e.g. mirostat's `gather`). Standard
    //    programs (no such op) stay on the batched fast path. ────────────────
    {
        SamplingIrBackend batched(/*batched_lowering=*/true);
        const int V = static_cast<int>(BENCH_VOCAB);

        // (a) argmax lowers batched → batched fast path.
        ProgramHandle ha = batched.get_or_compile(
            std::span<const std::uint8_t>(BENCH_ARGMAX, sizeof(BENCH_ARGMAX)));
        CHECK(ha != kInvalidProgram);
        CHECK(batched.program_is_batched(ha));  // batched fast path

        // (b) mirostat uses `gather` → batched lowering fails → M=1 fallback.
        //     Without the fallback this would be kInvalidProgram.
        ProgramHandle hm = batched.get_or_compile(
            std::span<const std::uint8_t>(BENCH_MIROSTAT, sizeof(BENCH_MIROSTAT)));
        if (hm == kInvalidProgram) {
            std::fprintf(stderr, "mirostat fallback compile failed: %s\n",
                         batched.last_error().c_str());
        }
        CHECK(hm != kInvalidProgram);             // fallback kept it compilable
        CHECK(!batched.program_is_batched(hm));   // ran the M=1 fallback path

        // Execute the M=1 fallback (num_rows=1, the MVP custom-sampler geometry):
        // bind logits + host inputs (μ, seed) + outputs (token, S); a dominant
        // logit ⇒ mirostat keeps + Gumbel-picks it.
        if (hm != kInvalidProgram) {
            const ProgramInterface& iface = batched.interface(hm);
            const int expect = 123;
            std::vector<std::uint16_t> logits(V, f32_to_bf16(0.0f));
            logits[expect] = f32_to_bf16(100.0f);  // dominant token
            CUdeviceptr d_logits = 0, d_token = 0, d_s = 0;
            CU(cuMemAlloc(&d_logits, static_cast<std::size_t>(V) * 2));
            CU(cuMemAlloc(&d_token, sizeof(std::int32_t)));
            CU(cuMemAlloc(&d_s, sizeof(float)));
            CU(cuMemcpyHtoD(d_logits, logits.data(), static_cast<std::size_t>(V) * 2));

            // Per-row host inputs (num_rows=1): μ large (keep all) + seed.
            std::vector<CUdeviceptr> hostbufs;
            std::vector<ResolvedInput> ri;
            for (const InputDecl& in : iface.inputs) {
                ResolvedInput r;
                r.input_id = in.input_id;
                r.cls = in.cls;
                r.intrinsic = in.intrinsic;
                r.elem_count = in.elem_count;
                r.present = true;
                if (in.cls == BindingClass::Intrinsic) {
                    r.device_ptr = reinterpret_cast<const void*>(d_logits);
                } else {
                    CUdeviceptr b = 0;
                    CU(cuMemAlloc(&b, 8));
                    float val = 100.0f;  // μ; seed reads as bits, any value ok
                    CU(cuMemcpyHtoD(b, &val, sizeof(val)));
                    hostbufs.push_back(b);
                    r.device_ptr = reinterpret_cast<const void*>(b);
                }
                ri.push_back(r);
            }
            // Outputs: Token + Scalar S.
            std::vector<void*> outs;
            for (const DeclaredOutput& od : iface.outputs) {
                outs.push_back(od.cls == OutputClass::Token
                                   ? reinterpret_cast<void*>(d_token)
                                   : reinterpret_cast<void*>(d_s));
            }

            LaunchArgs a;
            a.inputs = std::span<const ResolvedInput>(ri.data(), ri.size());
            a.output_ptrs = std::span<void* const>(outs.data(), outs.size());
            a.num_rows = 1;
            a.vocab_size = V;
            a.prng_offset = 0;
            batched.launch(hm, a, /*stream=*/nullptr);
            CU(cuCtxSynchronize());

            std::int32_t token = -1;
            CU(cuMemcpyDtoH(&token, d_token, sizeof(token)));
            std::fprintf(stderr, "mirostat (M=1 fallback) token = %d (expect %d)\n",
                         token, expect);
            CHECK(token == expect);

            for (CUdeviceptr b : hostbufs) CU(cuMemFree(b));
            CU(cuMemFree(d_logits));
            CU(cuMemFree(d_token));
            CU(cuMemFree(d_s));
        }
    }

    // ── v4 baked argmax via get_or_compile(bytecode, manifest): the binding-free
    //    v4 program + its attach manifest (pie_standard_samplers.h, EDSL-canonical)
    //    must compile through the manifest-aware path and run the batched [N,V]
    //    primitive — per-row argmax over the real qwen3 vocab. This is the v4
    //    ingestion delta owns (decode_v4 + manifest-folded cache key). ──────────
    {
        SamplingIrBackend backend(/*batched_lowering=*/true);
        const std::uint32_t V = 151936u;  // baked vocab

        StandardSamplerProgram prog =
            standard_sampler_program(StandardSamplerKind::Argmax, V);
        CHECK(prog.valid);
        CHECK(prog.bytecode != nullptr && prog.len > 0);
        CHECK(prog.manifest.size() == 1);  // slot[0] = Logits (Intrinsic)

        ProgramHandle h = backend.get_or_compile(
            std::span<const std::uint8_t>(prog.bytecode, prog.len), prog.manifest);
        if (h == kInvalidProgram) {
            std::fprintf(stderr, "v4 argmax compile failed: %s\n",
                         backend.last_error().c_str());
        }
        CHECK(h != kInvalidProgram);
        CHECK(backend.program_is_batched(h));  // batched [N,V] fast path

        // Same bytes + manifest ⇒ cache hit (same handle, manifest folded in key).
        CHECK(h == backend.get_or_compile(
                       std::span<const std::uint8_t>(prog.bytecode, prog.len),
                       prog.manifest));

        // Batched launch: [N, V] logits with a distinct argmax per row, output [N].
        const int N = 4;
        const std::size_t row_elems = static_cast<std::size_t>(V);
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(N) * row_elems,
                                          f32_to_bf16(0.0f));
        std::vector<std::int32_t> expect(N);
        for (int r = 0; r < N; ++r) {
            const int idx = (r + 1) * 1000;  // distinct per row
            logits[static_cast<std::size_t>(r) * row_elems + idx] = f32_to_bf16(50.0f);
            expect[r] = idx;
        }

        CUdeviceptr d_logits = 0, d_tokens = 0;
        CU(cuMemAlloc(&d_logits, logits.size() * 2));
        CU(cuMemAlloc(&d_tokens, static_cast<std::size_t>(N) * sizeof(std::int32_t)));
        CU(cuMemcpyHtoD(d_logits, logits.data(), logits.size() * 2));

        const ProgramInterface& iface = backend.interface(h);
        std::vector<ResolvedInput> ri;
        for (const InputDecl& in : iface.inputs) {
            ResolvedInput r;
            r.input_id = in.input_id;
            r.cls = in.cls;
            r.intrinsic = in.intrinsic;
            r.elem_count = in.elem_count;
            r.present = true;
            r.device_ptr = reinterpret_cast<const void*>(d_logits);  // [N,V] base
            ri.push_back(r);
        }
        std::vector<void*> outs{reinterpret_cast<void*>(d_tokens)};  // [N] token base

        LaunchArgs a;
        a.inputs = std::span<const ResolvedInput>(ri.data(), ri.size());
        a.output_ptrs = std::span<void* const>(outs.data(), outs.size());
        a.num_rows = N;
        a.vocab_size = static_cast<int>(V);
        a.prng_offset = 0;
        backend.launch(h, a, /*stream=*/nullptr);
        CU(cuCtxSynchronize());

        std::vector<std::int32_t> tokens(N, -1);
        CU(cuMemcpyDtoH(tokens.data(), d_tokens,
                        static_cast<std::size_t>(N) * sizeof(std::int32_t)));
        for (int r = 0; r < N; ++r) {
            std::fprintf(stderr, "v4 argmax [N,V] row %d token = %d (expect %d)\n",
                         r, tokens[r], expect[r]);
            CHECK(tokens[r] == expect[r]);
        }

        CU(cuMemFree(d_logits));
        CU(cuMemFree(d_tokens));
    }

    // ── v4 baked temp/min-p via get_or_compile(bytecode, manifest) + row_seeds:
    //    the PRODUCTION RNG path (#4 headline parity). Drive the canonical baked
    //    program (pie_standard_samplers.h, EDSL-authored) through the backend with
    //    the ambient seed bound via LaunchArgs.row_seeds, and assert token-exact
    //    vs launch_sample_temp_bf16 over the same per-row T/min_p/seed. stream:0 ⇒
    //    seed_eff_stream(S,0)==seed_eff(S) (the removal-gate axis). Validates
    //    delta's row_seeds bind end-to-end through the real backend. ─────────────
    {
        SamplingIrBackend backend(/*batched_lowering=*/true);
        const std::uint32_t V = 151936u;
        const int N = 16;

        // [N,V] varied bf16 logits (distinct per-row pattern).
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(N) * V);
        for (int r = 0; r < N; ++r)
            for (std::uint32_t j = 0; j < V; ++j)
                logits[static_cast<std::size_t>(r) * V + j] =
                    f32_to_bf16(static_cast<float>((j * 131u + r * 7u) % 997u) * 0.01f);
        std::vector<std::uint32_t> seeds(N);
        for (int r = 0; r < N; ++r) seeds[r] = 1234u + static_cast<std::uint32_t>(r);

        CUdeviceptr d_logits = 0, d_temps = 0, d_minps = 0, d_seeds = 0, d_ir = 0, d_hw = 0;
        CU(cuMemAlloc(&d_logits, logits.size() * 2));
        CU(cuMemAlloc(&d_temps, N * sizeof(float)));
        CU(cuMemAlloc(&d_minps, N * sizeof(float)));
        CU(cuMemAlloc(&d_seeds, N * sizeof(std::uint32_t)));
        CU(cuMemAlloc(&d_ir, N * sizeof(std::int32_t)));
        CU(cuMemAlloc(&d_hw, N * sizeof(std::int32_t)));
        CU(cuMemcpyHtoD(d_logits, logits.data(), logits.size() * 2));
        CU(cuMemcpyHtoD(d_seeds, seeds.data(), N * sizeof(std::uint32_t)));

        auto run_kind = [&](StandardSamplerKind kind, const char* nm, float minp_val) {
            std::vector<float> temps(N, 1.0f), minps(N, minp_val);
            CU(cuMemcpyHtoD(d_temps, temps.data(), N * sizeof(float)));
            CU(cuMemcpyHtoD(d_minps, minps.data(), N * sizeof(float)));

            StandardSamplerProgram prog = standard_sampler_program(kind, V);
            CHECK(prog.valid);
            ProgramHandle h = backend.get_or_compile(
                std::span<const std::uint8_t>(prog.bytecode, prog.len), prog.manifest);
            if (h == kInvalidProgram) {
                std::fprintf(stderr, "v4 %s compile failed: %s\n", nm,
                             backend.last_error().c_str());
            }
            CHECK(h != kInvalidProgram);
            CHECK(backend.program_is_batched(h));

            // Bind: Intrinsic logits + HostSubmit params (T key0 [+ min_p key1]);
            // the ambient per-row seed rides LaunchArgs.row_seeds (not an Op::Input).
            const ProgramInterface& iface = backend.interface(h);
            std::vector<ResolvedInput> ri;
            for (const InputDecl& in : iface.inputs) {
                ResolvedInput r;
                r.input_id = in.input_id;
                r.cls = in.cls;
                r.intrinsic = in.intrinsic;
                r.elem_count = in.elem_count;
                r.present = true;
                if (in.cls == BindingClass::Intrinsic)
                    r.device_ptr = reinterpret_cast<const void*>(d_logits);
                else if (in.host_key == 0)
                    r.device_ptr = reinterpret_cast<const void*>(d_temps);   // T
                else
                    r.device_ptr = reinterpret_cast<const void*>(d_minps);   // min_p
                ri.push_back(r);
            }
            void* outp = reinterpret_cast<void*>(d_ir);
            LaunchArgs a;
            a.inputs = std::span<const ResolvedInput>(ri.data(), ri.size());
            a.output_ptrs = std::span<void* const>(&outp, 1);
            a.num_rows = N;
            a.vocab_size = static_cast<int>(V);
            a.prng_offset = 0;
            a.row_seeds = reinterpret_cast<const void*>(d_seeds);  // ambient per-row S
            backend.launch(h, a, /*stream=*/nullptr);
            CU(cuCtxSynchronize());

            // Reference: the dedicated sample_temp kernel over the same params.
            launch_sample_temp_bf16(reinterpret_cast<const void*>(d_logits),
                                    reinterpret_cast<float*>(d_temps),
                                    reinterpret_cast<float*>(d_minps),
                                    reinterpret_cast<std::uint32_t*>(d_seeds),
                                    reinterpret_cast<std::int32_t*>(d_hw), N,
                                    static_cast<int>(V), /*stream=*/nullptr);
            CU(cuCtxSynchronize());

            std::vector<std::int32_t> ir(N), hw(N);
            CU(cuMemcpyDtoH(ir.data(), d_ir, N * sizeof(std::int32_t)));
            CU(cuMemcpyDtoH(hw.data(), d_hw, N * sizeof(std::int32_t)));
            int mism = 0;
            for (int r = 0; r < N; ++r)
                if (ir[r] != hw[r]) mism++;
            std::fprintf(stderr,
                         "v4 %s [N=%d,V=%u] row_seeds bind: ir-vs-sample_temp mism=%d\n",
                         nm, N, V, mism);
            CHECK(mism == 0);
        };

        run_kind(StandardSamplerKind::Temperature, "temp", 0.0f);
        run_kind(StandardSamplerKind::MinP, "min_p", 0.1f);

        CU(cuMemFree(d_logits));
        CU(cuMemFree(d_temps));
        CU(cuMemFree(d_minps));
        CU(cuMemFree(d_seeds));
        CU(cuMemFree(d_ir));
        CU(cuMemFree(d_hw));
    }

    // ── Custom-batch (num_rows>1) replay: an M=1-fallback program (mirostat's
    //    Gather) over N rows must run the single-row program PER ROW, sliced — each
    //    row reads its own logits/params and writes its own output. Validates
    //    delta's custom-batch slicing (the #10/#11 batched-custom prerequisite):
    //    instead of throwing at num_rows>1, the backend replays N sliced launches,
    //    token-identical to N separate num_rows=1 fires. ────────────────────────
    {
        SamplingIrBackend backend(/*batched_lowering=*/true);
        const int V = static_cast<int>(BENCH_VOCAB);
        const int N = 4;

        ProgramHandle h = backend.get_or_compile(
            std::span<const std::uint8_t>(BENCH_MIROSTAT, sizeof(BENCH_MIROSTAT)));
        CHECK(h != kInvalidProgram);
        CHECK(!backend.program_is_batched(h));  // gather → M=1 fallback

        const ProgramInterface& iface = backend.interface(h);

        // [N,V] logits: row r dominant (100.0) at a distinct idx_r = (r+1)*100.
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(N) * V, f32_to_bf16(0.0f));
        std::vector<std::int32_t> expect(N);
        for (int r = 0; r < N; ++r) {
            const int idx = (r + 1) * 100;
            logits[static_cast<std::size_t>(r) * V + idx] = f32_to_bf16(100.0f);
            expect[r] = idx;
        }
        CUdeviceptr d_logits = 0, d_tok = 0, d_s = 0;
        CU(cuMemAlloc(&d_logits, static_cast<std::size_t>(N) * V * 2));
        CU(cuMemAlloc(&d_tok, N * sizeof(std::int32_t)));
        CU(cuMemAlloc(&d_s, N * sizeof(float)));
        CU(cuMemcpyHtoD(d_logits, logits.data(), static_cast<std::size_t>(N) * V * 2));

        // Per-row host inputs ([N]): μ large (keep all) + seed; sliced per row.
        std::vector<CUdeviceptr> hostbufs;
        std::vector<ResolvedInput> ri;
        for (const InputDecl& in : iface.inputs) {
            ResolvedInput r;
            r.input_id = in.input_id;
            r.cls = in.cls;
            r.intrinsic = in.intrinsic;
            r.elem_count = in.elem_count;
            r.present = true;
            if (in.cls == BindingClass::Intrinsic) {
                r.device_ptr = reinterpret_cast<const void*>(d_logits);
            } else {
                CUdeviceptr b = 0;
                CU(cuMemAlloc(&b, N * sizeof(float)));
                std::vector<float> vals(N, 100.0f);  // μ large; seed reads as bits
                CU(cuMemcpyHtoD(b, vals.data(), N * sizeof(float)));
                hostbufs.push_back(b);
                r.device_ptr = reinterpret_cast<const void*>(b);
            }
            ri.push_back(r);
        }
        std::vector<void*> outs;
        for (const DeclaredOutput& od : iface.outputs) {
            outs.push_back(od.cls == OutputClass::Token
                               ? reinterpret_cast<void*>(d_tok)
                               : reinterpret_cast<void*>(d_s));
        }

        LaunchArgs a;
        a.inputs = std::span<const ResolvedInput>(ri.data(), ri.size());
        a.output_ptrs = std::span<void* const>(outs.data(), outs.size());
        a.num_rows = N;  // > 1 → custom-batch replay (no longer throws)
        a.vocab_size = V;
        a.prng_offset = 0;
        backend.launch(h, a, /*stream=*/nullptr);
        CU(cuCtxSynchronize());

        std::vector<std::int32_t> tok(N, -1);
        CU(cuMemcpyDtoH(tok.data(), d_tok, N * sizeof(std::int32_t)));
        for (int r = 0; r < N; ++r) {
            std::fprintf(stderr, "custom-batch mirostat row %d token = %d (expect %d)\n",
                         r, tok[r], expect[r]);
            CHECK(tok[r] == expect[r]);
        }

        for (CUdeviceptr b : hostbufs) CU(cuMemFree(b));
        CU(cuMemFree(d_logits));
        CU(cuMemFree(d_tok));
        CU(cuMemFree(d_s));
    }

    CU(cuCtxDestroy(ctx));

    if (g_failures == 0) {
        std::fprintf(stderr, "sampling_ir_backend: OK\n");
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_backend: %d failure(s)\n", g_failures);
    return 1;
}
