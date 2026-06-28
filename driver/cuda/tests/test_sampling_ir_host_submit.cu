// WS1a unit test (lane L4 / echo): per-fire submit-bound host-input binding.
//
// Drives `SamplingIrRuntime::try_run` with a program whose ONLY input is a
// `host(submit-bound)` F32 vector — `argmax(host_vec) -> Token`. This exercises
// the new path end-to-end: the runtime stages the per-fire host bytes into its
// device blob, binds them to the program's HostSubmit buffer, the JIT'd argmax
// runs, and the token lands in `pi.sampled`. Firing twice with different vectors
// proves the value refreshes each fire (no recompile, no stale binding) — the
// mechanism mirostat's µ and a grammar mask ride on.
//
// Self-contained: runtime + backend (codegen/reader/jit) + a minimal
// PersistentInputs. No workspace/logits needed (no Intrinsic input → ws unused).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "executor/persistent_inputs.hpp"
#include "sampling_ir/jit_backend.hpp"
#include "sampling_ir/runtime.hpp"

using namespace pie_cuda_driver;
using namespace pie_cuda_driver::sampling_ir;

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

void put_u8(std::vector<std::uint8_t>& b, std::uint8_t v) { b.push_back(v); }
void put_u16(std::vector<std::uint8_t>& b, std::uint16_t v) {
    b.push_back(v & 0xff);
    b.push_back((v >> 8) & 0xff);
}
void put_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xff);
}

// PSIR v2: argmax over a host(submit-bound) F32 Vector{len} → Token.
//   inputs: [ Input{id:0, F32 Vector{len}, Host{key=7, SubmitBound}} ]
//   slot 0: ops:[ ReduceArgmax(0) ], outputs:[ (value=1, kind=Token) ]
// `avail`: 0 = SubmitBound, 1 = LateBound.
std::vector<std::uint8_t> build_host_argmax_v2(std::uint32_t len,
                                               std::uint32_t key,
                                               std::uint8_t avail = 0) {
    std::vector<std::uint8_t> b;
    b.insert(b.end(), {'P', 'S', 'I', 'R'});
    put_u16(b, 2);       // version
    put_u16(b, 0);       // flags
    put_u32(b, 1);       // n_inputs
    put_u32(b, 1);       // n_slots
    // Input[0]
    put_u32(b, 0);       // id
    put_u8(b, 0);        // dtype F32
    put_u8(b, 1);        // shape Vector
    put_u32(b, len);     // a = len
    put_u32(b, 0);       // b
    put_u8(b, 2);        // binding Host
    put_u32(b, key);     // host key
    put_u8(b, avail);    // avail (0=SubmitBound, 1=LateBound)
    // Slot[0]
    put_u32(b, 1);       // n_ops
    put_u8(b, 0x33);     // ReduceArgmax
    put_u32(b, 0);       // operand v = id 0
    put_u32(b, 1);       // n_outputs
    put_u32(b, 1);       // output value id = 1
    put_u8(b, 0);        // kind Token
    return b;
}

// Run one fire with host vector `vals` bound under `key`; return the token.
std::int32_t fire(SamplingIrRuntime& rt, PersistentInputs& pi,
                  const std::vector<std::uint8_t>& bytecode,
                  const std::vector<float>& vals, std::uint32_t key) {
    const auto* raw = reinterpret_cast<const std::uint8_t*>(vals.data());
    SubmitInput si;
    si.key = key;
    si.data = raw;
    si.len_bytes = vals.size() * sizeof(float);

    FireContext ctx;
    ctx.program_bytecode = {bytecode.data(), bytecode.size()};
    ctx.submit_inputs = std::span<const SubmitInput>(&si, 1);
    ctx.logits = nullptr;  // no Intrinsic input → logits never read
    ctx.pi = &pi;
    ctx.vocab_size = static_cast<int>(vals.size());
    ctx.sample_row = 0;
    ctx.prng_offset = 0;
    ctx.stream = nullptr;

    const RunStatus st = rt.try_run(ctx);
    CHECK(st == RunStatus::Handled);
    cudaDeviceSynchronize();
    std::int32_t token = -1;
    cudaMemcpy(&token, pi.sampled.data(), sizeof(std::int32_t),
               cudaMemcpyDeviceToHost);
    return token;
}

std::int32_t host_argmax(const std::vector<float>& v) {
    std::int32_t best = 0;
    for (std::int32_t i = 1; i < static_cast<std::int32_t>(v.size()); ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return static_cast<std::uint16_t>(bits >> 16);
}

// PSIR v2: argmax(logits + mask) → Token, where logits is the bf16
// Intrinsic(Logits) and mask is a host(submit-bound) F32 vector. This is the
// grammar-masking / additive-bias shape the real mirostat & grammar inferlets
// use (logits intrinsic + submit-bound mask in ONE program).
//   inputs: [ Input{0, F32 Vector{vocab}, Intrinsic(Logits)},
//             Input{1, F32 Vector{vocab}, Host{key, SubmitBound}} ]
//   slot 0: ops:[ Add(0,1), ReduceArgmax(2) ], outputs:[ (value=3, Token) ]
std::vector<std::uint8_t> build_logits_plus_mask_v2(std::uint32_t vocab,
                                                    std::uint32_t key) {
    std::vector<std::uint8_t> b;
    b.insert(b.end(), {'P', 'S', 'I', 'R'});
    put_u16(b, 2);
    put_u16(b, 0);
    put_u32(b, 2);       // n_inputs
    put_u32(b, 1);       // n_slots
    // Input[0] — Intrinsic logits
    put_u32(b, 0);
    put_u8(b, 0);        // F32
    put_u8(b, 1);        // Vector
    put_u32(b, vocab);
    put_u32(b, 0);
    put_u8(b, 1);        // Intrinsic
    put_u8(b, 0);        // Logits
    // Input[1] — Host submit-bound mask
    put_u32(b, 1);
    put_u8(b, 0);        // F32
    put_u8(b, 1);        // Vector
    put_u32(b, vocab);
    put_u32(b, 0);
    put_u8(b, 2);        // Host
    put_u32(b, key);
    put_u8(b, 0);        // SubmitBound
    // Slot[0]
    put_u32(b, 2);       // n_ops
    put_u8(b, 0x10);     // Add
    put_u32(b, 0);       // a = logits
    put_u32(b, 1);       // b = mask
    put_u8(b, 0x33);     // ReduceArgmax
    put_u32(b, 2);       // v = Add result (id 2)
    put_u32(b, 1);       // n_outputs
    put_u32(b, 3);       // value id 3
    put_u8(b, 0);        // Token
    return b;
}

}  // namespace

int main() {
    // Establish the primary context so the runtime's runtime-API staging and
    // the backend's driver-API launch share one context.
    cudaSetDevice(0);
    cudaFree(nullptr);

    {
        SamplingIrBackend backend;
        SamplingIrRuntime rt;
        rt.set_backend(&backend);

        PersistentInputs pi;
        pi.sampled = DeviceBuffer<std::int32_t>::alloc(1);

        const std::uint32_t len = 128;
        const std::uint32_t key = 7;
        std::vector<std::uint8_t> bytecode = build_host_argmax_v2(len, key);

        // Fire 1: peak at index 42.
        std::vector<float> a(len, 0.f);
        a[42] = 9.0f;
        std::int32_t t1 = fire(rt, pi, bytecode, a, key);
        std::fprintf(stderr, "fire 1: token=%d expect=%d\n", t1, host_argmax(a));
        CHECK(t1 == host_argmax(a));
        CHECK(t1 == 42);

        // Fire 2: SAME program, DIFFERENT host vector (peak at 100) — proves the
        // submit-bound value refreshes per fire with no recompile.
        std::vector<float> b(len, 0.f);
        b[100] = 5.0f;
        b[7] = 3.0f;
        std::int32_t t2 = fire(rt, pi, bytecode, b, key);
        std::fprintf(stderr, "fire 2: token=%d expect=%d\n", t2, host_argmax(b));
        CHECK(t2 == host_argmax(b));
        CHECK(t2 == 100);

        // Fire 3: back to a third distribution (peak at 0) — boundary index.
        std::vector<float> c(len, -1.f);
        c[0] = 2.0f;
        std::int32_t t3 = fire(rt, pi, bytecode, c, key);
        std::fprintf(stderr, "fire 3: token=%d expect=%d\n", t3, host_argmax(c));
        CHECK(t3 == 0);

        // ── Combined shape: argmax(logits + mask) ───────────────────────
        // The real grammar/mirostat program: bf16 Intrinsic(Logits) + a
        // submit-bound F32 mask in ONE program. Proves intrinsic and
        // host-submit inputs bind together and the mask is applied on-device.
        const std::uint32_t vocab = 256;
        const std::uint32_t mkey = 11;
        std::vector<std::uint8_t> mask_prog = build_logits_plus_mask_v2(vocab, mkey);

        // Real device logits: natural argmax at 200.
        std::vector<std::uint16_t> h_logits(vocab);
        for (std::uint32_t i = 0; i < vocab; ++i)
            h_logits[i] = f32_to_bf16(static_cast<float>(i % 20));
        h_logits[200] = f32_to_bf16(50.0f);
        std::uint16_t* d_logits = nullptr;
        cudaMalloc(&d_logits, vocab * sizeof(std::uint16_t));
        cudaMemcpy(d_logits, h_logits.data(), vocab * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice);

        // Mask bans the natural argmax (−inf at 200) and boosts 137 → the
        // masked argmax must move to 137. (Grammar-style hard constraint.)
        std::vector<float> mask(vocab, 0.0f);
        mask[200] = -1e30f;
        mask[137] = 100.0f;

        SubmitInput msi;
        msi.key = mkey;
        msi.data = reinterpret_cast<const std::uint8_t*>(mask.data());
        msi.len_bytes = mask.size() * sizeof(float);

        FireContext mctx;
        mctx.program_bytecode = {mask_prog.data(), mask_prog.size()};
        mctx.submit_inputs = std::span<const SubmitInput>(&msi, 1);
        mctx.logits = d_logits;
        mctx.pi = &pi;
        mctx.vocab_size = static_cast<int>(vocab);
        mctx.sample_row = 0;
        mctx.prng_offset = 0;
        mctx.stream = nullptr;

        const RunStatus mst = rt.try_run(mctx);
        CHECK(mst == RunStatus::Handled);
        cudaDeviceSynchronize();
        std::int32_t mtok = -1;
        cudaMemcpy(&mtok, pi.sampled.data(), sizeof(std::int32_t),
                   cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "logits+mask: token=%d expect=137 (natural=200)\n", mtok);
        CHECK(mtok == 137);

        cudaFree(d_logits);

        // ── WS1b host-late: value supplied via late_value_inputs ────────
        // A host{late-bound} input resolved through the staged late-value path
        // (correctness path: value known by submit time). Same argmax program,
        // but declared LateBound — the runtime stages the late bytes and binds.
        const std::uint32_t lkey = 21;
        std::vector<std::uint8_t> late_prog = build_host_argmax_v2(len, lkey, /*avail=*/1);
        std::vector<float> lv(len, 0.f);
        lv[88] = 7.0f;
        {
            SubmitInput lsi;
            lsi.key = lkey;
            lsi.data = reinterpret_cast<const std::uint8_t*>(lv.data());
            lsi.len_bytes = lv.size() * sizeof(float);
            FireContext lctx;
            lctx.program_bytecode = {late_prog.data(), late_prog.size()};
            lctx.late_value_inputs = std::span<const SubmitInput>(&lsi, 1);
            lctx.logits = nullptr;
            lctx.pi = &pi;
            lctx.vocab_size = static_cast<int>(len);
            lctx.sample_row = 0;
            lctx.stream = nullptr;
            const RunStatus st = rt.try_run(lctx);
            CHECK(st == RunStatus::Handled);
            cudaDeviceSynchronize();
            std::int32_t tok = -1;
            cudaMemcpy(&tok, pi.sampled.data(), sizeof(tok), cudaMemcpyDeviceToHost);
            std::fprintf(stderr, "host-late supplied: token=%d expect=88\n", tok);
            CHECK(tok == 88);
        }

        // ── WS1b host-late miss = skip ──────────────────────────────────
        // Same LateBound program, but NO value supplied (no late_value_inputs,
        // no late_inputs) → SkippedLateBindMiss (discard + retry, fail loud).
        {
            FireContext lctx;
            lctx.program_bytecode = {late_prog.data(), late_prog.size()};
            lctx.logits = nullptr;
            lctx.pi = &pi;
            lctx.vocab_size = static_cast<int>(len);
            lctx.sample_row = 0;
            lctx.stream = nullptr;
            const RunStatus st = rt.try_run(lctx);
            std::fprintf(stderr, "host-late missing: status=%d (expect SkippedLateBindMiss=2)\n",
                         static_cast<int>(st));
            CHECK(st == RunStatus::SkippedLateBindMiss);
        }
    }

    if (g_failures == 0) {
        std::fprintf(stderr, "sampling_ir_host_submit: OK\n");
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_host_submit: %d failure(s)\n", g_failures);
    return 1;
}
