//===-- cuda_reference.cu - the Metal kernels' arithmetic, taken from CUDA -===//
//
// `crates/kernels-metal/kernels/**` was written by agents with no Metal
// compiler and no device in front of them. Its arithmetic has never been
// executed. `crates/kernels-cuda/kernels/**` is the SAME arithmetic in code
// that a CUDA host actually serves with, and it compiles and runs on the L40S
// this file was driven on. So the question these fixtures answer is not "is
// the Metal source plausible" -- reading it answers that -- it is "does the
// Metal kernel, on a device, produce the number the CUDA kernel produces from
// the same bits".
//
// That question needs the INPUT as much as the output, which is what makes
// this a fixture generator rather than a golden hash. `gemm_service/oracle.cu`
// beside this file in `driver-cuda/tests/oracle/` prints an fnv1a64 per shape
// because its inputs come from a recurrence both sides run; here the two sides
// are two different languages on two different machines, so the bits have to
// travel. Every array a case needs to be reproducible is written out: inputs,
// scalars, outputs.
//
// # What it drives, and against which Metal entry point
//
//   Metal (kernels-metal/kernels/...)        CUDA (kernels-cuda/kernels/...)
//   ---------------------------------------  ---------------------------------
//   mlp/packed.metal
//     packed_swiglu                          mlp::chunked_swiglu<bf16>
//     packed_swiglu_clamp                    mlp::chunked_swiglu_clamp<bf16>
//     packed_gptoss_swiglu                   mlp::chunked_gpt_oss_glu<bf16>
//     packed_geglu_tanh                      mlp::chunked_geglu_tanh<bf16>
//     packed_situ                            mlp::chunked_situ<bf16>
//   moe/route.metal
//     router_topk (softmax_over_all == 0)    moe::topk_softmax<bf16>
//                                            moe::topk_softmax_warp_x2<bf16>
//     router_topk_sigmoid                    moe::topk_sigmoid<bf16>, bias null
//     router_topk_sqrt_softplus              moe::topk_sqrtsoftplus<bf16>
//   rope/neox.metal
//     rope_neox_mb                           rope::rotate<false, false>
//     rope_neox_prop_mb                      rope::rotate_partial<bf16>
//     rope_neox_last_mb                      rope::rotate_partial_last
//     rope_neox_yarn_mb                      rope::rotate_yarn_original
//
// Every launch below uses the grid, the block and the dynamic shared size the
// Rust claim body for that point states -- `kernels-cuda/src/mlp.rs`,
// `src/moe.rs`, `src/rope.rs` -- because the argument ORDER and the parameter
// MEANINGS are stated there and nowhere else. `chunked_situ`'s
// `linear_beta` is `Mlp::situ`'s `up_cap`; `topk_softmax`'s `act`, `bias` and
// `hidden` are the fused form's and the row passes two nulls and a zero;
// `rotate`'s `cache_pairs` and `heads_per_block` are host arithmetic
// (`rope.rs::cache_pairs`, `::heads_per_block`) and are reproduced here.
//
// # The three places the two sides are NOT the same kernel
//
// 1. `rope_neox_mb` at a PARTIAL rotary has no CUDA twin. It pairs
//    `(i, i + rotary/2)` and divides the exponent by `rotary`, over a head
//    that is `head_dim` wide; `rotate` pairs `(i, i + head_dim/2)` over
//    `head_dim`, and `rotate_partial` does too. The arithmetic is
//    nevertheless reachable, because it depends only on `(pos, dim_pair)` and
//    the two loaded values: `rotate` driven at `head_dim = rotary` over a
//    buffer holding just the leading `rotary` channels of each head IS the
//    partial rotation, channel for channel. `neox_mb_partial` below is taken
//    that way and the repack is the only host arithmetic in this file that
//    is not a launch parameter. `neox_mb_full` needs no such argument.
//
// 2. `packed_gptoss_swiglu` spells the sigmoid as `g * (1 / (1 + exp(-a*g)))`
//    where `chunked_gpt_oss_glu` spells it `g / (1 + exp(-a*g))`, and Metal
//    reaches for `fast::exp` where CUDA reaches for `expf`. Both differences
//    are below the bf16 storage floor -- eight mantissa bits -- so the
//    recorded outputs are the right target, but they are a reason to compare
//    with a tolerance of an ulp or two of bf16 rather than bit for bit.
//    `precise::tanh` vs `tanhf` in `packed_geglu_tanh` / `packed_situ`, and
//    `fast::cos`/`fast::sin` vs `__sincosf` in every rope variant, are the
//    same kind of difference. `exp2(-d * log2(theta))` vs
//    `powf(theta, -d)` is one more.
//
// 3. `router_topk`'s `softmax_over_all != 0` arm -- softmax over ALL experts,
//    then select -- has no CUDA twin at all. `topk_softmax` renormalises over
//    the K it picked, which is the `== 0` arm. Both CUDA forms below are the
//    `== 0` arm; the other is not covered.
//
// `topk_softmax` and `topk_softmax_warp_x2` are BOTH recorded over the same
// logits and they are two different expressions of one answer: the block form
// takes the full softmax and renormalises the K, the warp form selects on the
// raw logits and exponentiates only the K. Metal's `router_topk` is spelled
// like the warp form. Their agreement is itself a reading of how much of the
// difference survives f32.
//
// # bf16 in, bf16 out
//
// Every bf16 input is generated in f32, rounded to bf16 with the same
// round-to-nearest-even `prelude/device.cuh::f32_to_bf16` uses, widened back,
// and the WIDENED value is what is both fed to the kernel and written to the
// fixture -- so the Metal side loads the identical sixteen bits. Outputs the
// kernel writes as bf16 are recorded as the widened f32 of those sixteen bits.
// A router's weights and a YaRN ramp bound are f32 on both sides and are
// recorded as they are.
//
// Regenerate with `run.sh` beside this file.
//
//===----------------------------------------------------------------------===//

#include "mlp/swiglu.cuh"
#include "moe/dsv4_routing.cuh"
#include "moe/topk_sigmoid.cuh"
#include "moe/topk_softmax.cuh"
#include "rope/rope.cuh"
#include "ssm/causal_conv1d.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using pie::bf16;

// ── the scalar layer, on the host ──────────────────────────────────────────
//
// `pie::bf16`'s constructors are `__device__`, so a host buffer is built as
// raw `uint16_t` and the DEVICE pointer is reinterpreted. `bf16` is a struct
// of one `unsigned short`: two bytes, aligned to two, standard layout.

uint32_t f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

float bits_f32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// `f32 -> bf16`, round-to-nearest-even. The host mirror of
/// `prelude/device.cuh::f32_to_bf16`, line for line.
uint16_t to_bf16(float f) {
    const uint32_t b = f32_bits(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((b >> 16) | 0x0040u);
    }
    const uint32_t rounding = 0x7fffu + ((b >> 16) & 1u);
    return static_cast<uint16_t>((b + rounding) >> 16);
}

/// `bf16 -> f32`. The host mirror of `prelude/device.cuh::bf16_to_f32`.
float from_bf16(uint16_t r) { return bits_f32(static_cast<uint32_t>(r) << 16); }

/// The value a bf16 round trip leaves behind. Everything this file feeds a
/// kernel as bf16 goes through here FIRST, so the fixture and the kernel see
/// one number.
float bf16_exact(float f) { return from_bf16(to_bf16(f)); }

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", cudaGetErrorString(e), what);
        std::exit(2);
    }
}

void sync(const char* what) {
    ck(cudaGetLastError(), what);
    ck(cudaDeviceSynchronize(), what);
}

// ── the inputs ─────────────────────────────────────────────────────────────
//
// splitmix64, seeded per case by name, so a case's numbers do not move when
// another case is added above it.

struct Rng {
    uint64_t s;

    explicit Rng(const char* name) : s(0x243f6a8885a308d3ull) {
        for (const char* p = name; *p; ++p) {
            s ^= static_cast<unsigned char>(*p);
            s *= 0x100000001b3ull;
        }
    }

    uint64_t next() {
        s += 0x9e3779b97f4a7c15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    /// Uniform on [0, 1).
    float unit() { return static_cast<float>((next() >> 40) * (1.0 / 16777216.0)); }

    /// Uniform on (-a, a), rounded to a bf16 the Metal side can load.
    float sym_bf16(float a) { return bf16_exact(a * (2.0f * unit() - 1.0f)); }

    /// Uniform on (-a, a), left at full f32 width. For operands that are
    /// `float` on BOTH sides -- a correction bias, a per-expert scale --
    /// where rounding to bf16 would hide a reader that loads the wrong width.
    float sym_f32(float a) { return a * (2.0f * unit() - 1.0f); }
};

// ── the fixture file ───────────────────────────────────────────────────────

class Fixture {
  public:
    Fixture(const std::string& dir, const char* name) {
        const std::string path = dir + "/" + name;
        f_ = std::fopen(path.c_str(), "wb");
        if (f_ == nullptr) {
            std::fprintf(stderr, "cannot write %s\n", path.c_str());
            std::exit(2);
        }
        path_ = path;
    }

    ~Fixture() {
        if (f_ != nullptr) std::fclose(f_);
    }

    void comment(const char* fmt, ...) {
        std::va_list ap;
        va_start(ap, fmt);
        std::fputs("# ", f_);
        std::vfprintf(f_, fmt, ap);
        std::fputc('\n', f_);
        va_end(ap);
    }

    void blank_comment() { std::fputs("#\n", f_); }

    void open_case(const char* name) { std::fprintf(f_, "case %s\n", name); }

    void f32(const char* name, const std::vector<float>& v) {
        std::fprintf(f_, "array %s f32 %zu\n", name, v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            std::fprintf(f_, "%08x%c", f32_bits(v[i]),
                         (i % 8 == 7 || i + 1 == v.size()) ? '\n' : ' ');
        }
    }

    void i32(const char* name, const std::vector<int32_t>& v) {
        std::fprintf(f_, "array %s i32 %zu\n", name, v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            std::fprintf(f_, "%08x%c", static_cast<uint32_t>(v[i]),
                         (i % 8 == 7 || i + 1 == v.size()) ? '\n' : ' ');
        }
    }

    void scalar_f32(const char* name, float v) { f32(name, std::vector<float>{v}); }
    void scalar_i32(const char* name, int32_t v) { i32(name, std::vector<int32_t>{v}); }

    const std::string& path() const { return path_; }

  private:
    std::FILE* f_ = nullptr;
    std::string path_;
};

/// The four lines every fixture opens with. What produced it, on what, when.
void header(Fixture& fx, const char* what) {
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    fx.comment("%s", what);
    fx.comment("produced by crates/driver-metal/tests/fixtures/cuda_reference.cu"
               " (regenerate with run.sh beside it)");
    fx.comment("gpu: %s, sm_%d%d", prop.name, prop.major, prop.minor);
    fx.comment("nvcc: %d.%d.%d", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__,
               __CUDACC_VER_BUILD__);
    fx.comment("built: %s", __DATE__);
    fx.blank_comment();
    fx.comment("Arrays are named for the CUDA kernel's own parameters, IN LOWER"
               " CASE: a name in this");
    fx.comment("format is [a-z0-9_], so `I`, `K`, `E`, `R` and `C` are written"
               " `i`, `k`, `e`, `r` and `c`.");
    fx.comment("Every f32 token is the little-endian bit pattern of the value;"
               " a bf16 operand is recorded");
    fx.comment("as the f32 its sixteen bits widen to, so the Metal side loads"
               " the same bits after narrowing.");
    fx.comment("A scalar is an array of length 1.");
    fx.blank_comment();
}

// ── device buffers ─────────────────────────────────────────────────────────

/// Upload an f32 host vector as bf16. Every element must ALREADY be a bf16
/// round trip (see `bf16_exact`); this asserts it rather than rounding
/// silently, because a value the fixture and the kernel disagree about is the
/// one failure this whole file exists to rule out.
bf16* upload_bf16(const std::vector<float>& host, const char* what) {
    std::vector<uint16_t> bits(host.size());
    for (size_t i = 0; i < host.size(); ++i) {
        bits[i] = to_bf16(host[i]);
        if (from_bf16(bits[i]) != host[i]) {
            std::fprintf(stderr,
                         "%s[%zu] = %.9g is not exactly representable in bf16\n",
                         what, i, static_cast<double>(host[i]));
            std::exit(2);
        }
    }
    void* d = nullptr;
    ck(cudaMalloc(&d, bits.size() * 2), what);
    ck(cudaMemcpy(d, bits.data(), bits.size() * 2, cudaMemcpyHostToDevice), what);
    return static_cast<bf16*>(d);
}

std::vector<float> download_bf16(const bf16* d, size_t n, const char* what) {
    std::vector<uint16_t> bits(n);
    ck(cudaMemcpy(bits.data(), d, n * 2, cudaMemcpyDeviceToHost), what);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = from_bf16(bits[i]);
    return out;
}

template <class T>
T* upload(const std::vector<T>& host, const char* what) {
    void* d = nullptr;
    ck(cudaMalloc(&d, host.size() * sizeof(T)), what);
    ck(cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
       what);
    return static_cast<T*>(d);
}

template <class T>
T* alloc(size_t n, const char* what) {
    void* d = nullptr;
    ck(cudaMalloc(&d, n * sizeof(T)), what);
    ck(cudaMemset(d, 0, n * sizeof(T)), what);
    return static_cast<T*>(d);
}

template <class T>
std::vector<T> download(const T* d, size_t n, const char* what) {
    std::vector<T> out(n);
    ck(cudaMemcpy(out.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), what);
    return out;
}

//===----------------------------------------------------------------------===//
// 1. The five packed MLP activations.
//===----------------------------------------------------------------------===//
//
// The packed row is `[gate | up]`, `I` wide each, gate FIRST -- which is what
// `chunked_*`'s `gate_offset<false>` says and what `packed.metal`'s header
// says, in the same words. `y` is `[N, I]`.
//
// `I = 40` and `N = 5`: neither is a multiple of the 256-wide block and
// neither is a power of two, so a kernel that confused `I` with the block
// width, or a row stride with `2 * I`, lands outside its row.
//
// The launch is `mlp.rs::elementwise_rows(rows, width)`: `dim3(N,
// ceil(I / 256))` blocks of 256, which is `blockIdx.x = n` and
// `blockIdx.y * blockDim.x + threadIdx.x = i` exactly as the kernel reads
// them.

constexpr int kMlpBlock = 256;
constexpr int kMlpRows = 5;
constexpr int kMlpI = 40;

dim3 mlp_grid(int rows, int width) {
    return dim3(static_cast<unsigned>(rows),
                static_cast<unsigned>((width + kMlpBlock - 1) / kMlpBlock), 1);
}

/// The packed activation for one case. `amp` is the half-width of the range
/// the values are drawn from; the clamped activations want one wide enough
/// that the clamp actually bites, which `require_clamped` then checks.
std::vector<float> packed_input(const char* case_name, float amp) {
    Rng rng(case_name);
    std::vector<float> packed(static_cast<size_t>(kMlpRows) * 2 * kMlpI);
    for (float& v : packed) v = rng.sym_bf16(amp);
    return packed;
}

void require_clamped(const char* case_name, const std::vector<float>& packed,
                     float limit) {
    int gate_hi = 0, up_hi = 0, up_lo = 0;
    for (int n = 0; n < kMlpRows; ++n) {
        for (int i = 0; i < kMlpI; ++i) {
            const float g = packed[static_cast<size_t>(n) * 2 * kMlpI + i];
            const float u = packed[static_cast<size_t>(n) * 2 * kMlpI + kMlpI + i];
            if (g > limit) ++gate_hi;
            if (u > limit) ++up_hi;
            if (u < -limit) ++up_lo;
        }
    }
    if (gate_hi == 0 || up_hi == 0 || up_lo == 0) {
        std::fprintf(stderr,
                     "%s: the clamp never bites (gate>limit %d, up>limit %d, "
                     "up<-limit %d) -- widen the range\n",
                     case_name, gate_hi, up_hi, up_lo);
        std::exit(2);
    }
}

void emit_mlp(Fixture& fx) {
    header(fx, "Reference vectors for kernels-metal/kernels/mlp/packed.metal.");
    fx.comment("Each case is one launch of the CUDA twin in"
               " kernels-cuda/kernels/mlp/swiglu.cuh over a");
    fx.comment("packed [N, 2I] activation, gate half FIRST, writing y at [N, I]."
               " N = %d, I = %d.", kMlpRows, kMlpI);
    fx.comment("Grid dim3(N, ceil(I/256)) x 256 threads -- mlp.rs::elementwise_rows.");
    fx.blank_comment();
    fx.comment("packed_swiglu        <- pie::mlp::chunked_swiglu<bf16>(packed, y, I=%d)",
               kMlpI);
    fx.comment("packed_swiglu_clamp  <- pie::mlp::chunked_swiglu_clamp<bf16>"
               "(packed, y, I=%d, limit=3)", kMlpI);
    fx.comment("packed_gptoss_swiglu <- pie::mlp::chunked_gpt_oss_glu<bf16>"
               "(packed, y, I=%d, limit=7, alpha=1.702)", kMlpI);
    fx.comment("packed_geglu_tanh    <- pie::mlp::chunked_geglu_tanh<bf16>"
               "(packed, y, I=%d)", kMlpI);
    fx.comment("packed_situ          <- pie::mlp::chunked_situ<bf16>"
               "(packed, y, I=%d, beta=2, linear_beta=1.5)", kMlpI);
    fx.comment("packed_situ_uncapped <- pie::mlp::chunked_situ<bf16>"
               "(packed, y, I=%d, beta=2, linear_beta=0)  -- the no-cap branch",
               kMlpI);
    fx.blank_comment();
    fx.comment("`linear_beta` is what `Mlp::situ` and packed.metal call `up_cap`;"
               " `I` is `intermediate`.");
    fx.blank_comment();

    const size_t y_n = static_cast<size_t>(kMlpRows) * kMlpI;
    const dim3 grid = mlp_grid(kMlpRows, kMlpI);

    // -- packed_swiglu -----------------------------------------------------
    {
        const char* name = "packed_swiglu";
        auto packed = packed_input(name, 5.0f);
        bf16* dp = upload_bf16(packed, name);
        bf16* dy = alloc<bf16>(y_n, name);
        pie::mlp::chunked_swiglu<bf16><<<grid, kMlpBlock>>>(dp, dy, kMlpI);
        sync(name);
        fx.open_case(name);
        fx.f32("packed", packed);
        fx.scalar_i32("i", kMlpI);
        fx.f32("y", download_bf16(dy, y_n, name));
        ck(cudaFree(dp), name);
        ck(cudaFree(dy), name);
    }

    // -- packed_swiglu_clamp -----------------------------------------------
    {
        const char* name = "packed_swiglu_clamp";
        const float limit = 3.0f;
        auto packed = packed_input(name, 6.0f);
        require_clamped(name, packed, limit);
        bf16* dp = upload_bf16(packed, name);
        bf16* dy = alloc<bf16>(y_n, name);
        pie::mlp::chunked_swiglu_clamp<bf16><<<grid, kMlpBlock>>>(dp, dy, kMlpI, limit);
        sync(name);
        fx.open_case(name);
        fx.f32("packed", packed);
        fx.scalar_i32("i", kMlpI);
        fx.scalar_f32("limit", limit);
        fx.f32("y", download_bf16(dy, y_n, name));
        ck(cudaFree(dp), name);
        ck(cudaFree(dy), name);
    }

    // -- packed_gptoss_swiglu ----------------------------------------------
    {
        const char* name = "packed_gptoss_swiglu";
        const float limit = 7.0f;
        const float alpha = 1.702f;
        auto packed = packed_input(name, 12.0f);
        require_clamped(name, packed, limit);
        bf16* dp = upload_bf16(packed, name);
        bf16* dy = alloc<bf16>(y_n, name);
        pie::mlp::chunked_gpt_oss_glu<bf16>
            <<<grid, kMlpBlock>>>(dp, dy, kMlpI, limit, alpha);
        sync(name);
        fx.open_case(name);
        fx.f32("packed", packed);
        fx.scalar_i32("i", kMlpI);
        fx.scalar_f32("limit", limit);
        fx.scalar_f32("alpha", alpha);
        fx.f32("y", download_bf16(dy, y_n, name));
        ck(cudaFree(dp), name);
        ck(cudaFree(dy), name);
    }

    // -- packed_geglu_tanh -------------------------------------------------
    {
        const char* name = "packed_geglu_tanh";
        auto packed = packed_input(name, 4.0f);
        bf16* dp = upload_bf16(packed, name);
        bf16* dy = alloc<bf16>(y_n, name);
        pie::mlp::chunked_geglu_tanh<bf16><<<grid, kMlpBlock>>>(dp, dy, kMlpI);
        sync(name);
        fx.open_case(name);
        fx.f32("packed", packed);
        fx.scalar_i32("i", kMlpI);
        fx.f32("y", download_bf16(dy, y_n, name));
        ck(cudaFree(dp), name);
        ck(cudaFree(dy), name);
    }

    // -- packed_situ, capped and not ---------------------------------------
    for (int capped = 1; capped >= 0; --capped) {
        const char* name = capped ? "packed_situ" : "packed_situ_uncapped";
        const float beta = 2.0f;
        const float linear_beta = capped ? 1.5f : 0.0f;
        auto packed = packed_input(name, 5.0f);
        bf16* dp = upload_bf16(packed, name);
        bf16* dy = alloc<bf16>(y_n, name);
        pie::mlp::chunked_situ<bf16>
            <<<grid, kMlpBlock>>>(dp, dy, kMlpI, beta, linear_beta);
        sync(name);
        fx.open_case(name);
        fx.f32("packed", packed);
        fx.scalar_i32("i", kMlpI);
        fx.scalar_f32("beta", beta);
        fx.scalar_f32("linear_beta", linear_beta);
        fx.f32("y", download_bf16(dy, y_n, name));
        ck(cudaFree(dp), name);
        ck(cudaFree(dy), name);
    }
}

//===----------------------------------------------------------------------===//
// 2. The routers.
//===----------------------------------------------------------------------===//
//
// `E = 40` experts, `K = 6` routed: neither divides 32, so a router that
// reduced within one simdgroup or that assumed the expert count fills its
// lanes picks a quarter-local winner. `rows = 5`.
//
// The logits are drawn distinct WITHIN a row and checked to be so, because a
// tie is resolved by index on both sides and a comparison that disagreed
// about the tiebreak would look like an arithmetic failure. A tie is a case
// worth having; it is not one of these.

constexpr int kRouterRows = 5;
constexpr int kRouterExperts = 40;
constexpr int kRouterK = 6;

/// Logits with no repeated bf16 pattern inside a row.
std::vector<float> router_logits(const char* case_name, int rows, int experts,
                                 float amp) {
    Rng rng(case_name);
    std::vector<float> logits(static_cast<size_t>(rows) * experts);
    for (int n = 0; n < rows; ++n) {
        for (int e = 0; e < experts; ++e) {
            for (int attempt = 0;; ++attempt) {
                const float v = rng.sym_bf16(amp);
                bool clash = false;
                for (int j = 0; j < e; ++j) {
                    if (logits[static_cast<size_t>(n) * experts + j] == v) {
                        clash = true;
                        break;
                    }
                }
                if (!clash) {
                    logits[static_cast<size_t>(n) * experts + e] = v;
                    break;
                }
                if (attempt > 1000) {
                    std::fprintf(stderr, "%s: cannot draw %d distinct bf16 logits\n",
                                 case_name, experts);
                    std::exit(2);
                }
            }
        }
    }
    return logits;
}

/// Refuse a fixture whose ranking has a tie in it: the scan's tiebreak is not
/// what these vectors are for.
void require_no_ties(const char* case_name, const std::vector<float>& score,
                     int rows, int experts) {
    for (int n = 0; n < rows; ++n) {
        for (int a = 0; a < experts; ++a) {
            for (int b = a + 1; b < experts; ++b) {
                if (score[static_cast<size_t>(n) * experts + a] ==
                    score[static_cast<size_t>(n) * experts + b]) {
                    std::fprintf(stderr, "%s: row %d ranks experts %d and %d equal\n",
                                 case_name, n, a, b);
                    std::exit(2);
                }
            }
        }
    }
}

void emit_router(Fixture& fx) {
    header(fx, "Reference vectors for the three routers in"
               " kernels-metal/kernels/moe/route.metal.");
    fx.comment("topk_softmax_block     <- pie::moe::topk_softmax<bf16>"
               "(logits, null, null, topk_idx, topk_w, num_experts=%d, K=%d,"
               " hidden=0)", kRouterExperts, kRouterK);
    fx.comment("topk_softmax_warp      <- pie::moe::topk_softmax_warp_x2<bf16>"
               "(logits, topk_idx, topk_w, num_experts=%d, K=%d) -- SAME logits",
               kRouterExperts, kRouterK);
    fx.comment("topk_sigmoid_renorm    <- pie::moe::topk_sigmoid<bf16>"
               "(logits, topk_idx, topk_w, bias=null, E=%d, K=%d, renormalize=1,"
               " routed_scaling_factor=2.5)", kRouterExperts, kRouterK);
    fx.comment("topk_sigmoid_plain     <- the same, renormalize=0,"
               " routed_scaling_factor=1.5");
    fx.comment("topk_sigmoid_fanout    <- the same at E=5, K=8: K > E, so the"
               " spare slots park on expert 0 at weight 0");
    fx.comment("topk_sqrtsoftplus_bias <- pie::moe::topk_sqrtsoftplus<bf16>"
               "(logits, topk_idx, topk_w, correction_bias, E=%d, K=%d,"
               " renormalize=1, routed_scaling_factor=1)", kRouterExperts, kRouterK);
    fx.comment("topk_sqrtsoftplus_plain<- the same with a zero bias,"
               " renormalize=0, routed_scaling_factor=1.5");
    fx.blank_comment();
    fx.comment("The block form is 1 block of 64 threads per row"
               " (moe.rs::router_lane); the warp form 1 block of 32;");
    fx.comment("the two ranked routers 1 block of 256 (moe.rs::rms).");
    fx.blank_comment();
    fx.comment("BOTH softmax cases are route.metal's `softmax_over_all == 0` arm"
               " -- a softmax over the");
    fx.comment("SELECTED k. topk_softmax renormalises the full softmax over its K"
               " and topk_softmax_warp_x2");
    fx.comment("exponentiates only the K, which is how router_topk spells it;"
               " the two agree to f32 rounding.");
    fx.comment("route.metal's `softmax_over_all != 0` arm has NO CUDA twin and is"
               " not covered here.");
    fx.comment("router_topk_sigmoid has no correction bias, so topk_sigmoid is"
               " driven with a null one.");
    fx.blank_comment();

    // -- the two softmax forms, over one set of logits ----------------------
    {
        auto logits = router_logits("topk_softmax", kRouterRows, kRouterExperts, 4.0f);
        require_no_ties("topk_softmax", logits, kRouterRows, kRouterExperts);
        bf16* dl = upload_bf16(logits, "topk_softmax");
        const size_t out_n = static_cast<size_t>(kRouterRows) * kRouterK;

        {
            const char* name = "topk_softmax_block";
            int32_t* di = alloc<int32_t>(out_n, name);
            float* dw = alloc<float>(out_n, name);
            pie::moe::topk_softmax<bf16><<<kRouterRows, 64>>>(
                dl, nullptr, nullptr, di, dw, kRouterExperts, kRouterK, 0);
            sync(name);
            fx.open_case(name);
            fx.f32("logits", logits);
            fx.scalar_i32("num_experts", kRouterExperts);
            fx.scalar_i32("k", kRouterK);
            fx.i32("topk_idx", download(di, out_n, name));
            fx.f32("topk_w", download(dw, out_n, name));
            ck(cudaFree(di), name);
            ck(cudaFree(dw), name);
        }
        {
            // PerLane = 2 covers 64 lanes' worth of experts, which is the rung
            // `topk_softmax.cu`'s ladder picks for 33..64 experts.
            const char* name = "topk_softmax_warp";
            int32_t* di = alloc<int32_t>(out_n, name);
            float* dw = alloc<float>(out_n, name);
            pie::moe::topk_softmax_warp_x2<bf16><<<kRouterRows, 32>>>(
                dl, di, dw, kRouterExperts, kRouterK);
            sync(name);
            fx.open_case(name);
            fx.f32("logits", logits);
            fx.scalar_i32("num_experts", kRouterExperts);
            fx.scalar_i32("k", kRouterK);
            fx.i32("topk_idx", download(di, out_n, name));
            fx.f32("topk_w", download(dw, out_n, name));
            ck(cudaFree(di), name);
            ck(cudaFree(dw), name);
        }
        ck(cudaFree(dl), "topk_softmax");
    }

    // -- topk_sigmoid, renormalised and not --------------------------------
    {
        auto logits = router_logits("topk_sigmoid", kRouterRows, kRouterExperts, 4.0f);
        std::vector<float> gate(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            gate[i] = 1.0f / (1.0f + std::exp(-logits[i]));
        }
        require_no_ties("topk_sigmoid", gate, kRouterRows, kRouterExperts);
        bf16* dl = upload_bf16(logits, "topk_sigmoid");
        const size_t out_n = static_cast<size_t>(kRouterRows) * kRouterK;

        struct Arm {
            const char* name;
            bool renormalize;
            float scaling;
        };
        const Arm arms[] = {{"topk_sigmoid_renorm", true, 2.5f},
                            {"topk_sigmoid_plain", false, 1.5f}};
        for (const Arm& arm : arms) {
            int32_t* di = alloc<int32_t>(out_n, arm.name);
            float* dw = alloc<float>(out_n, arm.name);
            pie::moe::topk_sigmoid<bf16><<<kRouterRows, 256>>>(
                dl, di, dw, nullptr, kRouterExperts, kRouterK, arm.renormalize,
                arm.scaling);
            sync(arm.name);
            fx.open_case(arm.name);
            fx.f32("logits", logits);
            fx.scalar_i32("e", kRouterExperts);
            fx.scalar_i32("k", kRouterK);
            fx.scalar_i32("renormalize", arm.renormalize ? 1 : 0);
            fx.scalar_f32("routed_scaling_factor", arm.scaling);
            fx.i32("topk_idx", download(di, out_n, arm.name));
            fx.f32("topk_w", download(dw, out_n, arm.name));
            ck(cudaFree(di), arm.name);
            ck(cudaFree(dw), arm.name);
        }
        ck(cudaFree(dl), "topk_sigmoid");
    }

    // -- topk_sigmoid with a fan-out wider than the expert count -----------
    //
    // `picks = min(K, E)` on both sides, and both park the remainder on
    // expert 0 at weight zero rather than repeating the last winner. This is
    // the one place the two ranked routers do NOT agree with each other:
    // `topk_sqrtsoftplus` has no `picks` and would run its scan K times past
    // the experts, so no fan-out case is taken for it.
    {
        const char* name = "topk_sigmoid_fanout";
        constexpr int kE = 5;
        constexpr int kK = 8;
        constexpr int kRows = 3;
        auto logits = router_logits(name, kRows, kE, 4.0f);
        std::vector<float> gate(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            gate[i] = 1.0f / (1.0f + std::exp(-logits[i]));
        }
        require_no_ties(name, gate, kRows, kE);
        bf16* dl = upload_bf16(logits, name);
        const size_t out_n = static_cast<size_t>(kRows) * kK;
        int32_t* di = alloc<int32_t>(out_n, name);
        float* dw = alloc<float>(out_n, name);
        pie::moe::topk_sigmoid<bf16>
            <<<kRows, 256>>>(dl, di, dw, nullptr, kE, kK, true, 1.0f);
        sync(name);
        fx.open_case(name);
        fx.f32("logits", logits);
        fx.scalar_i32("e", kE);
        fx.scalar_i32("k", kK);
        fx.scalar_i32("renormalize", 1);
        fx.scalar_f32("routed_scaling_factor", 1.0f);
        fx.i32("topk_idx", download(di, out_n, name));
        fx.f32("topk_w", download(dw, out_n, name));
        ck(cudaFree(dl), name);
        ck(cudaFree(di), name);
        ck(cudaFree(dw), name);
    }

    // -- topk_sqrtsoftplus, with a real bias and with a zero one -----------
    //
    // The bias is left at full f32 width: it is `const float*` on both sides,
    // and a reader that narrowed it to bf16 would produce a plausible ranking.
    {
        auto logits =
            router_logits("topk_sqrtsoftplus", kRouterRows, kRouterExperts, 4.0f);
        Rng bias_rng("topk_sqrtsoftplus_bias");
        std::vector<float> bias(kRouterExperts);
        for (float& b : bias) b = bias_rng.sym_f32(0.35f);
        std::vector<float> zero_bias(kRouterExperts, 0.0f);

        auto ssp = [](float x) {
            const float sp = x > 20.0f ? x : std::log1p(std::exp(x));
            return std::sqrt(std::fmax(sp, 0.0f));
        };
        std::vector<float> ranked(logits.size());
        for (int n = 0; n < kRouterRows; ++n) {
            for (int e = 0; e < kRouterExperts; ++e) {
                const size_t i = static_cast<size_t>(n) * kRouterExperts + e;
                ranked[i] = ssp(logits[i]) + bias[e];
            }
        }
        require_no_ties("topk_sqrtsoftplus_bias", ranked, kRouterRows, kRouterExperts);
        for (int n = 0; n < kRouterRows; ++n) {
            for (int e = 0; e < kRouterExperts; ++e) {
                const size_t i = static_cast<size_t>(n) * kRouterExperts + e;
                ranked[i] = ssp(logits[i]);
            }
        }
        require_no_ties("topk_sqrtsoftplus_plain", ranked, kRouterRows, kRouterExperts);

        bf16* dl = upload_bf16(logits, "topk_sqrtsoftplus");
        float* db = upload(bias, "topk_sqrtsoftplus");
        float* dz = upload(zero_bias, "topk_sqrtsoftplus");
        const size_t out_n = static_cast<size_t>(kRouterRows) * kRouterK;

        struct Arm {
            const char* name;
            float* bias_dev;
            const std::vector<float>* bias_host;
            bool renormalize;
            float scaling;
        };
        const Arm arms[] = {
            {"topk_sqrtsoftplus_bias", db, &bias, true, 1.0f},
            {"topk_sqrtsoftplus_plain", dz, &zero_bias, false, 1.5f}};
        for (const Arm& arm : arms) {
            int32_t* di = alloc<int32_t>(out_n, arm.name);
            float* dw = alloc<float>(out_n, arm.name);
            pie::moe::topk_sqrtsoftplus<bf16><<<kRouterRows, 256>>>(
                dl, di, dw, arm.bias_dev, kRouterExperts, kRouterK, arm.renormalize,
                arm.scaling);
            sync(arm.name);
            fx.open_case(arm.name);
            fx.f32("logits", logits);
            fx.f32("correction_bias", *arm.bias_host);
            fx.scalar_i32("e", kRouterExperts);
            fx.scalar_i32("k", kRouterK);
            fx.scalar_i32("renormalize", arm.renormalize ? 1 : 0);
            fx.scalar_f32("routed_scaling_factor", arm.scaling);
            fx.i32("topk_idx", download(di, out_n, arm.name));
            fx.f32("topk_w", download(dw, out_n, arm.name));
            ck(cudaFree(di), arm.name);
            ck(cudaFree(dw), arm.name);
        }
        ck(cudaFree(dl), "topk_sqrtsoftplus");
        ck(cudaFree(db), "topk_sqrtsoftplus");
        ck(cudaFree(dz), "topk_sqrtsoftplus");
    }
}

//===----------------------------------------------------------------------===//
// 3. RoPE neox.
//===----------------------------------------------------------------------===//
//
// `head_dim = 24`, `n_head = 5`, `N = 3` tokens, `rotary = 16` where a
// variant is partial. 24 is not 32 and 5 is not 4, so a head stride taken
// from the block width, or a token stride taken from one head, walks into the
// next row.
//
// The rotations are IN PLACE on both sides, so each case records `q_in` and
// `q_out` -- the same buffer before and after. `k` is not driven: every CUDA
// kernel here takes a `k` plane with its own head count and `neox.metal` turns
// ONE tensor, so `num_kv_heads` is zero throughout and the `k` branch is never
// reached.
//
// Positions are small on purpose. `__sincosf` is a fast intrinsic whose
// argument reduction loses accuracy as the angle grows -- rope.cuh's header
// tabulates it -- and `fast::cos`/`fast::sin` on the Metal side lose it
// differently. At `dim_pair = 0` the angle IS the position, so a position of
// 4096 would put the two sides' disagreement above the bf16 floor and turn a
// parity check into an accuracy check.

constexpr int kRopeTokens = 3;
constexpr int kRopeHeads = 5;
constexpr int kRopeHeadDim = 24;
constexpr int kRopeRotary = 16;
constexpr int kRopeBlock = 256;  // rope.rs::ROTATE_BLOCK
constexpr float kRopeTheta = 10000.0f;

/// `rope.rs::heads_per_block`.
int heads_per_block(int half) { return half >= kRopeBlock ? 1 : kRopeBlock / half; }

/// `rope.rs::cache_pairs`. `MAX_CACHED_PAIRS` is 4096 and every `half` here is
/// far below it.
int cache_pairs(int half) { return half <= 4096 ? half : 0; }

std::vector<float> rope_input(const char* case_name) {
    Rng rng(case_name);
    std::vector<float> q(static_cast<size_t>(kRopeTokens) * kRopeHeads * kRopeHeadDim);
    for (float& v : q) v = rng.sym_bf16(2.0f);
    return q;
}

std::vector<int32_t> rope_positions() { return {1, 6, 23}; }

/// The scalars every rope case carries. `base` is `log2(theta)` because that
/// is what `neox.metal` binds where CUDA binds `theta` itself -- one number,
/// two spellings, and writing both saves the reader from choosing which
/// rounding of the log to reproduce.
void rope_common(Fixture& fx, const std::vector<float>& q_in,
                 const std::vector<int32_t>& positions) {
    fx.f32("q_in", q_in);
    fx.i32("positions", positions);
    fx.scalar_i32("num_q_heads", kRopeHeads);
    fx.scalar_i32("head_dim", kRopeHeadDim);
    fx.scalar_f32("theta", kRopeTheta);
    fx.scalar_f32("base", std::log2(kRopeTheta));
}

void emit_rope(Fixture& fx) {
    header(fx, "Reference vectors for kernels-metal/kernels/rope/neox.metal.");
    fx.comment("N = %d tokens, n_head = %d, head_dim = %d, theta = %g,"
               " positions = {1, 6, 23}.", kRopeTokens, kRopeHeads, kRopeHeadDim,
               static_cast<double>(kRopeTheta));
    fx.comment("Every rotation is in place: `q_in` is the buffer as launched,"
               " `q_out` as it came back,");
    fx.comment("both [N, n_head, head_dim]. num_kv_heads is 0 throughout -- the"
               " Metal kernels turn one tensor.");
    fx.blank_comment();
    fx.comment("neox_mb_full             <- pie::rope::rotate<false, false>"
               "(q, k=q, positions, num_q_heads=%d, num_kv_heads=0, head_dim=%d,"
               " theta=%g, interleaved=false, cache_pairs, heads_per_block, ...)",
               kRopeHeads, kRopeHeadDim, static_cast<double>(kRopeTheta));
    fx.comment("neox_mb_partial          <- the same kernel at head_dim=%d over"
               " the leading %d channels of each head, then scattered back"
               " (see below)", kRopeRotary, kRopeRotary);
    fx.comment("neox_prop_mb             <- pie::rope::rotate_partial<bf16>"
               "(q, k=q, positions, position_delta=0, num_q_heads=%d,"
               " num_kv_heads=0, head_dim=%d, rotary_dim=%d, theta=%g)",
               kRopeHeads, kRopeHeadDim, kRopeRotary,
               static_cast<double>(kRopeTheta));
    fx.comment("neox_last_mb             <- pie::rope::rotate_partial_last"
               "(q, k=q, positions, num_q_heads=%d, num_kv_heads=0, head_dim=%d,"
               " rotary_dim=%d, theta=%g, inverse=false, interleaved=false,"
               " yarn_factor=1, 0, 0)", kRopeHeads, kRopeHeadDim, kRopeRotary,
               static_cast<double>(kRopeTheta));
    fx.comment("neox_last_mb_interleaved <- the same with interleaved=true");
    fx.comment("neox_yarn_mb             <- pie::rope::rotate_yarn_original"
               "(q, k=q, positions, num_q_heads=%d, num_kv_heads=0, head_dim=%d,"
               " theta=%g, factor, low_dim, high_dim, mscale, interleaved=false,"
               " heads_per_block, cache_pairs)", kRopeHeads, kRopeHeadDim,
               static_cast<double>(kRopeTheta));
    fx.comment("neox_yarn_mb_interleaved <- the same with interleaved=true");
    fx.blank_comment();
    fx.comment("THE PARTIAL CASE IS A REPACK, and it is the only host arithmetic"
               " in this file that is not");
    fx.comment("a launch parameter. rope_neox_mb at rotary < head_dim pairs"
               " (i, i + rotary/2) and divides");
    fx.comment("the exponent by rotary; no CUDA kernel does that -- `rotate` and"
               " `rotate_partial` both pair");
    fx.comment("(i, i + head_dim/2) over head_dim. The rotation depends only on"
               " (position, dim_pair) and the");
    fx.comment("two loaded values, so `rotate` at head_dim=%d over a [N, n_head,"
               " %d] buffer holding the leading", kRopeRotary, kRopeRotary);
    fx.comment("%d channels of each head IS that rotation, channel for channel."
               " q_in and q_out are recorded", kRopeRotary);
    fx.comment("at the FULL [N, n_head, %d] layout with the tail copied through"
               " untouched, which is what the", kRopeHeadDim);
    fx.comment("Metal kernel leaves behind.");
    fx.blank_comment();
    fx.comment("`base` is log2(theta): neox.metal binds the log where CUDA binds"
               " theta. `scale` is neox.metal's");
    fx.comment("position multiplier, which rotate/rotate_partial do not have"
               " -- 1.0 here, the value at which the");
    fx.comment("two are the same rotation. rope_neox_last_mb and"
               " rope_neox_yarn_mb take no scale on either side.");
    fx.blank_comment();

    const size_t q_n = static_cast<size_t>(kRopeTokens) * kRopeHeads * kRopeHeadDim;
    const auto positions = rope_positions();

    // -- neox_mb_full ------------------------------------------------------
    {
        const char* name = "neox_mb_full";
        auto q_in = rope_input(name);
        bf16* dq = upload_bf16(q_in, name);
        int32_t* dp = upload(positions, name);
        const int half = kRopeHeadDim / 2;
        const int pairs = cache_pairs(half);
        const int per_block = heads_per_block(half);
        const dim3 grid(kRopeTokens,
                        static_cast<unsigned>((kRopeHeads + per_block - 1) / per_block),
                        1);
        pie::rope::rotate<false, false>
            <<<grid, kRopeBlock, static_cast<unsigned>(pairs) * 2 * 4>>>(
                dq, dq, dp, kRopeHeads, 0, kRopeHeadDim, kRopeTheta, false, pairs,
                per_block, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, 0, 0);
        sync(name);
        fx.open_case(name);
        rope_common(fx, q_in, positions);
        fx.scalar_i32("rotary_dim", kRopeHeadDim);
        fx.scalar_f32("scale", 1.0f);
        fx.f32("q_out", download_bf16(dq, q_n, name));
        ck(cudaFree(dq), name);
        ck(cudaFree(dp), name);
    }

    // -- neox_mb_partial ---------------------------------------------------
    {
        const char* name = "neox_mb_partial";
        auto q_in = rope_input(name);

        // Gather the leading `rotary` channels of every head into a packed
        // [N, n_head, rotary] buffer, rotate THAT at head_dim = rotary, and
        // scatter the result back over the same channels.
        const size_t rot_n =
            static_cast<size_t>(kRopeTokens) * kRopeHeads * kRopeRotary;
        std::vector<float> rot(rot_n);
        for (int m = 0; m < kRopeTokens; ++m) {
            for (int h = 0; h < kRopeHeads; ++h) {
                for (int c = 0; c < kRopeRotary; ++c) {
                    rot[(static_cast<size_t>(m) * kRopeHeads + h) * kRopeRotary + c] =
                        q_in[(static_cast<size_t>(m) * kRopeHeads + h) * kRopeHeadDim +
                             c];
                }
            }
        }
        bf16* dq = upload_bf16(rot, name);
        int32_t* dp = upload(positions, name);
        const int half = kRopeRotary / 2;
        const int pairs = cache_pairs(half);
        const int per_block = heads_per_block(half);
        const dim3 grid(kRopeTokens,
                        static_cast<unsigned>((kRopeHeads + per_block - 1) / per_block),
                        1);
        pie::rope::rotate<false, false>
            <<<grid, kRopeBlock, static_cast<unsigned>(pairs) * 2 * 4>>>(
                dq, dq, dp, kRopeHeads, 0, kRopeRotary, kRopeTheta, false, pairs,
                per_block, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, 0, 0);
        sync(name);
        auto rotated = download_bf16(dq, rot_n, name);

        std::vector<float> q_out = q_in;
        for (int m = 0; m < kRopeTokens; ++m) {
            for (int h = 0; h < kRopeHeads; ++h) {
                for (int c = 0; c < kRopeRotary; ++c) {
                    q_out[(static_cast<size_t>(m) * kRopeHeads + h) * kRopeHeadDim +
                          c] =
                        rotated[(static_cast<size_t>(m) * kRopeHeads + h) *
                                    kRopeRotary +
                                c];
                }
            }
        }
        fx.open_case(name);
        rope_common(fx, q_in, positions);
        fx.scalar_i32("rotary_dim", kRopeRotary);
        fx.scalar_f32("scale", 1.0f);
        fx.f32("q_out", q_out);
        ck(cudaFree(dq), name);
        ck(cudaFree(dp), name);
    }

    // -- neox_prop_mb ------------------------------------------------------
    {
        const char* name = "neox_prop_mb";
        auto q_in = rope_input(name);
        bf16* dq = upload_bf16(q_in, name);
        int32_t* dp = upload(positions, name);
        pie::rope::rotate_partial<bf16><<<kRopeTokens, kRopeBlock>>>(
            dq, dq, dp, 0, kRopeHeads, 0, kRopeHeadDim, kRopeRotary, kRopeTheta);
        sync(name);
        fx.open_case(name);
        rope_common(fx, q_in, positions);
        fx.scalar_i32("rotary_dim", kRopeRotary);
        fx.scalar_f32("scale", 1.0f);
        fx.f32("q_out", download_bf16(dq, q_n, name));
        ck(cudaFree(dq), name);
        ck(cudaFree(dp), name);
    }

    // -- neox_last_mb, both pairings ---------------------------------------
    for (int interleaved = 0; interleaved <= 1; ++interleaved) {
        const char* name = interleaved ? "neox_last_mb_interleaved" : "neox_last_mb";
        auto q_in = rope_input(name);
        bf16* dq = upload_bf16(q_in, name);
        int32_t* dp = upload(positions, name);
        // `yarn_factor = 1` is `rope.rs::Yarn::NONE.factor`, and the kernel
        // takes its YaRN branch only above 1, so the ramp bounds are unread.
        pie::rope::rotate_partial_last<<<kRopeTokens, kRopeBlock>>>(
            dq, dq, dp, kRopeHeads, 0, kRopeHeadDim, kRopeRotary, kRopeTheta, false,
            interleaved != 0, 1.0f, 0.0f, 0.0f);
        sync(name);
        fx.open_case(name);
        rope_common(fx, q_in, positions);
        fx.scalar_i32("rotary_dim", kRopeRotary);
        fx.scalar_i32("interleaved", interleaved);
        fx.f32("q_out", download_bf16(dq, q_n, name));
        ck(cudaFree(dq), name);
        ck(cudaFree(dp), name);
    }

    // -- neox_yarn_mb, both pairings ---------------------------------------
    //
    // `low_dim` and `high_dim` are computed on the HOST by
    // `prelude/rope.cuh::yarn_original_ramp_bounds` and arrive as arguments;
    // `neox.metal` binds them the same way, so they are recorded rather than
    // re-derived. `mscale` is YaRN's attention temperature, 0.1*ln(factor)+1.
    for (int interleaved = 0; interleaved <= 1; ++interleaved) {
        const char* name = interleaved ? "neox_yarn_mb_interleaved" : "neox_yarn_mb";
        const float factor = 8.0f;
        const float beta_fast = 32.0f;
        const float beta_slow = 1.0f;
        const int original_max_position = 4096;
        const float mscale = 0.1f * std::log(factor) + 1.0f;
        float low_dim = 0.0f, high_dim = 0.0f;
        pie::yarn_original_ramp_bounds(kRopeHeadDim, kRopeTheta, beta_fast, beta_slow,
                                       original_max_position, low_dim, high_dim);

        auto q_in = rope_input(name);
        bf16* dq = upload_bf16(q_in, name);
        int32_t* dp = upload(positions, name);
        const int half = kRopeHeadDim / 2;
        const int pairs = cache_pairs(half);
        const int per_block = heads_per_block(half);
        const dim3 grid(kRopeTokens,
                        static_cast<unsigned>((kRopeHeads + per_block - 1) / per_block),
                        1);
        pie::rope::rotate_yarn_original
            <<<grid, kRopeBlock, static_cast<unsigned>(pairs) * 8>>>(
                dq, dq, dp, kRopeHeads, 0, kRopeHeadDim, kRopeTheta, factor, low_dim,
                high_dim, mscale, interleaved != 0, per_block, pairs);
        sync(name);
        fx.open_case(name);
        rope_common(fx, q_in, positions);
        fx.scalar_f32("factor", factor);
        fx.scalar_f32("low_dim", low_dim);
        fx.scalar_f32("high_dim", high_dim);
        fx.scalar_f32("mscale", mscale);
        fx.scalar_i32("interleaved", interleaved);
        fx.f32("q_out", download_bf16(dq, q_n, name));
        ck(cudaFree(dq), name);
        ck(cudaFree(dp), name);
    }
}

//===----------------------------------------------------------------------===//
// 4. The short causal convolution.
//===----------------------------------------------------------------------===//
//
// `causal_conv1d.metal`'s two entry points against the two CUDA kernels the
// Rust claim bodies for the same two points name -- `ssm.rs::causal_conv1d`
// fires `causal_conv1d_update_batched<bf16>` and
// `ssm.rs::causal_conv1d_chunked` fires `causal_conv1d_prefill_batched<bf16>`
// below eight requests, which is the arm three requests take.
//
//   causal_conv1d          <- ssm::causal_conv1d_update_batched<bf16>
//   causal_conv1d_chunked  <- ssm::causal_conv1d_prefill_batched<bf16>
//
// The slabs DO line up. Both are `[num_slots, K, C]` with `C` the fast axis
// and the oldest tap at row 0: CUDA reads `state_base + slot *
// slot_stride_elems` then `[k * C + c]` with `slot_stride_elems = K * C`, and
// `causal_conv1d.metal` reads `slots[r] * taps * chans + k * chans + col`.
// Same rectangle, same order, same slot stride. Three things do NOT line up,
// and each is handled here rather than papered over:
//
// 1. THE SLAB'S ELEMENT. CUDA holds the conv state in `T` -- bf16, since both
//    points refuse anything else -- and Metal holds it in `float`. Every
//    value this fixture puts in a slab is bf16-exact, so the two planes hold
//    the same number and the two kernels read the same tap. It also stays
//    true of the OUTPUT slab: everything either kernel writes there is a
//    value it just read from a bf16 plane or from `x`, so both sides' written
//    rows are bf16-exact too.
//
// 2. IN PLACE VS A SECOND PLANE. CUDA shifts the slab where it stands; Metal
//    reads `conv_state` and writes `new_conv_state`, because a Metal dispatch
//    makes no promise about the order two threadgroups touch a row. The two
//    agree because each thread owns one (request, channel) COLUMN for the
//    whole kernel and the in-place shift's reads always run ahead of its
//    writes -- `state[k] = state[k+1]` ascending in the step, and
//    `state[s] = state[Nr + s]` with `Nr > 0` in the chunk. So the fixture
//    records `conv_state` as launched and `new_conv_state` as the CUDA slab
//    came back.
//
//    THE METAL SIDE MUST SEED `new_conv_state` WITH `conv_state` BEFORE THE
//    DISPATCH. That is not an accommodation, it is the invariant
//    `Pool::carry_forward` maintains and the one `causal_conv1d_chunked`'s
//    early return on an empty window depends on -- `conv_chunked_empty` below
//    is the case that pins it.
//
// 3. THE BIAS. `causal_conv1d_update_batched` and
//    `causal_conv1d_prefill_batched` take a nullable `bias`; the Metal
//    kernels have no such operand, and both Rust claim bodies pass
//    `MaybeConst::<bf16>::none()`. Driven with a null bias, which is what
//    every caller of these two points does.
//
// One more spelling difference, of the kind §2 of this file's header lists:
// CUDA's `silu_f` is `z / (1 + __expf(-z))` and Metal's
// `causal_conv1d_silu` is `z / (1 + metal::exp(-z))` -- the fast intrinsic
// against the precise call, below the bf16 floor. And CUDA accumulates
// `acc += wv * xv` where Metal accumulates `acc += tap * w`, which is the
// same product in the other order and identical in IEEE.
//
// `C = 20` channels and `K = 4` taps (Qwen3.5's conv width) over `R = 3`
// requests in slots `{2, 0, 1}` of a three-slot pool -- a permutation, so a
// kernel that indexed the slab by the request rather than by the seat reads
// the wrong column. 20 is not a multiple of 8, so a channel axis someone
// vectorised runs off the end of a row.

constexpr int kConvChannels = 20;
constexpr int kConvWidth = 4;
constexpr int kConvRequests = 3;
constexpr int kConvSlots = 3;

/// One bf16-exact draw per (case, operand). The operand is part of the seed so
/// `x`, `weight` and the slab are three independent streams rather than three
/// slices of one.
std::vector<float> conv_draw(const char* case_name, const char* operand, size_t n) {
    Rng rng((std::string(case_name) + "/" + operand).c_str());
    std::vector<float> v(n);
    for (float& e : v) e = rng.sym_bf16(1.0f);
    return v;
}

std::vector<float> conv_weight(const char* case_name) {
    return conv_draw(case_name, "weight",
                     static_cast<size_t>(kConvChannels) * kConvWidth);
}

std::vector<float> conv_slab(const char* case_name) {
    return conv_draw(case_name, "conv_state",
                     static_cast<size_t>(kConvSlots) * kConvWidth * kConvChannels);
}

std::vector<float> conv_tokens(const char* case_name, int n) {
    return conv_draw(case_name, "x", static_cast<size_t>(n) * kConvChannels);
}

/// `slot_ids` is the CUDA operand, one seat per REQUEST;
/// `causal_conv1d.metal` binds one seat per TOKEN and reads the window's
/// first. Both are written out so neither side has to derive the other.
const std::vector<int32_t>& conv_slot_ids() {
    static const std::vector<int32_t> ids{2, 0, 1};
    return ids;
}

void emit_ssm(Fixture& fx) {
    header(fx, "Reference vectors for kernels-metal/kernels/ssm/causal_conv1d.metal.");
    fx.comment("C = %d channels, K = %d taps, R = %d requests in slots {2, 0, 1}"
               " of a %d-slot pool.", kConvChannels, kConvWidth, kConvRequests,
               kConvSlots);
    fx.blank_comment();
    fx.comment("conv_update        <- pie::ssm::causal_conv1d_update_batched<bf16>"
               "(x, weight, bias=null, state_base, slot_ids,"
               " slot_stride_elems=%d, y, R=%d, C=%d, K=%d)",
               kConvWidth * kConvChannels, kConvRequests, kConvChannels, kConvWidth);
    fx.comment("                      grid [ceil(C/256), R] x 256"
               " -- ssm.rs::causal_conv1d");
    fx.comment("conv_chunked       <- pie::ssm::causal_conv1d_prefill_batched<bf16>"
               "(x, weight, bias=null, y, state_out_base, slot_ids, qo_indptr,"
               " slot_stride_elems=%d, C=%d, K=%d, write_state=true, mask=null,"
               " commit_len=null)", kConvWidth * kConvChannels, kConvChannels,
               kConvWidth);
    fx.comment("                      grid [C, R] x 64 -- ssm.rs::causal_conv1d_chunked"
               " below its R >= 8 channel-tile arm");
    fx.comment("                      qo_indptr = {0, 3, 9, 14}: one window"
               " SHORTER than K, so the trailing-state write reads taps back"
               " off the slab");
    fx.comment("conv_chunked_empty <- the same kernel at qo_indptr = {0, 0, 4, 9}:"
               " request 0's window is empty and both sides return early");
    fx.blank_comment();
    fx.comment("THE SLAB IS [num_slots, K, C], oldest tap at row 0, C the fast"
               " axis -- the same rectangle on");
    fx.comment("both sides. CUDA holds it in bf16 and causal_conv1d.metal in"
               " float, so every value here is");
    fx.comment("bf16-exact and the two planes hold one number. CUDA shifts the"
               " slab in place and Metal writes");
    fx.comment("a second plane, so `conv_state` is the slab as launched and"
               " `new_conv_state` is the slab as it");
    fx.comment("came back. THE METAL SIDE MUST SEED new_conv_state WITH"
               " conv_state BEFORE DISPATCHING -- that is");
    fx.comment("what Pool::carry_forward leaves behind, and conv_chunked_empty"
               " is the case that depends on it.");
    fx.comment("`bias` is null: the Metal kernels have no such operand and both"
               " Rust claim bodies pass none.");
    fx.blank_comment();

    const size_t slab_n =
        static_cast<size_t>(kConvSlots) * kConvWidth * kConvChannels;
    const int32_t slot_stride = kConvWidth * kConvChannels;
    const auto slot_ids = conv_slot_ids();

    // -- conv_update -------------------------------------------------------
    {
        const char* name = "conv_update";
        auto weight = conv_weight(name);
        auto state = conv_slab(name);
        auto x = conv_tokens(name, kConvRequests);
        const size_t y_n = static_cast<size_t>(kConvRequests) * kConvChannels;

        bf16* dx = upload_bf16(x, name);
        bf16* dw = upload_bf16(weight, name);
        bf16* ds = upload_bf16(state, name);
        bf16* dy = alloc<bf16>(y_n, name);
        int32_t* dslot = upload(slot_ids, name);

        pie::ssm::causal_conv1d_update_batched<bf16>
            <<<dim3(static_cast<unsigned>((kConvChannels + 255) / 256),
                    kConvRequests, 1),
               256>>>(dx, dw, nullptr, ds, dslot, slot_stride, dy, kConvRequests,
                      kConvChannels, kConvWidth);
        sync(name);

        fx.open_case(name);
        fx.f32("x", x);
        fx.f32("weight", weight);
        fx.f32("conv_state", state);
        fx.i32("slot_ids", slot_ids);
        fx.scalar_i32("r", kConvRequests);
        fx.scalar_i32("c", kConvChannels);
        fx.scalar_i32("k", kConvWidth);
        fx.scalar_i32("slot_stride_elems", slot_stride);
        fx.f32("y", download_bf16(dy, y_n, name));
        fx.f32("new_conv_state", download_bf16(ds, slab_n, name));
        ck(cudaFree(dx), name);
        ck(cudaFree(dw), name);
        ck(cudaFree(ds), name);
        ck(cudaFree(dy), name);
        ck(cudaFree(dslot), name);
    }

    // -- conv_chunked, one full window set and one with an empty request ---
    struct Chunk {
        const char* name;
        std::vector<int32_t> indptr;
    };
    const Chunk chunks[] = {{"conv_chunked", {0, 3, 9, 14}},
                            {"conv_chunked_empty", {0, 0, 4, 9}}};
    for (const Chunk& chunk : chunks) {
        const char* name = chunk.name;
        const int n_tokens = chunk.indptr.back();
        auto weight = conv_weight(name);
        auto state = conv_slab(name);
        auto x = conv_tokens(name, n_tokens);
        const size_t y_n = static_cast<size_t>(n_tokens) * kConvChannels;

        // The per-TOKEN seat table `causal_conv1d.metal` binds. Constant
        // across a request's window, which is what lets it read `slots[begin]`.
        std::vector<int32_t> slots(static_cast<size_t>(n_tokens), 0);
        for (int r = 0; r < kConvRequests; ++r) {
            for (int t = chunk.indptr[r]; t < chunk.indptr[r + 1]; ++t) {
                slots[static_cast<size_t>(t)] = slot_ids[r];
            }
        }

        bf16* dx = upload_bf16(x, name);
        bf16* dw = upload_bf16(weight, name);
        bf16* ds = upload_bf16(state, name);
        bf16* dy = alloc<bf16>(y_n, name);
        int32_t* dslot = upload(slot_ids, name);
        std::vector<uint32_t> indptr_u32(chunk.indptr.begin(), chunk.indptr.end());
        uint32_t* dindptr = upload(indptr_u32, name);

        pie::ssm::causal_conv1d_prefill_batched<bf16>
            <<<dim3(kConvChannels, kConvRequests, 1), 64>>>(
                dx, dw, nullptr, dy, ds, dslot, dindptr, slot_stride, kConvChannels,
                kConvWidth, true, nullptr, nullptr);
        sync(name);

        fx.open_case(name);
        fx.f32("x", x);
        fx.i32("qo_indptr", chunk.indptr);
        fx.f32("weight", weight);
        fx.f32("conv_state", state);
        fx.i32("slot_ids", slot_ids);
        fx.i32("slots", slots);
        fx.scalar_i32("c", kConvChannels);
        fx.scalar_i32("k", kConvWidth);
        fx.scalar_i32("slot_stride_elems", slot_stride);
        fx.f32("y", download_bf16(dy, y_n, name));
        fx.f32("new_conv_state", download_bf16(ds, slab_n, name));
        ck(cudaFree(dx), name);
        ck(cudaFree(dw), name);
        ck(cudaFree(ds), name);
        ck(cudaFree(dy), name);
        ck(cudaFree(dslot), name);
        ck(cudaFree(dindptr), name);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    ck(cudaSetDevice(0), "cudaSetDevice");

    {
        Fixture fx(dir, "mlp_packed.txt");
        emit_mlp(fx);
        std::fprintf(stderr, "wrote %s\n", fx.path().c_str());
    }
    {
        Fixture fx(dir, "moe_router.txt");
        emit_router(fx);
        std::fprintf(stderr, "wrote %s\n", fx.path().c_str());
    }
    {
        Fixture fx(dir, "rope_neox.txt");
        emit_rope(fx);
        std::fprintf(stderr, "wrote %s\n", fx.path().c_str());
    }
    {
        Fixture fx(dir, "ssm_conv.txt");
        emit_ssm(fx);
        std::fprintf(stderr, "wrote %s\n", fx.path().c_str());
    }
    return 0;
}
