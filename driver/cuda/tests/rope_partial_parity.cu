// Guards for the two partial-rotary launchers. They test DIFFERENT PROPERTIES
// against DIFFERENT REFERENCES, and confusing the two is how this file went
// wrong once already:
//
//   * `launch_rope_partial_bf16` (default)  -- STRUCTURE: which channels
//     rotate, which pass through, what the frequency denominator is. Checked
//     against the HF definition in fp64, with a tolerance, because the
//     question is "does it rotate the right dims by the right angle" and the
//     historical bugs were whole-number-sized.
//
//   * `launch_rope_partial_vllm_table_bf16` (PIE_DEBUG_ROPE_VLLM_TABLE=1) -- BITS:
//     does it reproduce vLLM's `cos_sin_cache` exactly. Checked against that
//     table itself, with NO tolerance and NO transcendental evaluated here.
//
// REPRODUCING THIS WITHOUT BUILDING THE WHOLE DRIVER. `driver/cuda/CMakeLists.txt`
// sets no global CUDA flags and no numeric ones anywhere: no `-use_fast_math`,
// no `--fmad` override, no `-prec-div`/`-prec-sqrt`/`-ftz`. The only per-target
// CUDA options are `--extended-lambda --expt-relaxed-constexpr` plus warning
// flags, none of which affect arithmetic. So compiling just `kernels/rope.cu`
// and this file directly --
//
//   nvcc -std=c++17 -O2 -I<src> -I<tests> -gencode arch=compute_89,code=sm_89 \
//        -c kernels/rope.cu tests/rope_partial_parity.cu
//
// -- is numerically identical to the real target, and avoids building 77
// sources to check a rounding question. It also means `sincosf` is the accurate
// library routine and not silently mapped onto the `__sincosf` intrinsic, which
// a fast-math flag would do and which would make this whole file inert.
//
// The fp64 reference below is legitimate for the first and ILLEGITIMATE for
// the second. vLLM's table is a deliberately rounded bf16 value, so a kernel
// that is more accurate than it is further from parity, not closer. Do not
// carry the fp64 reference across into the vLLM checks; see the long note
// above them for what happened when an earlier draft did.
//
// ── The default path: structure ────────────────────────────────────────────
//
// WHY THIS EXISTS. Partial rotary rotates only the first `rotary_dim`
// channels of each head and leaves `[rotary_dim, head_dim)` untouched. The
// CUDA kernel had BOTH halves of that contract wrong — it used `head_dim` as
// the frequency denominator and `head_dim/2` as the pair offset — and the
// error survived because nothing tested this launcher. It survived twice: a
// comment above the kernel asserted the incorrect form was right and called
// the correct form "the previous draft [that] got it wrong", so the code had
// been changed INTO the bug and documented confidently.
//
// For Qwen3.6-27B (head_dim 256, partial_rotary_factor 0.25, rotary_dim 64)
// the consequences were:
//
//   * dims 32..63   left UNROTATED  — they are the second half of each pair
//   * dims 128..159 OVERWRITTEN     — they are pass-through
//   * frequency denominator 4x too large, angle off by up to 1.2e5 at j=31
//
// Observably that produced a systematic ~2.2 nat logit disagreement against
// vLLM on the same checkpoint, which is exactly zero at relative distance 0
// and grows with |m-n| — so a decode agreed for its first several tokens and
// then diverged, fluently, in a way that read as "the model is worse".
//
// The reference below is the HF definition, written out longhand rather than
// factored, so that a future edit cannot make reference and kernel wrong in
// the same direction. THE POINT OF THIS TEST IS THE THREE ASSERTIONS BELOW,
// not the tolerance:
//
//   1. channels [0, rotary_dim) rotate with pair offset rotary_dim/2
//   2. channels [rotary_dim, head_dim) are BYTE-IDENTICAL to the input
//   3. the frequency denominator is rotary_dim, not head_dim
//
// Assertion 2 is the cheap one that would have caught this on day one: the
// buggy kernel wrote to dims 128..159, and no tolerance is needed to see it.
//
// The kernel accumulates in fp32 and stores bf16, so rotated channels are
// compared with a tolerance. bf16 carries ~8 mantissa bits; a correct
// rotation of unit-scale values lands well inside 5e-3, while either of the
// two historical bugs lands orders of magnitude outside it.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/rope.hpp"
#include "rope_vllm_cos_sin_golden.hpp"

namespace kernels = pie_cuda_driver::kernels;

namespace {

int g_failures = 0;

std::uint16_t float_to_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(rounded >> 16);
}

float bf16_to_float(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// HF partial rotary, longhand. `rotate_half` is applied to the slice
// x[..., :rotary_dim], so the pair partner is rotary_dim/2 away and the
// frequency denominator is rotary_dim.
void reference_rope_partial(
    std::vector<float>& head, int head_dim, int rotary_dim, int pos, float theta) {
    const int angles = rotary_dim / 2;
    const std::vector<float> in = head;
    for (int j = 0; j < angles; ++j) {
        const double freq = std::pow(static_cast<double>(theta),
                                     -2.0 * static_cast<double>(j) /
                                         static_cast<double>(rotary_dim));
        const double ang = static_cast<double>(pos) * freq;
        const double c = std::cos(ang), s = std::sin(ang);
        const double a = in[j], b = in[j + angles];
        head[j] = static_cast<float>(a * c - b * s);
        head[j + angles] = static_cast<float>(b * c + a * s);
    }
    // [rotary_dim, head_dim) deliberately untouched.
}

// ── vLLM cos/sin-table parity (the `..._vllm_table` launcher) ──────────────
//
// WHAT THE REFERENCE IS. vLLM's `cos_sin_cache`, embedded verbatim in
// rope_vllm_cos_sin_golden.hpp. Not a formula evaluated here, not a
// recomputation, and deliberately NOT fp64 truth.
//
// WHY fp64 TRUTH IS THE WRONG REFERENCE, and must not be reintroduced. vLLM
// builds that cache once at init in fp32 and then does `cache.to(dtype)`,
// which rounds the whole table to bf16 and stores it; the `triton_mrope` path
// indexes those bf16 values with no fp32 cast and casts q/k down to match. The
// reference value is a rounded, lossy number BY CONSTRUCTION. An
// implementation that computed cos/sin more accurately would move AWAY from
// it. On this path a more accurate kernel is a WORSE kernel. Any assertion
// phrased as "how close is this to the true cosine" measures the wrong thing
// and will point the next reader in the wrong direction -- an earlier draft of
// this file asserted exactly that (it demanded the table path beat the default
// path against an fp64 reference) and had to be withdrawn, because the change
// this file guards cannot satisfy it and should not.
//
// The property under test is reproduction of the reference's BITS. Nothing in
// the vLLM checks below evaluates a transcendental at all.
//
// HOW THE TABLE IS READ BACK. The kernel exposes no table, so feed it a head
// whose rotated slice is a = 1.0, b = 0.0:
//
//   out[j]          = (1*cos) - (0*sin) = cos[j]
//   out[j+angles]   = (0*cos) + (1*sin) = sin[j]
//
// Every operand is already bf16 and every product is exact, so the output IS
// the table row, bit for bit, with no tolerance anywhere. It reads out the
// fp32-rotate path just as exactly as the bf16 one, so the same probe compares
// both against the same golden.
//
// TOLERANCE: NONE. Zero mismatches, no allowance for rounding-boundary ties.
// One bf16 ulp in cos can flip a token whose logits sit near a decision
// boundary, so an allowance would quietly bless a token flip. If this
// assertion goes red, the parity defect is real and small: report it with the
// positions and lanes below. Do not widen the bar to reach green.
constexpr int kGoldenHeadDim = 256;

// One head laid out so the rotated slice reads back as the cos/sin row.
void fill_table_probe_head(std::uint16_t* head, int head_dim, int rotary_dim) {
    const int angles = rotary_dim / 2;
    for (int d = 0; d < head_dim; ++d) head[d] = 0x0000;  // +0.0f
    for (int j = 0; j < angles; ++j) head[j] = 0x3f80;    // 1.0f
}

enum class Path { Default, VllmTable };

// Recovers [kRows][kRotaryDim] of whichever path is asked for.
std::vector<std::uint16_t> read_back_table(Path path) {
    namespace g = pie_rope_golden;
    const int tokens = g::kRows;
    const int total = tokens * kGoldenHeadDim;

    std::vector<std::uint16_t> buf(total);
    std::vector<std::int32_t> positions(tokens);
    for (int n = 0; n < tokens; ++n) {
        positions[n] = g::kCosSinCache[n].pos;
        fill_table_probe_head(buf.data() + n * kGoldenHeadDim, kGoldenHeadDim,
                              g::kRotaryDim);
    }

    void *dq = nullptr, *dk = nullptr;  // distinct: both are `__restrict__`
    std::int32_t* dpos = nullptr;
    cudaMalloc(&dq, total * sizeof(std::uint16_t));
    cudaMalloc(&dk, sizeof(std::uint16_t));
    cudaMalloc(&dpos, tokens * sizeof(std::int32_t));
    cudaMemcpy(dq, buf.data(), total * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, positions.data(), tokens * sizeof(std::int32_t),
               cudaMemcpyHostToDevice);

    if (path == Path::VllmTable) {
        kernels::launch_rope_partial_vllm_table_bf16(
            dq, dk, dpos, tokens, /*num_q_heads=*/1, /*num_kv_heads=*/0,
            kGoldenHeadDim, g::kRotaryDim, g::kTheta, /*stream=*/nullptr);
    } else {
        kernels::launch_rope_partial_bf16(
            dq, dk, dpos, tokens, /*num_q_heads=*/1, /*num_kv_heads=*/0,
            kGoldenHeadDim, g::kRotaryDim, g::kTheta, /*stream=*/nullptr);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(buf.data(), dq, total * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(dq); cudaFree(dk); cudaFree(dpos);

    std::vector<std::uint16_t> table(tokens * g::kRotaryDim);
    for (int n = 0; n < tokens; ++n)
        for (int e = 0; e < g::kRotaryDim; ++e)
            table[n * g::kRotaryDim + e] = buf[n * kGoldenHeadDim + e];
    return table;
}

bool listed(const pie_rope_golden::Entry* list, int count, int row, int entry) {
    for (int i = 0; i < count; ++i)
        if (list[i].row == row && list[i].entry == entry) return true;
    return false;
}

struct TableDiff {
    int total = 0;         // every mismatch; the bar is that this is zero
    int at_override = 0;   // at an entry carrying measured reference bits
    int elsewhere = 0;     // somewhere the two constructions were thought to agree
    int guards_broken = 0;
};

// Classifies mismatches but does NOT excuse any of them. The parity check
// requires `total == 0`; the split is a diagnostic.
//
// `at_override` means the driver disagreed at an entry where the reference is
// KNOWN to differ from a correctly-rounded construction -- most likely the
// driver has drifted toward correct rounding, which is the direction measured
// to be worse. `elsewhere` means it disagreed where the two constructions were
// believed identical, which is either a real regression or a fixture derived
// against the wrong reference build.
TableDiff diff_table(const std::vector<std::uint16_t>& got, bool report) {
    namespace g = pie_rope_golden;
    TableDiff d;
    int printed = 0;
    for (int n = 0; n < g::kRows; ++n) {
        for (int e = 0; e < g::kRotaryDim; ++e) {
            const std::uint16_t want = g::kCosSinCache[n].entry[e];
            const std::uint16_t have = got[n * g::kRotaryDim + e];
            const bool guard =
                listed(g::kRegressionGuards, g::kRegressionGuardCount, n, e);
            if (have == want) continue;
            ++d.total;
            const bool known = listed(g::kReferenceOverrides,
                                      g::kReferenceOverrideCount, n, e);
            if (known) ++d.at_override; else ++d.elsewhere;
            if (guard) ++d.guards_broken;
            if (!report || printed >= 12) continue;
            ++printed;
            std::printf("      %-10s pos=%-6d %s[%2d]  got=0x%04x want=0x%04x  "
                        "delta=%+d bf16 ulp%s\n",
                        known ? "[at-override]" : "[elsewhere]",
                        g::kCosSinCache[n].pos,
                        e < g::kAngles ? "cos" : "sin",
                        e < g::kAngles ? e : e - g::kAngles, have, want,
                        static_cast<int>(have) - static_cast<int>(want),
                        guard ? "   <== REGRESSION GUARD" : "");
        }
    }
    if (report && d.total > printed)
        std::printf("      ... and %d more\n", d.total - printed);
    return d;
}

struct Case {
    const char* label;
    int head_dim;
    int rotary_dim;
    int num_q_heads;
    int num_kv_heads;
    int num_tokens;
    float theta;
};

void run_case(const Case& c) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    const int total_q = c.num_tokens * c.num_q_heads * c.head_dim;
    const int total_k = c.num_tokens * c.num_kv_heads * c.head_dim;
    std::vector<std::uint16_t> q(total_q), k(total_k);
    std::vector<float> q_ref(total_q), k_ref(total_k);
    for (int i = 0; i < total_q; ++i) {
        const float v = dist(rng);
        q[i] = float_to_bf16(v);
        q_ref[i] = bf16_to_float(q[i]);
    }
    for (int i = 0; i < total_k; ++i) {
        const float v = dist(rng);
        k[i] = float_to_bf16(v);
        k_ref[i] = bf16_to_float(k[i]);
    }
    const std::vector<std::uint16_t> q_in = q, k_in = k;

    std::vector<std::int32_t> positions(c.num_tokens);
    for (int n = 0; n < c.num_tokens; ++n) positions[n] = 3 + 5 * n;

    // CPU reference, per head.
    for (int n = 0; n < c.num_tokens; ++n) {
        for (int h = 0; h < c.num_q_heads; ++h) {
            std::vector<float> head(q_ref.begin() + (n * c.num_q_heads + h) * c.head_dim,
                                    q_ref.begin() + (n * c.num_q_heads + h + 1) * c.head_dim);
            reference_rope_partial(head, c.head_dim, c.rotary_dim, positions[n], c.theta);
            std::copy(head.begin(), head.end(),
                      q_ref.begin() + (n * c.num_q_heads + h) * c.head_dim);
        }
        for (int h = 0; h < c.num_kv_heads; ++h) {
            std::vector<float> head(k_ref.begin() + (n * c.num_kv_heads + h) * c.head_dim,
                                    k_ref.begin() + (n * c.num_kv_heads + h + 1) * c.head_dim);
            reference_rope_partial(head, c.head_dim, c.rotary_dim, positions[n], c.theta);
            std::copy(head.begin(), head.end(),
                      k_ref.begin() + (n * c.num_kv_heads + h) * c.head_dim);
        }
    }

    void *dq = nullptr, *dk = nullptr;
    std::int32_t* dpos = nullptr;
    cudaMalloc(&dq, total_q * sizeof(std::uint16_t));
    cudaMalloc(&dk, total_k * sizeof(std::uint16_t));
    cudaMalloc(&dpos, c.num_tokens * sizeof(std::int32_t));
    cudaMemcpy(dq, q.data(), total_q * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), total_k * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, positions.data(), c.num_tokens * sizeof(std::int32_t),
               cudaMemcpyHostToDevice);

    kernels::launch_rope_partial_bf16(dq, dk, dpos, c.num_tokens, c.num_q_heads,
                                      c.num_kv_heads, c.head_dim, c.rotary_dim,
                                      c.theta, /*stream=*/nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(q.data(), dq, total_q * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(k.data(), dk, total_k * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(dq); cudaFree(dk); cudaFree(dpos);

    const double tol = 5e-3;
    double max_rot = 0.0;
    int passthrough_violations = 0;

    auto check = [&](const std::vector<std::uint16_t>& got,
                     const std::vector<std::uint16_t>& in,
                     const std::vector<float>& ref, int heads) {
        for (int n = 0; n < c.num_tokens; ++n) {
            for (int h = 0; h < heads; ++h) {
                const int base = (n * heads + h) * c.head_dim;
                for (int d = 0; d < c.head_dim; ++d) {
                    if (d < c.rotary_dim) {
                        max_rot = std::fmax(max_rot,
                            std::fabs(bf16_to_float(got[base + d]) - ref[base + d]));
                    } else if (got[base + d] != in[base + d]) {
                        // Assertion 2: pass-through channels must be BIT-identical.
                        // The historical bug wrote dims 128..159 here.
                        ++passthrough_violations;
                    }
                }
            }
        }
    };
    check(q, q_in, q_ref, c.num_q_heads);
    check(k, k_in, k_ref, c.num_kv_heads);

    const bool ok = (max_rot <= tol) && (passthrough_violations == 0);
    if (!ok) ++g_failures;
    std::printf("[%s] %-28s head_dim=%d rotary_dim=%d  max_rot=%.3e  passthrough_violations=%d\n",
                ok ? "ok" : "FAIL", c.label, c.head_dim, c.rotary_dim, max_rot,
                passthrough_violations);
}

// ── Is the fixture capable of catching anything? ───────────────────────────
//
// This slice has been blind TWICE, the same way both times.
//
//   v1 sampled 13 positions, and not one of its 832 entries lay within even
//   3 fp32 ulp of a bf16 rounding midpoint. A bf16 table can only disagree
//   near a midpoint, so it could not fail. It passed with zero tolerance and
//   proved nothing.
//
//   v2 added the four positions where DEVICE-side trig differed. Moving the
//   table build onto the host fixed all four and moved the residual to
//   entirely different positions -- so a slice holding only the old four was
//   blind again, one iteration later.
//
// The interesting positions are a property of the current implementation pair,
// not a constant. Two things are asserted here, before any GPU work, because a
// fixture that cannot fail is itself the defect:
//
//   1. the slice contains boundary-adjacent entries at all
//   2. it contains at least one entry MEASURED to differ, whenever the survey
//      found one -- if the survey found none, there is nothing to require
void run_golden_slice_is_not_blind() {
    namespace g = pie_rope_golden;
    constexpr int kFloor = 8;

    const bool has_boundary_entries = g::kEntriesWithinOneUlp >= kFloor;
    // Without at least one overridden entry, every bit in this fixture came
    // from our own construction and the parity check compares us against
    // ourselves. That is not a weak test, it is a circular one, and it is what
    // the fixture silently was until the divergent entries were measured.
    const bool is_reference_derived = g::kReferenceOverrideCount >= 1;
    const bool ok = has_boundary_entries && is_reference_derived;
    if (!ok) ++g_failures;

    std::printf("[%s] %-28s rows=%d entries=%d  boundary_adjacent<=1ulp=%d "
                "(>=%d)  reference_overrides=%d (>=1)  guards=%d\n",
                ok ? "ok" : "FAIL", "golden slice has teeth", g::kRows,
                g::kRows * g::kRotaryDim, g::kEntriesWithinOneUlp, kFloor,
                g::kReferenceOverrideCount, g::kRegressionGuardCount);
    // Printed every run. The reference's trig is a different LIBRARY on
    // different hosts -- Intel MKL VML on an x86 torch build, SLEEF only where
    // USE_MKL=OFF or __APPLE__ -- and their divergences sit at different
    // entries. A slice derived against the wrong one cannot fail. Loud beats
    // subtle.
    std::printf("      reference ISA: %s  (residual surveyed to position %d)\n",
                g::kReferenceIsa, g::kResidualSurveyMax);
}

// ── The parity assertion: knob-on must reproduce vLLM's table exactly ──────
void run_vllm_golden_table() {
    namespace g = pie_rope_golden;
    const std::vector<std::uint16_t> got = read_back_table(Path::VllmTable);
    const TableDiff d = diff_table(got, /*report=*/true);

    // The host-built table must actually have covered every position used. If
    // it ran short the kernel silently degrades to device trig, which is the
    // behaviour this path exists to replace -- so a green above would mean
    // nothing.
    const int capacity =
        kernels::rope_vllm_table_capacity_for(g::kTheta, g::kRotaryDim);
    const unsigned int oob =
        kernels::rope_vllm_table_oob_blocks(g::kTheta, g::kRotaryDim);
    int max_pos = 0;
    for (int n = 0; n < g::kRows; ++n)
        max_pos = std::max(max_pos, g::kCosSinCache[n].pos);

    // Zero is the bar. The classification above is a label, not an allowance:
    // widening this to `d.elsewhere == 0` would bless a token flip, since one
    // bf16 ulp in cos can move a logit across a decision boundary. Closing the
    // last entries bit-exactly would mean vendoring Intel oneMKL -- x86-only
    // and closed-source -- into a CUDA driver's host-side table builder, which
    // is a cost decision to be taken deliberately, not by relaxing a test.
    const bool ok = d.total == 0 && oob == 0u && capacity > max_pos;
    if (!ok) ++g_failures;
    std::printf("[%s] %-28s entries=%d  mismatches=%d (required: 0; "
                "at_override=%d elsewhere=%d guards_broken=%d)  capacity=%d "
                "(max_pos=%d)  device_trig_fallback_blocks=%u (required: 0)\n",
                ok ? "ok" : "FAIL", "vllm-table matches cache",
                g::kRows * g::kRotaryDim, d.total, d.at_override, d.elsewhere,
                d.guards_broken, capacity, max_pos, oob);
    // The trig backend changes the table's bits, so a mismatch count means
    // nothing without it. Expected on the deployment base (glibc 2.35):
    // exact -> 19, libm -> 18, of which only pos=13852 is inside the campaign
    // window. Neither is zero; closing the rest would mean vendoring oneMKL.
    std::printf("      host trig backend: %s   (exact is deterministic across "
                "C libraries; libm is not)\n",
                kernels::rope_vllm_table_trig_name());
    if (d.guards_broken > 0)
        std::printf("      REGRESSION: an entry the host-side table build "
                    "already fixed has come undone\n");
}

// ── The knob-off assertion: it must NOT reproduce vLLM's table ─────────────
//
// Without this, a silent regression that routed the knob back to the default
// kernel -- or a knob that never took effect -- would leave the check above
// passing for the wrong reason. The default path uses the `__sincosf` SFU
// intrinsic and a `theta^(-2j/d)` exponent form, so it disagrees with the
// reference table on a few percent of entries. The floor is set well below the
// campaign's measured disagreement rate (~3.8% of entries at corpus lengths,
// ~4.9% past position 20000) so this states a real property rather than a
// tuned threshold.
void run_default_path_differs() {
    namespace g = pie_rope_golden;
    const int entries = g::kRows * g::kRotaryDim;
    const std::vector<std::uint16_t> got = read_back_table(Path::Default);
    const int bad = diff_table(got, /*report=*/false).total;
    const int floor = entries / 100;  // 1%, vs ~4% measured
    const bool ok = bad >= floor;
    if (!ok) ++g_failures;
    std::printf("[%s] %-28s entries=%d  mismatches_vs_vllm=%d (required: >=%d, "
                "the knob must actually change something)\n",
                ok ? "ok" : "FAIL", "default path differs", entries, bad, floor);
}

// ── The bf16 rotate structure, on real data ────────────────────────────────
//
// The read-back probe above pins the TABLE but not the ROTATE: with a=1, b=0
// an fp32 rotate and a bf16 rotate agree trivially. This drives random bf16
// operands and reproduces the rotate exactly -- three separately-rounded
// operations per output, in Triton's operand order. No transcendental is
// evaluated here, so there is nothing to be tolerant about.
//
// THE REFERENCE COS/SIN COME FROM PIE'S OWN TABLE, read back off the device,
// NOT from the golden. That is deliberate and it is the difference between one
// defect reporting once and reporting three times.
//
// An earlier version sourced them from `g::kCosSinCache`. Because Pie's table
// differs from the reference at 19 entries, every one of those entries poisoned
// this check too -- amplified by 6 heads x 2 lanes per affected pair, it turned
// 19 table entries into 130 and 132 "rotate" failures, on rows that had nothing
// wrong with their rotation. A reader then had to run a separate probe to
// discover that the rotate check was failing for a table reason.
//
// Sourcing from Pie's own table makes this check answer exactly one question:
// given whatever cos/sin the kernel actually has, does it combine them with q/k
// correctly? Table-vs-reference parity is owned solely by
// `run_vllm_golden_table`, which is red at 19 and is where that fact belongs.
//
// This does NOT make the check circular, and that was verified rather than
// asserted -- three candidate kernels simulated against each other:
//
//   variant                 read-back table    random-operand rotate
//   correct bf16 rotate     identical          identical
//   fp32 rotate             IDENTICAL          differs, 2181/4000
//   swapped operand order   differs            differs, 4000/4000
//
// The middle row is the whole point. An fp32 rotate is INVISIBLE to the
// read-back -- with a = 1.0, b = 0.0 the products are exact and every variant
// agrees -- so if the extraction could launder a rotate defect, that is the one
// it would launder. It does not: under random operands the check catches it on
// more than half the pairs. A wrong pair offset or operand order is caught by
// both probes.
void run_vllm_rotate_structure(int head_dim,
                               const std::vector<std::uint16_t>& pie_table) {
    namespace g = pie_rope_golden;
    constexpr int q_heads = 4, kv_heads = 2;
    const int tokens = g::kRows;
    const int angles = g::kAngles;
    const int total_q = tokens * q_heads * head_dim;
    const int total_k = tokens * kv_heads * head_dim;

    std::mt19937 rng(4321);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<std::uint16_t> q(total_q), k(total_k);
    for (int i = 0; i < total_q; ++i) q[i] = float_to_bf16(dist(rng));
    for (int i = 0; i < total_k; ++i) k[i] = float_to_bf16(dist(rng));
    const std::vector<std::uint16_t> q_in = q, k_in = k;
    std::vector<std::uint16_t> q_ref = q, k_ref = k;

    std::vector<std::int32_t> positions(tokens);
    for (int n = 0; n < tokens; ++n) positions[n] = g::kCosSinCache[n].pos;

    auto rotate_ref = [&](std::vector<std::uint16_t>& buf, int heads) {
        for (int n = 0; n < tokens; ++n) {
            for (int h = 0; h < heads; ++h) {
                const int base = (n * heads + h) * head_dim;
                for (int j = 0; j < angles; ++j) {
                    const int row = n * g::kRotaryDim;
                    const float c = bf16_to_float(pie_table[row + j]);
                    const float s = bf16_to_float(pie_table[row + angles + j]);
                    const float a = bf16_to_float(buf[base + j]);
                    const float b = bf16_to_float(buf[base + j + angles]);
                    const float ac = bf16_to_float(float_to_bf16(a * c));
                    const float bs = bf16_to_float(float_to_bf16(b * s));
                    const float bc = bf16_to_float(float_to_bf16(b * c));
                    const float as = bf16_to_float(float_to_bf16(a * s));
                    buf[base + j] = float_to_bf16(ac - bs);
                    buf[base + j + angles] = float_to_bf16(bc + as);
                }
            }
        }
    };
    rotate_ref(q_ref, q_heads);
    rotate_ref(k_ref, kv_heads);

    void *dq = nullptr, *dk = nullptr;
    std::int32_t* dpos = nullptr;
    cudaMalloc(&dq, total_q * sizeof(std::uint16_t));
    cudaMalloc(&dk, total_k * sizeof(std::uint16_t));
    cudaMalloc(&dpos, tokens * sizeof(std::int32_t));
    cudaMemcpy(dq, q.data(), total_q * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), total_k * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, positions.data(), tokens * sizeof(std::int32_t),
               cudaMemcpyHostToDevice);
    kernels::launch_rope_partial_vllm_table_bf16(
        dq, dk, dpos, tokens, q_heads, kv_heads, head_dim, g::kRotaryDim,
        g::kTheta, /*stream=*/nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(q.data(), dq, total_q * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(k.data(), dk, total_k * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(dq); cudaFree(dk); cudaFree(dpos);

    int bit_mismatches = 0, passthrough_violations = 0;
    auto check = [&](const std::vector<std::uint16_t>& got,
                     const std::vector<std::uint16_t>& in,
                     const std::vector<std::uint16_t>& ref, int heads) {
        for (int n = 0; n < tokens; ++n)
            for (int h = 0; h < heads; ++h) {
                const int base = (n * heads + h) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    if (d >= g::kRotaryDim) {
                        // Assertion 2 holds on this path too.
                        if (got[base + d] != in[base + d]) ++passthrough_violations;
                    } else if (got[base + d] != ref[base + d]) {
                        ++bit_mismatches;
                    }
                }
            }
    };
    check(q, q_in, q_ref, q_heads);
    check(k, k_in, k_ref, kv_heads);

    const bool ok = bit_mismatches == 0 && passthrough_violations == 0;
    if (!ok) ++g_failures;
    std::printf("[%s] %-28s head_dim=%d rotary_dim=%d  bit_mismatches=%d  "
                "passthrough_violations=%d (required: 0/0)\n",
                ok ? "ok" : "FAIL", "vllm-table rotate", head_dim,
                g::kRotaryDim, bit_mismatches, passthrough_violations);
}
}  // namespace

int main() {
    // Without this, a run on a host with no CUDA device does not skip and does
    // not error -- every `cudaMalloc` fails silently, the kernels never launch,
    // and the checks compare uninitialized memory and print FAIL. A build host
    // and a genuine numeric regression then look identical in the log, which is
    // exactly the confusion this whole file exists to prevent.
    int devices = 0;
    const cudaError_t err = cudaGetDeviceCount(&devices);
    if (err != cudaSuccess || devices == 0) {
        std::printf("no CUDA device (%s) -- this test requires a GPU and did "
                    "NOT run\n", cudaGetErrorString(err));
        return 77;  // ctest's conventional "skipped"
    }

    // Fixture sanity first, genuinely before any GPU work: every input it reads
    // is `constexpr`, so it cannot depend on device state, and a fixture that
    // cannot fail should be reported before spending a single kernel launch.
    run_golden_slice_is_not_blind();

    const Case cases[] = {
        // The shape that was broken in production.
        {"qwen3.6-27b", 256, 64, 8, 2, 4, 1e7f},
        // A different partial factor, so a fix that hardcodes 64 is caught.
        {"partial-half", 128, 64, 4, 2, 3, 1e6f},
        {"partial-eighth", 256, 32, 4, 1, 2, 1e7f},
        // rotary_dim == head_dim: both historical bugs vanish here, which is
        // exactly why every full-rotary model stayed correct and this went
        // unnoticed. Kept so the fix cannot regress the full case.
        {"full-rotary", 128, 128, 4, 2, 3, 1e6f},
    };
    for (const auto& c : cases) run_case(c);

    // Parity against vLLM's own cos_sin_cache. Zero tolerance, both directions:
    // knob-on must reproduce it, knob-off must not.
    run_vllm_golden_table();
    run_default_path_differs();
    // Reference cos/sin for the rotate checks are Pie's OWN table, so a table
    // residual cannot masquerade as a rotate failure. See the note above
    // `run_vllm_rotate_structure`.
    const std::vector<std::uint16_t> pie_table = read_back_table(Path::VllmTable);
    run_vllm_rotate_structure(256, pie_table);   // production shape
    run_vllm_rotate_structure(128, pie_table);   // head_dim != 4 * rotary_dim

    if (g_failures != 0) {
        std::printf("\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
