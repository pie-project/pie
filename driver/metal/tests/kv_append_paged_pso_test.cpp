// Phase 1b/3 paged-KV bridge multi-batch PSO compile gate — REAL Metal
// shader compilation (runs on-device via `[MTLDevice newLibraryWithSource:]`,
// The driver compiles shaders at runtime, but this test needs
// NO checkpoint at all — just the `kernels/*.metal` sources on disk and a
// Metal device, both available on this Mac. Gated on Apple (this file only
// builds/links where RawMetalContext is real; a non-Apple CI box has nothing
// to compile shaders against, so this test is Apple-only, unlike the pure
// heap_layout/executor_geometry gates).
//
// Proves `load_multibatch_psos` — including the kv_append_paged PSO this
// session ADDED (metal_ptir_plan.md Phase 1b/3 review: "kv_append_paged has
// no PSO entry" was cited as a concrete missing piece) — compiles
// successfully against the real kernel sources: embed_gather_mb, rope_mb,
// gdn_core_slotted, sdpa_paged (d256 required, d512/gemma4 optional), and
// kv_append_paged. A real, decisive, on-hardware check that these shaders
// are syntactically valid Metal and bind-index-consistent with the
// `MultiBatchPsos`/`bind::` contracts in decode_abi.hpp — the actual
// encoder/DAG wiring to DISPATCH them in a live forward is the genuinely
// remaining gap (see decode_step_mb — not yet implemented this session);
// this test proves the compile-time half of that gap is now closed.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "decode_psos.hpp"
#include "mtl4_context.hpp"

using pie::metal::load_multibatch_psos;
using pie::metal::load_decode_psos;
using pie::metal::DecodeStepPsos;
using pie::metal::MultiBatchPsos;
using pie::metal::RawMetalContext;
using pie::metal::Grid;
using pie::metal::Threadgroup;
using pie::metal::SlotHandle;

namespace {
int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}

std::uint16_t bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

float from_bf16(std::uint16_t value) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(value) << 16;
    float result = 0;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
}  // namespace

int main() {
    std::printf("[multi-batch PSO compile gate: real Metal shader compilation]\n");

    std::string kernels_dir;
    if (const char* kd = std::getenv("PIE_METAL_KERNELS_DIR")) kernels_dir = kd;
#ifdef PIE_METAL_KERNELS_DIR_DEFAULT
    if (kernels_dir.empty()) kernels_dir = PIE_METAL_KERNELS_DIR_DEFAULT;
#endif
    if (!expect(!kernels_dir.empty(), "kernels_dir resolved (env or compiled-in default)")) {
        std::printf("\n==== kv_append_paged_pso_test: %d passed, %d failed ====\n", g_pass, g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    // A small heap is enough — this test only compiles PSOs, it never
    // allocates/binds a real decode heap.
    auto ctx = RawMetalContext::create(16u << 20);
    if (!expect(ctx != nullptr, "RawMetalContext::create succeeds")) {
        std::printf("\n==== kv_append_paged_pso_test: %d passed, %d failed ====\n", g_pass, g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    MultiBatchPsos psos;
    std::string err;
    DecodeStepPsos base;
    expect(load_decode_psos(*ctx, kernels_dir, base, /*with_argmax=*/false, &err,
                            /*fuse_residual=*/false, /*gdn_prep=*/true),
           "load_decode_psos compiles base kernels after MB row ABI additions (" + err + ")");
    const bool ok = load_multibatch_psos(*ctx, kernels_dir, psos, /*with_d512=*/true, &err);
    expect(ok, "load_multibatch_psos compiles successfully (" + err + ")");
    expect(psos.embed_mb.valid(), "embed_gather_mb_4bit_bfloat16_gs_64_b_4 compiled");
    expect(psos.rope_mb.valid(), "rope_neox_mb_bfloat16 compiled");
    expect(psos.gdn_slotted.valid(), "gdn_core_slotted_bfloat16 compiled");
    expect(psos.sdpa_paged.valid(), "sdpa_paged_decode_bfloat16_d_256 compiled");
    expect(psos.sdpa_paged_d512.valid(), "sdpa_paged_decode_bfloat16_d_512 (gemma4) compiled");
    expect(psos.kv_append_paged.valid(),
          "kv_append_paged_bfloat16 compiled — Phase 1b/3 review's cited gap "
          "(\"kv_append_paged has no PSO\") is closed at the compile level");
    expect(psos.valid(), "MultiBatchPsos::valid() (all required paged/slotted PSOs) is true");

    if (ok) {
        constexpr std::size_t width = 256;
        std::vector<SlotHandle> allocations;
        auto allocate = [&](std::size_t bytes) -> SlotHandle {
            SlotHandle handle = ctx->create_standalone_buffer(bytes);
            allocations.push_back(handle);
            return handle;
        };
        SlotHandle query = allocate(width * sizeof(std::uint16_t));
        SlotHandle keys = allocate(2 * width * sizeof(std::uint16_t));
        SlotHandle values = allocate(2 * width * sizeof(std::uint16_t));
        SlotHandle output = allocate(width * sizeof(std::uint16_t));
        SlotHandle gqa = allocate(sizeof(int));
        SlotHandle positions = allocate(sizeof(int));
        SlotHandle requests = allocate(sizeof(int));
        SlotHandle page_indices = allocate(sizeof(std::uint32_t));
        SlotHandle page_indptr = allocate(2 * sizeof(std::uint32_t));
        SlotHandle page_size = allocate(sizeof(int));
        SlotHandle heads = allocate(sizeof(int));
        SlotHandle scale = allocate(sizeof(float));
        SlotHandle mask = allocate(2);
        SlotHandle mask_stride = allocate(sizeof(std::uint32_t));
        SlotHandle mask_enabled = allocate(1);
        std::memset(query.contents(), 0, query.size);
        std::memset(keys.contents(), 0, keys.size);
        auto* value_data =
            static_cast<std::uint16_t*>(values.contents());
        for (std::size_t column = 0; column < width; ++column) {
            value_data[column] = bf16(1.0f);
            value_data[width + column] = bf16(3.0f);
        }
        *static_cast<int*>(gqa.contents()) = 1;
        *static_cast<int*>(positions.contents()) = 1;
        *static_cast<int*>(requests.contents()) = 0;
        *static_cast<std::uint32_t*>(page_indices.contents()) = 0;
        auto* indptr =
            static_cast<std::uint32_t*>(page_indptr.contents());
        indptr[0] = 0;
        indptr[1] = 1;
        *static_cast<int*>(page_size.contents()) = 2;
        *static_cast<int*>(heads.contents()) = 1;
        *static_cast<float*>(scale.contents()) = 1.0f;
        auto* mask_data = static_cast<std::uint8_t*>(mask.contents());
        mask_data[0] = 1;
        mask_data[1] = 0;
        *static_cast<std::uint32_t*>(mask_stride.contents()) = 2;
        *static_cast<std::uint8_t*>(mask_enabled.contents()) = 1;
        constexpr int ordinal = 97000;
        for (const auto& [index, handle] :
             std::vector<std::pair<std::uint8_t, SlotHandle>>{
                 {0, query}, {1, keys}, {2, values}, {3, output},
                 {4, gqa}, {5, positions}, {6, requests},
                 {7, page_indices}, {8, page_indptr}, {9, page_size},
                 {10, heads}, {11, scale}, {12, mask},
                 {13, mask_stride}, {14, mask_enabled}}) {
            ctx->arg_bind_ordinal(ordinal, index, handle);
        }
        ctx->make_resident();
        const auto masked_timing = ctx->run_step([&](auto& encoder) {
            encoder.set_pso(psos.sdpa_paged);
            encoder.set_argtable_ordinal(ordinal);
            encoder.dispatch(Grid{1024, 1, 1}, Threadgroup{1024, 1, 1});
        });
        const float masked =
            from_bf16(static_cast<std::uint16_t*>(output.contents())[0]);
        *static_cast<std::uint8_t*>(mask_enabled.contents()) = 0;
        std::memset(output.contents(), 0, output.size);
        const auto causal_timing = ctx->run_step([&](auto& encoder) {
            encoder.set_pso(psos.sdpa_paged);
            encoder.set_argtable_ordinal(ordinal);
            encoder.dispatch(Grid{1024, 1, 1}, Threadgroup{1024, 1, 1});
        });
        const float causal =
            from_bf16(static_cast<std::uint16_t*>(output.contents())[0]);
        expect(
            masked_timing.succeeded() && causal_timing.succeeded() &&
                std::abs(masked - 1.0f) < 0.02f &&
                std::abs(causal - 2.0f) < 0.02f,
            "sdpa_paged consumes the bound dense mask exactly "
            "(masked=" + std::to_string(masked) +
            ", unmasked=" + std::to_string(causal) + ")");
        ctx->release_argtable_ordinal(ordinal);
        for (const SlotHandle handle : allocations) {
            ctx->release_standalone_buffer(handle);
        }
    }

    std::printf("\n==== kv_append_paged_pso_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
