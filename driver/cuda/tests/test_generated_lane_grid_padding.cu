// Padded-lane-grid canary for the generated stage body (stage6-plan.md
// increment 3), on the `test_graph_padding_kv_canary` harness pattern: give
// the surplus work its OWN way to fail loudly, then prove it stayed silent.
//
// `launch_generated_stage` now sizes every lane-derived grid at the lattice
// bucket (`generated_lane_grid_bucket`) while the live lane count travels as
// device data (`PtirLaneTableHeader::lane_count`). The safety claim is that a
// block whose lane index falls in [live, bucket) exits on the guard BEFORE
// touching any lane-indexed table or output. This test launches three of the
// body's kernels exactly as the body does — bucket-sized grid, live-sized
// header — with the phantom lanes fully ARMED: valid-looking lane records,
// commit flags raised, value/dest pointers aimed at canary buffers, and (for
// readiness) a not-ready channel that would zero the phantom's commit flag.
// If any kernel's guard moved below a write, a canary flips and the test
// fails; a segfault-shaped failure (phantom pointers null) would prove
// nothing.
//
// Covered here: k_grouped_stage_readiness (the capture-stability target of
// scope item 3), k_generated_scan_f32, k_generated_attn_page_mask. The other
// lane-gridded kernels (generated NVRTC region, nucleus family, topk,
// envelope_dot, matmul) use the identical guard-first idiom —
// `header->lane_count` read before any lane table — and the NVRTC template's
// guard lives at compiler/codegen/runtime/cuda/fused_block1.cuh:18-19; they
// are exercised end-to-end by the PTIR suites.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "pipeline/generated/fused_runtime.cuh"

using namespace pie_cuda_driver::pipeline;
using namespace pie_cuda_driver::pipeline::generated;

namespace {

int g_fail = 0;

// fused_runtime.cuh (via cuda_check.hpp) already defines the throwing
// CUDA_CHECK; a test wants exit-code-2 semantics like the kv canary, so it
// carries its own differently-named macro instead of redefining that one.
void check(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess) {
        std::fprintf(
            stderr, "%s:%d: %s: %s\n", __FILE__, line, expression,
            cudaGetErrorString(error));
        std::exit(2);
    }
}

#define TEST_CUDA_CHECK(expr) check((expr), #expr, __LINE__)

void expect(bool ok, const char* what) {
    if (ok) {
        std::printf("  PASS  %s\n", what);
    } else {
        ++g_fail;
        std::printf("  FAIL  %s\n", what);
    }
}

constexpr std::uint32_t kLive = 3;
constexpr std::uint32_t kBucket = generated_lane_grid_bucket(kLive);
static_assert(kBucket == 4, "3 live lanes pad to a bucket of 4");

constexpr float kCanaryFloat = -777.25f;
constexpr std::uint8_t kCanaryByte = 0xa5;

template <class T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    TEST_CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
    TEST_CUDA_CHECK(cudaMemcpy(
        device, host.data(), host.size() * sizeof(T),
        cudaMemcpyHostToDevice));
    return device;
}

template <class T>
std::vector<T> download(const T* device, std::size_t count) {
    std::vector<T> host(count);
    TEST_CUDA_CHECK(cudaMemcpy(
        host.data(), device, count * sizeof(T),
        cudaMemcpyDeviceToHost));
    return host;
}

}  // namespace

int main() {
    // Live header; the grids below deliberately exceed it.
    const PtirLaneTableHeader header_h{
        PTIR_LANE_TABLE_ABI_VERSION, kLive, 0, 0};
    PtirLaneTableHeader* header = upload<PtirLaneTableHeader>({header_h});

    // One commit word per PADDED lane, all raised: a phantom lane is not
    // allowed to hide behind a lowered commit flag.
    std::uint32_t* commits =
        upload<std::uint32_t>(std::vector<std::uint32_t>(kBucket, 1u));

    // Armed lane records for every PADDED lane: the phantom's extents are
    // real, so if a kernel consulted the record instead of the header it
    // would happily do work.
    std::vector<PtirLaneRecord> lanes_h(kBucket);
    for (std::uint32_t lane = 0; lane < kBucket; ++lane) {
        std::memset(&lanes_h[lane], 0, sizeof(PtirLaneRecord));
        lanes_h[lane].commit_slot =
            reinterpret_cast<std::uint64_t>(commits + lane);
        lanes_h[lane].row_count = 2;
        lanes_h[lane].sampled_rows = 2;
        lanes_h[lane].logits_row_count = 2;
    }
    PtirLaneRecord* lanes = upload<PtirLaneRecord>(lanes_h);

    // ---- k_grouped_stage_readiness ------------------------------------
    // Live lane 1 and phantom lane 3 both demand a channel whose ring cell
    // is empty (not ready). The kernel must zero lane 1's commit (proof it
    // ran) and must NOT touch lane 3's (proof the guard held).
    std::vector<GroupedReadinessLane> readiness_h(kBucket);
    readiness_h[1].full_offset = 0;
    readiness_h[1].full_count = 1;
    readiness_h[3].full_offset = 0;
    readiness_h[3].full_count = 1;
    GroupedReadinessLane* readiness =
        upload<GroupedReadinessLane>(readiness_h);
    std::uint32_t* slots = upload<std::uint32_t>({0u});
    std::uint8_t* ring_full = upload<std::uint8_t>(
        std::vector<std::uint8_t>(kMaxRing, 0));  // slot 0: nothing full
    std::uint32_t* ring_head = upload<std::uint32_t>({0u});
    std::uint32_t* ring_tail = upload<std::uint32_t>({0u});
    std::uint32_t* ring_cap1 = upload<std::uint32_t>({4u});

    k_grouped_stage_readiness<<<(kBucket + 127) / 128, 128>>>(
        header, lanes, readiness, slots,
        ring_full, ring_head, ring_tail, ring_cap1, 0u);
    TEST_CUDA_CHECK(cudaGetLastError());
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    {
        const auto commits_after = download<std::uint32_t>(commits, kBucket);
        expect(commits_after[0] == 1u, "readiness: ready live lane kept");
        expect(commits_after[1] == 0u,
               "readiness: not-ready live lane zeroed (kernel ran)");
        expect(commits_after[2] == 1u, "readiness: ready live lane kept");
        expect(commits_after[3] == 1u,
               "readiness: armed PHANTOM lane untouched (guard held)");
    }
    // Re-raise lane 1 for the kernels below (they skip commit==0 lanes).
    {
        const std::uint32_t one = 1;
        TEST_CUDA_CHECK(cudaMemcpy(
            commits + 1, &one, sizeof(one), cudaMemcpyHostToDevice));
    }

    // ---- k_generated_scan_f32 -----------------------------------------
    // 2 rows x 4 columns per lane, value 0 -> value 1 cumulative sum. The
    // phantom lane's value-table entries point at a canary buffer that a
    // block past the guard would overwrite with prefix sums.
    constexpr std::uint32_t kRows = 2;
    constexpr std::uint32_t kColumns = 4;
    constexpr std::uint32_t kValueCount = 2;
    constexpr std::uint32_t kPerLane = kRows * kColumns;
    GroupedRowShape shape{};
    shape.rows.max_numel = kRows;
    shape.columns.max_numel = kColumns;
    shape.max_rows = kRows;
    shape.max_columns = kColumns;

    std::vector<float> scan_input_h(kLive * kPerLane);
    for (std::size_t index = 0; index < scan_input_h.size(); ++index) {
        scan_input_h[index] = static_cast<float>(index % 5) + 1.0f;
    }
    float* scan_input = upload<float>(scan_input_h);
    float* scan_output = upload<float>(
        std::vector<float>(kLive * kPerLane, 0.0f));
    float* scan_canary = upload<float>(
        std::vector<float>(2 * kPerLane, kCanaryFloat));

    std::vector<std::uint64_t> values_h(kBucket * kValueCount);
    for (std::uint32_t lane = 0; lane < kLive; ++lane) {
        values_h[lane * kValueCount + 0] =
            reinterpret_cast<std::uint64_t>(scan_input + lane * kPerLane);
        values_h[lane * kValueCount + 1] =
            reinterpret_cast<std::uint64_t>(scan_output + lane * kPerLane);
    }
    // Phantom lane: input reads the canary, output WRITES the canary.
    values_h[3 * kValueCount + 0] =
        reinterpret_cast<std::uint64_t>(scan_canary);
    values_h[3 * kValueCount + 1] =
        reinterpret_cast<std::uint64_t>(scan_canary + kPerLane);
    std::uint64_t* values = upload<std::uint64_t>(values_h);

    k_generated_scan_f32<<<kBucket * kRows, 1>>>(
        header, lanes, values, kValueCount, 0u, 1u, shape, false);
    TEST_CUDA_CHECK(cudaGetLastError());
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    {
        const auto out = download<float>(scan_output, kLive * kPerLane);
        bool live_ok = true;
        for (std::uint32_t lane = 0; lane < kLive && live_ok; ++lane) {
            for (std::uint32_t row = 0; row < kRows && live_ok; ++row) {
                float running = 0.0f;
                for (std::uint32_t column = 0; column < kColumns; ++column) {
                    const std::size_t at =
                        static_cast<std::size_t>(lane) * kPerLane +
                        row * kColumns + column;
                    running += scan_input_h[at];
                    if (out[at] != running) live_ok = false;
                }
            }
        }
        expect(live_ok, "scan: live lanes produced exact prefix sums");
        const auto canary = download<float>(scan_canary, 2 * kPerLane);
        bool canary_ok = true;
        for (const float value : canary) {
            if (value != kCanaryFloat) canary_ok = false;
        }
        expect(canary_ok,
               "scan: armed PHANTOM lane wrote nothing (guard held)");
    }

    // ---- k_generated_attn_page_mask -----------------------------------
    // Live lanes threshold a 4-page float mask into their keep bytes; the
    // phantom lane's dest aims at a canary byte buffer and its mask at
    // nonzero floats, so a block past the guard would raise keep bytes.
    constexpr std::uint32_t kPages = 4;
    std::vector<float> mask_h;
    for (std::uint32_t lane = 0; lane < kLive; ++lane) {
        for (std::uint32_t page = 0; page < kPages; ++page) {
            mask_h.push_back((lane + page) % 2 == 0 ? 1.0f : 0.0f);
        }
    }
    float* mask = upload<float>(mask_h);
    float* mask_canary_src = upload<float>(
        std::vector<float>(kPages, 1.0f));  // all-keep: maximally visible
    std::uint8_t* keep = upload<std::uint8_t>(
        std::vector<std::uint8_t>(kLive * kPages, 0xff));
    std::uint8_t* keep_canary = upload<std::uint8_t>(
        std::vector<std::uint8_t>(kPages, kCanaryByte));

    std::vector<GroupedLanePageMaskDevice> dests_h(kBucket);
    for (std::uint32_t lane = 0; lane < kLive; ++lane) {
        dests_h[lane] = {keep + lane * kPages, kPages, 0u};
    }
    dests_h[3] = {keep_canary, kPages, 0u};
    GroupedLanePageMaskDevice* dests =
        upload<GroupedLanePageMaskDevice>(dests_h);

    std::vector<std::uint64_t> mask_values_h(kBucket);
    for (std::uint32_t lane = 0; lane < kLive; ++lane) {
        mask_values_h[lane] =
            reinterpret_cast<std::uint64_t>(mask + lane * kPages);
    }
    mask_values_h[3] = reinterpret_cast<std::uint64_t>(mask_canary_src);
    std::uint64_t* mask_values = upload<std::uint64_t>(mask_values_h);

    k_generated_attn_page_mask<float><<<kBucket, kTier0Block>>>(
        header, lanes, dests, mask_values, 1u, 0u);
    TEST_CUDA_CHECK(cudaGetLastError());
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    {
        const auto keep_after =
            download<std::uint8_t>(keep, kLive * kPages);
        bool live_ok = true;
        for (std::size_t at = 0; at < keep_after.size(); ++at) {
            const std::uint8_t expected =
                mask_h[at] != 0.0f ? 1u : 0u;
            if (keep_after[at] != expected) live_ok = false;
        }
        expect(live_ok, "page mask: live lanes thresholded exactly");
        const auto canary =
            download<std::uint8_t>(keep_canary, kPages);
        bool canary_ok = true;
        for (const std::uint8_t byte : canary) {
            if (byte != kCanaryByte) canary_ok = false;
        }
        expect(canary_ok,
               "page mask: armed PHANTOM lane wrote nothing (guard held)");
    }

    if (g_fail == 0) {
        std::printf("test_generated_lane_grid_padding: all checks passed\n");
        return 0;
    }
    std::printf("test_generated_lane_grid_padding: %d FAILED\n", g_fail);
    return 1;
}
