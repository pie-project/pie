#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "pie_native/ptir/bound.hpp"
#include "pie_native/ptir/container.hpp"
#include "pie_native/ptir/plan.hpp"
#include "pipeline/dispatch.hpp"

using pie_cuda_driver::pipeline::Dispatch;
using pie_cuda_driver::pipeline::DispatchStats;
using pie_cuda_driver::pipeline::RetryableLaunchError;

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    }
}

std::string trim(const std::string& value) {
    const std::size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const std::size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string golden_field(const std::string& path, const std::string& key) {
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t separator = line.find(':');
        if (separator == std::string::npos) continue;
        if (trim(line.substr(0, separator)) == key) {
            return trim(line.substr(separator + 1));
        }
    }
    return {};
}

std::vector<std::uint8_t> hex_bytes(const std::string& value) {
    std::vector<std::uint8_t> result;
    result.reserve(value.size() / 2);
    for (std::size_t index = 0; index + 1 < value.size(); index += 2) {
        result.push_back(static_cast<std::uint8_t>(
            std::stoul(value.substr(index, 2), nullptr, 16)));
    }
    return result;
}

std::uint16_t bf16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

void publish_mask(const PieChannelEndpointBinding& binding) {
    auto* words = reinterpret_cast<std::uint64_t*>(binding.word_base);
    std::memset(
        reinterpret_cast<void*>(binding.mirror_base),
        0xff,
        binding.cell_bytes);
    std::atomic_ref<std::uint64_t>(words[binding.tail_word_index])
        .store(1, std::memory_order_release);
}

void publish_bytes(
    const PieChannelEndpointBinding& binding,
    const void* bytes,
    std::size_t size) {
    std::memcpy(
        reinterpret_cast<void*>(binding.mirror_base), bytes, size);
    auto* words = reinterpret_cast<std::uint64_t*>(binding.word_base);
    std::atomic_ref<std::uint64_t>(words[binding.tail_word_index])
        .store(1, std::memory_order_release);
}

std::vector<std::uint8_t> read_packed_bool(
    const PieChannelEndpointBinding& binding,
    std::size_t count) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(
        binding.mirror_base);
    std::vector<std::uint8_t> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = static_cast<std::uint8_t>(
            (bytes[index / 8] >> (index % 8)) & 1u);
    }
    return values;
}

std::uint64_t run_case(
    const std::string& golden_directory,
    bool partial,
    bool recurrent_state,
    bool duplicate_instance = false,
    bool direct_bf16 = false,
    std::uint32_t lane_count = 4,
    bool benchmark = false) {
    constexpr std::uint32_t vocab = 32;
    constexpr std::uint64_t no_ticket =
        std::numeric_limits<std::uint64_t>::max();
    const std::string path =
        golden_directory + "/section3_masked_gumbel.txt";
    const std::vector<std::uint8_t> container_bytes =
        hex_bytes(golden_field(path, "container"));
    const std::vector<std::uint8_t> sidecar_bytes =
        hex_bytes(golden_field(path, "sidecar"));
    const std::uint64_t hash =
        std::stoull(golden_field(path, "hash"), nullptr, 16);

    pie_native::ptir::container::Container container;
    pie_native::ptir::container::DecodeError decode_error;
    expect(
        pie_native::ptir::container::decode(
            container_bytes.data(), container_bytes.size(),
            container, &decode_error),
        "decode section3 fixture: " + decode_error.detail);
    expect(container.channels.size() == 5, "section3 channel count");

    Dispatch dispatch;
    std::string error;
    expect(
        dispatch.register_program(
            hash,
            pie_native::ByteSlice{
                container_bytes.data(), container_bytes.size()},
            pie_native::ByteSlice{
                sidecar_bytes.data(), sidecar_bytes.size()},
            &error) == PIE_STATUS_OK,
        "register grouped program: " + error);
    const DispatchStats registration_stats = dispatch.stats();
    expect(
        registration_stats.generated_compilations +
                registration_stats.generated_disk_hits !=
            0 &&
            registration_stats.generated_stage_cache_entries != 0 &&
            registration_stats.generated_program_cache_entries == 1,
        "registration compiles every generated fused region before publishing");

    std::vector<std::uint64_t> hashes(lane_count, hash);
    std::vector<std::uint64_t> instances(lane_count);
    std::vector<PieTerminalCell> terminals(lane_count);
    std::vector<PieTerminalCell*> terminal_ptrs(lane_count);
    std::vector<std::vector<PieChannelEndpointBinding>> endpoints(lane_count);
    std::vector<std::vector<std::uint64_t>> channel_ids(lane_count);
    std::vector<std::uint64_t> expected_heads;
    std::vector<std::uint64_t> expected_tails;
    std::vector<std::uint32_t> ticket_indptr{0};
    std::vector<std::uint32_t> sampling_indptr{0};
    std::vector<std::uint32_t> expected_tokens(lane_count);
    std::vector<std::uint32_t> row_counts(lane_count, 1);
    std::vector<std::uint32_t> token_counts(lane_count, 1);
    std::vector<std::uint32_t> kv_lens(lane_count, 1);
    std::vector<std::uint32_t> page_counts(lane_count, 1);
    std::vector<std::uint32_t> query_lens(lane_count, 1);
    std::vector<std::uint32_t> key_lens(lane_count, 1);

    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        if (duplicate_instance && lane != 0) {
            instances[lane] = instances[0];
            terminal_ptrs[lane] = &terminals[lane];
            expected_heads.insert(
                expected_heads.end(), {0, no_ticket, 0, 0, 0});
            expected_tails.insert(
                expected_tails.end(), {1, 0, no_ticket, 1, 1});
            ticket_indptr.push_back(
                static_cast<std::uint32_t>(expected_heads.size()));
            sampling_indptr.push_back(lane + 1);
            expected_tokens[lane] = (lane * 5 + 3) % vocab;
            continue;
        }
        endpoints[lane].resize(container.channels.size());
        channel_ids[lane].resize(container.channels.size());
        for (std::size_t dense = 0; dense < container.channels.size(); ++dense) {
            const auto& source = container.channels[dense];
            const std::uint64_t id =
                10000 + static_cast<std::uint64_t>(lane) * 100 + dense;
            channel_ids[lane][dense] = id;
            PieChannelDesc descriptor{};
            descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
            descriptor.channel_id = id;
            descriptor.shape = {source.shape.dims, source.shape.rank};
            descriptor.dtype = source.dtype;
            descriptor.host_role = source.host_role;
            descriptor.seeded = source.seeded;
            descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
            descriptor.capacity = source.capacity;
            descriptor.reader_wait_id = id * 2 + 1;
            descriptor.writer_wait_id = id * 2 + 2;
            expect(
                dispatch.register_channel(
                    descriptor, &endpoints[lane][dense], &error) ==
                    PIE_STATUS_OK,
                "register grouped channel: " + error);
        }

        const std::int32_t token = 1;
        const std::uint32_t length = 1;
        const std::uint32_t rng[2] = {1234 + lane, 0};
        const PieChannelValueDesc seed_values[] = {
            {
                channel_ids[lane][0],
                {
                    reinterpret_cast<const std::uint8_t*>(&token),
                    sizeof(token),
                },
            },
            {
                channel_ids[lane][3],
                {
                    reinterpret_cast<const std::uint8_t*>(&length),
                    sizeof(length),
                },
            },
            {
                channel_ids[lane][4],
                {
                    reinterpret_cast<const std::uint8_t*>(rng),
                    sizeof(rng),
                },
            },
        };
        instances[lane] = 500 + lane;
        PieInstanceBinding instance_binding{};
        expect(
            dispatch.bind_instance(
                instances[lane],
                hash,
                PIE_GEOMETRY_CLASS_HOST,
                900 + lane,
                channel_ids[lane],
                std::vector<PieChannelValueDesc>(
                    std::begin(seed_values), std::end(seed_values)),
                &instance_binding,
                &error) == PIE_STATUS_OK,
            "bind grouped instance: " + error);
        if (!partial || lane != 1) {
            publish_mask(endpoints[lane][2]);
        }

        terminal_ptrs[lane] = &terminals[lane];
        expected_heads.insert(
            expected_heads.end(), {0, no_ticket, 0, 0, 0});
        expected_tails.insert(
            expected_tails.end(), {1, 0, no_ticket, 1, 1});
        ticket_indptr.push_back(
            static_cast<std::uint32_t>(expected_heads.size()));
        sampling_indptr.push_back(lane + 1);
        expected_tokens[lane] = (lane * 5 + 3) % vocab;
    }

    std::vector<float> logits(
        static_cast<std::size_t>(lane_count) * vocab, -100.0f);
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        logits[static_cast<std::size_t>(lane) * vocab +
               expected_tokens[lane]] = 100.0f;
    }
    float* device_logits = nullptr;
    std::uint16_t* device_logits_bf16 = nullptr;
    std::vector<std::uint32_t> direct_rows(lane_count);
    if (direct_bf16) {
        // Four two-token requests all select request-relative row zero. The
        // composed mapping must therefore be {0,2,4,6}, not {0,0,0,0}.
        std::vector<float> source_logits(
            static_cast<std::size_t>(lane_count) * 2 * vocab,
            -100.0f);
        for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
            std::copy_n(
                logits.data() + static_cast<std::size_t>(lane) * vocab,
                vocab,
                source_logits.data() +
                    static_cast<std::size_t>(lane * 2) * vocab);
            direct_rows[lane] = lane * 2;
        }
        std::vector<std::uint16_t> bf16(source_logits.size());
        std::transform(
            source_logits.begin(), source_logits.end(), bf16.begin(),
            bf16_bits);
        cudaMalloc(
            &device_logits_bf16, bf16.size() * sizeof(std::uint16_t));
        cudaMemcpy(
            device_logits_bf16,
            bf16.data(),
            bf16.size() * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice);
    } else {
        cudaMalloc(&device_logits, logits.size() * sizeof(float));
        cudaMemcpy(
            device_logits, logits.data(), logits.size() * sizeof(float),
            cudaMemcpyHostToDevice);
    }

    pie_native::LaunchView view{};
    view.terminal_cells = pie_native::slice_from(
        terminal_ptrs.data(), terminal_ptrs.size());
    view.ptir_program_hashes = pie_native::slice_from_u64(
        hashes.data(), hashes.size());
    view.ptir_program_instances = pie_native::slice_from_u64(
        instances.data(), instances.size());
    view.sampling_indptr = pie_native::slice_from_u32(
        sampling_indptr.data(), sampling_indptr.size());
    view.channel_expected_head = pie_native::slice_from_u64(
        expected_heads.data(), expected_heads.size());
    view.channel_expected_tail = pie_native::slice_from_u64(
        expected_tails.data(), expected_tails.size());
    view.channel_ticket_indptr = pie_native::slice_from_u32(
        ticket_indptr.data(), ticket_indptr.size());
    view.ptir_row_counts =
        pie_native::slice_from_u32(row_counts.data(), row_counts.size());
    view.ptir_token_counts =
        pie_native::slice_from_u32(token_counts.data(), token_counts.size());
    view.ptir_kv_lens =
        pie_native::slice_from_u32(kv_lens.data(), kv_lens.size());
    view.ptir_page_counts =
        pie_native::slice_from_u32(page_counts.data(), page_counts.size());
    view.ptir_query_lens =
        pie_native::slice_from_u32(query_lens.data(), query_lens.size());
    view.ptir_key_lens =
        pie_native::slice_from_u32(key_lens.data(), key_lens.size());
    const std::uint32_t rs_slot = 0;
    if (recurrent_state) {
        view.rs_slot_ids = pie_native::slice_from_u32(&rs_slot, 1);
    }

    expect(
        dispatch.validate_launch(view, &error) == PIE_STATUS_OK,
        "validate grouped launch: " + error);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudaEvent_t benchmark_start = nullptr;
    cudaEvent_t benchmark_stop = nullptr;
    if (benchmark) {
        cudaEventCreate(&benchmark_start);
        cudaEventCreate(&benchmark_stop);
        cudaEventRecord(benchmark_start, stream);
    }
    bool retryable_preflight = false;
    bool ran = false;
    try {
        ran = dispatch.run(
            view, device_logits, vocab, stream, nullptr, PieCompletion{},
            device_logits_bf16,
            direct_bf16 ? direct_rows.data() : nullptr);
    } catch (const RetryableLaunchError&) {
        retryable_preflight = true;
    }
    if (partial && recurrent_state) {
        expect(
            retryable_preflight,
            "RS launch rejects incomplete readiness before state mutation");
        cudaStreamDestroy(stream);
        cudaFree(device_logits);
        cudaFree(device_logits_bf16);
        return 0;
    }
    expect(!retryable_preflight && ran, "run grouped launch");
    if (benchmark) {
        cudaEventRecord(benchmark_stop, stream);
    }
    cudaDeviceSynchronize();
    if (benchmark) {
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(
            &elapsed_ms, benchmark_start, benchmark_stop);
        std::printf(
            "generated grouped section3 B=%u: %.3f ms/fire\n",
            lane_count,
            elapsed_ms);
        cudaEventDestroy(benchmark_stop);
        cudaEventDestroy(benchmark_start);
    }

    const DispatchStats stats = dispatch.stats();
    expect(
        stats.generated_compilations ==
            registration_stats.generated_compilations,
        "first fire performs no generated-region compilation");
    if (direct_bf16) {
        expect(
            stats.direct_bf16_groups == 1 &&
                stats.direct_bf16_solo_materializations == 0,
            "direct BF16 rows avoid sampled-logits gather/materialization");
    }
    if (duplicate_instance) {
        expect(
            stats.grouped_lanes == lane_count &&
                stats.shared_slot_exclusions != 0 &&
                stats.ordered_alias_launches != 0,
            "duplicate instance lanes retain stable commit snapshots");
    } else if (recurrent_state) {
        expect(
            stats.grouped_lanes == lane_count,
            "RS lanes retain independent device commit predicates in one group");
    } else {
        expect(
            stats.grouped_lanes == lane_count &&
                stats.generated_fused_groups == 1 &&
                stats.generated_fused_body_launches == 1,
            "Dispatch executes one generated fused body for N=4");
    }
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        const std::uint32_t outcome =
            std::atomic_ref<std::uint32_t>(terminals[lane].outcome)
                .load(std::memory_order_acquire);
        const bool retry =
            (duplicate_instance && lane != 0) ||
            (partial && lane == 1);
        const std::string case_label =
            " lane=" + std::to_string(lane) +
            " partial=" + std::to_string(partial) +
            " rs=" + std::to_string(recurrent_state) +
            " duplicate=" + std::to_string(duplicate_instance) +
            " bf16=" + std::to_string(direct_bf16);
        expect(
            outcome ==
                (retry
                    ? PIE_TERMINAL_OUTCOME_RETRY
                    : PIE_TERMINAL_OUTCOME_SUCCESS),
            "per-lane terminal attribution" + case_label +
                " outcome=" + std::to_string(outcome));
        if (!retry) {
            const std::uint32_t token = *reinterpret_cast<const std::uint32_t*>(
                endpoints[lane][1].mirror_base);
            expect(
                token == expected_tokens[lane],
                "grouped Gumbel-max token parity" + case_label +
                    " expected=" + std::to_string(expected_tokens[lane]) +
                    " actual=" + std::to_string(token));
        }
    }
    cudaStreamDestroy(stream);
    cudaFree(device_logits);
    cudaFree(device_logits_bf16);
    return stats.grouped_body_op_launches;
}

std::vector<std::int32_t> run_nucleus_case(
    const std::string& golden_directory,
    bool direct_bf16) {
    constexpr std::uint32_t lane_count = 6;
    constexpr std::uint32_t vocab = 8;
    constexpr std::uint64_t no_ticket =
        std::numeric_limits<std::uint64_t>::max();
    const std::string path = golden_directory + "/nucleus_sample.txt";
    const auto container_bytes =
        hex_bytes(golden_field(path, "container"));
    const auto sidecar_bytes = hex_bytes(golden_field(path, "sidecar"));
    const auto hash = std::stoull(golden_field(path, "hash"), nullptr, 16);
    pie_native::ptir::container::Container container;
    pie_native::ptir::container::DecodeError decode_error;
    expect(
        pie_native::ptir::container::decode(
            container_bytes.data(), container_bytes.size(),
            container, &decode_error),
        "decode nucleus fixture: " + decode_error.detail);
    expect(container.channels.size() == 3, "nucleus channel count");

    Dispatch dispatch;
    std::string error;
    expect(
        dispatch.register_program(
            hash,
            {container_bytes.data(), container_bytes.size()},
            {sidecar_bytes.data(), sidecar_bytes.size()},
            &error) == PIE_STATUS_OK,
        "register nucleus program: " + error);

    std::vector<std::uint64_t> hashes(lane_count, hash);
    std::vector<std::uint64_t> instances(lane_count);
    std::vector<PieTerminalCell> terminals(lane_count);
    std::vector<PieTerminalCell*> terminal_ptrs(lane_count);
    std::vector<std::vector<PieChannelEndpointBinding>> endpoints(lane_count);
    std::vector<std::uint64_t> expected_heads;
    std::vector<std::uint64_t> expected_tails;
    std::vector<std::uint32_t> ticket_indptr{0};
    std::vector<std::uint32_t> sampling_indptr{0};
    std::vector<std::uint32_t> unit_extents(lane_count, 1);
    const float top_ps[lane_count] = {
        0.5f, 1.0f, 0.0f, 0.5f, 1.0f, 0.9f};
    const std::uint32_t rng_states[lane_count][2] = {
        {1234, 0},
        {1234, 1},
        {1300, 0},
        {1301, 0},
        {1302, 0},
        {1303, 0},
    };
    const std::int32_t expected_tokens[lane_count] = {
        0, 0, 0, 2, 5, 0};

    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        endpoints[lane].resize(container.channels.size());
        std::vector<std::uint64_t> ids(container.channels.size());
        for (std::size_t dense = 0; dense < container.channels.size(); ++dense) {
            const auto& source = container.channels[dense];
            const std::uint64_t id =
                40000 + static_cast<std::uint64_t>(lane) * 100 + dense;
            ids[dense] = id;
            PieChannelDesc descriptor{};
            descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
            descriptor.channel_id = id;
            descriptor.shape = {source.shape.dims, source.shape.rank};
            descriptor.dtype = source.dtype;
            descriptor.host_role = source.host_role;
            descriptor.seeded = source.seeded;
            descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
            descriptor.capacity = source.capacity;
            descriptor.reader_wait_id = id * 2 + 1;
            descriptor.writer_wait_id = id * 2 + 2;
            expect(
                dispatch.register_channel(
                    descriptor, &endpoints[lane][dense], &error) ==
                    PIE_STATUS_OK,
                "register nucleus channel: " + error);
        }
        const PieChannelValueDesc seeds[] = {
            {
                ids[0],
                {
                    reinterpret_cast<const std::uint8_t*>(rng_states[lane]),
                    sizeof(rng_states[lane]),
                },
            },
            {
                ids[1],
                {
                    reinterpret_cast<const std::uint8_t*>(&top_ps[lane]),
                    sizeof(float),
                },
            },
        };
        instances[lane] = 1000 + lane;
        PieInstanceBinding binding{};
        expect(
            dispatch.bind_instance(
                instances[lane],
                hash,
                PIE_GEOMETRY_CLASS_HOST,
                5000 + lane,
                ids,
                std::vector<PieChannelValueDesc>(
                    std::begin(seeds), std::end(seeds)),
                &binding,
                &error) == PIE_STATUS_OK,
            "bind nucleus instance: " + error);
        terminal_ptrs[lane] = &terminals[lane];
        expected_heads.insert(
            expected_heads.end(), {0, no_ticket, no_ticket});
        expected_tails.insert(
            expected_tails.end(), {no_ticket, no_ticket, 0});
        ticket_indptr.push_back(
            static_cast<std::uint32_t>(expected_heads.size()));
        sampling_indptr.push_back(lane + 1);
    }

    std::vector<float> logits(
        static_cast<std::size_t>(lane_count) * vocab, -100.0f);
    const float golden_logits[vocab] = {
        4.0f,
        4.0f,
        3.0f,
        2.0f,
        1.0f,
        0.0f,
        -1.0f,
        std::numeric_limits<float>::quiet_NaN(),
    };
    std::copy(
        std::begin(golden_logits),
        std::end(golden_logits),
        logits.begin());
    std::copy(
        std::begin(golden_logits),
        std::end(golden_logits),
        logits.begin() + vocab);
    logits[3 * vocab + 2] = 100.0f;
    logits[3 * vocab + 3] = 100.0f;
    logits[4 * vocab + 5] = 100.0f;
    logits[5 * vocab + 7] = std::numeric_limits<float>::quiet_NaN();
    float* device_logits = nullptr;
    std::uint16_t* device_logits_bf16 = nullptr;
    std::vector<std::uint32_t> direct_rows(lane_count);
    if (direct_bf16) {
        std::vector<std::uint16_t> bf16(logits.size());
        std::transform(
            logits.begin(), logits.end(), bf16.begin(), bf16_bits);
        std::iota(direct_rows.begin(), direct_rows.end(), 0u);
        cudaMalloc(
            &device_logits_bf16, bf16.size() * sizeof(std::uint16_t));
        cudaMemcpy(
            device_logits_bf16,
            bf16.data(),
            bf16.size() * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice);
    } else {
        cudaMalloc(&device_logits, logits.size() * sizeof(float));
        cudaMemcpy(
            device_logits, logits.data(), logits.size() * sizeof(float),
            cudaMemcpyHostToDevice);
    }

    pie_native::LaunchView view{};
    view.terminal_cells =
        pie_native::slice_from(terminal_ptrs.data(), terminal_ptrs.size());
    view.ptir_program_hashes =
        pie_native::slice_from_u64(hashes.data(), hashes.size());
    view.ptir_program_instances =
        pie_native::slice_from_u64(instances.data(), instances.size());
    view.sampling_indptr =
        pie_native::slice_from_u32(sampling_indptr.data(), sampling_indptr.size());
    view.channel_expected_head = pie_native::slice_from_u64(
        expected_heads.data(), expected_heads.size());
    view.channel_expected_tail = pie_native::slice_from_u64(
        expected_tails.data(), expected_tails.size());
    view.channel_ticket_indptr =
        pie_native::slice_from_u32(ticket_indptr.data(), ticket_indptr.size());
    view.ptir_row_counts = pie_native::slice_from_u32(
        unit_extents.data(), unit_extents.size());
    view.ptir_token_counts = pie_native::slice_from_u32(
        unit_extents.data(), unit_extents.size());
    view.ptir_kv_lens = pie_native::slice_from_u32(
        unit_extents.data(), unit_extents.size());
    view.ptir_page_counts = pie_native::slice_from_u32(
        unit_extents.data(), unit_extents.size());
    view.ptir_query_lens = pie_native::slice_from_u32(
        unit_extents.data(), unit_extents.size());
    view.ptir_key_lens = pie_native::slice_from_u32(
        unit_extents.data(), unit_extents.size());
    expect(
        dispatch.validate_launch(view, &error) == PIE_STATUS_OK,
        "validate nucleus launch: " + error);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    expect(
        dispatch.run(
            view, device_logits, vocab, stream, nullptr, PieCompletion{},
            device_logits_bf16,
            direct_bf16 ? direct_rows.data() : nullptr),
        "run nucleus launch");
    cudaDeviceSynchronize();
    const DispatchStats stats = dispatch.stats();
    expect(
        stats.grouped_lanes == lane_count &&
            stats.nucleus_library_groups == 1 &&
            stats.generated_fused_groups == 1 &&
            stats.generated_fused_body_launches == 3,
        "generic nucleus region composes generated producers/consumers "
        "with one stock library");
    const std::uint64_t expected_body_launches = 3;
    expect(
        stats.grouped_body_op_launches == expected_body_launches,
        "nucleus launch topology is generated + library + generated: expected=" +
            std::to_string(expected_body_launches) +
            " actual=" + std::to_string(stats.grouped_body_op_launches));
    std::vector<std::int32_t> tokens;
    tokens.reserve(lane_count);
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        const auto outcome =
            std::atomic_ref<std::uint32_t>(terminals[lane].outcome)
                .load(std::memory_order_acquire);
        expect(
            outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
            "nucleus lane commits");
        const auto token = *reinterpret_cast<const std::int32_t*>(
            endpoints[lane][2].mirror_base);
        tokens.push_back(token);
        expect(token == expected_tokens[lane], "exact nucleus token");
    }
    cudaStreamDestroy(stream);
    cudaFree(device_logits);
    cudaFree(device_logits_bf16);
    return tokens;
}

void run_structured_mask_golden(const std::string& golden_directory) {
    constexpr std::uint32_t lane_count = 2;
    constexpr std::uint64_t no_ticket =
        std::numeric_limits<std::uint64_t>::max();
    const std::string path =
        golden_directory + "/structured_masks.txt";
    const auto container_bytes =
        hex_bytes(golden_field(path, "container"));
    const auto sidecar_bytes =
        hex_bytes(golden_field(path, "sidecar"));
    const auto hash =
        std::stoull(golden_field(path, "hash"), nullptr, 16);
    pie_native::ptir::container::Container container;
    pie_native::ptir::container::DecodeError decode_error;
    expect(
        pie_native::ptir::container::decode(
            container_bytes.data(),
            container_bytes.size(),
            container,
            &decode_error),
        "decode structured-mask golden: " + decode_error.detail);
    expect(container.channels.size() == 7, "structured-mask channel count");

    Dispatch dispatch;
    std::string error;
    expect(
        dispatch.register_program(
            hash,
            {container_bytes.data(), container_bytes.size()},
            {sidecar_bytes.data(), sidecar_bytes.size()},
            &error) == PIE_STATUS_OK,
        "register structured-mask golden: " + error);
    std::vector<std::uint64_t> hashes(lane_count, hash);
    std::vector<std::uint64_t> instances(lane_count);
    std::vector<PieTerminalCell> terminals(lane_count);
    std::vector<PieTerminalCell*> terminal_ptrs(lane_count);
    std::vector<std::vector<PieChannelEndpointBinding>> endpoints(lane_count);
    std::vector<std::uint64_t> expected_heads;
    std::vector<std::uint64_t> expected_tails;
    std::vector<std::uint32_t> ticket_indptr{0};
    std::vector<std::uint32_t> sampling_indptr{0};
    std::vector<std::uint32_t> unit_extents(lane_count, 1);
    const std::uint32_t positions[2] = {3, 5};
    const std::uint32_t ancestors[6] = {0, 1, 2, 1, 2, 3};
    const std::uint32_t owners[4] = {0, 1, 2, 3};

    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        endpoints[lane].resize(container.channels.size());
        std::vector<std::uint64_t> ids(container.channels.size());
        for (std::size_t channel = 0;
             channel < container.channels.size();
             ++channel) {
            const auto& source = container.channels[channel];
            ids[channel] = 80000 + lane * 32 + channel;
            PieChannelDesc descriptor{};
            descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
            descriptor.channel_id = ids[channel];
            descriptor.shape = {source.shape.dims, source.shape.rank};
            descriptor.dtype = source.dtype;
            descriptor.host_role = source.host_role;
            descriptor.seeded = source.seeded;
            descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
            descriptor.capacity = source.capacity;
            descriptor.reader_wait_id = ids[channel] * 2 + 1;
            descriptor.writer_wait_id = ids[channel] * 2 + 2;
            expect(
                dispatch.register_channel(
                    descriptor, &endpoints[lane][channel], &error) ==
                    PIE_STATUS_OK,
                "register structured-mask channel: " + error);
        }
        const PieChannelValueDesc seeds[] = {
            {
                ids[0],
                {
                    reinterpret_cast<const std::uint8_t*>(positions),
                    sizeof(positions),
                },
            },
            {
                ids[1],
                {
                    reinterpret_cast<const std::uint8_t*>(ancestors),
                    sizeof(ancestors),
                },
            },
            {
                ids[2],
                {
                    reinterpret_cast<const std::uint8_t*>(owners),
                    sizeof(owners),
                },
            },
        };
        instances[lane] = 3000 + lane;
        PieInstanceBinding binding{};
        expect(
            dispatch.bind_instance(
                instances[lane],
                hash,
                PIE_GEOMETRY_CLASS_HOST,
                7000 + lane,
                ids,
                std::vector<PieChannelValueDesc>(
                    std::begin(seeds), std::end(seeds)),
                &binding,
                &error) == PIE_STATUS_OK,
            "bind structured-mask instance: " + error);
        terminal_ptrs[lane] = &terminals[lane];
        expected_heads.insert(
            expected_heads.end(),
            {0, 0, 0, no_ticket, no_ticket, no_ticket, no_ticket});
        expected_tails.insert(
            expected_tails.end(),
            {no_ticket, no_ticket, no_ticket, 0, 0, 0, 0});
        ticket_indptr.push_back(
            static_cast<std::uint32_t>(expected_heads.size()));
        sampling_indptr.push_back(0);
    }
    pie_native::LaunchView view{};
    view.terminal_cells =
        pie_native::slice_from(terminal_ptrs.data(), terminal_ptrs.size());
    view.ptir_program_hashes =
        pie_native::slice_from_u64(hashes.data(), hashes.size());
    view.ptir_program_instances =
        pie_native::slice_from_u64(instances.data(), instances.size());
    view.sampling_indptr =
        pie_native::slice_from_u32(sampling_indptr.data(), sampling_indptr.size());
    view.channel_expected_head = pie_native::slice_from_u64(
        expected_heads.data(), expected_heads.size());
    view.channel_expected_tail = pie_native::slice_from_u64(
        expected_tails.data(), expected_tails.size());
    view.channel_ticket_indptr =
        pie_native::slice_from_u32(ticket_indptr.data(), ticket_indptr.size());
    view.ptir_row_counts = pie_native::slice_from_u32(
        unit_extents.data(), unit_extents.size());
    view.ptir_token_counts = pie_native::slice_from_u32(
        unit_extents.data(), unit_extents.size());
    expect(
        dispatch.validate_launch(view, &error) == PIE_STATUS_OK,
        "validate structured-mask golden: " + error);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    expect(
        dispatch.run(
            view, nullptr, 0, stream, nullptr, PieCompletion{}),
        "run structured-mask golden");
    cudaDeviceSynchronize();
    const auto stats = dispatch.stats();
    expect(
        stats.grouped_lanes == lane_count &&
            stats.generated_fused_groups == 1 &&
            stats.generated_fused_body_launches == 1,
        "structured-mask golden uses one generated fused region");
    const std::vector<std::vector<std::uint8_t>> expected{
        {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1},
        {0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1},
        {1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1},
        {1, 1, 1, 0, 0, 1, 1, 1},
    };
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        const auto outcome =
            std::atomic_ref<std::uint32_t>(terminals[lane].outcome)
                .load(std::memory_order_acquire);
        expect(
            outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
            "structured-mask grouped lane commits");
        for (std::uint32_t output = 0; output < expected.size(); ++output) {
            expect(
                read_packed_bool(endpoints[lane][output + 3],
                                 expected[output].size()) ==
                    expected[output],
                "structured-mask authoritative output parity");
        }
    }
    cudaStreamDestroy(stream);
}

enum class WriteBoundCase {
    None,
    BelowLower,
    Sentinel,
};

void run_declared_phase_case(
    const std::string& golden_directory,
    bool fail_after_first_layer,
    WriteBoundCase write_bound_case = WriteBoundCase::None) {
    constexpr std::uint64_t no_ticket =
        std::numeric_limits<std::uint64_t>::max();
    const std::string path =
        golden_directory + "/staged_dispatch.txt";
    const auto container_bytes =
        hex_bytes(golden_field(path, "container"));
    const auto sidecar_bytes =
        hex_bytes(golden_field(path, "sidecar"));
    const auto hash =
        std::stoull(golden_field(path, "hash"), nullptr, 16);
    pie_native::ptir::container::Container container;
    pie_native::ptir::container::DecodeError decode_error;
    expect(
        pie_native::ptir::container::decode(
            container_bytes.data(), container_bytes.size(),
            container, &decode_error),
        "decode staged fixture: " + decode_error.detail);
    expect(
        container.stages.size() == 4 &&
            container.stages[0].stage == PTIR_STAGE_PROLOGUE &&
            container.stages[1].stage == PTIR_STAGE_ON_ATTN_PROJ &&
            container.stages[2].stage == PTIR_STAGE_ON_ATTN &&
            container.stages[3].stage == PTIR_STAGE_EPILOGUE,
        "staged fixture preserves all declared phase identities");

    std::string error;
    {
        Dispatch unsupported;
        expect(
            unsupported.register_program(
                hash,
                {container_bytes.data(), container_bytes.size()},
                {sidecar_bytes.data(), sidecar_bytes.size()},
                &error) == PIE_STATUS_UNSUPPORTED,
            "model without attention hook coverage rejects registration");
    }
    Dispatch dispatch;
    dispatch.set_attention_hook_coverage(true, 2);
    error.clear();
    expect(
        dispatch.register_program(
            hash,
            {container_bytes.data(), container_bytes.size()},
            {sidecar_bytes.data(), sidecar_bytes.size()},
            &error) == PIE_STATUS_OK,
        "register staged program: " + error);

    std::vector<PieChannelEndpointBinding> endpoints(
        container.channels.size());
    std::vector<std::uint64_t> channel_ids(container.channels.size());
    for (std::size_t dense = 0; dense < container.channels.size(); ++dense) {
        const auto& source = container.channels[dense];
        channel_ids[dense] =
            (fail_after_first_layer ? 200000 : 100000) + dense;
        PieChannelDesc descriptor{};
        descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
        descriptor.channel_id = channel_ids[dense];
        descriptor.shape = {source.shape.dims, source.shape.rank};
        descriptor.dtype = source.dtype;
        descriptor.host_role = source.host_role;
        descriptor.seeded = source.seeded;
        descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
        descriptor.capacity = source.capacity;
        descriptor.reader_wait_id = channel_ids[dense] * 2 + 1;
        descriptor.writer_wait_id = channel_ids[dense] * 2 + 2;
        expect(
            dispatch.register_channel(
                descriptor, &endpoints[dense], &error) ==
                PIE_STATUS_OK,
            "register staged channel: " + error);
    }

    const std::uint32_t source_token =
        write_bound_case == WriteBoundCase::Sentinel
            ? std::numeric_limits<std::uint32_t>::max()
            : write_bound_case == WriteBoundCase::BelowLower ? 0u : 2u;
    const std::uint32_t pages[2] = {0, 0};
    const std::uint32_t page_ptr[2] = {0, 1};
    const std::uint32_t one = 1;
    const std::uint32_t zero = 0;
    const float zero_f32 = 0.0f;
    const PieChannelValueDesc seed_values[] = {
        {channel_ids[0],
         {reinterpret_cast<const std::uint8_t*>(&source_token),
          sizeof(source_token)}},
        {channel_ids[2],
         {reinterpret_cast<const std::uint8_t*>(pages), sizeof(pages)}},
        {channel_ids[3],
         {reinterpret_cast<const std::uint8_t*>(page_ptr),
          sizeof(page_ptr)}},
        {channel_ids[4],
         {reinterpret_cast<const std::uint8_t*>(&one), sizeof(one)}},
        {channel_ids[5],
         {reinterpret_cast<const std::uint8_t*>(&zero), sizeof(zero)}},
        {channel_ids[6],
         {reinterpret_cast<const std::uint8_t*>(&zero), sizeof(zero)}},
        {channel_ids[7],
         {reinterpret_cast<const std::uint8_t*>(&zero), sizeof(zero)}},
        {channel_ids[8],
         {reinterpret_cast<const std::uint8_t*>(&zero), sizeof(zero)}},
        {channel_ids[9],
         {reinterpret_cast<const std::uint8_t*>(&zero_f32),
          sizeof(zero_f32)}},
    };
    const std::uint64_t instance_id =
        fail_after_first_layer ? 2000 : 1000;
    PieInstanceBinding binding{};
    // Classify-once: the staged program IS the decode envelope; the
    // fixed-decode composer refuses host-classified lanes (the runtime
    // ships the class on the wire and the driver trusts it after the
    // bind-time verification below).
    expect(
        dispatch.bind_instance(
            instance_id, hash, PIE_GEOMETRY_CLASS_DECODE_ENVELOPE,
            instance_id + 1, channel_ids,
            std::vector<PieChannelValueDesc>(
                std::begin(seed_values), std::end(seed_values)),
            &binding, &error) == PIE_STATUS_OK,
        "bind staged instance: " + error);
    publish_bytes(endpoints[10], &one, sizeof(one));

    const std::uint64_t hashes[] = {hash};
    const std::uint64_t instances[] = {instance_id};
    PieTerminalCell terminal{};
    PieTerminalCell* terminals[] = {&terminal};
    const std::uint32_t row_attribution[] = {0, 0};
    const std::uint32_t ticket_indptr[] = {
        0, static_cast<std::uint32_t>(container.channels.size())};
    const std::uint64_t expected_heads[] = {
        0, 0, no_ticket, no_ticket, no_ticket,
        0, 0, 0, 0, 0, 0,
        no_ticket, no_ticket, no_ticket, no_ticket,
    };
    const std::uint64_t expected_tails[] = {
        no_ticket, 0, no_ticket, no_ticket, no_ticket,
        no_ticket, no_ticket, 1, 1, 1, no_ticket,
        0, 0, 0, 0,
    };
    pie_native::LaunchView view{};
    view.ptir_program_hashes =
        pie_native::slice_from_u64(hashes, 1);
    view.ptir_program_instances =
        pie_native::slice_from_u64(instances, 1);
    view.terminal_cells =
        pie_native::slice_from(terminals, 1);
    view.ptir_program_row_indptr =
        pie_native::slice_from_u32(row_attribution, 2);
    view.channel_ticket_indptr =
        pie_native::slice_from_u32(ticket_indptr, 2);
    view.channel_expected_head =
        pie_native::slice_from_u64(
            expected_heads, std::size(expected_heads));
    view.channel_expected_tail =
        pie_native::slice_from_u64(
            expected_tails, std::size(expected_tails));
    const std::uint32_t translation[] = {0, 1, 2, 3};
    const std::uint32_t translation_indptr[] = {0, 4};
    view.kv_translation =
        pie_native::slice_from_u32(
            translation, std::size(translation));
    view.kv_translation_indptr =
        pie_native::slice_from_u32(
            translation_indptr, std::size(translation_indptr));
    const std::uint32_t fixed_position[] = {0};
    view.position_ids = pie_native::slice_from_u32(
        fixed_position, std::size(fixed_position));
    const std::uint64_t write_lower_bounds[] = {1};
    const std::uint64_t write_upper_bounds[] = {
        std::numeric_limits<std::uint64_t>::max()};
    if (write_bound_case != WriteBoundCase::None) {
        view.ptir_kv_write_lower_bounds = pie_native::slice_from_u64(
            write_lower_bounds, std::size(write_lower_bounds));
        view.ptir_kv_write_upper_bounds = pie_native::slice_from_u64(
            write_upper_bounds, std::size(write_upper_bounds));
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    auto launch = dispatch.begin(view, stream);
    std::uint32_t* fixed_tokens = nullptr;
    std::uint32_t* fixed_positions = nullptr;
    std::uint32_t* fixed_qo = nullptr;
    std::uint32_t* fixed_pages = nullptr;
    std::uint32_t* fixed_page_indptr = nullptr;
    std::uint32_t* fixed_last_page_lens = nullptr;
    std::uint32_t* fixed_w_page = nullptr;
    std::uint32_t* fixed_w_off = nullptr;
    std::uint8_t* fixed_valid = nullptr;
    cudaMalloc(&fixed_tokens, sizeof(std::uint32_t));
    cudaMalloc(&fixed_positions, sizeof(std::uint32_t));
    cudaMalloc(&fixed_qo, 2 * sizeof(std::uint32_t));
    cudaMalloc(&fixed_pages, 4 * sizeof(std::uint32_t));
    cudaMalloc(&fixed_page_indptr, 2 * sizeof(std::uint32_t));
    cudaMalloc(&fixed_last_page_lens, sizeof(std::uint32_t));
    cudaMalloc(&fixed_w_page, sizeof(std::uint32_t));
    cudaMalloc(&fixed_w_off, sizeof(std::uint32_t));
    cudaMalloc(&fixed_valid, sizeof(std::uint8_t));
    const pie_cuda_driver::pipeline::FixedDecodeDeviceBuffers fixed_buffers{
        .token_ids = fixed_tokens,
        .position_ids = fixed_positions,
        .qo_indptr = fixed_qo,
        .kv_page_indices = fixed_pages,
        .kv_page_indptr = fixed_page_indptr,
        .kv_last_page_lens = fixed_last_page_lens,
        .w_page = fixed_w_page,
        .w_off = fixed_w_off,
        .row_valid = fixed_valid,
        .token_capacity = 1,
        .request_capacity = 1,
        .page_capacity = 4,
        .dummy_page = 3,
    };
    expect(
        dispatch.enqueue_fixed_decode(
            view, 16, 4, fixed_buffers, &error, *launch),
        "fixed decode composes staged geometry on device: " + error);
    cudaStreamSynchronize(stream);
    std::uint32_t fixed_token = 0;
    std::uint8_t fixed_row_valid = 0;
    cudaMemcpy(
        &fixed_token, fixed_tokens, sizeof(fixed_token),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        &fixed_row_valid, fixed_valid, sizeof(fixed_row_valid),
        cudaMemcpyDeviceToHost);
    if (write_bound_case != WriteBoundCase::None) {
        expect(
            fixed_token == 0 && fixed_row_valid == 0,
            write_bound_case == WriteBoundCase::BelowLower
                ? "fixed decode rejects a write below the declaration lower bound"
                : "fixed decode keeps a sentinel row inactive");
        dispatch.abort(*launch, stream);
        auto envelope_launch = dispatch.begin(view, stream);

        std::uint32_t* envelope_tokens = nullptr;
        std::uint32_t* envelope_positions = nullptr;
        std::uint32_t* envelope_pages = nullptr;
        std::uint32_t* envelope_page_indptr = nullptr;
        std::uint32_t* envelope_last_page_lens = nullptr;
        std::uint8_t* envelope_valid = nullptr;
        cudaMalloc(&envelope_tokens, sizeof(std::uint32_t));
        cudaMalloc(&envelope_positions, sizeof(std::uint32_t));
        cudaMalloc(&envelope_pages, sizeof(std::uint32_t));
        cudaMalloc(&envelope_page_indptr, 2 * sizeof(std::uint32_t));
        cudaMalloc(&envelope_last_page_lens, sizeof(std::uint32_t));
        cudaMalloc(&envelope_valid, sizeof(std::uint8_t));
        const pie_cuda_driver::pipeline::DecodeEnvelopeDeviceBuffers
            envelope_buffers{
                .token_ids = envelope_tokens,
                .position_ids = envelope_positions,
                .kv_page_indices = envelope_pages,
                .kv_page_indptr = envelope_page_indptr,
                .kv_last_page_lens = envelope_last_page_lens,
                .row_valid = envelope_valid,
                .dummy_page = 3,
                .page_size = 16,
            };
        const std::uint32_t program_starts[] = {0};
        const std::uint32_t template_page_indptr[] = {0, 1};
        const std::uint32_t template_page = 0;
        auto enqueue_envelope = [&](auto& staged_launch) {
            cudaMemcpyAsync(
                envelope_pages, &template_page, sizeof(template_page),
                cudaMemcpyHostToDevice, stream);
            expect(
                dispatch.enqueue_decode_envelopes(
                    view,
                    std::span<const std::uint32_t>(program_starts, 1),
                    std::span<const std::uint32_t>(program_starts, 1),
                    std::span<const std::uint32_t>(
                        template_page_indptr, 2),
                    envelope_buffers, &error, staged_launch),
                "decode envelope composes staged geometry on device: " +
                    error);
        };

        enqueue_envelope(*envelope_launch);
        cudaStreamSynchronize(stream);
        std::uint8_t envelope_row_valid = 0;
        cudaMemcpy(
            &envelope_row_valid, envelope_valid,
            sizeof(envelope_row_valid), cudaMemcpyDeviceToHost);
        expect(
            envelope_row_valid == 0,
            write_bound_case == WriteBoundCase::BelowLower
                ? "decode envelope rejects a write below the declaration lower bound"
                : "decode envelope keeps a sentinel row inactive");

        const DispatchStats containment_stats = dispatch.stats();
        if (write_bound_case == WriteBoundCase::BelowLower) {
            expect(
                containment_stats.fixed_decode_chain_kills == 1 &&
                    containment_stats.decode_envelope_chain_kills == 1,
                "both CUDA composers fail-stop a live row below the lower bound "
                "(fixed=" +
                    std::to_string(
                        containment_stats.fixed_decode_chain_kills) +
                    ", envelope=" +
                    std::to_string(
                        containment_stats.decode_envelope_chain_kills) +
                    ")");
        } else {
            expect(
                containment_stats.fixed_decode_chain_kills == 0 &&
                    containment_stats.decode_envelope_chain_kills == 0,
                "sentinel rows bypass both CUDA lower-bound checks");
        }
        dispatch.abort(*envelope_launch, stream);

        cudaFree(envelope_tokens);
        cudaFree(envelope_positions);
        cudaFree(envelope_pages);
        cudaFree(envelope_page_indptr);
        cudaFree(envelope_last_page_lens);
        cudaFree(envelope_valid);
        cudaFree(fixed_tokens);
        cudaFree(fixed_positions);
        cudaFree(fixed_qo);
        cudaFree(fixed_pages);
        cudaFree(fixed_page_indptr);
        cudaFree(fixed_last_page_lens);
        cudaFree(fixed_w_page);
        cudaFree(fixed_w_off);
        cudaFree(fixed_valid);
        cudaStreamDestroy(stream);
        return;
    }
    expect(
        fixed_token == source_token && fixed_row_valid == 1,
        "fixed decode preserves the staged token and validity");
    cudaFree(fixed_tokens);
    cudaFree(fixed_positions);
    cudaFree(fixed_qo);
    cudaFree(fixed_pages);
    cudaFree(fixed_page_indptr);
    cudaFree(fixed_last_page_lens);
    cudaFree(fixed_w_page);
    cudaFree(fixed_w_off);
    cudaFree(fixed_valid);
    pie_native::ptir::ResolvedPrograms resolved;
    expect(
        dispatch.resolve_descriptors(
            view, 16, 4, resolved, &error, false, launch.get()) &&
            resolved.per_program.size() == 1 &&
            resolved.per_program[0].token_ids ==
                std::vector<std::uint32_t>{source_token},
        "prologue pending geometry resolves into the same forward: " +
            error);
    const DispatchStats descriptor_stats = dispatch.stats();
    expect(
        descriptor_stats.descriptor_readback_batches == 1 &&
            descriptor_stats.descriptor_readback_cells != 0 &&
            descriptor_stats.descriptor_readback_bytes != 0,
        "descriptor resolution uses one packed readback");

    const std::uint32_t sample_ptr[] = {0, 1};
    const std::uint32_t sample_start[] = {0};
    const std::uint32_t unit[] = {1};
    view.sampling_indptr =
        pie_native::slice_from_u32(sample_ptr, 2);
    view.ptir_sample_starts =
        pie_native::slice_from_u32(sample_start, 1);
    view.ptir_sample_counts =
        pie_native::slice_from_u32(unit, 1);
    view.ptir_row_counts = pie_native::slice_from_u32(unit, 1);
    view.ptir_token_counts = pie_native::slice_from_u32(unit, 1);
    view.ptir_kv_lens = pie_native::slice_from_u32(unit, 1);
    view.ptir_page_counts = pie_native::slice_from_u32(unit, 1);
    view.ptir_query_lens = pie_native::slice_from_u32(unit, 1);
    view.ptir_key_lens = pie_native::slice_from_u32(unit, 1);
    dispatch.update_launch_geometry(
        *launch, view, std::span<const std::uint32_t>(sample_start, 1));

    const float query_values[2][4] = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {10.0f, 20.0f, 30.0f, 40.0f},
    };
    std::uint16_t query_bf16[2][4]{};
    for (std::size_t layer = 0; layer < 2; ++layer) {
        for (std::size_t value = 0; value < 4; ++value) {
            query_bf16[layer][value] =
                bf16_bits(query_values[layer][value]);
        }
    }
    std::uint16_t* device_query = nullptr;
    cudaMalloc(&device_query, sizeof(query_bf16));
    cudaMemcpy(
        device_query, query_bf16, sizeof(query_bf16),
        cudaMemcpyHostToDevice);
    float* device_query_f32 = nullptr;
    cudaMalloc(&device_query_f32, sizeof(query_values[1]));
    cudaMemcpy(
        device_query_f32, query_values[1], sizeof(query_values[1]),
        cudaMemcpyHostToDevice);
    const std::uint32_t layers_to_run =
        fail_after_first_layer ? 1 : 2;
    for (std::uint32_t layer = 0; layer < layers_to_run; ++layer) {
        dispatch.execute_attention_phase(
            *launch, PTIR_STAGE_ON_ATTN_PROJ,
            layer == 0
                ? static_cast<const void*>(device_query)
                : static_cast<const void*>(device_query_f32),
            1, 4, layer, stream, layer != 0);
        dispatch.execute_attention_phase(
            *launch, PTIR_STAGE_ON_ATTN,
            layer == 0
                ? static_cast<const void*>(device_query)
                : static_cast<const void*>(device_query_f32),
            1, 4, layer, stream, layer != 0);
    }

    const float logits_host[4] = {0.0f, 1.0f, 9.0f, 2.0f};
    float* device_logits = nullptr;
    cudaMalloc(&device_logits, sizeof(logits_host));
    cudaMemcpy(
        device_logits, logits_host, sizeof(logits_host),
        cudaMemcpyHostToDevice);
    if (fail_after_first_layer) {
        bool rejected = false;
        try {
            dispatch.finish(
                *launch, view, device_logits, 4, stream,
                nullptr, PieCompletion{});
        } catch (const std::exception&) {
            rejected = true;
            dispatch.abort(*launch, stream);
        }
        cudaDeviceSynchronize();
        expect(rejected, "missing declared layer invocation rejects finish");
        for (std::size_t channel = 11; channel <= 14; ++channel) {
            auto* words = reinterpret_cast<std::uint64_t*>(
                endpoints[channel].word_base);
            expect(
                std::atomic_ref<std::uint64_t>(
                    words[endpoints[channel].tail_word_index])
                        .load(std::memory_order_acquire) == 0,
                "failed staged pass publishes no intermediate effects");
        }
    } else {
        expect(
            dispatch.finish(
                *launch, view, device_logits, 4, stream,
                nullptr, PieCompletion{}),
            "finish declared-phase launch");
        cudaDeviceSynchronize();
        expect(
            std::atomic_ref<std::uint32_t>(terminal.outcome).load(
                std::memory_order_acquire) ==
                PIE_TERMINAL_OUTCOME_SUCCESS,
            "declared-phase launch commits once");
        expect(
            *reinterpret_cast<const std::uint32_t*>(
                endpoints[11].mirror_base) == 2 &&
            *reinterpret_cast<const std::uint32_t*>(
                endpoints[12].mirror_base) == 2 &&
            std::abs(
                *reinterpret_cast<const float*>(
                    endpoints[13].mirror_base) -
                110.0f) < 1e-5f &&
            *reinterpret_cast<const std::uint32_t*>(
                endpoints[14].mirror_base) == 2,
            "Query/Layer run at layers {0,1} and epilogue publishes once");
    }
    cudaFree(device_logits);
    cudaFree(device_query_f32);
    cudaFree(device_query);
    cudaStreamDestroy(stream);
}

std::vector<std::uint8_t> run_beam_case(
    const std::string& golden_directory,
    bool direct_bf16,
    std::uint64_t* body_launches) {
    constexpr std::uint32_t lane_count = 2;
    constexpr std::uint32_t rows = 2;
    constexpr std::uint32_t vocab = 8;
    constexpr std::uint64_t no_ticket =
        std::numeric_limits<std::uint64_t>::max();
    const std::string path = golden_directory + "/beam_epilogue.txt";
    const auto container_bytes =
        hex_bytes(golden_field(path, "container"));
    const auto sidecar_bytes = hex_bytes(golden_field(path, "sidecar"));
    const auto hash = std::stoull(golden_field(path, "hash"), nullptr, 16);
    pie_native::ptir::container::Container container;
    pie_native::ptir::container::DecodeError decode_error;
    expect(
        pie_native::ptir::container::decode(
            container_bytes.data(), container_bytes.size(),
            container, &decode_error),
        "decode beam fixture: " + decode_error.detail);
    expect(container.channels.size() == 16, "beam channel count");
    pie_native::ptir::bound::Bound bound;
    std::string plan_error;
    expect(
        pie_native::ptir::bound::parse_sidecar(
            sidecar_bytes.data(), sidecar_bytes.size(), bound, &plan_error) &&
            bound.plans.size() == 1,
        "beam fixture carries one compiler region plan: " + plan_error);
    pie_native::ptir::plan::StagePlan decoded_plan;
    if (!bound.plans.empty()) {
        expect(
            pie_native::ptir::plan::decode(
                bound.plans[0].bytes.data(),
                bound.plans[0].bytes.size(),
                decoded_plan,
                &plan_error),
            "decode beam compiler region plan: " + plan_error);
    }
    std::size_t topk_regions = 0;
    for (const auto& region : decoded_plan.fused.regions) {
        if (!region.library ||
            region.library_op != PTIR_LIBRARY_TOP_K) {
            continue;
        }
        ++topk_regions;
        expect(
            region.nodes.size() == 1 &&
                decoded_plan.ops[region.nodes.front()].op.tag ==
                    PTIR_OP_TOP_K,
            "beam workload has an opcode-derived TopK library cut");
    }
    const auto has_opcode = [&](std::uint8_t tag) {
        return std::any_of(
            decoded_plan.ops.begin(), decoded_plan.ops.end(),
            [tag](const auto& normalized) {
                return normalized.op.tag == tag;
            });
    };
    expect(
        topk_regions == 1 &&
            has_opcode(PTIR_OP_DIV) &&
            has_opcode(PTIR_OP_REM) &&
            has_opcode(PTIR_OP_GATHER),
        "beam workload keeps index arithmetic and gathers as generic SSA");
    std::vector<std::uint8_t> consumes(container.channels.size(), 0);
    std::vector<std::uint8_t> publishes(container.channels.size(), 0);
    for (const auto& op : container.stages[0].ops) {
        if (op.tag == PTIR_OP_CHAN_TAKE) {
            consumes[static_cast<std::size_t>(op.chan)] = 1;
        } else if (op.tag == PTIR_OP_CHAN_PUT) {
            publishes[static_cast<std::size_t>(op.chan)] = 1;
        }
    }

    Dispatch dispatch;
    std::string error;
    expect(
        dispatch.register_program(
            hash,
            {container_bytes.data(), container_bytes.size()},
            {sidecar_bytes.data(), sidecar_bytes.size()},
            &error) == PIE_STATUS_OK,
        "register beam program: " + error);
    std::vector<std::uint64_t> hashes(lane_count, hash);
    std::vector<std::uint64_t> instances(lane_count);
    std::vector<PieTerminalCell> terminals(lane_count);
    std::vector<PieTerminalCell*> terminal_ptrs(lane_count);
    std::vector<std::vector<PieChannelEndpointBinding>> endpoints(lane_count);
    std::vector<std::uint64_t> expected_heads;
    std::vector<std::uint64_t> expected_tails;
    std::vector<std::uint32_t> ticket_indptr{0};
    std::vector<std::uint32_t> sampling_indptr{0};
    std::vector<std::uint32_t> row_extents(lane_count, rows);
    std::vector<std::uint32_t> unit_extents(lane_count, 1);

    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        endpoints[lane].resize(container.channels.size());
        std::vector<std::uint64_t> ids(container.channels.size());
        for (std::size_t dense = 0; dense < container.channels.size(); ++dense) {
            const auto& source = container.channels[dense];
            const std::uint64_t id =
                70000 + static_cast<std::uint64_t>(lane) * 100 + dense;
            ids[dense] = id;
            PieChannelDesc descriptor{};
            descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
            descriptor.channel_id = id;
            descriptor.shape = {source.shape.dims, source.shape.rank};
            descriptor.dtype = source.dtype;
            descriptor.host_role = source.host_role;
            descriptor.seeded = source.seeded;
            descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
            descriptor.capacity = source.capacity;
            descriptor.reader_wait_id = id * 2 + 1;
            descriptor.writer_wait_id = id * 2 + 2;
            expect(
                dispatch.register_channel(
                    descriptor, &endpoints[lane][dense], &error) ==
                    PIE_STATUS_OK,
                "register beam channel: " + error);
        }

        std::vector<std::vector<std::uint8_t>> payloads(12);
        auto set_payload = [&](std::size_t dense, const auto& values) {
            payloads[dense].resize(sizeof(values));
            std::memcpy(payloads[dense].data(), &values, sizeof(values));
        };
        const std::uint32_t c0[6] = {5, 6, 0, 5, 6, 0};
        const std::uint32_t c1[6] = {4, 2, 0, 4, 2, 0};
        const std::uint32_t c2[2] = {6, 6};
        const std::uint8_t c3[24] = {
            1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        };
        const std::uint32_t c4[2] = {6, 6};
        const std::uint32_t c5[2] = {2, 2};
        const std::uint32_t c6[2] = {6, 6};
        const std::uint32_t c7[2] = {2, 2};
        const std::uint32_t c8[2] = {6, 6};
        const std::uint32_t c9[2] = {2, 2};
        const std::int32_t c10[2] = {1, 2};
        const float c11[2] = {0.0f, 0.0f};
        set_payload(0, c0);
        set_payload(1, c1);
        set_payload(2, c2);
        set_payload(3, c3);
        set_payload(4, c4);
        set_payload(5, c5);
        set_payload(6, c6);
        set_payload(7, c7);
        set_payload(8, c8);
        set_payload(9, c9);
        set_payload(10, c10);
        set_payload(11, c11);
        std::vector<PieChannelValueDesc> seeds;
        for (std::size_t dense = 0; dense < payloads.size(); ++dense) {
            seeds.push_back({
                ids[dense],
                {payloads[dense].data(), payloads[dense].size()},
            });
        }
        instances[lane] = 2000 + lane;
        PieInstanceBinding binding{};
        expect(
            dispatch.bind_instance(
                instances[lane],
                hash,
                PIE_GEOMETRY_CLASS_HOST,
                6000 + lane,
                ids,
                seeds,
                &binding,
                &error) == PIE_STATUS_OK,
            "bind beam instance: " + error);
        const std::uint32_t host_input[2] = {7, 8};
        publish_bytes(
            endpoints[lane][12], host_input, sizeof(host_input));
        terminal_ptrs[lane] = &terminals[lane];
        for (std::size_t dense = 0; dense < container.channels.size(); ++dense) {
            const bool consumes_or_replaces_seed =
                consumes[dense] ||
                (container.channels[dense].seeded && publishes[dense]);
            expected_heads.push_back(
                consumes_or_replaces_seed ? 0 : no_ticket);
            expected_tails.push_back(
                publishes[dense]
                    ? (container.channels[dense].seeded ? 1 : 0)
                    : no_ticket);
        }
        ticket_indptr.push_back(
            static_cast<std::uint32_t>(expected_heads.size()));
        sampling_indptr.push_back((lane + 1) * rows);
    }

    const float one_lane[rows * vocab] = {
        0, 0, 0, 8, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 7, 0, 0,
    };
    std::vector<float> logits;
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        logits.insert(logits.end(), std::begin(one_lane), std::end(one_lane));
    }
    float* device_logits = nullptr;
    std::uint16_t* device_bf16 = nullptr;
    std::vector<std::uint32_t> direct_rows(lane_count * rows);
    if (direct_bf16) {
        std::vector<std::uint16_t> bf16(logits.size());
        std::transform(
            logits.begin(), logits.end(), bf16.begin(), bf16_bits);
        std::iota(direct_rows.begin(), direct_rows.end(), 0u);
        cudaMalloc(
            &device_bf16, bf16.size() * sizeof(std::uint16_t));
        cudaMemcpy(
            device_bf16, bf16.data(),
            bf16.size() * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice);
    } else {
        cudaMalloc(&device_logits, logits.size() * sizeof(float));
        cudaMemcpy(
            device_logits, logits.data(), logits.size() * sizeof(float),
            cudaMemcpyHostToDevice);
    }
    pie_native::LaunchView view{};
    view.terminal_cells =
        pie_native::slice_from(terminal_ptrs.data(), terminal_ptrs.size());
    view.ptir_program_hashes =
        pie_native::slice_from_u64(hashes.data(), hashes.size());
    view.ptir_program_instances =
        pie_native::slice_from_u64(instances.data(), instances.size());
    view.sampling_indptr =
        pie_native::slice_from_u32(sampling_indptr.data(), sampling_indptr.size());
    view.channel_expected_head = pie_native::slice_from_u64(
        expected_heads.data(), expected_heads.size());
    view.channel_expected_tail = pie_native::slice_from_u64(
        expected_tails.data(), expected_tails.size());
    view.channel_ticket_indptr =
        pie_native::slice_from_u32(ticket_indptr.data(), ticket_indptr.size());
    view.ptir_row_counts = pie_native::slice_from_u32(
        row_extents.data(), row_extents.size());
    view.ptir_token_counts = pie_native::slice_from_u32(
        unit_extents.data(), unit_extents.size());
    view.ptir_kv_lens = pie_native::slice_from_u32(
        unit_extents.data(), unit_extents.size());
    view.ptir_page_counts = pie_native::slice_from_u32(
        unit_extents.data(), unit_extents.size());
    view.ptir_query_lens = pie_native::slice_from_u32(
        unit_extents.data(), unit_extents.size());
    view.ptir_key_lens = pie_native::slice_from_u32(
        unit_extents.data(), unit_extents.size());
    expect(
        dispatch.validate_launch(view, &error) == PIE_STATUS_OK,
        "validate beam launch: " + error);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    expect(
        dispatch.run(
            view, device_logits, vocab, stream, nullptr, PieCompletion{},
            device_bf16,
            direct_bf16 ? direct_rows.data() : nullptr),
        "run beam launch");
    cudaDeviceSynchronize();
    const auto stats = dispatch.stats();
    if (body_launches != nullptr) {
        *body_launches = stats.grouped_body_op_launches;
    }
    expect(
        stats.grouped_lanes == lane_count &&
            stats.selection_library_groups == 1 &&
            stats.generated_fused_groups == 1 &&
            stats.generated_fused_body_launches > 1,
        "beam workload composes generated regions with the TopK library");
    std::vector<std::uint8_t> attribution;
    auto append = [&](const void* data, std::size_t bytes) {
        const auto* begin = static_cast<const std::uint8_t*>(data);
        attribution.insert(attribution.end(), begin, begin + bytes);
    };
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        const auto outcome =
            std::atomic_ref<std::uint32_t>(terminals[lane].outcome)
                .load(std::memory_order_acquire);
        expect(
            outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
            "beam lane attribution");
        const auto* tokens = reinterpret_cast<const std::int32_t*>(
            endpoints[lane][13].mirror_base);
        const auto* parents = reinterpret_cast<const std::uint32_t*>(
            endpoints[lane][14].mirror_base);
        const auto* values = reinterpret_cast<const float*>(
            endpoints[lane][15].mirror_base);
        const std::uint32_t topk_indices[rows] = {3, 13};
        expect(
            tokens[0] == static_cast<std::int32_t>(
                             topk_indices[0] % vocab) &&
                tokens[1] == static_cast<std::int32_t>(
                                 topk_indices[1] % vocab),
            "beam tokens equal generic TopK remainder reference");
        expect(
            parents[0] == topk_indices[0] / vocab &&
                parents[1] == topk_indices[1] / vocab,
            "beam parents equal generic TopK division reference");
        expect(
            std::abs(values[0] - -0.0023454318f) < 1e-7f &&
                std::abs(values[1] - -0.0063628945f) < 1e-7f,
            "beam values");
        append(tokens, rows * sizeof(*tokens));
        append(parents, rows * sizeof(*parents));
        append(values, rows * sizeof(*values));
    }
    cudaStreamDestroy(stream);
    cudaFree(device_logits);
    cudaFree(device_bf16);
    return attribution;
}

void run_mtp_direct_case(const std::string& golden_directory) {
    constexpr std::uint32_t lane_count = 2;
    constexpr std::uint32_t target_rows = 4;
    constexpr std::uint32_t draft_rows = 3;
    constexpr std::uint32_t vocab = 8;
    constexpr std::uint32_t logits_stride = 10;
    constexpr std::uint64_t no_ticket =
        std::numeric_limits<std::uint64_t>::max();
    const std::string path =
        golden_directory + "/mtp_verify_tail.txt";
    const auto container_bytes =
        hex_bytes(golden_field(path, "container"));
    const auto sidecar_bytes =
        hex_bytes(golden_field(path, "sidecar"));
    const auto hash =
        std::stoull(golden_field(path, "hash"), nullptr, 16);
    pie_native::ptir::container::Container container;
    pie_native::ptir::container::DecodeError decode_error;
    expect(
        pie_native::ptir::container::decode(
            container_bytes.data(), container_bytes.size(),
            container, &decode_error),
        "decode MtpLogits fixture: " + decode_error.detail);
    Dispatch dispatch;
    std::string error;
    expect(
        dispatch.register_program(
            hash,
            {container_bytes.data(), container_bytes.size()},
            {sidecar_bytes.data(), sidecar_bytes.size()},
            &error) == PIE_STATUS_OK,
        "register MtpLogits fixture: " + error);

    std::vector<std::uint64_t> hashes(lane_count, hash);
    std::vector<std::uint64_t> instances(lane_count);
    std::vector<PieTerminalCell> terminals(lane_count);
    std::vector<PieTerminalCell*> terminal_ptrs(lane_count);
    std::vector<std::vector<PieChannelEndpointBinding>> endpoints(lane_count);
    std::vector<std::uint64_t> expected_heads;
    std::vector<std::uint64_t> expected_tails;
    std::vector<std::uint32_t> ticket_indptr{0};
    std::vector<std::uint32_t> sample_ptr{0};
    std::vector<std::uint32_t> extents(lane_count, target_rows);
    const std::int32_t draft_tokens[draft_rows] = {3, 5, 6};
    const std::uint8_t mask[(target_rows * vocab + 7) / 8] = {
        0xff, 0xff, 0x04, 0xff,
    };
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        endpoints[lane].resize(container.channels.size());
        std::vector<std::uint64_t> ids(container.channels.size());
        for (std::size_t channel = 0;
             channel < container.channels.size();
             ++channel) {
            const auto& source = container.channels[channel];
            ids[channel] = 90000 + lane * 16 + channel;
            PieChannelDesc descriptor{};
            descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
            descriptor.channel_id = ids[channel];
            descriptor.shape = {source.shape.dims, source.shape.rank};
            descriptor.dtype = source.dtype;
            descriptor.host_role = source.host_role;
            descriptor.seeded = source.seeded;
            descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
            descriptor.capacity = source.capacity;
            descriptor.reader_wait_id = ids[channel] * 2 + 1;
            descriptor.writer_wait_id = ids[channel] * 2 + 2;
            expect(
                dispatch.register_channel(
                    descriptor, &endpoints[lane][channel], &error) ==
                    PIE_STATUS_OK,
                "register MtpLogits channel: " + error);
        }
        const PieChannelValueDesc seed{
            ids[0],
            {
                reinterpret_cast<const std::uint8_t*>(draft_tokens),
                sizeof(draft_tokens),
            },
        };
        instances[lane] = 4000 + lane;
        PieInstanceBinding binding{};
        expect(
            dispatch.bind_instance(
                instances[lane], hash, PIE_GEOMETRY_CLASS_HOST,
                8000 + lane, ids, {seed},
                &binding, &error) == PIE_STATUS_OK,
            "bind MtpLogits instance: " + error);
        publish_bytes(endpoints[lane][1], mask, sizeof(mask));
        terminal_ptrs[lane] = &terminals[lane];
        expected_heads.insert(
            expected_heads.end(), {0, 0, no_ticket, no_ticket});
        expected_tails.insert(
            expected_tails.end(), {1, no_ticket, 0, 0});
        ticket_indptr.push_back(
            static_cast<std::uint32_t>(expected_heads.size()));
        sample_ptr.push_back((lane + 1) * target_rows);
    }

    const float target[target_rows * vocab] = {
        0,0,0,9,0,0,0,0,
        0,0,0,0,0,9,0,0,
        0,0,1,0,0,0,9,0,
        0,0,0,0,9,0,0,0,
    };
    const float drafts[draft_rows * vocab] = {
        0,7,0,0,0,0,0,0,
        0,0,0,0,7,0,0,0,
        7,0,0,0,0,0,0,0,
    };
    const std::uint32_t total_target_rows = lane_count * target_rows;
    const std::uint32_t total_rows =
        total_target_rows + lane_count * draft_rows;
    std::vector<std::uint16_t> bf16(
        static_cast<std::size_t>(total_rows) * logits_stride,
        bf16_bits(100.0f));
    std::vector<std::uint32_t> direct_rows(total_target_rows);
    std::iota(direct_rows.begin(), direct_rows.end(), 0u);
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        for (std::uint32_t index = 0;
             index < target_rows * vocab;
             ++index) {
            bf16[
                (static_cast<std::size_t>(lane) * target_rows +
                 index / vocab) *
                    logits_stride +
                index % vocab] = bf16_bits(target[index]);
        }
        for (std::uint32_t index = 0;
             index < draft_rows * vocab;
             ++index) {
            bf16[
                (static_cast<std::size_t>(total_target_rows) +
                 static_cast<std::size_t>(lane) * draft_rows +
                 index / vocab) *
                    logits_stride +
                index % vocab] = bf16_bits(drafts[index]);
        }
    }
    std::uint16_t* device_bf16 = nullptr;
    cudaMalloc(&device_bf16, bf16.size() * sizeof(std::uint16_t));
    cudaMemcpy(
        device_bf16, bf16.data(),
        bf16.size() * sizeof(std::uint16_t),
        cudaMemcpyHostToDevice);
    std::vector<std::uint32_t> draft_starts{
        total_target_rows,
        total_target_rows + draft_rows,
    };
    std::vector<std::uint32_t> draft_counts(
        lane_count, draft_rows);
    pie_native::LaunchView view{};
    view.terminal_cells =
        pie_native::slice_from(terminal_ptrs.data(), terminal_ptrs.size());
    view.ptir_program_hashes =
        pie_native::slice_from_u64(hashes.data(), hashes.size());
    view.ptir_program_instances =
        pie_native::slice_from_u64(instances.data(), instances.size());
    view.sampling_indptr =
        pie_native::slice_from_u32(sample_ptr.data(), sample_ptr.size());
    view.channel_expected_head = pie_native::slice_from_u64(
        expected_heads.data(), expected_heads.size());
    view.channel_expected_tail = pie_native::slice_from_u64(
        expected_tails.data(), expected_tails.size());
    view.channel_ticket_indptr =
        pie_native::slice_from_u32(ticket_indptr.data(), ticket_indptr.size());
    view.ptir_row_counts =
        pie_native::slice_from_u32(extents.data(), extents.size());
    view.ptir_token_counts =
        pie_native::slice_from_u32(extents.data(), extents.size());
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    expect(
        dispatch.run(
            view, nullptr, logits_stride, stream, nullptr, PieCompletion{},
            device_bf16, direct_rows.data(), draft_starts, draft_counts,
            total_rows),
        "run dedicated direct-BF16 MtpLogits layout");
    cudaDeviceSynchronize();
    const auto stats = dispatch.stats();
    expect(
        stats.grouped_lanes == lane_count &&
            stats.direct_bf16_groups == 1 &&
            stats.direct_bf16_solo_materializations == 0,
        "MtpLogits uses one grouped dedicated-row launch");
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        const auto* accepted = reinterpret_cast<const std::int32_t*>(
            endpoints[lane][2].mirror_base);
        const auto* next_drafts = reinterpret_cast<const std::int32_t*>(
            endpoints[lane][3].mirror_base);
        expect(
            accepted[0] == 3 && accepted[1] == 5 &&
                accepted[2] == 2 && accepted[3] == -1,
            "MtpLogits target-row attribution: [" +
                std::to_string(accepted[0]) + "," +
                std::to_string(accepted[1]) + "," +
                std::to_string(accepted[2]) + "," +
                std::to_string(accepted[3]) + "]");
        expect(
            next_drafts[0] == 1 && next_drafts[1] == 4 &&
                next_drafts[2] == 0,
            "MtpLogits dedicated draft-row attribution");
    }
    cudaStreamDestroy(stream);
    cudaFree(device_bf16);
}

void run_parallel_signature_case(const std::string& golden_directory) {
    constexpr std::uint64_t no_ticket =
        std::numeric_limits<std::uint64_t>::max();
    struct Program {
        std::uint64_t hash = 0;
        std::uint64_t instance = 0;
        pie_native::ptir::container::Container container;
        std::vector<PieChannelEndpointBinding> endpoints;
        std::vector<std::uint64_t> channel_ids;
    };
    Dispatch dispatch;
    std::string error;
    auto prepare = [&](const std::string& name,
                       std::uint64_t channel_base,
                       std::uint64_t instance,
                       std::uint32_t geometry_class,
                       const std::vector<PieChannelValueDesc>& seeds) {
        Program result;
        const std::string path =
            golden_directory + "/" + name + ".txt";
        const auto canonical =
            hex_bytes(golden_field(path, "container"));
        const auto sidecar =
            hex_bytes(golden_field(path, "sidecar"));
        result.hash =
            std::stoull(golden_field(path, "hash"), nullptr, 16);
        pie_native::ptir::container::DecodeError decode_error;
        expect(
            pie_native::ptir::container::decode(
                canonical.data(),
                canonical.size(),
                result.container,
                &decode_error),
            "parallel-signature container decode: " +
                decode_error.detail);
        expect(
            dispatch.register_program(
                result.hash,
                {canonical.data(), canonical.size()},
                {sidecar.data(), sidecar.size()},
                &error) == PIE_STATUS_OK,
            "parallel-signature program registration: " + error);
        result.instance = instance;
        result.endpoints.resize(result.container.channels.size());
        result.channel_ids.resize(result.container.channels.size());
        for (std::size_t channel = 0;
             channel < result.container.channels.size();
             ++channel) {
            const auto& source = result.container.channels[channel];
            result.channel_ids[channel] = channel_base + channel;
            PieChannelDesc descriptor{};
            descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
            descriptor.channel_id = result.channel_ids[channel];
            descriptor.shape = {source.shape.dims, source.shape.rank};
            descriptor.dtype = source.dtype;
            descriptor.host_role = source.host_role;
            descriptor.seeded = source.seeded;
            descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
            descriptor.capacity = source.capacity;
            descriptor.reader_wait_id =
                result.channel_ids[channel] * 2 + 1;
            descriptor.writer_wait_id =
                result.channel_ids[channel] * 2 + 2;
            expect(
                dispatch.register_channel(
                    descriptor,
                    &result.endpoints[channel],
                    &error) == PIE_STATUS_OK,
                "parallel-signature channel registration: " + error);
        }
        std::vector<PieChannelValueDesc> bound_seeds = seeds;
        for (auto& seed : bound_seeds) {
            seed.channel_id =
                result.channel_ids[seed.channel_id];
        }
        PieInstanceBinding binding{};
        expect(
            dispatch.bind_instance(
                instance,
                result.hash,
                geometry_class,
                instance + 100,
                result.channel_ids,
                bound_seeds,
                &binding,
                &error) == PIE_STATUS_OK,
            "parallel-signature instance binding: " + error);
        return result;
    };

    const std::int32_t greedy_seed = 1;
    const PieChannelValueDesc greedy_seeds[] = {{
        0,
        {
            reinterpret_cast<const std::uint8_t*>(&greedy_seed),
            sizeof(greedy_seed),
        },
    }};
    Program greedy = prepare(
        "greedy_argmax",
        120000,
        9100,
        PIE_GEOMETRY_CLASS_HOST,
        {std::begin(greedy_seeds), std::end(greedy_seeds)});
    const std::int32_t section_token = 1;
    const std::uint32_t section_length = 1;
    const std::uint32_t section_rng[2] = {1234, 0};
    const PieChannelValueDesc section_seeds[] = {
        {
            0,
            {
                reinterpret_cast<const std::uint8_t*>(&section_token),
                sizeof(section_token),
            },
        },
        {
            3,
            {
                reinterpret_cast<const std::uint8_t*>(&section_length),
                sizeof(section_length),
            },
        },
        {
            4,
            {
                reinterpret_cast<const std::uint8_t*>(section_rng),
                sizeof(section_rng),
            },
        },
    };
    Program section = prepare(
        "section3_masked_gumbel",
        121000,
        9200,
        PIE_GEOMETRY_CLASS_HOST,
        {std::begin(section_seeds), std::end(section_seeds)});
    publish_mask(section.endpoints[2]);

    std::vector<std::uint64_t> hashes{greedy.hash, section.hash};
    std::vector<std::uint64_t> instances{
        greedy.instance, section.instance};
    std::vector<PieTerminalCell> terminals(2);
    std::vector<PieTerminalCell*> terminal_ptrs{
        &terminals[0], &terminals[1]};
    std::vector<std::uint32_t> sampling_indptr{0, 1, 2};
    std::vector<std::uint64_t> expected_heads{
        0, no_ticket,
        0, no_ticket, 0, 0, 0,
    };
    std::vector<std::uint64_t> expected_tails{
        1, 0,
        1, 0, no_ticket, 1, 1,
    };
    std::vector<std::uint32_t> ticket_indptr{0, 2, 7};
    std::vector<std::uint32_t> extents{1, 1};
    pie_native::LaunchView view{};
    view.terminal_cells =
        pie_native::slice_from(terminal_ptrs.data(), terminal_ptrs.size());
    view.ptir_program_hashes =
        pie_native::slice_from_u64(hashes.data(), hashes.size());
    view.ptir_program_instances =
        pie_native::slice_from_u64(instances.data(), instances.size());
    view.sampling_indptr =
        pie_native::slice_from_u32(
            sampling_indptr.data(), sampling_indptr.size());
    view.channel_expected_head =
        pie_native::slice_from_u64(
            expected_heads.data(), expected_heads.size());
    view.channel_expected_tail =
        pie_native::slice_from_u64(
            expected_tails.data(), expected_tails.size());
    view.channel_ticket_indptr =
        pie_native::slice_from_u32(
            ticket_indptr.data(), ticket_indptr.size());
    view.ptir_row_counts =
        pie_native::slice_from_u32(extents.data(), extents.size());
    view.ptir_token_counts =
        pie_native::slice_from_u32(extents.data(), extents.size());
    view.ptir_kv_lens =
        pie_native::slice_from_u32(extents.data(), extents.size());
    view.ptir_page_counts =
        pie_native::slice_from_u32(extents.data(), extents.size());
    view.ptir_query_lens =
        pie_native::slice_from_u32(extents.data(), extents.size());
    view.ptir_key_lens =
        pie_native::slice_from_u32(extents.data(), extents.size());

    std::vector<float> logits(2 * 32, -100.0f);
    logits[2] = 100.0f;
    logits[32 + 7] = 100.0f;
    float* device_logits = nullptr;
    cudaMalloc(&device_logits, logits.size() * sizeof(float));
    cudaMemcpy(
        device_logits,
        logits.data(),
        logits.size() * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    expect(
        dispatch.run(
            view,
            device_logits,
            32,
            stream,
            nullptr,
            PieCompletion{}),
        "parallel-signature launch");
    cudaDeviceSynchronize();
    const auto stats = dispatch.stats();
    expect(
        stats.generated_fused_groups == 2 &&
            stats.overlapped_groups == 2,
        "independent signatures execute concurrently on dedicated streams");
    const auto greedy_token =
        *reinterpret_cast<const std::int32_t*>(
            greedy.endpoints[1].mirror_base);
    const auto section_token_out =
        *reinterpret_cast<const std::int32_t*>(
            section.endpoints[1].mirror_base);
    expect(
        greedy_token == 2 && section_token_out == 7,
        "parallel signatures preserve per-program outputs");
    cudaStreamDestroy(stream);
    cudaFree(device_logits);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <golden-directory>\n", argv[0]);
        return 2;
    }
    const auto fp32_launches =
        run_case(argv[1], false, false, false, false, 4, true);
    run_case(argv[1], true, false);
    run_case(argv[1], false, true);
    run_case(argv[1], true, true);
    run_case(argv[1], false, false, true);
    const auto direct_bf16_launches =
        run_case(argv[1], false, false, false, true);
    run_case(argv[1], false, false, false, false, 1, true);
    run_case(argv[1], false, false, false, false, 2, true);
    run_case(argv[1], false, false, false, false, 4, true);
    run_case(argv[1], false, false, false, false, 8, true);
    expect(
        direct_bf16_launches == fp32_launches,
        "direct BF16 grouped fallback fuses conversion into consumers");
    const auto nucleus_fp32 =
        run_nucleus_case(argv[1], false);
    const auto nucleus_bf16 =
        run_nucleus_case(argv[1], true);
    expect(
        nucleus_fp32 == nucleus_bf16,
        "nucleus generated/library execution matches FP32 and direct BF16");
    run_structured_mask_golden(argv[1]);
    run_declared_phase_case(argv[1], false);
    run_declared_phase_case(argv[1], true);
    run_declared_phase_case(
        argv[1], false, WriteBoundCase::BelowLower);
    run_declared_phase_case(
        argv[1], false, WriteBoundCase::Sentinel);
    run_mtp_direct_case(argv[1]);
    std::uint64_t beam_fp32_launches = 0;
    std::uint64_t beam_bf16_launches = 0;
    const auto beam_fp32 = run_beam_case(
        argv[1], false, &beam_fp32_launches);
    const auto beam_bf16 = run_beam_case(
        argv[1], true, &beam_bf16_launches);
    expect(
        beam_fp32 == beam_bf16,
        "generic TopK/index/gather beam attribution supports direct BF16");
    expect(
        beam_fp32_launches == beam_bf16_launches,
        "generic beam launch scaling is independent of logits storage");
    run_parallel_signature_case(argv[1]);
    std::printf("PTIR grouped Dispatch: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
