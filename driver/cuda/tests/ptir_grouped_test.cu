#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pipeline/dispatch.hpp"
#include "pipeline/descriptor_resolve.hpp"
#include "pipeline/grouped_runtime.cuh"
#include "support/host_eval.hpp"

using namespace pie_cuda_driver::pipeline;
using namespace pie_native::ptir;

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    }
}

std::uint16_t bf16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

std::string trim(const std::string& value) {
    const std::size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const std::size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

[[maybe_unused]] std::vector<std::uint8_t> hex_bytes(
    const std::string& value) {
    std::vector<std::uint8_t> result;
    result.reserve(value.size() / 2);
    for (std::size_t index = 0; index + 1 < value.size(); index += 2) {
        result.push_back(static_cast<std::uint8_t>(
            std::stoul(value.substr(index, 2), nullptr, 16)));
    }
    return result;
}

[[maybe_unused]] std::string golden_field(
    const std::string& path, const std::string& key) {
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

Value value(
    ValueId id,
    TensorType type,
    ValueSource source) {
    Value result;
    result.id = id;
    result.type = std::move(type);
    result.source = source;
    return result;
}

Trace masked_trace(std::uint32_t vocab, std::uint32_t channel_offset) {
    Trace trace;
    for (std::uint32_t channel = 0; channel < channel_offset; ++channel) {
        Channel unused;
        unused.id = channel;
        unused.type = {Shape::vec(1), DType::U32};
        unused.capacity = 1;
        trace.channels.push_back(unused);
    }
    Channel mask;
    mask.id = channel_offset;
    mask.type = {Shape::vec(vocab), DType::Bool};
    mask.capacity = 1;
    mask.has_seed = true;
    Channel output;
    output.id = channel_offset + 1;
    output.type = {Shape::vec(1), DType::U32};
    output.capacity = 1;
    output.host_visible = true;
    output.host_reader = true;
    trace.channels.push_back(mask);
    trace.channels.push_back(output);

    const TensorType logits_type{Shape::mat(1, vocab), DType::F32};
    const TensorType mask_type{Shape::vec(vocab), DType::Bool};
    const TensorType scalar_f32{Shape::scalar(), DType::F32};
    const TensorType token_type{Shape::vec(1), DType::U32};
    Value logits = value(0, logits_type, ValueSource::Intrinsic);
    logits.intrinsic = Intrinsic::Logits;
    Value mask_value = value(1, mask_type, ValueSource::ChannelTake);
    mask_value.channel = channel_offset;
    Value negative = value(2, scalar_f32, ValueSource::Const);
    negative.lit = Literal::f32(-INFINITY);
    trace.values = {
        logits,
        mask_value,
        negative,
        value(3, logits_type, ValueSource::OpResult),
        value(4, token_type, ValueSource::OpResult),
    };
    Stage stage;
    stage.kind = StageKind::Epilogue;
    Op select;
    select.code = OpCode::Select;
    select.args = {1, 0, 2};
    select.result_type = logits_type;
    select.result_id = 3;
    Op argmax;
    argmax.code = OpCode::ReduceArgmax;
    argmax.args = {3};
    argmax.result_type = token_type;
    argmax.result_id = 4;
    stage.ops = {select, argmax};
    stage.puts = {{channel_offset + 1, 4}};
    stage.takes = {channel_offset};
    trace.stages = {stage};
    return trace;
}

plan::ValueType plan_type(
    std::uint8_t dtype,
    std::initializer_list<std::uint32_t> dimensions,
    std::uint8_t domain) {
    plan::ValueType result;
    result.dtype = dtype;
    result.domain = domain;
    for (std::uint32_t dimension : dimensions) {
        result.dims.push_back({false, dimension});
    }
    return result;
}

plan::ValueType sampled_rows_type(
    std::uint8_t dtype,
    std::initializer_list<std::uint32_t> trailing,
    std::uint8_t domain) {
    plan::ValueType result;
    result.dtype = dtype;
    result.domain = domain;
    result.dims.push_back({true, PTIR_EXTENT_SAMPLED_ROWS});
    for (const auto dimension : trailing) {
        result.dims.push_back({false, dimension});
    }
    return result;
}

plan::StagePlan masked_plan(
    std::uint32_t vocab,
    std::uint32_t channel_offset) {
    plan::StagePlan stage;
    stage.stage = PTIR_STAGE_EPILOGUE;
    stage.signature_hash = 0x7b93d629c474a45eULL;
    stage.signature.assign(
        {'m', 'a', 's', 'k', 'e', 'd', '-', 'a', 'r', 'g', 'm', 'a', 'x'});
    stage.channel_bindings = {channel_offset, channel_offset + 1};

    container::COp logits;
    logits.tag = PTIR_OP_INTRINSIC_VAL;
    logits.intr = PTIR_INTR_LOGITS;
    logits.dtype = PTIR_DT_F32;
    logits.results = 1;
    container::COp mask;
    mask.tag = PTIR_OP_CHAN_TAKE;
    mask.chan = 0;
    mask.results = 1;
    container::COp negative;
    negative.tag = PTIR_OP_CONST;
    negative.lit_dtype = PTIR_DT_F32;
    const float negative_infinity = -INFINITY;
    std::memcpy(
        &negative.lit_bits, &negative_infinity, sizeof(negative.lit_bits));
    negative.results = 1;
    container::COp select;
    select.tag = PTIR_OP_SELECT;
    select.args = {1, 0, 2};
    select.results = 1;
    container::COp argmax;
    argmax.tag = PTIR_OP_REDUCE_ARGMAX;
    argmax.args = {3};
    argmax.results = 1;
    container::COp put;
    put.tag = PTIR_OP_CHAN_PUT;
    put.chan = 1;
    put.args = {4};
    put.results = 0;
    stage.ops = {
        {logits, {0}},
        {mask, {1}},
        {negative, {2}},
        {select, {3}},
        {argmax, {4}},
        {put, {5}},
    };
    stage.value_types = {
        plan_type(PTIR_DT_F32, {1, vocab}, 2),
        plan_type(PTIR_DT_BOOL, {vocab}, 4),
        plan_type(PTIR_DT_F32, {}, 0),
        plan_type(PTIR_DT_F32, {1, vocab}, 2),
        plan_type(PTIR_DT_U32, {1}, 0),
    };
    stage.singleton.kind = 0;
    stage.fused.kind = 1;
    stage.fused.regions.push_back({
        false,
        0,
        PTIR_SCHEDULE_ONE_CTA_PER_ROW,
        {0, 1, 2, 3, 4},
        {},
        {4},
        {{1, 4}},
    });
    return stage;
}

plan::StagePlan gumbel_plan(std::uint32_t vocab) {
    plan::StagePlan stage;
    stage.stage = PTIR_STAGE_EPILOGUE;
    stage.signature_hash = 0xa60d4a89fa914a21ULL;
    stage.signature.assign(
        {'g', 'u', 'm', 'b', 'e', 'l', '-', 'm', 'a', 'x'});
    stage.channel_bindings = {0, 1};
    container::COp logits;
    logits.tag = PTIR_OP_INTRINSIC_VAL;
    logits.intr = PTIR_INTR_LOGITS;
    logits.dtype = PTIR_DT_F32;
    container::COp state;
    state.tag = PTIR_OP_CHAN_TAKE;
    state.chan = 0;
    container::COp rng;
    rng.tag = PTIR_OP_RNG_KEYED;
    rng.args = {1};
    rng.kind = 1;
    container::COp add;
    add.tag = PTIR_OP_ADD;
    add.args = {0, 2};
    container::COp argmax;
    argmax.tag = PTIR_OP_REDUCE_ARGMAX;
    argmax.args = {3};
    container::COp put;
    put.tag = PTIR_OP_CHAN_PUT;
    put.chan = 1;
    put.args = {4};
    put.results = 0;
    stage.ops = {
        {logits, {0}},
        {state, {1}},
        {rng, {2}},
        {add, {3}},
        {argmax, {4}},
        {put, {5}},
    };
    stage.value_types = {
        plan_type(PTIR_DT_F32, {1, vocab}, 2),
        plan_type(PTIR_DT_U32, {2}, 0),
        plan_type(PTIR_DT_F32, {1, vocab}, 2),
        plan_type(PTIR_DT_F32, {1, vocab}, 2),
        plan_type(PTIR_DT_U32, {1}, 0),
    };
    stage.singleton.kind = 0;
    stage.fused.kind = 1;
    stage.fused.regions.push_back({
        false,
        0,
        PTIR_SCHEDULE_ONE_CTA_PER_ROW,
        {0, 1, 2, 3, 4},
        {},
        {4},
        {{1, 4}},
    });
    return stage;
}

struct Fixture {
    DeviceChannelRegistry registry;
    std::vector<std::unique_ptr<PtirInstance>> instances;
    std::vector<std::unique_ptr<Trace>> traces;
    std::vector<plan::StagePlan> plans;
    std::vector<std::vector<PieChannelEndpointBinding>> endpoints;
    std::vector<std::vector<std::uint64_t>> channel_ids;
    std::uint64_t next_channel_id = 1000;

    std::size_t add_instance(
        std::uint32_t vocab,
        std::uint32_t channel_offset,
        const std::vector<std::uint8_t>& mask_seed) {
        traces.push_back(std::make_unique<Trace>(
            masked_trace(vocab, channel_offset)));
        plans.push_back(masked_plan(vocab, channel_offset));
        const Trace& trace = *traces.back();
        endpoints.emplace_back(trace.channels.size());
        channel_ids.emplace_back(trace.channels.size());
        for (std::size_t dense = 0; dense < trace.channels.size(); ++dense) {
            const Channel& channel = trace.channels[dense];
            const std::uint64_t id = next_channel_id++;
            channel_ids.back()[dense] = id;
            PieChannelDesc descriptor{};
            descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
            descriptor.channel_id = id;
            descriptor.shape = {
                channel.type.shape.dims.data(),
                channel.type.shape.dims.size(),
            };
            descriptor.dtype = static_cast<std::uint8_t>(channel.type.dtype);
            descriptor.host_role = channel.host_reader
                ? PIE_CHANNEL_HOST_ROLE_READER
                : PIE_CHANNEL_HOST_ROLE_NONE;
            descriptor.seeded = channel.has_seed ? 1 : 0;
            descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
            descriptor.capacity = channel.capacity;
            descriptor.reader_wait_id = id * 2 + 1;
            descriptor.writer_wait_id = id * 2 + 2;
            std::string error;
            expect(
                registry.register_endpoint(
                    descriptor, &endpoints.back()[dense], &error),
                "register endpoint: " + error);
        }
        std::vector<ChannelValue> seeds;
        seeds.push_back({
            channel_ids.back()[channel_offset],
            mask_seed,
        });
        std::string error;
        auto instance = std::make_unique<PtirInstance>(
            trace, &registry, channel_ids.back(), seeds, &error);
        expect(instance->ok(), "bind instance: " + error);
        instances.push_back(std::move(instance));
        return instances.size() - 1;
    }

    std::vector<DeviceHostChannelTicket> tickets(
        std::size_t instance_index,
        std::uint32_t channel_offset,
        bool ready = true) {
        PtirInstance& instance = *instances[instance_index];
        std::vector<DeviceHostChannelTicket> result;
        const std::uint32_t mask_slot =
            instance.view().slot(channel_offset);
        const std::uint32_t output_slot =
            instance.view().slot(channel_offset + 1);
        result.push_back(ticket(
            mask_slot,
            kTicketConsume | kTicketRequireInput,
            ready ? 0 : 1,
            kNoChannelTicket));
        result.push_back(ticket(
            output_slot,
            kTicketPublish,
            kNoChannelTicket,
            0));
        return result;
    }

    DeviceHostChannelTicket ticket(
        std::uint32_t slot,
        std::uint32_t flags,
        std::uint64_t head,
        std::uint64_t tail) {
        return DeviceHostChannelTicket{
            .slot = slot,
            .flags = flags,
            .expected_head = head,
            .expected_tail = tail,
            .words = registry.host_words(slot),
            .mirror = static_cast<const std::uint8_t*>(
                registry.host_mirror(slot)),
            .cells = static_cast<std::uint8_t*>(
                registry.cell_base(slot)),
            .cap1 = registry.capacity(slot) + 1,
            .wire_bytes = static_cast<std::uint32_t>(
                registry.wire_bytes(slot)),
            .native_bytes = static_cast<std::uint32_t>(
                registry.cell_bytes(slot)),
        };
    }

    std::uint32_t output(
        std::size_t instance_index,
        std::uint32_t channel_offset) {
        std::uint32_t result = UINT32_MAX;
        instances[instance_index]->view().read_committed(
            channel_offset + 1, &result, sizeof(result));
        return result;
    }
};

float* device_logits(
    const std::vector<std::vector<float>>& rows,
    std::uint32_t vocab) {
    std::vector<float> flat;
    for (const auto& row : rows) flat.insert(flat.end(), row.begin(), row.end());
    float* device = nullptr;
    cudaMalloc(&device, flat.size() * sizeof(float));
    cudaMemcpy(
        device, flat.data(), flat.size() * sizeof(float),
        cudaMemcpyHostToDevice);
    return device;
}

std::vector<std::uint32_t> run_solo(
    std::uint32_t lanes,
    std::uint32_t vocab,
    const std::vector<std::vector<float>>& logits) {
    Fixture fixture;
    fixture.instances.reserve(lanes);
    fixture.traces.reserve(lanes);
    fixture.plans.reserve(lanes);
    fixture.endpoints.reserve(lanes);
    fixture.channel_ids.reserve(lanes);
    const std::vector<std::uint8_t> mask(vocab, 1);
    float* device = device_logits(logits, vocab);
    std::vector<std::uint32_t> result;
    for (std::uint32_t lane = 0; lane < lanes; ++lane) {
        const std::size_t instance = fixture.add_instance(vocab, 0, mask);
        FireInputs inputs;
        inputs.logits = device + static_cast<std::size_t>(lane) * vocab;
        inputs.vocab = vocab;
        const PassResult pass = fixture.instances[instance]->fire(inputs);
        expect(pass.ok && pass.committed, "solo Tier 0 commits");
        result.push_back(fixture.output(instance, 0));
    }
    cudaFree(device);
    return result;
}

std::vector<std::uint32_t> run_grouped(
    std::uint32_t lanes,
    std::uint32_t vocab,
    const std::vector<std::vector<float>>& logits,
    std::int32_t unready_lane = -1,
    std::uint32_t* body_op_launches = nullptr,
    float* elapsed_ms = nullptr,
    GroupedTier0GraphCache* graph_cache = nullptr) {
    Fixture fixture;
    fixture.instances.reserve(lanes);
    fixture.traces.reserve(lanes);
    fixture.plans.reserve(lanes);
    fixture.endpoints.reserve(lanes);
    fixture.channel_ids.reserve(lanes);
    const std::vector<std::uint8_t> mask(vocab, 1);
    float* device = device_logits(logits, vocab);
    std::vector<std::vector<DeviceHostChannelTicket>> tickets(lanes);
    std::vector<GroupedLaneBinding> bindings;
    for (std::uint32_t lane = 0; lane < lanes; ++lane) {
        const std::size_t instance = fixture.add_instance(vocab, 0, mask);
        tickets[lane] = fixture.tickets(
            instance, 0, static_cast<std::int32_t>(lane) != unready_lane);
        bindings.push_back({
            .instance = fixture.instances[instance].get(),
            .plan = &fixture.plans[instance],
            .tickets = &tickets[lane],
            .logits_base = device,
            .logits_row_offset = lane,
            .logits_row_count = 1,
            .vocab = vocab,
            .program_index = lane,
        });
    }
    std::string reason;
    expect(grouped_stage_supported(bindings, &reason), "group supported: " + reason);
    cudaStream_t execution_stream = nullptr;
    if (graph_cache != nullptr) {
        cudaStreamCreateWithFlags(
            &execution_stream, cudaStreamNonBlocking);
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (elapsed_ms != nullptr) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, execution_stream);
    }
    GroupedLaunchResult launch = GroupedTier0Executor::run(
        bindings, execution_stream, graph_cache);
    if (elapsed_ms != nullptr) {
        cudaEventRecord(stop, execution_stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(elapsed_ms, start, stop);
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
    }
    cudaDeviceSynchronize();
    if (body_op_launches != nullptr) {
        *body_op_launches = launch.body_op_launches;
    }
    if (launch.device_tickets != nullptr &&
        !launch.device_tickets_persistent) {
        cudaFree(launch.device_tickets);
    }
    if (execution_stream != nullptr) cudaStreamDestroy(execution_stream);
    std::vector<std::uint32_t> result;
    for (std::uint32_t lane = 0; lane < lanes; ++lane) {
        std::uint32_t committed = 0;
        cudaMemcpy(
            &committed,
            fixture.instances[lane]->commit_device_flag(),
            sizeof(committed),
            cudaMemcpyDeviceToHost);
        if (static_cast<std::int32_t>(lane) == unready_lane) {
            expect(committed == 0, "unready lane retries");
            result.push_back(UINT32_MAX);
        } else {
            expect(committed == 1, "ready grouped lane commits");
            result.push_back(fixture.output(lane, 0));
        }
    }
    cudaFree(device);
    return result;
}

std::vector<std::vector<float>> test_logits(
    std::uint32_t lanes,
    std::uint32_t vocab) {
    std::vector<std::vector<float>> rows(
        lanes, std::vector<float>(vocab, -4.0f));
    for (std::uint32_t lane = 0; lane < lanes; ++lane) {
        rows[lane][(lane * 7 + 3) % vocab] = 9.0f;
        rows[lane][(lane * 11 + 5) % vocab] = 8.0f;
    }
    return rows;
}

void parity_cases() {
    constexpr std::uint32_t vocab = 64;
    std::uint32_t expected_launches = 0;
    for (std::uint32_t lanes : {1u, 2u, 4u, 8u}) {
        const auto logits = test_logits(lanes, vocab);
        const auto solo = run_solo(lanes, vocab, logits);
        std::uint32_t launches = 0;
        float elapsed_ms = 0.0f;
        const auto grouped = run_grouped(
            lanes, vocab, logits, -1, &launches, &elapsed_ms);
        expect(grouped == solo, "grouped vs solo N=" + std::to_string(lanes));
        if (expected_launches == 0) expected_launches = launches;
        expect(
            launches == expected_launches,
            "grouped body launches are independent of lane count");
        if (std::getenv("NV_SANITIZER_INJECTION_TRANSPORT_TYPE") == nullptr) {
            expect(
                elapsed_ms < 20.0f,
                "grouped post-forward event time remains bounded");
        }
        std::printf(
            "  grouped B=%u launches=%u event=%.3f ms\n",
            lanes, launches, elapsed_ms);
    }
}

void partial_readiness() {
    constexpr std::uint32_t lanes = 4;
    constexpr std::uint32_t vocab = 64;
    const auto logits = test_logits(lanes, vocab);
    const auto solo = run_solo(lanes, vocab, logits);
    const auto grouped = run_grouped(lanes, vocab, logits, 1);
    expect(grouped[0] == solo[0], "partial lane 0");
    expect(grouped[1] == UINT32_MAX, "partial lane 1 retries");
    expect(grouped[2] == solo[2], "partial lane 2");
    expect(grouped[3] == solo[3], "partial lane 3");
}

void fallback_graph_case() {
    constexpr std::uint32_t width = 64;
    DeviceChannelRegistry registry;
    Trace trace;
    Stage trace_stage;
    trace_stage.kind = StageKind::Epilogue;
    trace.stages = {trace_stage};
    std::string error;
    PtirInstance instance(
        trace, &registry, {}, {}, &error);
    expect(instance.ok(), "bind graph-only Tier0 instance: " + error);
    plan::StagePlan plan;
    plan.stage = PTIR_STAGE_EPILOGUE;
    plan.signature = {'f', 'a', 'l', 'l', 'b', 'a', 'c', 'k', '-', 'g'};
    plan.signature_hash = container::fnv1a64(
        plan.signature.data(), plan.signature.size());
    container::COp iota;
    iota.tag = PTIR_OP_IOTA;
    iota.imm = width;
    container::COp sum;
    sum.tag = PTIR_OP_REDUCE_SUM;
    sum.args = {0};
    plan.ops = {{iota, {0}}, {sum, {1}}};
    plan.value_types = {
        plan_type(PTIR_DT_U32, {width}, 3),
        plan_type(PTIR_DT_U32, {}, 0),
    };
    plan.singleton.kind = 0;
    plan.fused.kind = 1;
    std::vector<DeviceHostChannelTicket> tickets;
    GroupedLaneBinding binding{
        .instance = &instance,
        .plan = &plan,
        .tickets = &tickets,
        .logits_row_count = 0,
        .row_count = 1,
        .token_count = 1,
        .kv_len = 1,
        .page_count = 1,
        .query_len = 1,
        .key_len = 1,
        .vocab = 1,
    };
    GroupedTier0GraphCache cache(4);
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEvent_t begin = nullptr;
    cudaEvent_t middle = nullptr;
    cudaEvent_t end = nullptr;
    cudaEventCreate(&begin);
    cudaEventCreate(&middle);
    cudaEventCreate(&end);
    cudaEventRecord(begin, stream);
    auto first = GroupedTier0Executor::run(
        {binding}, stream, &cache);
    cudaEventRecord(middle, stream);
    auto second = GroupedTier0Executor::run(
        {binding}, stream, &cache);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float capture_ms = 0.0f;
    float replay_ms = 0.0f;
    cudaEventElapsedTime(&capture_ms, begin, middle);
    cudaEventElapsedTime(&replay_ms, middle, end);
    const auto metrics = cache.metrics();
    expect(
        first.body_op_launches == second.body_op_launches &&
            metrics.captures == 1 &&
            metrics.replays >= 1 && metrics.max_nodes >= 5,
        "steady grouped Tier0 fallback replays one complete pass graph");
    std::printf(
        "  tier0 pass graph capture=%.3f ms replay=%.3f ms "
        "captures=%llu replays=%llu nodes=%zu\n",
        capture_ms, replay_ms,
        static_cast<unsigned long long>(metrics.captures),
        static_cast<unsigned long long>(metrics.replays),
        metrics.max_nodes);
    cudaEventDestroy(end);
    cudaEventDestroy(middle);
    cudaEventDestroy(begin);
    cudaStreamDestroy(stream);
}

void graph_cache_memory_bound_case() {
    GroupedTier0GraphCache cache(8, 100);
    auto first = cache.get_or_create("first");
    expect(first != nullptr && cache.account(first, 60),
           "graph cache accepts first byte reservation");
    first.reset();
    auto second = cache.get_or_create("second");
    expect(second != nullptr && cache.account(second, 60),
           "graph cache accepts replacement byte reservation");
    const auto metrics = cache.metrics();
    expect(
        cache.size() == 1 && metrics.retained_bytes == 60 &&
            metrics.evictions == 1,
        "graph cache evicts by retained bytes, not only entry count");
}

void graph_cache_lock_order_case() {
    GroupedTier0GraphCache cache(8, 1ULL << 20);
    auto entry = cache.get_or_create("lock-order");
    std::atomic<bool> start{false};
    std::thread accounting([&] {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (int iteration = 0; iteration < 2000; ++iteration) {
            std::lock_guard<std::mutex> entry_lock(entry->mutex);
            ++entry->captures;
            cache.account(entry, 64);
        }
    });
    std::thread metrics([&] {
        start.store(true, std::memory_order_release);
        for (int iteration = 0; iteration < 2000; ++iteration) {
            static_cast<void>(cache.metrics());
        }
    });
    accounting.join();
    metrics.join();
    const auto snapshot = cache.metrics();
    expect(
        snapshot.captures == 2000 && snapshot.retained_bytes == 64,
        "graph-cache metrics snapshots never invert cache/entry lock order");
}

void integer_unary_cases() {
    constexpr std::uint32_t lane_count = 2;
    constexpr std::uint32_t width = 3;
    const std::int32_t signed_input[lane_count * width] = {
        std::numeric_limits<std::int32_t>::min(),
        16777217,
        -16777217,
        -1,
        0,
        1,
    };
    std::int32_t* input_device = nullptr;
    std::int32_t* neg_device = nullptr;
    std::int32_t* abs_device = nullptr;
    std::int32_t* sign_device = nullptr;
    std::uint32_t* commits_device = nullptr;
    cudaMalloc(&input_device, sizeof(signed_input));
    cudaMalloc(&neg_device, sizeof(signed_input));
    cudaMalloc(&abs_device, sizeof(signed_input));
    cudaMalloc(&sign_device, sizeof(signed_input));
    cudaMalloc(&commits_device, lane_count * sizeof(std::uint32_t));
    cudaMemcpy(
        input_device, signed_input, sizeof(signed_input),
        cudaMemcpyHostToDevice);
    const std::uint32_t commits[lane_count] = {1, 1};
    cudaMemcpy(
        commits_device, commits, sizeof(commits), cudaMemcpyHostToDevice);

    PtirLaneTableHeader header_host{
        PTIR_LANE_TABLE_ABI_VERSION, lane_count, 0, 0};
    PtirLaneRecord lanes_host[lane_count]{};
    std::uint64_t values_host[lane_count * 4]{};
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        lanes_host[lane].commit_slot = reinterpret_cast<std::uint64_t>(
            commits_device + lane);
        values_host[lane * 4 + 0] = reinterpret_cast<std::uint64_t>(
            input_device + lane * width);
        values_host[lane * 4 + 1] = reinterpret_cast<std::uint64_t>(
            neg_device + lane * width);
        values_host[lane * 4 + 2] = reinterpret_cast<std::uint64_t>(
            abs_device + lane * width);
        values_host[lane * 4 + 3] = reinterpret_cast<std::uint64_t>(
            sign_device + lane * width);
    }
    PtirLaneTableHeader* header_device = nullptr;
    PtirLaneRecord* lanes_device = nullptr;
    std::uint64_t* values_device = nullptr;
    cudaMalloc(&header_device, sizeof(header_host));
    cudaMalloc(&lanes_device, sizeof(lanes_host));
    cudaMalloc(&values_device, sizeof(values_host));
    cudaMemcpy(
        header_device, &header_host, sizeof(header_host),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        lanes_device, lanes_host, sizeof(lanes_host),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        values_device, values_host, sizeof(values_host),
        cudaMemcpyHostToDevice);
    const std::uint32_t blocks =
        grouped_grid(static_cast<std::uint64_t>(lane_count) * width);
    k_grouped_unary<std::int32_t><<<blocks, kTier0Block>>>(
        header_device, lanes_device, values_device, 4, 0, 1, width,
        UnKind::Neg);
    k_grouped_unary<std::int32_t><<<blocks, kTier0Block>>>(
        header_device, lanes_device, values_device, 4, 0, 2, width,
        UnKind::Abs);
    k_grouped_unary<std::int32_t><<<blocks, kTier0Block>>>(
        header_device, lanes_device, values_device, 4, 0, 3, width,
        UnKind::Sign);
    cudaDeviceSynchronize();
    std::int32_t neg[lane_count * width]{};
    std::int32_t absolute[lane_count * width]{};
    std::int32_t sign[lane_count * width]{};
    cudaMemcpy(neg, neg_device, sizeof(neg), cudaMemcpyDeviceToHost);
    cudaMemcpy(absolute, abs_device, sizeof(absolute), cudaMemcpyDeviceToHost);
    cudaMemcpy(sign, sign_device, sizeof(sign), cudaMemcpyDeviceToHost);
    for (std::size_t index = 0; index < lane_count * width; ++index) {
        const std::uint32_t bits =
            static_cast<std::uint32_t>(signed_input[index]);
        const std::int32_t wrapped_neg =
            static_cast<std::int32_t>(0u - bits);
        const std::int32_t wrapped_abs =
            signed_input[index] < 0 ? wrapped_neg : signed_input[index];
        expect(neg[index] == wrapped_neg, "grouped wrapping integer neg");
        expect(absolute[index] == wrapped_abs, "grouped wrapping integer abs");
        expect(
            sign[index] ==
                ((signed_input[index] > 0) - (signed_input[index] < 0)),
            "grouped exact integer sign");
    }
    cudaFree(values_device);
    cudaFree(lanes_device);
    cudaFree(header_device);
    cudaFree(commits_device);
    cudaFree(sign_device);
    cudaFree(abs_device);
    cudaFree(neg_device);
    cudaFree(input_device);
}

void integer_binary_cases() {
    constexpr std::uint32_t width = 4;
    const std::int32_t a_host[width] = {
        std::numeric_limits<std::int32_t>::max(),
        std::numeric_limits<std::int32_t>::min(),
        5,
        -5,
    };
    const std::int32_t b_host[width] = {1, -1, 0, 2};
    std::int32_t* a_device = nullptr;
    std::int32_t* b_device = nullptr;
    std::int32_t* add_device = nullptr;
    std::int32_t* div_device = nullptr;
    std::int32_t* rem_device = nullptr;
    std::uint32_t* commit_device = nullptr;
    cudaMalloc(&a_device, sizeof(a_host));
    cudaMalloc(&b_device, sizeof(b_host));
    cudaMalloc(&add_device, sizeof(a_host));
    cudaMalloc(&div_device, sizeof(a_host));
    cudaMalloc(&rem_device, sizeof(a_host));
    cudaMalloc(&commit_device, sizeof(std::uint32_t));
    cudaMemcpy(a_device, a_host, sizeof(a_host), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, sizeof(b_host), cudaMemcpyHostToDevice);
    const std::uint32_t one = 1;
    cudaMemcpy(commit_device, &one, sizeof(one), cudaMemcpyHostToDevice);
    PtirLaneTableHeader header_host{
        PTIR_LANE_TABLE_ABI_VERSION, 1, 0, 0};
    PtirLaneRecord lane_host{};
    lane_host.commit_slot =
        reinterpret_cast<std::uint64_t>(commit_device);
    const std::uint64_t values_host[5] = {
        reinterpret_cast<std::uint64_t>(a_device),
        reinterpret_cast<std::uint64_t>(b_device),
        reinterpret_cast<std::uint64_t>(add_device),
        reinterpret_cast<std::uint64_t>(div_device),
        reinterpret_cast<std::uint64_t>(rem_device),
    };
    PtirLaneTableHeader* header_device = nullptr;
    PtirLaneRecord* lane_device = nullptr;
    std::uint64_t* values_device = nullptr;
    cudaMalloc(&header_device, sizeof(header_host));
    cudaMalloc(&lane_device, sizeof(lane_host));
    cudaMalloc(&values_device, sizeof(values_host));
    cudaMemcpy(
        header_device, &header_host, sizeof(header_host),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        lane_device, &lane_host, sizeof(lane_host),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        values_device, values_host, sizeof(values_host),
        cudaMemcpyHostToDevice);
    const auto blocks = grouped_grid(width);
    k_grouped_binary<std::int32_t><<<blocks, kTier0Block>>>(
        header_device, lane_device, values_device, 5, 0, 1, 2, width,
        BinKind::Add, false, false);
    k_grouped_binary<std::int32_t><<<blocks, kTier0Block>>>(
        header_device, lane_device, values_device, 5, 0, 1, 3, width,
        BinKind::Div, false, false);
    k_grouped_binary<std::int32_t><<<blocks, kTier0Block>>>(
        header_device, lane_device, values_device, 5, 0, 1, 4, width,
        BinKind::Rem, false, false);
    cudaDeviceSynchronize();
    std::int32_t add[width]{};
    std::int32_t divide[width]{};
    std::int32_t remainder[width]{};
    cudaMemcpy(add, add_device, sizeof(add), cudaMemcpyDeviceToHost);
    cudaMemcpy(divide, div_device, sizeof(divide), cudaMemcpyDeviceToHost);
    cudaMemcpy(remainder, rem_device, sizeof(remainder), cudaMemcpyDeviceToHost);
    const std::int32_t expected_add[width] = {
        std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::max(),
        5,
        -3,
    };
    const std::int32_t expected_div[width] = {
        std::numeric_limits<std::int32_t>::max(),
        std::numeric_limits<std::int32_t>::min(),
        0,
        -2,
    };
    const std::int32_t expected_rem[width] = {0, 0, 0, -1};
    expect(
        std::equal(std::begin(add), std::end(add), std::begin(expected_add)),
        "grouped wrapping integer add");
    expect(
        std::equal(
            std::begin(divide), std::end(divide), std::begin(expected_div)),
        "grouped wrapping/zero integer div");
    expect(
        std::equal(
            std::begin(remainder),
            std::end(remainder),
            std::begin(expected_rem)),
        "grouped wrapping/zero integer rem");
    cudaFree(values_device);
    cudaFree(lane_device);
    cudaFree(header_device);
    cudaFree(commit_device);
    cudaFree(rem_device);
    cudaFree(div_device);
    cudaFree(add_device);
    cudaFree(b_device);
    cudaFree(a_device);
}

void grouped_integer_reduction_case() {
    constexpr std::uint32_t lane_count = 2;
    constexpr std::uint32_t width = 4;
    const std::int32_t input_host[lane_count * width] = {
        std::numeric_limits<std::int32_t>::max(),
        1,
        16777217,
        -16777217,
        std::numeric_limits<std::int32_t>::min(),
        -1,
        1,
        0,
    };
    std::int32_t* input_device = nullptr;
    std::int32_t* output_device = nullptr;
    std::uint32_t* commits_device = nullptr;
    cudaMalloc(&input_device, sizeof(input_host));
    cudaMalloc(&output_device, lane_count * sizeof(std::int32_t));
    cudaMalloc(&commits_device, lane_count * sizeof(std::uint32_t));
    cudaMemcpy(
        input_device, input_host, sizeof(input_host), cudaMemcpyHostToDevice);
    const std::uint32_t commits[lane_count] = {1, 1};
    cudaMemcpy(
        commits_device, commits, sizeof(commits), cudaMemcpyHostToDevice);
    PtirLaneTableHeader header_host{
        PTIR_LANE_TABLE_ABI_VERSION, lane_count, 0, 0};
    PtirLaneRecord lanes_host[lane_count]{};
    std::uint64_t values_host[lane_count * 2]{};
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        lanes_host[lane].commit_slot =
            reinterpret_cast<std::uint64_t>(commits_device + lane);
        values_host[lane * 2] = reinterpret_cast<std::uint64_t>(
            input_device + lane * width);
        values_host[lane * 2 + 1] = reinterpret_cast<std::uint64_t>(
            output_device + lane);
    }
    PtirLaneTableHeader* header_device = nullptr;
    PtirLaneRecord* lanes_device = nullptr;
    std::uint64_t* values_device = nullptr;
    cudaMalloc(&header_device, sizeof(header_host));
    cudaMalloc(&lanes_device, sizeof(lanes_host));
    cudaMalloc(&values_device, sizeof(values_host));
    cudaMemcpy(
        header_device, &header_host, sizeof(header_host),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        lanes_device, lanes_host, sizeof(lanes_host),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        values_device, values_host, sizeof(values_host),
        cudaMemcpyHostToDevice);
    k_grouped_reduce<std::int32_t><<<
        lane_count, kCanonicalReduceWidth>>>(
        header_device,
        lanes_device,
        values_device,
        2,
        0,
        1,
        1,
        width,
        RedKind::Sum);
    cudaDeviceSynchronize();
    std::int32_t actual[lane_count]{};
    cudaMemcpy(
        actual,
        output_device,
        sizeof(actual),
        cudaMemcpyDeviceToHost);
    const std::int32_t expected[lane_count] = {
        std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::min(),
    };
    expect(
        std::equal(
            std::begin(actual), std::end(actual), std::begin(expected)),
        "grouped integer reduction preserves wrapping dtype semantics");
    cudaFree(values_device);
    cudaFree(lanes_device);
    cudaFree(header_device);
    cudaFree(commits_device);
    cudaFree(output_device);
    cudaFree(input_device);
}

void grouped_integer_argmax_case() {
    const std::uint32_t u32_input[2] = {16777216u, 16777217u};
    const std::int32_t i32_input[2] = {-16777217, -16777216};
    std::uint32_t* u32_device = nullptr;
    std::int32_t* i32_device = nullptr;
    std::uint32_t* output_device = nullptr;
    std::uint32_t* commit_device = nullptr;
    cudaMalloc(&u32_device, sizeof(u32_input));
    cudaMalloc(&i32_device, sizeof(i32_input));
    cudaMalloc(&output_device, sizeof(std::uint32_t));
    cudaMalloc(&commit_device, sizeof(std::uint32_t));
    cudaMemcpy(
        u32_device, u32_input, sizeof(u32_input), cudaMemcpyHostToDevice);
    cudaMemcpy(
        i32_device, i32_input, sizeof(i32_input), cudaMemcpyHostToDevice);
    const std::uint32_t one = 1;
    cudaMemcpy(commit_device, &one, sizeof(one), cudaMemcpyHostToDevice);
    PtirLaneTableHeader header_host{
        PTIR_LANE_TABLE_ABI_VERSION, 1, 0, 0};
    PtirLaneRecord lane_host{};
    lane_host.commit_slot =
        reinterpret_cast<std::uint64_t>(commit_device);
    PtirLaneTableHeader* header_device = nullptr;
    PtirLaneRecord* lane_device = nullptr;
    std::uint64_t* values_device = nullptr;
    cudaMalloc(&header_device, sizeof(header_host));
    cudaMalloc(&lane_device, sizeof(lane_host));
    cudaMalloc(&values_device, 2 * sizeof(std::uint64_t));
    cudaMemcpy(
        header_device, &header_host, sizeof(header_host),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        lane_device, &lane_host, sizeof(lane_host),
        cudaMemcpyHostToDevice);
    auto run = [&](std::uint64_t input, auto kernel) {
        const std::uint64_t values_host[2] = {
            input, reinterpret_cast<std::uint64_t>(output_device)};
        cudaMemcpy(
            values_device,
            values_host,
            sizeof(values_host),
            cudaMemcpyHostToDevice);
        kernel();
        cudaDeviceSynchronize();
        std::uint32_t actual = 0;
        cudaMemcpy(
            &actual,
            output_device,
            sizeof(actual),
            cudaMemcpyDeviceToHost);
        expect(actual == 1, "grouped integer argmax compares exact dtype");
    };
    run(reinterpret_cast<std::uint64_t>(u32_device), [&] {
        k_grouped_reduce_argmax<std::uint32_t><<<1, kTier0Block>>>(
            header_device, lane_device, values_device, 2, 0, 1, 1, 2);
    });
    run(reinterpret_cast<std::uint64_t>(i32_device), [&] {
        k_grouped_reduce_argmax<std::int32_t><<<1, kTier0Block>>>(
            header_device, lane_device, values_device, 2, 0, 1, 1, 2);
    });
    cudaFree(values_device);
    cudaFree(lane_device);
    cudaFree(header_device);
    cudaFree(commit_device);
    cudaFree(output_device);
    cudaFree(i32_device);
    cudaFree(u32_device);
}

void cast_adversarial_case() {
    constexpr std::uint32_t width = 8;
    const float input_host[width] = {
        std::numeric_limits<float>::quiet_NaN(),
        -INFINITY,
        -1.5f,
        -0.0f,
        0.0f,
        1.9f,
        INFINITY,
        4294967296.0f,
    };
    float* input_device = nullptr;
    std::int32_t* i32_device = nullptr;
    std::uint32_t* u32_device = nullptr;
    std::uint8_t* bool_device = nullptr;
    std::uint32_t* commit_device = nullptr;
    cudaMalloc(&input_device, sizeof(input_host));
    cudaMalloc(&i32_device, width * sizeof(std::int32_t));
    cudaMalloc(&u32_device, width * sizeof(std::uint32_t));
    cudaMalloc(&bool_device, width);
    cudaMalloc(&commit_device, sizeof(std::uint32_t));
    cudaMemcpy(
        input_device, input_host, sizeof(input_host), cudaMemcpyHostToDevice);
    const std::uint32_t one = 1;
    cudaMemcpy(commit_device, &one, sizeof(one), cudaMemcpyHostToDevice);
    PtirLaneTableHeader header_host{
        PTIR_LANE_TABLE_ABI_VERSION, 1, 0, 0};
    PtirLaneRecord lane_host{};
    lane_host.commit_slot =
        reinterpret_cast<std::uint64_t>(commit_device);
    const std::uint64_t values_host[4] = {
        reinterpret_cast<std::uint64_t>(input_device),
        reinterpret_cast<std::uint64_t>(i32_device),
        reinterpret_cast<std::uint64_t>(u32_device),
        reinterpret_cast<std::uint64_t>(bool_device),
    };
    PtirLaneTableHeader* header_device = nullptr;
    PtirLaneRecord* lane_device = nullptr;
    std::uint64_t* values_device = nullptr;
    cudaMalloc(&header_device, sizeof(header_host));
    cudaMalloc(&lane_device, sizeof(lane_host));
    cudaMalloc(&values_device, sizeof(values_host));
    cudaMemcpy(
        header_device, &header_host, sizeof(header_host),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        lane_device, &lane_host, sizeof(lane_host),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        values_device, values_host, sizeof(values_host),
        cudaMemcpyHostToDevice);
    const auto blocks = grouped_grid(width);
    k_grouped_cast<float, std::int32_t><<<blocks, kTier0Block>>>(
        header_device, lane_device, values_device, 4, 0, 1, width);
    k_grouped_cast<float, std::uint32_t><<<blocks, kTier0Block>>>(
        header_device, lane_device, values_device, 4, 0, 2, width);
    k_grouped_cast_bool<float><<<blocks, kTier0Block>>>(
        header_device, lane_device, values_device, 4, 0, 3, width);
    cudaDeviceSynchronize();
    std::int32_t i32[width]{};
    std::uint32_t u32[width]{};
    std::uint8_t boolean[width]{};
    cudaMemcpy(i32, i32_device, sizeof(i32), cudaMemcpyDeviceToHost);
    cudaMemcpy(u32, u32_device, sizeof(u32), cudaMemcpyDeviceToHost);
    cudaMemcpy(boolean, bool_device, sizeof(boolean), cudaMemcpyDeviceToHost);
    const std::int32_t expected_i32[width] = {
        0,
        std::numeric_limits<std::int32_t>::min(),
        -1,
        0,
        0,
        1,
        std::numeric_limits<std::int32_t>::max(),
        std::numeric_limits<std::int32_t>::max(),
    };
    const std::uint32_t expected_u32[width] = {
        0, 0, 0, 0, 0, 1,
        std::numeric_limits<std::uint32_t>::max(),
        std::numeric_limits<std::uint32_t>::max(),
    };
    const std::uint8_t expected_bool[width] = {1, 1, 1, 0, 0, 1, 1, 1};
    expect(
        std::equal(
            std::begin(i32), std::end(i32), std::begin(expected_i32)),
        "F32 to I32 saturating cast");
    expect(
        std::equal(
            std::begin(u32), std::end(u32), std::begin(expected_u32)),
        "F32 to U32 saturating cast");
    expect(
        std::equal(
            std::begin(boolean),
            std::end(boolean),
            std::begin(expected_bool)),
        "numeric to Bool cast");
    cudaFree(values_device);
    cudaFree(lane_device);
    cudaFree(header_device);
    cudaFree(commit_device);
    cudaFree(bool_device);
    cudaFree(u32_device);
    cudaFree(i32_device);
    cudaFree(input_device);
}

void topk_adversarial_case() {
    constexpr std::uint32_t width = 8;
    const float input_host[width] = {
        INFINITY,
        std::numeric_limits<float>::quiet_NaN(),
        INFINITY,
        1.0f,
        -0.0f,
        0.0f,
        -INFINITY,
        std::numeric_limits<float>::quiet_NaN(),
    };
    const std::uint32_t expected_indices[width] = {
        0, 2, 3, 4, 5, 6, 1, 7,
    };
    float* input_device = nullptr;
    float* values_output_device = nullptr;
    std::uint32_t* indices_output_device = nullptr;
    std::uint32_t* commit_device = nullptr;
    cudaMalloc(&input_device, sizeof(input_host));
    cudaMalloc(&values_output_device, sizeof(input_host));
    cudaMalloc(
        &indices_output_device, sizeof(expected_indices));
    cudaMalloc(&commit_device, sizeof(std::uint32_t));
    const std::uint32_t committed = 1;
    cudaMemcpy(
        input_device, input_host, sizeof(input_host), cudaMemcpyHostToDevice);
    cudaMemcpy(
        commit_device, &committed, sizeof(committed), cudaMemcpyHostToDevice);
    PtirLaneTableHeader header_host{
        PTIR_LANE_TABLE_ABI_VERSION, 1, 0, 0};
    PtirLaneRecord lane_host{};
    lane_host.commit_slot =
        reinterpret_cast<std::uint64_t>(commit_device);
    const std::uint64_t values_host[3] = {
        reinterpret_cast<std::uint64_t>(input_device),
        reinterpret_cast<std::uint64_t>(values_output_device),
        reinterpret_cast<std::uint64_t>(indices_output_device),
    };
    PtirLaneTableHeader* header_device = nullptr;
    PtirLaneRecord* lane_device = nullptr;
    std::uint64_t* values_device = nullptr;
    cudaMalloc(&header_device, sizeof(header_host));
    cudaMalloc(&lane_device, sizeof(lane_host));
    cudaMalloc(&values_device, sizeof(values_host));
    cudaMemcpy(
        header_device, &header_host, sizeof(header_host),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        lane_device, &lane_host, sizeof(lane_host),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        values_device, values_host, sizeof(values_host),
        cudaMemcpyHostToDevice);
    k_grouped_topk<<<1, kTier0Block>>>(
        header_device,
        lane_device,
        nullptr,
        nullptr,
        values_device,
        3,
        0,
        1,
        2,
        1,
        width,
        width,
        false,
        GroupedDynamicShape{width, 1, 0xff, {0, 0, 0}},
        GroupedRowShape{
            GroupedDynamicShape{1, 1, 0xff, {0, 0, 0}},
            GroupedDynamicShape{width, 1, 0xff, {0, 0, 0}},
            1,
            width,
        },
        width,
        0);
    cudaDeviceSynchronize();
    float output_values[width]{};
    std::uint32_t output_indices[width]{};
    cudaMemcpy(
        output_values,
        values_output_device,
        sizeof(output_values),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        output_indices,
        indices_output_device,
        sizeof(output_indices),
        cudaMemcpyDeviceToHost);
    expect(
        std::equal(
            std::begin(output_indices),
            std::end(output_indices),
            std::begin(expected_indices)),
        "TopK exact tie/infinity/NaN order");
    for (std::uint32_t index = 0; index < width; ++index) {
        expect(
            std::memcmp(
                &output_values[index],
                &input_host[expected_indices[index]],
                sizeof(float)) == 0,
            "TopK preserves selected value bits");
    }
    cudaFree(values_device);
    cudaFree(lane_device);
    cudaFree(header_device);
    cudaFree(commit_device);
    cudaFree(indices_output_device);
    cudaFree(values_output_device);
    cudaFree(input_device);
}

void scalable_nucleus_production_case() {
    auto run = [](std::uint32_t vocab, bool adversarial) {
        std::vector<float> logits(vocab, 0.0f);
        float top_p = 1.0f;
        if (adversarial) {
            for (std::uint32_t index = 0; index < vocab; ++index) {
                logits[index] =
                    (index & 7u) == 0
                        ? std::numeric_limits<float>::quiet_NaN()
                        : 0.0f;
            }
            logits[3] = 2.0f;
            logits[4] = 2.0f;
            top_p = 0.7f;
        }
        const float maximum = host_eval::canonical_reduce(
            logits.data(),
            vocab,
            -std::numeric_limits<float>::infinity(),
            host_eval::canonical_max);
        std::vector<float> exponentials(vocab);
        std::transform(
            logits.begin(),
            logits.end(),
            exponentials.begin(),
            [maximum](float value) {
                return std::exp(value - maximum);
            });
        const float sum = host_eval::canonical_reduce(
            exponentials.data(),
            vocab,
            0.0f,
            [](float left, float right) { return left + right; });
        std::vector<float> probabilities(vocab);
        std::transform(
            exponentials.begin(),
            exponentials.end(),
            probabilities.begin(),
            [sum](float value) { return value / sum; });
        const std::uint32_t state[2] = {1234, 1};
        float* logits_device = nullptr;
        float* probabilities_device = nullptr;
        float* top_p_device = nullptr;
        std::uint32_t* state_device = nullptr;
        std::uint32_t* output_device = nullptr;
        std::uint32_t* commit_device = nullptr;
        cudaMalloc(&logits_device, logits.size() * sizeof(float));
        cudaMalloc(
            &probabilities_device,
            probabilities.size() * sizeof(float));
        cudaMalloc(&top_p_device, sizeof(float));
        cudaMalloc(&state_device, sizeof(state));
        cudaMalloc(&output_device, sizeof(std::uint32_t));
        cudaMalloc(&commit_device, sizeof(std::uint32_t));
        cudaMemcpy(
            logits_device, logits.data(), logits.size() * sizeof(float),
            cudaMemcpyHostToDevice);
        cudaMemcpy(
            top_p_device, &top_p, sizeof(top_p), cudaMemcpyHostToDevice);
        cudaMemcpy(
            state_device, state, sizeof(state), cudaMemcpyHostToDevice);
        const std::uint32_t one = 1;
        cudaMemcpy(
            commit_device, &one, sizeof(one), cudaMemcpyHostToDevice);

        PtirLaneTableHeader header_host{
            PTIR_LANE_TABLE_ABI_VERSION, 1, 0, 0};
        PtirLaneRecord lane_host{};
        lane_host.logits_row_count = 1;
        lane_host.sampled_rows = 1;
        lane_host.commit_slot =
            reinterpret_cast<std::uint64_t>(commit_device);
        const std::uint64_t values_host[5] = {
            reinterpret_cast<std::uint64_t>(logits_device),
            reinterpret_cast<std::uint64_t>(top_p_device),
            reinterpret_cast<std::uint64_t>(state_device),
            reinterpret_cast<std::uint64_t>(probabilities_device),
            reinterpret_cast<std::uint64_t>(output_device),
        };
        PtirLaneTableHeader* header_device = nullptr;
        PtirLaneRecord* lane_device = nullptr;
        std::uint64_t* values_device = nullptr;
        cudaMalloc(&header_device, sizeof(header_host));
        cudaMalloc(&lane_device, sizeof(lane_host));
        cudaMalloc(&values_device, sizeof(values_host));
        cudaMemcpy(
            header_device, &header_host, sizeof(header_host),
            cudaMemcpyHostToDevice);
        cudaMemcpy(
            lane_device, &lane_host, sizeof(lane_host),
            cudaMemcpyHostToDevice);
        cudaMemcpy(
            values_device, values_host, sizeof(values_host),
            cudaMemcpyHostToDevice);

        std::uint64_t* keys_in = nullptr;
        std::uint64_t* keys_out = nullptr;
        std::uint32_t* indices_in = nullptr;
        std::uint32_t* indices_out = nullptr;
        std::uint32_t* offsets = nullptr;
        cudaMalloc(&keys_in, vocab * sizeof(std::uint64_t));
        cudaMalloc(&keys_out, vocab * sizeof(std::uint64_t));
        cudaMalloc(&indices_in, vocab * sizeof(std::uint32_t));
        cudaMalloc(&indices_out, vocab * sizeof(std::uint32_t));
        cudaMalloc(&offsets, 2 * sizeof(std::uint32_t));
        const std::uint32_t offsets_host[2] = {0, vocab};
        cudaMemcpy(
            offsets, offsets_host, sizeof(offsets_host),
            cudaMemcpyHostToDevice);
        std::size_t temp_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, temp_bytes,
            keys_in, keys_out, indices_in, indices_out,
            static_cast<int>(vocab), 1, offsets, offsets + 1);
        void* temp = nullptr;
        cudaMalloc(&temp, temp_bytes);
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        GroupedNucleusLaunch launch{
            0, 1, 2, 4, UINT32_MAX, 1, vocab, 1, 0};
        k_grouped_nucleus_probabilities<<<1, kCanonicalReduceWidth>>>(
            header_device,
            lane_device,
            nullptr,
            nullptr,
            values_device,
            5,
            probabilities_device,
            launch);
        k_grouped_nucleus_sort_keys<<<
            grouped_grid(vocab), kTier0Block>>>(
            header_device, lane_device, probabilities_device,
            keys_in, indices_in, 1, vocab);
        cub::DeviceSegmentedRadixSort::SortPairs(
            temp, temp_bytes,
            keys_in, keys_out, indices_in, indices_out,
            static_cast<int>(vocab), 1, offsets, offsets + 1);
        k_grouped_nucleus_sorted_finish<<<1, 1>>>(
            header_device, lane_device, nullptr, nullptr, nullptr,
            values_device, 5, probabilities_device,
            indices_out, launch);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        std::uint32_t actual = UINT32_MAX;
        cudaMemcpy(
            &actual, output_device, sizeof(actual), cudaMemcpyDeviceToHost);
        std::vector<std::uint32_t> sorted(vocab);
        cudaMemcpy(
            sorted.data(),
            indices_out,
            sorted.size() * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost);
        std::vector<std::uint32_t> reference_order(vocab);
        std::iota(reference_order.begin(), reference_order.end(), 0u);
        std::stable_sort(
            reference_order.begin(),
            reference_order.end(),
            [&](std::uint32_t left, std::uint32_t right) {
                const float a = probabilities[left];
                const float b = probabilities[right];
                const bool a_nan = std::isnan(a);
                const bool b_nan = std::isnan(b);
                if (a_nan != b_nan) return !a_nan;
                if (a_nan) return left < right;
                if (a == b) return left < right;
                return a > b;
            });
        expect(
            sorted == reference_order,
            "scalable nucleus stable NaN/tie/signed-zero order");
        float exclusive = 0.0f;
        std::vector<std::uint8_t> included(vocab, 0);
        for (const auto index : reference_order) {
            if (!(exclusive < top_p)) break;
            included[index] = 1;
            exclusive += probabilities[index];
        }
        std::uint32_t expected = 0;
        bool have_expected = false;
        const auto noise = host_eval::rng_keyed(
            state[0], state[1], vocab, true);
        float best = -std::numeric_limits<float>::infinity();
        for (std::uint32_t index = 0; index < vocab; ++index) {
            const float score = included[index] != 0
                ? logits[index] + noise[index]
                : -std::numeric_limits<float>::infinity();
            if (!std::isnan(score) &&
                (!have_expected || score > best ||
                 (score == best && index < expected))) {
                expected = index;
                best = score;
                have_expected = true;
            }
        }
        expect(
            actual == expected,
            "scalable nucleus production-vocab token parity");
        if (std::getenv("NV_SANITIZER_INJECTION_TRANSPORT_TYPE") == nullptr) {
            expect(
                elapsed_ms < 1000.0f,
                "scalable nucleus production-vocab event timing");
        }
        std::printf(
            "  nucleus V=%u adversarial=%d: %.3f ms\n",
            vocab,
            adversarial ? 1 : 0,
            elapsed_ms);
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(temp);
        cudaFree(offsets);
        cudaFree(indices_out);
        cudaFree(indices_in);
        cudaFree(keys_out);
        cudaFree(keys_in);
        cudaFree(values_device);
        cudaFree(lane_device);
        cudaFree(header_device);
        cudaFree(commit_device);
        cudaFree(output_device);
        cudaFree(state_device);
        cudaFree(top_p_device);
        cudaFree(probabilities_device);
        cudaFree(logits_device);
    };
    run(151936, false);
    run(248320, true);
}

void scalable_nucleus_batch_measurements() {
    constexpr std::uint32_t vocab = 151936;
    for (const std::uint32_t lane_count : {1u, 2u, 4u, 8u}) {
        std::vector<std::uint16_t> logits(
            static_cast<std::size_t>(lane_count) * vocab,
            bf16_bits(-10.0f));
        std::vector<std::uint32_t> expected(lane_count);
        for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
            expected[lane] = 17 + lane * 97;
            logits[
                static_cast<std::size_t>(lane) * vocab +
                expected[lane]] = bf16_bits(10.0f);
        }
        std::vector<float> top_p(lane_count, 0.9f);
        std::vector<std::uint32_t> states(lane_count * 2);
        std::vector<std::uint32_t> commits(lane_count, 1);
        for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
            states[lane * 2] = 1234 + lane;
            states[lane * 2 + 1] = lane;
        }

        std::uint16_t* logits_device = nullptr;
        float* top_p_device = nullptr;
        std::uint32_t* states_device = nullptr;
        std::uint32_t* outputs_device = nullptr;
        std::uint32_t* commits_device = nullptr;
        float* probabilities_device = nullptr;
        cudaMalloc(
            &logits_device, logits.size() * sizeof(std::uint16_t));
        cudaMalloc(&top_p_device, top_p.size() * sizeof(float));
        cudaMalloc(
            &states_device, states.size() * sizeof(std::uint32_t));
        cudaMalloc(
            &outputs_device, lane_count * sizeof(std::uint32_t));
        cudaMalloc(
            &commits_device, lane_count * sizeof(std::uint32_t));
        cudaMalloc(
            &probabilities_device,
            static_cast<std::size_t>(lane_count) * vocab * sizeof(float));
        cudaMemcpy(
            logits_device, logits.data(),
            logits.size() * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice);
        cudaMemcpy(
            top_p_device, top_p.data(),
            top_p.size() * sizeof(float),
            cudaMemcpyHostToDevice);
        cudaMemcpy(
            states_device, states.data(),
            states.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice);
        cudaMemcpy(
            commits_device, commits.data(),
            commits.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice);

        std::vector<std::uint64_t> row_pointers(lane_count);
        for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
            row_pointers[lane] = reinterpret_cast<std::uint64_t>(
                logits_device + static_cast<std::size_t>(lane) * vocab);
        }
        std::uint64_t* row_pointers_device = nullptr;
        cudaMalloc(
            &row_pointers_device,
            row_pointers.size() * sizeof(std::uint64_t));
        cudaMemcpy(
            row_pointers_device,
            row_pointers.data(),
            row_pointers.size() * sizeof(std::uint64_t),
            cudaMemcpyHostToDevice);

        PtirLaneTableHeader header_host{
            PTIR_LANE_TABLE_ABI_VERSION, lane_count, 0,
            kGroupedLaneFlagBf16Rows};
        std::vector<PtirLaneRecord> lanes_host(lane_count);
        constexpr std::uint32_t value_count = 4;
        std::vector<std::uint64_t> values_host(
            static_cast<std::size_t>(lane_count) * value_count);
        for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
            lanes_host[lane].logits_base =
                reinterpret_cast<std::uint64_t>(
                    row_pointers_device + lane);
            lanes_host[lane].logits_row_count = 1;
            lanes_host[lane].sampled_rows = 1;
            lanes_host[lane].commit_slot =
                reinterpret_cast<std::uint64_t>(
                    commits_device + lane);
            const std::size_t base =
                static_cast<std::size_t>(lane) * value_count;
            values_host[base] = 0;
            values_host[base + 1] =
                reinterpret_cast<std::uint64_t>(top_p_device + lane);
            values_host[base + 2] =
                reinterpret_cast<std::uint64_t>(
                    states_device + lane * 2);
            values_host[base + 3] =
                reinterpret_cast<std::uint64_t>(
                    outputs_device + lane);
        }
        PtirLaneTableHeader* header_device = nullptr;
        PtirLaneRecord* lanes_device = nullptr;
        std::uint64_t* values_device = nullptr;
        cudaMalloc(&header_device, sizeof(header_host));
        cudaMalloc(
            &lanes_device,
            lanes_host.size() * sizeof(PtirLaneRecord));
        cudaMalloc(
            &values_device,
            values_host.size() * sizeof(std::uint64_t));
        cudaMemcpy(
            header_device, &header_host, sizeof(header_host),
            cudaMemcpyHostToDevice);
        cudaMemcpy(
            lanes_device, lanes_host.data(),
            lanes_host.size() * sizeof(PtirLaneRecord),
            cudaMemcpyHostToDevice);
        cudaMemcpy(
            values_device, values_host.data(),
            values_host.size() * sizeof(std::uint64_t),
            cudaMemcpyHostToDevice);

        const std::size_t items =
            static_cast<std::size_t>(lane_count) * vocab;
        std::uint64_t* keys_in = nullptr;
        std::uint64_t* keys_out = nullptr;
        std::uint32_t* indices_in = nullptr;
        std::uint32_t* indices_out = nullptr;
        std::uint32_t* offsets = nullptr;
        cudaMalloc(&keys_in, items * sizeof(std::uint64_t));
        cudaMalloc(&keys_out, items * sizeof(std::uint64_t));
        cudaMalloc(&indices_in, items * sizeof(std::uint32_t));
        cudaMalloc(&indices_out, items * sizeof(std::uint32_t));
        cudaMalloc(
            &offsets,
            (static_cast<std::size_t>(lane_count) + 1) *
                sizeof(std::uint32_t));
        std::vector<std::uint32_t> offsets_host(lane_count + 1);
        for (std::uint32_t lane = 0; lane <= lane_count; ++lane) {
            offsets_host[lane] = lane * vocab;
        }
        cudaMemcpy(
            offsets, offsets_host.data(),
            offsets_host.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice);
        std::size_t temp_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, temp_bytes,
            keys_in, keys_out, indices_in, indices_out,
            static_cast<int>(items), static_cast<int>(lane_count),
            offsets, offsets + 1);
        void* temp = nullptr;
        cudaMalloc(&temp, temp_bytes);
        GroupedNucleusLaunch launch{
            0, 1, 2, 3, UINT32_MAX, 1, vocab, 1, 1};
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        k_grouped_nucleus_probabilities<<<
            lane_count, kCanonicalReduceWidth>>>(
            header_device,
            lanes_device,
            nullptr,
            nullptr,
            values_device,
            value_count,
            probabilities_device,
            launch);
        k_grouped_nucleus_sort_keys<<<
            grouped_grid(items), kTier0Block>>>(
            header_device,
            lanes_device,
            probabilities_device,
            keys_in,
            indices_in,
            1,
            vocab);
        cub::DeviceSegmentedRadixSort::SortPairs(
            temp, temp_bytes,
            keys_in, keys_out, indices_in, indices_out,
            static_cast<int>(items), static_cast<int>(lane_count),
            offsets, offsets + 1);
        k_grouped_nucleus_sorted_finish<<<
            (lane_count + 127) / 128, 128>>>(
            header_device,
            lanes_device,
            nullptr,
            nullptr,
            nullptr,
            values_device,
            value_count,
            probabilities_device,
            indices_out,
            launch);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        std::vector<std::uint32_t> actual(lane_count);
        cudaMemcpy(
            actual.data(), outputs_device,
            actual.size() * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost);
        expect(
            actual == expected,
            "production-vocab direct BF16 nucleus B=" +
                std::to_string(lane_count));
        if (std::getenv("NV_SANITIZER_INJECTION_TRANSPORT_TYPE") == nullptr) {
            expect(
                elapsed_ms < 1000.0f,
                "production-vocab nucleus batch event timing");
        }
        std::printf(
            "  nucleus B=%u V=%u direct_bf16=1: %.3f ms\n",
            lane_count, vocab, elapsed_ms);

        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(temp);
        cudaFree(offsets);
        cudaFree(indices_out);
        cudaFree(indices_in);
        cudaFree(keys_out);
        cudaFree(keys_in);
        cudaFree(values_device);
        cudaFree(lanes_device);
        cudaFree(header_device);
        cudaFree(row_pointers_device);
        cudaFree(probabilities_device);
        cudaFree(commits_device);
        cudaFree(outputs_device);
        cudaFree(states_device);
        cudaFree(top_p_device);
        cudaFree(logits_device);
    }
}

void nucleus_region_input_source_case() {
    constexpr std::uint32_t vocab = 8;
    const std::uint32_t rng_state[2] = {1234, 0};
    const float top_p = 0.01f;
    std::vector<float> materialized(vocab, -20.0f);
    materialized[3] = 20.0f;
    std::vector<std::uint16_t> direct(vocab, bf16_bits(-20.0f));
    direct[6] = bf16_bits(20.0f);

    std::uint32_t* rng_device = nullptr;
    float* top_p_device = nullptr;
    float* materialized_device = nullptr;
    std::uint16_t* direct_device = nullptr;
    std::uint32_t* output_device = nullptr;
    std::uint32_t* commit_device = nullptr;
    std::uint64_t* row_table_device = nullptr;
    std::uint64_t* values_device = nullptr;
    PtirLaneTableHeader* header_device = nullptr;
    PtirLaneRecord* lane_device = nullptr;
    cudaMalloc(&rng_device, sizeof(rng_state));
    cudaMalloc(&top_p_device, sizeof(top_p));
    cudaMalloc(
        &materialized_device, materialized.size() * sizeof(float));
    cudaMalloc(&direct_device, direct.size() * sizeof(std::uint16_t));
    cudaMalloc(&output_device, sizeof(std::uint32_t));
    cudaMalloc(&commit_device, sizeof(std::uint32_t));
    cudaMalloc(&row_table_device, sizeof(std::uint64_t));
    cudaMalloc(&values_device, 5 * sizeof(std::uint64_t));
    cudaMalloc(&header_device, sizeof(PtirLaneTableHeader));
    cudaMalloc(&lane_device, sizeof(PtirLaneRecord));
    cudaMemcpy(
        rng_device, rng_state, sizeof(rng_state), cudaMemcpyHostToDevice);
    cudaMemcpy(
        top_p_device, &top_p, sizeof(top_p), cudaMemcpyHostToDevice);
    cudaMemcpy(
        materialized_device, materialized.data(),
        materialized.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(
        direct_device, direct.data(),
        direct.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    const std::uint32_t one = 1;
    cudaMemcpy(
        commit_device, &one, sizeof(one), cudaMemcpyHostToDevice);
    const std::uint64_t direct_row =
        reinterpret_cast<std::uint64_t>(direct_device);
    cudaMemcpy(
        row_table_device, &direct_row, sizeof(direct_row),
        cudaMemcpyHostToDevice);
    const std::uint64_t values[5] = {
        reinterpret_cast<std::uint64_t>(rng_device),
        reinterpret_cast<std::uint64_t>(top_p_device),
        0,
        reinterpret_cast<std::uint64_t>(materialized_device),
        reinterpret_cast<std::uint64_t>(output_device),
    };
    cudaMemcpy(
        values_device, values, sizeof(values), cudaMemcpyHostToDevice);
    const PtirLaneTableHeader header{
        PTIR_LANE_TABLE_ABI_VERSION, 1, 0, 0};
    PtirLaneRecord lane{};
    lane.logits_base =
        reinterpret_cast<std::uint64_t>(row_table_device);
    lane.logits_row_count = 1;
    lane.sampled_rows = 1;
    lane.commit_slot =
        reinterpret_cast<std::uint64_t>(commit_device);
    cudaMemcpy(
        header_device, &header, sizeof(header), cudaMemcpyHostToDevice);
    cudaMemcpy(
        lane_device, &lane, sizeof(lane), cudaMemcpyHostToDevice);

    GroupedNucleusLaunch launch{
        3, 1, 0, 4, UINT32_MAX, 1, vocab, 1, 0};
    k_grouped_nucleus_sample<<<1, kCanonicalReduceWidth>>>(
        header_device, lane_device, nullptr, nullptr, nullptr,
        values_device, 5, launch);
    std::uint32_t materialized_token = UINT32_MAX;
    cudaMemcpy(
        &materialized_token, output_device, sizeof(materialized_token),
        cudaMemcpyDeviceToHost);
    launch.logits_kind = 1;
    k_grouped_nucleus_sample<<<1, kCanonicalReduceWidth>>>(
        header_device, lane_device, nullptr, nullptr, nullptr,
        values_device, 5, launch);
    std::uint32_t direct_token = UINT32_MAX;
    cudaMemcpy(
        &direct_token, output_device, sizeof(direct_token),
        cudaMemcpyDeviceToHost);
    expect(
        materialized_token == 3 && direct_token == 6,
        "nucleus region uses materialized generic logits and raw direct rows");

    cudaFree(lane_device);
    cudaFree(header_device);
    cudaFree(values_device);
    cudaFree(row_table_device);
    cudaFree(commit_device);
    cudaFree(output_device);
    cudaFree(direct_device);
    cudaFree(materialized_device);
    cudaFree(top_p_device);
    cudaFree(rng_device);
}

void ragged_layout_kernel_case() {
    constexpr std::uint32_t lane_count = 2;
    constexpr std::uint32_t columns = 4;
    constexpr std::uint32_t max_rows = 4;
    constexpr std::uint32_t max_numel = max_rows * columns;
    constexpr std::uint32_t value_count = 7;
    const std::uint32_t rows[lane_count] = {3, 4};
    std::vector<float> input(lane_count * max_numel, 0.0f);
    std::vector<std::uint32_t> indices(lane_count * max_numel, 0);
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        for (std::uint32_t index = 0; index < rows[lane] * columns; ++index) {
            input[lane * max_numel + index] =
                static_cast<float>(lane * 100 + index);
            indices[lane * max_numel + index] = index;
        }
    }
    float* input_device = nullptr;
    std::uint32_t* indices_device = nullptr;
    float* transpose_device = nullptr;
    float* sort_values_device = nullptr;
    std::uint32_t* sort_indices_device = nullptr;
    float* scatter_device = nullptr;
    std::uint32_t* commits_device = nullptr;
    cudaMalloc(&input_device, input.size() * sizeof(float));
    cudaMalloc(&indices_device, indices.size() * sizeof(std::uint32_t));
    cudaMalloc(&transpose_device, input.size() * sizeof(float));
    cudaMalloc(&sort_values_device, input.size() * sizeof(float));
    cudaMalloc(&sort_indices_device, indices.size() * sizeof(std::uint32_t));
    cudaMalloc(&scatter_device, input.size() * sizeof(float));
    cudaMalloc(&commits_device, lane_count * sizeof(std::uint32_t));
    cudaMemset(
        transpose_device, 0, input.size() * sizeof(float));
    cudaMemset(
        sort_values_device, 0, input.size() * sizeof(float));
    cudaMemset(
        sort_indices_device, 0,
        indices.size() * sizeof(std::uint32_t));
    cudaMemset(
        scatter_device, 0, input.size() * sizeof(float));
    cudaMemcpy(
        input_device,
        input.data(),
        input.size() * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        indices_device,
        indices.data(),
        indices.size() * sizeof(std::uint32_t),
        cudaMemcpyHostToDevice);
    const std::uint32_t commits[lane_count] = {1, 1};
    cudaMemcpy(
        commits_device,
        commits,
        sizeof(commits),
        cudaMemcpyHostToDevice);
    PtirLaneTableHeader header_host{
        PTIR_LANE_TABLE_ABI_VERSION, lane_count, 0, 0};
    PtirLaneRecord lanes_host[lane_count]{};
    std::uint64_t values_host[lane_count * value_count]{};
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        lanes_host[lane].sampled_rows = rows[lane];
        lanes_host[lane].row_count = rows[lane];
        lanes_host[lane].commit_slot =
            reinterpret_cast<std::uint64_t>(commits_device + lane);
        const auto value_offset =
            static_cast<std::size_t>(lane) * value_count;
        const auto element_offset =
            static_cast<std::size_t>(lane) * max_numel;
        values_host[value_offset + 0] =
            reinterpret_cast<std::uint64_t>(input_device + element_offset);
        values_host[value_offset + 1] = reinterpret_cast<std::uint64_t>(
            transpose_device + element_offset);
        values_host[value_offset + 2] = reinterpret_cast<std::uint64_t>(
            sort_values_device + element_offset);
        values_host[value_offset + 3] = reinterpret_cast<std::uint64_t>(
            sort_indices_device + element_offset);
        values_host[value_offset + 4] = reinterpret_cast<std::uint64_t>(
            indices_device + element_offset);
        values_host[value_offset + 5] =
            reinterpret_cast<std::uint64_t>(input_device + element_offset);
        values_host[value_offset + 6] = reinterpret_cast<std::uint64_t>(
            scatter_device + element_offset);
    }
    PtirLaneTableHeader* header_device = nullptr;
    PtirLaneRecord* lanes_device = nullptr;
    std::uint64_t* values_device = nullptr;
    cudaMalloc(&header_device, sizeof(header_host));
    cudaMalloc(&lanes_device, sizeof(lanes_host));
    cudaMalloc(&values_device, sizeof(values_host));
    cudaMemcpy(
        header_device,
        &header_host,
        sizeof(header_host),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        lanes_device,
        lanes_host,
        sizeof(lanes_host),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        values_device,
        values_host,
        sizeof(values_host),
        cudaMemcpyHostToDevice);
    const GroupedDynamicShape shape{
        max_numel, columns, PTIR_EXTENT_SAMPLED_ROWS, {0, 0, 0}};
    k_grouped_transpose<float><<<
        grouped_grid(lane_count * max_numel), kTier0Block>>>(
        header_device,
        lanes_device,
        values_device,
        value_count,
        0,
        1,
        GroupedRowShape{
            GroupedDynamicShape{
                max_rows, 1, PTIR_EXTENT_SAMPLED_ROWS, {0, 0, 0}},
            GroupedDynamicShape{columns, columns, 0xff, {0, 0, 0}},
            max_rows,
            columns,
        });
    k_grouped_topk<<<lane_count, kTier0Block>>>(
        header_device,
        lanes_device,
        nullptr,
        nullptr,
        values_device,
        value_count,
        0,
        2,
        3,
        1,
        max_numel,
        max_numel,
        true,
        shape,
        GroupedRowShape{
            GroupedDynamicShape{1, 1, 0xff, {0, 0, 0}},
            shape,
            1,
            max_numel,
        },
        columns,
        0);
    k_grouped_scatter<float, std::uint32_t, false><<<
        lane_count, kTier0Block>>>(
        header_device,
        lanes_device,
        values_device,
        value_count,
        0,
        4,
        5,
        6,
        shape,
        shape,
        shape,
        false);
    cudaDeviceSynchronize();
    std::vector<float> transpose(input.size());
    std::vector<float> sort_values(input.size());
    std::vector<std::uint32_t> sort_indices(indices.size());
    std::vector<float> scatter(input.size());
    cudaMemcpy(
        transpose.data(),
        transpose_device,
        transpose.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        sort_values.data(),
        sort_values_device,
        sort_values.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        sort_indices.data(),
        sort_indices_device,
        sort_indices.size() * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        scatter.data(),
        scatter_device,
        scatter.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        const auto base = static_cast<std::size_t>(lane) * max_numel;
        const auto actual = rows[lane] * columns;
        for (std::uint32_t row = 0; row < rows[lane]; ++row) {
            for (std::uint32_t column = 0; column < columns; ++column) {
                expect(
                    transpose[
                        base + column * rows[lane] + row] ==
                        input[base + row * columns + column],
                    "ragged transpose compact layout");
            }
        }
        for (std::uint32_t index = 0; index < actual; ++index) {
            expect(
                sort_indices[base + index] == actual - index - 1 &&
                    sort_values[base + index] ==
                        input[base + actual - index - 1],
                "ragged global SortDesc");
            expect(
                scatter[base + index] == input[base + index],
                "ragged ScatterSet");
        }
    }
    cudaFree(values_device);
    cudaFree(lanes_device);
    cudaFree(header_device);
    cudaFree(commits_device);
    cudaFree(scatter_device);
    cudaFree(sort_indices_device);
    cudaFree(sort_values_device);
    cudaFree(transpose_device);
    cudaFree(indices_device);
    cudaFree(input_device);
}

void register_rule_case() {
    constexpr std::uint32_t lane_count = 2;
    Trace trace;
    Channel state;
    state.id = 0;
    state.type = {Shape::vec(1), DType::U32};
    state.capacity = 1;
    state.has_seed = true;
    Channel output;
    output.id = 1;
    output.type = {Shape::vec(1), DType::U32};
    output.capacity = 1;
    output.host_visible = true;
    output.host_reader = true;
    trace.channels = {state, output};
    Stage stage;
    stage.kind = StageKind::Epilogue;
    stage.takes = {0};
    stage.reads = {0};
    stage.puts = {{0, 0}, {0, 3}, {1, 4}};
    trace.stages = {stage};

    plan::StagePlan plan;
    plan.stage = PTIR_STAGE_EPILOGUE;
    plan.signature_hash = 0x98f69a334b63a1d2ULL;
    plan.signature.assign(
        {'r', 'e', 'g', 'i', 's', 't', 'e', 'r', '-', 'r', 'u', 'l', 'e'});
    plan.channel_bindings = {0, 1};
    container::COp seven;
    seven.tag = PTIR_OP_CONST;
    seven.lit_dtype = PTIR_DT_U32;
    seven.lit_bits = 7;
    container::COp put_seven;
    put_seven.tag = PTIR_OP_CHAN_PUT;
    put_seven.chan = 0;
    put_seven.args = {0};
    put_seven.results = 0;
    container::COp take;
    take.tag = PTIR_OP_CHAN_TAKE;
    take.chan = 0;
    container::COp one;
    one.tag = PTIR_OP_CONST;
    one.lit_dtype = PTIR_DT_U32;
    one.lit_bits = 1;
    container::COp add;
    add.tag = PTIR_OP_ADD;
    add.args = {1, 2};
    container::COp put_eight;
    put_eight.tag = PTIR_OP_CHAN_PUT;
    put_eight.chan = 0;
    put_eight.args = {3};
    put_eight.results = 0;
    container::COp read;
    read.tag = PTIR_OP_CHAN_READ;
    read.chan = 0;
    container::COp put_output;
    put_output.tag = PTIR_OP_CHAN_PUT;
    put_output.chan = 1;
    put_output.args = {4};
    put_output.results = 0;
    plan.ops = {
        {seven, {0}},
        {put_seven, {1}},
        {take, {2}},
        {one, {3}},
        {add, {4}},
        {put_eight, {5}},
        {read, {6}},
        {put_output, {7}},
    };
    for (int value = 0; value < 5; ++value) {
        plan.value_types.push_back(plan_type(PTIR_DT_U32, {1}, 0));
    }
    plan.singleton.kind = 0;
    plan.fused.kind = 1;

    DeviceChannelRegistry registry;
    std::vector<std::unique_ptr<PtirInstance>> instances;
    std::vector<std::vector<PieChannelEndpointBinding>> endpoints(lane_count);
    std::vector<std::vector<DeviceHostChannelTicket>> tickets(lane_count);
    std::vector<GroupedLaneBinding> bindings;
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        endpoints[lane].resize(2);
        std::vector<std::uint64_t> ids = {
            30000 + lane * 10,
            30001 + lane * 10,
        };
        for (std::uint32_t dense = 0; dense < 2; ++dense) {
            const Channel& channel = trace.channels[dense];
            PieChannelDesc descriptor{};
            descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
            descriptor.channel_id = ids[dense];
            descriptor.shape = {
                channel.type.shape.dims.data(),
                channel.type.shape.dims.size(),
            };
            descriptor.dtype = static_cast<std::uint8_t>(channel.type.dtype);
            descriptor.host_role = channel.host_reader
                ? PIE_CHANNEL_HOST_ROLE_READER
                : PIE_CHANNEL_HOST_ROLE_NONE;
            descriptor.seeded = channel.has_seed;
            descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
            descriptor.capacity = 1;
            descriptor.reader_wait_id = ids[dense] * 2 + 1;
            descriptor.writer_wait_id = ids[dense] * 2 + 2;
            std::string error;
            expect(
                registry.register_endpoint(
                    descriptor, &endpoints[lane][dense], &error),
                "register-rule endpoint: " + error);
        }
        const std::uint32_t seed = 5;
        std::vector<ChannelValue> seeds{{
            ids[0],
            std::vector<std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(&seed),
                reinterpret_cast<const std::uint8_t*>(&seed) + sizeof(seed)),
        }};
        std::string error;
        auto instance = std::make_unique<PtirInstance>(
            trace, &registry, ids, seeds, &error);
        expect(instance->ok(), "register-rule instance: " + error);
        const std::uint32_t state_slot = instance->view().slot(0);
        const std::uint32_t output_slot = instance->view().slot(1);
        tickets[lane] = {
            DeviceHostChannelTicket{
                .slot = state_slot,
                .flags = kTicketConsume | kTicketPublish | kTicketRequireInput,
                .expected_head = 0,
                .expected_tail = 1,
                .words = registry.host_words(state_slot),
                .mirror = static_cast<const std::uint8_t*>(
                    registry.host_mirror(state_slot)),
                .cells = static_cast<std::uint8_t*>(
                    registry.cell_base(state_slot)),
                .cap1 = 2,
                .wire_bytes = 4,
                .native_bytes = 4,
            },
            DeviceHostChannelTicket{
                .slot = output_slot,
                .flags = kTicketPublish,
                .expected_head = kNoChannelTicket,
                .expected_tail = 0,
                .words = registry.host_words(output_slot),
                .mirror = static_cast<const std::uint8_t*>(
                    registry.host_mirror(output_slot)),
                .cells = static_cast<std::uint8_t*>(
                    registry.cell_base(output_slot)),
                .cap1 = 2,
                .wire_bytes = 4,
                .native_bytes = 4,
            },
        };
        instances.push_back(std::move(instance));
        bindings.push_back({
            .instance = instances.back().get(),
            .plan = &plan,
            .tickets = &tickets[lane],
            .logits_row_count = 1,
            .vocab = 1,
            .program_index = lane,
        });
    }
    std::string reason;
    expect(
        grouped_stage_supported(bindings, &reason),
        "register-rule group supported: " + reason);
    GroupedLaunchResult launch = GroupedTier0Executor::run(bindings, nullptr);
    cudaDeviceSynchronize();
    if (launch.device_tickets != nullptr) cudaFree(launch.device_tickets);
    for (auto& instance : instances) {
        instance->view().sync_host_rings();
        std::uint32_t state_value = 0;
        instance->view().read_committed(0, &state_value, sizeof(state_value));
        expect(
            state_value == 5,
            "put-then-take preserves unread seeded state");
        expect(
            !instance->view().committed_full(1),
            "put-then-take readiness miss publishes no output");
    }
}

void structured_mask_grouped_case() {
    constexpr std::uint32_t lane_count = 2;
    plan::StagePlan plan;
    plan.stage = PTIR_STAGE_EPILOGUE;
    plan.signature_hash = 0x93a47126df71c805ULL;
    plan.signature.assign(
        {'s', 't', 'r', 'u', 'c', 't', 'u', 'r', 'e', 'd', '-', 'm', 'a', 's', 'k'});
    plan.channel_bindings = {0, 1, 2, 3};
    container::COp positions;
    positions.tag = PTIR_OP_CHAN_TAKE;
    positions.chan = 0;
    container::COp causal;
    causal.tag = PTIR_OP_CAUSAL_MASK;
    causal.args = {0};
    causal.imm = 6;
    container::COp sliding;
    sliding.tag = PTIR_OP_SLIDING_WINDOW_MASK;
    sliding.args = {0};
    sliding.imm = 6;
    sliding.imm2 = 3;
    container::COp sink;
    sink.tag = PTIR_OP_SINK_WINDOW_MASK;
    sink.args = {0};
    sink.imm = 6;
    sink.imm2 = 1;
    sink.imm3 = 3;
    plan.ops = {
        {positions, {0}},
        {causal, {1}},
        {sliding, {2}},
        {sink, {3}},
    };
    for (std::uint32_t channel = 1; channel < 4; ++channel) {
        container::COp put;
        put.tag = PTIR_OP_CHAN_PUT;
        put.chan = channel;
        put.args = {channel};
        put.results = 0;
        plan.ops.push_back({put, {static_cast<std::uint32_t>(plan.ops.size())}});
    }
    plan.value_types = {
        plan_type(PTIR_DT_U32, {2}, 1),
        plan_type(PTIR_DT_BOOL, {2, 6}, 4),
        plan_type(PTIR_DT_BOOL, {2, 6}, 4),
        plan_type(PTIR_DT_BOOL, {2, 6}, 4),
    };
    plan.singleton.kind = 0;
    plan.fused.kind = 1;

    Trace trace;
    const Shape shapes[] = {
        Shape::vec(2),
        Shape::mat(2, 6),
        Shape::mat(2, 6),
        Shape::mat(2, 6),
    };
    for (std::uint32_t channel = 0; channel < 4; ++channel) {
        Channel value;
        value.id = channel;
        value.type = {
            shapes[channel],
            channel == 0 ? DType::U32 : DType::Bool,
        };
        value.capacity = 1;
        value.has_seed = channel == 0;
        value.host_visible = channel != 0;
        value.host_reader = channel != 0;
        trace.channels.push_back(value);
    }
    Stage stage;
    stage.kind = StageKind::Epilogue;
    stage.takes = {0};
    stage.puts = {{1, 1}, {2, 2}, {3, 3}};
    for (std::uint32_t result = 1; result < 4; ++result) {
        Op op;
        op.result_id = result;
        op.result_count = 1;
        if (result == 1) {
            op.code = OpCode::CausalMask;
            op.args = {0};
            op.imm = 6;
        } else if (result == 2) {
            op.code = OpCode::SlidingWindowMask;
            op.args = {0};
            op.imm = 6;
            op.imm2 = 3;
        } else {
            op.code = OpCode::SinkWindowMask;
            op.args = {0};
            op.imm = 6;
            op.imm2 = 1;
            op.imm3 = 3;
        }
        stage.ops.push_back(op);
    }
    trace.stages = {stage};
    trace.ports = {
        {.port = descriptor::kPortPositions, .channel = 0},
        {.port = descriptor::kPortAttnMask, .channel = 2},
    };
    const auto causal_descriptor =
        detail::structured_mask_descriptor(trace, 1);
    const auto sliding_descriptor =
        detail::structured_mask_descriptor(trace, 2);
    const auto sink_descriptor =
        detail::structured_mask_descriptor(trace, 3);
    expect(
        causal_descriptor.kind == StructuredMaskKind::Causal &&
            causal_descriptor.key_len == 6 &&
            sliding_descriptor.kind ==
                StructuredMaskKind::SlidingWindow &&
            sliding_descriptor.window == 3 &&
            sink_descriptor.kind == StructuredMaskKind::SinkWindow &&
            sink_descriptor.sink == 1 && sink_descriptor.window == 3,
        "semantic mask descriptors bypass dense composition when eligible");

    const std::uint32_t position_values[2] = {3, 5};
    const std::uint8_t expected_causal[12] = {
        1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1};
    const std::uint8_t expected_sliding[12] = {
        0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1};
    const std::uint8_t expected_sink[12] = {
        1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1};

    DeviceChannelRegistry registry;
    std::vector<std::unique_ptr<PtirInstance>> instances;
    std::vector<std::vector<PieChannelEndpointBinding>> endpoints(
        lane_count, std::vector<PieChannelEndpointBinding>(4));
    std::vector<std::vector<DeviceHostChannelTicket>> tickets(lane_count);
    std::vector<GroupedLaneBinding> bindings;
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        std::vector<std::uint64_t> ids(4);
        for (std::uint32_t channel = 0; channel < 4; ++channel) {
            ids[channel] = 67000 + lane * 16 + channel;
            const Channel& source = trace.channels[channel];
            PieChannelDesc descriptor{};
            descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
            descriptor.channel_id = ids[channel];
            descriptor.shape = {
                source.type.shape.dims.data(),
                source.type.shape.dims.size(),
            };
            descriptor.dtype = static_cast<std::uint8_t>(source.type.dtype);
            descriptor.host_role = source.host_reader
                ? PIE_CHANNEL_HOST_ROLE_READER
                : PIE_CHANNEL_HOST_ROLE_NONE;
            descriptor.seeded = source.has_seed;
            descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
            descriptor.capacity = 1;
            descriptor.reader_wait_id = ids[channel] * 2 + 1;
            descriptor.writer_wait_id = ids[channel] * 2 + 2;
            std::string error;
            expect(
                registry.register_endpoint(
                    descriptor, &endpoints[lane][channel], &error),
                "register structured mask channel: " + error);
        }
        auto bytes = [](const void* data, std::size_t size) {
            const auto* begin = static_cast<const std::uint8_t*>(data);
            return std::vector<std::uint8_t>(begin, begin + size);
        };
        std::vector<ChannelValue> seeds{
            {ids[0], bytes(position_values, sizeof(position_values))},
        };
        std::string error;
        auto instance = std::make_unique<PtirInstance>(
            trace, &registry, ids, seeds, &error);
        expect(instance->ok(), "bind structured mask instance: " + error);
        if (lane == 0) {
            FireGeometry geometry;
            expect(
                resolve_fire_geometry(
                    trace,
                    instance->view(),
                    16,
                    geometry,
                    &error,
                    true) &&
                    geometry.mask.empty() &&
                    !geometry.has_mask &&
                    geometry.structured_mask.kind ==
                        StructuredMaskKind::SlidingWindow,
                "eligible semantic descriptor skips dense mask read/materialization");
        }
        for (std::uint32_t channel = 0; channel < 4; ++channel) {
            const std::uint32_t slot = instance->view().slot(channel);
            tickets[lane].push_back({
                .slot = slot,
                .flags = channel == 0
                    ? kTicketConsume | kTicketRequireInput
                    : kTicketPublish,
                .expected_head = channel == 0 ? 0 : kNoChannelTicket,
                .expected_tail = channel == 0 ? kNoChannelTicket : 0,
                .words = registry.host_words(slot),
                .mirror = static_cast<const std::uint8_t*>(
                    registry.host_mirror(slot)),
                .cells = static_cast<std::uint8_t*>(
                    registry.cell_base(slot)),
                .cap1 = 2,
                .wire_bytes = static_cast<std::uint32_t>(
                    instance->view().cell_bytes(channel)),
                .native_bytes = static_cast<std::uint32_t>(
                    instance->view().cell_bytes(channel)),
            });
        }
        instances.push_back(std::move(instance));
        bindings.push_back({
            .instance = instances.back().get(),
            .plan = &plan,
            .tickets = &tickets[lane],
            .logits_row_count = 1,
            .vocab = 1,
            .program_index = lane,
        });
    }
    std::string reason;
    expect(
        grouped_stage_supported(bindings, &reason),
        "structured masks use grouped exact fallback: " + reason);
    auto launch = GroupedTier0Executor::run(bindings, nullptr);
    cudaDeviceSynchronize();
    if (launch.device_tickets != nullptr) cudaFree(launch.device_tickets);
    for (auto& instance : instances) {
        instance->view().sync_host_rings();
        std::uint8_t actual[12]{};
        instance->view().read_committed(1, actual, 12);
        expect(
            std::equal(std::begin(actual), std::end(actual),
                       std::begin(expected_causal)),
            "grouped causal mask parity");
        instance->view().read_committed(2, actual, 12);
        expect(
            std::equal(std::begin(actual), std::end(actual),
                       std::begin(expected_sliding)),
            "grouped sliding mask parity");
        instance->view().read_committed(3, actual, 12);
        expect(
            std::equal(std::begin(actual), std::end(actual),
                       std::begin(expected_sink)),
            "grouped sink-window mask parity");
    }
}

void ragged_rows_case() {
    constexpr std::uint32_t lane_count = 3;
    constexpr std::uint32_t vocab = 8;
    const std::uint32_t row_counts[lane_count] = {0, 2, 4};
    plan::StagePlan plan;
    plan.stage = PTIR_STAGE_EPILOGUE;
    plan.signature_hash = 0x21a4f52be0d2c881ULL;
    plan.signature.assign(
        {'r', 'a', 'g', 'g', 'e', 'd', '-', 'a', 'r', 'g', 'm', 'a', 'x'});
    plan.channel_bindings = {0};
    container::COp logits;
    logits.tag = PTIR_OP_INTRINSIC_VAL;
    logits.intr = PTIR_INTR_LOGITS;
    logits.results = 1;
    container::COp argmax;
    argmax.tag = PTIR_OP_REDUCE_ARGMAX;
    argmax.args = {0};
    argmax.results = 1;
    container::COp sum;
    sum.tag = PTIR_OP_REDUCE_SUM;
    sum.args = {1};
    sum.results = 1;
    container::COp put;
    put.tag = PTIR_OP_CHAN_PUT;
    put.chan = 0;
    put.args = {2};
    put.results = 0;
    plan.ops = {
        {logits, {0}},
        {argmax, {1}},
        {sum, {2}},
        {put, {3}},
    };
    plan.value_types = {
        sampled_rows_type(PTIR_DT_F32, {vocab}, 2),
        sampled_rows_type(PTIR_DT_U32, {}, 0),
        plan_type(PTIR_DT_U32, {}, 0),
    };
    plan.singleton.kind = 0;
    plan.fused.kind = 1;

    DeviceChannelRegistry registry;
    std::vector<Trace> traces(lane_count);
    std::vector<std::unique_ptr<PtirInstance>> instances;
    std::vector<PieChannelEndpointBinding> endpoints(lane_count);
    std::vector<std::vector<DeviceHostChannelTicket>> tickets(lane_count);
    std::vector<GroupedLaneBinding> bindings;
    std::uint32_t total_rows = 0;
    for (const auto rows : row_counts) total_rows += rows;
    std::vector<float> logits_host(
        static_cast<std::size_t>(total_rows) * vocab, -50.0f);
    std::uint32_t row_offset = 0;
    std::uint32_t expected[lane_count]{};
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        Trace& trace = traces[lane];
        Channel output;
        output.id = 0;
        output.type = {Shape::vec(1), DType::U32};
        output.capacity = 1;
        output.host_visible = true;
        output.host_reader = true;
        trace.channels = {output};
        Stage stage;
        stage.kind = StageKind::Epilogue;
        stage.puts = {{0, 2}};
        trace.stages = {stage};
        const std::uint64_t channel_id = 60000 + lane;
        PieChannelDesc descriptor{};
        descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
        descriptor.channel_id = channel_id;
        descriptor.shape = {
            output.type.shape.dims.data(),
            output.type.shape.dims.size(),
        };
        descriptor.dtype = static_cast<std::uint8_t>(DType::U32);
        descriptor.host_role = PIE_CHANNEL_HOST_ROLE_READER;
        descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
        descriptor.capacity = 1;
        descriptor.reader_wait_id = channel_id * 2 + 1;
        descriptor.writer_wait_id = channel_id * 2 + 2;
        std::string error;
        expect(
            registry.register_endpoint(
                descriptor, &endpoints[lane], &error),
            "register ragged output: " + error);
        auto instance = std::make_unique<PtirInstance>(
            trace,
            &registry,
            std::vector<std::uint64_t>{channel_id},
            std::vector<ChannelValue>{},
            &error);
        expect(instance->ok(), "bind ragged instance: " + error);
        const std::uint32_t slot = instance->view().slot(0);
        tickets[lane] = {{
            .slot = slot,
            .flags = kTicketPublish,
            .expected_head = kNoChannelTicket,
            .expected_tail = 0,
            .words = registry.host_words(slot),
            .mirror = static_cast<const std::uint8_t*>(
                registry.host_mirror(slot)),
            .cells = static_cast<std::uint8_t*>(registry.cell_base(slot)),
            .cap1 = 2,
            .wire_bytes = static_cast<std::uint32_t>(
                sizeof(std::uint32_t)),
            .native_bytes = static_cast<std::uint32_t>(
                sizeof(std::uint32_t)),
        }};
        instances.push_back(std::move(instance));
        bindings.push_back({
            .instance = instances.back().get(),
            .plan = &plan,
            .tickets = &tickets[lane],
            .logits_row_offset = row_offset,
            .logits_row_count = row_counts[lane],
            .vocab = vocab,
            .program_index = lane,
        });
        for (std::uint32_t row = 0; row < row_counts[lane]; ++row) {
            const std::uint32_t token =
                (lane * 3 + row * 2 + 1) % vocab;
            expected[lane] += token;
            logits_host[
                static_cast<std::size_t>(row_offset + row) * vocab + token] =
                50.0f;
        }
        row_offset += row_counts[lane];
    }
    std::vector<std::uint16_t> logits_bf16(logits_host.size());
    std::transform(
        logits_host.begin(), logits_host.end(), logits_bf16.begin(), bf16_bits);
    std::uint16_t* logits_device = nullptr;
    cudaMalloc(
        &logits_device, logits_bf16.size() * sizeof(std::uint16_t));
    cudaMemcpy(
        logits_device,
        logits_bf16.data(),
        logits_bf16.size() * sizeof(std::uint16_t),
        cudaMemcpyHostToDevice);
    std::vector<std::vector<std::uint64_t>> row_tables(lane_count);
    row_offset = 0;
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        for (std::uint32_t row = 0; row < row_counts[lane]; ++row) {
            row_tables[lane].push_back(reinterpret_cast<std::uint64_t>(
                logits_device +
                static_cast<std::size_t>(row_offset + row) * vocab));
        }
        bindings[lane].logits_bf16_rows = &row_tables[lane];
        row_offset += row_counts[lane];
    }
    std::string reason;
    expect(
        grouped_stage_supported(bindings, &reason),
        "direct-BF16 sampled-row {0,2,4} lanes remain one grouped signature: " +
            reason);
    auto launch = GroupedTier0Executor::run(bindings, nullptr);
    cudaDeviceSynchronize();
    if (launch.device_tickets != nullptr) cudaFree(launch.device_tickets);
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        instances[lane]->view().sync_host_rings();
        std::uint32_t actual = UINT32_MAX;
        instances[lane]->view().read_committed(
            0, &actual, sizeof(actual));
        expect(
            actual == expected[lane],
            "ragged sampled-row reduction/effect attribution lane " +
                std::to_string(lane));
    }
    expect(
        launch.body_op_launches == 3,
        "ragged sampled rows launch once per executable op/effect, not once per lane");
    cudaFree(logits_device);
}

void fp32_dynamic_root_stride_case() {
    constexpr std::uint32_t lane_count = 2;
    constexpr std::uint32_t rows = 3;
    constexpr std::uint32_t logical_vocab = 5;
    constexpr std::uint32_t physical_stride = 8;
    constexpr std::uint32_t value_count = 1;
    std::vector<float> source(
        static_cast<std::size_t>(lane_count) * rows * physical_stride,
        10000.0f);
    std::vector<float> expected(
        static_cast<std::size_t>(lane_count) * rows * logical_vocab);
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t column = 0;
                 column < logical_vocab;
                 ++column) {
                const float value = static_cast<float>(
                    lane * 100 + row * 10 + column);
                source[
                    (static_cast<std::size_t>(lane) * rows + row) *
                        physical_stride +
                    column] = value;
                expected[
                    (static_cast<std::size_t>(lane) * rows + row) *
                        logical_vocab +
                    column] = value;
            }
        }
    }

    float* device_source = nullptr;
    float* device_output = nullptr;
    PtirLaneTableHeader* device_header = nullptr;
    PtirLaneRecord* device_lanes = nullptr;
    std::uint64_t* device_values = nullptr;
    std::uint64_t* device_sources = nullptr;
    cudaMalloc(&device_source, source.size() * sizeof(float));
    cudaMalloc(&device_output, expected.size() * sizeof(float));
    cudaMalloc(&device_header, sizeof(PtirLaneTableHeader));
    cudaMalloc(&device_lanes, lane_count * sizeof(PtirLaneRecord));
    cudaMalloc(&device_values, lane_count * value_count * sizeof(std::uint64_t));
    cudaMalloc(&device_sources, lane_count * sizeof(std::uint64_t));
    cudaMemcpy(
        device_source, source.data(), source.size() * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemset(device_output, 0, expected.size() * sizeof(float));

    const PtirLaneTableHeader header{
        PTIR_LANE_TABLE_ABI_VERSION, lane_count, 0, 0};
    PtirLaneRecord lanes[lane_count]{};
    std::uint64_t values[lane_count]{};
    std::uint64_t sources[lane_count]{};
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        lanes[lane].sampled_rows = rows;
        values[lane] = reinterpret_cast<std::uint64_t>(
            device_output +
            static_cast<std::size_t>(lane) * rows * logical_vocab);
        sources[lane] = reinterpret_cast<std::uint64_t>(
            device_source +
            static_cast<std::size_t>(lane) * rows * physical_stride);
    }
    cudaMemcpy(
        device_header, &header, sizeof(header), cudaMemcpyHostToDevice);
    cudaMemcpy(
        device_lanes, lanes, sizeof(lanes), cudaMemcpyHostToDevice);
    cudaMemcpy(
        device_values, values, sizeof(values), cudaMemcpyHostToDevice);
    cudaMemcpy(
        device_sources, sources, sizeof(sources), cudaMemcpyHostToDevice);
    const GroupedDynamicShape shape{
        static_cast<std::uint64_t>(rows) * logical_vocab,
        logical_vocab,
        PTIR_EXTENT_SAMPLED_ROWS,
        {},
    };
    k_grouped_copy_dynamic_root<<<1, 128>>>(
        device_header,
        device_lanes,
        device_values,
        value_count,
        0,
        device_sources,
        shape,
        sizeof(float),
        logical_vocab,
        physical_stride);
    cudaDeviceSynchronize();
    std::vector<float> actual(expected.size());
    cudaMemcpy(
        actual.data(), device_output, actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    expect(
        actual == expected,
        "FP32 dynamic-root copy honors physical stride for every logical row");
    cudaFree(device_sources);
    cudaFree(device_values);
    cudaFree(device_lanes);
    cudaFree(device_header);
    cudaFree(device_output);
    cudaFree(device_source);
}

void zero_length_gather_case() {
    PtirLaneTableHeader header{
        PTIR_LANE_TABLE_ABI_VERSION, 1, 0, 0};
    PtirLaneRecord lane{};
    lane.token_count = 0;
    std::uint32_t* commit = nullptr;
    std::uint32_t* indices = nullptr;
    float* output = nullptr;
    PtirLaneTableHeader* device_header = nullptr;
    PtirLaneRecord* device_lane = nullptr;
    std::uint64_t* device_values = nullptr;
    cudaMalloc(&commit, sizeof(std::uint32_t));
    cudaMalloc(&indices, 3 * sizeof(std::uint32_t));
    cudaMalloc(&output, 3 * sizeof(float));
    cudaMalloc(&device_header, sizeof(header));
    cudaMalloc(&device_lane, sizeof(lane));
    cudaMalloc(&device_values, 3 * sizeof(std::uint64_t));
    const std::uint32_t one = 1;
    const std::uint32_t zeros[3] = {0, 0, 0};
    const float nonzero[3] = {1.0f, 2.0f, 3.0f};
    cudaMemcpy(commit, &one, sizeof(one), cudaMemcpyHostToDevice);
    cudaMemcpy(indices, zeros, sizeof(zeros), cudaMemcpyHostToDevice);
    cudaMemcpy(output, nonzero, sizeof(nonzero), cudaMemcpyHostToDevice);
    lane.commit_slot = reinterpret_cast<std::uint64_t>(commit);
    const std::uint64_t values[3] = {
        0,
        reinterpret_cast<std::uint64_t>(indices),
        reinterpret_cast<std::uint64_t>(output),
    };
    cudaMemcpy(
        device_header, &header, sizeof(header), cudaMemcpyHostToDevice);
    cudaMemcpy(
        device_lane, &lane, sizeof(lane), cudaMemcpyHostToDevice);
    cudaMemcpy(
        device_values, values, sizeof(values), cudaMemcpyHostToDevice);
    const GroupedDynamicShape source_shape{
        0, 2, PTIR_EXTENT_TOKEN_COUNT, {}};
    const GroupedDynamicShape axis0_shape{
        0, 1, PTIR_EXTENT_TOKEN_COUNT, {}};
    const GroupedDynamicShape fixed_three{3, 1, 0xff, {}};
    k_grouped_gather_axis0<float, std::uint32_t><<<1, 32>>>(
        device_header, device_lane, device_values, 3, 0, 1, 2,
        source_shape, axis0_shape, fixed_three, fixed_three);
    float actual[3]{};
    cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost);
    expect(
        std::all_of(std::begin(actual), std::end(actual),
                    [](float value) { return value == 0.0f; }),
        "zero-length grouped Gather zero-fills without division");

    cudaMemcpy(output, nonzero, sizeof(nonzero), cudaMemcpyHostToDevice);
    k_grouped_direct_gather<std::uint32_t><<<1, 32>>>(
        device_header, device_lane, nullptr, nullptr,
        device_values, 3, 1, 2, source_shape, axis0_shape,
        fixed_three, fixed_three, 8, 1, false);
    cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost);
    expect(
        std::all_of(std::begin(actual), std::end(actual),
                    [](float value) { return value == 0.0f; }),
        "zero-length direct-BF16 Gather zero-fills without division");
    cudaFree(device_values);
    cudaFree(device_lane);
    cudaFree(device_header);
    cudaFree(output);
    cudaFree(indices);
    cudaFree(commit);
}

void ragged_reshape_put_case() {
    constexpr std::uint32_t lane_count = 2;
    plan::StagePlan plan;
    plan.stage = PTIR_STAGE_EPILOGUE;
    plan.signature = {'r', 'a', 'g', 'g', 'e', 'd', '-', 'r', 'e', 's', 'h',
                      'a', 'p', 'e'};
    plan.signature_hash = container::fnv1a64(
        plan.signature.data(), plan.signature.size());
    plan.channel_bindings = {0};
    container::COp iota;
    iota.tag = PTIR_OP_IOTA;
    iota.results = 1;
    container::COp reshape;
    reshape.tag = PTIR_OP_RESHAPE;
    reshape.args = {0};
    reshape.results = 1;
    container::COp put;
    put.tag = PTIR_OP_CHAN_PUT;
    put.chan = 0;
    put.args = {1};
    put.results = 0;
    plan.ops = {{iota, {0}}, {reshape, {1}}, {put, {2}}};
    plan::ValueType token_vector;
    token_vector.dtype = PTIR_DT_U32;
    token_vector.domain = 3;
    token_vector.dims.push_back({true, PTIR_EXTENT_TOKEN_COUNT});
    plan.value_types = {token_vector, token_vector};
    plan.singleton.kind = 0;
    plan.fused.kind = 1;

    DeviceChannelRegistry registry;
    std::vector<Trace> traces(lane_count);
    std::vector<std::unique_ptr<PtirInstance>> instances;
    std::vector<std::vector<DeviceHostChannelTicket>> tickets(lane_count);
    std::vector<GroupedLaneBinding> bindings;
    for (std::uint32_t lane_index = 0;
         lane_index < lane_count;
         ++lane_index) {
        Channel output;
        output.id = 0;
        output.type = {Shape::vec(4), DType::U32};
        output.capacity = 1;
        output.host_visible = true;
        output.host_reader = true;
        traces[lane_index].channels = {output};
        Stage stage;
        stage.kind = StageKind::Epilogue;
        stage.puts = {{0, 1}};
        traces[lane_index].stages = {stage};
        const std::uint64_t channel_id = 64500 + lane_index;
        PieChannelDesc descriptor{};
        descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
        descriptor.channel_id = channel_id;
        descriptor.shape = {
            output.type.shape.dims.data(),
            output.type.shape.dims.size(),
        };
        descriptor.dtype = static_cast<std::uint8_t>(DType::U32);
        descriptor.host_role = PIE_CHANNEL_HOST_ROLE_READER;
        descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
        descriptor.capacity = 1;
        descriptor.reader_wait_id = channel_id * 2 + 1;
        descriptor.writer_wait_id = channel_id * 2 + 2;
        PieChannelEndpointBinding endpoint{};
        std::string error;
        expect(
            registry.register_endpoint(descriptor, &endpoint, &error),
            "register ragged reshape output: " + error);
        instances.push_back(std::make_unique<PtirInstance>(
            traces[lane_index],
            &registry,
            std::vector<std::uint64_t>{channel_id},
            std::vector<ChannelValue>{},
            &error));
        expect(
            instances.back()->ok(),
            "bind ragged reshape instance: " + error);
        const auto slot = instances.back()->view().slot(0);
        tickets[lane_index] = {{
            .slot = slot,
            .flags = kTicketPublish,
            .expected_head = kNoChannelTicket,
            .expected_tail = 0,
            .words = registry.host_words(slot),
            .mirror = static_cast<const std::uint8_t*>(
                registry.host_mirror(slot)),
            .cells = static_cast<std::uint8_t*>(
                registry.cell_base(slot)),
            .cap1 = 2,
            .wire_bytes = 4 * sizeof(std::uint32_t),
            .native_bytes = 4 * sizeof(std::uint32_t),
        }};
        bindings.push_back({
            .instance = instances.back().get(),
            .plan = &plan,
            .tickets = &tickets[lane_index],
            .logits_row_count = 1,
            .token_count = lane_index == 0 ? 2u : 4u,
            .vocab = 1,
        });
    }
    std::string reason;
    expect(
        grouped_stage_supported(bindings, &reason),
        "ragged reshape-to-fixed-cell remains grouped: " + reason);
    auto launch = GroupedTier0Executor::run(bindings, nullptr);
    cudaDeviceSynchronize();
    if (launch.device_tickets != nullptr) cudaFree(launch.device_tickets);
    const std::uint32_t expected[lane_count][4] = {
        {0, 1, 0, 0},
        {0, 1, 2, 3},
    };
    for (std::uint32_t lane_index = 0;
         lane_index < lane_count;
         ++lane_index) {
        instances[lane_index]->view().sync_host_rings();
        std::uint32_t actual[4] = {UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                   UINT32_MAX};
        instances[lane_index]->view().read_committed(
            0, actual, sizeof(actual));
        expect(
            std::equal(
                std::begin(actual), std::end(actual),
                std::begin(expected[lane_index])),
            "ragged reshape materializes and zero-pads the full channel cell");
    }
    expect(
        launch.body_op_launches == 3,
        "fixed-cell reshape adds exactly one safe materialization launch");
}

void token_count_extent_case() {
    plan::StagePlan plan;
    plan.stage = PTIR_STAGE_EPILOGUE;
    plan.signature_hash = 0x5abf44a0c61e8001ULL;
    plan.signature.assign(
        {'t', 'o', 'k', 'e', 'n', '-', 'c', 'o', 'u', 'n', 't'});
    plan.channel_bindings = {0};
    container::COp iota;
    iota.tag = PTIR_OP_IOTA;
    iota.results = 1;
    container::COp sum;
    sum.tag = PTIR_OP_REDUCE_SUM;
    sum.args = {0};
    sum.results = 1;
    container::COp put;
    put.tag = PTIR_OP_CHAN_PUT;
    put.chan = 0;
    put.args = {1};
    put.results = 0;
    plan.ops = {{iota, {0}}, {sum, {1}}, {put, {2}}};
    plan::ValueType token_vector;
    token_vector.dtype = PTIR_DT_U32;
    token_vector.domain = 3;
    token_vector.dims.push_back({true, PTIR_EXTENT_TOKEN_COUNT});
    plan.value_types = {
        token_vector,
        plan_type(PTIR_DT_U32, {}, 0),
    };
    plan.singleton.kind = 0;
    plan.fused.kind = 1;

    Trace trace;
    Channel output;
    output.id = 0;
    output.type = {Shape::vec(1), DType::U32};
    output.capacity = 1;
    output.host_visible = true;
    output.host_reader = true;
    trace.channels = {output};
    Stage stage;
    stage.kind = StageKind::Epilogue;
    stage.puts = {{0, 1}};
    trace.stages = {stage};
    DeviceChannelRegistry registry;
    const std::uint64_t channel_id = 64000;
    PieChannelDesc descriptor{};
    descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
    descriptor.channel_id = channel_id;
    descriptor.shape = {
        output.type.shape.dims.data(),
        output.type.shape.dims.size(),
    };
    descriptor.dtype = static_cast<std::uint8_t>(DType::U32);
    descriptor.host_role = PIE_CHANNEL_HOST_ROLE_READER;
    descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
    descriptor.capacity = 1;
    descriptor.reader_wait_id = channel_id * 2 + 1;
    descriptor.writer_wait_id = channel_id * 2 + 2;
    PieChannelEndpointBinding endpoint{};
    std::string error;
    expect(
        registry.register_endpoint(descriptor, &endpoint, &error),
        "register token-count output: " + error);
    PtirInstance instance(
        trace,
        &registry,
        std::vector<std::uint64_t>{channel_id},
        std::vector<ChannelValue>{},
        &error);
    expect(instance.ok(), "bind token-count instance: " + error);
    const auto slot = instance.view().slot(0);
    std::vector<DeviceHostChannelTicket> tickets{{
        .slot = slot,
        .flags = kTicketPublish,
        .expected_head = kNoChannelTicket,
        .expected_tail = 0,
        .words = registry.host_words(slot),
        .mirror = static_cast<const std::uint8_t*>(
            registry.host_mirror(slot)),
        .cells = static_cast<std::uint8_t*>(registry.cell_base(slot)),
        .cap1 = 2,
        .wire_bytes = static_cast<std::uint32_t>(
            sizeof(std::uint32_t)),
        .native_bytes = static_cast<std::uint32_t>(
            sizeof(std::uint32_t)),
    }};
    GroupedLaneBinding binding{
        .instance = &instance,
        .plan = &plan,
        .tickets = &tickets,
        .logits_row_count = 1,
        .row_count = 2,
        .token_count = 4,
        .kv_len = 9,
        .page_count = 3,
        .query_len = 4,
        .key_len = 9,
        .vocab = 1,
    };
    expect(
        grouped_extent(PTIR_EXTENT_SAMPLED_ROWS, binding) == 1 &&
            grouped_extent(PTIR_EXTENT_ROW_COUNT, binding) == 2 &&
            grouped_extent(PTIR_EXTENT_TOKEN_COUNT, binding) == 4 &&
            grouped_extent(PTIR_EXTENT_KV_LEN, binding) == 9 &&
            grouped_extent(PTIR_EXTENT_PAGE_COUNT, binding) == 3 &&
            grouped_extent(PTIR_EXTENT_QUERY_LEN, binding) == 4 &&
            grouped_extent(PTIR_EXTENT_KEY_LEN, binding) == 9,
        "lane ABI keeps sampled/token/KV/page/query/key extents distinct");
    const auto static_candidate =
        plan_type(PTIR_DT_F32, {4, 8}, 2);
    const auto static_shape =
        grouped_dynamic_shape(static_candidate, {binding});
    expect(
        grouped_numel(static_candidate, binding) == 32 &&
            grouped_rows(static_candidate, binding) == 4 &&
            static_shape.max_numel == 32 &&
            static_shape.extent == 0xff,
        "explicit [B,V] RNG/broadcast/candidate shapes stay static");
    std::string reason;
    expect(
        grouped_stage_supported({binding}, &reason),
        "token-count extent is available independently of sampled rows: " +
            reason);
    auto shorter_extent = binding;
    shorter_extent.token_count = 2;
    const std::uint64_t short_channel_id = channel_id + 1;
    descriptor.channel_id = short_channel_id;
    descriptor.reader_wait_id = short_channel_id * 2 + 1;
    descriptor.writer_wait_id = short_channel_id * 2 + 2;
    PieChannelEndpointBinding short_endpoint{};
    expect(
        registry.register_endpoint(
            descriptor, &short_endpoint, &error),
        "register token-count=2 output: " + error);
    PtirInstance short_instance(
        trace,
        &registry,
        std::vector<std::uint64_t>{short_channel_id},
        std::vector<ChannelValue>{},
        &error);
    expect(short_instance.ok(), "bind token-count=2 instance: " + error);
    const auto short_slot = short_instance.view().slot(0);
    std::vector<DeviceHostChannelTicket> short_tickets{{
        .slot = short_slot,
        .flags = kTicketPublish,
        .expected_head = kNoChannelTicket,
        .expected_tail = 0,
        .words = registry.host_words(short_slot),
        .mirror = static_cast<const std::uint8_t*>(
            registry.host_mirror(short_slot)),
        .cells = static_cast<std::uint8_t*>(
            registry.cell_base(short_slot)),
        .cap1 = 2,
        .wire_bytes = static_cast<std::uint32_t>(
            sizeof(std::uint32_t)),
        .native_bytes = static_cast<std::uint32_t>(
            sizeof(std::uint32_t)),
    }};
    shorter_extent.instance = &short_instance;
    shorter_extent.tickets = &short_tickets;
    expect(
        grouped_stage_supported(
            {shorter_extent, binding}, &reason),
        "token_count {2,4} lanes remain one grouped signature: " + reason);
    const auto ragged_launch =
        GroupedTier0Executor::run({shorter_extent, binding}, nullptr);
    cudaDeviceSynchronize();
    if (ragged_launch.device_tickets != nullptr) {
        cudaFree(ragged_launch.device_tickets);
    }
    short_instance.view().sync_host_rings();
    instance.view().sync_host_rings();
    std::uint32_t short_actual = UINT32_MAX;
    std::uint32_t actual = UINT32_MAX;
    short_instance.view().read_committed(
        0, &short_actual, sizeof(short_actual));
    instance.view().read_committed(0, &actual, sizeof(actual));
    expect(
        short_actual == 1 && actual == 6 &&
            ragged_launch.body_op_launches == 3,
        "grouped {2,4} iota/reductions preserve extents and effects");
    plan::StagePlan symbolic_sink = plan;
    symbolic_sink.value_types[1] = token_vector;
    binding.plan = &symbolic_sink;
    expect(
        !grouped_stage_supported({binding}, &reason) &&
            reason.find("channel value size") != std::string::npos,
        "symbolic channel sink cannot overrun a fixed channel cell");
    binding.plan = &plan;
    binding.token_count = kUnavailableGroupedExtent;
    expect(
        !grouped_stage_supported({binding}, &reason) &&
            reason.find("unavailable") != std::string::npos,
        "unavailable symbolic extents reject grouping");
}

void hierarchical_reduction_case() {
    constexpr std::uint32_t lane_count = 2;
    constexpr std::uint32_t width = 4096;
    plan::StagePlan plan;
    plan.stage = PTIR_STAGE_EPILOGUE;
    plan.signature_hash = 0xf257a41978d21a0cULL;
    plan.signature.assign(
        {'h', 'i', 'e', 'r', 'a', 'r', 'c', 'h', 'i', 'c', 'a', 'l'});
    plan.channel_bindings = {0, 1, 2};
    container::COp input;
    input.tag = PTIR_OP_CHAN_TAKE;
    input.chan = 0;
    input.results = 1;
    container::COp sum;
    sum.tag = PTIR_OP_REDUCE_SUM;
    sum.args = {0};
    sum.results = 1;
    container::COp argmax;
    argmax.tag = PTIR_OP_REDUCE_ARGMAX;
    argmax.args = {0};
    argmax.results = 1;
    container::COp put;
    put.tag = PTIR_OP_CHAN_PUT;
    put.chan = 1;
    put.args = {1};
    put.results = 0;
    container::COp put_argmax;
    put_argmax.tag = PTIR_OP_CHAN_PUT;
    put_argmax.chan = 2;
    put_argmax.args = {2};
    put_argmax.results = 0;
    plan.ops = {
        {input, {0}},
        {sum, {1}},
        {argmax, {2}},
        {put, {3}},
        {put_argmax, {4}},
    };
    plan.value_types = {
        plan_type(PTIR_DT_F32, {1, width}, 0),
        plan_type(PTIR_DT_F32, {1}, 0),
        plan_type(PTIR_DT_U32, {1}, 0),
    };
    plan.singleton.kind = 0;
    plan.fused.kind = 1;

    DeviceChannelRegistry registry;
    std::vector<Trace> traces(lane_count);
    std::vector<std::unique_ptr<PtirInstance>> instances;
    std::vector<std::vector<PieChannelEndpointBinding>> endpoints(
        lane_count, std::vector<PieChannelEndpointBinding>(3));
    std::vector<std::vector<DeviceHostChannelTicket>> tickets(lane_count);
    std::vector<GroupedLaneBinding> bindings;
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        Trace& trace = traces[lane];
        Channel source;
        source.id = 0;
        source.type = {Shape::mat(1, width), DType::F32};
        source.capacity = 1;
        source.has_seed = true;
        Channel output;
        output.id = 1;
        output.type = {Shape::vec(1), DType::F32};
        output.capacity = 1;
        output.host_visible = true;
        output.host_reader = true;
        Channel token;
        token.id = 2;
        token.type = {Shape::vec(1), DType::U32};
        token.capacity = 1;
        token.host_visible = true;
        token.host_reader = true;
        trace.channels = {source, output, token};
        Stage stage;
        stage.kind = StageKind::Epilogue;
        stage.takes = {0};
        stage.puts = {{1, 1}, {2, 2}};
        trace.stages = {stage};
        const std::vector<std::uint64_t> ids = {
            65000 + lane * 2,
            65001 + lane * 2,
            66000 + lane,
        };
        for (std::uint32_t dense = 0; dense < 3; ++dense) {
            const Channel& channel = trace.channels[dense];
            PieChannelDesc descriptor{};
            descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
            descriptor.channel_id = ids[dense];
            descriptor.shape = {
                channel.type.shape.dims.data(),
                channel.type.shape.dims.size(),
            };
            descriptor.dtype = static_cast<std::uint8_t>(channel.type.dtype);
            descriptor.host_role = channel.host_reader
                ? PIE_CHANNEL_HOST_ROLE_READER
                : PIE_CHANNEL_HOST_ROLE_NONE;
            descriptor.seeded = channel.has_seed;
            descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
            descriptor.capacity = 1;
            descriptor.reader_wait_id = ids[dense] * 2 + 1;
            descriptor.writer_wait_id = ids[dense] * 2 + 2;
            std::string error;
            expect(
                registry.register_endpoint(
                    descriptor, &endpoints[lane][dense], &error),
                "register hierarchical channel: " + error);
        }
        std::vector<float> values(width, 1.0f);
        if (lane == 1) {
            for (std::uint32_t index = 1; index < width; index += 2) {
                values[index] = -1.0f;
            }
            values[3077] = 5.0f;
        }
        std::vector<ChannelValue> seeds{{
            ids[0],
            std::vector<std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(values.data()),
                reinterpret_cast<const std::uint8_t*>(
                    values.data() + values.size())),
        }};
        std::string error;
        auto instance = std::make_unique<PtirInstance>(
            trace, &registry, ids, seeds, &error);
        expect(instance->ok(), "bind hierarchical instance: " + error);
        const auto input_slot = instance->view().slot(0);
        const auto output_slot = instance->view().slot(1);
        const auto token_slot = instance->view().slot(2);
        tickets[lane] = {
            {
                .slot = input_slot,
                .flags = kTicketConsume | kTicketRequireInput,
                .expected_head = 0,
                .expected_tail = kNoChannelTicket,
                .words = registry.host_words(input_slot),
                .mirror = static_cast<const std::uint8_t*>(
                    registry.host_mirror(input_slot)),
                .cells = static_cast<std::uint8_t*>(
                    registry.cell_base(input_slot)),
                .cap1 = 2,
                .wire_bytes = static_cast<std::uint32_t>(
                    width * sizeof(float)),
                .native_bytes = static_cast<std::uint32_t>(
                    width * sizeof(float)),
            },
            {
                .slot = output_slot,
                .flags = kTicketPublish,
                .expected_head = kNoChannelTicket,
                .expected_tail = 0,
                .words = registry.host_words(output_slot),
                .mirror = static_cast<const std::uint8_t*>(
                    registry.host_mirror(output_slot)),
                .cells = static_cast<std::uint8_t*>(
                    registry.cell_base(output_slot)),
                .cap1 = 2,
                .wire_bytes = static_cast<std::uint32_t>(sizeof(float)),
                .native_bytes = static_cast<std::uint32_t>(sizeof(float)),
            },
            {
                .slot = token_slot,
                .flags = kTicketPublish,
                .expected_head = kNoChannelTicket,
                .expected_tail = 0,
                .words = registry.host_words(token_slot),
                .mirror = static_cast<const std::uint8_t*>(
                    registry.host_mirror(token_slot)),
                .cells = static_cast<std::uint8_t*>(
                    registry.cell_base(token_slot)),
                .cap1 = 2,
                .wire_bytes = static_cast<std::uint32_t>(
                    sizeof(std::uint32_t)),
                .native_bytes = static_cast<std::uint32_t>(
                    sizeof(std::uint32_t)),
            },
        };
        instances.push_back(std::move(instance));
        bindings.push_back({
            .instance = instances.back().get(),
            .plan = &plan,
            .tickets = &tickets[lane],
            .logits_row_count = 1,
            .vocab = width,
            .program_index = lane,
        });
    }
    auto launch = GroupedTier0Executor::run(bindings, nullptr);
    cudaDeviceSynchronize();
    if (launch.device_tickets != nullptr) cudaFree(launch.device_tickets);
    const float expected[lane_count] = {
        static_cast<float>(width),
        6.0f,
    };
    const std::uint32_t expected_argmax[lane_count] = {0, 3077};
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        instances[lane]->view().sync_host_rings();
        float actual = 0.0f;
        instances[lane]->view().read_committed(
            1, &actual, sizeof(actual));
        std::uint32_t actual_argmax = UINT32_MAX;
        instances[lane]->view().read_committed(
            2, &actual_argmax, sizeof(actual_argmax));
        expect(
            std::memcmp(&actual, &expected[lane], sizeof(float)) == 0,
            "hierarchical canonical reduction");
        expect(
            actual_argmax == expected_argmax[lane],
            "hierarchical NaN-safe argmax");
    }
}

void exclusion_cases() {
    constexpr std::uint32_t vocab = 32;
    Fixture fixture;
    const std::vector<std::uint8_t> mask(vocab, 1);
    const std::size_t instance = fixture.add_instance(vocab, 0, mask);
    auto tickets = fixture.tickets(instance, 0);
    auto& view = fixture.instances[instance]->view();
    detail::PortCellCache cached_cells;
    auto& cached = cached_cells[view.slot(0)];
    cached.bytes.resize(view.cell_bytes(0));
    std::vector<std::uint8_t> cached_output;
    std::string cached_error;
    expect(
        !detail::read_port_cell(
            view,
            0,
            cached_output,
            &cached_error,
            nullptr,
            &cached_cells) &&
            cached_error.find("not ready") != std::string::npos,
        "descriptor cache preserves empty-channel readiness");
    cached.ready = 1;
    expect(
        detail::read_port_cell(
            view,
            0,
            cached_output,
            &cached_error,
            nullptr,
            &cached_cells),
        "descriptor cache serves bytes only after readiness");
    std::vector<GroupedLaneBinding> aliased(2);
    for (GroupedLaneBinding& lane : aliased) {
        lane.instance = fixture.instances[instance].get();
        lane.plan = &fixture.plans[instance];
        lane.tickets = &tickets;
        lane.logits_row_count = 1;
        lane.vocab = vocab;
    }
    std::string reason;
    expect(
        !grouped_stage_supported(aliased, &reason) &&
            reason.find("shared") != std::string::npos,
        "shared slot alias is excluded");
    expect(
        !grouped_dispatch_supported({aliased.front()}, true, &reason) &&
            reason.find("recurrent") != std::string::npos,
        "RS lane is excluded");
    plan::StagePlan unsupported = fixture.plans[instance];
    unsupported.ops[3].op.tag = PTIR_OP_KERNEL_CALL;
    GroupedLaneBinding unsupported_lane = aliased.front();
    unsupported_lane.plan = &unsupported;
    expect(
        !grouped_stage_supported({unsupported_lane}, &reason),
        "second-party kernel remains an explicit model/library boundary");
    plan::StagePlan mtp_logits = fixture.plans[instance];
    mtp_logits.ops[0].op.intr = PTIR_INTR_MTP_LOGITS;
    mtp_logits.value_types[0] =
        plan_type(PTIR_DT_F32, {3, vocab}, 2);
    GroupedLaneBinding mtp_lane = aliased.front();
    mtp_lane.plan = &mtp_logits;
    const std::vector<std::uint64_t> mtp_rows(3, 1);
    mtp_lane.mtp_logits_bf16_rows = &mtp_rows;
    expect(
        grouped_stage_supported({mtp_lane}, &reason),
        "static-K MtpLogits accepts a dedicated grouped draft-row table: " +
            reason);
    plan::StagePlan mtp_drafts = mtp_logits;
    mtp_drafts.ops[0].op.intr = PTIR_INTR_MTP_DRAFTS;
    mtp_lane.plan = &mtp_drafts;
    mtp_lane.mtp_logits_bf16_rows = nullptr;
    expect(
        !grouped_stage_supported({mtp_lane}, &reason) &&
            reason.find("intrinsic") != std::string::npos,
        "MtpDrafts remains unsupported without dedicated storage");
    plan::StagePlan collision = fixture.plans[instance];
    collision.signature.push_back('!');
    GroupedLaneBinding collision_lane = aliased.front();
    collision_lane.plan = &collision;
    expect(
        !grouped_stage_supported(
            {aliased.front(), collision_lane}, &reason) &&
            reason.find("plan") != std::string::npos,
        "equal signature hashes with different canonical bytes do not group");
}

void aggregate_mtp_group_case() {
    constexpr std::uint32_t lane_count = 2;
    constexpr std::uint32_t drafts = 20;
    constexpr std::uint32_t vocab = 8;
    plan::StagePlan plan;
    plan.stage = PTIR_STAGE_EPILOGUE;
    plan.signature = {'m', 't', 'p', '-', '4', '0'};
    plan.signature_hash = container::fnv1a64(
        plan.signature.data(), plan.signature.size());
    plan.channel_bindings = {0};
    container::COp mtp;
    mtp.tag = PTIR_OP_INTRINSIC_VAL;
    mtp.intr = PTIR_INTR_MTP_LOGITS;
    container::COp argmax;
    argmax.tag = PTIR_OP_REDUCE_ARGMAX;
    argmax.args = {0};
    container::COp sum;
    sum.tag = PTIR_OP_REDUCE_SUM;
    sum.args = {1};
    container::COp put;
    put.tag = PTIR_OP_CHAN_PUT;
    put.chan = 0;
    put.args = {2};
    put.results = 0;
    plan.ops = {{mtp, {0}}, {argmax, {1}}, {sum, {2}}, {put, {3}}};
    plan.value_types = {
        plan_type(PTIR_DT_F32, {drafts, vocab}, 2),
        plan_type(PTIR_DT_U32, {drafts}, 1),
        plan_type(PTIR_DT_U32, {}, 0),
    };
    plan.singleton.kind = 0;
    plan.fused.kind = 1;

    std::vector<std::uint16_t> logits(
        lane_count * drafts * vocab, bf16_bits(-10.0f));
    std::uint32_t expected[lane_count]{};
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        for (std::uint32_t row = 0; row < drafts; ++row) {
            const std::uint32_t token = (lane + row) % vocab;
            logits[
                (static_cast<std::size_t>(lane) * drafts + row) *
                    vocab +
                token] = bf16_bits(10.0f);
            expected[lane] += token;
        }
    }
    std::uint16_t* device_logits = nullptr;
    cudaMalloc(&device_logits, logits.size() * sizeof(std::uint16_t));
    cudaMemcpy(
        device_logits, logits.data(),
        logits.size() * sizeof(std::uint16_t),
        cudaMemcpyHostToDevice);

    DeviceChannelRegistry registry;
    std::vector<Trace> traces(lane_count);
    std::vector<std::unique_ptr<PtirInstance>> instances;
    std::vector<std::vector<DeviceHostChannelTicket>> tickets(lane_count);
    std::vector<std::vector<std::uint64_t>> mtp_rows(lane_count);
    std::vector<std::vector<std::uint64_t>> ordinary_rows(lane_count);
    std::vector<GroupedLaneBinding> bindings;
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        Channel output;
        output.id = 0;
        output.type = {Shape::vec(1), DType::U32};
        output.capacity = 1;
        output.host_visible = true;
        output.host_reader = true;
        traces[lane].channels = {output};
        Stage trace_stage;
        trace_stage.kind = StageKind::Epilogue;
        trace_stage.puts = {{0, 2}};
        traces[lane].stages = {trace_stage};
        const std::uint64_t channel_id = 68000 + lane;
        PieChannelDesc descriptor{};
        descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
        descriptor.channel_id = channel_id;
        descriptor.shape = {
            output.type.shape.dims.data(),
            output.type.shape.dims.size(),
        };
        descriptor.dtype = static_cast<std::uint8_t>(DType::U32);
        descriptor.host_role = PIE_CHANNEL_HOST_ROLE_READER;
        descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
        descriptor.capacity = 1;
        descriptor.reader_wait_id = channel_id * 2 + 1;
        descriptor.writer_wait_id = channel_id * 2 + 2;
        PieChannelEndpointBinding endpoint{};
        std::string error;
        expect(
            registry.register_endpoint(descriptor, &endpoint, &error),
            "register aggregate MTP output: " + error);
        instances.push_back(std::make_unique<PtirInstance>(
            traces[lane], &registry,
            std::vector<std::uint64_t>{channel_id},
            std::vector<ChannelValue>{}, &error));
        const auto slot = instances.back()->view().slot(0);
        tickets[lane] = {{
            .slot = slot,
            .flags = kTicketPublish,
            .expected_head = kNoChannelTicket,
            .expected_tail = 0,
            .words = registry.host_words(slot),
            .mirror = static_cast<const std::uint8_t*>(
                registry.host_mirror(slot)),
            .cells = static_cast<std::uint8_t*>(
                registry.cell_base(slot)),
            .cap1 = 2,
            .wire_bytes = sizeof(std::uint32_t),
            .native_bytes = sizeof(std::uint32_t),
        }};
        ordinary_rows[lane] = {
            reinterpret_cast<std::uint64_t>(
                device_logits +
                static_cast<std::size_t>(lane) * drafts * vocab),
        };
        for (std::uint32_t row = 0; row < drafts; ++row) {
            mtp_rows[lane].push_back(reinterpret_cast<std::uint64_t>(
                device_logits +
                (static_cast<std::size_t>(lane) * drafts + row) *
                    vocab));
        }
        bindings.push_back({
            .instance = instances.back().get(),
            .plan = &plan,
            .tickets = &tickets[lane],
            .logits_bf16_rows = &ordinary_rows[lane],
            .mtp_logits_bf16_rows = &mtp_rows[lane],
            .logits_row_count = 1,
            .row_count = drafts,
            .token_count = drafts,
            .vocab = vocab,
            .logits_stride = vocab,
        });
    }
    std::string reason;
    expect(
        grouped_stage_supported(bindings, &reason),
        "40 aggregate MTP rows remain one group: " + reason);
    auto launch = GroupedTier0Executor::run(bindings, nullptr);
    cudaDeviceSynchronize();
    if (launch.device_tickets != nullptr) cudaFree(launch.device_tickets);
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        instances[lane]->view().sync_host_rings();
        std::uint32_t actual = UINT32_MAX;
        instances[lane]->view().read_committed(0, &actual, sizeof(actual));
        expect(
            actual == expected[lane],
            "aggregate MTP attribution lane " + std::to_string(lane));
    }
    expect(
        launch.body_op_launches == 3,
        "aggregate MTP launch count scales by signature, not program");
    cudaFree(device_logits);
}

void cross_program_signature() {
    constexpr std::uint32_t vocab = 32;
    Fixture fixture;
    const std::vector<std::uint8_t> mask(vocab, 1);
    const std::size_t first = fixture.add_instance(vocab, 0, mask);
    const std::size_t second = fixture.add_instance(vocab, 1, mask);
    auto first_tickets = fixture.tickets(first, 0);
    auto second_tickets = fixture.tickets(second, 1);
    const auto logits = test_logits(2, vocab);
    float* device = device_logits(logits, vocab);
    std::vector<GroupedLaneBinding> lanes{
        {
            .instance = fixture.instances[first].get(),
            .plan = &fixture.plans[first],
            .tickets = &first_tickets,
            .logits_base = device,
            .logits_row_offset = 0,
            .logits_row_count = 1,
            .vocab = vocab,
            .program_index = 0,
        },
        {
            .instance = fixture.instances[second].get(),
            .plan = &fixture.plans[second],
            .tickets = &second_tickets,
            .logits_base = device,
            .logits_row_offset = 1,
            .logits_row_count = 1,
            .vocab = vocab,
            .program_index = 1,
        },
    };
    fixture.traces[first]->channels[0].extern_dir =
        PTIR_EXTERN_IMPORT;
    fixture.traces[second]->channels[0].extern_dir =
        PTIR_EXTERN_IMPORT;
    expect(
        fixture.plans[first].signature == fixture.plans[second].signature &&
            fixture.plans[first].channel_bindings !=
                fixture.plans[second].channel_bindings,
        "different program bindings share canonical signature");
    std::string reason;
    expect(
        grouped_stage_supported(lanes, &reason),
        "distinct extern bindings retain grouped fallback: " + reason);
    GroupedLaunchResult launch = GroupedTier0Executor::run(lanes, nullptr);
    cudaDeviceSynchronize();
    if (launch.device_tickets != nullptr) cudaFree(launch.device_tickets);
    expect(fixture.output(first, 0) == 3, "cross-program lane 0");
    expect(fixture.output(second, 1) == 10, "cross-program lane 1");
    cudaFree(device);
}

#if 0  // The Dispatch integration variant lives in ptir_grouped_dispatch_test.cpp.
void dispatch_grouping(
    const std::string& golden_directory,
    bool partial,
    bool recurrent_state) {
    constexpr std::uint32_t lane_count = 4;
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
    expect(
        !container_bytes.empty() && !sidecar_bytes.empty(),
        "load section3 grouped fixture");

    container::Container container;
    container::DecodeError decode_error;
    expect(
        container::decode(
            container_bytes.data(), container_bytes.size(),
            container, &decode_error),
        "decode section3 fixture: " + decode_error.detail);
    expect(container.channels.size() == 5, "section3 channel layout");

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
        "dispatch register grouped program: " + error);

    std::vector<std::uint64_t> hashes(lane_count, hash);
    std::vector<std::uint64_t> instances(lane_count);
    std::vector<PieTerminalCell> terminals(lane_count);
    std::vector<PieTerminalCell*> terminal_ptrs(lane_count);
    std::vector<std::uint64_t> expected_heads;
    std::vector<std::uint64_t> expected_tails;
    std::vector<std::uint32_t> ticket_indptr{0};
    std::vector<std::uint32_t> sampling_indptr{0};
    std::vector<std::vector<PieChannelEndpointBinding>> endpoints(lane_count);
    std::vector<std::vector<std::uint64_t>> channel_ids(lane_count);
    std::vector<std::uint32_t> expected_tokens(lane_count);

    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        endpoints[lane].resize(container.channels.size());
        channel_ids[lane].resize(container.channels.size());
        for (std::size_t dense = 0; dense < container.channels.size(); ++dense) {
            const container::CChannel& source = container.channels[dense];
            channel_ids[lane][dense] =
                10000 + static_cast<std::uint64_t>(lane) * 100 + dense;
            PieChannelDesc descriptor{};
            descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
            descriptor.channel_id = channel_ids[lane][dense];
            descriptor.shape = {
                source.shape.dims,
                source.shape.rank,
            };
            descriptor.dtype = source.dtype;
            descriptor.host_role = source.host_role;
            descriptor.seeded = source.seeded;
            descriptor.extern_dir = PIE_CHANNEL_EXTERN_NONE;
            descriptor.capacity = source.capacity;
            descriptor.reader_wait_id = descriptor.channel_id * 2 + 1;
            descriptor.writer_wait_id = descriptor.channel_id * 2 + 2;
            expect(
                dispatch.register_channel(
                    descriptor, &endpoints[lane][dense], &error) ==
                    PIE_STATUS_OK,
                "dispatch register grouped channel: " + error);
        }
        const std::int32_t token_seed = 1;
        const std::uint32_t length_seed = 1;
        const std::uint32_t rng_seed[2] = {1234 + lane, 0};
        const PieChannelValueDesc seeds[] = {
            {
                channel_ids[lane][0],
                {
                    reinterpret_cast<const std::uint8_t*>(&token_seed),
                    sizeof(token_seed),
                },
            },
            {
                channel_ids[lane][3],
                {
                    reinterpret_cast<const std::uint8_t*>(&length_seed),
                    sizeof(length_seed),
                },
            },
            {
                channel_ids[lane][4],
                {
                    reinterpret_cast<const std::uint8_t*>(rng_seed),
                    sizeof(rng_seed),
                },
            },
        };
        PieInstanceBinding binding{};
        instances[lane] = 500 + lane;
        expect(
            dispatch.bind_instance(
                instances[lane], hash, 900 + lane, channel_ids[lane],
                std::vector<PieChannelValueDesc>(
                    std::begin(seeds), std::end(seeds)),
                &binding, &error) == PIE_STATUS_OK,
            "dispatch bind grouped instance: " + error);

        if (!partial || lane != 1) {
            PieChannelEndpointBinding& mask = endpoints[lane][2];
            auto* words =
                reinterpret_cast<std::uint64_t*>(mask.word_base);
            std::memset(
                reinterpret_cast<void*>(mask.mirror_base),
                0xff, mask.cell_bytes);
            std::atomic_ref<std::uint64_t>(
                words[mask.tail_word_index])
                .store(1, std::memory_order_release);
        }

        terminal_ptrs[lane] = &terminals[lane];
        expected_heads.insert(
            expected_heads.end(),
            {
                0,
                no_ticket,
                0,
                0,
                0,
            });
        expected_tails.insert(
            expected_tails.end(),
            {
                1,
                0,
                no_ticket,
                1,
                1,
            });
        ticket_indptr.push_back(
            static_cast<std::uint32_t>(expected_heads.size()));
        sampling_indptr.push_back(lane + 1);
        expected_tokens[lane] = (lane * 5 + 3) % vocab;
    }

    std::vector<float> logits(
        static_cast<std::size_t>(lane_count) * vocab, -6.0f);
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        logits[static_cast<std::size_t>(lane) * vocab +
               expected_tokens[lane]] = 20.0f;
    }
    float* device_logits = nullptr;
    cudaMalloc(&device_logits, logits.size() * sizeof(float));
    cudaMemcpy(
        device_logits, logits.data(), logits.size() * sizeof(float),
        cudaMemcpyHostToDevice);

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
    const std::uint32_t rs_slot = 0;
    if (recurrent_state) {
        view.rs_slot_ids = pie_native::slice_from_u32(&rs_slot, 1);
    }
    expect(
        dispatch.validate_launch(view, &error) == PIE_STATUS_OK,
        "validate grouped dispatch: " + error);
    setenv("PIE_CUDA_DISABLE_PTIR_GENERATED", "1", 1);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    expect(
        dispatch.run(
            view, device_logits, vocab, stream, nullptr, PieCompletion{}),
        "dispatch grouped run");
    cudaDeviceSynchronize();
    unsetenv("PIE_CUDA_DISABLE_PTIR_GENERATED");

    const DispatchStats stats = dispatch.stats();
    if (recurrent_state) {
        expect(
            stats.grouped_lanes == 0 && stats.rs_exclusions != 0,
            "Dispatch excludes RS lanes");
    } else {
        expect(
            stats.grouped_lanes == lane_count &&
                stats.grouped_tier0_groups == 1,
            "Dispatch grouped four compatible programs");
    }
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        const std::uint32_t outcome =
            std::atomic_ref<std::uint32_t>(terminals[lane].outcome)
                .load(std::memory_order_acquire);
        if (partial && lane == 1) {
            expect(
                outcome == PIE_TERMINAL_OUTCOME_RETRY,
                "Dispatch attributes partial retry");
        } else {
            expect(
                outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
                "Dispatch attributes successful grouped lane");
        }
    }
    cudaStreamDestroy(stream);
    cudaFree(device_logits);
}
#endif

}  // namespace

int main(int argc, char** argv) {
    cuInit(0);
    cudaFree(nullptr);
    parity_cases();
    fallback_graph_case();
    graph_cache_memory_bound_case();
    graph_cache_lock_order_case();
    partial_readiness();
    integer_unary_cases();
    integer_binary_cases();
    grouped_integer_reduction_case();
    grouped_integer_argmax_case();
    cast_adversarial_case();
    topk_adversarial_case();
    scalable_nucleus_production_case();
    scalable_nucleus_batch_measurements();
    nucleus_region_input_source_case();
    ragged_layout_kernel_case();
    register_rule_case();
    structured_mask_grouped_case();
    ragged_rows_case();
    fp32_dynamic_root_stride_case();
    zero_length_gather_case();
    ragged_reshape_put_case();
    token_count_extent_case();
    hierarchical_reduction_case();
    exclusion_cases();
    aggregate_mtp_group_case();
    cross_program_signature();
    static_cast<void>(argc);
    static_cast<void>(argv);
    std::printf("PTIR grouped vertical slice: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
