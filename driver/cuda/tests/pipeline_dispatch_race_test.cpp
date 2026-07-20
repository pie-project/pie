// PTIR dispatch concurrency gate (direct_ffi_new_plan.md §14 gate 9).
//
// One scheduler thread (this thread) drives same-instance RUN-AHEAD fires —
// fire N+1 is prepared and launched while fire N's completion callback may
// still be executing on the CUDA host-func thread — interleaved with
// register_channel growth (registry vector reallocation) and bind/close of an
// unrelated instance. The completion callback's §7 contract is that it only
// dereferences pointers precomputed at enqueue time (pinned word blocks,
// terminal cells, the commit-flag snapshot) plus the notify table, so none of
// this scheduler-thread churn may race it.
//
// Built normally this is a crash/UAF smoke; built with -fsanitize=thread it
// is the TSAN gate refereeing the callback-thread vs scheduler-thread
// interleaving. Needs a GPU. Run from driver/cuda/tests (golden fixtures).

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "pie_native/ptir/container.hpp"
#include "pipeline/dispatch.hpp"

using pie_cuda_driver::pipeline::Dispatch;

namespace {

const char* g_stage = "startup";

bool expect(bool cond, const char* msg) {
    if (!cond) std::fprintf(stderr, "FAIL [%s]: %s\n", g_stage, msg);
    return cond;
}

std::string trim(const std::string& s) {
    const std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const std::size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<std::uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        out.push_back(static_cast<std::uint8_t>(
            std::stoul(hex.substr(i, 2), nullptr, 16)));
    }
    return out;
}

std::string golden_field(const std::string& path, const std::string& key) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        if (trim(line.substr(0, colon)) != key) continue;
        return trim(line.substr(colon + 1));
    }
    return "";
}

std::atomic<std::uint64_t> g_notifies{0};

extern "C" void count_notify(void*, std::uint64_t, std::uint64_t) {
    g_notifies.fetch_add(1, std::memory_order_relaxed);
}

// Register the golden trace's channels under fresh global ids starting at
// `first_id`; fills `endpoints` and returns the ids.
std::vector<std::uint64_t> register_channels(
    Dispatch& dispatch,
    const pie_native::ptir::container::Container& container,
    std::uint64_t first_id,
    std::vector<PieChannelEndpointBinding>& endpoints,
    std::string& err) {
    std::vector<std::uint64_t> ids(container.channels.size());
    endpoints.resize(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ids[i] = first_id + i;
        const auto& source = container.channels[i];
        PieChannelDesc desc{};
        desc.abi_version = PIE_DRIVER_ABI_VERSION;
        desc.channel_id = ids[i];
        desc.shape = {.ptr = source.shape.dims, .len = source.shape.rank};
        desc.dtype = source.dtype;
        desc.host_role = source.host_role;
        desc.seeded = source.seeded;
        desc.extern_dir = PIE_CHANNEL_EXTERN_NONE;
        desc.capacity = source.capacity;
        desc.reader_wait_id = 2 * ids[i] + 1;
        desc.writer_wait_id = 2 * ids[i] + 2;
        if (dispatch.register_channel(desc, &endpoints[i], &err) !=
            PIE_STATUS_OK) {
            return {};
        }
    }
    return ids;
}

// Simulated host take: release-advance the reader endpoint's head word so a
// capacity-1 output channel never blocks the run-ahead burst. The mirror
// value itself is not read — this stress cares about the wake/word protocol.
void advance_reader_head(const PieChannelEndpointBinding& endpoint) {
    auto* words = reinterpret_cast<std::uint64_t*>(endpoint.word_base);
    std::atomic_ref<std::uint64_t> head(words[endpoint.head_word_index]);
    head.store(head.load(std::memory_order_acquire) + 1,
               std::memory_order_release);
}

// Direct host put (ABI v2): wire bytes into the writer ring cell, then the
// release-published tail word — exactly what the runtime's put does.
void ring_put(const PieChannelEndpointBinding& endpoint, const void* wire,
              std::size_t bytes) {
    auto* words = reinterpret_cast<std::uint64_t*>(endpoint.word_base);
    std::atomic_ref<std::uint64_t> tail(words[endpoint.tail_word_index]);
    const std::uint64_t sequence = tail.load(std::memory_order_acquire);
    const std::uint64_t cap1 = static_cast<std::uint64_t>(endpoint.capacity) + 1;
    auto* cell = reinterpret_cast<std::uint8_t*>(endpoint.mirror_base) +
        (sequence % cap1) * endpoint.cell_bytes;
    std::memcpy(cell, wire, bytes);
    tail.store(sequence + 1, std::memory_order_release);
}

}  // namespace

int main() {
    const std::string golden = "../tests/golden-ptir/greedy_argmax.txt";
    const auto bytes = hex_to_bytes(golden_field(golden, "container"));
    const auto sidecar = hex_to_bytes(golden_field(golden, "sidecar"));
    const std::uint64_t hash =
        std::stoull(golden_field(golden, "hash"), nullptr, 16);
    if (!expect(!bytes.empty() && !sidecar.empty() && hash != 0,
                "load golden PTIR")) return 1;

    setenv("PIE_CUDA_FORCE_RETRY_ONCE", "1", 1);
    Dispatch dispatch;
    std::string err;
    g_stage = "register_program";
    if (!expect(dispatch.register_program(
                    hash,
                    pie_native::ByteSlice{bytes.data(), bytes.size()},
                    pie_native::ByteSlice{sidecar.data(), sidecar.size()},
                    &err) == PIE_STATUS_OK,
                err.c_str())) return 1;

    g_stage = "decode";
    pie_native::ptir::container::Container container;
    pie_native::ptir::container::DecodeError decode_error;
    if (!expect(pie_native::ptir::container::decode(
                    bytes.data(), bytes.size(), container, &decode_error),
                decode_error.detail.c_str())) return 1;

    g_stage = "register_channels";
    std::vector<PieChannelEndpointBinding> endpoints;
    const std::vector<std::uint64_t> channel_ids =
        register_channels(dispatch, container, 1000, endpoints, err);
    if (!expect(!channel_ids.empty(), err.c_str())) return 1;
    for (std::size_t i = 0; i < container.channels.size(); ++i) {
        std::fprintf(stderr, "chan %zu: host_role=%u seeded=%u capacity=%u\n",
                     i, container.channels[i].host_role,
                     container.channels[i].seeded,
                     container.channels[i].capacity);
    }

    const std::uint8_t seed_bytes[4] = {1, 0, 0, 0};
    const PieChannelValueDesc seed{
        .channel_id = channel_ids[0],
        .bytes = {.ptr = seed_bytes, .len = sizeof(seed_bytes)},
    };
    g_stage = "bind_instance";
    PieInstanceBinding binding{};
    if (!expect(dispatch.bind_instance(
                    /*instance_id=*/77, hash, PIE_GEOMETRY_CLASS_HOST,
                    /*pacing_wait_id=*/1234,
                    channel_ids, {seed}, &binding, &err) == PIE_STATUS_OK,
                err.c_str())) return 1;

    PieRuntimeCallbacks runtime{};
    runtime.abi_version = PIE_DRIVER_ABI_VERSION;
    runtime.ctx = nullptr;
    runtime.notify = count_notify;

    constexpr std::uint32_t kVocab = 8;
    float* d_logits = nullptr;
    cudaMalloc(&d_logits, kVocab * sizeof(float));
    const float logits[kVocab] = {0, 1, 9, 2, 0, 0, 0, 3};
    cudaMemcpy(d_logits, logits, sizeof(logits), cudaMemcpyHostToDevice);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // Terminal cells must stay stable until their callback retires them —
    // preallocate every fire's cell up front (no reallocation).
    constexpr int kIterations = 96;
    constexpr int kFiresPerIteration = 2;  // same-instance run-ahead pair
    static PieTerminalCell terminals[kIterations * kFiresPerIteration]{};
    const std::uint64_t instance_ids[1] = {binding.instance_id};
    const std::uint32_t sampling_indptr[2] = {0, 1};
    const std::uint32_t channel_ticket_indptr[2] = {0, 2};
    constexpr std::uint64_t kNoTicket = std::numeric_limits<std::uint64_t>::max();

    // Forced benign non-commit: terminal RETRY, no ring actuals move. The same
    // logical payload then commits normally.
    {
        PieTerminalCell retry_terminal{};
        PieTerminalCell* retry_cells[1] = {&retry_terminal};
        pie_native::LaunchView retry_view{};
        retry_view.terminal_cells = pie_native::slice_from(
            const_cast<PieTerminalCell* const*>(retry_cells), 1);
        retry_view.sampling_indptr =
            pie_native::slice_from_u32(sampling_indptr, 2);
        retry_view.ptir_program_hashes =
            pie_native::slice_from_u64(&hash, 1);
        retry_view.ptir_program_instances =
            pie_native::slice_from_u64(instance_ids, 1);
        const std::uint64_t retry_heads[2] = {0, kNoTicket};
        const std::uint64_t retry_tails[2] = {kNoTicket, 0};
        retry_view.channel_expected_head =
            pie_native::slice_from_u64(retry_heads, 2);
        retry_view.channel_expected_tail =
            pie_native::slice_from_u64(retry_tails, 2);
        retry_view.channel_ticket_indptr =
            pie_native::slice_from_u32(channel_ticket_indptr, 2);
        auto* reader_words =
            reinterpret_cast<std::uint64_t*>(endpoints[1].word_base);
        const std::uint64_t tail_before =
            std::atomic_ref<std::uint64_t>(
                reader_words[endpoints[1].tail_word_index])
                .load(std::memory_order_acquire);

        if (!expect(dispatch.run(
                        retry_view, d_logits, kVocab, stream, &runtime,
                        PieCompletion{.wait_id = 399,
                                      .target_epoch = 1,
                                      .terminal_cell = nullptr}),
                    "forced retry run accepted")) {
            return 1;
        }
        cudaDeviceSynchronize();
        if (!expect(
                retry_terminal.outcome == PIE_TERMINAL_OUTCOME_RETRY,
                "forced non-commit publishes RETRY")) {
            return 1;
        }
        if (!expect(
                std::atomic_ref<std::uint64_t>(
                    reader_words[endpoints[1].tail_word_index])
                        .load(std::memory_order_acquire) == tail_before,
                "RETRY publishes no reader tail")) {
            return 1;
        }
        unsetenv("PIE_CUDA_FORCE_RETRY_ONCE");

        PieTerminalCell committed_terminal{};
        PieTerminalCell* committed_cells[1] = {&committed_terminal};
        retry_view.terminal_cells = pie_native::slice_from(
            const_cast<PieTerminalCell* const*>(committed_cells), 1);
        if (!expect(dispatch.run(
                        retry_view, d_logits, kVocab, stream, &runtime,
                        PieCompletion{.wait_id = 400,
                                      .target_epoch = 1,
                                      .terminal_cell = nullptr}),
                    "retry successor run accepted")) {
            return 1;
        }
        cudaDeviceSynchronize();
        if (!expect(
                committed_terminal.outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
                "same payload commits after RETRY")) {
            return 1;
        }
        advance_reader_head(endpoints[1]);
        const std::uint32_t next_token[1] = {1};
        ring_put(endpoints[0], next_token, sizeof(next_token));
    }

    std::uint64_t churn_channel_id = 50000;
    std::uint64_t churn_instance_id = 200;

    // Ballast: fill the registry to just under its current slot capacity
    // (observed through stats, so this never silently decays if the
    // initial capacity moves) so the live churn inside the loop crosses it
    // while fires are in flight, forcing grow() to reallocate the shared
    // head/tail/full arrays under load (RV-27). These stay registered
    // until after the final drain.
    g_stage = "ballast_register";
    const std::uint64_t initial_slot_capacity =
        dispatch.stats().channel_slot_capacity;
    if (!expect(initial_slot_capacity > 0, "registry reports capacity")) {
        return 1;
    }
    std::vector<std::uint64_t> live_churn_ids;
    while (live_churn_ids.size() + channel_ids.size() + 4 <
           initial_slot_capacity) {
        std::vector<PieChannelEndpointBinding> ballast_endpoints;
        const auto ballast_ids = register_channels(
            dispatch, container, churn_channel_id, ballast_endpoints, err);
        if (!expect(!ballast_ids.empty(), err.c_str())) return 1;
        churn_channel_id += ballast_ids.size();
        live_churn_ids.insert(
            live_churn_ids.end(), ballast_ids.begin(), ballast_ids.end());
    }

    std::uint64_t next_sequence = 1;
    int fired = 0;
    for (int i = 0; i < kIterations; ++i) {
        // Two back-to-back fires: fire N+1's prep (pull, snapshot copy,
        // schedule_host_publish) overlaps fire N's callback.
        for (int burst = 0; burst < kFiresPerIteration; ++burst) {
            const std::uint64_t fire_sequence =
                next_sequence + static_cast<std::uint64_t>(burst);
            PieTerminalCell* cell = &terminals[fired++];
            PieTerminalCell* cells[1] = {cell};
            pie_native::LaunchView view{};
            view.terminal_cells = pie_native::slice_from(
                const_cast<PieTerminalCell* const*>(cells), 1);
            view.sampling_indptr =
                pie_native::slice_from_u32(sampling_indptr, 2);
            view.ptir_program_hashes = pie_native::slice_from_u64(&hash, 1);
            view.ptir_program_instances =
                pie_native::slice_from_u64(instance_ids, 1);
            const std::uint64_t ticket_heads[2] = {fire_sequence, kNoTicket};
            const std::uint64_t ticket_tails[2] = {kNoTicket, fire_sequence};
            view.channel_expected_head =
                pie_native::slice_from_u64(ticket_heads, 2);
            view.channel_expected_tail =
                pie_native::slice_from_u64(ticket_tails, 2);
            view.channel_ticket_indptr =
                pie_native::slice_from_u32(channel_ticket_indptr, 2);
            g_stage = "validate_launch";
            const int validate_rc = dispatch.validate_launch(view, &err);
            if (!expect(validate_rc == PIE_STATUS_OK, err.c_str())) return 1;
            const PieCompletion completion{
                .wait_id = 400 + static_cast<std::uint64_t>(fired),
                .target_epoch = 1,
                .terminal_cell = nullptr,
            };
            g_stage = "run";
            if (!expect(dispatch.run(view, d_logits, kVocab, stream, &runtime,
                                     completion),
                        "run accepted")) return 1;
        }

        // Scheduler-thread churn concurrent with the genuinely in-flight
        // fires above (no sync yet): registry growth under load. Churn
        // channels stay LIVE across iterations, so the table crosses the
        // initial 1024-slot capacity mid-loop and grow() reallocates the
        // shared head/tail/full arrays while kernels advance them through
        // pointers baked at launch (RV-27) — the committed-per-pair check
        // below is what catches a lost update, and ASAN/TSAN builds catch
        // the use-after-free.
        g_stage = "churn_register";
        std::vector<PieChannelEndpointBinding> churn_endpoints;
        const auto churn_ids = register_channels(
            dispatch, container, churn_channel_id, churn_endpoints, err);
        if (!expect(!churn_ids.empty(), err.c_str())) return 1;
        churn_channel_id += churn_ids.size();
        for (const std::uint64_t id : churn_ids) {
            live_churn_ids.push_back(id);
        }

        // … and bind/close of an unrelated instance, also under load.
        g_stage = "churn_bind";
        PieInstanceBinding churn_binding{};
        const PieChannelValueDesc churn_seed{
            .channel_id = churn_ids[0],
            .bytes = {.ptr = seed_bytes, .len = sizeof(seed_bytes)},
        };
        if (!expect(dispatch.bind_instance(
                        churn_instance_id++, hash, PIE_GEOMETRY_CLASS_HOST,
                        /*pacing_wait_id=*/4321,
                        churn_ids, {churn_seed}, &churn_binding,
                        &err) == PIE_STATUS_OK,
                    err.c_str())) return 1;
        dispatch.close_instance(churn_binding.instance_id);

        cudaDeviceSynchronize();
        int committed = 0;
        for (int burst = 0; burst < kFiresPerIteration; ++burst) {
            const auto outcome = std::atomic_ref<std::uint32_t>(
                terminals[fired - kFiresPerIteration + burst].outcome)
                                     .load(std::memory_order_acquire);
            committed += outcome == PIE_TERMINAL_OUTCOME_SUCCESS ? 1 : 0;
        }
        if (!expect(committed == 1, "one ordered fire commits per capacity-1 pair")) {
            return 1;
        }
        advance_reader_head(endpoints[1]);
        const std::uint32_t token[1] = {1};
        ring_put(endpoints[0], token, sizeof(token));
        next_sequence += static_cast<std::uint64_t>(committed);
    }

    g_stage = "drain";
    cudaDeviceSynchronize();
    // The coverage this stress claims: the churn actually forced at least
    // one registry growth while fires were in flight.
    if (!expect(dispatch.stats().channel_slot_capacity > initial_slot_capacity,
                "registry grew under in-flight load")) return 1;
    if (!expect(g_notifies.load(std::memory_order_relaxed) >=
                    static_cast<std::uint64_t>(fired),
                "every accepted fire notified")) return 1;
    // Every terminal cell settled (SUCCESS or FAILED, never PENDING).
    for (int i = 0; i < fired; ++i) {
        const auto outcome =
            std::atomic_ref<std::uint32_t>(terminals[i].outcome)
                .load(std::memory_order_acquire);
        if (!expect(outcome != PIE_TERMINAL_OUTCOME_PENDING,
                    "terminal cell settled")) return 1;
    }

    dispatch.close_instance(binding.instance_id);
    g_stage = "close_live_churn";
    for (const std::uint64_t id : live_churn_ids) {
        if (!expect(dispatch.close_channel(id, &err) == PIE_STATUS_OK,
                    err.c_str())) return 1;
    }
    for (const std::uint64_t id : channel_ids) {
        if (!expect(dispatch.close_channel(id, &err) == PIE_STATUS_OK,
                    err.c_str())) return 1;
    }
    cudaStreamDestroy(stream);
    cudaFree(d_logits);
    std::printf("test_ptir_dispatch_race: OK (%d fires, %llu notifies)\n",
                fired,
                static_cast<unsigned long long>(
                    g_notifies.load(std::memory_order_relaxed)));
    return 0;
}
