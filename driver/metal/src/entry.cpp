#include "entry.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.hpp"
#include "pie_native/abi_validation.hpp"
#include "pie_native/ptir_channels.hpp"
#include "ptir/host_interp.hpp"

namespace {

namespace interp = pie_metal_driver::ptir_host;

struct ProgramRecord {
    std::uint64_t program_id = 0;
    std::uint64_t program_hash = 0;
    std::vector<std::uint8_t> canonical;
    std::vector<std::uint8_t> sidecar;
    std::vector<pie_native::PtirChannelDecl> channels;
    // Decoded execution plan; `plan.executable == false` carries the launch
    // rejection reason (registration itself stays lenient).
    interp::ExecPlan plan;
};

struct InstanceRecord {
    std::uint64_t instance_id = 0;
    std::uint64_t program_id = 0;
    std::uint64_t program_hash = 0;
    std::vector<std::uint64_t> channel_ids;
    std::uint64_t fire_seq = 0;
    interp::InterpInstance interp;
};

struct ChannelRecord {
    PieChannelDesc desc{};
    std::vector<std::uint32_t> shape;
    std::string extern_name;
    std::vector<std::uint8_t> mirror;
    std::vector<std::uint64_t> words;
    std::unordered_map<std::uint64_t, std::uint8_t> attachments;
    // §4.3 writer-ring cursors: cells moved into the interp (`pulled`) and
    // consumed cells whose head word is published (`reserved_head`). A bound
    // seed spends as one take without a head-word publish (`seed_credit`).
    std::uint64_t pulled = 0;
    std::uint64_t reserved_head = 0;
    bool seed_credit = false;
    // Extern channels share one interp ring across the exporting and
    // importing instance (created at first bind).
    std::shared_ptr<interp::ChannelState> shared_state;

    std::size_t numel() const {
        std::size_t n = 1;
        for (const std::uint32_t d : shape) n *= d;
        return n;
    }
    interp::DType program_dtype() const {
        switch (desc.dtype) {
            case PIE_CHANNEL_DTYPE_I32: return interp::DType::I32;
            case PIE_CHANNEL_DTYPE_U32: return interp::DType::U32;
            case PIE_CHANNEL_DTYPE_BOOL: return interp::DType::Bool;
            default: return interp::DType::F32;
        }
    }
};

void store_word(std::vector<std::uint64_t>& words, std::size_t index, std::uint64_t value) {
    std::atomic_ref<std::uint64_t>(words[index]).store(value, std::memory_order_release);
}

std::uint64_t load_word(const std::vector<std::uint64_t>& words, std::size_t index) {
    // libc++ lacks the const atomic_ref specialization; the underlying word is
    // never actually const (driver-owned storage).
    return std::atomic_ref<std::uint64_t>(const_cast<std::uint64_t&>(words[index]))
        .load(std::memory_order_acquire);
}

void publish_terminal(PieTerminalCell* cell, std::uint32_t outcome) {
    if (cell == nullptr) return;
    cell->reserved0 = 0;
    std::atomic_ref<std::uint32_t>(cell->outcome).store(outcome, std::memory_order_release);
}

// One scheduled writer-ring consume: `target_head == reserved_head` marks a
// seed-credit spend (no head-word publish).
struct WriterConsume {
    std::uint64_t channel_id = 0;
    std::size_t dense = 0;
    std::uint64_t target_head = 0;
};

struct ModelFacts {
    std::uint32_t vocab_size = 32000;
    std::uint32_t max_model_len = 8192;
    std::string arch_name = "llama";
    bool has_linear_attn = false;
};

ModelFacts read_model_facts(const std::string& hf_path) {
    ModelFacts facts;
    if (hf_path.empty()) return facts;
    const std::filesystem::path cfg =
        std::filesystem::path(hf_path) / "config.json";
    std::ifstream f(cfg);
    if (!f) return facts;
    try {
        nlohmann::json j;
        f >> j;
        if (j.contains("vocab_size") && j["vocab_size"].is_number_integer()) {
            facts.vocab_size = j["vocab_size"].get<std::uint32_t>();
        }
        if (j.contains("max_position_embeddings") &&
            j["max_position_embeddings"].is_number_integer()) {
            facts.max_model_len = j["max_position_embeddings"].get<std::uint32_t>();
        }
        if (j.contains("architectures") && j["architectures"].is_array() &&
            !j["architectures"].empty()) {
            std::string a = j["architectures"][0].get<std::string>();
            for (auto& c : a) c = static_cast<char>(std::tolower(c));
            const std::string suffix = "forcausallm";
            if (a.size() > suffix.size() &&
                a.compare(a.size() - suffix.size(), suffix.size(), suffix) == 0) {
                a.erase(a.size() - suffix.size());
            }
            if (!a.empty()) facts.arch_name = a;
        }
        const nlohmann::json& tc =
            (j.contains("text_config") && j["text_config"].is_object())
                ? j["text_config"]
                : j;
        if (tc.contains("linear_num_value_heads") &&
            tc["linear_num_value_heads"].is_number_integer() &&
            tc["linear_num_value_heads"].get<int>() > 0) {
            facts.has_linear_attn = true;
        }
        if (tc.contains("layer_types") && tc["layer_types"].is_array()) {
            for (const auto& t : tc["layer_types"]) {
                if (t.is_string() && t.get<std::string>() == "linear_attention") {
                    facts.has_linear_attn = true;
                    break;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] warning: failed to parse "
                  << cfg.string() << ": " << e.what() << "\n";
    }
    return facts;
}

std::string build_caps_json(const pie_metal_driver::Config& cfg,
                            const ModelFacts& facts) {
    const bool rs_cache_required = facts.has_linear_attn;
    const std::uint32_t rs_cache_slots =
        rs_cache_required ? cfg.batching.max_forward_requests : 0u;
    const std::uint32_t rs_cache_slot_bytes = 0u;
    const std::uint32_t max_forward_requests =
        rs_cache_required
            ? std::min(cfg.batching.max_forward_requests, rs_cache_slots)
            : cfg.batching.max_forward_requests;
    nlohmann::json caps = {
        {"abi_version", PIE_DRIVER_ABI_VERSION},
        {"total_pages", cfg.batching.total_pages},
        {"kv_page_size", cfg.batching.kv_page_size},
        {"swap_pool_size", cfg.batching.cpu_pages},
        {"rs_cache_required", rs_cache_required},
        {"rs_cache_slots", rs_cache_slots},
        {"rs_cache_slot_bytes", rs_cache_slot_bytes},
        {"max_forward_tokens", cfg.batching.max_forward_tokens},
        {"max_forward_requests", max_forward_requests},
        {"max_page_refs", cfg.batching.total_pages},
        {"arch_name", facts.arch_name},
        {"vocab_size", facts.vocab_size},
        {"max_model_len", facts.max_model_len},
        {"activation_dtype", "bf16"},
        {"snapshot_dir", cfg.model.hf_path},
        {"storage_backend", "metal"},
        {"max_tile_bytes", 0},
        {"preferred_alignment", 0},
        {"mxfp4_moe_policy", ""},
        {"native_mxfp4_moe", false},
    };
    return caps.dump();
}

class MetalDriver {
  public:
    int initialize(const std::string& config_path, const PieRuntimeCallbacks& runtime) {
        cfg_ = pie_metal_driver::load_config(config_path);
        facts_ = read_model_facts(cfg_.model.hf_path);
        caps_json_ = build_caps_json(cfg_, facts_);
        runtime_ = runtime;
        return PIE_STATUS_OK;
    }

    void fill_caps(PieDriverCaps* caps) const {
        if (caps == nullptr) return;
        caps->json_bytes = reinterpret_cast<const std::uint8_t*>(caps_json_.data());
        caps->json_len = caps_json_.size();
    }

    int register_program(const PieProgramDesc& program, std::uint64_t* program_id) {
        auto found = program_ids_by_hash_.find(program.program_hash);
        if (found != program_ids_by_hash_.end()) {
            if (program_id != nullptr) *program_id = found->second;
            return PIE_STATUS_OK;
        }
        if (program.canonical_bytes.len == 0)
            return PIE_STATUS_INVALID_ARGUMENT;

        ProgramRecord record;
        record.program_id = next_program_id_++;
        record.program_hash = program.program_hash;
        record.canonical.assign(program.canonical_bytes.ptr,
                                program.canonical_bytes.ptr + program.canonical_bytes.len);
        std::string decode_error;
        if (!pie_native::decode_ptir_channels(
                record.canonical.data(), record.canonical.size(),
                record.channels, &decode_error)) {
            std::cerr << "[pie-driver-metal] register_program: "
                      << decode_error << "\n";
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        if (program.sidecar_bytes.ptr != nullptr && program.sidecar_bytes.len > 0) {
            record.sidecar.assign(program.sidecar_bytes.ptr,
                                  program.sidecar_bytes.ptr + program.sidecar_bytes.len);
        }
        if (!interp::build_exec_plan(record.canonical.data(), record.canonical.size(),
                                     record.sidecar.data(), record.sidecar.size(),
                                     record.plan, &decode_error)) {
            std::cerr << "[pie-driver-metal] register_program: " << decode_error << "\n";
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        program_ids_by_hash_[record.program_hash] = record.program_id;
        if (program_id != nullptr) *program_id = record.program_id;
        programs_.emplace(record.program_id, std::move(record));
        return PIE_STATUS_OK;
    }

    int register_channel(const PieChannelDesc& channel,
                         PieChannelEndpointBinding* binding) {
        if (channels_.find(channel.channel_id) != channels_.end()) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        pie_native::PtirChannelDecl geometry;
        geometry.dtype = channel.dtype;
        geometry.dims.assign(channel.shape.ptr, channel.shape.ptr + channel.shape.len);
        const std::uint32_t cell_bytes = geometry.cell_bytes();
        if (cell_bytes == 0) return PIE_STATUS_INVALID_ARGUMENT;
        ChannelRecord record;
        record.desc = channel;
        record.shape.assign(channel.shape.ptr, channel.shape.ptr + channel.shape.len);
        record.desc.shape.ptr = record.shape.data();
        if (channel.extern_name.len != 0) {
            record.extern_name.assign(
                reinterpret_cast<const char*>(channel.extern_name.ptr),
                channel.extern_name.len);
        }
        record.desc.extern_name.ptr =
            reinterpret_cast<const std::uint8_t*>(record.extern_name.data());
        record.mirror.assign(
            static_cast<std::size_t>(cell_bytes) * (channel.capacity + 1), 0);
        record.words.assign(4, 0);
        auto [it, inserted] =
            channels_.emplace(channel.channel_id, std::move(record));
        if (!inserted) return PIE_STATUS_INVALID_ARGUMENT;
        ChannelRecord& stored = it->second;
        stored.desc.shape.ptr = stored.shape.data();
        stored.desc.extern_name.ptr =
            reinterpret_cast<const std::uint8_t*>(stored.extern_name.data());
        *binding = PieChannelEndpointBinding{
            .channel_id = channel.channel_id,
            .mirror_base = reinterpret_cast<std::uint64_t>(stored.mirror.data()),
            .word_base = reinterpret_cast<std::uint64_t>(stored.words.data()),
            .mirror_bytes = stored.mirror.size(),
            .word_bytes = stored.words.size() * sizeof(std::uint64_t),
            .cell_bytes = static_cast<std::uint32_t>(cell_bytes),
            .capacity = channel.capacity,
            .head_word_index = 0,
            .tail_word_index = 1,
            .poison_word_index = 2,
            .closed_word_index = 3,
        };
        return PIE_STATUS_OK;
    }

    int bind_instance(const PieInstanceDesc& instance, PieInstanceBinding* binding) {
        if (programs_.find(instance.program_id) == programs_.end())
            return PIE_STATUS_INVALID_ARGUMENT;
        const std::uint64_t instance_id =
            instance.requested_instance_id != 0 ? instance.requested_instance_id : next_instance_id_++;
        if (instances_.find(instance_id) != instances_.end())
            return PIE_STATUS_INVALID_ARGUMENT;
        const ProgramRecord& program = programs_.at(instance.program_id);
        if (instance.channel_ids.len != program.channels.size()) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        {
            std::unordered_set<std::uint64_t> unique_ids(
                instance.channel_ids.ptr,
                instance.channel_ids.ptr + instance.channel_ids.len);
            if (unique_ids.size() != instance.channel_ids.len) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            std::unordered_set<std::uint64_t> seeded_ids;
            for (std::size_t i = 0; i < instance.seed_values.len; ++i) {
                const PieChannelValueDesc& seed = instance.seed_values.ptr[i];
                const auto id = std::find(
                    instance.channel_ids.ptr,
                    instance.channel_ids.ptr + instance.channel_ids.len,
                    seed.channel_id);
                if (id == instance.channel_ids.ptr + instance.channel_ids.len ||
                    !seeded_ids.insert(seed.channel_id).second) {
                    return PIE_STATUS_INVALID_ARGUMENT;
                }
                const std::size_t channel =
                    static_cast<std::size_t>(id - instance.channel_ids.ptr);
                if (!program.channels[channel].seeded ||
                    seed.bytes.len != program.channels[channel].cell_bytes() ||
                    seed.bytes.ptr == nullptr) {
                    return PIE_STATUS_INVALID_ARGUMENT;
                }
            }
        }
        for (std::size_t i = 0; i < instance.channel_ids.len; ++i) {
            auto endpoint = channels_.find(instance.channel_ids.ptr[i]);
            if (endpoint == channels_.end()) return PIE_STATUS_INVALID_ARGUMENT;
            const auto& decl = program.channels[i];
            const auto& endpoint_desc = endpoint->second.desc;
            if (endpoint_desc.dtype != decl.dtype ||
                endpoint_desc.capacity != decl.capacity ||
                endpoint_desc.host_role != decl.host_role ||
                endpoint_desc.seeded != static_cast<std::uint8_t>(decl.seeded) ||
                endpoint->second.shape != decl.dims ||
                (decl.extern_dir == PIE_CHANNEL_EXTERN_NONE
                     ? endpoint_desc.extern_dir != PIE_CHANNEL_EXTERN_NONE ||
                           !endpoint->second.attachments.empty()
                     : endpoint_desc.extern_dir == PIE_CHANNEL_EXTERN_NONE ||
                           endpoint->second.extern_name != decl.extern_name ||
                           std::any_of(
                               endpoint->second.attachments.begin(),
                               endpoint->second.attachments.end(),
                               [&](const auto& attachment) {
                                   return attachment.second == decl.extern_dir;
                               }))) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
        InstanceRecord record;
        record.instance_id = instance_id;
        record.program_id = instance.program_id;
        record.program_hash = program.program_hash;
        record.channel_ids.assign(
            instance.channel_ids.ptr,
            instance.channel_ids.ptr + instance.channel_ids.len);

        // Seeds are per-instance data (D2): they pre-fill the interp ring, not
        // the host-owned wire ring. A seeded host-Writer channel additionally
        // carries one take of credit that spends without a head-word publish
        // (plan §4.3, the dummy driver's scheme).
        std::map<std::uint32_t, interp::Value> seeds;
        for (std::size_t s = 0; s < instance.seed_values.len; ++s) {
            const PieChannelValueDesc& seed = instance.seed_values.ptr[s];
            const auto id = std::find(record.channel_ids.begin(), record.channel_ids.end(),
                                      seed.channel_id);
            const auto dense = static_cast<std::uint32_t>(id - record.channel_ids.begin());
            ChannelRecord& endpoint = channels_.at(seed.channel_id);
            interp::Value value;
            if (!interp::decode_wire(seed.bytes.ptr, seed.bytes.len, endpoint.program_dtype(),
                                     endpoint.numel(), value)) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            seeds.emplace(dense, std::move(value));
            if (endpoint.desc.host_role == PIE_CHANNEL_HOST_ROLE_WRITER) {
                endpoint.seed_credit = true;
            }
        }
        std::map<std::uint32_t, std::shared_ptr<interp::ChannelState>> externs;
        for (std::size_t i = 0; i < record.channel_ids.size(); ++i) {
            if (program.channels[i].extern_dir == PIE_CHANNEL_EXTERN_NONE) continue;
            ChannelRecord& endpoint = channels_.at(record.channel_ids[i]);
            if (endpoint.shared_state == nullptr) {
                endpoint.shared_state = std::make_shared<interp::ChannelState>();
                endpoint.shared_state->capacity = endpoint.desc.capacity;
                endpoint.shared_state->last =
                    interp::zeros(endpoint.program_dtype(), endpoint.numel());
            }
            externs.emplace(static_cast<std::uint32_t>(i), endpoint.shared_state);
        }
        record.interp = interp::make_instance(program.plan, externs, seeds);

        for (std::size_t i = 0; i < record.channel_ids.size(); ++i) {
            channels_.at(record.channel_ids[i])
                .attachments.emplace(instance_id, program.channels[i].extern_dir);
        }
        if (binding != nullptr) {
            std::memset(binding, 0, sizeof(*binding));
            binding->instance_id = instance_id;
        }
        instances_[instance_id] = std::move(record);
        return PIE_STATUS_OK;
    }

    // ABI v2 launch: host puts already live in the endpoint's shared ring
    // (words[1] is the producer tail). Channel-plane PTIR programs execute on
    // the host interpreter; §4.3 pulls + availability rejects synchronously,
    // §4.4 publishes reader tails / writer heads / terminals and notifies the
    // per-channel wait slots, then the batch slot exactly once. Programs that
    // need the model forward (intrinsics, per-layer taps, kernel calls) stay
    // UNSUPPORTED until the Metal executor is wired.
    int launch(const PieLaunchDesc& launch, PieCompletion completion) {
        std::vector<InstanceRecord*> members;
        members.reserve(launch.instance_ids.len);
        for (std::size_t i = 0; i < launch.instance_ids.len; ++i) {
            const auto instance_it = instances_.find(launch.instance_ids.ptr[i]);
            if (instance_it == instances_.end()) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            members.push_back(&instance_it->second);
        }
        for (const InstanceRecord* member : members) {
            const ProgramRecord& program = programs_.at(member->program_id);
            if (!program.plan.executable) {
                std::cerr << "[pie-driver-metal] launch: instance " << member->instance_id
                          << ": " << program.plan.reject_reason << "\n";
                return PIE_STATUS_UNSUPPORTED;
            }
        }

        // §4.3: pull host-published writer-ring entries into the interp, then
        // validate availability for every member BEFORE accepting anything —
        // aggregated across members sharing an endpoint. A missing put is a
        // guest ordering bug and rejects synchronously (no epoch, no poison).
        std::vector<std::vector<WriterConsume>> consumes(members.size());
        std::unordered_map<ChannelRecord*, std::uint64_t> planned_extra;
        std::unordered_set<ChannelRecord*> planned_seed_spend;
        for (std::size_t m = 0; m < members.size(); ++m) {
            InstanceRecord& member = *members[m];
            const ProgramRecord& program = programs_.at(member.program_id);
            for (std::size_t dense = 0; dense < member.channel_ids.size(); ++dense) {
                ChannelRecord& endpoint = channels_.at(member.channel_ids[dense]);
                if (endpoint.desc.host_role != PIE_CHANNEL_HOST_ROLE_WRITER) continue;
                const int rc = pull_writer_ring(member, program, dense, endpoint);
                if (rc != PIE_STATUS_OK) return rc;
                if (!program.plan.takes_channel(static_cast<std::uint32_t>(dense))) continue;
                const std::uint64_t tail = load_word(endpoint.words, 1);
                std::uint64_t& extra = planned_extra[&endpoint];
                const bool credit =
                    endpoint.seed_credit && planned_seed_spend.count(&endpoint) == 0;
                const std::uint64_t reserved = endpoint.reserved_head + extra;
                const std::uint64_t available =
                    (tail > reserved ? tail - reserved : 0) + (credit ? 1 : 0);
                if (available < 1) {
                    std::cerr << "[pie-driver-metal] launch: channel "
                              << member.channel_ids[dense]
                              << " has no host input for this fire (put must happen "
                                 "before submit)\n";
                    return PIE_STATUS_INVALID_ARGUMENT;
                }
                if (credit) {
                    planned_seed_spend.insert(&endpoint);
                    consumes[m].push_back({member.channel_ids[dense], dense, reserved});
                } else {
                    ++extra;
                    consumes[m].push_back({member.channel_ids[dense], dense, reserved + 1});
                }
            }
        }
        // Every member validated — commit the scheduled reservations.
        for (const auto& member_consumes : consumes) {
            for (const WriterConsume& consume : member_consumes) {
                ChannelRecord& endpoint = channels_.at(consume.channel_id);
                if (endpoint.seed_credit && consume.target_head == endpoint.reserved_head) {
                    endpoint.seed_credit = false;
                } else {
                    endpoint.reserved_head = consume.target_head;
                }
            }
        }

        // Execute and settle each member; publication order per §4.4: channel
        // words while settling, then terminal cells, then per-channel
        // notifies, then the batch notify exactly once.
        std::vector<std::uint32_t> outcomes(members.size(), PIE_TERMINAL_OUTCOME_SUCCESS);
        std::vector<std::pair<std::uint64_t, std::uint64_t>> notifications;
        for (std::size_t m = 0; m < members.size(); ++m) {
            InstanceRecord& member = *members[m];
            const ProgramRecord& program = programs_.at(member.program_id);
            member.fire_seq += 1;
            std::string failure;
            if (!run_member(member, program, consumes[m], notifications, failure)) {
                std::cerr << "[pie-driver-metal] instance " << member.instance_id
                          << " launch failed: " << failure << "\n";
                poison_instance(member, notifications);
                outcomes[m] = PIE_TERMINAL_OUTCOME_FAILED;
            }
        }
        for (std::size_t m = 0; m < members.size(); ++m) {
            publish_terminal(launch.terminal_cells.ptr[m], outcomes[m]);
        }
        for (const auto& [wait_id, epoch] : notifications) {
            notify(wait_id, epoch);
        }
        notify(completion.wait_id, completion.target_epoch);
        return PIE_STATUS_OK;
    }

    int copy_kv(const PieKvCopyDesc&, PieCompletion completion) {
        static_cast<void>(completion);
        return PIE_STATUS_UNSUPPORTED;
    }
    int copy_state(const PieStateCopyDesc&, PieCompletion completion) {
        static_cast<void>(completion);
        return PIE_STATUS_UNSUPPORTED;
    }
    int resize_pool(const PiePoolResizeDesc&, PieCompletion completion) {
        static_cast<void>(completion);
        return PIE_STATUS_UNSUPPORTED;
    }

    int close_instance(std::uint64_t instance_id) {
        auto it = instances_.find(instance_id);
        if (it == instances_.end()) return PIE_STATUS_CLOSED;
        for (const std::uint64_t channel_id : it->second.channel_ids) {
            auto channel = channels_.find(channel_id);
            if (channel != channels_.end()) {
                channel->second.attachments.erase(instance_id);
            }
        }
        instances_.erase(it);
        return PIE_STATUS_OK;
    }

    int close_channel(std::uint64_t channel_id) {
        auto it = channels_.find(channel_id);
        if (it == channels_.end()) return PIE_STATUS_CLOSED;
        if (!it->second.attachments.empty()) return PIE_STATUS_INVALID_ARGUMENT;
        std::atomic_ref<std::uint64_t>(it->second.words[3]).store(
            1, std::memory_order_release);
        channels_.erase(it);
        return PIE_STATUS_OK;
    }

  private:
    void notify(std::uint64_t wait_id, std::uint64_t epoch) const {
        if (wait_id == 0 || epoch == 0 || runtime_.notify == nullptr) return;
        runtime_.notify(runtime_.ctx, wait_id, epoch);
    }

    // §4.3 driver pull: move host-published writer-ring cells (mirror cells up
    // to the release-published tail word) into the interp ring, advancing
    // `pulled`. Interp backpressure leaves the remainder for a later fire.
    int pull_writer_ring(InstanceRecord& member, const ProgramRecord& program,
                         std::size_t dense, ChannelRecord& endpoint) {
        const std::size_t cell =
            interp::wire_cell_bytes(endpoint.program_dtype(), endpoint.numel());
        const std::uint64_t cap1 = static_cast<std::uint64_t>(endpoint.desc.capacity) + 1;
        for (;;) {
            const std::uint64_t tail = load_word(endpoint.words, 1);
            if (endpoint.pulled >= tail) return PIE_STATUS_OK;
            const std::size_t offset = static_cast<std::size_t>(endpoint.pulled % cap1) * cell;
            interp::Value value;
            if (!interp::decode_wire(endpoint.mirror.data() + offset, cell,
                                     endpoint.program_dtype(), endpoint.numel(), value)) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            switch (interp::host_put(member.interp, program.plan,
                                     static_cast<std::uint32_t>(dense), std::move(value))) {
                case interp::HostOp::Ok:
                    endpoint.pulled += 1;
                    break;
                case interp::HostOp::WouldBlock:
                    return PIE_STATUS_OK;
                default:
                    std::cerr << "[pie-driver-metal] writer ring pull failed for channel "
                              << endpoint.desc.channel_id << "\n";
                    return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    // One member's fire: interp step, then §4.4 word publication — consumed
    // writer heads (release + writer wake), produced reader cells + tails
    // (release + reader wake). Returns false with `failure` set on any fault;
    // the caller poisons.
    bool run_member(InstanceRecord& member, const ProgramRecord& program,
                    const std::vector<WriterConsume>& member_consumes,
                    std::vector<std::pair<std::uint64_t, std::uint64_t>>& notifications,
                    std::string& failure) {
        const interp::StepResult report = interp::step(member.interp, program.plan);
        if (!report.ok) {
            failure = report.error;
            return false;
        }
        if (!report.committed) {
            failure = "accepted launch missed its readiness invariant at channel " +
                      std::to_string(report.missed_channel);
            return false;
        }
        for (const auto& consume : member_consumes) {
            ChannelRecord& endpoint = channels_.at(consume.channel_id);
            const std::uint64_t previous = load_word(endpoint.words, 0);
            if (consume.target_head > previous) {
                store_word(endpoint.words, 0, consume.target_head);
                notifications.emplace_back(endpoint.desc.writer_wait_id, consume.target_head);
            }
        }
        for (std::size_t dense = 0; dense < member.channel_ids.size(); ++dense) {
            ChannelRecord& endpoint = channels_.at(member.channel_ids[dense]);
            if (endpoint.desc.host_role != PIE_CHANNEL_HOST_ROLE_READER) continue;
            for (;;) {
                interp::Value value;
                const interp::HostOp rc = interp::host_take(
                    member.interp, program.plan, static_cast<std::uint32_t>(dense), value);
                if (rc == interp::HostOp::WouldBlock) break;
                if (rc != interp::HostOp::Ok) {
                    failure = "host take failed for channel " +
                              std::to_string(endpoint.desc.channel_id);
                    return false;
                }
                const std::uint64_t tail = load_word(endpoint.words, 1);
                const std::uint64_t head = load_word(endpoint.words, 0);
                if (tail - head >= endpoint.desc.capacity) {
                    failure = "channel " + std::to_string(endpoint.desc.channel_id) +
                              " has no reserved output capacity";
                    return false;
                }
                const std::size_t cell =
                    interp::wire_cell_bytes(endpoint.program_dtype(), endpoint.numel());
                const std::uint64_t cap1 =
                    static_cast<std::uint64_t>(endpoint.desc.capacity) + 1;
                interp::encode_wire(
                    value, endpoint.mirror.data() +
                               static_cast<std::size_t>(tail % cap1) * cell);
                store_word(endpoint.words, 1, tail + 1);
                store_word(endpoint.words, 2, 0);
                notifications.emplace_back(endpoint.desc.reader_wait_id, tail + 1);
            }
        }
        return true;
    }

    // D4 failure settlement: poison every attached channel's word (epoch
    // `tail + 1`, first poison wins device-side by construction) and queue the
    // role's wake so parked waiters observe Poisoned, not Empty.
    void poison_instance(InstanceRecord& member,
                         std::vector<std::pair<std::uint64_t, std::uint64_t>>& notifications) {
        member.interp.poisoned = true;
        for (const std::uint64_t channel_id : member.channel_ids) {
            ChannelRecord& endpoint = channels_.at(channel_id);
            const std::uint64_t tail = load_word(endpoint.words, 1);
            const std::uint64_t poison_epoch = std::max<std::uint64_t>(tail + 1, 1);
            store_word(endpoint.words, 2, poison_epoch);
            const std::uint64_t wait_id =
                endpoint.desc.host_role == PIE_CHANNEL_HOST_ROLE_READER
                    ? endpoint.desc.reader_wait_id
                    : (endpoint.desc.host_role == PIE_CHANNEL_HOST_ROLE_WRITER
                           ? endpoint.desc.writer_wait_id
                           : 0);
            notifications.emplace_back(wait_id, poison_epoch);
        }
    }

    pie_metal_driver::Config cfg_{};
    ModelFacts facts_{};
    std::string caps_json_;
    PieRuntimeCallbacks runtime_{};
    std::uint64_t next_program_id_ = 1;
    std::uint64_t next_instance_id_ = 1;
    std::unordered_map<std::uint64_t, ProgramRecord> programs_;
    std::unordered_map<std::uint64_t, std::uint64_t> program_ids_by_hash_;
    std::unordered_map<std::uint64_t, InstanceRecord> instances_;
    std::unordered_map<std::uint64_t, ChannelRecord> channels_;
};

PieDriver* create_driver_impl(const PieDriverCreateDesc* desc, PieDriverCaps* caps) {
    std::memset(caps, 0, sizeof(*caps));
    const std::string config_path(
        reinterpret_cast<const char*>(desc->config_bytes.ptr),
        desc->config_bytes.len);
    auto driver = std::make_unique<MetalDriver>();
    const int rc = driver->initialize(config_path, desc->runtime);
    if (rc != PIE_STATUS_OK) return nullptr;
    driver->fill_caps(caps);
    return reinterpret_cast<PieDriver*>(driver.release());
}

MetalDriver* as_driver(PieDriver* driver) {
    return reinterpret_cast<MetalDriver*>(driver);
}

}  // namespace

extern "C" PieDriver* pie_metal_create(const PieDriverCreateDesc* desc,
                                       PieDriverCaps* caps) {
    if (pie_native::abi::validate_create_desc(desc, caps) != PIE_STATUS_OK) {
        return nullptr;
    }
    try {
        return create_driver_impl(desc, caps);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] create: " << e.what() << "\n";
        return nullptr;
    } catch (...) {
        std::cerr << "[pie-driver-metal] create: unknown exception\n";
        return nullptr;
    }
}

extern "C" int32_t pie_metal_register_program(PieDriver* driver,
                                              const PieProgramDesc* program,
                                              std::uint64_t* program_id) {
    const int status = pie_native::abi::validate_program_desc(program, program_id);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->register_program(*program, program_id);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_register_channel(
    PieDriver* driver,
    const PieChannelDesc* channel,
    PieChannelEndpointBinding* binding) {
    const int status = pie_native::abi::validate_channel_desc(channel, binding);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->register_channel(*channel, binding);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_bind_instance(PieDriver* driver,
                                           const PieInstanceDesc* instance,
                                           PieInstanceBinding* binding) {
    const int status = pie_native::abi::validate_instance_desc(instance, binding);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->bind_instance(*instance, binding);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_launch(PieDriver* driver,
                                    const PieLaunchDesc* launch,
                                    PieCompletion completion) {
    const int status = pie_native::abi::validate_launch_desc(launch);
    if (status != PIE_STATUS_OK) return status;
    const int completion_status =
        pie_native::abi::validate_completion(completion, false);
    if (completion_status != PIE_STATUS_OK) return completion_status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->launch(*launch, completion);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_copy_kv(PieDriver* driver,
                                     const PieKvCopyDesc* copy,
                                     PieCompletion completion) {
    const int status = pie_native::abi::validate_kv_copy_desc(copy);
    if (status != PIE_STATUS_OK) return status;
    const int completion_status =
        pie_native::abi::validate_completion(completion, true);
    if (completion_status != PIE_STATUS_OK) return completion_status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->copy_kv(*copy, completion);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_copy_state(PieDriver* driver,
                                        const PieStateCopyDesc* copy,
                                        PieCompletion completion) {
    const int status = pie_native::abi::validate_state_copy_desc(copy);
    if (status != PIE_STATUS_OK) return status;
    const int completion_status =
        pie_native::abi::validate_completion(completion, true);
    if (completion_status != PIE_STATUS_OK) return completion_status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->copy_state(*copy, completion);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_resize_pool(PieDriver* driver,
                                         const PiePoolResizeDesc* resize,
                                         PieCompletion completion) {
    const int status = pie_native::abi::validate_pool_resize_desc(resize);
    if (status != PIE_STATUS_OK) return status;
    const int completion_status =
        pie_native::abi::validate_completion(completion, true);
    if (completion_status != PIE_STATUS_OK) return completion_status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->resize_pool(*resize, completion);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_close_instance(PieDriver* driver,
                                            std::uint64_t instance_id) {
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->close_instance(instance_id);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_close_channel(PieDriver* driver,
                                           std::uint64_t channel_id) {
    if (driver == nullptr || channel_id == 0) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->close_channel(channel_id);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" void pie_metal_destroy(PieDriver* driver) {
    delete as_driver(driver);
}
