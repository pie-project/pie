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
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.hpp"
#include "pie_native/abi_validation.hpp"
#include "pie_native/ptir_channels.hpp"

namespace {

struct ProgramRecord {
    std::uint64_t program_id = 0;
    std::uint64_t program_hash = 0;
    std::vector<std::uint8_t> canonical;
    std::vector<std::uint8_t> sidecar;
    std::vector<pie_native::PtirChannelDecl> channels;
};

struct InstanceRecord {
    std::uint64_t instance_id = 0;
    std::uint64_t program_id = 0;
    std::uint64_t program_hash = 0;
    std::vector<std::uint64_t> channel_ids;
    std::uint64_t fire_seq = 0;
};

struct ChannelRecord {
    PieChannelDesc desc{};
    std::vector<std::uint32_t> shape;
    std::string extern_name;
    std::vector<std::uint8_t> mirror;
    std::vector<std::uint64_t> words;
    std::unordered_map<std::uint64_t, std::uint8_t> attachments;
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
    int initialize(const std::string& config_path) {
        cfg_ = pie_metal_driver::load_config(config_path);
        facts_ = read_model_facts(cfg_.model.hf_path);
        caps_json_ = build_caps_json(cfg_, facts_);
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
        for (std::size_t s = 0; s < instance.seed_values.len; ++s) {
            const PieChannelValueDesc& seed = instance.seed_values.ptr[s];
            ChannelRecord& endpoint = channels_.at(seed.channel_id);
            std::memcpy(endpoint.mirror.data(), seed.bytes.ptr, seed.bytes.len);
            endpoint.words[1] = 1;
        }
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

    int launch(const PieLaunchDesc& launch, PieCompletion completion) {
        static_cast<void>(completion);
        for (std::size_t i = 0; i < launch.instance_ids.len; ++i) {
            const auto instance_it = instances_.find(launch.instance_ids.ptr[i]);
            if (instance_it == instances_.end()) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            const InstanceRecord& instance = instance_it->second;
            const ProgramRecord& program = programs_.at(instance.program_id);
            std::unordered_set<std::uint64_t> put_ids;
            const std::uint32_t lo = launch.host_put_indptr.len == 0
                ? 0
                : launch.host_put_indptr.ptr[i];
            const std::uint32_t hi = launch.host_put_indptr.len == 0
                ? 0
                : launch.host_put_indptr.ptr[i + 1];
            for (std::uint32_t e = lo; e < hi; ++e) {
                const PieChannelValueDesc& value =
                    launch.ptir_host_put_values.ptr[e];
                const auto id = std::find(
                    instance.channel_ids.begin(),
                    instance.channel_ids.end(),
                    value.channel_id);
                if (id == instance.channel_ids.end() ||
                    !put_ids.insert(value.channel_id).second) {
                    return PIE_STATUS_INVALID_ARGUMENT;
                }
                const std::size_t channel =
                    static_cast<std::size_t>(id - instance.channel_ids.begin());
                if (program.channels[channel].host_role !=
                        pie_native::PTIR_HOST_WRITER ||
                    value.bytes.len != program.channels[channel].cell_bytes()) {
                    return PIE_STATUS_INVALID_ARGUMENT;
                }
            }
        }
        return PIE_STATUS_UNSUPPORTED;
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
    pie_metal_driver::Config cfg_{};
    ModelFacts facts_{};
    std::string caps_json_;
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
    const int rc = driver->initialize(config_path);
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
