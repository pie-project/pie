#include "entry.hpp"

#include <algorithm>
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
    std::vector<PieChannelWait> channel_waits;
    std::vector<PieChannelBinding> channel_bindings;
    std::vector<std::uint8_t> mirror;
    std::vector<std::uint64_t> words;
    std::uint64_t fire_seq = 0;
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

constexpr std::uint32_t pacing_word_index() noexcept { return 0; }
constexpr std::uint32_t head_word_index(std::uint32_t channel) noexcept {
    return 1 + 3 * channel;
}
constexpr std::uint32_t tail_word_index(std::uint32_t channel) noexcept {
    return 2 + 3 * channel;
}
constexpr std::uint32_t poison_word_index(std::uint32_t channel) noexcept {
    return 3 + 3 * channel;
}
constexpr std::uint32_t word_count(std::uint32_t channel_count) noexcept {
    return 1 + 3 * channel_count;
}

class MetalDriver {
  public:
    explicit MetalDriver(const PieDriverCreateDesc& desc) : runtime_(desc.runtime) {}

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

    int bind_instance(const PieInstanceDesc& instance, PieInstanceBinding* binding) {
        if (programs_.find(instance.program_id) == programs_.end())
            return PIE_STATUS_INVALID_ARGUMENT;
        const std::uint64_t instance_id =
            instance.requested_instance_id != 0 ? instance.requested_instance_id : next_instance_id_++;
        InstanceRecord record;
        record.instance_id = instance_id;
        record.program_id = instance.program_id;
        record.program_hash = programs_.at(instance.program_id).program_hash;
        const ProgramRecord& program = programs_.at(instance.program_id);
        if (instance.channel_ids.len != program.channels.size() ||
            instance.channel_waits.len != program.channels.size()) {
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
                    seed.bytes.len != program.channels[channel].cell_bytes()) {
                    return PIE_STATUS_INVALID_ARGUMENT;
                }
            }
        }
        record.channel_ids.assign(
            instance.channel_ids.ptr,
            instance.channel_ids.ptr + instance.channel_ids.len);
        record.channel_waits.assign(
            instance.channel_waits.ptr,
            instance.channel_waits.ptr + instance.channel_waits.len);
        std::uint64_t mirror_offset = 0;
        record.channel_bindings.reserve(record.channel_ids.size());
        std::uint32_t reader_rank = 0;
        for (std::size_t i = 0; i < record.channel_ids.size(); ++i) {
            const auto& decl = program.channels[i];
            if (decl.host_role != pie_native::PTIR_HOST_READER) continue;
            const std::uint32_t cell_bytes = decl.cell_bytes();
            record.channel_bindings.push_back(PieChannelBinding{
                .channel_id = record.channel_ids[i],
                .cell_bytes = cell_bytes,
                .capacity = decl.capacity,
                .mirror_offset = mirror_offset,
                .head_word_index = head_word_index(reader_rank),
                .tail_word_index = tail_word_index(reader_rank),
                .poison_word_index = poison_word_index(reader_rank),
                .reserved = 0,
            });
            mirror_offset +=
                static_cast<std::uint64_t>(cell_bytes) * (decl.capacity + 1);
            ++reader_rank;
        }
        record.mirror.assign(static_cast<std::size_t>(mirror_offset), 0);
        record.words.assign(word_count(static_cast<std::uint32_t>(record.channel_bindings.size())), 0);
        for (std::size_t s = 0; s < instance.seed_values.len; ++s) {
            const PieChannelValueDesc& seed = instance.seed_values.ptr[s];
            for (const PieChannelBinding& channel : record.channel_bindings) {
                if (channel.channel_id != seed.channel_id) continue;
                const std::size_t n = std::min<std::size_t>(
                    channel.cell_bytes, seed.bytes.len);
                if (n > 0 && seed.bytes.ptr != nullptr) {
                    std::memcpy(record.mirror.data() + channel.mirror_offset,
                                seed.bytes.ptr, n);
                    record.words[channel.tail_word_index] = 1;
                }
                break;
            }
        }
        if (binding != nullptr) {
            std::memset(binding, 0, sizeof(*binding));
            binding->instance_id = instance_id;
            binding->mirror_base = reinterpret_cast<std::uint64_t>(record.mirror.data());
            binding->word_base = reinterpret_cast<std::uint64_t>(record.words.data());
            binding->channel_count =
                static_cast<std::uint32_t>(record.channel_bindings.size());
            binding->word_count = word_count(binding->channel_count);
            binding->mirror_bytes = record.mirror.size();
            binding->word_bytes = record.words.size() * sizeof(std::uint64_t);
            binding->channels.ptr = record.channel_bindings.data();
            binding->channels.len = record.channel_bindings.size();
        }
        instances_[instance_id] = std::move(record);
        return PIE_STATUS_OK;
    }

    int launch(const PieLaunchDesc& launch, PieCompletion completion) {
        for (std::size_t i = 0; i < launch.instance_ids.len; ++i) {
            if (instances_.find(launch.instance_ids.ptr[i]) == instances_.end())
                return PIE_STATUS_INVALID_ARGUMENT;
        }
        if (launch.instance_ids.len == 0) {
            notify_inline(completion);
            return PIE_STATUS_OK;
        }
        for (std::size_t i = 0; i < launch.instance_ids.len; ++i) {
            const InstanceRecord& instance =
                instances_.at(launch.instance_ids.ptr[i]);
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
        const bool have_host_puts =
            launch.host_put_indptr.ptr != nullptr &&
            launch.host_put_indptr.len == launch.instance_ids.len + 1 &&
            (launch.ptir_host_put_values.len == 0 ||
             launch.ptir_host_put_values.ptr != nullptr);
        for (std::size_t i = 0; i < launch.instance_ids.len; ++i) {
            InstanceRecord& instance = instances_.at(launch.instance_ids.ptr[i]);
            ++instance.fire_seq;
            instance.words[pacing_word_index()] = instance.fire_seq;
            if (!have_host_puts) continue;
            const std::uint32_t lo = launch.host_put_indptr.ptr[i];
            const std::uint32_t hi = launch.host_put_indptr.ptr[i + 1];
            for (std::uint32_t e = lo; e < hi; ++e) {
                const PieChannelValueDesc& value = launch.ptir_host_put_values.ptr[e];
                for (std::size_t c = 0; c < instance.channel_bindings.size(); ++c) {
                    const PieChannelBinding& channel = instance.channel_bindings[c];
                    if (channel.channel_id != value.channel_id) continue;
                    if (value.bytes.ptr != nullptr && !instance.mirror.empty()) {
                        const std::size_t n = std::min<std::size_t>(
                            channel.cell_bytes, value.bytes.len);
                        std::memcpy(instance.mirror.data() + channel.mirror_offset,
                                    value.bytes.ptr, n);
                    }
                    instance.words[channel.head_word_index] = 0;
                    instance.words[channel.tail_word_index] = 1;
                    instance.words[channel.poison_word_index] = 0;
                    break;
                }
            }
        }
        notify_inline(completion);
        return PIE_STATUS_OK;
    }

    int copy_kv(const PieKvCopyDesc&, PieCompletion completion) {
        notify_inline(completion);
        return PIE_STATUS_OK;
    }
    int copy_state(const PieStateCopyDesc&, PieCompletion completion) {
        notify_inline(completion);
        return PIE_STATUS_OK;
    }
    int resize_pool(const PiePoolResizeDesc&, PieCompletion completion) {
        notify_inline(completion);
        return PIE_STATUS_OK;
    }

    int close_instance(std::uint64_t instance_id) {
        auto it = instances_.find(instance_id);
        if (it == instances_.end()) return PIE_STATUS_CLOSED;
        instances_.erase(it);
        return PIE_STATUS_OK;
    }

  private:
    void notify_inline(PieCompletion completion) const {
        if (runtime_.notify != nullptr && completion.wait_id != 0) {
            runtime_.notify(runtime_.ctx, completion.wait_id, completion.target_epoch);
        }
    }

    PieRuntimeCallbacks runtime_{};
    pie_metal_driver::Config cfg_{};
    ModelFacts facts_{};
    std::string caps_json_;
    std::uint64_t next_program_id_ = 1;
    std::uint64_t next_instance_id_ = 1;
    std::unordered_map<std::uint64_t, ProgramRecord> programs_;
    std::unordered_map<std::uint64_t, std::uint64_t> program_ids_by_hash_;
    std::unordered_map<std::uint64_t, InstanceRecord> instances_;
};

PieDriver* create_driver_impl(const PieDriverCreateDesc* desc, PieDriverCaps* caps) {
    std::memset(caps, 0, sizeof(*caps));
    const std::string config_path(
        reinterpret_cast<const char*>(desc->config_bytes.ptr),
        desc->config_bytes.len);
    auto driver = std::make_unique<MetalDriver>(*desc);
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

extern "C" void pie_metal_destroy(PieDriver* driver) {
    delete as_driver(driver);
}
