#include "entry.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.hpp"
#include "pie_native/abi_validation.hpp"
#include "pie_native/launch_view.hpp"
#include "pie_native/ptir_channels.hpp"
#include "ptir/host_interp.hpp"
#include "ptir/descriptor_resolve.hpp"
#include "executor/executor.hpp"
#include "executor/executor_worker.hpp"
#include "decode_abi.hpp"

namespace {

namespace interp = pie_metal_driver::ptir_host;
namespace executor = pie_metal_driver::executor;
namespace raw_metal = pie_metal_driver::raw_metal;

// RS_FLAG_RESET (runtime/engine/src/driver/frame.rs) — bit 0 of
// `rs_slot_flags[m]` marks a fresh recurrent-state slot for this fire.
constexpr std::uint8_t kRsFlagReset = 1;

// Directory the compiled-in default .metal kernel library resolves to
// (driver/metal/CMakeLists.txt), overridable at run time.
#ifndef PIE_METAL_KERNELS_DIR_DEFAULT
#define PIE_METAL_KERNELS_DIR_DEFAULT ""
#endif

std::string metal_kernels_dir() {
    if (const char* env = std::getenv("PIE_METAL_KERNELS_DIR")) return std::string(env);
    return PIE_METAL_KERNELS_DIR_DEFAULT;
}

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

// Phase 3 (review item 1): one member of a launch, deep-copied at preflight so
// the owned worker job can settle it AFTER `pie_metal_launch` has returned.
// Holds no borrowed launch-array pointers — the forward descriptor (if any) is
// a fully-owned copy, and the instance is re-resolved by id on the worker (so a
// close racing the in-flight job is detected, not a use-after-free). The
// terminal cell pointer is the ABI-leased cell (valid until the batch notify).
struct LaunchMember {
    std::uint64_t instance_id = 0;
    std::vector<WriterConsume> consumes;
    bool needs_forward = false;
    int fwd_slot = -1;            // index into LaunchJobData::fwd_descs, or -1
    std::string build_err;       // desc-build failure → poison this member
    PieTerminalCell* terminal_cell = nullptr;
};

// Phase 3 (review item 1): the entire owned payload of one accepted launch,
// enqueued to the executor worker. `pie_metal_launch` deep-copies this under
// the state mutex during preflight, posts it, and returns — the worker later
// runs forward + interp step + settlement + publication from it.
struct LaunchJobData {
    std::vector<LaunchMember> members;
    std::vector<executor::MemberForwardDesc> fwd_descs;  // owned; index == fwd_slot
    PieCompletion completion{};
};

// Read the launch's flat forward-field arrays into the shared reader (same
// type CUDA's entry.cpp builds), so the CSR slicing below reads identically
// across both drivers. Metal Phase 1 does not yet touch the image/audio/mask
// fields; they stay default-empty.
pie_native::LaunchView build_launch_view(const PieLaunchDesc& launch) {
    pie_native::LaunchView view{};
    view.token_ids = pie_native::slice_from_u32(launch.token_ids.ptr, launch.token_ids.len);
    view.position_ids = pie_native::slice_from_u32(launch.position_ids.ptr, launch.position_ids.len);
    view.kv_page_indices =
        pie_native::slice_from_u32(launch.kv_page_indices.ptr, launch.kv_page_indices.len);
    view.kv_page_indptr =
        pie_native::slice_from_u32(launch.kv_page_indptr.ptr, launch.kv_page_indptr.len);
    view.kv_last_page_lens =
        pie_native::slice_from_u32(launch.kv_last_page_lens.ptr, launch.kv_last_page_lens.len);
    view.qo_indptr = pie_native::slice_from_u32(launch.qo_indptr.ptr, launch.qo_indptr.len);
    view.rs_slot_ids = pie_native::slice_from_u32(launch.rs_slot_ids.ptr, launch.rs_slot_ids.len);
    view.rs_slot_flags = pie_native::slice_from_u8(launch.rs_slot_flags.ptr, launch.rs_slot_flags.len);
    view.sampling_indices =
        pie_native::slice_from_u32(launch.sampling_indices.ptr, launch.sampling_indices.len);
    view.sampling_indptr =
        pie_native::slice_from_u32(launch.sampling_indptr.ptr, launch.sampling_indptr.len);
    // WorkingSet page translation (kv_refact.md flattened-table model), CSR
    // partitioned per `instance_ids` (one segment per launch member — same
    // indexing as every other per-member CSR field here). Device-geometry
    // members map their channel-resolved Pages/WSlot ids through this
    // (§Phase 2 review fix A; mirrors CUDA's ptir_dispatch.cu:733-753).
    view.kv_translation =
        pie_native::slice_from_u32(launch.kv_translation.ptr, launch.kv_translation.len);
    view.kv_translation_indptr = pie_native::slice_from_u32(launch.kv_translation_indptr.ptr,
                                                           launch.kv_translation_indptr.len);
    return view;
}

// Slice member `m`'s forward fields out of the batch-wide CSR view (§5.2).
// `member_count` is `members.size()` for THIS launch — every CSR indptr
// slice must carry exactly `member_count + 1` entries when present.
//
// `resolved` (Phase 2, C3): when non-null, the member's token/position/KV-
// page/readout fields come from the descriptor-resolved `FireGeometry`
// INSTEAD of the wire CSR slices — a device-geometry program's wire span is
// an empty placeholder (`pie_native::LaunchView`'s own doc comment); the
// channel-resolved geometry is the only truth for those fields. The
// recurrent-state slot bookkeeping (rs_slot_id/rs_reset) is unrelated to
// device-geometry classification and always comes from the wire.
bool build_member_forward_desc(const pie_native::LaunchView& view, std::size_t m,
                               std::size_t member_count, bool has_linear_attn,
                               const pie_native::ptir::FireGeometry* resolved,
                               executor::MemberForwardDesc& desc, std::string& err) {
    if (resolved != nullptr) {
        desc.token_ids = resolved->token_ids;
        desc.position_ids = resolved->position_ids;
        desc.kv_pages = resolved->kv_page_indices;
        desc.kv_last_page_len =
            resolved->kv_last_page_lens.empty() ? 0 : resolved->kv_last_page_lens.back();
        desc.readout_local_indices = resolved->sampling_indices;
        // Explicit KV write descriptor (Phase 1b review fix B): propagate
        // it through instead of silently dropping it — MetalExecutor::
        // forward is responsible for rejecting it honestly (the paged-KV
        // write kernel has no encoder integration in this build).
        desc.has_write_desc = resolved->has_write_desc;
        desc.w_page = resolved->w_page;
        desc.w_off = resolved->w_off;
        desc.requires_paged = true;
    } else {
        if (view.qo_indptr.size() != member_count + 1) {
            err = "launch is missing qo_indptr for a forward-needing member";
            return false;
        }
        const std::uint32_t* qo = view.qo_indptr.data();
        const std::uint32_t qb = qo[m], qe = qo[m + 1];
        if (qe < qb || qe > view.token_ids.size() || qe > view.position_ids.size()) {
            err = "malformed qo_indptr/token_ids for this member";
            return false;
        }
        desc.token_ids.assign(view.token_ids.data() + qb, view.token_ids.data() + qe);
        desc.position_ids.assign(view.position_ids.data() + qb, view.position_ids.data() + qe);

        if (!view.kv_page_indptr.empty()) {
            if (view.kv_page_indptr.size() != member_count + 1) {
                err = "malformed kv_page_indptr for this launch";
                return false;
            }
            const std::uint32_t* kp = view.kv_page_indptr.data();
            const std::uint32_t pb = kp[m], pe = kp[m + 1];
            if (pe < pb || pe > view.kv_page_indices.size()) {
                err = "malformed kv_page_indices for this member";
                return false;
            }
            desc.kv_pages.assign(view.kv_page_indices.data() + pb, view.kv_page_indices.data() + pe);
            if (view.kv_last_page_lens.size() == member_count) {
                desc.kv_last_page_len = view.kv_last_page_lens.data()[m];
            }
        }
    }

    desc.has_rs_slot = view.rs_slot_ids.size() == member_count &&
                       view.rs_slot_flags.size() == member_count;
    if (has_linear_attn && !desc.has_rs_slot) {
        err = "missing recurrent-state slot assignment for a hybrid-attention model";
        return false;
    }
    if (desc.has_rs_slot) {
        desc.rs_slot_id = view.rs_slot_ids.data()[m];
        desc.rs_reset = (view.rs_slot_flags.data()[m] & kRsFlagReset) != 0;
    }

    if (resolved == nullptr && !view.sampling_indptr.empty()) {
        if (view.sampling_indptr.size() != member_count + 1) {
            err = "malformed sampling_indptr for this launch";
            return false;
        }
        const std::uint32_t* sp = view.sampling_indptr.data();
        const std::uint32_t sb = sp[m], se = sp[m + 1];
        if (se < sb || se > view.sampling_indices.size()) {
            err = "malformed sampling_indices for this member";
            return false;
        }
        desc.readout_local_indices.assign(view.sampling_indices.data() + sb,
                                          view.sampling_indices.data() + se);
    }
    return true;
}

struct ModelFacts {
    std::uint32_t vocab_size = 32000;
    std::uint32_t max_model_len = 8192;
    std::string arch_name = "llama";
    bool has_linear_attn = false;
};

// Phase 1a (metal_ptir_plan.md §5.4, §12 "Caps honesty"): the Metal forward
// is ONE resident linear-sequence RawMetalDecoder — a fixed
// `max_ctx_ = 4096` KV/GDN ring (decoder.hpp), not the runtime's
// multi-tenant paged pool. Shared between `build_caps_json` (what we
// ADVERTISE) and the Phase 2 descriptor resolver's `validate_fire_geometry`
// page-range check (what we ENFORCE) so both always agree.
constexpr std::uint32_t kMetalPhase1aMaxCtxTokens = 4096;

std::uint32_t effective_total_pages(const pie_metal_driver::Config& cfg, bool rs_cache_required) {
    const std::uint32_t kv_page_size = std::max<std::uint32_t>(1u, cfg.batching.kv_page_size);
    // Ceil-divide: the ring must hold kMetalPhase1aMaxCtxTokens tokens even
    // when kv_page_size does not divide it evenly (a floor division would
    // under-report by up to one page).
    return rs_cache_required ? (kMetalPhase1aMaxCtxTokens + kv_page_size - 1) / kv_page_size
                             : cfg.batching.total_pages;
}

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
    // Reporting the generic multi-request config values here would let the
    // scheduler build launches this driver structurally cannot honor, so
    // cap them truthfully whenever the checkpoint needs the forward
    // (GDN-hybrid, i.e. this increment's qwen3.6 target) — capping only
    // ever narrows a config/model limit down to what the ring can hold,
    // never inflates a limit that was already smaller.
    //
    // The paged DAG is allocated/bound for exactly these bounded capacities.
    // Do not echo arbitrary config values: doing so would advertise IO/scratch
    // rows or GDN slots that do not exist in the resident decoder.
    constexpr std::uint32_t kMetalPagedMaxForwardRequests =
        executor::kPagedMaxForwardRequests;
    constexpr std::uint32_t kMetalPagedMaxForwardTokens =
        executor::kPagedMaxForwardTokens;
    // Phase 1b: the GDN recurrent-state region is genuinely sized for
    // `raw_metal::executor::kPhase1bRsSlots` addressable slots (heap_layout.hpp
    // plan_heap sizes State region as `max_slots * per_slot_bytes`, and
    // MetalExecutor::setup() sets `DecodeGeometry.max_slots` to exactly this
    // constant) — copy_state can genuinely address any of them. This does
    // NOT relax max_forward_requests: Phase 1b still runs one forward
    // request synchronously; the extra slots exist purely as addressable
    // copy_state destinations/sources (e.g. warm-starting/branching a
    // resident sequence's state), not concurrent forward execution.
    const std::uint32_t rs_cache_slots =
        rs_cache_required ? executor::kPhase1bRsSlots : 0u;
    // Static per-slot byte formula mirrors RawMetalDecoder::rs_slot_bytes()
    // exactly (conv_state + conv_state_out + recurrent_state per GDN layer),
    // computed here from the shipped qwen3.6 DecodeGeometry{} defaults
    // directly since no live executor/decoder exists yet at capabilities-
    // build time (mirrors how vocab_size is cross-checked without a live
    // decoder).
    std::uint32_t rs_cache_slot_bytes = 0u;
    if (rs_cache_required) {
        const raw_metal::DecodeGeometry g{};
        const std::uint64_t conv_stride = g.gdn_conv_stride_bytes();
        const std::uint64_t recur_stride = g.gdn_recurrent_stride_bytes();
        int gdn_layers = 0;
        for (int l = 0; l < g.n_layers; ++l) {
            if (!raw_metal::DecodeGeometry::is_full_attn(l)) ++gdn_layers;
        }
        rs_cache_slot_bytes = static_cast<std::uint32_t>(
            std::uint64_t(gdn_layers) * (2 * conv_stride + recur_stride));
    }
    const std::uint32_t max_forward_requests =
        rs_cache_required ? std::min(cfg.batching.max_forward_requests,
                                     kMetalPagedMaxForwardRequests)
                          : cfg.batching.max_forward_requests;
    const std::uint32_t max_forward_tokens =
        rs_cache_required
            ? std::min(cfg.batching.max_forward_tokens, kMetalPagedMaxForwardTokens)
            : cfg.batching.max_forward_tokens;
    const std::uint32_t max_model_len =
        rs_cache_required ? std::min(facts.max_model_len, kMetalPhase1aMaxCtxTokens)
                          : facts.max_model_len;
    const std::uint32_t total_pages = effective_total_pages(cfg, rs_cache_required);
    nlohmann::json caps = {
        {"abi_version", PIE_DRIVER_ABI_VERSION},
        {"total_pages", total_pages},
        {"kv_page_size", cfg.batching.kv_page_size},
        {"swap_pool_size", cfg.batching.cpu_pages},
        {"rs_cache_required", rs_cache_required},
        {"rs_cache_slots", rs_cache_slots},
        {"rs_cache_slot_bytes", rs_cache_slot_bytes},
        {"max_forward_tokens", max_forward_tokens},
        {"max_forward_requests", max_forward_requests},
        {"max_page_refs", rs_cache_required
                              ? total_pages * max_forward_requests
                              : total_pages},
        {"arch_name", facts.arch_name},
        {"vocab_size", facts.vocab_size},
        {"max_model_len", max_model_len},
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
    // Tear the RawMetalDecoder (Metal device/heap/PSO objects) down ON the
    // worker thread that created + exclusively drove them (Phase 3, §7 thread-
    // affinity). `worker_.run` drains behind any still-queued jobs first (FIFO),
    // so all in-flight launches/control ops settle before the executor is freed;
    // both the null-check and the reset run on the worker so `executor_` is
    // never touched off that thread. The destructor body runs before member
    // destructors, so `executor_` is already null when its unique_ptr member is
    // destroyed and `worker_` is still alive here.
    ~MetalDriver() {
        worker_.run([this] {
            if (executor_ != nullptr) executor_.reset();
        });
    }

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
        std::lock_guard<std::mutex> lock(state_mutex_);
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
        std::lock_guard<std::mutex> lock(state_mutex_);
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
        std::lock_guard<std::mutex> lock(state_mutex_);
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

    // ABI v2 launch (Phase 3, review item 1 — ASYNC). SYNCHRONOUS PREFLIGHT
    // under `state_mutex_`: validate instance/program ids + static rules, run
    // the §4.3 writer-ring pull + availability check (a missing put still
    // rejects synchronously with INVALID_ARGUMENT — no epoch, no poison), and
    // RESERVE the consumed inputs. Then deep-copy the accepted batch into an
    // owned `LaunchJobData` (each forward member's descriptor is a fully-owned
    // copy; no launch-array pointer is retained), POST it to the executor
    // worker, and RETURN — `pie_metal_launch` never waits for the GPU forward
    // or settlement. The worker (`run_launch_job`) runs forward + interp step +
    // §4.4 publication (channel words → terminals → per-channel notifies → the
    // batch notify, exactly once) off the caller thread. Non-forward
    // (channel-plane C1) members settle the same way, just without a forward.
    int launch(const PieLaunchDesc& launch, PieCompletion completion) {
        std::unique_lock<std::mutex> lock_holder(state_mutex_);
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
        // Phase 2 (C3): at most one device-geometry program per launch batch
        // — the same structural constraint the runtime's scheduler already
        // upholds (metal_ptir_plan.md §6); a defensive re-check here so a
        // scheduling bug fails the launch loudly instead of resolving two
        // programs' geometry against one shared forward.
        {
            std::size_t device_geometry_count = 0;
            for (const InstanceRecord* member : members) {
                const ProgramRecord& program = programs_.at(member->program_id);
                if (interp::is_device_geometry_trace(program.plan.trace)) ++device_geometry_count;
            }
            if (device_geometry_count > 1) {
                std::cerr << "[pie-driver-metal] launch: " << device_geometry_count
                          << " device-geometry programs in one batch (at most one is supported)\n";
                return PIE_STATUS_INVALID_ARGUMENT;
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

        // Phase 3 (review item 1): preflight is done — deep-copy the accepted
        // launch into an OWNED job and POST it to the executor worker, then
        // return WITHOUT waiting. The worker runs the (possibly multi-ms) GPU
        // forward + interp step + settlement + publication; `pie_metal_launch`
        // must not block the engine's scheduler thread on any of that. Every
        // per-member forward descriptor is a fully-owned copy built here (under
        // the state mutex, instances alive), so the job borrows no launch-array
        // pointer after this returns. Instances are re-resolved by id on the
        // worker so a close racing the in-flight job is handled, not UAF'd.
        const pie_native::LaunchView view = build_launch_view(launch);
        auto job = std::make_shared<LaunchJobData>();
        job->completion = completion;
        job->members.resize(members.size());
        for (std::size_t m = 0; m < members.size(); ++m) {
            LaunchMember& lm = job->members[m];
            lm.instance_id = members[m]->instance_id;
            lm.consumes = std::move(consumes[m]);
            lm.terminal_cell = launch.terminal_cells.ptr[m];
            const ProgramRecord& program = programs_.at(members[m]->program_id);
            if (!program.plan.needs_forward()) continue;
            lm.needs_forward = true;
            executor::MemberForwardDesc desc;
            std::string build_err;
            if (build_forward_desc_for_member(view, m, members.size(), *members[m], program, desc,
                                              build_err)) {
                lm.fwd_slot = static_cast<int>(job->fwd_descs.size());
                job->fwd_descs.push_back(std::move(desc));
            } else {
                lm.build_err = build_err;  // poisoned individually; not in the batch
            }
        }
        lock_holder.unlock();  // never hold state_mutex_ across the worker post/return
        worker_.post([this, job] { run_launch_job(job); });
        return PIE_STATUS_OK;
    }

    // Phase 3 (review item 1): the OWNED, async settlement of one accepted
    // launch, executed on the executor worker thread. Publishes in the §4.4
    // order (channel words while settling → terminal cells → per-channel
    // notifies → the batch notify exactly once), regardless of any fault. The
    // GPU forward runs WITHOUT the state mutex (so it never blocks a concurrent
    // launch preflight); only the interp step + channel settlement take the
    // mutex. Item 3: exceptions from the forward or a member's settlement are
    // caught and translated to that member's terminal FAILED with the original
    // what() diagnostic — never swallowed, never left pending.
    void run_launch_job(std::shared_ptr<LaunchJobData> job) {
        const std::size_t M = job->members.size();
        std::vector<std::uint32_t> outcomes(M, PIE_TERMINAL_OUTCOME_SUCCESS);
        std::vector<std::pair<std::uint64_t, std::uint64_t>> notifications;

        // ── Phase 1: GPU forward (no mutex; executor is worker-owned) ──
        std::vector<executor::LogitsOut> fwd_outs;
        std::vector<std::uint8_t> fwd_ok;
        std::vector<std::string> fwd_err;
        std::string setup_err;
        bool executor_ready = true;
        bool has_forward = false;
        for (const LaunchMember& lm : job->members) {
            if (lm.needs_forward && lm.build_err.empty()) has_forward = true;
        }
        if (has_forward) {
            try {
                executor_ready = ensure_executor(setup_err);  // inline on the worker
                if (executor_ready) {
                    executor_->forward_batch(job->fwd_descs, fwd_outs, fwd_ok, fwd_err);
                }
            } catch (const std::exception& e) {
                executor_ready = false;
                setup_err = std::string("forward raised: ") + e.what();
            } catch (...) {
                executor_ready = false;
                setup_err = "forward raised: unknown exception";
            }
        }

        // ── Phase 2: interp step + channel settlement (under the state mutex) ──
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            for (std::size_t m = 0; m < M; ++m) {
                LaunchMember& lm = job->members[m];
                auto it = instances_.find(lm.instance_id);
                if (it == instances_.end()) {
                    // Instance closed after acceptance (guest ordering bug): fail
                    // its terminal rather than dereference freed interp state.
                    std::cerr << "[pie-driver-metal] launch: instance " << lm.instance_id
                              << " was closed before its accepted fire settled\n";
                    outcomes[m] = PIE_TERMINAL_OUTCOME_FAILED;
                    continue;
                }
                InstanceRecord& member = it->second;
                const ProgramRecord& program = programs_.at(member.program_id);
                member.fire_seq += 1;
                std::string failure;
                interp::PassInputs pass_in{};
                bool ok = true;
                if (lm.needs_forward) {
                    if (!lm.build_err.empty()) {
                        ok = false;
                        failure = lm.build_err;
                    } else if (!executor_ready) {
                        ok = false;
                        failure = "Metal executor setup failed: " + setup_err;
                    } else {
                        const int si = lm.fwd_slot;
                        if (si < 0 || fwd_ok[static_cast<std::size_t>(si)] == 0) {
                            ok = false;
                            failure = si >= 0 ? fwd_err[static_cast<std::size_t>(si)]
                                              : "forward member was not scheduled";
                        } else {
                            const executor::LogitsOut& lo = fwd_outs[static_cast<std::size_t>(si)];
                            pass_in.logits = lo.data.data();
                            pass_in.rows = lo.rows;
                            pass_in.vocab = lo.vocab;
                            pass_in.mtp_draft_row = -1;  // MtpLogits falls back to row 0
                        }
                    }
                }
                try {
                    if (ok &&
                        !run_member(member, program, lm.consumes, pass_in, notifications, failure)) {
                        ok = false;
                    }
                } catch (const std::exception& e) {
                    ok = false;
                    failure = std::string("settlement raised: ") + e.what();
                } catch (...) {
                    ok = false;
                    failure = "settlement raised: unknown exception";
                }
                if (!ok) {
                    std::cerr << "[pie-driver-metal] instance " << member.instance_id
                              << " launch failed: " << failure << "\n";
                    poison_instance(member, notifications);
                    outcomes[m] = PIE_TERMINAL_OUTCOME_FAILED;
                }
            }
        }

        // ── Phase 3: publication (no mutex — leased terminal cells + notify
        //    callbacks): terminals, per-channel notifies, then the batch notify
        //    exactly once. Always runs, so a fault above still settles here. ──
        for (std::size_t m = 0; m < M; ++m) {
            publish_terminal(job->members[m].terminal_cell, outcomes[m]);
        }
        for (const auto& [wait_id, epoch] : notifications) notify(wait_id, epoch);
        notify(job->completion.wait_id, job->completion.target_epoch);
    }

    // Phase 1b/3 paged-KV bridge: real, page-addressable KV pool copy — see
    // RawMetalDecoder::copy_kv_pages/copy_kv_cells (memcpy over the SEPARATE
    // Shared-storage NHD standalone pool — genuinely page-addressable,
    // sized to exactly what caps advertises, see ensure_executor). Only the
    // Metal-resident domain is supported (no host-pinned swap pool exists
    // in this build) — a cross-domain request is honestly UNSUPPORTED
    // rather than silently misinterpreted.
    // Phase 3 (review items 1/2): control ops run their executor-touching body
    // on the worker (behind any queued launches, FIFO), so they never race an
    // in-flight forward and all RawMetalDecoder use stays on one thread. The
    // thin public wrapper just forwards to the worker and returns its status;
    // `worker_.run` rethrows a job exception, which the extern "C" wrapper maps
    // to DRIVER_ERROR.
    int copy_kv(const PieKvCopyDesc& copy, PieCompletion completion) {
        int status = PIE_STATUS_OK;
        worker_.run([&] { status = copy_kv_impl(copy, completion); });
        return status;
    }

    int copy_kv_impl(const PieKvCopyDesc& copy, PieCompletion completion) {
        if (!facts_.has_linear_attn) {
            // Non-hybrid architectures are out of scope for this Metal
            // increment entirely (MetalExecutor::setup refuses them) — no
            // executor, no pool, nothing to copy.
            std::cerr << "[pie-driver-metal] copy_kv: UNSUPPORTED — this increment only "
                         "supports the qwen3.6 (GDN-hybrid) checkpoint geometry\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        if (copy.src_domain != PIE_MEMORY_DOMAIN_METAL_SHARED ||
            copy.dst_domain != PIE_MEMORY_DOMAIN_METAL_SHARED) {
            std::cerr << "[pie-driver-metal] copy_kv: UNSUPPORTED — only same-domain "
                         "(PIE_MEMORY_DOMAIN_METAL_SHARED) copies are supported; there is no "
                         "host-pinned swap pool in this build\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        std::string err;
        if (!ensure_executor(err)) {
            std::cerr << "[pie-driver-metal] copy_kv: " << err << "\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        if (executor_->kv_pool_total_pages() == 0) {
            std::cerr << "[pie-driver-metal] copy_kv: UNSUPPORTED — no paged KV pool is "
                         "allocated (config total_pages/kv_page_size produced a zero-sized "
                         "pool, or allocation failed at executor setup; see prior stderr)\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        if (copy.src_page_ids.len != copy.dst_page_ids.len) {
            std::cerr << "[pie-driver-metal] copy_kv: src/dst page id count mismatch\n";
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        if (copy.src_page_ids.len > 0) {
            const std::vector<std::uint32_t> src(copy.src_page_ids.ptr,
                                                  copy.src_page_ids.ptr + copy.src_page_ids.len);
            const std::vector<std::uint32_t> dst(copy.dst_page_ids.ptr,
                                                  copy.dst_page_ids.ptr + copy.dst_page_ids.len);
            std::string copy_err;
            if (!executor_->copy_kv_pages(src, dst, &copy_err)) {
                std::cerr << "[pie-driver-metal] copy_kv: " << copy_err << "\n";
                return PIE_STATUS_DRIVER_ERROR;
            }
        }
        if (copy.cells.len > 0) {
            std::vector<executor::MetalExecutor::KvMoveCell> cells;
            cells.reserve(copy.cells.len);
            for (std::size_t i = 0; i < copy.cells.len; ++i) {
                const PieKvMoveCell& c = copy.cells.ptr[i];
                cells.push_back({c.dst_page_id, c.dst_token_offset, c.src_page_id,
                                c.src_token_offset});
            }
            std::string cell_err;
            if (!executor_->copy_kv_cells(cells, &cell_err)) {
                std::cerr << "[pie-driver-metal] copy_kv: " << cell_err << "\n";
                return PIE_STATUS_DRIVER_ERROR;
            }
        }
        publish_terminal(completion.terminal_cell, PIE_TERMINAL_OUTCOME_SUCCESS);
        notify(completion.wait_id, completion.target_epoch);
        return PIE_STATUS_OK;
    }

    int copy_state(const PieStateCopyDesc& copy, PieCompletion completion) {
        int status = PIE_STATUS_OK;
        worker_.run([&] { status = copy_state_impl(copy, completion); });
        return status;
    }

    int copy_state_impl(const PieStateCopyDesc& copy, PieCompletion completion) {
        if (!facts_.has_linear_attn) {
            std::cerr << "[pie-driver-metal] copy_state: UNSUPPORTED — this checkpoint has "
                         "no GDN/linear-attention layers, so there is no recurrent state to "
                         "copy\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        std::string err;
        if (!ensure_executor(err)) {
            std::cerr << "[pie-driver-metal] copy_state: " << err << "\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        const std::uint32_t rs_slots = executor_->rs_slots();
        for (std::size_t i = 0; i < copy.slot_ranges.len; ++i) {
            const PieStateCopyRange& range = copy.slot_ranges.ptr[i];
            if (range.src_slot_id >= rs_slots || range.dst_slot_id >= rs_slots) {
                std::cerr << "[pie-driver-metal] copy_state: slot id out of range [0, "
                          << rs_slots << ")\n";
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
        for (std::size_t i = 0; i < copy.slot_ranges.len; ++i) {
            const PieStateCopyRange& range = copy.slot_ranges.ptr[i];
            std::string copy_err;
            if (!executor_->copy_state(range.src_slot_id, range.dst_slot_id, &copy_err)) {
                std::cerr << "[pie-driver-metal] copy_state: " << copy_err << "\n";
                return PIE_STATUS_DRIVER_ERROR;
            }
        }
        publish_terminal(completion.terminal_cell, PIE_TERMINAL_OUTCOME_SUCCESS);
        notify(completion.wait_id, completion.target_epoch);
        return PIE_STATUS_OK;
    }

    int resize_pool(const PiePoolResizeDesc& resize, PieCompletion completion) {
        int status = PIE_STATUS_OK;
        worker_.run([&] { status = resize_pool_impl(resize, completion); });
        return status;
    }

    int resize_pool_impl(const PiePoolResizeDesc& resize, PieCompletion completion) {
        if (resize.pool_id != 0) {
            std::cerr << "[pie-driver-metal] resize_pool: UNSUPPORTED — only pool_id 0 (the "
                         "one paged KV pool this bridge creates) exists\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        if (!facts_.has_linear_attn) {
            std::cerr << "[pie-driver-metal] resize_pool: UNSUPPORTED — this increment only "
                         "supports the qwen3.6 (GDN-hybrid) checkpoint geometry\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        std::string err;
        if (!ensure_executor(err)) {
            std::cerr << "[pie-driver-metal] resize_pool: " << err << "\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        const std::uint32_t current_pages = executor_->kv_pool_total_pages();
        if (current_pages == 0) {
            std::cerr << "[pie-driver-metal] resize_pool: UNSUPPORTED — no paged KV pool is "
                         "allocated (config total_pages/kv_page_size produced a zero-sized "
                         "pool, or allocation failed at executor setup; see prior stderr)\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        if (resize.target_pages > std::numeric_limits<std::uint32_t>::max()) {
            std::cerr << "[pie-driver-metal] resize_pool: target_pages exceeds u32 range\n";
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        const auto target_pages = static_cast<std::uint32_t>(resize.target_pages);
        // A shrink is only accepted if the caller's `unmap_ranges` fully
        // covers every page in [target_pages, current_pages) — this driver
        // has no independent way to know which physical pages the runtime
        // still considers live, so it never silently truncates without
        // that explicit attestation (never "reject if REALLY needed"; a
        // partial attestation is rejected too, precisely).
        bool unmapped_tail_pages = true;
        if (target_pages < current_pages) {
            std::vector<bool> unmapped(current_pages - target_pages, false);
            for (std::size_t i = 0; i < resize.unmap_ranges.len; ++i) {
                const PiePoolRange& r = resize.unmap_ranges.ptr[i];
                for (std::uint64_t p = r.page_index; p < r.page_index + r.page_count; ++p) {
                    if (p >= target_pages && p < current_pages) unmapped[p - target_pages] = true;
                }
            }
            unmapped_tail_pages =
                std::all_of(unmapped.begin(), unmapped.end(), [](bool b) { return b; });
        }
        std::string resize_err;
        if (!executor_->resize_kv_pool(target_pages, unmapped_tail_pages, &resize_err)) {
            std::cerr << "[pie-driver-metal] resize_pool: " << resize_err << "\n";
            return target_pages < current_pages && !unmapped_tail_pages
                       ? PIE_STATUS_UNSUPPORTED
                       : PIE_STATUS_DRIVER_ERROR;
        }
        publish_terminal(completion.terminal_cell, PIE_TERMINAL_OUTCOME_SUCCESS);
        notify(completion.wait_id, completion.target_epoch);
        return PIE_STATUS_OK;
    }

    // Phase 3 (review items 1/2): a close runs on the worker BEHIND any queued
    // launches (FIFO), so `close_sequence` (executor state) runs on the owning
    // thread and can never race an in-flight forward or its settlement. The
    // map mutation takes the state mutex (shared with launch preflight).
    int close_instance(std::uint64_t instance_id) {
        int status = PIE_STATUS_OK;
        worker_.run([&] { status = close_instance_impl(instance_id); });
        return status;
    }

    int close_instance_impl(std::uint64_t instance_id) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto it = instances_.find(instance_id);
        if (it == instances_.end()) return PIE_STATUS_CLOSED;
        for (const std::uint64_t channel_id : it->second.channel_ids) {
            auto channel = channels_.find(channel_id);
            if (channel != channels_.end()) {
                channel->second.attachments.erase(instance_id);
            }
        }
        // Release Metal executor residency BEFORE erasing the instance so a
        // later, different sequence's fresh fire is not rejected as
        // "another sequence is resident" (executor::MetalExecutor
        // `validate_linear_sequence_geometry`; a no-op if this instance
        // never ran a forward, or a different sequence is resident).
        if (executor_ != nullptr) executor_->close_sequence(instance_id);
        instances_.erase(it);
        return PIE_STATUS_OK;
    }

    int close_channel(std::uint64_t channel_id) {
        int status = PIE_STATUS_OK;
        worker_.run([&] { status = close_channel_impl(channel_id); });
        return status;
    }

    int close_channel_impl(std::uint64_t channel_id) {
        std::lock_guard<std::mutex> lock(state_mutex_);
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

    // C2 (forward-needing) members only: lazily creates the executor on the
    // first forward-needing launch (mirrors CUDA's lazy `ptir_dispatch`,
    // §5.1). For a device-geometry (C3) program, resolves its descriptor-
    // port channels into a `FireGeometry` BEFORE the forward (Phase 2,
    // W1.1) and feeds that instead of the wire CSR fields; a not-ready
    // descriptor channel fails the fire here (no dummy-run) — the caller
    // poisons only this member. Otherwise slices the wire fields as before
    // (C2, Phase 1). Either way, runs the forward and binds `pass_in` to
    // the materialized logits. `logits_out` must outlive `pass_in`'s use.
    // Lazily create (and one-time-`setup()`) the MetalExecutor on demand —
    // shared by the forward path and the Phase 1b control ops (copy_state
    // et al.) which also need a live RawMetalDecoder to operate on, even if
    // no forward has run yet in this process.
    bool ensure_executor(std::string& err) {
        if (executor_ != nullptr) return true;
        executor::SetupConfig setup_cfg;
        setup_cfg.checkpoint_dir = cfg_.model.hf_path;
        setup_cfg.kernels_dir = metal_kernels_dir();
        setup_cfg.arch_name = facts_.arch_name;
        setup_cfg.vocab_size = facts_.vocab_size;
        setup_cfg.has_linear_attn = facts_.has_linear_attn;
        // Phase 1b/3 paged-KV bridge: size the REAL pool to exactly what
        // caps advertises (effective_total_pages — the same capped value
        // build_caps_json reports), so copy_kv/resize_pool never operate
        // over a pool bigger or smaller than what the driver claims to have.
        setup_cfg.total_pages = effective_total_pages(cfg_, facts_.has_linear_attn);
        setup_cfg.kv_page_size = cfg_.batching.kv_page_size;
        setup_cfg.max_forward_tokens = cfg_.batching.max_forward_tokens;
        setup_cfg.max_forward_requests = cfg_.batching.max_forward_requests;
        // Create + `setup()` the executor ON THE WORKER THREAD (Phase 3, §7):
        // RawMetalDecoder::setup builds the Metal device/heap/PSOs, which must
        // be created on the same thread that will later drive them.
        bool ok = false;
        std::string setup_err;
        std::unique_ptr<executor::MetalExecutor> candidate;
        worker_.run([&] {
            candidate = std::make_unique<executor::MetalExecutor>();
            ok = candidate->setup(setup_cfg, &setup_err);
        });
        if (!ok) {
            err = "Metal executor setup failed: " + setup_err;
            return false;
        }
        executor_ = std::move(candidate);
        return true;
    }

    // C2 (forward-needing) members only: builds the executor forward
    // descriptor for member `m`. For a device-geometry (C3) program, resolves
    // its descriptor-port channels into a `FireGeometry` BEFORE the forward
    // (Phase 2, W1.1) and feeds that instead of the wire CSR fields; a
    // not-ready descriptor channel fails the fire here (no dummy-run) — the
    // caller poisons only this member. Otherwise slices the wire fields as
    // before (C2, Phase 1). The actual forward is run for the WHOLE batch at
    // once by `forward_batch` (Phase 3, §7); this only prepares one member's
    // descriptor + arbitrates its device geometry.
    bool build_forward_desc_for_member(const pie_native::LaunchView& view, std::size_t m,
                                       std::size_t member_count, InstanceRecord& member,
                                       const ProgramRecord& program,
                                       executor::MemberForwardDesc& desc, std::string& failure) {
        desc.sequence_id = member.instance_id;

        pie_native::ptir::FireGeometry resolved;
        const bool device_geometry = interp::is_device_geometry_trace(program.plan.trace);
        if (device_geometry) {
            if (!interp::resolve_fire_geometry(program.plan.trace, member.interp,
                                              cfg_.batching.kv_page_size, resolved, &failure)) {
                return false;
            }
            // WorkingSet page translation (Phase 2 review fix A): apply
            // BEFORE validation, so validate_fire_geometry checks the
            // PHYSICAL page ids the forward will actually use.
            if (view.kv_translation_indptr.size() == member_count + 1) {
                const std::uint32_t lo = view.kv_translation_indptr.data()[m];
                const std::uint32_t hi = view.kv_translation_indptr.data()[m + 1];
                if (hi > lo && hi <= view.kv_translation.size()) {
                    interp::translate_kv_pages(view.kv_translation.data() + lo, hi - lo, resolved);
                }
                // hi <= lo (empty segment): legacy pass-through, ids stay physical.
            }
            const std::uint32_t device_pages =
                effective_total_pages(cfg_, facts_.has_linear_attn);
            if (!pie_native::ptir::validate_fire_geometry(
                    resolved, device_pages, cfg_.batching.kv_page_size, &failure)) {
                return false;
            }
        }
        return build_member_forward_desc(view, m, member_count, facts_.has_linear_attn,
                                         device_geometry ? &resolved : nullptr, desc, failure);
    }

    // One member's fire: interp step, then §4.4 word publication — consumed
    // writer heads (release + writer wake), produced reader cells + tails
    // (release + reader wake). Returns false with `failure` set on any fault;
    // the caller poisons. `pass_in` is empty for C1 (channel-plane-only)
    // members; C2 members carry the forward's materialized logits (§5.3).
    bool run_member(InstanceRecord& member, const ProgramRecord& program,
                    const std::vector<WriterConsume>& member_consumes,
                    const interp::PassInputs& pass_in,
                    std::vector<std::pair<std::uint64_t, std::uint64_t>>& notifications,
                    std::string& failure) {
        const interp::StepResult report = interp::step(member.interp, program.plan, pass_in);
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
    // Phase 3 (§7, D4): coherent single mutex over ALL driver state (programs/
    // instances/channels + the executor pointer). Every ABI entry point locks
    // it, so registry/channel/instance mutation is race-free and a close/copy
    // can never race an in-flight forward's settlement. Declared BEFORE the
    // worker + executor so it outlives them at teardown.
    std::mutex state_mutex_;
    // Phase 3 (§7, D4): the single thread that owns every RawMetalDecoder /
    // RawMetalContext touch. Executor setup, forward_batch, and the copy/
    // resize control ops all run here (via `worker_.run`), giving Metal
    // command-queue thread-affinity + FIFO serialization. Declared BEFORE
    // `executor_` so the worker is still alive while `executor_` is destroyed.
    executor::ExecutorWorker worker_;
    // Lazily created on the first forward-needing (C2) launch (§5.1); zero
    // Metal dependencies leak into this translation unit beyond the plain
    // executor::MetalExecutor interface. Only ever touched on `worker_`'s
    // thread once created (see ensure_executor / the executor call sites).
    std::unique_ptr<executor::MetalExecutor> executor_;
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
