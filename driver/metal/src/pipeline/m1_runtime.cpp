#include "pipeline/m1_runtime.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "pie_native/ptir/ptir_abi.h"
#include "observability.hpp"
#include "pipeline/m1_codegen.hpp"

namespace pie::metal::pipeline {

std::string encode_m1_cache_identity(
    std::uint64_t device,
    std::uint64_t signature,
    M1CacheIdentityVersions versions) {
    std::array<std::uint8_t, 22> bytes{};
    auto put_le = [&](std::size_t offset, std::uint64_t value, std::size_t width) {
        for (std::size_t byte = 0; byte < width; ++byte) {
            bytes[offset + byte] =
                static_cast<std::uint8_t>(value >> (byte * 8));
        }
    };
    bytes[0] = 1;  // BackendKind::Metal
    put_le(1, device, sizeof(device));
    put_le(9, versions.compiler, sizeof(versions.compiler));
    put_le(11, signature, sizeof(signature));
    bytes[19] = 0;  // generic row bucket
    bytes[20] = 0;  // lane-count-one generic bucket
    bytes[21] = 0;  // SemanticMode::Exact

    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (std::uint8_t byte : bytes) {
        stream << std::setw(2) << static_cast<unsigned>(byte);
    }
    stream << "-v"
           << std::setw(4) << static_cast<unsigned>(versions.compiler)
           << std::setw(4) << static_cast<unsigned>(versions.region_plan)
           << std::setw(8) << versions.lane_table
           << std::setw(4) << static_cast<unsigned>(versions.emitter);
    return stream.str();
}

namespace {

constexpr std::size_t kMaxStageCacheEntries = 64;
constexpr std::size_t kMaxProgramCacheEntries = 64;
constexpr std::size_t kMaxNegativeEntries = 64;
constexpr std::size_t kMaxRegionsPerStage = 256;
constexpr std::size_t kMaxRegionsPerProgram = 1024;
constexpr std::size_t kMaxScratchBytes = 512u << 20;
constexpr std::uint64_t kNoTicket = ~std::uint64_t{0};
constexpr std::uint32_t kM3ChannelValid = 1u << 0;
constexpr std::uint32_t kM3ChannelNeedsFull = 1u << 1;
constexpr std::uint32_t kM3ChannelNeedsEmpty = 1u << 2;
constexpr std::uint32_t kM3ChannelTake = 1u << 3;
constexpr std::uint32_t kM3ChannelPut = 1u << 4;
constexpr std::uint32_t kM3ChannelRetryIneligible = 1u << 5;

struct DeviceStatus {
    std::uint32_t state = 0;
    std::uint32_t fault = 0;
    std::uint32_t reserved0 = 0;
    std::uint32_t reserved1 = 0;
};

struct DeviceValueDesc {
    std::uint32_t len = 1;
    std::uint32_t rows = 1;
    std::uint32_t last = 1;
    std::uint32_t rank = 0;
    std::uint32_t dtype = 0;
    std::uint32_t dims[4] = {1, 1, 1, 1};
};

struct DeviceOpParams {
    std::uint32_t tag = 0;
    std::uint32_t a0 = 0;
    std::uint32_t a1 = 0;
    std::uint32_t a2 = 0;
    std::uint32_t o0 = 0;
    std::uint32_t o1 = 0;
    std::uint32_t imm = 0;
    std::uint32_t imm2 = 0;
    std::uint32_t imm3 = 0;
    std::uint32_t kind = 0;
    std::uint32_t pred_tag = 0;
    std::uint32_t lit_dtype = 0;
    std::uint32_t lit_bits = 0;
    std::uint32_t channel_slot = 0;
    std::uint32_t intr = 0;
    std::uint32_t sink_bytes = 0;
};

static_assert(sizeof(PtirLaneTableHeader) == 16);
static_assert(sizeof(PtirLaneRecord) == 72);
static_assert(sizeof(PtirLaneChannelSlot) == 32);
static_assert(sizeof(DeviceStatus) == 16);
static_assert(sizeof(DeviceValueDesc) == 36);
static_assert(sizeof(DeviceOpParams) == 64);

std::size_t align_up(std::size_t value, std::size_t alignment = 256) {
    return (value + alignment - 1) / alignment * alignment;
}

SlotHandle subhandle(
    const SlotHandle& base,
    std::size_t offset,
    std::size_t size) {
    return {
        .buffer = base.buffer,
        .contents_ptr =
            static_cast<std::uint8_t*>(base.contents()) + offset,
        .gpu_address = base.gpu_address + offset,
        .offset = base.offset + offset,
        .size = size,
    };
}

SlotHandle external_handle(const SharedStorage& storage) {
    return {
        .buffer = storage.native_buffer,
        .contents_ptr = storage.contents,
        .gpu_address = storage.gpu_address,
        .offset = 0,
        .size = storage.size,
    };
}

std::uint64_t fnv1a64(const std::uint8_t* data, std::size_t size) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= data[index];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

std::string hex64(std::uint64_t value) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << value;
    return stream.str();
}

std::string encode_cache_identity(
    std::uint64_t device,
    std::uint64_t signature) {
    return encode_m1_cache_identity(
        device,
        signature,
        {
            .compiler = PTIR_COMPILER_VERSION,
            .region_plan = PTIR_REGION_PLAN_VERSION,
            .lane_table = PTIR_LANE_TABLE_ABI_VERSION,
            .emitter = kMetalM1EmitterVersion,
        });
}

std::uint32_t symbolic_extent(
    std::uint32_t role,
    const M1RuntimeExtents& extents) {
    switch (role) {
        case PTIR_EXTENT_KV_LEN: return extents.kv_len;
        case PTIR_EXTENT_PAGE_COUNT: return extents.page_count;
        case PTIR_EXTENT_ROW_COUNT: return extents.row_count;
        case PTIR_EXTENT_TOKEN_COUNT: return extents.token_count;
        case PTIR_EXTENT_SAMPLED_ROWS: return extents.sampled_rows;
        case PTIR_EXTENT_QUERY_LEN: return extents.query_len;
        case PTIR_EXTENT_KEY_LEN: return extents.key_len;
        default: return 1;
    }
}

bool describe_value(
    const pie_native::ptir::plan::ValueType& type,
    const M1RuntimeExtents& extents,
    DeviceValueDesc& descriptor) {
    descriptor = {};
    descriptor.dtype = type.dtype;
    descriptor.rank =
        static_cast<std::uint32_t>(std::min<std::size_t>(type.dims.size(), 4));
    std::uint64_t length = 1;
    for (std::uint32_t dimension = 0; dimension < descriptor.rank; ++dimension) {
        const auto& source = type.dims[dimension];
        const std::uint32_t value =
            source.symbolic ? symbolic_extent(source.value, extents)
                            : source.value;
        descriptor.dims[dimension] = value;
        if (value == 0 ||
            length >
                std::numeric_limits<std::uint32_t>::max() /
                    value) {
            return false;
        }
        length *= value;
    }
    descriptor.len = static_cast<std::uint32_t>(length);
    descriptor.rows = 1;
    if (descriptor.rank >= 2) {
        std::uint64_t rows = 1;
        for (std::uint32_t dimension = 0;
             dimension + 1 < descriptor.rank;
             ++dimension) {
            if (rows >
                std::numeric_limits<std::uint32_t>::max() /
                    descriptor.dims[dimension]) {
                return false;
            }
            rows *= descriptor.dims[dimension];
        }
        descriptor.rows = static_cast<std::uint32_t>(rows);
    }
    descriptor.last =
        descriptor.rows == 0 ? 0 : descriptor.len / descriptor.rows;
    return true;
}

std::size_t value_bytes(const DeviceValueDesc& descriptor) {
    return std::max<std::size_t>(
        descriptor.dtype == PTIR_DT_BOOL ? descriptor.len
                                         : descriptor.len * 4,
        4);
}

std::size_t wire_value_bytes(const DeviceValueDesc& descriptor) {
    return std::max<std::size_t>(
        descriptor.dtype == PTIR_DT_BOOL
            ? (descriptor.len + 7) / 8
            : descriptor.len * 4,
        1);
}

std::uint64_t combined_signature(
    const std::vector<pie_native::ptir::plan::StagePlan>& plans) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(plans.size() * sizeof(std::uint64_t));
    for (const auto& plan : plans) {
        for (int byte = 0; byte < 8; ++byte) {
            bytes.push_back(
                static_cast<std::uint8_t>(
                    plan.signature_hash >> (byte * 8)));
        }
    }
    return fnv1a64(bytes.data(), bytes.size());
}

}  // namespace

M1ResolvedShape resolve_m1_shape_for_test(
    const pie_native::ptir::plan::ValueType& type,
    const M1RuntimeExtents& extents) {
    DeviceValueDesc descriptor;
    if (!describe_value(type, extents, descriptor)) return {};
    return {descriptor.len, descriptor.rows, descriptor.last};
}

M1RuntimeExtents m1_extents_from_forward_desc(
    const batch::MemberForwardDesc& desc,
    std::uint32_t sampled_rows) {
    return {
        .kv_len = desc.kv_len,
        .page_count = desc.page_count,
        .row_count = desc.row_count,
        .token_count = desc.token_count,
        .sampled_rows = sampled_rows,
        .query_len = desc.query_len,
        .key_len = desc.key_len,
    };
}

M1RuntimeExtents m3_extents_from_forward_desc(
    const batch::MemberForwardDesc& desc) {
    return m1_extents_from_forward_desc(
        desc,
        static_cast<std::uint32_t>(
            desc.readout_local_indices.size()));
}

M1DeviceInputs m1_singleton_fallback_inputs(
    const batch::LogitsOut& output,
    const batch::MemberForwardDesc& desc,
    int mtp_draft_row) {
    M1DeviceInputs inputs;
    inputs.logits_bf16 = SlotHandle{
        .buffer = output.device_buffer,
        .contents_ptr = output.device_contents,
        .gpu_address = output.device_gpu_address,
        .offset = 0,
        .size = static_cast<std::size_t>(output.device_bytes),
    };
    inputs.logits_row_offset = output.device_row_offset;
    inputs.logits_row_count = output.rows;
    inputs.vocab = output.vocab;
    inputs.mtp_draft_row = mtp_draft_row;
    inputs.extents =
        m1_extents_from_forward_desc(desc, output.rows);
    return inputs;
}

struct M1RegionExecutable {
    M1OpMeta operation;
    Pso pso{};
    int ordinal = -1;
};

struct M2FusedRegionExecutable {
    pie_native::ptir::plan::Region region;
    Pso pso{};
    int ordinal = -1;
};

struct M3GroupedRegionExecutable {
    pie_native::ptir::plan::Region region;
    Pso pso{};
    bool parallel_nucleus = false;
    bool parallel_topk = false;
};

struct M1StageExecutable {
    std::vector<M1RegionExecutable> regions;
    std::vector<M2FusedRegionExecutable> fused_regions;
    std::vector<M3GroupedRegionExecutable> grouped_regions;
    std::vector<M3GroupedRegionExecutable> grouped_singleton_regions;
    bool fused_supported = false;
    bool grouped_supported = false;
    std::string fused_reason;
    std::string grouped_reason;
    std::string cache_identity;
    std::vector<std::uint8_t> canonical_signature;
};

struct M1ProgramStage {
    std::shared_ptr<M1StageExecutable> executable;
    pie_native::ptir::plan::StagePlan plan;
};

struct M1ProgramExecutable {
    std::uint64_t program_hash = 0;
    std::vector<std::uint8_t> canonical_bytes;
    std::vector<M1ProgramStage> stages;
    std::vector<M1ChannelEffect> effects;
    Pso readiness{};
    Pso commit{};
    Pso grouped_readiness{};
    Pso grouped_commit{};
    std::string grouped_reason;
    int readiness_ordinal = -1;
    int commit_ordinal = -1;
    bool requires_m2_placement = false;
};

struct M1PreparedFire {
    std::shared_ptr<M1ProgramExecutable> program;
    std::vector<std::shared_ptr<ChannelState>> channels;
    std::vector<batch::ChannelTicket> tickets;
    std::vector<SlotHandle> committed_cells;
    std::vector<SlotHandle> pending_cells;
    std::vector<SlotHandle> word_handles;
    std::vector<SlotHandle> m1_external_handles;
    SlotHandle status{};
    SlotHandle lane_table{};
    bool resource_accounted = false;
};

struct M2EncodedRegion {
    Pso pso{};
    int ordinal = -1;
    std::array<SlotHandle, 7> fixed{};
    std::vector<SlotHandle> channels;
};

struct M2CommandPlan {
    std::shared_ptr<M1PreparedFire> fire;
    RawMetalContext* target = nullptr;
    SlotHandle scratch{};
    SlotHandle descriptors{};
    SlotHandle parameters{};
    SlotHandle offsets{};
    std::vector<SlotHandle> external_handles;
    std::vector<M2EncodedRegion> pre;
    std::vector<M2EncodedRegion> post;
    int readiness_ordinal = -1;
    int commit_ordinal = -1;
    SlotHandle logits_base{};
    std::uint32_t logits_vocab = 0;
};

struct M3ChannelMeta {
    std::uint64_t words = 0;
    std::uint32_t capacity = 0;
    std::uint32_t flags = 0;
};

struct M3GroupLayout {
    std::uint32_t lane_count = 0;
    std::uint32_t value_count = 0;
    std::uint32_t scratch_stride = 0;
    std::uint32_t temporary_offset = 0;
    std::uint32_t vocab = 0;
    std::uint32_t reserved[3] = {0, 0, 0};
};

struct M3RowMeta {
    std::uint32_t offset = 0;
    std::uint32_t count = 0;
    std::uint32_t mtp_offset = 0;
    std::uint32_t reserved = 0;
};

struct M3EncodedRegion {
    Pso pso{};
    int ordinal = -1;
    std::array<SlotHandle, 11> fixed{};
    Grid grid{};
    Threadgroup threadgroup{};
    bool library = false;
    bool parallel_selection = false;
};

struct M3StageCommand {
    SlotHandle descriptors{};
    SlotHandle parameters{};
    SlotHandle offsets{};
    SlotHandle scratch{};
    SlotHandle layout{};
    SlotHandle bindings{};
    SlotHandle lane_indices{};
    std::vector<M3EncodedRegion> regions;
    std::uint8_t stage = PTIR_STAGE_EPILOGUE;
    bool singleton_fallback = false;
};

struct M3GroupCommand {
    std::vector<M3LaneCandidate> candidates;
    RawMetalContext* target = nullptr;
    SlotHandle lane_table{};
    SlotHandle statuses{};
    SlotHandle channel_meta{};
    SlotHandle pending_flags{};
    SlotHandle row_meta{};
    SlotHandle row_indices{};
    std::vector<M3StageCommand> stages;
    std::vector<SlotHandle> external_handles;
    Pso readiness{};
    Pso commit{};
    int readiness_ordinal = -1;
    int commit_ordinal = -1;
    M3GroupStats stats{};
    void* timestamp_heap = nullptr;
    std::chrono::steady_clock::time_point post_begin{};
};

M1PrepareOutcome check_readiness_host(
    const std::shared_ptr<M1ProgramExecutable>& program,
    const std::vector<std::shared_ptr<ChannelState>>& channels,
    const std::vector<batch::ChannelTicket>& tickets,
    std::string& error) {
    if (!program || channels.size() != program->effects.size() ||
        tickets.size() != channels.size()) {
        error = "Metal M1 fire/channel layout mismatch";
        return M1PrepareOutcome::Failed;
    }
    for (std::size_t channel = 0; channel < channels.size(); ++channel) {
        if (!channels[channel]) {
            error = "Metal M1 fire has a null channel";
            return M1PrepareOutcome::Failed;
        }
        const ChannelState& state = *channels[channel];
        const auto& ticket = tickets[channel];
        const auto& effect = program->effects[channel];
        const std::uint64_t head = state.head();
        const std::uint64_t tail = state.tail();
        if (state.poison() != 0 || state.closed() != 0 || tail < head) {
            error = "Metal M1 host readiness fault " +
                    std::to_string(0x200 + channel);
            return M1PrepareOutcome::Failed;
        }
        if ((ticket.expected_head != kNoTicket &&
             head != ticket.expected_head) ||
            (effect.requires_full && tail <= head) ||
            (effect.requires_empty &&
             tail - head >= effect.capacity)) {
            error = "Metal M1 host readiness retry " +
                    std::to_string(0x300 + channel);
            return M1PrepareOutcome::Retry;
        }
        if (effect.put) {
            const std::uint64_t credit =
                effect.take && effect.requires_full ? 1u : 0u;
            if (ticket.expected_tail == kNoTicket ||
                tail != ticket.expected_tail ||
                tail - head >=
                    static_cast<std::uint64_t>(effect.capacity) + credit) {
                error = "Metal M1 host readiness retry " +
                        std::to_string(0x500 + channel);
                return M1PrepareOutcome::Retry;
            }
        }
    }
    return M1PrepareOutcome::Ready;
}

std::uint8_t m3_schedule_bucket(const M1RuntimeExtents& extents) {
    const std::uint32_t rows =
        std::max(extents.sampled_rows, extents.row_count);
    return rows <= 1
               ? 0
               : static_cast<std::uint8_t>(
                     32 - std::countl_zero(rows - 1));
}

std::string m3_stage_key(
    const M1ProgramStage& stage,
    const M1RuntimeExtents& extents) {
    if (stage.plan.signature.empty()) return {};
    std::string key(
        reinterpret_cast<const char*>(stage.plan.signature.data()),
        stage.plan.signature.size());
    key.push_back(static_cast<char>(m3_schedule_bucket(extents)));
    return key;
}

std::size_t m3_used_channel_slots(
    const pie_native::ptir::plan::StagePlan& stage) {
    std::size_t count = 0;
    for (const auto& normalized : stage.ops) {
        if (normalized.op.chan >= 0) {
            count = std::max(
                count,
                static_cast<std::size_t>(
                    normalized.op.chan) +
                    1);
        }
    }
    return count;
}

std::uint32_t m3_channel_flags(
    const M1ChannelEffect& effect,
    bool retry_ineligible) {
    std::uint32_t flags = kM3ChannelValid;
    if (effect.requires_full) flags |= kM3ChannelNeedsFull;
    if (effect.requires_empty) flags |= kM3ChannelNeedsEmpty;
    if (effect.take) flags |= kM3ChannelTake;
    if (effect.put) flags |= kM3ChannelPut;
    if (retry_ineligible) flags |= kM3ChannelRetryIneligible;
    return flags;
}

static_assert(sizeof(M3ChannelMeta) == 16);
static_assert(sizeof(M3GroupLayout) == 32);
static_assert(sizeof(M3RowMeta) == 16);

void bind_m2_effect(M2CommandPlan& command, int ordinal) {
    command.target->arg_bind_ordinal(
        ordinal, 0, command.fire->status);
    command.target->arg_bind_ordinal(
        ordinal, 1, command.fire->lane_table);
    for (std::size_t channel = 0;
         channel < command.fire->word_handles.size();
         ++channel) {
        command.target->arg_bind_ordinal(
            ordinal,
            static_cast<std::uint8_t>(channel + 2),
            command.fire->word_handles[channel]);
    }
}

void bind_m2_region(
    M2CommandPlan& command,
    const M2EncodedRegion& region) {
    for (std::size_t index = 0; index < region.fixed.size(); ++index) {
        command.target->arg_bind_ordinal(
            region.ordinal,
            static_cast<std::uint8_t>(index),
            region.fixed[index]);
    }
    for (std::size_t index = 0; index < region.channels.size(); ++index) {
        command.target->arg_bind_ordinal(
            region.ordinal,
            static_cast<std::uint8_t>(7 + index),
            region.channels[index]);
    }
}

void bind_m3_effect(M3GroupCommand& command, int ordinal) {
    command.target->arg_bind_ordinal(
        ordinal, 0, command.lane_table);
    command.target->arg_bind_ordinal(
        ordinal, 1, command.channel_meta);
}

void bind_m3_region(
    M3GroupCommand& command,
    const M3EncodedRegion& region) {
    for (std::size_t index = 0; index < region.fixed.size(); ++index) {
        command.target->arg_bind_ordinal(
            region.ordinal,
            static_cast<std::uint8_t>(index),
            region.fixed[index]);
    }
}

class PsoCompileTransaction {
  public:
    PsoCompileTransaction(RawMetalContext& context, int& next_ordinal)
        : context_(context),
          next_ordinal_(next_ordinal),
          initial_ordinal_(next_ordinal) {
        created_.reserve(kMaxRegionsPerProgram * 4 + 4);
    }

    ~PsoCompileTransaction() {
        if (committed_) return;
        for (auto pso = created_.rbegin(); pso != created_.rend(); ++pso) {
            context_.release_pso(*pso);
        }
        next_ordinal_ = initial_ordinal_;
    }

    void track(Pso pso) {
        if (pso.valid()) created_.push_back(pso);
    }

    void commit() { committed_ = true; }

  private:
    RawMetalContext& context_;
    int& next_ordinal_;
    int initial_ordinal_;
    std::vector<Pso> created_;
    bool committed_ = false;
};

struct M1Runtime::Impl {
    struct CompileFault {
        std::string function_substring;
        std::string error;
        std::uint32_t remaining = 0;
    };

    std::unique_ptr<RawMetalContext> context;
    std::filesystem::path cache_dir;
    std::string runtime_template;
    int next_ordinal = 100000;
    std::unordered_map<std::string, std::shared_ptr<M1StageExecutable>>
        stage_cache;
    std::unordered_map<
        std::uint64_t,
        std::shared_ptr<M1ProgramExecutable>>
        programs;
    std::unordered_map<std::uint64_t, std::string> negative;
    M1CacheStats stats;
    M3GroupStats m3_stats;
    Pso grouped_readiness{};
    Pso grouped_commit{};
    std::vector<CompileFault> compile_faults;
    std::size_t max_program_cache_entries = kMaxProgramCacheEntries;

    bool compile_cached(
        const std::string& source,
        const std::string& function,
        const std::filesystem::path& archive,
        Pso& pso,
        std::string& error,
        PsoCompileTransaction& transaction) {
        for (auto& fault : compile_faults) {
            if (fault.remaining != 0 &&
                function.find(fault.function_substring) !=
                    std::string::npos) {
                --fault.remaining;
                error = fault.error;
                pso = {};
                return false;
            }
        }
        std::error_code ec;
        std::filesystem::create_directories(archive.parent_path(), ec);
        if (ec) {
            error = "cannot create Metal PTIR cache directory " +
                    archive.parent_path().string() + ": " + ec.message();
            pso = {};
            return false;
        }
        bool hit = false;
        pso = context->compile_ptir_pso_cached(
            source, function, archive.string(), &hit, &error);
        if (!pso.valid()) return false;
        transaction.track(pso);
        if (hit) ++stats.persistent_hits;
        else ++stats.compilations;
        return true;
    }

    void remember_negative(std::uint64_t hash, const std::string& error) {
        if (negative.size() >= kMaxNegativeEntries) {
            negative.erase(negative.begin());
        }
        negative[hash] = error;
        stats.negative_entries = negative.size();
    }

    void bind_effect_kernel(
        int ordinal,
        const std::shared_ptr<M1PreparedFire>& fire) {
        context->arg_bind_ordinal(ordinal, 0, fire->status);
        context->arg_bind_ordinal(ordinal, 1, fire->lane_table);
        for (std::size_t channel = 0; channel < fire->word_handles.size();
             ++channel) {
            context->arg_bind_ordinal(
                ordinal,
                static_cast<std::uint8_t>(channel + 2),
                fire->word_handles[channel]);
        }
    }
};

std::string default_m1_cache_dir() {
    if (const char* override_path = std::getenv("PIE_METAL_PTIR_CACHE_DIR")) {
        if (*override_path != 0) return override_path;
    }
    if (const char* home = std::getenv("HOME")) {
        return (
            std::filesystem::path(home) / "Library" / "Caches" / "pie" /
            "metal" / "ptir-m1")
            .string();
    }
    return ".pie-metal-ptir-cache";
}

M1Runtime::M1Runtime(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
M1Runtime::~M1Runtime() = default;

std::unique_ptr<M1Runtime> M1Runtime::create(
    const std::string& kernels_dir,
    const std::string& cache_dir,
    std::string& error) {
    auto impl = std::make_unique<Impl>();
    impl->context = RawMetalContext::create(4u << 20);
    if (!impl->context) {
        error = "unable to create Metal M1 context";
        return nullptr;
    }
    impl->context->make_resident();
    impl->cache_dir =
        cache_dir.empty() ? default_m1_cache_dir() : cache_dir;
    const std::filesystem::path runtime_path =
        std::filesystem::path(kernels_dir) / "ptir_m1_runtime.metal";
    if (!read_ptir_msl_source(
            runtime_path.string(), impl->runtime_template, &error) ||
        impl->runtime_template.empty()) {
        return nullptr;
    }
    if (const char* injected =
            std::getenv("PIE_METAL_PTIR_TEST_FAIL_COMPILE_ONCE")) {
        if (*injected != 0) {
            impl->compile_faults.push_back({
                .function_substring = {},
                .error = injected,
                .remaining = 1,
            });
        }
    }
    return std::unique_ptr<M1Runtime>(new M1Runtime(std::move(impl)));
}

std::shared_ptr<M1ProgramExecutable> M1Runtime::compile_program(
    std::uint64_t program_hash,
    const ExecPlan& plan,
    std::string& error,
    std::span<const std::uint8_t> canonical_bytes,
    M1CompileFailureKind* failure_kind) {
    if (failure_kind != nullptr) {
        *failure_kind = M1CompileFailureKind::None;
    }
    if (const auto found = impl_->programs.find(program_hash);
        found != impl_->programs.end()) {
        if (!canonical_bytes.empty() &&
            (found->second->canonical_bytes.empty() ||
             found->second->canonical_bytes.size() !=
                 canonical_bytes.size() ||
             !std::equal(
                 canonical_bytes.begin(),
                 canonical_bytes.end(),
                 found->second->canonical_bytes.begin()))) {
            error = "Metal M1 program hash collision";
            if (failure_kind != nullptr) {
                *failure_kind = M1CompileFailureKind::Deterministic;
            }
            return nullptr;
        }
        ++impl_->stats.memory_hits;
        return found->second;
    }
    if (const auto found = impl_->negative.find(program_hash);
        found != impl_->negative.end()) {
        ++impl_->stats.negative_hits;
        error = found->second;
        if (failure_kind != nullptr) {
            *failure_kind = M1CompileFailureKind::Deterministic;
        }
        return nullptr;
    }
    auto reject_deterministic = [&](const std::string& reason) {
        error = reason;
        if (failure_kind != nullptr) {
            *failure_kind = M1CompileFailureKind::Deterministic;
        }
        impl_->remember_negative(program_hash, reason);
        return std::shared_ptr<M1ProgramExecutable>{};
    };
    auto reject_retryable = [&](const std::string& reason) {
        error = reason;
        if (failure_kind != nullptr) {
            *failure_kind = M1CompileFailureKind::Retryable;
        }
        return std::shared_ptr<M1ProgramExecutable>{};
    };
    if (!plan.executable) {
        return reject_deterministic(
            plan.reject_reason.empty()
                ? "Metal M1 plan is not executable"
                : plan.reject_reason);
    }
    if (plan.bound.version < 2 ||
        (plan.region_plans.empty() && !plan.trace.stages.empty())) {
        return reject_deterministic(
            "Metal M1 requires PTIB v2 with PTRP v4/compiler v3 plans");
    }
    if (plan.region_plans.size() != plan.trace.stages.size()) {
        return reject_deterministic("Metal M1 plan/stage count mismatch");
    }
    bool requires_m2_placement = false;
    if (plan.needs_forward()) {
        for (const auto& stage : plan.region_plans) {
            if (stage.stage == PTIR_STAGE_PROLOGUE) {
                requires_m2_placement = true;
            } else if (stage.stage != PTIR_STAGE_EPILOGUE) {
                return reject_deterministic(
                    "Metal rejects forward-needing per-layer stages");
            }
        }
        std::vector<bool> descriptor_channels(
            plan.trace.channels.size(), false);
        for (const auto& port : plan.trace.ports) {
            if (!port.is_const && port.channel < descriptor_channels.size()) {
                descriptor_channels[port.channel] = true;
            }
        }
        for (const auto& stage : plan.region_plans) {
            if (stage.stage != PTIR_STAGE_PROLOGUE) continue;
            for (const auto& normalized : stage.ops) {
                const auto& op = normalized.op;
                if (op.tag != PTIR_OP_CHAN_PUT || op.chan < 0 ||
                    static_cast<std::size_t>(op.chan) >=
                        stage.channel_bindings.size()) {
                    continue;
                }
                const std::uint32_t dense =
                    stage.channel_bindings[op.chan];
                if (dense < descriptor_channels.size() &&
                    descriptor_channels[dense]) {
                    return reject_deterministic(
                        "Metal M2 cannot consume a prologue pending value "
                        "through a host-resolved descriptor");
                }
            }
        }
    }
    if (plan.trace.channels.size() > kMetalM1MaxChannels) {
        return reject_deterministic(
            "Metal M1 supports at most " +
            std::to_string(kMetalM1MaxChannels) +
            " channel slots per lane");
    }
    if (impl_->programs.size() >= impl_->max_program_cache_entries) {
        return reject_retryable("Metal M1 program executable cache is full");
    }

    auto executable = std::make_shared<M1ProgramExecutable>();
    executable->program_hash = program_hash;
    executable->canonical_bytes.assign(
        canonical_bytes.begin(), canonical_bytes.end());
    executable->requires_m2_placement = requires_m2_placement;
    executable->effects.resize(plan.trace.channels.size());
    for (std::size_t channel = 0; channel < plan.trace.channels.size();
         ++channel) {
        auto& effect = executable->effects[channel];
        effect.capacity = plan.trace.channels[channel].capacity;
        effect.take = plan.takes_channel(static_cast<std::uint32_t>(channel));
        effect.put = plan.puts_channel(static_cast<std::uint32_t>(channel));
        const auto readiness = std::find_if(
            plan.bound.readiness.begin(),
            plan.bound.readiness.end(),
            [channel](const auto& entry) {
                return entry.chan == channel;
            });
        if (readiness != plan.bound.readiness.end()) {
            effect.requires_full =
                readiness->dir ==
                pie_native::ptir::container::Direction::NeedsFull;
            effect.requires_empty =
                readiness->dir ==
                pie_native::ptir::container::Direction::NeedsEmpty;
        }
    }

    std::size_t total_regions = 0;
    std::vector<std::vector<M1OpMeta>> validated_operations;
    validated_operations.reserve(plan.region_plans.size());
    for (const auto& stage_plan : plan.region_plans) {
        for (const std::uint32_t binding : stage_plan.channel_bindings) {
            if (binding >= plan.trace.channels.size()) {
                return reject_deterministic(
                    "Metal M1 stage channel binding is out of range");
            }
        }
        if (stage_plan.singleton.regions.size() > kMaxRegionsPerStage ||
            total_regions + stage_plan.singleton.regions.size() >
                kMaxRegionsPerProgram) {
            return reject_deterministic(
                "Metal M1 singleton executable exceeds the bounded region cache");
        }
        total_regions += stage_plan.singleton.regions.size();
        std::vector<M1OpMeta> operations;
        std::string validation_error;
        if (!validate_singleton_plan(
                stage_plan, operations, validation_error)) {
            return reject_deterministic(validation_error);
        }
        validated_operations.push_back(std::move(operations));
    }

    PsoCompileTransaction transaction(
        *impl_->context, impl_->next_ordinal);
    std::unordered_map<
        std::string,
        std::shared_ptr<M1StageExecutable>>
        pending_stages;

    for (std::size_t stage_index = 0;
         stage_index < plan.region_plans.size();
         ++stage_index) {
        const auto& stage_plan = plan.region_plans[stage_index];
        const auto& operations = validated_operations[stage_index];
        const std::string identity = encode_cache_identity(
            impl_->context->device_cache_id(),
            stage_plan.signature_hash);
        if (const auto found = impl_->stage_cache.find(identity);
            found != impl_->stage_cache.end()) {
            if (found->second->canonical_signature !=
                stage_plan.signature) {
                return reject_deterministic(
                    "Metal M1 stage signature hash collision");
            }
            ++impl_->stats.memory_hits;
            executable->stages.push_back(
                {found->second, stage_plan});
            continue;
        }
        if (const auto found = pending_stages.find(identity);
            found != pending_stages.end()) {
            if (found->second->canonical_signature !=
                stage_plan.signature) {
                return reject_deterministic(
                    "Metal M1 stage signature hash collision");
            }
            ++impl_->stats.memory_hits;
            executable->stages.push_back(
                {found->second, stage_plan});
            continue;
        }
        if (impl_->stage_cache.size() + pending_stages.size() >=
            kMaxStageCacheEntries) {
            return reject_retryable(
                "Metal M1 stage executable cache is full");
        }
        auto stage = std::make_shared<M1StageExecutable>();
        stage->cache_identity = identity;
        stage->canonical_signature = stage_plan.signature;
        const std::filesystem::path directory =
            impl_->cache_dir / identity;
        for (std::size_t region = 0; region < operations.size(); ++region) {
            const std::string function =
                "ptir_m1_" + hex64(stage_plan.signature_hash) + "_r" +
                std::to_string(region);
            const std::string source = emit_singleton_region_msl(
                impl_->runtime_template,
                function,
                operations[region].op.tag);
            Pso pso;
            if (!impl_->compile_cached(
                    source,
                    function,
                    directory / ("region-" + std::to_string(region) + ".mtl4archive"),
                    pso,
                    error,
                    transaction)) {
                return reject_retryable(
                    "Metal M1 compile failed for " +
                    std::string(pie_native::ptir::op_name(
                        static_cast<pie_native::ptir::OpCode>(
                            operations[region].op.tag))) +
                    ": " + error);
            }
            stage->regions.push_back(
                {operations[region], pso, impl_->next_ordinal++});
        }
        stage->fused_supported =
            stage_plan.channel_bindings.size() <=
            kMetalM2MaxFusedChannels;
        if (!stage->fused_supported) {
            stage->fused_reason =
                "fused region exceeds the 12-channel direct-binding limit";
        } else {
            for (std::size_t region = 0;
                 region < stage_plan.fused.regions.size();
                 ++region) {
                const std::string function =
                    "ptir_m2_" + hex64(stage_plan.signature_hash) + "_r" +
                    std::to_string(region);
                std::string source;
                std::string emit_error;
                if (!emit_fused_region_msl(
                        impl_->runtime_template,
                        function,
                        stage_plan,
                        stage_plan.fused.regions[region],
                        source,
                        emit_error)) {
                    stage->fused_supported = false;
                    stage->fused_reason = emit_error;
                    stage->fused_regions.clear();
                    break;
                }
                Pso pso;
                if (!impl_->compile_cached(
                        source,
                        function,
                        directory /
                            ("fused-" + std::to_string(region) +
                             ".mtl4archive"),
                        pso,
                        error,
                        transaction)) {
                    return reject_retryable(
                        "Metal M2 fused compile failed: " + error);
                }
                stage->fused_regions.push_back({
                    stage_plan.fused.regions[region],
                    pso,
                    impl_->next_ordinal++,
                });
            }
        }
        for (std::size_t region = 0;
             region < stage_plan.singleton.regions.size();
             ++region) {
            const std::string function =
                "ptir_m3s_" + hex64(stage_plan.signature_hash) + "_r" +
                std::to_string(region);
            std::string source;
            std::string emit_error;
            if (!emit_grouped_fused_region_msl(
                    impl_->runtime_template,
                    function,
                    stage_plan,
                    stage_plan.singleton.regions[region],
                    source,
                    emit_error)) {
                return reject_deterministic(
                    "Metal M3 grouped singleton emission failed: " +
                    emit_error);
            }
            Pso pso;
            if (!impl_->compile_cached(
                    source,
                    function,
                    directory /
                        ("grouped-singleton-" + std::to_string(region) +
                         ".mtl4archive"),
                    pso,
                    error,
                    transaction)) {
                return reject_retryable(
                    "Metal M3 grouped singleton compile failed: " +
                    error);
            }
            stage->grouped_singleton_regions.push_back({
                stage_plan.singleton.regions[region],
                pso,
            });
        }

        stage->grouped_supported = true;
        for (std::size_t region = 0;
             region < stage_plan.fused.regions.size();
             ++region) {
            const auto& region_plan =
                stage_plan.fused.regions[region];
            const bool parallel_nucleus =
                region_plan.library &&
                region_plan.library_op ==
                    PTIR_LIBRARY_NUCLEUS_SAMPLE;
            const bool parallel_topk =
                region_plan.library &&
                region_plan.library_op == PTIR_LIBRARY_TOP_K &&
                !region_plan.nodes.empty() &&
                stage_plan.ops[region_plan.nodes.front()].op.tag ==
                    PTIR_OP_TOP_K;
            const std::string function =
                "ptir_m3_" + hex64(stage_plan.signature_hash) + "_r" +
                std::to_string(region);
            std::string source;
            std::string emit_error;
            const bool emitted =
                parallel_nucleus
                    ? emit_grouped_nucleus_msl(
                          impl_->runtime_template,
                          function,
                          stage_plan,
                          region_plan,
                          source,
                          emit_error)
                    : parallel_topk
                    ? emit_grouped_topk_msl(
                          impl_->runtime_template,
                          function,
                          stage_plan,
                          region_plan,
                          source,
                          emit_error)
                    : emit_grouped_fused_region_msl(
                          impl_->runtime_template,
                          function,
                          stage_plan,
                          region_plan,
                          source,
                          emit_error);
            if (!emitted) {
                stage->grouped_supported = false;
                stage->grouped_reason = emit_error;
                stage->grouped_regions.clear();
                break;
            }
            Pso pso;
            if (!impl_->compile_cached(
                    source,
                    function,
                    directory /
                        ("grouped-" + std::to_string(region) +
                         ".mtl4archive"),
                    pso,
                    error,
                    transaction)) {
                return reject_retryable(
                    "Metal M3 grouped compile failed: " + error);
            }
            stage->grouped_regions.push_back({
                region_plan,
                pso,
                parallel_nucleus,
                parallel_topk,
            });
        }
        pending_stages.emplace(identity, stage);
        executable->stages.push_back(
            {std::move(stage), stage_plan});
    }
    if (executable->requires_m2_placement) {
        for (const auto& stage : executable->stages) {
            if (!stage.executable->fused_supported) {
                return reject_deterministic(
                    "Metal M2-required stage has no fused executable: " +
                    stage.executable->fused_reason);
            }
        }
    }

    const std::uint64_t effect_signature =
        combined_signature(plan.region_plans);
    const std::string effect_identity = encode_cache_identity(
        impl_->context->device_cache_id(), effect_signature);
    const std::filesystem::path effect_dir =
        impl_->cache_dir / effect_identity;
    const std::string readiness_name =
        "ptir_m1_" + hex64(effect_signature) + "_ready";
    const std::string commit_name =
        "ptir_m1_" + hex64(effect_signature) + "_commit";
    if (!impl_->compile_cached(
            emit_readiness_msl(readiness_name, executable->effects),
            readiness_name,
            effect_dir / "readiness.mtl4archive",
            executable->readiness,
            error,
            transaction)) {
        return reject_retryable(
            "Metal M1 readiness compile failed: " + error);
    }
    if (!impl_->compile_cached(
            emit_commit_msl(commit_name, executable->effects),
            commit_name,
            effect_dir / "commit.mtl4archive",
            executable->commit,
            error,
            transaction)) {
        return reject_retryable(
            "Metal M1 commit compile failed: " + error);
    }
    const std::string grouped_ready_name =
        "ptir_m3_generic_ready_v" +
        std::to_string(kMetalM1EmitterVersion);
    const std::string grouped_commit_name =
        "ptir_m3_generic_commit_v" +
        std::to_string(kMetalM1EmitterVersion);
    const std::string grouped_effect_identity =
        grouped_ready_name + "|" + grouped_commit_name;
    const std::filesystem::path grouped_effect_dir =
        impl_->cache_dir /
        encode_cache_identity(
            impl_->context->device_cache_id(),
            fnv1a64(
                reinterpret_cast<const std::uint8_t*>(
                    grouped_effect_identity.data()),
                grouped_effect_identity.size()));
    Pso grouped_readiness = impl_->grouped_readiness;
    Pso grouped_commit = impl_->grouped_commit;
    if (!grouped_readiness.valid() &&
        !impl_->compile_cached(
            emit_grouped_readiness_msl(grouped_ready_name),
            grouped_ready_name,
            grouped_effect_dir / "readiness.mtl4archive",
            grouped_readiness,
            error,
            transaction)) {
        return reject_retryable(
            "Metal M3 grouped readiness compile failed: " + error);
    }
    if (!grouped_commit.valid() &&
        !impl_->compile_cached(
            emit_grouped_commit_msl(grouped_commit_name),
            grouped_commit_name,
            grouped_effect_dir / "commit.mtl4archive",
            grouped_commit,
            error,
            transaction)) {
        return reject_retryable(
            "Metal M3 grouped commit compile failed: " + error);
    }
    executable->grouped_readiness = grouped_readiness;
    executable->grouped_commit = grouped_commit;
    executable->readiness_ordinal = impl_->next_ordinal++;
    executable->commit_ordinal = impl_->next_ordinal++;

    for (auto& [identity, stage] : pending_stages) {
        impl_->stage_cache.emplace(identity, std::move(stage));
    }
    impl_->grouped_readiness = grouped_readiness;
    impl_->grouped_commit = grouped_commit;
    impl_->programs.emplace(program_hash, executable);
    transaction.commit();
    impl_->stats.stage_entries = impl_->stage_cache.size();
    impl_->stats.program_entries = impl_->programs.size();
    return executable;
}

M1PrepareOutcome M1Runtime::prepare(
    const std::shared_ptr<M1ProgramExecutable>& program,
    const std::vector<std::shared_ptr<ChannelState>>& channels,
    const std::vector<batch::ChannelTicket>& tickets,
    std::shared_ptr<M1PreparedFire>& prepared,
    std::string& error) {
    prepared.reset();
    const M1PrepareOutcome readiness =
        check_readiness_host(program, channels, tickets, error);
    if (readiness != M1PrepareOutcome::Ready) return readiness;
    auto fire = std::make_shared<M1PreparedFire>();
    fire->program = program;
    fire->channels = channels;
    fire->tickets = tickets;
    fire->status =
        impl_->context->acquire_transient_buffer(sizeof(DeviceStatus));
    const std::size_t lane_bytes =
        sizeof(PtirLaneTableHeader) + sizeof(PtirLaneRecord) +
        channels.size() * sizeof(PtirLaneChannelSlot);
    fire->lane_table =
        impl_->context->acquire_transient_buffer(lane_bytes);
    if (!fire->status.valid() || !fire->lane_table.valid()) {
        error = "Metal M1 lane allocation failed";
        release(fire);
        return M1PrepareOutcome::Failed;
    }
    std::memset(fire->status.contents(), 0, fire->status.size);
    std::memset(fire->lane_table.contents(), 0, fire->lane_table.size);

    auto* header =
        static_cast<PtirLaneTableHeader*>(fire->lane_table.contents());
    *header = {
        .abi_version = PTIR_LANE_TABLE_ABI_VERSION,
        .lane_count = 1,
        .channel_slots_per_lane =
            static_cast<std::uint32_t>(channels.size()),
        .flags = 0,
    };
    auto* record = reinterpret_cast<PtirLaneRecord*>(
        static_cast<std::uint8_t*>(fire->lane_table.contents()) +
        sizeof(PtirLaneTableHeader));
    record->channel_slot_offset = 0;
    record->commit_slot = fire->status.gpu_address;
    auto* slots = reinterpret_cast<PtirLaneChannelSlot*>(
        reinterpret_cast<std::uint8_t*>(record) + sizeof(PtirLaneRecord));

    fire->committed_cells.resize(channels.size());
    fire->pending_cells.resize(channels.size());
    fire->word_handles.resize(channels.size());
    for (std::size_t channel = 0; channel < channels.size(); ++channel) {
        const auto& state = *channels[channel];
        const auto& ticket = tickets[channel];
        const std::uint64_t head =
            ticket.expected_head != kNoTicket ? ticket.expected_head
                                              : state.head();
        const std::uint64_t tail =
            ticket.expected_tail != kNoTicket ? ticket.expected_tail
                                              : state.tail();
        const SlotHandle cells = external_handle(state.cells);
        const SlotHandle words = external_handle(state.words);
        fire->committed_cells[channel] = subhandle(
            cells,
            static_cast<std::size_t>(head % state.cap1) * state.cell_bytes,
            state.cell_bytes);
        fire->pending_cells[channel] = subhandle(
            cells,
            static_cast<std::size_t>(tail % state.cap1) * state.cell_bytes,
            state.cell_bytes);
        fire->word_handles[channel] = words;
        impl_->context->use_external_buffer(cells);
        impl_->context->use_external_buffer(words);
        fire->m1_external_handles.push_back(cells);
        fire->m1_external_handles.push_back(words);
        slots[channel] = {
            .committed_cell =
                fire->committed_cells[channel].gpu_address,
            .pending_cell = fire->pending_cells[channel].gpu_address,
            .expected_head = ticket.expected_head,
            .expected_tail = ticket.expected_tail,
        };
    }

    fire->resource_accounted = true;
    m1_prepared_resource_counters().acquire(
        fire->m1_external_handles.size(), 2);
    prepared = std::move(fire);
    return M1PrepareOutcome::Ready;
}

M1ExecuteOutcome M1Runtime::execute(
    const std::shared_ptr<M1PreparedFire>& fire,
    const M1DeviceInputs& inputs,
    std::string& error,
    M1ExecutionMode mode) {
    if (!fire || !fire->program) {
        error = "Metal M1 execute called without a prepared fire";
        return M1ExecuteOutcome::Failed;
    }
    impl_->bind_effect_kernel(
        fire->program->readiness_ordinal, fire);
    impl_->bind_effect_kernel(
        fire->program->commit_ordinal, fire);

    std::size_t region_count = 0;
    std::vector<DeviceValueDesc> descriptors;
    for (const auto& stage : fire->program->stages) {
        region_count += stage.executable->regions.size();
        for (const auto& type : stage.plan.value_types) {
            DeviceValueDesc descriptor;
            if (!describe_value(type, inputs.extents, descriptor)) {
                error = "Metal M1 resolved value shape exceeds u32";
                return M1ExecuteOutcome::Failed;
            }
            descriptors.push_back(descriptor);
        }
    }
    if (descriptors.empty()) descriptors.push_back(DeviceValueDesc{});

    std::vector<std::size_t> value_offsets(descriptors.size(), 0);
    std::size_t scratch_bytes = 256;
    std::size_t max_value_len = 1;
    for (std::size_t value = 0; value < descriptors.size(); ++value) {
        scratch_bytes = align_up(scratch_bytes);
        value_offsets[value] = scratch_bytes;
        scratch_bytes += align_up(value_bytes(descriptors[value]));
        max_value_len =
            std::max<std::size_t>(max_value_len, descriptors[value].len);
    }
    const std::size_t temporary_bytes =
        align_up(max_value_len * sizeof(std::uint32_t) * 4);
    const std::size_t temporary_offset = align_up(scratch_bytes);
    scratch_bytes = temporary_offset + temporary_bytes;
    if (scratch_bytes > kMaxScratchBytes ||
        scratch_bytes < temporary_offset) {
        error = "Metal M1 fire scratch exceeds the 512 MiB bound";
        return M1ExecuteOutcome::Failed;
    }

    SlotHandle scratch =
        impl_->context->acquire_transient_buffer(scratch_bytes);
    SlotHandle descriptor_buffer = impl_->context->acquire_transient_buffer(
        descriptors.size() * sizeof(DeviceValueDesc));
    SlotHandle parameter_buffer = impl_->context->acquire_transient_buffer(
        std::max<std::size_t>(region_count, 1) * sizeof(DeviceOpParams));
    SlotHandle offset_buffer = impl_->context->acquire_transient_buffer(
        descriptors.size() * sizeof(std::uint32_t));
    if (!scratch.valid() || !descriptor_buffer.valid() ||
        !parameter_buffer.valid() || !offset_buffer.valid()) {
        error = "Metal M1 scratch allocation failed";
        if (scratch.valid()) impl_->context->recycle_transient_buffer(scratch);
        if (descriptor_buffer.valid())
            impl_->context->recycle_transient_buffer(descriptor_buffer);
        if (parameter_buffer.valid())
            impl_->context->recycle_transient_buffer(parameter_buffer);
        if (offset_buffer.valid())
            impl_->context->recycle_transient_buffer(offset_buffer);
        return M1ExecuteOutcome::Failed;
    }
    std::memcpy(
        descriptor_buffer.contents(),
        descriptors.data(),
        descriptors.size() * sizeof(DeviceValueDesc));
    auto* parameters =
        static_cast<DeviceOpParams*>(parameter_buffer.contents());
    auto* device_offsets =
        static_cast<std::uint32_t*>(offset_buffer.contents());
    for (std::size_t value = 0; value < value_offsets.size(); ++value) {
        device_offsets[value] =
            static_cast<std::uint32_t>(value_offsets[value]);
    }

    auto value_handle = [&](std::size_t global_value) {
        return subhandle(
            scratch,
            value_offsets[global_value],
            value_bytes(descriptors[global_value]));
    };
    const SlotHandle dummy = subhandle(scratch, 0, 256);
    const SlotHandle temporary = subhandle(
        scratch, temporary_offset, temporary_bytes);
    const bool borrowed_logits = inputs.logits_bf16.valid();
    if (borrowed_logits) {
        impl_->context->use_external_buffer(inputs.logits_bf16);
    }

    auto* lane_record = reinterpret_cast<PtirLaneRecord*>(
        static_cast<std::uint8_t*>(fire->lane_table.contents()) +
        sizeof(PtirLaneTableHeader));
    lane_record->logits_base = inputs.logits_bf16.gpu_address;
    lane_record->logits_row_offset = inputs.logits_row_offset;
    lane_record->logits_row_count = inputs.logits_row_count;
    lane_record->kv_len = inputs.extents.kv_len;
    lane_record->page_count = inputs.extents.page_count;
    lane_record->row_count = inputs.extents.row_count;
    lane_record->token_count = inputs.extents.token_count;
    lane_record->sampled_rows = inputs.extents.sampled_rows;
    lane_record->query_len = inputs.extents.query_len;
    lane_record->key_len = inputs.extents.key_len;

    std::vector<bool> pending(fire->channels.size(), false);
    StepTiming timing;
    std::size_t stage_value_base = 0;
    std::size_t parameter_index = 0;
    std::vector<std::size_t> stage_value_bases;
    std::vector<std::size_t> stage_parameter_bases;
    SlotHandle fused_logits = dummy;
    std::vector<bool> fused_pending(fire->channels.size(), false);
    for (const auto& stage : fire->program->stages) {
        stage_value_bases.push_back(stage_value_base);
        stage_parameter_bases.push_back(parameter_index);
        const SlotHandle stage_descriptors = subhandle(
            descriptor_buffer,
            stage_value_base * sizeof(DeviceValueDesc),
            std::max<std::size_t>(
                stage.plan.value_types.size() * sizeof(DeviceValueDesc),
                sizeof(DeviceValueDesc)));
        for (const auto& region : stage.executable->regions) {
            const auto& op = region.operation.op;
            const std::uint32_t local_result =
                region.operation.result_base;
            DeviceOpParams& params = parameters[parameter_index];
            params.tag = op.tag;
            params.imm =
                op.tag == PTIR_OP_INTRINSIC_VAL
                    ? inputs.vocab
                    : op.imm;
            params.imm2 =
                op.tag == PTIR_OP_INTRINSIC_VAL &&
                        (op.intr == PTIR_INTR_MTP_LOGITS ||
                         op.intr == PTIR_INTR_MTP_DRAFTS) &&
                        inputs.mtp_draft_row >= 0
                    ? static_cast<std::uint32_t>(
                          inputs.mtp_draft_row)
                    : op.imm2;
            params.imm3 = op.imm3;
            params.kind = op.kind;
            params.pred_tag = op.pred_tag;
            params.lit_dtype = op.lit_dtype;
            params.lit_bits = op.lit_bits;
            params.channel_slot =
                op.chan >= 0 ? static_cast<std::uint32_t>(op.chan) : 0;
            params.intr = op.intr;
            params.sink_bytes = 0;
            params.a0 = !op.args.empty() ? op.args[0] : 0;
            params.a1 =
                op.args.size() > 1
                    ? op.args[1]
                    : (op.tag == PTIR_OP_PIVOT_THRESHOLD
                           ? op.pred_payload
                           : 0);
            params.a2 =
                op.args.size() > 2 ? op.args[2] : 0;
            params.o0 = op.results > 0
                            ? local_result
                            : params.a0;
            params.o1 = op.results > 1 ? params.o0 + 1 : params.o0;

            SlotHandle a0 = dummy, a1 = dummy, a2 = dummy;
            SlotHandle o0 = dummy, o1 = dummy;
            if (!op.args.empty())
                a0 = value_handle(stage_value_base + params.a0);
            if (op.args.size() > 1 ||
                op.tag == PTIR_OP_PIVOT_THRESHOLD) {
                a1 = value_handle(stage_value_base + params.a1);
            }
            if (op.args.size() > 2)
                a2 = value_handle(stage_value_base + params.a2);
            if (op.results > 0)
                o0 = value_handle(stage_value_base + params.o0);
            if (op.results > 1)
                o1 = value_handle(stage_value_base + params.o1);

            if (op.tag == PTIR_OP_CHAN_TAKE ||
                op.tag == PTIR_OP_CHAN_READ) {
                if (op.chan < 0 ||
                    static_cast<std::size_t>(op.chan) >=
                        stage.plan.channel_bindings.size()) {
                    error = "Metal M1 channel root binding is invalid";
                    goto cleanup_failure;
                }
                const std::size_t dense =
                    stage.plan.channel_bindings[op.chan];
                a0 = pending[dense] ? fire->pending_cells[dense]
                                    : fire->committed_cells[dense];
            } else if (op.tag == PTIR_OP_CHAN_PUT) {
                if (op.chan < 0 ||
                    static_cast<std::size_t>(op.chan) >=
                        stage.plan.channel_bindings.size()) {
                    error = "Metal M1 channel sink binding is invalid";
                    goto cleanup_failure;
                }
                const std::size_t dense =
                    stage.plan.channel_bindings[op.chan];
                if (wire_value_bytes(
                        descriptors[
                            stage_value_base + params.a0]) >
                    fire->pending_cells[dense].size) {
                    error = "Metal M1 channel sink exceeds fixed cell size";
                    goto cleanup_failure;
                }
                params.sink_bytes = static_cast<std::uint32_t>(
                    fire->pending_cells[dense].size);
                o0 = fire->pending_cells[dense];
                // PTIR channels have register semantics within a pass: a
                // subsequent take/read observes the pending last put.
                pending[dense] = true;
            } else if (op.tag == PTIR_OP_INTRINSIC_VAL) {
                if (!inputs.logits_bf16.valid() ||
                    inputs.logits_row_count == 0 ||
                    inputs.vocab == 0) {
                    error = "Metal M1 logits intrinsic is unbound";
                    goto cleanup_failure;
                }
                const auto& output_descriptor =
                    descriptors[
                        stage_value_base + params.o0];
                const std::uint32_t rows_needed =
                    op.intr == PTIR_INTR_MTP_DRAFTS
                        ? output_descriptor.len
                        : (output_descriptor.len +
                           inputs.vocab - 1) /
                              inputs.vocab;
                const std::uint32_t local_row =
                    (op.intr == PTIR_INTR_MTP_LOGITS ||
                     op.intr == PTIR_INTR_MTP_DRAFTS) &&
                            inputs.mtp_draft_row >= 0
                        ? static_cast<std::uint32_t>(
                              inputs.mtp_draft_row)
                        : 0u;
                if (local_row > inputs.logits_row_count ||
                    rows_needed >
                        inputs.logits_row_count - local_row) {
                    error =
                        "Metal M1 intrinsic row range exceeds bound logits";
                    goto cleanup_failure;
                }
                const std::size_t byte_offset =
                    static_cast<std::size_t>(
                        inputs.logits_row_offset) *
                    inputs.vocab * sizeof(std::uint16_t);
                const std::size_t required_bytes =
                    static_cast<std::size_t>(
                        local_row + rows_needed) *
                    inputs.vocab * sizeof(std::uint16_t);
                if (byte_offset > inputs.logits_bf16.size ||
                    required_bytes >
                        inputs.logits_bf16.size - byte_offset) {
                    error =
                        "Metal M1 intrinsic exceeds logits buffer";
                    goto cleanup_failure;
                }
                a0 = subhandle(
                    inputs.logits_bf16,
                    byte_offset,
                    inputs.logits_bf16.size - byte_offset);
            }

            const SlotHandle params_handle = subhandle(
                parameter_buffer,
                parameter_index * sizeof(DeviceOpParams),
                sizeof(DeviceOpParams));
            impl_->context->arg_bind_ordinal(
                region.ordinal, 0, fire->status);
            impl_->context->arg_bind_ordinal(
                region.ordinal, 1, stage_descriptors);
            impl_->context->arg_bind_ordinal(region.ordinal, 2, a0);
            impl_->context->arg_bind_ordinal(region.ordinal, 3, a1);
            impl_->context->arg_bind_ordinal(region.ordinal, 4, a2);
            impl_->context->arg_bind_ordinal(region.ordinal, 5, o0);
            impl_->context->arg_bind_ordinal(region.ordinal, 6, o1);
            impl_->context->arg_bind_ordinal(
                region.ordinal, 7, temporary);
            impl_->context->arg_bind_ordinal(
                region.ordinal, 8, params_handle);
            ++parameter_index;
        }
        stage_value_base += stage.plan.value_types.size();
    }

    if (inputs.logits_bf16.valid() && inputs.vocab != 0) {
        const std::size_t byte_offset =
            static_cast<std::size_t>(inputs.logits_row_offset) *
            inputs.vocab * sizeof(std::uint16_t);
        fused_logits = subhandle(
            inputs.logits_bf16,
            byte_offset,
            inputs.logits_bf16.size - byte_offset);
    }
    for (std::size_t stage_index = 0;
         stage_index < fire->program->stages.size();
         ++stage_index) {
        const auto& stage = fire->program->stages[stage_index];
        if (!stage.executable->fused_supported) continue;
        const std::size_t value_base = stage_value_bases[stage_index];
        const std::size_t parameter_base =
            stage_parameter_bases[stage_index];
        const SlotHandle stage_descriptors = subhandle(
            descriptor_buffer,
            value_base * sizeof(DeviceValueDesc),
            std::max<std::size_t>(
                stage.plan.value_types.size() * sizeof(DeviceValueDesc),
                sizeof(DeviceValueDesc)));
        const SlotHandle stage_parameters = subhandle(
            parameter_buffer,
            parameter_base * sizeof(DeviceOpParams),
            std::max<std::size_t>(
                stage.plan.ops.size() * sizeof(DeviceOpParams),
                sizeof(DeviceOpParams)));
        const SlotHandle stage_offsets = subhandle(
            offset_buffer,
            value_base * sizeof(std::uint32_t),
            std::max<std::size_t>(
                stage.plan.value_types.size() * sizeof(std::uint32_t),
                sizeof(std::uint32_t)));
        for (const auto& fused : stage.executable->fused_regions) {
            impl_->context->arg_bind_ordinal(
                fused.ordinal, 0, fire->status);
            impl_->context->arg_bind_ordinal(
                fused.ordinal, 1, stage_descriptors);
            impl_->context->arg_bind_ordinal(
                fused.ordinal, 2, stage_parameters);
            impl_->context->arg_bind_ordinal(
                fused.ordinal, 3, stage_offsets);
            impl_->context->arg_bind_ordinal(
                fused.ordinal, 4, scratch);
            impl_->context->arg_bind_ordinal(
                fused.ordinal, 5, temporary);
            impl_->context->arg_bind_ordinal(
                fused.ordinal, 6, fused_logits);
            for (std::size_t local = 0;
                 local < stage.plan.channel_bindings.size();
                 ++local) {
                const std::size_t dense =
                    stage.plan.channel_bindings[local];
                impl_->context->arg_bind_ordinal(
                    fused.ordinal,
                    static_cast<std::uint8_t>(7 + local * 2),
                    fused_pending[dense]
                        ? fire->pending_cells[dense]
                        : fire->committed_cells[dense]);
                impl_->context->arg_bind_ordinal(
                    fused.ordinal,
                    static_cast<std::uint8_t>(8 + local * 2),
                    fire->pending_cells[dense]);
            }
            for (const std::uint32_t node : fused.region.nodes) {
                const auto& op = stage.plan.ops[node].op;
                if (op.tag == PTIR_OP_CHAN_PUT) {
                    const std::size_t dense =
                        stage.plan.channel_bindings[op.chan];
                    fused_pending[dense] = true;
                }
            }
        }
    }

    static_cast<DeviceStatus*>(fire->status.contents())->state = 0;
    timing = impl_->context->run_step([&](StepEncoder& encoder) {
        encoder.set_pso(fire->program->readiness);
        encoder.set_argtable_ordinal(
            fire->program->readiness_ordinal);
        encoder.dispatch(Grid{1, 1, 1}, Threadgroup{1, 1, 1});
        encoder.barrier(BarrierVisibility::Device);
        for (const auto& stage : fire->program->stages) {
            if (mode == M1ExecutionMode::Fused &&
                stage.executable->fused_supported) {
                for (const auto& region :
                     stage.executable->fused_regions) {
                    encoder.set_pso(region.pso);
                    encoder.set_argtable_ordinal(region.ordinal);
                    encoder.dispatch(
                        Grid{1, 1, 1}, Threadgroup{1, 1, 1});
                    encoder.barrier(BarrierVisibility::Device);
                }
            } else {
                for (const auto& region : stage.executable->regions) {
                    encoder.set_pso(region.pso);
                    encoder.set_argtable_ordinal(region.ordinal);
                    encoder.dispatch(
                        Grid{1, 1, 1}, Threadgroup{1, 1, 1});
                    encoder.barrier(BarrierVisibility::Device);
                }
            }
        }
        encoder.set_pso(fire->program->commit);
        encoder.set_argtable_ordinal(fire->program->commit_ordinal);
        encoder.dispatch(Grid{1, 1, 1}, Threadgroup{1, 1, 1});
        });

    {
        const DeviceStatus status =
            *static_cast<const DeviceStatus*>(fire->status.contents());
        impl_->context->recycle_transient_buffer(scratch);
        impl_->context->recycle_transient_buffer(descriptor_buffer);
        impl_->context->recycle_transient_buffer(parameter_buffer);
        impl_->context->recycle_transient_buffer(offset_buffer);
        if (borrowed_logits) {
            impl_->context->release_external_buffer(inputs.logits_bf16);
        }
        if (!timing.succeeded()) {
            error =
                "Metal M1 command timed out before its completion fence";
            return M1ExecuteOutcome::Failed;
        }
        if (status.state == 4) return M1ExecuteOutcome::Committed;
        if (status.state == 2) {
            error =
                "Metal M1 execution readiness retry " +
                std::to_string(status.fault);
            return M1ExecuteOutcome::Retry;
        }
        error =
            "Metal M1 generated op fault " + std::to_string(status.fault);
        return M1ExecuteOutcome::Failed;
    }

cleanup_failure:
    impl_->context->recycle_transient_buffer(scratch);
    impl_->context->recycle_transient_buffer(descriptor_buffer);
    impl_->context->recycle_transient_buffer(parameter_buffer);
    impl_->context->recycle_transient_buffer(offset_buffer);
    if (borrowed_logits) {
        impl_->context->release_external_buffer(inputs.logits_bf16);
    }
    return M1ExecuteOutcome::Failed;
}

bool M1Runtime::prepare_m2_command(
    const std::shared_ptr<M1PreparedFire>& fire,
    const M1DeviceInputs& inputs,
    RawMetalContext& target,
    std::shared_ptr<M2CommandPlan>& command,
    std::string& error) {
    command.reset();
    if (!fire || !fire->program) {
        error = "Metal M2 requires a prepared M1 fire";
        return false;
    }
    for (const auto& stage : fire->program->stages) {
        if (!stage.executable->fused_supported) {
            error = stage.executable->fused_reason;
            return false;
        }
        if (stage.plan.stage != PTIR_STAGE_PROLOGUE &&
            stage.plan.stage != PTIR_STAGE_EPILOGUE) {
            error = "Metal M2 cannot place per-layer stages";
            return false;
        }
    }

    std::vector<DeviceValueDesc> descriptors;
    std::vector<std::size_t> stage_value_bases;
    std::vector<std::size_t> stage_parameter_bases;
    std::size_t parameter_count = 0;
    for (const auto& stage : fire->program->stages) {
        stage_value_bases.push_back(descriptors.size());
        stage_parameter_bases.push_back(parameter_count);
        parameter_count += stage.plan.ops.size();
        for (const auto& type : stage.plan.value_types) {
            DeviceValueDesc descriptor;
            if (!describe_value(type, inputs.extents, descriptor)) {
                error = "Metal M2 resolved value shape exceeds u32";
                return false;
            }
            descriptors.push_back(descriptor);
        }
    }
    if (descriptors.empty()) descriptors.push_back(DeviceValueDesc{});
    for (std::size_t stage_index = 0;
         stage_index < fire->program->stages.size();
         ++stage_index) {
        const auto& stage = fire->program->stages[stage_index];
        std::uint32_t result_base = 0;
        for (const auto& normalized : stage.plan.ops) {
            const auto& op = normalized.op;
            if (op.tag == PTIR_OP_INTRINSIC_VAL) {
                if (!inputs.logits_bf16.valid() ||
                    inputs.logits_row_count == 0 ||
                    inputs.vocab == 0) {
                    error = "Metal M2 logits intrinsic is unbound";
                    return false;
                }
                const auto& descriptor =
                    descriptors[
                        stage_value_bases[stage_index] +
                        result_base];
                const std::uint32_t rows_needed =
                    op.intr == PTIR_INTR_MTP_DRAFTS
                        ? descriptor.len
                        : (descriptor.len + inputs.vocab - 1) /
                              inputs.vocab;
                const std::uint32_t row_offset =
                    (op.intr == PTIR_INTR_MTP_LOGITS ||
                     op.intr == PTIR_INTR_MTP_DRAFTS) &&
                            inputs.mtp_draft_row >= 0
                        ? static_cast<std::uint32_t>(
                              inputs.mtp_draft_row)
                        : 0u;
                if (row_offset > inputs.logits_row_count ||
                    rows_needed >
                        inputs.logits_row_count - row_offset) {
                    error =
                        "Metal M2 intrinsic row range exceeds bound logits";
                    return false;
                }
                const std::size_t byte_offset =
                    static_cast<std::size_t>(
                        inputs.logits_row_offset) *
                    inputs.vocab * sizeof(std::uint16_t);
                const std::size_t required_bytes =
                    static_cast<std::size_t>(
                        row_offset + rows_needed) *
                    inputs.vocab * sizeof(std::uint16_t);
                if (byte_offset > inputs.logits_bf16.size ||
                    required_bytes >
                        inputs.logits_bf16.size - byte_offset) {
                    error =
                        "Metal M2 intrinsic exceeds logits buffer";
                    return false;
                }
            } else if (op.tag == PTIR_OP_CHAN_PUT) {
                const std::size_t dense =
                    stage.plan.channel_bindings[op.chan];
                if (wire_value_bytes(
                        descriptors[
                            stage_value_bases[stage_index] +
                            op.args[0]]) >
                    fire->pending_cells[dense].size) {
                    error = "Metal M2 channel sink exceeds fixed cell size";
                    return false;
                }
            }
            result_base += op.results;
        }
    }

    std::vector<std::size_t> value_offsets(descriptors.size(), 0);
    std::size_t scratch_bytes = 256;
    std::size_t max_value_len = 1;
    for (std::size_t value = 0; value < descriptors.size(); ++value) {
        scratch_bytes = align_up(scratch_bytes);
        value_offsets[value] = scratch_bytes;
        scratch_bytes += align_up(value_bytes(descriptors[value]));
        max_value_len =
            std::max<std::size_t>(max_value_len, descriptors[value].len);
    }
    const std::size_t temporary_bytes =
        align_up(max_value_len * sizeof(std::uint32_t) * 4);
    const std::size_t temporary_offset = align_up(scratch_bytes);
    scratch_bytes = temporary_offset + temporary_bytes;
    if (scratch_bytes > kMaxScratchBytes ||
        scratch_bytes < temporary_offset) {
        error = "Metal M2 fire scratch exceeds the 512 MiB bound";
        return false;
    }

    auto result = std::make_shared<M2CommandPlan>();
    result->fire = fire;
    result->target = &target;
    result->readiness_ordinal = impl_->next_ordinal++;
    result->commit_ordinal = impl_->next_ordinal++;
    result->logits_base = inputs.logits_bf16;
    result->logits_vocab = inputs.vocab;
    result->scratch = target.acquire_transient_buffer(scratch_bytes);
    result->descriptors = target.acquire_transient_buffer(
        descriptors.size() * sizeof(DeviceValueDesc));
    result->parameters = target.acquire_transient_buffer(
        std::max<std::size_t>(parameter_count, 1) *
        sizeof(DeviceOpParams));
    result->offsets = target.acquire_transient_buffer(
        descriptors.size() * sizeof(std::uint32_t));
    auto cleanup = [&] {
        if (result->scratch.valid())
            target.recycle_transient_buffer(result->scratch);
        if (result->descriptors.valid())
            target.recycle_transient_buffer(result->descriptors);
        if (result->parameters.valid())
            target.recycle_transient_buffer(result->parameters);
        if (result->offsets.valid())
            target.recycle_transient_buffer(result->offsets);
    };
    if (!result->scratch.valid() || !result->descriptors.valid() ||
        !result->parameters.valid() || !result->offsets.valid()) {
        error = "Metal M2 command-buffer scratch allocation failed";
        cleanup();
        return false;
    }
    std::memcpy(
        result->descriptors.contents(),
        descriptors.data(),
        descriptors.size() * sizeof(DeviceValueDesc));
    auto* offsets =
        static_cast<std::uint32_t*>(result->offsets.contents());
    for (std::size_t value = 0; value < value_offsets.size(); ++value) {
        offsets[value] = static_cast<std::uint32_t>(value_offsets[value]);
    }
    auto* parameters =
        static_cast<DeviceOpParams*>(result->parameters.contents());
    for (std::size_t stage_index = 0;
         stage_index < fire->program->stages.size();
         ++stage_index) {
        const auto& stage = fire->program->stages[stage_index];
        std::uint32_t result_base = 0;
        for (std::size_t node = 0; node < stage.plan.ops.size(); ++node) {
            const auto& op = stage.plan.ops[node].op;
            auto& params =
                parameters[stage_parameter_bases[stage_index] + node];
            params.tag = op.tag;
            params.a0 = !op.args.empty() ? op.args[0] : 0;
            params.a1 =
                op.args.size() > 1
                    ? op.args[1]
                    : (op.tag == PTIR_OP_PIVOT_THRESHOLD
                           ? op.pred_payload
                           : 0);
            params.a2 = op.args.size() > 2 ? op.args[2] : 0;
            params.o0 = op.results > 0 ? result_base : params.a0;
            params.o1 = op.results > 1 ? result_base + 1 : params.o0;
            params.imm =
                op.tag == PTIR_OP_INTRINSIC_VAL
                    ? inputs.vocab
                    : op.imm;
            params.imm2 =
                op.tag == PTIR_OP_INTRINSIC_VAL &&
                        (op.intr == PTIR_INTR_MTP_LOGITS ||
                         op.intr == PTIR_INTR_MTP_DRAFTS) &&
                        inputs.mtp_draft_row >= 0
                    ? static_cast<std::uint32_t>(
                          inputs.mtp_draft_row)
                    : op.imm2;
            params.imm3 = op.imm3;
            params.kind = op.kind;
            params.pred_tag = op.pred_tag;
            params.lit_dtype = op.lit_dtype;
            params.lit_bits = op.lit_bits;
            params.channel_slot =
                op.chan >= 0 ? static_cast<std::uint32_t>(op.chan) : 0;
            params.intr = op.intr;
            params.sink_bytes = 0;
            if (op.tag == PTIR_OP_CHAN_PUT) {
                const std::size_t dense =
                    stage.plan.channel_bindings[op.chan];
                params.sink_bytes = static_cast<std::uint32_t>(
                    fire->pending_cells[dense].size);
            }
            result_base += op.results;
        }
    }

    target.use_external_buffer(fire->status);
    target.use_external_buffer(fire->lane_table);
    result->external_handles.push_back(fire->status);
    result->external_handles.push_back(fire->lane_table);
    target.arg_bind_ordinal(
        result->readiness_ordinal, 0, fire->status);
    target.arg_bind_ordinal(
        result->readiness_ordinal, 1, fire->lane_table);
    target.arg_bind_ordinal(
        result->commit_ordinal, 0, fire->status);
    target.arg_bind_ordinal(
        result->commit_ordinal, 1, fire->lane_table);
    for (std::size_t channel = 0;
         channel < fire->word_handles.size();
         ++channel) {
        target.use_external_buffer(fire->word_handles[channel]);
        target.use_external_buffer(fire->committed_cells[channel]);
        target.use_external_buffer(fire->pending_cells[channel]);
        result->external_handles.push_back(fire->word_handles[channel]);
        result->external_handles.push_back(
            fire->committed_cells[channel]);
        result->external_handles.push_back(
            fire->pending_cells[channel]);
        target.arg_bind_ordinal(
            result->readiness_ordinal,
            static_cast<std::uint8_t>(channel + 2),
            fire->word_handles[channel]);
        target.arg_bind_ordinal(
            result->commit_ordinal,
            static_cast<std::uint8_t>(channel + 2),
            fire->word_handles[channel]);
    }
    const SlotHandle temporary = subhandle(
        result->scratch, temporary_offset, temporary_bytes);
    std::vector<bool> pending(fire->channels.size(), false);
    for (std::size_t stage_index = 0;
         stage_index < fire->program->stages.size();
         ++stage_index) {
        const auto& stage = fire->program->stages[stage_index];
        const std::size_t value_base = stage_value_bases[stage_index];
        const SlotHandle stage_descriptors = subhandle(
            result->descriptors,
            value_base * sizeof(DeviceValueDesc),
            std::max<std::size_t>(
                stage.plan.value_types.size() * sizeof(DeviceValueDesc),
                sizeof(DeviceValueDesc)));
        const SlotHandle stage_parameters = subhandle(
            result->parameters,
            stage_parameter_bases[stage_index] *
                sizeof(DeviceOpParams),
            std::max<std::size_t>(
                stage.plan.ops.size() * sizeof(DeviceOpParams),
                sizeof(DeviceOpParams)));
        const SlotHandle stage_offsets = subhandle(
            result->offsets,
            value_base * sizeof(std::uint32_t),
            std::max<std::size_t>(
                stage.plan.value_types.size() * sizeof(std::uint32_t),
                sizeof(std::uint32_t)));
        SlotHandle logits = result->scratch;
        if (inputs.logits_bf16.valid()) {
            const std::size_t offset =
                static_cast<std::size_t>(inputs.logits_row_offset) *
                inputs.vocab * sizeof(std::uint16_t);
            logits = subhandle(
                inputs.logits_bf16,
                offset,
                inputs.logits_bf16.size - offset);
        }
        for (const auto& fused : stage.executable->fused_regions) {
            M2EncodedRegion encoded;
            encoded.pso = fused.pso;
            encoded.ordinal = impl_->next_ordinal++;
            encoded.fixed = {
                fire->status,
                stage_descriptors,
                stage_parameters,
                stage_offsets,
                result->scratch,
                temporary,
                logits,
            };
            for (std::size_t local = 0;
                 local < stage.plan.channel_bindings.size();
                 ++local) {
                const std::size_t dense =
                    stage.plan.channel_bindings[local];
                encoded.channels.push_back(
                    pending[dense] ? fire->pending_cells[dense]
                                   : fire->committed_cells[dense]);
                encoded.channels.push_back(
                    fire->pending_cells[dense]);
            }
            for (const std::uint32_t node : fused.region.nodes) {
                const auto& op = stage.plan.ops[node].op;
                if (op.tag == PTIR_OP_CHAN_PUT) {
                    pending[stage.plan.channel_bindings[op.chan]] = true;
                }
            }
            auto& placement =
                stage.plan.stage == PTIR_STAGE_PROLOGUE
                    ? result->pre
                    : result->post;
            placement.push_back(std::move(encoded));
        }
    }
    static_cast<DeviceStatus*>(fire->status.contents())->state = 0;
    command = std::move(result);
    return true;
}

void M1Runtime::set_m2_logits_row(
    const std::shared_ptr<M2CommandPlan>& command,
    std::uint32_t row) {
    if (!command || !command->logits_base.valid() ||
        command->logits_vocab == 0) {
        return;
    }
    const std::size_t offset =
        static_cast<std::size_t>(row) * command->logits_vocab *
        sizeof(std::uint16_t);
    if (offset >= command->logits_base.size) return;
    const SlotHandle logits = subhandle(
        command->logits_base,
        offset,
        command->logits_base.size - offset);
    for (auto& region : command->pre) region.fixed[6] = logits;
    for (auto& region : command->post) region.fixed[6] = logits;
    auto* lane = reinterpret_cast<PtirLaneRecord*>(
        static_cast<std::uint8_t*>(
            command->fire->lane_table.contents()) +
        sizeof(PtirLaneTableHeader));
    lane->logits_row_offset = row;
}

void M1Runtime::encode_m2_pre(
    const std::shared_ptr<M2CommandPlan>& command,
    StepEncoder& encoder) {
    if (!command) return;
    bind_m2_effect(
        *command, command->readiness_ordinal);
    encoder.set_pso(command->fire->program->readiness);
    encoder.set_argtable_ordinal(
        command->readiness_ordinal);
    encoder.dispatch(Grid{1, 1, 1}, Threadgroup{1, 1, 1});
    encoder.barrier(BarrierVisibility::Device);
    for (const auto& region : command->pre) {
        bind_m2_region(*command, region);
        encoder.set_pso(region.pso);
        encoder.set_argtable_ordinal(region.ordinal);
        encoder.dispatch(Grid{1, 1, 1}, Threadgroup{1, 1, 1});
        encoder.barrier(BarrierVisibility::Device);
    }
}

void M1Runtime::encode_m2_post(
    const std::shared_ptr<M2CommandPlan>& command,
    StepEncoder& encoder) {
    if (!command) return;
    for (const auto& region : command->post) {
        bind_m2_region(*command, region);
        encoder.set_pso(region.pso);
        encoder.set_argtable_ordinal(region.ordinal);
        encoder.dispatch(Grid{1, 1, 1}, Threadgroup{1, 1, 1});
        encoder.barrier(BarrierVisibility::Device);
    }
    bind_m2_effect(
        *command, command->commit_ordinal);
    encoder.set_pso(command->fire->program->commit);
    encoder.set_argtable_ordinal(
        command->commit_ordinal);
    encoder.dispatch(Grid{1, 1, 1}, Threadgroup{1, 1, 1});
}

M1ExecuteOutcome M1Runtime::finish_m2_command(
    std::shared_ptr<M2CommandPlan>& command,
    std::string& error) {
    if (!command || command->target == nullptr) {
        error = "Metal M2 command was not prepared";
        return M1ExecuteOutcome::Failed;
    }
    const DeviceStatus status =
        *static_cast<const DeviceStatus*>(
            command->fire->status.contents());
    for (const SlotHandle& handle : command->external_handles) {
        command->target->release_external_buffer(handle);
    }
    command->target->release_argtable_ordinal(
        command->readiness_ordinal);
    command->target->release_argtable_ordinal(
        command->commit_ordinal);
    for (const auto& region : command->pre)
        command->target->release_argtable_ordinal(region.ordinal);
    for (const auto& region : command->post)
        command->target->release_argtable_ordinal(region.ordinal);
    command->target->recycle_transient_buffer(command->scratch);
    command->target->recycle_transient_buffer(command->descriptors);
    command->target->recycle_transient_buffer(command->parameters);
    command->target->recycle_transient_buffer(command->offsets);
    command.reset();
    if (status.state == 4) return M1ExecuteOutcome::Committed;
    if (status.state == 2) return M1ExecuteOutcome::Retry;
    error = "Metal M2 fused execution fault " +
            std::to_string(status.fault);
    return M1ExecuteOutcome::Failed;
}

bool M1Runtime::prepare_m3_group(
    const std::vector<M3LaneCandidate>& candidates,
    RawMetalContext& target,
    std::shared_ptr<M3GroupCommand>& command,
    std::string& error) {
    command.reset();
    if (candidates.empty() || candidates.size() > 64) {
        error = "Metal M3 group lane count must be in [1,64]";
        return false;
    }
    if (!impl_->grouped_readiness.valid() ||
        !impl_->grouped_commit.valid()) {
        error = "Metal M3 grouped executable is unavailable";
        return false;
    }
    std::unordered_set<const ChannelState*> aliases;
    std::size_t channel_stride = 0;
    std::size_t total_logit_rows = 0;
    for (const auto& candidate : candidates) {
        if (!candidate.fire || !candidate.fire->program) {
            error = "Metal M3 candidate has no prepared program";
            return false;
        }
        if (candidate.fire->program->stages.empty()) {
            error = "Metal M3 cannot group an empty program";
            return false;
        }
        const M1PrepareOutcome readiness = check_readiness_host(
            candidate.fire->program,
            candidate.fire->channels,
            candidate.fire->tickets,
            error);
        if (readiness != M1PrepareOutcome::Ready) {
            error =
                readiness == M1PrepareOutcome::Retry
                    ? "Metal M3 group aborted by definitive host readiness"
                    : "Metal M3 group failed definitive host readiness: " +
                          error;
            return false;
        }
        channel_stride = std::max(
            channel_stride,
            candidate.fire->program->effects.size());
        if (!candidate.inputs.logits_rows.empty() &&
            candidate.inputs.logits_rows.size() !=
                candidate.inputs.logits_row_count) {
            error = "Metal M3 explicit logits row map size mismatch";
            return false;
        }
        if (candidate.inputs.mtp_draft_row >
            static_cast<int>(candidate.inputs.logits_row_count)) {
            error = "Metal M3 MTP row base exceeds logits row map";
            return false;
        }
        total_logit_rows += candidate.inputs.logits_row_count;
        if (total_logit_rows >
            std::numeric_limits<std::uint32_t>::max()) {
            error = "Metal M3 logits row map exceeds u32";
            return false;
        }
        for (const auto& channel : candidate.fire->channels) {
            if (!aliases.insert(channel.get()).second) {
                error = "Metal M3 shared-channel alias requires ordered solo execution";
                return false;
            }
        }
    }
    auto group = std::make_shared<M3GroupCommand>();
    group->candidates = candidates;
    group->target = &target;
    group->readiness = impl_->grouped_readiness;
    group->commit = impl_->grouped_commit;
    group->readiness_ordinal = impl_->next_ordinal++;
    group->commit_ordinal = impl_->next_ordinal++;
    group->timestamp_heap = target.create_timestamp_heap(2);
    const std::size_t lane_count = candidates.size();
    const std::size_t lane_bytes =
        sizeof(PtirLaneTableHeader) +
        lane_count * sizeof(PtirLaneRecord) +
        lane_count * channel_stride * sizeof(PtirLaneChannelSlot);
    group->lane_table = target.acquire_transient_buffer(lane_bytes);
    group->statuses =
        target.acquire_transient_buffer(lane_count * sizeof(DeviceStatus));
    group->channel_meta = target.acquire_transient_buffer(
        std::max<std::size_t>(
            lane_count * channel_stride * sizeof(M3ChannelMeta), 1));
    group->pending_flags = target.acquire_transient_buffer(
        std::max<std::size_t>(lane_count * channel_stride, 1));
    group->row_meta = target.acquire_transient_buffer(
        lane_count * sizeof(M3RowMeta));
    group->row_indices = target.acquire_transient_buffer(
        std::max<std::size_t>(
            total_logit_rows * sizeof(std::uint32_t), 1));
    auto release_group = [&] {
        if (group->lane_table.valid())
            target.recycle_transient_buffer(group->lane_table);
        if (group->statuses.valid())
            target.recycle_transient_buffer(group->statuses);
        if (group->channel_meta.valid())
            target.recycle_transient_buffer(group->channel_meta);
        if (group->pending_flags.valid())
            target.recycle_transient_buffer(group->pending_flags);
        if (group->row_meta.valid())
            target.recycle_transient_buffer(group->row_meta);
        if (group->row_indices.valid())
            target.recycle_transient_buffer(group->row_indices);
        for (auto& stage : group->stages) {
            if (stage.descriptors.valid())
                target.recycle_transient_buffer(stage.descriptors);
            if (stage.parameters.valid())
                target.recycle_transient_buffer(stage.parameters);
            if (stage.offsets.valid())
                target.recycle_transient_buffer(stage.offsets);
            if (stage.scratch.valid())
                target.recycle_transient_buffer(stage.scratch);
            if (stage.layout.valid())
                target.recycle_transient_buffer(stage.layout);
            if (stage.bindings.valid())
                target.recycle_transient_buffer(stage.bindings);
            if (stage.lane_indices.valid())
                target.recycle_transient_buffer(stage.lane_indices);
        }
        for (const auto& handle : group->external_handles)
            target.release_external_buffer(handle);
        if (group->timestamp_heap != nullptr)
            target.release_timestamp_heap(group->timestamp_heap);
    };
    if (!group->lane_table.valid() || !group->statuses.valid() ||
        !group->channel_meta.valid() || !group->pending_flags.valid() ||
        !group->row_meta.valid() || !group->row_indices.valid()) {
        error = "Metal M3 lane-table allocation failed";
        release_group();
        return false;
    }
    std::memset(group->lane_table.contents(), 0, group->lane_table.size);
    std::memset(group->statuses.contents(), 0, group->statuses.size);
    std::memset(
        group->channel_meta.contents(), 0, group->channel_meta.size);
    std::memset(
        group->pending_flags.contents(), 0, group->pending_flags.size);
    std::memset(group->row_meta.contents(), 0, group->row_meta.size);
    auto* header =
        static_cast<PtirLaneTableHeader*>(group->lane_table.contents());
    *header = {
        .abi_version = PTIR_LANE_TABLE_ABI_VERSION,
        .lane_count = static_cast<std::uint32_t>(lane_count),
        .channel_slots_per_lane =
            static_cast<std::uint32_t>(channel_stride),
        .flags = 0,
    };
    auto* records = reinterpret_cast<PtirLaneRecord*>(
        static_cast<std::uint8_t*>(group->lane_table.contents()) +
        sizeof(PtirLaneTableHeader));
    auto* slots = reinterpret_cast<PtirLaneChannelSlot*>(
        reinterpret_cast<std::uint8_t*>(records) +
        lane_count * sizeof(PtirLaneRecord));
    auto* metadata =
        static_cast<M3ChannelMeta*>(group->channel_meta.contents());
    auto* row_meta =
        static_cast<M3RowMeta*>(group->row_meta.contents());
    auto* row_indices = static_cast<std::uint32_t*>(
        group->row_indices.contents());
    std::size_t row_cursor = 0;
    bool ragged = false;
    const std::uint32_t first_rows =
        candidates.front().inputs.logits_row_count;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
        const auto& candidate = candidates[lane];
        const auto& fire = candidate.fire;
        const auto& inputs = candidate.inputs;
        ragged = ragged ||
                 inputs.logits_row_count != first_rows;
        const std::uint64_t mask =
            inputs.logits_row_count == 0
                ? 0
                : (inputs.logits_row_count >= 64
                       ? ~std::uint64_t{0}
                       : (std::uint64_t{1}
                              << inputs.logits_row_count) -
                             1);
        records[lane] = {
            .logits_base = inputs.logits_bf16.gpu_address,
            .logits_row_offset = inputs.logits_row_offset,
            .logits_row_count = inputs.logits_row_count,
            .kv_len = inputs.extents.kv_len,
            .page_count = inputs.extents.page_count,
            .row_count = inputs.extents.row_count,
            .token_count = inputs.extents.token_count,
            .sampled_rows = inputs.extents.sampled_rows,
            .query_len = inputs.extents.query_len,
            .key_len = inputs.extents.key_len,
            .channel_slot_offset =
                static_cast<std::uint32_t>(lane * channel_stride),
            .rng_state = 0,
            .commit_slot =
                group->statuses.gpu_address +
                lane * sizeof(DeviceStatus),
            .active_row_mask = mask,
        };
        row_meta[lane] = {
            .offset = static_cast<std::uint32_t>(row_cursor),
            .count = inputs.logits_row_count,
            .mtp_offset =
                static_cast<std::uint32_t>(
                    std::max(inputs.mtp_draft_row, 0)),
            .reserved = 0,
        };
        for (std::uint32_t row = 0;
             row < inputs.logits_row_count;
             ++row) {
            row_indices[row_cursor++] =
                inputs.logits_rows.empty()
                    ? inputs.logits_row_offset + row
                    : inputs.logits_rows[row];
        }
        for (std::size_t channel = 0;
             channel < fire->program->effects.size();
             ++channel) {
            const std::size_t index =
                lane * channel_stride + channel;
            slots[index] = {
                .committed_cell =
                    fire->committed_cells[channel].gpu_address,
                .pending_cell =
                    fire->pending_cells[channel].gpu_address,
                .expected_head = fire->tickets[channel].expected_head,
                .expected_tail = fire->tickets[channel].expected_tail,
            };
            metadata[index] = {
                .words = fire->word_handles[channel].gpu_address,
                .capacity = static_cast<std::uint32_t>(
                    fire->channels[channel]->capacity),
                .flags = m3_channel_flags(
                    fire->program->effects[channel],
                    candidate.retry_ineligible),
            };
            for (const SlotHandle handle : {
                     fire->word_handles[channel],
                     fire->committed_cells[channel],
                     fire->pending_cells[channel],
                 }) {
                target.use_external_buffer(handle);
                group->external_handles.push_back(handle);
            }
        }
    }
    if (ragged) header->flags |= 1u;

    struct StageRef {
        std::size_t lane = 0;
        std::size_t stage = 0;
    };
    std::map<std::string, std::vector<StageRef>> pre_groups;
    std::map<std::string, std::vector<StageRef>> post_groups;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
        const auto& stages = candidates[lane].fire->program->stages;
        std::uint32_t pre_count = 0;
        std::uint32_t post_count = 0;
        for (std::size_t stage = 0; stage < stages.size(); ++stage) {
            const auto& program_stage = stages[stage];
            auto* groups =
                program_stage.plan.stage == PTIR_STAGE_PROLOGUE
                    ? &pre_groups
                    : program_stage.plan.stage == PTIR_STAGE_EPILOGUE
                          ? &post_groups
                          : nullptr;
            if (groups == nullptr) {
                error = "Metal M3 cannot place a per-layer stage";
                release_group();
                return false;
            }
            if (program_stage.plan.stage == PTIR_STAGE_PROLOGUE)
                ++pre_count;
            else
                ++post_count;
            const std::string key = m3_stage_key(
                program_stage, candidates[lane].inputs.extents);
            if (key.empty()) {
                error = "Metal M3 stage has no canonical signature";
                release_group();
                return false;
            }
            (*groups)[key].push_back({lane, stage});
        }
        if (pre_count > 1 || post_count > 1) {
            error =
                "Metal M3 requires at most one stage per pass boundary";
            release_group();
            return false;
        }
    }
    std::vector<std::vector<StageRef>> stage_groups;
    stage_groups.reserve(pre_groups.size() + post_groups.size());
    for (auto& [key, refs] : pre_groups) {
        static_cast<void>(key);
        stage_groups.push_back(std::move(refs));
    }
    for (auto& [key, refs] : post_groups) {
        static_cast<void>(key);
        stage_groups.push_back(std::move(refs));
    }

    for (const auto& stage_refs : stage_groups) {
        const StageRef first_ref = stage_refs.front();
        const auto& program_stage =
            candidates[first_ref.lane]
                .fire->program->stages[first_ref.stage];
        const auto& plan = program_stage.plan;
        const auto& code = program_stage.executable;
        const std::size_t binding_count =
            m3_used_channel_slots(plan);
        const bool use_fused =
            code->grouped_supported && !code->grouped_regions.empty();
        const auto& regions =
            use_fused ? code->grouped_regions
                      : code->grouped_singleton_regions;
        if (regions.empty()) {
            error = code->grouped_reason.empty()
                        ? "Metal M3 has no grouped singleton fallback"
                        : code->grouped_reason;
            release_group();
            return false;
        }

        const std::size_t stage_lane_count =
            stage_refs.size();
        std::vector<std::vector<DeviceValueDesc>> lane_descriptors(
            stage_lane_count);
        std::vector<std::size_t> max_bytes(
            plan.value_types.size(), 4);
        std::size_t max_value_len = 1;
        std::uint32_t maximum_rows = 1;
        for (std::size_t lane = 0;
             lane < stage_lane_count;
             ++lane) {
            const StageRef ref = stage_refs[lane];
            const auto& lane_plan =
                candidates[ref.lane]
                    .fire->program->stages[ref.stage]
                    .plan;
            if (candidates[ref.lane].inputs.vocab !=
                candidates[first_ref.lane].inputs.vocab) {
                error = "Metal M3 stage vocab mismatch";
                release_group();
                return false;
            }
            if (lane_plan.value_types.size() !=
                    plan.value_types.size() ||
                lane_plan.ops.size() != plan.ops.size() ||
                lane_plan.channel_bindings.size() <
                    binding_count) {
                error =
                    "Metal M3 canonical stage schema mismatch stage=" +
                    std::to_string(plan.stage) + " values=" +
                    std::to_string(lane_plan.value_types.size()) + "/" +
                    std::to_string(plan.value_types.size()) + " ops=" +
                    std::to_string(lane_plan.ops.size()) + "/" +
                    std::to_string(plan.ops.size()) + " bindings=" +
                    std::to_string(
                        lane_plan.channel_bindings.size()) +
                    "/>=" + std::to_string(binding_count);
                release_group();
                return false;
            }
            for (std::size_t value = 0;
                 value < plan.value_types.size();
                 ++value) {
                DeviceValueDesc descriptor;
                if (!describe_value(
                        lane_plan.value_types[value],
                        candidates[ref.lane].inputs.extents,
                        descriptor)) {
                    error =
                        "Metal M3 resolved value shape exceeds u32";
                    release_group();
                    return false;
                }
                lane_descriptors[lane].push_back(descriptor);
                max_bytes[value] =
                    std::max(max_bytes[value], value_bytes(descriptor));
                max_value_len = std::max<std::size_t>(
                    max_value_len, descriptor.len);
                maximum_rows = std::max(
                    maximum_rows, descriptor.rows);
            }
            for (const auto& normalized : lane_plan.ops) {
                const auto& op = normalized.op;
                if (op.tag != PTIR_OP_CHAN_PUT || op.args.empty() ||
                    op.chan < 0 ||
                    static_cast<std::size_t>(op.chan) >=
                        lane_plan.channel_bindings.size()) {
                    continue;
                }
                const std::size_t dense =
                    lane_plan.channel_bindings[op.chan];
                if (dense >=
                        candidates[ref.lane]
                            .fire->pending_cells.size() ||
                    wire_value_bytes(
                        lane_descriptors[lane][op.args[0]]) >
                        candidates[ref.lane]
                            .fire->pending_cells[dense]
                            .size) {
                    error =
                        "Metal M3 channel sink exceeds fixed cell size";
                    release_group();
                    return false;
                }
            }
        }
        std::vector<std::uint32_t> offsets(plan.value_types.size(), 0);
        std::size_t scratch_stride = 256;
        for (std::size_t value = 0; value < offsets.size(); ++value) {
            scratch_stride = align_up(scratch_stride);
            offsets[value] =
                static_cast<std::uint32_t>(scratch_stride);
            scratch_stride += align_up(max_bytes[value]);
        }
        const std::size_t temporary_offset = align_up(scratch_stride);
        scratch_stride = temporary_offset +
                         align_up(
                             max_value_len * sizeof(std::uint32_t) * 4);
        if (scratch_stride > kMaxScratchBytes) {
            error = "Metal M3 per-lane scratch exceeds 512 MiB";
            release_group();
            return false;
        }
        M3StageCommand stage;
        stage.stage = plan.stage;
        stage.singleton_fallback = !use_fused;
        stage.descriptors = target.acquire_transient_buffer(
            stage_lane_count * plan.value_types.size() *
            sizeof(DeviceValueDesc));
        stage.parameters = target.acquire_transient_buffer(
            std::max<std::size_t>(
                stage_lane_count * plan.ops.size(), 1) *
            sizeof(DeviceOpParams));
        stage.offsets = target.acquire_transient_buffer(
            std::max<std::size_t>(offsets.size(), 1) *
            sizeof(std::uint32_t));
        stage.scratch = target.acquire_transient_buffer(
            stage_lane_count * scratch_stride);
        stage.layout =
            target.acquire_transient_buffer(sizeof(M3GroupLayout));
        stage.bindings = target.acquire_transient_buffer(
            std::max<std::size_t>(
                stage_lane_count * binding_count, 1) *
            sizeof(std::uint32_t));
        stage.lane_indices = target.acquire_transient_buffer(
            stage_lane_count * sizeof(std::uint32_t));
        if (!stage.descriptors.valid() || !stage.parameters.valid() ||
            !stage.offsets.valid() || !stage.scratch.valid() ||
            !stage.layout.valid() || !stage.bindings.valid() ||
            !stage.lane_indices.valid()) {
            error = "Metal M3 stage allocation failed";
            group->stages.push_back(std::move(stage));
            release_group();
            return false;
        }
        auto* destination = static_cast<DeviceValueDesc*>(
            stage.descriptors.contents());
        for (std::size_t lane = 0;
             lane < stage_lane_count;
             ++lane) {
            std::copy(
                lane_descriptors[lane].begin(),
                lane_descriptors[lane].end(),
                destination + lane * plan.value_types.size());
        }
        if (!offsets.empty()) {
            std::memcpy(
                stage.offsets.contents(),
                offsets.data(),
                offsets.size() * sizeof(std::uint32_t));
        }
        auto* binding_map =
            static_cast<std::uint32_t*>(stage.bindings.contents());
        auto* lane_indices =
            static_cast<std::uint32_t*>(
                stage.lane_indices.contents());
        for (std::size_t lane = 0;
             lane < stage_lane_count;
             ++lane) {
            const StageRef ref = stage_refs[lane];
            const auto& lane_plan =
                candidates[ref.lane]
                    .fire->program->stages[ref.stage]
                    .plan;
            if (lane_plan.channel_bindings.size() <
                binding_count) {
                error = "Metal M3 stage binding schema mismatch";
                group->stages.push_back(std::move(stage));
                release_group();
                return false;
            }
            std::copy(
                lane_plan.channel_bindings.begin(),
                lane_plan.channel_bindings.begin() +
                    binding_count,
                binding_map + lane * binding_count);
            lane_indices[lane] =
                static_cast<std::uint32_t>(ref.lane);
        }
        auto* parameters =
            static_cast<DeviceOpParams*>(stage.parameters.contents());
        for (std::size_t lane = 0;
             lane < stage_lane_count;
             ++lane) {
            const StageRef ref = stage_refs[lane];
            const auto& lane_plan =
                candidates[ref.lane]
                    .fire->program->stages[ref.stage]
                    .plan;
            std::uint32_t result_base = 0;
            for (std::size_t node = 0;
                 node < lane_plan.ops.size();
                 ++node) {
                const auto& op = lane_plan.ops[node].op;
                auto& params =
                    parameters[lane * plan.ops.size() + node];
                params.tag = op.tag;
                params.a0 = !op.args.empty() ? op.args[0] : 0;
                params.a1 =
                    op.args.size() > 1
                        ? op.args[1]
                        : (op.tag == PTIR_OP_PIVOT_THRESHOLD
                               ? op.pred_payload
                               : 0);
                params.a2 =
                    op.args.size() > 2 ? op.args[2] : 0;
                params.o0 =
                    op.results > 0 ? result_base : params.a0;
                params.o1 =
                    op.results > 1
                        ? result_base + 1
                        : params.o0;
                params.imm =
                    op.tag == PTIR_OP_INTRINSIC_VAL
                        ? candidates[ref.lane].inputs.vocab
                        : op.imm;
                params.imm2 =
                    op.tag == PTIR_OP_INTRINSIC_VAL &&
                            (op.intr ==
                                 PTIR_INTR_MTP_LOGITS ||
                             op.intr ==
                                 PTIR_INTR_MTP_DRAFTS) &&
                            candidates[ref.lane]
                                    .inputs.mtp_draft_row >= 0
                        ? static_cast<std::uint32_t>(
                              candidates[ref.lane]
                                  .inputs.mtp_draft_row)
                        : op.imm2;
                params.imm3 = op.imm3;
                params.kind = op.kind;
                params.pred_tag = op.pred_tag;
                params.lit_dtype = op.lit_dtype;
                params.lit_bits = op.lit_bits;
                params.channel_slot =
                    op.chan >= 0
                        ? static_cast<std::uint32_t>(op.chan)
                        : 0;
                params.intr = op.intr;
                params.sink_bytes = 0;
                if (op.tag == PTIR_OP_CHAN_PUT) {
                    const std::size_t dense =
                        lane_plan.channel_bindings[op.chan];
                    params.sink_bytes = static_cast<std::uint32_t>(
                        candidates[ref.lane]
                            .fire->pending_cells[dense]
                            .size);
                }
                result_base += op.results;
            }
        }
        *static_cast<M3GroupLayout*>(stage.layout.contents()) = {
            .lane_count =
                static_cast<std::uint32_t>(stage_lane_count),
            .value_count =
                static_cast<std::uint32_t>(plan.value_types.size()),
            .scratch_stride =
                static_cast<std::uint32_t>(scratch_stride),
            .temporary_offset =
                static_cast<std::uint32_t>(temporary_offset),
            .vocab = candidates[first_ref.lane].inputs.vocab,
            .reserved = {
                static_cast<std::uint32_t>(
                    binding_count),
                maximum_rows,
                static_cast<std::uint32_t>(
                    plan.ops.size()),
            },
        };
        for (const auto& region : regions) {
            M3EncodedRegion encoded;
            encoded.pso = region.pso;
            encoded.ordinal = impl_->next_ordinal++;
            encoded.fixed = {
                group->lane_table,
                stage.descriptors,
                stage.parameters,
                stage.offsets,
                stage.scratch,
                stage.layout,
                stage.bindings,
                group->pending_flags,
                stage.lane_indices,
                group->row_meta,
                group->row_indices,
            };
            encoded.library = region.region.library;
            encoded.parallel_selection =
                region.parallel_nucleus || region.parallel_topk;
            if (region.parallel_nucleus || region.parallel_topk) {
                const std::uint64_t thread_count =
                    static_cast<std::uint64_t>(
                        stage_lane_count) *
                    maximum_rows * 256u;
                if (thread_count >
                    std::numeric_limits<std::uint32_t>::max()) {
                    error =
                        "Metal M3 parallel library launch exceeds u32 grid";
                    group->stages.push_back(std::move(stage));
                    release_group();
                    return false;
                }
                encoded.grid = {
                    static_cast<std::uint32_t>(thread_count),
                    1,
                    1,
                };
                encoded.threadgroup = {256, 1, 1};
            } else {
                encoded.grid = {
                    static_cast<std::uint32_t>(
                        stage_lane_count),
                    1,
                    1,
                };
                encoded.threadgroup = {1, 1, 1};
            }
            stage.regions.push_back(encoded);
        }
        group->stages.push_back(std::move(stage));
    }
    group->stats.lanes = lane_count;
    static_cast<PtirLaneTableHeader*>(
        group->lane_table.contents())->flags |=
        ragged ? 1u : 0u;
    command = std::move(group);
    return true;
}

void M1Runtime::encode_m3_pre(
    const std::shared_ptr<M3GroupCommand>& command,
    StepEncoder& encoder) {
    if (!command) return;
    bind_m3_effect(*command, command->readiness_ordinal);
    encoder.set_pso(command->readiness);
    encoder.set_argtable_ordinal(command->readiness_ordinal);
    encoder.dispatch(
        Grid{static_cast<std::uint32_t>(command->candidates.size()), 1, 1},
        Threadgroup{1, 1, 1});
    ++command->stats.readiness_launches;
    encoder.barrier(BarrierVisibility::Device);
    for (const auto& stage : command->stages) {
        if (stage.stage != PTIR_STAGE_PROLOGUE) continue;
        for (const auto& region : stage.regions) {
            bind_m3_region(*command, region);
            encoder.set_pso(region.pso);
            encoder.set_argtable_ordinal(region.ordinal);
            encoder.dispatch(region.grid, region.threadgroup);
            ++command->stats.body_launches;
            if (region.library) ++command->stats.library_launches;
            if (region.parallel_selection)
                ++command->stats.parallel_selection_launches;
            if (stage.singleton_fallback) {
                ++command->stats.singleton_fallback_launches;
            }
            encoder.barrier(BarrierVisibility::Device);
        }
    }
}

void M1Runtime::encode_m3_post(
    const std::shared_ptr<M3GroupCommand>& command,
    StepEncoder& encoder) {
    if (!command) return;
    command->post_begin = std::chrono::steady_clock::now();
    encoder.mark_timestamp(command->timestamp_heap, 0);
    for (const auto& stage : command->stages) {
        if (stage.stage != PTIR_STAGE_EPILOGUE) continue;
        for (const auto& region : stage.regions) {
            bind_m3_region(*command, region);
            encoder.set_pso(region.pso);
            encoder.set_argtable_ordinal(region.ordinal);
            encoder.dispatch(region.grid, region.threadgroup);
            ++command->stats.body_launches;
            if (region.library) ++command->stats.library_launches;
            if (region.parallel_selection)
                ++command->stats.parallel_selection_launches;
            if (stage.singleton_fallback) {
                ++command->stats.singleton_fallback_launches;
            }
            encoder.barrier(BarrierVisibility::Device);
        }
    }
    bind_m3_effect(*command, command->commit_ordinal);
    encoder.set_pso(command->commit);
    encoder.set_argtable_ordinal(command->commit_ordinal);
    encoder.dispatch(
        Grid{static_cast<std::uint32_t>(command->candidates.size()), 1, 1},
        Threadgroup{1, 1, 1});
    ++command->stats.commit_launches;
    encoder.mark_timestamp(command->timestamp_heap, 1);
}

std::vector<M1ExecuteOutcome> M1Runtime::finish_m3_group(
    std::shared_ptr<M3GroupCommand>& command,
    std::string& error) {
    std::vector<M1ExecuteOutcome> outcomes;
    if (!command || !command->target) {
        error = "Metal M3 group was not prepared";
        return outcomes;
    }
    const auto* statuses =
        static_cast<const DeviceStatus*>(command->statuses.contents());
    outcomes.reserve(command->candidates.size());
    for (std::size_t lane = 0; lane < command->candidates.size(); ++lane) {
        if (statuses[lane].state == 4)
            outcomes.push_back(M1ExecuteOutcome::Committed);
        else if (statuses[lane].state == 2)
            outcomes.push_back(M1ExecuteOutcome::Retry);
        else
            outcomes.push_back(M1ExecuteOutcome::Failed);
    }
    std::uint64_t timestamps[2] = {0, 0};
    command->target->resolve_timestamps(
        command->timestamp_heap, 2, timestamps);
    if (timestamps[1] > timestamps[0]) {
        command->stats.post_forward_critical_ns =
            timestamps[1] - timestamps[0];
    } else if (
        command->post_begin !=
        std::chrono::steady_clock::time_point{}) {
        command->stats.post_forward_critical_ns =
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() -
                    command->post_begin)
                    .count());
    }
    for (const auto& handle : command->external_handles)
        command->target->release_external_buffer(handle);
    command->target->release_argtable_ordinal(
        command->readiness_ordinal);
    command->target->release_argtable_ordinal(
        command->commit_ordinal);
    command->target->recycle_transient_buffer(command->lane_table);
    command->target->recycle_transient_buffer(command->statuses);
    command->target->recycle_transient_buffer(command->channel_meta);
    command->target->recycle_transient_buffer(command->pending_flags);
    command->target->recycle_transient_buffer(command->row_meta);
    command->target->recycle_transient_buffer(command->row_indices);
    command->target->release_timestamp_heap(command->timestamp_heap);
    for (auto& stage : command->stages) {
        for (const auto& region : stage.regions)
            command->target->release_argtable_ordinal(region.ordinal);
        command->target->recycle_transient_buffer(stage.descriptors);
        command->target->recycle_transient_buffer(stage.parameters);
        command->target->recycle_transient_buffer(stage.offsets);
        command->target->recycle_transient_buffer(stage.scratch);
        command->target->recycle_transient_buffer(stage.layout);
        command->target->recycle_transient_buffer(stage.bindings);
        command->target->recycle_transient_buffer(stage.lane_indices);
    }
    impl_->m3_stats.body_launches += command->stats.body_launches;
    impl_->m3_stats.readiness_launches +=
        command->stats.readiness_launches;
    impl_->m3_stats.library_launches += command->stats.library_launches;
    impl_->m3_stats.parallel_selection_launches +=
        command->stats.parallel_selection_launches;
    impl_->m3_stats.singleton_fallback_launches +=
        command->stats.singleton_fallback_launches;
    impl_->m3_stats.commit_launches += command->stats.commit_launches;
    impl_->m3_stats.lanes += command->stats.lanes;
    impl_->m3_stats.post_forward_critical_ns +=
        command->stats.post_forward_critical_ns;
    command.reset();
    return outcomes;
}

M3GroupStats M1Runtime::m3_stats() const {
    return impl_->m3_stats;
}

std::vector<std::uint64_t> M1Runtime::m3_active_masks_for_test(
    const std::shared_ptr<M3GroupCommand>& command) const {
    std::vector<std::uint64_t> masks;
    if (!command) return masks;
    const auto* lanes = reinterpret_cast<const PtirLaneRecord*>(
        static_cast<const std::uint8_t*>(
            command->lane_table.contents()) +
        sizeof(PtirLaneTableHeader));
    for (std::size_t lane = 0; lane < command->candidates.size(); ++lane)
        masks.push_back(lanes[lane].active_row_mask);
    return masks;
}

void M1Runtime::release(std::shared_ptr<M1PreparedFire>& prepared) {
    if (!prepared) return;
    if (prepared->resource_accounted) {
        m1_prepared_resource_counters().release(
            prepared->m1_external_handles.size(), 2);
        prepared->resource_accounted = false;
    }
    for (const SlotHandle& handle : prepared->m1_external_handles) {
        impl_->context->release_external_buffer(handle);
    }
    if (prepared->status.valid())
        impl_->context->recycle_transient_buffer(prepared->status);
    if (prepared->lane_table.valid())
        impl_->context->recycle_transient_buffer(prepared->lane_table);
    prepared.reset();
}

M1CacheStats M1Runtime::cache_stats() const {
    M1CacheStats result = impl_->stats;
    result.stage_entries = impl_->stage_cache.size();
    result.program_entries = impl_->programs.size();
    result.negative_entries = impl_->negative.size();
    return result;
}

RawMetalContext& M1Runtime::context() { return *impl_->context; }

bool M1Runtime::requires_m2_placement(
    const std::shared_ptr<M1ProgramExecutable>& program) const {
    return program != nullptr && program->requires_m2_placement;
}

std::string M1Runtime::m3_stage_group_key(
    const std::shared_ptr<M1ProgramExecutable>& program,
    std::uint8_t stage,
    const M1RuntimeExtents& extents) const {
    if (!program) return {};
    const auto found = std::find_if(
        program->stages.begin(),
        program->stages.end(),
        [stage](const auto& candidate) {
            return candidate.plan.stage == stage;
        });
    return found == program->stages.end()
               ? std::string{}
               : m3_stage_key(*found, extents);
}

void M1Runtime::inject_stage_cache_entry_for_test(
    std::uint64_t signature_hash,
    std::vector<std::uint8_t> canonical_signature) {
    const std::string identity = encode_cache_identity(
        impl_->context->device_cache_id(), signature_hash);
    auto stage = std::make_shared<M1StageExecutable>();
    stage->cache_identity = identity;
    stage->canonical_signature = std::move(canonical_signature);
    impl_->stage_cache[identity] = std::move(stage);
}

void M1Runtime::inject_compile_failure_for_test(
    std::string function_substring,
    std::string error,
    std::uint32_t count) {
    impl_->compile_faults.push_back({
        .function_substring = std::move(function_substring),
        .error = std::move(error),
        .remaining = count,
    });
}

void M1Runtime::set_program_cache_capacity_for_test(
    std::size_t capacity) {
    impl_->max_program_cache_entries = capacity;
}

}  // namespace pie::metal::pipeline
