#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "batch/compose.hpp"
#include "mtl4_context.hpp"
#include "pipeline/interp.hpp"

namespace pie::metal::pipeline {

enum class M1PrepareOutcome { Ready, Retry, Failed };
enum class M1ExecuteOutcome { Committed, Retry, Failed };
enum class M1ExecutionMode { Singleton, Fused };
enum class M1CompileFailureKind { None, Deterministic, Retryable };

struct M1CacheIdentityVersions {
    std::uint16_t compiler = 0;
    std::uint16_t region_plan = 0;
    std::uint32_t lane_table = 0;
    std::uint16_t emitter = 0;
};

std::string encode_m1_cache_identity(
    std::uint64_t device,
    std::uint64_t signature,
    M1CacheIdentityVersions versions);

struct M1RuntimeExtents {
    std::uint32_t kv_len = 1;
    std::uint32_t page_count = 1;
    std::uint32_t row_count = 1;
    std::uint32_t token_count = 1;
    std::uint32_t sampled_rows = 1;
    std::uint32_t query_len = 1;
    std::uint32_t key_len = 1;
};

M1RuntimeExtents m1_extents_from_forward_desc(
    const batch::MemberForwardDesc& desc,
    std::uint32_t sampled_rows);
M1RuntimeExtents m3_extents_from_forward_desc(
    const batch::MemberForwardDesc& desc);

struct M1ResolvedShape {
    std::uint32_t len = 0;
    std::uint32_t rows = 0;
    std::uint32_t row_len = 0;
};

M1ResolvedShape resolve_m1_shape_for_test(
    const pie_native::ptir::plan::ValueType& type,
    const M1RuntimeExtents& extents);

struct M1DeviceInputs {
    SlotHandle logits_bf16{};
    std::uint32_t logits_row_offset = 0;
    std::uint32_t logits_row_count = 0;
    std::uint32_t vocab = 0;
    std::vector<std::uint32_t> logits_rows;
    int mtp_draft_row = -1;
    M1RuntimeExtents extents{};
};

M1DeviceInputs m1_singleton_fallback_inputs(
    const batch::LogitsOut& output,
    const batch::MemberForwardDesc& desc,
    int mtp_draft_row);

struct M1CacheStats {
    std::uint64_t memory_hits = 0;
    std::uint64_t persistent_hits = 0;
    std::uint64_t compilations = 0;
    std::uint64_t negative_hits = 0;
    std::size_t stage_entries = 0;
    std::size_t program_entries = 0;
    std::size_t negative_entries = 0;
};

struct M1ProgramExecutable;
struct M1PreparedFire;
struct M2CommandPlan;
struct M3GroupCommand;

struct M3LaneCandidate {
    std::shared_ptr<M1PreparedFire> fire;
    M1DeviceInputs inputs;
    bool retry_ineligible = false;
};

struct M3GroupStats {
    std::uint64_t readiness_launches = 0;
    std::uint64_t body_launches = 0;
    std::uint64_t library_launches = 0;
    std::uint64_t parallel_selection_launches = 0;
    std::uint64_t singleton_fallback_launches = 0;
    std::uint64_t commit_launches = 0;
    std::uint64_t lanes = 0;
    std::uint64_t post_forward_critical_ns = 0;
};

class M1Runtime {
  public:
    static std::unique_ptr<M1Runtime> create(
        const std::string& kernels_dir,
        const std::string& cache_dir,
        std::string& error);
    ~M1Runtime();

    M1Runtime(const M1Runtime&) = delete;
    M1Runtime& operator=(const M1Runtime&) = delete;

    std::shared_ptr<M1ProgramExecutable> compile_program(
        std::uint64_t program_hash,
        const ExecPlan& plan,
        std::string& error,
        std::span<const std::uint8_t> canonical_bytes = {},
        M1CompileFailureKind* failure_kind = nullptr);

    M1PrepareOutcome prepare(
        const std::shared_ptr<M1ProgramExecutable>& program,
        const std::vector<std::shared_ptr<ChannelState>>& channels,
        const std::vector<batch::ChannelTicket>& tickets,
        std::shared_ptr<M1PreparedFire>& prepared,
        std::string& error);

    M1ExecuteOutcome execute(
        const std::shared_ptr<M1PreparedFire>& prepared,
        const M1DeviceInputs& inputs,
        std::string& error,
        M1ExecutionMode mode = M1ExecutionMode::Singleton);

    void release(std::shared_ptr<M1PreparedFire>& prepared);

    bool prepare_m2_command(
        const std::shared_ptr<M1PreparedFire>& prepared,
        const M1DeviceInputs& inputs,
        RawMetalContext& target_context,
        std::shared_ptr<M2CommandPlan>& command,
        std::string& error);
    void encode_m2_pre(
        const std::shared_ptr<M2CommandPlan>& command,
        StepEncoder& encoder);
    void encode_m2_post(
        const std::shared_ptr<M2CommandPlan>& command,
        StepEncoder& encoder);
    void set_m2_logits_row(
        const std::shared_ptr<M2CommandPlan>& command,
        std::uint32_t row);
    M1ExecuteOutcome finish_m2_command(
        std::shared_ptr<M2CommandPlan>& command,
        std::string& error);

    std::string m3_stage_group_key(
        const std::shared_ptr<M1ProgramExecutable>& program,
        std::uint8_t stage,
        const M1RuntimeExtents& extents) const;
    bool prepare_m3_group(
        const std::vector<M3LaneCandidate>& lanes,
        RawMetalContext& target_context,
        std::shared_ptr<M3GroupCommand>& command,
        std::string& error);
    void encode_m3_pre(
        const std::shared_ptr<M3GroupCommand>& command,
        StepEncoder& encoder);
    void encode_m3_post(
        const std::shared_ptr<M3GroupCommand>& command,
        StepEncoder& encoder);
    std::vector<M1ExecuteOutcome> finish_m3_group(
        std::shared_ptr<M3GroupCommand>& command,
        std::string& error);
    M3GroupStats m3_stats() const;
    std::vector<std::uint64_t> m3_active_masks_for_test(
        const std::shared_ptr<M3GroupCommand>& command) const;

    M1CacheStats cache_stats() const;
    RawMetalContext& context();
    bool requires_m2_placement(
        const std::shared_ptr<M1ProgramExecutable>& program) const;
    void inject_stage_cache_entry_for_test(
        std::uint64_t signature_hash,
        std::vector<std::uint8_t> canonical_signature);
    void inject_compile_failure_for_test(
        std::string function_substring,
        std::string error,
        std::uint32_t count = 1);
    void set_program_cache_capacity_for_test(std::size_t capacity);

  private:
    struct Impl;
    explicit M1Runtime(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

std::string default_m1_cache_dir();

}  // namespace pie::metal::pipeline
