#pragma once

#include "batch/batch_schedule.hpp"
#include "batch/decode_timing.hpp"
#include "batch/forward.hpp"
#include "model/qwen3_5/geometry.hpp"
#include "store/kv_pool.hpp"

namespace pie::metal::batch {

struct NativeAccess {
    static bool setup(
        MetalExecutor& executor,
        const std::string& checkpoint_dir,
        const std::string& kernels_dir,
        const DecodeGeometry& geometry,
        std::string* error) {
        return executor.setup_native(
            checkpoint_dir,
            kernels_dir,
            geometry,
            error);
    }

    static bool setup_kv_pool(
        MetalExecutor& executor,
        std::uint32_t total_pages,
        std::uint32_t page_size,
        std::string* error) {
        return executor.setup_kv_pool_native(
            total_pages,
            page_size,
            error);
    }

    static void reset_state(MetalExecutor& executor) {
        executor.reset_state_native();
    }

    static void reset_state(MetalExecutor& executor, std::uint32_t slot) {
        executor.reset_state_native(slot);
    }

    static bool copy_state_slot(
        MetalExecutor& executor,
        std::uint32_t src_slot,
        std::uint32_t dst_slot,
        std::string* error) {
        return executor.copy_state_slot_native(src_slot, dst_slot, error);
    }

    static StepTiming step(
        MetalExecutor& executor,
        std::uint32_t token_id,
        std::uint32_t position,
        std::uint32_t slot = 0) {
        return executor.step_native(token_id, position, slot);
    }

    static bool run_batch_step(
        MetalExecutor& executor,
        const BatchSchedule& schedule,
        const BatchStepInputs& inputs,
        std::string* error) {
        return executor.run_batch_step_native(schedule, inputs, error);
    }

    static std::uint64_t paged_bind_generation(
        const MetalExecutor& executor) {
        return executor.paged_bind_generation_native();
    }

    static const KvPagePool& kv_pool(const MetalExecutor& executor) {
        return executor.kv_pool_native();
    }

    static int vocab(const MetalExecutor& executor) {
        return executor.vocab_native();
    }

    static void copy_logits_f32(
        const MetalExecutor& executor,
        float* output) {
        executor.copy_logits_f32_native(output);
    }

    static void copy_batch_logits_f32(
        const MetalExecutor& executor,
        std::uint32_t token_row,
        float* output) {
        executor.copy_batch_logits_f32_native(token_row, output);
    }

    static std::uint32_t argmax(const MetalExecutor& executor) {
        return executor.argmax_native();
    }
};

}  // namespace pie::metal::batch
