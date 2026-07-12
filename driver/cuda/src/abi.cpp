// The C ABI boundary for the CUDA driver: the 11 frozen `pie_cuda_*` exports
// (see `interface/driver/include/pie_driver_abi.h`), ABI-level argument
// validation, and the opaque `PieDriver*` handle <-> `pie::cuda::Context`
// mapping. Everything else (composition, device state, registries, launch
// composition) lives in `pie::cuda::Context` (`context.{hpp,cpp}`).
#include "context.hpp"

#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "pie_native/abi_validation.hpp"

namespace {

pie::cuda::Context* as_context(PieDriver* driver) {
    return reinterpret_cast<pie::cuda::Context*>(driver);
}

PieDriver* create_context(const PieDriverCreateDesc& desc, PieDriverCaps* caps) {
    std::memset(caps, 0, sizeof(*caps));
    const std::string config_path(
        reinterpret_cast<const char*>(desc.config_bytes.ptr),
        desc.config_bytes.len);
    auto context = std::make_unique<pie::cuda::Context>();
    if (context->initialize(config_path, desc.runtime) != PIE_STATUS_OK) {
        return nullptr;
    }
    context->fill_caps(caps);
    return reinterpret_cast<PieDriver*>(context.release());
}

}  // namespace

extern "C" PieDriver* pie_cuda_create(const PieDriverCreateDesc* desc,
                                       PieDriverCaps* caps) {
    if (pie_native::abi::validate_create_desc(desc, caps) != PIE_STATUS_OK) {
        return nullptr;
    }
    try {
        return create_context(*desc, caps);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] create: " << e.what() << "\n";
        return nullptr;
    } catch (...) {
        std::cerr << "[pie-driver-cuda] create: unknown exception\n";
        return nullptr;
    }
}

extern "C" int32_t pie_cuda_register_program(PieDriver* driver,
                                              const PieProgramDesc* program,
                                              std::uint64_t* program_id) {
    const int status = pie_native::abi::validate_program_desc(program, program_id);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->register_program(*program, program_id);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_register_channel(
    PieDriver* driver,
    const PieChannelDesc* channel,
    PieChannelEndpointBinding* binding) {
    const int status = pie_native::abi::validate_channel_desc(channel, binding);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->register_channel(*channel, binding);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_bind_instance(PieDriver* driver,
                                           const PieInstanceDesc* instance,
                                           PieInstanceBinding* binding) {
    const int status = pie_native::abi::validate_instance_desc(instance, binding);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->bind_instance(*instance, binding);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_launch(PieDriver* driver,
                                    const PieLaunchDesc* launch,
                                    PieCompletion completion) {
    const int status = pie_native::abi::validate_launch_desc(launch);
    if (status != PIE_STATUS_OK) return status;
    const int completion_status =
        pie_native::abi::validate_completion(completion, false);
    if (completion_status != PIE_STATUS_OK) return completion_status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->launch(*launch, completion);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_copy_kv(PieDriver* driver,
                                     const PieKvCopyDesc* copy,
                                     PieCompletion completion) {
    const int status = pie_native::abi::validate_kv_copy_desc(copy);
    if (status != PIE_STATUS_OK) return status;
    const int completion_status =
        pie_native::abi::validate_completion(completion, true);
    if (completion_status != PIE_STATUS_OK) return completion_status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->copy_kv(*copy, completion);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_copy_state(PieDriver* driver,
                                        const PieStateCopyDesc* copy,
                                        PieCompletion completion) {
    const int status = pie_native::abi::validate_state_copy_desc(copy);
    if (status != PIE_STATUS_OK) return status;
    const int completion_status =
        pie_native::abi::validate_completion(completion, true);
    if (completion_status != PIE_STATUS_OK) return completion_status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->copy_state(*copy, completion);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_resize_pool(PieDriver* driver,
                                         const PiePoolResizeDesc* resize,
                                         PieCompletion completion) {
    const int status = pie_native::abi::validate_pool_resize_desc(resize);
    if (status != PIE_STATUS_OK) return status;
    const int completion_status =
        pie_native::abi::validate_completion(completion, true);
    if (completion_status != PIE_STATUS_OK) return completion_status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->resize_pool(*resize, completion);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_close_instance(PieDriver* driver,
                                            std::uint64_t instance_id) {
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->close_instance(instance_id);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_close_channel(PieDriver* driver,
                                          std::uint64_t channel_id) {
    if (driver == nullptr || channel_id == 0) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->close_channel(channel_id);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" void pie_cuda_destroy(PieDriver* driver) {
    delete as_context(driver);
}
