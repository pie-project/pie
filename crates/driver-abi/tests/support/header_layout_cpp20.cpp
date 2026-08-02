#include <cstddef>
#include <type_traits>

#include "pie_driver_abi.h"

#define PIE_LAYOUT(type, size, align) \
    static_assert(sizeof(type) == (size), "sizeof(" #type ")"); \
    static_assert(alignof(type) == (align), "alignof(" #type ")");

#define PIE_FIELD(type, field, offset) \
    static_assert(offsetof(type, field) == (offset), "offsetof(" #type ", " #field ")");

#include "layout_contract.inc"

using pie_create_fn = PieDriver *(*)(const PieDriverCreateDesc *, PieDriverCaps *);
using pie_register_program_fn = int32_t (*)(PieDriver *, const PieProgramDesc *, uint64_t *);
using pie_bind_instance_fn = int32_t (*)(PieDriver *, const PieInstanceDesc *, PieInstanceBinding *);
using pie_launch_fn = int32_t (*)(PieDriver *, const PieFrameDesc *, PieCompletion);
using pie_copy_kv_fn = int32_t (*)(PieDriver *, const PieKvCopyDesc *, PieCompletion);
using pie_copy_state_fn = int32_t (*)(PieDriver *, const PieStateCopyDesc *, PieCompletion);
using pie_resize_pool_fn = int32_t (*)(PieDriver *, const PiePoolResizeDesc *, PieCompletion);
using pie_close_instance_fn = int32_t (*)(PieDriver *, uint64_t);
using pie_destroy_fn = void (*)(PieDriver *);

static_assert(std::is_same_v<decltype(&pie_cuda_create), pie_create_fn>);
static_assert(std::is_same_v<decltype(&pie_cuda_register_program), pie_register_program_fn>);
static_assert(std::is_same_v<decltype(&pie_cuda_bind_instance), pie_bind_instance_fn>);
static_assert(std::is_same_v<decltype(&pie_cuda_launch), pie_launch_fn>);
static_assert(std::is_same_v<decltype(&pie_cuda_copy_kv), pie_copy_kv_fn>);
static_assert(std::is_same_v<decltype(&pie_cuda_copy_state), pie_copy_state_fn>);
static_assert(std::is_same_v<decltype(&pie_cuda_resize_pool), pie_resize_pool_fn>);
static_assert(std::is_same_v<decltype(&pie_cuda_close_instance), pie_close_instance_fn>);
static_assert(std::is_same_v<decltype(&pie_cuda_destroy), pie_destroy_fn>);

static_assert(std::is_same_v<decltype(&pie_metal_create), pie_create_fn>);
static_assert(std::is_same_v<decltype(&pie_metal_register_program), pie_register_program_fn>);
static_assert(std::is_same_v<decltype(&pie_metal_bind_instance), pie_bind_instance_fn>);
static_assert(std::is_same_v<decltype(&pie_metal_launch), pie_launch_fn>);
static_assert(std::is_same_v<decltype(&pie_metal_copy_kv), pie_copy_kv_fn>);
static_assert(std::is_same_v<decltype(&pie_metal_copy_state), pie_copy_state_fn>);
static_assert(std::is_same_v<decltype(&pie_metal_resize_pool), pie_resize_pool_fn>);
static_assert(std::is_same_v<decltype(&pie_metal_close_instance), pie_close_instance_fn>);
static_assert(std::is_same_v<decltype(&pie_metal_destroy), pie_destroy_fn>);

int pie_driver_abi_header_layout_cpp20() { return 0; }
