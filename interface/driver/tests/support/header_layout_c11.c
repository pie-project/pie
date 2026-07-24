#include <stddef.h>

#include "pie_driver_abi.h"

#define PIE_LAYOUT(type, size, align) \
    _Static_assert(sizeof(type) == (size), "sizeof(" #type ")"); \
    _Static_assert(_Alignof(type) == (align), "_Alignof(" #type ")");

#define PIE_FIELD(type, field, offset) \
    _Static_assert(offsetof(type, field) == (offset), "offsetof(" #type ", " #field ")");

#include "layout_contract.inc"

typedef PieDriver *(*pie_create_fn)(const PieDriverCreateDesc *, PieDriverCaps *);
typedef int32_t (*pie_register_program_fn)(PieDriver *, const PieProgramDesc *, uint64_t *);
typedef int32_t (*pie_bind_instance_fn)(PieDriver *, const PieInstanceDesc *, PieInstanceBinding *);
typedef int32_t (*pie_launch_fn)(PieDriver *, const PieFrameDesc *, PieCompletion);
typedef int32_t (*pie_copy_kv_fn)(PieDriver *, const PieKvCopyDesc *, PieCompletion);
typedef int32_t (*pie_copy_state_fn)(PieDriver *, const PieStateCopyDesc *, PieCompletion);
typedef int32_t (*pie_resize_pool_fn)(PieDriver *, const PiePoolResizeDesc *, PieCompletion);
typedef int32_t (*pie_close_instance_fn)(PieDriver *, uint64_t);
typedef void (*pie_destroy_fn)(PieDriver *);

static pie_create_fn const expect_cuda_create = &pie_cuda_create;
static pie_register_program_fn const expect_cuda_register_program = &pie_cuda_register_program;
static pie_bind_instance_fn const expect_cuda_bind_instance = &pie_cuda_bind_instance;
static pie_launch_fn const expect_cuda_launch = &pie_cuda_launch;
static pie_copy_kv_fn const expect_cuda_copy_kv = &pie_cuda_copy_kv;
static pie_copy_state_fn const expect_cuda_copy_state = &pie_cuda_copy_state;
static pie_resize_pool_fn const expect_cuda_resize_pool = &pie_cuda_resize_pool;
static pie_close_instance_fn const expect_cuda_close_instance = &pie_cuda_close_instance;
static pie_destroy_fn const expect_cuda_destroy = &pie_cuda_destroy;

static pie_create_fn const expect_metal_create = &pie_metal_create;
static pie_register_program_fn const expect_metal_register_program = &pie_metal_register_program;
static pie_bind_instance_fn const expect_metal_bind_instance = &pie_metal_bind_instance;
static pie_launch_fn const expect_metal_launch = &pie_metal_launch;
static pie_copy_kv_fn const expect_metal_copy_kv = &pie_metal_copy_kv;
static pie_copy_state_fn const expect_metal_copy_state = &pie_metal_copy_state;
static pie_resize_pool_fn const expect_metal_resize_pool = &pie_metal_resize_pool;
static pie_close_instance_fn const expect_metal_close_instance = &pie_metal_close_instance;
static pie_destroy_fn const expect_metal_destroy = &pie_metal_destroy;

int pie_driver_abi_header_layout_c11(void) {
    return expect_cuda_create != NULL && expect_metal_create != NULL;
}
