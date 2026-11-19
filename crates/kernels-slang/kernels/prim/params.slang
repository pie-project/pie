#pragma once

// A kernel's scalars: a push constant on SPIR-V (engine-vulkan pushes them,
// and a recording keeps them), a uniform block at `binding_`, the slot after
// the buffers, on WGSL (engine-wgpu binds them there). Either way the fields
// are the scalars in the order the host passes them.
//
// `PIE_PARAMS_FIRED` declares the same block, and on SPIR-V the same fields
// again as a uniform at `binding_`, which engine-vulkan backs with the fire
// block: the per-fire memory a recording reads what a fire changes from. A
// field read as `PIE_FIRE(field)` (a row count, an element count, a GEMM's
// M, a split's rows) comes from there, so a recording's commands do not
// carry it; on WGSL it is the field itself.
#ifdef PIE_WGSL
#define PIE_PARAMS(binding_, type_, name_) [[vk::binding(binding_)]] ConstantBuffer<type_> name_
#define PIE_PARAMS_FIRED(binding_, type_, name_) PIE_PARAMS(binding_, type_, name_)
#define PIE_FIRE(field_) params.field_
#else
#define PIE_PARAMS(binding_, type_, name_) [[vk::push_constant]] ConstantBuffer<type_> name_
#define PIE_PARAMS_FIRED(binding_, type_, name_) \
    [[vk::push_constant]] ConstantBuffer<type_> name_; \
    [[vk::binding(binding_)]] ConstantBuffer<type_> pie_fire
#define PIE_FIRE(field_) pie_fire.field_
#endif
