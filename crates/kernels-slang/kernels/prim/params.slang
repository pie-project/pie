#pragma once

// A kernel's scalars: a push constant on SPIR-V (engine-vulkan pushes them),
// a uniform block at `binding_`, the slot after the buffers, on WGSL
// (engine-wgpu binds them there). Either way the fields are the scalars in
// the order the host passes them.
#ifdef PIE_WGSL
#define PIE_PARAMS(binding_, type_, name_) [[vk::binding(binding_)]] ConstantBuffer<type_> name_
#else
#define PIE_PARAMS(binding_, type_, name_) [[vk::push_constant]] ConstantBuffer<type_> name_
#endif
