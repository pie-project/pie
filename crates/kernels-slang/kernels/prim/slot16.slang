#pragma once

#include "prim/bf16.slang"

// bf16 values one to a 16-bit slot: SPIR-V's 16-bit storage, which WGSL
// lacks. A SPIR-V kernel lands single values here (a row of an odd width, a
// lane's one row); its WGSL form lands words, two values each, and its host
// refuses what a word would tear.

#if !defined(PIE_WGSL)

#define PIE_SLOTS_RO(binding_, name_) [[vk::binding(binding_)]] StructuredBuffer<uint16_t> name_
#define PIE_SLOTS_RW(binding_, name_) [[vk::binding(binding_)]] RWStructuredBuffer<uint16_t> name_

float slot_at(StructuredBuffer<uint16_t> buf, uint i) {
    return bf16_to_f32(uint(buf[i]));
}

void slot_put(RWStructuredBuffer<uint16_t> buf, uint i, float v) {
    buf[i] = uint16_t(f32_to_bf16(v));
}

#endif
