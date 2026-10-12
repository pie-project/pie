#ifndef PIE_VULKAN_DIT_TYPES_SLANG
#define PIE_VULKAN_DIT_TYPES_SLANG

#include "common/bf16.slang"

// The activation element: bf16 unless PIE_X_F32.
#ifdef PIE_X_F32
#define PIE_XT float
#define PIE_XL(v) (v)
#define PIE_XS(f) (f)
#else
#define PIE_XT uint16_t
#define PIE_XL(v) bf16_to_f32(v)
#define PIE_XS(f) f32_to_bf16(f)
#endif

// A modulation plane's element: bf16 unless PIE_M_F32.
#ifdef PIE_M_F32
#define PIE_MT float
#define PIE_ML(v) (v)
#else
#define PIE_MT uint16_t
#define PIE_ML(v) bf16_to_f32(v)
#endif

#endif
