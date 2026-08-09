#pragma once
// Stub <cuda_bf16.h>: only the storage type is ever named host-side.
struct __nv_bfloat16 { unsigned short x; };
struct __nv_bfloat162 { __nv_bfloat16 x, y; };
