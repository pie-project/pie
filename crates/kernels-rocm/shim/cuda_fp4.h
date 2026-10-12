

#pragma once

typedef unsigned char __nv_fp4_storage_t;

typedef unsigned char __nv_fp4x2_storage_t;

typedef unsigned short __nv_fp4x4_storage_t;

typedef enum __nv_fp4_interpretation_t {
    __NV_E2M1 = 0,
} __nv_fp4_interpretation_t;

struct __nv_fp4_e2m1 {
    __nv_fp4_storage_t __x;
};

struct __nv_fp4x2_e2m1 {
    __nv_fp4x2_storage_t __x;
};

struct __nv_fp4x4_e2m1 {
    __nv_fp4x4_storage_t __x;
};
