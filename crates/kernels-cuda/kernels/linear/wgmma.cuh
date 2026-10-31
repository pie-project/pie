#pragma once

#include "linear/grouped_fp8.cuh"
#include "linear/moe_w4.cuh"
#include "linear/w8a16.cuh"

namespace pie::linear {

namespace wgfp8 {

__device__ __forceinline__ unsigned long long desc(const void* smem) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    unsigned long long d = (at & 0x3FFFFu) >> 4;   // start address
    d |= 1ull << 16;                               // leading offset (unread: swizzled K-major)
    d |= static_cast<unsigned long long>(1024 >> 4) << 32;  // 8 rows of 128 bytes
    d |= 1ull << 62;                               // 128-byte swizzle
    return d;
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(at), "l"(src));
}

template <int R>
struct Mma;

template <>
struct Mma<32> {
    static __device__ __forceinline__ void run(float (&d)[16], unsigned long long a,
                                               unsigned long long b, int acc) {
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1;\n}\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
              "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]),
              "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
            : "l"(a), "l"(b), "r"(acc));
    }
};

template <>
struct Mma<64> {
    static __device__ __forceinline__ void run(float (&d)[32], unsigned long long a,
                                               unsigned long long b, int acc) {
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,"
            "%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
              "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]),
              "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]),
              "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
              "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]),
              "+f"(d[31])
            : "l"(a), "l"(b), "r"(acc));
    }
};

}

namespace wgfp4 {

// m64n64k16 with A, the warp's 16 rows, from registers: the a0..a3 of
// mma.sync's m16n8k16, so the register dequant feeds it as it feeds mma.sync.
__device__ __forceinline__ void mma(float (&d)[32], const unsigned (&a)[4],
                                    unsigned long long b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,"
        "%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, 1, 1, 1, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]),
          "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]),
          "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
          "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]),
          "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}

// m64n128k16, `mma`'s A from registers against 128 B columns.
__device__ __forceinline__ void mma(float (&d)[64], const unsigned (&a)[4],
                                    unsigned long long b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, 1, 1, 1, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}

// Holds `a`'s registers to here: an MMA in flight still reads them.
__device__ __forceinline__ void hold(unsigned (&a)[4]) {
    asm volatile("" : "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])::"memory");
}

}

// namespace wgfp4

namespace wgfp4 {

// The byte of `w` that `sel` (`0x4440 | c`) picks: its two e2m1 codes as
// `w4mma::e2m1_pair` lands them (the low code the pair's low bf16).
__device__ __forceinline__ unsigned byte_codes(unsigned w, unsigned sel) {
    const unsigned b = __byte_perm(w, 0u, sel);
    return ((b * 0x01001000u) & 0x80008000u) | ((b * 0x00040040u) & 0x01C001C0u);
}

}

}
