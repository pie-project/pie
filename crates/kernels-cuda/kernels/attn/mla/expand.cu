#include <cuda_bf16.h>

#include "prim/device.cuh"

// A latent attention's prefill expanded to one key and value head a query
// head (what vLLM's MLA prefill runs): the fire's latent rows gathered out
// of their pages into rows of their own, page by page in the order the
// page table names them; the `kv_b` product of those rows (per head the
// `nope` key part, then the value); and the key rows packed as that part
// beside the row's rope part, which every head shares.

namespace pie::attn::mla {

typedef unsigned int u32;

// Compact page `c` (the page table's `c`-th entry) lands at cells
// `[c * page, (c + 1) * page)`: its latent rows (`rank` wide) at `lat`, its
// rope rows (`rope` wide) at `kpe`. Sixteen bytes a thread. Thread 0 also
// lands `live = [0, requests]`, and the first `pages` threads the identity
// page table the expanded planes are read through.
__global__ void gather(
    const __nv_bfloat16* __restrict__ ckv_pages,
    const __nv_bfloat16* __restrict__ kpe_pages,
    const int* __restrict__ page_indices,
    int pages,
    int page,
    int rank,
    int rope,
    __nv_bfloat16* __restrict__ lat,
    __nv_bfloat16* __restrict__ kpe,
    int* __restrict__ ident,
    int* __restrict__ live,
    int requests)
{
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at == 0) {
        live[0] = 0;
        live[1] = requests;
    }
    if (at < pages) ident[at] = static_cast<int>(at);
    const int runs = (rank + rope) / 8;
    const long long cells = static_cast<long long>(pages) * page;
    if (at >= cells * runs) return;
    const long long cell = at / runs;
    const int run = static_cast<int>(at % runs);
    const int c = static_cast<int>(cell / page);
    const int t = static_cast<int>(cell % page);
    const long long src = static_cast<long long>(page_indices[c]) * page + t;
    if (run * 8 < rank) {
        *reinterpret_cast<uint4*>(lat + cell * rank + run * 8) =
            *reinterpret_cast<const uint4*>(ckv_pages + src * rank + run * 8);
    } else {
        const int r = run * 8 - rank;
        *reinterpret_cast<uint4*>(kpe + cell * rope + r) =
            *reinterpret_cast<const uint4*>(kpe_pages + src * rope + r);
    }
}

// Key rows `[cells][heads][nope + rope]`: each head's `nope` part of the
// expanded row (`[heads][nope + v]` a cell), then the cell's rope part.
// Sixteen bytes a thread.
__global__ void pack_keys(
    const __nv_bfloat16* __restrict__ expanded,
    const __nv_bfloat16* __restrict__ kpe,
    __nv_bfloat16* __restrict__ keys,
    long long cells,
    int heads,
    int nope,
    int v,
    int rope)
{
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int width = nope + rope;
    const int runs = width / 8;
    if (at >= cells * heads * runs) return;
    const long long row = at / runs;
    const int run = static_cast<int>(at % runs);
    const long long cell = row / heads;
    const int h = static_cast<int>(row % heads);
    const int d = run * 8;
    const uint4 x = d < nope
        ? *reinterpret_cast<const uint4*>(expanded + (cell * heads + h) * (nope + v) + d)
        : *reinterpret_cast<const uint4*>(kpe + cell * rope + (d - nope));
    *reinterpret_cast<uint4*>(keys + row * width + d) = x;
}

// Query rows `[rows][heads][nope + rope]` from their two halves. Sixteen
// bytes a thread.
__global__ void pack_queries(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    __nv_bfloat16* __restrict__ q,
    long long rows,
    int heads,
    int nope,
    int rope)
{
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int width = nope + rope;
    const int runs = width / 8;
    if (at >= rows * heads * runs) return;
    const long long row = at / runs;
    const int run = static_cast<int>(at % runs);
    const long long r = row / heads;
    const int h = static_cast<int>(row % heads);
    const int d = run * 8;
    const uint4 x = d < nope
        ? *reinterpret_cast<const uint4*>(q_nope + (r * heads + h) * nope + d)
        : *reinterpret_cast<const uint4*>(q_pe + (r * heads + h) * rope + (d - nope));
    *reinterpret_cast<uint4*>(q + row * width + d) = x;
}

}  // namespace pie::attn::mla
