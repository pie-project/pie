#pragma once

#include "prim/device.cuh"

namespace pie {

// Where a fixed state row lives: a row splits into `pieces` equal pieces of
// `atoms` atoms each (a head, a channel, a frame row: whatever the reading op
// never cuts), and piece `p` of the row a lane names sits in chunk
// `ids[row * pieces + p]`, `off` elements into it. Chunk `c` is chunk
// `c % seg_chunks` of segment `c / seg_chunks` (`segs[...]`, a byte address),
// `chunk` elements apart; with no segment table every chunk is in the one
// segment at `base`. A negative first id names no row.
//
// One segment, one piece of every atom, the slot as the chunk and its stride
// as the chunk's span is the pool's slot layout: `base + slot * stride`.
struct StateRows {
    u64 base;
    const u64* segs;
    const i32* ids;
    i64 chunk;
    i64 off;
    i32 seg_chunks;
    i32 pieces;
    i32 atoms;
    i32 pad_;

    // Whether the fire names a pool at all.
    __device__ __forceinline__ bool live() const {
        return base != 0 || segs != nullptr;
    }

    // The ids of row `r`'s pieces.
    __device__ __forceinline__ const i32* row(int r) const {
        return ids + static_cast<i64>(r) * pieces;
    }

    // Row `r`'s first chunk id: negative where the lane holds no state.
    __device__ __forceinline__ int first(int r) const {
        return ids[static_cast<i64>(r) * pieces];
    }

    // The first element of the piece chunk `cid` holds.
    template <class T>
    __device__ __forceinline__ T* chunk_at(int cid) const {
        if (segs == nullptr) {
            return reinterpret_cast<T*>(base) + static_cast<i64>(cid) * chunk + off;
        }
        const int s = cid / seg_chunks;
        return reinterpret_cast<T*>(segs[s])
            + static_cast<i64>(cid - s * seg_chunks) * chunk + off;
    }

    // The first element of piece `p` of the row whose ids are `rid`.
    template <class T>
    __device__ __forceinline__ T* piece(const i32* rid, int p) const {
        return chunk_at<T>(rid[p]);
    }

    // The first element of atom `a` of the row whose ids are `rid`, atoms
    // `atom_elems` apart inside a piece.
    template <class T>
    __device__ __forceinline__ T* atom(const i32* rid, int a, i64 atom_elems) const {
        const int p = a / atoms;
        return piece<T>(rid, p) + static_cast<i64>(a - p * atoms) * atom_elems;
    }
};

}
