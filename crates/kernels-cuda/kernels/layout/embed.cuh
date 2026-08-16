//===-- embed.cuh - the token-embedding gathers ----------------------===//
//
// Two `__global__`s: the flat embedding lookup and the vocab-sharded one.
// `embed.cu` includes this file and keeps both launchers, so exactly ONE
// definition of each exists in the tree -- a split and not a copy, because
// two definitions that agree today are two that disagree after the first
// edit, and `norm/altup_aux` shipped a release proving it.
//
// # Only one of the two is a row
//
// `embed_vocab_shard` is templated over `T` and instantiated by a row:
// one block per token, `LaunchRule::RouteRows`, no host arithmetic anywhere
// in its launcher. `Elem<T>::from_f32(0.0f)` replaced a bare
// `f32_to_bf16(0.0f)` -- the zero an out-of-shard token writes has to be the
// zero of whatever `T` is, and `elementwise.cuh` documents the conversion
// trick.
//
// `embed` is NOT a row and is not templated. Its `VEC` parameter is chosen on
// the HOST from a run-time test the device cannot make -- `hidden % 8 == 0`
// and both `weight` and `y` 16-byte aligned -- and the element count it
// launches over is `num_tokens * (vec ? hidden/8 : hidden)`, an extent that
// depends on the answer. No `Source` in `kernels/src/lib.rs` produces "is
// this pointer 16-byte aligned", and `new-horizon.md` §10.5 refuses an
// invented one. It stays here because the file is the family's device text,
// not because a row will find it.
//
// The vectorised form is not an optimisation to drop: at decode the
// token-per-block form issued 24 dependent 2-byte loads from 8 blocks and ran
// at 8 GB/s -- the row it reads is a random offset into the largest tensor in
// the model, so the access is a cold TLB miss whose latency only a wide grid
// hides.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

// `Elem<T>` names the prelude's, which is what the kernels below spell.
template <class T>
using Elem = ::pie::Elem<T>;

// One block per token. Threads stride across `hidden`. Bounds-clamp the
// token id so a runaway wire payload can't OOB-read. (Out-of-vocab → 0 row.)
//
// `VEC` gathers eight bf16 per thread through a 16-byte load. The row this
// reads is a random offset into a table that is the largest tensor in the
// model, so the access is a cold TLB miss whose latency only a wide grid
// hides: at decode the token-per-block form issued 24 dependent 2-byte loads
// from 8 blocks and ran at 8 GB/s. Flattening (token, chunk) into one grid
// gives the memory system every row at once. Pure copy either way.
template <bool VEC>
__global__ void embed(
    const i32* __restrict__ token_ids,
    const bf16* __restrict__ weight,
    bf16* __restrict__ y,
    int hidden, int vocab, int num_tokens, int per_row)
{
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * per_row) return;
    const int n = idx / per_row;
    const int h = idx % per_row;

    const i32 tid_raw = token_ids[n];
    const int tid = (tid_raw >= 0 && tid_raw < vocab) ? tid_raw : 0;
    const bf16* row = weight + static_cast<long long>(tid) * hidden;
    bf16* out = y + static_cast<long long>(n) * hidden;

    if constexpr (VEC) {
        reinterpret_cast<float4*>(out)[h] =
            reinterpret_cast<const float4*>(row)[h];
    } else {
        out[h] = row[h];
    }
}

/// Tensor-parallel embedding: this rank owns rows `[vocab_offset,
/// vocab_offset + local_vocab)` and writes ZERO for every other token, so an
/// all-reduce over the ranks reconstructs the full table. One block per token
/// -- `LaunchRule::RouteRows`.
template <class T>
__global__ void embed_vocab_shard(
    const i32* __restrict__ token_ids,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden, int local_vocab, int vocab_offset)
{
    const int n = blockIdx.x;
    const i32 tid_raw = token_ids[n];
    const int local_tid = tid_raw - vocab_offset;
    const bool in_shard = local_tid >= 0 && local_tid < local_vocab;
    const T* row =
        weight + static_cast<long long>(in_shard ? local_tid : 0) * hidden;
    T* out = y + static_cast<long long>(n) * hidden;

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        out[h] = in_shard ? row[h] : Elem<T>::from_f32(0.0f);
    }
}

}  // namespace pie::layout
