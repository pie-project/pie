#pragma once

// A decode step's carried rows: the rows a lane buffered past its committed
// state and replays ahead of its own, read from its rs buffer pages through
// the staged tables, and the length of its run (carried then own) the state
// is committed at. `kernels-cuda`'s `ssm/carry.cuh` lays the same bytes; the
// engine stages one set of tables for both.

#include "prim/device.hip"

namespace pie::attn {

// A lane of the staged rs tables, as `runtime/channel.hip`'s `RsLane` lays
// it: where the lane's rows land in its rs buffer pages, how many buffered
// rows it carries ahead of its own (and where their tokens start in the
// replay table), and its place in the fire's extended run.
struct CarryLane {
    u32 at;
    u32 replay;
    u32 replay_off;
    u32 page_off;
    u32 ext;
};

static_assert(sizeof(CarryLane) == 20, "CarryLane: `channel::RsLane` mirrors this layout");

// A beam group's fold (BEAM-PAGE-LISTS §11.3): its representative lane (any
// of its lanes; they share the set's state), where its fold rows start in
// the replay table, and how many. The table's folding groups lead it; the
// first entry of no rows ends it.
struct GroupFold {
    u32 lane;
    u32 fold_off;
    u32 rows;
};

static_assert(sizeof(GroupFold) == 12, "GroupFold: the Rust `ssm::GroupFold` mirrors this layout");

constexpr u32 CARRY_NONE = 0xffffffffu;

// The Rust `ssm::Carry`, passed by value. `lanes` null: the fire buffers
// nothing, every lane carries none and commits all it runs. `plane_off[i]`
// and `plane_stride[i]` place the op's i-th buffered plane in a page's slot;
// Four planes at most: a KDA layer reads its forget and beta rows, keys
// and corrected values.
struct Carry {
    const CarryLane* lanes;
    const u64* pages;
    const u32* replay_tok;
    const int* commit_len;
    u64 plane_off[4];
    u64 plane_stride[4];
    u32 page_tokens;
    u32 pad_;

    __device__ __forceinline__ CarryLane lane(int rl) const {
        if (lanes == nullptr) return CarryLane{CARRY_NONE, 0u, 0u, 0u, 0u};
        return lanes[rl];
    }

    // The rows lane `l` carries ahead of its own.
    __device__ __forceinline__ int carried(const CarryLane& l) const { return static_cast<int>(l.replay); }

    // How many rows of the lane's run (`run` long: carried then own) the
    // state is committed at.
    __device__ __forceinline__ int commit(int rl, int run) const {
        if (commit_len == nullptr) return run;
        const int c = commit_len[rl];
        return c < run ? c : run;
    }

    // The page slot buffer token `tok` lies in, and its row in it.
    __device__ __forceinline__ void slot_of(const CarryLane& l, u32 tok, u64& slot, u32& row) const {
        slot = pages[l.page_off + tok / page_tokens];
        row = tok % page_tokens;
    }

    template <class T>
    __device__ __forceinline__ T* plane_at(u64 slot, u32 row, int p) const {
        return reinterpret_cast<T*>(slot + plane_off[p] + static_cast<u64>(row) * plane_stride[p]);
    }

    // The page slot and row of the lane's j-th carried row.
    __device__ __forceinline__ void carried_at(const CarryLane& l, int j, u64& slot, u32& row) const {
        slot_of(l, replay_tok[l.replay_off + j], slot, row);
    }

    // The page slot and row of a group fold's j-th row (§11.3): token
    // `replay_tok[fold_off + j]` through the representative lane's pages.
    __device__ __forceinline__ void fold_at(const CarryLane& l, u32 fold_off, int j, u64& slot, u32& row) const {
        slot_of(l, replay_tok[fold_off + j], slot, row);
    }

    // Plane `p` of the lane's j-th carried row.
    template <class T>
    __device__ __forceinline__ const T* carried_row(const CarryLane& l, int j, int p) const {
        u64 slot;
        u32 row;
        carried_at(l, j, slot, row);
        return plane_at<const T>(slot, row, p);
    }

    // Plane `p` of the row the lane's own row `e` lands at; null where the
    // lane lands none.
    template <class T>
    __device__ __forceinline__ T* own_row(const CarryLane& l, int e, int p) const {
        if (lanes == nullptr || l.at == CARRY_NONE) return nullptr;
        u64 slot;
        u32 row;
        slot_of(l, l.at + static_cast<u32>(e), slot, row);
        return plane_at<T>(slot, row, p);
    }

};

}  // namespace pie::attn
