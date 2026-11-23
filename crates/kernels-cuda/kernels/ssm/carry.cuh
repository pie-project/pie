#pragma once

#include "ssm/state.cuh"
#include "runtime/rs_rows.cuh"

namespace pie::ssm {

// A lane of the staged rs tables, as `runtime/channel.cu`'s `RsLane` lays
// it: where the lane's rows land in its rs buffer pages, how many buffered
// rows it carries ahead of its own (and where their spans start in the
// replay table), and its place in the fire's extended run.
struct CarryLane {
    u32 at;
    u32 replay;
    u32 replay_off;
    u32 page_off;
    u32 ext;
};

static_assert(sizeof(CarryLane) == 20, "CarryLane: `channel::RsLane` mirrors this layout");

// A beam group's fold (BEAM-PAGE-LISTS §11.3): its representative lane, where
// its fold rows' spans start in the replay table, and how many rows (none
// folds nothing).
struct GroupFold {
    u32 lane;
    u32 fold_off;
    u32 rows;
};

static_assert(sizeof(GroupFold) == 12, "GroupFold: `ssm::GroupFold` mirrors this layout");

constexpr u32 CARRY_NONE = 0xffffffffu;

// A lane's `page_off` with this bit set names its entry of the fire's lane
// rows (`Carry::lane_rows`, `RsRows`) in its low bits: its rows resolve on
// the device, through its own pages or a beam's group's list.
constexpr u32 CARRY_ROWS = 0x80000000u;

// The buffer token of the j-th row of a run of spans from `off`: the replay
// table holds (start, count) pairs of buffer tokens, in order, which
// together count the run's rows (a lane alone on its set, or a beam over its
// own list, replays one span; a lane sharing its set by mask, the spans its
// mask admits).
__device__ __forceinline__ u32 span_tok(const u32* spans, u32 off, u32 j)
{
    for (;; off += 2u) {
        const u32 n = spans[off + 1u];
        if (j < n) return spans[off] + j;
        j -= n;
    }
}

// A decode step's carried rows (the Rust `ssm::Carry`): the rows a lane
// buffered past its committed state and reads ahead of its own, from its
// rs buffer pages through the staged tables, and the length of its run
// (carried then own) the state is committed at. `lanes` null: the fire
// buffers nothing, every lane carries none and commits all it runs.
//
// `plane[i]` is where the op's i-th buffered plane lies in a page's slot.
struct Carry {
    const CarryLane* lanes;
    const u64* pages;
    // Each lane's replayed rows as (start, count) spans ([`span_tok`]).
    const u32* replay_spans;
    const int* commit_len;
    u64 plane_off[4];
    u64 plane_stride[4];
    u32 page_tokens;
    u32 pad_;
    // A decode step's room to split a lane's carried rows across blocks:
    // a counter a unit, then each part's partial sums (`gdn_step_carried`);
    // null where the step splits none.
    unsigned char* split;
    // The fire's lanes' rows (`RsRows`), read by a lane whose `page_off`
    // carries `CARRY_ROWS`.
    const RsRows* lane_rows;

    // The page slot buffer token `tok` of lane `l` lies in, and its row in it.
    __device__ __forceinline__ void locate(const CarryLane& l, u32 tok, u64& slot, u32& row) const {
        if (l.page_off & CARRY_ROWS) {
            lane_rows[l.page_off & ~CARRY_ROWS].locate(tok, slot, row);
            return;
        }
        slot = pages[l.page_off + tok / page_tokens];
        row = tok % page_tokens;
    }

    __device__ __forceinline__ CarryLane lane(int rl) const {
        if (lanes == nullptr) return CarryLane{CARRY_NONE, 0u, 0u, 0u, 0u};
        return lanes[rl];
    }

    // The rows lane `l` carries ahead of its own.
    __device__ __forceinline__ int carried(const CarryLane& l) const {
        return static_cast<int>(l.replay);
    }

    // How many rows of the lane's run (`run` long: carried then own) the
    // state is committed at.
    __device__ __forceinline__ int commit(int rl, int run) const {
        if (commit_len == nullptr) return run;
        const int c = commit_len[rl];
        return c < run ? c : run;
    }

    // The page row buffer token `tok` of plane `p` lies at.
    template <class T>
    __device__ __forceinline__ T* at_tok(const CarryLane& l, u32 tok, int p) const {
        u64 slot;
        u32 row;
        locate(l, tok, slot, row);
        return reinterpret_cast<T*>(slot + plane_off[p] + static_cast<u64>(row) * plane_stride[p]);
    }

    // The page row of the lane's j-th carried row of plane `p`.
    template <class T>
    __device__ __forceinline__ const T* carried_row(const CarryLane& l, int j, int p) const {
        return at_tok<const T>(l, span_tok(replay_spans, l.replay_off, static_cast<u32>(j)), p);
    }

    // The page row the lane's own row `e` lands at, of plane `p`; null where
    // the lane lands none.
    template <class T>
    __device__ __forceinline__ T* own_row(const CarryLane& l, int e, int p) const {
        if (lanes == nullptr || l.at == CARRY_NONE) return nullptr;
        return at_tok<T>(l, l.at + static_cast<u32>(e), p);
    }

    // The page slot of the lane's j-th carried row, and its token in it:
    // what `plane_at` reads any plane of the row from.
    __device__ __forceinline__ void carried_at(const CarryLane& l, int j, u64& slot, u32& tok) const {
        locate(l, span_tok(replay_spans, l.replay_off, static_cast<u32>(j)), slot, tok);
    }

    // The page slot of a beam group's j-th fold row (BEAM-PAGE-LISTS §11.3)
    // and its token in it: the j-th of the spans from `fold_off`, read through
    // the group's representative lane's rs pages as a carried row is.
    __device__ __forceinline__ void fold_at(const CarryLane& l, u32 fold_off, int j, u64& slot, u32& tok) const {
        locate(l, span_tok(replay_spans, fold_off, static_cast<u32>(j)), slot, tok);
    }

    template <class T>
    __device__ __forceinline__ T* plane_at(u64 slot, u32 tok, int p) const {
        return reinterpret_cast<T*>(slot + plane_off[p] + static_cast<u64>(tok) * plane_stride[p]);
    }
};

}
