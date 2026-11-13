#include <metal_stdlib>
using namespace metal;

// How a fixed state bank is addressed: through a segment table and a chunk id
// per piece of each lane's record, never as `slot * stride` into one slab.
//
// `map` is a small device table of ulongs, in elements of the bank it rides
// with (the bound bank pointer is the origin of every segment):
//
//   [0] slot_stride   elements from one slot of a chunk to the next
//   [1] chunk_slots   slots one chunk holds
//   [2] chunk_stride  elements from one chunk of a segment to the next
//   [3] seg_chunks    chunks one segment holds
//   [4] piece_atoms   atoms one piece of a record holds (a head of GDN, KDA
//                     or Mamba-2, a channel of a conv, a frame of a frame
//                     cache); a piece never cuts an atom
//   [5] pieces        pieces one record is cut into
//   [6] [7] [8]       chunk_slots, seg_chunks and piece_atoms as divisor
//                     words (`kernels_metal::tensor::divisor_word`): every
//                     division a lookup takes is a multiply-high and two
//                     shifts, not a runtime integer division
//   [9 + s]           segment s's first element
//
// A record's ids are `pieces` consecutive ids, one a piece: a lane's row of
// the slot table is `slots + lane * pieces`, a snapshot's
// `snap + PIE_RS_SNAPSHOTS + n * pieces`.
//
// The degenerate table, `{stride, 1, stride, ~0u, ~0u, 1, ..., 0}` (one segment,
// a chunk a slot, one piece a record), puts slot `id` at `id * stride` with
// its atoms in order: the slab every kernel read before, bit for bit.

#define PIE_STATE_MAP_HEAD 9

// `n / d` from `d`'s divisor word (exact for every `n`).
inline uint pie_div(uint n, ulong word) {
  const uint t = mulhi(n, uint(word));
  return (t + ((n - t) >> (uint(word >> 32) & 0xffu))) >> (uint(word >> 40) & 0xffu);
}

inline uint pie_state_pieces(const device ulong* map) {
  return uint(map[5]);
}

inline uint pie_state_piece_atoms(const device ulong* map) {
  return uint(map[4]);
}

// The first element of the slot `id` names.
inline ulong pie_state_slot(const device ulong* map, uint id) {
  const uint chunk = pie_div(id, map[6]);
  const uint seg = pie_div(chunk, map[7]);
  return map[PIE_STATE_MAP_HEAD + seg] + ulong(chunk - seg * uint(map[3])) * map[2] +
         ulong(id - chunk * uint(map[1])) * map[0];
}

// The first element of atom `atom` (each `atom_elems` long, atoms outermost
// in a piece) of the record whose piece ids start at `ids`.
template <typename I>
inline ulong pie_state_atom(const device ulong* map, const device I* ids, uint atom,
                            ulong atom_elems) {
  const uint piece = pie_div(atom, map[8]);
  return pie_state_slot(map, uint(ids[piece])) +
         ulong(atom - piece * pie_state_piece_atoms(map)) * atom_elems;
}

// Where atom `atom` of a record whose atom axis is innermost (a conv's
// `[history][channels]`) sits: the piece's first element, the piece's row
// pitch in atoms and the atom's column in it. A piece of a record with
// `atoms` atoms in all is `[rows][min(piece_atoms, atoms)]`.
struct pie_state_column {
  ulong base;
  uint pitch;
  uint col;
};

template <typename I>
inline pie_state_column pie_state_inner(const device ulong* map, const device I* ids, uint atom,
                                        uint atoms) {
  const uint per = pie_state_piece_atoms(map);
  const uint piece = pie_div(atom, map[8]);
  return pie_state_column{pie_state_slot(map, uint(ids[piece])), min(per, atoms),
                          atom - piece * per};
}

// A lane's row of a snapshot table: `PIE_RS_SNAPSHOTS` rows into the lane,
// then that many records' piece ids.
inline uint pie_state_snap_row(const device ulong* map) {
  return 1 + pie_state_pieces(map);
}
