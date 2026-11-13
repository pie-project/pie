#pragma once

#include <metal_stdlib>

// The weight cache's expert table (`kernels_metal::linear::moe::BankTable`):
// one GPU address a piece of an expert of a routed bank's plane, entry
// `expert * pieces + piece`; piece `p` holds the plane's output rows
// `p * piece_rows ..` of the expert, laid as in the bank.
struct ExpertPlaneAt {
  const device char *at;
};

// Where the kernel addresses expert `e`'s plane as if it were contiguous:
// `bank + e * expert_bytes` where `pieces == 0`, else the address that puts
// output row `row`'s piece where the bank would (its first row less the
// rows before the piece), so every row of that piece is read from the
// address its offset from this base names. A kernel that reads rows
// `row .. row + n` this way keeps them inside one piece.
inline const device char *expert_plane(const device char *bank, const device ExpertPlaneAt *table,
                                       int pieces, int piece_rows, ulong expert_bytes,
                                       ulong row_bytes, uint e, uint row) {
  if (pieces <= 0) {
    return bank + ulong(e) * expert_bytes;
  }
  const uint piece = row / uint(piece_rows);
  return table[e * uint(pieces) + piece].at - ulong(piece) * ulong(piece_rows) * row_bytes;
}

// A batched routed kernel's table arguments (after its tile experts), and
// their types for an explicit instantiation.
#define PIE_EXPERT_TABLE_ARGS                                                  \
  const device ExpertPlaneAt *codes_at [[buffer(13)]],                         \
      const device ExpertPlaneAt *scales_at [[buffer(14)]],                    \
      const device ExpertPlaneAt *biases_at [[buffer(15)]],                    \
      const device ExpertPlaneAt *bias_at [[buffer(16)]],                      \
      const constant int &pieces [[buffer(17)]],                               \
      const constant int &piece_rows [[buffer(18)]]
#define PIE_EXPERT_TABLE_TYPES                                                 \
  const device ExpertPlaneAt *, const device ExpertPlaneAt *,                  \
      const device ExpertPlaneAt *, const device ExpertPlaneAt *,              \
      const constant int &, const constant int &
