// Packed-bitmask logit-mask semantics — port of `crates/inferlet/src/mask.rs`,
// byte-identical to the engine's `0x65 MaskApply` op.
//
// A logit mask is one bit per vocabulary token, packed into `ceil(vocab/32)`
// u32 words: bit `1` = allowed. Token `j`'s bit is word `j >>> 5`, bit
// `j & 31`. The host grammar matcher hands such a mask out; `unpackMask`
// turns it into the `[vocab] bool` cell a `maskedArgmax` epilogue reads.

/** Number of u32 words a packed mask for `vocab` tokens occupies. */
export function maskWords(vocab: number): number {
  return Math.ceil(vocab / 32);
}

/** An all-allowed packed mask (every bit `1`). */
export function allAllowed(vocab: number): Uint32Array {
  return new Uint32Array(maskWords(vocab)).fill(0xffff_ffff);
}

/** Whether token `j` is allowed; tokens past the mask's coverage read as
 * disallowed (a padded output vocabulary decodes to no token there). */
export function bitAllowed(mask: ArrayLike<number>, j: number): boolean {
  const word = j >>> 5;
  return word < mask.length && ((mask[word] >>> (j & 31)) & 1) === 1;
}

/** Pack an allowed-token id list into a packed bitmask; ids `>= vocab` are ignored. */
export function packAllowed(vocab: number, allowed: ArrayLike<number>): Uint32Array {
  const mask = new Uint32Array(maskWords(vocab));
  for (let i = 0; i < allowed.length; i++) {
    const j = allowed[i];
    if (j >= 0 && j < vocab) mask[j >>> 5] |= 1 << (j & 31);
  }
  return mask;
}

/** Expand a packed mask into one boolean per token. An empty `packed` means
 * the constraint is inactive, so everything is allowed. */
export function unpackMask(packed: ArrayLike<number>, vocab: number): boolean[] {
  if (packed.length === 0) return new Array<boolean>(vocab).fill(true);
  const out = new Array<boolean>(vocab);
  for (let j = 0; j < vocab; j++) out[j] = bitAllowed(packed, j);
  return out;
}

/** Argmax over `logits` with the packed mask applied (disallowed = -inf;
 * ties to the lowest index; all-disallowed returns 0). */
export function applyMaskArgmax(logits: ArrayLike<number>, mask: ArrayLike<number>): number {
  let bestIdx = 0;
  let bestVal = -Infinity;
  for (let j = 0; j < logits.length; j++) {
    const v = bitAllowed(mask, j) ? logits[j] : -Infinity;
    if (v > bestVal) {
      bestVal = v;
      bestIdx = j;
    }
  }
  return bestIdx;
}
