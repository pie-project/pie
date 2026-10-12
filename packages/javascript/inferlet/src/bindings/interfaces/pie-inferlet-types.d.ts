/** @module Interface pie:inferlet/types@0.3.0 **/
export type Error = string;
export type Blob = Uint8Array;
/**
 * ── Shape-typed tensor primitives (absorbed from the former `tensor`
 * interface; shared with the `forward` interface via
 * `use types.{shape, dtype, data}`). These are pure type aliases — the
 * device-resident tensor/program resources (the removed PSIR sampler front
 * door) are gone; programmable sampling now lives entirely in `forward`.
 * Tensor dimensions. Rank is `shape.len()`: `[]` is scalar, `[n]` is a
 * vector, `[m, n]` is a matrix. All dimensions are static; dynamic/ragged
 * buffers are not represented.
 */
export type Shape = Uint32Array;
/**
 * Element type of a tensor payload. Closed set, no payloads — an `enum`,
 * like `model.forward-kind`.
 * # Variants
 * 
 * ## `"f32"`
 * 
 * ## `"i32"`
 * 
 * ## `"u32"`
 * 
 * ## `"bool"`
 */
export type Dtype = 'f32' | 'i32' | 'u32' | 'bool';
/**
 * Packed little-endian tensor payload. The runtime validates byte length
 * against the tensor shape/dtype.
 */
export type Data = Uint8Array;
