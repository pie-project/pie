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
export type Dtype = DtypeF32 | DtypeI32 | DtypeU32 | DtypeBool;
export interface DtypeF32 {
  tag: 'f32',
}
export interface DtypeI32 {
  tag: 'i32',
}
export interface DtypeU32 {
  tag: 'u32',
}
export interface DtypeBool {
  tag: 'bool',
}
/**
 * Packed little-endian tensor payload. The runtime validates byte length
 * against the tensor shape/dtype.
 */
export type Data = Uint8Array;
