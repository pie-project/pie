//! The interpreter's SSA value cell and the channel wire codec.
//!
//! A running program moves tensors between ops and through channel rings. This
//! module owns the two shapes those tensors take: [`Value`], the in-memory cell
//! an op reads and writes, and the wire encoding a channel ring stores.
//!
//! # Why a dedicated cell type rather than [`driver_api::plan::LaunchValue`]
//!
//! `LaunchValue` is a *declaration* — a value id, a dtype byte, a shape. It says
//! nothing about the numbers a value holds at run time. [`Value`] is the
//! run-time counterpart: the actual lane vector. Keeping them separate is what
//! lets the launch package stay a plain description the driver never mutates
//! while the interpreter churns through [`Value`]s.
//!
//! # The `Act` dtype never reaches a cell
//!
//! Channel declarations may carry the late-bound activation dtype
//! ([`PIE_CHANNEL_DTYPE_ACT`], wire byte `4`), which programs see as `f32`.
//! [`tensor_ir::DType`] has no `Act` variant precisely because a *value* is
//! never `Act` — it is always one of the four concrete types. So every dtype
//! byte that crosses into this module is folded through [`concrete_dtype`]
//! first, and `Act` becomes [`DType::F32`] at that single point. Carrying an
//! `Act` variant into the cell would force every arithmetic arm to re-answer
//! "but what if it is `Act`?" when the answer is always "treat it as `f32`".
//!
//! # The bool asymmetry is load-bearing
//!
//! A bool tensor is **bit-packed on the wire** (8 lanes per byte) but
//! **one byte per lane in memory** ([`Value::Bool`]). The wire form is what a
//! future generated kernel reads directly out of a shared channel buffer, so it
//! must stay dense; the in-memory form is what the elementwise ops index lane by
//! lane, so it must stay addressable. [`decode_wire`] and [`encode_wire`] are
//! the only places that bridge the two, and a test pins the asymmetry because
//! getting it wrong is invisible until a bool channel is read back.

use driver_api::local::PIE_CHANNEL_DTYPE_ACT;
use tensor_ir::DType;

/// The concrete cell dtype a wire dtype byte names.
///
/// Folds the activation dtype ([`PIE_CHANNEL_DTYPE_ACT`]) to [`DType::F32`],
/// which is how programs observe it, and defaults any unrecognised byte to
/// [`DType::F32`] as well. The default is not a guess about correctness: the
/// launch package is ABI-validated before it reaches the driver, so an
/// out-of-range dtype byte cannot occur in a real package; mapping it to `f32`
/// keeps this a total function so callers never have to thread an error through
/// a value they know is well-typed.
#[must_use]
pub fn concrete_dtype(byte: u8) -> DType {
    match DType::from_wire(byte) {
        Some(dtype) => dtype,
        // `Act` (byte 4) and any other out-of-range byte materialize as `f32`.
        None if byte == PIE_CHANNEL_DTYPE_ACT => DType::F32,
        None => DType::F32,
    }
}

/// One SSA value / channel cell: a lane vector tagged by its dtype.
///
/// An enum rather than a struct-of-four-vectors (the C++ shape) because exactly
/// one lane vector is ever live, and the enum makes that unrepresentable-if-
/// wrong: there is no way to hold `i32` lanes while claiming to be `f32`. Bool
/// lanes are one byte each (`0`/`1`), never bit-packed — see the module docs.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// 32-bit float lanes.
    F32(Vec<f32>),
    /// Signed 32-bit integer lanes.
    I32(Vec<i32>),
    /// Unsigned 32-bit integer lanes.
    U32(Vec<u32>),
    /// Boolean lanes, one byte per lane (`0` or `1`).
    Bool(Vec<u8>),
}

impl Value {
    /// This cell's element type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Value::F32(_) => DType::F32,
            Value::I32(_) => DType::I32,
            Value::U32(_) => DType::U32,
            Value::Bool(_) => DType::Bool,
        }
    }

    /// The number of lanes.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Value::F32(v) => v.len(),
            Value::I32(v) => v.len(),
            Value::U32(v) => v.len(),
            Value::Bool(v) => v.len(),
        }
    }

    /// Whether the cell has no lanes. Present because [`clippy`] asks any type
    /// with [`Value::len`] to offer it; a zero-lane cell is a real state (an
    /// empty reduction result), not just an API formality.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A zero-filled cell of `numel` lanes (at least one).
    ///
    /// The floor of one lane mirrors the reference interpreter: a scalar is a
    /// one-lane tensor, and a zero-length cell would make every broadcast that
    /// reads lane `0` index out of bounds. The default arm is `f32`, matching
    /// the wire decoder's fallback.
    #[must_use]
    pub fn zeros(dtype: DType, numel: usize) -> Value {
        let n = numel.max(1);
        match dtype {
            DType::I32 => Value::I32(vec![0; n]),
            DType::U32 => Value::U32(vec![0; n]),
            DType::Bool => Value::Bool(vec![0; n]),
            DType::F32 => Value::F32(vec![0.0; n]),
        }
    }

    /// The lanes reinterpreted as `f32`, the common numeric working type.
    ///
    /// Integer and bool lanes are *converted*, not bit-cast: `bool` becomes
    /// `0.0`/`1.0` and integers round-trip through `as f32`. This is the input
    /// side of every float-path op, so it never fails — a caller that reaches a
    /// float op with a bool operand still gets sensible `0.0`/`1.0` lanes.
    #[must_use]
    pub fn lanes_f32(&self) -> Vec<f32> {
        match self {
            Value::F32(v) => v.clone(),
            Value::I32(v) => v.iter().map(|&x| x as f32).collect(),
            Value::U32(v) => v.iter().map(|&x| x as f32).collect(),
            Value::Bool(v) => v.iter().map(|&x| if x != 0 { 1.0 } else { 0.0 }).collect(),
        }
    }

    /// The lanes widened to `i64`, the common integer working type.
    ///
    /// `i64` rather than `i32` so the integer arithmetic ops have headroom for
    /// intermediate sums without wrapping before the result is narrowed back to
    /// the declared dtype by [`Value::from_i64`]. Float lanes truncate toward
    /// zero, matching a C++ `static_cast<int64_t>`.
    #[must_use]
    pub fn lanes_i64(&self) -> Vec<i64> {
        match self {
            Value::I32(v) => v.iter().map(|&x| i64::from(x)).collect(),
            Value::U32(v) => v.iter().map(|&x| i64::from(x)).collect(),
            Value::Bool(v) => v.iter().map(|&x| i64::from(x != 0)).collect(),
            Value::F32(v) => v.iter().map(|&x| x as i64).collect(),
        }
    }

    /// Narrow `i64` working lanes back into a cell of `dtype`.
    ///
    /// The inverse of [`Value::lanes_i64`] for the integer ops. `Bool` maps
    /// `x != 0`; the float arm converts rather than reinterprets so an integer
    /// op that declares an `f32` result (the reference interpreter allows it)
    /// still yields the numeric value, not its bit pattern.
    #[must_use]
    pub fn from_i64(dtype: DType, x: &[i64]) -> Value {
        match dtype {
            DType::U32 => Value::U32(x.iter().map(|&v| v as u32).collect()),
            DType::Bool => Value::Bool(x.iter().map(|&v| u8::from(v != 0)).collect()),
            DType::F32 => Value::F32(x.iter().map(|&v| v as f32).collect()),
            DType::I32 => Value::I32(x.iter().map(|&v| v as i32).collect()),
        }
    }
}

/// The scalar-broadcast index rule shared by every elementwise op: a length-one
/// operand contributes lane `0` to every output lane, otherwise lanes line up.
///
/// A free function, not inlined at each call site, because "does this operand
/// broadcast?" is one decision and every binary op must make it the same way —
/// a site that forgets the `len == 1` case silently reads out of bounds on a
/// scalar operand.
#[must_use]
pub fn pick(len: usize, i: usize) -> usize {
    if len == 1 { 0 } else { i }
}

/// Whether a cell matches a declared `(dtype byte, shape)` — the same dtype
/// (folding `Act`) and the same lane count (floored at one).
///
/// The floor of one lane matches [`Value::zeros`]: a scalar declaration has
/// `numel == 1`, and an empty declared shape still expects a one-lane cell.
/// Used to gate host puts against a channel's declared type so a wrong-shaped
/// value is refused at the boundary rather than corrupting a ring cell.
#[must_use]
pub fn value_matches(v: &Value, dtype_byte: u8, dims: &[u32]) -> bool {
    v.dtype() == concrete_dtype(dtype_byte) && v.len() as u64 == super::shape_numel(dims).max(1)
}

/// The wire byte count of a cell of `numel` lanes and `dtype`.
///
/// Bool packs 8 lanes per byte (`ceil(numel / 8)`); every other dtype is 4
/// bytes per lane. This is the one size both ends of the codec must agree on,
/// so [`decode_wire`] checks the incoming slice against it rather than trusting
/// the length.
#[must_use]
pub fn wire_cell_bytes(dtype: DType, numel: usize) -> usize {
    if dtype == DType::Bool {
        numel.div_ceil(8)
    } else {
        numel * 4
    }
}

/// Decode a wire cell into a [`Value`], or `None` if the byte length does not
/// match the declared shape.
///
/// Returns `Option` rather than the C++ out-param-plus-bool: a length mismatch
/// is the *only* failure, and a caller either has a value or does not. Bool
/// lanes are unpacked from bits to bytes here — the inverse of [`encode_wire`]
/// — which is the whole reason a bool channel can be indexed lane by lane
/// downstream.
#[must_use]
pub fn decode_wire(bytes: &[u8], dtype: DType, numel: usize) -> Option<Value> {
    if bytes.len() != wire_cell_bytes(dtype, numel) {
        return None;
    }
    Some(match dtype {
        DType::Bool => {
            let mut out = vec![0u8; numel];
            for (j, lane) in out.iter_mut().enumerate() {
                *lane = (bytes[j / 8] >> (j % 8)) & 1;
            }
            Value::Bool(out)
        }
        DType::I32 => Value::I32(
            bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        DType::U32 => Value::U32(
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        DType::F32 => Value::F32(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
    })
}

/// Encode a [`Value`] into `dst`, which must be exactly
/// [`wire_cell_bytes`] long for the value's dtype and lane count.
///
/// The bool arm bit-packs (and must zero the destination first, since a lane of
/// `0` writes no bit); every other arm is a little-endian lane copy. Metal's
/// only targets are little-endian, so the explicit `to_le_bytes` is the wire
/// contract, not a byte-order guess.
///
/// # Panics
///
/// If `dst` is shorter than [`wire_cell_bytes`] for this value. Callers size
/// the buffer from the same function, so a short slice is a caller bug, not
/// input the codec should tolerate.
pub fn encode_wire(v: &Value, dst: &mut [u8]) {
    match v {
        Value::Bool(b) => {
            let packed = b.len().div_ceil(8);
            for byte in &mut dst[..packed] {
                *byte = 0;
            }
            for (j, &lane) in b.iter().enumerate() {
                if lane != 0 {
                    dst[j / 8] |= 1u8 << (j % 8);
                }
            }
        }
        Value::I32(v) => copy_le(v.iter().flat_map(|x| x.to_le_bytes()), dst),
        Value::U32(v) => copy_le(v.iter().flat_map(|x| x.to_le_bytes()), dst),
        Value::F32(v) => copy_le(v.iter().flat_map(|x| x.to_le_bytes()), dst),
    }
}

/// Write a little-endian byte stream into `dst` lane by lane.
fn copy_le(bytes: impl Iterator<Item = u8>, dst: &mut [u8]) {
    for (slot, byte) in dst.iter_mut().zip(bytes) {
        *slot = byte;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_activation_dtype_byte_folds_to_f32_because_a_cell_is_never_act() {
        assert_eq!(
            concrete_dtype(PIE_CHANNEL_DTYPE_ACT),
            DType::F32,
            "Act must present as f32; a cell that stayed Act would force every op to special-case it"
        );
        assert_eq!(concrete_dtype(0), DType::F32);
        assert_eq!(concrete_dtype(1), DType::I32);
        assert_eq!(concrete_dtype(2), DType::U32);
        assert_eq!(concrete_dtype(3), DType::Bool);
    }

    #[test]
    fn a_bool_cell_is_bit_packed_on_the_wire_and_byte_per_lane_in_memory() {
        // Lanes 0,2,4,6,8 set: in memory that is nine bytes; on the wire it is
        // two bytes, 0b0101_0101 then 0b0000_0001.
        let lanes = vec![1u8, 0, 1, 0, 1, 0, 1, 0, 1];
        let value = Value::Bool(lanes.clone());
        assert_eq!(
            wire_cell_bytes(DType::Bool, 9),
            2,
            "nine bool lanes must pack into ceil(9/8)=2 wire bytes, not nine"
        );
        let mut wire = vec![0u8; 2];
        encode_wire(&value, &mut wire);
        assert_eq!(
            wire,
            vec![0b0101_0101u8, 0b0000_0001u8],
            "bit j of byte j/8 must carry lane j; a byte-per-lane wire form would break kernel reads"
        );
        let decoded = decode_wire(&wire, DType::Bool, 9).expect("length matches");
        assert_eq!(
            decoded,
            Value::Bool(lanes),
            "decode must unpack bits back to one byte per lane"
        );
    }

    #[test]
    fn decode_rejects_a_slice_whose_length_disagrees_with_the_shape() {
        assert!(
            decode_wire(&[0u8; 3], DType::F32, 1).is_none(),
            "a 3-byte slice cannot be one f32 lane (4 bytes); the codec must refuse, not read past"
        );
        assert!(
            decode_wire(&[0u8; 1], DType::Bool, 9).is_none(),
            "nine bool lanes need two packed bytes; one byte must be rejected"
        );
    }

    #[test]
    fn f32_lanes_round_trip_through_the_wire_little_endian() {
        let value = Value::F32(vec![1.5, -2.0, f32::INFINITY]);
        let mut wire = vec![0u8; wire_cell_bytes(DType::F32, 3)];
        encode_wire(&value, &mut wire);
        assert_eq!(
            &wire[0..4],
            &1.5f32.to_le_bytes(),
            "lane 0 must be little-endian"
        );
        let decoded = decode_wire(&wire, DType::F32, 3).expect("length matches");
        assert_eq!(
            decoded, value,
            "f32 lanes must survive an encode/decode round trip"
        );
    }

    #[test]
    fn lanes_i64_narrowed_back_preserves_integer_values() {
        let value = Value::I32(vec![-5, 0, 7]);
        let widened = value.lanes_i64();
        assert_eq!(widened, vec![-5i64, 0, 7]);
        assert_eq!(
            Value::from_i64(DType::I32, &widened),
            value,
            "i32 -> i64 -> i32 must be the identity for in-range values"
        );
        assert_eq!(
            Value::from_i64(DType::Bool, &[0, 3, -1]),
            Value::Bool(vec![0, 1, 1]),
            "narrowing to bool is x != 0, not a low-bit truncation"
        );
    }

    #[test]
    fn pick_broadcasts_a_scalar_operand_but_not_a_vector() {
        assert_eq!(
            pick(1, 7),
            0,
            "a length-one operand always contributes lane 0"
        );
        assert_eq!(
            pick(8, 7),
            7,
            "a full-length operand lines up lane for lane"
        );
    }
}
