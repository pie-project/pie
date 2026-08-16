//! Turning a row and a list of values into the two things a dispatch needs.
//!
//! This is where Vulkan diverges from Metal, and the divergence is not
//! cosmetic. Metal binds every operand to a numbered buffer slot, scalars
//! included — a `setBytes` at index `k` for the `k`-th argument — so a driver
//! there has one run to fill and the row's order IS the answer. Vulkan splits
//! the same list in two: buffers become descriptors in a set, scalars become
//! bytes at computed offsets inside a push block, and NEITHER run is indexed by
//! the operand's position in the row.
//!
//! So the mapping has to be computed, and computing it wrongly has no symptom.
//! A descriptor bound one slot over reads the wrong tensor and produces
//! numbers. A scalar packed at the wrong offset reads a stride where a head
//! count belongs and produces numbers. Nothing returns an error and no layer
//! complains, which is why the arithmetic lives here, in one place, checked —
//! rather than at each of the call sites that would otherwise each get it
//! slightly right.
//!
//! # No per-kernel branch
//!
//! [`pack`] matches on the operand's KIND and never on the kernel's name. That
//! is the same claim `driver-metal`'s dispatch makes and the same reason: a
//! symbol is a name, the table states its ABI, and a row the table already has
//! needs no code written to receive it. A `match` on entrypoint here would be
//! a list to keep in step with a list that already exists.
//!
//! # Who calls this
//!
//! Nothing in a serving path, today. The live dispatch binds from the
//! COMPILED MODULE instead — [`crate::binding::Params`] reads whether a
//! kernel takes its scalars as a push block or as a struct in a storage
//! buffer off the SPIR-V, because the row and the module disagree about that
//! and the module is the one the GPU obeys.
//!
//! This half is kept because it is the part that is provable without a
//! device, and it is the statement the module is checked AGAINST. But an ABI
//! derivation with no caller is one that rots without symptom, so
//! `every_row_packs_and_every_scalar_lands_on_its_own_field` in `tests/rules`
//! packs all 44 rows that state operands and reads every scalar back out of
//! the block it landed in. The two `unreachable!`s below are unreachable
//! twice over: the counts cannot disagree, and no serving path would reach
//! them if they did.
//!
//! # What a caller still owes
//!
//! [`Call::buffers`] holds the caller's own buffer handles in BINDING order,
//! not addresses — this crate does not resolve names to allocations, because
//! that is the plan's half of the work and it needs `model-compiler`. What
//! this settles is the part that is a property of the ROW, and therefore
//! provable with no device and no plan.

use kernels::{KernelSig, Ty};
use kernels_vulkan::Binding;

/// One value a caller supplies for one operand.
///
/// The scalar kinds are separate variants rather than one integer because the
/// widths differ and the check that matters is exactly the width one: a
/// [`Ty::Usize`] value handed to a [`Ty::I32`] operand is eight bytes going
/// into a four-byte slot, which either truncates or writes over its neighbour
/// depending on where in the block it lands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    /// A device allocation, named by whatever index the caller keys its own
    /// buffers on. Opaque here on purpose: this module decides ORDER, and the
    /// caller decides what a buffer is.
    Buffer(u32),
    /// A 32-bit signed scalar.
    I32(i32),
    /// A 32-bit unsigned scalar. Distinct from [`Value::I32`] because the row
    /// distinguishes them, and a row that says `U32` and receives a negative
    /// number is a bug the caller wants named.
    U32(u32),
    /// A 32-bit float.
    F32(f32),
    /// A 64-bit stride or extent, which is what [`Ty::Usize`] means here.
    Usize(u64),
}

impl Value {
    /// What this value is, for an error message.
    const fn kind(self) -> &'static str {
        match self {
            Self::Buffer(_) => "a buffer",
            Self::I32(_) => "an i32",
            Self::U32(_) => "a u32",
            Self::F32(_) => "an f32",
            Self::Usize(_) => "a usize",
        }
    }

    /// Whether this value may stand where an operand of `ty` is wanted.
    fn fits(self, ty: Ty) -> bool {
        match self {
            // Every buffer kind the table uses -- read/write, and the typed
            // arrays. Matched on the KIND rather than on a list of names so
            // that a row growing a new array type does not silently land in
            // the push block.
            Self::Buffer(_) => matches!(
                ty,
                Ty::Buf
                    | Ty::BufMut
                    | Ty::I32s
                    | Ty::I64s
                    | Ty::U32s
                    | Ty::U8s
                    | Ty::F32s
                    | Ty::F32sMut
                    | Ty::I32sMut
                    | Ty::U32sMut
                    | Ty::U8sMut
            ),
            Self::I32(_) => matches!(ty, Ty::I32),
            // `InPacked` too, and that is not a convenience. The table spells
            // it `::std::uint32_t` -- it is a `u32` FIELD of a struct some
            // earlier buffer binds, so a driver supplies a value of exactly
            // that width and the only thing unusual about it is where the
            // value goes, which `Binding::Packed` decides and this does not.
            Self::U32(_) => matches!(ty, Ty::U32 | Ty::InPacked),
            Self::F32(_) => matches!(ty, Ty::F32),
            Self::Usize(_) => matches!(ty, Ty::Usize | Ty::I64),
        }
    }

    /// This value's bytes, little-endian, at its own width.
    fn bytes(self) -> Vec<u8> {
        match self {
            // A buffer has no bytes in the push block. Reaching here means
            // `fits` was not consulted, which is why it is a panic and not a
            // zero-filled slot: a silently empty scalar is the failure this
            // whole module exists to prevent.
            Self::Buffer(_) => unreachable!("a buffer does not go in the push block"),
            Self::I32(v) => v.to_le_bytes().to_vec(),
            Self::U32(v) => v.to_le_bytes().to_vec(),
            Self::F32(v) => v.to_le_bytes().to_vec(),
            Self::Usize(v) => v.to_le_bytes().to_vec(),
        }
    }
}

/// What a dispatch needs from the row, and nothing else.
// No `Eq`: a packed operand can be an `f32`, and a driver comparing two calls
// for equality of a float is asking a question with no useful answer.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    /// The caller's buffers in DESCRIPTOR order, which is not the row's order.
    pub buffers: Vec<u32>,
    /// The push block, packed and padded to the size the layout declares.
    pub push: Vec<u8>,
    /// The operands the row said the driver must supply into a packed struct
    /// rather than into a slot of their own, as `(operand index, value)`.
    ///
    /// Handed back rather than dropped, and rather than folded into the push
    /// block. [`Binding::Packed`] is a field of a buffer some earlier operand
    /// already binds, so folding it into the push run would push a word no
    /// shader reads AND leave the struct member unwritten -- one mistake
    /// producing two wrong things. The caller writes these while filling that
    /// buffer, and an empty list is the normal case.
    pub packed: Vec<(usize, Value)>,
}

/// Why a list of values is not a call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mismatch {
    /// The row states no operands at all.
    ///
    /// UNSTATED, not nullary: a row with an empty list has not been filled in
    /// yet, and 292 of this table's 480 entrypoints are in that state. A
    /// driver may still launch one -- `driver-metal` falls back to the lowered
    /// plan's own argument order -- but it may not do so from HERE, because
    /// there is nothing here to do it from. Answering with an empty call would
    /// be a dispatch that binds nothing and reports success.
    Unstated,
    /// A different number of values than the row has operands.
    Arity {
        /// What the row states.
        row: usize,
        /// What the caller supplied.
        given: usize,
    },
    /// A value of the wrong kind for the operand it was given to.
    Kind {
        /// Which operand, by index in the row.
        at: usize,
        /// That operand's name, as the row spells it.
        name: &'static str,
        /// What the row wants there.
        wants: Ty,
        /// What arrived.
        given: &'static str,
    },
}

impl core::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unstated => write!(
                f,
                "the row states no operands, so it cannot say what a call looks like"
            ),
            Self::Arity { row, given } => {
                write!(f, "the row has {row} operands and {given} were supplied")
            }
            Self::Kind {
                at,
                name,
                wants,
                given,
            } => write!(f, "operand {at} (`{name}`) wants {wants:?} and got {given}"),
        }
    }
}

impl core::error::Error for Mismatch {}

/// Split a row's operands into descriptors and a push block.
///
/// # Errors
///
/// [`Mismatch`], and nothing partial: a call binds whole or not at all,
/// because a dispatch with some operands placed is a dispatch that reads
/// whatever was in the others.
pub fn pack(sig: &KernelSig, values: &[Value]) -> Result<Call, Mismatch> {
    if sig.operands.is_empty() {
        return Err(Mismatch::Unstated);
    }
    if sig.operands.len() != values.len() {
        return Err(Mismatch::Arity {
            row: sig.operands.len(),
            given: values.len(),
        });
    }

    // Every kind checked BEFORE anything is placed. Checking as it goes would
    // return a half-built call on the failing operand, and the caller most
    // likely to ignore the error is the one that would then dispatch it.
    for (at, (op, value)) in sig.operands.iter().zip(values).enumerate() {
        if !value.fits(op.ty) {
            return Err(Mismatch::Kind {
                at,
                name: op.name,
                wants: op.ty,
                given: value.kind(),
            });
        }
    }

    let layout = kernels_vulkan::push_layout(sig);
    // Sized from the row and zero-filled, rather than grown as fields are
    // written. The block has PADDING -- an 8-byte stride after a 4-byte count
    // starts at 8, not 4 -- and a block built by appending would close those
    // gaps and shift every field after them.
    let mut push = vec![0u8; kernels_vulkan::push_size(sig) as usize];
    let mut buffers = Vec::new();
    let mut packed = Vec::new();

    for (at, (binding, value)) in kernels_vulkan::bindings(sig)
        .into_iter()
        .zip(values)
        .enumerate()
    {
        match binding {
            Binding::Buffer(_) => match value {
                Value::Buffer(b) => buffers.push(*b),
                // Unreachable: `fits` refused every non-buffer against a
                // buffer kind above, and `bindings` reads the same kinds.
                other => unreachable!("`{}` is {} at a buffer slot", sig.name, other.kind()),
            },
            Binding::Push(field) => {
                let Some(field) = layout.get(field as usize) else {
                    unreachable!("`{}` has a push field outside its own layout", sig.name)
                };
                let bytes = value.bytes();
                // The row's width, not the value's. They agree because `fits`
                // required it, and asserting it here is what makes a future
                // kind whose width nobody thought about a panic in a test
                // rather than a neighbouring field overwritten on a GPU.
                assert_eq!(
                    bytes.len() as u32,
                    field.size,
                    "`{}` operand `{}` is {} bytes and its slot is {}",
                    sig.name,
                    field.name,
                    bytes.len(),
                    field.size
                );
                let at = field.offset as usize;
                push[at..at + bytes.len()].copy_from_slice(&bytes);
            }
            Binding::Packed => packed.push((at, *value)),
        }
    }

    Ok(Call {
        buffers,
        push,
        packed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one row in the table that needs a value with no slot of its own.
    fn packed_row() -> Option<&'static KernelSig> {
        kernels_vulkan::KERNELS
            .iter()
            .find(|s| s.operands.iter().any(|o| matches!(o.ty, Ty::InPacked)))
    }

    /// A value for whatever kind an operand wants, so a test can build a call
    /// without naming a specific row's shape.
    fn value_for(ty: Ty) -> Value {
        match ty {
            Ty::I32 => Value::I32(7),
            Ty::U32 | Ty::InPacked => Value::U32(7),
            Ty::F32 => Value::F32(0.5),
            Ty::Usize | Ty::I64 => Value::Usize(7),
            _ => Value::Buffer(0),
        }
    }

    fn call_for(sig: &KernelSig) -> Result<Call, Mismatch> {
        let values: Vec<Value> = sig.operands.iter().map(|o| value_for(o.ty)).collect();
        pack(sig, &values)
    }

    #[test]
    fn a_row_splits_into_the_two_runs_a_dispatch_needs() {
        let sig = kernels::sig_in(kernels_vulkan::KERNELS, "kv_append").expect("the row is stated");
        let call = call_for(sig).expect("every operand is the kind the row wants");
        assert_eq!(
            call.buffers.len() as u32,
            kernels_vulkan::buffer_count(sig),
            "the descriptor run"
        );
        assert_eq!(
            call.push.len() as u32,
            kernels_vulkan::push_size(sig),
            "the push run, padded to the size the layout declares"
        );
    }

    /// The block is the size the LAYOUT says, not the sum of its fields.
    ///
    /// `kv_append` is `head_dim: i32` then two `usize` strides, so the strides
    /// start at 8 and 16 and the block is 24 bytes with four of them padding.
    /// A block built by appending would be 20, every stride would be four
    /// bytes early, and the shader would read half of one and half of the
    /// next -- a number, in the right place, that is not a stride.
    #[test]
    fn padding_is_left_where_the_layout_leaves_it() {
        let sig = kernels::sig_in(kernels_vulkan::KERNELS, "kv_append").expect("stated");
        let call = call_for(sig).expect("packs");
        let fields = kernels_vulkan::push_layout(sig);
        let sum: u32 = fields.iter().map(|f| f.size).sum();
        assert!(
            call.push.len() as u32 > sum,
            "this row was chosen because its block has padding; if it stopped \
             having any the test proves nothing"
        );
        for f in &fields {
            let at = f.offset as usize;
            assert_ne!(
                &call.push[at..at + f.size as usize],
                &vec![0u8; f.size as usize][..],
                "`{}` was not written",
                f.name
            );
        }
    }

    /// A scalar lands at its own offset and nowhere else.
    #[test]
    fn each_scalar_is_written_where_its_field_says() {
        let sig = kernels::sig_in(kernels_vulkan::KERNELS, "kv_append").expect("stated");
        let values: Vec<Value> = sig
            .operands
            .iter()
            .map(|o| match o.ty {
                Ty::I32 => Value::I32(0x0102_0304),
                Ty::Usize => Value::Usize(0x0a0b_0c0d_0e0f_1011),
                other => value_for(other),
            })
            .collect();
        let call = pack(sig, &values).expect("packs");
        assert_eq!(&call.push[0..4], &0x0102_0304i32.to_le_bytes());
        assert_eq!(&call.push[8..16], &0x0a0b_0c0d_0e0f_1011u64.to_le_bytes());
        assert_eq!(
            &call.push[4..8],
            &[0, 0, 0, 0],
            "the padding between them is not written through"
        );
    }

    /// An unstated row refuses rather than answering with an empty call.
    #[test]
    fn an_unstated_row_will_not_pretend_to_be_nullary() {
        let sig = kernels_vulkan::KERNELS
            .iter()
            .find(|s| s.operands.is_empty())
            .expect("292 of them are");
        assert_eq!(pack(sig, &[]), Err(Mismatch::Unstated));
    }

    #[test]
    fn a_call_with_the_wrong_number_of_values_is_refused() {
        let sig = kernels::sig_in(kernels_vulkan::KERNELS, "kv_append").expect("stated");
        assert_eq!(
            pack(sig, &[Value::Buffer(0)]),
            Err(Mismatch::Arity {
                row: sig.operands.len(),
                given: 1
            })
        );
    }

    /// A scalar where a buffer belongs is named, not placed.
    ///
    /// The refusal that matters most. Nothing downstream could tell: a
    /// descriptor run one short binds every later tensor one slot early, and
    /// the shader reads keys as values.
    #[test]
    fn a_scalar_standing_in_for_a_buffer_is_refused() {
        let sig = kernels::sig_in(kernels_vulkan::KERNELS, "kv_append").expect("stated");
        let mut values: Vec<Value> = sig.operands.iter().map(|o| value_for(o.ty)).collect();
        values[0] = Value::I32(1);
        assert!(matches!(
            pack(sig, &values),
            Err(Mismatch::Kind { at: 0, .. })
        ));
    }

    /// And an eight-byte scalar in a four-byte slot.
    ///
    /// The one that would not truncate but would overwrite: `head_dim` is four
    /// bytes at offset 0 and the next field starts at 8, so eight bytes
    /// written there survive the assert only by destroying the padding -- and
    /// on a row where the next field started at 4, its value.
    #[test]
    fn a_wide_scalar_in_a_narrow_slot_is_refused() {
        let sig = kernels::sig_in(kernels_vulkan::KERNELS, "kv_append").expect("stated");
        let mut values: Vec<Value> = sig.operands.iter().map(|o| value_for(o.ty)).collect();
        let at = sig
            .operands
            .iter()
            .position(|o| matches!(o.ty, Ty::I32))
            .expect("`head_dim`");
        values[at] = Value::Usize(1);
        assert!(matches!(pack(sig, &values), Err(Mismatch::Kind { .. })));
    }

    /// A packed operand takes neither run, and is handed back.
    ///
    /// Folding it into the push block is the defect `Binding::Packed` exists
    /// to make unrepresentable: it would push a word no shader reads AND leave
    /// the struct member unwritten.
    #[test]
    fn a_packed_operand_takes_no_slot_and_is_not_lost() {
        let Some(sig) = packed_row() else {
            return;
        };
        let call = call_for(sig).expect("packs");
        assert_eq!(call.packed.len(), 1, "the row has exactly one");
        assert_eq!(
            call.buffers.len() as u32,
            kernels_vulkan::buffer_count(sig),
            "it took no descriptor"
        );
        assert_eq!(
            call.push.len() as u32,
            kernels_vulkan::push_size(sig),
            "and no push field"
        );
    }

    /// Every stated row in the table packs, with no arm written for any of it.
    ///
    /// The claim the module is for. 188 rows, ten operand kinds, and a `match`
    /// on kind rather than on name -- so a row the table already has needs no
    /// code here to receive it.
    #[test]
    fn every_stated_row_packs_from_its_own_description() {
        let mut packed = 0;
        for sig in kernels_vulkan::KERNELS {
            if sig.operands.is_empty() {
                continue;
            }
            let call = call_for(sig).unwrap_or_else(|e| panic!("`{}`: {e}", sig.name));
            assert_eq!(call.buffers.len() as u32, kernels_vulkan::buffer_count(sig));
            assert_eq!(call.push.len() as u32, kernels_vulkan::push_size(sig));
            packed += 1;
        }
        // The floor falls with the crossing and with nothing else: a row a
        // family retires is a row this path will never pack again, and the
        // check that has to survive is "every row it CAN pack, it packs from
        // its own description" -- not a count.
        let floor = 40usize.saturating_sub(kernels_vulkan::retired_rows().len());
        assert!(packed >= floor, "only {packed} rows were packed");
    }
}
