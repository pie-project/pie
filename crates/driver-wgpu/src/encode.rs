//! [`Encode`] over a real adapter: what a routine body dispatches through.
//!
//! `kernels-wgpu` declares routines and cannot run them — it depends on
//! `kernels` and nothing else, so it names no adapter, no buffer and no
//! pipeline. It states an entrypoint and a number of LANES; everything from
//! there is this crate's.
//!
//! # What this does that a body must not
//!
//! **The division into workgroups.** `@workgroup_size` is declared in the WGSL
//! and recovered by reflection, so the divisor is a property of the shader
//! text rather than of the launch. A body that divided by it would carry a
//! second copy of a number it cannot see; here it is one `div_ceil` per axis,
//! in one place, exactly as [`crate::geometry::groups`] has always done it.
//!
//! **The split of arguments into bindings and scalars.** A routine hands over
//! its argument list in signature order and this reads the split off the
//! VARIANTS: [`ArgValue::Buffer`] is a `@group(0)` entry and everything else
//! is a word of the `@group(1)` uniform block. That is the property the
//! routine shape buys — under `kernel!` the same split was stated a second
//! time, as an operand's `Ty`, and two statements of it could disagree.

use core::cell::RefCell;

use kernels::routine::Refusal;
use kernels_wgpu::routine::{ArgValue, Encode, Fire};

use crate::binding::Bound;
use crate::device::{Buffer, Device, Pipelines, Recorded};
use crate::lowering::hold::Facts;
use crate::serve::Modules;
use kernels_wgpu::Capability;

/// One fire's worth of adapter, handed to a routine body.
///
/// Borrows rather than owns: a shell already has the device, the pipeline
/// cache and the fire's buffers, and this is a view over them for the length
/// of one call.
pub struct Encoder<'a, M: Modules> {
    device: &'a Device,
    /// `Pipelines::get` takes `&mut self` and [`Encode::fire`] takes
    /// `&self`, because the machinery hands a body `&B::Ctx`. Interior
    /// mutability is the whole of the difference, and it is confined here.
    pipelines: RefCell<&'a mut Pipelines>,
    modules: &'a M,
    tier: Capability,
    /// Handle to buffer. A routine's [`ArgValue::Buffer`] is an index into
    /// this: the caller decides what a buffer is, which is the same contract
    /// `crate::lowering` has always had.
    buffers: &'a [&'a Buffer],
    /// WHAT THIS FIRE ANSWERS, for a body that asks.
    ///
    /// `None` on an encoder built to dispatch alone, which has no statement
    /// behind it: a body that asks on one gets [`Refusal::Unstated`], which is
    /// the honest answer rather than an invented zero. The shape is
    /// `lowering::routine::Planner`'s, because the question is the same one.
    answers: Option<(&'a RefCell<crate::lowering::hold::Handles<'a>>, Facts)>,
}

impl<'a, M: Modules> Encoder<'a, M> {
    /// A view for one call.
    pub fn new(
        device: &'a Device,
        pipelines: &'a mut Pipelines,
        modules: &'a M,
        tier: Capability,
        buffers: &'a [&'a Buffer],
    ) -> Self {
        Self {
            device,
            pipelines: RefCell::new(pipelines),
            modules,
            tier,
            buffers,
            answers: None,
        }
    }

    /// The same encoder, able to ANSWER a body that asks.
    ///
    /// `handles` is a cell because ANSWERING MINTS: a staged fact takes a
    /// handle, and the caller reads the same list back afterwards.
    #[must_use]
    pub fn answering(
        mut self,
        handles: &'a RefCell<crate::lowering::hold::Handles<'a>>,
        facts: Facts,
    ) -> Self {
        self.answers = Some((handles, facts));
        self
    }
}

/// The scalar arguments of a call, as the uniform block's bytes.
///
/// Little-endian at each argument's own width, in signature order, with
/// [`ArgValue::Usize`] two `u32` words low-first — WGSL has no 64-bit integer
/// and the shader reads it as a `vec2<u32>`.
///
/// # Each scalar is ALIGNED to its own width, and one of them is eight
///
/// WGSL gives `vec2<u32>` an alignment of 8, so a `Usize` following a
/// four-byte scalar starts at the next multiple of eight and leaves a
/// four-byte HOLE. A packer that concatenated would put every field after it
/// four bytes early — an `i32` extent read as the high half of a stride, and
/// the stride read as something else again. Numbers, not errors.
///
/// This is not a hypothetical and it is not this backend's discovery:
/// `driver-vulkan` shipped the concatenating version and measured it on
/// `attn/kv_write.slang`, whose push block is `{ int head_dim; PIE_STRIDE
/// k_head_stride; ... }` — twenty bytes packed against a twenty-four byte
/// range. Its own row-driven packer had applied the rule for as long as it had
/// existed; the routine path packs from a SIGNATURE and had never met a
/// `uint2`.
///
/// No ported routine here takes a `Usize` yet — all ten uses are in `attn`,
/// which has not crossed — so this is the rule arriving before the family that
/// needs it rather than after.
fn block(args: &[ArgValue]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    for a in args {
        // Pad to this scalar's own alignment first. Four for everything the
        // shader reads as a scalar, eight for the `vec2<u32>`.
        let align = match a {
            ArgValue::Usize(_) => 8,
            _ => 4,
        };
        while !out.len().is_multiple_of(align) {
            out.push(0);
        }
        match a {
            // A HANDLE CONTRIBUTES NO BYTES, shaped or not. `Shaped` carries
            // the rectangle a statement gave the operand, which the marks read
            // and the uniform block does not — it is a `@group(0)` entry by
            // the same rule `Buffer` is. A RAISED VIEW is host data the body
            // already read; it reaches no shader and packs nothing.
            ArgValue::Buffer(_) | ArgValue::Shaped { .. } | ArgValue::Raised(_) => {}
            ArgValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
            ArgValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
            ArgValue::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
            ArgValue::Usize(v) => {
                let (lo, hi) = (
                    u32::try_from(v & 0xffff_ffff).unwrap_or(u32::MAX),
                    u32::try_from(v >> 32).unwrap_or(u32::MAX),
                );
                out.extend_from_slice(&lo.to_le_bytes());
                out.extend_from_slice(&hi.to_le_bytes());
            }
        }
    }
    out
}

impl<M: Modules> Encode for Encoder<'_, M> {
    // AN ENCODER ANSWERS THROUGH THE SAME BINDER THE COLUMN WENT THROUGH,
    // exactly as `lowering::routine::Planner` does. `Env` left the parameter
    // list, so a fact only the fire can answer is no longer bound into `args`
    // before the body runs; the body asks, and this is `kernels::bind::one`
    // entered at one argument instead of a list.
    //
    // `RefCell` because answering MINTS -- a staged fact takes a handle, which
    // is a mutation of the handle vector -- and the body holds only a `&self`.
    fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
        let (handles, facts) = self.answers.ok_or(Refusal::Unstated {
            what: "a fact, on an encoder with no fire behind it",
        })?;
        crate::lowering::bind::one(ty, source, &mut handles.borrow_mut(), facts)
    }

    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
        // A body that had nothing to do should have refused; a zero here would
        // become `dispatch_workgroups(0, 1, 1)`, which is legal WebGPU that
        // runs nothing and reports success.
        if fire.lanes.contains(&0) {
            return Err(Refusal::Grid {
                what: "the lanes a routine asked for",
                at: 0,
            });
        }

        let source = self
            .modules
            .at(fire.file, fire.entrypoint, self.tier)
            .ok_or(Refusal::Undeclared)?;

        let mut pipelines = self.pipelines.borrow_mut();
        let pipeline = pipelines
            .get(self.device, fire.entrypoint, self.tier, &source)
            .map_err(|_| Refusal::Device {
                why: "the module would not build a pipeline",
            })?;

        // The divisor, off the shader rather than off a body.
        let local = pipeline.module().local;
        let groups = [
            fire.lanes[0].div_ceil(local.at(0)),
            fire.lanes[1].div_ceil(local.at(1)),
            fire.lanes[2].div_ceil(local.at(2)),
        ];

        let mut bound = Vec::new();
        for a in args {
            // A `Shaped` HANDLE IS STILL A HANDLE — the rectangle beside it is
            // what the marks read, and a dispatch binds the allocation. An
            // `if let` over `Buffer` alone skipped every shaped operand
            // silently, which is the defect `driver-vulkan` names at length.
            let handle = match a {
                ArgValue::Buffer(h) => *h,
                ArgValue::Shaped { handle, .. } => *handle,
                _ => continue,
            };
            let buffer = self
                .buffers
                .get(handle as usize)
                .ok_or(Refusal::Absent { what: "a buffer" })?;
            bound.push(Bound::whole(*buffer));
        }
        let uniform = block(args);

        self.device
            .run_all(&[Recorded {
                pipeline,
                buffers: &bound,
                uniform: &uniform,
                groups,
            }])
            .map(|_| ())
            .map_err(|_| Refusal::Device {
                why: "the device refused the dispatch",
            })
    }
}

#[cfg(test)]
mod tests {
    use super::block;
    use kernels_wgpu::routine::ArgValue;

    /// A buffer contributes no bytes and a `Usize` contributes two words.
    ///
    /// The uniform block is built from the argument list's VARIANTS, which is
    /// what the routine shape buys: under `kernel!` the same split was stated
    /// a second time as an operand's `Ty`, and the failure mode of the two
    /// disagreeing was a scalar written over its neighbour.
    #[test]
    fn the_block_is_the_scalars_at_their_own_alignments_and_nothing_else() {
        let bytes = block(&[
            ArgValue::Buffer(3),
            ArgValue::I32(-2),
            ArgValue::Buffer(0),
            ArgValue::U32(7),
            ArgValue::F32(1.5),
            ArgValue::Usize(0x0000_0002_0000_0001),
        ]);
        assert_eq!(
            bytes.len(),
            4 + 4 + 4 + 4 + 8,
            "two buffers contribute nothing, three scalars contribute twelve, \
             and the usize contributes eight AFTER four bytes of padding -- \
             `vec2<u32>` is eight-aligned in WGSL"
        );
        assert_eq!(&bytes[0..4], (-2i32).to_le_bytes());
        assert_eq!(&bytes[4..8], 7u32.to_le_bytes());
        assert_eq!(&bytes[8..12], 1.5f32.to_le_bytes());
        assert_eq!(
            &bytes[12..16],
            [0u8; 4],
            "the HOLE. `driver-vulkan` shipped the version without it and \
             measured twenty bytes packed against a twenty-four byte range on \
             `attn/kv_write.slang`; every field after the stride was read four \
             bytes early, which is numbers rather than errors."
        );
        assert_eq!(&bytes[16..20], 1u32.to_le_bytes(), "the low word first");
        assert_eq!(&bytes[20..24], 2u32.to_le_bytes(), "then the high word");

        // And a `Usize` that is ALREADY aligned gets no padding: the rule is
        // an alignment and not an unconditional gap.
        let flush = block(&[ArgValue::I32(1), ArgValue::U32(2), ArgValue::Usize(3)]);
        assert_eq!(flush.len(), 4 + 4 + 8);
        assert_eq!(&flush[8..12], 3u32.to_le_bytes());
    }
}
