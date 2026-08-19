//! This backend's instantiation of the `kernels` routine machinery.
//!
//! A routine is an ordinary `fn` that computes a launch and dispatches it.
//! Everything a row used to STATE — the launch rule, the grid parameter, the
//! head parameter, the operand list — is code in the body, and the table row
//! is derived from the signature by [`routine!`].
//!
//! This file is a PORT of `kernels-wgpu/src/routine.rs`, not a design. Two
//! backends with the same constraint answering it two ways is the drift this
//! whole refactor exists to remove, so where Vulkan has no reason to differ it
//! does not. `.wiki/kernel-x/vulkan-refactor.md` §4 and §5 record which of
//! these decisions are wgpu's and which are this backend's.
//!
//! # Why the context is a trait and not a struct
//!
//! `kernels-cuda-new`'s `Ctx` is a struct: it owns the NVRTC cache and the
//! cuBLAS handles, so the crate that declares the routines also launches them.
//!
//! This crate cannot be that. Its `[dependencies]` is `kernels` and nothing
//! else — `ash` is a dev-dependency, deliberately — while the `ash::Device`,
//! the command buffer being recorded, the descriptor pool and every
//! `vk::Pipeline` live in `driver-vulkan`. So the thing a body dispatches
//! through is a TRAIT the driver implements ([`Encode`]), and `Backend::Ctx`
//! is `dyn Encode`. The `?Sized` bound that makes this legal is already on
//! the shared machinery; `kernels-wgpu` needed it first, for this reason.
//!
//! # What is Vulkan's alone
//!
//! **The tier, which the BODY resolves.** One entrypoint compiles to up to
//! three `.spv` modules — `Baseline`, `Fp16`, `Coopmat` — named by
//! [`crate::Capability::module`] as the entrypoint plus a tag. A body asks
//! [`Encode::best`] for the ceiling this device advertises, hands it to
//! [`crate::module::path`], and fires the artifact that comes back. The file
//! it names is the file that runs.
//!
//! It was the other way round until `driver-vulkan`'s `Modules::code` walked
//! `Capability::PREFERENCE` behind the body, and that walk could not reach a
//! tiered artifact at all: 146 cooperative-matrix modules and 20 fp16 ones
//! were dead on every device from the first commit, found by measuring prefill
//! rather than by anything failing. A body that names its module cannot have
//! that bug.
//!
//! A body still cannot name a tier the device lacks — that is an unadvertised
//! cooperative-matrix shape and a SIGSEGV inside `vkCreateComputePipelines`
//! with the validation layer entirely silent — because the only tier it can
//! compose with is the one `best()` handed it.
//!
//! The FILE beside the entrypoint is the artifact, not a source: `build.rs`
//! here emits one `.spv` per entrypoint per tier, so `norm/rms.slang` is a
//! build input and `rms_single_row_bfloat16.coopmat.spv` is what a fire loads.
//! wgpu states a `.wgsl` in the same position for the opposite reason — one
//! source holds many entrypoints and it compiles at run time.

#[cfg(test)]
use kernels::Ty;
#[cfg(test)]
use kernels::routine::Arg;
use kernels::routine::{Backend, Extent, Refusal};
use kernels::shader::ShaderValue;

pub use crate::Capability;

/// This backend, as the machinery names it.
///
/// A marker: never constructed, carrying only the two concrete types the
/// `kernels` machinery is generic over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vulkan;

impl Backend for Vulkan {
    type Value = ArgValue;
    type Ctx<'a> = dyn Encode + 'a;

    // THIS PLANE MINTS REGION-SHAPED VALUES NOW, and that is §7 of
    // `.wiki/migration.md` settled. It used to bind addresses alone and refuse
    // here, so its marks answered zero for `rows` and `width` and every
    // rectangle had to arrive as a separate parameter keyed to `keys::Width`,
    // `keys::InWidth` or `keys::OutWidth0`. The widths were always there --
    // `Holds::in_width` and `out_width` answered them for a `Kind::InWidth`
    // slot -- they simply reached a parameter instead of the operand they
    // describe.
    fn region(value: &ArgValue) -> Result<Extent, Refusal> {
        match *value {
            ArgValue::Buffer { rows, width, .. } => Ok(Extent { rows, width }),
            // `Absent`, not `Kind`: a plain handle carries no `Ty` mismatch,
            // just no shape to report.
            _ => Err(Refusal::Absent {
                what: "a region's shape: the bound value carries only a handle",
            }),
        }
    }
}

/// One value a caller supplies for one argument.
///
/// The scalar kinds are separate variants rather than one integer because the
/// widths differ and the check that matters is exactly the width one: a
/// [`Ty::Usize`] value handed to a [`Ty::I32`] argument is eight bytes going
/// into a four-byte slot, which either truncates or writes over its neighbour
/// depending on where in the push block it lands. Vulkan's push range is
/// packed by the driver from this list, so nothing downstream would report it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    /// A device allocation, named by whatever index the caller keys its own
    /// buffers on, and whether the launcher may WRITE through it.
    ///
    /// The handle is opaque on purpose: this crate cannot name a `vk::Buffer`
    /// and does not need to. The driver binds handles to its own buffers
    /// before it calls, and a routine only passes them through.
    ///
    /// The flag is Vulkan's alone -- `ShaderValue::buffer_mut` defaults to
    /// dropping it and the other two backends take the default. This driver
    /// puts a barrier between two dispatches only when they touch the same
    /// bytes and decides that from writability; barriering every neighbouring
    /// pair costs 8 microseconds each on this card, measured at 3.8 ms of a
    /// 7.2 ms decode. Under `kernel!` the fact came off the row's operand
    /// types, and a routine states it in the argument TYPE instead -- so the
    /// `Buf`/`Buf` distinction has to survive the trip through a value,
    /// which is all the driver sees.
    ///
    /// # ONE VARIANT, BECAUSE A SECOND ONE LOST BOTH FACTS
    ///
    /// The rectangle used to ride a separate `Shaped` variant, minted by
    /// `kernels::bind` for every operand slot. Two things went wrong with it,
    /// and neither could fail loudly.
    ///
    /// `Shaped` had no `writes`, and `ShaderValue::buffer_at` and
    /// `buffer_mut_at` had identical bodies -- so every operand that carried a
    /// shape arrived with its DIRECTION erased, which is the one fact the
    /// barrier decision reads.
    ///
    /// And `crate::encode`'s recording pass splits arguments with
    /// `if let ArgValue::Buffer { .. }`. An `if let` does not have to be
    /// exhaustive, so `Shaped` did not fail to compile there: it was silently
    /// skipped, never pushed to `buffers`, and never written into a descriptor
    /// set. `kernels::bind`'s `shaped` is on the path for `Kind::In`,
    /// `Kind::Out` and `Source::Alias` -- every operand of every routine --
    /// so only weights, which still minted a plain buffer, reached the shader.
    ///
    /// Merging the two is what makes both impossible to reintroduce: there is
    /// one buffer variant, it always carries its direction, and a rectangle it
    /// was never given is `0`, which [`Extent`] already treats as absent.
    Buffer {
        /// The caller's index for the allocation.
        handle: u32,
        /// Whether the shader may write through this binding.
        writes: bool,
        /// Rows in this launch's rectangle. Zero where the binder had none to
        /// give — a weight, or a plane's own re-emission through
        /// [`kernels::Bind::arg`].
        rows: i32,
        /// Elements per row. Zero where the statement gave none.
        width: i32,
    },
    /// A 32-bit signed scalar.
    I32(i32),
    /// A 32-bit unsigned scalar.
    U32(u32),
    /// A 32-bit float.
    F32(f32),
    /// A 64-bit stride or extent, which is what [`Ty::Usize`] means here.
    Usize(u64),
}

/// NO ABSENT VALUE TO MINT, and the default is the whole of the answer.
///
/// This plane binds HANDLES: every operand a statement places resolves to one,
/// and a statement that placed none produces no value at all rather than an
/// empty one. So `Option<M>` here always unpacks as `Some`, which is what
/// [`kernels::routine::Absent`]'s default says. A plane that later grows a
/// sentinel handle overrides both halves together.
impl kernels::routine::Absent for ArgValue {}

impl ArgValue {
    /// What this value is, for a refusal to name.
    #[must_use]
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Buffer { .. } => "a buffer",
            Self::I32(_) => "an i32",
            Self::U32(_) => "a u32",
            Self::F32(_) => "an f32",
            Self::Usize(_) => "a usize",
        }
    }
}

// THE PLANE'S OWN `Fire` STOOD HERE. It is `kernels::routine::Fire` now,
// which CUDA states too -- the four facts were always the same four, and the
// only difference was that CUDA passed them positionally. See that type for
// what the shared `lanes`/`group` pair means.


/// What a routine body dispatches through.
///
/// Implemented by `driver-vulkan`. The split is the crate boundary: a body
/// knows the entrypoint and the extent, and the driver knows what a buffer is,
/// which descriptor set a binding lands in, how the push block is packed, and
/// which capability tier this device can be given.
pub trait Encode {
    /// Run one dispatch.
    ///
    /// `args` is the routine's own argument list, in signature order — which
    /// is the SHADER's binding order, not the trace's. The two differ for
    /// 2,898 of this tree's 3,992 rectangles, and reconciling them is the
    /// caller's job, not this one's.
    ///
    /// The implementor separates buffers from scalars by variant, which is
    /// what makes the split derivable rather than stated twice.
    ///
    /// # Errors
    ///
    /// Whatever the device or the binding refused, as a [`Refusal`].
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal>;

    /// ONE VALUE, RESOLVED FROM THE COLUMN'S OWN VOCABULARY.
    ///
    /// What a body reaches through `ctx.ask::<C, keys::X>()` and
    /// `ctx.params()`. The ANSWERING side is unchanged by this: the driver
    /// already resolves a `(Ty, Source)` pair for every argument it binds --
    /// `kernels::bind::one`, over its own `Holds` -- and this is that call,
    /// made for a body instead of for a column.
    ///
    /// It exists because most of what used to be an `Env` parameter was
    /// checkpoint configuration the statement now carries as a `Const`, and
    /// what is left -- the batch, the plan, the allocator -- needed an ANSWER
    /// rather than a parameter.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unstated`] when this backend answers no such fact, and
    /// whatever the fact's own absence means otherwise.
    fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal>;

    /// The highest tier THIS DEVICE advertises.
    ///
    /// A body picks its own artifact — [`crate::module::path`] steps down from
    /// this to the best tier the build compiled — so the ceiling has to reach
    /// the body rather than stopping at the driver. It is the device's answer
    /// and never the text's: a body may not name a tier it did not get from
    /// here, because a module declaring a capability the device left disabled
    /// faults inside `vkCreateComputePipelines` with the validation layer
    /// silent.
    ///
    /// This is what `kernels-cuda`'s bodies have always had in a different
    /// spelling: `mma_supported(..)` asks whether a shape fits the tile, then
    /// the body fires one kernel or the other. The question differs, the shape
    /// of the answer does not.
    ///
    /// Defaults to [`Capability::Baseline`], which is what a test double that
    /// records fires rather than running them should say.
    fn best(&self) -> Capability {
        Capability::Baseline
    }
}

/// This backend's value, as the shared operand types read it.
///
/// The ten operand types live in [`kernels::shader`], not per-backend, because
/// the vocabulary is closed and identical across metal/vulkan/wgpu. What is
/// NOT shared is the value, [`ArgValue`]; this impl is the whole of what a
/// shared operand type needs to know about it.
impl ShaderValue for ArgValue {
    fn as_buffer(self) -> Option<u32> {
        match self {
            Self::Buffer { handle, .. } => Some(handle),
            _ => None,
        }
    }
    fn as_i32(self) -> Option<i32> {
        match self {
            Self::I32(v) => Some(v),
            _ => None,
        }
    }
    fn as_u32(self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(v),
            _ => None,
        }
    }
    fn as_f32(self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }
    fn as_usize(self) -> Option<u64> {
        match self {
            Self::Usize(v) => Some(v),
            _ => None,
        }
    }
    fn as_extent(self) -> Option<(i32, i32)> {
        match self {
            Self::Buffer { rows, width, .. } => Some((rows, width)),
            _ => None,
        }
    }
    fn buffer_at(handle: u32, rows: i32, width: i32) -> Self {
        Self::Buffer { handle, writes: false, rows, width }
    }
    fn buffer_mut_at(handle: u32, rows: i32, width: i32) -> Self {
        Self::Buffer { handle, writes: true, rows, width }
    }
    fn buffer(handle: u32) -> Self {
        Self::Buffer { handle, writes: false, rows: 0, width: 0 }
    }
    fn buffer_mut(handle: u32) -> Self {
        Self::Buffer { handle, writes: true, rows: 0, width: 0 }
    }
    fn i32(v: i32) -> Self {
        Self::I32(v)
    }
    fn u32(v: u32) -> Self {
        Self::U32(v)
    }
    fn f32(v: f32) -> Self {
        Self::F32(v)
    }
    fn usize(v: u64) -> Self {
        Self::Usize(v)
    }
}

/// How Slang spells the twelve, as this tree's shaders declare them.
///
/// Every string is read out of `kernels/`, not invented; `PIE_ACT` is the
/// activation type the tier substitutes, which is why the two opaque
/// buffers name it rather than a concrete element type.
///
/// `USIZE` is empty on purpose: Slang has `uint64_t` under the shader-int64
/// capability, but nothing in this tree declares one — every live use is an
/// extent the driver resolves and narrows before it reaches a push block.
impl kernels::shader::Lang for Vulkan {
    const BUF: &'static str = "StructuredBuffer<PIE_ACT>";
    const BUF_MUT: &'static str = "RWStructuredBuffer<PIE_ACT>";
    const I32S: &'static str = "StructuredBuffer<int>";
    const U32S: &'static str = "StructuredBuffer<uint>";
    const U8S: &'static str = "StructuredBuffer<uint8_t>";
    const F32S: &'static str = "StructuredBuffer<float>";
    const F32S_MUT: &'static str = "RWStructuredBuffer<float>";
    // WHAT `PIE_ACT` EXPANDS TO, and the reason the two spellings above are
    // not the same string. `common/bf16.slang` defines it `uint16_t` -- Slang
    // has no bf16, so the sixteen bits are declared as an integer and
    // `bf16_to_f32` shifts them into place. A signature naming the element
    // gets the expansion; one naming an opaque buffer gets the macro, which
    // is a spelling no `.slang` line contains.
    const BF16S: &'static str = "StructuredBuffer<uint16_t>";
    const BF16S_MUT: &'static str = "RWStructuredBuffer<uint16_t>";
    const F16S: &'static str = "StructuredBuffer<uint16_t>";
    const F16S_MUT: &'static str = "RWStructuredBuffer<uint16_t>";
    const I32: &'static str = "int";
    const U32: &'static str = "uint";
    const F32: &'static str = "float";
    const USIZE: &'static str = "";
    const IN_PACKED: &'static str = "uint";
}

/// The operand vocabulary, from the crate that holds it once.
///
/// Re-exported rather than named through `kernels::shader` at every use, so a
/// body's signature reads as this backend's own and a family file imports one
/// module.
pub use kernels::shader::{Bind, InPacked, Tensor, Usize, bf16, f16};

/// What a routine body dispatches through, spelled as a body writes it.
///
/// `dyn Encode + 'a`, and the lifetime is why [`Backend::Ctx`] is a generic
/// associated type rather than a plain one. A Vulkan `Encode` BORROWS
/// everything it needs — the `ash::Device`, the command buffer mid-recording,
/// the descriptor pool, the pipeline cache — all of which live in the
/// driver's frame, so an implementor is never `'static` and a bare
/// `dyn Encode` (which means `dyn Encode + 'static`) could not name one.
/// Bodies take `&Ctx<'_>`.
pub type Ctx<'a> = dyn Encode + 'a;

// THE ASKING SIDE, over the one method the driver implements.
//
// `Asks` is blanket-implemented for every `Answers`, so this is the whole of
// what a plane states to give its bodies `ctx.ask::<C, keys::X>()`,
// `ctx.params()` and `ctx.absent()`.
impl kernels::routine::Answers<Vulkan> for Ctx<'_> {
    fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
        Encode::resolve(self, ty, source)
    }
}

/// One routine, in this backend's instantiation of the machinery.
pub type Routine = kernels::routine::Routine<Vulkan>;



/// The two launch rules a shader-plane body states its rectangle with.
///
/// Re-exported and not re-written. Both of these, and the `rectangle` check
/// under them, stood here character for character in `kernels-metal`,
/// `kernels-vulkan` and `kernels/src/shader.rs` — one function in three
/// copies, reached by three different paths, so a body could be told apart by
/// which spelling it used. There is one now, and it is the shared one.
pub use kernels::shader::{elementwise, elementwise_rows};

/// The fact keys a BODY asks the runtime with — `ctx.ask::<i32, keys::Rows>()`.
///
/// Not what a signature binds its scalars from any more: a scalar the
/// checkpoint fixes is a `Const` the statement carries, and a key names only
/// what a fire decides.
pub use kernels::keys;
pub use kernels::routine::{Const, Fire, In, InOut, Out};

/// Where a body turns an entrypoint and a tier into the artifact it fires.
pub use crate::module::path as module_path;

/// What a body asks the runtime for, once `Env` is out of the parameter list.
pub use kernels::routine::{Answers, Asks};


#[cfg(test)]
mod tests {
    use super::*;

    /// Every argument kind refuses a value of another kind, by position (the
    /// width check described on [`ArgValue`]).
    #[test]
    fn an_argument_refuses_a_value_of_another_kind() {
        assert_eq!(
            <i32 as Arg<Vulkan>>::unpack(&ArgValue::Usize(1), 3),
            Err(Refusal::Kind {
                at: 3,
                want: Ty::I32
            })
        );
        assert_eq!(
            <Usize as Arg<Vulkan>>::unpack(&ArgValue::I32(1), 0),
            Err(Refusal::Kind {
                at: 0,
                want: Ty::Usize
            })
        );
        assert_eq!(
            <Tensor<bf16> as Arg<Vulkan>>::unpack(&ArgValue::F32(1.0), 7),
            Err(Refusal::Kind {
                at: 7,
                want: Ty::Bf16s
            })
        );
        // And a buffer handle is accepted by every buffer kind, because the
        // handle carries no element type -- the TYPE does, which is what makes
        // a body handed the wrong one fail to compile rather than at runtime.
        assert_eq!(
            <Tensor<bf16> as Arg<Vulkan>>::unpack(
                &ArgValue::Buffer {
                    handle: 9,
                    writes: false,
                    rows: 0,
                    width: 0
                },
                0
            ),
            Ok(Tensor::<bf16>::new(9))
        );
        assert_eq!(
            <Tensor<u32> as Arg<Vulkan>>::unpack(
                &ArgValue::Buffer {
                    handle: 9,
                    writes: false,
                    rows: 0,
                    width: 0
                },
                0
            ),
            Ok(Tensor::<u32>::new(9))
        );
    }

    /// `InPacked` takes a `u32`'s value and is not a `u32`.
    ///
    /// It is the one kind whose difference is not width but PLACE: the value
    /// rides a field of a struct an earlier buffer binds. A signature that
    /// spelled it `u32` would be one the driver cannot tell apart from a push
    /// block scalar.
    #[test]
    fn a_packed_field_is_a_u32_that_is_not_the_u32_kind() {
        assert_eq!(
            <InPacked as Arg<Vulkan>>::unpack(&ArgValue::U32(5), 0),
            Ok(InPacked(5))
        );
        assert_ne!(
            <InPacked as Arg<Vulkan>>::TY,
            <u32 as Arg<Vulkan>>::TY,
            "the two are the same width and different arguments"
        );
    }

    /// The ELEMENT decides the `Ty`, and nothing else does.
    #[test]
    fn a_tensor_takes_its_argument_type_from_its_element() {
        // `Tensor<i32>` IS THE CARRIER NOW, and it has no provenance to
        // assert: `Env` and `Provenance` are both deleted, so what is left
        // to check is that the element decides the `Ty`.
        assert_eq!(<Tensor<i32> as Arg<Vulkan>>::TY, Ty::I32s);
        assert_eq!(<Tensor<u8> as Arg<Vulkan>>::TY, Ty::U8s);
    }

    /// The declared spellings are the shader tree's, not invented.
    ///
    /// Each of these was read out of `kernels/` before it was written here,
    /// and the count beside it is how many declarations use it. The point of
    /// `SPELLING` is a future generated cross-check against the real module,
    /// which is worth nothing if the strings were guessed.
    #[test]
    fn the_spellings_are_the_ones_the_slang_tree_declares() {
        // ONE CONST, ONE VALUE. The line under the first of these used to
        // assert the SAME const equalled `"RWStructuredBuffer<PIE_ACT>"` --
        // two contradictory claims about `<Buf as Arg<Vulkan>>::SPELLING`,
        // which is a single string. It is the residue of the `Buf`/`BufMut`
        // merge: the second line read `BufMut` until `BufMut` stopped
        // existing and the rename swept it into its neighbour. The direction
        // now rides the MARK, so the writable spelling is `Lang::BF16S_MUT`
        // and no type names it. `Tensor<bf16>`'s own reading routes through
        // `Element::SPELL`, which is `Lang::BF16S` and not `Lang::BUF` --
        // Slang has no bf16 storage type, so both halves it and `f16` alike
        // as `uint16_t`, which is why this is not `PIE_ACT`: that macro is
        // `common/bf16.slang`'s own name for the same expansion, and the
        // element decides the `Ty` on a string the SIGNATURE never sees.
        assert_eq!(<Tensor<bf16> as Arg<Vulkan>>::SPELLING, "StructuredBuffer<uint16_t>");
        assert_eq!(
            <Vulkan as kernels::shader::Lang>::BF16S_MUT,
            "RWStructuredBuffer<uint16_t>"
        );
        assert_eq!(<Tensor<i32> as Arg<Vulkan>>::SPELLING, "StructuredBuffer<int>");
        assert_eq!(<Tensor<u32> as Arg<Vulkan>>::SPELLING, "StructuredBuffer<uint>");
        assert_eq!(<Tensor<u8> as Arg<Vulkan>>::SPELLING, "StructuredBuffer<uint8_t>");
        // Empty, and deliberately: nothing in the tree declares a 64-bit
        // shader integer, so there is no spelling to record. A guess here
        // would be worse than the gap.
        assert_eq!(<Usize as Arg<Vulkan>>::SPELLING, "");
    }
}

// THE PLANE'S `routine!` WRAPPER STOOD HERE AND HAS NO CALLERS.
//
// It filled this backend in so a membership list could name only the
// `fn`. There is no membership list: `#[routine]` builds the row beside
// the `fn` and a distributed slice collects it, so the only caller of
// `kernels::routine!` is the attribute, which names the backend through
// `crate::Plane`.
