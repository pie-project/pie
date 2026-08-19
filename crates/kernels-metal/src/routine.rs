//! This backend's instantiation of the `kernels` routine machinery.
//!
//! A routine is an ordinary `fn` that computes a launch and dispatches it.
//! Everything a row used to STATE — the launch rule, the grid parameter, the
//! is derived from the signature by [`routine!`]. It mirrors
//! `kernels-vulkan/src/routine.rs` where Metal has no reason to differ. The
//! places it does:
//!
//! * **The context is a trait, not a struct.** This crate depends on `kernels`
//!   and nothing else, while the `MTLDevice`, the command buffer, the argument
//!   table and every compiled pipeline live in `driver-metal`. So a body
//!   dispatches through [`Encode`] and `Backend::Ctx` is `dyn Encode`.
//! * **A [`Fire`] names its FILE.** There is no ahead-of-time shader build:
//!   `driver-metal` compiles from `(path, entry name)` at run time, and the
//!   path is not derivable from the entrypoint — `quant/qmm_t.metal` holds 54
//!   of them.
//! * **A [`Fire`] states its THREADGROUP.** MSL declares no workgroup size in
//!   the shader text, so there is nothing to reflect and no divisor to
//!   recover; `dispatchThreads:threadsPerThreadgroup:` takes both numbers and
//!   the kernel's own reductions depend on the second.
//! * **There is no capability tier.** One source compiles to one pipeline;
//!   what varies is the entrypoint, which a body spells.
//!
//! A buffer carries no writability, though it looks like it should: Metal's
//! ordering comes from the encoder, so `Buf` versus `Buf` stays a property
//! of the signature's TYPE — which is what makes a body handed the wrong one
//! fail to compile.

#[cfg(test)]
use kernels::Ty;
#[cfg(test)]
use kernels::routine::Arg;
use kernels::routine::{Backend, Extent, Refusal};
use kernels::shader::ShaderValue;

/// This backend, as the machinery names it. A marker: never constructed,
/// carrying only the two concrete types the `kernels` machinery is generic over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Metal;

impl Backend for Metal {
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
            ArgValue::Shaped { rows, width, .. } => Ok(Extent { rows, width }),
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
/// widths differ and the check that matters is exactly the width one. Metal's
/// scalars ride a per-dispatch parameter run packed into `ParamSlot`s of a
/// stated byte width; a [`Ty::Usize`] value landing in a four-byte slot either
/// truncates or writes over its neighbour, unreported.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    /// A device allocation, named by whatever index the caller keys its own
    /// buffers on. The handle is opaque on purpose: this crate cannot name an
    /// `MTLBuffer` or a GPU address and does not need to — the driver resolves
    /// handles to its own `Slice`s before it calls.
    Buffer(u32),
    /// The same, bound at a [`Ty::Buf`] or [`Ty::F32s`] argument.
    ///
    /// # The direction has to ride on the VALUE
    ///
    /// [`ShaderValue::buffer_mut`] exists for this and Metal took its
    /// default, so `driver-metal`'s `directed` had to recover the direction
    /// by indexing the SIGNATURE at the value's position. That is only right
    /// when a routine's dispatch list is its parameter list, and twenty-three
    /// of them are not: an entrypoint with holes in its buffer numbering
    /// fills them with a `pad` taken once and bound many times, so every
    /// argument after the first hole reads its neighbour's type.
    ///
    /// `gdn_core_recurrent_prefill` was the one that showed: `core_out` sits
    /// at slot 3, the signature's third parameter is `pre_q`, and the scan's
    /// only real output was declared a READ. The encoder saw no hazard
    /// between it and the `gated_rms` that consumes it, ran the two at once,
    /// and qwen3.6 answered differently every time it was asked the same
    /// question. `qmm_splitk_reduce` declared its `y` the same way, and
    /// `cast_qmm_input_bfloat16_to_float16` declared `half_out` not at all.
    BufferMut(u32),
    /// A 32-bit signed scalar.
    I32(i32),
    /// A 32-bit unsigned scalar.
    U32(u32),
    /// A 32-bit float.
    F32(f32),
    /// A 64-bit stride or extent, which is what [`Ty::Usize`] means here.
    Usize(u64),
    /// A DEVICE ALLOCATION AND THE RECTANGLE THE STATEMENT GAVE IT.
    ///
    /// Minted by [`kernels::bind`] for an operand slot, and consumed by the
    /// mark that unpacks it: `In<Tensor<bf16>>` keeps the shape, so `x.width`
    /// is where a body reads its own operand's pitch. That is what took
    /// `Width`, `InWidth` and `OutWidth0` off 337 parameter lists -- they were
    /// a fact the operand beside them already implied, and the only reason
    /// they could not come off the mark was that this plane bound addresses
    /// alone.
    ///
    /// # It never reaches `Encode::fire`
    ///
    /// A body re-emits its operands through [`kernels::Bind::arg`], which
    /// mints a plain [`Self::Buffer`]. So the shape exists between `bind` and
    /// `unpack` and nowhere else, and no encoder has to know about it.
    Shaped {
        /// The caller's index for the allocation.
        handle: u32,
        /// Rows in this launch's rectangle.
        rows: i32,
        /// Elements per row. Zero where the statement gave none.
        width: i32,
    },
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
            Self::Buffer(_) => "a buffer",
            Self::Shaped { .. } => "a buffer",
            Self::BufferMut(_) => "a writable buffer",
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
/// Implemented by `driver-metal`. The split is the crate boundary: a body
/// knows the entrypoint, the file and the extent, and the driver knows what a
/// buffer is, which argument-table index a binding lands in, how the scalar
/// run is packed, and which pipelines a fire needs compiled.
///
/// `driver-metal` separates PLANNING from ENCODING: the argument table is
/// sized from the whole plan, pipelines are compiled in one batch, and the
/// plan is fingerprinted for indirect-command-buffer replay, all before an
/// encoder exists. So `dispatch` APPENDS a `Dispatch` to the plan — a body
/// replaces `plan_one`, not `encode_one`, and the plan stays data.
pub trait Encode {
    /// Record one dispatch.
    ///
    /// `args` is the routine's own argument list, in signature order — which
    /// is the SHADER's binding order, not the trace's. The two differ for most
    /// of this tree's rectangles, and reconciling them is the caller's job.
    /// The implementor separates buffers from scalars by variant, which makes
    /// the split derivable rather than stated twice.
    ///
    /// # Errors
    ///
    /// Whatever the binding refused, as a [`Refusal`].
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
}

/// This backend's value, as the shared operand types read it.
///
/// The ten operand types live in [`kernels::shader`] because the vocabulary is
/// closed and identical in metal, vulkan and wgpu. What is NOT shared is the
/// value, which is [`ArgValue`], and this impl is the whole of what a shared
/// operand type needs to know about it.
impl ShaderValue for ArgValue {
    fn as_buffer(self) -> Option<u32> {
        match self {
            Self::Shaped { handle, .. } => Some(handle),
            Self::Buffer(handle) | Self::BufferMut(handle) => Some(handle),
            _ => None,
        }
    }
    fn buffer_mut(handle: u32) -> Self {
        Self::BufferMut(handle)
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
            Self::Shaped { rows, width, .. } => Some((rows, width)),
            _ => None,
        }
    }
    fn buffer_at(handle: u32, rows: i32, width: i32) -> Self {
        Self::Shaped { handle, rows, width }
    }
    fn buffer_mut_at(handle: u32, rows: i32, width: i32) -> Self {
        Self::Shaped { handle, rows, width }
    }
    fn buffer(handle: u32) -> Self {
        Self::Buffer(handle)
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

/// How MSL spells the twelve, as this tree's shaders declare them.
///
/// Every string was counted in `kernels/` before it was written here:
/// `const device T*` 248 declarations, `device T*` 91, `const device int*` 58,
/// `const constant int&` 259, `const constant float&` 48, `constant uint&` 2.
/// Two of the ten have TWO live spellings and the more common one is recorded:
/// `const device uint32_t*` (62) over `const device uint*` (44), and
/// `const device uint8_t*` (38) over `const device uchar*` (22) — same MSL
/// types, so a cross-check built on `SPELLING` has to accept both.
///
/// `USIZE` is empty on purpose: MSL has `ulong`, but nothing in this tree
/// declares one — every live use is an extent the driver resolves on the host
/// and narrows before it reaches a parameter slot.
impl kernels::shader::Lang for Metal {
    const BUF: &'static str = "const device T*";
    const BUF_MUT: &'static str = "device T*";
    const I32S: &'static str = "const device int*";
    const U32S: &'static str = "const device uint32_t*";
    const U8S: &'static str = "const device uint8_t*";
    const F32S: &'static str = "const device float*";
    const F32S_MUT: &'static str = "device float*";
    // THE ONE LANGUAGE OF THE THREE THAT HAS THE TYPE. MSL declares `bfloat`
    // and this tree writes it 129 times; `device T*` is the TEMPLATE
    // parameter, written 346 times, and a signature that named the element
    // was spelling the second where the kernel meant the first.
    const BF16S: &'static str = "const device bfloat*";
    const BF16S_MUT: &'static str = "device bfloat*";
    const F16S: &'static str = "const device half*";
    const F16S_MUT: &'static str = "device half*";
    const I32: &'static str = "const constant int&";
    const U32: &'static str = "constant uint&";
    const F32: &'static str = "const constant float&";
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
/// associated type: a Metal `Encode` BORROWS what it appends to — the plan
/// under construction, and the resolver the operands were bound through — so
/// an implementor is never `'static`. Bodies take `&Ctx<'_>`.
pub type Ctx<'a> = dyn Encode + 'a;

// THE ASKING SIDE, over the one method the driver implements.
//
// `Asks` is blanket-implemented for every `Answers`, so this is the whole of
// what a plane states to give its bodies `ctx.ask::<C, keys::X>()`,
// `ctx.params()` and `ctx.absent()`.
impl kernels::routine::Answers<Metal> for Ctx<'_> {
    fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
        Encode::resolve(self, ty, source)
    }
}

/// One routine, in this backend's instantiation of the machinery.
pub type Routine = kernels::routine::Routine<Metal>;



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

/// What a body asks the runtime for, once `Env` is out of the parameter list.
pub use kernels::routine::{Answers, Asks};


#[cfg(test)]
mod tests {
    use super::*;

    /// Every argument kind refuses a value of another kind, by position.
    ///
    /// The check that matters is the WIDTH one, and it is the reason the
    /// scalar kinds are separate variants at all: a `Usize` is eight bytes and
    /// an `I32` is four, so a value that crossed between them either truncates
    /// or writes over its neighbour, and `driver-metal` would report neither.
    #[test]
    fn an_argument_refuses_a_value_of_another_kind() {
        assert_eq!(
            <i32 as Arg<Metal>>::unpack(&ArgValue::Usize(1), 3),
            Err(Refusal::Kind {
                at: 3,
                want: Ty::I32
            })
        );
        assert_eq!(
            <Usize as Arg<Metal>>::unpack(&ArgValue::I32(1), 0),
            Err(Refusal::Kind {
                at: 0,
                want: Ty::Usize
            })
        );
        assert_eq!(
            <Tensor<bf16> as Arg<Metal>>::unpack(&ArgValue::F32(1.0), 7),
            Err(Refusal::Kind {
                at: 7,
                want: Ty::Bf16s
            })
        );
        // And a buffer handle is accepted by every buffer kind, because the
        // handle carries no element type -- the TYPE does, which is what makes
        // a body handed the wrong one fail to compile rather than at runtime.
        assert_eq!(
            <Tensor<bf16> as Arg<Metal>>::unpack(&ArgValue::Buffer(9), 0),
            Ok(Tensor::<bf16>::new(9))
        );
        assert_eq!(
            <Tensor<u32> as Arg<Metal>>::unpack(&ArgValue::Buffer(9), 0),
            Ok(Tensor::<u32>::new(9))
        );
    }

    /// A writable buffer is a different VALUE from a read-only one.
    ///
    /// This asserted the opposite, on the reasoning that "Metal's ordering is
    /// the encoder's, so this backend takes `ShaderValue::buffer_mut`'s
    /// default and the distinction stays where it is checkable -- in the
    /// signature's type". Metal's ordering IS the encoder's, and the encoder
    /// reads `Touches`, and `Touches` is built from the values: a driver that
    /// wanted the signature's type had to index the signature at the value's
    /// POSITION, which is the same list only for a routine that binds its
    /// parameters in order. Twenty-three of them bind a `pad` at the holes in
    /// their entrypoint's buffer numbering and do not.
    ///
    /// `ssm::gdn_core_recurrent_prefill` is the one that showed: its
    /// `core_out` was declared a read, the `gated_rms` consuming it ran
    /// alongside it, and qwen3.6 answered the same two-token prompt
    /// differently every time. Both facts are still stated -- the type in the
    /// signature, the direction on the value -- and the second is the one the
    /// driver can use.
    #[test]
    fn writability_is_a_value_and_not_only_a_type() {
        assert_ne!(
            <ArgValue as ShaderValue>::buffer_mut(4),
            <ArgValue as ShaderValue>::buffer(4)
        );
        assert_eq!(
            <ArgValue as ShaderValue>::buffer_mut(4),
            ArgValue::BufferMut(4)
        );
        assert_ne!(
            <In<Tensor<bf16>> as Arg<Metal>>::TY,
            <Out<Tensor<bf16>> as Arg<Metal>>::TY
        );
        // And a mutable value still unpacks as its handle, because a routine
        // that receives one reads a handle and not a direction.
        assert_eq!(
            <Tensor<bf16> as Arg<Metal>>::unpack(&ArgValue::BufferMut(9), 0),
            Ok(Tensor::<bf16>::new(9))
        );
    }

    /// `InPacked` takes a `u32`'s value and is not a `u32`.
    ///
    /// It is the one kind whose difference is not width but PLACE: the value
    /// rides a field of a struct an earlier buffer binds.
    #[test]
    fn a_packed_field_is_a_u32_that_is_not_the_u32_kind() {
        assert_eq!(
            <InPacked as Arg<Metal>>::unpack(&ArgValue::U32(5), 0),
            Ok(InPacked(5))
        );
        assert_ne!(
            <InPacked as Arg<Metal>>::TY,
            <u32 as Arg<Metal>>::TY,
            "the two are the same width and different arguments"
        );
    }

    /// The ELEMENT decides the `Ty`, and nothing else does.
    #[test]
    fn a_tensor_takes_its_argument_type_from_its_element() {
        // `Tensor<i32>` IS THE CARRIER NOW, and it has no provenance to
        // assert: `Env` and `Provenance` are both deleted, so what is left
        // to check is that the element decides the `Ty`.
        assert_eq!(<Tensor<i32> as Arg<Metal>>::TY, Ty::I32s);
        assert_eq!(<Tensor<u8> as Arg<Metal>>::TY, Ty::U8s);
    }

    /// The declared spellings are the shader tree's, not invented. The point of
    /// `SPELLING` is a generated cross-check against the real shader, which is
    /// worth nothing if the strings were guessed.
    #[test]
    fn the_spellings_are_the_ones_the_msl_tree_declares() {
        // ONE CONST, ONE VALUE. This used to assert the SAME const equalled
        // BOTH `"const device T*"` (`Buf`'s spelling) AND `"device T*"`
        // (`BufMut`'s) -- two contradictory claims about a single string,
        // the residue of the `Buf`/`BufMut` merge: `Tensor<bf16>` is the
        // ACTIVATION element, which `Lang::BF16S`/`BF16S_MUT` name, not the
        // generic `Lang::BUF`/`BUF_MUT` a bare opaque buffer would have
        // routed through. `Arg<Metal>::SPELLING` for `Tensor<E>` reads
        // `E`'s own spell off `Lang`, and for `bf16` that spell is `BF16S`,
        // never `BUF` -- `BUF`/`BUF_MUT` are what a `Tensor<E>` for some
        // OTHER, unnamed element would have spelled, and nothing in this
        // tree carries one. The direction now rides the MARK, so the
        // writable spelling is `Lang::BF16S_MUT` and no type names it.
        assert_eq!(
            <Tensor<bf16> as Arg<Metal>>::SPELLING,
            "const device bfloat*"
        );
        assert_eq!(<Metal as kernels::shader::Lang>::BF16S_MUT, "device bfloat*");
        assert_eq!(<Tensor<i32> as Arg<Metal>>::SPELLING, "const device int*");
        assert_eq!(<Tensor<u32> as Arg<Metal>>::SPELLING, "const device uint32_t*");
        assert_eq!(<Tensor<u8> as Arg<Metal>>::SPELLING, "const device uint8_t*");
        // The element the tree declares 129 times, which `Buf` withheld.
        // `bf16`/`f16` are `Element` markers now, not `Arg<Metal>` carriers
        // -- `Tensor<bf16>`/`Tensor<f16>` are -- so the concrete spelling is
        // read off `Lang` directly, the same way `BF16S_MUT` is two lines up.
        assert_eq!(<Metal as kernels::shader::Lang>::BF16S, "const device bfloat*");
        assert_eq!(<Metal as kernels::shader::Lang>::F16S, "const device half*");
        // Empty, and deliberately: nothing in the tree declares a 64-bit
        // shader integer, so there is no spelling to record.
        assert_eq!(<Usize as Arg<Metal>>::SPELLING, "");
    }
}

// THE PLANE'S `routine!` WRAPPER STOOD HERE AND HAS NO CALLERS.
//
// It filled this backend in so a membership list could name only the
// `fn`. There is no membership list: `#[routine]` builds the row beside
// the `fn` and a distributed slice collects it, so the only caller of
// `kernels::routine!` is the attribute, which names the backend through
// `crate::Plane`.
