//! The shader backends' operand vocabulary, written once.
//!
//! Identical in metal, vulkan and wgpu, so it is declared here rather than
//! three times. The count is [`COUNT`], asserted rather than narrated, and it
//! may rise as more signatures are written down.
//!
//! What is NOT shared is the value a backend binds, so the impls are generic
//! over [`ShaderValue`], which asks only how to read a buffer handle or a
//! scalar out of one.
//!
//! CUDA does not use this: its vocabulary has pointer arrays, by-value
//! aggregates and a `void*` ABI that drops the operand type.

use crate::Ty;
use crate::routine::{Arg, Backend, Refusal};

/// How many operand types this module declares.
///
/// Asserted by `the_vocabulary_is_the_size_this_module_says_it_is`, so the
/// sentence above cannot drift from the list below — which it did once, from
/// ten to twelve, when `ssm` crossed and brought the float arrays with it.
///
/// It had drifted again by the time the census could run: the duplicated
/// `mod tests` below kept this file's tests from compiling, so `12` sat here
/// over a vocabulary of TEN -- `BufMut`, `F32sMut`, `I32sMut` and `U32sMut`
/// were deleted when the direction moved to the mark and this did not follow.
/// The two half-width elements bring it back to twelve, which is now a
/// measured number rather than a coincidence.
pub const COUNT: usize = 12;

/// The little a shared operand type needs to know about a backend's value.
///
/// Implemented by each shader backend for its own `ArgValue`. Every method is
/// total: a value that is not of the asked-for kind answers `None`, and the
/// operand type turns that into [`Refusal::Kind`] naming the position.
pub trait ShaderValue: Copy {
    /// The buffer handle this value names, if it names one.
    fn as_buffer(self) -> Option<u32>;
    /// The 32-bit signed scalar this value is, if it is one.
    fn as_i32(self) -> Option<i32>;
    /// The 32-bit unsigned scalar this value is, if it is one.
    fn as_u32(self) -> Option<u32>;
    /// The 32-bit float this value is, if it is one.
    fn as_f32(self) -> Option<f32>;
    /// The 64-bit extent this value is, if it is one.
    fn as_usize(self) -> Option<u64>;

    /// The raised HOST object this value carries, if it carries one — the
    /// address of a driver-built view (`PagedKvView`, a plan carve). Carried
    /// as an address rather than a pointer so the value stays `Send` where a
    /// plane needs it; the ends cast. Default `None`: a plane that never
    /// binds a view answers honestly.
    fn as_raised(self) -> Option<usize> {
        None
    }

    /// A value carrying a raised host object's address. The panic default is
    /// for planes that declare no views; a plane whose routines take
    /// `In<Struct<..>>` overrides with its own variant.
    #[must_use]
    fn raised(addr: usize) -> Self {
        let _ = addr;
        panic!("this plane binds no raised views");
    }

    /// The rectangle the statement gave this operand, if it gave one.
    ///
    /// `(rows, width)`, and both zero is the honest answer for a statement
    /// that stated neither -- [`crate::Region`] refuses on a zero width and
    /// `Layout::packed(0, 0)` is a legal empty, so the absent case lands on
    /// the value every reader already checks.
    ///
    /// # Why a handle plane answers this at all
    ///
    /// Because `Width`, `InWidth` and `OutWidth0` were 337 uses of a fact the
    /// operand beside them already implied, and the reason they could not come
    /// off the mark was that only CUDA minted region-shaped values. Now
    /// [`bind`](crate::bind::bind) mints them here too, out of the same
    /// `Holds::in_width`/`out_width` a `Kind::InWidth` slot already read.
    fn as_extent(self) -> Option<(i32, i32)> {
        None
    }

    /// A value naming a buffer the launcher only reads.
    fn buffer(handle: u32) -> Self;

    /// The same, carrying the rectangle the statement gave it.
    ///
    /// Defaults to dropping the shape, which is what a plane that has nowhere
    /// to put it must do. A plane that overrides gets `x.width` in its bodies.
    #[must_use]
    fn buffer_at(handle: u32, rows: i32, width: i32) -> Self {
        let _ = (rows, width);
        Self::buffer(handle)
    }

    /// [`Self::buffer_mut`], carrying the rectangle. See [`Self::buffer_at`].
    #[must_use]
    fn buffer_mut_at(handle: u32, rows: i32, width: i32) -> Self {
        let _ = (rows, width);
        Self::buffer_mut(handle)
    }
    /// A value naming a buffer the launcher may WRITE through.
    ///
    /// Defaults to [`Self::buffer`]: wgpu binds every storage entry the same
    /// way and metal reads writability elsewhere.
    ///
    /// `driver-vulkan` overrides it, and the reason is measured: it barriers
    /// two dispatches only when they touch the same bytes, decided from which
    /// operands a launch may WRITE. Barriering every neighbouring pair costs
    /// 8 microseconds each on an RTX 4090 -- 3.8 ms of a 7.2 ms decode. A
    /// routine states the direction in the argument TYPE, and this is how that
    /// reaches a driver that sees only values.
    ///
    /// Not overriding it loses nothing: every buffer reads as written, which
    /// is the coarse and SAFE direction.
    #[must_use]
    fn buffer_mut(handle: u32) -> Self {
        Self::buffer(handle)
    }
    /// A value carrying a 32-bit signed scalar.
    fn i32(v: i32) -> Self;
    /// A value carrying a 32-bit unsigned scalar.
    fn u32(v: u32) -> Self;
    /// A value carrying a 32-bit float.
    fn f32(v: f32) -> Self;
    /// A value carrying a 64-bit extent.
    fn usize(v: u64) -> Self;
    /// A value carrying a 64-bit signed scalar.
    ///
    /// Defaults to [`Self::i32`] truncated, because no shader plane has a
    /// 64-bit scalar carrier: WGSL and Slang have no `i64` a kernel takes by
    /// value, so a signature that spelled one could not have reached them.
    /// CUDA overrides it -- `long long` is a real parameter width there.
    #[must_use]
    fn i64(v: i64) -> Self {
        Self::i32(v as i32)
    }
    /// A value carrying a one-bit flag.
    ///
    /// Defaults to a zero-or-one `i32`, which is how the three shader planes
    /// carry every flag: a uniform block has no one-byte cell. CUDA
    /// overrides it, because `bool` is one byte in the ABI and a four-byte
    /// write there is a write over the neighbouring parameter.
    #[must_use]
    fn bool(v: bool) -> Self {
        Self::i32(i32::from(v))
    }
}

/// How one shader LANGUAGE spells the operand types.
///
/// The vocabulary is one closed set; the SPELLINGS are not, there being three
/// languages. One shared constant would have to be one of the three and
/// therefore wrong in the other two.
///
/// Implemented on the backend MARKER and not on its value: the value is a
/// runtime binding, the spelling a property of the text the kernel is in.
pub trait Lang: Backend {
    /// A read-only opaque buffer.
    const BUF: &'static str;
    /// A writable opaque buffer.
    const BUF_MUT: &'static str;
    /// A read-only array of `i32`.
    const I32S: &'static str;
    /// A read-only array of `u32`.
    const U32S: &'static str;
    /// A read-only array of `u8`.
    const U8S: &'static str;
    /// A read-only array of `f32`.
    const F32S: &'static str;
    /// An array of `f32` the launcher may WRITE through.
    ///
    /// Two spellings and not one because two of the three languages really do
    /// have two: Slang writes `RWStructuredBuffer<float>` against
    /// `StructuredBuffer<float>` and MSL writes `device float*` against `const
    /// device float*`. WGSL does NOT -- the element type is `array<f32>` for
    /// both and the access lives in the `var<storage, read_write>` that
    /// declares the binding, not in the type. So `Wgpu` spells these two the
    /// same, and that is the language rather than an omission.
    const F32S_MUT: &'static str;
    /// A read-only array of the ACTIVATION element.
    ///
    /// Spelled per language because no language has a bf16 storage type and
    /// each works around it differently: Slang declares the sixteen bits as
    /// `uint16_t`, MSL has a real `bfloat`, and WGSL has no 16-bit type at all
    /// and packs pairs into `array<u32>`. Three workarounds, one element.
    ///
    /// This is what [`Self::BUF`] used to carry. `BUF` spells the activation
    /// on every plane -- `StructuredBuffer<PIE_ACT>`, `const device T*` -- so
    /// the type a signature named "an opaque buffer" was never opaque; it was
    /// the activation with its name withheld. See [`crate::shader::bf16`].
    const BF16S: &'static str;
    /// [`Self::BF16S`], where the launcher may write through it.
    const BF16S_MUT: &'static str;
    /// A read-only array of the OTHER half-width float, where a launcher
    /// casts into it rather than reading the activation as it stands.
    ///
    /// `quant`'s `*_fp16_precast` family is the whole of it: it narrows bf16
    /// to fp16 once and multiplies in fp16 after, so its scratch is a
    /// different element from the activation beside it. The two are both
    /// sixteen bits and Slang declares both `uint16_t`, which is exactly why
    /// the SIGNATURE has to distinguish them -- the shader cannot.
    const F16S: &'static str;
    /// [`Self::F16S`], where the launcher may write through it.
    const F16S_MUT: &'static str;
    /// A 32-bit signed scalar.
    const I32: &'static str;
    /// A 32-bit unsigned scalar.
    const U32: &'static str;
    /// A 32-bit float.
    const F32: &'static str;
    /// A 64-bit extent, as the shader receives it.
    ///
    /// Empty for a language whose kernels here declare no 64-bit integer and
    /// receive the value already narrowed — an empty spelling says "this
    /// backend has not written one down", which is what [`Arg::SPELLING`]'s
    /// own doc reserves it for, and is honest where a guess would not be.
    const USIZE: &'static str;
    /// A `u32` that rides a field of a bound struct.
    const IN_PACKED: &'static str;
}

pub use crate::routine::Bind;

/// WHAT A SHADER-PLANE TENSOR IS MADE OF.
///
/// The element half of [`Tensor`], and the whole of what used to be seven
/// ad-hoc buffer names — `Buf`, `bf16`, `f16`, `I32s`, `U32s`, `U8s`, `F32s`.
/// They were one constructor spelled seven ways, and the cost of that was
/// measured: `kernels-wgpu`'s `is_buffer` listed five of them twice each and
/// omitted `Bf16s`, which is what a naming habit costs when the names are not
/// related by construction.
///
/// `Tensor<f32>` and `f32` are related by construction. `F32s` and `f32` were
/// related by a habit.
pub trait Element: 'static {
    /// The [`Ty`] a READ of a tensor of this element binds as.
    const TY_CONST: Ty;
    /// The [`Ty`] a WRITE binds as.
    ///
    /// The value cannot tell the two apart — `ArgValue::Buffer(handle)` either
    /// way — and the TABLE must, because the driver reads it to decide hazards
    /// and barriers. The mark says which; this is the pair it picks from.
    const TY_MUT: Ty;
    /// Which of [`Lang`]'s spellings names it.
    const SPELL: Spell;
}

/// One element's place in a [`Lang`]'s spelling table.
///
/// An enum and not a `&'static str`, because the string is the LANGUAGE's and
/// the element is the tensor's: Slang says `StructuredBuffer<uint16_t>` where
/// MSL says `const device bfloat*`, and one shared constant would have to be
/// one of the three and therefore wrong in the other two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spell {
    /// The activation element.
    Bf16,
    /// The other half-width float, which is NOT the activation.
    F16,
    /// A float array.
    F32,
    /// A signed 32-bit array.
    I32,
    /// An unsigned 32-bit array — the CSR/indptr family.
    U32,
    /// A byte array — the per-row validity masks.
    U8,
}

/// How `B` spells a READ of this element.
const fn spell_read<B: Lang>(s: Spell) -> &'static str {
    match s {
        Spell::Bf16 => B::BF16S,
        Spell::F16 => B::F16S,
        Spell::F32 => B::F32S,
        Spell::I32 => B::I32S,
        Spell::U32 => B::U32S,
        Spell::U8 => B::U8S,
    }
}

// THERE IS NO `spell_write`. A `RWStructuredBuffer<T>` and a
// `StructuredBuffer<T>` are one VALUE -- `ArgValue::Buffer(handle)` either way
// -- and which way the launch drives the operand is the MARK's answer, carried
// in the `Ty` column as `Bf16s` against `Bf16sMut`. `Lang`'s `*_MUT` spellings
// stay declared because a generated cross-check against the real shader
// declaration needs them; nothing in the signature picks between them.

/// Declare one element.
macro_rules! element {
    ($(#[$m:meta])* $name:ty, $ty:expr, $ty_mut:expr, $spell:ident) => {
        impl Element for $name {
            const TY_CONST: Ty = $ty;
            const TY_MUT: Ty = $ty_mut;
            const SPELL: Spell = Spell::$spell;
        }
    };
}

/// THE ACTIVATION ELEMENT.
///
/// A marker and not a number: nothing host-side holds a bf16, and the point of
/// the type is that a signature can NAME the element the shader declares.
/// `Buf` was never opaque — Slang's [`Lang::BUF`] is
/// `StructuredBuffer<PIE_ACT>` and MSL's is `const device T*` — so 381 vulkan
/// operands, 379 wgpu ones and 389 metal ones declared "a buffer" where the
/// text declared an activation, while CUDA spelled the same operand
/// `In<bf16>` throughout.
///
/// Lowercase, and matching CUDA's: `kernels_cuda::jit::abi::bf16` is the
/// element a CUDA signature names and this is the element a shader signature
/// names. They cannot be ONE type — [`crate::routine::Elem`] has a single
/// `Read` carrier and CUDA's is a pointer where a shader's is a binding index
/// — but they can read alike, and a reader crossing the two files should not
/// have to learn a second word for one element.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct bf16;

/// The other half-width float, which is NOT the activation.
///
/// Its own element because `quant`'s `*_fp16_precast` routines hold both at
/// once — `half_in` beside `x` — and Slang spells them the same, `uint16_t`.
/// A signature that called both the activation would be stating that the
/// narrowing does not happen.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct f16;

element!(bf16, Ty::Bf16s, Ty::Bf16sMut, Bf16);
element!(f16, Ty::F16s, Ty::F16sMut, F16);
element!(f32, Ty::F32s, Ty::F32sMut, F32);
element!(i32, Ty::I32s, Ty::I32sMut, I32);
element!(u32, Ty::U32s, Ty::U32sMut, U32);
// NEITHER WGSL NOR SLANG HAS AN 8-BIT STORAGE TYPE, so the shader reads a byte
// array as packed 32-bit words. The DECLARED width is what makes a mismatch an
// error here rather than a stride bug on the device.
element!(u8, Ty::U8s, Ty::U8sMut, U8);

/// A DEVICE ARRAY OF `E`, as a shader plane binds one.
///
/// One constructor over the same element set the scalars use, and the reason
/// it had to exist is [`crate::routine::Const`]: once the mark stopped
/// implying buffer-ness — `Const<Tensor<bf16>>` is a weight and `Const<i32>` a
/// scalar the statement carries — the CARRIER had to say which it was.
/// `Tensor<E>` is that saying.
///
/// It holds the driver's handle and the rectangle the statement gave it. The
/// rectangle is why [`Backend::region`] is implemented on these planes at all:
/// with `Width`, `InWidth` and `OutWidth0` off the parameter list, `x.width`
/// is where a body reads its own operand's pitch, and 606 uses collapse into
/// the marks.
#[derive(Debug)]
pub struct Tensor<E: Element> {
    /// The driver's handle for the allocation.
    pub handle: u32,
    /// Which element it holds — a marker, never a value.
    held: core::marker::PhantomData<E>,
}

// `Clone`/`Copy`/`PartialEq` BY HAND, BECAUSE `derive` PUTS THE BOUND ON THE
// PARAMETER. A derived `Copy` here would ask for `E: Copy` -- the ELEMENT --
// when an element is a marker nothing copies and the only field is a `u32`.
impl<E: Element> Clone for Tensor<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Element> Copy for Tensor<E> {}

impl<E: Element> PartialEq for Tensor<E> {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}
impl<E: Element> Eq for Tensor<E> {}

impl<E: Element> Tensor<E> {
    /// This handle, as a tensor of `E`.
    #[must_use]
    pub const fn new(handle: u32) -> Self {
        Self {
            handle,
            held: core::marker::PhantomData,
        }
    }
}

impl<B: Lang, E: Element> Arg<B> for Tensor<E>
where
    B::Value: ShaderValue,
{
    const TY: Ty = E::TY_CONST;
    const SPELLING: &'static str = spell_read::<B>(E::SPELL);

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        value.as_buffer().map(Self::new).ok_or(Refusal::Kind {
            at,
            want: E::TY_CONST,
        })
    }
}

impl<V: ShaderValue, E: Element> Bind<V> for Tensor<E> {
    fn arg(self) -> V {
        V::buffer(self.handle)
    }
}

// AND THE SAME HANDLE, BOUND AS A WRITE. One type serves both directions here
// -- `Elem::Read` and `Elem::Write` are both `Tensor<E>` -- so the direction
// can only come from WHICH TRAIT the mark reaches for. `Out` and `InOut` reach
// for this one; `In` and `Const` reach for `Bind` above.
impl<V: ShaderValue, E: Element> crate::routine::BindMut<V> for Tensor<E> {
    fn arg_mut(self) -> V {
        V::buffer_mut(self.handle)
    }
}

// A HANDLE IS AN ELEMENT WITH ONE CARRIER, and that is the whole of what
// distinguishes this plane from CUDA. A pointee has two forms -- `*const E` to
// read and `*mut E` to write -- and a binding index has one:
// `ArgValue::Buffer(handle)` either way. So `Read` and `Write` are both `Self`,
// and the `Ty` still splits, because the TABLE must say which way the launch
// drives the operand even where the value cannot.
impl<E: Element> crate::routine::Elem for Tensor<E> {
    type Read = Self;
    type Write = Self;

    // A HANDLE DOES NOT MOVE. The driver binds a whole buffer and the shader
    // indexes into it, so a windowed view reaches the device as a scalar
    // rather than as an offset carrier -- which is what these planes already
    // do. Answering with the handle unchanged is the honest reading of
    // "advance a thing that has no inside".
    unsafe fn advance_read(read: Self::Read, _elems: usize) -> Self::Read {
        read
    }

    unsafe fn advance_write(write: Self::Write, _elems: usize) -> Self::Write {
        write
    }

    // EMPTY: a C++ spelling is CUDA's, and these planes state theirs through
    // `Arg::SPELLING`, which is the one that can see the `Lang`.
    const CPP_CONST: &'static str = "";
    const CPP_MUT: &'static str = "";
    const TY_CONST: Ty = E::TY_CONST;
    const TY_MUT: Ty = E::TY_MUT;
}

// A TENSOR IS THE WEIGHT RUN'S CARRIER. `Const<Tensor<bf16>>` claims the next
// weight and inherits the named-bank chain `Weight` had; a scalar `Const`
// claims the next slot of the params run. One mark, two runs, decided here.
impl<E: Element> crate::routine::ConstRun for Tensor<E> {
    const RUN: crate::routine::Claim = crate::routine::Claim::Weight;
    const TY: Ty = E::TY_CONST;
    type Held = Self;
}

/// Its own type rather than `u64` so the width is stated where the argument
/// is. Neither WGSL nor Slang has a 64-bit integer in these kernels, so the
/// shader reads it as two 32-bit words, low first — and a value that arrived
/// as eight bytes and was bound as four is the failure this distinguishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usize(
    /// The value.
    pub u64,
);

// A RAISED VIEW'S CARRIER. `In<Struct<T>>` holds `*const T::Value`, and the
// blanket `In` impl requires the carrier itself to be an `Arg` -- CUDA
// bridges raw pointers through its ABI, and this is the shader planes'
// bridge: the address rides [`ShaderValue::as_raised`], and the spelling is
// empty because a raised object never reaches shader text (the BODY reads
// the view on the host and binds its fields).
impl<B: Lang, V: 'static> Arg<B> for *const V
where
    B::Value: ShaderValue,
{
    const TY: Ty = Ty::Raised;
    const SPELLING: &'static str = "";

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        value
            .as_raised()
            .map(|addr| addr as *const V)
            .ok_or(Refusal::Kind {
                at,
                want: Ty::Raised,
            })
    }
}

impl<B: Lang> Arg<B> for Usize
where
    B::Value: ShaderValue,
{
    const TY: Ty = Ty::Usize;
    const SPELLING: &'static str = B::USIZE;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        value.as_usize().map(Self).ok_or(Refusal::Kind {
            at,
            want: Ty::Usize,
        })
    }
}

impl<V: ShaderValue> Bind<V> for Usize {
    fn arg(self) -> V {
        V::usize(self.0)
    }
}

/// A `u32` that rides a FIELD of a struct some earlier buffer binds, rather
/// than the scalar block.
///
/// The width is a `u32`'s; the difference is WHERE the value goes, which the
/// driver's binding decides and a routine does not. One row uses it, in all
/// three backends — `refactor-bigplan.md` §10 leaves open whether it should
/// stay a type at all once the ports are done.
/// A `u32` that rides a FIELD of a struct some earlier buffer binds, rather
/// than the scalar block.
///
/// The width is a `u32`'s; the difference is WHERE the value goes, which the
/// driver's binding decides and a routine does not. One row uses it, in all
/// three backends — `refactor-bigplan.md` §10 leaves open whether it should
/// stay a type at all once the ports are done.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InPacked(
    /// The value.
    pub u32,
);

impl<B: Lang> Arg<B> for InPacked
where
    B::Value: ShaderValue,
{
    const TY: Ty = Ty::InPacked;
    const SPELLING: &'static str = B::IN_PACKED;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        value.as_u32().map(Self).ok_or(Refusal::Kind {
            at,
            want: Ty::InPacked,
        })
    }
}

impl<V: ShaderValue> Bind<V> for InPacked {
    fn arg(self) -> V {
        V::u32(self.0)
    }
}

/// `LaunchRule::Elementwise`: one lane per element of the whole rectangle.
///
/// The lane arithmetic is shared and the bodies are not: the backends have a
/// right to differ about tiles and workgroup sizes, while the GRID arithmetic
/// already agreed character for character.
///
/// # Errors
///
/// [`Refusal::Empty`] when either extent is zero or negative, and
/// [`Refusal::Grid`] when the product does not fit a `u32`.
///
/// Refusals rather than clamps because both fail SILENTLY otherwise: a
/// dispatch of zero runs nothing and reports success, and a product that
/// WRAPPED covers a fraction of the rectangle and also reports success. Hence
/// the product in `u64`. A zero extent arrives honestly -- a routed expert
/// that won no tokens has zero rows -- so it is a value, not a panic.
pub fn elementwise(width: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    let [w, r] = rectangle(width, rows)?;
    let n = u64::from(w) * u64::from(r);
    let n = u32::try_from(n).map_err(|_| Refusal::Grid {
        what: "width * rows",
        at: i64::try_from(n).unwrap_or(i64::MAX),
    })?;
    Ok([n, 1, 1])
}

/// `LaunchRule::ElementwiseRows`: the rows on their own axis.
///
/// # Errors
///
/// [`Refusal::Empty`], as [`elementwise`]. No `Grid` refusal is possible:
/// neither extent is multiplied, and each already fits a `u32`.
pub fn elementwise_rows(width: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    let [w, r] = rectangle(width, rows)?;
    Ok([w, r, 1])
}

/// The extents both rules share, checked once, and named separately.
///
/// Which extent was empty, rather than that their product was: a caller
/// reading `nothing to launch: width * rows is zero` has to work out which of
/// the two it was.
fn rectangle(width: i32, rows: i32) -> Result<[u32; 2], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([width.unsigned_abs(), rows.unsigned_abs()])
}

/// Declare one scalar operand over an existing Rust type.
macro_rules! scalar_arg {
    ($rust:ty, $ty:expr, $read:ident, $make:ident, $spelling:ident) => {
        impl<B: Lang> Arg<B> for $rust
        where
            B::Value: ShaderValue,
        {
            const TY: Ty = $ty;
            const SPELLING: &'static str = B::$spelling;

            fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
                value.$read().ok_or(Refusal::Kind { at, want: $ty })
            }
        }

        impl<V: ShaderValue> Bind<V> for $rust {
            fn arg(self) -> V {
                V::$make(self)
            }
        }
    };
}

scalar_arg!(i32, Ty::I32, as_i32, i32, I32);
scalar_arg!(u32, Ty::U32, as_u32, u32, U32);
scalar_arg!(f32, Ty::F32, as_f32, f32, F32);

#[cfg(test)]
mod tests {
    use super::*;

    /// The vocabulary is the size this module says it is.
    ///
    /// The count is prose otherwise, and this module's prose said "ten types,
    /// no eleventh" for as long as it took another backend to cross a family
    /// and need two more. A census that reads the declarations cannot say that.
    #[test]
    fn the_vocabulary_is_the_size_this_module_says_it_is() {
        let src = include_str!("shader.rs");
        let declared = src.matches("\nelement!(").count()
            + src.matches("\nscalar_arg!(").count()
            // The ones written out by hand, because their unpacking is not
            // the shape either macro stamps. Matched on the `Arg<B> for` and
            // not on the bound: the bound was `Backend` when this was written
            // and is `Lang` now, and a census that names a trait it does not
            // own counts zero of them without saying so.
            + src
                .lines()
                .filter(|l| l.starts_with("impl") && l.contains("Arg<B> for "))
                .count();
        assert_eq!(
            declared, COUNT,
            "this module declares {declared} operand types and `COUNT` says \
             {COUNT}. Update it, and the sentence in the module docs that \
             quotes it -- a vocabulary that grew without either moving is how \
             the last count went stale."
        );
    }

    /// A rectangle whose product does not fit a `u32` is refused, not wrapped.
    ///
    /// This is the check the shared version exists for. `kernels-wgpu`'s
    /// first copy of this arithmetic multiplied two `u32`s -- which panics in
    /// debug and WRAPS in release, and a wrapped product is a grid covering a
    /// fraction of the rectangle that dispatches cleanly and reports success.
    /// That is `driver-metal`'s quarter-prefill defect arrived at by a
    /// different road, and nothing downstream reports either.
    #[test]
    fn a_rectangle_too_large_for_a_u32_is_refused_rather_than_wrapped() {
        // 65536 x 65536 is exactly 2^32: the smallest product that does not
        // fit, and the one that wraps to ZERO -- a grid that runs nothing.
        assert_eq!(
            elementwise(65536, 65536),
            Err(Refusal::Grid {
                what: "width * rows",
                at: 1 << 32
            })
        );
        // And the axis form cannot reach it, because it multiplies nothing.
        assert_eq!(elementwise_rows(65536, 65536), Ok([65536, 65536, 1]));
    }

    /// An empty rectangle names WHICH extent was empty.
    ///
    /// A caller told only that `width * rows` is zero has to work out which of
    /// the two it was, and one of them is ordinary -- a routed expert that won
    /// no tokens has zero rows and a caller wants `Ok`-shaped silence for it.
    #[test]
    fn an_empty_rectangle_names_the_extent_that_was_empty() {
        assert_eq!(elementwise(0, 8), Err(Refusal::Empty { what: "width" }));
        assert_eq!(elementwise(8, 0), Err(Refusal::Empty { what: "rows" }));
        assert_eq!(
            elementwise_rows(-1, 8),
            Err(Refusal::Empty { what: "width" })
        );
        assert_eq!(elementwise(64, 7), Ok([448, 1, 1]));
        assert_eq!(elementwise_rows(64, 7), Ok([64, 7, 1]));
    }

    /// A backend value, as small as [`ShaderValue`] allows.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum V {
        Buffer(u32),
        I32(i32),
        U32(u32),
        F32(f32),
        Usize(u64),
    }

    impl ShaderValue for V {
        fn as_buffer(self) -> Option<u32> {
            match self {
                Self::Buffer(h) => Some(h),
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

    #[derive(Clone, Copy, Debug)]
    struct Shader;

    impl Backend for Shader {
        type Value = V;
        type Ctx<'a> = ();
    }

    /// This harness binds spellings, and a spelling is never absent.
    impl crate::routine::Absent for V {}

    /// Spellings that are recognisably no real language's, so a test that
    /// asserted one by accident reads as obviously wrong.
    impl Lang for Shader {
        const BUF: &'static str = "buf";
        const BUF_MUT: &'static str = "buf_mut";
        const I32S: &'static str = "i32s";
        const U32S: &'static str = "u32s";
        const U8S: &'static str = "u8s";
        const F32S: &'static str = "f32s";
        const F32S_MUT: &'static str = "f32s_mut";
        const BF16S: &'static str = "bf16s";
        const BF16S_MUT: &'static str = "bf16s_mut";
        const F16S: &'static str = "f16s";
        const F16S_MUT: &'static str = "f16s_mut";
        const I32: &'static str = "i32";
        const U32: &'static str = "u32";
        const F32: &'static str = "f32";
        const USIZE: &'static str = "usize";
        const IN_PACKED: &'static str = "in_packed";
    }

    /// Every operand refuses a value of another kind, by position.
    ///
    /// The check that matters is the WIDTH one, and it is why the scalar kinds
    /// are separate at all: a `Usize` is eight bytes and an `I32` is four, so
    /// a value that crossed between them either truncates or writes over its
    /// neighbour depending on where in the block it lands. Nothing downstream
    /// would report either.
    #[test]
    fn an_operand_refuses_a_value_of_another_kind() {
        assert_eq!(
            <i32 as Arg<Shader>>::unpack(&V::Usize(1), 3),
            Err(Refusal::Kind {
                at: 3,
                want: Ty::I32
            })
        );
        assert_eq!(
            <Usize as Arg<Shader>>::unpack(&V::I32(1), 0),
            Err(Refusal::Kind {
                at: 0,
                want: Ty::Usize
            })
        );
        assert_eq!(
            <Tensor<bf16> as Arg<Shader>>::unpack(&V::F32(1.0), 7),
            Err(Refusal::Kind {
                at: 7,
                want: Ty::Bf16s
            })
        );
    }

    /// A handle carries no element type; the TYPE does.
    ///
    /// Which is what makes a body handed the wrong buffer kind fail to
    /// compile rather than at runtime — the handle would have been accepted.
    #[test]
    fn a_buffer_handle_is_accepted_by_every_buffer_kind() {
        assert_eq!(
            <Tensor<bf16> as Arg<Shader>>::unpack(&V::Buffer(9), 0),
            Ok(Tensor::<bf16>::new(9))
        );
        assert_eq!(
            <Tensor<u32> as Arg<Shader>>::unpack(&V::Buffer(9), 0),
            Ok(Tensor::<u32>::new(9))
        );
        assert_eq!(
            <Tensor<u8> as Arg<Shader>>::unpack(&V::Buffer(9), 0),
            Ok(Tensor::<u8>::new(9))
        );
    }

    /// Binding and unpacking are inverse, for every kind.
    #[test]
    fn what_a_body_binds_is_what_a_signature_recovers() {
        assert_eq!(Bind::<V>::arg(Tensor::<bf16>::new(4)), V::Buffer(4));
        // `BufMut(4)` STOOD HERE and was the same assertion twice over: the
        // two types merged into one when the direction moved to the mark, so
        // the rename left a second `Buf`. The element carriers are what a
        // second line here can now say something new about.
        assert_eq!(Bind::<V>::arg(Tensor::<bf16>::new(4)), V::Buffer(4));
        assert_eq!(Bind::<V>::arg(Tensor::<f16>::new(4)), V::Buffer(4));
        assert_eq!(Bind::<V>::arg(-3i32), V::I32(-3));
        assert_eq!(Bind::<V>::arg(3u32), V::U32(3));
        assert_eq!(Bind::<V>::arg(0.5f32), V::F32(0.5));
        assert_eq!(Bind::<V>::arg(Usize(1 << 40)), V::Usize(1 << 40));
        // `InPacked` binds as a `u32`: the width is a `u32`'s and only the
        // PLACE differs, which the driver decides.
        assert_eq!(Bind::<V>::arg(InPacked(7)), V::U32(7));
    }

    /// `InPacked` takes a `u32`'s value and is not the `u32` operand.
    #[test]
    fn a_packed_field_is_a_u32_that_is_not_the_u32_operand() {
        assert_eq!(
            <InPacked as Arg<Shader>>::unpack(&V::U32(5), 0),
            Ok(InPacked(5))
        );
        assert_ne!(
            <InPacked as Arg<Shader>>::TY,
            <u32 as Arg<Shader>>::TY,
            "the two are the same width and different operands"
        );
    }

    /// THE ELEMENT DECIDES THE ARGUMENT TYPE, and nothing else does.
    ///
    /// This used to assert what `Env` claimed about a SUPPLIER — that a keyed
    /// wrapper carried `Provenance::Env` while its carrier carried `Trace`.
    /// Both the wrapper and the provenance column are deleted: every parameter
    /// is the statement's now, and a fact only the fire can answer is asked
    /// for in the body rather than declared beside the operands.
    ///
    /// What is left to check is the fact that replaced it — one constructor
    /// over the scalar element set, with the element choosing the `Ty` and the
    /// mark choosing the direction.
    #[test]
    fn a_tensor_takes_its_argument_type_from_its_element() {
        assert_eq!(<Tensor<i32> as Arg<Shader>>::TY, Ty::I32s);
        assert_eq!(<Tensor<u8> as Arg<Shader>>::TY, Ty::U8s);
        assert_eq!(<Tensor<f32> as Arg<Shader>>::TY, Ty::F32s);
        // The DIRECTION is the mark's, and the element carries both readings
        // for it to pick from.
        assert_eq!(<Tensor<i32> as crate::Elem>::TY_MUT, Ty::I32sMut);
        assert_eq!(Bind::<V>::arg(Tensor::<bf16>::new(2)), V::Buffer(2));
    }
}
