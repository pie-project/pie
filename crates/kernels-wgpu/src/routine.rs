//! This backend's instantiation of the `kernels` routine machinery.
//!
//! A routine is an ordinary `fn` that computes a launch and dispatches it.
//! Everything a row used to STATE -- the launch rule, the grid parameter, the
//! head parameter, the operand list -- is code in the body, and the table row
//! is derived from the signature by [`macro@crate::routine`].
//!
//! This crate depends on `kernels` and nothing else -- no `wgpu`, no adapter,
//! no device -- so the table and the shaders build on any machine that can
//! build Rust. The thing a body dispatches through is therefore a TRAIT the
//! driver implements ([`Encode`]), and `Backend::Ctx` is `dyn Encode`, which
//! is why `Backend::Ctx` is `?Sized`: the machinery only ever names the
//! context behind a reference, and a `Sized` bound would force a `wgpu`
//! dependency here.
//!
//! A body decides the entrypoint STRING and the workgroup counts, which makes
//! a kernel whose grid follows no rule expressible without inventing one.

use kernels::routine::{Backend, Extent, Refusal};
use kernels::shader::ShaderValue;

/// This backend, as the machinery names it.
///
/// A marker: never constructed, carrying only the two concrete types the
/// `kernels` machinery is generic over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wgpu;

impl Backend for Wgpu {
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
/// widths differ and the check that matters is exactly the width one: a
/// [`kernels::Ty::Usize`] value handed to a [`kernels::Ty::I32`] argument is eight bytes going
/// into a four-byte slot, which either truncates or writes over its neighbour
/// depending on where in the block it lands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    /// A device allocation, named by whatever index the caller keys its own
    /// buffers on. Opaque on purpose: this crate cannot name a `wgpu::Buffer`
    /// and does not need to. The driver binds handles to its own buffers
    /// before it calls, and a routine only passes them through.
    Buffer(u32),
    /// A 32-bit signed scalar.
    I32(i32),
    /// A 32-bit unsigned scalar.
    U32(u32),
    /// A 32-bit float.
    F32(f32),
    /// A 64-bit stride or extent, which is what [`kernels::Ty::Usize`] means here.
    ///
    /// WGSL has no 64-bit integer of any kind, so the shader reads this as a
    /// `vec2<u32>` — two words, low first. The width is why it is its own
    /// variant.
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
/// Implemented by `driver-wgpu`. The split is the crate boundary: a body knows
/// the entrypoint and the grid, and the driver knows what a buffer is, which
/// bind group a scalar rides, and how to submit.
pub trait Encode {
    /// Run one dispatch.
    ///
    /// `args` is the routine's own argument list, in signature order. The
    /// implementor separates buffers from scalars by variant — which is what
    /// makes the split derivable rather than stated twice.
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
}

/// This backend's value, as the shared operand types read it.
///
/// The ten operand types live in [`kernels::shader`] because the vocabulary is
/// closed and identical in metal, vulkan and wgpu -- see
/// `.wiki/kernel-x/refactor-bigplan.md` §7. What is NOT shared is the value,
/// which is this enum, and this impl is the whole of what a shared type needs
/// to know about it.
impl ShaderValue for ArgValue {
    fn as_buffer(self) -> Option<u32> {
        match self {
            Self::Shaped { handle, .. } => Some(handle),
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

/// How WGSL spells the twelve.
///
/// A storage binding's access mode for the opaque buffers, the element type
/// for the arrays, and `vec2<u32>` for a 64-bit extent, which WGSL has no
/// scalar for and these kernels read as two words, low first.
impl kernels::shader::Lang for Wgpu {
    const BUF: &'static str = "read";
    const BUF_MUT: &'static str = "read_write";
    const I32S: &'static str = "array<i32>";
    const U32S: &'static str = "array<u32>";
    const U8S: &'static str = "array<u8>";
    const F32S: &'static str = "array<f32>";
    const F32S_MUT: &'static str = "array<f32>";
    // `array<u32>`, AND IT IS NOT A PLACEHOLDER. WGSL has no 16-bit scalar at
    // all, so an activation buffer really is declared as words and the shader
    // unpacks halves -- `rope/neox.wgsl`'s `x: array<u32>` is the whole of it.
    // The access mode rides the `var<storage, ..>` and not the type, which is
    // why these two agree, exactly as `F32S`/`F32S_MUT` do.
    const BF16S: &'static str = "array<u32>";
    const BF16S_MUT: &'static str = "array<u32>";
    const F16S: &'static str = "array<u32>";
    const F16S_MUT: &'static str = "array<u32>";
    const I32: &'static str = "i32";
    const U32: &'static str = "u32";
    const F32: &'static str = "f32";
    const USIZE: &'static str = "vec2<u32>";
    const IN_PACKED: &'static str = "u32";
}

/// The operand vocabulary, from the crate that holds it once. Re-exported
/// rather than named through `kernels::shader` at every use, so a body's
/// signature reads as this backend's own and a family file imports one module.
pub use kernels::shader::{Bind, InPacked, Tensor, Usize, bf16, f16};

/// The two launch rules a body states its rectangle with, on this plane's own
/// path — the same names `kernels-metal` and `kernels-vulkan` re-export, so a
/// body reads the same on all three.
pub use kernels::shader::{elementwise, elementwise_rows};

/// What a routine body dispatches through, spelled as a body writes it.
///
/// `dyn Encode + 'a`, and the lifetime is why `Backend::Ctx` is a generic
/// associated type: a wgpu `Encode` BORROWS the device, the pipeline cache and
/// the fire's buffers from the caller's frame, so an implementor is never
/// `'static` and a plain `dyn Encode` could not name it.
pub type Ctx<'a> = dyn Encode + 'a;

// THE ASKING SIDE, over the one method the driver implements.
//
// `Asks` is blanket-implemented for every `Answers`, so this is the whole of
// what a plane states to give its bodies `ctx.ask::<C, keys::X>()`,
// `ctx.params()` and `ctx.absent()`.
impl kernels::routine::Answers<Wgpu> for Ctx<'_> {
    fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
        Encode::resolve(self, ty, source)
    }
}

/// One routine, in this backend's instantiation of the machinery.
pub type Routine = kernels::routine::Routine<Wgpu>;


/// The fact keys a BODY asks the runtime with — `ctx.ask::<i32, keys::Rows>()`.
///
/// Not what a signature binds its scalars from any more: a scalar the
/// checkpoint fixes is a `Const` the statement carries, and a key names only
/// what a fire decides.
pub use kernels::keys;
pub use kernels::routine::{Const, Fire, In, InOut, Out};

/// What a body asks the runtime for, once `Env` is out of the parameter list.
pub use kernels::routine::{Answers, Asks};


// THE PLANE'S `routine!` WRAPPER STOOD HERE AND HAS NO CALLERS.
//
// It filled this backend in so a membership list could name only the
// `fn`. There is no membership list: `#[routine]` builds the row beside
// the `fn` and a distributed slice collects it, so the only caller of
// `kernels::routine!` is the attribute, which names the backend through
// `crate::Plane`.
