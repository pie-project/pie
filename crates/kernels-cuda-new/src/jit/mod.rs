//! The per-symbol JIT: a root, an instantiation, and one entry point.
//!
//! What changed from `runtime::cache` is the granularity. A UNIT carried a row
//! list because the whole set of instantiations had to be enumerated before
//! anything could be compiled, and one fire of one FA2 symbol compiled its
//! unit's ten rows. Compilation is per instantiation now: the enumeration has
//! no reader left, so it is not data any more.
//!
//! The pieces:
//!
//! * [`Root`] — device text and what a compile of it needs.
//! * [`Ctx`] — what a routine body launches through.
//! * [`ArgValue`] — one bound argument, feature-free, because a routine body
//!   is feature-free.
//! * [`Cuda`] — the marker carrying those last two to the `kernels`
//!   machinery.

mod arg;
#[cfg(feature = "_cuda")]
pub mod cache;
mod ctx;
#[cfg(feature = "_cuda")]
mod launch;
mod root;
#[cfg(feature = "_cuda")]
mod scratch;
pub mod value;

pub use ctx::{Ctx, Cuda, Launch};
pub use root::{Headers, Root, Toolchain};
pub use value::ArgValue;

/// One routine's row, from its `fn` and nothing else.
///
/// The backend's three-line wrapper over [`kernels::routine!`], with [`Cuda`]
/// filled in so a declaration names only the `fn`:
///
/// ```ignore
/// pub static ROUTINES: &[Routine] = &[
///     routine!(rope_bf16, in_place = &[(0, 0), (1, 1)]),
///     routine!(rope_write_kv_bf16, whole),
/// ];
/// ```
#[macro_export]
macro_rules! routine {
    ($body:ident $(, $($fact:tt)*)?) => {
        ::kernels::routine!($crate::jit::Cuda, $body $(, $($fact)*)?)
    };
}

/// One routine, in this backend's instantiation of the machinery.
pub type Routine = kernels::routine::Routine<Cuda>;

/// One family's routines, and the namespace its trace symbols sit in.
///
/// A `Routine`'s name is its `fn`'s name, which is what makes the table
/// underivable-from-anything-else; a trace names `rope::rope_bf16`. The
/// namespace is the difference, and it is stated ONCE per family rather than
/// spelled into 200 routines that would then be free to disagree with it.
pub struct Family {
    /// What a trace prefixes this family's symbols with.
    pub namespace: &'static str,
    /// The routines, in declaration order.
    pub routines: &'static [Routine],
}

impl Family {
    /// The routine a trace symbol names, if this family declares it.
    #[must_use]
    pub fn routine(&self, symbol: &str) -> Option<&'static Routine> {
        let tail = symbol.strip_prefix(self.namespace)?.strip_prefix("::")?;
        self.routines.iter().find(|r| r.name == tail)
    }

    /// One routine's trace symbol.
    #[must_use]
    pub fn symbol(&self, routine: &Routine) -> String {
        format!("{}::{}", self.namespace, routine.name)
    }
}
