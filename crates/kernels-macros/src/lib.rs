//! The attributes the kernel crates are declared with.
//!
//! `#[points]` reads a family trait and states its point table; `#[claims]`
//! reads a plane's impl and states which of those points it answers
//! (.wiki/baker.md).
//!
//! # `#[routine]` STOOD HERE, AND THE FOUR PLANES FOLDED IT AWAY
//!
//! It was the per-launcher row those two replace, family by family: an
//! attribute on a `fn` that derived a thirteen-column `Routine` — the
//! trace name, the namespace off `module_path!()`, the argument types and
//! their `Source`s off `KernelFn`, the names-and-nullability column off
//! the syntax, the `canon` role, the `out(..)` shape rules — and
//! registered it into a `linkme` distributed slice each plane collected.
//!
//! Every column's reader died before the attribute did. The trace-time
//! shape walk that read `out(..)` went at R4e with `Signature` and the
//! `KernelSig` projection; the three shader drivers that read `derived`
//! left the workspace at R3; the by-name registry's last caller was a
//! model-loader name-drift test, which asks the compiler now. What was
//! left was `canon` — four `(claim, symbol)` pairs across cuda and metal,
//! for points no `#[claims]` block answers yet — and four pairs is a
//! `const` in each plane crate (`kernels_cuda::CANON`,
//! `kernels_metal::CANON`), not a proc macro, a registry and a linker
//! section.
//!

mod claims;
mod points;

use proc_macro::TokenStream;
use syn::{Error, ItemImpl, ItemTrait, parse_macro_input};

/// Declare a family: the trait as written, and the point table it states.
///
/// ```ignore
/// #[points]
/// pub trait Norm: Plane {
///     fn rmsnorm<T: Scalar>(&self, x: In<Self::Tensor<T>>, eps: f32,
///                           y: Out<Self::Tensor<T>>) -> Result<(), Refusal> { .. }
/// }
/// // pub const NORM_POINTS: &[Point] = &[Point { name: "norm.rmsnorm", .. }];
/// ```
///
/// The trait is re-emitted unchanged — the default bodies are the family's,
/// hand-written, and this reads them without touching them. The table names
/// `Point`, `Slot`, `Mark`, `Dtype` and `Prim` unqualified: it lands in the
/// family's own module, and that is the vocabulary a family is declared in.
#[proc_macro_attribute]
pub fn points(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return no_arguments("points", attr);
    }
    let item = parse_macro_input!(item as ItemTrait);
    points::expand(item)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// Declare what a plane answers: the impl as written, and the point names it
/// overrides. What it leaves to the family's default body is the backlog.
#[proc_macro_attribute]
pub fn claims(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return no_arguments("claims", attr);
    }
    let item = parse_macro_input!(item as ItemImpl);
    claims::expand(item)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn no_arguments(attr_name: &str, attr: TokenStream) -> TokenStream {
    Error::new(
        proc_macro2::TokenStream::from(attr)
            .into_iter()
            .next()
            .map_or_else(proc_macro2::Span::call_site, |t| t.span()),
        format!("`#[{attr_name}]` reads the item below it and takes no arguments"),
    )
    .into_compile_error()
    .into()
}
