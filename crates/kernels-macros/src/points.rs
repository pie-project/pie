//! `#[points]` — a family's point table, read off the trait that declares it.
//!
//! The macro only READS: the trait is re-emitted byte for byte, default
//! bodies and all, and the table lands beside it.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Error, FnArg, GenericParam, Ident, ItemTrait, Pat, PathSegment, ReceiverKind, ReturnType,
    Signature, TraitItem, TraitItemFn, Type, TypeParamBound, WherePredicate, spanned::Spanned,
};

pub fn expand(item: ItemTrait) -> Result<TokenStream, Error> {
    let family = snake(&item.ident);
    let table = Ident::new(
        &format!("{}_POINTS", family.to_uppercase()),
        item.ident.span(),
    );
    let points = item
        .items
        .iter()
        .filter_map(|i| match i {
            TraitItem::Fn(f) => Some(f),
            _ => None,
        })
        .map(|f| point(&family, f))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(quote! {
        #item

        pub const #table: &[Point] = &[#(#points),*];
    })
}

/// The trait's CamelCase as a point name's first half: `MoeRouter` →
/// `moe_router`, `MLP` → `mlp`.
pub(crate) fn snake(ident: &Ident) -> String {
    let spelled: Vec<char> = ident.to_string().chars().collect();
    let mut out = String::with_capacity(spelled.len() + 4);
    for (i, c) in spelled.iter().enumerate() {
        let breaks = i > 0
            && c.is_uppercase()
            && (!spelled[i - 1].is_uppercase()
                || spelled.get(i + 1).is_some_and(char::is_ascii_lowercase));
        if breaks {
            out.push('_');
        }
        out.extend(c.to_lowercase());
    }
    out
}

fn point(family: &str, f: &TraitItemFn) -> Result<TokenStream, Error> {
    let name = format!("{family}.{}", f.sig.ident);
    let axes = axes(&name, &f.sig)?;
    answers(&name, &f.sig)?;
    let mut args = f.sig.inputs.iter();
    match args.next() {
        Some(FnArg::Receiver(r)) if matches!(r.kind, ReceiverKind::Reference(_, _, None)) => {}
        _ => {
            return Err(Error::new(
                f.sig.span(),
                format!("`{name}` takes `&self` first: the plane the point fires on"),
            ));
        }
    }
    let slots = args
        .map(|a| slot(&name, a, &axes))
        .collect::<Result<Vec<_>, _>>()?;
    let n = axes.len();
    Ok(quote!(Point { name: #name, axes: #n, slots: &[#(#slots),*] }))
}

/// The method's scalar axes, in declaration order — the cartesian a dispatch
/// shim writes, and what a `Self::Tensor<T>` payload indexes into.
fn axes(point: &str, sig: &Signature) -> Result<Vec<Ident>, Error> {
    let mut out = Vec::new();
    for p in &sig.generics.params {
        let GenericParam::Type(t) = p else {
            return Err(Error::new(
                p.span(),
                format!(
                    "`{point}`: a point's generics are its scalar axes and nothing else \
                     — write `<T: Scalar>`"
                ),
            ));
        };
        let elsewhere = sig
            .generics
            .where_clause
            .iter()
            .flat_map(|w| &w.predicates)
            .filter_map(|p| match p {
                WherePredicate::Type(w) => match &w.bounded_ty {
                    Type::Path(path) if path.path.is_ident(&t.ident) => Some(&w.bounds),
                    _ => None,
                },
                _ => None,
            })
            .flatten();
        if !t.bounds.iter().chain(elsewhere).any(is_scalar) {
            return Err(Error::new(
                t.span(),
                format!(
                    "`{point}`: axis `{}` states no bound — a point quantifies over `Scalar`",
                    t.ident
                ),
            ));
        }
        out.push(t.ident.clone());
    }
    Ok(out)
}

fn is_scalar(b: &TypeParamBound) -> bool {
    matches!(b, TypeParamBound::Trait(t) if t.path.segments.last().is_some_and(|s| s.ident == "Scalar"))
}

/// Every point answers `Result<(), Refusal>`: an unclaimed one is a backlog
/// row, so there is no other answer to give.
fn answers(point: &str, sig: &Signature) -> Result<(), Error> {
    let stated = match &sig.output {
        ReturnType::Type(_, ty) => match &**ty {
            Type::Path(p) => p.path.segments.last().is_some_and(|s| {
                s.ident == "Result"
                    && match args(s).as_slice() {
                        [Type::Tuple(unit), Type::Path(r)] => {
                            unit.elems.is_empty()
                                && r.path.segments.last().is_some_and(|s| s.ident == "Refusal")
                        }
                        _ => false,
                    }
            }),
            _ => false,
        },
        ReturnType::Default => false,
    };
    if stated {
        return Ok(());
    }
    Err(Error::new(
        sig.output.span(),
        format!("`{point}` answers `Result<(), Refusal>`, and a point answers nothing else"),
    ))
}

fn slot(point: &str, arg: &FnArg, axes: &[Ident]) -> Result<TokenStream, Error> {
    let FnArg::Typed(pt) = arg else {
        return Err(Error::new(
            arg.span(),
            format!("`{point}`: `&self` is the first parameter and occurs once"),
        ));
    };
    let Pat::Ident(id) = &*pt.pat else {
        return Err(Error::new(
            pt.pat.span(),
            format!("`{point}`: a slot is named, and this one is a pattern"),
        ));
    };
    let name = id.ident.to_string();
    let (mark, dtype) = marked(point, &name, &pt.ty, axes)?;
    Ok(quote!(Slot { name: #name, mark: Mark::#mark, dtype: #dtype }))
}

/// The slot's mark and its payload dtype. A host scalar wears no mark, which
/// is what makes it one.
fn marked(
    point: &str,
    slot: &str,
    ty: &Type,
    axes: &[Ident],
) -> Result<(Ident, TokenStream), Error> {
    if let Some(p) = prim(ty) {
        return Ok((
            Ident::new("Scalar", ty.span()),
            quote!(Dtype::Fixed(Prim::#p)),
        ));
    }
    let marked = match ty {
        Type::Path(p) => p.path.segments.last().filter(|s| {
            matches!(
                s.ident.to_string().as_str(),
                "In" | "InOut" | "Out" | "Const" | "Cache"
            )
        }),
        _ => None,
    };
    let Some(seg) = marked else {
        return Err(Error::new(
            ty.span(),
            format!(
                "`{point}`: slot `{slot}` is neither marked nor a host scalar — an operand is \
                 `In`, `InOut`, `Out` or `Const` of `Self::Tensor<..>`, a pool row is `Cache` \
                 of the plane's view, a scalar is a bare `f32`, `i32`, `u32` or `bool`"
            ),
        ));
    };
    let carried = args(seg);
    let [payload] = carried.as_slice() else {
        return Err(Error::new(
            seg.span(),
            format!(
                "`{point}`: mark `{}` on slot `{slot}` carries one payload",
                seg.ident
            ),
        ));
    };
    // `Cache` carries a POOL'S VIEW, and a view is not a rectangle of
    // elements: the two payload readers are different for that reason, and
    // a cache slot's dtype column is `Opaque` rather than an axis.
    let dtype = if seg.ident == "Cache" {
        pool(point, slot, payload)?
    } else {
        tensor(point, slot, payload, axes)?
    };
    Ok((seg.ident.clone(), dtype))
}

/// `Self::Recurrent` or `Self::Pages` — the plane's view of a pool row, and
/// the two payloads a `Cache` mark carries. ONE ASSOCIATED TYPE PER POOL,
/// not one per family: the recurrent slabs the mixers keep and the paged KV
/// the latent, index and pooled families address are two pools, and every
/// family reading either names the pool's own view here.
fn pool(point: &str, slot: &str, ty: &Type) -> Result<TokenStream, Error> {
    let named = match ty {
        Type::Path(p)
            if p.qself.is_none()
                && p.path.segments.len() == 2
                && p.path.segments[0].ident == "Self" =>
        {
            p.path.segments.last().filter(|s| {
                (s.ident == "Recurrent" || s.ident == "Pages") && s.arguments.is_none()
            })
        }
        _ => None,
    };
    if named.is_some() {
        return Ok(quote!(Dtype::Opaque));
    }
    Err(Error::new(
        ty.span(),
        format!(
            "`{point}`: `Cache` on slot `{slot}` carries the pool's own view, and the \
             floor declares two — `Self::Recurrent` and `Self::Pages`"
        ),
    ))
}

/// `Self::Tensor<T>` → `Dtype::Generic(i)` for the method's i-th axis;
/// `Self::Tensor<f32>` → `Dtype::Fixed(Prim::F32)`.
fn tensor(point: &str, slot: &str, ty: &Type, axes: &[Ident]) -> Result<TokenStream, Error> {
    let payload = match ty {
        Type::Path(p)
            if p.qself.is_none()
                && p.path.segments.len() == 2
                && p.path.segments[0].ident == "Self" =>
        {
            p.path.segments.last().filter(|s| s.ident == "Tensor")
        }
        _ => None,
    };
    let Some(seg) = payload else {
        return Err(Error::new(
            ty.span(),
            format!(
                "`{point}`: a mark on slot `{slot}` carries the plane's payload, \
                 `Self::Tensor<..>` — a host scalar is bare and wears no mark"
            ),
        ));
    };
    let carried = args(seg);
    let [dtype] = carried.as_slice() else {
        return Err(Error::new(
            seg.span(),
            format!("`{point}`: `Self::Tensor` on slot `{slot}` carries one dtype"),
        ));
    };
    if let Type::Path(p) = dtype
        && let Some(i) = p
            .path
            .get_ident()
            .and_then(|id| axes.iter().position(|a| a == id))
    {
        return Ok(quote!(Dtype::Generic(#i)));
    }
    if let Some(p) = tensor_prim(dtype) {
        return Ok(quote!(Dtype::Fixed(Prim::#p)));
    }
    Err(Error::new(
        dtype.span(),
        format!(
            "`{point}`: slot `{slot}` carries a dtype that is neither an axis of this method \
             nor fixed — write `Self::Tensor<T>` for `<T: Scalar>`, or `Self::Tensor<f32>`"
        ),
    ))
}

/// A host scalar's spelling, as the `Prim` variant that names it.
fn prim(ty: &Type) -> Option<Ident> {
    let Type::Path(p) = ty else { return None };
    let id = p.path.get_ident()?;
    let variant = match id.to_string().as_str() {
        "f32" => "F32",
        "i32" => "I32",
        "u32" => "U32",
        "bool" => "Bool",
        _ => return None,
    };
    Some(Ident::new(variant, id.span()))
}

/// A fixed TENSOR element's spelling. A rectangle carries elements a host
/// scalar run never does — `Self::Tensor<u8>` is the byte mask a selection
/// writes — so the two readers are different, and a bare `u8` parameter
/// stays what it always was: not a scalar this floor knows how to run.
fn tensor_prim(ty: &Type) -> Option<Ident> {
    if let Some(p) = prim(ty) {
        return Some(p);
    }
    let Type::Path(p) = ty else { return None };
    let id = p.path.get_ident()?;
    (id == "u8").then(|| Ident::new("U8", id.span()))
}

/// The type arguments of `Foo<A, B>`, dropping lifetimes and consts.
fn args(seg: &PathSegment) -> Vec<&Type> {
    match &seg.arguments {
        syn::PathArguments::AngleBracketed(a) => a
            .args
            .iter()
            .filter_map(|g| match g {
                syn::GenericArgument::Type(t) => Some(t),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}
