//! `#[routine]` — a kernel's operand table, derived from its own signature.
//!
//! `routine!` receives a path, which cannot be introspected: by then the
//! parameter names live in a `fn` the macro never sees. An attribute macro
//! sits on that `fn` and sees the whole signature, which is why this second
//! macro crate exists.
//!
//! Facts are spelled as types (`Env<keys::RmsEps>`) rather than const generic
//! strings because `&'static str` is not a legal const generic parameter type
//! and `adt_const_params` is unstable.
//!
//! # What it derives
//!
//! | parameter | derives |
//! |---|---|
//! | `#[source(...)]` on it | that `Source`, with short spellings folded |
//! | `In<N, T>`, `InRow<N, T>`, `InSlot<N, T>` | `Slot(In, N)` |
//! | `Out<N, T>`, `OutRow<N, T>`, `OutSlot<N, T>` | `Slot(Out, N)` |
//! | `Bank<N, T>` | `Slot(Weight, N)` |
//! | `Weight<0, T>`, `Weight<1, T>` | `Named(keys::NamedWeight[2])` |
//! | `Aux<N, T>` | `Slot(Aux, N)` |
//! | `Param<N, T>`, `ParamF32<N>` | `Slot(Param[F32], N)`, consuming no operand index |
//! | `Unbound<T>` | `None`, consuming no index |
//! | `Env<keys::F>` | `<keys::F as Fact>::SOURCE` |
//! | bare `*const T` | `Slot(In, n)`, counting const pointers from 0 |
//! | bare `*mut T` | `Slot(Out, n)`, counting mut pointers from 0 |
//! | `Or<T>` | as the unwrapped `T`, and `nullable` |
//! | anything else | `None` |
//!
//! A stated index sets the counter to `N + 1` rather than consuming the next
//! number, so a bare pointer after a stated `In<1, _>` continues the run.
//!
//! This macro counts pointers and cannot see a `ROUTINES` row, so it gets
//! in-place launchers wrong: an aliased buffer occupies an input slot and an
//! output slot, making the `*const` the statement's second input. `bind`'s
//! `operands` corrects for it there, where `in_place` is stated.

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Error, FnArg, ItemFn, Pat, PatType, Type, parse_macro_input, spanned::Spanned,
};

/// Derive an operand table from the `fn`'s own signature.
///
/// Emits the function unchanged, helper attributes stripped, beside a marker
/// of the same name carrying [`kernels::Derivation`]. See the module docs for
/// the derivation table.
#[proc_macro_attribute]
pub fn routine(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut f = parse_macro_input!(item as ItemFn);
    let derived = match derive_all(&mut f) {
        Ok(d) => d,
        Err(e) => return e.to_compile_error().into(),
    };

    let name = &f.sig.ident;
    let vis = &f.vis;
    let doc = format!(
        "`{name}`'s operands, derived by `#[routine]` from its signature.\n\n\
         An uninhabited marker in the TYPE namespace wearing the function's \
         name -- a unit struct would take the VALUE namespace too and collide \
         with the `fn`. It exists so that \
         `routine!` can reach the column through `<{name} as \
         ::kernels::Derivation>::DERIVED` from the `fn` path alone."
    );

    quote! {
        #f

        #[doc = #doc]
        #[allow(non_camel_case_types)]
        #vis enum #name {}

        impl ::kernels::Derivation for #name {
            const DERIVED: &'static [::kernels::Derived] = &[#(#derived),*];
        }
    }
    .into()
}

/// Walk the parameters, stripping helper attributes as it goes.
fn derive_all(f: &mut ItemFn) -> Result<Vec<proc_macro2::TokenStream>, Error> {
    let mut out = Vec::new();
    let (mut ins, mut outs) = (0u8, 0u8);

    // The function's own type parameters, so `Env<T>` in a generic launcher
    // is not mistaken for an unqualified fact. Read from the signature rather
    // than allowlisted, since `T` is a convention and not a rule.
    let generics: Vec<String> = f
        .sig
        .generics
        .type_params()
        .map(|p| p.ident.to_string())
        .collect();

    for (i, arg) in f.sig.inputs.iter_mut().enumerate() {
        // The context is the backend's, not the statement's, and never in
        // the argument run: `KernelFn::invoke` takes it beside `args`.
        if i == 0 {
            continue;
        }
        let FnArg::Typed(pt) = arg else {
            return Err(Error::new(arg.span(), "a routine takes no `self`"));
        };
        let name = param_name(pt)?;
        let explicit = take_source_attr(pt)?;
        let (shape, nullable) = classify(&pt.ty);
        let literal = take_lit_attr(pt)?;
        if explicit.is_some() && literal.is_some() {
            return Err(Error::new(
                pt.span(),
                "`#[source(..)]` and `#[lit(..)]` both answer where a \
                 parameter comes from; a parameter has one origin",
            ));
        }
        // Whether the signature said where this comes from or the macro
        // counted its way to it: a binder correcting a guess must not also
        // correct a statement.
        let mut stated = false;

        // Precedence: attribute, then signature, then position. The attribute
        // is the escape hatch and wins; letting position beat a wrapper would
        // make the wrapper decorative.
        let source = if let Some(s) = explicit {
            stated = true;
            s
        } else if let Some(s) = literal {
            stated = true;
            s
        } else if let Some(s) = stated_source(&pt.ty, &mut ins, &mut outs, &generics) {
            stated = true;
            s
        } else {
            match shape {
                Shape::Env => quote!(::core::option::Option::None),
                Shape::ConstPtr => {
                    let at = ins;
                    ins += 1;
                    quote!(::core::option::Option::Some(::kernels::Source::Slot(::kernels::Kind::In, #at)))
                }
                Shape::MutPtr => {
                    let at = outs;
                    outs += 1;
                    quote!(::core::option::Option::Some(::kernels::Source::Slot(::kernels::Kind::Out, #at)))
                }
                Shape::Opaque => quote!(::core::option::Option::None),
            }
        };

        out.push(quote! {
            ::kernels::Derived {
                name: #name,
                nullable: #nullable,
                source: #source,
                stated: #stated,
            }
        });
    }
    Ok(out)
}

/// What the macro can tell about a parameter's type.
enum Shape {
    /// `Env<_>`; its name says which fact.
    Env,
    /// `*const T`, read through.
    ConstPtr,
    /// `*mut T`, written through.
    MutPtr,
    /// A scalar, a handle, a plan — nothing a statement places.
    Opaque,
}

/// Peel `Or<_>` (which only adds nullability) and read what is underneath.
fn classify(ty: &Type) -> (Shape, bool) {
    match ty {
        Type::Ptr(p) => (
            // syn 3 made this an enum rather than an `Option<Token![mut]>`:
            // a raw pointer is `*const` or `*mut` and never neither.
            match p.mutability {
                syn::PointerMutability::Mut(_) => Shape::MutPtr,
                syn::PointerMutability::Const(_) => Shape::ConstPtr,
            },
            false,
        ),
        Type::Path(p) => {
            let Some(seg) = p.path.segments.last() else {
                return (Shape::Opaque, false);
            };
            match seg.ident.to_string().as_str() {
                // `Env` states who supplies, not whether it can be absent,
                // so nullability comes from what it wraps.
                "Env" => (Shape::Env, inner(seg).is_some_and(|t| classify(&t).1)),
                // The slot wrappers. Their source comes from `stated_source`;
                // this reads them only for `nullable`, which belongs to the
                // wrapped type — landing them on `Opaque` would silently
                // demote a nullable result to a required one.
                "In" | "InOut" | "Out" | "InRow" | "OutRow" | "InSlot" | "OutSlot" | "Weight"
                | "Bank" | "Unbound" | "Aux" | "Param" => {
                    inner(seg).map_or((Shape::Opaque, false), |t| classify(&t))
                }
                // The two shapes the ABI marks nullable. A nullable read is
                // spelled `MaybeConst`, which is why only the `Option` form
                // lands on `MutPtr`.
                "Option" => (Shape::MutPtr, true),
                "MaybeConst" => (Shape::ConstPtr, true),
                _ => (Shape::Opaque, false),
            }
        }
        _ => (Shape::Opaque, false),
    }
}

/// Rewrite a mark's short variant spelling into `Source::Slot(Kind, u8)`.
///
/// The attribute grammar keeps the short spellings the variants no longer
/// have. Anything not in the list passes through untouched.
fn fold_indexed(e: &syn::Expr) -> proc_macro2::TokenStream {
    const KINDS: [&str; 9] = [
        "In", "Out", "Weight", "Param", "ParamF32", "Aux", "OutWidth", "InWidth", "OutElements",
    ];
    if let syn::Expr::Call(call) = e
        && let syn::Expr::Path(p) = &*call.func
        && let Some(seg) = p.path.segments.last()
        && KINDS.contains(&seg.ident.to_string().as_str())
        && call.args.len() == 1
    {
        let kind = &seg.ident;
        let at = &call.args[0];
        return quote!(::core::option::Option::Some(::kernels::Source::Slot(::kernels::Kind::#kind, #at)));
    }
    // A nullary variant name is a key name: `#[source(KvKeys)]` names a
    // `keys::` type. The mapping is one-to-one but not the identity —
    // `WeightNamed` is declared by `keys::NamedWeight`, `WeightNamed2` by
    // `NamedWeight2`.
    if let syn::Expr::Path(p) = e
        && let Some(seg) = p.path.segments.last()
        && seg.arguments.is_none()
    {
        let raw = seg.ident.to_string();
        if raw == "Unbound" {
            return quote!(::core::option::Option::None);
        }
        {
            let key = match raw.as_str() {
                "WeightNamed" => "NamedWeight".to_owned(),
                "WeightNamed2" => "NamedWeight2".to_owned(),
                other => other.to_owned(),
            };
            let id = syn::Ident::new(&key, seg.ident.span());
            return quote!(::core::option::Option::Some(::kernels::Source::Named(
                <::kernels::keys::#id as ::kernels::keys::Fact>::KEY
            )));
        }
    }
    quote!(::core::option::Option::Some(::kernels::Source::#e))
}

/// The `Source` a wrapper states, or `None` for a type that states none.
///
/// A stated index sets the counter rather than consuming the next number: it
/// moves to `N + 1` and positional inference picks up from there, so a bare
/// `*const T` after a stated `In<1, _>` continues the run rather than
/// restarting it. Wrappers that state a fact with no position touch nothing.
fn stated_source(
    ty: &Type,
    ins: &mut u8,
    outs: &mut u8,
    generics: &[String],
) -> Option<proc_macro2::TokenStream> {
    let Type::Path(p) = ty else {
        return None;
    };
    // The last segment, so `Width<slot::Out<0>>` and a `use`d
    // `Width<Out<0>>` read the same.
    let seg = p.path.segments.last()?;
    match seg.ident.to_string().as_str() {
        // `InOut` derives `In`: a `Source` says which operand, never what is
        // done to it.
        "In" | "InOut" => {
            let at = const_val(seg)?;
            *ins = at + 1;
            Some(quote!(::core::option::Option::Some(::kernels::Source::Slot(::kernels::Kind::In, #at))))
        }
        "Out" => {
            let at = const_val(seg)?;
            *outs = at + 1;
            Some(quote!(::core::option::Option::Some(::kernels::Source::Slot(::kernels::Kind::Out, #at))))
        }
        // The width these carry costs no variant: a `Source` says which
        // operand, never how big it is.
        "InRow" => {
            let at = const_val(seg)?;
            *ins = at + 1;
            Some(quote!(::core::option::Option::Some(::kernels::Source::Slot(::kernels::Kind::In, #at))))
        }
        "OutRow" => {
            let at = const_val(seg)?;
            *outs = at + 1;
            Some(quote!(::core::option::Option::Some(::kernels::Source::Slot(::kernels::Kind::Out, #at))))
        }
        "InSlot" => {
            let at = const_val(seg)?;
            *ins = at + 1;
            Some(quote!(::core::option::Option::Some(::kernels::Source::Slot(::kernels::Kind::In, #at))))
        }
        "OutSlot" => {
            let at = const_val(seg)?;
            *outs = at + 1;
            Some(quote!(::core::option::Option::Some(::kernels::Source::Slot(::kernels::Kind::Out, #at))))
        }
        // A refusal, not a mark. Touches neither counter, which is the
        // point: a bare pointer would have consumed an input slot.
        "Unbound" => Some(quote!(::core::option::Option::None)),
        // The positional bank. Named and positional read different tables and
        // the index cannot tell them apart, since both have a slot 1.
        "Bank" => {
            let at = const_val(seg)?;
            Some(quote!(::core::option::Option::Some(::kernels::Source::Slot(::kernels::Kind::Weight, #at))))
        }
        // The named bank: `Facts` carries `w_named` and `w_named2` and
        // nothing else, so there is no third. A third refuses rather than
        // answering `None`, which would read as "no wrapper stated a source"
        // and let `Weight<2, _>` fall to positional inference — a parameter
        // that says "weight" out loud, silently bound to an operand.
        "Weight" => match const_val(seg)? {
            0 => Some(quote!(::core::option::Option::Some(::kernels::Source::Named(<kernels::keys::NamedWeight as kernels::keys::Fact>::KEY)))),
            1 => Some(quote!(::core::option::Option::Some(::kernels::Source::Named(<kernels::keys::NamedWeight2 as kernels::keys::Fact>::KEY)))),
            // Formatted here rather than with `concat!`/`stringify!`: `#n`
            // quotes as a suffixed literal, giving "Weight<2u8, _>".
            n => {
                let msg = format!(
                    "Weight<{n}, _> has no bank. `Facts` carries `w_named` and \
                     `w_named2` and nothing else, so only Weight<0, _> and \
                     Weight<1, _> name one. For the POSITIONAL bank -- \
                     `b.args[spec.n_in + spec.n_out + {n}]`, which is what \
                     mla_absorb binds from -- write #[source(Weight({n}))]; the \
                     wrapper spells the NAMED bank and the two are different \
                     reads."
                );
                Some(quote!(::core::compile_error!(#msg)))
            }
        },
        "Aux" => const_arg(seg).map(|n| quote!(::core::option::Option::Some(::kernels::Source::Slot(::kernels::Kind::Aux, #n)))),
        // Touches neither counter: `ins`/`outs` walk the operand run, while
        // `N` here indexes `spec.params[]`. Advancing an operand counter from
        // a param would shift every bare pointer after it one slot along.
        "Param" => const_arg(seg).map(|n| quote!(::core::option::Option::Some(::kernels::Source::Slot(::kernels::Kind::Param, #n)))),
        "ParamF32" => const_arg(seg).map(|n| quote!(::core::option::Option::Some(::kernels::Source::Slot(::kernels::Kind::ParamF32, #n)))),
        // The fact is the type, so the macro decides nothing: it emits a
        // path and the fact's own declaration answers it.
        //
        // The `keys::` prefix is required; a `use`d shorthand is not
        // supported. A bare capitalised `Env<T>` is a driver-owned handle
        // the driver hands over whole, so `Env<Buf>` and `Env<RmsEps>` would
        // otherwise look identical and mean opposite things.
        "Env" => {
            let fact = inner(seg)?;
            let Type::Path(fp) = &fact else {
                return None;
            };
            let n = fp.path.segments.len();
            if n >= 2 && fp.path.segments[n - 2].ident == "keys" {
                return Some(quote!(::core::option::Option::Some(
                    <#fact as ::kernels::keys::Fact>::SOURCE
                )));
            }

            // An unqualified capitalised `Env<X>` is a build stop, because
            // the failure is otherwise silent: it falls past this arm to
            // `Shape::Env`, which answers `None`, and nothing rescues it.
            //
            // The handle allowlist is written rather than computed because a
            // handle and a fact are both `Env`. Generic parameters need no
            // entry; they are checked against the function's own generics.
            if n == 1 && seg_is_bare(&fp.path.segments[0]) {
                let id = fp.path.segments[0].ident.to_string();
                const HANDLES: [&str; 4] = ["Buf", "I32s", "PrefillPlan", "DecodePlan"];
                if !HANDLES.contains(&id.as_str()) && !generics.contains(&id) {
                    let msg = format!(
                        "`Env<{id}>` is unqualified. If {id} is a fact, write \
                         `Env<keys::{id}>` -- `stated_source` reads the `keys::` \
                         segment, and without it this falls through to no \
                         source at all -- kilimanjaro2 Stage 5 deleted the \
                         parameter-name table that used to rescue it, so the \
                         binding is simply lost with no diagnostic. If {id} \
                         is a driver-owned handle, add it beside `Buf`, `I32s`, \
                         `PrefillPlan` and `DecodePlan` in `stated_source` and \
                         say what owns it."
                    );
                    return Some(quote!(::core::compile_error!(#msg)));
                }
            }
            None
        }
        _ => None,
    }
}

/// A path segment with no generic arguments and a capitalised initial.
///
/// Both halves matter: a fact takes no parameters, so the arguments rule out
/// `Env<Option<NonNull<T>>>`, and the case rules out `Env<i32>`.
fn seg_is_bare(seg: &syn::PathSegment) -> bool {
    matches!(seg.arguments, syn::PathArguments::None)
        && seg
            .ident
            .to_string()
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_uppercase())
}

/// The `usize` const a marker carries, as the `u8` every `Source` index is.
///
/// `None` rather than an error for a non-literal: the caller's fallback of
/// position, then no source, is the honest answer for a signature this macro
/// cannot read.
fn const_arg(seg: &syn::PathSegment) -> Option<proc_macro2::TokenStream> {
    const_val(seg).map(|n| quote!(#n))
}

/// The `usize` const a marker carries, as a number this macro can branch on.
///
/// [`const_arg`] emits it; this reads it, because `Weight` chooses a variant
/// rather than interpolating an index and so needs the value at macro time.
fn const_val(seg: &syn::PathSegment) -> Option<u8> {
    let syn::PathArguments::AngleBracketed(a) = &seg.arguments else {
        return None;
    };
    a.args.iter().find_map(|g| match g {
        syn::GenericArgument::Const(syn::Expr::Lit(l)) => match &l.lit {
            syn::Lit::Int(i) => i.base10_parse::<u8>().ok(),
            _ => None,
        },
        _ => None,
    })
}

/// `Or<*const T>` → `*const T`.
fn inner(seg: &syn::PathSegment) -> Option<Type> {
    let syn::PathArguments::AngleBracketed(a) = &seg.arguments else {
        return None;
    };
    a.args.iter().find_map(|g| match g {
        syn::GenericArgument::Type(t) => Some(t.clone()),
        _ => None,
    })
}

/// `Named(..)`, in any of the three shapes an attribute can spell it: a call
/// `Named("x")`, a bare path `Named`, or a qualified
/// `kernels::Source::Named("x")`. Checking the last segment catches all
/// three.
fn is_named(e: &syn::Expr) -> bool {
    let path = match e {
        syn::Expr::Call(c) => match &*c.func {
            syn::Expr::Path(p) => &p.path,
            _ => return false,
        },
        syn::Expr::Path(p) => &p.path,
        _ => return false,
    };
    path.segments.last().is_some_and(|s| s.ident == "Named")
}

/// The parameter's own identifier, which is the fact's name.
fn param_name(pt: &PatType) -> Result<String, Error> {
    match &*pt.pat {
        Pat::Ident(id) => Ok(id.ident.to_string()),
        other => Err(Error::new(
            other.span(),
            "a routine's parameters are named; this one is a pattern",
        )),
    }
}

/// Read and remove `#[source(...)]`, so the emitted `fn` compiles.
///
/// This is the hole in `Source`'s key discipline. Facts otherwise reach
/// `Source::Named` through a `keys::` type that cannot be misspelled, but
/// this emits `::kernels::Source::#e` verbatim — so a typo'd
/// `Named("rms_esp")` compiles, derives, and refuses at bind time as
/// `Unstated`, indistinguishable from a fact the driver really lacks. Hence
/// the refusal below. Safe only once deleted, which waits on the last
/// `#[source(..)]` mark.
fn take_source_attr(pt: &mut PatType) -> Result<Option<proc_macro2::TokenStream>, Error> {
    let mut found = None;
    let mut err = None;
    pt.attrs.retain(|a| {
        if !a.path().is_ident("source") {
            return true;
        }
        match a.parse_args::<syn::Expr>() {
            Ok(e) if is_named(&e) => {
                err = Some(Error::new(
                    e.span(),
                    "`#[source(Named(..))]` writes a fact's key as a string \
                     literal, which is the one spelling a typo survives: it \
                     compiles, derives, and refuses at bind time as \
                     `Unstated`, indistinguishable from a fact the driver \
                     really lacks. Say the fact as a type -- \
                     `Env<keys::TheFact>` -- and mint the key in \
                     `kernels/src/keys.rs` if it does not exist",
                ));
            }
            // The escape hatch survives because a handful of marks are
            // inexpressible as wrappers; `kernels-cuda`'s `stated_columns`
            // test pins each with the class that stops it.
            Ok(e) => found = Some(fold_indexed(&e)),
            Err(e) => err = Some(e),
        }
        false
    });
    err.map_or(Ok(found), Err)
}


/// Read and remove `#[lit(..)]`, the one spelling that derives a
/// [`kernels::Source::Lit`].
///
/// An attribute rather than a type because the value is the whole content of
/// the mark and a type cannot carry it — a constant has no type distinguishing
/// it from a fact. Unlike `#[source(Named(..))]` this cannot be silently
/// wrong: the literal is checked against the parameter's own type here, at
/// expansion.
fn take_lit_attr(pt: &mut PatType) -> Result<Option<proc_macro2::TokenStream>, Error> {
    let ty = (*pt.ty).clone();
    let mut found = None;
    let mut err = None;
    pt.attrs.retain(|a| {
        if !a.path().is_ident("lit") {
            return true;
        }
        match a.parse_args::<syn::Expr>().and_then(|e| lit_of(&e, &ty)) {
            Ok(t) => found = Some(t),
            Err(e) => err = Some(e),
        }
        false
    });
    err.map_or(Ok(found), Err)
}

/// `#[lit(..)]`'s argument as a `Source::Lit`, checked against the parameter.
///
/// The agreement check is why this happens at expansion: `Lit::I32` on an
/// `f32` parameter would mint `ArgValue::I32` against a declared `Ty::F32`,
/// and `abi_admits` would refuse a row that reads correctly.
fn lit_of(e: &syn::Expr, ty: &Type) -> Result<proc_macro2::TokenStream, Error> {
    let some = |v: proc_macro2::TokenStream| {
        quote!(::core::option::Option::Some(::kernels::Source::Lit(::kernels::Lit::#v)))
    };
    let disagree = |want: &str| {
        let got = scalar_name(ty).unwrap_or_else(|| "that type".to_owned());
        Error::new(
            e.span(),
            format!(
                "`#[lit(..)]` wrote a {want} onto a `{got}` parameter. The \
                 literal binds as an `ArgValue` of its own type and \
                 `abi_admits` compares it against the parameter's declared \
                 `Ty`, so this row would refuse at bind time for a reason \
                 nothing in the signature explains"
            ),
        )
    };
    // A path and not a literal: Rust has no null literal, and `0` is not one.
    if let syn::Expr::Path(p) = e
        && p.path.is_ident("null")
    {
        return if is_ptr_param(ty) {
            Ok(some(quote!(Null)))
        } else {
            Err(disagree("null"))
        };
    }
    let (neg, lit) = match e {
        syn::Expr::Lit(l) => (false, &l.lit),
        syn::Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => match &*u.expr {
            syn::Expr::Lit(l) => (true, &l.lit),
            other => return Err(Error::new(other.span(), "`#[lit(..)]` takes a literal")),
        },
        other => {
            return Err(Error::new(
                other.span(),
                "`#[lit(..)]` takes a literal, or `null` on a pointer",
            ));
        }
    };
    let scalar = scalar_name(ty);
    match lit {
        syn::Lit::Bool(b) if !neg => {
            if scalar.as_deref() != Some("bool") {
                return Err(disagree("bool"));
            }
            let v = b.value;
            Ok(some(quote!(Bool(#v))))
        }
        syn::Lit::Int(i) => {
            if scalar.as_deref() != Some("i32") {
                return Err(disagree("i32"));
            }
            let n: i32 = i.base10_parse()?;
            let n = if neg { -n } else { n };
            Ok(some(quote!(I32(#n))))
        }
        syn::Lit::Float(x) => {
            if scalar.as_deref() != Some("f32") {
                return Err(disagree("f32"));
            }
            let v: f32 = x.base10_parse()?;
            let v = if neg { -v } else { v };
            Ok(some(quote!(F32(#v))))
        }
        other => Err(Error::new(
            other.span(),
            "`Lit` holds a bool, an `i32`, an `f32` or a null; a string or a \
             character has no `ArgValue`",
        )),
    }
}

/// The scalar a `#[lit(..)]` has to agree with, `Env<_>` peeled.
fn scalar_name(ty: &Type) -> Option<String> {
    let Type::Path(p) = ty else {
        return None;
    };
    let seg = p.path.segments.last()?;
    if seg.ident == "Env" {
        return inner(seg).as_ref().and_then(scalar_name);
    }
    matches!(seg.arguments, syn::PathArguments::None).then(|| seg.ident.to_string())
}

/// Whether a null can land here: a raw pointer, or one inside a wrapper.
fn is_ptr_param(ty: &Type) -> bool {
    match ty {
        Type::Ptr(_) => true,
        Type::Path(p) => p.path.segments.last().is_some_and(|s| {
            matches!(s.ident.to_string().as_str(), "Env" | "Unbound" | "MaybeConst")
                && inner(s).as_ref().is_some_and(is_ptr_param)
        }),
        _ => false,
    }
}
