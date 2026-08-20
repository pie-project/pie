//! `#[routine]` — a kernel's row, built beside the `fn` it is derived from.
//!
//! # Why an attribute and not just `routine!`
//!
//! `routine!` receives a PATH. By then the parameter names live in a `fn` the
//! macro never sees, and neither does the module. An attribute sits on the
//! `fn` and sees both, which is the whole of what this crate is for.
//!
//! # What it derives
//!
//! | column | from |
//! |---|---|
//! | the trace name | the `fn`'s name, plus its instantiation: `rmsnorm` + `bf16` |
//! | the namespace | `module_path!()`, expanded at the `fn` |
//! | `args`, `sides`, `sources`, `spelling` | `KernelFn`, off the parameter TYPES |
//! | `in_place` | the `InOut` marks, through `Source::Alias` |
//! | `name`, `nullable` | the signature, which is the only thing left a type cannot say |
//!
//! # What it no longer derives, and why that matters
//!
//! A SOURCE. It used to compute one per parameter by reading SYNTAX -- a
//! table of marks, a `keys::` prefix rule, an allowlist of four handle types,
//! and a `compile_error!` for `Weight<2, _>` because `Facts` has two named
//! banks. All of it existed because a reader of syntax cannot tell a fact from
//! a handle, or know how many weight slots there are.
//!
//! Meanwhile `KernelFn::SOURCES` computed the same column off the TYPES, and
//! the three shader planes bound from that one. Two columns, two readers, and
//! a dump of both found rows where they disagreed. The types answer now, and
//! everything in that paragraph is deleted.

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{
    Error, FnArg, ItemFn, Pat, PatType, Type, parse_macro_input, spanned::Spanned,
};

/// Declare a routine: its column, its trace name and its namespace.
///
/// # What the attribute takes
///
/// ```ignore
/// #[routine]                       // a concrete `fn`: the name is its own
/// #[routine(bf16)]                 // a generic one: instantiated, named `<fn>_bf16`
/// #[routine(bf16, whole)]          // plus the facts no signature carries
/// ```
///
/// # Why the name is not written
///
/// It used to be, on the ROW: `routine!(rmsnorm_bf16 = rmsnorm::<bf16>)`. The
/// macro's own doc defended that as necessary because `stringify!` would
/// answer `rmsnorm::<bf16>`, which no trace can state -- true of a `macro_
/// rules!` reading a path, and not true here, where the parts arrive
/// separately and the name is `<fn>_<generics>`.
///
/// The rule was checked before it was relied on: across all 74 generic rows
/// the written name equalled that composition at every one, and no base was
/// instantiated twice. So the aliasing form was defending a capability -- one
/// body answering several symbols -- that nothing in the tree uses.
#[proc_macro_attribute]
pub fn routine(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut f = parse_macro_input!(item as ItemFn);
    let spec = match Spec::parse(attr.into()) {
        Ok(s) => s,
        Err(e) => return e.to_compile_error().into(),
    };
    // THE MARK CHECK RUNS FIRST, BECAUSE `derive_all` CONSUMES `#[unbound]`.
    // The attribute is the stated absence and it is stripped at expansion --
    // rustc has no such attribute -- so a check that ran afterwards would see
    // every escape as an oversight and refuse the whole tree.
    let by_path_early = spec.has("untraced");
    if !(spec.has("uncolumned") || by_path_early)
        && let Err(e) = every_parameter_is_a_mark(&f)
    {
        return e.to_compile_error().into();
    }
    let derived = match derive_all(&mut f) {
        Ok(d) => d,
        Err(e) => return e.to_compile_error().into(),
    };
    // A ROUTINE'S ARITY IS ITS KERNEL'S ABI, not a style choice. The device
    // signature fixes how many arguments there are and no host decision can
    // reduce them, so the lint has nothing to tell a routine.
    //
    // Emitted here rather than written at the site, which is what it was: 50
    // per-`fn` copies and 33 module-level ones. The module-level form is the
    // reason to move it -- `#![allow(..)]` covers a whole FILE, so a private
    // helper that really had grown twelve arguments was covered too. This
    // covers routines and nothing else.
    f.attrs.push(syn::parse_quote!(#[allow(clippy::too_many_arguments)]));

    // THE ELEMENT BOUND IS THE PLANE'S, NOT THE SIGNATURE'S.
    //
    // A generic routine names an element type and every plane needs something
    // of it -- CUDA that it can be instantiated in device text AND bound
    // through a pointer ABI, the shader planes only that it is an `Elem`. Left
    // to the signature, that came out as `where T: crate::jit::abi::Pointee`
    // on fifty-seven CUDA routines and nothing at all on the other three, so a
    // `where` line was a backend fingerprint: read one and you knew the file.
    //
    // Each plane says it once, as `crate::RoutineElem`, and this adds it to
    // every type parameter that does not already carry a bound. A routine
    // needing MORE than the plane's minimum still writes its own -- the one
    // that also wants `MaybeConst<T>: Abi` keeps saying so -- because this
    // only fills an EMPTY bound list and never edits a stated one.
    for p in f.sig.generics.type_params_mut() {
        if p.bounds.is_empty() {
            p.bounds.push(syn::parse_quote!(crate::RoutineElem));
        }
    }

    let base = f.sig.ident.clone();
    let vis = f.vis.clone();
    // THE TRACE NAME, COMPOSED. `rmsnorm` + `bf16` is `rmsnorm_bf16`, which is
    // the string a text states and a `Family` used to prefix.
    let trace = spec.trace_name(&base.to_string());
    // `rmsnorm_bf16` -> `RMSNORM_BF16_ROUTINE`.
    //
    // THE SUFFIX IS LOAD-BEARING. A family often keeps an entrypoint table
    // under the routine's own name in caps -- `static QMV_FAST: [&str; 6]`
    // beside `fn qmv_fast` -- and forty of them collide without it. The row is
    // reached through the distributed slice rather than by name, so the
    // spelling costs a reader nothing.
    let row = syn::Ident::new(&format!("{}_ROUTINE", trace.to_uppercase()), base.span());
    let body = spec.body(&base);
    // `untraced` AND `uncolumned` ARE NOT BUILDER FLAGS. The first says NO
    // TRACE REACHES THIS SYMBOL, so the row carries no argument table and its
    // body refuses a dispatch by string; the second says the row deliberately
    // carries no operand column. Both change what the row IS, so they are read
    // here and the rest are appended.
    //
    // It was `driver_bound` and the rename is the point: `kernels-*` may not
    // know which caller reaches it. Whether a TRACE does is a fact about the
    // routine -- it is what having a column means -- and whether a driver then
    // calls the `fn` by path is that driver's business. The old name put the
    // consumer in the producer's vocabulary, and the flag it names has never
    // meant anything else.
    let by_path = spec.has("untraced");
    let uncolumned = spec.has("uncolumned") || by_path;
    // EVERY PARAMETER OF A COLUMNED ROUTINE MUST BE A MARK, because the binder
    // resolves the column and a mark is the only thing that says where a value
    // comes from. A bare `i32` derives no source, so `bind` refuses it with
    // `Unstated` -- at run time, on a fire, having compiled and shipped.
    let facts: Vec<&syn::Ident> = spec
        .facts()
        .iter()
        .filter(|f| *f != "untraced" && *f != "uncolumned")
        .collect();
    let asked = asked_facts(&f);
    // `Derivation::DERIVED` IS ALWAYS FULL, EVEN ON AN UNCOLUMNED ROW.
    //
    // It carries the parameter's own NAME and whether a null may land there,
    // both read off the syntax -- facts that are true of a `untraced`
    // launcher exactly as they are of a bound one, and the only thing holding
    // its arguments apart. `layout::envelope_update_appended` is the case: five
    // adjacent `i32`s whose ORDER is the launch's, where a permutation is a
    // wrong stride rather than a type error, and the names are what pin it.
    //
    // What an uncolumned row does not have is a SOURCE column, which is a
    // claim about who BINDS each argument -- and nothing binds these.
    let column = quote!(&[#(#derived),*]);
    // AN UNCOLUMNED ROW HAS NO SOURCE COLUMN EITHER, and asking for one is not
    // merely empty -- it is a compile error. `sources` goes through `KernelFn`,
    // which requires every parameter to be an `Arg`, and the whole point of
    // `untraced` is that some are not: `lora_qkv_correction` takes a
    // `Staged<'_>`, an aggregate the arm builds and no value can carry.
    let sources = if uncolumned {
        quote!(&[])
    } else {
        quote!(::kernels::routine::sources::<crate::Plane, _, _>(#body))
    };
    let asked_list = if uncolumned {
        quote!(&[])
    } else {
        quote!(&[#(<#asked as ::kernels::keys::Fact>::KEY),*])
    };
    // THE ROW CARRIES THE COLUMN. `routine!` cannot reach it: it is handed a
    // BODY expression, and the names-and-nullability column is read off the
    // SYNTAX and hangs on the marker type this attribute also emits. So the
    // attribute, which has both in hand, attaches it -- and `Stated::derived`
    // downstream is what tells `arity_problem` which operands are optional.
    let with_column = quote!(
        .derived(<#base as ::kernels::Derivation>::DERIVED)
        .asking(<#base as ::kernels::Derivation>::ASKED)
    );
    // The module the `fn` is written in, unless the attribute says otherwise.
    // See `Spec::namespace` for the one file that says otherwise and why.
    let ns = match &spec.namespace {
        Some(ns) => quote!(#ns),
        None => quote!(::kernels::routine::namespace(::core::module_path!())),
    };
    let row_expr = if by_path {
        quote!(::kernels::untraced!(
            crate::Plane,
            #trace,
            #body,
            namespace = #ns
            #(, #facts)*
        )#with_column)
    } else {
        quote!(::kernels::routine!(
            crate::Plane,
            #trace,
            #body,
            namespace = #ns
            #(, #facts)*
        )#with_column)
    };

    let doc = format!(
        "`{trace}`'s operands, derived by `#[routine]` from its signature.\n\n An uninhabited marker in the TYPE namespace wearing the function's name -- a unit struct would take the VALUE namespace too and collide with the `fn`."
    );

    quote! {
        #f

        #[doc = #doc]
        #[allow(non_camel_case_types)]
        #vis enum #base {}

        impl ::kernels::Derivation for #base {
            const DERIVED: &'static [::kernels::Derived] = #column;
            const SOURCES: &'static [::core::option::Option<::kernels::Source>] = #sources;
            // THE FACTS THE BODY ASKS FOR, SCANNED OUT OF THE BODY.
            //
            // The derived column lost its `Env` half, and with it the check
            // that found `rope_scale`, `rope_theta` and `mscale` unanswered on
            // Vulkan: a driver test used to walk `SOURCES` and ask *"does this
            // backend answer every fact its own kernels name"*. `ctx.ask::<_,
            // K>()` is a call, not a declaration, and is not statically
            // enumerable -- so `#[routine]`, which already parses the whole
            // `ItemFn`, collects the turbofishes instead.
            //
            // Same fidelity as the parameter run for a fact asked in the body,
            // and it cannot drift from the calls. It MISSES a fact asked inside
            // a helper, which is a real step down from a type-system guarantee
            // to a syntactic one, accepted deliberately rather than discovered
            // later.
            const ASKED: &'static [&'static str] = #asked_list;
        }

        // THE ROW, BESIDE THE `fn`, AND REGISTERED BY EXISTING.
        //
        // `#[distributed_slice]` puts it in a link section the linker gathers,
        // so there is no list to add it to and no list to forget it in. That
        // was the last hand-written thing about a routine: a membership line
        // whose omission left the routine compiled, correct and unreachable,
        // with nothing to report it.
        //
        // The namespace comes from `module_path!()`, which expands HERE --
        // which is what retired `Family`: it was a container whose whole
        // content was one derivable string attached to a group.
        //
        // `crate::Plane` is the backend, aliased by each kernels crate, so
        // this attribute serves all four rather than naming CUDA's.
        #[::linkme::distributed_slice(crate::ROUTINES)]
        #[allow(non_upper_case_globals)]
        static #row: ::kernels::routine::Routine<crate::Plane> = #row_expr;
    }
    .into()
}

/// What the attribute was given: an instantiation and the stated facts.
struct Spec {
    /// The generic arguments to instantiate with, in order.
    generics: Vec<syn::Type>,
    /// `whole`, `depth_prefix_plan`, `uncolumned` — the facts no signature
    /// carries. `in_place` IS NOT AMONG THEM: it derives from `InOut` now.
    facts: Vec<syn::Ident>,
    /// The prefix a trace spells, where it is not the `fn`'s own module.
    ///
    /// EMPTY MEANS `module_path!()`, which is right for every routine that
    /// lives in the family it belongs to. `driver_internal.rs` is the file
    /// that does not: four host programs there are named by statements as
    /// `attn::`, `layout::`, `mlp::` and `ssm::`, because the lowering emits
    /// those strings, and the file collects them by CALLER rather than by
    /// family. A `routine!` line in each family module used to name them by
    /// bare identifier; a row registers where its `fn` is written now, so the
    /// namespace has to be said where the `fn` is.
    namespace: Option<String>,
}

impl Spec {
    fn parse(attr: proc_macro2::TokenStream) -> Result<Self, Error> {
        let (mut generics, mut facts) = (Vec::new(), Vec::new());
        let mut namespace = None;
        if attr.is_empty() {
            return Ok(Self { generics, facts, namespace });
        }
        let parsed = syn::parse2::<Args>(attr)?;
        for a in parsed.0 {
            match a {
                Arg::Fact(id) => {
                    if id == "in_place" {
                        return Err(Error::new(
                            id.span(),
                            "`in_place` is not stated any more: mark the parameter that \
                             wears both slots `InOut<_>` and the pair derives from it",
                        ));
                    }
                    facts.push(id);
                }
                Arg::Generic(t) => generics.push(t),
                Arg::Namespace(ns) => namespace = Some(ns),
            }
        }
        Ok(Self { generics, facts, namespace })
    }

    /// `rmsnorm` + `[bf16]` -> `rmsnorm_bf16`.
    ///
    /// Const generics do not join the name: `rope::<bf16, 256>` is
    /// `rope_bf16`, because 256 is a tuning number and not a point on the
    /// dtype axis a trace names.
    fn trace_name(&self, base: &str) -> String {
        let mut out = base.to_string();
        for g in &self.generics {
            if let syn::Type::Path(p) = g
                && let Some(seg) = p.path.segments.last()
                && seg.arguments.is_none()
            {
                out.push('_');
                out.push_str(&seg.ident.to_string().to_lowercase());
            }
        }
        out
    }

    /// The expression `routine!` invokes: the `fn`, instantiated if generic.
    fn body(&self, base: &syn::Ident) -> proc_macro2::TokenStream {
        if self.generics.is_empty() {
            quote!(#base)
        } else {
            let g = &self.generics;
            quote!(#base::<#(#g),*>)
        }
    }

    fn facts(&self) -> &[syn::Ident] {
        &self.facts
    }

    /// Whether the attribute stated `name`.
    fn has(&self, name: &str) -> bool {
        self.facts.iter().any(|f| f == name)
    }
}

/// The attribute's comma-separated arguments.
struct Args(Vec<Arg>);

/// One argument: a fact to state, a namespace to state it under, or a type to
/// instantiate with.
enum Arg {
    Fact(syn::Ident),
    Generic(syn::Type),
    /// `namespace = "attn"` — the prefix a TRACE spells, where it is not the
    /// module the `fn` lives in.
    Namespace(String),
}

/// One `namespace = "..."`, or a bare item.
enum Item {
    Namespace(syn::LitStr),
    Bare(syn::Type),
}

impl syn::parse::Parse for Item {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        if input.peek(syn::Ident) && input.peek2(syn::Token![=]) {
            let key: syn::Ident = input.parse()?;
            if key != "namespace" {
                return Err(syn::Error::new(
                    key.span(),
                    "the only keyed argument is `namespace = \"..\"`",
                ));
            }
            let _: syn::Token![=] = input.parse()?;
            return Ok(Self::Namespace(input.parse()?));
        }
        Ok(Self::Bare(input.parse()?))
    }
}

impl syn::parse::Parse for Args {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let raw =
            syn::punctuated::Punctuated::<Item, syn::Token![,]>::parse_terminated(input)?;
        let mut named = Vec::new();
        let mut items = Vec::new();
        for item in raw {
            match item {
                Item::Namespace(lit) => named.push(Arg::Namespace(lit.value())),
                Item::Bare(t) => items.push(t),
            }
        }
        // A BARE LOWERCASE WORD IS A FACT AND EVERYTHING ELSE IS A TYPE, and
        // the case is what tells them apart: `whole` is a fact, `bf16` is a
        // type. Rust's own convention does the work, and a mistake either way
        // is a name that does not resolve rather than a silent reading.
        const FACTS: [&str; 7] = [
            "whole", "depth_prefix_plan", "uncolumned", "untraced", "no_join",
            "internal", "driver",
        ];
        Ok(Self(
            named
                .into_iter()
                .chain(items.into_iter().map(|t| match &t {
                    syn::Type::Path(p)
                        if p.qself.is_none()
                            && p.path.segments.len() == 1
                            && p.path.segments[0].arguments.is_none()
                            && FACTS.contains(&p.path.segments[0].ident.to_string().as_str()) =>
                    {
                        Arg::Fact(p.path.segments[0].ident.clone())
                    }
                    _ => Arg::Generic(t),
                }))
                .collect(),
        ))
    }
}

/// Walk the parameters, stripping helper attributes as it goes.
fn derive_all(f: &mut ItemFn) -> Result<Vec<proc_macro2::TokenStream>, Error> {
    let mut out = Vec::new();
    for (i, arg) in f.sig.inputs.iter_mut().enumerate() {
        // The context is the backend's, not the statement's, and never in
        // the argument run: `KernelFn::invoke` takes it beside `args`.
        if i == 0 {
            continue;
        }
        let FnArg::Typed(pt) = arg else {
            return Err(Error::new(arg.span(), "a routine takes no `self`"));
        };
        // `#[unbound]` IS CONSUMED HERE. It says one thing to
        // `every_parameter_is_a_mark` and nothing to the column, so it must not
        // survive into the emitted `fn` -- rustc has no such attribute.
        pt.attrs.retain(|a| !a.path().is_ident("unbound"));
        let name = param_name(pt)?;
        let nullable = is_nullable(&pt.ty);
        out.push(quote! {
            ::kernels::Derived { name: #name, nullable: #nullable }
        });
    }
    Ok(out)
}

/// Refuse a parameter that no mark claims.
///
/// # Why this is an error and not a lint
///
/// Because the alternative is a run-time refusal. A columned row goes through
/// `operands`, which reads a source per parameter; a bare scalar has none, so
/// the fire refuses with `Unstated { what: "an argument whose signature does
/// not say where it comes from" }` -- and only for the statements that reach
/// that routine, only once they run.
///
/// # There is no routine-level escape
///
/// There was, for one commit: `#[routine(unbound_params)]`, pinning the
/// ninety-five parameters that wore this shape when the check was written. It
/// is gone, because the escape belongs at the PARAMETER and already exists
/// there — a routine-level opt-out said it about a whole signature, and would
/// have taken the next parameter added to that routine with it.
///
/// # THE ESCAPE, STATED: `#[unbound]`
///
/// A parameter the ARM supplies and no column could ever bind says so, at the
/// parameter, in one word:
///
/// ```ignore
/// #[unbound] plan: PrefillPlan,
/// ```
///
/// `attention_flashinfer_prefill`'s own doc is why it has to exist: *"the arm
/// builds this plan from the host CSR mirrors on the way in; nothing published
/// it before the fire, so `operand()` has nothing to read and there is no
/// column to answer"*. That is a fact about the value, not an oversight, and
/// the attribute is what tells the two apart. It is stripped at expansion and
/// changes nothing but this check.
///
/// # AND THE UNSTATED POINTER, WHICH NEEDS NO WORD
///
/// `Env<T, keys::Unstated>` used to spell *"nothing supplies this"* about ONE
/// argument, and both went with `Env`. What replaces them is the ABSENCE of a
/// mark on a POINTER, which is exactly the scope `keys::Unstated` documented
/// for itself: *"For POINTERS. A scalar with no source already derives none,
/// so the mark would be noise; a bare `*const T` is different — it would be
/// counted into the next input slot"*. There is no counting to escape now, so
/// the bare pointer can simply say it.
///
/// The tree already records where this lands, on the other side: fifty-three
/// parameters over twenty-three symbols, each a `Bound { arm: None, unbound:
/// Some(reason) }` in `driver-cuda/src/bind/arms/`. A bare SCALAR is still
/// refused, because a scalar the statement carries has a mark now — `Const`
/// — and one it does not is an oversight rather than a stated absence.
///
/// The ninety-five resolved to three spellings, and which one is a fact to be
/// established rather than guessed -- a wrong guess binds the wrong slot
/// silently. `bind/arms/*.rs` had already established most of them: thirty-one
/// symbols there carry `unbound: Some(reason)`, and twenty-three of those were
/// exactly these routines.
fn every_parameter_is_a_mark(f: &ItemFn) -> Result<(), Error> {
    const MARKS: [&str; 4] = ["In", "Out", "InOut", "Const"];
    for (i, arg) in f.sig.inputs.iter().enumerate() {
        // The context is the backend's and never in the argument run.
        if i == 0 {
            continue;
        }
        let FnArg::Typed(pt) = arg else { continue };
        let named = match &*pt.ty {
            Type::Path(p) => p
                .path
                .segments
                .last()
                .is_some_and(|s| MARKS.contains(&s.ident.to_string().as_str())),
            // A BARE POINTER IS THE STATED ABSENCE. See the doc above.
            Type::Ptr(_) => true,
            _ => false,
        };
        // `#[unbound]`, FOR THE ABSENCES A POINTER CANNOT SPELL: a by-value
        // plan aggregate, a window the arm computed. Stripped below.
        let named = named || pt.attrs.iter().any(|a| a.path().is_ident("unbound"));
        // `MaybeConst<T>` is a pointer wearing a nullability, and `Option<
        // NonNull<T>>` is the same fact spelled by the standard library.
        let named = named
            || matches!(&*pt.ty, Type::Path(p) if p.path.segments.last().is_some_and(|s| {
                matches!(s.ident.to_string().as_str(), "MaybeConst" | "Option")
            }));
        if !named {
            let name = param_name(pt).unwrap_or_else(|_| "_".to_owned());
            return Err(Error::new(
                pt.span(),
                format!(
                    "`{name}` is not a mark, so nothing says where it comes from. \
                     A columned routine binds every parameter through its source, \
                     and this one derives none -- the fire refuses it as `Unstated` \
                     at run time. There are four marks and every one of them is a \
                     QUALITY: `In` reads, `Out` writes, `InOut` does both at one \
                     address, and `Const` is what the statement placed and the \
                     launch only reads -- a weight when its carrier is a \
                     `Tensor<E>`, a scalar of the params run when it is an `i32`, \
                     an `f32` or a `bool`. A fact only THIS FIRE can answer is not \
                     a parameter at all: ask for it in the body with \
                     `ctx.ask::<carrier, keys::X>()`."
                ),
            ));
        }
    }
    Ok(())
}

/// Whether a null can land at this parameter.
///
/// The one thing left that a TYPE cannot answer for the column, and it is
/// answered here syntactically for a reason: `Provenance::Either` marks the
/// nullable ABI spellings, but it also marks `keys::Unstated`, and the two
/// are 3 parameters and 54. Reading nullability off the provenance would turn
/// fifty-one required operands into ones the binder may silently answer with
/// a null.
fn is_nullable(ty: &Type) -> bool {
    let Type::Path(p) = ty else {
        return false;
    };
    let Some(seg) = p.path.segments.last() else {
        return false;
    };
    match seg.ident.to_string().as_str() {
        // AN `Option` IS THE ANSWER, WHATEVER IT WRAPS.
        //
        // `Option<Const<Tensor<f32>>>` is a bias a checkpoint may not carry
        // and `Option<NonNull<T>>` is the raw pointer under one; both mean the
        // same thing to the column, which is that the binder may put a null
        // here rather than refuse. It used to read only the second, because
        // absence was spelled on the POINTEE — `Const<Tensor<MaybeConst<T>>>`
        // — and an `Option` around a mark could not occur.
        "Option" => true,
        // A mark says nothing about absence, and `Tensor<E>` says nothing
        // either, being the CONSTRUCTOR the carrier is spelled with rather
        // than the carrier's own shape. Both are walked through in case the
        // `Option` sits further in.
        "In" | "Out" | "InOut" | "Const" | "Tensor" => {
            inner(seg).is_some_and(|t| is_nullable(&t))
        }
        _ => false,
    }
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


/// Every `keys::X` a `ctx.ask::<C, keys::X>()` in this body names.
///
/// A SYNTACTIC SCAN, and it says so: the turbofish is read out of the token
/// stream, so a fact asked through a helper `fn` in another module is not
/// found. See `Derivation::ASKED` for why that trade was taken.
///
/// The path is kept whole -- `keys::Rows`, `crate::keys::Rows`, whatever the
/// body wrote -- because the emitted const resolves it in the body's own
/// module, where it already resolves.
fn asked_facts(f: &ItemFn) -> Vec<syn::Path> {
    let mut out: Vec<syn::Path> = Vec::new();
    collect_asks(f.block.to_token_stream(), &mut out);
    out
}

/// Walk a token stream for `ask :: < .. , path >`, recursing into groups.
fn collect_asks(ts: proc_macro2::TokenStream, out: &mut Vec<syn::Path>) {
    use proc_macro2::TokenTree;
    let trees: Vec<TokenTree> = ts.into_iter().collect();
    let mut i = 0;
    while i < trees.len() {
        if let TokenTree::Group(g) = &trees[i] {
            collect_asks(g.stream(), out);
        }
        // `ask` `::` `<` .. `>` -- the angle-bracketed run is not a `Group`,
        // so it is gathered by scanning to the matching `>`.
        if let TokenTree::Ident(id) = &trees[i]
            && id == "ask"
            && matches!(trees.get(i + 1), Some(TokenTree::Punct(p)) if p.as_char() == ':')
        {
            let mut j = i + 2;
            while j < trees.len() {
                if matches!(&trees[j], TokenTree::Punct(p) if p.as_char() == '<') {
                    break;
                }
                j += 1;
            }
            let (mut depth, mut k, mut last) = (0i32, j, Vec::new());
            let mut piece: Vec<TokenTree> = Vec::new();
            while k < trees.len() {
                match &trees[k] {
                    TokenTree::Punct(p) if p.as_char() == '<' => {
                        depth += 1;
                        if depth > 1 {
                            piece.push(trees[k].clone());
                        }
                    }
                    TokenTree::Punct(p) if p.as_char() == '>' => {
                        depth -= 1;
                        if depth == 0 {
                            last = piece.clone();
                            break;
                        }
                        piece.push(trees[k].clone());
                    }
                    // THE CARRIER IS THE FIRST ARGUMENT AND THE QUESTION THE
                    // SECOND, in the order `Env<T, K>` named them. Only the
                    // question is a fact, so each comma at depth one starts
                    // the run over and the LAST run is the one kept.
                    TokenTree::Punct(p) if p.as_char() == ',' && depth == 1 => {
                        piece.clear();
                    }
                    t => piece.push(t.clone()),
                }
                k += 1;
            }
            if !last.is_empty() {
                let stream: proc_macro2::TokenStream = last.into_iter().collect();
                if let Ok(path) = syn::parse2::<syn::Path>(stream)
                    && !out.iter().any(|p| p == &path)
                {
                    out.push(path);
                }
            }
        }
        i += 1;
    }
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






