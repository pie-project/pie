//! pie-driver-abi derive macro.
//!
//! `#[schema(...)]` applied to a Rust struct or enum makes that type
//! the canonical wire schema. The macro emits:
//!
//!   * rkyv derives — `Archive` + `Serialize` + `Deserialize` (the
//!     archived form is the on-wire layout).
//!   * Reader accessors — `pie_<type>_<field>(*const ArchivedT)` for
//!     each field; uniform shape across structs and enums.
//!   * Parse entry — `pie_parse_<type>(bytes, len) -> *const ArchivedT`.
//!   * Descriptor type + builder — `#[repr(C)] PieXDesc` mirrors the
//!     schema in C-friendly form; `pie_build_<type>` consumes it and
//!     writes wire bytes.
//!   * PyO3 wrapper — `register_<type>(m)` adds a `Py<T>` class with
//!     getters for every field and a `parse(bytes)` classmethod.
//!
//! ## Container args
//!
//! Only trait-derive shortcuts:
//!   * `default` — `#[derive(Default)]`
//!   * `copy` — `#[derive(Copy)]`
//!   * `eq` — `#[derive(PartialEq, Eq)]`
//!   * `partial_eq` — `#[derive(PartialEq)]` only (for types with floats)
//!
//! Symbol prefixes derive mechanically from the type name in
//! `snake_case`. There are no field-level attributes — every field is
//! exposed; the type-driven classification handles primitives, `Vec<T>`,
//! `Option<T>`, `String`, nested schema types, and enum variants.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TS2;
use quote::{format_ident, quote};
use syn::{
    Data, DataEnum, DataStruct, DeriveInput, Fields, GenericArgument, Ident, PathArguments, Type,
    Variant,
};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[proc_macro_attribute]
pub fn schema(_args: TokenStream, input: TokenStream) -> TokenStream {
    let mut parsed = match syn::parse::<DeriveInput>(input) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };

    let derive_block = quote! {
        #[derive(::rkyv::Archive, ::rkyv::Serialize, ::rkyv::Deserialize, Debug, Clone)]
        #[rkyv(derive(Debug))]
    };

    // Compute the C-ABI/PyO3 surface from the original definition (it reads the
    // `#[schema(pod)]` field markers) before stripping those markers off the
    // re-emitted type — `schema` is an item attribute, so a leftover field-level
    // `#[schema(pod)]` would be an unknown attribute to the compiler.
    let extras = match &parsed.data {
        Data::Struct(s) => emit_struct(&parsed.ident, s),
        Data::Enum(e) => emit_enum(&parsed.ident, e),
        _ => syn::Error::new_spanned(&parsed.ident, "#[schema] only supports structs and enums")
            .to_compile_error(),
    };
    strip_pod_attrs(&mut parsed);

    quote! {
        #derive_block
        #parsed
        #extras
    }
    .into()
}

/// Remove `#[schema(pod)]` field markers from a parsed item so it re-emits as a
/// plain definition (the marker is consumed by classification, not the compiler).
fn strip_pod_attrs(input: &mut DeriveInput) {
    fn strip(attrs: &mut Vec<syn::Attribute>) {
        attrs.retain(|a| !a.path().is_ident("schema"));
    }
    match &mut input.data {
        Data::Struct(s) => {
            for f in s.fields.iter_mut() {
                strip(&mut f.attrs);
            }
        }
        Data::Enum(e) => {
            for v in e.variants.iter_mut() {
                for f in v.fields.iter_mut() {
                    strip(&mut f.attrs);
                }
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Naming
// ---------------------------------------------------------------------------

/// Convert CamelCase → snake_case. Underscore is inserted before an
/// uppercase letter only when the previous char is lowercase, or when
/// the previous is uppercase and the next is lowercase ("ABc" → "a_bc").
/// Digits never trigger an underscore on the right ("D2H" → "d2h",
/// "ZoInit" → "zo_init", "TopKTopP" → "top_k_top_p").
fn snake(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    let mut out = String::with_capacity(name.len() + 2);
    for i in 0..chars.len() {
        let c = chars[i];
        if c.is_uppercase() && i > 0 {
            let prev = chars[i - 1];
            let next = chars.get(i + 1).copied();
            let prev_is_lower = prev.is_lowercase();
            let prev_is_upper = prev.is_uppercase();
            let next_is_lower = next.map(|n| n.is_lowercase()).unwrap_or(false);
            if prev_is_lower || (prev_is_upper && next_is_lower) {
                out.push('_');
            }
        }
        if c.is_uppercase() {
            for lc in c.to_lowercase() {
                out.push(lc);
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn upper_snake(name: &str) -> String {
    snake(name).to_uppercase()
}

// ---------------------------------------------------------------------------
// Field type classification
// ---------------------------------------------------------------------------

enum FieldKind {
    /// `u32`, `u64`, `i32`, `i64`, `f32`.
    PrimScalar(Ident),
    /// `bool` — wire-serialized as itself, read out as `u8`.
    BoolScalar,
    /// `Vec<T>` for primitive `T`.
    SlicePrim(Ident),
    /// `Vec<bool>`.
    SliceBool,
    /// `Vec<NestedSchemaType>`.
    SliceNested(Ident),
    /// `String`.
    BareString,
    /// Nested `#[schema]` struct or enum (referenced by type name).
    Nested(Ident),
    /// A flat-POD field marked `#[schema(pod)]` — a `#[repr(C)]`/`#[repr(u8)]`
    /// type from `crate::pod`. Archived form == native, so it is embedded by
    /// value in the descriptor and read by cast (no Pie<T>Desc, no Py<T>).
    Pod(Ident),
    /// `Vec<T>` where `T` is a `#[schema(pod)]` flat-POD type. Stored as a
    /// `(ptr, len)` slice of the POD element type (no nested descriptor).
    PodSlice(Ident),
}

/// True when a field carries `#[schema(pod)]`, marking its type as a flat-POD
/// wire type (defined in `crate::pod`, not via `#[schema]`).
fn is_pod_field(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        if !a.path().is_ident("schema") {
            return false;
        }
        let mut found = false;
        // `#[schema(pod)]` — the sole field-level form.
        let _ = a.parse_nested_meta(|m| {
            if m.path.is_ident("pod") {
                found = true;
            }
            Ok(())
        });
        found
    })
}

/// Classify a field, honoring a `#[schema(pod)]` marker: a POD `Vec<T>` becomes
/// [`FieldKind::PodSlice`], a bare POD type becomes [`FieldKind::Pod`].
fn classify_field(field: &syn::Field) -> Result<FieldKind, syn::Error> {
    if is_pod_field(&field.attrs) {
        if let Some(inner) = strip_generic(&field.ty, "Vec") {
            let name = type_ident(inner).ok_or_else(|| {
                syn::Error::new_spanned(&field.ty, "#[schema(pod)] Vec<T> needs a named element")
            })?;
            return Ok(FieldKind::PodSlice(name));
        }
        let name = type_ident(&field.ty).ok_or_else(|| {
            syn::Error::new_spanned(&field.ty, "#[schema(pod)] needs a named type")
        })?;
        return Ok(FieldKind::Pod(name));
    }
    classify(&field.ty)
}

fn classify(ty: &Type) -> Result<FieldKind, syn::Error> {
    if let Some(inner) = strip_generic(ty, "Vec") {
        if let Some(elem) = primitive_ident(inner) {
            if elem == "bool" {
                return Ok(FieldKind::SliceBool);
            }
            return Ok(FieldKind::SlicePrim(elem));
        }
        if let Some(name) = type_ident(inner) {
            return Ok(FieldKind::SliceNested(name));
        }
        return Err(syn::Error::new_spanned(ty, "unsupported Vec<T>"));
    }
    if strip_generic(ty, "Option").is_some() {
        return Err(syn::Error::new_spanned(
            ty,
            "Option<T> is not supported in #[schema]; use a sentinel value (e.g. 0 or -1) \
             and document its meaning. The C header `pie_driver_abi.h` mirrors `Pie<T>Desc` \
             byte-for-byte, and adding has-flags forces a fragile field-order convention \
             between Rust and C.",
        ));
    }
    if is_string(ty) {
        return Ok(FieldKind::BareString);
    }
    if let Some(id) = primitive_ident(ty) {
        if id == "bool" {
            return Ok(FieldKind::BoolScalar);
        }
        return Ok(FieldKind::PrimScalar(id));
    }
    if let Some(name) = type_ident(ty) {
        return Ok(FieldKind::Nested(name));
    }
    Err(syn::Error::new_spanned(ty, "unsupported field type"))
}

fn primitive_ident(ty: &Type) -> Option<Ident> {
    let id = type_ident(ty)?;
    matches!(
        id.to_string().as_str(),
        "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" | "f32" | "f64" | "bool"
    )
    .then_some(id)
}

fn type_ident(ty: &Type) -> Option<Ident> {
    if let Type::Path(tp) = ty
        && tp.qself.is_none()
        && tp.path.segments.len() == 1
    {
        return Some(tp.path.segments[0].ident.clone());
    }
    None
}

fn strip_generic<'a>(ty: &'a Type, wrap: &str) -> Option<&'a Type> {
    let Type::Path(tp) = ty else { return None };
    if tp.qself.is_some() || tp.path.segments.len() != 1 {
        return None;
    }
    let seg = &tp.path.segments[0];
    if seg.ident != wrap {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    if args.args.len() != 1 {
        return None;
    }
    let GenericArgument::Type(inner) = &args.args[0] else {
        return None;
    };
    Some(inner)
}

fn is_string(ty: &Type) -> bool {
    matches!(type_ident(ty), Some(id) if id == "String")
}

// ---------------------------------------------------------------------------
// Struct emission
// ---------------------------------------------------------------------------

fn emit_struct(name: &Ident, data: &DataStruct) -> TS2 {
    let desc = format_ident!("Pie{}Desc", name);

    let fields: Vec<(Ident, &syn::Field)> = match &data.fields {
        Fields::Named(fs) => fs
            .named
            .iter()
            .filter_map(|f| f.ident.clone().map(|n| (n, f)))
            .collect(),
        _ => {
            return syn::Error::new_spanned(name, "#[schema] struct must have named fields")
                .to_compile_error();
        }
    };

    let mut desc_fields: Vec<TS2> = Vec::new();
    let mut build_extracts: Vec<TS2> = Vec::new();
    let mut struct_inits: Vec<TS2> = Vec::new();
    let mut kinds: Vec<(Ident, FieldKind)> = Vec::new();

    for (fname, field) in &fields {
        let kind = match classify_field(field) {
            Ok(k) => k,
            Err(e) => return e.to_compile_error(),
        };
        desc_fields.push(emit_desc_field(fname, &kind));
        let (extract, init) = emit_build_extract(fname, &kind);
        build_extracts.push(extract);
        struct_inits.push(init);
        kinds.push((fname.clone(), kind));
    }

    // Python-free protocol layer (#13): the only surface is the in-proc
    // Rust↔C++ Desc — a `repr(C) Pie<T>Desc` (ptr+len per Vec), the zero-copy
    // `ToDesc` (`T::as_desc`), and `FromDesc` (`T::from_desc`). The rkyv
    // accessors (`pie_parse_*`/`pie_build_*`/readers) and the PyO3 wrappers are
    // gone — C++ reads Descs directly and out-of-proc is Rust↔Rust rkyv.
    let desc_block = emit_desc_struct(&desc, &desc_fields);
    let view_block = emit_struct_view(name, &desc, &kinds);
    let from_desc_block = emit_struct_from_desc(name, &desc, &build_extracts, &struct_inits);

    quote! {
        #desc_block
        #view_block
        #from_desc_block
    }
}

// ---------------------------------------------------------------------------
// Descriptor + builder emission
// ---------------------------------------------------------------------------

fn emit_desc_struct(desc: &Ident, fields: &[TS2]) -> TS2 {
    // Desc structs are POD: raw pointers, primitives, embedded sub-Descs.
    // Copy + Clone make them cheap to embed by value (parent enum descs
    // copy variant sub-descs); Default zero-initializes inactive enum
    // variant slots. Zero is a valid bit pattern for every field type
    // (null ptr, 0 len, 0 scalar).
    quote! {
        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct #desc {
            #(#fields,)*
        }

        impl Default for #desc {
            #[inline]
            fn default() -> Self {
                // SAFETY: every field is POD with a valid all-zero bit
                // pattern (null ptr, 0 len, 0 scalar, sub-Desc whose
                // fields are themselves zero-valid).
                unsafe { ::core::mem::zeroed() }
            }
        }
    }
}

fn emit_desc_field(field: &Ident, kind: &FieldKind) -> TS2 {
    let ptr_name = format_ident!("{}_ptr", field);
    let len_name = format_ident!("{}_len", field);
    match kind {
        FieldKind::BoolScalar => quote!(pub #field: u8),
        FieldKind::PrimScalar(t) => quote!(pub #field: #t),
        FieldKind::SlicePrim(elem) => quote! {
            pub #ptr_name: *const #elem,
            pub #len_name: usize
        },
        FieldKind::SliceBool => quote! {
            pub #ptr_name: *const u8,
            pub #len_name: usize
        },
        FieldKind::SliceNested(name) => {
            let nested_desc = format_ident!("Pie{}Desc", name);
            quote! {
                pub #ptr_name: *const #nested_desc,
                pub #len_name: usize
            }
        }
        FieldKind::BareString => quote! {
            pub #ptr_name: *const u8,
            pub #len_name: usize
        },
        FieldKind::Nested(name) => {
            let nested_desc = format_ident!("Pie{}Desc", name);
            quote!(pub #field: #nested_desc)
        }
        // Flat-POD: embed the concrete `#[repr(C)]`/`#[repr(u8)]` type by value
        // (it IS its own descriptor — no Pie<T>Desc).
        FieldKind::Pod(name) => quote!(pub #field: #name),
        FieldKind::PodSlice(elem) => quote! {
            pub #ptr_name: *const #elem,
            pub #len_name: usize
        },
    }
}

/// Emit the per-field tokens used inside `pie_build_<type>` to extract a
/// native Rust value from the C descriptor. Returns `(extract, init)` —
/// `extract` declares a `let f = ...` binding; `init` is the
/// `f: <expr>,` line inside the native struct literal.
fn emit_build_extract(field: &Ident, kind: &FieldKind) -> (TS2, TS2) {
    let ptr_name = format_ident!("{}_ptr", field);
    let len_name = format_ident!("{}_len", field);
    let binding = format_ident!("__f_{}", field);
    match kind {
        FieldKind::BoolScalar => {
            let e = quote! { let #binding = d.#field != 0; };
            (e, quote! { #field: #binding })
        }
        FieldKind::PrimScalar(_) => {
            let e = quote! { let #binding = d.#field; };
            (e, quote! { #field: #binding })
        }
        FieldKind::SlicePrim(elem) => {
            let e = quote! {
                let #binding: ::std::vec::Vec<#elem> = unsafe {
                    if d.#ptr_name.is_null() || d.#len_name == 0 {
                        ::std::vec::Vec::new()
                    } else {
                        ::core::slice::from_raw_parts(d.#ptr_name, d.#len_name).to_vec()
                    }
                };
            };
            (e, quote! { #field: #binding })
        }
        FieldKind::SliceBool => {
            let e = quote! {
                let #binding: ::std::vec::Vec<bool> = unsafe {
                    if d.#ptr_name.is_null() || d.#len_name == 0 {
                        ::std::vec::Vec::new()
                    } else {
                        ::core::slice::from_raw_parts(d.#ptr_name, d.#len_name)
                            .iter()
                            .map(|b| *b != 0)
                            .collect()
                    }
                };
            };
            (e, quote! { #field: #binding })
        }
        FieldKind::SliceNested(name) => {
            let from_fn = format_ident!("__pie_{}_from_desc", snake(&name.to_string()));
            let e = quote! {
                let #binding: ::std::vec::Vec<#name> = unsafe {
                    if d.#ptr_name.is_null() || d.#len_name == 0 {
                        ::std::vec::Vec::new()
                    } else {
                        (0..d.#len_name)
                            .map(|i| #from_fn(&*d.#ptr_name.add(i)))
                            .collect()
                    }
                };
            };
            (e, quote! { #field: #binding })
        }
        FieldKind::BareString => {
            let e = quote! {
                let #binding: String = unsafe {
                    if d.#ptr_name.is_null() || d.#len_name == 0 {
                        String::new()
                    } else {
                        let bytes = ::core::slice::from_raw_parts(d.#ptr_name, d.#len_name);
                        String::from_utf8_lossy(bytes).into_owned()
                    }
                };
            };
            (e, quote! { #field: #binding })
        }
        FieldKind::Nested(name) => {
            let from_fn = format_ident!("__pie_{}_from_desc", snake(&name.to_string()));
            let e = quote! {
                let #binding: #name = #from_fn(&d.#field);
            };
            (e, quote! { #field: #binding })
        }
        // Flat-POD: the descriptor holds the value directly — just copy it.
        FieldKind::Pod(_) => {
            let e = quote! { let #binding = d.#field; };
            (e, quote! { #field: #binding })
        }
        FieldKind::PodSlice(elem) => {
            let e = quote! {
                let #binding: ::std::vec::Vec<#elem> = unsafe {
                    if d.#ptr_name.is_null() || d.#len_name == 0 {
                        ::std::vec::Vec::new()
                    } else {
                        ::core::slice::from_raw_parts(d.#ptr_name, d.#len_name).to_vec()
                    }
                };
            };
            (e, quote! { #field: #binding })
        }
    }
}

/// Emit `FromDesc` for a struct: `T::from_desc(&Pie<T>Desc) -> T`, the inverse
/// of `as_desc`, used on the in-proc response path (C++ writes the Desc, the
/// runtime converts it to the native type). The internal `__pie_<t>_from_desc`
/// free fn carries the recursion (nested types call it); the public method
/// wraps it.
fn emit_struct_from_desc(name: &Ident, desc: &Ident, extracts: &[TS2], inits: &[TS2]) -> TS2 {
    let from_fn = format_ident!("__pie_{}_from_desc", snake(&name.to_string()));
    quote! {
        #[doc(hidden)]
        pub fn #from_fn(d: &#desc) -> #name {
            unsafe {
                #(#extracts)*
                #name { #(#inits,)* }
            }
        }

        impl #name {
            /// Reconstruct a native `#name` from its in-proc descriptor.
            #[inline]
            pub fn from_desc(d: &#desc) -> #name {
                #from_fn(d)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Direct-FFI view emission — TODO
//
// Goal: `pie_<type>_view(&native) -> Pie<T>View<'_>` that builds a Desc
// whose slice pointers alias native's heap, materializing temporary
// `Vec<NestedDesc>` holders only where needed. Skips rkyv archive
// entirely — the in-process FFI path can pass `&view.desc` directly.
//
// Open design question: enum view return-types are heterogeneous
// (unit enums → u8; data enums with newtype variants → view with
// holder enum; data enums with only inline-struct variants → bare
// Desc). The macro needs a uniform interface across struct fields
// classified as `Nested(T)` without semantic info about T.
//
// Likely shape:
//   - Emit Pie<T>View<'a> for every #[schema] type (uniform).
//   - For unit enums, the view is `pub struct PieEView<'a> { pub desc: u8, _m: PhantomData<&'a E> }`.
//   - For data enums, an internal holder enum keeps active-variant nested views alive.
//   - Struct fields of type T always call `pie_<t>_view(&native.f)` and extract `view.desc`.
//
// Deferred until the FFI vtable migration is scoped — there's no
// caller for these views yet.
// ---------------------------------------------------------------------------

/// Direct-FFI view emission for structs. Every nested-type field —
/// whether the inner type is a struct, a data enum, or a unit enum —
/// uses the same uniform protocol: call `pie_<t>_view(&native.f)`,
/// grab `.desc` (which is `Copy`), and hold the returned view in our
/// holder so its allocations stay alive.
///
/// `Vec<T>` for any `#[schema]` `T` materializes a `Vec<Pie<T>Desc>`
/// (cheap copy since `Pie<T>Desc: Copy`) whose pointer feeds the
/// parent desc's `ptr/len` pair; the underlying `Vec<Pie<T>View<'a>>`
/// stays in the holder.
fn emit_struct_view(name: &Ident, desc: &Ident, kinds: &[(Ident, FieldKind)]) -> TS2 {
    let prefix = snake(&name.to_string());
    let view_ty = format_ident!("Pie{}View", name);
    let view_fn = format_ident!("pie_{}_view", prefix);

    let mut holder_decls: Vec<TS2> = Vec::new();
    let mut holder_names: Vec<Ident> = Vec::new();
    let mut holder_inits: Vec<TS2> = Vec::new();
    let mut desc_inits: Vec<TS2> = Vec::new();

    for (fname, kind) in kinds {
        let ptr_name = format_ident!("{}_ptr", fname);
        let len_name = format_ident!("{}_len", fname);
        let view_holder = format_ident!("__h_view_{}", fname);
        let desc_holder = format_ident!("__h_desc_{}", fname);

        match kind {
            FieldKind::BoolScalar => {
                desc_inits.push(quote! { #fname: ::core::primitive::u8::from(native.#fname) });
            }
            FieldKind::PrimScalar(_) => {
                desc_inits.push(quote! { #fname: native.#fname });
            }
            FieldKind::SlicePrim(elem) => {
                desc_inits.push(quote! {
                    #ptr_name: native.#fname.as_ptr() as *const #elem,
                    #len_name: native.#fname.len()
                });
            }
            FieldKind::SliceBool => {
                desc_inits.push(quote! {
                    #ptr_name: native.#fname.as_ptr() as *const u8,
                    #len_name: native.#fname.len()
                });
            }
            FieldKind::SliceNested(nested) => {
                let nested_view_ty = format_ident!("Pie{}View", nested);
                let nested_desc_ty = format_ident!("Pie{}Desc", nested);
                let nested_view_fn = format_ident!("pie_{}_view", snake(&nested.to_string()));
                holder_decls.push(quote! {
                    #[allow(dead_code)]
                    #view_holder: ::std::vec::Vec<#nested_view_ty<'a>>
                });
                holder_decls.push(quote! {
                    #[allow(dead_code)]
                    #desc_holder: ::std::vec::Vec<#nested_desc_ty>
                });
                holder_names.push(view_holder.clone());
                holder_names.push(desc_holder.clone());
                holder_inits.push(quote! {
                    let #view_holder: ::std::vec::Vec<#nested_view_ty<'a>> =
                        native.#fname.iter().map(#nested_view_fn).collect();
                    let #desc_holder: ::std::vec::Vec<#nested_desc_ty> =
                        #view_holder.iter().map(|v| v.desc).collect();
                });
                desc_inits.push(quote! {
                    #ptr_name: #desc_holder.as_ptr(),
                    #len_name: #desc_holder.len()
                });
            }
            FieldKind::BareString => {
                desc_inits.push(quote! {
                    #ptr_name: native.#fname.as_ptr() as *const u8,
                    #len_name: native.#fname.len()
                });
            }
            FieldKind::Nested(nested) => {
                let nested_view_ty = format_ident!("Pie{}View", nested);
                let nested_view_fn = format_ident!("pie_{}_view", snake(&nested.to_string()));
                holder_decls.push(quote! {
                    #[allow(dead_code)]
                    #view_holder: #nested_view_ty<'a>
                });
                holder_names.push(view_holder.clone());
                holder_inits.push(quote! {
                    let #view_holder = #nested_view_fn(&native.#fname);
                });
                desc_inits.push(quote! {
                    #fname: #view_holder.desc
                });
            }
            // Flat-POD: copy the value into the desc; the slice form aliases
            // native's heap directly (the element type IS the C type).
            FieldKind::Pod(_) => {
                desc_inits.push(quote! { #fname: native.#fname });
            }
            FieldKind::PodSlice(elem) => {
                desc_inits.push(quote! {
                    #ptr_name: native.#fname.as_ptr() as *const #elem,
                    #len_name: native.#fname.len()
                });
            }
        }
    }

    quote! {
        /// Direct-FFI view of [`#name`]: builds a `Pie<T>Desc` aliasing
        /// `native`'s data. Holders keep any nested-view allocations
        /// alive for the view's lifetime. Pointers in `desc` are
        /// invalid once the view drops.
        pub struct #view_ty<'a> {
            pub desc: #desc,
            #(#holder_decls,)*
            _marker: ::core::marker::PhantomData<&'a #name>,
        }

        /// Build the view from a borrowed native value. Zero rkyv
        /// serialization; suitable for in-process FFI handoff. Internal
        /// recursion helper; callers use [`#name::as_desc`].
        #[doc(hidden)]
        pub fn #view_fn<'a>(native: &'a #name) -> #view_ty<'a> {
            #(#holder_inits)*
            let desc = #desc {
                #(#desc_inits,)*
            };
            #view_ty {
                desc,
                #(#holder_names,)*
                _marker: ::core::marker::PhantomData,
            }
        }

        impl #name {
            /// Zero-copy in-proc descriptor view: a `Pie<T>Desc` whose `Vec`
            /// fields alias this value's memory (the runtime hands `&v.desc`
            /// to the C++ driver). Valid only while the returned view lives.
            #[inline]
            pub fn as_desc<'a>(&'a self) -> #view_ty<'a> {
                #view_fn(self)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Enum emission
// ---------------------------------------------------------------------------

enum VariantShape<'a> {
    Unit,
    /// `Variant(Inner)`. The bool is `true` when the inner is `#[schema(pod)]`
    /// (a flat-POD type embedded by value, not a nested `#[schema]` type).
    Newtype(&'a Type, bool),
}

fn classify_variant(v: &Variant) -> Result<VariantShape<'_>, syn::Error> {
    match &v.fields {
        Fields::Unit => Ok(VariantShape::Unit),
        Fields::Unnamed(f) if f.unnamed.len() == 1 => Ok(VariantShape::Newtype(
            &f.unnamed[0].ty,
            is_pod_field(&f.unnamed[0].attrs),
        )),
        // Named-field (inline-struct) variants are no longer supported: the only
        // such enum (`Sampler`) was flattened to primitive SoA arrays, so the
        // data-enum surface is unit + single-field-newtype only.
        _ => Err(syn::Error::new_spanned(
            v,
            "#[schema] enum variants must be unit or single-field newtype \
             (named-field / inline-struct variants are no longer supported — \
             flatten them to parallel primitive fields)",
        )),
    }
}

fn emit_enum(name: &Ident, data: &DataEnum) -> TS2 {
    let prefix = snake(&name.to_string());

    let mut shapes: Vec<(Ident, VariantShape)> = Vec::new();
    for v in &data.variants {
        match classify_variant(v) {
            Ok(s) => shapes.push((v.ident.clone(), s)),
            Err(e) => return e.to_compile_error(),
        }
    }
    // A fieldless (all-unit) enum is a flat-POD wire type: `#[repr(u8)]` + rkyv
    // in `crate::pod`, read by cast and referenced via `#[schema(pod)]`. The
    // full reader/descriptor/PyO3 surface is only for the rich data enums, so
    // `#[schema]` no longer accepts an all-unit enum.
    if shapes.iter().all(|(_, s)| matches!(s, VariantShape::Unit)) {
        return syn::Error::new_spanned(
            name,
            "#[schema] no longer supports all-unit enums — make it a plain \
             `#[repr(u8)]` + `#[derive(rkyv::Archive, ...)]` enum in `crate::pod` \
             and reference it from rich types with `#[schema(pod)]`",
        )
        .to_compile_error();
    }
    emit_data_enum(name, &prefix, &shapes)
}

fn emit_data_enum(name: &Ident, prefix: &str, shapes: &[(Ident, VariantShape)]) -> TS2 {
    let from_fn = format_ident!("__pie_{}_from_desc", prefix);
    let desc = format_ident!("Pie{}Desc", name);

    // Descriptor: u8 kind + one embedded field per newtype variant. (Data enums
    // are unit + single-field-newtype only; the old inline-struct field union is
    // gone.) The generated `pie_driver_abi.h` mirrors this order byte-for-byte;
    // `tests/desc_layout.rs` pins exact offsets.
    let mut desc_fields: Vec<TS2> = vec![quote!(pub kind: u8)];
    for (vname, shape) in shapes {
        match shape {
            VariantShape::Newtype(ty, is_pod) => {
                let Some(inner_name) = type_ident(ty) else {
                    continue;
                };
                let fname = format_ident!("{}", snake(&vname.to_string()));
                // POD inner is embedded by value (it IS its own descriptor);
                // rich inner embeds the generated Pie<Inner>Desc.
                if *is_pod {
                    desc_fields.push(quote!(pub #fname: #inner_name));
                } else {
                    let nested_desc = format_ident!("Pie{}Desc", inner_name);
                    desc_fields.push(quote!(pub #fname: #nested_desc));
                }
            }
            VariantShape::Unit => {}
        }
    }

    // from_desc: dispatch on kind, construct variant.
    let mut from_arms: Vec<TS2> = Vec::new();
    for (i, (vname, shape)) in shapes.iter().enumerate() {
        let i = i as u8;
        let arm = match shape {
            VariantShape::Unit => quote! { #i => #name::#vname, },
            VariantShape::Newtype(ty, is_pod) => {
                let Some(inner_name) = type_ident(ty) else {
                    return syn::Error::new_spanned(
                        name,
                        "newtype variant must contain a named type",
                    )
                    .to_compile_error();
                };
                let fname = format_ident!("{}", snake(&vname.to_string()));
                // POD inner: the descriptor holds the value — copy it.
                if *is_pod {
                    quote! { #i => #name::#vname(d.#fname), }
                } else {
                    let from_inner =
                        format_ident!("__pie_{}_from_desc", snake(&inner_name.to_string()));
                    quote! { #i => #name::#vname(#from_inner(&d.#fname)), }
                }
            }
        };
        from_arms.push(arm);
    }
    let first_variant = &shapes[0].0;
    let first_default = match &shapes[0].1 {
        VariantShape::Unit => quote!(#name::#first_variant),
        _ => quote!(panic!("invalid descriptor kind")),
    };

    let view_block = emit_data_enum_view(name, prefix, &desc, shapes);

    // `PIE_<ENUM>_<VARIANT>` discriminant consts: C++ reads `desc.kind` (u8)
    // and compares against these (cbindgen emits them into the header).
    let consts: Vec<TS2> = shapes
        .iter()
        .enumerate()
        .map(|(i, (v, _))| {
            let cname = format_ident!(
                "PIE_{}_{}",
                upper_snake(&name.to_string()),
                upper_snake(&v.to_string())
            );
            let i = i as u8;
            quote! { pub const #cname: u8 = #i; }
        })
        .collect();

    quote! {
        #(#consts)*

        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct #desc {
            #(#desc_fields,)*
        }

        impl Default for #desc {
            #[inline]
            fn default() -> Self {
                // SAFETY: every field is POD with valid all-zero bit
                // pattern (null ptr, 0 len, 0 scalar, sub-Desc whose
                // fields are zero-valid transitively).
                unsafe { ::core::mem::zeroed() }
            }
        }

        #[doc(hidden)]
        pub fn #from_fn(d: &#desc) -> #name {
            match d.kind {
                #(#from_arms)*
                _ => #first_default,
            }
        }

        impl #name {
            /// Reconstruct a native `#name` from its in-proc descriptor.
            #[inline]
            pub fn from_desc(d: &#desc) -> #name {
                #from_fn(d)
            }
        }

        #view_block
    }
}

/// Emit `Pie<E>View<'a>` + `pie_<e>_view` for a data enum. The view
/// holds a tagged holder enum so that newtype-variant nested views
/// (with their own holder allocations) stay alive as long as the
/// parent view does. For inline-struct / unit variants no holder is
/// needed — the holder enum's `None` arm covers them.
fn emit_data_enum_view(
    name: &Ident,
    prefix: &str,
    desc: &Ident,
    shapes: &[(Ident, VariantShape)],
) -> TS2 {
    let view_ty = format_ident!("Pie{}View", name);
    let view_fn = format_ident!("pie_{}_view", prefix);
    let holder_enum = format_ident!("__Pie{}Holder", name);

    // Holder-enum variants: one per *rich* newtype variant (its sub-view owns
    // allocations that must outlive the parent), plus a catch-all `None`. POD
    // newtype variants need no holder — they are copied into the desc by value.
    let mut holder_variants: Vec<TS2> = Vec::new();
    let mut needs_lifetime = false;
    for (vname, shape) in shapes {
        if let VariantShape::Newtype(ty, is_pod) = shape {
            if *is_pod {
                continue;
            }
            let Some(inner) = type_ident(ty) else {
                continue;
            };
            let inner_view = format_ident!("Pie{}View", inner);
            holder_variants.push(quote! { #vname(#inner_view<'a>) });
            needs_lifetime = true;
        }
    }
    holder_variants.push(quote! { None });

    // Build the per-variant arms of the view fn: fill the discriminant
    // byte, copy the active variant's nested desc into the embedded
    // sub-desc field (other variants stay Default::default()), and
    // record the holder.
    let mut view_arms: Vec<TS2> = Vec::new();
    for (i, (vname, shape)) in shapes.iter().enumerate() {
        let kind = i as u8;
        let arm = match shape {
            VariantShape::Unit => quote! {
                #name::#vname => {
                    let mut d = <#desc as ::core::default::Default>::default();
                    d.kind = #kind;
                    (d, #holder_enum::None)
                }
            },
            VariantShape::Newtype(ty, is_pod) => {
                let Some(inner) = type_ident(ty) else {
                    return syn::Error::new_spanned(
                        name,
                        "newtype variant must contain a named type",
                    )
                    .to_compile_error();
                };
                let fname = format_ident!("{}", snake(&vname.to_string()));
                if *is_pod {
                    // POD inner: copy the value into the desc, no holder needed.
                    quote! {
                        #name::#vname(inner) => {
                            let mut d = <#desc as ::core::default::Default>::default();
                            d.kind = #kind;
                            d.#fname = *inner;
                            (d, #holder_enum::None)
                        }
                    }
                } else {
                    let inner_view_fn = format_ident!("pie_{}_view", snake(&inner.to_string()));
                    quote! {
                        #name::#vname(inner) => {
                            let h = #inner_view_fn(inner);
                            let mut d = <#desc as ::core::default::Default>::default();
                            d.kind = #kind;
                            d.#fname = h.desc;
                            (d, #holder_enum::#vname(h))
                        }
                    }
                }
            }
        };
        view_arms.push(arm);
    }

    let _holder_needs_lifetime = needs_lifetime;
    let lifetime_decl = quote!(<'a>); // always parameterize for uniform call sites

    quote! {
        #[allow(non_camel_case_types, dead_code)]
        enum #holder_enum #lifetime_decl {
            #(#holder_variants,)*
            _Unused(::core::marker::PhantomData<&'a ()>),
        }

        /// Direct-FFI view of a data enum. `desc` carries the
        /// discriminant + active variant's sub-desc; the internal
        /// holder keeps any nested view allocations alive.
        pub struct #view_ty<'a> {
            pub desc: #desc,
            #[allow(dead_code)]
            __h: #holder_enum<'a>,
            _marker: ::core::marker::PhantomData<&'a #name>,
        }

        /// Build the data-enum view from a borrowed native value.
        /// Internal recursion helper; callers use [`#name::as_desc`].
        #[doc(hidden)]
        pub fn #view_fn<'a>(native: &'a #name) -> #view_ty<'a> {
            let (desc, __h) = match native {
                #(#view_arms)*
            };
            #view_ty { desc, __h, _marker: ::core::marker::PhantomData }
        }

        impl #name {
            /// Zero-copy in-proc descriptor view (see [`#name`]). Valid only
            /// while the returned view lives.
            #[inline]
            pub fn as_desc<'a>(&'a self) -> #view_ty<'a> {
                #view_fn(self)
            }
        }
    }
}
