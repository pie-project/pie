//! pie-bridge schema macro.
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
    Data, DataEnum, DataStruct, DeriveInput, Fields, GenericArgument, Ident, PathArguments, Token,
    Type, Variant, parse::Parser, punctuated::Punctuated,
};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[proc_macro_attribute]
pub fn schema(_args: TokenStream, input: TokenStream) -> TokenStream {
    let parsed = match syn::parse::<DeriveInput>(input) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };

    let derive_block = quote! {
        #[derive(::rkyv::Archive, ::rkyv::Serialize, ::rkyv::Deserialize, Debug, Clone)]
        #[rkyv(derive(Debug))]
    };

    let extras = match &parsed.data {
        Data::Struct(s) => emit_struct(&parsed.ident, s),
        Data::Enum(e) => emit_enum(&parsed.ident, e),
        _ => syn::Error::new_spanned(&parsed.ident, "#[schema] only supports structs and enums")
            .to_compile_error(),
    };

    quote! {
        #derive_block
        #parsed
        #extras
    }
    .into()
}

// ---------------------------------------------------------------------------
// schema_module!(T1, T2, ...) — emit `pub fn register_schema_types(m)`
// that calls each `register_<t>(m)?;`. The caller writes the
// `#[pymodule]` wrapper and can add extra (non-schema) classes
// alongside the schema registrations.
// ---------------------------------------------------------------------------

#[proc_macro]
pub fn schema_module(input: TokenStream) -> TokenStream {
    let parser = Punctuated::<Ident, Token![,]>::parse_terminated;
    let parsed = match parser.parse(input) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };
    let calls: Vec<TS2> = parsed
        .iter()
        .map(|name| {
            let mod_name = format_ident!("__py_{}", snake(&name.to_string()));
            let reg = format_ident!("register_{}", snake(&name.to_string()));
            quote! { crate::schema::#mod_name::#reg(m)?; }
        })
        .collect();
    quote! {
        /// Register every `#[schema]` type's `Py<T>` class on the
        /// given module. Call from your own `#[pymodule]`.
        pub fn register_schema_types(
            m: &::pyo3::Bound<'_, ::pyo3::types::PyModule>,
        ) -> ::pyo3::PyResult<()> {
            #(#calls)*
            Ok(())
        }
    }
    .into()
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
             and document its meaning. The C header `pie_bridge.h` mirrors `Pie<T>Desc` \
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
    let prefix = snake(&name.to_string());
    let archived = format_ident!("Archived{}", name);
    let desc = format_ident!("Pie{}Desc", name);
    let build_fn = format_ident!("pie_build_{}", prefix);
    let size_fn = format_ident!("pie_size_{}", prefix);
    let parse_fn = format_ident!("pie_parse_{}", prefix);

    let fields: Vec<(Ident, &Type)> = match &data.fields {
        Fields::Named(fs) => fs
            .named
            .iter()
            .filter_map(|f| f.ident.clone().map(|n| (n, &f.ty)))
            .collect(),
        _ => {
            return syn::Error::new_spanned(name, "#[schema] struct must have named fields")
                .to_compile_error();
        }
    };

    let mut readers: Vec<TS2> = Vec::new();
    let mut desc_fields: Vec<TS2> = Vec::new();
    let mut build_extracts: Vec<TS2> = Vec::new();
    let mut struct_inits: Vec<TS2> = Vec::new();
    let mut py_methods: Vec<TS2> = Vec::new();
    let mut kinds: Vec<(Ident, FieldKind)> = Vec::new();

    for (fname, ty) in &fields {
        let kind = match classify(ty) {
            Ok(k) => k,
            Err(e) => return e.to_compile_error(),
        };
        readers.push(emit_struct_reader(&archived, &prefix, fname, &kind));
        desc_fields.push(emit_desc_field(fname, &kind));
        let (extract, init) = emit_build_extract(fname, &kind);
        build_extracts.push(extract);
        struct_inits.push(init);
        py_methods.push(emit_py_method(&prefix, fname, &kind));
        kinds.push((fname.clone(), kind));
    }

    let parse_block = emit_parse_fn(name, &parse_fn, &archived);
    let build_block = emit_struct_builder(
        name,
        &desc,
        &build_fn,
        &size_fn,
        &build_extracts,
        &struct_inits,
    );
    let desc_block = emit_desc_struct(&desc, &desc_fields);
    let view_block = emit_struct_view(name, &desc, &kinds);
    let pyclass = emit_pyclass_struct(name, &archived, &parse_fn, &py_methods);

    quote! {
        #(#readers)*
        #parse_block
        #desc_block
        #build_block
        #view_block
        #pyclass
    }
}

fn emit_struct_reader(archived: &Ident, prefix: &str, field: &Ident, kind: &FieldKind) -> TS2 {
    let fn_base = format_ident!("pie_{}_{}", prefix, field);
    match kind {
        FieldKind::BoolScalar => quote! {
            #[cfg(feature = "cabi")]
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #fn_base(owner: *const #archived) -> u8 {
                ::core::primitive::u8::from(unsafe { (*owner).#field })
            }
        },
        FieldKind::PrimScalar(t) => quote! {
            #[cfg(feature = "cabi")]
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #fn_base(owner: *const #archived) -> #t {
                unsafe { (*owner).#field }.into()
            }
        },
        FieldKind::SlicePrim(elem) => quote! {
            #[cfg(feature = "cabi")]
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #fn_base(
                owner: *const #archived,
                out_ptr: *mut *const #elem,
                out_len: *mut usize,
            ) {
                let v = unsafe { &(*owner).#field };
                unsafe {
                    *out_ptr = v.as_ptr() as *const #elem;
                    *out_len = v.len();
                }
            }
        },
        FieldKind::SliceBool => quote! {
            #[cfg(feature = "cabi")]
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #fn_base(
                owner: *const #archived,
                out_ptr: *mut *const u8,
                out_len: *mut usize,
            ) {
                let v = unsafe { &(*owner).#field };
                unsafe {
                    *out_ptr = v.as_ptr() as *const u8;
                    *out_len = v.len();
                }
            }
        },
        FieldKind::SliceNested(name) => {
            let elem_archived = format_ident!("Archived{}", name);
            let len_fn = format_ident!("pie_{}_{}_len", prefix, field);
            let at_fn = format_ident!("pie_{}_{}_at", prefix, field);
            quote! {
                #[cfg(feature = "cabi")]
                #[unsafe(no_mangle)]
                pub unsafe extern "C" fn #len_fn(owner: *const #archived) -> usize {
                    unsafe { (*owner).#field.len() }
                }
                #[cfg(feature = "cabi")]
                #[unsafe(no_mangle)]
                pub unsafe extern "C" fn #at_fn(
                    owner: *const #archived,
                    i: usize,
                ) -> *const #elem_archived {
                    let v = unsafe { &(*owner).#field };
                    if i < v.len() { &v[i] as *const _ } else { ::core::ptr::null() }
                }
            }
        }
        FieldKind::BareString => quote! {
            #[cfg(feature = "cabi")]
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #fn_base(
                owner: *const #archived,
                out_ptr: *mut *const ::core::ffi::c_char,
                out_len: *mut usize,
            ) {
                let s = unsafe { &(*owner).#field };
                unsafe {
                    *out_ptr = s.as_ptr() as *const ::core::ffi::c_char;
                    *out_len = s.len();
                }
            }
        },
        FieldKind::Nested(name) => {
            let inner_archived = format_ident!("Archived{}", name);
            quote! {
                #[cfg(feature = "cabi")]
                #[unsafe(no_mangle)]
                pub unsafe extern "C" fn #fn_base(owner: *const #archived) -> *const #inner_archived {
                    unsafe { &(*owner).#field as *const _ }
                }
            }
        }
    }
}

fn emit_parse_fn(name: &Ident, parse_fn: &Ident, archived: &Ident) -> TS2 {
    let _ = name;
    // # Safety
    //
    // The emitted `pie_parse_<type>` uses `rkyv::access_unchecked` —
    // it does no validation. This is sound for pie because every
    // producer of these bytes (the runtime, the macro's builder, the
    // shmem ring) is *the same crate version*: bytes are either
    // produced in-process and handed across an FFI vtable, or
    // exchanged through `pie_bridge::ipc::ShmemServer` between
    // processes that already agreed on `SCHEMA_HASH` at handshake. A
    // mismatched producer fails the schema-hash check before any
    // payload is sent.
    //
    // If you ever need to parse rkyv bytes from a *partially trusted*
    // source (e.g. a user-supplied file or network input), use
    // `pie_bridge::wire::parse_request` / `parse_response` instead —
    // those use the checked variant and return a `WireError` on bad
    // bytes.
    quote! {
        #[cfg(feature = "cabi")]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #parse_fn(bytes: *const u8, len: usize) -> *const #archived {
            if bytes.is_null() || len == 0 {
                return ::core::ptr::null();
            }
            let slice = unsafe { ::core::slice::from_raw_parts(bytes, len) };
            // SAFETY: see the module-level note above — trusted producer.
            unsafe { ::rkyv::access_unchecked::<#archived>(slice) as *const #archived }
        }
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
        #[cfg(feature = "cabi")]
        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct #desc {
            #(#fields,)*
        }

        #[cfg(feature = "cabi")]
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
    }
}

fn emit_struct_builder(
    name: &Ident,
    desc: &Ident,
    build_fn: &Ident,
    size_fn: &Ident,
    extracts: &[TS2],
    inits: &[TS2],
) -> TS2 {
    let from_fn = format_ident!("__pie_{}_from_desc", snake(&name.to_string()));
    // Tier 1: write directly into out_buf via rkyv's `Buffer` writer.
    // Eliminates rkyv's internal `AlignedVec` allocation and the trailing
    // memcpy that the old `to_bytes` + `copy_nonoverlapping` path needed.
    // Alignment requirement on `out_buf` is unchanged — it must be
    // aligned for `pie_parse_<type>` to read the result back regardless.
    //
    // Sizing API: `pie_size_<type>(desc)` returns the encoded byte
    // count *without* writing — useful when a caller wants to size an
    // output buffer (or a shmem slot) before allocating. Internally it
    // serializes through a discard writer; the cost is one full rkyv
    // serialize-and-throw-away. Use only when you actually need the
    // size up front — for the common case where the caller already
    // has a sufficiently-large buffer, call `pie_build_<type>`
    // directly and check the return value.
    quote! {
        #[cfg(feature = "cabi")]
        #[doc(hidden)]
        pub fn #from_fn(d: &#desc) -> #name {
            unsafe {
                #(#extracts)*
                #name { #(#inits,)* }
            }
        }

        #[cfg(feature = "cabi")]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #build_fn(
            desc_ptr: *const #desc,
            out_buf: *mut u8,
            out_buf_cap: usize,
        ) -> usize {
            if desc_ptr.is_null() || out_buf.is_null() {
                return 0;
            }
            let d = unsafe { &*desc_ptr };
            let native = #from_fn(d);
            let slice = unsafe { ::core::slice::from_raw_parts_mut(out_buf, out_buf_cap) };
            let writer = ::rkyv::ser::writer::Buffer::from(slice);
            match ::rkyv::api::high::to_bytes_in::<_, ::rkyv::rancor::Error>(&native, writer) {
                Ok(buf) => buf.len(),
                Err(_) => 0,
            }
        }

        /// Returns the encoded size in bytes for the given descriptor,
        /// or 0 on serialization failure. Useful for sizing a target
        /// buffer before calling `pie_build_<type>`. Performs a full
        /// serialize-and-discard internally — prefer to call the
        /// builder directly when you already have a sufficient buffer.
        #[cfg(feature = "cabi")]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #size_fn(desc_ptr: *const #desc) -> usize {
            if desc_ptr.is_null() {
                return 0;
            }
            let d = unsafe { &*desc_ptr };
            let native = #from_fn(d);
            match ::rkyv::to_bytes::<::rkyv::rancor::Error>(&native) {
                Ok(buf) => buf.len(),
                Err(_) => 0,
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
        }
    }

    quote! {
        /// Direct-FFI view of [`#name`]: builds a `Pie<T>Desc` aliasing
        /// `native`'s data. Holders keep any nested-view allocations
        /// alive for the view's lifetime. Pointers in `desc` are
        /// invalid once the view drops.
        #[cfg(feature = "cabi")]
        pub struct #view_ty<'a> {
            pub desc: #desc,
            #(#holder_decls,)*
            _marker: ::core::marker::PhantomData<&'a #name>,
        }

        /// Build the view from a borrowed native value. Zero rkyv
        /// serialization; suitable for in-process FFI handoff.
        #[cfg(feature = "cabi")]
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
    }
}

// ---------------------------------------------------------------------------
// Enum emission
// ---------------------------------------------------------------------------

enum VariantShape<'a> {
    Unit,
    Newtype(&'a Type),
    Inline(Vec<(Ident, &'a Type)>),
}

fn classify_variant(v: &Variant) -> Result<VariantShape<'_>, syn::Error> {
    match &v.fields {
        Fields::Unit => Ok(VariantShape::Unit),
        Fields::Unnamed(f) if f.unnamed.len() == 1 => Ok(VariantShape::Newtype(&f.unnamed[0].ty)),
        Fields::Named(f) => Ok(VariantShape::Inline(
            f.named
                .iter()
                .filter_map(|f| f.ident.clone().map(|n| (n, &f.ty)))
                .collect(),
        )),
        _ => Err(syn::Error::new_spanned(
            v,
            "#[schema] enum variants must be unit, single-field newtype, or named-field struct",
        )),
    }
}

fn emit_enum(name: &Ident, data: &DataEnum) -> TS2 {
    let prefix = snake(&name.to_string());
    let archived = format_ident!("Archived{}", name);

    let mut shapes: Vec<(Ident, VariantShape)> = Vec::new();
    for v in &data.variants {
        match classify_variant(v) {
            Ok(s) => shapes.push((v.ident.clone(), s)),
            Err(e) => return e.to_compile_error(),
        }
    }
    let all_unit = shapes.iter().all(|(_, s)| matches!(s, VariantShape::Unit));

    if all_unit {
        return emit_unit_enum(name, &prefix, &archived, &shapes);
    }
    emit_data_enum(name, &prefix, &archived, &shapes)
}

fn emit_unit_enum(
    name: &Ident,
    prefix: &str,
    archived: &Ident,
    shapes: &[(Ident, VariantShape)],
) -> TS2 {
    let value_fn = format_ident!("pie_{}_value", prefix);
    let view_fn = format_ident!("pie_{}_view", prefix);
    let view_ty = format_ident!("Pie{}View", name);
    let from_fn = format_ident!("__pie_{}_from_desc", prefix);
    let desc = format_ident!("Pie{}Desc", name);

    // Discriminant getter: match the archived form.
    let arms: Vec<TS2> = shapes
        .iter()
        .enumerate()
        .map(|(i, (v, _))| {
            let i = i as u8;
            quote! { #archived::#v => #i, }
        })
        .collect();

    // from_desc(u8) -> Native enum
    let from_arms: Vec<TS2> = shapes
        .iter()
        .enumerate()
        .map(|(i, (v, _))| {
            let i = i as u8;
            quote! { #i => #name::#v, }
        })
        .collect();
    let first_variant = &shapes[0].0;

    // view: native enum -> u8 discriminant
    let view_arms: Vec<TS2> = shapes
        .iter()
        .enumerate()
        .map(|(i, (v, _))| {
            let i = i as u8;
            quote! { #name::#v => #i, }
        })
        .collect();

    let pyclass = emit_pyclass_unit_enum(name, archived);

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

        #[cfg(feature = "cabi")]
        pub type #desc = u8;

        #[cfg(feature = "cabi")]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #value_fn(owner: *const #archived) -> u8 {
            match unsafe { &*owner } {
                #(#arms)*
            }
        }

        #[cfg(feature = "cabi")]
        #[doc(hidden)]
        pub fn #from_fn(d: &u8) -> #name {
            match *d {
                #(#from_arms)*
                _ => #name::#first_variant,
            }
        }

        /// Direct-FFI view of a unit enum. `desc` is the u8 discriminant.
        /// Uniform with struct/data-enum views so a parent view can
        /// embed `view.desc` regardless of the inner type's shape.
        #[cfg(feature = "cabi")]
        pub struct #view_ty<'a> {
            pub desc: #desc,
            _marker: ::core::marker::PhantomData<&'a #name>,
        }

        /// Build the unit-enum view; `view.desc` is the discriminant.
        #[cfg(feature = "cabi")]
        pub fn #view_fn<'a>(native: &'a #name) -> #view_ty<'a> {
            #view_ty {
                desc: match native {
                    #(#view_arms)*
                },
                _marker: ::core::marker::PhantomData,
            }
        }

        #pyclass
    }
}

fn emit_data_enum(
    name: &Ident,
    prefix: &str,
    archived: &Ident,
    shapes: &[(Ident, VariantShape)],
) -> TS2 {
    let kind_fn = format_ident!("pie_{}_kind", prefix);
    let from_fn = format_ident!("__pie_{}_from_desc", prefix);
    let desc = format_ident!("Pie{}Desc", name);

    // kind() — discriminant match
    let kind_arms: Vec<TS2> = shapes
        .iter()
        .enumerate()
        .map(|(i, (v, shape))| {
            let i = i as u8;
            let pat = match shape {
                VariantShape::Unit => quote!(#archived::#v),
                VariantShape::Newtype(_) => quote!(#archived::#v(_)),
                VariantShape::Inline(_) => quote!(#archived::#v { .. }),
            };
            quote! { #pat => #i, }
        })
        .collect();

    // pie_<enum>_as_<variant> for newtype variants
    let mut newtype_dispatchers: Vec<TS2> = Vec::new();
    for (v, shape) in shapes {
        if let VariantShape::Newtype(ty) = shape {
            let Some(inner_name) = type_ident(ty) else {
                continue;
            };
            let inner_archived = format_ident!("Archived{}", inner_name);
            let as_fn = format_ident!("pie_{}_as_{}", prefix, snake(&v.to_string()));
            newtype_dispatchers.push(quote! {
                #[cfg(feature = "cabi")]
                #[unsafe(no_mangle)]
                pub unsafe extern "C" fn #as_fn(owner: *const #archived) -> *const #inner_archived {
                    match unsafe { &*owner } {
                        #archived::#v(inner) => inner as *const _,
                        _ => ::core::ptr::null(),
                    }
                }
            });
        }
    }

    // Union of inline-struct fields across all variants. Same name in
    // multiple variants → one accessor that pattern-matches all.
    let mut field_index: Vec<(Ident, &Type, Vec<&Ident>)> = Vec::new();
    for (vname, shape) in shapes {
        if let VariantShape::Inline(items) = shape {
            for (fname, ty) in items {
                if let Some(slot) = field_index.iter_mut().find(|(n, _, _)| n == fname) {
                    slot.2.push(vname);
                } else {
                    field_index.push((fname.clone(), ty, vec![vname]));
                }
            }
        }
    }

    let mut field_accessors: Vec<TS2> = Vec::new();
    for (fname, ty, variants) in &field_index {
        match enum_field_accessor(archived, prefix, fname, ty, variants) {
            Ok(t) => field_accessors.push(t),
            Err(e) => return e.to_compile_error(),
        }
    }

    // Descriptor: u8 kind + embedded fields for newtype variants + flat
    // field union for inline-struct variants.
    let mut desc_fields: Vec<TS2> = vec![quote!(pub kind: u8)];
    for (vname, shape) in shapes {
        match shape {
            VariantShape::Newtype(ty) => {
                let Some(inner_name) = type_ident(ty) else {
                    continue;
                };
                let nested_desc = format_ident!("Pie{}Desc", inner_name);
                let fname = format_ident!("{}", snake(&vname.to_string()));
                desc_fields.push(quote!(pub #fname: #nested_desc));
            }
            VariantShape::Unit => {}
            VariantShape::Inline(_) => {}
        }
    }
    // Add inline-struct fields to desc (union). Walked in source order
    // across variants — Option<T> is no longer a supported FieldKind, so
    // there's no "Option-last" reshuffle to keep in sync with the C
    // header. The hand-written `pie_bridge.h` mirrors this order
    // byte-for-byte; `tests/desc_layout.rs` pins exact offsets.
    for (fname, ty, _) in &field_index {
        match classify(ty) {
            Ok(k) => desc_fields.push(emit_desc_field(fname, &k)),
            Err(e) => return e.to_compile_error(),
        }
    }

    // from_desc: dispatch on kind, construct variant.
    let mut from_arms: Vec<TS2> = Vec::new();
    for (i, (vname, shape)) in shapes.iter().enumerate() {
        let i = i as u8;
        let arm = match shape {
            VariantShape::Unit => quote! { #i => #name::#vname, },
            VariantShape::Newtype(ty) => {
                let Some(inner_name) = type_ident(ty) else {
                    return syn::Error::new_spanned(
                        name,
                        "newtype variant must contain a named type",
                    )
                    .to_compile_error();
                };
                let from_inner =
                    format_ident!("__pie_{}_from_desc", snake(&inner_name.to_string()));
                let fname = format_ident!("{}", snake(&vname.to_string()));
                quote! { #i => #name::#vname(#from_inner(&d.#fname)), }
            }
            VariantShape::Inline(items) => {
                let inits: Vec<TS2> = items
                    .iter()
                    .map(|(fname, ty)| {
                        let kind =
                            classify(ty).unwrap_or(FieldKind::PrimScalar(format_ident!("u32")));
                        let expr = build_inline_field_extract(fname, &kind);
                        quote! { #fname: #expr }
                    })
                    .collect();
                quote! { #i => #name::#vname { #(#inits,)* }, }
            }
        };
        from_arms.push(arm);
    }
    let first_variant = &shapes[0].0;
    let first_default = match &shapes[0].1 {
        VariantShape::Unit => quote!(#name::#first_variant),
        _ => quote!(panic!("invalid descriptor kind")),
    };

    let pyclass = emit_pyclass_data_enum(name, archived, prefix, shapes, &field_index);
    let view_block = emit_data_enum_view(name, prefix, &desc, shapes);

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

        #[cfg(feature = "cabi")]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #kind_fn(owner: *const #archived) -> u8 {
            match unsafe { &*owner } {
                #(#kind_arms)*
            }
        }

        #(#newtype_dispatchers)*
        #(#field_accessors)*

        #[cfg(feature = "cabi")]
        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct #desc {
            #(#desc_fields,)*
        }

        #[cfg(feature = "cabi")]
        impl Default for #desc {
            #[inline]
            fn default() -> Self {
                // SAFETY: every field is POD with valid all-zero bit
                // pattern (null ptr, 0 len, 0 scalar, sub-Desc whose
                // fields are zero-valid transitively).
                unsafe { ::core::mem::zeroed() }
            }
        }

        #[cfg(feature = "cabi")]
        #[doc(hidden)]
        pub fn #from_fn(d: &#desc) -> #name {
            match d.kind {
                #(#from_arms)*
                _ => #first_default,
            }
        }

        #view_block

        #pyclass
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

    // Holder-enum variants: one per newtype variant, plus a catch-all `None`.
    let mut holder_variants: Vec<TS2> = Vec::new();
    let mut needs_lifetime = false;
    for (vname, shape) in shapes {
        if let VariantShape::Newtype(ty) = shape {
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
            VariantShape::Newtype(ty) => {
                let Some(inner) = type_ident(ty) else {
                    return syn::Error::new_spanned(
                        name,
                        "newtype variant must contain a named type",
                    )
                    .to_compile_error();
                };
                let inner_view_fn = format_ident!("pie_{}_view", snake(&inner.to_string()));
                let fname = format_ident!("{}", snake(&vname.to_string()));
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
            VariantShape::Inline(items) => {
                // Fill the flat-field union for the active variant.
                let mut inits: Vec<TS2> = Vec::new();
                for (fname, ty) in items {
                    let k = match classify(ty) {
                        Ok(k) => k,
                        Err(e) => return e.to_compile_error(),
                    };
                    let init = build_inline_field_view_init(fname, &k);
                    inits.push(init);
                }
                let bind_fields: Vec<TS2> = items.iter().map(|(f, _)| quote! { #f }).collect();
                quote! {
                    #name::#vname { #(#bind_fields,)* } => {
                        let mut d = <#desc as ::core::default::Default>::default();
                        d.kind = #kind;
                        #(#inits)*
                        (d, #holder_enum::None)
                    }
                }
            }
        };
        view_arms.push(arm);
    }

    let _holder_needs_lifetime = needs_lifetime;
    let lifetime_decl = quote!(<'a>); // always parameterize for uniform call sites

    quote! {
        #[cfg(feature = "cabi")]
        #[allow(non_camel_case_types, dead_code)]
        enum #holder_enum #lifetime_decl {
            #(#holder_variants,)*
            _Unused(::core::marker::PhantomData<&'a ()>),
        }

        /// Direct-FFI view of a data enum. `desc` carries the
        /// discriminant + active variant's sub-desc; the internal
        /// holder keeps any nested view allocations alive.
        #[cfg(feature = "cabi")]
        pub struct #view_ty<'a> {
            pub desc: #desc,
            #[allow(dead_code)]
            __h: #holder_enum<'a>,
            _marker: ::core::marker::PhantomData<&'a #name>,
        }

        /// Build the data-enum view from a borrowed native value.
        #[cfg(feature = "cabi")]
        pub fn #view_fn<'a>(native: &'a #name) -> #view_ty<'a> {
            let (desc, __h) = match native {
                #(#view_arms)*
            };
            #view_ty { desc, __h, _marker: ::core::marker::PhantomData }
        }
    }
}

/// Per-inline-struct-variant field init: write `d.<f> = ...` for the
/// active variant. Mirrors `emit_desc_field` but for a *value*
/// assignment instead of a struct decl. Borrows from the matched
/// variant's fields (slice ptrs alias the native Vec).
fn build_inline_field_view_init(field: &Ident, kind: &FieldKind) -> TS2 {
    let ptr_name = format_ident!("{}_ptr", field);
    let len_name = format_ident!("{}_len", field);
    match kind {
        FieldKind::BoolScalar => quote! {
            d.#field = ::core::primitive::u8::from(*#field);
        },
        FieldKind::PrimScalar(_) => quote! {
            d.#field = *#field;
        },
        FieldKind::SlicePrim(elem) => quote! {
            d.#ptr_name = #field.as_ptr() as *const #elem;
            d.#len_name = #field.len();
        },
        FieldKind::SliceBool => quote! {
            d.#ptr_name = #field.as_ptr() as *const u8;
            d.#len_name = #field.len();
        },
        FieldKind::BareString => quote! {
            d.#ptr_name = #field.as_ptr() as *const u8;
            d.#len_name = #field.len();
        },
        FieldKind::SliceNested(_) | FieldKind::Nested(_) => {
            // Inline-struct-variant fields holding nested-schema types
            // would need their own view holder. None of our current
            // enums hit this; emit a compile error if a future schema
            // change introduces it.
            quote! {
                compile_error!("nested-schema field inside inline-struct enum variant is not supported by direct-FFI view yet");
            }
        }
    }
}

/// Build the per-variant field-extraction expression used in
/// `from_desc` for inline-struct variants.
fn build_inline_field_extract(field: &Ident, kind: &FieldKind) -> TS2 {
    let ptr_name = format_ident!("{}_ptr", field);
    let len_name = format_ident!("{}_len", field);
    match kind {
        FieldKind::BoolScalar => quote! { d.#field != 0 },
        FieldKind::PrimScalar(_) => quote! { d.#field },
        FieldKind::SlicePrim(elem) => quote! {
            unsafe {
                if d.#ptr_name.is_null() || d.#len_name == 0 {
                    ::std::vec::Vec::<#elem>::new()
                } else {
                    ::core::slice::from_raw_parts(d.#ptr_name, d.#len_name).to_vec()
                }
            }
        },
        FieldKind::SliceBool => quote! {
            unsafe {
                if d.#ptr_name.is_null() || d.#len_name == 0 {
                    ::std::vec::Vec::<bool>::new()
                } else {
                    ::core::slice::from_raw_parts(d.#ptr_name, d.#len_name)
                        .iter()
                        .map(|b| *b != 0)
                        .collect()
                }
            }
        },
        FieldKind::SliceNested(name) => {
            let from_fn = format_ident!("__pie_{}_from_desc", snake(&name.to_string()));
            quote! {
                unsafe {
                    if d.#ptr_name.is_null() || d.#len_name == 0 {
                        ::std::vec::Vec::<#name>::new()
                    } else {
                        (0..d.#len_name)
                            .map(|i| #from_fn(&*d.#ptr_name.add(i)))
                            .collect()
                    }
                }
            }
        }
        FieldKind::BareString => quote! {
            unsafe {
                if d.#ptr_name.is_null() || d.#len_name == 0 {
                    String::new()
                } else {
                    let bytes = ::core::slice::from_raw_parts(d.#ptr_name, d.#len_name);
                    String::from_utf8_lossy(bytes).into_owned()
                }
            }
        },
        FieldKind::Nested(name) => {
            let from_fn = format_ident!("__pie_{}_from_desc", snake(&name.to_string()));
            quote! { #from_fn(&d.#field) }
        }
    }
}

/// Emit a C-ABI accessor for a field that appears in one or more
/// inline-struct variants of a data enum. The accessor pattern-matches
/// the archived enum and returns the value when present, default otherwise.
fn enum_field_accessor(
    archived: &Ident,
    prefix: &str,
    field: &Ident,
    ty: &Type,
    variants: &[&Ident],
) -> Result<TS2, syn::Error> {
    let kind = classify(ty)?;
    let fn_base = format_ident!("pie_{}_{}", prefix, field);

    // Patterns that match the relevant variants.
    let pats: Vec<TS2> = variants
        .iter()
        .map(|v| quote!(#archived::#v { #field, .. }))
        .collect();

    Ok(match kind {
        FieldKind::BoolScalar => quote! {
            #[cfg(feature = "cabi")]
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #fn_base(owner: *const #archived) -> u8 {
                match unsafe { &*owner } {
                    #(#pats)|* => ::core::primitive::u8::from(*#field),
                    _ => 0,
                }
            }
        },
        FieldKind::PrimScalar(t) => quote! {
            #[cfg(feature = "cabi")]
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #fn_base(owner: *const #archived) -> #t {
                match unsafe { &*owner } {
                    #(#pats)|* => (*#field).into(),
                    _ => <#t as ::core::default::Default>::default(),
                }
            }
        },
        FieldKind::SlicePrim(elem) => quote! {
            #[cfg(feature = "cabi")]
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #fn_base(
                owner: *const #archived,
                out_ptr: *mut *const #elem,
                out_len: *mut usize,
            ) {
                match unsafe { &*owner } {
                    #(#pats)|* => unsafe {
                        *out_ptr = #field.as_ptr() as *const #elem;
                        *out_len = #field.len();
                    },
                    _ => unsafe {
                        *out_ptr = ::core::ptr::null();
                        *out_len = 0;
                    },
                }
            }
        },
        FieldKind::SliceBool => quote! {
            #[cfg(feature = "cabi")]
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #fn_base(
                owner: *const #archived,
                out_ptr: *mut *const u8,
                out_len: *mut usize,
            ) {
                match unsafe { &*owner } {
                    #(#pats)|* => unsafe {
                        *out_ptr = #field.as_ptr() as *const u8;
                        *out_len = #field.len();
                    },
                    _ => unsafe {
                        *out_ptr = ::core::ptr::null();
                        *out_len = 0;
                    },
                }
            }
        },
        FieldKind::BareString => quote! {
            #[cfg(feature = "cabi")]
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #fn_base(
                owner: *const #archived,
                out_ptr: *mut *const ::core::ffi::c_char,
                out_len: *mut usize,
            ) {
                match unsafe { &*owner } {
                    #(#pats)|* => unsafe {
                        *out_ptr = #field.as_ptr() as *const ::core::ffi::c_char;
                        *out_len = #field.len();
                    },
                    _ => unsafe {
                        *out_ptr = ::core::ptr::null();
                        *out_len = 0;
                    },
                }
            }
        },
        FieldKind::SliceNested(_) | FieldKind::Nested(_) => {
            return Err(syn::Error::new_spanned(
                ty,
                "nested enum field in inline-struct variant not yet supported",
            ));
        }
    })
}

// ---------------------------------------------------------------------------
// PyO3 wrappers
// ---------------------------------------------------------------------------

fn emit_pyclass_struct(name: &Ident, archived: &Ident, parse_fn: &Ident, methods: &[TS2]) -> TS2 {
    let py_name = format_ident!("Py{}", name);
    let mod_name = format_ident!("__py_{}", snake(&name.to_string()));
    let register = format_ident!("register_{}", snake(&name.to_string()));
    let display = name.to_string();

    quote! {
        #[cfg(feature = "python")]
        #[doc(hidden)]
        pub mod #mod_name {
            use ::pyo3::prelude::*;
            use ::pyo3::types::PyBytes;
            use crate::schema::#archived;

            // `bytes: Py<PyBytes>` (was `Arc<Vec<u8>>`) — zero-copy
            // parse: we hold a refcounted handle to the Python bytes
            // object the caller passed in, rather than memcpying its
            // contents into a Rust Vec. The PyBytes buffer is
            // immutable on the Python side, so the `ptr` aliasing the
            // archive inside it stays valid for the lifetime of this
            // wrapper.
            #[::pyo3::pyclass(name = #display, unsendable)]
            pub struct #py_name {
                pub(crate) bytes: ::pyo3::Py<PyBytes>,
                pub(crate) ptr: usize,
            }

            impl #py_name {
                #[allow(dead_code)]
                pub(crate) fn from_nested(
                    bytes: ::pyo3::Py<PyBytes>,
                    ptr: *const #archived,
                ) -> Self {
                    Self { bytes, ptr: ptr as usize }
                }
            }

            #[::pyo3::pymethods]
            impl #py_name {
                #[staticmethod]
                fn parse(data: &::pyo3::Bound<'_, PyBytes>) -> ::pyo3::PyResult<Self> {
                    let buf = data.as_bytes();
                    let ptr = unsafe { crate::schema::#parse_fn(buf.as_ptr(), buf.len()) };
                    if ptr.is_null() {
                        return Err(::pyo3::exceptions::PyValueError::new_err(
                            "invalid rkyv buffer",
                        ));
                    }
                    Ok(Self {
                        bytes: data.clone().unbind(),
                        ptr: ptr as usize,
                    })
                }

                #(#methods)*
            }

            pub fn #register(m: &::pyo3::Bound<'_, ::pyo3::types::PyModule>) -> ::pyo3::PyResult<()> {
                m.add_class::<#py_name>()
            }
        }
    }
}

fn emit_pyclass_unit_enum(name: &Ident, archived: &Ident) -> TS2 {
    let py_name = format_ident!("Py{}", name);
    let mod_name = format_ident!("__py_{}", snake(&name.to_string()));
    let register = format_ident!("register_{}", snake(&name.to_string()));
    let value_fn = format_ident!("pie_{}_value", snake(&name.to_string()));
    let display = name.to_string();

    quote! {
        #[cfg(feature = "python")]
        #[doc(hidden)]
        pub mod #mod_name {
            use ::pyo3::prelude::*;
            use ::pyo3::types::PyBytes;
            use crate::schema::#archived;

            #[::pyo3::pyclass(name = #display, unsendable)]
            pub struct #py_name {
                pub(crate) bytes: ::pyo3::Py<PyBytes>,
                pub(crate) ptr: usize,
            }

            impl #py_name {
                #[allow(dead_code)]
                pub(crate) fn from_nested(
                    bytes: ::pyo3::Py<PyBytes>,
                    ptr: *const #archived,
                ) -> Self {
                    Self { bytes, ptr: ptr as usize }
                }
            }

            #[::pyo3::pymethods]
            impl #py_name {
                #[getter]
                fn value(&self) -> ::pyo3::PyResult<u8> {
                    Ok(unsafe { crate::schema::#value_fn(self.ptr as *const _) })
                }
            }

            pub fn #register(m: &::pyo3::Bound<'_, ::pyo3::types::PyModule>) -> ::pyo3::PyResult<()> {
                m.add_class::<#py_name>()
            }
        }
    }
}

fn emit_pyclass_data_enum(
    name: &Ident,
    archived: &Ident,
    prefix: &str,
    shapes: &[(Ident, VariantShape)],
    field_index: &[(Ident, &Type, Vec<&Ident>)],
) -> TS2 {
    let py_name = format_ident!("Py{}", name);
    let mod_name = format_ident!("__py_{}", prefix);
    let register = format_ident!("register_{}", prefix);
    let kind_fn = format_ident!("pie_{}_kind", prefix);
    let display = name.to_string();

    // as_<variant> wrappers for newtype variants.
    let mut as_methods: Vec<TS2> = Vec::new();
    for (v, shape) in shapes {
        if let VariantShape::Newtype(ty) = shape {
            let Some(inner_name) = type_ident(ty) else {
                continue;
            };
            let inner_snake = snake(&inner_name.to_string());
            let inner_py = format_ident!("Py{}", inner_name);
            let inner_mod = format_ident!("__py_{}", inner_snake);
            let as_fn = format_ident!("pie_{}_as_{}", prefix, snake(&v.to_string()));
            let method = format_ident!("as_{}", snake(&v.to_string()));
            as_methods.push(quote! {
                fn #method(&self, py: ::pyo3::Python<'_>)
                    -> ::pyo3::PyResult<Option<crate::#inner_mod::#inner_py>>
                {
                    let p = unsafe { crate::schema::#as_fn(self.ptr as *const _) };
                    if p.is_null() {
                        Ok(None)
                    } else {
                        Ok(Some(crate::#inner_mod::#inner_py::from_nested(
                            self.bytes.clone_ref(py),
                            p,
                        )))
                    }
                }
            });
        }
    }

    // Per-field getters (inline-struct variants).
    let mut field_methods: Vec<TS2> = Vec::new();
    for (fname, ty, _) in field_index {
        let kind = match classify(ty) {
            Ok(k) => k,
            Err(e) => return e.to_compile_error(),
        };
        field_methods.push(emit_py_method(prefix, fname, &kind));
    }

    quote! {
        #[cfg(feature = "python")]
        #[doc(hidden)]
        pub mod #mod_name {
            use ::pyo3::prelude::*;
            use ::pyo3::types::PyBytes;
            use crate::schema::#archived;

            #[::pyo3::pyclass(name = #display, unsendable)]
            pub struct #py_name {
                pub(crate) bytes: ::pyo3::Py<PyBytes>,
                pub(crate) ptr: usize,
            }

            impl #py_name {
                #[allow(dead_code)]
                pub(crate) fn from_nested(
                    bytes: ::pyo3::Py<PyBytes>,
                    ptr: *const #archived,
                ) -> Self {
                    Self { bytes, ptr: ptr as usize }
                }
            }

            #[::pyo3::pymethods]
            impl #py_name {
                #[getter]
                fn kind(&self) -> ::pyo3::PyResult<u8> {
                    Ok(unsafe { crate::schema::#kind_fn(self.ptr as *const _) })
                }

                #(#as_methods)*
                #(#field_methods)*
            }

            pub fn #register(m: &::pyo3::Bound<'_, ::pyo3::types::PyModule>) -> ::pyo3::PyResult<()> {
                m.add_class::<#py_name>()
            }
        }
    }
}

fn emit_py_method(prefix: &str, field: &Ident, kind: &FieldKind) -> TS2 {
    let method_name = format_ident!("{}", field);
    let fn_base = format_ident!("pie_{}_{}", prefix, field);
    match kind {
        FieldKind::BoolScalar => quote! {
            #[getter]
            fn #method_name(&self) -> ::pyo3::PyResult<bool> {
                Ok((unsafe { crate::schema::#fn_base(self.ptr as *const _) }) != 0)
            }
        },
        FieldKind::PrimScalar(t) => quote! {
            #[getter]
            fn #method_name(&self) -> ::pyo3::PyResult<#t> {
                Ok(unsafe { crate::schema::#fn_base(self.ptr as *const _) })
            }
        },
        FieldKind::SlicePrim(elem) => {
            // Map Rust primitive idents to NumPy dtype names. Matches
            // rkyv's archived primitive layout (native byte order, no
            // padding) — `numpy.frombuffer` reads the buffer with this
            // dtype and returns a zero-copy view over `self.bytes`.
            let dtype = match elem.to_string().as_str() {
                "u8" => "uint8",
                "i8" => "int8",
                "u16" => "uint16",
                "i16" => "int16",
                "u32" => "uint32",
                "i32" => "int32",
                "u64" => "uint64",
                "i64" => "int64",
                "f32" => "float32",
                "f64" => "float64",
                _ => "uint8",
            };
            quote! {
                #[getter]
                fn #method_name<'py>(
                    &self,
                    py: ::pyo3::Python<'py>,
                ) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::PyAny>> {
                    let mut p: *const #elem = ::core::ptr::null();
                    let mut n: usize = 0;
                    unsafe {
                        crate::schema::#fn_base(self.ptr as *const _, &mut p, &mut n);
                    }
                    crate::python::slice_to_numpy(
                        py,
                        &self.bytes,
                        p as *const u8,
                        n,
                        #dtype,
                    )
                }
            }
        }
        FieldKind::SliceBool => quote! {
            // rkyv archives `Vec<bool>` as one byte per element (not
            // bit-packed); NumPy `bool_` is also 1 byte. Zero-copy.
            #[getter]
            fn #method_name<'py>(
                &self,
                py: ::pyo3::Python<'py>,
            ) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::PyAny>> {
                let mut p: *const u8 = ::core::ptr::null();
                let mut n: usize = 0;
                unsafe {
                    crate::schema::#fn_base(self.ptr as *const _, &mut p, &mut n);
                }
                crate::python::slice_to_numpy(py, &self.bytes, p, n, "bool_")
            }
        },
        FieldKind::SliceNested(name) => {
            let inner_snake = snake(&name.to_string());
            let inner_py = format_ident!("Py{}", name);
            let inner_mod = format_ident!("__py_{}", inner_snake);
            let len_fn = format_ident!("pie_{}_{}_len", prefix, field);
            let at_fn = format_ident!("pie_{}_{}_at", prefix, field);
            let len_method = format_ident!("{}_len", field);
            let at_method = format_ident!("{}_at", field);
            quote! {
                #[getter]
                fn #len_method(&self) -> ::pyo3::PyResult<usize> {
                    Ok(unsafe { crate::schema::#len_fn(self.ptr as *const _) })
                }
                fn #at_method(&self, py: ::pyo3::Python<'_>, i: usize)
                    -> ::pyo3::PyResult<crate::#inner_mod::#inner_py>
                {
                    let p = unsafe { crate::schema::#at_fn(self.ptr as *const _, i) };
                    if p.is_null() {
                        Err(::pyo3::exceptions::PyIndexError::new_err(format!(
                            "index {} out of range", i
                        )))
                    } else {
                        Ok(crate::#inner_mod::#inner_py::from_nested(
                            self.bytes.clone_ref(py),
                            p,
                        ))
                    }
                }
            }
        }
        FieldKind::BareString => quote! {
            // Skips the intermediate `Rust String` allocation + the UTF-8
            // lossy scan that `String::from_utf8_lossy` did. The schema
            // type's `String` was already valid UTF-8, and rkyv preserves
            // the bytes verbatim. A full zero-copy str is impossible (the
            // CPython str object has its own storage and encoding), so
            // `PyString::new` still copies into Python's str — but it's
            // one allocation instead of two.
            #[getter]
            fn #method_name<'py>(
                &self,
                py: ::pyo3::Python<'py>,
            ) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::types::PyString>> {
                let mut p: *const ::core::ffi::c_char = ::core::ptr::null();
                let mut n: usize = 0;
                unsafe {
                    crate::schema::#fn_base(self.ptr as *const _, &mut p, &mut n);
                }
                if p.is_null() || n == 0 {
                    Ok(::pyo3::types::PyString::new(py, ""))
                } else {
                    // SAFETY: rkyv preserves the bytes of `String`
                    // verbatim; the source was valid UTF-8.
                    let s = unsafe {
                        let bytes = ::core::slice::from_raw_parts(p as *const u8, n);
                        ::std::str::from_utf8_unchecked(bytes)
                    };
                    Ok(::pyo3::types::PyString::new(py, s))
                }
            }
        },
        FieldKind::Nested(name) => {
            let inner_snake = snake(&name.to_string());
            let inner_py = format_ident!("Py{}", name);
            let inner_mod = format_ident!("__py_{}", inner_snake);
            quote! {
                #[getter]
                fn #method_name(&self, py: ::pyo3::Python<'_>)
                    -> ::pyo3::PyResult<crate::#inner_mod::#inner_py>
                {
                    let p = unsafe { crate::schema::#fn_base(self.ptr as *const _) };
                    Ok(crate::#inner_mod::#inner_py::from_nested(
                        self.bytes.clone_ref(py),
                        p,
                    ))
                }
            }
        }
    }
}
