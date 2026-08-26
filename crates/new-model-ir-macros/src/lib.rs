//! `#[derive(Operands)]`: def-use edges read straight off an op enum's field
//! declarations, so there is no parallel operand list to drift out of sync.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::{Data, DeriveInput, Error, Field, Fields, Ident, Meta, Result, Type, parse_macro_input};

/// Derives `Operands` for an enum whose variants have named fields.
///
/// The generated impl names the trait and the id type as `crate::Operands`
/// and `crate::ValueId`; the consuming crate must re-export both at its root.
///
/// Field classification, per variant:
/// - a field typed `ValueId`, `Vec<ValueId>`, or `Option<ValueId>` with no
///   attribute is an input;
/// - `#[out]` marks an output;
/// - `#[out(alias = other)]` marks an output that overwrites the input
///   `other` — an InOut kept as two SSA ids plus an `aliases()` entry;
/// - fields of any other type (`u32`, `f32`, `Option<u32>`, …) are ignored.
///
/// `name()` returns `"enum.variant"` with both parts snake_cased:
/// `Gemm::Matmul` → `"gemm.matmul"`, `Cache::KvAppend` → `"cache.kv_append"`.
///
/// Caveat: types are matched by token spelling, so don't alias `ValueId`
/// (or spell it with a path) — the derive won't recognize it.
#[proc_macro_derive(Operands, attributes(out))]
pub fn derive_operands(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(&input).unwrap_or_else(Error::into_compile_error).into()
}

fn expand(input: &DeriveInput) -> Result<TokenStream2> {
    let Data::Enum(data) = &input.data else {
        return Err(Error::new_spanned(&input.ident, "Operands derives only on enums"));
    };

    let (mut inputs, mut outputs, mut aliases, mut names) = (vec![], vec![], vec![], vec![]);
    for variant in &data.variants {
        let ops = classify(variant)?;
        let v = &variant.ident;
        let name = format!("{}.{}", snake(&input.ident), snake(v));
        inputs.push(push_arm(v, &ops.inputs));
        outputs.push(push_arm(v, &ops.outputs));
        aliases.push(alias_arm(v, &ops.aliases));
        names.push(quote! { Self::#v { .. } => #name, });
    }

    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics crate::Operands for #ident #ty_generics #where_clause {
            fn inputs(&self, sink: &mut Vec<crate::ValueId>) {
                match self { #(#inputs)* }
            }
            fn outputs(&self, sink: &mut Vec<crate::ValueId>) {
                match self { #(#outputs)* }
            }
            fn aliases(&self, sink: &mut Vec<(crate::ValueId, crate::ValueId)>) {
                match self { #(#aliases)* }
            }
            fn name(&self) -> &'static str {
                match self { #(#names)* }
            }
        }
    })
}

/// How a `ValueId`-carrying field holds its ids.
enum Kind {
    One,
    Many,
    Maybe,
}

/// One variant's operands: `(field, kind)` per input and output, and the
/// `(out, aliased in)` pairs.
struct Ops {
    inputs: Vec<(Ident, Kind)>,
    outputs: Vec<(Ident, Kind)>,
    aliases: Vec<(Ident, Ident)>,
}

fn classify(variant: &syn::Variant) -> Result<Ops> {
    let Fields::Named(fields) = &variant.fields else {
        return Err(Error::new_spanned(variant, "Operands variants must have named fields"));
    };

    let mut ops = Ops { inputs: vec![], outputs: vec![], aliases: vec![] };
    for field in &fields.named {
        let ident = field.ident.clone().unwrap();
        match (out_attr(field)?, kind(&field.ty)) {
            (None, Some(k)) => ops.inputs.push((ident, k)),
            (None, None) => {}
            (Some(_), None) => {
                return Err(Error::new_spanned(
                    &field.ty,
                    "#[out] field must be a ValueId, Vec<ValueId>, or Option<ValueId>",
                ));
            }
            (Some(None), Some(k)) => ops.outputs.push((ident, k)),
            (Some(Some(target)), Some(Kind::One)) => {
                ops.aliases.push((ident.clone(), target));
                ops.outputs.push((ident, Kind::One));
            }
            (Some(Some(_)), Some(_)) => {
                return Err(Error::new_spanned(&field.ty, "an aliasing output must be a plain ValueId"));
            }
        }
    }

    for (out, target) in &ops.aliases {
        if !ops.inputs.iter().any(|(f, k)| f == target && matches!(k, Kind::One)) {
            return Err(Error::new(
                target.span(),
                format!("`{out}` aliases `{target}`, which is not an un-attributed ValueId field of this variant"),
            ));
        }
    }
    Ok(ops)
}

/// `None` = no `#[out]`; `Some(None)` = `#[out]`; `Some(Some(f))` = `#[out(alias = f)]`.
fn out_attr(field: &Field) -> Result<Option<Option<Ident>>> {
    let Some(attr) = field.attrs.iter().find(|a| a.path().is_ident("out")) else {
        return Ok(None);
    };
    if matches!(attr.meta, Meta::Path(_)) {
        return Ok(Some(None));
    }
    let mut alias = None;
    attr.parse_nested_meta(|meta| {
        if !meta.path.is_ident("alias") {
            return Err(meta.error("expected `alias = <field>`"));
        }
        alias = Some(meta.value()?.parse()?);
        Ok(())
    })?;
    match alias {
        Some(target) => Ok(Some(Some(target))),
        None => Err(Error::new_spanned(attr, "expected `#[out]` or `#[out(alias = <field>)]`")),
    }
}

/// Matches the type by token spelling — the §2 caveat: don't alias `ValueId`.
fn kind(ty: &Type) -> Option<Kind> {
    match ty.to_token_stream().to_string().replace(' ', "").as_str() {
        "ValueId" => Some(Kind::One),
        "Vec<ValueId>" => Some(Kind::Many),
        "Option<ValueId>" => Some(Kind::Maybe),
        _ => None,
    }
}

/// One `inputs`/`outputs` arm: destructure the named fields, elide the rest,
/// push every id each field holds.
fn push_arm(variant: &Ident, fields: &[(Ident, Kind)]) -> TokenStream2 {
    let names = fields.iter().map(|(f, _)| f);
    let pushes = fields.iter().map(|(f, k)| match k {
        Kind::One => quote! { sink.push(*#f); },
        Kind::Many => quote! { sink.extend_from_slice(#f); },
        Kind::Maybe => quote! { if let Some(id) = #f { sink.push(*id); } },
    });
    quote! { Self::#variant { #(#names,)* .. } => { #(#pushes)* } }
}

/// One `aliases` arm: `(out, the input it overwrites)` per `#[out(alias = …)]`.
fn alias_arm(variant: &Ident, pairs: &[(Ident, Ident)]) -> TokenStream2 {
    let mut names: Vec<&Ident> = vec![];
    for (out, input) in pairs {
        for f in [out, input] {
            if !names.contains(&f) {
                names.push(f);
            }
        }
    }
    let pushes = pairs.iter().map(|(out, input)| quote! { sink.push((*#out, *#input)); });
    quote! { Self::#variant { #(#names,)* .. } => { #(#pushes)* } }
}

/// `KvAppend` → `kv_append`, `RmsnormPerHeadPlusOne` → `rmsnorm_per_head_plus_one`.
fn snake(ident: &Ident) -> String {
    let mut out = String::new();
    for c in ident.to_string().chars() {
        if c.is_ascii_uppercase() && !out.is_empty() {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}
