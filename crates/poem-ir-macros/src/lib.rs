//! `#[derive(Op)]`: an op group's enum states each op's operands once, on
//! its fields, and the derive writes [`Operands`] from them.
//!
//! ```ignore
//! #[derive(Op)]
//! #[op(group = "collective")]
//! pub enum Collective {
//!     AllReduce {
//!         #[input] buf: ValueId,
//!         #[output(alias = buf)] buf_out: ValueId,
//!     },
//!     ...
//! }
//! ```
//!
//! - The name is `<group>.<variant in snake case>`; it is never written.
//! - `#[input]` and `#[output]` mark operands, in field order. A field is
//!   anything that implements `poem_ir::operands::Slot` (`ValueId`,
//!   `Option<ValueId>`, `Vec<ValueId>`, arrays, attribute types that carry
//!   values).
//! - `#[input(I32)]` names the operand's expected type, a constant on
//!   `poem_ir::check::Expect`; the validator checks it under the field's
//!   name.
//! - `#[output(alias = f)]` says the output is input `f` updated in place.
//!
//! [`Operands`]: https://docs.rs/pie-poem-ir

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Fields, Ident, LitStr, parse_macro_input};

#[proc_macro_derive(Op, attributes(op, input, output))]
pub fn derive_op(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

struct Operand {
    field: Ident,
    output: bool,
    want: Option<Ident>,
    alias: Option<Ident>,
}

struct Variant {
    ident: Ident,
    name: String,
    operands: Vec<Operand>,
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let group = group(input)?;
    let Data::Enum(data) = &input.data else {
        return Err(syn::Error::new(
            input.span(),
            "#[derive(Op)] goes on an op group's enum",
        ));
    };
    let mut variants = Vec::new();
    for variant in &data.variants {
        let fields = match &variant.fields {
            Fields::Named(fields) => fields.named.iter().collect(),
            Fields::Unit => Vec::new(),
            Fields::Unnamed(_) => {
                return Err(syn::Error::new(
                    variant.span(),
                    "an op's operands are named fields",
                ));
            }
        };
        let mut operands = Vec::new();
        for field in fields {
            let ident = field.ident.clone().expect("named field");
            for attr in &field.attrs {
                let output = attr.path().is_ident("output");
                if !output && !attr.path().is_ident("input") {
                    continue;
                }
                let mut operand = Operand {
                    field: ident.clone(),
                    output,
                    want: None,
                    alias: None,
                };
                if !matches!(attr.meta, syn::Meta::Path(_)) {
                    attr.parse_nested_meta(|meta| {
                        if meta.path.is_ident("alias") {
                            if !output {
                                return Err(meta.error("only an output aliases an input"));
                            }
                            operand.alias = Some(meta.value()?.parse()?);
                        } else if let Some(want) = meta.path.get_ident() {
                            if operand.want.replace(want.clone()).is_some() {
                                return Err(meta.error("one expected type an operand"));
                            }
                        } else {
                            return Err(
                                meta.error("expected an `Expect` constant or `alias = field`")
                            );
                        }
                        Ok(())
                    })?;
                }
                operands.push(operand);
            }
        }
        for operand in &operands {
            if let Some(alias) = &operand.alias
                && !operands.iter().any(|o| !o.output && o.field == *alias)
            {
                return Err(syn::Error::new(
                    alias.span(),
                    format!("`{alias}` is not an #[input] of this op"),
                ));
            }
        }
        variants.push(Variant {
            ident: variant.ident.clone(),
            name: format!("{group}.{}", snake(&variant.ident.to_string())),
            operands,
        });
    }

    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let arm = |variant: &Variant, body: TokenStream2, fields: Vec<&Ident>| {
        let v = &variant.ident;
        quote! { Self::#v { #(#fields,)* .. } => { #body } }
    };

    let side = |output: bool| {
        variants.iter().map(move |variant| {
            let fields: Vec<&Ident> = variant
                .operands
                .iter()
                .filter(|o| o.output == output)
                .map(|o| &o.field)
                .collect();
            let body = quote! { #(::poem_ir::operands::Slot::push(#fields, sink);)* };
            arm(variant, body, fields.clone())
        })
    };
    let inputs = side(false);
    let outputs = side(true);

    let aliases = variants.iter().map(|variant| {
        let pairs: Vec<(&Ident, &Ident)> = variant
            .operands
            .iter()
            .filter_map(|o| o.alias.as_ref().map(|a| (&o.field, a)))
            .collect();
        let mut fields: Vec<&Ident> = Vec::new();
        for field in pairs.iter().flat_map(|(o, i)| [*o, *i]) {
            if !fields.contains(&field) {
                fields.push(field);
            }
        }
        let body = pairs.iter().map(|(out, input)| {
            quote! {
                if let (Some(out), Some(input)) = (
                    ::poem_ir::operands::Slot::one(#out),
                    ::poem_ir::operands::Slot::one(#input),
                ) {
                    sink.push((out, input));
                }
            }
        });
        arm(variant, quote! { #(#body)* }, fields)
    });

    let names = variants.iter().map(|variant| {
        let name = LitStr::new(&variant.name, Span::call_site());
        arm(variant, quote! { #name }, Vec::new())
    });

    let ports = variants.iter().map(|variant| {
        let wanted: Vec<&Operand> = variant
            .operands
            .iter()
            .filter(|o| o.want.is_some())
            .collect();
        let fields: Vec<&Ident> = wanted.iter().map(|o| &o.field).collect();
        let body = wanted.iter().map(|o| {
            let field = &o.field;
            let want = o.want.as_ref().expect("filtered");
            let port = LitStr::new(&field.to_string(), field.span());
            quote! {
                ::poem_ir::operands::Slot::each(#field, &mut |id| {
                    sink.push((#port, id, ::poem_ir::check::Expect::#want));
                });
            }
        });
        arm(variant, quote! { #(#body)* }, fields)
    });

    let empty = variants.is_empty().then(|| quote! { _ => unreachable!() });

    Ok(quote! {
        impl #impl_generics ::poem_ir::Operands for #ident #ty_generics #where_clause {
            fn inputs(&self, sink: &mut Vec<::poem_ir::ValueId>) {
                match self { #(#inputs)* #empty }
            }
            fn outputs(&self, sink: &mut Vec<::poem_ir::ValueId>) {
                match self { #(#outputs)* #empty }
            }
            fn aliases(&self, sink: &mut Vec<(::poem_ir::ValueId, ::poem_ir::ValueId)>) {
                match self { #(#aliases)* #empty }
            }
            fn name(&self) -> &'static str {
                match self { #(#names)* #empty }
            }
            fn ports(
                &self,
                sink: &mut Vec<(&'static str, ::poem_ir::ValueId, ::poem_ir::check::Expect)>,
            ) {
                match self { #(#ports)* #empty }
            }
        }
    })
}

fn group(input: &DeriveInput) -> syn::Result<String> {
    let mut group = None;
    for attr in &input.attrs {
        if !attr.path().is_ident("op") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("group") {
                let lit: LitStr = meta.value()?.parse()?;
                group = Some(lit.value());
                Ok(())
            } else {
                Err(meta.error("expected `group = \"...\"`"))
            }
        })?;
    }
    group.ok_or_else(|| syn::Error::new(input.span(), "#[op(group = \"...\")] names the group"))
}

/// `AllReduce` → `all_reduce`, `TopK` → `top_k`.
fn snake(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}
