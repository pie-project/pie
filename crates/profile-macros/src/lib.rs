//! `#[derive(Decision)]`: one Rust type declares one device-profile decision.
//!
//! ```ignore
//! /// Dense affine-quantised matmul.
//! #[derive(Clone, Debug, PartialEq, Decision)]
//! #[decision(id = "kernel.qmm", shape(rows; k, n, bits: set(2, 4, 8)), default = "qmv")]
//! enum Qmm {
//!     /// One threadgroup per output row.
//!     Qmv { #[constant(set(1, 2, 4, 8), default = 2)] rows_max: u32 },
//!     #[requires("family >= 10")]
//!     #[partial]
//!     QmmMpp { #[constant(default = false)] packed: bool },
//! }
//! ```
//!
//! - An enum is a **choice**; each variant an alternative named in
//!   snake case. A field is a constant: `#[constant(set(..) | range(a..=b) |
//!   word(a, b) | words(a, b), default = <literal>, launch | specialize)]`.
//! - A one-field tuple struct is a **constant**:
//!   `#[decision(id = .., constant(range(1..=8)), default = 4)]`.
//! - A unit struct is a **flag**: `#[decision(id = .., flag, default = false)]`.
//! - `shape(dynamic, ..; fixed, ..)` declares variables; each may say
//!   `: set(..)`, `: range(a..=b)`, `: enum(a, b)`, `: bool`, and a dynamic
//!   one `: derived`. A struct `<Type>At` with one field per variable is
//!   generated as the point the ladder is evaluated at; an integer variable
//!   that says `optional` is an `Option<i64>` there, `None` where unknown.
//! - `fits = path` names `fn(&Self, &<Type>At) -> bool` for partial
//!   alternatives.
//! - `tune = search | price | engine | never` says how `pie tune` treats
//!   the decision; without it, `never`.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as Tokens};
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    Attribute, Data, DeriveInput, Expr, ExprLit, Fields, Ident, Lit, LitStr, Path, Token, Type,
    parenthesized,
};

#[proc_macro_derive(
    Decision,
    attributes(decision, constant, requires, partial, consistent_with)
)]
pub fn derive_decision(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(t) => t.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn doc(attrs: &[Attribute]) -> String {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            syn::Meta::NameValue(nv) => match &nv.value {
                Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) => Some(s.value().trim().to_string()),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn snake(ident: &Ident) -> String {
    let mut out = String::new();
    for (i, c) in ident.to_string().chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// A domain or constant range spelled in an attribute.
#[derive(Clone)]
enum Values {
    Any,
    Set(Vec<i64>),
    Range(Box<(Expr, Expr)>),
    Names(Vec<String>),
    Bool,
}

fn parse_ints(input: ParseStream) -> syn::Result<Vec<i64>> {
    let content;
    parenthesized!(content in input);
    let items = Punctuated::<Expr, Token![,]>::parse_terminated(&content)?;
    items.iter().map(int_of).collect()
}

fn int_of(e: &Expr) -> syn::Result<i64> {
    match ungroup(e) {
        Expr::Lit(ExprLit {
            lit: Lit::Int(i), ..
        }) => i.base10_parse(),
        Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => Ok(-int_of(&u.expr)?),
        _ => Err(syn::Error::new_spanned(e, "expected an integer literal")),
    }
}

fn parse_names(input: ParseStream) -> syn::Result<Vec<String>> {
    let content;
    parenthesized!(content in input);
    let items = Punctuated::<Ident, Token![,]>::parse_terminated(&content)?;
    Ok(items
        .iter()
        .map(|i| i.to_string().trim_start_matches("r#").to_string())
        .collect())
}

fn parse_range(input: ParseStream) -> syn::Result<Values> {
    let content;
    parenthesized!(content in input);
    let range: syn::ExprRange = content.parse()?;
    if !matches!(range.limits, syn::RangeLimits::Closed(_)) {
        return Err(syn::Error::new_spanned(
            &range,
            "write an inclusive range, `lo..=hi`",
        ));
    }
    match (range.start, range.end) {
        (Some(lo), Some(hi)) => Ok(Values::Range(Box::new((*lo, *hi)))),
        _ => Err(syn::Error::new_spanned(
            Ident::new("range", Span::call_site()),
            "a range has both ends",
        )),
    }
}

/// `set(..)`, `range(..)`, or a list of names after one of `names`; `None`
/// for any other word.
fn parse_values(word: &str, names: &[&str], input: ParseStream) -> syn::Result<Option<Values>> {
    Ok(Some(match word {
        "set" => Values::Set(parse_ints(input)?),
        "range" => parse_range(input)?,
        w if names.contains(&w) => Values::Names(parse_names(input)?),
        _ => return Ok(None),
    }))
}

struct VarDecl {
    name: String,
    values: Values,
    dynamic: bool,
    derived: bool,
    optional: bool,
}

fn parse_var(input: ParseStream, dynamic: bool) -> syn::Result<VarDecl> {
    let name: Ident = input.parse()?;
    let mut var = VarDecl {
        name: name.to_string(),
        values: Values::Any,
        dynamic,
        derived: false,
        optional: false,
    };
    if input.peek(Token![:]) {
        input.parse::<Token![:]>()?;
        loop {
            let word: Ident = if input.peek(Token![enum]) {
                input.parse::<Token![enum]>()?;
                Ident::new("enum", Span::call_site())
            } else {
                input.parse()?
            };
            let w = word.to_string();
            match w.as_str() {
                "derived" => var.derived = true,
                "optional" => var.optional = true,
                "bool" => var.values = Values::Bool,
                _ => {
                    var.values = parse_values(&w, &["enum"], input)?.ok_or_else(|| {
                        syn::Error::new_spanned(
                            word,
                            "expected derived, optional, set(..), range(..), enum(..) or bool",
                        )
                    })?;
                }
            }
            if !(input.peek(Ident) || input.peek(Token![enum])) {
                break;
            }
        }
    }
    if var.optional && matches!(var.values, Values::Names(_) | Values::Bool) {
        return Err(syn::Error::new_spanned(
            name,
            "only an integer variable is optional",
        ));
    }
    Ok(var)
}

struct DecisionAttr {
    id: LitStr,
    vars: Vec<VarDecl>,
    default: Option<Expr>,
    fits: Option<Path>,
    flag: bool,
    constant: Option<Values>,
    tune: Option<Ident>,
}

impl Parse for DecisionAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut attr = DecisionAttr {
            id: LitStr::new("", Span::call_site()),
            vars: Vec::new(),
            default: None,
            fits: None,
            flag: false,
            constant: None,
            tune: None,
        };
        let mut seen_id = false;
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            match key.to_string().as_str() {
                "id" => {
                    input.parse::<Token![=]>()?;
                    attr.id = input.parse()?;
                    seen_id = true;
                }
                "default" => {
                    input.parse::<Token![=]>()?;
                    attr.default = Some(input.parse()?);
                }
                "fits" => {
                    input.parse::<Token![=]>()?;
                    attr.fits = Some(input.parse()?);
                }
                "flag" => attr.flag = true,
                "tune" => {
                    input.parse::<Token![=]>()?;
                    let word: Ident = input.parse()?;
                    if !["search", "price", "engine", "never"].contains(&word.to_string().as_str())
                    {
                        return Err(syn::Error::new_spanned(
                            word,
                            "expected search, price, engine or never",
                        ));
                    }
                    attr.tune = Some(word);
                }
                "constant" => {
                    let content;
                    parenthesized!(content in input);
                    let mut values = Values::Any;
                    if !content.is_empty() {
                        let word: Ident = content.parse()?;
                        values = parse_values(&word.to_string(), &["word", "words"], &content)?
                            .ok_or_else(|| {
                                syn::Error::new_spanned(
                                    word,
                                    "expected set(..), range(..) or word(..)",
                                )
                            })?;
                    }
                    attr.constant = Some(values);
                }
                "shape" => {
                    let content;
                    parenthesized!(content in input);
                    // Variables before `;` are dynamic; without `;`, all are fixed.
                    let mut dynamic = true;
                    while !content.is_empty() {
                        if content.peek(Token![;]) {
                            content.parse::<Token![;]>()?;
                            dynamic = false;
                            continue;
                        }
                        attr.vars.push(parse_var(&content, dynamic)?);
                        if content.peek(Token![,]) {
                            content.parse::<Token![,]>()?;
                        }
                    }
                    if dynamic {
                        for v in &mut attr.vars {
                            v.dynamic = false;
                        }
                    }
                }
                _ => {
                    return Err(syn::Error::new_spanned(
                        key,
                        "expected id, shape, default, fits, flag, tune or constant(..)",
                    ));
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        if !seen_id {
            return Err(input.error("a decision needs `id = \"domain.name\"`"));
        }
        Ok(attr)
    }
}

fn domain_tokens(values: &Values) -> Tokens {
    match values {
        Values::Any => quote!(::profile::Domain::Int {
            lo: 0,
            hi: i64::MAX
        }),
        Values::Set(v) => quote!(::profile::Domain::Ints(&[#(#v),*])),
        Values::Range(r) => {
            let (lo, hi) = &**r;
            quote!(::profile::Domain::Int { lo: #lo, hi: #hi })
        }
        Values::Names(n) => quote!(::profile::Domain::Enum(&[#(#n),*])),
        Values::Bool => quote!(::profile::Domain::Bool),
    }
}

/// `e` without the invisible group a `macro_rules!` fragment wraps it in.
fn ungroup(e: &Expr) -> &Expr {
    match e {
        Expr::Group(g) => ungroup(&g.expr),
        Expr::Paren(p) => ungroup(&p.expr),
        other => other,
    }
}

/// A Rust literal as the TOML a profile would spell.
fn toml_of(e: &Expr) -> syn::Result<String> {
    match ungroup(e) {
        Expr::Lit(ExprLit { lit, .. }) => match lit {
            Lit::Int(i) => Ok(i.base10_digits().to_string()),
            Lit::Float(f) => {
                let d = f.base10_digits().to_string();
                Ok(if d.contains('.') || d.contains('e') {
                    d
                } else {
                    format!("{d}.0")
                })
            }
            Lit::Bool(b) => Ok(b.value.to_string()),
            Lit::Str(s) => Ok(format!("\"{}\"", s.value())),
            _ => Err(syn::Error::new_spanned(
                e,
                "a default is an integer, float, bool or string",
            )),
        },
        Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => {
            Ok(format!("-{}", toml_of(&u.expr)?))
        }
        Expr::Array(a) => Ok(format!(
            "[{}]",
            a.elems
                .iter()
                .map(toml_of)
                .collect::<syn::Result<Vec<_>>>()?
                .join(", ")
        )),
        _ => Err(syn::Error::new_spanned(e, "a default is a literal")),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Ty {
    Int,
    Float,
    Bool,
    StaticStr,
    Owned,
    Words,
}

fn ty_of(ty: &Type) -> syn::Result<Ty> {
    let text = quote!(#ty).to_string().replace(' ', "");
    let t = match text.as_str() {
        "u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize" => Ty::Int,
        "f32" | "f64" => Ty::Float,
        "bool" => Ty::Bool,
        "&'staticstr" => Ty::StaticStr,
        "String" => Ty::Owned,
        "Vec<String>" => Ty::Words,
        _ => {
            return Err(syn::Error::new_spanned(
                ty,
                "a constant is an integer, float, bool, &'static str, String or Vec<String>",
            ));
        }
    };
    Ok(t)
}

struct ConstDecl {
    spec: Tokens,
    convert: Tokens,
}

/// A constant's spec and the expression that reads it from `v: &Value`.
fn constant(
    name: &str,
    doc: &str,
    ty: &Type,
    values: Values,
    default: Option<&Expr>,
    sort: Option<bool>,
    span_of: &dyn quote::ToTokens,
) -> syn::Result<ConstDecl> {
    let t = ty_of(ty)?;
    let (kind, specialize) = match (&values, t) {
        (Values::Set(v), Ty::Int) => (quote!(::profile::Kind::Ints(&[#(#v),*])), true),
        (Values::Range(r), Ty::Int) => {
            let (lo, hi) = &**r;
            (quote!(::profile::Kind::Int { lo: #lo, hi: #hi }), false)
        }
        (Values::Any, Ty::Int) => (
            quote!(::profile::Kind::Int {
                lo: 0,
                hi: i64::MAX
            }),
            false,
        ),
        (Values::Range(r), Ty::Float) => {
            let (lo, hi) = &**r;
            (
                quote!(::profile::Kind::Float { lo: #lo as f64, hi: #hi as f64 }),
                false,
            )
        }
        (Values::Any, Ty::Float) => (
            quote!(::profile::Kind::Float {
                lo: f64::NEG_INFINITY,
                hi: f64::INFINITY
            }),
            false,
        ),
        (Values::Any, Ty::Bool) => (quote!(::profile::Kind::Bool), true),
        (Values::Names(n), Ty::StaticStr | Ty::Owned) => {
            (quote!(::profile::Kind::Word(&[#(#n),*])), true)
        }
        (Values::Names(n), Ty::Words) => (quote!(::profile::Kind::Words(&[#(#n),*])), false),
        _ => {
            return Err(syn::Error::new_spanned(
                span_of,
                "these values do not fit this field's type",
            ));
        }
    };
    let specialize = sort.unwrap_or(specialize);
    let sort_tokens = if specialize {
        quote!(::profile::Sort::Specialize)
    } else {
        quote!(::profile::Sort::Launch)
    };
    // A constant decision's default is the decision's; an alternative's
    // constant carries its own.
    let default = match default {
        Some(e) => toml_of(e)?,
        None if name == "value" => String::new(),
        None => {
            return Err(syn::Error::new_spanned(
                span_of,
                "a constant needs `default = <literal>`",
            ));
        }
    };
    let convert = match t {
        Ty::Int => {
            quote!(<#ty>::try_from(v.as_int()).expect("a registry-checked integer fits its field"))
        }
        Ty::Float => quote!(v.as_float() as #ty),
        Ty::Bool => quote!(v.as_bool()),
        Ty::Owned => quote!(v.as_word().to_string()),
        Ty::Words => quote!(v.as_words()),
        Ty::StaticStr => {
            let Values::Names(n) = &values else {
                unreachable!()
            };
            quote!({
                const NAMES: &[&str] = &[#(#n),*];
                NAMES.iter().copied().find(|n| *n == v.as_word()).expect("a registry-checked word")
            })
        }
    };
    Ok(ConstDecl {
        spec: quote!(::profile::Constant {
            name: #name,
            kind: #kind,
            sort: #sort_tokens,
            doc: #doc,
            default: #default,
        }),
        convert,
    })
}

/// `#[constant(...)]` on a field.
fn field_constant(field: &syn::Field) -> syn::Result<ConstDecl> {
    let name = field.ident.as_ref().expect("named").to_string();
    let mut values = Values::Any;
    let mut default = None;
    let mut sort = None;
    for attr in field.attrs.iter().filter(|a| a.path().is_ident("constant")) {
        attr.parse_args_with(|input: ParseStream| {
            while !input.is_empty() {
                let key: Ident = input.parse()?;
                let word = key.to_string();
                match word.as_str() {
                    "default" => {
                        input.parse::<Token![=]>()?;
                        default = Some(input.parse::<Expr>()?);
                    }
                    "launch" => sort = Some(false),
                    "specialize" => sort = Some(true),
                    _ => {
                        values = parse_values(&word, &["word", "words"], input)?.ok_or_else(|| {
                            syn::Error::new_spanned(
                                key,
                                "expected set, range, word, words, default, launch or specialize",
                            )
                        })?;
                    }
                }
                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                }
            }
            Ok(())
        })?;
    }
    constant(
        &name,
        &doc(&field.attrs),
        &field.ty,
        values,
        default.as_ref(),
        sort,
        field,
    )
}

fn str_attr(attrs: &[Attribute], name: &str) -> syn::Result<Vec<String>> {
    attrs
        .iter()
        .filter(|a| a.path().is_ident(name))
        .map(|a| a.parse_args::<LitStr>().map(|s| s.value()))
        .collect()
}

fn expand(input: &DeriveInput) -> syn::Result<Tokens> {
    let attr = input
        .attrs
        .iter()
        .find(|a| a.path().is_ident("decision"))
        .ok_or_else(|| {
            syn::Error::new_spanned(&input.ident, "add #[decision(id = \"domain.name\", ...)]")
        })?
        .parse_args::<DecisionAttr>()?;
    let ident = &input.ident;
    let vis = &input.vis;
    let id = &attr.id;
    let type_doc = doc(&input.attrs);

    let var_specs = attr.vars.iter().map(|v| {
        let name = &v.name;
        let domain = domain_tokens(&v.values);
        let (dynamic, derived) = (v.dynamic, v.derived);
        quote!(::profile::Var { name: #name, domain: #domain, dynamic: #dynamic, derived: #derived })
    });

    let at_ident = format_ident!("{}At", ident);
    let (at_type, at_def) = if attr.vars.is_empty() {
        (quote!(::profile::Anywhere), quote!())
    } else {
        let fields = attr.vars.iter().map(|v| {
            let f = format_ident!("{}", v.name);
            match v.values {
                Values::Names(_) => quote!(pub #f: &'static str),
                Values::Bool => quote!(pub #f: bool),
                _ if v.optional => quote!(pub #f: Option<i64>),
                _ => quote!(pub #f: i64),
            }
        });
        let arms = attr.vars.iter().enumerate().map(|(i, v)| {
            let f = format_ident!("{}", v.name);
            match &v.values {
                Values::Names(n) => quote!(#i => Some({
                    const NAMES: &[&str] = &[#(#n),*];
                    NAMES.iter().position(|n| *n == self.#f)
                        .unwrap_or_else(|| panic!("`{}` is not a value of `{}`", self.#f, stringify!(#f))) as i64
                })),
                Values::Bool => quote!(#i => Some(i64::from(self.#f))),
                _ if v.optional => quote!(#i => self.#f),
                _ => quote!(#i => Some(self.#f)),
            }
        });
        let doc = format!("The shape `{}` is chosen at.", id.value());
        (
            quote!(#at_ident),
            quote! {
                #[doc = #doc]
                #[derive(Clone, Copy, Debug, PartialEq, Eq)]
                #vis struct #at_ident { #(#fields),* }
                impl ::profile::Point for #at_ident {
                    fn coord(&self, var: usize) -> Option<i64> {
                        match var { #(#arms,)* _ => unreachable!("no variable {var}") }
                    }
                }
            },
        )
    };

    let (shape, build, default_toml) = match &input.data {
        Data::Enum(data) => {
            if attr.flag || attr.constant.is_some() {
                return Err(syn::Error::new_spanned(
                    ident,
                    "an enum is a choice; flags and constants are structs",
                ));
            }
            let mut alts = Vec::new();
            let mut arms = Vec::new();
            for (index, variant) in data.variants.iter().enumerate() {
                let name = snake(&variant.ident);
                let vdoc = doc(&variant.attrs);
                let requires = str_attr(&variant.attrs, "requires")?.join(" and ");
                let partial = variant.attrs.iter().any(|a| a.path().is_ident("partial"));
                let consistent = str_attr(&variant.attrs, "consistent_with")?;
                let vident = &variant.ident;
                let (consts, build) = match &variant.fields {
                    Fields::Unit => (Vec::new(), quote!(Self::#vident)),
                    Fields::Named(named) => {
                        let mut specs = Vec::new();
                        let mut inits = Vec::new();
                        for (i, field) in named.named.iter().enumerate() {
                            let c = field_constant(field)?;
                            specs.push(c.spec);
                            let f = field.ident.as_ref().expect("named");
                            let convert = c.convert;
                            inits.push(quote!(#f: { let v = &constants[#i]; #convert }));
                        }
                        (specs, quote!(Self::#vident { #(#inits),* }))
                    }
                    Fields::Unnamed(_) => {
                        return Err(syn::Error::new_spanned(
                            variant,
                            "an alternative's constants are named fields",
                        ));
                    }
                };
                alts.push(quote!(::profile::Alternative {
                    name: #name,
                    doc: #vdoc,
                    constants: &[#(#consts),*],
                    requires: #requires,
                    partial: #partial,
                    consistent_with: &[#(#consistent),*],
                }));
                arms.push(
                    quote!(::profile::Pick::Alternative { index: #index, constants } => {
                        let _ = constants;
                        #build
                    }),
                );
            }
            let default = match &attr.default {
                Some(Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                })) => {
                    let v = s.value();
                    if !v.is_empty() && v.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                        format!("\"{v}\"")
                    } else {
                        v
                    }
                }
                Some(e) => {
                    return Err(syn::Error::new_spanned(
                        e,
                        "a choice's default is a string: an alternative name, or TOML",
                    ));
                }
                None => {
                    return Err(syn::Error::new_spanned(
                        ident,
                        "a choice needs `default = \"..\"`",
                    ));
                }
            };
            (
                quote!(::profile::Shape::Choice(&[#(#alts),*])),
                quote!(match pick { #(#arms)* other => unreachable!("`{}` cannot build from {other:?}", #id) }),
                default,
            )
        }
        Data::Struct(data) => match (&data.fields, attr.flag, &attr.constant) {
            (Fields::Unit, true, None) => {
                let default = match attr.default.as_ref().map(ungroup) {
                    None => "false".to_string(),
                    Some(Expr::Lit(ExprLit {
                        lit: Lit::Bool(b), ..
                    })) => b.value.to_string(),
                    Some(Expr::Lit(ExprLit {
                        lit: Lit::Str(s), ..
                    })) => format!("\"{}\"", s.value()),
                    Some(e) => {
                        return Err(syn::Error::new_spanned(
                            e,
                            "a flag's default is a bool or a `when` string",
                        ));
                    }
                };
                (
                    quote!(::profile::Shape::Flag),
                    quote!(match pick {
                        ::profile::Pick::On => Self,
                        other => unreachable!("flag `{}` cannot build from {other:?}", #id),
                    }),
                    default,
                )
            }
            (Fields::Unnamed(f), false, Some(values)) if f.unnamed.len() == 1 => {
                let field = &f.unnamed[0];
                let values = values.clone();
                let default = match &attr.default {
                    Some(Expr::Lit(ExprLit {
                        lit: Lit::Str(s), ..
                    })) if s.value().trim_start().starts_with('[') => s.value(),
                    Some(e) => toml_of(e)?,
                    None => {
                        return Err(syn::Error::new_spanned(
                            ident,
                            "a constant needs `default = <literal>`, or a ladder as a TOML string",
                        ));
                    }
                };
                let c = constant("value", "", &field.ty, values, None, None, field)?;
                let spec = c.spec;
                let convert = c.convert;
                (
                    quote!(::profile::Shape::Constant(#spec)),
                    quote!(match pick { ::profile::Pick::Value(v) => Self(#convert), other => unreachable!("`{}` cannot build from {other:?}", #id) }),
                    default,
                )
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    ident,
                    "a struct decision is a unit struct with `flag`, or a one-field tuple struct with `constant(..)`",
                ));
            }
        },
        Data::Union(_) => {
            return Err(syn::Error::new_spanned(
                ident,
                "a union cannot be a decision",
            ));
        }
    };

    let tune = match attr.tune.as_ref().map(|w| w.to_string()).as_deref() {
        Some("search") => quote!(::profile::Tune::Search),
        Some("price") => quote!(::profile::Tune::Price),
        Some("engine") => quote!(::profile::Tune::Engine),
        _ => quote!(::profile::Tune::Never),
    };

    let fits = match &attr.fits {
        Some(path) => quote!(fn fits(&self, at: &Self::At) -> bool { #path(self, at) }),
        None => quote!(),
    };

    Ok(quote! {
        #at_def
        impl ::profile::Decision for #ident {
            type At = #at_type;

            fn spec() -> &'static ::profile::Spec {
                static SPEC: ::profile::Spec = ::profile::Spec {
                    id: #id,
                    doc: #type_doc,
                    shape: #shape,
                    vars: &[#(#var_specs),*],
                    default: #default_toml,
                    tune: #tune,
                };
                &SPEC
            }

            fn build(pick: &::profile::Pick) -> Self {
                #build
            }

            #fits
        }
    })
}
