//! Procedural macros for the inferlet library.
//!
//! Provides the `#[inferlet::main]` attribute macro for defining inferlet entry points.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, Attribute, Expr, ExprLit, FnArg, GenericArgument, Ident, ItemFn, Lit,
    LitStr, Meta, Pat, PathArguments, PatType, Type,
};

/// Reads the package name from Pie.toml.
fn read_package_name() -> Result<String, String> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map_err(|_| "CARGO_MANIFEST_DIR not set".to_string())?;

    let pie_toml_path = std::path::PathBuf::from(&manifest_dir).join("Pie.toml");
    let pie_toml_content = std::fs::read_to_string(&pie_toml_path).map_err(|_| {
        "Failed to read Pie.toml - make sure it exists next to Cargo.toml".to_string()
    })?;

    let pie_config: toml::Value = pie_toml_content
        .parse()
        .map_err(|e| format!("Failed to parse Pie.toml: {e}"))?;

    pie_config["package"]["name"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "Missing [package].name in Pie.toml".to_string())
}

/// Converts a package name like "text-completion" to a valid Rust identifier "text_completion".
fn to_rust_ident(name: &str) -> syn::Ident {
    syn::Ident::new(&name.replace('-', "_"), Span::call_site())
}

/// Returns `true` if `ty` is exactly `String`.
fn is_string(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.is_ident("String"))
}

/// Extracts the inner type `T` from `Result<T>` or `Result<T, E>`.
fn result_inner(ty: &Type) -> Option<&Type> {
    let Type::Path(p) = ty else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "Result" { return None; }
    let PathArguments::AngleBracketed(args) = &seg.arguments else { return None };
    match args.args.first()? {
        GenericArgument::Type(inner) => Some(inner),
        _ => None,
    }
}

/// Marks an async function as the inferlet entry point.
///
/// The macro inspects the function signature and generates the appropriate
/// JSON serialization bridge:
///
/// - **Input**: if the parameter type is not `String`, the raw JSON input
///   string is deserialized via `serde_json::from_str`.
/// - **Output**: if the `Result<T>` inner type is not `String`, the return
///   value is serialized via `serde_json::to_string`.
///
/// All four combinations of typed/raw input × typed/raw output are supported.
///
/// ```ignore
/// #[inferlet::main]
/// async fn main(input: MyInput) -> Result<MyOutput> { .. }
///
/// #[inferlet::main]
/// async fn main(input: String) -> Result<String> { .. }
/// ```
#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input_fn = parse_macro_input!(item as ItemFn);
    let inner_fn_name = syn::Ident::new("__pie_main_inner", input_fn.sig.ident.span());

    if input_fn.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            input_fn.sig.ident,
            "#[inferlet::main] can only be used on async functions",
        )
        .to_compile_error()
        .into();
    }

    let package_name = match read_package_name() {
        Ok(n) => n,
        Err(e) => return syn::Error::new(Span::call_site(), e).to_compile_error().into(),
    };
    let package_ident = to_rust_ident(&package_name);

    // Inline WIT for this inferlet's export interface
    let export_wit = format!(
        r#"
package pie:{package_name};

interface run {{
    run: func(input: string) -> result<string, string>;
}}

world inferlet {{
    export run;
}}
"#
    );

    // --- Detect input/output conventions ---

    let first_param_ty = input_fn.sig.inputs.first().and_then(|arg| {
        if let FnArg::Typed(PatType { ty, .. }) = arg { Some(ty.as_ref()) } else { None }
    });
    let typed_input = first_param_ty.map_or(false, |ty| !is_string(ty));

    let typed_output = match &input_fn.sig.output {
        syn::ReturnType::Type(_, ty) => result_inner(ty).map_or(false, |t| !is_string(t)),
        _ => false,
    };

    // --- Build code-gen fragments ---

    // Deserialization runs in sync context (before block_on) to avoid 'static issues.
    let input_prep = if typed_input {
        quote! {
            let typed_input = ::inferlet::serde_json::from_str(&input)
                .map_err(|e| format!("Failed to parse JSON input: {e}"))?;
        }
    } else {
        quote! { let typed_input = input; }
    };

    let output_transform = if typed_output {
        quote! {
            match result {
                Ok(v) => ::inferlet::serde_json::to_string(&v)
                    .map_err(|e| format!("Failed to serialize output: {e}")),
                Err(e) => Err(e),
            }
        }
    } else {
        quote! { result }
    };

    // Rename user's function so we can wrap it
    input_fn.sig.ident = inner_fn_name.clone();

    let expanded = quote! {
        mod __pie_export {
            ::inferlet::wit_bindgen::generate!({
                inline: #export_wit,
                world: "inferlet",
                pub_export_macro: true,
                runtime_path: "::inferlet::wit_bindgen::rt",
            });
        }

        #input_fn

        struct __PieMain;

        impl __pie_export::exports::pie::#package_ident::run::Guest for __PieMain {
            fn run(input: String) -> std::result::Result<String, String> {
                #input_prep
                let result = inferlet::wstd::runtime::block_on(async {
                    #inner_fn_name(typed_input).await
                });
                let _ = std::io::Write::flush(&mut std::io::stdout());
                let _ = std::io::Write::flush(&mut std::io::stderr());
                #output_transform
            }
        }

        __pie_export::export!(__PieMain with_types_in __pie_export);
    };

    expanded.into()
}

// =============================================================================
// `#[tool]` — derive a `Tool` impl from an async fn signature
// =============================================================================

/// Maps a primitive Rust type to a JSON Schema fragment. Returns `None`
/// for unsupported types — the caller emits a compile error pointing at
/// the offending param.
fn primitive_to_schema(ty: &Type) -> Option<&'static str> {
    let Type::Path(p) = ty else { return None };
    let seg = p.path.segments.last()?;
    if !seg.arguments.is_empty() {
        return None;
    }
    Some(match seg.ident.to_string().as_str() {
        "String" => r#"{"type":"string"}"#,
        "bool" => r#"{"type":"boolean"}"#,
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize" => {
            r#"{"type":"integer"}"#
        }
        "f32" | "f64" => r#"{"type":"number"}"#,
        _ => return None,
    })
}

/// Extracts the description text. Prefers `#[tool("...")]` attribute arg;
/// otherwise concatenates `///` doc lines.
fn extract_description(attr_args: &TokenStream2, attrs: &[Attribute]) -> Result<String, String> {
    if !attr_args.is_empty() {
        let lit: LitStr = syn::parse2(attr_args.clone())
            .map_err(|_| "expected a string literal as #[tool(...)] argument".to_string())?;
        return Ok(lit.value());
    }

    let mut lines: Vec<String> = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        let Meta::NameValue(nv) = &attr.meta else { continue };
        let Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) = &nv.value else { continue };
        let line = s.value();
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            lines.push(trimmed.to_string());
        }
    }
    if lines.is_empty() {
        return Err(
            "no description: add a `///` doc comment or `#[tool(\"...\")]` argument".into(),
        );
    }
    Ok(lines.join(" "))
}

/// Marks an async fn as a tool. Generates a unit struct of the same name
/// that implements `inferlet::tools::Tool`, plus `call(args: &str)` and
/// `call_typed(...)` associated functions.
///
/// The description comes from the `///` doc comment, or from
/// `#[tool("override")]` if you need it to differ from the rustdoc.
///
/// Param types must be `String`, `bool`, an integer (`i32`, `u64`, …),
/// or a float (`f32`, `f64`). Anything richer needs a hand-written
/// `Tool` impl.
///
/// ```ignore
/// /// Search the web for current information.
/// #[tool]
/// async fn web_search(query: String) -> Result<String> { /* body */ }
///
/// // Static metadata:
/// assert_eq!(<web_search as inferlet::tools::Tool>::name(&web_search), "web_search");
///
/// // JSON dispatch (used by tool-call dispatchers):
/// let _ = web_search::call(r#"{"query": "rust"}"#).await?;
///
/// // Typed dispatch (direct invocation):
/// let _ = web_search::call_typed("rust".into()).await?;
/// ```
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr2: TokenStream2 = attr.into();
    let input_fn = parse_macro_input!(item as ItemFn);

    // ── Validate signature ───────────────────────────────────────────────
    if input_fn.sig.asyncness.is_none() {
        return syn::Error::new_spanned(&input_fn.sig.ident, "#[tool] requires an async fn")
            .to_compile_error()
            .into();
    }
    if !input_fn.sig.generics.params.is_empty() {
        return syn::Error::new_spanned(
            &input_fn.sig.generics,
            "#[tool] does not support generic fns",
        )
        .to_compile_error()
        .into();
    }

    // ── Description ──────────────────────────────────────────────────────
    let description = match extract_description(&attr2, &input_fn.attrs) {
        Ok(d) => d,
        Err(e) => {
            return syn::Error::new_spanned(&input_fn.sig.ident, e)
                .to_compile_error()
                .into();
        }
    };

    // ── Param walk ───────────────────────────────────────────────────────
    let mut param_idents: Vec<Ident> = Vec::new();
    let mut param_types: Vec<Box<Type>> = Vec::new();
    let mut schema_props: Vec<String> = Vec::new();
    let mut schema_required: Vec<String> = Vec::new();

    for arg in &input_fn.sig.inputs {
        let pt = match arg {
            FnArg::Receiver(_) => {
                return syn::Error::new_spanned(arg, "#[tool] does not support `self` parameters")
                    .to_compile_error()
                    .into();
            }
            FnArg::Typed(pt) => pt,
        };
        let ident = match &*pt.pat {
            Pat::Ident(pi) => pi.ident.clone(),
            other => {
                return syn::Error::new_spanned(other, "#[tool] params must be plain identifiers")
                    .to_compile_error()
                    .into();
            }
        };
        let frag = match primitive_to_schema(&pt.ty) {
            Some(f) => f,
            None => {
                return syn::Error::new_spanned(
                    &pt.ty,
                    "#[tool] supports only primitive params (String, bool, integer, float)",
                )
                .to_compile_error()
                .into();
            }
        };
        let name_str = ident.to_string();
        schema_props.push(format!(r#""{}":{}"#, name_str, frag));
        schema_required.push(format!(r#""{}""#, name_str));
        param_idents.push(ident);
        param_types.push(pt.ty.clone());
    }

    let schema_str = format!(
        r#"{{"type":"object","properties":{{{}}},"required":[{}]}}"#,
        schema_props.join(","),
        schema_required.join(","),
    );

    // ── Code-gen ─────────────────────────────────────────────────────────
    let fn_ident = input_fn.sig.ident.clone();
    let fn_name_str = fn_ident.to_string();
    let body = input_fn.block;
    let ret = input_fn.sig.output;
    let args_struct = format_ident!("__{}_Args", fn_ident);

    // Reconstruct the typed param list for call_typed.
    let typed_params: Vec<TokenStream2> = param_idents
        .iter()
        .zip(param_types.iter())
        .map(|(id, ty)| quote! { #id: #ty })
        .collect();

    let expanded = quote! {
        #[allow(non_camel_case_types)]
        pub struct #fn_ident;

        impl ::inferlet::tools::Tool for #fn_ident {
            fn name(&self) -> &'static str { #fn_name_str }
            fn description(&self) -> &'static str { #description }
            fn schema(&self) -> &'static str { #schema_str }
        }

        #[allow(non_camel_case_types)]
        #[derive(::inferlet::serde::Deserialize)]
        #[serde(crate = "::inferlet::serde")]
        struct #args_struct {
            #( #param_idents: #param_types, )*
        }

        impl #fn_ident {
            /// Invoke with JSON args. Used by tool-call dispatchers.
            pub async fn call(args: &str) -> ::inferlet::Result<String> {
                let parsed: #args_struct = ::inferlet::serde_json::from_str(args)
                    .map_err(|e| format!(concat!("tool `", #fn_name_str, "` arg parse: {}"), e))?;
                Self::call_typed(#( parsed.#param_idents ),*).await
            }

            /// Invoke with typed args — same shape as the original fn.
            pub async fn call_typed(#( #typed_params ),*) #ret #body
        }
    };

    expanded.into()
}
