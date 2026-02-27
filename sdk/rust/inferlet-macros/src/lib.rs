//! Procedural macros for the inferlet library.
//!
//! Provides the `#[inferlet::main]` attribute macro for defining inferlet entry points.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{parse_macro_input, FnArg, GenericArgument, ItemFn, PathArguments, PatType, Type};

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
