//! Procedural macros for the inferlet library.
//!
//! Provides the `#[inferlet::main]` attribute macro for defining inferlet entry points.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

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

/// Converts a package name like "text-completion" to a valid Rust identifier "text_completion"
fn to_rust_ident(name: &str) -> syn::Ident {
    let sanitized = name.replace('-', "_");
    syn::Ident::new(&sanitized, Span::call_site())
}

/// Marks an async function as the inferlet entry point.
///
/// The function should have the signature:
/// ```ignore
/// async fn main(args: Vec<String>) -> inferlet::Result<String>
/// ```
///
/// # Example
/// ```ignore
/// use inferlet::Result;
///
/// #[inferlet::main]
/// async fn main(args: Vec<String>) -> Result<String> {
///     Ok("Hello, world!".to_string())
/// }
/// ```
#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let mut input_fn = parse_macro_input!(item as ItemFn);
    let original_fn_name = input_fn.sig.ident.clone();
    let inner_fn_name = syn::Ident::new("__pie_main_inner", original_fn_name.span());

    // Ensure the function is async
    if input_fn.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            input_fn.sig.ident,
            "The #[inferlet::main] attribute can only be used on async functions",
        )
        .to_compile_error()
        .into();
    }

    // Read package name from Pie.toml
    let package_name = match read_package_name() {
        Ok(name) => name,
        Err(e) => {
            return syn::Error::new(Span::call_site(), e)
                .to_compile_error()
                .into();
        }
    };
    let package_ident = to_rust_ident(&package_name);

    // Generate inline WIT for the export interface.
    // Each inferlet exports: pie:{package_name}/run
    let export_wit = format!(
        r#"
package pie:{package_name};

interface run {{
    run: func(args: list<string>) -> result<string, string>;
}}

world inferlet {{
    export run;
}}
"#
    );

    // Rename the user's function so that we can call it from our generated code.
    input_fn.sig.ident = inner_fn_name.clone();

    // Generate a wrapper type that implements the Guest trait from the generated export bindings.
    let expanded = quote! {
        // Generate export bindings with package-specific namespace
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
            fn run(args: Vec<String>) -> std::result::Result<String, String> {
                let result = inferlet::wstd::runtime::block_on(async { #inner_fn_name(args).await });
                let _ = std::io::Write::flush(&mut std::io::stdout());
                let _ = std::io::Write::flush(&mut std::io::stderr());
                result
            }
        }

        __pie_export::export!(__PieMain with_types_in __pie_export);
    };

    expanded.into()
}
