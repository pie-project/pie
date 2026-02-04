//! Procedural macros for the inferlet library.
//!
//! Provides the `#[inferlet::main]` attribute macro for defining inferlet entry points.

use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

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

    // Rename the user's function so that we can call it from our generated code.
    input_fn.sig.ident = inner_fn_name.clone();

    // Generate a wrapper type that implements the Run trait.
    // The new WIT signature: run(args: list<string>) -> result<string, error>
    let expanded = quote! {
        #input_fn

        struct __PieMain;

        impl inferlet::exports::pie::core::run::Guest for __PieMain {
            fn run(args: Vec<String>) -> std::result::Result<String, String> {
                inferlet::wstd::runtime::block_on(async { #inner_fn_name(args).await })
            }
        }

        inferlet::export!(__PieMain with_types_in inferlet);
    };

    expanded.into()
}
