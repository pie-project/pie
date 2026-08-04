//! Procedural macros for the inferlet library.
//!
//! Provides the `#[inferlet::main]` attribute macro for defining inferlet entry points.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, FnArg, GenericArgument, ItemFn, PatType, PathArguments, Type};

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

    // Deserialize the JSON input into the user's typed parameter.
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
        #input_fn

        struct __PieMain;

        impl ::inferlet::exports::pie::inferlet::run::Guest for __PieMain {
            async fn run(input: String) -> std::result::Result<String, String> {
                #input_prep
                let result = #inner_fn_name(typed_input).await;
                let _ = std::io::Write::flush(&mut std::io::stdout());
                let _ = std::io::Write::flush(&mut std::io::stderr());
                #output_transform
            }
        }

        ::inferlet::export!(__PieMain with_types_in ::inferlet);
    };

    expanded.into()
}

