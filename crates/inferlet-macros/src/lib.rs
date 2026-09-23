use proc_macro::TokenStream;
use quote::quote;
use syn::{FnArg, GenericArgument, ItemFn, PatType, PathArguments, Type, parse_macro_input};

fn is_string(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.is_ident("String"))
}

fn result_inner(ty: &Type) -> Option<&Type> {
    let Type::Path(p) = ty else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "Result" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    match args.args.first()? {
        GenericArgument::Type(inner) => Some(inner),
        _ => None,
    }
}

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

    let first_param_ty = input_fn.sig.inputs.first().and_then(|arg| {
        if let FnArg::Typed(PatType { ty, .. }) = arg {
            Some(ty.as_ref())
        } else {
            None
        }
    });
    let typed_input = first_param_ty.is_some_and(|ty| !is_string(ty));

    let typed_output = match &input_fn.sig.output {
        syn::ReturnType::Type(_, ty) => result_inner(ty).is_some_and(|t| !is_string(t)),
        _ => false,
    };

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

    input_fn.sig.ident = inner_fn_name.clone();

    let expanded = quote! {
        #input_fn

        const __PIE_PACKAGE_TEXT: &str =
            concat!(env!("CARGO_PKG_NAME"), "@", env!("CARGO_PKG_VERSION"));
        #[used]
        #[unsafe(link_section = "pie.package")]
        static __PIE_PACKAGE: [u8; __PIE_PACKAGE_TEXT.len()] = {
            let text = __PIE_PACKAGE_TEXT.as_bytes();
            let mut bytes = [0u8; __PIE_PACKAGE_TEXT.len()];
            let mut i = 0;
            while i < bytes.len() {
                bytes[i] = text[i];
                i += 1;
            }
            bytes
        };

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
