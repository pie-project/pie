use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Type, parse_macro_input};

#[proc_macro_derive(Facts)]
pub fn derive_facts(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let Data::Struct(data) = &input.data else {
        return error(name, "Facts derives only on a struct of bool fields");
    };
    let Fields::Named(fields) = &data.fields else {
        return error(name, "Facts derives only on named fields");
    };

    let mut constructors = Vec::new();
    let mut bits = Vec::new();
    let mut names = Vec::new();
    for (i, field) in fields.named.iter().enumerate() {
        if !matches!(&field.ty, Type::Path(p) if p.path.is_ident("bool")) {
            return error(name, "every Facts field must be a bool");
        }
        let ident = field.ident.as_ref().unwrap();
        let bit = i as u8;
        let fact = ident.to_string();
        constructors.push(quote! {
            #[must_use]
            pub fn #ident() -> ::model_dsl::Predicate {
                ::model_dsl::Predicate::fact(#bit, #fact)
            }
        });
        bits.push(quote! { |= (self.#ident as u64) << #bit });
        names.push(fact.clone());
    }

    quote! {
        impl #name {
            #(#constructors)*
        }

        impl ::model_dsl::FactWord for #name {
            const NAMES: &'static [&'static str] = &[#(#names),*];

            fn word(&self) -> u64 {
                let mut word = 0u64;
                #(word #bits;)*
                word
            }
        }
    }
    .into()
}

fn error(name: &syn::Ident, message: &str) -> TokenStream {
    syn::Error::new(name.span(), message)
        .to_compile_error()
        .into()
}
