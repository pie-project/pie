mod claims;
mod points;

use proc_macro::TokenStream;
use syn::{Error, ItemImpl, ItemTrait, parse_macro_input};

#[proc_macro_attribute]
pub fn points(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return no_arguments("points", attr);
    }
    let item = parse_macro_input!(item as ItemTrait);
    points::expand(item)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

#[proc_macro_attribute]
pub fn claims(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return no_arguments("claims", attr);
    }
    let item = parse_macro_input!(item as ItemImpl);
    claims::expand(item)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn no_arguments(attr_name: &str, attr: TokenStream) -> TokenStream {
    Error::new(
        proc_macro2::TokenStream::from(attr)
            .into_iter()
            .next()
            .map_or_else(proc_macro2::Span::call_site, |t| t.span()),
        format!("`#[{attr_name}]` reads the item below it and takes no arguments"),
    )
    .into_compile_error()
    .into()
}
