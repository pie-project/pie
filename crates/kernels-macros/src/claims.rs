//! `#[claims]` — the points a plane answers, read off the impl that answers
//! them. What the block does not override keeps the family's default body,
//! which is the backlog row.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Error, Ident, ImplItem, ItemImpl, spanned::Spanned};

use crate::points::snake;

pub fn expand(item: ItemImpl) -> Result<TokenStream, Error> {
    let Some((path, _)) = &item.trait_ else {
        return Err(Error::new(
            item.self_ty.span(),
            "`#[claims]` reads a family impl: `impl Norm for Ctx<'_>`",
        ));
    };
    let Some(last) = path.segments.last() else {
        return Err(Error::new(path.span(), "`#[claims]` wants a named family"));
    };
    let family = snake(&last.ident);
    let table = Ident::new(
        &format!("{}_CLAIMS", family.to_uppercase()),
        last.ident.span(),
    );
    // Membership needs no check: rustc refuses a method the family never
    // declared, so everything in the block is a point.
    let claimed = item.items.iter().filter_map(|i| match i {
        ImplItem::Fn(f) => Some(format!("{family}.{}", f.sig.ident)),
        _ => None,
    });
    Ok(quote! {
        #item

        pub const #table: &[&str] = &[#(#claimed),*];
    })
}
