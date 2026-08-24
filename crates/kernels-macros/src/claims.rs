use proc_macro2::TokenStream;
use quote::quote;
use syn::{Error, Ident, ImplItem, ItemImpl, spanned::Spanned};

use crate::points::{point, snake, strip_shape};

pub fn expand(item: ItemImpl) -> Result<TokenStream, Error> {
    let Some((path, _)) = &item.trait_ else {
        return tier2(item);
    };
    let Some(last) = path.segments.last() else {
        return Err(Error::new(path.span(), "`#[claims]` wants a named family"));
    };
    let family = snake(&last.ident);
    let table = Ident::new(
        &format!("{}_CLAIMS", family.to_uppercase()),
        last.ident.span(),
    );

    let claimed = item.items.iter().filter_map(|i| match i {
        ImplItem::Fn(f) => Some(format!("{family}.{}", f.sig.ident)),
        _ => None,
    });
    Ok(quote! {
        #item

        pub const #table: &[&str] = &[#(#claimed),*];
    })
}

fn tier2(mut item: ItemImpl) -> Result<TokenStream, Error> {
    let points = item
        .items
        .iter()
        .filter_map(|i| match i {
            ImplItem::Fn(f) => Some(f),
            _ => None,
        })
        .map(|f| point(f.sig.ident.to_string(), &f.attrs, &f.sig))
        .collect::<Result<Vec<_>, _>>()?;

    for i in &mut item.items {
        if let ImplItem::Fn(f) = i {
            strip_shape(&mut f.attrs);
        }
    }
    Ok(quote! {
        #item

        pub const TIER2_POINTS: &[Point] = &[#(#points),*];
    })
}
