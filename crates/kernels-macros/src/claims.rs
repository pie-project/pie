//! `#[claims]` — the points a plane answers, read off the impl that answers
//! them. What the block does not override keeps the family's default body,
//! which is the backlog row.
//!
//! TWO KINDS OF IMPL, AND THE DIFFERENCE IS TIER. A family impl
//! (`impl Norm for Ctx<'_>`) answers points some other crate DECLARED, so
//! what this reads off it is a list of NAMES — the floor already holds their
//! slots. An inherent impl (`impl Ctx<'_>`) is the plane's tier-2 surface,
//! which no floor declares, so what this reads off it is the POINTS
//! THEMSELVES. See [`tier2`].

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Error, Ident, ImplItem, ItemImpl, spanned::Spanned};

use crate::points::{point, snake};

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

/// The plane's TIER-2 SURFACE: inherent methods on its own `Ctx`.
///
/// `.wiki/baker.md`: *"Tier-2 = an inherent method on `Ctx`. No trait, no
/// floor entry — an inherent impl can only live in the plane crate, which is
/// the whole rule."* This is the reading of that block, and it emits ONE
/// table where a family impl emits a name list.
///
/// # Why one table and not two
///
/// A tier-1 point stands in TWO tables because it states two different facts
/// held by two different crates: the floor's `*_POINTS` says what every plane
/// owes, and a plane's `*_CLAIMS` says which of those it answers — so a point
/// with a declaration and no claim is a measured backlog row.
///
/// A tier-2 point has no floor entry to owe anything against. Declaring it
/// and claiming it are the same act at the same site, so a claim table beside
/// this one could say nothing the point list does not, and a backlog row is
/// not a thing it can have. That asymmetry IS the tier distinction, and one
/// table is what it looks like written down.
///
/// The slots are read by [`point`], the same reader `#[points]` uses, so a
/// tier-2 declaration is spelled in exactly the vocabulary a floor
/// declaration is — `In<Self::Tensor<T>>`, `Cache<Self::Pages>`, a bare
/// `f32` — and the generated dispatch reads its columns by the same rule.
fn tier2(item: ItemImpl) -> Result<TokenStream, Error> {
    // EVERY FN IN THE BLOCK IS A POINT, which is what makes the table
    // complete. A family impl gets this from rustc (a method the trait never
    // declared does not compile); an inherent block would happily hold a
    // private helper, so the rule is stated instead: a helper goes in a block
    // of its own, and this one holds the surface.
    let points = item
        .items
        .iter()
        .filter_map(|i| match i {
            ImplItem::Fn(f) => Some(f),
            _ => None,
        })
        .map(|f| point(f.sig.ident.to_string(), &f.sig))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(quote! {
        #item

        pub const TIER2_POINTS: &[Point] = &[#(#points),*];
    })
}
