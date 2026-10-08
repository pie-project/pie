//! The drafters published apart from the models they draft for, as the
//! packages state them: a draft head's repository, the repository of the model
//! it drafts for, and the deployment the two are served as.

pub use poem::star::Published;

/// Every drafter a package states it was published for.
pub fn all() -> impl Iterator<Item = &'static Published> {
    crate::star::packages()
        .iter()
        .flat_map(|package| package.manifest().published.iter())
}

/// The drafter `drafter` published for the repository `target`.
#[must_use]
pub fn lookup(target: &str, drafter: &str) -> Option<&'static Published> {
    let wanted = target.to_ascii_lowercase().replace("--", "/");
    all().find(|p| {
        p.drafter.eq_ignore_ascii_case(drafter) && p.target.to_ascii_lowercase() == wanted
    })
}

/// Every drafter published for the repository `target`.
pub fn for_target(target: &str) -> impl Iterator<Item = &'static Published> {
    let wanted = target.to_ascii_lowercase().replace("--", "/");
    all().filter(move |p| p.target.to_ascii_lowercase() == wanted)
}
