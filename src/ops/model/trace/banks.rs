//! `banks.json`: the adapter banks a trace registers, the weights a trainer
//! trains. A bank's name is its struct path; where that path is
//! `layers.{l}.{stanza}.{site}.{bank}`, its entry also states the layer, the
//! stanza and the site.

use anyhow::{Result, bail};
use poem::Param;
use serde_json::{Value, json};

/// `banks.json` for the model `model` of the package `package`: every bank
/// with its shape and dtype, and the slots axis 0 of each holds.
pub fn list(package: &str, model: &str, banks: &[&Param]) -> Result<Value> {
    let slots = banks
        .first()
        .and_then(|bank| bank.shape.first().copied())
        .unwrap_or(0);
    let mut listed = Vec::new();
    for bank in banks {
        if bank.shape.first() != Some(&slots) {
            bail!(
                "the bank `{}` is {:?}, whose axis 0 is not the {slots} slots of the others",
                bank.name,
                bank.shape
            );
        }
        let layer = match bank.name.split('.').collect::<Vec<_>>()[..] {
            ["layers", layer, ..] => layer.parse::<u32>().ok(),
            _ => None,
        };
        let (stanza, site) = stanza_of(&bank.name).unzip();
        listed.push(json!({
            "name": bank.name,
            "shape": bank.shape,
            "dtype": poem::star::word(bank.dtype),
            "layer": layer,
            "stanza": stanza,
            "site": site,
        }));
    }
    Ok(json!({
        "format": "pie-stanza.banks/1",
        "package": package,
        "model": model,
        "slots": slots,
        "banks": listed,
    }))
}

/// The stanza and the site a bank's name `layers.{l}.{stanza}.{site}.{bank}`
/// states; none where the name has another shape.
pub fn stanza_of(name: &str) -> Option<(&str, &str)> {
    match name.split('.').collect::<Vec<_>>()[..] {
        ["layers", _, stanza, site, _] => Some((stanza, site)),
        _ => None,
    }
}
