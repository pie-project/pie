//! **The precision a row states is the precision its trace declares.**
//!
//! `models::PRECISIONS` is hand-written, and a hand-written table beside a
//! generated one is worth exactly as much as the thing holding them together.
//! Two things hold this one:
//!
//! * `precision_disagreements` — the completeness check, in BOTH directions,
//!   so a deleted catalog row cannot keep its entry forever;
//! * this file — the AGREEMENT check, which is the one that catches a wrong
//!   entry rather than a missing one.
//!
//! # Why the agreement check is worth having
//!
//! The tempting way to fill the table is to slice the SKU name before `-kv-`,
//! and it is wrong for every MoE row: `gptoss-20b-bf16-mxfp4-kv-bf16` is bf16
//! dense and mxfp4 experts, `glm5-a12b-bf16-bf16-kv-bf16` has two bf16
//! segments. An entry filled that way is wrong SILENTLY — the stamp states
//! `bf16` for a file whose experts are mxfp4, and `Stamp::check` passes it
//! against a deployment that computed `bf16` the same wrong way.
//!
//! So the trace is asked instead, in both directions. A token in the table
//! must be WITNESSED by some param's dtype, and a quantized param must be
//! NAMED by some token. Neither direction alone is enough: the first lets a
//! row forget its experts, the second lets a row claim a form it does not
//! hold.
//!
//! This is not a classification — it does not compute the string, which is
//! the thing that would need a table keyed on dtypes and would rot the day a
//! new dtype landed, as five such tables did when 2-bit opened. It checks a
//! string somebody wrote.

use model_dsl::{Dtype, Platform};

/// Which token, if any, a stored weight dtype is evidence for.
///
/// `None` is "this dtype says nothing about precision" — a raw plane, whose
/// presence is compatible with every row, since every model has norms and
/// biases whatever its banks are.
fn witnesses(dtype: Dtype) -> Option<&'static str> {
    match dtype {
        // The MLX affine family, at every width it ships. The 8-bit row is
        // `mlxu4`'s too: `Dtype::U8g64` is the same scheme at the width the
        // MoE router gates use, which is that type's whole argument.
        Dtype::U4g64 | Dtype::U4g32 | Dtype::U4g64tiled | Dtype::U8g64 => Some("mlxu4"),
        Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128 => Some("mlxu2"),
        Dtype::Mxfp4 => Some("mxfp4"),
        _ => None,
    }
}

#[test]
fn the_table_and_the_catalog_name_the_same_rows() {
    let faults = models::precision_disagreements();
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

#[test]
fn every_row_states_a_precision_its_trace_agrees_with() {
    let mut faults: Vec<String> = Vec::new();
    for (sku, _, trace_of, _) in models::catalog() {
        let Some(precision) = models::precision_of(sku) else {
            continue; // the sibling test above owns this case
        };
        let stated: Vec<&str> = precision.split('-').collect();
        // The traces are read at BOTH platforms, because a placement is
        // resolved per shell (§J4c) and precision must not be: a repack moves
        // no value, so `U4g64tiled` and `U4g64` are one token and a row whose
        // precision changed with the reading would be a row whose artifact
        // could not be named.
        for platform in [Platform::Cuda, Platform::Metal] {
            let trace = trace_of(platform);
            let mut held: Vec<&str> = trace
                .params
                .iter()
                .filter_map(|param| witnesses(param.dtype))
                .collect();
            held.sort_unstable();
            held.dedup();

            for token in &stated {
                if *token == "bf16" {
                    // `bf16` is the absence of a quantized bank, so it is
                    // witnessed by the OTHERS being absent rather than by any
                    // dtype of its own — checked below, where it is a
                    // statement about the whole set.
                    continue;
                }
                if !held.contains(token) {
                    faults.push(format!(
                        "`{sku}` states precision {precision:?}, and no param of its \
                         {platform:?} trace is stored in a form that witnesses \
                         {token:?}"
                    ));
                }
            }
            for token in &held {
                if !stated.contains(token) {
                    let witness = trace
                        .params
                        .iter()
                        .find(|param| witnesses(param.dtype) == Some(token))
                        .map(|param| param.name.clone())
                        .unwrap_or_default();
                    faults.push(format!(
                        "`{sku}` states precision {precision:?} and its {platform:?} \
                         trace holds `{witness}` in a form that witnesses {token:?}, \
                         which the stated precision does not name"
                    ));
                }
            }
            if stated == ["bf16"] && !held.is_empty() {
                faults.push(format!(
                    "`{sku}` states plain `bf16` and its {platform:?} trace holds \
                     {held:?}"
                ));
            }
        }
    }
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
