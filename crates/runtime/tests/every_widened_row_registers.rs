// A tensor-parallel group identifies its artifact by the 1-rank row and
// selects `<base>-tp<ranks>`; `runtime::model::register` then looks that name
// up in ROWS. A catalog row the table does not carry boots every rank and
// refuses at registration, so the table states each widened row, and states
// it as its 1-rank sibling does where the table carries the sibling.
#[test]
fn every_widened_row_registers() {
    let mut faults = Vec::new();
    for row in models::skus().filter(|row| row.recipe.tp > 1) {
        let Some(widened) = runtime::model::row(&row.name) else {
            faults.push(format!(
                "`{}` is a {}-rank catalog row and `runtime::model::ROWS` has no entry for it",
                row.name, row.recipe.tp
            ));
            continue;
        };
        let base = models::Recipe {
            tp: 1,
            ..row.recipe
        }
        .name();
        let Some(base) = runtime::model::row(&base) else {
            continue;
        };
        if (widened.layers, widened.vocab, widened.arch) != (base.layers, base.vocab, base.arch) {
            faults.push(format!(
                "`{}` states ({}, {}, {}) and `{}` states ({}, {}, {}); the ranks cut \
                 the widths, not the depth, the vocabulary or the arch",
                widened.id,
                widened.layers,
                widened.vocab,
                widened.arch,
                base.id,
                base.layers,
                base.vocab,
                base.arch
            ));
        }
    }
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
