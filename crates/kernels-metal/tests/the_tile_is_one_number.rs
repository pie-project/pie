//! **The tile that spells the name is the tile that sizes the grid.**
//!
//! `.wiki/kernel-x/metal-refactor.md` §2.1 is titled *"One tile, chosen twice,
//! in two crates, compared nowhere"*, and §9 lists settling it as the thing
//! the refactor **owed first**. This is what settling it turned into.
//!
//! # What it was
//!
//! `affine_qmm_t` is stamped over `(group x bits x bm x bn)`, so the tile is
//! part of the entrypoint NAME: `affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32`.
//! The model text wrote one tile into the name -- `QMM_TILE: (u32, u32) =
//! (32, 32)` on `LlamaLikeMetalFacts::qmm_tile` -- and the driver computed
//! another from geometry and never read the name: `bm = shapes::qmm_bm(rows)`,
//! the widest of `[16, 32, 64]` at or under the row count, and `bn =
//! widest_column_tile(width)`, the widest of `[16, 32, 64]` dividing it.
//!
//! They differ on shapes this repository actually serves. Llama-3.2-1B's
//! projections are 2048 wide, so `widest_column_tile(2048)` is 64 while the
//! text says 32; at a 2048-row prefill `qmm_bm(2048)` is 64 and the text says
//! 32. `StepEncoder::dispatch` takes THREADS, so the threadgroup count is
//! `[width / bn, rows / bm, 1]`, and a kernel compiled for a 32x32 tile handed
//! a grid enumerated for 64x64 covers a quarter of its output.
//!
//! The doc was careful not to claim that was a live wrong answer -- the MLX
//! oracle passed on exactly that shape -- and asked for one assert in
//! `plan_one`: refuse unless the symbol ends in `_bm_{bm}_bn_{bn}` for the
//! pair the rule just computed. *"Either it fires -- and the refactor has
//! found a silent correctness defect of the first order -- or it does not, and
//! we have learned what reconciles them."*
//!
//! # What happened instead
//!
//! Neither. `plan_one` and `Rule::Qmm` are gone, `qmm_bm` and
//! `widest_column_tile` are gone with them -- `grep` finds one mention of the
//! first in a comment and nothing at all of the second -- and `bm` and `bn`
//! arrive at `qmm_t` as `Const<i32>` parameters of the routine. One value
//! spells the name and one value sizes the grid, and it is the same value:
//!
//! ```ignore
//! Fire::at(QMM_FILE, qmm_name("", *group, *bits, *bm, *bn)?)
//!     .apply(Grid::of(qmm_grid(n, *bn, m, *bm, 1)?, QMM_GROUP))
//! ```
//!
//! That is the doc's own last paragraph, arrived at: *"in a body there is one
//! `bn`. It is chosen once, and the same value spells the name and sizes the
//! grid. The disagreement is not fixed; it is **unstateable**."*
//!
//! # Why there is still a test
//!
//! Because "unstateable" is a property of how these fourteen lines are
//! WRITTEN, and nothing was holding it. A `bn` recomputed in a body -- from
//! `y.width`, from a rung table, from anything -- restores the defect exactly,
//! and it restores it in the form the doc describes: silently, on the shapes
//! that divide differently from the ones the texts state, with the oracle
//! green until a deployment arrives at the wrong width.
//!
//! So this reads the fourteen fires and requires, textually, that the last two
//! arguments of the name composer are the same expressions the grid is given.
//! Not the same VALUES -- a test cannot run a plan here -- the same source
//! text, which is stronger: it forbids a second computation rather than
//! catching one that happens to differ today.
//!
//! # What it does not check
//!
//! That the tile the model text states is one the shader tree stamps.
//! `qmm_name` refuses a point off its axes and
//! `composed_names_are_stamped` holds the whole product against `STAMPED`,
//! which is that question asked in the two places it can be answered.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// The composers whose last two arguments are `(bm, bn)`.
///
/// Both take the tile last, and that is the only thing this needs to know
/// about them: `qmm_name(form, group, bits, bm, bn)` and
/// `qmm_precast_name(before, after, bm, bn)` differ in arity and agree in
/// tail. A third composer that puts the tile somewhere else has to be added
/// here, and until it is, the count below fails.
const COMPOSERS: &[&str] = &["qmm_name(", "qmm_precast_name("];

/// The arguments of `call(..)` starting at the byte after its opening paren,
/// split on top-level commas.
fn arguments(text: &str, open: usize) -> Option<Vec<String>> {
    let bytes = text.as_bytes();
    let (mut depth, mut in_str, mut i) = (1i32, false, open + 1);
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    while i < bytes.len() {
        let c = bytes[i] as char;
        match c {
            '"' => in_str = !in_str,
            // NOT `<`. `Const<i32>` never appears in a call's arguments, and
            // counting angle brackets would make `a < b` unbalanced.
            '(' | '[' if !in_str => depth += 1,
            ')' | ']' if !in_str => {
                depth -= 1;
                if depth == 0 {
                    out.push(cur.trim().to_owned());
                    return Some(out);
                }
            }
            ',' if !in_str && depth == 1 => {
                out.push(cur.trim().to_owned());
                cur.clear();
                i += 1;
                continue;
            }
            _ => {}
        }
        cur.push(c);
        i += 1;
    }
    None
}

/// Every `(line, composer arguments, grid arguments)` in `quant.rs`.
fn fires() -> Vec<(usize, Vec<String>, Vec<String>)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/quant.rs");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        // Comments carry the composer's name in prose all over this module,
        // and a doc example is not a fire.
        let code = line.split_once("//").map_or(line, |(before, _)| before);
        if !code.contains("Fire::at(QMM_FILE") {
            continue;
        }
        let Some(composer) = COMPOSERS.iter().find_map(|c| code.find(c).map(|at| at + c.len() - 1))
        else {
            continue;
        };
        let Some(named) = arguments(code, composer) else { continue };
        let Some(grid) = code.find("qmm_grid(").and_then(|at| arguments(code, at + "qmm_grid".len()))
        else {
            continue;
        };
        out.push((n + 1, named, grid));
    }
    out
}

/// **The name's tile and the grid's tile are the same source text.**
///
/// See the header. `qmm_grid(n, bn, m, bm, splits)` takes the column tile
/// second and the row tile fourth, and both composers take `(bm, bn)` last,
/// so the pairing is `named[len-2] == grid[3]` and `named[len-1] == grid[1]`.
#[test]
fn the_tile_that_spells_the_name_is_the_tile_that_sizes_the_grid() {
    let fires = fires();
    let mut wrong: Vec<String> = Vec::new();
    for (line, named, grid) in &fires {
        if named.len() < 2 || grid.len() < 4 {
            wrong.push(format!(
                "  quant.rs:{line}: read {} name argument(s) and {} grid \
                 argument(s), which is not a shape this can compare",
                named.len(),
                grid.len()
            ));
            continue;
        }
        let (name_bm, name_bn) = (&named[named.len() - 2], &named[named.len() - 1]);
        let (grid_bn, grid_bm) = (&grid[1], &grid[3]);
        if name_bm != grid_bm {
            wrong.push(format!(
                "  quant.rs:{line}: the name is spelled with a row tile of \
                 `{name_bm}` and the grid is sized with `{grid_bm}`"
            ));
        }
        if name_bn != grid_bn {
            wrong.push(format!(
                "  quant.rs:{line}: the name is spelled with a column tile of \
                 `{name_bn}` and the grid is sized with `{grid_bn}`"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} fire(s) spell one tile and enumerate another. `StepEncoder::\
         dispatch` takes threads, so the threadgroup count is \
         `[width / bn, rows / bm, 1]`: a kernel compiled for a 32x32 tile \
         handed a grid enumerated for 64x64 covers a quarter of its \
         output.\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    // A NUMBER AND AN ASSERTION. A scan that stopped matching would compare
    // nothing and agree with everything, which is the failure this whole
    // family of tests is written against.
    assert_eq!(fires.len(), 14, "the affine matmul fires in `quant.rs`");
}

/// **The tile reaches a fire as a parameter, and is not computed beside it.**
///
/// The check above forbids the two expressions DISAGREEING. This forbids the
/// shape that made them able to: a tile derived inside the body from the
/// geometry, which is what `qmm_bm(rows)` and `widest_column_tile(width)`
/// were. Two expressions that agree textually are still two expressions if
/// each is a call.
///
/// So every tile spelled at a fire must be a plain `*bm`/`*bn` deref of the
/// routine's own `Const<i32>` parameter, or a module constant in SCREAMING
/// case -- `WIDE_BN`, which is the one column tile the wide forms are stamped
/// at, stated once at its definition and not recomputed either.
#[test]
fn a_tile_is_a_parameter_or_a_stated_constant_and_never_a_computation() {
    let fires = fires();
    let mut wrong: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (line, named, _) in &fires {
        for tile in &named[named.len().saturating_sub(2)..] {
            seen.insert(tile.clone());
            let deref = tile
                .strip_prefix('*')
                .is_some_and(|rest| rest.chars().all(|c| c.is_lowercase() || c == '_'));
            let stated = tile.chars().all(|c| c.is_ascii_uppercase() || c == '_') && !tile.is_empty();
            if !deref && !stated {
                wrong.push(format!(
                    "  quant.rs:{line}: the tile is spelled `{tile}`, which is \
                     neither a parameter deref nor a stated constant"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} tile(s) are computed where they are spelled. That is the shape \
         `.wiki/kernel-x/metal-refactor.md` §2.1 describes: a tile chosen \
         twice cannot be compared, and the second chooser wins the grid while \
         the first wins the name.\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    assert_eq!(
        seen,
        ["*bm", "*bn", "WIDE_BN"].iter().map(|s| (*s).to_owned()).collect(),
        "the whole vocabulary of tiles this module spells"
    );
}
