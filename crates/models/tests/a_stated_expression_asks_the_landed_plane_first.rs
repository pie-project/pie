//! **A `read_expr` ARGUMENT MAY NOT NEED `?`**, because the `?` fires before
//! the verb it is an argument to.
//!
//! Every read verb in `checkpoint_dsl::Builder` consults
//! `holds_the_landed_plane` first: a file that already carries this weight
//! under this text's own name is an artifact `pie model import` wrote, the
//! transform has run, and the plane binds with `read_own`. `read_expr`
//! consults it too — and one step too late, because it takes an `Expr`
//! ALREADY MADE and the making happens at the call site.
//!
//! For an expression assembled out of names and constants that ordering never
//! shows. For one that reads the checkpoint to decide its own shape it is
//! fatal, and it was: `qwen_3`'s `squeezed` asks which of the two depthwise
//! conv1d spellings a file uses and answers `Error::Missing` when neither is
//! there, which is exactly what a promoted artifact says. So
//!
//! ```text
//! b.read_expr(&g.conv, squeezed(src, n("linear_attn.conv1d.weight"))?)?;
//! ```
//!
//! refused an artifact whose `layer.N.conv` was sitting in the file, and
//! `mlx-community/Qwen3.5-0.8B-4bit` came out of `pie model import` as 673
//! objects that this same build could not identify. Four call sites across
//! three family texts had the shape; `Builder::read_derived` takes the
//! expression as a thunk and forces it on the far side of the check.
//!
//! # Why the rule is `?` and not "reads the source"
//!
//! A textual scan cannot see whether a helper consults the checkpoint —
//! `squeezed(src, ..)` does and `norm(name)` does not, and the next one may
//! take the source by another name or reach it through `self`. What it CAN
//! see is fallibility, and fallibility is the honest proxy: an expression
//! that cannot fail did not have to ask anything, and one that can is one
//! whose question the landed check may have made unnecessary. The rule is
//! therefore a little wider than the defect, which is the direction a rule
//! about a silent failure should err in — and it costs nothing, because the
//! fix is to name the other verb.
//!
//! The precedent for scanning text at all is `checkpoint/tests/citations.rs`,
//! which reads these same files for the same reason: the property is about
//! what is WRITTEN, and no type in the tree carries it.

use std::path::{Path, PathBuf};

fn import_texts() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&root).expect("the family texts are here") {
        let path = entry.expect("a readable entry").path().join("import.rs");
        if path.is_file() {
            out.push(path);
        }
    }
    out.sort();
    assert!(
        out.len() >= 7,
        "the tree ships seven family texts and this found {}",
        out.len()
    );
    out
}

/// The argument list of one `read_expr(` call, from the open paren to the
/// paren that closes it.
///
/// Counted rather than matched to a line ending, because three of the call
/// sites wrap across lines and one of them nests two more calls inside.
/// String and char literals are not tracked: no argument in these texts
/// contains an unbalanced paren inside a literal, and a scan that started
/// guessing about literals would be a parser.
fn arguments(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut at = 0usize;
    while let Some(found) = text[at..].find("read_expr(") {
        let open = at + found + "read_expr(".len() - 1;
        let line = text[..open].lines().count();
        let mut depth = 0i32;
        let mut end = open;
        for (offset, byte) in bytes[open..].iter().enumerate() {
            match byte {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + offset;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push((line, text[open + 1..end].to_string()));
        at = end.max(open + 1);
    }
    out
}

#[test]
fn no_stated_expression_is_built_by_a_question_mark() {
    let mut sites = 0usize;
    let mut faults: Vec<String> = Vec::new();
    for path in import_texts() {
        let text = std::fs::read_to_string(&path).expect("a readable family text");
        for (line, argument) in arguments(&text) {
            sites += 1;
            if argument.contains('?') {
                faults.push(format!(
                    "{}:{line} builds its expression with `?`:\n    {}",
                    path.display(),
                    argument.split_whitespace().collect::<Vec<_>>().join(" "),
                ));
            }
        }
    }
    assert!(
        faults.is_empty(),
        "a `read_expr` argument that can fail is evaluated BEFORE the verb \
         asks whether the plane is already landed, so it refuses an artifact \
         `pie model import` wrote out of this same text. Use \
         `Builder::read_derived`, which takes the expression as a thunk.\n\n{}",
        faults.join("\n"),
    );
    // The scan found call sites at all. A rename of the verb, or a helper
    // that stopped being written this way, would otherwise leave this test
    // passing over nothing.
    assert!(
        sites >= 20,
        "the scan found only {sites} `read_expr` call sites, which is fewer \
         than the tree has — the scan, not the tree, is what changed"
    );
}
