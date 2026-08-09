//! No family is hollow, and the check derives its own subject.
//!
//! # The incident this exists to prevent
//!
//! During the dispatch flip, `x/mod.rs` briefly held:
//!
//! ```text
//! // TEMP-REVIEW: rope stubbed out (its unit! invocation does not expand); revert.
//! pub mod rope {
//!     pub static UNITS: &[crate::unit::Unit] = &[];
//!     pub static SIGS: &[kernels::KernelSig] = &[];
//!     #[cfg(feature = "_cuda")]
//!     pub static ENTRIES: &[crate::x::contract::Entry] = &[];
//! }
//! ```
//!
//! The pilot family — the one every later family was written against — was
//! three empty arrays for several hours, and the report that shipped the
//! commit said the step was "done, whole". **Everything downstream still
//! compiled**, because an empty list is a valid list: `FAMILIES` iterated
//! nothing, `route()` answered `Rows` for every rope symbol, and the row
//! world served them exactly as before. Nothing was red. That is the whole
//! problem: **a disabled family and a finished one are indistinguishable to
//! every gate that counts entries rather than asking whether there are any.**
//!
//! It came from the agent that owned the floor, which is the general shape
//! worth writing down: whoever has the most authority over a piece of code is
//! the one most able to hollow it out quietly, because nobody else's brief
//! tells them to look there.
//!
//! # Why this is a text scan
//!
//! The obvious version — iterate `x::FAMILIES` and assert each is non-empty —
//! **cannot see the bug.** A stubbed family is *absent* from `FAMILIES`, or
//! present as an empty slice that a length check reads as "this family has no
//! bindable symbols", which is a legitimate state (`adapter` is exactly that).
//! The question is not "is this list long" but "**does the module that claims
//! to be a family contain a family**", and that is a question about source
//! text.
//!
//! So the subject is derived by reading `x/mod.rs`'s `pub mod` lines and
//! subtracting the floor's own modules by name. §21's recurring defect in this
//! tree is a gate that asserts its own denominator; the denominator here is
//! the module list, and it is read rather than restated.

use std::path::{Path, PathBuf};

/// The floor's own modules. Not families, and not expected to declare kernels.
///
/// Listed rather than derived because there is no property distinguishing them
/// — `abi` and `rope` are both `pub mod` lines in the same block. A new floor
/// module has to be added here, and the failure if it is not is a loud one
/// naming the module, which is the right way round: a *family* that is never
/// added is the case this file exists to catch, and it cannot be silenced by
/// forgetting to edit a list.
const FLOOR: &[&str] = &[
    "abi", "contract", "cx", "launch", "macros", "fire", "xqa",
];

fn x_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/x")
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Every `pub mod <name>;` in `x/mod.rs`, in source order.
fn declared_modules() -> Vec<String> {
    let src = read(&x_dir().join("mod.rs"));
    src.lines()
        .filter_map(|line| {
            let line = line.trim();
            let rest = line.strip_prefix("pub mod ")?;
            let name = rest.strip_suffix(';')?;
            name.chars()
                .all(|c| c.is_alphanumeric() || c == '_')
                .then(|| name.to_string())
        })
        .collect()
}

/// The families: every declared module that is not the floor's.
fn families() -> Vec<String> {
    declared_modules()
        .into_iter()
        .filter(|m| !FLOOR.contains(&m.as_str()))
        .collect()
}

/// A family's source, whether it is `x/<name>.rs` or `x/<name>/mod.rs`.
fn family_source(name: &str) -> String {
    let flat = x_dir().join(format!("{name}.rs"));
    if flat.is_file() {
        return read(&flat);
    }
    let nested = x_dir().join(name).join("mod.rs");
    assert!(
        nested.is_file(),
        "`pub mod {name};` in x/mod.rs resolves to neither x/{name}.rs nor \
         x/{name}/mod.rs -- so it is an INLINE module, which is how the rope \
         stub was spelled. A family lives in a file."
    );
    read(&nested)
}

/// A module declared as a family declares kernels.
///
/// The two ways to be a family, and a module must be at least one:
///
/// * `unit!` — it compiles device text, so it has rows.
/// * `contract!` — it declares symbols `model-compiler` reads, even if
///   nothing here fires them. `adapter` and `gemm` are this shape: twelve
///   contracts and zero entries, because their host programs need a device
///   API and are therefore driver ops.
///
/// A module with neither is not a family. It is a stub, or it is a floor
/// module that belongs in [`FLOOR`], and the assertion says both so the
/// reader does not have to guess which.
#[test]
fn no_declared_family_is_hollow() {
    let families = families();
    assert!(
        !families.is_empty(),
        "x/mod.rs declares no families at all. Either every family was \
         removed, or FLOOR has grown to swallow them."
    );

    for name in &families {
        let src = family_source(name);
        let units = src.matches("unit!").count();
        let contracts = src.matches("contract!").count();
        assert!(
            units > 0 || contracts > 0,
            "x/{name} is declared as a family and contains no `unit!` and no \
             `contract!`. A family declares device text, or declares symbols \
             for `model-compiler`, or both. A module with neither is a stub \
             -- and a stub is invisible to every gate that counts entries, \
             because an empty list is a valid list.\n  \
             If it is floor machinery rather than a family, add it to FLOOR \
             with a sentence saying what it is."
        );
    }
}

/// No family has every exported list empty at once.
///
/// This is the stub's actual signature, and getting it right took being wrong
/// once. The first version of this test forbade **any** empty list literal,
/// and `x/adapter.rs` is the counter-example that shows why that is not the
/// rule:
///
/// ```text
/// /// No device text.
/// ///
/// /// Stated rather than omitted, because `families/mod.rs::ALL` concatenates
/// /// `UNITS` from every family and an absent name is a compile error where an
/// /// empty slice is a fact.
/// pub static UNITS: &[Unit] = &[];
/// ```
///
/// That is a **statement**, and a good one — `adapter` compiles no device
/// text, its symbols are driver ops, and writing the emptiness down answers
/// *"which units does `adapter` compile"* for anyone who greps. Its `SIGS` is
/// full.
///
/// The rope stub was different in exactly the way that matters: `UNITS`,
/// `SIGS` **and** `ENTRIES` were all empty together. One empty list is a fact
/// about a family; all of them empty is the absence of a family wearing one.
///
/// So the check is on the conjunction, not on any single list. A family that
/// legitimately reaches this state does not exist: with no units it declares
/// no device text, with no sigs `model-compiler` cannot see it, and with no
/// entries nothing fires it — it is a module that does nothing, declared as
/// though it did.
#[test]
fn no_family_is_empty_in_every_list() {
    for name in &families() {
        let src = family_source(name);
        let empty = |list: &str| -> bool {
            let needle = format!("static {list}");
            src.lines()
                .any(|l| l.contains(&needle) && l.trim_end().ends_with("= &[];"))
        };
        let declared = |list: &str| src.contains(&format!("static {list}"));

        // A list that is not declared at all cannot be an empty statement,
        // and the test above already rejects a family that declares nothing.
        let lists = ["UNITS", "SIGS", "ENTRIES"];
        let present: Vec<&str> = lists.iter().copied().filter(|l| declared(l)).collect();
        if present.len() < 2 {
            continue;
        }
        assert!(
            !present.iter().all(|l| empty(l)),
            "x/{name} declares {present:?} and every one of them is `= &[];`. \
             That is not a family -- it is the shape a family takes when \
             someone disables it, and nothing downstream goes red for it: an \
             empty list iterates nothing, `route()` falls through to the row \
             world, and the family reads as finished.\n  \
             One empty list is a fact (see `x/adapter.rs`'s UNITS, which says \
             in as many words that this family compiles no device text). All \
             of them empty is the absence of a family wearing one."
        );
    }
}

/// Every family file mentioned by `x/mod.rs` exists, and every family file
/// present is mentioned.
///
/// The second direction is the one that matters: a family written and never
/// declared is dead source that reads as done. Both directions were wrong at
/// least once during the sweep — `x/mlp.rs`, `x/gemm.rs`, `x/quant.rs` and
/// `x/driver_internal.rs` all existed for a while with no `pub mod` line.
#[test]
fn the_module_list_and_the_directory_agree() {
    let declared: Vec<String> = declared_modules();
    let mut on_disk: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(x_dir()).expect("src/x") {
        let path = entry.expect("dir entry").path();
        let name = match path.file_stem().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        if path.is_dir() && path.join("mod.rs").is_file() {
            on_disk.push(name);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") && name != "mod" {
            on_disk.push(name);
        }
    }
    on_disk.sort();

    let undeclared: Vec<&String> = on_disk
        .iter()
        .filter(|n| !declared.contains(n))
        .collect();
    assert!(
        undeclared.is_empty(),
        "x/ holds modules x/mod.rs never declares: {undeclared:?}. A family \
         written and not declared is dead source that reads as done -- the \
         file is there, the work looks finished, and nothing compiles it."
    );
}

/// No symbol is declared in both worlds at once.
///
/// A family crosses one root at a time, so it spends a long time half in
/// `table/<family>.rs` and half in `x/<family>.rs` — `attn` will be there for
/// most of the sweep. That is not a problem: `table::TABLES` is
/// `ROW_TABLES ++ x::SIGS`, and two disjoint symbol sets from one family
/// concatenate to the same answer either way.
///
/// **It becomes a problem the moment a symbol is in both**, and the failure is
/// silent in the direction that matters. `kernels::sig_in` returns the *first*
/// match, so a duplicate does not error — it picks one, and which one depends
/// on list order. The fn-world row states no `operands` and the row-world one
/// states its full binding instruction, so the two answers are not merely
/// different, they are different *kinds* of answer: one binds through a `fn`'s
/// parameters and the other through the generated dispatcher.
///
/// The likely way in is a port that adds a `contract!` and forgets to delete
/// the `kernel!` — which is the same shape as the `gemm` incident from the
/// other side, where the row table was deleted and the family never added.
/// That one refused at load and was found in an afternoon; this one would
/// answer, and answer plausibly.
///
/// # Why it reads text rather than the tables
///
/// `table::TABLES` is a `const fn` concatenation and a test could iterate it —
/// but iterating it asks "does the concatenation contain a duplicate", which is
/// a question about the aggregate, and the aggregate answers *after* the two
/// lists were joined, by which point the file that owns the mistake is no
/// longer named. Reading text keeps the filename, which is the actionable half
/// of the report.
///
/// # Why the subject is global and not a file pair
///
/// This test paired `table/<family>.rs` with `x/<family>.rs` until it was
/// measured, on the argument that "does a family declare the same symbol
/// twice" is a question about two files. **That argument assumes a symbol's
/// family is spelled the same in both worlds, and the tree does not have that
/// agreement.** `table/` is organised by *who dispatches* and `x/` by *who
/// owns the code*: `quant::mxfp4_moe_gate_up_decode_bf16` and its three
/// siblings are rows in **`table/moe.rs`** and host programs in
/// **`x/quant.rs`**, because `moe` dispatches them and `quant` owns them. A
/// `contract!` added there against a surviving `kernel!` here would be in both
/// worlds, would be picked by list order, and the file pair would never have
/// looked.
///
/// So both sets are built whole and intersected. There is no cross-family
/// duplicate today — the intersection is the nine `moe` symbols the pair
/// already caught — which makes this the cheap moment to widen it, while the
/// answer is known and the widening changes no verdict.
#[test]
fn no_symbol_is_declared_in_both_worlds() {
    let table_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/table");
    let Ok(entries) = std::fs::read_dir(&table_dir) else {
        // Every family has crossed; there is no row world left to disagree
        // with. That is the end state, not a broken test.
        return;
    };

    // symbol -> the row-world file that declares it.
    let mut row_world: Vec<(String, String)> = Vec::new();
    for entry in entries {
        let path = entry.expect("dir entry").path();
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if name == "mod" || path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = read(&path);
        // `kernel!(ident "symbol", …)` — the row world's declaration.
        for rest in src.split("kernel!(").skip(1) {
            let Some(open) = rest.find('"') else { continue };
            let Some(close) = rest[open + 1..].find('"').map(|c| c + open + 1) else {
                continue;
            };
            row_world.push((rest[open + 1..close].to_string(), format!("table/{name}.rs")));
        }
    }
    if row_world.is_empty() {
        return;
    }

    // Read fn-world once, not once per symbol: the walk is ~30 files and the
    // row set is ~60 symbols, and a gate that is quadratic in the thing it
    // guards gets deleted the first time someone profiles the test suite.
    let fn_world: Vec<(String, String)> = fn_world_files()
        .into_iter()
        .map(|path| {
            let shown = path
                .strip_prefix(x_dir().parent().expect("src/"))
                .unwrap_or(&path)
                .display()
                .to_string();
            (shown, read(&path))
        })
        .collect();

    for (symbol, table_file) in row_world {
        // `contract!` writes `symbol: "…"`; a `bind!` arm writes `"…" =>`.
        // Both are fn-world declarations and both make `sig_in` answer.
        let needles = [format!("symbol: \"{symbol}\""), format!("\"{symbol}\" =>")];
        for (shown, src) in &fn_world {
            if !needles.iter().any(|n| src.contains(n.as_str())) {
                continue;
            }
            panic!(
                "`{symbol}` is declared in BOTH {table_file} and {shown} \
                 -- the row world and fn-world both claim it.\n  \
                 `sig_in` returns the first match, so this does not error: it \
                 picks one by list order. The fn-world row states no operands \
                 and binds through a `fn`'s parameters; the row-world one \
                 states its full binding instruction and binds through the \
                 generated dispatcher. Two different KINDS of answer, chosen \
                 by accident.\n  \
                 The two filenames need not name the same family: `table/` is \
                 organised by who dispatches and `x/` by who owns the code.\n  \
                 A crossing deletes the `kernel!` in the same change that adds \
                 the `contract!`."
            );
        }
    }
}

/// Every fn-world source file, flat and nested.
///
/// A family is `x/<name>.rs` or `x/<name>/**.rs`, and the nested form is not
/// rare — `gemm` has been a directory since it crossed. Walking beats globbing
/// a fixed depth because a family that grows a third level would otherwise
/// leave this test silently smaller, which is the failure mode a gate can
/// least afford.
fn fn_world_files() -> Vec<std::path::PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(&x_dir(), &mut out);
    out.sort();
    out
}
