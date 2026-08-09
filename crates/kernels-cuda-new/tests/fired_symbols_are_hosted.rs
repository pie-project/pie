//! `x::fire::fire_ex` panics when no unit hosts the symbol it is handed, so a
//! symbol a fire site names and no `unit!` declares is a runtime abort that
//! nothing else in this crate's tests would reach.

use std::path::Path;

/// The first argument of every `raw::*` / `fire::fire*` call that is a string
/// literal, plus every `fam::name#arm` literal anywhere -- an arm suffix only
/// ever names an instantiation, never a service.
fn fired(src: &str, out: &mut Vec<String>) {
    let b = src.as_bytes();
    let lit = |from: usize| -> Option<(String, usize)> {
        let end = from + b[from..].iter().position(|&c| c == b'"')?;
        Some((String::from_utf8_lossy(&b[from..end]).into_owned(), end))
    };
    let mut i = 0;
    while i < b.len() {
        if b[i..].starts_with(b"raw::") || b[i..].starts_with(b"fire::fire") {
            if let Some(open) = b[i..].iter().position(|&c| c == b'(') {
                let mut j = i + open + 1;
                while j < b.len() && b[j].is_ascii_whitespace() {
                    j += 1;
                }
                if b.get(j) == Some(&b'"')
                    && let Some((s, _)) = lit(j + 1)
                {
                    out.push(s);
                }
            }
        }
        if b[i] == b'"'
            && let Some((s, end)) = lit(i + 1)
        {
            if s.contains('#') && s.contains("::") && !s.contains(' ') {
                out.push(s);
            }
            i = end;
        }
        i += 1;
    }
}

fn walk(dir: &Path, out: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).expect("`src/x` is readable") {
        let path = entry.expect("a readable entry").path();
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            fired(&std::fs::read_to_string(&path).expect("a readable file"), out);
        }
    }
}

#[test]
fn every_symbol_a_fire_site_names_is_hosted_by_a_unit() {
    let mut symbols = Vec::new();
    walk(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src/x"), &mut symbols);
    symbols.sort_unstable();
    symbols.dedup();
    assert!(symbols.len() > 150, "the scan found {} symbols; it stopped matching", symbols.len());

    let missing: Vec<&String> = symbols
        .iter()
        .filter(|s| kernels_cuda_new::unit::unit_of(s).is_none())
        .collect();
    assert!(
        missing.is_empty(),
        "{} symbol(s) a fire site names are in no JIT unit, and firing one aborts: {missing:#?}",
        missing.len()
    );
}
