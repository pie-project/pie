//! The CSL kernel library: every op one function over pointers and runtime
//! sizes, so a program's phases are calls and its code size does not grow
//! with the number of ops in a fire. Each function (with the helpers only it
//! uses) is one file, `kernels/<group>/<name>.csl`, looked up by its path
//! (`elemwise/rmsnorm`) and called as `k_<name>`; a program carries only the
//! units its phases call, since a PE holds 48 KB of code and data together.
//! CSL has no namespaces and a program is one text, so `k_<name>` is unique
//! across the tree.

include!(concat!(env!("OUT_DIR"), "/units.rs"));

/// The unit at `path` (`elemwise/rmsnorm`): its call name and its text.
#[must_use]
pub fn unit(path: &str) -> Option<(&'static str, &'static str)> {
    UNITS
        .iter()
        .find(|(p, ..)| *p == path)
        .map(|(_, name, text)| (*name, *text))
}

/// The prelude every program carries: base DSDs, the dot reducer, copies.
#[must_use]
pub fn prelude() -> &'static str {
    unit("prim/prelude").expect("the prelude unit").1
}

/// The parameter types of CSL function `name` (`fn name(a: [*]f32, n: i32,
/// flag: bool) void` gives `["[*]f32", "i32", "bool"]`), found in whichever
/// unit declares it; `None` when no such function exists.
pub fn signature(name: &str) -> Option<Vec<String>> {
    let marker = format!("fn {name}(");
    let text = UNITS
        .iter()
        .map(|(.., t)| *t)
        .find(|t| t.contains(&marker))?;
    let start = text.find(&marker)? + marker.len();
    let rest = &text[start..];
    let end = rest.find(") ")?;
    let params = &rest[..end];
    Some(
        params
            .split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(|p| p.split_once(':').map_or(p, |(_, t)| t).trim().to_string())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_are_found_by_path() {
        assert!(prelude().contains("fn k_round_bf16"));
        assert_eq!(
            signature("k_matmul").unwrap(),
            ["[*]f32", "[*]f32", "[*]f32", "i32", "i32", "i32"]
        );
        assert_eq!(signature("k_round_bf16").unwrap(), ["[*]f32", "i32"]);
        assert!(signature("k_attend").unwrap().len() > 30);
        assert!(signature("nope").is_none());
        let (name, text) = unit("linear/matmul").unwrap();
        assert_eq!(name, "k_matmul");
        assert!(text.starts_with("// y[m, n]"));
        assert!(unit("linear/lora").unwrap().1.contains("fn k_lora("));
        assert!(unit("k_lora").is_none());
    }

    /// A unit holds the function it is named for (`prim/` excepted: its
    /// units are helper sets, the prelude and the packed-bf16 codec), and no
    /// function is declared in two units.
    #[test]
    fn every_unit_declares_its_name_and_no_name_twice() {
        let mut seen = std::collections::BTreeMap::new();
        for (path, name, text) in UNITS {
            assert!(
                path.starts_with("prim/") || text.contains(&format!("fn {name}(")),
                "{path} declares no `fn {name}(`"
            );
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("fn ") {
                    let f = rest.split('(').next().expect("a name").to_string();
                    if let Some(prior) = seen.insert(f.clone(), *path) {
                        panic!("fn `{f}` is in {prior} and {path}");
                    }
                }
            }
        }
    }
}
