//! The CSL kernel library (`kernels.csl`), served section by section so a
//! program carries only the functions its phases call.

const SOURCE: &str = include_str!("kernels.csl");

/// The library section named `name` (its `// @@ name` block), or none.
#[must_use]
pub fn section(name: &str) -> Option<&'static str> {
    let marker = format!("// @@ {name}\n");
    let start = SOURCE.find(&marker)? + marker.len();
    let rest = &SOURCE[start..];
    let end = rest.find("\n// @@ ").map_or(rest.len(), |at| at + 1);
    Some(&rest[..end])
}

/// The prelude every program carries: base DSDs, the dot reducer, copies.
#[must_use]
pub fn prelude() -> &'static str {
    section("k_prelude").expect("the prelude section")
}

/// The parameter types of kernel `name` (`fn name(a: [*]f32, n: i32, flag:
/// bool) void` gives `["[*]f32", "i32", "bool"]`), found in its own section
/// or the prelude; `None` when no such function exists.
pub fn signature(name: &str) -> Option<Vec<String>> {
    let marker = format!("fn {name}(");
    let start = SOURCE.find(&marker)? + marker.len();
    let rest = &SOURCE[start..];
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
    fn sections_are_found_by_name() {
        assert!(prelude().contains("fn k_round_bf16"));
        assert_eq!(
            signature("k_matmul").unwrap(),
            ["[*]f32", "[*]f32", "[*]f32", "i32", "i32", "i32"]
        );
        assert_eq!(signature("k_round_bf16").unwrap(), ["[*]f32", "i32"]);
        assert!(signature("k_attend").unwrap().len() > 30);
        assert!(signature("nope").is_none());
        assert!(section("k_matmul").unwrap().starts_with("// y[m, n]"));
        assert!(section("k_lora").unwrap().contains("fn k_lora("));
        assert!(section("nope").is_none());
    }
}
