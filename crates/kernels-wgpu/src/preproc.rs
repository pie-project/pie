//! The variant expander: what a `-D` is when the language has no preprocessor.
//!
//! ## Why this exists at all
//!
//! An entrypoint in this crate is a **module**: one `.wgsl` source plus one set
//! of defines. That is the shape `kernels-vulkan` uses, which is the shape
//! llama.cpp's `vulkan-shaders-gen` uses, and it is not a stylistic choice —
//! a pipeline-overridable constant cannot vary a TYPE, a binding layout, or a
//! loop trip count that has to be known when the module is parsed. Sixty of the
//! 480 entrypoints differ in exactly those ways.
//!
//! GLSL gets this for free: `glslc -DPIE_BITS=4` is a preprocessor that already
//! exists. **WGSL has no preprocessor at all** — no `#if`, no `#include`, no
//! `#define` — so a WGSL backend either gives up compile-time variants or
//! brings its own. This module is the second answer, in about three hundred
//! lines of Rust with no dependency, which is a price worth naming out loud:
//! it is the whole cost of the toolchain the other two backends pay an SDK for.
//!
//! ## What it is not
//!
//! It is not C. It does not expand function-like macros, it does not
//! concatenate tokens, and it does not substitute identifiers textually. Those
//! are the parts of a preprocessor that make a shader tree unreadable, and none
//! of them are needed — WGSL has `const`, and an abstract-numeric `const`
//! coerces into an array size, a `@workgroup_size` and an arithmetic
//! expression alike. So a numeric define becomes a real `const` declaration in
//! a prelude, and the language does the rest:
//!
//! ```text
//! // pie:instantiate rms_single_row_bfloat16 N_READS=4
//! ```
//!
//! puts `const N_READS = 4;` at the top of the module, and the body writes
//! `N_READS` as an ordinary identifier that an editor, a formatter and `naga`
//! all understand. Nothing is hidden behind a substitution.
//!
//! What conditional compilation IS still needed for is structure: a variant
//! that binds a different buffer set, or takes a different branch of an
//! algorithm, cannot be a `const`. [`expand`] therefore keeps `//#if`, and
//! nothing else.
//!
//! ## Directives are comments, on purpose
//!
//! Every directive is prefixed `//#`, so a file with directives in it is still
//! syntactically a WGSL comment stream. `naga` will not parse an unexpanded
//! file — the `//#if` bodies would both be present — but an editor highlights
//! it, `wgsl-analyzer` reads it, and a diff against `kernels-vulkan`'s `.comp`
//! lines up. A leading `#` would have cost all three for nothing.
//!
//! | directive | meaning |
//! |---|---|
//! | `//#include "common/bf16.inc.wgsl"` | splice a fragment, once per module |
//! | `//#if <cond>` / `//#elif <cond>` / `//#else` / `//#endif` | keep one arm |
//! | `// pie:instantiate NAME [@tier] K=V ...` | declare a variant of this file |
//!
//! A condition is `defined(K)`, `!defined(K)`, or a comparison `K <op> <int>`
//! with `op` one of `== != < <= > >=`, joined by `&&` and `||` with `&&`
//! binding tighter. There are no parentheses and there is no arithmetic. That
//! is deliberately less than C: every condition in the tree fits it, and the
//! grammar is small enough that this module can be read in one sitting, which
//! is the property that matters for something with no test but its own.
//!
//! ## One parser, not two
//!
//! `kernels-vulkan` parses its `pie:instantiate` lines in `build.rs` and again
//! in `scripts/vulkan-kernel-audit.py`, and its own notes call the duplication
//! "intentional" — the build and the audit must not be able to disagree. That
//! is a good property bought at a bad price. Here the parser is a library
//! function: `build.rs` calls it through `#[path]`, the tests call it, and a
//! driver compiling a module at pipeline-creation time calls the very same
//! code. They cannot disagree because there is only one of them.

use std::collections::BTreeMap;
use std::fmt;

/// One `// pie:instantiate` line: a variant this file declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variant {
    /// The entrypoint name — one of the 480 the table's axes produce.
    pub entrypoint: String,
    /// The capability tier, from a `@tag` after the name. Baseline if absent.
    pub tier: crate::Capability,
    /// The defines, in the order the line states them.
    pub defines: BTreeMap<String, String>,
    /// The 1-based line the directive sits on, for an error a person can find.
    pub line: usize,
}

/// A `//#`-prefixed line, once recognised.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Directive {
    /// `//#include "path"`.
    Include(String),
    /// `//#if <cond>`.
    If(String),
    /// `//#elif <cond>`.
    Elif(String),
    /// `//#else`.
    Else,
    /// `//#endif`.
    Endif,
}

/// What a source can be wrong about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Malformed {
    /// A directive this module does not know.
    Unknown { line: usize, what: String },
    /// `//#elif`, `//#else` or `//#endif` with no `//#if` open.
    Unopened { line: usize, what: String },
    /// A `//#if` that runs off the end of the file.
    Unclosed { line: usize },
    /// `//#elif` after `//#else`.
    AfterElse { line: usize },
    /// A condition the grammar in this module's docs does not cover.
    Uncondition { line: usize, cond: String },
    /// `//#include` naming a file the tree does not have.
    Unincluded { line: usize, path: String },
    /// A `// pie:instantiate` line with no entrypoint name.
    Unnamed { line: usize },
    /// A `@tier` tag [`crate::Capability::from_tag`] does not know.
    Untiered { line: usize, tag: String },
    /// A define with no `=`.
    Undefined { line: usize, what: String },
}

impl fmt::Display for Malformed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown { line, what } => write!(f, "line {line}: unknown directive `{what}`"),
            Self::Unopened { line, what } => {
                write!(f, "line {line}: `{what}` with no `//#if` open")
            }
            Self::Unclosed { line } => write!(f, "line {line}: `//#if` is never closed"),
            Self::AfterElse { line } => write!(f, "line {line}: `//#elif` after `//#else`"),
            Self::Uncondition { line, cond } => {
                write!(
                    f,
                    "line {line}: `{cond}` is not a condition this expander reads"
                )
            }
            Self::Unincluded { line, path } => write!(f, "line {line}: no such include `{path}`"),
            Self::Unnamed { line } => {
                write!(f, "line {line}: `pie:instantiate` names no entrypoint")
            }
            Self::Untiered { line, tag } => write!(f, "line {line}: `@{tag}` is not a tier"),
            Self::Undefined { line, what } => write!(f, "line {line}: `{what}` is not `KEY=VALUE`"),
        }
    }
}

impl std::error::Error for Malformed {}

/// The `// pie:instantiate` lines of one source, in file order.
///
/// A file with none declares no entrypoints, which is what a `.inc.wgsl`
/// fragment is. That is not an error and the caller should not treat it as one.
///
/// # Errors
///
/// [`Malformed::Unnamed`], [`Malformed::Untiered`] or [`Malformed::Undefined`]
/// when a directive is present and unreadable — never for a file that simply
/// has none.
pub fn instantiations(text: &str) -> Result<Vec<Variant>, Malformed> {
    let mut out = Vec::new();
    for (at, raw) in text.lines().enumerate() {
        let line = at + 1;
        let Some(rest) = directive_body(raw, "pie:instantiate") else {
            continue;
        };
        let mut words = rest.split_whitespace();
        let Some(entrypoint) = words.next() else {
            return Err(Malformed::Unnamed { line });
        };

        let mut tier = crate::Capability::Baseline;
        let mut defines = BTreeMap::new();
        for word in words {
            if let Some(tag) = word.strip_prefix('@') {
                tier = crate::Capability::from_tag(tag).ok_or_else(|| Malformed::Untiered {
                    line,
                    tag: tag.to_owned(),
                })?;
                continue;
            }
            let Some((key, value)) = word.split_once('=') else {
                return Err(Malformed::Undefined {
                    line,
                    what: word.to_owned(),
                });
            };
            defines.insert(key.to_owned(), value.to_owned());
        }

        out.push(Variant {
            entrypoint: entrypoint.to_owned(),
            tier,
            defines,
            line,
        });
    }
    Ok(out)
}

/// The body of a `// <keyword>` line, or `None` if this line is not one.
///
/// Leading whitespace is allowed, and so is the `//#` spelling the structural
/// directives use — `directive_body(l, "#if")` reads `//#if`.
fn directive_body<'a>(raw: &'a str, keyword: &str) -> Option<&'a str> {
    let rest = raw.trim_start().strip_prefix("//")?.trim_start();
    let rest = rest.strip_prefix(keyword)?;
    // `//#if` must not match `//#ifdef`, and `// pie:instantiateX` must not
    // match `pie:instantiate`. A keyword ends at a boundary or at the line.
    if rest.is_empty() || rest.starts_with(char::is_whitespace) {
        Some(rest.trim())
    } else {
        None
    }
}

/// Expand one source into the WGSL a variant compiles from.
///
/// `includes` resolves an `//#include` path to a fragment's text; a tree on
/// disk and a tree embedded in the rlib both satisfy it, which is what lets
/// `build.rs` and a running driver share this function.
///
/// The result is a prelude of `const` declarations — one for every define whose
/// value is a number — followed by the source with its `//#if` arms resolved
/// and its includes spliced. Non-numeric defines get no `const`, because there
/// is nothing in WGSL for them to be; they exist for `defined()` and for
/// comparison, and a body that wanted one as a value would be a body wanting a
/// substitution this expander deliberately does not do.
///
/// # Errors
///
/// Any [`Malformed`] the source or its includes contain.
pub fn expand(
    text: &str,
    defines: &BTreeMap<String, String>,
    includes: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Malformed> {
    let mut prelude = String::new();

    prelude.push_str("// Generated by kernels-wgpu's expander. Do not edit.\n");
    for (key, value) in defines {
        if is_number(value) {
            prelude.push_str(&format!("const {key} = {value};\n"));
        } else {
            prelude.push_str(&format!(
                "// define {key}={value} (not a number: no const)\n"
            ));
        }
    }

    let mut body = String::new();
    let mut seen = Vec::new();
    splice(text, defines, includes, &mut seen, &mut body)?;

    Ok(hoist_enables(&prelude, &body))
}

/// Move every `enable` directive above the generated `const` prelude.
///
/// WGSL requires all `enable` directives to precede every declaration in the
/// module — they are directives, not statements, and `naga` refuses one that
/// follows a `const`. The prelude is a run of `const` declarations, so a
/// fragment that opens with `enable subgroups;` lands *after* them and the
/// module stops parsing.
///
/// That is a latent failure rather than a live one: no variant in the tree
/// carries both a numeric define and a tier whose include enables an extension.
/// It is fixed here anyway, because the day one does the error would arrive as
/// "expected declaration, found `enable`" pointing at a line the author did not
/// write, in a file the author cannot see, generated by a step that is supposed
/// to be invisible.
///
/// Hoisting is safe in a way that reordering generally is not: an `enable` has
/// no operands, no ordering relationship with any other directive, and a
/// duplicate is legal. So the only thing this can change is a module that was
/// going to fail.
fn hoist_enables(prelude: &str, body: &str) -> String {
    let mut enables: Vec<&str> = Vec::new();
    let mut rest = String::with_capacity(body.len());

    for line in body.lines() {
        if line.trim_start().starts_with("enable ") {
            let line = line.trim();
            if !enables.contains(&line) {
                enables.push(line);
            }
            continue;
        }
        rest.push_str(line);
        rest.push('\n');
    }

    let mut out = String::with_capacity(prelude.len() + rest.len() + 64);
    for enable in enables {
        out.push_str(enable);
        out.push('\n');
    }
    out.push_str(prelude);
    out.push_str(&rest);
    out
}

/// Whether a define's value is something WGSL can hold in a `const`.
///
/// Integers, floats and the `u`/`i`/`f` suffixes WGSL spells them with. A
/// hexadecimal literal counts, because a mask is a natural thing for a variant
/// to carry.
fn is_number(value: &str) -> bool {
    let body = value
        .trim_end_matches(['u', 'i', 'f'])
        .trim_start_matches('-');
    if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        return !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit());
    }
    !body.is_empty()
        && body.chars().all(|c| c.is_ascii_digit() || c == '.')
        && body.chars().filter(|c| *c == '.').count() <= 1
        && body.chars().any(|c| c.is_ascii_digit())
}

/// One source's worth of splicing, recursing through its includes.
fn splice(
    text: &str,
    defines: &BTreeMap<String, String>,
    includes: &dyn Fn(&str) -> Option<String>,
    seen: &mut Vec<String>,
    out: &mut String,
) -> Result<(), Malformed> {
    // One frame per open `//#if`: (this arm is live, some earlier arm was, saw
    // an `//#else`).
    struct Frame {
        live: bool,
        taken: bool,
        elsed: bool,
        line: usize,
    }
    let mut stack: Vec<Frame> = Vec::new();

    for (at, raw) in text.lines().enumerate() {
        let line = at + 1;
        let live = stack.iter().all(|f| f.live);

        if let Some(cond) = directive_body(raw, "#if") {
            let take = live && truth(cond, defines, line)?;
            stack.push(Frame {
                live: take,
                taken: take,
                elsed: false,
                line,
            });
            continue;
        }
        if let Some(cond) = directive_body(raw, "#elif") {
            let Some(frame) = stack.last_mut() else {
                return Err(Malformed::Unopened {
                    line,
                    what: "//#elif".to_owned(),
                });
            };
            if frame.elsed {
                return Err(Malformed::AfterElse { line });
            }
            // The enclosing arms, which `live` above already folded in, minus
            // this frame's own -- an `//#elif` inside a dead `//#if` is dead
            // whatever it says, and must not be evaluated as if it were not.
            let outer = stack[..stack.len() - 1].iter().all(|f| f.live);
            let frame = stack.last_mut().expect("just checked");
            let take = outer && !frame.taken && truth(cond, defines, line)?;
            frame.live = take;
            frame.taken |= take;
            continue;
        }
        if directive_body(raw, "#else").is_some() {
            let outer = stack
                .len()
                .checked_sub(1)
                .map(|n| stack[..n].iter().all(|f| f.live));
            let Some(outer) = outer else {
                return Err(Malformed::Unopened {
                    line,
                    what: "//#else".to_owned(),
                });
            };
            let frame = stack
                .last_mut()
                .expect("outer proved the stack is not empty");
            if frame.elsed {
                return Err(Malformed::AfterElse { line });
            }
            frame.elsed = true;
            frame.live = outer && !frame.taken;
            frame.taken = true;
            continue;
        }
        if directive_body(raw, "#endif").is_some() {
            if stack.pop().is_none() {
                return Err(Malformed::Unopened {
                    line,
                    what: "//#endif".to_owned(),
                });
            }
            continue;
        }
        if let Some(path) = directive_body(raw, "#include") {
            if !live {
                continue;
            }
            let path = path.trim().trim_matches('"').to_owned();
            // Once per module, not once per `//#include`. Two files including
            // the same fragment is the normal case, and a second copy of a
            // `fn` is a WGSL redefinition error rather than a warning.
            if seen.contains(&path) {
                continue;
            }
            let Some(body) = includes(&path) else {
                return Err(Malformed::Unincluded { line, path });
            };
            seen.push(path);
            splice(&body, defines, includes, seen, out)?;
            continue;
        }
        if let Some(what) = unknown_directive(raw) {
            return Err(Malformed::Unknown { line, what });
        }

        if live {
            out.push_str(raw);
            out.push('\n');
        }
    }

    match stack.first() {
        Some(frame) => Err(Malformed::Unclosed { line: frame.line }),
        None => Ok(()),
    }
}

/// A `//#`-prefixed line none of the known directives claimed.
///
/// Caught rather than passed through, because a typo in `//#endif` that is
/// treated as a comment silently deletes the rest of a shader.
fn unknown_directive(raw: &str) -> Option<String> {
    let rest = raw.trim_start().strip_prefix("//")?;
    let rest = rest.strip_prefix('#')?;
    let word: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    Some(format!("//#{word}"))
}

/// Whether a condition holds, under the grammar this module's docs state.
fn truth(cond: &str, defines: &BTreeMap<String, String>, line: usize) -> Result<bool, Malformed> {
    let mut any = false;
    for clause in cond.split("||") {
        let mut all = true;
        for term in clause.split("&&") {
            all &= term_truth(term.trim(), defines, line)?;
        }
        any |= all;
    }
    Ok(any)
}

/// One `&&`-joined term.
fn term_truth(
    term: &str,
    defines: &BTreeMap<String, String>,
    line: usize,
) -> Result<bool, Malformed> {
    let uncondition = || Malformed::Uncondition {
        line,
        cond: term.to_owned(),
    };

    if let Some(rest) = term.strip_prefix('!') {
        return Ok(!term_truth(rest.trim(), defines, line)?);
    }
    if let Some(rest) = term.strip_prefix("defined") {
        let key = rest
            .trim()
            .strip_prefix('(')
            .and_then(|r| r.strip_suffix(')'))
            .ok_or_else(uncondition)?;
        return Ok(defines.contains_key(key.trim()));
    }

    // `>=` and `<=` must be tried before `>` and `<`, or `A >= 1` parses as
    // `A > (= 1)` and compares against a value that is not a number.
    for op in ["==", "!=", ">=", "<=", ">", "<"] {
        let Some((key, want)) = term.split_once(op) else {
            continue;
        };
        let key = key.trim();
        let want = want.trim();
        let Some(have) = defines.get(key) else {
            // An undefined key compares as absent rather than as zero. C would
            // say zero; C is wrong about this often enough that llama.cpp's
            // shader tree carries a lint for it, and a false `==` is the safe
            // reading for a variant that was never instantiated.
            return Ok(op == "!=");
        };
        let (Ok(have), Ok(want)) = (have.trim().parse::<i64>(), want.parse::<i64>()) else {
            // A non-numeric comparison is a string one, which `==`/`!=` can
            // still answer and the orderings cannot.
            return match op {
                "==" => Ok(have.trim() == want),
                "!=" => Ok(have.trim() != want),
                _ => Err(uncondition()),
            };
        };
        return Ok(match op {
            "==" => have == want,
            "!=" => have != want,
            ">=" => have >= want,
            "<=" => have <= want,
            ">" => have > want,
            "<" => have < want,
            _ => unreachable!("the list above is the list matched here"),
        });
    }

    Err(uncondition())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defines(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    fn no_includes(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn a_directive_line_is_a_comment_a_wgsl_tool_can_read() {
        // The whole reason for the `//#` spelling: strip the directives and
        // what is left is still a file, rather than a file with `#` in it.
        assert_eq!(
            directive_body("//#if PIE_BITS == 4", "#if"),
            Some("PIE_BITS == 4")
        );
        assert_eq!(directive_body("  //#endif", "#endif"), Some(""));
        assert_eq!(
            directive_body("// pie:instantiate x A=1", "pie:instantiate"),
            Some("x A=1")
        );
        // A keyword ends at a boundary: `//#ifdef` is not `//#if`.
        assert_eq!(directive_body("//#ifdef X", "#if"), None);
    }

    #[test]
    fn an_instantiate_line_carries_a_name_a_tier_and_defines() {
        let text = "// pie:instantiate sdpa_paged_mma_bfloat16_d_64 @subgroup PIE_HEAD_DIM=64\n";
        let got = instantiations(text).expect("the line is well formed");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].entrypoint, "sdpa_paged_mma_bfloat16_d_64");
        assert_eq!(got[0].tier, crate::Capability::Subgroup);
        assert_eq!(got[0].defines, defines(&[("PIE_HEAD_DIM", "64")]));
        assert_eq!(got[0].line, 1);
    }

    #[test]
    fn a_file_with_no_directives_declares_no_entrypoints() {
        // What a `.inc.wgsl` fragment is. Not an error.
        let got = instantiations("fn f() -> f32 { return 1.0; }\n").expect("no directives");
        assert!(got.is_empty());
    }

    #[test]
    fn a_numeric_define_becomes_a_const_and_a_word_does_not() {
        let out = expand(
            "// body\n",
            &defines(&[("N_READS", "4"), ("T", "bf16")]),
            &no_includes,
        )
        .expect("no directives to go wrong");
        assert!(out.contains("const N_READS = 4;"), "{out}");
        assert!(
            !out.contains("const T ="),
            "a word is not a WGSL value: {out}"
        );
    }

    #[test]
    fn only_the_live_arm_survives() {
        let text = "\
//#if defined(PIE_STRIDED)
strided
//#else
flat
//#endif
";
        let strided = expand(text, &defines(&[("PIE_STRIDED", "1")]), &no_includes).unwrap();
        assert!(strided.contains("strided") && !strided.contains("flat"));

        let flat = expand(text, &defines(&[]), &no_includes).unwrap();
        assert!(flat.contains("flat") && !flat.contains("strided"));
    }

    #[test]
    fn an_elif_chain_takes_the_first_true_arm_and_no_other() {
        let text = "\
//#if PIE_BITS == 4
four
//#elif PIE_BITS == 8
eight
//#else
other
//#endif
";
        for (bits, want, gone) in [("4", "four", "eight"), ("8", "eight", "four")] {
            let out = expand(text, &defines(&[("PIE_BITS", bits)]), &no_includes).unwrap();
            assert!(
                out.contains(want) && !out.contains(gone),
                "bits={bits}: {out}"
            );
            assert!(!out.contains("other"));
        }
        let out = expand(text, &defines(&[("PIE_BITS", "2")]), &no_includes).unwrap();
        assert!(out.contains("other"));
    }

    /// The bug this expander was most likely to have.
    ///
    /// An `//#elif` inside a DEAD `//#if` must stay dead whatever it says. A
    /// naive implementation evaluates it against the defines and revives the
    /// arm, which splices two mutually exclusive bodies into one module — and
    /// because both halves are valid WGSL on their own, the failure is a
    /// duplicate-definition error a long way from the cause.
    #[test]
    fn a_nested_arm_of_a_dead_branch_stays_dead() {
        let text = "\
//#if defined(OUTER)
//#if A == 1
inner_a
//#elif A == 2
inner_b
//#else
inner_c
//#endif
//#endif
after
";
        let out = expand(text, &defines(&[("A", "2")]), &no_includes).unwrap();
        assert!(!out.contains("inner_a"), "{out}");
        assert!(!out.contains("inner_b"), "the outer arm is dead: {out}");
        assert!(!out.contains("inner_c"), "{out}");
        assert!(
            out.contains("after"),
            "the file continues past the frame: {out}"
        );
    }

    #[test]
    fn an_include_is_spliced_once_however_many_ask_for_it() {
        let text = "//#include \"a.inc.wgsl\"\n//#include \"a.inc.wgsl\"\nbody\n";
        let out = expand(text, &defines(&[]), &|path| {
            (path == "a.inc.wgsl").then(|| "fragment\n".to_owned())
        })
        .unwrap();
        assert_eq!(
            out.matches("fragment").count(),
            1,
            "a second copy is a redefinition: {out}"
        );
        assert!(out.contains("body"));
    }

    #[test]
    fn an_include_inside_a_dead_arm_is_not_read() {
        let text = "//#if defined(NEVER)\n//#include \"missing.inc.wgsl\"\n//#endif\nbody\n";
        let out = expand(text, &defines(&[]), &no_includes).expect("the dead arm is not resolved");
        assert!(out.contains("body"));
    }

    /// An `enable` is hoisted above the generated `const` prelude.
    ///
    /// WGSL requires every `enable` to precede every declaration. The prelude
    /// is declarations, so without the hoist a fragment that enables an
    /// extension produces a module that does not parse — and the error points
    /// at a generated line the author never wrote.
    #[test]
    fn an_enable_directive_is_hoisted_above_the_prelude() {
        let out = expand(
            "enable subgroups;\nfn f() {}\n",
            &defines(&[("N_READS", "4")]),
            &no_includes,
        )
        .expect("nothing to go wrong");

        let enable = out
            .find("enable subgroups;")
            .expect("the directive survives");
        let konst = out.find("const N_READS").expect("the prelude survives");
        assert!(
            enable < konst,
            "an `enable` after a declaration does not parse:\n{out}",
        );
        assert_eq!(
            out.matches("enable subgroups;").count(),
            1,
            "hoisted once, not duplicated: {out}",
        );
    }

    /// Two includes that both enable the same extension yield one directive.
    ///
    /// Legal either way — WGSL permits a duplicate `enable` — but a module
    /// whose head is a wall of repeats is a module nobody reads.
    #[test]
    fn a_repeated_enable_is_collapsed() {
        let out = expand(
            "enable f16;\nenable f16;\nfn f() {}\n",
            &defines(&[]),
            &no_includes,
        )
        .expect("nothing to go wrong");
        assert_eq!(out.matches("enable f16;").count(), 1, "{out}");
    }

    #[test]
    fn a_malformed_source_names_its_line() {
        let cases: &[(&str, Malformed)] = &[
            (
                "//#endif\n",
                Malformed::Unopened {
                    line: 1,
                    what: "//#endif".into(),
                },
            ),
            ("//#if defined(X)\n", Malformed::Unclosed { line: 1 }),
            (
                "//#if defined(X)\n//#else\n//#elif defined(Y)\n//#endif\n",
                Malformed::AfterElse { line: 3 },
            ),
            (
                "//#if X ~ 1\n//#endif\n",
                Malformed::Uncondition {
                    line: 1,
                    cond: "X ~ 1".into(),
                },
            ),
            (
                "//#ifdef X\n",
                Malformed::Unknown {
                    line: 1,
                    what: "//#ifdef".into(),
                },
            ),
            (
                "//#include \"nope.inc.wgsl\"\n",
                Malformed::Unincluded {
                    line: 1,
                    path: "nope.inc.wgsl".into(),
                },
            ),
        ];
        for (text, want) in cases {
            let got = expand(text, &defines(&[]), &no_includes).unwrap_err();
            assert_eq!(got, *want, "for {text:?}");
        }
    }

    #[test]
    fn a_typo_in_a_directive_is_an_error_and_not_a_comment() {
        // The failure this prevents: `//#endfi` read as prose leaves the frame
        // open, and the rest of the shader disappears into a dead arm.
        let err = expand("//#if defined(X)\n//#endfi\n", &defines(&[]), &no_includes).unwrap_err();
        assert_eq!(
            err,
            Malformed::Unknown {
                line: 2,
                what: "//#endfi".into()
            }
        );
    }

    #[test]
    fn the_condition_grammar_covers_what_the_tree_uses() {
        let d = defines(&[("PIE_BITS", "4"), ("PIE_GROUP", "64"), ("T", "bf16")]);
        for (cond, want) in [
            ("defined(PIE_BITS)", true),
            ("!defined(PIE_MXFP4)", true),
            ("PIE_BITS == 4", true),
            ("PIE_BITS != 4", false),
            ("PIE_GROUP >= 64", true),
            ("PIE_GROUP > 64", false),
            ("PIE_GROUP <= 32", false),
            ("T == bf16", true),
            ("PIE_BITS == 4 && PIE_GROUP == 64", true),
            ("PIE_BITS == 8 || PIE_GROUP == 64", true),
            ("PIE_BITS == 8 && PIE_GROUP == 64 || defined(T)", true),
            // An undefined key is ABSENT, not zero.
            ("PIE_MXFP4 == 0", false),
            ("PIE_MXFP4 != 0", true),
        ] {
            assert_eq!(truth(cond, &d, 1).unwrap(), want, "for `{cond}`");
        }
    }
}
