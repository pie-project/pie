//! Structural checks on every source the emitters produce.
//!
//! The goldens pin the bytes of the cases that were dumped once; they say
//! nothing about a case nobody dumped. Anything an emitter produces on a new
//! path -- a region shape the corpus never built, a channel count nobody
//! tried -- is unchecked until a device compiler sees it, and there is no
//! device here.
//!
//! So this walks the whole corpus through both backends and checks the
//! properties any C-family source has to have regardless of what it says.
//! That is a weaker claim than "this compiles", but it is checkable without a
//! toolchain and it covers cases the goldens do not: an unbalanced brace from
//! a `push_str` that forgot its closer, an unterminated string from a
//! `format!` that interpolated a newline, a kernel emitted with no entry
//! point in it.
//!
//! It does not check semantics, types, or declaration order. A source can
//! pass everything here and still be rejected by `nvrtc`.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;

use pie_codegen::program::{Backend, EmittedKernel, emit_program};
use pie_ir::validate::bind;
use pie_plan::compile_bound;

use msl_corpus::{
    GOLDEN_NAMES, extended_traces, golden_container, golden_profile, synthetic_traces,
};

/// Every kernel the corpus produces, tagged with where it came from.
fn every_emitted_kernel() -> Vec<(String, Backend, EmittedKernel)> {
    let mut traces: Vec<_> = GOLDEN_NAMES
        .iter()
        .map(|name| {
            (
                (*name).to_string(),
                golden_container(name),
                golden_profile(name),
            )
        })
        .collect();
    for (name, container, profile) in synthetic_traces().into_iter().chain(extended_traces()) {
        traces.push((name.to_string(), container, profile));
    }

    let mut out = Vec::new();
    for (name, container, profile) in traces {
        // The `neg_*` goldens exist to fail binding; they contribute nothing.
        let Ok(bound) = bind(container, profile) else {
            continue;
        };
        let stages = compile_bound(&bound);
        for backend in [Backend::Cuda, Backend::Metal] {
            for kernel in emit_program(backend, &stages, &bound) {
                out.push((name.clone(), backend, kernel));
            }
        }
    }
    out
}

/// Bracket depth after skipping comments and literals, or the first structural
/// complaint.
///
/// Written as a scanner rather than a `count('{')` because both backends emit
/// braces inside string literals (`"{...}"` in printf-style device asserts)
/// and inside comments, and counting those would make the check report
/// failures that are not there.
fn bracket_report(source: &str) -> Result<(), String> {
    let bytes = source.as_bytes();
    let mut stack: Vec<(char, usize)> = Vec::new();
    let mut line = 1usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i] as char;
        let next = bytes.get(i + 1).map(|b| *b as char);
        if c == '\n' {
            line += 1;
            i += 1;
            continue;
        }
        if c == '/' && next == Some('/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && next == Some('*') {
            let start = line;
            i += 2;
            loop {
                if i + 1 >= bytes.len() {
                    return Err(format!("unterminated block comment opened on line {start}"));
                }
                if bytes[i] == b'\n' {
                    line += 1;
                }
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            continue;
        }
        if c == '"' || c == '\'' {
            let quote = bytes[i];
            let start = line;
            i += 1;
            loop {
                if i >= bytes.len() || bytes[i] == b'\n' {
                    return Err(format!("unterminated literal on line {start}"));
                }
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == quote {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }
        match c {
            '{' | '(' | '[' => stack.push((c, line)),
            '}' | ')' | ']' => {
                let want = match c {
                    '}' => '{',
                    ')' => '(',
                    _ => '[',
                };
                match stack.pop() {
                    Some((open, _)) if open == want => {}
                    Some((open, opened)) => {
                        return Err(format!(
                            "line {line}: `{c}` closes `{open}` opened on line {opened}"
                        ));
                    }
                    None => return Err(format!("line {line}: `{c}` with nothing open")),
                }
            }
            _ => {}
        }
        i += 1;
    }
    match stack.first() {
        Some((open, opened)) => Err(format!("`{open}` opened on line {opened} never closes")),
        None => Ok(()),
    }
}

#[test]
fn every_emitted_source_is_structurally_closed() {
    let kernels = every_emitted_kernel();
    // 861 today. A floor, not a pin: it catches the harness silently
    // stopping at zero kernels, which would make everything below vacuous.
    assert!(
        kernels.len() > 800,
        "the corpus produced only {} kernels; the harness is not reaching the emitters",
        kernels.len()
    );
    for (trace, backend, kernel) in &kernels {
        if kernel.source.is_empty() {
            continue;
        }
        if let Err(complaint) = bracket_report(&kernel.source) {
            panic!(
                "{trace} {backend:?} stage {} region {} ({}): {complaint}",
                kernel.stage_index, kernel.region_index, kernel.entry_name
            );
        }
    }
}

/// A kernel is either a source with an entry point in it, or a refusal with a
/// reason. `EmittedKernel::new` is what makes that true, and the drivers join
/// on it: an empty source with an empty error reads as a kernel that emitted
/// successfully to nothing.
#[test]
fn every_emitted_kernel_is_a_source_or_a_reason() {
    for (trace, backend, kernel) in every_emitted_kernel() {
        let where_ = format!(
            "{trace} {backend:?} stage {} region {}",
            kernel.stage_index, kernel.region_index
        );
        assert_ne!(
            kernel.source.is_empty(),
            kernel.error.is_empty(),
            "{where_}: source and error must be exactly one of the two \
             (source {} bytes, error {:?})",
            kernel.source.len(),
            kernel.error
        );
        if kernel.source.is_empty() {
            assert!(
                kernel.entry_name.is_empty(),
                "{where_}: a refusal still named an entry point {:?}",
                kernel.entry_name
            );
            continue;
        }
        assert!(
            !kernel.entry_name.is_empty(),
            "{where_}: emitted a source with no entry name"
        );
        assert!(
            kernel.source.contains(&kernel.entry_name),
            "{where_}: source does not define its own entry point {:?}",
            kernel.entry_name
        );
    }
}

/// The scanner has to be wrong in the safe direction: braces inside literals
/// and comments are not structure. Both emitters put them there.
#[test]
fn the_bracket_scanner_ignores_literals_and_comments() {
    assert!(bracket_report("void f() { }\n").is_ok());
    assert!(bracket_report("const char* s = \"{{{\";\n").is_ok());
    assert!(bracket_report("/* } } } */ void f() {}\n").is_ok());
    assert!(bracket_report("// }\nvoid f() {}\n").is_ok());
    assert!(bracket_report("char c = '}';\n").is_ok());
    assert!(bracket_report("const char* s = \"\\\"}\";\n").is_ok());

    assert!(bracket_report("void f() {\n").is_err());
    assert!(bracket_report("void f() }\n").is_err());
    assert!(bracket_report("void f( ) { ] }\n").is_err());
    assert!(bracket_report("const char* s = \"abc;\n").is_err());
    assert!(bracket_report("/* never closed\n").is_err());
}
