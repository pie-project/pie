//! Two claims about the device text this crate owns, rehomed from the archive.
//!
//! Both lived in `kernels-cuda/tests/sources.rs` until that crate was deleted,
//! and neither was ever really about it. The first walked THIS crate's `kernels/`
//! from inside that one; the second was quantified over a set this crate
//! produces. A guard whose subject and whose host are two different crates
//! survives only until one of them is deleted, and then it goes silently —
//! which is what made finding them the precondition for the deletion rather
//! than a step in it.
//!
//! # Why the counts here are derived and not pinned
//!
//! At the move the walk saw 121 `.cuh` holding 371 `__global__` definitions
//! under 371 distinct qualified names. Those three numbers are recorded here
//! as an observation with a date on it and are asserted NOWHERE: a number that
//! names a length is the defect this tree has caught most often, because it
//! agrees with the tree on the day it is written and afterwards only ever
//! agrees with the past. Every assertion below is structural — it compares two
//! things the same walk derived, so it stays true as the tree grows and fails
//! only when the property genuinely breaks.

use std::path::{Path, PathBuf};

/// This crate's device sources.
fn kernels_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels")
}

/// Every `.cuh` under `kernels/`, as absolute paths.
///
/// Panics rather than returning empty if the root is missing. The version of
/// this walk in the archive returned `Vec::new()` on a failed `read_dir` and
/// skipped unreadable files with `if let Ok(..)`, so a moved directory or a
/// permission error would have made every caller pass by finding nothing.
fn device_headers() -> Vec<PathBuf> {
    let root = kernels_dir();
    assert!(
        root.is_dir(),
        "{root:?} does not read. Every assertion in this file is quantified \
         over the files under it, so a missing root makes all of them \
         vacuously true — the failure this walk is written to refuse."
    );
    let mut out = Vec::new();
    walk(&root, &mut out);
    out.sort();
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{dir:?} reads: {e}"));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("{dir:?} entry reads: {e}"))
            .path();
        if path.is_dir() {
            if path
                .file_name()
                .is_some_and(|n| n == "third_party" || n == "vendor")
            {
                continue;
            }
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "cuh") {
            out.push(path);
        }
    }
}

/// A `__global__` may be defined once.
///
/// Two definitions of one name is a half-finished migration: one copy gets
/// edited and the other drifts, each right for whichever half of the tree its
/// tests exercise. `norm/altup_aux` shipped exactly that for a release with
/// every gate green, which is why this is a test and not a review note.
///
/// Names are compared QUALIFIED. A bare leaf is not an identity — `k_matmul`
/// is a `kernels::ptir` template and an anonymous-namespace helper in `model`,
/// and those are two kernels that share a spelling, which is what a namespace
/// is for. Comparing leaves would report them and teach a reader to skip this.
#[test]
fn no_global_is_defined_twice() {
    let headers = device_headers();
    let mut seen: Vec<(String, PathBuf)> = Vec::new();
    let mut clashes: Vec<String> = Vec::new();

    for path in &headers {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?} reads: {e}"));
        let lines: Vec<&str> = text.lines().collect();
        let mut ns = String::new();
        for (at, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed.strip_prefix("namespace ")
                && let Some(named) = rest.split(&[' ', '{'][..]).next()
                && !named.is_empty()
            {
                ns = named.to_string();
            }
            let Some(after) = opens_a_global(trimmed) else {
                continue;
            };
            let leaf = kernel_leaf(after, &lines[at + 1..]).unwrap_or_else(|| {
                panic!(
                    "{}:{} opens a `__global__` and no name can be read out of \
                     it, so this definition would be counted nowhere. That is \
                     the shape of every gap this scanner has had: the clash \
                     assertion below is quantified over what was READ, so a \
                     definition it cannot parse is one it silently agrees is \
                     unique.",
                    path.display(),
                    at + 1
                )
            });
            let name = format!("{ns}::{leaf}");
            if let Some((_, first)) = seen.iter().find(|(n, _)| *n == name) {
                clashes.push(format!(
                    "{name}: {} and {}",
                    first.display(),
                    path.display()
                ));
            } else {
                seen.push((name, path.clone()));
            }
        }
    }

    // Anti-vacuity, both halves derived by the walk that just ran rather than
    // compared against a remembered number. The first fails if the tree stops
    // holding device headers; the second fails if it holds them and the
    // scanner stops recognising a definition — a changed `__global__` spelling
    // or a formatter putting the return type on its own line. Either one turns
    // the assertion below into a statement about nothing.
    //
    // Neither covers the PARTIAL version of the second, where the scanner
    // reads most definitions and silently drops a few; that is what
    // `kernel_leaf` returning `None` is made a panic for, and what its doc
    // records seven of.
    assert!(
        !headers.is_empty(),
        "{:?} holds no `.cuh` at all, so this test passes by looking at nothing.",
        kernels_dir()
    );
    assert!(
        !seen.is_empty(),
        "the walk read {} device headers and found no `__global__ void` in \
         any of them. The files are there and the scanner is not seeing \
         them, which is the quietest way for this test to stop testing.",
        headers.len()
    );

    assert!(
        clashes.is_empty(),
        "a `__global__` is defined in two places, so a migration was left \
         half-done and the two copies can drift. {} definitions under {} \
         distinct names across {} headers:\n  {}",
        seen.len() + clashes.len(),
        seen.len(),
        headers.len(),
        clashes.join("\n  ")
    );
}

/// What follows the `__global__` keyword on a line that opens a definition.
///
/// `CUBIN_EXPORT` is XQA's export macro and is the only thing in this tree
/// that precedes the keyword (`xqa/mha.cuh:1549`, `:2783`). Anchoring at the
/// start of the trimmed line is what keeps prose out: every `__global__` in a
/// comment sits behind `//`, `///` or a block comment's `*`.
fn opens_a_global(trimmed: &str) -> Option<&str> {
    let rest = trimmed
        .strip_prefix("CUBIN_EXPORT ")
        .unwrap_or(trimmed)
        .strip_prefix("__global__")?;
    // `__global__foo` is an identifier that starts with the keyword's letters
    // and is not the keyword.
    (!rest.starts_with(|c: char| c.is_alphanumeric() || c == '_')).then_some(rest)
}

/// The kernel's leaf name, reading on into `rest` when the declaration wraps.
///
/// # The four shapes this was measured against
///
/// A declaration is `__global__`, then `void` and an optional
/// `__launch_bounds__(…)` in EITHER order, then the name — and any of it may
/// be on a later line:
///
/// 1. `__global__ void name(` — 346 sites, the ordinary one.
/// 2. `__global__ void __launch_bounds__(512, 1)` with the name on the next
///    line — `comm/vllm_custom_all_reduce.cuh:214`, `:236`.
/// 3. `__global__ __launch_bounds__(…) void name(` — five sites, among them
///    `attention/prefill.cuh:2369` and `attn/attention_mla_naive.cuh:372`.
/// 4. `CUBIN_EXPORT __global__` / `#endif` / `void` / `name(` —
///    `xqa/mha.cuh:1549`, which is why directives are dropped below.
///
/// Shape 2 is why this exists: reading the leading identifier gave
/// `__launch_bounds__` for both vllm kernels and reported them as one name
/// defined twice. Shapes 3 and 4 were found while fixing it, and had been
/// skipped in silence — seven definitions this test believed it had checked.
///
/// Returns `None` when no name can be read, which the caller turns into a
/// failure. A `__global__` that cannot be parsed must not be passed over: an
/// unread definition is one the clash check agrees is unique.
fn kernel_leaf(after: &str, rest: &[&str]) -> Option<String> {
    // Preprocessor lines are dropped rather than parsed. They are line-based
    // and a declaration can be interrupted by one (shape 4), so a directive
    // in the middle is not part of the declaration in any sense that matters
    // here.
    let mut more = rest.iter().filter(|l| !l.trim_start().starts_with('#'));
    let mut acc = after.to_string();
    loop {
        match read_leaf(&acc) {
            Read::Name(leaf) => return Some(leaf),
            Read::Incomplete => {
                acc.push(' ');
                acc.push_str(more.next()?);
            }
        }
    }
}

/// The outcome of reading a name out of one accumulated declaration.
enum Read {
    /// The leaf name.
    Name(String),
    /// The text ran out before a name could be read, so the declaration wraps
    /// and the caller should join the next line.
    Incomplete,
}

/// Read past the qualifiers to the name, if all of both are present.
fn read_leaf(text: &str) -> Read {
    let mut text = text.trim_start();
    loop {
        if text.is_empty() {
            return Read::Incomplete;
        }
        // The return type, which sits before the attribute at shape 2's sites
        // and after it at shape 3's.
        if let Some(rest) = text.strip_prefix("void")
            && !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_')
        {
            text = rest.trim_start();
            continue;
        }
        if let Some(rest) = text.strip_prefix("__launch_bounds__") {
            let Some(rest) = skip_group(rest.trim_start()) else {
                return Read::Incomplete;
            };
            text = rest.trim_start();
            continue;
        }
        let leaf: String = text
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if leaf.is_empty() {
            return Read::Incomplete;
        }
        return Read::Name(leaf);
    }
}

/// Everything after the parenthesised group `text` opens with, or `None` if
/// it does not open with one or the group does not close inside `text`.
///
/// Counted rather than searched for the first `)`, because
/// `__launch_bounds__(KTraits::NUM_THREADS)` is flat but
/// `__launch_bounds__(kThreads, PIE_MLA_MMA_MINBLK)` is one macro expansion
/// away from not being.
fn skip_group(text: &str) -> Option<&str> {
    if !text.starts_with('(') {
        return None;
    }
    let mut depth = 0usize;
    for (at, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[at + 1..]);
                }
            }
            _ => {}
        }
    }
    None
}

/// This tree declares into `pie` and into nothing else, and CUDA is the
/// measurement that says why the root has to exist.
///
/// # The experiment this test is the residue of
///
/// The namespace was `pie_cuda_driver::kernels::<family>::device` and two of
/// those four segments were dead: the first named a C++ archive that had been
/// deleted, and `device` separated device text from host text in a tree that
/// has held nothing but device text since. Removing both was mechanical. The
/// question was whether the ROOT had to survive it at all, or whether a family
/// could open at global scope and let a launch name the same string a trace
/// does -- `norm::scalar_mul`, spelled once for all four planes.
///
/// It cannot, and the answer took one compile to arrive. NVRTC pre-includes
/// CUDA's math API, which declares
///
///     __device__ double norm(int dim, double const* p);
///
/// at global scope (`crt/math_functions.h:2335` in 13.0). `namespace norm {`
/// is then a redeclaration of an existing name, and every root under `norm/`
/// -- six of the 126 -- failed to compile with *"norm has already been
/// declared in the current scope"*. The other 120 were fine, which is the
/// trap: the collision is invisible until the one family that happens to
/// share a spelling with a builtin gets compiled, and `norm` is not a name
/// this crate may rename -- eight crates lower traces to `norm::…`.
///
/// So the global namespace is CUDA's, and this crate's device text is a guest
/// in a translation unit that also holds 22k lines of internalised upstream.
/// `pie` is the whole of what this crate opens there.
///
/// # What is allowed through, and why each one has to be
///
/// A declaration at global scope is allowed only if it CANNOT be anywhere
/// else. Two shapes qualify, and the walk below recognises exactly them:
///
/// * `extern "C"` — a symbol the host or the driver resolves by name.
///   `pie_xqa_smem_size` is read out of the module by the launch that sizes
///   XQA's shared memory, and `cudaGraphSetConditional` is the driver's own,
///   declared here because NVRTC carries no `<cuda_device_runtime_api.h>`.
///   A namespace would mangle the first and rename the second.
/// * a name this tree impersonates on NVIDIA's behalf — `__nv_bfloat16` in
///   `pie_mma.cuh`, which exists so upstream text compiles against the
///   prelude's `bf16`. The whole point of the name is that it is NVIDIA's, so
///   it lives where NVIDIA's would.
///
/// The internalised trees are not held to any of this: they were written for
/// a compiler with real CUDA headers and they declare what they declare. That
/// is the boundary this crate keeps against them, and it is the reason the
/// root is worth its five characters.
#[test]
fn the_tree_declares_into_pie_and_nothing_else() {
    let ours: Vec<PathBuf> = device_headers()
        .into_iter()
        .filter(|p| !internalised(p))
        .collect();
    assert!(
        !ours.is_empty(),
        "the walk found no device header this crate owns, so every assertion \
         below is a statement about nothing."
    );

    let mut foreign_namespaces: Vec<String> = Vec::new();
    let mut loose_declarations: Vec<String> = Vec::new();
    let mut roots = 0usize;

    for path in &ours {
        let text = without_comments(
            &std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?} reads: {e}")),
        );
        let mut depth = 0i32;
        let mut statement = String::new();

        for (at, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            let opens = trimmed.chars().filter(|c| *c == '{').count() as i32;
            let closes = trimmed.chars().filter(|c| *c == '}').count() as i32;

            if depth == 0 && !trimmed.is_empty() && !trimmed.starts_with('#') {
                if let Some(rest) = trimmed.strip_prefix("namespace ")
                    && !rest.contains('=')
                {
                    let named = rest.split(&[' ', '{'][..]).next().unwrap_or("");
                    if named == "pie" || named.starts_with("pie::") {
                        roots += 1;
                    } else if !IMPERSONATED_NAMESPACES.contains(&named) {
                        foreign_namespaces.push(format!(
                            "{}:{}: {trimmed}",
                            path.display(),
                            at + 1
                        ));
                    }
                } else if !trimmed.starts_with('}') && !trimmed.starts_with('{') {
                    statement.push_str(trimmed);
                    statement.push(' ');
                }
            }

            if statement.contains(';') {
                let whole = statement.trim().to_string();
                if !crosses_by_c_linkage(&whole) && !impersonates_nvidia(&whole) {
                    loose_declarations.push(format!("{}:{}: {whole}", path.display(), at + 1));
                }
                statement.clear();
            }
            depth += opens - closes;
        }
    }

    assert!(
        roots > 0,
        "no header opened `namespace pie`, so the walk is reading something \
         other than this tree's device text."
    );
    assert!(
        foreign_namespaces.is_empty(),
        "a namespace other than `pie` opens at global scope. CUDA owns that \
         scope -- `norm` is already a function there -- so a family opened in \
         it compiles until the day its name collides with a builtin, and then \
         only for the roots that include it:\n  {}",
        foreign_namespaces.join("\n  ")
    );
    assert!(
        loose_declarations.is_empty(),
        "a declaration reaches global scope without being one of the two \
         things that have to: an `extern \"C\"` symbol the host or the driver \
         resolves by name, or a name this tree impersonates for NVIDIA. Put \
         it in `pie`:\n  {}",
        loose_declarations.join("\n  ")
    );
}

/// The internalised FlashInfer and XQA trees, by the directory that names
/// their provenance. They are somebody else's text and they keep their own
/// global scope; the rule above is this crate's about this crate's.
fn internalised(path: &Path) -> bool {
    path.components()
        .any(|c| c.as_os_str() == "flashinfer" || c.as_os_str() == "xqa")
}

/// The namespaces this tree opens on NVIDIA's behalf rather than its own.
///
/// One, and it is the namespace half of the same impersonation `shim/`
/// performs with filenames: `pie_mma.cuh` defines `nvcuda::wmma::fragment`
/// and the `load_matrix_sync`/`mma_sync` family over the prelude's `bf16`,
/// because upstream text includes `<mma.h>` and NVRTC carries no such header.
/// A name that exists to BE NVIDIA's cannot be spelled anywhere but where
/// NVIDIA's is, which is why this list is a list and not a violation.
const IMPERSONATED_NAMESPACES: &[&str] = &["nvcuda"];

/// `text` with comments and literals blanked, so a `{` in prose does not move
/// the brace depth and the word `namespace` in a sentence is not an opener.
fn without_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '/' if chars.peek() == Some(&'/') => {
                for c in chars.by_ref() {
                    if c == '\n' {
                        out.push('\n');
                        break;
                    }
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                let mut last = ' ';
                for c in chars.by_ref() {
                    if c == '\n' {
                        out.push('\n');
                    }
                    if last == '*' && c == '/' {
                        break;
                    }
                    last = c;
                }
            }
            '"' | '\'' => {
                // Literals are kept, minus any braces inside them: `extern
                // "C"` is a linkage spelling the walk above reads, and a `{`
                // in a `printf` format would move the brace depth.
                let quote = c;
                out.push(quote);
                let mut escaped = false;
                for c in chars.by_ref() {
                    if escaped {
                        escaped = false;
                    } else if c == '\\' {
                        escaped = true;
                    } else if c == quote {
                        break;
                    }
                    out.push(if c == '{' || c == '}' { ' ' } else { c });
                }
                out.push(quote);
            }
            _ => out.push(c),
        }
    }
    out
}

/// Whether a global-scope declaration is one the host or the driver resolves
/// by name, which is the one linkage a namespace would break.
fn crosses_by_c_linkage(statement: &str) -> bool {
    statement.starts_with("extern \"C\"")
}

/// Whether a global-scope declaration is a name this tree wears on NVIDIA's
/// behalf, which is a name that has to be spelled where NVIDIA's is.
fn impersonates_nvidia(statement: &str) -> bool {
    statement
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .any(|word| {
            word.starts_with("__nv_") || word.starts_with("__half") || word == "__nv_bfloat16"
        })
}
