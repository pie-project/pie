//! Reading a `.metal` source with its local `#include`s spliced in.
//!
//! Metal's runtime shader compiler resolves no filesystem includes of its
//! own. `newLibraryWithSource:` is handed one string and that string is the
//! whole translation unit, so a `#include "rms_params.h"` in a kernel is not
//! a lookup the compiler will perform -- it is text this module has to put
//! there. That is the only way two `.metal` files can share a definition
//! instead of restating it, and restating it is what the 4-bit codecs did
//! before this existed.
//!
//! Angle-bracket includes are left alone: those are Metal's own headers
//! (`<metal_stdlib>`), which the compiler does resolve.
//!
//! # Why this module is portable
//!
//! Nothing here touches Metal, so nothing here needs a GPU or an Apple host.
//! It is string manipulation over files, which means it is the part of the
//! shell most worth testing and the part a GPU-gated test suite would never
//! run. The loader is injected ([`splice_with`]) so the tests do not need a
//! filesystem either.
//!
//! # And why the batch is here too
//!
//! [`Batch`] reads a whole load's worth of sources and computes the key its
//! compiled pipelines are cached under. Both halves are string work over
//! files, and the key in particular has to be decided by the resolved text --
//! which is a fact about include splicing, not about Metal. Keeping it beside
//! the splicer is what lets a test prove that editing a header invalidates
//! the cache without a GPU in the room.

use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// How deep a `#include` chain may nest before the shell calls it a cycle.
///
/// A bound is required rather than merely prudent: this module resolves
/// includes itself, so a header that includes itself is an infinite loop
/// here, where in a real preprocessor it would be a diagnostic.
pub const MAX_INCLUDE_DEPTH: usize = 8;

/// The directive this module acts on. Quoted form only.
const DIRECTIVE: &str = "#include \"";

/// Read `path` and splice its local `#include`s in, recursively.
pub fn read_source(path: impl AsRef<Path>) -> Result<String> {
    splice_with(path, |p| std::fs::read_to_string(p))
}

/// [`read_source`] against a caller-supplied loader.
///
/// The loader receives the path as resolved -- relative to the directory of
/// the file doing the including, which is what a C preprocessor does with the
/// quoted form and therefore what the kernel sources are written against.
pub fn splice_with<L>(path: impl AsRef<Path>, mut load: L) -> Result<String>
where
    L: FnMut(&Path) -> io::Result<String>,
{
    let path = path.as_ref();
    let mut seen = HashSet::new();
    // The root counts as already spliced. Without this a header that includes
    // the file including it would be spliced a second time -- the one cycle
    // the dedup below would not catch, because it only ever records what it
    // descends INTO.
    seen.insert(dedup_key(path));
    splice_at(path, 0, &mut seen, &mut load)
}

/// The identity a file is deduplicated by.
///
/// Canonicalized when the filesystem will say, so that `kernels/x.h` and
/// `kernels/../kernels/x.h` are one file rather than two. When it will not --
/// an injected loader in a test, a path that does not exist -- the path as
/// written is used, which is the best available answer and is stable within
/// one splice.
fn dedup_key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn splice_at<L>(
    path: &Path,
    depth: usize,
    seen: &mut HashSet<PathBuf>,
    load: &mut L,
) -> Result<String>
where
    L: FnMut(&Path) -> io::Result<String>,
{
    if depth > MAX_INCLUDE_DEPTH {
        return Err(Error::IncludeTooDeep {
            path: path.to_path_buf(),
            limit: MAX_INCLUDE_DEPTH,
        });
    }

    let mut source = load(path).map_err(|source| Error::ShaderRead {
        path: path.to_path_buf(),
        source,
    })?;

    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();

    // Rebuilt rather than spliced in place. The C++ this replaces mutates the
    // string under the cursor it is searching with, which works but ties the
    // scan position to the length of every replacement; building the output
    // forward means the scan only ever moves forward over the ORIGINAL text,
    // so a header that itself contains the characters `#include "` cannot
    // make the outer scan revisit it.
    let mut out = String::with_capacity(source.len());
    let mut cursor = 0usize;

    while let Some(rel) = source[cursor..].find(DIRECTIVE) {
        let at = cursor + rel;

        // Column zero only, matching the C++ shell: a directive is a directive
        // only at the start of a line, so the same characters inside a string
        // literal or a comment are left alone. Note this also declines an
        // INDENTED `#include`, which a real preprocessor would honour. That is
        // deliberate and matches the shipped behaviour -- the kernel sources
        // are all written flush-left, and widening it here would change which
        // text a shipped shader compiles from.
        if at != 0 && source.as_bytes()[at - 1] != b'\n' {
            out.push_str(&source[cursor..at + DIRECTIVE.len()]);
            cursor = at + DIRECTIVE.len();
            continue;
        }

        let name_at = at + DIRECTIVE.len();
        let Some(rel_close) = source[name_at..].find('"') else {
            return Err(Error::UnterminatedInclude {
                path: path.to_path_buf(),
                offset: at,
            });
        };
        let close = name_at + rel_close;

        out.push_str(&source[cursor..at]);

        let included_path = dir.join(&source[name_at..close]);

        // Each distinct file is spliced at most once per translation unit.
        //
        // Splicing turns an include into text, and text carries no `#pragma
        // once` -- the pragma is a property of a FILE the compiler opened, and
        // after splicing there is no file. So a diamond (two headers that both
        // include a third) would emit the third one's definitions twice and
        // fail to compile, with an error naming a redefinition at a line
        // number in a source no one wrote. No kernel has a diamond today, so
        // this changes nothing now; it is here because the first one to add a
        // shared header would otherwise find out this way.
        let key = dedup_key(&included_path);
        if seen.insert(key) {
            out.push_str(&splice_at(&included_path, depth + 1, seen, load)?);
        }

        cursor = close + 1;
    }

    out.push_str(&source[cursor..]);
    source = out;
    Ok(source)
}

/// One entry point wanted out of one source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// The `.metal` file to compile.
    pub path: PathBuf,
    /// The kernel function to build a pipeline for.
    pub function: String,
}

impl Request {
    /// A request for `function` out of `path`.
    pub fn new(path: impl Into<PathBuf>, function: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            function: function.into(),
        }
    }
}

/// A batch of requests with each distinct source read exactly once.
///
/// A load asks for tens of entry points out of a handful of files, and a file
/// is a translation unit: two entry points in the same file are two pipelines
/// off one library. Reading and splicing per request would do that work once
/// per entry point instead of once per file.
///
/// The batch is also what the archive is keyed on, and that is the reason the
/// sources are kept rather than discarded after the read. The C++ reads every
/// file twice for this -- once to compute the key and once to compile -- and
/// the two reads are not guaranteed to see the same bytes.
#[derive(Debug)]
pub struct Batch {
    /// Distinct paths, in first-seen order.
    paths: Vec<PathBuf>,
    /// The resolved source of each path, or why it could not be read.
    sources: Vec<Result<String>>,
    /// For each request, its index into `paths`.
    library: Vec<usize>,
    /// The requested entry point of each request.
    functions: Vec<String>,
}

impl Batch {
    /// Read every distinct source in `requests`.
    ///
    /// Never fails as a whole. A file that cannot be read fails the requests
    /// that name it and no others, because one missing kernel is not a reason
    /// to refuse the twenty that are present.
    pub fn load(requests: &[Request]) -> Self {
        Self::load_with(requests, |path| read_source(path))
    }

    /// [`Batch::load`] against a caller-supplied reader, for tests.
    pub fn load_with<L>(requests: &[Request], mut read: L) -> Self
    where
        L: FnMut(&Path) -> Result<String>,
    {
        let mut paths: Vec<PathBuf> = Vec::new();
        let mut sources: Vec<Result<String>> = Vec::new();
        let mut library = Vec::with_capacity(requests.len());
        let mut functions = Vec::with_capacity(requests.len());
        for request in requests {
            let index = paths.iter().position(|p| *p == request.path);
            let index = match index {
                Some(index) => index,
                None => {
                    paths.push(request.path.clone());
                    sources.push(read(&request.path));
                    paths.len() - 1
                }
            };
            library.push(index);
            functions.push(request.function.clone());
        }
        Self {
            paths,
            sources,
            library,
            functions,
        }
    }

    /// How many requests this batch was built from.
    #[must_use]
    pub fn len(&self) -> usize {
        self.library.len()
    }

    /// Whether the batch has no requests.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.library.is_empty()
    }

    /// The distinct source files, in first-seen order.
    #[must_use]
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// The resolved source for library `index`.
    #[must_use]
    pub fn source(&self, index: usize) -> Option<&Result<String>> {
        self.sources.get(index)
    }

    /// The library index and entry point of request `index`.
    #[must_use]
    pub fn request(&self, index: usize) -> Option<(usize, &str)> {
        Some((
            *self.library.get(index)?,
            self.functions.get(index)?.as_str(),
        ))
    }

    /// A key naming exactly this batch, for use as an archive filename.
    ///
    /// Every input that can change the compiled binaries goes in: which entry
    /// points were asked for out of which files, in order, and the RESOLVED
    /// text of each of those files. Resolved and not the file's size and
    /// mtime, because a source that includes another would otherwise keep its
    /// key when the included file changed and be served a pipeline built from
    /// the old definition -- which is worse than a slow start, since it looks
    /// like it worked.
    ///
    /// `salt` is for what the caller knows and this module does not: the GPU
    /// the binaries are for and the language dialect they were compiled as.
    /// A cache shared between two of either would otherwise collide.
    ///
    /// A file that could not be read hashes as its own distinct marker. If it
    /// hashed as nothing, a batch with a missing file and a batch with an
    /// empty one would be the same batch.
    #[must_use]
    pub fn key(&self, salt: u64) -> u64 {
        const OFFSET: u64 = 14_695_981_039_346_656_037;
        let mut hash = OFFSET;
        let mut eat = |bytes: &[u8]| {
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(1_099_511_628_211);
            }
        };
        eat(&salt.to_le_bytes());
        for (library, function) in self.library.iter().zip(&self.functions) {
            eat(&library.to_le_bytes());
            eat(function.as_bytes());
            // Lengths, so that ("ab","c") and ("a","bc") are not one string.
            eat(&function.len().to_le_bytes());
        }
        for source in &self.sources {
            match source {
                Ok(text) => {
                    eat(&[1]);
                    eat(text.as_bytes());
                    eat(&text.len().to_le_bytes());
                }
                Err(_) => eat(&[0]),
            }
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A loader over an in-memory file table, so these tests describe include
    /// resolution rather than a temporary directory.
    fn table(files: &[(&str, &str)]) -> impl FnMut(&Path) -> io::Result<String> + use<> {
        let map: HashMap<PathBuf, String> = files
            .iter()
            .map(|(k, v)| (PathBuf::from(*k), (*v).to_string()))
            .collect();
        move |p: &Path| {
            map.get(p)
                .cloned()
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, p.display().to_string()))
        }
    }

    #[test]
    fn splices_a_quoted_include() {
        let out = splice_with(
            "k/a.metal",
            table(&[
                ("k/a.metal", "pre\n#include \"b.h\"\npost\n"),
                ("k/b.h", "BODY"),
            ]),
        )
        .expect("splices");
        assert_eq!(out, "pre\nBODY\npost\n");
    }

    #[test]
    fn resolves_relative_to_the_including_file() {
        let out = splice_with(
            "k/sub/a.metal",
            table(&[("k/sub/a.metal", "#include \"b.h\"\n"), ("k/sub/b.h", "OK")]),
        )
        .expect("splices");
        assert_eq!(out, "OK\n");
    }

    #[test]
    fn leaves_angle_bracket_includes_alone() {
        let src = "#include <metal_stdlib>\nbody\n";
        let out = splice_with("k/a.metal", table(&[("k/a.metal", src)])).expect("splices");
        assert_eq!(out, src);
    }

    #[test]
    fn ignores_a_directive_that_is_not_at_column_zero() {
        // Both of these are text, not directives: one is inside a string
        // literal, the other is indented.
        let src = "const char* s = \"#include \\\"x.h\\\"\";\n  #include \"y.h\"\n";
        let out = splice_with("k/a.metal", table(&[("k/a.metal", src)])).expect("splices");
        assert_eq!(out, src, "no include should have been resolved");
    }

    #[test]
    fn splices_nested_includes() {
        let out = splice_with(
            "k/a.metal",
            table(&[
                ("k/a.metal", "#include \"b.h\"\n"),
                ("k/b.h", "B<#include \"c.h\"\n>B"),
                ("k/c.h", "C"),
            ]),
        )
        .expect("splices");
        // `#include "c.h"` inside b.h is not at column zero, so it stays text.
        assert_eq!(out, "B<#include \"c.h\"\n>B\n");
    }

    #[test]
    fn splices_a_shared_header_once() {
        let out = splice_with(
            "k/a.metal",
            table(&[
                ("k/a.metal", "#include \"x.h\"\n#include \"y.h\"\n"),
                ("k/x.h", "#include \"common.h\"\nX"),
                ("k/y.h", "#include \"common.h\"\nY"),
                ("k/common.h", "COMMON"),
            ]),
        )
        .expect("splices");
        assert_eq!(
            out.matches("COMMON").count(),
            1,
            "a diamond must not emit the shared header twice: {out}"
        );
    }

    /// Dedup gives quoted includes the same semantics an include guard does,
    /// so a self-include resolves to nothing rather than recursing. The claim
    /// under test is that it TERMINATES; that it also produces the right text
    /// is the assertion after it.
    #[test]
    fn a_self_include_resolves_to_nothing() {
        let out = splice_with(
            "k/a.metal",
            table(&[("k/a.metal", "before\n#include \"a.metal\"\nafter\n")]),
        )
        .expect("a self-include is a no-op, not an error");
        assert_eq!(out, "before\n\nafter\n");
    }

    #[test]
    fn a_mutual_cycle_terminates() {
        let out = splice_with(
            "k/a.metal",
            table(&[
                ("k/a.metal", "A\n#include \"b.h\"\n"),
                ("k/b.h", "B\n#include \"a.metal\"\n"),
            ]),
        )
        .expect("the cycle is cut by dedup");
        // "A\n" + b.h's "B\n" + b.h's post-directive "\n" + a.metal's own "\n".
        assert_eq!(out, "A\nB\n\n\n");
    }

    /// The root is spliced once even when a header reaches back for it -- the
    /// one cycle dedup would miss if it only recorded what it descends into.
    #[test]
    fn a_header_cannot_re_splice_the_root() {
        let out = splice_with(
            "k/a.metal",
            table(&[
                ("k/a.metal", "#include \"b.h\"\nROOT\n"),
                ("k/b.h", "#include \"a.metal\"\n"),
            ]),
        )
        .expect("splices");
        assert_eq!(out.matches("ROOT").count(), 1, "root spliced twice: {out}");
    }

    /// Dedup makes unbounded recursion impossible, so the depth bound only
    /// ever fires on a chain of DISTINCT files. It is kept at the C++ shell's
    /// limit rather than removed: a chain that deep is a shader nobody meant
    /// to write, and the bound is what turns it into a message.
    #[test]
    fn refuses_a_chain_deeper_than_the_limit() {
        let mut files: Vec<(String, String)> =
            vec![("k/a.metal".to_string(), "#include \"h0.h\"\n".to_string())];
        // h0 -> h1 -> ... -> h9: ten levels below the root, past the bound.
        for i in 0..10 {
            files.push((format!("k/h{i}.h"), format!("#include \"h{}.h\"\n", i + 1)));
        }
        files.push(("k/h10.h".to_string(), "END".to_string()));
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let err = splice_with("k/a.metal", table(&refs)).expect_err("too deep");
        assert!(matches!(err, Error::IncludeTooDeep { .. }), "{err}");
    }

    #[test]
    fn accepts_a_chain_at_the_limit() {
        let mut files: Vec<(String, String)> =
            vec![("k/a.metal".to_string(), "#include \"h0.h\"\n".to_string())];
        // h0..h6 is depth 1..7, and h7 is depth 8 -- exactly the bound.
        for i in 0..7 {
            files.push((format!("k/h{i}.h"), format!("#include \"h{}.h\"\n", i + 1)));
        }
        files.push(("k/h7.h".to_string(), "END".to_string()));
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let out = splice_with("k/a.metal", table(&refs)).expect("exactly at the bound is fine");
        assert!(out.contains("END"), "{out}");
    }

    #[test]
    fn reports_an_unterminated_include() {
        let err = splice_with("k/a.metal", table(&[("k/a.metal", "#include \"b.h\n")]))
            .expect_err("no closing quote");
        assert!(matches!(err, Error::UnterminatedInclude { .. }), "{err}");
    }

    #[test]
    fn reports_a_missing_header_with_its_path() {
        let err = splice_with(
            "k/a.metal",
            table(&[("k/a.metal", "#include \"gone.h\"\n")]),
        )
        .expect_err("header does not exist");
        match err {
            Error::ShaderRead { path, .. } => assert_eq!(path, PathBuf::from("k/gone.h")),
            other => panic!("expected ShaderRead, got {other}"),
        }
    }

    #[test]
    fn reports_a_missing_root_with_its_path() {
        let err = splice_with("k/nope.metal", table(&[])).expect_err("root does not exist");
        assert!(matches!(err, Error::ShaderRead { .. }), "{err}");
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use std::collections::HashMap;

    /// A reader over an in-memory file table.
    fn files(entries: &[(&str, &str)]) -> impl FnMut(&Path) -> Result<String> + use<> {
        let map: HashMap<PathBuf, String> = entries
            .iter()
            .map(|(k, v)| (PathBuf::from(*k), (*v).to_string()))
            .collect();
        move |path: &Path| {
            map.get(path)
                .cloned()
                .ok_or_else(|| Error::ShaderRead {
                    path: path.to_path_buf(),
                    source: io::Error::new(io::ErrorKind::NotFound, "absent"),
                })
        }
    }

    /// The same table, as the `io::Result` loader `splice_with` wants.
    fn raw(entries: &[(&str, &str)]) -> impl FnMut(&Path) -> io::Result<String> + use<> {
        let map: HashMap<PathBuf, String> = entries
            .iter()
            .map(|(k, v)| (PathBuf::from(*k), (*v).to_string()))
            .collect();
        move |path: &Path| {
            map.get(path)
                .cloned()
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, path.display().to_string()))
        }
    }

    fn requests(pairs: &[(&str, &str)]) -> Vec<Request> {
        pairs.iter().map(|(p, f)| Request::new(*p, *f)).collect()
    }

    #[test]
    fn a_file_named_by_several_requests_is_read_once() {
        let mut reads = 0;
        let mut read = files(&[("a.metal", "A"), ("b.metal", "B")]);
        let batch = Batch::load_with(
            &requests(&[
                ("a.metal", "one"),
                ("b.metal", "two"),
                ("a.metal", "three"),
            ]),
            |path| {
                reads += 1;
                read(path)
            },
        );
        assert_eq!(reads, 2, "three requests out of two files is two reads");
        assert_eq!(batch.paths().len(), 2);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.request(0), Some((0, "one")));
        assert_eq!(batch.request(1), Some((1, "two")));
        assert_eq!(
            batch.request(2),
            Some((0, "three")),
            "the third request shares the first request's library"
        );
    }

    #[test]
    fn one_unreadable_file_fails_only_its_own_requests() {
        let batch = Batch::load_with(
            &requests(&[("here.metal", "k"), ("gone.metal", "k")]),
            files(&[("here.metal", "A")]),
        );
        assert!(matches!(batch.source(0), Some(Ok(text)) if text == "A"));
        assert!(matches!(
            batch.source(1),
            Some(Err(Error::ShaderRead { .. }))
        ));
    }

    #[test]
    fn editing_a_source_changes_the_key() {
        let asked = requests(&[("a.metal", "k")]);
        let before = Batch::load_with(&asked, files(&[("a.metal", "kernel void k() {}")]));
        let after = Batch::load_with(&asked, files(&[("a.metal", "kernel void k() { ; }")]));
        assert_ne!(before.key(0), after.key(0));
    }

    #[test]
    fn editing_an_included_file_changes_the_key() {
        // The whole reason the key is over RESOLVED text: `a.metal` itself is
        // byte for byte identical in both, and only the header moved.
        let asked = requests(&[("a.metal", "k")]);
        let root = "#include \"h.h\"\nkernel void k() {}";
        let before = Batch::load_with(&asked, |path| {
            splice_with(path, raw(&[("a.metal", root), ("h.h", "#define N 1")]))
        });
        let after = Batch::load_with(&asked, |path| {
            splice_with(path, raw(&[("a.metal", root), ("h.h", "#define N 2")]))
        });
        assert_ne!(
            before.key(0),
            after.key(0),
            "an archive keyed to the old header would serve a stale pipeline"
        );
    }

    #[test]
    fn asking_for_a_different_entry_point_changes_the_key() {
        let one = Batch::load_with(&requests(&[("a.metal", "one")]), files(&[("a.metal", "A")]));
        let two = Batch::load_with(&requests(&[("a.metal", "two")]), files(&[("a.metal", "A")]));
        assert_ne!(one.key(0), two.key(0));
    }

    #[test]
    fn the_order_of_the_requests_is_part_of_the_key() {
        // The results are positional, so a batch asked in a different order
        // is a different batch even though it builds the same pipelines.
        let entries = [("a.metal", "A"), ("b.metal", "B")];
        let forward = Batch::load_with(
            &requests(&[("a.metal", "x"), ("b.metal", "y")]),
            files(&entries),
        );
        let backward = Batch::load_with(
            &requests(&[("b.metal", "y"), ("a.metal", "x")]),
            files(&entries),
        );
        assert_ne!(forward.key(0), backward.key(0));
    }

    #[test]
    fn the_salt_is_part_of_the_key() {
        let batch = Batch::load_with(&requests(&[("a.metal", "k")]), files(&[("a.metal", "A")]));
        assert_ne!(
            batch.key(1),
            batch.key(2),
            "two GPUs, or two dialects, must not share an archive"
        );
    }

    #[test]
    fn a_missing_file_is_not_an_empty_one() {
        let asked = requests(&[("a.metal", "k")]);
        let missing = Batch::load_with(&asked, files(&[]));
        let empty = Batch::load_with(&asked, files(&[("a.metal", "")]));
        assert_ne!(missing.key(0), empty.key(0));
    }

    #[test]
    fn an_empty_batch_is_empty() {
        let batch = Batch::load_with(&[], files(&[]));
        assert!(batch.is_empty());
        assert_eq!(batch.request(0), None);
    }
}
