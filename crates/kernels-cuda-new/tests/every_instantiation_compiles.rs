//! Every template-id a routine body names is one NVRTC can lower.
//!
//! The refactor deleted the row lists, so nothing enumerates the crate's
//! instantiations any more and nothing checked them. A body names its
//! template-id as a string; a typo in one is an NVRTC error at the first fire
//! of that kernel on a GPU, which is the worst place to find it. Some 750 of
//! these strings were transcribed by hand during the port, and until this
//! fixture ran, not one of them had been through a compiler.
//!
//! The crate declares its roots two ways, so this reads them two ways:
//!
//! * **Written** — `Root::new("norm/rmsnorm", include_str!(..), ..)` with a
//!   `mod inst` beside it. Fifty-nine of these. They are read out of the
//!   SOURCE rather than kept in a list here, because a list here is one more
//!   thing to forget to update; a new constant is covered the moment it is
//!   written. The cost is a parser, and a parser that quietly stopped matching
//!   would turn this into a test that passes by finding nothing — so every
//!   `Root::new` occurrence is reconciled against what was recovered, and a
//!   declaration the parser cannot read fails rather than skips.
//!
//! * **Computed** — the two lattices, where a `const fn` or a macro stamps a
//!   root per member out of one text under a different `-D` set. Fifty-six
//!   FA2 points and five XQA ones, with no literal to read. Those come from
//!   the crate's own public statics, which is better than parsing: `DECODE`,
//!   `PREFILL` and `ROOTS` carry both the root and its arms, so the fixture
//!   asks the crate what it built rather than guessing.
//!
//! Needs `libnvrtc`, not a device: `nvrtcCompileProgram` targets an
//! architecture, it does not talk to one. Skips with a message when the
//! library will not load, so a box without CUDA still runs the suite.

#![cfg(feature = "_cuda")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kernels_cuda_new::jit::{Headers, Root, Toolchain};
use kernels_cuda_new::runtime::nvrtc;
use kernels_cuda_new::source::Header;

// ===========================================================================
// One compile, however its root was declared
// ===========================================================================

/// One root and the template-ids to ask it for.
struct Job {
    /// Where it came from, for a message that can be acted on.
    site: String,
    /// The root's name, as a diagnostic will spell it.
    name: String,
    /// The device text.
    text: String,
    /// NVRTC options this root needs.
    options: Vec<String>,
    /// The header set its `#include`s resolve against.
    headers: &'static [Header],
    /// The template-ids handed to `nvrtcAddNameExpression`.
    wanted: Vec<String>,
}

impl Job {
    /// The same job, as the crate's compiler takes it.
    fn from_root(site: String, root: &Root, wanted: Vec<String>) -> Self {
        Self {
            site,
            name: root.name.to_owned(),
            text: root.text.to_owned(),
            options: root.options.iter().map(|&o| o.to_owned()).collect(),
            headers: root.header_set(),
            wanted,
        }
    }

    /// Ask NVRTC for every one of them.
    fn compile(&self, arch: &str) -> Result<(), String> {
        let options: Vec<&str> = self.options.iter().map(String::as_str).collect();
        let job = nvrtc::Job {
            name: Box::leak(self.name.clone().into_boxed_str()),
            source: self.text.clone(),
            arch,
            options: &options,
            headers: self.headers,
            // Not the root's own `.since`: this asks whether the toolchain
            // that IS here can lower the symbol, and a floor would answer a
            // different question by refusing before the compile.
            floor: Toolchain::ANY,
            wanted: &self.wanted,
            device_link: options.iter().any(|o| o.contains("relocatable-device-code")),
        };
        nvrtc::compile_text(&job).map(|_| ()).map_err(|why| why.to_string())
    }
}

// ===========================================================================
// The written roots: read out of `src/x`
// ===========================================================================

/// A Rust string literal's value: continuations joined, escapes resolved.
///
/// `"a\` + newline + spaces + `b"` is `"ab"` — the backslash eats the newline
/// AND the indentation after it, which is how every long template-id in this
/// crate is written.
fn unescape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            Some('\n') => {
                chars.next();
                while chars.next_if(|c| c.is_whitespace()).is_some() {}
            }
            Some('"' | '\\') => out.push(chars.next().expect("peeked")),
            _ => out.push(c),
        }
    }
    out
}

/// Every `"..."` literal in `text`, unescaped, in source order.
///
/// Comments are skipped rather than scanned. They are prose about kernels, and
/// prose about kernels quotes code: the doc comment over `xqa`'s `mod inst`
/// explains that `kernel_mha` is `extern "C"`, and a scanner that took every
/// pair of quotes handed NVRTC `C` as a name expression.
fn literals(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                i = text[i..].find('\n').map_or(bytes.len(), |e| i + e + 1);
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                i = text[i + 2..].find("*/").map_or(bytes.len(), |e| i + 2 + e + 2);
            }
            b'"' => {
                let start = i + 1;
                let mut j = start;
                while j < bytes.len() && bytes[j] != b'"' {
                    j += if bytes[j] == b'\\' { 2 } else { 1 };
                }
                if j >= bytes.len() {
                    break;
                }
                out.push(unescape(&text[start..j]));
                i = j + 1;
            }
            _ => i += 1,
        }
    }
    out
}

/// The contents of the balanced group opening at the first `open` in `text`,
/// and the byte offset just past its close.
fn group(text: &str, open: char, close: char) -> Option<(&str, usize)> {
    let at = text.find(open)?;
    let mut depth = 0usize;
    for (i, c) in text[at..].char_indices() {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some((&text[at + 1..at + i], at + i + 1));
            }
        }
    }
    None
}

/// One written root, as the source declares it.
struct Written {
    site: String,
    name: String,
    /// The file `include_str!` names, resolved against the declaring file.
    text: PathBuf,
    options: Vec<String>,
    upstream: bool,
    wanted: Vec<String>,
}

/// What one file yielded.
#[derive(Default)]
struct Read {
    /// The roots whose arguments could be reconstructed.
    roots: Vec<Written>,
    /// Sites of `Root::new` calls whose arguments are computed, not written.
    unread: Vec<String>,
    /// Instantiations from a file that declares no root of its own. A family
    /// may put its host program in a submodule file (`gemm/gemv.rs`) while the
    /// root stays with the family (`gemm.rs`); these are attached to the
    /// parent module's root once every file has been read.
    loose: Vec<String>,
}

/// Every template-id the first `mod inst` in `span` names.
fn instantiations(span: &str) -> Vec<String> {
    span.find("mod inst")
        .and_then(|i| group(&span[i..], '{', '}').map(|(body, _)| literals(body)))
        .unwrap_or_default()
}

/// Pull every root out of one file, each with the `mod inst` beside it.
fn scrape(file: &Path) -> Read {
    let text = std::fs::read_to_string(file).expect("a readable source file");
    let dir = file.parent().expect("a file has a parent");
    let show = file.file_name().expect("a named file").to_string_lossy().into_owned();
    let line_of = |at: usize| text[..at].bytes().filter(|&b| b == b'\n').count() + 1;

    let starts: Vec<usize> = text.match_indices("Root::new").map(|(at, _)| at).collect();
    if starts.is_empty() {
        return Read { loose: instantiations(&text), ..Read::default() };
    }

    // Read every declaration before pairing any with a `mod inst`, because a
    // root's instantiations are the ones between it and the next READABLE
    // root: a computed root nested in a `const fn` sits between `xqa`'s
    // metadata root and that root's own module, and stopping there loses it.
    let mut read: Vec<(usize, usize, Option<Written>)> = Vec::new();
    for &start in &starts {
        let (args, past) = group(&text[start..], '(', ')').expect("`Root::new` is a call");
        let after = start + past;

        // The two arguments that must be literal for a compile to be
        // reconstructible: the root's name, and the path `include_str!` reads.
        let name = literals(args).into_iter().next();
        let include = args.find("include_str!(").map(|i| {
            literals(&args[i..]).into_iter().next().expect("`include_str!` takes a literal path")
        });
        let (Some(name), Some(include)) = (name, include) else {
            read.push((start, after, None));
            continue;
        };

        // The chain runs to the `;` closing the static. `;\n` and not `;`, so
        // that a `;` inside a chained argument cannot end it early.
        let end = text[after..].find(";\n").map_or(text.len(), |e| after + e);
        let chain = &text[after..end];
        let options = chain.find(".options(").map_or_else(Vec::new, |i| {
            let (inner, _) = group(&chain[i..], '(', ')').expect("`.options` is a call");
            literals(inner)
        });

        read.push((start, end, Some(Written {
            site: format!("{show}:{}", line_of(start)),
            name,
            text: dir.join(&include),
            options,
            upstream: chain.contains(".upstream()"),
            wanted: Vec::new(),
        })));
    }

    let readable: Vec<usize> = read.iter().filter(|(.., r)| r.is_some()).map(|&(s, ..)| s).collect();
    let mut out = Read::default();
    for (start, end, written) in read {
        let Some(mut written) = written else {
            out.unread.push(format!("{show}:{}", line_of(start)));
            continue;
        };
        let next = readable.iter().copied().find(|&s| s > start).unwrap_or(text.len());
        written.wanted = instantiations(&text[end.min(next)..next]);
        out.roots.push(written);
    }
    out
}

fn walk(dir: &Path, out: &mut Vec<(PathBuf, Read)>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("a readable directory")
        .map(|e| e.expect("a readable entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let read = scrape(&path);
            out.push((path, read));
        }
    }
}

/// The roots whose arguments are computed rather than written, and how each
/// lattice is reached instead.
///
/// The reconciliation below turns any OTHER unreadable declaration into a
/// failure, so a third lattice cannot arrive uncompiled and unmentioned.
const COMPUTED: &[(&str, &str)] = &[
    ("xqa.rs", "`mha_root` stamps five members; reached through `xqa::ROOTS`"),
    ("fa2.rs", "two macros stamp fifty-six; reached through `fa2::{DECODE, PREFILL}`"),
];

/// The roots that name no instantiation, and why each has none to name.
///
/// A root with nothing to compile is nearly always a mistake — the `mod inst`
/// drifted away from it, or the last body that fired it was deleted — so one
/// is a failure unless it is written down here.
const NO_INSTANTIATIONS: &[(&str, &str)] = &[
    (
        "attn/pack_dense_mask",
        "neither packer has a host program: the root is carried so the text \
         stays reachable, and nothing in this crate compiles it",
    ),
    (
        "attn/attention_mla_fa2",
        "the six template-ids `mla_fa2::SYMBOLS` names were never transcribed, \
         so the module has trace symbols, a grid, three arms and their measured \
         sizes — and no instantiation to compile. `smem_echo.rs` reaches the \
         root through its `__device__` echoes, which need no `mod inst`; the \
         kernels themselves are unreachable until someone writes the strings. \
         Nothing has noticed because `driver-cuda` refuses `KvStyle::Mla` at \
         model load (`bind/arms/attn.rs`), so no trace names any of the six",
    ),
];

/// Every written root in `src/x`, with its instantiations.
fn written() -> Vec<Job> {
    let mut files = Vec::new();
    walk(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src/x"), &mut files);

    // A submodule's instantiations belong to the root its parent module
    // declares. Done before anything is checked, so a root is not reported
    // empty for instantiations that are merely in the next file over.
    let loose: Vec<(PathBuf, Vec<String>)> = files
        .iter()
        .filter(|(_, read)| !read.loose.is_empty())
        .map(|(path, read)| {
            let dir = path.parent().expect("a submodule file has a parent");
            (dir.with_extension("rs"), read.loose.clone())
        })
        .collect();

    let (mut roots, mut unread): (Vec<Written>, Vec<String>) = (Vec::new(), Vec::new());
    for (path, mut read) in files {
        for (owner, wanted) in &loose {
            if *owner == path && read.roots.len() == 1 {
                read.roots[0].wanted.extend(wanted.iter().cloned());
            }
        }
        roots.append(&mut read.roots);
        unread.append(&mut read.unread);
    }

    // The parser is this fixture's own subject. If it stopped matching a
    // declaration form, every assertion would pass by finding nothing, so
    // every `Root::new` it could not read is named and accounted for.
    unread.retain(|site| !COMPUTED.iter().any(|(file, _)| site.starts_with(file)));
    assert!(
        unread.is_empty(),
        "the parser could not read {} `Root::new` declaration(s), so they went \
         uncompiled: {unread:?}\nEither the declaration form changed, or a new \
         computed root belongs in `COMPUTED`.",
        unread.len()
    );

    let bodiless: Vec<String> = roots
        .iter()
        .filter(|r| r.wanted.is_empty() && !NO_INSTANTIATIONS.iter().any(|(n, _)| *n == r.name))
        .map(|r| format!("{} ({})", r.name, r.site))
        .collect();
    assert!(
        bodiless.is_empty(),
        "{} root(s) name no instantiation, which means either the `mod inst` \
         moved away from its root or the root has no reader left: {bodiless:?}",
        bodiless.len()
    );

    roots
        .into_iter()
        .filter(|r| !r.wanted.is_empty())
        .map(|r| Job {
            text: std::fs::read_to_string(&r.text).unwrap_or_else(|why| {
                panic!("{}: `{}`'s text at {:?}: {why}", r.site, r.name, r.text)
            }),
            headers: if r.upstream { Headers::LibraryAndUpstream } else { Headers::Library }.set(),
            site: r.site,
            name: r.name,
            options: r.options,
            wanted: r.wanted,
        })
        .collect()
}

// ===========================================================================
// The computed roots: asked of the crate
// ===========================================================================

/// Both lattices, each point with the arms it was built to answer.
fn computed() -> Vec<Job> {
    use kernels_cuda_new::x::{fa2, xqa};

    let mut out = Vec::new();
    for point in &fa2::DECODE {
        let site = format!("fa2::DECODE hd{} g{}", point.head_dim, point.group_size);
        let wanted = point.arms.iter().map(|&a| a.to_owned()).collect();
        out.push(Job::from_root(site, &point.root, wanted));
    }
    for point in &fa2::PREFILL {
        let site = format!(
            "fa2::PREFILL hd{} q{} kv{}",
            point.head_dim, point.cta_tile_q, point.num_mma_kv
        );
        let wanted = point.arms.iter().map(|&a| a.to_owned()).collect();
        out.push(Job::from_root(site, &point.root, wanted));
    }
    for (nth, root) in xqa::ROOTS.iter().enumerate() {
        let site = format!("xqa::ROOTS[{nth}]");
        out.push(Job::from_root(site, root, vec![xqa::inst::MHA[nth].to_owned()]));
    }
    out
}

// ===========================================================================

#[test]
fn every_instantiation_a_body_names_compiles() {
    let Ok(have) = nvrtc::version() else {
        eprintln!("SKIPPED: libnvrtc will not load, so nothing here can be compiled");
        return;
    };
    let arch = kernels_cuda_new::jit::cache::arch().unwrap_or("compute_89");

    let (written, computed) = (written(), computed());
    let count = |jobs: &[Job]| jobs.iter().map(|j| j.wanted.len()).sum::<usize>();
    eprintln!(
        "nvrtc {have} targeting {arch}: {} written roots ({} instantiations), \
         {} computed ({})",
        written.len(),
        count(&written),
        computed.len(),
        count(&computed)
    );

    // One thread per root. The lattices are FlashInfer, whose points take tens
    // of seconds each and would otherwise make this too slow to leave on.
    let jobs: Vec<Job> = written.into_iter().chain(computed).collect();
    let failed: BTreeMap<String, String> = std::thread::scope(|scope| {
        let running: Vec<_> =
            jobs.iter().map(|job| scope.spawn(move || (job, job.compile(arch)))).collect();
        running
            .into_iter()
            .filter_map(|handle| match handle.join().expect("a compile thread") {
                (_, Ok(())) => None,
                (job, Err(why)) => Some((format!("{} ({})", job.name, job.site), why)),
            })
            .collect()
    });

    assert!(
        failed.is_empty(),
        "{} of {} roots would not compile:\n\n{}",
        failed.len(),
        jobs.len(),
        failed
            .iter()
            .map(|(what, why)| format!("── {what} ──\n{why}\n"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
