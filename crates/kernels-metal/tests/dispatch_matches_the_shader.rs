//! What a routine BINDS, against what the entrypoint DECLARES.
//!
//! `Arg::SPELLING` exists, and `kernels-metal/src/routine.rs` says what for:
//!
//! > The point of `SPELLING` is a generated cross-check against the real
//! > shader, which is worth nothing if the strings were guessed.
//!
//! The strings were not guessed -- `the_spellings_are_the_ones_the_msl_tree_declares`
//! holds each one against a count taken from `kernels/` -- but the cross-check
//! it was generated FOR was never written. This file is that cross-check.
//!
//! # Why it could not be built on `SPELLING` alone
//!
//! A routine's SIGNATURE and its DISPATCH LIST are two lists, and
//! `driver-metal/src/lowering/routine.rs` spends a section on the twenty-three
//! places they differ: an entrypoint that numbers its buffers with holes fills
//! them with a `pad` taken once in the signature and bound five times in the
//! dispatch, and `gdn_core` reaches three of its eleven buffers through
//! `ctx.ask` rather than through a parameter. `Routine::spelling` is the
//! signature's; the shader declares the dispatch's. Comparing the first pair
//! is comparing two lists that were never meant to be equal.
//!
//! So this invokes the BODY. Every row in `ROUTINES` is called against a
//! recording `Encode` that answers every ask and records each `ctx.fire` --
//! its `(file, entrypoint)` and the ordered `ArgValue` run it bound -- and
//! THAT list is the one the shader can be held against, because it is the one
//! the encoder would have bound.
//!
//! # What it catches, and the two failures that are the reason it exists
//!
//! `gdn_core_bfloat16` lost its `constant GdnCoreParams&` and grew eleven
//! `const constant int&` in its place. Every scalar index moved and `slot_ids`
//! moved with them, and `driver-metal/tests/device_gdn.rs` went on staging a
//! packed struct at 11 -- handing a POINTER to a kernel declaring
//! `const constant int& Dk` and leaving ten scalars bound to nothing. Nothing
//! refused: the pipeline builds, the dispatch encodes, the kernel reads a
//! garbage extent and writes nothing. A stale binding table on this plane
//! fails SILENTLY, because a signature that is N separate buffer indices has
//! no arity to get wrong at the call.
//!
//! `gdn_core_recurrent_prefill` is the other. Its `core_out` sits at buffer 3
//! and was bound as a READ, so the encoder saw no hazard between the scan and
//! the `gated_rms` that consumes it, ran the two at once, and qwen3.6 answered
//! a two-token prompt thirteen logits differently every time it was asked.
//! `qmm_splitk_reduce` declared its `y` the same way.
//!
//! Both are one question -- *does argument N look like what buffer N
//! declares?* -- asked of the kind, the scalar width, and the direction.
//!
//! There is a second question the first cannot reach. A routine may bind
//! NOTHING at a slot: `ctx.absent()` mints a handle with no allocation, and at
//! a pointer slot that is a perfectly well-shaped `const device T*` which
//! happens to address no memory. `kv_append_paged`'s buffer 15 was that, and
//! it was only ever visible because the declaration there is a SCALAR. So
//! `a_null_lands_only_where_somebody_has_read_the_guard` asks the eight
//! pointer cases separately, against a list that names the compile-time flag
//! keeping each one unread.
//!
//! # What it does NOT check
//!
//! The ELEMENT of a pointer. A template is written `const device T*` and the
//! instantiation supplies `bfloat`, so the definition this parses names a type
//! parameter rather than a type.
//!
//! The parser is not what stops it. `instantiate_*` carries the concrete list
//! and the preprocessor below already expands those calls, so the shader half
//! of an element comparison is one short step away. The DISPATCH half is not
//! there at all: `ArgValue` is a handle, a shape and a scalar, and nothing on
//! it says bf16 or f32. A body binds `In<Tensor<bf16>>` and what arrives here
//! is `Shaped { handle, rows, width }` -- the element was a fact of the MARK,
//! and it is spent by the time the value is made. Whoever wants this check
//! should widen the value, not the parser.
//!
//! Pointer-vs-scalar, scalar WIDTH and direction are all recoverable from the
//! definition, and all three are where the failures were.
//!
//! What CAN be asked without widening anything is the element the shader tree
//! states twice. A stamp writes its type into the host name and passes it as a
//! template argument, by hand, on one macro line, and
//! `a_stamps_name_and_the_type_it_instantiates_are_the_same_type` holds the
//! two against each other for 437 of the 470 stamps. That is not the check
//! above -- it cannot see a bind at all -- but it covers the half of the
//! element question that has ever been observed to drift, because a row in a
//! forty-line instantiation table is a thing people copy.
//!
//! Nor does it check an index no entrypoint declares. A hole is a `pad`, and a
//! pad is bound deliberately.
//!
//! # The invariant that came back with the preprocessor
//!
//! Reaching a declaration meant expanding the macros that stamp entrypoint
//! names, and that set is exactly what `tests/entrypoints.rs` records as
//! having left the tree: the set equality `.wiki/kernel-x/metal-refactor.md`
//! §9 lists first needed a C preprocessor, arrived for a while as a committed
//! `entrypoints.generated.txt`, and went when that file did. It is asserted
//! again at the bottom of this file, against the shader tree directly. That
//! was not the reason to write this, and it is the better half of what it
//! turned out to be worth.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use kernels::routine::Refusal;
use kernels::{Source, Ty};
use kernels_metal::ROUTINES;
use kernels_metal::routine::{ArgValue, Encode};

/// What an argument LOOKS LIKE, at the resolution both sides can state.
///
/// The element is deliberately absent -- see the header. What survives is the
/// three distinctions that were actually got wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `const device T*` -- a buffer the kernel reads.
    Read,
    /// `device T*` -- a buffer the kernel writes.
    Write,
    /// A 32-BIT INTEGER SCALAR, however it is spelled.
    ///
    /// `const constant int&` and `constant uint&` are ONE kind here, and the
    /// collapse is the repo's own criterion: `ArgValue` says the scalar kinds
    /// are separate variants *"because the widths differ and the check that
    /// matters is exactly the width one"*, and these two do not differ in
    /// width. Ten sites bind a `Const<i32>` at a `constant uint&` -- the seven
    /// `rms_*`, both `gated_rms_*`, `vnorm_single_row` and both
    /// `shared_expert_combine*` -- every one of them an axis size or an
    /// element count, whose nonnegative `i32` has the `uint`'s bits. Splitting
    /// them would report ten deliberate spellings and nothing else.
    Int,
    /// `const constant float&`. NOT collapsed into [`Self::Int`]: same width,
    /// different interpretation, and a kernel reading an integer's bits as a
    /// float is the failure this distinction is kept for.
    F32,
}

impl Kind {
    /// How the shader's declaration reads.
    ///
    /// `constant` before `device` on purpose: `const constant int&` contains
    /// the word `const` and so does `const device int*`, and a match on
    /// `const` first would call every scalar a read buffer.
    ///
    /// A `constant X&` is a SCALAR only when `X` is a primitive. `constant
    /// ArgmaxParams&` is the packed convention -- one buffer holding every
    /// field -- and it is bound with an ADDRESS, so it reads as a buffer here.
    /// `packed_params_cover_the_struct.rs` records that no model text binds one
    /// any more; the sampler and `ptir` still do, which is why the case is
    /// live rather than historical.
    fn of_declaration(spelling: &str) -> Option<Self> {
        let s = spelling.split_whitespace().collect::<Vec<_>>().join(" ");
        if s.contains("constant") && s.contains('&') {
            return Some(if s.contains("float") {
                Self::F32
            } else if s.contains("int") || s.contains("unsigned") || s.contains("size_t")
                || s.contains("short")
            {
                Self::Int
            } else {
                // A struct, taken by address.
                Self::Read
            });
        }
        if s.contains('*') {
            // BOTH ORDERS ARE MSL AND BOTH MEAN READ-ONLY. `quant/transcode.metal`
            // writes `device const uchar* payload`, where every other file
            // writes `const device T*`, so a test for a leading `const` called
            // all five of its parameters writes and reported three entrypoints
            // that were never wrong.
            return Some(if s.contains("const") {
                Self::Read
            } else {
                Self::Write
            });
        }
        None
    }

    /// How the bound value reads. `Shaped` is a read: it is what
    /// `kernels::bind` mints for an operand slot, and a body re-emits it
    /// through `Bind::arg`, which is the immutable spelling.
    fn of_value(v: ArgValue) -> Option<Self> {
        Some(match v {
            ArgValue::Buffer(_) | ArgValue::Shaped { .. } => Self::Read,
            ArgValue::BufferMut(_) => Self::Write,
            ArgValue::I32(_) | ArgValue::U32(_) => Self::Int,
            ArgValue::F32(_) => Self::F32,
            ArgValue::Usize(_) => return None,
        })
    }
}

/// One entrypoint's declared arguments: `buffer index -> kind`.
type Declared = BTreeMap<usize, Kind>;

/// The shader tree, parsed once: `file -> template name -> declaration`, and
/// the `host_name` prefixes each template is stamped out under.
#[derive(Default)]
struct Shaders {
    /// `(file, template) -> declared arguments`.
    templates: BTreeMap<(String, String), Declared>,
    /// `(file, host_name prefix) -> template`, longest prefix winning at
    /// lookup. `instantiate_gdn_core` writes `[[host_name("gdn_core_" #name)]]`
    /// above `gdn_core<itype>`, so the prefix is a literal and the template is
    /// the identifier that follows it.
    stamps: BTreeMap<(String, String), String>,
    /// `(file, host_name prefix) -> the `<...>` the stamp instantiates`.
    args: BTreeMap<(String, String), String>,
}

/// The shader directory this crate publishes.
fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels")
}

/// The text with `//` and `/* */` comments blanked out.
///
/// NOT decoration. `kv_write.metal` annotates its parameters --
/// `const device T* k_new [[buffer(0)]], // [n_kv_heads, head_dim]` -- and the
/// comma that ends the parameter comes BEFORE the comment, so a split on
/// commas hands the next parameter its predecessor's note. `v_new` then read
/// as `// [n_kv_heads, head_dim] const device T*`, which does not begin with
/// `const`, and this file called a read buffer a write on four entrypoints
/// before the comments came out.
///
/// Blanked rather than removed, so every byte offset this file computes still
/// points where it did — which is also why it works in BYTES: blanking a
/// multi-byte character one `char` at a time would shorten the text and move
/// every offset after it.
fn without_comments(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(b' ');
                i += 1;
            }
            continue;
        }
        if bytes[i..].starts_with(b"/*") {
            while i < bytes.len() && out.len() < bytes.len() && !bytes[i..].starts_with(b"*/") {
                out.push(if bytes[i] == b'\n' { b'\n' } else { b' ' });
                i += 1;
            }
            while i < bytes.len() && bytes[i..].starts_with(b"*/") {
                out.push(b' ');
                out.push(b' ');
                i += 2;
            }
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| text.to_string())
}

/// Split a parenthesised parameter list on TOP-LEVEL commas.
///
/// `[[buffer(11)]]` has a comma-free interior but `uint3` attributes and
/// nested attribute parens do not, so this tracks depth rather than splitting
/// on every comma.
fn top_level_commas(list: &str) -> Vec<String> {
    let (mut out, mut depth, mut cur) = (Vec::new(), 0i32, String::new());
    for c in list.chars() {
        match c {
            '(' | '[' | '<' => {
                depth += 1;
                cur.push(c);
            }
            ')' | ']' | '>' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth == 0 => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// The text from `open`'s matching `(` to its `)`, or `None` if unbalanced.
fn balanced(text: &str, open: usize) -> Option<&str> {
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return text.get(open + 1..i);
                }
            }
            _ => {}
        }
    }
    None
}

/// A function-like `#define`, with its parameters and its body.
struct Define {
    /// The macro's own name.
    name: String,
    /// Its formal parameters, in order.
    params: Vec<String>,
    /// Everything after the parameter list, with continuations joined.
    body: String,
}

/// Every function-like `#define` in one file.
///
/// Object-like defines are skipped: they carry no parameter list, so nothing
/// downstream substitutes into them.
fn defines(text: &str) -> Vec<Define> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(hit) = text[at..].find("#define ") {
        let start = at + hit + "#define ".len();
        at = start;
        let rest = &text[start..];
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() || !rest[name.len()..].starts_with('(') {
            continue;
        }
        let Some(params_src) = balanced(text, start + name.len()) else { continue };
        let params: Vec<String> = params_src
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();
        // The body runs to the first newline NOT preceded by a `\`.
        let body_start = start + name.len() + params_src.len() + 2;
        let mut i = body_start;
        let bytes = text.as_bytes();
        while i < bytes.len() {
            if bytes[i] == b'\n' {
                let mut j = i;
                while j > body_start && (bytes[j - 1] == b' ' || bytes[j - 1] == b'\t') {
                    j -= 1;
                }
                if j == body_start || bytes[j - 1] != b'\\' {
                    break;
                }
            }
            i += 1;
        }
        let body = text[body_start..i.min(text.len())].replace('\\', " ");
        out.push(Define { name, params, body });
        at = i;
    }
    out
}

/// Every expansion of every function-like macro in the file, concatenated.
///
/// A MINIMAL PREPROCESSOR, and only because this tree's entrypoint names and
/// two of its parameter lists are BUILT by one. `quant/qmv.metal` writes the
/// buffer list once inside `gptoss_qmv_kernel(name, ...)` and stamps four
/// templates out of it, then names each instantiation through a second macro
/// as `[[host_name(#host "_" #name "_gs_" #gs "_b_" #b)]]`; `attn/sdpa_paged.metal`
/// composes `fn "_" #name "_d_" #d`. Nothing about those names or those
/// declarations is readable without substituting, which is why
/// `tests/entrypoints.rs` records that the shader half of its own invariant
/// needed a C preprocessor and left the tree.
///
/// It is not one: it substitutes ONE level, twice, and understands `#param`
/// (stringized) and a bare parameter. That is everything the shader tree uses,
/// and a macro it cannot expand leaves an entrypoint unresolved -- which is
/// counted rather than assumed away, in
/// [`both_halves_of_the_comparison_are_there`].
fn expansions(text: &str, defs: &[Define]) -> String {
    // A USE INSIDE A `#define` IS NOT A USE. `instantiate_qmm_t_splitk`'s body
    // calls `instantiate_qmm_t_splitk_named(name, ptype, gs, bm, bk, bn, b)`,
    // and expanding THAT hands the inner macro its outer's parameter names as
    // arguments -- so the stamp comes out `affine_qmm_t_splitk_bfloat16_gs_gs_
    // b_b_bm_bm_bn_bn`, the parameter written where its value belongs.
    //
    // The real nesting still works and is why rounds exist: the outer macro's
    // CALL SITE expands with concrete arguments, and the text that comes out
    // carries the inner call with those arguments and no `#define` at all. So
    // the next round expands the castings while this skips the mould.
    let moulds = macro_bodies(text);
    let mut out = String::new();
    for def in defs {
        let mut at = 0usize;
        while let Some(hit) = text[at..].find(&def.name) {
            let start = at + hit;
            at = start + def.name.len();
            if moulds.iter().any(|(a, b)| (*a..*b).contains(&start)) {
                continue;
            }
            // A use, not the definition, and not a longer identifier that
            // happens to contain this one.
            let before = text[..start].chars().next_back().unwrap_or(' ');
            if before.is_alphanumeric() || before == '_' {
                continue;
            }
            if text[..start].trim_end().ends_with("#define") {
                continue;
            }
            if !text[at..].starts_with('(') {
                continue;
            }
            let Some(args_src) = balanced(text, at) else { continue };
            let args: Vec<String> = top_level_commas(args_src)
                .into_iter()
                .map(|a| a.trim().to_string())
                .collect();
            if args.len() != def.params.len() {
                continue;
            }
            let mut body = def.body.clone();
            for (param, arg) in def.params.iter().zip(&args) {
                body = substitute(&body, param, arg);
            }
            out.push('\n');
            out.push_str(&body);
        }
    }
    out
}

/// Replace `#param` with the argument QUOTED, and a whole-word `param` with
/// the argument as written.
///
/// `#` is C's stringizing operator, and the difference matters: `#name` with
/// `name = bfloat16` is the literal `"bfloat16"` that joins an entrypoint's
/// spelling, while a bare `itype` is the TYPE `bfloat` that joins a
/// declaration.
fn substitute(body: &str, param: &str, arg: &str) -> String {
    let hashed = format!("#{param}");
    let quoted = format!("\"{}\"", arg.trim_matches('"'));
    let body = body.replace(&hashed, &quoted);
    let mut out = String::with_capacity(body.len());
    let bytes = body.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let is_word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
        if body[i..].starts_with(param)
            && (i == 0 || !is_word(bytes[i - 1]))
            && bytes
                .get(i + param.len())
                .is_none_or(|&c| !is_word(c))
        {
            out.push_str(arg);
            i += param.len();
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// Every macro expansion the file produces, to a fixed depth.
///
/// ROUNDS, NOT ONE PASS. `attn/sdpa_paged.metal` stacks two layers --
/// `instantiate_sdpa_paged(bfloat16, bfloat, 128, 128)` expands to
/// `instantiate_sdpa_paged_impl("sdpa_paged_decode", ...)`, and only the inner
/// one carries the `[[host_name(...)]]` -- so a single pass produced the
/// `_p32` spellings, which ARE written directly, and none of the four plain
/// ones. Three rounds is one more than the tree currently nests; the round
/// that adds nothing stops it.
fn expanded(text: &str) -> String {
    let defs = defines(text);
    let (mut acc, mut cur) = (String::new(), text.to_string());
    for _ in 0..3 {
        let next = expansions(&cur, &defs);
        if next.trim().is_empty() {
            break;
        }
        acc.push('\n');
        acc.push_str(&next);
        cur = next;
    }
    acc
}

/// The byte range of every `#define` body, so a stamp can be told from its
/// own mould.
///
/// `instantiate_qmm_t` writes `[[host_name("affine_qmm_t_bfloat16_gs_" #gs
/// "_b_" #b ...)]]` INSIDE the define. Read literally, the adjacent literals
/// concatenate over the `#gs` gaps and produce
/// `affine_qmm_t_bfloat16_gs__b__bm__bn_` -- a name with the shape of an
/// entrypoint and none of its numbers, sitting in the stamp set beside the
/// thirty concrete ones the same macro really writes. Thirty-five of the
/// thirty-five names that reached no row were this, and none of them was a
/// shader.
///
/// The expansions carry no `#define`, so nothing there is skipped: what this
/// removes is the mould and not the castings.
fn macro_bodies(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(hit) = text[at..].find("#define ") {
        let start = at + hit;
        let mut i = start;
        while i < bytes.len() {
            if bytes[i] == b'\n' {
                let mut j = i;
                while j > start && (bytes[j - 1] == b' ' || bytes[j - 1] == b'\t') {
                    j -= 1;
                }
                if j == start || bytes[j - 1] != b'\\' {
                    break;
                }
            }
            i += 1;
        }
        out.push((start, i));
        at = i.max(start + 1);
    }
    out
}

/// Read one `.metal` file into templates and stamps.
///
/// A TEMPLATE DEFINITION is `[[kernel]] void name(` with `[[buffer(N)]]` in
/// its list; an INSTANTIATION is `name<itype>(` inside an `instantiate_*`
/// macro and carries no buffer numbers. Counting the second as the first
/// doubled the census the first time `scripts/metal-kernel-audit.py` was run
/// by hand, and it would produce empty declarations here.
fn read_shader(file: &str, text: &str, out: &mut Shaders) {
    let moulds = macro_bodies(text);
    // TWO SPELLINGS. `quant/transcode.metal` writes `kernel void
    // mxfp4_dequant_bf16(` with no attribute brackets at all, and scanning for
    // `[[kernel]]` alone skipped all three of its entrypoints.
    let mut marks: Vec<usize> = Vec::new();
    for (needle, skip) in [("[[kernel]]", "[[kernel]]".len()), ("kernel void", "kernel".len())] {
        let mut at = 0usize;
        while let Some(hit) = text[at..].find(needle) {
            marks.push(at + hit + skip);
            at = at + hit + needle.len();
        }
    }
    marks.sort_unstable();
    marks.dedup();
    for start in marks {
        // The same mould that stamps a name also DECLARES one.
        // `gptoss_qmv_kernel` writes `[[kernel]] void name(` with the whole
        // buffer list under it, so read where it stands the template is called
        // `name` -- an entrypoint no row can declare, because there is no such
        // kernel. Its expansions carry `qmv_tail` and `qmv_routed` with the
        // identical list, which is where the declarations actually come from.
        if moulds.iter().any(|(a, b)| (*a..*b).contains(&start)) {
            continue;
        }
        // ANCHOR ON `void`, NOT ON THE NEXT `(`. A second attribute may sit
        // between the two -- `[[kernel]]
        // [[max_total_threads_per_threadgroup(1024)]] void sdpa_paged_decode(`
        // -- and its own parenthesis is the first one after `[[kernel]]`. Every
        // `sdpa_paged_*` and `sdpa_paged_mma_*` template was skipped on that,
        // which is eight entrypoints this file silently did not compare.
        let Some(vpos) = text[start..].find("void") else { continue };
        let after_void = start + vpos + "void".len();
        let rest = &text[after_void..];
        let lead = rest.len() - rest.trim_start().len();
        let name: String = rest
            .trim_start()
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        let name_end = after_void + lead + name.len();
        // `gdn_core<itype>(` is an INSTANTIATION inside an `instantiate_*`
        // macro, not a definition: it carries no buffer numbers, and counting
        // it as a definition would overwrite the real one with an empty list.
        if text[name_end..].trim_start().starts_with('<') {
            continue;
        }
        let Some(paren) = text[name_end..].find('(') else { continue };
        let Some(list) = balanced(text, name_end + paren) else { continue };
        let mut declared = Declared::new();
        for param in top_level_commas(list) {
            let Some(b) = param.find("[[buffer(") else { continue };
            let rest = &param[b + "[[buffer(".len()..];
            let Some(close) = rest.find(')') else { continue };
            let Ok(index) = rest[..close].trim().parse::<usize>() else { continue };
            // The declaration is everything before the attribute, minus the
            // parameter's own name.
            let decl = param[..b].trim();
            let decl = decl.rsplit_once(char::is_whitespace).map_or(decl, |(t, _)| t);
            if let Some(kind) = Kind::of_declaration(decl) {
                declared.insert(index, kind);
            }
        }
        if !declared.is_empty() {
            out.templates.insert((file.to_string(), name.clone()), declared);
        }
    }

    // The stamps: `[[host_name(<expr>)]] [[kernel]] void <template><...>`.
    //
    // EVERY adjacent literal, not the first. C concatenates them and so does
    // the compiler that reads this: `#host "_" #name "_gs_" #gs "_b_" #b` is
    // one name in five pieces, and taking only the first recorded a stamp
    // called `_` -- which no entrypoint starts with, so the whole `qmv` and
    // `sdpa_paged_tiled` families resolved to nothing and were never compared.
    let mut at = 0usize;
    while let Some(hit) = text[at..].find("[[host_name") {
        let here = at + hit;
        let start = here + "[[host_name".len();
        at = start;
        if moulds.iter().any(|(a, b)| (*a..*b).contains(&here)) {
            continue;
        }
        let Some(expr) = balanced(text, start) else { continue };
        let mut literal = String::new();
        let mut rest = expr;
        while let Some(q1) = rest.find('"') {
            let after = &rest[q1 + 1..];
            let Some(q2) = after.find('"') else { break };
            literal.push_str(&after[..q2]);
            rest = &after[q2 + 1..];
        }
        if literal.is_empty() {
            continue;
        }
        let after = &text[start + expr.len()..];
        let Some(vpos) = after.find("void") else { continue };
        let tail = &after[vpos + "void".len()..];
        let ident: String = tail
            .trim_start()
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !ident.is_empty() {
            let after_ident = tail.trim_start().len() - tail.trim_start()[ident.len()..].len();
            let rest2 = &tail.trim_start()[after_ident..];
            let targs = if rest2.starts_with('<') {
                let mut depth = 0i32;
                let mut end = 0usize;
                for (i, c) in rest2.char_indices() {
                    if c == '<' { depth += 1; }
                    if c == '>' { depth -= 1; if depth == 0 { end = i; break; } }
                }
                rest2[1..end].to_string()
            } else {
                String::new()
            };
            out.args.insert((file.to_string(), literal.clone()), targs);
            out.stamps
                .insert((file.to_string(), literal), ident);
        }
    }
}

/// Every `.metal` under the shader directory, parsed.
fn shaders() -> Shaders {
    let root = kernels_dir();
    let mut out = Shaders::default();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("the shader directory is readable") {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("metal") {
                continue;
            }
            let rel = path
                .strip_prefix(&root)
                .expect("under the root")
                .to_string_lossy()
                .into_owned();
            let text = std::fs::read_to_string(&path).expect("a readable shader");
            let text = without_comments(&text);
            // The expansions go in the SAME pass and after the original, so a
            // macro-built template or stamp is read by exactly the parser that
            // reads a hand-written one.
            let whole = format!("{text}\n{}", expanded(&text));
            read_shader(&rel, &whole, &mut out);
        }
    }
    out
}

impl Shaders {
    /// The declaration a fired `(file, entrypoint)` resolves to.
    ///
    /// By STEM, longest first, which is `driver-metal`'s own rule: both
    /// `gdn_core_bfloat16` and `gdn_core_slotted_bfloat16` start with
    /// `gdn_core_`, and only the longer stamp tells them apart.
    fn declaration(&self, file: &str, entrypoint: &str) -> Option<&Declared> {
        let mut best: Option<(usize, &String)> = None;
        for ((f, literal), template) in &self.stamps {
            if f != file || !entrypoint.starts_with(literal.as_str()) {
                continue;
            }
            if best.is_none_or(|(n, _)| literal.len() > n) {
                best = Some((literal.len(), template));
            }
        }
        if let Some((_, template)) = best
            && let Some(d) = self.templates.get(&(file.to_string(), template.clone()))
        {
            return Some(d);
        }
        // A point written out longhand rather than stamped by a macro: the
        // entrypoint IS the template.
        self.templates.get(&(file.to_string(), entrypoint.to_string()))
    }
}

/// Records what a body would have bound, and answers whatever it asks.
#[derive(Default)]
struct Recorder {
    /// `(file, entrypoint, the ordered run)`.
    fires: RefCell<Vec<(&'static str, &'static str, Vec<ArgValue>)>>,
    /// What every scalar answer is filled with on this pass.
    fill: i32,
}

impl Encode for Recorder {
    fn fire(&self, fire: kernels::routine::Fire, args: &[ArgValue]) -> Result<(), Refusal> {
        self.fires
            .borrow_mut()
            .push((fire.file, fire.entrypoint, args.to_vec()));
        Ok(())
    }

    /// GENEROUS BY CONSTRUCTION. A refusal here is a routine this file did not
    /// reach, and an unreached routine is coverage lost silently -- so every
    /// fact is answered, and the shape of the answer follows the `Ty` alone.
    ///
    /// ONE SOURCE IS READ, and it is the one that is not a question.
    /// `ctx.absent()` resolves `Source::Lit(Lit::Null)`, which a real binder
    /// turns into a handle with no allocation -- so a value that came from
    /// there is marked with [`NIL`] and the assertion below can ask where the
    /// nulls land.
    fn resolve(&self, ty: Ty, source: Source) -> Result<ArgValue, Refusal> {
        if matches!(source, Source::Lit(kernels::Lit::Null)) {
            return Ok(ArgValue::Shaped { handle: NIL, rows: 0, width: 0 });
        }
        Ok(value_for(ty, self.fill))
    }
}

/// The handle this file gives an `absent()` bind, so it can be told from a
/// real one. Nothing dereferences a handle here, so any value not otherwise
/// minted will do.
const NIL: u32 = u32::MAX;

/// A plausible value for one argument type.
///
/// The handle is an index and nothing here dereferences it; what matters is
/// the VARIANT, because that is the half of the comparison this file makes.
fn value_for(ty: Ty, fill: i32) -> ArgValue {
    match ty {
        Ty::I32 => ArgValue::I32(fill),
        Ty::F32 => ArgValue::F32(fill as f32),
        Ty::U32 => ArgValue::U32(fill.unsigned_abs()),
        Ty::I64 | Ty::Usize => ArgValue::Usize(fill.unsigned_abs().into()),
        _ => ArgValue::Shaped {
            handle: 0,
            rows: fill,
            width: fill,
        },
    }
}

/// Every fire every routine makes, over a spread of scalar fillings.
///
/// ONE FILLING IS NOT ENOUGH and the spread is not decoration: a body
/// validates its own geometry and refuses what cannot be dispatched, so a
/// single `0` reaches almost nothing (`Refusal::Empty { what: "v_dim" }`) and
/// a single `128` misses every routine whose scalar selects a compiled tiling
/// -- `gdn_core_recurrent_prefill` is stamped for nine `(lanes, vrows)` points
/// and `128` is not one of them. Firing each row under several fillings
/// collects the union, which is why the count below is larger than the number
/// of routines.
fn every_fire() -> Vec<(&'static str, &'static str, Vec<ArgValue>)> {
    let mut all = Vec::new();
    for fill in [1, 2, 4, 8, 16, 32, 64, 128] {
        for row in ROUTINES {
            let rec = Recorder {
                fill,
                ..Recorder::default()
            };
            let values: Vec<ArgValue> = row.args.iter().map(|t| value_for(*t, fill)).collect();
            let ctx: &kernels_metal::routine::Ctx<'_> = &rec;
            // A refusal is expected and carries nothing: this is a synthetic
            // filling, not a plan.
            let _ = (row.body)(ctx, &values);
            all.extend(rec.fires.into_inner());
        }
    }
    all.sort_by(|a, b| (a.0, a.1, a.2.len()).cmp(&(b.0, b.1, b.2.len())));
    all.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1 && a.2 == b.2);
    all
}

/// Where a bound argument does not have its buffer's shape, and why.
///
/// `(entrypoint stem, buffer index)`. Stated rather than skipped by a rule: a
/// rule would also excuse the next one, and the next one is the failure this
/// file exists to report.
///
/// IT IS EMPTY, AND IT HELD ONE ENTRY. `kv_append_paged`'s buffer 15 is
/// `const constant int& src_row_stride` and the routine bound `ctx.absent()`
/// there -- a `Lit::Null`, a handle with no allocation, at a slot the shader
/// really declares and really reads. It survived on the shader's own guard:
/// `kv_write.metal:73` takes `src_row_stride > 0 ? src_row_stride :
/// row_stride`, so a zero selects the packed `[N, n_kv_heads, head_dim]` a
/// decode hands over. The entry recorded the residual risk rather than arguing
/// it away -- nothing guaranteed a nil bind reads as zero.
///
/// The routine binds a stated `0` now, so the slot agrees with its declaration
/// and there is nothing to excuse. What the exception bought was the reason to
/// look: six slots beside it are holes the shader declares nothing at, and the
/// seventh had been wearing their clothes since a row named it `ring_15`.
///
/// The machinery stays for the next real one. An empty list is the strongest
/// thing this constant can say.
const DELIBERATE: &[(&str, usize)] = &[];

/// The question the GDN failure was: is argument N shaped like buffer N?
#[test]
fn every_bound_argument_matches_the_buffer_it_lands_on() {
    let shaders = shaders();
    let mut wrong: Vec<String> = Vec::new();
    let mut short: Vec<String> = Vec::new();
    let mut checked = 0usize;
    let mut excused = 0usize;

    for (file, entrypoint, args) in every_fire() {
        let Some(declared) = shaders.declaration(file, entrypoint) else {
            continue;
        };
        for (&index, &want) in declared {
            let Some(&got) = args.get(index) else {
                short.push(format!(
                    "  {entrypoint} ({file}): declares [[buffer({index})]] and the dispatch \
                     binds {} argument(s), so it is left UNBOUND -- the kernel reads a \
                     garbage extent and no refusal reports it.",
                    args.len()
                ));
                continue;
            };
            let Some(got) = Kind::of_value(got) else {
                continue;
            };
            checked += 1;
            if got != want {
                if DELIBERATE
                    .iter()
                    .any(|(stem, at)| *at == index && entrypoint.starts_with(stem))
                {
                    excused += 1;
                    continue;
                }
                wrong.push(format!(
                    "  {entrypoint} ({file}): [[buffer({index})]] is declared {want:?} and the \
                     dispatch binds {got:?}."
                ));
            }
        }
    }

    assert!(
        short.is_empty(),
        "an entrypoint declares an argument the dispatch never binds:\n{}",
        short.join("\n")
    );
    assert!(
        wrong.is_empty(),
        "a bound argument does not have the shape of the buffer it lands on -- a pointer \
         where the kernel declares a scalar reads a garbage extent, and a read where it \
         declares a write loses the encoder's hazard:\n{}",
        wrong.join("\n")
    );
    assert!(
        checked > 0,
        "nothing was compared, so this file is green because it looked at NOTHING: either no \
         routine fired or no entrypoint resolved to a template"
    );
    // A RATCHET IS A NUMBER PLUS AN ASSERTION. `DELIBERATE` names a slot that
    // this file must still REACH: an entry nothing excuses is an exception
    // whose subject moved, and it would go on excusing whatever landed on that
    // index next. Empty, it has no subject to lose.
    assert!(
        DELIBERATE.is_empty() || excused > 0,
        "`DELIBERATE` excuses {} slot(s) and none of them was reached, so the entry describes a \
         dispatch this file no longer sees and is excusing nothing",
        DELIBERATE.len()
    );
}

/// The guard the assertion above needs: it is green when it compares nothing,
/// and the two ways to compare nothing are a shader tree that did not parse
/// and a routine set that did not fire.
#[test]
fn both_halves_of_the_comparison_are_there() {
    let shaders = shaders();
    assert!(
        shaders.templates.len() >= 80,
        "the shader tree parsed {} entrypoint definitions, and the tree carried EIGHTY-SIX \
         when this number was written -- a parse that suddenly sees fewer has lost a spelling, \
         not a file",
        shaders.templates.len()
    );
    let fires = every_fire();
    assert!(
        fires.len() >= 440,
        "only {} distinct dispatches were recorded from {} routines, against the 464 measured \
         when this number was written; a body that refuses every filling is coverage this file \
         silently does not have, and the way it goes missing is a `value_for` arm that cannot \
         answer some `Ty` -- the routine then refuses on KIND, before it ever fires",
        fires.len(),
        ROUTINES.len()
    );
    let resolved = fires
        .iter()
        .filter(|(f, e, _)| shaders.declaration(f, e).is_some())
        .count();
    assert!(
        resolved == fires.len(),
        "{resolved} of {} recorded dispatches resolved to a shader definition. EVERY one of them \
         did when this was written, which is the strongest form this guard has: an entrypoint \
         the parser cannot find is not compared, and a dispatch that is not compared is exactly \
         the hole this file exists to close",
        fires.len()
    );
}

/// The names the shader tree writes as ENTRYPOINTS, as opposed to templates.
///
/// A `[[host_name("...")]]` stamp is one. A `kernel void name(` with nothing
/// stamping it is one -- `quant/transcode.metal` and the routing kernels are
/// spelled that way, eleven of them. A template that a stamp names is NOT one:
/// `gdn_core<itype>` is a body three stamps point at, and counting it would be
/// counting a function nobody can dispatch.
fn stamped(shaders: &Shaders) -> BTreeSet<String> {
    let bodies: BTreeSet<&String> = shaders.stamps.values().collect();
    let mut named: BTreeSet<String> =
        shaders.stamps.keys().map(|(_, n)| n.clone()).collect();
    for (_, template) in shaders.templates.keys() {
        if !bodies.contains(template) {
            named.insert(template.clone());
        }
    }
    named
}

/// The set equality of `.wiki/kernel-x/metal-refactor.md` §9, the half that
/// costs a
/// device to find out about.
///
/// > every entrypoint in `kernels/` resolves to exactly one (row, axis point),
/// > and every (row, axis point) to exactly one entrypoint
///
/// `tests/entrypoints.rs` records this as unrecoverable: the shader half of
/// the comparison is an `instantiate_*` expansion, so it arrived as a
/// committed `entrypoints.generated.txt` that a Python script wrote, and when
/// that artifact went so did "the only hermetic view a `cargo test` had of what
/// the shaders instantiate". The preprocessor above is that view, and this is
/// the comparison coming back.
///
/// THIS direction is the expensive one. A row whose axes over-generate names a
/// pipeline that was never compiled, and the shape of that failure is a nil
/// pipeline at the moment a model first reaches the point -- not at build, not
/// at load, but partway through somebody's generate. Four hundred and
/// eighty-one names, matched EXACTLY and not by prefix, with no exceptions.
#[test]
fn every_entrypoint_the_table_declares_is_one_the_shader_tree_writes() {
    let shaders = shaders();
    let mut named: BTreeSet<String> =
        shaders.stamps.keys().map(|(_, n)| n.clone()).collect();
    for (_, template) in shaders.templates.keys() {
        named.insert(template.clone());
    }
    let table: BTreeSet<String> = kernels_metal::entrypoints().into_iter().collect();
    let missing: Vec<&str> = table
        .iter()
        .filter(|name| !named.contains(*name))
        .map(String::as_str)
        .collect();
    assert!(
        missing.is_empty(),
        "the table declares {} entrypoint(s) that no shader in `kernels/` writes, so each is a \
         nil pipeline waiting for the first model that reaches its axis point:\n{}",
        missing.len(),
        missing.join("\n")
    );
    assert_eq!(
        table.len(),
        481,
        "the table's axis product is {} and it was 481 when this comparison was restored; the \
         number is here so that a row that stops generating is as loud as one that generates too \
         much",
        table.len()
    );
}

/// The other direction, and it is a zero.
///
/// A shader name no row declares is a kernel nothing can dispatch: compiled
/// into the library, spending its build time, and dead. Together with the
/// assertion above this is invariant (1) as set EQUALITY -- 481 names on each
/// side and the same 481 -- which is what `entrypoints.generated.txt` used to
/// state and what nothing in `cargo test` has stated since it went.
///
/// It read thirty-five failures on the first pass and every one was the
/// preprocessor's, in three kinds. A `[[host_name]]` inside a `#define` body
/// concatenates its literals straight over the `#param` gaps and produces
/// `affine_qmm_t_bias_bfloat16_gs__b__bm__bn_`. Expanding a macro CALL that
/// sits inside another macro's body hands the inner one its outer's parameter
/// NAMES, so `..._splitk_bfloat16_gs_gs_b_b_bm_bm_bn_bn` is a parameter
/// written where its value belongs. And a mould declares a TEMPLATE as well as
/// stamping a name: `gptoss_qmv_kernel` writes `[[kernel]] void name(`, so the
/// tree appeared to hold a kernel called `name`.
///
/// All three are the same mistake -- reading a macro's mould as one of its
/// castings -- and `macro_bodies` is the one answer to it. The expansions
/// carry no `#define`, so the castings are untouched.
#[test]
fn a_shader_name_that_no_row_declares_is_a_kernel_nothing_can_reach() {
    let shaders = shaders();
    let named = stamped(&shaders);
    let table: BTreeSet<String> = kernels_metal::entrypoints().into_iter().collect();
    let orphan: Vec<&str> = named
        .iter()
        .filter(|name| {
            !table.contains(*name) && !table.iter().any(|t| t.starts_with(name.as_str()))
        })
        .map(String::as_str)
        .collect();
    assert!(
        orphan.is_empty(),
        "{} shader name(s) reach no entrypoint the table declares. Either a kernel is compiled \
         that no row can dispatch, or `macro_bodies` has stopped telling a mould from a casting \
         and these are half-expanded names rather than shaders:\n{}",
        orphan.len(),
        orphan.join("\n")
    );
    assert_eq!(
        named.len(),
        table.len(),
        "the shader tree writes {} entrypoint name(s) and the table declares {}. Neither list \
         has anything in it the other lacks -- the two assertions above say so -- so a \
         difference here is one side holding a name TWICE under two spellings",
        named.len(),
        table.len()
    );
}

/// Every slot where a routine binds NOTHING and the entrypoint declares
/// something, with the name of the thing it declares.
///
/// `(entrypoint, buffer index, the parameter's name in the shader)`.
///
/// `ctx.absent()` is a routine saying *"this cell is occupied and empty"*, and
/// on this plane the binder mints a handle with no allocation for it. Most of
/// them land at an index no entrypoint declares -- six of `kv_append_paged`'s
/// seven are holes in a `[[buffer(n)]]` numbering, and a hole is unread by
/// construction. These eight are the other case: the slot IS declared, so
/// whether the null is read depends on the kernel and not on the ABI.
///
/// All eight are one pattern. A template serves two instantiations and a
/// `MLX_MTL_CONST bool` decides which of them reads the slot, so the arm that
/// dereferences the pointer is not compiled into the variant bound here.
/// `qmv.metal:527` is the clearest: `Codec::zero_point ? U(bi_row[...]) :
/// U(0)`, and mxfp4's codec states `zero_point = false`.
///
/// THIS IS NOT THE SAME CLAIM AS A HOLE, which is why it is a list and not a
/// rule. `kv_append_paged`'s buffer 15 spent a long time being read as one of
/// its neighbours' holes while the shader declared and read it on every token;
/// the guard that saved it was in the kernel, and nobody had checked. An entry
/// here is a promise that somebody read the guard.
const NIL_AT_A_DECLARED_SLOT: &[(&str, usize, &str)] = &[
    // `bias`, and the guard is `BIASED`, a `bool` template parameter of
    // `qmv_gptoss_impl`. The load is `qmv.metal:541`, inside it.
    //
    // Read the two lines above that one before trusting this: 535 copies the
    // null into `bias_row` unconditionally, which is a copy and not a load,
    // and 536 offsets it -- `bias_row += expert_ids[sel] * out_vec_size` --
    // under `BIASED && ROUTED`. So nothing arithmetics on the null either.
    // A null POINTER that gets offset before anyone checks the flag would be
    // undefined and would look exactly like this from the bind side.
    ("affine_qmv_routed_bfloat16_gs_64_b_4", 7, "bias"),
    // `biases`, the affine codec's zero points. mxfp4 has none --
    // `Mxfp4::zero_point` is `MLX_MTL_CONST bool` and false at qmv.metal:405 --
    // so the load at qmv.metal:527 is not compiled into this instantiation.
    ("mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4", 2, "biases"),
    // `per_expert_scale`, read at `moe/route.metal:200` under `SCALED`, a
    // `bool` template parameter. The declaration at 88 carries the shader
    // author's own note: the slot is positional, so it "has to hold an
    // address whether or not `SCALED` dereferences it". This is what holding
    // one looks like when nobody dereferences it.
    ("router_topk_bfloat16", 3, "per_expert_scale"),
    // `sinks`, read only by the `_sink` instantiations. gpt-oss states
    // attention sinks and nothing else in `texts()` does. `WITH_SINK` is a
    // `bool` template parameter of all three families
    // (`sdpa_paged.metal:71`, `366`, `626`) and every one of the three loads
    // -- 258, 282, 612 -- sits inside `if (WITH_SINK)`.
    ("sdpa_paged_decode_bfloat16_d_64", 16, "sinks"),
    ("sdpa_paged_decode_bfloat16_d_128", 16, "sinks"),
    ("sdpa_paged_mma_bfloat16_d_64", 16, "sinks"),
    ("sdpa_paged_tiled_bfloat16_d_64", 16, "sinks"),
    ("sdpa_paged_tiled_bfloat16_d_128", 16, "sinks"),
];

/// A null bound where the kernel declares a parameter, held against the list
/// of the ones somebody has read the guard for.
///
/// The kind comparison above cannot see this. An `absent()` at a pointer slot
/// IS a pointer -- `Kind::Read` against `const device T*`, agreeing perfectly
/// -- and the thing that is wrong with it is not its shape but that it
/// addresses nothing. `kv_append_paged`'s buffer 15 is the failure this is
/// shaped around, and that one was only visible because the declaration
/// happened to be a SCALAR.
///
/// So the question is asked directly: where does a null land on something
/// declared? Eight places, all of them a compile-time flag away from being
/// read, and a ninth is a new one to go and read the guard for.
///
/// All four guards were read at their load sites rather than inferred from
/// the instantiation names, which is the difference between this list and a
/// plausible story: `bias` in particular is copied and offset before the flag
/// is checked, and a null that gets pointer arithmetic done to it looks
/// identical from the bind side to one that does not.
#[test]
fn a_null_lands_only_where_somebody_has_read_the_guard() {
    let shaders = shaders();
    let known: BTreeSet<(&str, usize)> = NIL_AT_A_DECLARED_SLOT
        .iter()
        .map(|(name, index, _)| (*name, *index))
        .collect();
    let mut found: BTreeSet<(String, usize)> = BTreeSet::new();
    for (file, entrypoint, args) in every_fire() {
        let Some(declared) = shaders.declaration(file, entrypoint) else { continue };
        for (index, value) in args.iter().enumerate() {
            if matches!(value, ArgValue::Shaped { handle: NIL, .. })
                && declared.contains_key(&index)
            {
                found.insert((entrypoint.to_string(), index));
            }
        }
    }
    let fresh: Vec<String> = found
        .iter()
        .filter(|(name, index)| !known.contains(&(name.as_str(), *index)))
        .map(|(name, index)| format!("{name} [[buffer({index})]]"))
        .collect();
    assert!(
        fresh.is_empty(),
        "a routine binds `ctx.absent()` at {} slot(s) the entrypoint DECLARES, and no entry says \
         why the kernel does not read them. A null is not a hole: find the guard that makes it \
         unread and put it in `NIL_AT_A_DECLARED_SLOT`, or source the argument:\n{}",
        fresh.len(),
        fresh.join("\n")
    );
    let stale: Vec<String> = NIL_AT_A_DECLARED_SLOT
        .iter()
        .filter(|(name, index, _)| !found.contains(&((*name).to_string(), *index)))
        .map(|(name, index, what)| format!("{name} [[buffer({index})]] {what}"))
        .collect();
    assert!(
        stale.is_empty(),
        "{} entr(y/ies) excuse a null nothing binds any more. An exception whose subject moved \
         goes on excusing whatever lands there next:\n{}",
        stale.len(),
        stale.join("\n")
    );
}


/// How a stamp's NAME spells the type its template is handed.
///
/// Three spellings, and the table is the measurement rather than a rule:
/// these are the ones 437 stamps actually use. A fourth arriving fails the
/// test below, which is the point -- whoever adds `_float16_` says here what
/// it means, once, instead of leaving the next reader to infer it from a
/// filename.
///
/// A spelling may name MORE THAN ONE type, which is why the right-hand side
/// is a list. `f32_bfloat16` is not a contradiction and not a third type: it
/// is a split-K pair, the model at bfloat16 and the partials that get summed
/// at f32. `affine_qmm_t_splitk<bfloat, float, ...>` passes both;
/// `affine_qmm_t_splitk_fp16_precast<float, ...>` passes only the partial and
/// writes `const device bfloat* scales` into its declaration; and
/// `qmm_splitk_reduce<bfloat, float>` passes both in the other order. One
/// name, three arities, and the only thing all three agree on is the SET.
///
/// Matching longest-first is what keeps `f32_bfloat16` from reading as plain
/// `bfloat16` and refusing the `float` beside it.
const SPELLINGS: &[(&str, &[&str])] = &[
    ("f32_bfloat16", &["bfloat", "float"]),
    ("bfloat16", &["bfloat"]),
    // `attn/split_qkv.metal` alone writes `instantiate_split_qkv(bf16, bfloat)`.
    // Same type, shorter name, and no reason beyond the file that wrote it.
    ("bf16", &["bfloat"]),
];

/// The CODEC policies a stamp may be handed, and the tokens its name owes.
///
/// `qmv.metal`'s tail and routed families take the codec as a type rather
/// than as `bits`: `affine_qmv_tail<AffineU8, bfloat, 64>` gets its unpacking,
/// its scale layout and its zero-point handling from the struct. So the name's
/// `_b_8` and the argument `AffineU8` are two statements of one fact, written
/// on the same macro line, and the failure when they disagree is not a
/// rounding difference -- a kernel handed `AffineU4` for a `_b_8` row reads
/// eight 4-bit weights out of a word that holds four 8-bit ones and decodes
/// the whole tensor as noise.
///
/// The tokens are what the name must CARRY, not its whole shape. `Mxfp4` owes
/// only its prefix: MXFP4 is 4-bit by construction and its `_b_4` is the same
/// fact a third time.
const CODECS: &[(&str, &[&str])] = &[
    ("AffineU4", &["affine", "b_4"]),
    ("AffineU8", &["affine", "b_8"]),
    ("Mxfp4", &["mxfp4"]),
];

/// Whether `literal` carries `token` as a whole `_`-delimited run.
fn carries(literal: &str, token: &str) -> bool {
    let mut at = 0usize;
    while let Some(hit) = literal[at..].find(token) {
        let start = at + hit;
        let end = start + token.len();
        let left = start == 0 || literal.as_bytes()[start - 1] == b'_';
        let right = end == literal.len() || literal.as_bytes()[end] == b'_';
        if left && right {
            return true;
        }
        at = end;
    }
    false
}

/// The templates that fix their element type instead of taking it.
///
/// `affine_qmm_t_fp16_precast<int group_size, ...>` starts its parameter list
/// at an `int`: the quantized weight is `uint32_t`, the scales and biases are
/// `bfloat`, the activation is `half` and the output is `bfloat`, all written
/// into the declaration. So there is no type argument to compare a name
/// against, and the name's `bfloat16` describes the OUTPUT rather than a
/// parameter anything passes.
///
/// This is not an excuse for a check that could not be made to work. It is
/// the boundary of what the shader tree states twice: a fact written once
/// cannot disagree with itself, and 33 of 470 stamps are in that position.
const TYPE_IS_NOT_AN_ARGUMENT: &[&str] = &[
    "affine_qmm_t_fp16_precast",
    "affine_qmm_t_bias_fp16_precast",
    "affine_qmm_t_residual_fp16_precast",
    "affine_qmm_t_strided_fp16_precast",
];

/// The types a stamp's name carries, by its longest spelling.
fn advertised(literal: &str) -> Option<&'static [&'static str]> {
    let mut best: Option<(usize, &'static [&'static str])> = None;
    for (token, ty) in SPELLINGS {
        if carries(literal, token) && best.is_none_or(|(n, _)| token.len() > n) {
            best = Some((token.len(), ty));
        }
    }
    best.map(|(_, ty)| ty)
}

/// **A stamp's name and the type it instantiates are the same type.**
///
/// This is the ELEMENT question, asked where the answer exists.
///
/// [`every_bound_argument_matches_the_buffer_it_lands_on`] cannot ask it. A
/// bind carries a handle, and `ArgValue` has no element: `const device
/// bfloat*` and `const device half*` are the same `Kind::Read` and the same
/// four-byte handle, so a routine that hands a bf16 allocation to a half
/// kernel binds something this file calls correct. Widening the value to
/// carry an element is a cross-cutting change to a type five backends'
/// machinery is generic over, and it has not been made.
///
/// But the shader tree states the element TWICE, and two statements of one
/// fact can be held against each other. `instantiate_qmm_t_splitk_fp16_precast
/// (f32_bfloat16, float, 16)` writes the type into the host name and passes
/// it as a template argument, and the two are written by hand, in a table of
/// forty rows, one row per line, differing in two columns. That is the exact
/// shape a copy-paste gets wrong.
///
/// What it would cost is not subtle. `affine_qmm_t_splitk_fp16_precast<P>`
/// declares `device P* y`, so a row that says `bfloat16` and passes `float`
/// produces a kernel writing FOUR bytes per element into an allocation the
/// driver sized for two. Not a wrong number -- a write past the end of the
/// output, into whatever the allocator put next.
#[test]
fn a_stamps_name_and_the_type_it_instantiates_are_the_same_type() {
    let shaders = shaders();
    let mut wrong: Vec<String> = Vec::new();
    let mut compared = 0usize;
    let mut untyped = 0usize;

    for ((file, literal), args) in &shaders.args {
        let passed: Vec<&str> = args
            .split(',')
            .map(str::trim)
            .filter(|a| {
                !a.is_empty()
                    && a.chars().all(|c| c.is_alphanumeric() || c == '_')
                    && !a.starts_with(|c: char| c.is_ascii_digit())
                    && *a != "true"
                    && *a != "false"
            })
            .collect();
        if passed.is_empty() {
            untyped += 1;
            let template = shaders.stamps.get(&(file.clone(), literal.clone()));
            let known = template.is_some_and(|t| TYPE_IS_NOT_AN_ARGUMENT.contains(&t.as_str()));
            if !known {
                wrong.push(format!(
                    "  {file} `{literal}`: instantiates {template:?} with no \
                     type argument, and that template is not one of the four \
                     that fix their element. Either it grew a type this \
                     cannot read, or the list above is short."
                ));
            }
            continue;
        }
        let Some(want) = advertised(literal) else {
            wrong.push(format!(
                "  {file} `{literal}`: passes {passed:?} and its name carries \
                 no spelling `SPELLINGS` knows. Add the spelling and say \
                 what it means."
            ));
            continue;
        };
        compared += 1;
        // CONTAINMENT, NOT EQUALITY, and the asymmetry is the templates'.
        // A type the name does not mention is a type nothing selects by, so
        // it is caught. A type the name mentions and the stamp does not pass
        // is `affine_qmm_t_splitk_fp16_precast`'s `bfloat`, written into the
        // declaration instead -- so the reverse direction would refuse a
        // correct row.
        for ty in &passed {
            if want.contains(ty) {
                continue;
            }
            if let Some((_, owed)) = CODECS.iter().find(|(c, _)| c == ty) {
                let short: Vec<&str> = owed.iter().copied().filter(|t| !carries(literal, t)).collect();
                if !short.is_empty() {
                    wrong.push(format!(
                        "  {file} `{literal}`: handed the codec `{ty}`, whose \
                         name owes {short:?}"
                    ));
                }
                continue;
            }
            wrong.push(format!(
                "  {file} `{literal}`: the name advertises {want:?} and \
                 the template is handed `{ty}`, which is neither one of \
                 those nor a codec `CODECS` knows"
            ));
        }
    }

    assert!(
        wrong.is_empty(),
        "{} stamp(s) name one type and instantiate another. Both halves are \
         written by hand on the same line; when they disagree the name is \
         what the driver picks by and the argument is what runs.\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    // A NUMBER AND AN ASSERTION. Both move only when the tree does, and a
    // comparison that quietly stops comparing is the failure this file's
    // `both_halves_of_the_comparison_are_there` exists to catch.
    assert_eq!(compared, 437, "the stamps whose name and type arguments agree");
    assert_eq!(untyped, 33, "the stamps whose template fixes its element");
    assert_eq!(compared + untyped, shaders.args.len(), "every stamp is one or the other");
}
