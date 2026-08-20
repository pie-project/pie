//! The table's product, against itself and against the shader paths.
//!
//! This is the check `.wiki/kernel-x/metal-refactor.md` §9 lists first, as
//! "unchanged in form", and states as:
//!
//! > every entrypoint in `kernels/` resolves to exactly one (row, axis point),
//! > and every (row, axis point) to exactly one entrypoint
//!
//! THE CITATION USED TO READ `.wiki/kernel-metal-refactor.md` §6, invariant
//! (1). There is no such file -- the surviving document is under `kernel-x/`
//! -- and neither the numbered form nor that wording is anywhere in `.wiki/`
//! today. Whatever numbered them is gone, and the sentence above survives
//! only here and in the two siblings that quote it. It is left standing
//! because it is still exactly what this file asserts; what is corrected is
//! the pointer, which named nothing and would have sent the next reader
//! looking for a document to reconcile against.
//!
//! It is not checked HERE, but it is checked again. The shader half of the
//! comparison needs a C preprocessor — the axis product lives in
//! `instantiate_*` macros and nothing else writes it down — so it used to
//! arrive as a committed `entrypoints.generated.txt` that
//! `scripts/metal-kernel-audit.py` wrote and this file diffed against the
//! table. That artifact was deleted, and for a while nothing compared the two
//! sets, in a test or out of one.
//!
//! `tests/dispatch_matches_the_shader.rs` carries a minimal preprocessor now,
//! written for a different question, and the set comparison came back with it
//! in full: 481 names on each side and the same 481, matched EXACTLY and not
//! by prefix, with no exception on either. So both halves are hermetic in
//! `cargo test` again — the expensive one, a row whose axes over-generate and
//! surface as a nil pipeline partway through somebody's generate, and the
//! cheap one, a compiled kernel no row can dispatch.
//!
//! The Vulkan and WGSL siblings never lost their halves, and the difference is
//! the shading language rather than the effort: there a variant is DECLARED on
//! a `// pie:instantiate` line, so the set is a parse. Here it is a macro
//! expansion, which is why the thing that reads it is three hundred lines in
//! another file.
//!
//! What holds below is everything that reads the table, the routine name
//! tables, and the shader tree's file names: a typo in a hand-written routine
//! spelling is red here rather than a nil pipeline on a device.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// Every entrypoint the TABLE declares — its axis product, expanded.
///
/// This was the shader tree's set, read from `entrypoints.generated.txt`. The
/// two are held equal again, by
/// `dispatch_matches_the_shader::every_entrypoint_the_table_declares_is_one_the_shader_tree_writes`
/// — so the readers below that ask "is this name real?" mean the shaders too,
/// but they mean it by way of that file and not by anything asserted here.
fn from_the_table() -> BTreeSet<String> {
    kernels_metal::entrypoints().into_iter().collect()
}

// The shader-vs-table comparison lived here, reading a committed census, and
// went when the census did. It lives in `dispatch_matches_the_shader.rs` now,
// reading the shader tree directly: the shader set is an `instantiate_*`
// expansion and a preprocessor is the only thing that produces it, so the
// comparison follows the preprocessor rather than the other way round.
//
// A `//` tombstone and not a `///` one: it documents nothing, and as a doc
// comment it attached itself to the item below and read as that item's
// description.

/// Two families claiming one entrypoint would make the census a set of 480
/// while ten lists sum to 481, and [`from_the_table`] cannot see it: a
/// duplicate is absorbed by the set it builds.
///
/// This asked it of ROWS, and `sig_in` was the reason -- two rows over one
/// name made the lookup order-dependent. Every row is retired and the lookup
/// is `driver-metal`'s `crossed`, which resolves by STEM and cannot be
/// ambiguous by construction (the longest wins). What survives is the census
/// question: a name stated twice is a shader counted twice, and the 481
/// below would then be describing 480 shaders.
#[test]
fn no_two_files_stamp_the_same_entrypoint() {
    // OVER THE EXPANSION, not over ten hand-written tables. The tables are
    // gone; `build.rs` reads the `instantiate_*` macros, so what this holds up
    // is a property of the SHADER TREE rather than of a list somebody kept.
    //
    // Two files stamping one name is a pipeline whose source depends on which
    // file the driver happened to open, which is a silent wrong kernel.
    let mut seen: std::collections::BTreeMap<&str, &str> = Default::default();
    for (file, name) in kernels_metal::STAMPED {
        if let Some(first) = seen.insert(name, file) {
            panic!("`{name}` is stamped by both `{first}` and `{file}`");
        }
    }
    assert_eq!(seen.len(), kernels_metal::STAMPED.len());
}

/// The row count is load-bearing prose in three documents, so it is pinned
/// rather than described. Change it here when a kernel is added, deliberately.
///
/// It has earned its keep once already: 99/480 became 98/479 when the census
/// learned that a wrapped `template` parameter list still declares a template,
/// so `affine_qmm_t_aligned` was a BODY and never a dispatchable name. The set
/// comparison above passed either way — it compares the table to whatever the
/// census says — and this is the assertion that made the correction visible.
///
/// Back to 99/480 deliberately: `split_qkv_bf16` is a NEW kernel, written
/// because the Metal text names a QKV split and CUDA's answer to that —
/// a kernel the driver launches that no text has to name — is the category
/// this backend refuses to grow.
///
/// 100/481 deliberately: `add_bias` is a NEW kernel on both this side and the
/// Vulkan one, added in the same diff. The Qwen-2 family carries q/k/v
/// projection biases, `LlamaLikeFacts::qkv_bias` has always said so, and the
/// shared Metal text omitted the op for one reason only -- no Metal kernel
/// added a bias, so there was no symbol to name. That is a wrong ANSWER rather
/// than a missing kernel: the biases are small, the text stays fluent without
/// them, and nothing downstream can tell.
///
/// It held 100/481 against a change that was RIGHT and still had to come out.
/// `sdpa_paged_mma` gained a `_d_128` point: the shader was always written for
/// the width, and what had kept it uninstantiated was a comment pricing three
/// tiles at `KT=64` (40 KB, over budget) when the file instantiates `KT=16`
/// (16 KB). The device agreed -- the pipeline builds. What this number then
/// said is the part that was not obvious: the list is not this backend's
/// alone. `kernels-vulkan` and `kernels-wgpu` pin the same 100/481 in tests of
/// their own, so one width added here is three shaders and three tables, and
/// the siblings are Slang and WGSL. The instantiation was reverted and the
/// reasoning left in the shader.
///
/// Those three numbers are now the whole of the cross-backend claim. The
/// entrypoint-for-entrypoint diff that backed it read the three crates'
/// committed censuses, and they are deleted.
///
/// # 100 became 101, and the three backends no longer agree
///
/// `rms_rope` -- the fused per-head norm and NEOX rotation -- is a Vulkan
/// kernel. It is named here because Vulkan consumes the metal-flavoured plan
/// text and `model-ir` resolves every launched symbol through THIS crate's
/// census, and it is named here ONLY: there is no `ENTRYPOINTS` row and no
/// `.metal` body, so `entrypoints()` is still 481. `kernels-wgpu` is
/// untouched and still pins 100.
///
/// So the claim this file has always made -- three backends, one hundred
/// kernels -- is now 101/101/100, and that is not a bookkeeping slip to be
/// tidied away by adding the name to wgpu too. Adding a name to a backend
/// that has no shader behind it is exactly the silence these counts exist to
/// break. The honest state is that one backend has a kernel the other two do
/// not, and the honest way to close it is to write the kernel twice more, not
/// to write the name twice more.
#[test]
fn the_table_is_one_hundred_and_one_kernels_over_four_hundred_and_eighty_one_entrypoints() {
    // Rows PLUS retired, because the hundred is a claim about the shader tree
    // and rows are no longer the only thing that names it. A family that
    // crosses moves its names from the left term to the right and the sum is
    // unchanged -- which is what makes this the line that catches a row
    // deleted before its routine lands, rather than a line that has to be
    // edited every time one does.
    assert_eq!(
        kernels_metal::KERNELS.len() + kernels_metal::retired_rows().len(),
        101
    );
    assert!(kernels_metal::KERNELS.is_empty(), "every family crossed");
    assert_eq!(kernels_metal::entrypoints().len(), 481);
}

/// Every entrypoint the census names is one that has RETIRED its row.
///
/// This asked `kernels::sig_in` to resolve each of the 481 against `KERNELS`,
/// which was the lookup `model-ir` used. That table is empty: every family
/// crossed, and the resolver is `driver-metal`'s `crossed`, which matches the
/// stem the routine registry states and lives in a crate this one cannot
/// call. Asking `sig_in` here now would pass by resolving nothing.
///
/// So it asks the question that is still answerable from this crate, and it
/// is the one that catches the failure the original was built for: a name in
/// the census with no home. Before, a name no row declared was a shader the
/// table had lost; now it is a shader whose family neither kept a row nor
/// listed it as retired, and either way it is a name that reaches no
/// dispatcher.
#[test]
fn every_entrypoint_the_census_names_belongs_to_a_family_that_retired_it() {
    let retired: std::collections::BTreeSet<&str> = kernels_metal::retired().into_iter().collect();
    for name in from_the_table() {
        assert!(
            retired.contains(name.as_str()),
            "`{name}` is in the census and no family's retired list names it"
        );
    }
    assert_eq!(retired.len(), 481, "the whole census is retired");
}

/// Every entrypoint a routine can NAME is one the table declares.
///
/// A routine picks its spelling from a table -- `moe.rs` carries seventy-two
/// of them across three tilings -- and a name that is not there is not an
/// error at the call. `newFunctionWithName:` returns nil at run time, inside a
/// fire, after the plan was accepted and the pipelines batch-compiled. So the
/// sweep belongs here, where a typo is red before it is a fault.
///
/// The three tables are swept whole rather than sampled: they are written out
/// by hand precisely because a name assembled from a template is the defect
/// this plane forbids, and a hand-written list is a list with typos in it
/// until something reads every line.
///
/// The set swept against was the shader census; it is the KERNELS table now.
/// For a typo in the routine tables — which is what this catches — the two are
/// interchangeable, since the names come from a different hand either way.
#[test]
fn every_entrypoint_a_routed_matmul_can_name_is_one_the_table_carries() {
    let have = from_the_table();
    let mut swept = 0usize;
    for (group, bits) in [(32, 4), (32, 8), (64, 4), (64, 8), (128, 4), (128, 8)] {
        for m in [16, 32, 64] {
            for n in [16, 32, 64] {
                let name =
                    format!("affine_qmm_t_routed_bfloat16_gs_{group}_b_{bits}_bm_{m}_bn_{n}");
                assert!(have.contains(&name), "`{name}` is named and not compiled");
                swept += 1;
            }
        }
    }
    for m in [16, 32, 64] {
        for n in [16, 32, 64] {
            for name in [
                format!("affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_{m}_bn_{n}"),
                format!("mxfp4_qmm_t_routed_bias_bfloat16_bm_{m}_bn_{n}"),
            ] {
                assert!(have.contains(&name), "`{name}` is named and not compiled");
                swept += 1;
            }
        }
    }
    assert_eq!(swept, 72, "fifty-four affine, nine pre-cast, nine MXFP4");
}

/// Every shader path any routine in this crate spells names a file that
/// exists.
///
/// Metal answers `newLibraryWithSource:`/`newFunctionWithName:` for a module
/// it does not have with **nil**, not with an error, and the routines are
/// batch-compiled after a plan has already been accepted -- so a misspelled
/// path surfaces as a pipeline that is silently absent at encode time, on a
/// device, far from the line that wrote it. `attn.rs` named
/// `"attn/softcap.metal"` for the whole of its first hour; the file is
/// `attn/logit_softcap.metal`.
///
/// This reads the SOURCE rather than the routine table because a `Fire` is
/// built inside a dispatch: nothing can enumerate every one of them without
/// calling every routine with arguments it would have to invent.
#[test]
fn every_shader_path_the_routines_spell_is_a_file_on_disk() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut checked = 0usize;
    let mut missing: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(root.join("src")).expect("a src directory") {
        let path = entry.expect("an entry").path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("a readable module");
        for (n, line) in src.lines().enumerate() {
            // `file: X.metal"` and `const Y_FILE: &str = "X.metal"` alike:
            // any string literal in this crate ending in `.metal` is a path
            // handed to the shader loader.
            let Some(close) = line.find(".metal\"") else {
                continue;
            };
            let Some(open) = line[..close].rfind('"') else {
                continue;
            };
            let named = &line[open + 1..close + ".metal".len()];
            checked += 1;
            if !root.join("kernels").join(named).is_file() {
                let file = path.file_name().expect("a name").to_string_lossy();
                missing.push(format!("{file}:{}: {named}", n + 1));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "these shader paths name no file under kernels/: {missing:#?}"
    );
    // 40, THEN 481, NOW 36, AND THE TRIP DOWN IS THE POINT. It was 40 against
    // a hundred rows each stating one `file`; it went to 481 when `ENTRYPOINTS`
    // began stating a path per INSTANTIATED name; and it is 36 now that
    // `ENTRYPOINTS` is gone -- the shaders say what they stamp, so a Rust table
    // repeating it was 481 chances to disagree. What is left is one literal per
    // shader FILE, which is the only part a routine actually spells.
    //
    // A floor of 400 survived that deletion by three commits, green the whole
    // time it was measuring nothing, and it is an equality now for exactly that
    // reason. A path that names no file is still what this test is for --
    // `device_kernels.rs` dereferences every one, and a wrong one is a pipeline
    // silently absent on a device.
    assert_eq!(
        checked, 36,
        "the `.metal` literals this crate spells; a change here is a shader \
         file added or removed, and the list above is what checks them"
    );
}

/// Every entrypoint the attention routines can name is one the table declares.
///
/// The same sweep `moe.rs`'s routed matmuls get, and for the same reason: the
/// tables are written by hand because `Fire::entrypoint` is a `&'static str`,
/// and a name assembled from a template is the defect this whole plane exists
/// to prevent. A typo here is a nil pipeline on a device.
#[test]
fn every_entrypoint_an_attention_routine_can_name_is_one_the_table_declares() {
    let have = from_the_table();
    let named: Vec<&str> = kernels_metal::attn::PAGED_DECODE
        .iter()
        .chain(kernels_metal::attn::PAGED_DECODE_SINK.iter())
        .chain(kernels_metal::attn::PAGED_TILED.iter())
        .chain(kernels_metal::attn::PAGED_TILED_SINK.iter())
        .chain(kernels_metal::attn::PAGED_TILED_STRIDED.iter())
        .chain(kernels_metal::attn::PAGED_MMA.iter())
        .chain(kernels_metal::attn::PAGED_MMA_SINK.iter())
        .chain(kernels_metal::attn::VECTOR_DECODE.iter())
        .chain(kernels_metal::attn::VECTOR_SWA.iter())
        .chain(kernels_metal::attn::VECTOR_SINK.iter())
        .copied()
        .collect();
    let absent: Vec<&&str> = named.iter().filter(|n| !have.contains(**n)).collect();
    assert!(
        absent.is_empty(),
        "these attention entrypoints are named by a routine and instantiated \
         by no shader: {absent:#?}"
    );
    assert_eq!(named.len(), 19, "every table above was swept");
}

/// Every entrypoint a quantised projection can name is one the table declares.
///
/// 303 names across nineteen tables, and they are written out because
/// `Fire::entrypoint` is a `&'static str`. Assembling them -- `format!("{}_gs_
/// {group}_b_{bits}", stem)` -- is the defect this plane exists to prevent,
/// and it is worse here than anywhere: g64/b8 and g128/b4 pack to identical
/// SHAPES, so a module chosen for the wrong pair unpacks fluent nonsense
/// instead of failing.
#[test]
fn every_entrypoint_a_quantised_projection_can_name_is_one_the_tree_stamps() {
    // THE TABLES IT WALKED ARE GONE, AND SO IS THE INDIRECTION. It chained
    // nineteen `[&str; N]`s -- the axis product, written out -- and compared
    // them to the census. `quant::composable` runs the same axes through the
    // composers a fire calls, so what is compared is what a fire can reach.
    //
    // `PIE_GROUP` and `PIE_BITS` are a COORDINATE and not a label: g64/b8 and
    // g128/b4 pack to identical SHAPES, so a module chosen for the wrong pair
    // unpacks fluent nonsense instead of failing. That is what makes this
    // sweep worth its breadth.
    let have = from_the_table();
    let missing: Vec<&str> = kernels_metal::quant::composable()
        .into_iter()
        .filter(|name| !have.contains(*name))
        .collect();
    assert!(
        missing.is_empty(),
        "{} composable projection name(s) are not in the census:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

/// This crate's Rust modules, where routine bodies and census lists both live.
fn module_sources() -> Vec<PathBuf> {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&src).expect("a src directory") {
        let path = entry.expect("a readable entry").path();
        if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    out.sort();
    assert!(
        out.len() > 5,
        "expected this crate's modules, found {out:?}"
    );
    out
}

/// Every entrypoint AXIS in this crate, by name: `PAGED_DECODE`, `AFFINE_QMM`,
/// `MXFP4_QMM` and the thirty-odd others the bodies index.
///
/// A declaration qualifies when its type names `&str` and is a list. The
/// census lists are `&[(&str, &str)]` and are excluded by the parenthesis:
/// they are what the axes are compared AGAINST, and reading them as axes would
/// make the comparison compare a list to itself.
///
/// Both spellings are read, because both are in the tree: `[&str; 1]` on one
/// line (`PAGED_TILED_SINK`) and `&[&str]` over many (`AFFINE_QMM`). The wgpu
/// sibling reads only the second and would silently return an EMPTY list for
/// the first, which is the worse failure — an empty axis fires nothing and
/// agrees with everything.
fn entrypoint_tables() -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let push = |out: &mut BTreeMap<String, Vec<String>>,
                open: &mut Option<(String, Vec<String>)>,
                text: &str| {
        let (name, mut acc) = open.take().expect("an open declaration");
        acc.extend(text.split('"').skip(1).step_by(2).map(str::to_owned));
        if text.trim_end().ends_with("];") {
            out.insert(name, acc);
        } else {
            *open = Some((name, acc));
        }
    };
    for path in module_sources() {
        let text = std::fs::read_to_string(&path).expect("a readable module");
        let mut open: Option<(String, Vec<String>)> = None;
        for line in text.lines() {
            let code = line.split_once("//").map_or(line, |(before, _)| before);
            if open.is_some() {
                push(&mut out, &mut open, code);
                continue;
            }
            let trimmed = code.trim();
            let trimmed = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
            let Some(rest) = trimmed
                .strip_prefix("static ")
                .or_else(|| trimmed.strip_prefix("const "))
            else {
                continue;
            };
            let Some((name, tail)) = rest.split_once(':') else {
                continue;
            };
            let Some((ty, init)) = tail.split_once('=') else {
                continue;
            };
            if !ty.contains("&str") || !ty.contains('[') || ty.contains('(') {
                continue;
            }
            open = Some((name.trim().to_owned(), Vec::new()));
            push(&mut out, &mut open, init);
        }
    }
    // NINETEEN, AND IT WAS A FLOOR OF TWENTY. The 19 lattice tables that used
    // to spell one entrypoint per axis product are retired -- a name is its
    // axis values, so `qmm_name` builds it -- and what is scraped here now is
    // the axes THEMSELVES, one array per axis rather than one per product. The
    // floor outlived its subject by a hair: twenty tables became nineteen and
    // the assertion that was meant to catch a scraper going blind caught the
    // deletion instead.
    assert_eq!(
        out.len(),
        19,
        "expected this crate's entrypoint axes, found {:?}",
        out.keys().collect::<Vec<_>>(),
    );
    out
}

/// The entrypoints the second argument of one `Fire::at(file, entrypoint)`
/// call can name, or `None` if no static reading can tell.
///
/// Three shapes, and the plane forbids a fourth: a literal names itself,
/// `TABLE[point(..)]` names every string in `TABLE` because the index picks a
/// head dimension or a quantisation point and every point on the axis is a
/// real entrypoint, and a COMPOSER call names every point of its lattice.
///
/// # Why a composer is read by the crate and not by this scanner
///
/// The nineteen lattice tables are gone: a name is its axis values, so
/// `qmm_name("", *group, *bits, *bm, *bn)?` builds it. That is not the defect
/// this plane exists to prevent -- it is a `&'static str` from
/// `kernels::jit::symbol`, chosen from an axis each of whose points is
/// checked, not a `format!` a caller can spell anything into.
///
/// It is also not something to teach a text scanner, and `quant.rs`'s own
/// header says why: expanding the call here "would put the lattice into the
/// scanner -- a second place the axes are written, which is the thing the
/// tables were deleted for". So the crate answers, in `composable()`, by
/// running the composers over the axes; `composed_names_are_stamped` holds
/// that product against `STAMPED`; and what is left for this file is to
/// notice the shape and count it, so a composer arriving where nothing checks
/// it is still a failure here.
fn fired_names(value: &str, tables: &BTreeMap<String, Vec<String>>) -> Option<Vec<String>> {
    if let Some(rest) = value.strip_prefix('"') {
        return Some(vec![rest.split('"').next()?.to_owned()]);
    }
    if composer(value) {
        return Some(Vec::new());
    }
    let (name, _) = value.split_once('[')?;
    tables.get(name.trim()).cloned()
}

/// Whether this argument is a call to one of `quant.rs`'s name composers.
///
/// By NAME, and the four are written out rather than matched on a `_name`
/// suffix, because the excuse is not "it looks like a composer" -- it is
/// "`composable()` walks this one's whole lattice". A fifth composer that
/// `composable()` does not walk has to be added in both places or it fails
/// here, which is the coupling worth having.
fn composer(value: &str) -> bool {
    ["qmm_name(", "qmm_precast_name(", "qmv_name(", "qmv_wide_strided_name("]
        .iter()
        .any(|c| value.starts_with(c))
}

/// Every `Fire::at(file, entrypoint)` call read out of source, with the line
/// each opens on and the RAW TEXT of its second argument.
///
/// A body now spells its point as `Fire::at(file, entrypoint).apply(geometry)`
/// — a CALL, not the `Fire { file: .., entrypoint: .., .. }` struct literal
/// (nor an `entrypoint: ..,` labelled line) this scan used to anchor on: `rg
/// 'entrypoint:' crates/kernels-metal/src` is empty over the whole crate,
/// every family having crossed to the call form. So the region tracked is the
/// call's own parentheses, counted from `Fire::at(`'s own opening one, and
/// the two arguments split on the first comma AT THAT DEPTH — so a comma
/// inside `head_point(*head_dim, &PAGED_DIMS)`, itself an argument, does not
/// end the first one early. `kernels-wgpu`'s sibling test solved the same
/// problem first; this is that solution, trimmed to what this file needs (no
/// `.apply(..)` geometry is read here, so there is nothing to swallow).
fn fire_entrypoints(text: &str) -> Vec<(usize, String)> {
    let code: String = text
        .lines()
        .map(|l| l.split_once("//").map_or(l, |(before, _)| before))
        .collect::<Vec<_>>()
        .join("\n");
    let bytes = code.as_bytes();

    /// The index one past the delimiter matching `bytes[open]`, itself one of
    /// `( [ {`. A quoted string's own brackets do not count, so a table name
    /// or an entrypoint literal holding one — none do today, but the scan
    /// should not break the day one needs a `[` in its name — is inert.
    fn matching_close(bytes: &[u8], open: usize) -> usize {
        let mut depth = 1i32;
        let mut in_str = false;
        let mut i = open + 1;
        while depth > 0 {
            match bytes[i] {
                b'"' => in_str = !in_str,
                b'(' | b'[' | b'{' if !in_str => depth += 1,
                b')' | b']' | b'}' if !in_str => depth -= 1,
                _ => {}
            }
            i += 1;
        }
        i
    }

    /// The first comma this argument list holds at ITS OWN depth, i.e. not
    /// inside a nested call or index.
    fn top_level_comma(bytes: &[u8]) -> Option<usize> {
        let mut depth = 0i32;
        let mut in_str = false;
        for (i, &b) in bytes.iter().enumerate() {
            match b {
                b'"' => in_str = !in_str,
                b'(' | b'[' | b'{' if !in_str => depth += 1,
                b')' | b']' | b'}' if !in_str => depth -= 1,
                b',' if !in_str && depth == 0 => return Some(i),
                _ => {}
            }
        }
        None
    }

    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(rel) = code[from..].find("Fire::at(") {
        let start = from + rel;
        let open = start + "Fire::at".len();
        let close = matching_close(bytes, open);
        let args = &code[open + 1..close - 1];
        let entrypoint = match top_level_comma(args.as_bytes()) {
            Some(c) => args[c + 1..].trim(),
            None => args.trim(),
        };
        let line = code[..start].matches('\n').count() + 1;
        out.push((line, entrypoint.to_owned()));
        from = close;
    }
    out
}

/// Census names that no body in this crate fires.
///
/// Each is compiled by `device_kernels.rs` and dispatched by nothing, which is
/// a real cost — a pipeline built per device for a shader that never runs —
/// and each is deliberate. All four are explained where they are:
///
/// - `sdpa_paged_decode_bfloat16_d_{64,128}_p32` and `_d_64_p32_sg8` set
///   `FAST_FULL`, which deletes the window and all three mask operands from
///   the body; `attn.rs`'s [`PAGED_DECODE`] omits them and a test beside it
///   asserts the omission, so a fire that reached one would pass masks to a
///   kernel that has none.
/// - `silu_mul_strided_bfloat16` is the one of `mlp.rs`'s five that did not
///   cross. The reason was once the argument vocabulary — it declares
///   `row_pitch` at buffer 4 with buffer 3 empty, and a positional list could
///   not express a hole — and `pad` answered that; twenty-one routines now bind
///   an address at an index their shader does not declare. What keeps it dark
///   is that no text names it and no statement produces a row pitch, which is
///   what `driver-metal`'s `DARK` says.
///
/// The list being SHORT is the point. Every name on it is a shader compiled on
/// every device for nothing, so it should shrink as the vocabulary grows, and
/// a name arriving here without a reason written down is a routine that was
/// deleted while its census row stayed.
const UNFIRED: &[&str] = &[
    "sdpa_paged_decode_bfloat16_d_128_p32",
    "sdpa_paged_decode_bfloat16_d_64_p32",
    "sdpa_paged_decode_bfloat16_d_64_p32_sg8",
    "silu_mul_strided_bfloat16",
];

/// A retired family's stated census is exactly what its bodies FIRE.
///
/// Every row is retired, so `ENTRYPOINTS` is no longer a check on the table —
/// it IS the table. `entrypoints()` returns nothing else, `device_kernels.rs`
/// compiles what it returns, and `driver-metal` resolves stems against it. A
/// row could not drift this way: its `axes` GENERATED its entrypoints. Ten
/// hand-written lists totalling 481 lines can, and every sweep keyed on
/// `entrypoints()` would follow the drift rather than fail on it.
///
/// The sibling tests above ask a related but different question: they rebuild
/// an axis product with `format!` in the test and check the census carries it,
/// which catches a census missing a point. This reads what the bodies LITERALLY
/// SPELL, which catches the other three ways the pair can part — a typo in a
/// census line, a routine firing a name nothing compiles, and an axis entry
/// deleted from under a fire.
#[test]
fn every_entrypoint_a_body_fires_is_one_the_tree_stamps() {
    // THE TEN HAND-WRITTEN CENSUSES ARE GONE, and with them the drift this
    // test was written to catch. It compared each family's `ENTRYPOINTS` list
    // against the `Fire::at` calls in the same file, because "a row could not
    // drift this way -- its `axes` GENERATED its entrypoints. Ten hand-written
    // lists totalling 481 lines can."
    //
    // `build.rs` expands the shader tree's `instantiate_*` macros, so the
    // census generates again -- from the `.metal` files rather than from a
    // row -- and the drift it named is not reachable. What is still worth
    // holding is the OTHER half: a body firing a name no shader stamps, which
    // `newFunctionWithName:` answers nil for on a device rather than failing
    // here.
    //
    // Per module still, so a failure names the file to open.
    let stamped: BTreeSet<&str> = kernels_metal::STAMPED.iter().map(|(_, n)| *n).collect();
    let tables = entrypoint_tables();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut swept = 0usize;
    let mut composed = 0usize;
    for module in [
        "attn.rs", "layout.rs", "mlp.rs", "moe.rs", "norm.rs", "ptir.rs", "quant.rs",
        "rope.rs", "sample.rs", "ssm.rs",
    ] {
        let path = root.join(module);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let mut fired: BTreeSet<String> = BTreeSet::new();
        for (line, value) in fire_entrypoints(&text) {
            if composer(&value) {
                composed += 1;
            }
            match fired_names(&value, &tables) {
                Some(names) => fired.extend(names),
                None => panic!(
                    "{module}:{line}: `{value}` is neither a literal nor an \
                     index into an axis of literals, so no static reading of \
                     this crate can say which entrypoints it fires",
                ),
            }
        }
        let uncompiled: Vec<&String> =
            fired.iter().filter(|n| !stamped.contains(n.as_str())).collect();
        assert!(
            uncompiled.is_empty(),
            "`{module}` fires {} entrypoint(s) no `.metal` file stamps, and \
             `newFunctionWithName:` answers nil for those on a device rather \
             than failing here: {uncompiled:#?}",
            uncompiled.len(),
        );
        swept += fired.len();
    }
    // A vacuity guard. A `fire_entrypoints` that stopped matching would sweep
    // nothing and agree with everything, which is the failure mode this file's
    // own header warns about twice.
    assert!(
        swept > 100,
        "only {swept} fired entrypoint(s) were read out of ten modules; the \
         scan has stopped matching",
    );
    // A NUMBER AND AN ASSERTION, for the shape this file does not read
    // itself. Nineteen fires spell a composed name, `composable()` walks
    // 291 of them, and `composed_names_are_stamped` is where they are held.
    // The count moving is a composer added or removed, and either way the
    // question is whether that other file walked it.
    assert_eq!(
        composed, 19,
        "the fires whose name is composed rather than spelled; \
         `composable()` is what answers for these and \
         `composed_names_are_stamped` is what checks it"
    );
}

/// **A stamped entrypoint that no body fires is a kernel nothing can reach.**
///
/// The other direction, and it went dark when the ten hand-written censuses
/// did. [`UNFIRED`] outlived the test that read it -- `cargo clippy` called it
/// dead code, which is the honest description of an excuse list nothing
/// consults.
///
/// What made it hard to bring back is that a body no longer spells most of
/// its names: 291 of the 481 come out of `qmm_name` and its three siblings,
/// so reading the sources alone answers 186 and calls the rest unreachable.
/// The crate's own `composable()` is the missing term -- the same function
/// `composed_names_are_stamped` uses, walking the same axes a fire walks --
/// and the union of "what the modules spell" with "what the composers reach"
/// is 477 of 481, exactly.
///
/// Exactly the four [`UNFIRED`] already named, with nothing new hiding behind
/// them. Three are `sdpa_paged_decode`'s `_p32` points and one is
/// `silu_mul_strided`; see that constant for what each is waiting on.
///
/// This is not the same question as
/// `what_is_stamped_beyond_the_composed_family_is_a_known_number`, which pins
/// 190 as a COUNT. A count moving tells you something changed. This says
/// which shader compiles for nobody.
#[test]
fn a_stamped_entrypoint_no_body_fires_is_a_kernel_nothing_can_reach() {
    let stamped: BTreeSet<&str> = kernels_metal::STAMPED.iter().map(|(_, n)| *n).collect();
    let tables = entrypoint_tables();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut fired: BTreeSet<String> = BTreeSet::new();
    for module in [
        "attn.rs", "layout.rs", "mlp.rs", "moe.rs", "norm.rs", "ptir.rs", "quant.rs",
        "rope.rs", "sample.rs", "ssm.rs",
    ] {
        let path = root.join(module);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        for (_, value) in fire_entrypoints(&text) {
            if let Some(names) = fired_names(&value, &tables) {
                fired.extend(names);
            }
        }
    }
    // THE COMPOSED TERM. Without it this reports 291 dead kernels, every one
    // of them fired by a `qmm_name` call the scanner deliberately does not
    // expand. `composable()` is the crate answering for itself, which is the
    // same arrangement `fired_names` documents.
    fired.extend(kernels_metal::quant::composable().into_iter().map(str::to_owned));

    let dead: Vec<&str> = stamped
        .iter()
        .copied()
        .filter(|n| !fired.contains(*n) && !UNFIRED.contains(n))
        .collect();
    assert!(
        dead.is_empty(),
        "{} entrypoint(s) are stamped by a shader and fired by no body. Each \
         is a translation unit this backend compiles on every device for \
         nobody, and a name that reaches no dispatcher is the half of a typo \
         `newFunctionWithName:` cannot report.\n{dead:#?}",
        dead.len(),
    );
    // AND THE EXCUSE LIST DOES NOT OUTLIVE ITS SUBJECT, which is the failure
    // that put `UNFIRED` in front of clippy in the first place.
    let stale: Vec<&&str> = UNFIRED
        .iter()
        .filter(|n| fired.contains(**n) || !stamped.contains(*n))
        .collect();
    assert!(
        stale.is_empty(),
        "{} name(s) are argued unfired and are either fired now or stamped by \
         nothing: {stale:#?}",
        stale.len(),
    );
    // A NUMBER AND AN ASSERTION.
    assert_eq!(fired.len(), 477, "the names the modules spell and the composers reach");
    assert_eq!(stamped.len(), 481, "the names the shader tree stamps");
}
