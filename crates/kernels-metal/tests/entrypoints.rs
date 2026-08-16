//! The table's product, against itself and against the shader paths.
//!
//! This is invariant (1) of `.wiki/kernel-metal-refactor.md` §6:
//!
//! > every entrypoint in `kernels/` resolves to exactly one (row, axis point),
//! > and every (row, axis point) to exactly one entrypoint
//!
//! It is no longer checked here. The shader half of the comparison needed a C
//! preprocessor — the axis product lives in `instantiate_*` macros and nothing
//! else writes it down — so it arrived as a committed
//! `entrypoints.generated.txt` that `scripts/metal-kernel-audit.py` wrote and
//! this file diffed against the table. That artifact is deleted, and with it
//! the only hermetic view a `cargo test` had of what the shaders instantiate.
//!
//! The Vulkan and WGSL siblings kept their halves of the same invariant, and
//! the difference is the shading language rather than the effort: there a
//! variant is DECLARED on a `// pie:instantiate` line, so the set is a parse.
//! Here it is a macro expansion. `scripts/metal-kernel-audit.py --table` was
//! the way to compare the two sets, and it is retired — it read the table's
//! half by running `examples/entrypoints.rs`, which is deleted with the rest
//! of `examples/`. Nothing compares them now, in a test or out of one.
//!
//! What still holds below is everything that reads the table, the routine name
//! tables, and the shader tree's file names: a typo in a hand-written routine
//! spelling is still red here rather than a nil pipeline on a device. What is
//! NOT held is the set itself — a shader instantiating a name no row declares,
//! or a row whose axes over-generate one no shader stamps, is green
//! everywhere.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// Every entrypoint the TABLE declares — its axis product, expanded.
///
/// This was the shader tree's set, read from `entrypoints.generated.txt`. The
/// two were held equal, so the readers below that ask "is this name real?" are
/// unchanged in what they accept; what is gone is the test that made them
/// equal, and no assertion here should be read as checking the shaders again.
fn from_the_table() -> BTreeSet<String> {
    kernels_metal::entrypoints().into_iter().collect()
}

/// The shader-vs-table comparison lived here: every entrypoint the census
/// listed had to be one a row declares, and every one a row declares had to be
/// in the census.
///
/// It is deleted with the census it read. `scripts/metal-kernel-audit.py
/// --table` performs the same comparison, which is where it has to live now —
/// the shader set is an `instantiate_*` expansion and a preprocessor is the
/// only thing that produces it.

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
fn no_two_families_claim_the_same_entrypoint() {
    let mut seen: std::collections::BTreeSet<&str> = Default::default();
    let mut total = 0usize;
    for family in kernels_metal::RETIRED {
        for (file, name) in *family {
            total += 1;
            assert!(
                seen.insert(name),
                "`{name}` is claimed by two families; this one says {file}"
            );
        }
    }
    assert_eq!(total, seen.len());
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
#[test]
fn the_table_is_one_hundred_kernels_over_four_hundred_and_eighty_one_entrypoints() {
    // Rows PLUS retired, because the hundred is a claim about the shader tree
    // and rows are no longer the only thing that names it. A family that
    // crosses moves its names from the left term to the right and the sum is
    // unchanged -- which is what makes this the line that catches a row
    // deleted before its routine lands, rather than a line that has to be
    // edited every time one does.
    assert_eq!(
        kernels_metal::KERNELS.len() + kernels_metal::retired_rows().len(),
        100
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
    // Was 40, against a hundred rows each stating one `file`. The rows are
    // retired and `ENTRYPOINTS` states a path per INSTANTIATED name, so the
    // same scan now reaches all 481 and this floor moved with it. That is
    // not inflation: every one of those paths is dereferenced by
    // `device_kernels.rs`, and a wrong one there is a pipeline that is
    // silently absent on a device.
    assert!(
        checked > 400,
        "only {checked} shader paths were found, which means the scan stopped \
         seeing them rather than that the crate stopped naming them"
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
fn every_entrypoint_a_quantised_projection_can_name_is_one_the_table_declares() {
    use kernels_metal::quant::*;
    let have = from_the_table();
    let named: Vec<&str> = QMM_T
        .iter()
        .chain(QMM_T_BIAS.iter())
        .chain(QMM_T_RESIDUAL.iter())
        .chain(QMM_T_FP16_PRECAST.iter())
        .chain(QMM_T_BIAS_FP16_PRECAST.iter())
        .chain(QMM_T_RESIDUAL_FP16_PRECAST.iter())
        .chain(QMM_T_SPLITK.iter())
        .chain(QMM_T_SPLITK_F32.iter())
        .chain(QMM_T_SPLITK_FP16_PRECAST.iter())
        .chain(QMM_T_SPLITK_FP16_PRECAST_F32.iter())
        .chain(QMM_T_STRIDED.iter())
        .chain(QMM_T_STRIDED_RESIDUAL.iter())
        .chain(QMM_T_STRIDED_FP16_PRECAST.iter())
        .chain(QMM_T_STRIDED_FP16_PRECAST_RESIDUAL.iter())
        .chain(QMV_FAST.iter())
        .chain(QMV_FAST_RESIDUAL.iter())
        .chain(QMV_TAIL.iter())
        .chain(QMV_TAIL_BIAS.iter())
        .chain(QMV_WIDE_STRIDED.iter())
        .copied()
        // The twelve a routine spells as a literal rather than indexing: the
        // five hand-written `wm`/`wn` tiles, the two split-K reductions, the
        // two casts and the three codecs. They are here so the sweep covers
        // every name this family can hand a pipeline, not only the tabulated
        // ones.
        .chain([
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
            "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
            "qmm_splitk_reduce_bfloat16",
            "qmm_splitk_reduce_f32_bfloat16",
            "cast_qmm_input_bfloat16_to_float16",
            "cast_qmm_input_strided_bfloat16_to_float16",
            "affine_encode_u4_bf16",
            "affine_encode_u4_f32",
            "mxfp4_dequant_bf16",
        ])
        .collect();
    let absent: Vec<&&str> = named.iter().filter(|n| !have.contains(**n)).collect();
    assert!(
        absent.is_empty(),
        "these quantised entrypoints are named by a routine and instantiated \
         by no shader: {absent:#?}"
    );
    assert_eq!(
        named.len(),
        303,
        "every table above, and the twelve literals"
    );
}
