//! The launch ABI pilot: `rope`'s twelve launchers, proven by the C++ compiler.
//!
//! Every other area in this crate is proven by a differential oracle — build
//! the C++, sweep both over a grid, require byte-identical output, pin the
//! C++'s hash. That protocol does not reach the launcher boundary, and the
//! reason is worth stating: a differential oracle proves that two
//! implementations of a *described* contract agree. The launchers have no
//! described contract. `KernelSig` carried a kernel's name, its plan, its
//! capabilities and its sink, and not one word about how to call it.
//!
//! So the ABI pilot does not port a launcher. It writes the contract down —
//! `KernelSig::operands` — and then proves the writing.
//!
//! ## The proof
//!
//! `kernels_cuda::abi::emit_c_shim` generates one `extern "C"` function per
//! row whose body CALLS the launcher, with the real `rope.hpp` in scope. The
//! generated translation unit is then compiled. A row that misstates an
//! operand's type, constness, width, position, or the arity of the whole list
//! does not compile, because C++ overload resolution is deciding — not a
//! string comparison, and not a golden that could drift.
//!
//! This is STRICTER than a golden, and it is worth being explicit about why:
//! a golden proves the two sides agreed on the grid that was swept, and a
//! mutation suite estimates how much of the contract the grid reached. Here
//! there is no grid and nothing to estimate. The compiler checks the entire
//! signature or refuses the file.
//!
//! Consequently this file pins no hash and has no `mutate.sh`. What replaces
//! them is [`a_wrong_row_does_not_compile`], which corrupts a row and
//! requires the compile to FAIL — the same question a mutation suite asks
//! ("would the proof notice?"), answered exactly instead of statistically.

#![cfg(feature = "_cuda")]

use std::sync::atomic::{AtomicU64, Ordering};

use std::path::{Path, PathBuf};
use std::process::Command;

use driver_cuda_new::launch::{
    AttentionWorkspaceView, HopperPrefillPlan, KvCacheLayerView, MlaCacheLayerView,
    YarnOriginalParams,
};
use kernels::{KernelSig, Operand, Ty};
use kernels_cuda::abi::Record;

/// Where `kernels-cuda`'s sources are, relative to this crate.
fn csrc() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels-cuda/csrc/src")
}

/// The header the pilot's rows are declared in.
const ROPE_HPP: &str = "rope/rope.hpp";

/// Compile a generated shim against the real headers.
///
/// `-fsyntax-only`: nothing is linked, so this needs neither the built
/// archive nor nvcc. And the only CUDA name `rope.hpp` uses is
/// `cudaStream_t`, which `tests/oracle/launch_abi/stub/` supplies — so it needs
/// no CUDA toolkit either, which is what the CI job that runs this crate
/// promises. The stub directory is the ONLY include path added besides
/// `csrc/src`, so the answer does not depend on which CUDA is installed.
///
/// The scratch directory is per CALL, not per process. Test binaries run
/// their cases on threads of one process, so a pid-named directory is shared
/// state: two cases race on `shim.cpp`, and the one that reads a neighbour's
/// text is answered about the wrong shim. That failure is silent in the only
/// direction that matters — a corrupted row compiles, because what actually
/// got compiled was the good one — so it reads as "the proof is not
/// watching" rather than as a harness bug.
fn compile(shim: &str) -> Result<(), String> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "pie-launch-abi-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let src = dir.join("shim.cpp");
    std::fs::write(&src, shim).expect("write shim");

    let stub = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle/launch_abi/stub");
    let out = Command::new("g++")
        .arg("-std=c++20")
        .arg("-fsyntax-only")
        .arg(format!("-I{}", stub.display()))
        .arg(format!("-I{}", csrc().display()))
        .arg(&src)
        .output()
        .expect("g++ must be available");
    let _ = std::fs::remove_dir_all(&dir);
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

/// The toolkit's include directory, if this machine has one — `build.rs`'s
/// `cuda_home()` resolution, which is where the pairing matters: a test that
/// looked somewhere else would prove a different build than the one shipped.
fn cuda_include() -> Option<PathBuf> {
    let home = std::env::var_os("CUDA_HOME")
        .or_else(|| std::env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let inc = home.join("include");
    inc.join("cuda_runtime.h").exists().then_some(inc)
}

/// [`compile`] with extra include directories, for the one case whose
/// headers the stub cannot serve.
fn compile_with(shim: &str, extra: &[PathBuf]) -> Result<(), String> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "pie-launch-abi-x{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let src = dir.join("shim.cpp");
    std::fs::write(&src, shim).expect("write shim");

    let stub = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle/launch_abi/stub");
    let mut cmd = Command::new("g++");
    cmd.arg("-std=c++20").arg("-fsyntax-only");
    for e in extra {
        cmd.arg(format!("-I{}", e.display()));
    }
    // The stub goes LAST: where the toolkit has the real header, the real
    // one wins, and the stub only fills what it does not.
    let out = cmd
        .arg(format!("-I{}", csrc().display()))
        .arg(format!("-I{}", stub.display()))
        .arg(&src)
        .output()
        .expect("g++ must be available");
    let _ = std::fs::remove_dir_all(&dir);
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

fn rope_shim(table: &'static [KernelSig]) -> String {
    kernels_cuda::abi::emit_c_shim(&[table], &[ROPE_HPP]).expect("no entry-point collisions")
}

/// The headers `attn`'s rows are declared in.
///
/// Every `attn/*.hpp`, plus `gemm/gemm.hpp` — two of `attn`'s rows are the MLA
/// absorb pair, which live in `gemm.hpp` because they are `cublas` calls. A
/// family is not proven while two of its rows sit outside the shim, so the
/// list follows the ROWS rather than the directory.
fn attn_headers() -> Vec<String> {
    let mut hs: Vec<String> = std::fs::read_dir(csrc().join("attn"))
        .expect("attn/")
        .filter_map(|e| {
            let n = e.ok()?.file_name().into_string().ok()?;
            n.ends_with(".hpp").then(|| format!("attn/{n}"))
        })
        .collect();
    hs.sort();
    hs.push("gemm/gemm.hpp".into());
    hs
}

fn attn_shim(table: &'static [KernelSig]) -> String {
    let hs = attn_headers();
    let refs: Vec<&str> = hs.iter().map(String::as_str).collect();
    kernels_cuda::abi::emit_c_shim(&[table], &refs).expect("no entry-point collisions")
}

/// The headers `norm`'s rows are declared in — every `norm/*.hpp`.
fn norm_headers() -> Vec<String> {
    let mut hs: Vec<String> = std::fs::read_dir(csrc().join("norm"))
        .expect("norm/")
        .filter_map(|e| {
            let n = e.ok()?.file_name().into_string().ok()?;
            n.ends_with(".hpp").then(|| format!("norm/{n}"))
        })
        .collect();
    hs.sort();
    hs
}

/// `norm`'s rows, proven the same way `rope`'s and `attn`'s are.
///
/// Twenty-eight launchers across seven headers, and the family every
/// other one leans on: a wrong row here is a wrong argument in an arm
/// that four executors reach.
#[test]
fn every_norm_row_states_its_launcher_exactly() {
    let table = kernels_cuda::norm::KERNELS;
    let stated = table.iter().filter(|k| !k.operands.is_empty()).count();
    assert_eq!(
        stated,
        table.len(),
        "{} of {} norm rows are unstated, so the shim silently skips them",
        table.len() - stated,
        table.len()
    );
    let hs = norm_headers();
    let refs: Vec<&str> = hs.iter().map(String::as_str).collect();
    let shim = kernels_cuda::abi::emit_c_shim(&[table], &refs)
        .expect("no entry-point collisions");
    if let Err(err) = compile(&shim) {
        panic!(
            "the generated shim does not compile, so a row misstates its \
             launcher:\n{err}"
        );
    }
}

/// `mlp`'s rows. Sixteen activations across two headers, and the family
/// whose default arguments make a hand-written binding easiest to get
/// wrong -- `gpt_oss_glu_bf16` alone carries three.
#[test]
fn every_mlp_row_states_its_launcher_exactly() {
    let table = kernels_cuda::mlp::KERNELS;
    let stated = table.iter().filter(|k| !k.operands.is_empty()).count();
    assert_eq!(
        stated,
        table.len(),
        "{} of {} mlp rows are unstated, so the shim silently skips them",
        table.len() - stated,
        table.len()
    );
    let shim =
        kernels_cuda::abi::emit_c_shim(&[table], &["mlp/swiglu.hpp", "mlp/gaussian_topk.hpp"])
            .expect("no entry-point collisions");
    if let Err(err) = compile(&shim) {
        panic!(
            "the generated shim does not compile, so a row misstates its \
             launcher:\n{err}"
        );
    }
}

/// `quant`'s and `moe`'s rows, and the STATED half of `layout`'s and
/// `gemm`'s.
///
/// `gemm`'s scaled entry points are here because 1b made them
/// spellable: the storage a weight is in used to reach the dispatcher
/// inside a `WeightView`, a descriptor the driver BUILT from a
/// per-layer struct no statement mentioned. Assembling it inside the
/// launcher and taking its fields flat is what let a row describe the
/// call at all -- a struct is not something the operand vocabulary can
/// state, and giving it a kind would have stated nothing.
///
/// Not a whole-family assertion like `norm`'s and `mlp`'s, because two
/// of these families carry rows the shim cannot reach yet and saying so
/// is better than a count that quietly excludes them:
///
///   * `dist::` and `comm::` name METHODS on `NcclComm`, not free
///     launchers, so `::pie_cuda_driver::kernels::dist::all_reduce_bf16`
///     does not exist to forward to. They need free wrappers first.
///   * `gemm`'s remaining rows take a `WeightView` or pointer arrays,
///     which the operand vocabulary has no kind for yet.
///
/// What IS stated compiles, which is the claim this test can make.
#[test]
fn the_stated_quant_layout_gemm_and_moe_rows_describe_their_launchers() {
    let tables: [&'static [KernelSig]; 4] = [
        kernels_cuda::quant::KERNELS,
        kernels_cuda::layout::KERNELS,
        kernels_cuda::gemm::KERNELS,
        kernels_cuda::moe::KERNELS,
    ];
    let headers = [
        "quant/dequant_fp4.hpp",
        "quant/dequant_fp8.hpp",
        "quant/dequant_wna16.hpp",
        "quant/dtype_cast.hpp",
        "quant/mxfp4_marlin.hpp",
        "layout/embed.hpp",
        "layout/gather_rows.hpp",
        "layout/slot_ops.hpp",
        "layout/split_gate_up.hpp",
        "layout/deinterleave.hpp",
        "gemm/gemm.hpp",
        "gemm/gemv.hpp",
        "comm/custom_all_reduce.hpp",
        "moe/dsv4_routing.hpp",
        "moe/moe_dispatch.hpp",
        "moe/moe_grouped_gemm.hpp",
        "moe/flashinfer_moe.hpp",
        "sample/argmax.hpp",
        "../third_party/marlin_moe/marlin_moe_wrapper.hpp",
        "moe/topk_sigmoid.hpp",
        "moe/topk_softmax.hpp",
    ];
    let shim = kernels_cuda::abi::emit_c_shim(&tables, &headers)
        .expect("no entry-point collisions");
    if let Err(err) = compile(&shim) {
        panic!(
            "the generated shim does not compile, so a row misstates its \
             launcher:\n{err}"
        );
    }
}

/// `ssm`'s rows — the largest single family, and the one whose ten
/// recurrence spellings differ only by which state dtype and whether the
/// heads are grouped.
///
/// Ten near-identical argument lists is exactly where a hand-written
/// binding goes wrong quietly: `state_base` is `float*` in six of them
/// and `void*` in four, and the two are the same pointer at a call site.
#[test]
fn every_ssm_row_states_its_launcher_exactly() {
    let table = kernels_cuda::ssm::KERNELS;
    // THREE rows stay unstated, and naming them is the point of pinning
    // the count rather than asserting it away:
    //
    //
    // The two `build_nemotron_moe_ptrs_*` builders came IN with the
    // pointer-array kinds -- they take `const void* const*` for the
    // weights they read and `void**` for the arrays they fill, and only
    // `BufArray` vs `BufArrayOutMut` makes the difference a compile
    // error instead of a builder writing an array it was handed to
    // read.
    let stated = table.iter().filter(|k| !k.operands.is_empty()).count();
    assert_eq!(
        stated,
        table.len(),
        "{} of {} ssm rows are unstated, so the shim silently skips them",
        table.len() - stated,
        table.len()
    );
    let shim = kernels_cuda::abi::emit_c_shim(
        &[table],
        &[
            "ssm/causal_conv1d.hpp",
            "ssm/flashinfer_mamba.hpp",
            "ssm/gated_delta_net.hpp",
            "ssm/kda.hpp",
            "ssm/nemotron_h.hpp",
        ],
    )
    .expect("no entry-point collisions");
    if let Err(err) = compile(&shim) {
        panic!(
            "the generated shim does not compile, so a row misstates its \
             launcher:\n{err}"
        );
    }
}



/// The pilot itself: every stated `rope` row describes its launcher exactly.
#[test]
fn every_rope_row_states_its_launcher_exactly() {
    let shim = rope_shim(kernels_cuda::rope::KERNELS);
    if let Err(err) = compile(&shim) {
        panic!(
            "the generated shim does not compile, so a row misstates its \
             launcher:\n{err}\n--- shim ---\n{shim}"
        );
    }
}

/// The same proof at family scale: all fifty `attn` rows, ~700 operands.
///
/// `rope` was twelve rows of scalars and buffers. `attn` is what the ABI
/// actually has to survive: views passed BY VALUE, plan caches passed as
/// `const&` to a type the header never defines, a `cublasHandle_t` where a
/// stream would be, and both halves of every const/mut pointer pair. If the
/// vocabulary in `kernels::Ty` were short of any of that, this would not
/// compile — which is the point of running it as one shim rather than fifty.
#[test]
fn every_attn_row_states_its_launcher_exactly() {
    let table = kernels_cuda::attn::KERNELS;
    let stated = table.iter().filter(|k| !k.operands.is_empty()).count();
    assert_eq!(
        stated,
        table.len(),
        "{} of {} attn rows are unstated, so the shim silently skips them",
        table.len() - stated,
        table.len()
    );
    let shim = attn_shim(table);
    if let Err(err) = compile(&shim) {
        panic!(
            "the generated shim does not compile, so a row misstates its \
             launcher:\n{err}"
        );
    }
}

/// Every launcher `rope.hpp` declares has a row.
///
/// The other half of the crate's invariant, and the half a generated shim
/// cannot reach: emitting from the table proves each ROW is real, never that
/// the table is complete. This is what the pilot found — twelve declarations
/// against ten rows, with `rope_bf16` and `rope_partial_bf16_position_delta`
/// present in the header, called by the driver, and named nowhere in the
/// table the compiler plans against.
#[test]
fn every_launcher_the_header_declares_has_a_row() {
    let text = std::fs::read_to_string(csrc().join(ROPE_HPP)).expect("rope.hpp");
    let declared: Vec<String> = text
        .lines()
        .filter_map(|l| l.strip_prefix("void "))
        .filter_map(|l| l.split_once('('))
        .map(|(name, _)| name.trim().to_string())
        .collect();
    assert!(
        declared.len() >= 12,
        "the scan found {} declarations, so its shape assumption broke",
        declared.len()
    );

    let missing: Vec<&String> = declared
        .iter()
        .filter(|d| {
            !kernels_cuda::rope::KERNELS
                .iter()
                .any(|k| k.symbol == format!("rope::{d}"))
        })
        .collect();
    assert!(missing.is_empty(), "declared but not in the table: {missing:?}");
}

/// Why a launcher `attn`'s headers declare is allowed to have no row.
#[derive(Clone, Copy, PartialEq)]
enum NoRow {
    /// A prepare. The table carries these as `needs = Prepare::*` on the
    /// dispatch that obligates them, not as rows of their own — see the
    /// `kernels` crate's own table of what each declaration replaces.
    Prepare,
    /// One-time device work at startup. Real work, but not part of any
    /// forward, so no declaration ever names it.
    Warmup,
    /// The driver calls it; `dsl::cuda` does not name it. The table's subject
    /// is the planner's vocabulary — `model`'s `kernels_table` asserts the
    /// two are EQUAL — so a launcher the driver reaches for on its own is
    /// correctly absent. `split_qkv_bf16` is the loud case: 390 call sites.
    DriverInternal,
    /// Only sibling `.cu` files in this crate call it. Checked below rather
    /// than believed.
    KernelsInternal,
}

/// Every launcher `attn`'s headers declare is a row, or is one of four
/// documented kinds of not-a-row.
///
/// The rope pilot could assert the flat thing — every declaration has a row —
/// because for `rope` it is true. For `attn` it is false BY DESIGN, and a
/// test that asserted it anyway would have to be deleted rather than
/// answered. What is actually load-bearing is that no launcher joins these
/// headers without someone deciding which kind it is: 77 declarations against
/// 50 rows is not a gap to close, it is 27 decisions, and this is where they
/// are written down.
///
/// `KernelsInternal` is not taken on trust — the claim is "the driver never
/// calls this", the driver's sources are next door, and so it is checked.
#[test]
fn every_attn_launcher_is_a_row_or_a_stated_exception() {
    #[rustfmt::skip]
    let exceptions: &[(&str, NoRow)] = &[
        ("plan_attention_flashinfer_decode_bf16",       NoRow::Prepare),
        ("plan_attention_flashinfer_prefill_bf16",      NoRow::Prepare),
        ("plan_attention_flashinfer_prefill_sm90_bf16", NoRow::Prepare),
        ("plan_attention_mla_bf16",                     NoRow::Prepare),
        ("prepare_attention_xqa_decode_bf16",           NoRow::Prepare),
        ("set_decode_plan_int_base",                    NoRow::Prepare),
        ("xqa_decode_bf16_warmup_current_device",       NoRow::Warmup),
        ("xqa_decode_bf16_gqa5_warmup_current_device",  NoRow::Warmup),
        ("split_qkv_bf16",                              NoRow::DriverInternal),
        ("split_qkv_bf16_devwin",                       NoRow::DriverInternal),
        ("pack_dense_mask",                             NoRow::DriverInternal),
        ("pack_structured_mask",                        NoRow::DriverInternal),
        ("copy_kv_cells_bf16",                          NoRow::DriverInternal),
        ("attention_flashinfer_prefill_bf16",           NoRow::KernelsInternal),
        ("attention_flashinfer_prefill_custom_bf16",    NoRow::KernelsInternal),
        ("dispatch_attention_flashinfer_decode_capture_bf16", NoRow::KernelsInternal),
        ("dispatch_attention_flashinfer_prefill_custom_bf16", NoRow::KernelsInternal),
        ("attention_mtp_history_bf16",                  NoRow::KernelsInternal),
        ("attention_naive_bf16",                        NoRow::KernelsInternal),
        ("attention_naive_paged_custom",                NoRow::KernelsInternal),
        ("attention_naive_paged_decode",                NoRow::KernelsInternal),
        ("attention_xqa_decode_bf16",                   NoRow::KernelsInternal),
        ("add_ape_f32",                                 NoRow::KernelsInternal),
        ("attention_compressed_bf16",                   NoRow::KernelsInternal),
        ("average_pool_bf16",                           NoRow::KernelsInternal),
        ("dsv4_compress_gather_bf16",                   NoRow::KernelsInternal),
        ("gated_softmax_pool_bf16",                     NoRow::KernelsInternal),
        ("write_kv_to_pages_at_positions_bf16",         NoRow::KernelsInternal),
        ("write_mla_to_pages_bf16",                     NoRow::KernelsInternal),
    ];

    let declared = declared_launchers();
    assert!(
        declared.len() >= 77,
        "the scan found {} declarations, so its shape assumption broke",
        declared.len()
    );

    let has_row = |n: &str| {
        kernels_cuda::attn::KERNELS
            .iter()
            .any(|k| k.symbol == format!("attn::{n}") || k.symbol.ends_with(&format!("::{n}")))
    };
    let undecided: Vec<&String> = declared
        .iter()
        .filter(|d| !has_row(d) && !exceptions.iter().any(|(n, _)| n == d))
        .collect();
    assert!(
        undecided.is_empty(),
        "declared in attn/, no row, and no stated reason: {undecided:?}"
    );

    let stale: Vec<&str> = exceptions
        .iter()
        .map(|(n, _)| *n)
        .filter(|n| !declared.iter().any(|d| d == n))
        .collect();
    assert!(stale.is_empty(), "exception for a launcher no header declares: {stale:?}");

    // The `KernelsInternal` claim, checked against the driver's sources.
    // THE DRIVER IS THIS CRATE NOW. This used to read
    // `../driver-cuda/csrc/src`; when that tree was deleted the scan
    // found zero bytes and refused to pass, which is what the vacuity
    // guard below is for. The claim did not change with the language: a
    // launcher the DRIVER never calls is the question, and the driver is
    // whichever shell reaches the kernels.
    let driver = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut driver_text = String::new();
    collect_sources(&driver, &mut driver_text);
    assert!(
        driver_text.len() > 500_000,
        "only {} bytes of driver source found, so the check is vacuous",
        driver_text.len()
    );
    let wrong: Vec<&str> = exceptions
        .iter()
        .filter(|(_, why)| *why == NoRow::KernelsInternal)
        .map(|(n, _)| *n)
        .filter(|n| mentions_word(&driver_text, n))
        .collect();
    assert!(
        wrong.is_empty(),
        "called `KernelsInternal` but the driver calls it, so it is really \
         DriverInternal or a missing row: {wrong:?}"
    );
}

/// Every `void` launcher declared across `attn/*.hpp`, by name.
fn declared_launchers() -> Vec<String> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(csrc().join("attn")).expect("attn/") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("hpp") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("header");
        for line in text.lines() {
            let Some(rest) = line.strip_prefix("void ") else { continue };
            let Some((name, _)) = rest.split_once('(') else { continue };
            if name.chars().all(|c| c.is_alphanumeric() || c == '_') && !name.is_empty() {
                out.push(name.to_string());
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn collect_sources(dir: &Path, out: &mut String) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_sources(&p, out);
        } else if matches!(
            p.extension().and_then(|e| e.to_str()),
            // `rs` since the driver became Rust; the C++ extensions
            // stay because this walker also reads `csrc/` trees.
            Some("cpp" | "hpp" | "cu" | "inc" | "rs")
        ) && let Ok(t) = std::fs::read_to_string(&p)
        {
            out.push_str(&t);
            out.push('\n');
        }
    }
}

/// `needle` appears in `hay` as a whole identifier.
///
/// Substring matching would answer the wrong question here: every
/// `attention_naive_bf16` is inside some `attention_naive_bf16_something`,
/// and a check that cannot tell them apart cannot fail usefully.
fn mentions_word(hay: &str, needle: &str) -> bool {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    hay.match_indices(needle).any(|(i, _)| {
        let before = hay[..i].chars().next_back().is_none_or(|c| !ident(c));
        let after = hay[i + needle.len()..].chars().next().is_none_or(|c| !ident(c));
        before && after
    })
}

/// A row with no operands is not a row that takes none.
///
/// The distinction matters while the table is being filled a family at a
/// time: emitting an unstated row as a nullary `extern "C"` would generate a
/// call the compiler rejects for the wrong reason, and would make "this
/// family is done" indistinguishable from "this family is empty".
///
/// Asked of a SYNTHETIC row rather than of whichever family happens to be
/// empty today. It used to point at `attn`, and filling `attn` turned a
/// passing test into a failing one without anything being wrong — the check
/// is about `stated()`, so its subject should be too.
#[test]
fn an_unstated_row_is_skipped_rather_than_called_with_nothing() {
    let stated = kernels_cuda::rope::KERNELS
        .iter()
        .find(|k| !k.operands.is_empty())
        .expect("some rope row is stated");
    let row: &'static [KernelSig] =
        Vec::leak(vec![KernelSig { operands: &[], ..*stated }]);
    let shim = rope_shim(row);
    assert!(
        !shim.contains("extern \"C\""),
        "the row states no operands, so nothing should be emitted:\n{shim}"
    );
}

/// Corrupting a row must break the build — the mutation suite, answered
/// exactly.
///
/// Each case changes ONE thing a hand-written binding gets wrong, and every
/// one of them has to be caught. The last two are the interesting ones: they
/// are not type errors, they are an operand list of the right types in the
/// wrong ORDER, which is precisely the failure a `void*`-flattened ABI cannot
/// see and this one can.
#[test]
fn a_wrong_row_does_not_compile() {
    let base = kernels_cuda::rope::KERNELS
        .iter()
        .find(|k| k.symbol == "rope::qk_rmsnorm_rope_bf16")
        .expect("the pilot row");

    // `q_weight`/`k_weight` are `const void*`; `positions` is `const i32*`;
    // the extents are `int` and the two rates are `float`.
    let ops = base.operands;
    let swap = |i: usize, j: usize| {
        let mut v: Vec<Operand> = ops.to_vec();
        v.swap(i, j);
        v
    };
    let retype = |i: usize, ty: Ty| {
        let mut v: Vec<Operand> = ops.to_vec();
        v[i].ty = ty;
        v
    };

    let cases: Vec<(&str, Vec<Operand>)> = vec![
        ("a written buffer is claimed to be read-only", retype(0, Ty::Buf)),
        ("a read-only weight is claimed to be written", retype(2, Ty::BufMut)),
        ("positions loses its element type", retype(4, Ty::Buf)),
        ("an extent is widened to a float", retype(5, Ty::F32)),
        ("a rate is narrowed to an int", retype(9, Ty::I32)),
        ("the stream is dropped", ops[..ops.len() - 1].to_vec()),
        ("an operand is invented", {
            let mut v = ops.to_vec();
            v.insert(5, Operand { name: "extra", ty: Ty::I32, nullable: false, source: kernels::Source::Unbound });
            v
        }),
        ("q and k_weight trade places", swap(0, 3)),
        ("an extent and a rate trade places", swap(6, 9)),
    ];

    for (what, operands) in cases {
        let leaked: &'static [Operand] = Vec::leak(operands);
        let row: &'static [KernelSig] = Vec::leak(vec![KernelSig {
            name: base.name,
            symbol: base.symbol,
            whole: base.whole,
            needs: base.needs,
            lacks: base.lacks,
            sink: base.sink,
            in_place: base.in_place,
            depth_prefix_plan: base.depth_prefix_plan,
            operands: leaked,
            returns: base.returns,
            axes: base.axes,
            file: base.file,
            launch: base.launch,
        }]);
        assert!(
            compile(&rope_shim(row)).is_err(),
            "a row where {what} still compiled, so the proof is not watching"
        );
    }
}

/// The control: a change that is NOT a mistake must still compile.
///
/// Without this the test above passes for a build that is broken for some
/// unrelated reason, and every mutation registers as caught. Renaming an
/// operand is the right control here because the name is prose — the table
/// says so — so a rename must be invisible to the compiler while touching
/// exactly the text the mutations touch.
#[test]
fn renaming_an_operand_is_not_a_mistake() {
    let base = kernels_cuda::rope::KERNELS
        .iter()
        .find(|k| k.symbol == "rope::qk_rmsnorm_rope_bf16")
        .expect("the pilot row");
    let renamed: &'static [Operand] = Vec::leak(
        base.operands
            .iter()
            .enumerate()
            .map(|(i, o)| Operand {
                name: Box::leak(format!("arg{i}").into_boxed_str()),
                ..*o
            })
            .collect::<Vec<_>>(),
    );
    let row: &'static [KernelSig] = Vec::leak(vec![KernelSig {
        name: base.name,
        symbol: base.symbol,
        whole: base.whole,
        needs: base.needs,
        lacks: base.lacks,
        sink: base.sink,
        in_place: base.in_place,
        depth_prefix_plan: base.depth_prefix_plan,
        operands: renamed,
        returns: base.returns,
        axes: base.axes,
        file: base.file,
        launch: base.launch,
    }]);
    if let Err(err) = compile(&rope_shim(row)) {
        panic!("the control failed to compile, so the mutations prove nothing:\n{err}");
    }
}

/// The Rust bindings declare exactly what the C++ shim defines.
///
/// Both are generated from one row, so this cannot fail by drift; what it
/// pins is that the two emitters agree on the ENTRY POINT spelling, which is
/// the one string the linker matches on and the one thing neither compiler
/// checks.
#[test]
fn the_rust_bindings_name_the_symbols_the_shim_defines() {
    let shim = rope_shim(kernels_cuda::rope::KERNELS);
    let rs = kernels_cuda::abi::emit_rust_bindings(&[kernels_cuda::rope::KERNELS]);
    for k in kernels_cuda::rope::KERNELS {
        let entry = kernels_cuda::abi::entry_name(k.symbol);
        assert!(shim.contains(&format!("void {entry}(")), "{entry} not defined");
        assert!(rs.contains(&format!("fn {entry}(")), "{entry} not declared");
    }
}

/// The two ways the SHIM BUILD can fail that no per-family case reaches.
///
/// Both were found by the `origin/rewrite` merge, which is the point: a
/// family's rows compiling alone is not the bridge building, and until this
/// case existed the first thing to notice either was `build.rs` under
/// `--features bridge` — a build that needs the CUDA toolkit and so runs in
/// one CI job, late.
///
/// **A directory nobody proves.** `build.rs` names eleven family directories
/// by hand, and `comm` was not among them: the two collective rows live in
/// `gemm.rs`, their launchers live in `csrc/src/comm/`, and no
/// `every_*_row_states_its_launcher_exactly` case owns that directory. Here
/// the include set is READ from the tree rather than typed, so a row whose
/// family has a directory is covered the moment the directory exists.
///
/// **A name Rust cannot spell.** The C++ side accepts an operand called
/// `ref`; `emit_rust_bindings` writes it into an `extern "C"` block and the
/// block does not parse. Operand names are the row author's to choose
/// (`renaming_an_operand_is_not_a_mistake`), which is exactly why the
/// choice has to be checked on BOTH sides.
#[test]
fn the_whole_bridge_generates_for_both_sides() {
    let mut headers: Vec<String> = Vec::new();
    let mut dirs: Vec<String> = std::fs::read_dir(csrc())
        .expect("csrc/src")
        .filter_map(|e| {
            let e = e.ok()?;
            e.path().is_dir().then(|| e.file_name().into_string().ok())?
        })
        .collect();
    dirs.sort();
    for dir in &dirs {
        headers.extend(headers_in(dir));
    }
    headers.push("../third_party/marlin_moe/marlin_moe_wrapper.hpp".into());
    assert!(
        dirs.iter().any(|d| d == "comm"),
        "the directory scan found no `comm/`, so the case that motivated it \
         would pass vacuously: {dirs:?}"
    );

    let tables: &[&'static [kernels::KernelSig]] =
        &[kernels_cuda::KERNELS, kernels_cuda::driver_internal::DRIVER_KERNELS];
    let refs: Vec<&str> = headers.iter().map(String::as_str).collect();
    let shim = kernels_cuda::abi::emit_c_shim(tables, &refs).expect("no entry-point collisions");

    // The WHOLE bridge needs the real toolkit headers, and that is the
    // difference from every per-family case above. Those compile against
    // `tests/oracle/launch_abi/stub/`, which supplies the two CUDA names
    // they touch and nothing else — which is what lets this crate's CI job
    // run them with no toolkit installed. The vision towers include
    // `<cuda_bf16.h>`, so the union cannot be served that way.
    //
    // So this half runs where `build.rs` would run, and says so where it
    // cannot. Skipping loudly beats either alternative: dropping the
    // toolkit-dependent headers would make the union stop being the union,
    // and requiring a toolkit would take the whole file out of the
    // toolkit-free job for one case.
    match cuda_include() {
        Some(inc) => {
            if let Err(err) = compile_with(&shim, &[inc]) {
                panic!(
                    "the whole-bridge shim does not compile, so a row's launcher \
                     is not reachable from the headers `build.rs` includes:\n{err}"
                );
            }
        }
        None => eprintln!(
            "SKIPPED the shim half: no CUDA include directory (set CUDA_HOME or \
             CUDA_PATH). The operand-name half below still ran."
        ),
    }

    // The Rust half. `KEYWORDS` is the raw-identifier set: a name in it is
    // valid C++ and invalid Rust, which is the whole asymmetry.
    #[rustfmt::skip]
    const KEYWORDS: &[&str] = &[
        "as", "break", "const", "continue", "crate", "dyn", "else", "enum",
        "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop",
        "match", "mod", "move", "mut", "pub", "ref", "return", "self", "Self",
        "static", "struct", "super", "trait", "true", "type", "unsafe", "use",
        "where", "while", "async", "await", "box", "final", "macro",
        "override", "priv", "try", "typeof", "unsized", "virtual", "yield",
    ];
    let mut bad = Vec::new();
    for table in tables {
        for k in table.iter() {
            for operand in k.operands {
                if KEYWORDS.contains(&operand.name) {
                    bad.push(format!("{}: `{}`", k.symbol, operand.name));
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "these operand names are Rust keywords, so the generated bindings do \
         not parse — rename them in the row: {bad:?}"
    );
}

// ---------------------------------------------------------------------------
// Records: the operands that are neither a scalar nor a pointer.
// ---------------------------------------------------------------------------

/// Every mirrored record, with the offsets its Rust side computes.
fn records() -> Vec<Record> {
    vec![
        kernels_cuda::record!(KvCacheLayerView => "::pie_cuda_driver::KvCacheLayerView" {
            layer, source_layer, num_pages, page_size, num_kv_heads, head_dim,
            scheme, storage_dtype, block_size,
            k_pages, v_pages, k_scales, v_scales, k_bf16_pages, v_bf16_pages,
            k_env_min, k_env_max,
            hnd_layout, native_bf16,
        }),
        kernels_cuda::record!(AttentionWorkspaceView => "::pie_cuda_driver::AttentionWorkspaceView" {
            float_buffer, float_bytes, int_buffer, int_bytes, page_locked_int,
        }),
        kernels_cuda::record!(MlaCacheLayerView => "::pie_cuda_driver::MlaCacheLayerView" {
            layer, num_pages, page_size, kv_lora_rank, qk_rope_head_dim,
            ckv_pages, kpe_pages,
        }),
        kernels_cuda::record!(HopperPrefillPlan => "::pie_cuda_driver::kernels::attn::HopperPrefillPlan" {
            qo_tile_indices_offset, qo_indptr_offset, kv_indptr_offset,
            qo_len_offset, kv_len_offset, head_indices_offset,
            work_indptr_offset, batch_indices_offset,
            same_schedule_for_all_heads,
            total_tokens, num_requests, num_q_heads, num_kv_heads, head_dim,
            page_size, window_left, causal, valid,
        }),
        kernels_cuda::record!(YarnOriginalParams => "::pie_cuda_driver::kernels::attn::YarnOriginalParams" {
            factor, beta_fast, beta_slow, attention_factor, original_max_position,
        }),
    ]
}

/// The headers the mirrored records are declared in.
///
/// One list, used by every layout case including the mutation ones. That
/// matters more than it looks: those cases assert a TU fails to compile, so a
/// missing include would make them pass for the wrong reason — the mutation
/// would never be what broke it. When `records()` grew from one to five this
/// was the bug, and this is the shape that does not have it.
const MIRROR_HPPS: &[&str] = &[
    "attn/kv_cache_view.hpp",
    "attention_workspace_view.hpp",
    "attn/mla_cache_view.hpp",
    "attn/attention_flashinfer_hopper.hpp",
    "attn/mla_paged.hpp",
];

/// A `#[repr(C)]` mirror really does have the C++ record's layout.
///
/// This is the claim that decides whether a POD operand is a port or a
/// wrapper. If it holds, `KvCacheLayerView` crosses the boundary as itself —
/// no accessor shims, no field-by-field constructor, no copy — and every
/// other descriptor in the launcher surface is the same kind of thing.
#[test]
fn the_mirrors_have_the_layout_the_cpp_has() {
    let tu = kernels_cuda::abi::emit_layout_assertions(&records(), MIRROR_HPPS);
    if let Err(err) = compile(&tu) {
        panic!("a mirror disagrees with the C++ record:\n{err}\n--- tu ---\n{tu}");
    }
}

/// One way a mirror can drift from the record it claims to describe.
type Mutation = Box<dyn Fn(&mut Record)>;

/// And the proof notices when it stops being true.
///
/// The mutation suite for the layout claim. Each case is a way a mirror can
/// drift from a record that is edited on the other side of the boundary, and
/// the last is the one `sizeof` alone would miss: a member APPENDED to the
/// C++ lands in the tail padding an 8-aligned record already has, so size,
/// alignment and every existing offset all still agree. Only the member-count
/// binding catches it.
#[test]
fn a_wrong_mirror_does_not_compile() {
    let bad = |mutate: &dyn Fn(&mut Record)| {
        let mut rs = records();
        mutate(&mut rs[0]);
        compile(&kernels_cuda::abi::emit_layout_assertions(&rs, MIRROR_HPPS))
    };

    let cases: Vec<(&str, Mutation)> = vec![
        ("the record is one byte bigger", Box::new(|r: &mut Record| r.size += 1)),
        ("the record is over-aligned", Box::new(|r: &mut Record| r.align *= 2)),
        (
            "a field moves by one byte",
            Box::new(|r: &mut Record| r.fields[3].1 += 1),
        ),
        (
            "two fields of the same width trade places",
            Box::new(|r: &mut Record| {
                let (a, b) = (r.fields[9].1, r.fields[10].1);
                r.fields[9].1 = b;
                r.fields[10].1 = a;
            }),
        ),
        (
            "a field the C++ does not have is claimed",
            Box::new(|r: &mut Record| r.fields.push(("no_such_field", 0))),
        ),
        (
            "a field the C++ HAS is dropped",
            Box::new(|r: &mut Record| {
                r.fields.pop();
            }),
        ),
    ];

    for (what, mutate) in cases {
        assert!(
            bad(&*mutate).is_err(),
            "a mirror where {what} still compiled, so the proof is not watching"
        );
    }
}

/// The control for the layout proof: renaming the BINDINGS is not a mistake.
///
/// The member-count check binds positionally, so the names it invents carry
/// no claim. If changing them broke the build, the mutation above that drops
/// a field would be "caught" for the wrong reason.
#[test]
fn the_member_count_check_does_not_depend_on_field_names() {
    let mut rs = records();
    let n = rs[0].fields.len();
    for (i, f) in rs[0].fields.iter_mut().enumerate() {
        f.0 = Box::leak(format!("z{}", n - i).into_boxed_str());
    }
    // The offsetof asserts DO name fields, so drop them and keep the rest:
    // what is under test is the binding, not the offsets.
    let tu = kernels_cuda::abi::emit_layout_assertions(&rs, MIRROR_HPPS)
        .lines()
        .filter(|l| !l.contains("offsetof"))
        .filter(|l| !l.contains("is not at"))
        .collect::<Vec<_>>()
        .join("\n");
    if let Err(err) = compile(&tu) {
        panic!("the control failed, so the layout mutations prove nothing:\n{err}");
    }
}

/// The member-count binding is load-bearing, not decoration.
///
/// Dropping the last field is the case the docs on
/// `emit_layout_assertions` claim only the binding can see. That claim is
/// worth checking rather than asserting: here the same mutation is compiled
/// twice, once with the binding and once with it stripped, and the stripped
/// build has to SUCCEED. If it failed, `sizeof` would already have been
/// catching this and the binding would be ceremony.
#[test]
fn without_the_binding_a_dropped_field_would_go_unnoticed() {
    let mut rs = records();
    rs[0].fields.pop();
    let tu = kernels_cuda::abi::emit_layout_assertions(&rs, MIRROR_HPPS);
    assert!(compile(&tu).is_err(), "with the binding, this must fail");

    let without = tu
        .lines()
        .take_while(|l| !l.contains("Exactly"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        without.contains("sizeof") && without.contains("offsetof"),
        "the stripped translation unit must still make the other two claims"
    );
    if let Err(err) = compile(&without) {
        panic!("sizeof/offsetof already caught it, so the binding is ceremony:\n{err}");
    }
}

// ── restored after the origin/rewrite merge ──────────────────────────
//
// The merge resolved this file to the remote's side wholesale, which took
// the remote's new generation proofs and dropped five claims written on
// this branch. They are back, unchanged except where the remote's own work
// made an assertion stale — noted at the line it changed.

fn headers_in(dir: &str) -> Vec<String> {
    let mut hs: Vec<String> = std::fs::read_dir(csrc().join(dir))
        .expect("family directory")
        .filter_map(|e| {
            let n = e.ok()?.file_name().into_string().ok()?;
            n.ends_with(".hpp").then(|| format!("{dir}/{n}"))
        })
        .collect();
    hs.sort();
    hs
}

fn shim_over(table: &'static [KernelSig], headers: &[String]) -> String {
    let refs: Vec<&str> = headers.iter().map(String::as_str).collect();
    kernels_cuda::abi::emit_c_shim(&[table], &refs).expect("no entry-point collisions")
}

/// Every row of a family states its launcher exactly, and every row is stated.
///
/// Factored out because the check has two halves and only the second one is
/// the compiler's. `emit_c_shim` SKIPS a row with no operands, so a family
/// where half the rows are blank produces a shim that compiles and proves
/// nothing — the count assertion is what keeps "it compiled" from meaning
/// "some of it compiled".
fn prove_family(family: &str, table: &'static [KernelSig], headers: &[String]) {
    let stated = table.iter().filter(|k| !k.operands.is_empty()).count();
    assert_eq!(
        stated,
        table.len(),
        "{} of {} {family} rows are unstated, so the shim silently skips them",
        table.len() - stated,
        table.len()
    );
    if let Err(err) = compile(&shim_over(table, headers)) {
        panic!(
            "the generated {family} shim does not compile, so a row misstates \
             its launcher:\n{err}"
        );
    }
}

/// The one `sample` row. Its weight is the table's only `const int8_t*`
/// (`I8s`) — a fused GEMV+argmax over an int8 lm_head with per-channel fp32
/// scales.
#[test]
fn every_sample_row_states_its_launcher_exactly() {
    prove_family("sample", kernels_cuda::sample::KERNELS, &headers_in("sample"));
}

/// Across EVERY table, the rows without operands are exactly the ones with
/// a written reason — and nothing else.
///
/// This is the fill campaign's closing claim. Each family proof asserts its
/// own table; this one says no family was skipped and no future row can
/// quietly join the exception list.
///
/// It earned that description on the `origin/rewrite` merge, which brought
/// four rows this list did not have. None of them was an oversight and none
/// was a pseudo-symbol either, so the list stopped being "the known three"
/// and started carrying WHY — which is the shape it should have had, since
/// the three kinds below close differently:
///
/// * `NotACppFunction` never closes. The symbol names an operation of a
///   declared executor — a `cudaMemcpyAsync` pair, a staged LoRA apply —
///   and there is no function to describe.
/// * `SecondNamespaceRoot` closes on a LAYERING decision, written out at
///   the `dist::` rows in `gemm.rs`: `abi::cpp_path` spells one root
///   (`::pie_cuda_driver::kernels::`), and these launchers live beside it
///   rather than under it — `::pie_cuda_driver::marlin_moe` for the
///   vendored MoE GEMM, a driver-side namespace for the collectives. Each
///   HAS a statable signature; what is missing is which side of the
///   boundary the symbol admits to being on.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Unstated {
    NotACppFunction,
    SecondNamespaceRoot,
}

#[test]
fn the_unstated_rows_are_exactly_the_ones_with_a_written_reason() {
    #[rustfmt::skip]
    let expected: &[(&str, Unstated)] = &[
        ("dist::all_reduce_bf16",                        Unstated::SecondNamespaceRoot),
        ("dist::all_reduce_bf16_out",                    Unstated::SecondNamespaceRoot),
        ("dist::all_gather_bf16",                        Unstated::SecondNamespaceRoot),
        ("marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16", Unstated::SecondNamespaceRoot),
        ("qwen35_verify_stash_store",                    Unstated::NotACppFunction),
        ("qwen35_verify_stash_load",                     Unstated::NotACppFunction),
        ("pie_lora_qkv_correction",                      Unstated::NotACppFunction),
    ];

    let unstated: Vec<&str> = kernels_cuda::KERNELS
        .iter()
        .filter(|k| k.operands.is_empty())
        .map(|k| k.symbol)
        .collect();
    let named: Vec<&str> = expected.iter().map(|(s, _)| *s).collect();
    assert_eq!(
        unstated, named,
        "an unstated row with no written reason is an unfilled row"
    );

    // The `SecondNamespaceRoot` claim, checked rather than believed: a
    // symbol whose family DOES resolve under the one root the shim spells
    // is not blocked on a namespace, it is blocked on somebody writing it.
    for (symbol, why) in expected {
        if *why != Unstated::SecondNamespaceRoot {
            continue;
        }
        let family = symbol.split("::").next().expect("a namespaced symbol");
        assert!(
            !csrc().join(family).is_dir(),
            "`{symbol}` is called `SecondNamespaceRoot`, but `csrc/src/{family}/` \
             exists — so it resolves under the root the shim already spells, and \
             the row is simply unfilled"
        );
    }
}

/// Every row of the DRIVER-INTERNAL table states its launcher exactly.
///
/// The second table (`driver_internal::DRIVER_KERNELS`): launchers the
/// driver fires with no DSL statement behind them — the `DriverInternal`
/// kind the attn exhaustiveness test classifies, made callable for the
/// executor without ever joining the DSL-surface table that `model`'s
/// `kernels_table` holds to equality. Same proof as every family: the shim
/// calls the launcher, the compiler decides.
#[test]
fn every_driver_internal_row_states_its_launcher_exactly() {
    let mut headers = headers_in("layout");
    headers.extend(headers_in("attn"));
    headers.extend(headers_in("norm"));
    // `ssm` and `mlp` joined when the qwen3_5 declaration stopped naming
    // kernels: four launchers a semantic kind now picks live in those two
    // directories, and this test is what proved their rows.
    headers.extend(headers_in("ssm"));
    headers.extend(headers_in("mlp"));
    headers.push("gemm/gemm.hpp".into());
    // The VL tower rows' flat launchers — the C++-struct tower headers
    // stay out (their `std::vector` members are the wrappers' business,
    // not the shim's).
    headers.push("vision/qwen3_vl_tower_c.hpp".into());
    headers.push("vision/gemma4_towers_c.hpp".into());
    prove_family(
        "driver-internal",
        kernels_cuda::driver_internal::DRIVER_KERNELS,
        &headers,
    );
}

/// Every launcher declared across a family's headers, by name.
///
/// `returns` lists the return-type spellings the scan accepts. `attn`'s
/// launchers are all `void`; `norm` also declares two `bool` autotuner
/// probes, and a void-only scan would not merely miscount — it would never
/// ask what kind of not-a-row the probes are, which is the question the
/// exception tests exist to force.
fn declared_in(dir: &str, returns: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(csrc().join(dir)).expect("family directory") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("hpp") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("header");
        for line in text.lines() {
            let Some(rest) = returns.iter().find_map(|r| line.strip_prefix(r)) else { continue };
            let Some((name, _)) = rest.split_once('(') else { continue };
            if name.chars().all(|c| c.is_alphanumeric() || c == '_') && !name.is_empty() {
                out.push(name.to_string());
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Why a launcher `norm`'s headers declare is allowed to have no row.
///
/// Disjoint from [`NoRow`] because `norm`'s not-a-rows are different KINDS
/// of thing, not the same kinds with different members — and an enum shared
/// across families would invite classifying a new `norm` launcher as
/// `Prepare` when `norm` has no prepares.
#[derive(Clone, Copy, PartialEq)]
enum NormNoRow {
    /// The EMITTER chooses this launcher from a semantic op (`RmsNorm`,
    /// `AddBias`) rather than a trace recording it as a `Launch`. There is
    /// no DSL statement behind it, so a row would claim a surface that does
    /// not exist. Checked below rather than believed: the name must appear
    /// in the emitter sources.
    EmitterChosen,
    /// An autotuner probe: returns `bool` and has zero driver call sites.
    /// Both halves checked below.
    AutotunerProbe,
}

/// Every launcher `norm`'s headers declare is a row, or is one of two
/// documented kinds of not-a-row.
///
/// The `attn` twin above owns the long rationale; what is specific to `norm`
/// is the arithmetic — 32 declarations against 26 rows is 6 decisions — and
/// that BOTH exception kinds are checkable, so neither is taken on trust.
/// `rmsnorm_bf16` is the loud case here the way `split_qkv_bf16` was for
/// `attn`: 1,337 call sites, every one of them emitter-chosen.
#[test]
fn every_norm_launcher_is_a_row_or_a_stated_exception() {
    #[rustfmt::skip]
    let exceptions: &[(&str, NormNoRow)] = &[
        ("rmsnorm_bf16",               NormNoRow::EmitterChosen),
        ("add_bias_bf16",              NormNoRow::EmitterChosen),
        ("rmsnorm_gemma_bf16",         NormNoRow::EmitterChosen),
        ("rmsnorm_gated_fp32_in_bf16", NormNoRow::EmitterChosen),
        ("rmsnorm_bf16_tuned",         NormNoRow::AutotunerProbe),
        ("rmsnorm_rasr_tuned",         NormNoRow::AutotunerProbe),
    ];

    // `bool ` is load-bearing: the probes are the only non-`void` launchers,
    // and a void-only scan would count 30 and never see them.
    let declared = declared_in("norm", &["void ", "bool "]);
    assert!(
        declared.len() >= 32,
        "the scan found {} declarations, so its shape assumption broke",
        declared.len()
    );

    let has_row = |n: &str| {
        kernels_cuda::norm::KERNELS
            .iter()
            .any(|k| k.symbol == format!("norm::{n}"))
    };
    let undecided: Vec<&String> = declared
        .iter()
        .filter(|d| !has_row(d) && !exceptions.iter().any(|(n, _)| n == d))
        .collect();
    assert!(
        undecided.is_empty(),
        "declared in norm/, no row, and no stated reason: {undecided:?}"
    );

    let stale: Vec<&str> = exceptions
        .iter()
        .map(|(n, _)| *n)
        .filter(|n| !declared.iter().any(|d| d == n))
        .collect();
    assert!(stale.is_empty(), "exception for a launcher no header declares: {stale:?}");

    // The `EmitterChosen` claim, checked against the emitter's sources.
    // The file set is `scripts/kernel-vocabulary-audit.py`'s `emitted`
    // scan: every `emit.rs` under `model/src`, plus the compiler's lowering.
    let mut emitter_text = String::new();
    collect_files_named(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../model/src"),
        "emit.rs",
        &mut emitter_text,
    );
    if let Ok(t) = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../model-compiler/src/lower.rs"),
    ) {
        emitter_text.push_str(&t);
    }
    assert!(
        emitter_text.len() > 10_000,
        "only {} bytes of emitter source found, so the check is vacuous",
        emitter_text.len()
    );
    let unchosen: Vec<&str> = exceptions
        .iter()
        .filter(|(_, why)| *why == NormNoRow::EmitterChosen)
        .map(|(n, _)| *n)
        .filter(|n| !mentions_word(&emitter_text, n))
        .collect();
    assert!(
        unchosen.is_empty(),
        "called `EmitterChosen` but no emitter names it, so it is really a \
         missing row or some other kind of not-a-row: {unchosen:?}"
    );

    // The `AutotunerProbe` claim, both halves. A probe RETURNS its verdict —
    let norm_text: String = norm_headers()
        .iter()
        .map(|h| std::fs::read_to_string(csrc().join(h)).expect("header"))
        .collect();
    let probes: Vec<&str> = exceptions
        .iter()
        .filter(|(_, why)| *why == NormNoRow::AutotunerProbe)
        .map(|(n, _)| *n)
        .collect();
    for probe in &probes {
        assert!(
            norm_text.contains(&format!("bool {probe}(")),
            "called `AutotunerProbe` but `{probe}` does not return bool, so \
             it is not a probe"
        );
    }
    // — and the driver never calls one.
    // THE DRIVER IS THIS CRATE NOW. This used to read
    // `../driver-cuda/csrc/src`; when that tree was deleted the scan
    // found zero bytes and refused to pass, which is what the vacuity
    // guard below is for. The claim did not change with the language: a
    // launcher the DRIVER never calls is the question, and the driver is
    // whichever shell reaches the kernels.
    let driver = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut driver_text = String::new();
    collect_sources(&driver, &mut driver_text);
    assert!(
        driver_text.len() > 500_000,
        "only {} bytes of driver source found, so the check is vacuous",
        driver_text.len()
    );
    let called: Vec<&&str> = probes.iter().filter(|n| mentions_word(&driver_text, n)).collect();
    assert!(
        called.is_empty(),
        "called `AutotunerProbe` but the driver mentions it, so its \"zero \
         driver call sites\" is stale: {called:?}"
    );
}

/// Concatenate every file named `name` under `dir`, recursively.
///
/// [`collect_sources`] filters by extension; this filters by exact file
/// name, because "the emitter sources" is a claim about which FILES choose
/// kernels, not about a language.
fn collect_files_named(dir: &Path, name: &str, out: &mut String) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_files_named(&p, name, out);
        } else if p.file_name().and_then(|n| n.to_str()) == Some(name)
            && let Ok(t) = std::fs::read_to_string(&p)
        {
            out.push_str(&t);
            out.push('\n');
        }
    }
}

/// The two by-value `enum class` mirrors carry the C++'s own discriminants.
///
/// Enums are not records — `record!` has no fields to walk — so the claim is
/// made directly, in the layout proof's own style: every number below is read
/// off the RUST side (`as i32`, `size_of`) and baked into a generated TU as
/// `static_assert`s against the real headers. A reordered C++ enum fails here
/// rather than routing gemma-4's experts through qwen's activation.
#[test]
fn the_enum_mirrors_carry_the_cpp_discriminants() {
    use driver_cuda_new::launch::{MoeActivation, Mxfp4RowSelect};
    let tu = format!(
        "#include <cstdint>\n\
         #include \"moe/flashinfer_moe.hpp\"\n\
         #include \"quant/mxfp4_marlin.hpp\"\n\
         using ::pie_cuda_driver::kernels::moe::MoeActivation;\n\
         using ::pie_cuda_driver::kernels::quant::Mxfp4RowSelect;\n\
         static_assert(sizeof(MoeActivation) == {});\n\
         static_assert(static_cast<int>(MoeActivation::Relu2) == {});\n\
         static_assert(static_cast<int>(MoeActivation::Swiglu) == {});\n\
         static_assert(static_cast<int>(MoeActivation::Geglu) == {});\n\
         static_assert(sizeof(Mxfp4RowSelect) == {});\n\
         static_assert(static_cast<int>(Mxfp4RowSelect::Identity) == {});\n\
         static_assert(static_cast<int>(Mxfp4RowSelect::Even) == {});\n\
         static_assert(static_cast<int>(Mxfp4RowSelect::Odd) == {});\n",
        core::mem::size_of::<MoeActivation>(),
        MoeActivation::Relu2 as i32,
        MoeActivation::Swiglu as i32,
        MoeActivation::Geglu as i32,
        core::mem::size_of::<Mxfp4RowSelect>(),
        Mxfp4RowSelect::Identity as i32,
        Mxfp4RowSelect::Even as i32,
        Mxfp4RowSelect::Odd as i32,
    );
    if let Err(err) = compile(&tu) {
        panic!("an enum mirror disagrees with the C++:\n{err}");
    }
}

