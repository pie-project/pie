//! The launch ABI proof: a row's operand list, proven by the C++ compiler.
//!
//! # `rope` WAS THE PILOT AND IS NO LONGER HERE
//!
//! This file was written around `rope`'s twelve rows and named them in every
//! case. `rope` has crossed into fn-world (`.wiki/kernel-x/northstar.md` §5
//! step 3): its host programs are `kernels-cuda-new/src/x/rope.rs`, beside
//! the `rope.cuh` they fire, and its contracts state no `operands` at all.
//! There is no row left to emit a shim from, so six cases went with it —
//! `every_rope_row_states_its_launcher_exactly`,
//! `every_launcher_the_header_declares_has_a_row`,
//! `an_unstated_row_is_skipped_rather_than_called_with_nothing`,
//! `a_wrong_row_does_not_compile`, `renaming_an_operand_is_not_a_mistake`
//! and `the_rust_bindings_name_the_symbols_the_shim_defines`, plus the
//! `rope_shim` helper and the `ROPE_HPP` constant.
//!
//! **What replaces the proof for `rope`**: `kernels_cuda_new::x::abi`'s
//! `typecheck_tu` emits a TU that names each declared device function
//! through `Abi::CPP`, so the `__global__` and the Rust `raw::` stub are
//! checked against each other by the same compiler for the same reason.
//! §6.1 of the north star is the argument: *"nothing is written twice"*
//! means *"nothing is written twice **unchecked**"*, and the typecheck TU is
//! the oracle that holds the two sides together.
//!
//! **What is NOT replaced, and is a real loss**: `a_wrong_row_does_not_
//! compile` was the mutation suite for `emit_c_shim` ITSELF, which still
//! serves ten families. It was keyed on `rope::qk_rmsnorm_rope_bf16` by
//! OPERAND INDEX (`retype(0, Ty::Buf)`, `retype(2, Ty::BufMut)`, two
//! `swap`s), so re-keying it means choosing a row from a family that is
//! neither `device::JIT_DISPATCHED` nor `execution::RUST_SERVED` — the two
//! filters `emit_c_shim` applies — and reading its operand order. That is a
//! choice that has to be made against a build, and this change was written
//! without one. It is left undone deliberately rather than re-keyed on a
//! guess: a mutation suite pointed at the wrong row passes and proves
//! nothing, which is worse than an absent one.
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
//! `kernels_cuda_new::abi::emit_c_shim` generates one `extern "C"` function per
//! row whose body CALLS the launcher, with the family's real `.hpp` in scope.
//! The
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
//! Consequently this file pins no hash and has no `mutate.sh`. What replaced
//! them was `a_wrong_row_does_not_compile`, which corrupted a row and
//! required the compile to FAIL — the same question a mutation suite asks
//! ("would the proof notice?"), answered exactly instead of statistically.
//! See the note at the top of this header for where that case went.

#![cfg(feature = "_cuda")]

use std::sync::atomic::{AtomicU64, Ordering};

use std::path::{Path, PathBuf};
use std::process::Command;

use driver_cuda::bind::abi::{
    AttentionWorkspaceView, HopperPrefillPlan, KvCacheLayerView, MlaCacheLayerView,
    YarnOriginalParams,
};
// `Operand` and `Ty` left with the mutation suite — see the header.
use kernels::KernelSig;
use kernels_cuda_new::abi::Record;

/// Where `kernels-cuda`'s sources are, relative to this crate.
fn csrc() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels-cuda/csrc/src")
}

/// Every tree that currently holds a `.cu` or `.cuh` launcher body.
///
/// TWO, AND THE SECOND ONE IS A HOLE THIS CLOSES rather than a place things
/// belong. `csrc()` is the archive. `driver-cuda/csrc` holds
/// `attn/attention_flashinfer.cu`, `attn/plan_lifecycle.cpp`,
/// `supergraph.cu` and the three towers, which earlier passes moved out of the
/// archive because they could not be compiled by NVRTC. That reasoning is
/// right about NVRTC and wrong about the destination — host code is Rust, and
/// C++ in this crate is staging awaiting a rewrite, not a home.
///
/// Either way the scans below have to read it. Every `Orphaned` claim in the
/// exception list means NOTHING CALLS THIS, and it is checked by reading every
/// `.cu` body and looking for a call. A scan that reads one tree while
/// launchers live in two finds fewer callers than exist, so `Orphaned` gets
/// easier to claim exactly as files move — and it fails SILENTLY, by passing.
/// That is the opposite of what this list is for. Adding the second root costs
/// nothing and removes the incentive.
fn cu_roots() -> Vec<PathBuf> {
    vec![csrc(), Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc")]
}

/// Compile a generated shim against the real headers.
///
/// `-fsyntax-only`: nothing is linked, so this needs neither the built
/// archive nor nvcc. And the only CUDA name these headers use is
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
    kernels_cuda_new::abi::emit_c_shim(&[table], &refs, &kernels_cuda_new::device::jit_dispatched())
        .expect("no entry-point collisions")
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

/// `norm`'s rows, proven the same way `attn`'s are.
///
/// Twenty-eight launchers across seven headers, and the family every
/// other one leans on: a wrong row here is a wrong argument in an arm
/// that four executors reach.
#[test]
fn every_norm_row_states_its_launcher_exactly() {
    let table = kernels_cuda_new::table::norm::KERNELS;
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
    let shim = kernels_cuda_new::abi::emit_c_shim(
        &[table],
        &refs,
        &kernels_cuda_new::device::jit_dispatched(),
    )
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
    let table = kernels_cuda_new::table::mlp::KERNELS;
    let stated = table.iter().filter(|k| !k.operands.is_empty()).count();
    assert_eq!(
        stated,
        table.len(),
        "{} of {} mlp rows are unstated, so the shim silently skips them",
        table.len() - stated,
        table.len()
    );
    let shim = kernels_cuda_new::abi::emit_c_shim(
        &[table],
        // `mlp/swiglu.hpp` was here and is DELETED with `mlp/swiglu.cu` --
        // the whole of `csrc/src/mlp/`. All twelve `table::mlp` rows are in
        // `device::JIT_DISPATCHED`, so `emit_c_shim` skips every one of them
        // and this shim is now a file of includes and no bodies. The
        // assertion above it still bites: a row that stopped stating its
        // operands would fail the count, and the count is what this test is
        // actually for.
        &[],
        &kernels_cuda_new::device::jit_dispatched(),
    )
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
        kernels_cuda_new::table::quant::KERNELS,
        kernels_cuda_new::table::layout::KERNELS,
        kernels_cuda_new::table::gemm::KERNELS,
        kernels_cuda_new::table::moe::KERNELS,
    ];
    let headers = [
        // `quant/dequant_fp4.hpp`, `quant/dequant_fp8.hpp` and
        // `quant/dequant_wna16.hpp` were here and are DELETED with their
        // `.cu` files -- every launcher in the three is
        // `device::JIT_DISPATCHED`, so the shim emitted nothing for any of
        // them and only the includes were left.
        // `quant/dtype_cast.hpp` was here and is DELETED with
        // `quant/dtype_cast.cu`. Both its launchers -- `cast_fp32_to_bf16`
        // and `scale_rows_bf16` -- are `device::JIT_DISPATCHED` rows fired
        // from `driver_cuda::fire::dtype_cast`, so the shim emitted nothing
        // for them and only the include was left.
        // `quant/mxfp4_marlin.hpp` and `quant/quant_bf16_to_mxfp4.hpp` went
        // the same way with their `.cu` files.
        // `quant/quant_bf16_to_fp8.hpp` was the last of them and is DELETED
        // too, which empties `csrc/src/quant/` and takes every `quant`
        // include out of this list. Its note read: *"The ENCODE-side
        // quantizer whose header did land here: its rows arrived with the
        // encode kernels and the shim named functions it had not been shown
        // until this list gained the include."* Those rows were the three in
        // `table/driver_internal.rs`, and they are deleted -- the driver
        // reaches the kernels through `driver_cuda::fire::quant_int8` now.
        // `table::quant::KERNELS` above stays in the table list: its rows are
        // all routed, so the shim emits nothing and there is nothing left to
        // include for them.
        "layout/embed.hpp",
        // `layout/slot_ops.hpp` and `layout/deinterleave.hpp` were here and
        // are DELETED with their `.cu` files. `copy_if_valid_slot` and
        // `split_q_gate_bf16` are JIT rows; `concat_bf16_rows`,
        // `deinterleave_rows_bf16` and `deinterleave_vec_bf16` had no
        // consumer in any language and their `table::layout` rows went with
        // the files -- see `new-horizon.md` §54.
        "gemm/gemm.hpp",
        // `gemm/gemv.hpp` was here and is DELETED with `gemm/gemv.cu`: the
        // kernels are NVRTC's now (`kernels-cuda-new`'s `gemm/gemv` unit)
        // and the launcher is `driver_cuda::fire::gemv`. Nothing in
        // `table::gemm::KERNELS` named it -- the shim only ever included it
        // because the header list is written by hand -- so removing it
        // changes no emitted declaration.
        // `comm/custom_all_reduce.hpp` was here and is DELETED with
        // `comm/custom_all_reduce.cu` and `comm/custom_all_reduce_stub.cpp`,
        // which empties `csrc/src/comm/`. The `.cu` was a 664-line HOST
        // PROGRAM -- zero `__global__`, zero `<<<>>>` -- and the whole
        // lifecycle is `driver_cuda::fire::all_reduce`. Both its symbols are
        // `execution::RUST_SERVED`, so `emit_c_shim` drops both entries and
        // the include had nothing left to declare. `Ty::CustomAllReduce`
        // survives in `kernels::Ty` unchanged: the rows still take an opaque
        // `car` handle, and `KernelSig` is unchanged by design.
        // `moe/dsv4_routing.hpp` and `moe/moe_dispatch.hpp` were here and are
        // DELETED with their `.cu` files. All nine of their symbols are
        // `execution::RUST_SERVED`, so `emit_c_shim` drops every entry and
        // the two includes had nothing left to declare; the launchers are
        // `driver_cuda::fire::dsv4_routing` and
        // `driver_cuda::fire::moe_dispatch`. `moe_dispatch.hpp`'s one
        // non-launcher export, `moe_aligned_block()`, had no C++ caller and
        // is `fire::moe_dispatch::moe_aligned_block`.
        "moe/moe_grouped_gemm.hpp",
        "moe/flashinfer_moe.hpp",
        // `sample/argmax.hpp` was here and is DELETED with
        // `sample/argmax.cu`, the whole of `csrc/src/sample/`. Its one
        // surviving symbol is `execution::RUST_SERVED`, so `emit_c_shim`
        // drops the entry and the include had nothing left to declare;
        // `driver_cuda::fire::lm_head_argmax` is the launcher.
        "moe/topk_softmax.hpp",
    ];
    let shim = kernels_cuda_new::abi::emit_c_shim(
        &tables,
        &headers,
        &kernels_cuda_new::device::jit_dispatched(),
    )
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
    let table = kernels_cuda_new::table::ssm::KERNELS;
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
    let shim = kernels_cuda_new::abi::emit_c_shim(
        &[table],
        // NO HEADERS AT ALL, AND THAT IS THE END STATE RATHER THAN A GAP.
        // The comment this replaces read: *"ONE header, and the eleven that
        // used to need the other three are the reason there is one.
        // `ssm/{causal_conv1d,gated_delta_net,kda}.hpp` are deleted with
        // their `.cu`s: every launcher they declared is
        // `execution::RUST_SERVED` now, `emit_c_shim` skips a `RUST_SERVED`
        // row, and a header declaring nothing the shim calls is an include
        // that can only break the build later."*
        //
        // `ssm/nemotron_h.hpp` has now gone the same way and `csrc/src/ssm/`
        // with it, so this shim is EMPTY: every `table::ssm` row is either
        // `device::JIT_DISPATCHED` or `execution::RUST_SERVED`. The test
        // stays because it still makes a claim -- the `stated` assertion
        // above is not vacuous, and an ssm row that ever comes back
        // unrouted will emit a body with no header to declare it and fail
        // here rather than at link.
        &[] as &[&str],
        &kernels_cuda_new::device::jit_dispatched(),
    )
    .expect("no entry-point collisions");
    if let Err(err) = compile(&shim) {
        panic!(
            "the generated shim does not compile, so a row misstates its \
             launcher:\n{err}"
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
    let table = kernels_cuda_new::table::attn::KERNELS;
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
    /// **Nothing calls it at all** — not the driver, not a sibling `.cu`, not
    /// `dsl::cuda`. It had a row until `new-horizon.md` §28.4 measured the
    /// row as a second name for a job a reached row already does, or until
    /// §38 measured the row's whole consumer set as empty, and the row went;
    /// the launcher stays because §10.10 says a launcher goes only when its
    /// WHOLE consumer set has gone.
    ///
    /// Checked exactly as `KernelsInternal` is, and then harder: the driver
    /// must not mention it AND no `.cu` in `csrc` may call it. And then
    /// harder again — it must name at least one [`Keeper`], and every
    /// keeper it names is opened and read. This doc used to assert the
    /// keeper in prose, for all of them at once: *"its last consumer is
    /// `sources.rs`' `EXPECTED = 401` `<<<>>>` census"*. §38.10 measured
    /// that and it was true of **one of the three** entries it was written
    /// for. Two of them hold no `<<<>>>` at all. That is the §34 lesson
    /// arriving one level down: a reason stated once for a list is not a
    /// reason checked per entry.
    Orphaned(&'static [Keeper]),
}

/// **What is still holding an [`NoRow::Orphaned`] launcher, checked.**
///
/// An exception list is a silencer unless the exception costs something to
/// state. `Orphaned` already pays once — nothing may call it, and both the
/// driver and every `.cu` body are read to prove that. A keeper is the
/// second half and the more useful one: it says what a DELETION would cost,
/// which is the number a reader actually needs, and it is wrong loudly
/// rather than quietly.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Keeper {
    /// **Its body holds `<<<>>>` that `kernels-cuda/tests/sources.rs`'
    /// `EXPECTED` counts.** Deleting it moves 401, which is the one number
    /// this session is not allowed to move without saying so.
    Launches,
    /// **It is the ONLY caller of the named launcher.** Deleting it is a
    /// two-launcher edit, not a one-function edit: the named one goes from
    /// `KernelsInternal` to `Orphaned` in the same commit.
    SoleCallerOf(&'static str),
    /// **A probe cites its defining `.cu` by path**, so re-adding the row is
    /// a one-line edit rather than an archaeology exercise. §31's precedent:
    /// the probe is what makes a row cheap to re-add, and a launcher with no
    /// declaration would not be cheap at all.
    Probed(&'static str),
    /// **Nothing holds it.** The honest empty answer, and it is the one that
    /// says *delete me* — so it is checked to be TRUE rather than
    /// convenient: a `Backlog` entry whose body launches is refused, because
    /// that is the lie that would make a deletion look free when it costs a
    /// `<<<>>>`.
    Backlog,
}

/// Every launcher `attn`'s headers declare is a row, or is one of five
/// documented kinds of not-a-row.
///
/// The rope pilot could assert the flat thing — every declaration has a row —
/// because for `rope` it is true. For `attn` it is false BY DESIGN, and a
/// test that asserted it anyway would have to be deleted rather than
/// answered. What is actually load-bearing is that no launcher joins these
/// headers without someone deciding which kind it is: **78 declarations
/// against 40 rows is not a gap to close, it is 38 decisions**, and this is
/// where they are written down. Seven of the 38 arrived at once, when
/// §28.4's audit measured seven rows as duplicates and deleted them, and two
/// more when §38 measured two rows' consumer sets as empty — a decision
/// moving from "row" to "stated reason" is the same decision, restated, and
/// the count above moves one for one as it does.
///
/// `KernelsInternal` is not taken on trust — the claim is "the driver never
/// calls this", the driver's sources are next door, and so it is checked.
/// `Orphaned` is the stronger claim and gets the stronger check: nothing at
/// all calls it, so both the driver AND every `.cu` body in `csrc` are read.
#[test]
fn every_attn_launcher_is_a_row_or_a_stated_exception() {
    #[rustfmt::skip]
    let exceptions: &[(&str, NoRow)] = &[
        ("plan_attention_flashinfer_decode_bf16",       NoRow::Prepare),
        ("plan_attention_flashinfer_prefill_bf16",      NoRow::Prepare),
        ("plan_attention_flashinfer_prefill_sm90_bf16", NoRow::Prepare),
        ("plan_attention_mla_bf16",                     NoRow::Prepare),
        // `prepare_attention_xqa_decode_bf16` STOOD HERE and is gone, with
        // the exception, because the launcher is gone. It is
        // `driver-cuda/src/fire/xqa.rs::prepare_decode` now -- Rust firing
        // `attn/attention_xqa.cuh` through NVRTC -- and its `__global__` was
        // the last one the `kernels-cuda` archive held. The exception said
        // `NoRow::Prepare`: no row names it because a `Prepare` is an
        // obligation the TABLE states and the DRIVER discharges, and this is
        // the first one where the driver actually does. Removed rather than
        // retargeted: `mentions_word` below reads all of
        // `crates/driver-cuda/src`, so leaving the name here while
        // `fire/xqa.rs` cites it in a comment would fail the `Orphaned` arm
        // for a launcher that is not orphaned but ported.
        ("set_decode_plan_int_base",                    NoRow::Prepare),
        ("split_qkv_bf16",                              NoRow::DriverInternal),
        // `split_qkv_bf16_devwin` stood here and is GONE with
        // `attn/split_packed.cu`. Its row is no longer `driver_internal`
        // either -- it moved to `table::attn` so `RUST_SERVED` could take it.
        // `pack_dense_mask` and `pack_structured_mask` stood here and are
        // GONE with `attn/pack_dense_mask.cu`, its `.hpp` and their two
        // `table::driver_internal` rows. Empty consumer set on all five
        // channels; not ported, per §60.1.
        // `copy_kv_cells_bf16` STOOD HERE and is ported, so it is removed
        // rather than retargeted — `set_decode_plan_int_base` above records
        // the same rule and the same reason. `mentions_word` below reads all
        // of `crates/driver-cuda/src`, and `fire/kv_paged.rs` both defines
        // `copy_kv_cells_bf16` and cites the name in its doc, so leaving the
        // entry here would fail the `Orphaned` arm for a launcher that is not
        // orphaned but gone. Its `table/driver_internal.rs` row went in the
        // same edit; that file carries the evidence.
        ("attention_flashinfer_prefill_bf16",           NoRow::KernelsInternal),
        // ORPHANED BY THIS SESSION'S OWN DELETION, and left standing on
        // purpose. `attention_flashinfer_prefill_custom_bf16` was
        // `KernelsInternal` and the sibling that called it was
        // `attention_flashinfer_prefill_custom`, which §44 deleted as a
        // closed `Backlog`. So the pair went from "one dead caller" to "no
        // caller" in one edit — §41's orphaned-at-one-remove, produced
        // rather than found, and the reason a transitive audit has to be
        // re-run after a deletion and not only before.
        //
        // It is NOT deleted with it, and the asymmetry is the point. The two
        // `Backlog` entries §44 collected were host code this tree can
        // restate: a `dim3`, a guard, a forward. This one is a FlashInfer
        // custom-mask prefill dispatch — a `switch` over `kernels.def` into
        // vendored templates whose host geometry is precisely what §44
        // measured as unstateable in the `LaunchRule` vocabulary. A `Backlog`
        // keeper claims "nothing holds it", and something does hold this: the
        // cost of writing it again. That is not a `Keeper` variant and must
        // not become one for a single entry (§10.5), so it is stated here in
        // the kind that is still literally true — no driver call, no row —
        // and handed to whoever ports the FlashInfer prefill.
        ("attention_flashinfer_prefill_custom_bf16",    NoRow::KernelsInternal),
        ("dispatch_attention_flashinfer_decode_capture_bf16", NoRow::KernelsInternal),
        ("dispatch_attention_flashinfer_prefill_custom_bf16", NoRow::KernelsInternal),
        // `write_mla_to_pages_bf16` stood here and is GONE with
        // `attn/mla_paged.cu`. It was dead on all five channels and its only
        // caller was the `write_mla_to_pages` forwarder in the same file;
        // that forwarder is now `driver-cuda/src/fire/mla_paged.rs` and holds
        // the `<<<>>>` directly.
        // ── §44's deletions ──────────────────────────────────────────
        // FOURTEEN entries stood here and are gone with their launchers.
        // Two `Warmup` (`xqa_decode_bf16_warmup_current_device` and its
        // gqa5 half), eleven `KernelsInternal`
        // (`attention_mtp_history_bf16`, `attention_naive_bf16`,
        // `attention_naive_paged_custom`, `attention_naive_paged_decode`,
        // `attention_xqa_decode_bf16`, `add_ape_f32`,
        // `attention_compressed_bf16`, `average_pool_bf16`,
        // `dsv4_compress_gather_bf16`, `gated_softmax_pool_bf16`,
        // `write_kv_to_pages_at_positions_bf16`) and two `Orphaned`
        // (`write_kv_to_pages_bf16_devwin`,
        // `attention_mtp_paged_history_bf16`).
        //
        // `KernelsInternal` says "only sibling `.cu` files call it", and for
        // all eleven the audit measured WHICH siblings: none that anything
        // reaches. `attention_compressed_bf16` is the shape of it — five
        // unpaged DSv4 launchers where the only caller of four of them was
        // the fifth. A cycle of dead callers reads as `KernelsInternal` from
        // inside and as nothing at all from outside, which is why the claim
        // had to be checked transitively and not one hop.
        //
        // The two `Orphaned` are §38's own keepers, honoured rather than
        // ignored: `Keeper::Launches` said deleting them moves
        // `kernels-cuda/tests/sources.rs`' `EXPECTED`, and it does — the
        // number is RE-DERIVED from the tree in the same change, which is
        // what that keeper was asking for.
        //
        // `merge_attention_states_bf16` below is NOT among them, for the
        // reason its own keeper gives, and it still holds.
        // §28.4's seven: each had a row, each row was a second name for a
        // job a reached row already does, and none of the seven wrappers in
        // `dsl.rs` had a caller. The four below are still reached from a
        // sibling `.cu`; the three under `Orphaned` are reached from
        // nowhere at all.
        // `("write_kv_to_pages_bf16", NoRow::KernelsInternal)` WAS HERE and
        // is deleted with the launcher. Same reason the list already records
        // for `copy_kv_cells_bf16` and `set_decode_plan_int_base`:
        // `mentions_word` reads all of `driver-cuda/src`, and
        // `fire::kv_paged::write_kv_to_pages_bf16` is that word — a ported
        // name left standing here fails the `Orphaned` arm for a launcher
        // that is not orphaned but simply no longer C++.
        ("dispatch_attention_flashinfer_decode_bf16",   NoRow::KernelsInternal),
        ("dispatch_attention_flashinfer_prefill_sm90_bf16", NoRow::KernelsInternal),
        ("attention_naive_paged_bf16",                  NoRow::KernelsInternal),
        // §38.10's two `Backlog` entries — "Nothing holds them. They are
        // deletions waiting for their evidence" — are gone, with their
        // launchers. The evidence was the `Backlog` claim itself, which this
        // test already checks (`!body.contains("<<<")`), plus the audit
        // finding no caller on any channel. A `Backlog` that is never
        // collected is an exception list being the thing it exists to stop.
        // §38's two: not duplicates — each had a real, still-true wall in
        // front of it (`merge_attention_states_bf16` a two-geometry host
        // `if` in `cascade.cuh:638-666`, `attention_mtp_paged_history_bf16`
        // a three-way switch on `max_global_tokens + history_steps > 8192`)
        // and a consumer set that was empty on every channel §28.2 lists.
        // A wall in front of a door nobody opens is not a wall, so the rows
        // went and the launchers stayed. Their last consumers are named:
        // `attention_mtp_paged_history_bf16` is the ONLY caller of
        // `attention_mtp_history_bf16` above, so deleting it would orphan
        // two launchers and move `sources.rs`' `EXPECTED` off 401;
        // `merge_attention_states_bf16` is cited by file AND LINE from
        // `kernels-cuda-new/examples/vendor_probe.rs:199`, which is what
        // makes re-adding the row cheap if a model ever wants it.
        ("merge_attention_states_bf16",
            NoRow::Orphaned(&[Keeper::Probed("kernels-cuda-new/examples/vendor_probe.rs")])),
    ];

    let declared = declared_launchers();
    // 78 before §44, 58 after. The floor moves DOWN as the archive is
    // retired and it must, because it is a tripwire on the SCANNER and not
    // on the tree: it fires when `declared_launchers` stops finding
    // declarations for a reason that is not a deletion — a header renamed, a
    // `void` on the wrong line, a directory moved. Left at 77 it would have
    // fired on twenty honest deletions and said "the shape assumption broke"
    // about a change that broke nothing, which is the failure mode a floor
    // has instead of a bug.
    //
    // 55 -> 50, AND THE SENTENCE THAT PAYS FOR IT. The scan finds 53 now.
    // Two of the five that left are not this pass's -- 55 was already above
    // the tree when this pass started, because concurrent passes had taken
    // declarations out of `attn/*.hpp` without lowering the number that
    // counts them, so this assertion was RED AT HEAD and would have been
    // blamed on whoever touched it next. One is this pass's:
    // `prepare_attention_xqa_decode_bf16` left `attention_xqa.hpp` for
    // `driver-cuda/src/fire/xqa.rs`. 50 rather than 53 deliberately: `attn/`
    // is being emptied and a floor set flush against the current count is a
    // floor that has to be edited by every deletion, which is how it came to
    // be wrong. Three of headroom is the width of one more file's worth of
    // launchers, and below 50 the right question really is whether the
    // scanner still works.
    assert!(
        declared.len() >= 50,
        "the scan found {} declarations, so its shape assumption broke",
        declared.len()
    );

    let has_row = |n: &str| {
        kernels_cuda_new::table::attn::KERNELS
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
    assert!(
        stale.is_empty(),
        "exception for a launcher no header declares: {stale:?}"
    );

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
        .filter(|(_, why)| *why == NoRow::KernelsInternal || matches!(why, NoRow::Orphaned(_)))
        .map(|(n, _)| *n)
        .filter(|n| mentions_word(&driver_text, n))
        .collect();
    assert!(
        wrong.is_empty(),
        "called `KernelsInternal` but the driver calls it, so it is really \
         DriverInternal or a missing row: {wrong:?}"
    );

    // The `Orphaned` claim is the stronger one and it is checked in full:
    // NOTHING calls these. The driver was ruled out above; here every `.cu`
    // in `csrc` is read and a call — a mention that is not the declaration,
    // the definition, or a string literal — is a failure, because a launcher
    // with a live caller is not orphaned, it is `KernelsInternal`.
    let mut cu_text = String::new();
    for root in cu_roots() {
        collect_cu_bodies(&root, &mut cu_text);
    }
    assert!(
        cu_text.len() > 100_000,
        "only {} bytes of .cu body found, so the orphan check is vacuous",
        cu_text.len()
    );
    let not_orphans: Vec<&str> = exceptions
        .iter()
        .filter(|(_, why)| matches!(why, NoRow::Orphaned(_)))
        .map(|(n, _)| *n)
        .filter(|n| mentions_word(&cu_text, n))
        .collect();
    assert!(
        not_orphans.is_empty(),
        "called `Orphaned` but a sibling `.cu` calls it, so it is really \
         `KernelsInternal`: {not_orphans:?}"
    );

    // And every `Orphaned` names at least one `Keeper`, and every keeper is
    // opened and read. This is the half that makes the exception cost
    // something to state: `Orphaned` says what does NOT reach it, a keeper
    // says what a deletion WOULD cost, and only the second is a number.
    let mut checked = 0;
    for (name, why) in exceptions {
        let NoRow::Orphaned(keepers) = why else {
            continue;
        };
        assert!(
            !keepers.is_empty(),
            "`{name}` is `Orphaned` and names no keeper. Say what still holds it, or say \
             `Keeper::Backlog` and mean it"
        );
        let (file, body) = cu_definition(name).unwrap_or_else(|| {
            panic!("`{name}` is `Orphaned` but no `.cu` under `csrc` defines it")
        });
        for keeper in *keepers {
            match keeper {
                Keeper::Launches => assert!(
                    body.contains("<<<"),
                    "`{name}` claims `Launches` and its body in {file} holds no `<<<>>>`, so \
                     `sources.rs`' EXPECTED would not move if it went. That is a `Backlog`"
                ),
                Keeper::SoleCallerOf(target) => {
                    let inside = call_sites_in(&body, target);
                    let everywhere = orphan_call_sites(target);
                    assert!(
                        inside > 0,
                        "`{name}` claims to be the sole caller of `{target}` and its body in \
                         {file} does not call it at all"
                    );
                    assert_eq!(
                        inside, everywhere,
                        "`{name}` claims to be the SOLE caller of `{target}`, and `{target}` \
                         is called {everywhere} time(s) across `csrc` of which {inside} are \
                         here. Deleting `{name}` would not orphan `{target}` after all"
                    );
                }
                Keeper::Probed(probe) => {
                    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(probe);
                    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
                        panic!("`{name}` cites the probe `{probe}`, which does not open")
                    });
                    let leaf = Path::new(&file)
                        .file_name()
                        .and_then(|f| f.to_str())
                        .expect("a .cu file name");
                    assert!(
                        text.contains(leaf),
                        "`{name}` claims `{probe}` cites it, and that file never names \
                         `{leaf}`. A citation nobody opens is the thing this whole column \
                         exists to stop being"
                    );
                }
                Keeper::Backlog => {
                    assert_eq!(
                        *keepers,
                        &[Keeper::Backlog],
                        "`{name}` says `Backlog` beside a real keeper; `Backlog` means the \
                         list is empty, so it may not pad one"
                    );
                    assert!(
                        !body.contains("<<<"),
                        "`{name}` says nothing holds it and its body in {file} launches. \
                         That is the one `Backlog` lie worth checking: it would make a \
                         deletion look free when it costs a `<<<>>>` off EXPECTED = 401"
                    );
                }
            }
            checked += 1;
        }
    }
    // ONE, and it used to be six. The other five were collected rather than
    // lost: two `Backlog` entries became deletions, `write_kv_to_pages_bf16_devwin`'s
    // `Launches` became a deletion and a re-derived `EXPECTED`, and
    // `attention_mtp_paged_history_bf16`'s `Launches` + `SoleCallerOf` pair
    // became a two-launcher deletion of exactly the kind that keeper
    // described. What is left is `merge_attention_states_bf16`'s
    // `Probed`, which is the one keeper naming a consumer that still exists.
    //
    // The floor is worth keeping at one rather than deleting: its job is to
    // catch the `Orphaned` column going empty of keepers while entries
    // remain, which would mean the second half of the claim had quietly
    // stopped being made. It is not a target.
    assert!(
        checked >= 1,
        "only {checked} keeper(s) read, so the keeper check is vacuous"
    );
}

/// The `.cu` body of a launcher: from its `void <name>(` line to the first
/// line that is exactly `}`, which is how every launcher in this tree closes.
fn cu_definition(name: &str) -> Option<(String, String)> {
    fn walk(dir: &Path, name: &str, out: &mut Option<(String, String)>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk(&p, name, out);
            } else if p.extension().and_then(|e| e.to_str()) == Some("cu")
                && out.is_none()
                && let Ok(text) = std::fs::read_to_string(&p)
            {
                let lines: Vec<&str> = text.lines().collect();
                let opens = format!("void {name}(");
                for (i, line) in lines.iter().enumerate() {
                    if line.trim_start().starts_with(&opens) {
                        let end = lines[i..]
                            .iter()
                            .position(|l| l.trim_end() == "}")
                            .map_or(lines.len(), |k| i + k + 1);
                        *out = Some((p.display().to_string(), lines[i..end].join("\n")));
                        return;
                    }
                }
            }
        }
    }
    let mut out = None;
    for root in cu_roots() {
        walk(&root, name, &mut out);
        if out.is_some() {
            break;
        }
    }
    out
}

/// Calls to `target` inside one body — `target(`, not on a comment line and
/// not the line that declares or defines it.
fn call_sites_in(body: &str, target: &str) -> usize {
    let opens = format!("void {target}(");
    let call = format!("{target}(");
    body.lines()
        .filter(|l| {
            let t = l.trim_start();
            !t.starts_with("//") && !t.starts_with(&opens) && t.contains(&call)
        })
        .count()
}

/// The same count over every `.cu` under `csrc`, so "sole" can be checked
/// rather than believed.
fn orphan_call_sites(target: &str) -> usize {
    let mut text = String::new();
    for root in cu_roots() {
        collect_cu_bodies(&root, &mut text);
    }
    call_sites_in(&text, target)
}

/// Every `.cu` under `csrc`, with its own declarations, definitions and
/// string literals removed — what is left is call sites.
///
/// A launcher's definition names it and so does its header; neither is a
/// caller, and a check that could not tell them apart would report every
/// launcher in the tree as reached. The three lines dropped are: a line
/// opening `void <name>(`, which is a definition or a declaration, and any
/// line whose occurrence of the name sits inside a `"`-quoted string, which
/// is an error message.
fn collect_cu_bodies(dir: &Path, out: &mut String) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_cu_bodies(&p, out);
        } else if p.extension().and_then(|e| e.to_str()) == Some("cu")
            && let Ok(t) = std::fs::read_to_string(&p)
        {
            for line in t.lines() {
                let trimmed = line.trim_start();
                if trimmed.starts_with("void ") || trimmed.starts_with("//") {
                    continue;
                }
                // Strip `"`-quoted spans, so an error message naming the
                // launcher it is thrown from is not read as a call.
                let mut kept = String::with_capacity(line.len());
                let mut in_str = false;
                for c in line.chars() {
                    if c == '"' {
                        in_str = !in_str;
                        continue;
                    }
                    if !in_str {
                        kept.push(c);
                    }
                }
                out.push_str(&kept);
                out.push('\n');
            }
        }
    }
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
            let Some(rest) = line.strip_prefix("void ") else {
                continue;
            };
            let Some((name, _)) = rest.split_once('(') else {
                continue;
            };
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
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
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
        let after = hay[i + needle.len()..]
            .chars()
            .next()
            .is_none_or(|c| !ident(c));
        before && after
    })
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
/// `gemm.rs`, their launchers lived in `csrc/src/comm/` (which is now
/// DELETED — `fire::all_reduce` is the launcher and both rows are
/// `execution::RUST_SERVED`), and no
/// `every_*_row_states_its_launcher_exactly` case owns that directory. Here
/// the include set is READ from the tree rather than typed, so a row whose
/// family has a directory is covered the moment the directory exists — and
/// stops being included the moment it does not, which is the same property
/// read the other way.
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
            e.path()
                .is_dir()
                .then(|| e.file_name().into_string().ok())?
        })
        .collect();
    dirs.sort();
    for dir in &dirs {
        headers.extend(headers_in(dir));
    }
    assert!(
        dirs.iter().any(|d| d == "comm"),
        "the directory scan found no `comm/`, so the case that motivated it \
         would pass vacuously: {dirs:?}"
    );

    let tables: &[&'static [kernels::KernelSig]] = &[
        kernels_cuda_new::table::KERNELS,
        kernels_cuda_new::table::driver_internal::DRIVER_KERNELS,
    ];
    let refs: Vec<&str> = headers.iter().map(String::as_str).collect();
    let shim = kernels_cuda_new::abi::emit_c_shim(
        tables,
        &refs,
        &kernels_cuda_new::device::jit_dispatched(),
    )
    .expect("no entry-point collisions");

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
        kernels_cuda_new::record!(KvCacheLayerView => "::pie_cuda_driver::KvCacheLayerView" {
            layer, source_layer, num_pages, page_size, num_kv_heads, head_dim,
            scheme, storage_dtype, block_size,
            k_pages, v_pages, k_scales, v_scales, k_bf16_pages, v_bf16_pages,
            k_env_min, k_env_max,
            hnd_layout, native_bf16,
        }),
        kernels_cuda_new::record!(AttentionWorkspaceView => "::pie_cuda_driver::AttentionWorkspaceView" {
            float_buffer, float_bytes, int_buffer, int_bytes, page_locked_int,
        }),
        kernels_cuda_new::record!(MlaCacheLayerView => "::pie_cuda_driver::MlaCacheLayerView" {
            layer, num_pages, page_size, kv_lora_rank, qk_rope_head_dim,
            ckv_pages, kpe_pages,
        }),
        kernels_cuda_new::record!(HopperPrefillPlan => "::pie_cuda_driver::kernels::attn::HopperPrefillPlan" {
            qo_tile_indices_offset, qo_indptr_offset, kv_indptr_offset,
            qo_len_offset, kv_len_offset, head_indices_offset,
            work_indptr_offset, batch_indices_offset,
            same_schedule_for_all_heads,
            total_tokens, num_requests, num_q_heads, num_kv_heads, head_dim,
            page_size, window_left, causal, valid,
        }),
        kernels_cuda_new::record!(YarnOriginalParams => "::pie_cuda_driver::kernels::attn::YarnOriginalParams" {
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
    // `attn/mla_paged.hpp` stood here and is DELETED with its `.cu`. The
    // `MlaCacheLayerView` mirror is still asserted through
    // `attn/mla_cache_view.hpp`, which is where the record is DEFINED; the
    // launcher header only included it.
];

/// A `#[repr(C)]` mirror really does have the C++ record's layout.
///
/// This is the claim that decides whether a POD operand is a port or a
/// wrapper. If it holds, `KvCacheLayerView` crosses the boundary as itself —
/// no accessor shims, no field-by-field constructor, no copy — and every
/// other descriptor in the launcher surface is the same kind of thing.
#[test]
fn the_mirrors_have_the_layout_the_cpp_has() {
    let tu = kernels_cuda_new::abi::emit_layout_assertions(&records(), MIRROR_HPPS);
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
        compile(&kernels_cuda_new::abi::emit_layout_assertions(
            &rs,
            MIRROR_HPPS,
        ))
    };

    let cases: Vec<(&str, Mutation)> = vec![
        (
            "the record is one byte bigger",
            Box::new(|r: &mut Record| r.size += 1),
        ),
        (
            "the record is over-aligned",
            Box::new(|r: &mut Record| r.align *= 2),
        ),
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
    let tu = kernels_cuda_new::abi::emit_layout_assertions(&rs, MIRROR_HPPS)
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
    let tu = kernels_cuda_new::abi::emit_layout_assertions(&rs, MIRROR_HPPS);
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
    kernels_cuda_new::abi::emit_c_shim(&[table], &refs, &kernels_cuda_new::device::jit_dispatched())
        .expect("no entry-point collisions")
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
    prove_family(
        "sample",
        kernels_cuda_new::table::sample::KERNELS,
        &headers_in("sample"),
    );
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
///   rather than under it — a driver-side namespace for the collectives.
///   Each HAS a statable signature; what is missing is which side of the
///   boundary the symbol admits to being on.
///
///   It also closes the OTHER way, and did once: `marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16`
///   sat here for `::pie_cuda_driver::marlin_moe`, and the row is now gone —
///   nothing called `dsl::cuda::mxfp4_moe_gemm_w4a16`, and
///   `weights/plan.rs`'s `native_mxfp4_moe = false` means nothing plans the
///   lowering it served. An unstated row can be answered by deleting it.
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
        ("qwen35_verify_stash_store",                    Unstated::NotACppFunction),
        ("qwen35_verify_stash_load",                     Unstated::NotACppFunction),
        ("pie_lora_qkv_correction",                      Unstated::NotACppFunction),
    ];

    let unstated: Vec<&str> = kernels_cuda_new::table::KERNELS
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
    // not the shim's). BOTH ROW SETS ARE GONE: all three towers are Rust
    // and `driver-cuda/csrc/vision/` is deleted, so these two headers now
    // declare launchers nothing rows and nothing calls. They stay pushed
    // because `prove_family` proves ROW -> declaration and not the reverse,
    // and a header list that is a superset costs a parse.
    headers.push("vision/qwen3_vl_tower_c.hpp".into());
    headers.push("vision/gemma4_towers_c.hpp".into());
    prove_family(
        "driver-internal",
        kernels_cuda_new::table::driver_internal::DRIVER_KERNELS,
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
            let Some(rest) = returns.iter().find_map(|r| line.strip_prefix(r)) else {
                continue;
            };
            let Some((name, _)) = rest.split_once('(') else {
                continue;
            };
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
    /// **Nothing calls it at all** — not the driver, not a sibling `.cu`, not
    /// the emitter, not `dsl::cuda`. It had a row until `new-horizon.md`
    /// §28.4 measured the row as a second name for a job a reached row
    /// already does, and the row went; the launcher stays because §10.10
    /// says a launcher goes only when its WHOLE consumer set has gone, and
    /// this one's last consumer is `kernels-cuda/tests/sources.rs`'
    /// `EXPECTED = 401` `<<<>>>` census. That is a backlog entry, and this
    /// is where it is written down.
    ///
    /// The same kind as `attn`'s [`NoRow::Orphaned`] and checked the same
    /// way, in full: the driver must not mention it, no `.cu` in `csrc` may
    /// call it, and no emitter may choose it.
    Orphaned,
}

/// Every launcher `norm`'s headers declare is a row, or is one of three
/// documented kinds of not-a-row.
///
/// The `attn` twin above owns the long rationale; what is specific to `norm`
/// is the arithmetic — 32 declarations against 24 rows is 8 decisions — and
/// that ALL THREE exception kinds are checkable, so none is taken on trust.
/// `rmsnorm_bf16` is the loud case here the way `split_qkv_bf16` was for
/// `attn`: 1,337 call sites, every one of them emitter-chosen.
#[test]
fn every_norm_launcher_is_a_row_or_a_stated_exception() {
    #[rustfmt::skip]
    let exceptions: &[(&str, NormNoRow)] = &[
        ("rmsnorm_bf16",                    NormNoRow::EmitterChosen),
        ("add_bias_bf16",                   NormNoRow::EmitterChosen),
        ("rmsnorm_gated_fp32_in_bf16",      NormNoRow::EmitterChosen),
        // `rmsnorm_gemma_bf16` was `EmitterChosen` here until §43. It still
        // is chosen by `lower.rs`; what changed is that the SYMBOL is routed
        // (`device::JIT_DISPATCHED`), so the shim forwards to nothing, no
        // root reached the ahead-of-time launcher, and §43 deleted it. A
        // routed row needs no exception because it declares no launcher.
        //
        // `rmsnorm_bf16_tuned` and `rmsnorm_rasr_tuned` were the two
        // `AutotunerProbe`s and `add_bias_bf16_strided` /
        // `residual_add_scale_rmsnorm_bf16` the two `Orphaned`s. §41's
        // transitive audit measured all four as reachable from no root at
        // all and §43 deleted them. The `Orphaned` doc named their last
        // consumer as `sources.rs`' `<<<>>>` census; a census counts, it does
        // not call, and that is the whole reason the kind existed.
        //
        // All three kinds stay in the enum. `EmitterChosen` still has three
        // members, and the other two are checkable claims that cost nothing
        // while empty and would otherwise have to be reinvented -- with the
        // checks -- the next time a launcher earns one.
    ];

    // `bool ` is load-bearing: the probes are the only non-`void` launchers,
    // and a void-only scan would never see them.
    let declared = declared_in("norm", &["void ", "bool "]);

    // A SHAPE CHECK, not a census. This asserted a floor on the count, and
    // the count now falls every time a kernel migrates: `scalar_mul`'s
    // launcher went with its `.cu`, then all six of `altup_aux`'s
    // (`new-horizon.md` §10.8, §10.11). A number that has to be edited on
    // every migration is a number nobody reads -- and the thing it was
    // guarding against is a scan that silently matched NOTHING.
    //
    // So the guard names two launchers that are still there instead. Both
    // are `void` since §43: `rmsnorm_bf16_tuned` was the `bool` witness and
    // it was the last non-`void` launcher `norm` had, so the `"bool "` prefix
    // above is now a scan for a shape that is absent rather than a shape that
    // is present. It stays because a probe is exactly the thing that comes
    // back, and a scan that stopped looking would not notice the day it does.
    for expected in ["rmsnorm_bf16", "rmsnorm_strided_bf16"] {
        assert!(
            declared.iter().any(|d| d == expected),
            "the scan found {} declarations and `{expected}` is not among \
             them, so its shape assumption broke",
            declared.len()
        );
    }

    let has_row = |n: &str| {
        kernels_cuda_new::table::norm::KERNELS
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
    assert!(
        stale.is_empty(),
        "exception for a launcher no header declares: {stale:?}"
    );

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

    // The `Orphaned` claim's first half, against the SAME emitter text: a
    // launcher an emitter chooses is `EmitterChosen`, not an orphan, and
    // `add_bias_bf16_strided` sits one whole-word match away from
    // `add_bias_bf16`, which `lower.rs` does choose. `mentions_word` is what
    // keeps those two apart.
    let chosen_after_all: Vec<&str> = exceptions
        .iter()
        .filter(|(_, why)| *why == NormNoRow::Orphaned)
        .map(|(n, _)| *n)
        .filter(|n| mentions_word(&emitter_text, n))
        .collect();
    assert!(
        chosen_after_all.is_empty(),
        "called `Orphaned` but an emitter chooses it, so it is really \
         `EmitterChosen`: {chosen_after_all:?}"
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
    let called: Vec<&&str> = probes
        .iter()
        .filter(|n| mentions_word(&driver_text, n))
        .collect();
    assert!(
        called.is_empty(),
        "called `AutotunerProbe` but the driver mentions it, so its \"zero \
         driver call sites\" is stale: {called:?}"
    );

    // The `Orphaned` claim's other two halves, on the same two texts the
    // kinds above are checked against plus one more. The driver was just
    // read, so reuse it; then every `.cu` body in `csrc`, which is the half
    // `KernelsInternal` exists to name.
    let orphans: Vec<&str> = exceptions
        .iter()
        .filter(|(_, why)| *why == NormNoRow::Orphaned)
        .map(|(n, _)| *n)
        .collect();
    let driver_calls: Vec<&&str> = orphans
        .iter()
        .filter(|n| mentions_word(&driver_text, n))
        .collect();
    assert!(
        driver_calls.is_empty(),
        "called `Orphaned` but the driver mentions it, so it is really \
         `DriverInternal` or a missing row: {driver_calls:?}"
    );
    let mut cu_text = String::new();
    for root in cu_roots() {
        collect_cu_bodies(&root, &mut cu_text);
    }
    assert!(
        cu_text.len() > 100_000,
        "only {} bytes of .cu body found, so the orphan check is vacuous",
        cu_text.len()
    );
    let not_orphans: Vec<&&str> = orphans
        .iter()
        .filter(|n| mentions_word(&cu_text, n))
        .collect();
    assert!(
        not_orphans.is_empty(),
        "called `Orphaned` but a sibling `.cu` calls it, so it is really \
         `KernelsInternal`: {not_orphans:?}"
    );
}

/// Concatenate every file named `name` under `dir`, recursively.
///
/// [`collect_sources`] filters by extension; this filters by exact file
/// name, because "the emitter sources" is a claim about which FILES choose
/// kernels, not about a language.
fn collect_files_named(dir: &Path, name: &str, out: &mut String) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
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
///
/// # `Mxfp4RowSelect` left this TU, and did not leave the test
///
/// It was checked here against `quant/mxfp4_marlin.hpp`, which is DELETED
/// with `quant/mxfp4_marlin.cu`. Its three values did not stop existing: they
/// cross into device code as a plain `int` (`families::quant:481`), and
/// `kernels-cuda-new/csrc/src/quant/mxfp4_marlin.cuh:70-72` names them
/// `kRowSelectIdentity`, `kRowSelectEven` and `kRowSelectOdd` precisely so
/// the mapping is one grep. So the claim moved rather than dying:
/// [`the_mxfp4_row_select_mirror_matches_the_device_text`] reads those three
/// lines out of the `.cuh` and compares them to the Rust enum. It is a weaker
/// proof than a `static_assert` — text, not a compiler — and it is the
/// strongest one available once the host header is gone, which is the trade
/// every row that leaves the archive makes.
#[test]
fn the_enum_mirrors_carry_the_cpp_discriminants() {
    use driver_cuda::bind::abi::MoeActivation;
    let tu = format!(
        "#include <cstdint>\n\
         #include \"moe/flashinfer_moe.hpp\"\n\
         using ::pie_cuda_driver::kernels::moe::MoeActivation;\n\
         static_assert(sizeof(MoeActivation) == {});\n\
         static_assert(static_cast<int>(MoeActivation::Relu2) == {});\n\
         static_assert(static_cast<int>(MoeActivation::Swiglu) == {});\n\
         static_assert(static_cast<int>(MoeActivation::Geglu) == {});\n",
        core::mem::size_of::<MoeActivation>(),
        MoeActivation::Relu2 as i32,
        MoeActivation::Swiglu as i32,
        MoeActivation::Geglu as i32,
    );
    if let Err(err) = compile(&tu) {
        panic!("an enum mirror disagrees with the C++:\n{err}");
    }
}

/// `Mxfp4RowSelect`'s three discriminants still match the device text.
///
/// The half of [`the_enum_mirrors_carry_the_cpp_discriminants`] that lost its
/// header. `abi::Mxfp4RowSelect` is `#[repr(i32)]` and reaches the kernel as
/// `ArgValue::I32`, so what has to hold is that Rust's `Even` and the
/// `.cuh`'s `kRowSelectEven` are the same NUMBER — the enum's C++ type is no
/// longer part of the story, because there is no longer a C++ caller to have
/// one.
///
/// Reads the `.cuh` at build time rather than parsing it at run time so a
/// renamed constant is a missing `include_str!` match and not a silent pass.
#[test]
fn the_mxfp4_row_select_mirror_matches_the_device_text() {
    use driver_cuda::bind::abi::Mxfp4RowSelect;
    let cuh = include_str!("../../kernels-cuda-new/csrc/src/quant/mxfp4_marlin.cuh");
    for (name, value) in [
        ("kRowSelectIdentity", Mxfp4RowSelect::Identity as i32),
        ("kRowSelectEven", Mxfp4RowSelect::Even as i32),
        ("kRowSelectOdd", Mxfp4RowSelect::Odd as i32),
    ] {
        let wanted = format!("constexpr int {name} = {value};");
        assert!(
            cuh.contains(&wanted),
            "`quant/mxfp4_marlin.cuh` does not say `{wanted}`, so the Rust mirror and the \
             device text disagree about which half of an interleaved MXFP4 bank a row \
             reads. `select_row` takes the number, not the enum, so nothing else catches \
             this."
        );
    }
}
