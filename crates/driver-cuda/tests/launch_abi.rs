//! The launch ABI proof: a row's operand list, proven by the C++ compiler.
//!
//! # `rope` WAS THE PILOT AND IS NO LONGER HERE
//!
//! This file was written around `rope`'s twelve rows and named them in every
//! case. `rope` has crossed into fn-world (`.wiki/kernel-x/northstar.md` §5
//! step 3): its host programs are `kernels-cuda-new/src/x/rope.rs`, beside
//! the `rope.cuh` they fire, and its contracts state no `operands` at all.
//! There is no row left to emit a shim from, so six cases named `rope` and
//! **four of the six went with it** —
//! `every_rope_row_states_its_launcher_exactly`,
//! `every_launcher_the_header_declares_has_a_row`,
//! `a_wrong_row_does_not_compile` and
//! `renaming_an_operand_is_not_a_mistake`, plus the `rope_shim` helper and
//! the `ROPE_HPP` constant.
//!
//! **The other two did not, and the distinction is the whole of what this
//! header got wrong once.** A case that NAMED `rope` is not the same thing as
//! a case whose SUBJECT was `rope`:
//!
//! * `an_unstated_row_is_skipped_rather_than_called_with_nothing` is about
//!   `abi::stated()`, and its own doc said so — *"the check is about
//!   `stated()`, so its subject should be too"*. `rope` was only where it
//!   borrowed a realistic row from. `stated()` dropping an empty operand list
//!   is now the third of `x::SIGS`' three shim-dropping mechanisms and the
//!   one **every** ported family is carried by, so the case is more
//!   load-bearing than it was, not less. Re-keyed at `table::KERNELS`.
//! * `the_rust_bindings_name_the_symbols_the_shim_defines` is about two
//!   emitters agreeing on `entry_name`, which is the one string the linker
//!   matches on. Re-keyed at `table::KERNELS`, over the rows the shim
//!   actually defines — the two emitters do not filter identically and the
//!   intersection is the set the claim is about.
//!
//! Both are keyed at the AGGREGATE now rather than at a family, deliberately:
//! `table::KERNELS` spans `ROW_TABLES ++ x::SIGS`, so it cannot rot as the §5
//! step-5 sweep empties one family after another — which is exactly how these
//! six came to be broken and unnoticed in the first place.
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
//! That decision stands, and it now has the measurement it was missing: the
//! surviving donor pool is **three rows, all `attn`, and no one of them
//! carries the nine cases' vocabulary**. The full sweep — per family, per
//! filter, with the two rows that would have to be used together and why
//! `csrc/src/norm/` disqualifies twenty-six more — is written where the two
//! tests stood, together with the `Ty`-keyed re-key that dissolves the
//! index-rot this paragraph is about.
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
};
// `YarnOriginalParams` left with its record — its C++ declaration went with
// `attn/mla_paged.hpp` and there is nothing here to assert it against. The
// Rust type is still live; see the note in `records()`.
// `Operand` and `Ty` left with the mutation suite — see the header.
use kernels::KernelSig;
use kernels_cuda_new::abi::Record;

/// Where `kernels-cuda`'s sources are, relative to this crate.
fn csrc() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels-cuda/csrc/src")
}

/// Every tree that currently holds a `.cu` or `.cuh` launcher body.
///
/// ONE NOW, AND THE HOLE IS FILLED RATHER THAN CLOSED. This used to return
/// two: `csrc()`, the archive, and `driver-cuda/csrc`, which held
/// `attn/attention_flashinfer.cu`, `attn/plan_lifecycle.cpp`, `supergraph.cu`
/// and the three towers — files earlier passes moved out of the archive
/// because they could not be compiled by NVRTC. The doc that added the second
/// root said the destination was wrong and that C++ in this crate was staging
/// awaiting a rewrite, not a home. **The rewrite happened.** `supergraph.cu`
/// and the towers went earlier; `attention_flashinfer.cu` (1,258 lines) and
/// `plan_lifecycle.cpp` (105) went with the FA2 dispatches, and
/// `crates/driver-cuda/csrc/` does not exist. A root that names a directory
/// with no files in it is not a scan, it is a comment.
///
/// The reason the second root was added still stands and is why this is a
/// deletion rather than a rename: every `Orphaned` claim in the exception
/// list means NOTHING CALLS THIS, checked by reading every `.cu` body and
/// looking for a call, and a scan that reads one tree while launchers live in
/// two finds fewer callers than exist — failing SILENTLY, by passing. That
/// incentive is gone by the launchers being gone, not by the scan being
/// widened, which is the stronger of the two ways to remove it.
///
/// `collect_cu_bodies` returns quietly on a missing directory, so leaving the
/// dead root would also have "worked". It would have worked the way the
/// one-root scan worked before the second was added: by finding nothing and
/// saying nothing. The vacuity floors below are what stop that, and they are
/// re-measured against one root at their assertions.
fn cu_roots() -> Vec<PathBuf> {
    vec![csrc()]
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
    // `attn/` is the last family in the archive and will go the same way the
    // other nine did. `headers_in`'s header carries the argument for treating
    // its absence as a fact.
    let Ok(entries) = std::fs::read_dir(csrc().join("attn")) else {
        return Vec::new();
    };
    let mut hs: Vec<String> = entries
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
    // `norm/` has left the archive; this is `headers_in("norm")` spelled out,
    // and it returns the empty set for the reason that function's header
    // gives.
    let Ok(entries) = std::fs::read_dir(csrc().join("norm")) else {
        return Vec::new();
    };
    let mut hs: Vec<String> = entries
        .filter_map(|e| {
            let n = e.ok()?.file_name().into_string().ok()?;
            n.ends_with(".hpp").then(|| format!("norm/{n}"))
        })
        .collect();
    hs.sort();
    hs
}

// `every_norm_row_states_its_launcher_exactly` STOOD HERE, and it is deleted
// for the reason the `mlp` note below gives, which now applies to it too:
// §5 step 5 took `norm` into fn-world and `x::norm`'s twenty-eight contracts
// state NO operands, so the assertion would read "0 of 28 rows are stated"
// and fail on the change that made it right.
//
// Its text was: *"`norm`'s rows, proven the same way `attn`'s are. Twenty-eight
// launchers across seven headers, and the family every other one leans on: a
// wrong row here is a wrong argument in an arm that four executors reach."*
// It asserted that every one of `table::norm`'s rows states its `operands`,
// then compiled the shim those operands generate.
//
// TWENTY-EIGHT was the count of LAUNCHERS across the seven `norm/*.hpp`
// headers; `table::norm::KERNELS` held twenty-SIX rows, and the two the
// table never had -- `norm::add_bias_bf16` and
// `norm::rmsnorm_gated_fp32_in_bf16` -- are contracts in `x::norm` now, which
// is how the port found them. The launcher-side claim this test made survives
// in `every_norm_launcher_is_a_row_or_a_stated_exception` below, which reads
// the same seven
// headers and requires every declaration in them to have a row or a stated
// reason; it reads `x::norm::SIGS` for that half now.
//
// The C-shim half is not replaced and does not need to be: a contract-derived
// row states no operands, `emit_c_shim` drops it, and the shim for `norm`
// would be a file of includes and no bodies -- the same empty compile the
// `mlp` note records below.

// `every_mlp_row_states_its_launcher_exactly` STOOD HERE, and it is deleted
// rather than repaired, because §5 step 5 makes its assertion the OPPOSITE of
// what is now correct.
//
// Its text was: *"`mlp`'s rows. Sixteen activations across two headers, and
// the family whose default arguments make a hand-written binding easiest to
// get wrong -- `gpt_oss_glu_bf16` alone carries three."* It asserted that
// every one of `table::mlp`'s twelve rows states its `operands`, then
// compiled the shim those operands generate.
//
// `x::mlp`'s twelve contracts state NO operands — that is the third of the
// three shim-dropping mechanisms and it is the point of the port, not a
// regression — so the assertion would now read "0 of 12 rows are stated" and
// fail on the change that made it right. `emit_c_shim` was already emitting
// nothing here (the comment inside said so: *"this shim is now a file of
// includes and no bodies"*), so the compile half tested an empty file.
//
// The claim the test actually protected — that a stated row describes a real
// launcher — survived in `every_norm_row_states_its_launcher_exactly` until
// §5 step 5 took `norm` too (see the note above this one), and survives in
// `the_stated_quant_layout_and_gemm_rows_describe_their_launchers` below,
// which is deliberately NOT a whole-family assertion for exactly this reason.
// What replaces it for `mlp` is `driver-cuda/build.rs`'s `armless` check and
// the `x::mlp` host programs themselves: `gpt_oss_glu_bf16`'s three defaults
// are now three Rust parameters with types, which is the ladder §1 draws.

/// `quant`'s, `layout`'s and `gemm`'s rows — all three DERIVED now, which
/// is a state this test has to be read carefully in.
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
///
/// # `moe` was the fourth table and is gone, and with it the last STATED
/// row this walk had
///
/// `table::moe::KERNELS` held four routed `quant::` decode GEMVs after §5
/// step 5 emptied it of `moe::` rows — filed there because **`table/` was
/// organised by who DISPATCHES and `x/` by who owns the code** — and they
/// were the only entries left in this array that stated `operands`. They
/// are `x::quant` contracts now, and `x::quant::SIGS` was already the first
/// element, so nothing is appended in their place.
///
/// **So `shim_over` is walking three derived lists and emitting nothing at
/// all.** That is not a failure and it is not a pass either: the compile
/// half now compiles an empty file, exactly as `the_mlp_rows_*` did before
/// it was deleted two notes above. It is kept rather than deleted for one
/// reason — the header list below is still a real claim, that every header
/// these families' launchers live in is includable — and it will stop being
/// worth keeping the moment that claim moves somewhere that re-derives it.
#[test]
fn the_stated_quant_layout_and_gemm_rows_describe_their_launchers() {
    // `table::gemm::KERNELS` was the third; `x::gemm::SIGS` is the same
    // twelve symbols derived from the `contract!` block.
    //
    // `table::layout::KERNELS` was the second and `x::layout::SIGS` is its
    // seven, derived the same way. Both derived lists state no `operands`,
    // so `shim_over` emits nothing for either and they are here only so the
    // walk still sees every symbol these headers cover — which is what makes
    // `"layout/embed.hpp"` below an include for a family whose rows no
    // longer reach it, rather than an include that lost its rows silently.
    let tables: [&'static [KernelSig]; 3] = [
        // `table::quant::KERNELS` was the first; `x::quant::SIGS` is its
        // FIFTEEN symbols derived from the `contract!` block, stating no
        // `operands` for the same reason `layout`'s and `gemm`'s do. It was
        // eleven until the four `table::moe` tenants joined it.
        kernels_cuda_new::x::quant::SIGS,
        kernels_cuda_new::x::layout::SIGS,
        kernels_cuda_new::x::gemm::SIGS,
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
        // `x::quant::SIGS` above stays in the table list: its rows state no
        // operands at all now, so the shim emits nothing and there is nothing
        // left to include for them. The sentence held for a different reason
        // before the port — the rows were all routed — and both readings end
        // at the same empty shim, which is what made the swap safe.
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

// `every_ssm_row_states_its_launcher_exactly` STOOD HERE, and it is deleted
// for the reason the `norm` and `mlp` notes above give, which now applies to
// it too: §5 step 5 took `ssm`'s five roots into fn-world and `x::ssm`'s
// twenty-seven contracts state NO operands, so the assertion would read
// "0 of 27 rows are stated" and fail on the change that made it right.
//
// Its text was: *"`ssm`'s rows — the largest single family, and the one whose
// ten recurrence spellings differ only by which state dtype and whether the
// heads are grouped. Ten near-identical argument lists is exactly where a
// hand-written binding goes wrong quietly: `state_base` is `float*` in six of
// them and `void*` in four, and the two are the same pointer at a call
// site."*
//
// THAT CLAIM SURVIVES AND IS NOW A COMPILE ERROR RATHER THAN AN ASSERTION,
// which is the whole of what the port bought here. The ten spellings are ten
// `pub unsafe fn`s in `x::ssm::gated_delta_net`, six taking `*mut f32` and
// four `*mut c_void`, and `unit!`'s generic binding group ties each symbol to
// the instantiation that matches — so handing a `float*` state to a bf16 row
// no longer type-checks. No test is needed for a thing the compiler refuses.
//
// The C-shim half needed no replacement even before the port: this test's own
// comment recorded that the shim it compiled was ALREADY EMPTY, because every
// `table::ssm` row was either `device::JIT_DISPATCHED` or
// `execution::RUST_SERVED` and `emit_c_shim` skips both. It was compiling a
// file of no includes and no bodies.

/// The same proof at family scale: all fifty `attn` rows, ~700 operands.
///
/// **`table::attn::KERNELS` IS EMPTY AND THIS TEST HAS NO INPUT LEFT.** All
/// forty-one rows crossed into `kernels_cuda_new::x::attn` under §5 step 5,
/// the last being `attn::qkv_decode_qk_norm_rope_write_kv_bf16`, and a
/// contract states no `operands` — so `emit_c_shim` has nothing to emit for
/// this family by construction rather than by omission. The early return is
/// explicit because a shim generated from zero rows compiles trivially, and
/// a proof that passes because its subject is gone should say which.
///
/// This is the ROW WORLD's test and it retires with the row interpreters in
/// north star step 6. Left in place, not deleted, for the reason
/// `table::attn` itself is: it is one of two consumers keeping that module
/// declared, and step 6 should retire the pair together.
///
/// `rope` was twelve rows of scalars and buffers. `attn` was what the ABI
/// actually had to survive: views passed BY VALUE, plan caches passed as
/// `const&` to a type the header never defines, a `cublasHandle_t` where a
/// stream would be, and both halves of every const/mut pointer pair. If the
/// vocabulary in `kernels::Ty` had been short of any of that, this would not
/// have compiled — which was the point of running it as one shim rather than
/// fifty.
#[test]
fn every_attn_row_states_its_launcher_exactly() {
    let table = kernels_cuda_new::table::attn::KERNELS;
    if table.is_empty() {
        return;
    }
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
        // `x::attn::kv_paged::write_kv_to_pages_bf16` is that word — a ported
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

    // ACCOUNTED-FOR, NOT "HAS A ROW". `table::attn::KERNELS` is empty — the
    // family crossed — so the row-world predicate this used to be would call
    // every declared launcher undecided. The question the test asks is
    // unchanged (*is this declaration accounted for anywhere, or is it a
    // launcher nobody decided about*); what changed is where the answer
    // lives. Three registries, and the union is a strict superset of the old
    // predicate:
    //
    //   `unit::unit_of`      a JIT-hosted device row, which is what a
    //                        crossed launcher becomes
    //   `table::KERNELS`     `ROW_TABLES ++ x::SIGS`, so both a surviving row
    //                        and a `contract!`-derived one
    //   `ends_with("::{n}")` the cross-family case this always had: a
    //                        launcher declared in `attn/` whose symbol is
    //                        another family's
    let has_row = |n: &str| {
        let sym = format!("attn::{n}");
        kernels_cuda_new::unit::unit_of(&sym).is_some()
            || kernels_cuda_new::table::KERNELS
                .iter()
                .any(|k| k.symbol == sym || k.symbol.ends_with(&format!("::{n}")))
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
    // 60 KB, WAS 100 KB, AND THE MOVE IS THE POINT OF THE NUMBER. The old
    // floor was measured with `driver-cuda/csrc` in `cu_roots`. That tree is
    // deleted, and this scan — comment lines, `void <name>(` lines and
    // `"`-quoted spans removed — now reads 73,055 bytes from the archive
    // alone. A floor left at 100_000 would fail for the one reason a vacuity
    // guard must never fail for: the tree really did get smaller. A floor
    // raised to 73_000 would break on the next honest deletion. 60_000 is
    // ~82% of what is there, which is a scan that stopped, not a tree that
    // shrank — and `attention_xqa*`, the largest bodies left, are owned by
    // another pass, so this number is expected to move again with them.
    assert!(
        cu_text.len() > 60_000,
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
    // Same as `attn_headers`: the directory's absence is the family finishing.
    let Ok(entries) = std::fs::read_dir(csrc().join("attn")) else {
        return out;
    };
    for entry in entries {
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
///
/// # Its subject did not leave with `rope`, it got PROMOTED
///
/// The donor was `table::rope::KERNELS` and `rope` is fn-world's pilot, so
/// the row is gone. The rule is not: `abi::stated()` dropping a row with an
/// empty operand list is the THIRD of `x::SIGS`' three shim-dropping
/// mechanisms, and `x/mod.rs` names it *"the mechanism every ported row is
/// carried by"*. Every family that crosses adds twelve-odd rows whose only
/// protection from being emitted as a nullary `extern "C"` is this one
/// predicate — so a test that was about a corner of the fill campaign is now
/// the guard on the whole §5 step-5 sweep.
///
/// So the donor is re-keyed at `table::KERNELS`, the aggregate, rather than
/// at a family. A family-keyed donor is a citation that rots the day that
/// family crosses, which is the failure that brought this test here; the
/// aggregate cannot, because it is what `check_plan` reads and it spans both
/// worlds by construction (`ROW_TABLES ++ x::SIGS`).
///
/// The second half is the same predicate asked of REAL unstated rows instead
/// of a synthetic one, and it carries its own non-emptiness guard. An empty
/// list is a valid list: `assert!(!shim.contains("extern \"C\""))` over a
/// `SIGS` that had gone to zero rows would pass while checking nothing, and
/// a passing assertion over nothing is the one failure this tree has already
/// shipped once.
#[test]
fn an_unstated_row_is_skipped_rather_than_called_with_nothing() {
    let stated = kernels_cuda_new::table::KERNELS
        .iter()
        .find(|k| !k.operands.is_empty())
        .expect("some row of the aggregate table is stated");
    let row: &'static [KernelSig] = Vec::leak(vec![KernelSig {
        operands: &[],
        ..*stated
    }]);
    // `rope_shim` went with `rope`. The headers are irrelevant to what is
    // being asked — nothing is emitted, so nothing needs declaring — and
    // `shim_over` is the family-agnostic spelling the rest of the file uses.
    let shim = shim_over(row, &[]);
    assert!(
        !shim.contains("extern \"C\""),
        "the row states no operands, so nothing should be emitted:\n{shim}"
    );

    // AND THE REAL ONES. `x::rope::SIGS` is twelve contracts whose derived
    // rows state no `operands` by construction — the thing the synthetic row
    // above imitates — so the same predicate must drop all twelve.
    let ported = kernels_cuda_new::x::rope::SIGS;
    assert!(
        !ported.is_empty(),
        "`x::rope::SIGS` is empty, so the assertion below would pass over \
         nothing. Either the pilot family lost its contracts or this test is \
         pointed at the wrong list; both are defects and neither is a green run"
    );
    assert!(
        ported.iter().all(|k| k.operands.is_empty()),
        "a fn-world contract derived a row WITH operands, so `stated()` would \
         emit a `pie_k_*` entry forwarding to a launcher that is a Rust `fn`"
    );
    let ported_shim = shim_over(ported, &[]);
    assert!(
        !ported_shim.contains("extern \"C\""),
        "a ported family still emits a shim entry:\n{ported_shim}"
    );
}

// THE MUTATION SUITE AND ITS CONTROL ARE RETIRED HERE, UNRESOLVED, AND THAT
// IS DELIBERATE. Both were keyed on `rope::qk_rmsnorm_rope_bf16`, and `rope`
// is fn-world's pilot.
//
// `a_wrong_row_does_not_compile` read:
//
// > Corrupting a row must break the build — the mutation suite, answered
// > exactly.
// >
// > Each case changes ONE thing a hand-written binding gets wrong, and every
// > one of them has to be caught. The last two are the interesting ones: they
// > are not type errors, they are an operand list of the right types in the
// > wrong ORDER, which is precisely the failure a `void*`-flattened ABI
// > cannot see and this one can.
//
// It built NINE corruptions of the pilot row and required every one to fail
// the C++ compile: a written buffer claimed read-only (`retype(0, Buf)`), a
// read-only weight claimed written (`retype(2, BufMut)`), positions losing
// its element type (`retype(4, Buf)`), an extent widened to a float
// (`retype(5, F32)`), a rate narrowed to an int (`retype(9, I32)`), the
// stream dropped, an operand invented at index 5, `q` and `k_weight` trading
// places (`swap(0, 3)`) and an extent trading with a rate (`swap(6, 9)`).
// `renaming_an_operand_is_not_a_mistake` was its control: rename all eleven
// operands, which is prose, and require the same shim to still compile.
//
// # The subject is ALIVE, so this is a loss and not a close
//
// `emit_c_shim` still serves every row family, and the mutation suite is the
// only thing in the tree that asks whether the proof is watching. This file's
// header already recorded the decision not to re-key on a guess; what follows
// is the measurement that decision was missing, so the next attempt starts
// from numbers instead of from a survey.
//
// # Why it is not re-keyed, measured rather than asserted
//
// `emit_c_shim` applies three filters — `abi::stated()` (a non-empty operand
// list), `device::JIT_DISPATCHED`, and `execution::RUST_SERVED` — so a donor
// must survive all three, and its family's `.hpp` must still be in
// `csrc/src/` for the control to compile. Swept 2026-08-14 against
// `table/{attn,norm,moe}.rs`, `device::JIT_DISPATCHED` (37 symbols) and
// `execution::RUST_SERVED` (51):
//
//   * `attn` — 36 rows, **3 survive all three filters**.
//   * `norm` — 26 rows survive the filters and NONE is a candidate:
//     `csrc/src/norm/` is deleted, so `norm_headers()` is empty and there is
//     no declaration for a shim body to call.
//   * `moe` — 24 rows, **0 survive**.
//
// The three surviving `attn` rows do not carry the vocabulary between them,
// and they fail in opposite directions:
//
//   * `attn::attention_xqa_decode_bf16_prepared` (13 operands) has `Buf`,
//     `BufMut`, `I32` and `F32` — and NO typed device array, so "positions
//     loses its element type" has no subject.
//   * `attn::attn_score_fold_heads` (9) has `F32s`, `I32s`, `U32s`,
//     `F32sMut` and `I32` — and no `Buf`, no `BufMut`, no scalar `F32`, so
//     five of the nine cases have no subject.
//   * `attn::dispatch_attention_mla_bf16` (14) carries two opaque plan/view
//     types and no scalar `F32`.
//
// So a re-key needs TWO donors and would still be pointed at `attn`, which
// `x/attn.rs` and `x/xqa.rs` say is mid-crossing. Splitting a nine-case
// mutation suite across two rows of a family that is leaving is how the
// citation rots again, and this file's `HELD`-table lesson — a gate whose
// denominator is a set the claimant supplies — is the same shape.
//
// # What a re-key should do instead, when there is a build to check it with
//
// Key on `Ty`, not on index. The suite failed to survive its family because
// `retype(4, Buf)` is a claim about `rope::qk_rmsnorm_rope_bf16`'s operand 4;
// `retype(first_of(Ty::I32s), Ty::Buf)` is a claim about the vocabulary, and
// the vocabulary does not move when a family does. Select the donor at run
// time as *the first row of `table::KERNELS` that survives the three filters
// and carries a `BufMut`, a `Buf`, a typed device array, a scalar `I32` and a
// scalar `F32`*, `.expect()` with those five kinds named, and derive all nine
// cases from it. That donor is `None` today, which is why this is a record
// and not a patch.
//
// One fact makes the re-key sound the moment a donor exists, and it is worth
// carrying because it is not obvious: the emitted body forwards **through a
// function pointer of the row's exact type**, and `emit_c_shim`'s own header
// says why — *"A function-pointer initialisation admits NO parameter
// conversions ... a direct call is checked by overload resolution, and
// overload resolution accepts `void*` where the callee takes `const void*`."*
// So every one of the nine really is a compile error, including the two
// swaps: `int`/`float` would convert silently in a direct call and cannot in
// a pointer initialisation. A re-key does not have to re-establish that.
//
// The control goes with the suite, and only with it. A control's whole job is
// to fail when the mutations pass for an unrelated reason; kept alone it
// would assert that one arbitrary row compiles, which
// `every_attn_row_states_its_launcher_exactly` already asserts for all of
// them. Re-land the two together or neither.

/// The Rust bindings declare exactly what the C++ shim defines.
///
/// Both are generated from one row, so this cannot fail by drift; what it
/// pins is that the two emitters agree on the ENTRY POINT spelling, which is
/// the one string the linker matches on and the one thing neither compiler
/// checks.
///
/// # Re-keyed off `rope` and onto the aggregate, and the loop gained a filter
///
/// The donor was `table::rope::KERNELS` and `rope` crossed into fn-world, so
/// the rows are gone. The claim is not: `entry_name` is still the one string
/// the linker matches on, and `emit_c_shim` and `emit_rust_bindings` are
/// still two emitters that can disagree about it. `table::KERNELS` is the
/// right subject because it cannot rot as families cross — it spans
/// `ROW_TABLES ++ x::SIGS` by construction — and because a disagreement is a
/// link error in whichever binary states the symbol first, which has nothing
/// to do with which family the row is in.
///
/// **The two emitters do not filter identically, and iterating the table raw
/// would fail on that rather than on a disagreement.** Both apply
/// `abi::stated()` and both skip `execution::RUST_SERVED`; only
/// `emit_c_shim` also skips `device::JIT_DISPATCHED`, because a routed row
/// has no host launcher to forward to while its Rust declaration is still
/// legitimate (`bind::jit::fire` is its path, and `driver-cuda/build.rs`'s
/// "a declaration with no definition is only legitimate for a routed row"
/// check is where that is enforced). So the loop walks the INTERSECTION —
/// the rows the shim actually defines — which is exactly the set on which
/// "the two emitters agree" is a statement at all.
///
/// The intersection is checked for emptiness first. It is one row per family
/// still in the archive and it shrinks with every port; when it reaches zero
/// this test passes over nothing, and it must say so rather than go green.
#[test]
fn the_rust_bindings_name_the_symbols_the_shim_defines() {
    let table = kernels_cuda_new::table::KERNELS;
    let jit = kernels_cuda_new::device::jit_dispatched();
    let defined: Vec<&'static KernelSig> = table
        .iter()
        .filter(|k| !k.operands.is_empty())
        .filter(|k| !jit.iter().any(|d| d.sig.symbol == k.symbol))
        .filter(|k| !kernels_cuda_new::execution::RUST_SERVED.contains(&k.symbol))
        .collect();
    assert!(
        !defined.is_empty(),
        "no row survives the shim's three filters, so the loop below would \
         check nothing. The archive has no entry points left — which is the \
         end of the migration and is the right moment to DELETE this test \
         with a record, not to let it report green over an empty set"
    );

    // `rope_shim` went with `rope`; `shim_over` is the family-agnostic
    // spelling, and the headers are irrelevant here — this asks what the
    // emitters NAME, not whether the bodies compile, which is
    // `every_attn_row_states_its_launcher_exactly`'s question.
    let shim = shim_over(table, &[]);
    let rs = kernels_cuda_new::abi::emit_rust_bindings(&[table]);
    for k in defined {
        let entry = kernels_cuda_new::abi::entry_name(k.symbol);
        assert!(
            shim.contains(&format!("void {entry}(")),
            "{entry} not defined"
        );
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

    let tables: &[&'static [kernels::KernelSig]] = &[kernels_cuda_new::table::KERNELS];
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
        // `YarnOriginalParams` STOOD HERE AND ITS C++ RECORD DOES NOT EXIST.
        //
        // It was declared in `attn/mla_paged.hpp`, deleted with its `.cu` by
        // `244df6054`. `MIRROR_HPPS` lost that entry in the same change, and
        // the comment left there reasoned about `MlaCacheLayerView` finding
        // another home in `attn/mla_cache_view.hpp` — correctly — and did not
        // ask the same question about the second record the header declared.
        // There is now no `struct YarnOriginalParams` anywhere under either
        // `csrc` tree.
        //
        // **This reproduced, exactly, the bug `MIRROR_HPPS`' own doc was
        // written to prevent.** One positive case red on an undeclared type,
        // and the three mutation cases GREEN for the wrong reason — they
        // assert a TU fails to compile, and it was failing on the missing
        // declaration rather than on the mutation.
        // `records_are_declared_in_the_headers_that_are_included` is the
        // check that would have caught the deletion, and it names the record.
        //
        // It is removed rather than repointed because there is nothing to
        // repoint to, and the layout claim has no counterparty left:
        //
        //   * the row that spelled it, `attn::mla_prepare_bf16`, is in
        //     `execution::RUST_SERVED`, so `emit_c_shim` emits no entry and
        //     the generated shim never names the type;
        //   * the fn-world launcher `driver-cuda/src/fire/mla_paged.rs` does
        //     not pass a struct at all — `yarn_factor`, `yarn_low_dim`,
        //     `yarn_high_dim` and `yarn_mscale` cross as four separate
        //     `ArgValue::F32`, which is the shape
        //     `kernels-cuda-new/csrc/src/attn/mla_paged.cuh:251-254` declares;
        //   * nothing under `kernels-cuda-new/csrc` declares or uses it.
        //
        // The Rust `bind::abi::YarnOriginalParams` stays, and its remaining
        // user is `bind::abi::ffi` — which is `#[cfg(feature = "bridge")]`,
        // so the mirror retires on exactly the schedule these headers do.
        // `fire/mla_paged.rs` DID NOT use it, and that file is now deleted:
        // `attn::mla_prepare_bf16` crossed into fn-world, its host program is
        // `kernels_cuda_new::x::attn::mla_prepare_bf16`, and its `Option`
        // parameter is `x::Yarn` -- the same five fields, spelled once for
        // the whole crate instead of once per driver module.
        //
        // `kernels::Ty::YarnOriginalParams` still spells
        // `"const ::pie_cuda_driver::kernels::attn::YarnOriginalParams*"` for
        // the shim. It is inert, not broken, and the reason has CHANGED: it
        // used to be that the row was `RUST_SERVED` so `emit_c_shim` dropped
        // the entry; now there is no row at all, and nothing in `table::attn`
        // names this `Ty`. It would become a shim compile error the moment a
        // row that names it came back.
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

/// The five headers this test compiles, and they OUTLIVE THE ARCHIVE.
///
/// [`MIRROR_HPPS`] is four entries; the transitive closure of the generated TU
/// is five — `attn/kv_cache_view.hpp` includes `tensor.hpp`. All five live in
/// `crates/kernels-cuda/csrc`, which north star step 6 deletes when `bridge`
/// goes and `ROW_TABLES` empties.
///
/// **This test does not go with it.** It needs `g++`, `-I csrc`, and the
/// two-file stub tree beside it; it needs no nvcc, no `bridge`, no `native`
/// and nothing from the archive's CMakeLists. So the step-6 sweep will find
/// five headers whose only `#include`rs are dead archive text and conclude
/// they are unreachable — a grep for `#include "tensor.hpp"` cannot see this
/// consumer, because the include lines exist only in the text
/// [`kernels_cuda_new::abi::emit_layout_assertions`] generates.
///
/// # The rule, both halves, because this is now the only copy in a crate that
/// # survives
///
/// The first half was written during the archive sweep: **an `#include` from
/// a translation unit that cannot be compiled is not a consumer.** It retired
/// several keep-claims resting on includes from deleted `.cu` files and from
/// `oracle.cpp`s whose `run.sh` die at their first `cp` — see
/// `tests/oracle_census.rs`.
///
/// This test is the **dual, and it was found by the first half getting these
/// five wrong**: a translation unit that CAN be compiled need not contain a
/// literal `#include` anywhere in the tree. Both halves are the same mistake —
/// counting text instead of counting compilations — and they fail in opposite
/// directions. The first overcounts an edge that exists and can never be
/// traversed. The second undercounts an edge that is traversed every
/// `cargo test` and cannot be grepped.
///
/// The write-ups that stood beside the affected headers, in the archive's
/// `CMakeLists.txt` and in `csrc/src/attention_workspace_view.hpp`, are in
/// files scheduled for deletion; the CMakeLists copy was already lost to a
/// concurrent rewrite within the hour. **This is the copy that is meant to
/// survive**, and it is here rather than in a document for the reason
/// `weights/plan.rs` records: a measurement beside the assertion that depends
/// on it is read by the person who needs it.
///
/// The check is path existence, so it fires BEFORE `g++` does and says why,
/// instead of leaving a compile error whose cause is three indirections away.
/// It is also self-retiring in the useful direction: move a mirror's C++
/// record to a surviving crate and this fails, which is the moment to ask
/// whether the header still needs to exist at all.
const CLOSURE_SURVIVES_STEP_6: &[&str] = &[
    "attn/kv_cache_view.hpp",
    "attention_workspace_view.hpp",
    "attn/mla_cache_view.hpp",
    "attn/attention_flashinfer_hopper.hpp",
    // Not in `MIRROR_HPPS`; reached through `attn/kv_cache_view.hpp`, which
    // is why a list of the four would have missed it.
    "tensor.hpp",
];

/// The headers the layout assertions need are still on disk.
#[test]
fn the_mirrored_headers_outlive_the_archive_that_holds_them() {
    let root = csrc();
    let gone: Vec<&str> = CLOSURE_SURVIVES_STEP_6
        .iter()
        .copied()
        .filter(|h| !root.join(h).is_file())
        .collect();
    assert!(
        gone.is_empty(),
        "{gone:?} is missing from `kernels-cuda/csrc/src`.\n  \
         These are compiled by `the_mirrors_have_the_layout_the_cpp_has`, \
         through `#include` lines that `emit_layout_assertions` GENERATES — \
         so nothing in the tree greps as including them and a sweep reading \
         `#include` edges alone will call them unreachable. They are not: \
         this test needs only `g++` and the stub tree, and survives the \
         archive's deletion at north star step 6.\n  They do NOT need a home \
         outside `kernels-cuda`: see \
         `the_mirrors_have_the_layout_the_cpp_has` — after `bridge` nothing \
         makes a competing claim about these layouts, so this suite and these \
         headers retire together rather than moving."
    );
}

/// The other generated include set, and the one nothing was watching.
///
/// `kernels-cuda/build.rs::shim()` — feature `native`, which `bridge` is the
/// only thing that turns on — builds the production `libpie_launch_shim.a`.
/// Its `includes()` does **not** ask which rows survived: it `read_dir`s each
/// family directory and takes EVERY `*.hpp`, then `emit_c_shim` writes one
/// `#include "<dir>/<name>"` line per entry. So a header dropped into
/// `csrc/src/attn/` joins a production compile with nothing naming it.
///
/// That is the same blind spot as [`CLOSURE_SURVIVES_STEP_6`] and it was
/// found a turn later, on the same file: `attention_workspace_view.hpp` was
/// twice written up as having no compilable consumer, and it has TWO — this
/// shim and the g++ TU above. `attn/attention_xqa.hpp` and
/// `attn/attention_flashinfer.hpp` were written up as dead and are compiled
/// here every `native` build.
///
/// The general statement, which is what makes it checkable rather than
/// anecdotal: **`kernels_cuda_new::abi` is the tree's `#include` generator.**
/// Three emitters write `#include` text — `emit_c_shim`,
/// `emit_device_typecheck`, `emit_layout_assertions` — and no grep can
/// consult them, because their filenames are Rust `&[&str]`s and `read_dir`
/// results.
///
/// # The two compilers end on different schedules
///
/// The shim dies when `ROW_TABLES` empties and `bridge` goes. The g++ TU
/// above does not: it needs `g++`, the two-file stub tree and `-I csrc/src`,
/// and nothing from the archive's CMakeLists. So membership in `MIRROR_HPPS`
/// is what decides whether an `attn` header survives step 6, and that is not
/// arbitrary — a header is in it because a `#[repr(C)]` mirror claims to have
/// its layout, and the mirror is what crosses. `attention_flashinfer_hopper`
/// being in it is what discharges the constraint that it must outlive the
/// non-sm90 stub it defined.
///
/// This list is the shim-only remainder: in the generated shim, not in
/// `MIRROR_HPPS`, and therefore deleted with `bridge` rather than moved.
const SHIM_ONLY_ATTN_HPPS: &[&str] = &["attention_flashinfer.hpp", "attention_xqa.hpp"];

/// Every header the production shim compiles has been classified.
///
/// The partition is the point. `MIRROR_HPPS` says which `attn` headers a
/// Rust mirror pins; [`SHIM_ONLY_ATTN_HPPS`] says which are host declarations
/// the shim forwards through. A header in neither is one that joined a
/// production compile by being dropped into a directory, which is exactly the
/// way all three of this file's misses happened.
///
/// Self-retiring: when `csrc/src/attn/` empties, both lists must be empty and
/// this says so rather than passing vacuously.
///
/// # Why `attn` and not the whole shim set
///
/// `includes()` sweeps twelve directories and three still hold headers:
/// `attn` (5), `vision` (5), `rope` (1). Only `attn`'s are partitioned here,
/// because only `attn` has headers on both sides of step 6 — some pinned by a
/// `#[repr(C)]` mirror and some not. `vision` and `rope` have no mirror at
/// all, so every one of theirs is shim-only and goes with `bridge`; a table
/// listing them would have one column and assert nothing. If a `vision` or
/// `rope` record ever gains a mirror, that is the moment this partition wants
/// to cover all three directories.
#[test]
fn no_attn_header_joins_the_generated_shim_unclassified() {
    let dir = csrc().join("attn");
    let mut found: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{dir:?} is readable: {e}"))
        .filter_map(|e| {
            let n = e.ok()?.file_name().to_string_lossy().into_owned();
            n.ends_with(".hpp").then_some(n)
        })
        .collect();
    found.sort();

    let classified = |n: &str| {
        MIRROR_HPPS.iter().any(|m| m.strip_prefix("attn/") == Some(n))
            || SHIM_ONLY_ATTN_HPPS.contains(&n)
    };
    let unclassified: Vec<&String> = found.iter().filter(|n| !classified(n.as_str())).collect();
    assert!(
        unclassified.is_empty(),
        "csrc/src/attn/{unclassified:?} is compiled by the production shim and \
         classified nowhere.\n  `kernels-cuda/build.rs::includes()` sweeps \
         EVERY `*.hpp` in this directory into the generated `shim.cpp` — it \
         does not read the row tables — so a header lands in a `cc` compile \
         with nothing in the tree naming it.\n  Say which it is: a mirror \
         header goes in `MIRROR_HPPS` and survives step 6 because \
         `launch_abi` compiles it with g++; a host declaration goes in \
         `SHIM_ONLY_ATTN_HPPS` and is deleted with `bridge`."
    );

    let stale: Vec<&&str> =
        SHIM_ONLY_ATTN_HPPS.iter().filter(|n| !found.iter().any(|f| f.as_str() == **n)).collect();
    assert!(
        stale.is_empty(),
        "{stale:?} is listed shim-only and is not in csrc/src/attn/.\n  \
         Drop the entry — the header is gone and the row describes nothing."
    );
}

/// Every mirrored record is DECLARED in a header the assertions include.
///
/// The gate that was missing. `records()` names a C++ type;
/// [`MIRROR_HPPS`] names the headers `emit_layout_assertions` `#include`s.
/// Nothing tied the two together, so deleting a header could orphan a record
/// — and did: `attn/mla_paged.hpp` went with its `.cu` in `244df6054` and
/// took `struct YarnOriginalParams` with it, while the record stayed in
/// `records()` for two more sweeps.
///
/// # Why the compiler was not enough
///
/// It is what `MIRROR_HPPS`' own doc warns about, and the warning did not
/// save it. [`the_mirrors_have_the_layout_the_cpp_has`] goes red on an
/// undeclared type, which is loud but says only "g++ did not like the
/// generated TU". The three mutation cases go **green**, because they assert
/// the TU fails to compile and it does — on the missing declaration, not on
/// the mutation. So the net effect of orphaning a record is one confusing
/// failure and three tests that stop testing anything, which is strictly
/// worse than the plain failure.
///
/// This fires first, before any compiler runs, and names the record and the
/// headers it was looked for in.
///
/// # What it asserts
///
/// That the tag — the last `::` segment of the record's C++ path — appears
/// as `struct <Tag>` in one of the `MIRROR_HPPS` texts. Not a parse: a
/// declaration this suite depends on is a plain `struct X {` line in a
/// hand-written header, and a gate whose subject is approximately right
/// reports failures that are approximately real. If a record is ever
/// declared some other way (a template, a `using`), this fires and the right
/// answer is to say so here rather than to loosen the match.
#[test]
fn records_are_declared_in_the_headers_that_are_included() {
    let texts: Vec<(&str, String)> = MIRROR_HPPS
        .iter()
        .map(|h| {
            let p = csrc().join(h);
            let t = std::fs::read_to_string(&p)
                .unwrap_or_else(|e| panic!("{p:?} is a mirror header and does not read: {e}"));
            (*h, t)
        })
        .collect();

    for r in records() {
        let tag = r.cpp.rsplit("::").next().unwrap_or(r.cpp);
        let needle = format!("struct {tag} ");
        let brace = format!("struct {tag}{{");
        assert!(
            texts.iter().any(|(_, t)| t.contains(&needle) || t.contains(&brace)),
            "`{}` is asserted against a C++ record that is declared in none of \
             MIRROR_HPPS {MIRROR_HPPS:?}.\n  A record whose header was deleted \
             does not just fail: the mutation cases assert a TU FAILS to \
             compile, so they pass on the missing declaration and stop testing \
             the mutation.\n  Either add the header that declares `{tag}` to \
             MIRROR_HPPS, or — if nothing declares it any more — delete the \
             record. The layout claim needs a counterparty; without one there \
             is nothing for the mirror to disagree with.",
            r.cpp
        );
    }
}

/// A `#[repr(C)]` mirror really does have the C++ record's layout.
///
/// This is the claim that decides whether a POD operand is a port or a
/// wrapper. If it holds, `KvCacheLayerView` crosses the boundary as itself —
/// no accessor shims, no field-by-field constructor, no copy — and every
/// other descriptor in the launcher surface is the same kind of thing.
///
/// # This claim RETIRES WITH `bridge`, and the headers go with the archive
///
/// The question asked of step 6 was whether the five headers
/// [`CLOSURE_SURVIVES_STEP_6`] names need a home outside `kernels-cuda`,
/// since `del-archive` deletes the crate they live in. **They do not**, and
/// the reason is that this claim's subject is an ABI rather than a file.
///
/// Measured, after `bridge`:
///
///   * the generated shim is the only thing that ever passed these structs
///     across a C++ boundary, and it dies with `native`;
///   * `kernels-cuda-new/csrc` — the carried NVRTC set — DECLARES none of
///     them. `attention_flashinfer_common.cuh` names three and has zero
///     includers; `attention_xqa.cuh` names `AttentionWorkspaceView` once,
///     in a comment about host arithmetic;
///   * a struct that really does cross to NVRTC goes through `by_value!`,
///     and the tree has exactly one instance — `x/xqa.rs`'s `KvCacheList`,
///     a vendored XQA type under `csrc/src/attn/xqa/`, none of these.
///
/// So after `bridge` there is no second description of these layouts, and
/// moving the headers into `driver-cuda/tests/` would keep this suite
/// checking a Rust struct against a transcription owned by the test that
/// checks it. That is the `attn/attention_naive_paged.cu` defect this
/// migration already retired once: its `static_assert`s were kept as "the
/// only mechanical link in the chain" after the pair they compared had
/// stopped being the live pair.
///
/// The live pair for a struct that crosses is Rust ↔ carried device text,
/// and `by_value!` is where that is asserted. This suite is the live pair
/// for as long as the shim compiles, and not one commit longer.
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

/// The `.hpp` a family declares its ahead-of-time launchers in, or **nothing
/// at all** once the family has left the archive.
///
/// # A missing directory is a fact, not an error
///
/// This used to `.expect("family directory")`, and that was right while every
/// family had one. `norm/`, `ssm/`, `mlp/`, `sample/`, `quant/`, `gemm/`,
/// `comm/` and the rest are now gone from `kernels-cuda/csrc/src` — their
/// kernels are NVRTC's and their host programs are Rust — so the read panics
/// for a family that finished rather than a family that broke. The panic is
/// also misattributed: it fires in whichever test runs first and names a
/// directory, not the deletion that removed it.
///
/// So absence returns the empty set. What that means downstream is exact and
/// worth stating rather than leaving to be inferred: `emit_c_shim` writes one
/// entry per row whose declaration it can find, so a family with no headers
/// contributes no entries — which is the correct answer for rows now carried
/// by `device::JIT_DISPATCHED` or `execution::RUST_SERVED`, and the same
/// answer the shim would give if the headers were present and empty.
///
/// **The cost is that a test over an empty set is vacuous and still green.**
/// That is the honest trade here — the alternative is a hard-coded list of
/// which families are supposed to still exist, which is a second denominator
/// to keep in step, and §21's recurring defect in this tree is exactly a gate
/// asserting its own denominator. The countdown is `kernels-cuda/tests/
/// sources.rs`'s census; it is that file's job to notice a family leaving,
/// and it re-derives by walking.
fn headers_in(dir: &str) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(csrc().join(dir)) else {
        return Vec::new();
    };
    let mut hs: Vec<String> = entries
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

// The one `sample` row IS GONE, and this test with it.
//
// It read:
//
// > The one `sample` row. Its weight is the table's only `const int8_t*`
// > (`I8s`) — a fused GEMV+argmax over an int8 lm_head with per-channel
// > fp32 scales.
//
// §5 step 5 took `sample` into fn-world as `x::sample`, whose one contract
// derives a row that states no `operands` — so `prove_family`'s first
// assertion ("`n` of `m` rows are unstated, so the shim silently skips
// them") is now true BY CONSTRUCTION for every fn-world row and says
// nothing. There is also no shim to compile: `sample`'s launcher is
// `x::sample::lm_head_gemv_argmax_int8`, a Rust `fn`, and the C++ entry
// point it used to prove has no caller.
//
// **The `I8s` measurement survives** and is worth keeping in reach: it was
// this table's only `const int8_t*` operand, and it is now the `*const i8`
// parameter of `x::sample`'s `lm_head_gemv_argmax_int8` declaration —
// checked by the typecheck translation unit against the `.cuh` rather than
// by a shim compile, which is the same proof one layer down.

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
    // `prove_family("driver-internal", DRIVER_KERNELS, &headers)` STOOD HERE.
    //
    // It proved that every `driver_internal` row's symbol had a matching
    // `extern "C"` declaration in the archive's headers. §5 step 5 deleted
    // the rows: the six launchers are `fn`s in `x::driver_internal` with no
    // `contract!`, they are called directly from Rust, and there is no
    // declaration for them to agree with. The two `headers.push` lines above
    // are kept because the vision headers are still parsed by the callers
    // below.
    let _ = &headers;
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
    // A family that has left the archive declares nothing, and that is the
    // answer rather than a panic. See `headers_in`'s header for the argument
    // and for what an empty set costs the tests that read it.
    let Ok(entries) = std::fs::read_dir(csrc().join(dir)) else {
        return out;
    };
    for entry in entries {
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

    // `table::norm::KERNELS` until §5 step 5 took `norm` into fn-world.
    // `x::norm::SIGS` is the same set of symbols, derived from the
    // `contract!` block instead of written by hand -- and it is TWO LONGER,
    // because `norm::add_bias_bf16` and `norm::rmsnorm_gated_fp32_in_bf16`
    // are lowered symbols that never had an ahead-of-time row and reached
    // the driver only through `families::norm`'s JIT rows. The port gave
    // them contracts, because a symbol a lowering can state and no contract
    // declares reaches `Route::Unknown` and refuses the model at load.
    //
    // SO ALL THREE `EmitterChosen` EXCEPTIONS NOW HAVE A ROW. That does not
    // fail anything -- `undecided` only shrinks, and `stale` asks whether a
    // header still DECLARES the launcher, not whether a row exists -- but it
    // does mean the exceptions list is no longer the reason those three are
    // decided. It is kept, and kept honest by the `EmitterChosen` check at
    // the bottom, which reads `lower.rs` and is the claim that actually
    // matters: these three are chosen by the emitter, and now they are also
    // declared in `x::norm`, which is the same statement made twice rather
    // than a contradiction.
    let has_row = |n: &str| {
        kernels_cuda_new::x::norm::SIGS
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
    // 60 KB, WAS 100 KB, AND THE MOVE IS THE POINT OF THE NUMBER. The old
    // floor was measured with `driver-cuda/csrc` in `cu_roots`. That tree is
    // deleted, and this scan — comment lines, `void <name>(` lines and
    // `"`-quoted spans removed — now reads 73,055 bytes from the archive
    // alone. A floor left at 100_000 would fail for the one reason a vacuity
    // guard must never fail for: the tree really did get smaller. A floor
    // raised to 73_000 would break on the next honest deletion. 60_000 is
    // ~82% of what is there, which is a scan that stopped, not a tree that
    // shrank — and `attention_xqa*`, the largest bodies left, are owned by
    // another pass, so this number is expected to move again with them.
    assert!(
        cu_text.len() > 60_000,
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

/// `MoeActivation`'s three variants each still name a kernel that exists.
///
/// # This was `the_enum_mirrors_carry_the_cpp_discriminants`, and it was RED
///
/// It generated a TU that `#include`d `moe/flashinfer_moe.hpp` and
/// `static_assert`ed `sizeof(MoeActivation)` and the three discriminants
/// against that header. **The header was deleted in `efaad26b4` — "NVCC
/// LEAVES THE BUILD: the fused CUTLASS leg retired, 66 files deleted".**
/// Neither include path [`compile`] passes can supply it: the stub directory
/// holds `cuda_runtime.h` and `cublas_v2.h` and nothing else, and [`csrc`]
/// has no `moe/` directory at all. So `g++` failed on a missing file, every
/// time, and the failure arrived as
///
/// ```text
/// an enum mirror disagrees with the C++:
/// ```
///
/// A discriminant-mismatch diagnosis for a deleted-header cause. This tree's
/// recurring instrument failure is that breakage and pass produce the same
/// observation; this is the polarity reversed and it is worse, because the
/// message actively points the reader at the Rust enum — the one part of the
/// arrangement that was still correct.
///
/// # The claim moved, and NOT the way `Mxfp4RowSelect`'s did
///
/// [`the_mxfp4_row_select_mirror_matches_the_device_text`] is the sibling
/// migrated in the same pass, and its shape is `include_str!` the `.cuh` and
/// match `constexpr int kRowSelectEven = 1;`. **That shape is not available
/// here**, and the reason is the finding rather than an obstacle: there is no
/// number left to match. `csrc/src/moe/` is nine headers and forty-two
/// `__global__`s and not one of them takes an activation selector — no enum,
/// no `constexpr int kAct*`, no `int act` parameter. `act` in those signatures
/// is always `const T* __restrict__ act`, an activation TENSOR. `moe::silu` is
/// a `__tile__` inline in `moe_fused_tile.cuh`, which declares no `__global__`
/// at all.
///
/// The three activations did not die with the enum. They became three
/// separate `__global__` templates in `mlp/swiglu.cuh`, each with its own
/// row — so the selection that used to be a VALUE passed to one kernel is now
/// the SYMBOL of which kernel fires:
///
/// | variant | `__global__` | row |
/// |---|---|---|
/// | `Relu2` | `swiglu.cuh:247` | `mlp::relu2_bf16` (`x/mlp.rs:229`) |
/// | `Swiglu` | `swiglu.cuh:135` | `mlp::swiglu_bf16` (`x/mlp.rs:182`) |
/// | `Geglu` | `swiglu.cuh:230` | `mlp::geglu_tanh_bf16` (`x/mlp.rs:220`) |
///
/// So what has to hold is no longer "Rust's `1` is C++'s `Swiglu`" — there is
/// no C++ `MoeActivation` to be `1`, in either `csrc/` tree — but that **each
/// variant still names a kernel this tree has**, at both ends: the device text
/// declares it, and [`kernels_cuda_new::unit::unit_of`] says some unit will
/// compile it. A variant whose kernel was renamed or retired is then a named
/// failure here rather than a `MoeActivation::Geglu` nothing can dispatch.
///
/// # Telling a missing header from a wrong discriminant
///
/// The defect above was that one panic covered both causes. This separates
/// them structurally, at four levels with four messages:
///
/// 1. **A moved or deleted header is a COMPILE error.** `include_str!`
///    resolves at build time, so `swiglu.cuh` going away cannot reach a
///    runtime message at all — which is the whole reason the sibling reads
///    the `.cuh` instead of `#include`ing it.
/// 2. **A file that exists but is not the one meant** fails the namespace
///    assertion, which names the namespace and no variant.
/// 3. **A renamed kernel** fails naming the variant, the symbol and the exact
///    `__global__` spelling looked for.
/// 4. **A row that left the JIT** fails `unit_of`, naming the row and saying
///    which of the two ends moved.
///
/// # The half of `sizeof` that survived
///
/// The old TU asserted `sizeof(MoeActivation)` against the C++. There is no
/// C++ side left, so what remains is the Rust-internal claim the mirror rests
/// on: `#[repr(i32)]` is four bytes, and [`kernels::Ty::MoeActivation`]'s
/// `rust()` spelling has to be a four-byte integer for a row ever carrying
/// that kind to marshal into a `void**` cell against its parameter.
///
/// **It is `"u32"` today against an `#[repr(i32)]` mirror.** That is a real
/// disagreement and it is invisible: `Ty::MoeActivation` has ZERO writers —
/// its only row was `moe::flashinfer_cutlass_moe_bf16` and it went in
/// `0dc8e9e9b`, one commit before the header — so nothing marshals one, and
/// the signedness cannot bite until a row returns. The assertion below is
/// deliberately "a four-byte integer, either sign": pinning `"u32"` would
/// freeze the disagreement, and pinning `"i32"` would be red until somebody
/// fixed it. It goes red on a WIDTH change, which is the half that corrupts a
/// launch rather than a value.
#[test]
fn the_moe_activation_mirror_names_kernels_that_exist() {
    use driver_cuda::bind::abi::MoeActivation;

    let cuh = include_str!("../../kernels-cuda-new/csrc/src/mlp/swiglu.cuh");
    assert!(
        cuh.contains("namespace pie_cuda_driver::kernels::mlp::device {"),
        "`mlp/swiglu.cuh` does not open `pie_cuda_driver::kernels::mlp::device`, so it is \
         not the header the `MoeActivation` variants were mapped onto and every assertion \
         below is about the wrong file. This is the WRONG-FILE failure and it is \
         deliberately not one of the per-variant ones -- a variant naming a kernel that \
         moved has a different fix from a header that moved."
    );

    for (variant, global, row) in [
        (MoeActivation::Relu2, "relu2", "mlp::relu2_bf16"),
        (MoeActivation::Swiglu, "swiglu", "mlp::swiglu_bf16"),
        (MoeActivation::Geglu, "geglu_tanh", "mlp::geglu_tanh_bf16"),
    ] {
        // The trailing `(` is what separates `swiglu` from `swiglu_clamp`,
        // which is the neighbouring `__global__` in this very file and the
        // one a bare substring test would accept for `Swiglu`.
        let decl = format!("__global__ void {global}(");
        assert!(
            cuh.contains(&decl),
            "`MoeActivation::{variant:?}` selects `mlp::device::{global}`, and \
             `mlp/swiglu.cuh` does not say `{decl}`. The activation the variant names has \
             been renamed or retired in the DEVICE TEXT, so a caller reaching for it holds \
             a discriminant and no kernel."
        );
        assert!(
            kernels_cuda_new::unit::unit_of(row).is_some(),
            "`MoeActivation::{variant:?}` selects the row `{row}` and no unit hosts it, so \
             the symbol cannot be JIT-compiled. `mlp/swiglu.cuh` still declares \
             `{global}` -- this is the ROW leaving the JIT, not the kernel leaving the \
             header, and the fix is in `x/mlp.rs` rather than in `csrc/`."
        );
    }

    assert_eq!(
        core::mem::size_of::<MoeActivation>(),
        4,
        "`abi::MoeActivation` is `#[repr(i32)]` and must stay four bytes: it crossed BY \
         VALUE, and a `cuLaunchKernel` cell narrower than its parameter mis-marshals every \
         argument after it."
    );
    let spelling = kernels::Ty::MoeActivation.rust();
    assert!(
        matches!(spelling, "i32" | "u32"),
        "`Ty::MoeActivation.rust()` is `{spelling}`, which is not a four-byte integer. The \
         mirror is `#[repr(i32)]`, so a row carrying this kind would put a cell of one \
         width against a parameter of another. Either SIGN is accepted here on purpose -- \
         see this test's doc: `u32` against an `i32` mirror is a real disagreement, it is \
         unreachable while the kind has no writers, and pinning either spelling would make \
         this test the thing that has to change when it is fixed."
    );
}

/// `Mxfp4RowSelect`'s three discriminants still match the device text.
///
/// The half of the old `the_enum_mirrors_carry_the_cpp_discriminants` that
/// lost its header FIRST — the other half lost its own in `efaad26b4` and is
/// now [`the_moe_activation_mirror_names_kernels_that_exist`], so the pairing
/// this sentence used to draw has collapsed: **neither mirror has a C++ enum
/// to be compared against any more, and both read text.** They did not land in
/// the same place, and the difference is worth the sentence. `Mxfp4RowSelect`
/// kept its three NUMBERS — the kernels take the underlying `int` and
/// `mxfp4_marlin.cuh` names the values — so its claim is still an equality.
/// `MoeActivation` did not: its activations became three separate `__global__`
/// templates, so its claim became existence, not equality.
///
/// `abi::Mxfp4RowSelect` is `#[repr(i32)]` and reaches the kernel as
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

/// The two `ArgError` copies say the SAME thing about the same operand.
///
/// # A mirror is only a mirror while something looks
///
/// `driver_cuda::bind::device::ArgError` and
/// `kernels_cuda_new::runtime::args::ArgError` are two independent `Display`
/// impls over the same three variants, and each one's `is_pointer` carries a
/// comment naming the other and saying the lists *"are byte-identical and
/// must stay so"*. Nothing looked. The `Unsupported` clause had drifted
/// already: it named `Ty::Stream` and not `Ty::CublasHandle`, in both copies,
/// which is the failure mode where a mirror stays consistent by being equally
/// wrong on both sides — so equality alone is not the whole check and the
/// clauses are asserted by CONTENT below as well.
///
/// The two handles are not interchangeable and the messages must differ: a
/// stream is `cuLaunchKernel`'s sixth parameter, a cuBLAS handle belongs to
/// the service that issues the launch. `Ty::Dtype` is the control — a kind
/// with no clause, where both copies must still agree and neither may invent
/// one, so a passing run cannot be explained by every message being the same
/// string.
///
/// Since `849be7f2e` neither handle reaches a JIT'd launch: `Unit::source`
/// calls `Unit::typecheck`, so `abi::device_typecheck` refuses the whole row
/// set before NVRTC sees it. These messages are the last resort for a path
/// that did not go through a `Unit`, and a last resort that lies is worse
/// than one that is missing.
#[test]
fn both_arg_error_copies_diagnose_a_handle_the_same_way() {
    use driver_cuda::bind::device::ArgError as Theirs;
    use kernels::Ty;
    use kernels_cuda_new::runtime::args::ArgError as Ours;

    let cases: &[(Ty, Option<&'static str>)] = &[
        (Ty::Stream, Some("a stream is a launch argument")),
        (Ty::CublasHandle, Some("a cuBLAS handle is the service's")),
        // The control. If this one grew a clause the two `contains` checks
        // above would still pass and the test would be asserting nothing
        // about WHICH kinds are called out.
        (Ty::Dtype, None),
    ];

    // Symbol and operand held fixed so that `ty` is the only variable, and
    // the `assert_ne!` at the end is about the KIND rather than about two
    // messages that were never going to match anyway.
    const SYMBOL: &str = "gemm::act_x_wt_bf16";
    const OPERAND: &str = "handle";

    let mut rendered = Vec::new();
    for &(ty, clue) in cases {
        let theirs = Theirs::Unsupported { symbol: SYMBOL, operand: OPERAND, ty }.to_string();
        let ours = Ours::Unsupported { symbol: SYMBOL, operand: OPERAND, ty }.to_string();
        assert_eq!(
            theirs, ours,
            "the two `ArgError::Unsupported` copies disagree about `{ty:?}`. They are a \
             mirror by hand -- see either `is_pointer`'s comment -- so one crate now \
             tells a caller something the other does not. Change both or neither."
        );
        match clue {
            Some(clue) => assert!(
                ours.contains(clue) && ours.ends_with("so this row is unported"),
                "`{ty:?}` lost its clause. A handle in a row is not an unsupported type, \
                 it is a row that has not been ported -- `execution`'s `RUST_SERVED` doc \
                 for the cuBLAS one, `cuLaunchKernel`'s signature for the stream -- and \
                 the bare message sends the reader to the marshaller instead: {ours}"
            ),
            None => assert!(
                !ours.contains(" -- "),
                "`{ty:?}` grew a clause it should not have, which makes the two \
                 assertions above pass without discriminating anything: {ours}"
            ),
        }
        rendered.push(ours);
    }

    assert_ne!(
        rendered[0], rendered[1],
        "the two handle kinds render identically, so the message says a handle was found \
         rather than which one -- and the two belong to different owners: the launch, and \
         the service that issues it"
    );
}

/// The two `is_pointer` lists admit the SAME kinds, as a set.
///
/// # The sibling of the test above, and the half it did not cover
///
/// `both_arg_error_copies_diagnose_a_handle_the_same_way` asserts the two
/// crates SAY the same thing. This asserts they DO the same thing. Each
/// `is_pointer` carries a comment naming the other and stating the lists
/// *"are byte-identical and must stay so"*, with the consequence spelled out:
/// a `Ty` accepted by one and refused by the other is a symbol that binds in
/// one crate and not the other.
///
/// **They had diverged, and the sentence was the only thing guarding them.**
/// `Ty::Bf16sMut` and `Ty::F16sMut` went into `kernels_cuda_new`'s list when
/// the variants were minted and never into `driver_cuda`'s. Nothing noticed,
/// because `x::abi`'s `ptr_abi!(bf16, …)` tagged every `*mut bf16` parameter
/// `Ty::BufMut` and no row could present the kinds. The moment that tag moved
/// — 269 operand positions across 172 rows — `driver_cuda::bind::device`
/// would have taken `Args::bind`'s catch-all arm and answered
/// *"which a device entry point cannot take"* about an ordinary device
/// pointer.
///
/// # Why the text and not the behaviour
///
/// Both `is_pointer`s are private `const fn`s, so there is nothing to call.
/// `Args::bind` is reachable but answers for one `Ty` at a time and only
/// through a `KernelSig`, which would make the population of this test *the
/// kinds some row happens to state* — the same "measured against itself"
/// shape that let the divergence live. `include_str!` takes the whole list
/// from each file, so a kind added to one and not the other is caught whether
/// or not any row states it yet.
///
/// It also buys the property `#[cfg]` and dead code cannot: a moved or
/// renamed file is a COMPILE error here, not a runtime message about
/// something else. That is the distinction `the_enum_mirrors_carry_the_cpp_discriminants`
/// was migrated for.
#[test]
fn both_is_pointer_lists_admit_the_same_kinds() {
    /// The `matches!` arm list out of one `is_pointer`, as a sorted set of
    /// variant names.
    ///
    /// Deliberately not a string compare of the two bodies. The lists are
    /// hand-written in two crates and their ORDER is not the claim — a reader
    /// who groups the buffer kinds differently in one file has changed
    /// nothing about which kinds bind. What must agree is the set, and saying
    /// so is what keeps this test from going red for a reason it does not
    /// name.
    fn kinds(src: &str, file: &str) -> Vec<String> {
        let at = src
            .find("const fn is_pointer(ty: Ty) -> bool {")
            .unwrap_or_else(|| panic!("{file} has no `is_pointer`, so this test reads nothing"));
        let body = &src[at..];
        let end = body.find("\n}\n").unwrap_or_else(|| panic!("{file}'s `is_pointer` never ends"));
        let mut out: Vec<String> = body[..end]
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .flat_map(|l| l.split('|'))
            .filter_map(|t| t.trim().trim_end_matches(',').strip_prefix("Ty::"))
            .map(|t| t.split_whitespace().next().unwrap_or(t).to_string())
            .collect();
        out.sort_unstable();
        out.dedup();
        // The extractor proves it can return non-zero before any comparison
        // is believed. Two lists that both parsed to nothing are equal, and a
        // rename of the function or a reformat of the `matches!` would
        // produce exactly that.
        assert!(
            out.len() > 15,
            "{file}'s `is_pointer` parsed to {} kinds ({out:?}). Two empty lists compare \
             equal, so this test would pass by failing to read either one",
            out.len()
        );
        out
    }

    let ours = kinds(
        include_str!("../../kernels-cuda-new/src/runtime/args.rs"),
        "kernels-cuda-new/src/runtime/args.rs",
    );
    let theirs = kinds(
        include_str!("../src/bind/device.rs"),
        "driver-cuda/src/bind/device.rs",
    );

    let only_ours: Vec<&str> = ours
        .iter()
        .filter(|k| !theirs.iter().any(|t| t == *k))
        .map(String::as_str)
        .collect();
    let only_theirs: Vec<&str> = theirs
        .iter()
        .filter(|k| !ours.iter().any(|t| t == *k))
        .map(String::as_str)
        .collect();
    assert!(
        only_ours.is_empty() && only_theirs.is_empty(),
        "the two `is_pointer` lists disagree. Only in `kernels-cuda-new`: {only_ours:?}; \
         only in `driver-cuda`: {only_theirs:?}. A `Ty` in one list and not the other is \
         a row that binds in one crate and is refused in the other, with a message \
         saying a device entry point cannot take it -- change both or neither"
    );

    // AND THE TWO KINDS THIS TEST WAS WRITTEN FOR ARE IN BOTH, named rather
    // than left to the set comparison. Two lists that had BOTH lost them
    // would satisfy everything above, and that is the state the tree was in
    // for `driver-cuda` alone.
    for want in ["Bf16sMut", "F16sMut", "Bf16s", "F16s"] {
        assert!(
            ours.iter().any(|k| k.as_str() == want) && theirs.iter().any(|k| k.as_str() == want),
            "`Ty::{want}` is not bound as a pointer by both crates. `x::abi` tags \
             `*mut bf16` `Ty::Bf16sMut` and 269 operand positions state it"
        );
    }
}
