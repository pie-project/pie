#pragma once

// The attention scratch, as a kernel sees it.
//
// The buffers themselves are owned by an `AttentionWorkspace` in the driver
// (`driver-cuda/csrc/src/attention_workspace.hpp`), which is where they
// belong: allocating them, rotating the pinned plan-staging slots and
// fencing those slots on CUDA events are scheduler concerns, sized by the
// driver's run-ahead depth. None of that is a kernel's business, and the
// measurement says the kernels agree -- across all of `attn/` they read
// exactly the five values below and call nothing else.
//
// So the class stays home and only this crosses. The same one-way rule as
// `attn/kv_cache_view.hpp`: the driver reaches down, nothing here reaches
// up. It is also what lets a launcher be called from a driver that has no
// C++ objects at all -- this is standard-layout, so a `#[repr(C)]` mirror
// on the Rust side is a provable equivalent rather than a hopeful one.

#include <cstddef>

namespace pie_cuda_driver {

struct AttentionWorkspaceView {
    /// Device scratch FlashInfer accumulates split-KV partials into.
    void* float_buffer;
    /// Size of `float_buffer`. Kernels check their budget against it.
    std::size_t float_bytes;
    /// Device scratch holding per-request scheduling metadata (request
    /// indices, KV tile indices, `o_indptr`, chunk sizes).
    void* int_buffer;
    /// Size of `int_buffer`.
    std::size_t int_bytes;
    /// Pinned host mirror of `int_buffer`, staged by a plan and uploaded by
    /// the driver. This is the active slot: which one that is rotates per
    /// step, and the rotation is not visible from here.
    void* page_locked_int;
};

}  // namespace pie_cuda_driver

// ── Status: ZERO compilable consumers. Its last live edge is gone ─────────
//
// `attn/attention_mla.hpp` and `attn/attention_mla.cu` are DELETED. The
// census below was taken when one of the four edges was still live; it is
// kept verbatim because it is the measurement that decided this file's fate,
// and the fourth line is now the same kind of dead as the other three.
//
// Measured across the whole workspace, this header had four `#include`rs and
// three of them were dead text:
//
//   attn/attention_flashinfer_hopper.hpp   dead: no compilable includer
//   attn/attention_xqa.hpp                 dead: its two includers are
//                                          `oracle.cpp`s in DEAD oracles
//   attn/attention_flashinfer.hpp          dead: its only includer is
//                                          `attention_flashinfer_common.cuh`,
//                                          which has zero includers
//   attn/attention_mla.hpp                 LIVE: `attention_mla.cu:1`
//
// The rule that separates them, written out at `csrc/CMakeLists.txt` near
// :926: AN `#include` FROM A TRANSLATION UNIT THAT CANNOT BE COMPILED IS NOT
// A CONSUMER. Three of the four edges above exist as text and can never be
// traversed by a compiler, which makes them exactly as much evidence for
// keeping this file as no edge at all.
//
// So the count that mattered was ONE, and `attention_mla.cu` was the
// archive's last `.cu` with `<<<>>>` in it -- the two launches that stood
// between this tree and nvcc-zero. It landed: its FA2 arm is
// `kernels-cuda-new/src/x/attn.rs::mla_fa2`, its naive arm is
// `driver-cuda/src/fire/mla_naive.rs`, and its row crossed to
// `x::attn::ATTENTION_MLA`, which is what took the shim entry the `.cu` was
// the definition of.
//
// **THE COUNT IS NOW ZERO AND THIS FILE GOES WITH THE ARCHIVE.** It is left
// on disk in the same change that emptied it, deliberately: its three
// remaining includers are still on disk too, and deleting a header whose
// includers exist is a different measurement from deleting the one
// translation unit that compiled them. There is no unit to port it to,
// because everything it describes is a host-side view of a workspace the
// driver now plans in Rust.
//
// This block exists so that the next reader who greps for includers of this
// file counts four and stops -- and now, so that a reader who finds four
// edges and no compiler knows all four are text.

// ── CORRECTION: the table above is WRONG, and the way it is wrong is the ──
// ── most useful thing measured this week ─────────────────────────────────
//
// It says this header's one live edge was `attention_mla.hpp`, and that when
// `attention_mla.cu` crossed there would be zero compilable consumers. The
// `.cu` and the `.hpp` are now both deleted -- `sweep-attn` crossed
// `attn::dispatch_attention_mla_bf16` and the tree is at nvcc-zero -- so by
// that table this file should now be reachable by nothing.
//
// IT IS COMPILED ON EVERY `cargo test`, and so are four others:
//
//   driver-cuda/tests/launch_abi.rs
//     MIRROR_HPPS = [ attn/kv_cache_view.hpp,
//                     attention_workspace_view.hpp,
//                     attn/mla_cache_view.hpp,
//                     attn/attention_flashinfer_hopper.hpp ]
//     -> kernels_cuda_new::abi::emit_layout_assertions writes an `#include`
//        line per entry into a generated `shim.cpp` of `static_assert`s, and
//        `compile()` runs `g++ -std=c++20 -fsyntax-only` on it against
//        `tests/oracle/launch_abi/stub` (two files: `cuda_runtime.h`,
//        `cublas_v2.h`) plus `-I csrc`.
//
// Transitive closure of that TU, measured: the four above plus `tensor.hpp`,
// reached through `kv_cache_view.hpp`. Five headers, all with a real
// compilable consumer, and no unresolved include in the closure.
//
// # Why the sweep missed it
//
// The rule was right: an `#include` from a translation unit that cannot be
// compiled is not a consumer. What it needed was its DUAL, which nobody had
// written down: **a translation unit that CAN be compiled need not contain a
// literal `#include` anywhere in the tree.** These four are named in a Rust
// `const &[&str]`, and the include lines exist only in generated text. A grep
// for `#include "attention_workspace_view.hpp"` finds three dead headers and
// misses the one consumer that runs.
//
// Both halves of the rule are about the same mistake -- counting text instead
// of counting compilations -- and they fail in opposite directions. The first
// half overcounts: an edge that exists and can never be traversed. The second
// half undercounts: an edge that is traversed and does not exist to grep.
//
// # What this means for step 6
//
// **These five survive the archive.** `launch_abi.rs` needs `g++`, a two-file
// stub tree, and `-I csrc`. It needs no nvcc, no `bridge`, no `native`, and
// nothing from this archive's CMakeLists. When `ROW_TABLES` empties and
// `bridge` goes, that test still compiles these headers and still checks that
// `#[repr(C)]` mirrors match the C++ records -- which is the claim that
// decides whether a POD operand crosses as itself.
//
// So `.cpp`-zero and nvcc-zero DO NOT imply this file can be deleted, and the
// thing that compiles it is not the archive. Deleting it silently turns
// `the_mirrors_have_the_layout_the_cpp_has` and every mutation case beside it
// into a compile failure with no obvious cause -- and `MIRROR_HPPS`'s own doc
// records that exact bug happening once already: "those cases assert a TU
// fails to compile, so a missing include would make them pass for the wrong
// reason."
//
// Still dead, and unchanged by this: `attention_xqa.hpp` (two includers, both
// `oracle.cpp` in oracles `oracle_census.rs` lists DEAD) and
// `attn/attention_flashinfer.hpp` (one includer,
// `attention_flashinfer_common.cuh`, which has none).

// ── SECOND CORRECTION: there is a SECOND live compiler, and it is the ─────
// ── production one. The paragraph directly above is wrong. ───────────────
//
// Re-measured after `sweep-attn` reached nvcc-zero. Two sentences in the
// block above are false and this one names them:
//
//   "IT IS COMPILED ON EVERY `cargo test`, and so are four others"
//        -- true, and it is also compiled on every `cargo build` that turns
//        `native` on, by a compiler that block never looked at.
//   "Still dead, and unchanged by this: `attention_xqa.hpp` … and
//    `attn/attention_flashinfer.hpp`"
//        -- both are LIVE. Neither has a written `#include` from anything
//        compilable; both are compiled anyway.
//
// # The second compiler
//
//   kernels-cuda/build.rs::shim()             (feature `native`)
//     includes()          read_dir's twelve family directories and takes
//                         EVERY `*.hpp` it finds -- unfiltered, not driven
//                         by which rows survive
//     emit_c_shim(...)    writes one `#include "<dir>/<name>.hpp"` line per
//                         entry into a generated `shim.cpp`
//     cc::Build           compiles it -- HOST C++, not nvcc -- into
//                         `libpie_launch_shim.a`
//
// `csrc/src/attn/` holds five `.hpp` and no other file. All five are in that
// generated include list. Three of them -- `attention_flashinfer.hpp`,
// `attention_flashinfer_hopper.hpp`, `attention_xqa.hpp` -- open with
// `#include "attention_workspace_view.hpp"`, resolved by the `-I csrc/src`
// that `build.rs` publishes. So the three edges the first block called dead
// text are traversed by a production compile every time `bridge` is on.
//
// # This is the same blind spot, a THIRD time
//
// Turn one: counted written `#include`s in `.cu`/`.cpp` and said no unit
// reaches it. Turn two: found `launch_abi.rs` generating include lines into
// a g++ TU. Turn three: this, `build.rs` generating include lines into a
// `cc` TU. All three misses have one cause, and it is now nameable rather
// than anecdotal:
//
//   `kernels_cuda_new::abi` IS THE TREE'S `#include` GENERATOR. It has three
//   emitters that write `#include` text -- `emit_c_shim` (abi.rs:204),
//   `emit_device_typecheck` (abi.rs:447), `emit_layout_assertions`
//   (abi.rs:677). ANY answer to "who includes this header" that does not
//   consult all three is an undercount, and grep cannot consult them because
//   the filenames live in Rust `&[&str]`s and `read_dir` results.
//
// # What compiles each of these after step 6
//
// The two compilers end on DIFFERENT schedules, which is the whole reason
// the answer is a table and not a sentence. `bridge` is the only thing that
// turns `native` on, so the shim dies when `ROW_TABLES` empties.
// `launch_abi.rs` needs g++, `tests/oracle/launch_abi/stub` (two files) and
// `-I csrc/src`; it needs no nvcc, no `bridge` and nothing from this
// archive's CMakeLists, so it does not.
//
//   attention_workspace_view.hpp   shim + launch_abi   SURVIVES (launch_abi)
//   tensor.hpp                     shim + launch_abi   SURVIVES (launch_abi)
//   attn/kv_cache_view.hpp         shim + launch_abi   SURVIVES (launch_abi)
//   attn/mla_cache_view.hpp        shim + launch_abi   SURVIVES (launch_abi)
//   attn/attention_flashinfer_hopper.hpp
//                                  shim + launch_abi   SURVIVES (launch_abi)
//   attn/attention_xqa.hpp         shim only           DIES with `bridge`
//   attn/attention_flashinfer.hpp  shim only           DIES with `bridge`
//
// The discriminator is membership in `launch_abi.rs`' `MIRROR_HPPS`, and
// that is not arbitrary: a header is in it because a Rust `#[repr(C)]`
// mirror claims to have its layout, and that claim outlives every C++
// compile because the mirror is what crosses. A header not in it declares
// only host declarations the shim forwarded through, and when there is no
// shim there is nothing to declare to.
//
// **That reconciles the constraint nobody had reconciled**:
// `attention_flashinfer_hopper.hpp` was required to outlive the stub that
// was its non-sm90 definition. It does, and not by exemption -- it is a
// mirror header, `launch_abi.rs` compiles it, and the thing that compiles it
// is not the archive.
//
// # What must move before THIS file can go
//
// Nothing, and that is the finding rather than an absence of one. It is not
// waiting on a port: it declares a host-side view of a workspace the driver
// already plans in Rust, and its remaining job is to be the C++ side of a
// layout assertion. It goes when `MIRROR_HPPS` stops naming it, which is
// when `driver-cuda` stops carrying a `#[repr(C)]` mirror of an attention
// workspace -- and deleting it before then turns
// `the_mirrors_have_the_layout_the_cpp_has` and every mutation case beside
// it into a compile failure with no obvious cause.
//
// This file is therefore NOT the status `attention_flashinfer_common.cuh`
// has. That one has includers and no compiler. This one has a compiler and
// (in written text) no includer. They are the two halves of the same rule,
// failing in opposite directions, and they get opposite treatments.
