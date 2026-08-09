//! The device text, carried in the binary: the headers an `#include`
//! resolves against, and the roots a compile is handed.
//!
//! # The rule this file is
//!
//! > **No include path on disk. Includes resolve against a header set carried
//! > in the binary, or they do not resolve at all.**
//!
//! The first authored kernel header compiled by NVRTC in this tree got there
//! by being given nothing to include: `nvrtcCreateProgram` was called with
//! zero headers and zero include names, so `altup_aux.cuh` spelled its own
//! bf16 conversions rather than reaching for `cuda_bf16.h`. That was the right
//! price for one file. It is the wrong price for a family of them — the second
//! family to widen a bf16 restates the arithmetic, and
//! `driver-metal/src/layout/shader.rs` records where that ends: *"restating it
//! is what the 4-bit codecs did before this existed."*
//!
//! Metal solved it by SPLICING — reading the includer, finding the directive,
//! pasting the text. CUDA does not have to. `nvrtcCreateProgram` takes
//! `headers[]` and `includeNames[]`, an in-memory virtual filesystem, so the
//! compiler resolves the directive itself against text this module hands it.
//! That is strictly better than splicing and the reasons are worth stating,
//! because they are why this is not a port of the Metal module:
//!
//! * **The authored source keeps its `#include` lines**, so it stays
//!   nvcc-compilable — which is what lets `kernels-cuda`'s
//!   `abi::emit_device_typecheck` go on turning a drifted row into a build
//!   error instead of a failed fire.
//! * **Include guards work.** A diamond — two headers including a third — is
//!   one definition, because `#pragma once` is evaluated by a preprocessor
//!   rather than approximated by a `HashSet` of paths the splicer has seen.
//! * **Nothing is read from disk.** [`include_str!`] puts every byte in the
//!   binary at build time, so a header cannot be missing, cannot be stale
//!   relative to the rows it was built with, and cannot be found on a machine
//!   that has a CUDA toolkit and not found on one that does not.
//!
//! That last one is the crown jewel this whole crate is arranged around. This
//! crate builds and runs with no CUDA toolkit — layers 1 and 2 do not even
//! ask for a driver — and buying includes with `-I /usr/local/cuda/include`
//! would trade that for a convenience, and would also make the compiled
//! result depend on a host fact the cache key cannot see.
//!
//! # Why the text is no longer borrowed
//!
//! Every path below used to point into `kernels-cuda/csrc/src`, the AOT
//! crate's tree, rather than a `csrc/` of this crate's own. That was the same
//! seam `table` is: while both the AOT and the JIT path must run, **one file
//! is one contract**, and a copied `.cuh` is two contracts waiting to
//! disagree — one compiled by nvcc into an archive, one compiled by NVRTC at
//! run time, and nothing but a reviewer's memory holding them to the same
//! arithmetic. `norm/altup_aux` was two for a whole release, with every test
//! passing on whichever half it exercised.
//!
//! That reasoning held and its conclusion flipped when the JIT became the
//! larger reader: thirty-eight units and roughly a hundred and forty rows
//! compile out of this text on an L40S, against an archive that still needs
//! it but no longer authors it. So the `.cuh` files MOVED — fifty-seven of
//! them, every family, plus the prelude — and the paths below are local. The
//! rule they were protecting is untouched, because a move is not a copy:
//! there is still exactly one definition of every kernel in the tree, and
//! `kernels-cuda`'s `tests/sources.rs` still fails the build if a second
//! appears.
//!
//! The archive finds them through an include directory rather than a
//! dependency. When that was written the reason was arithmetic —
//! `kernels-cuda-new` depended on `kernels-cuda`, so the other direction was
//! a cycle. The tables have since moved here and the edge inverted, and the
//! answer did not change: this crate has no `links` key, so a dependency on
//! it carries no `DEP_*` and can tell a build script nothing about where its
//! files are. `csrc/CMakeLists.txt` states the whole of it.
//!
//! # What used to be deliberately NOT here
//!
//! `cuda_fp16.h`, `cuda_bf16.h`, `cuda_fp8.h`, `cuda_fp4.h`,
//! `cooperative_groups.h`. NVRTC
//! does not bundle the CUDA device headers before 13.3, so including one
//! meant **vendoring** it — a redistribution decision with a `NOTICE` entry
//! and a pinned device ABI behind it, not something a refactor gets to decide
//! on the way past.
//!
//! It was decided, and the answer was neither: the shims in `csrc/shim` are
//! written against the instructions rather than copied out of the toolkit,
//! and they wear NVIDIA's filenames only where the includer is upstream
//! source we do not own (`new-horizon.md` §13.4). They are entries in
//! [`DEVICE_HEADERS`] like any other file in the tree, because the mechanism
//! never cared which headers it carried — which is the argument for having
//! built the set before the decision rather than after it.
//!
//! They sat at the root of `csrc/src` until `csrc/` was re-cut by role, with
//! the six that arrived by vendoring — `cstdint`, `type_traits`, `bit`,
//! `cuda.h`, `cuda_runtime.h`, `boost/math/ccmath/fabs.hpp` — under
//! `csrc/vendor` doing the identical job from the other directory. The
//! fourteen are one thing and now sit in one place, [`SHIM`] — eight of
//! ours and six of theirs. Not one name changed, because for these files the
//! name IS the contract.
//!
//! # Names are relative to the tree the file sits in
//!
//! One spelling, two resolvers — and the second resolver is not the first
//! one's synonym. `#include "pie_device.cuh"` is what NVRTC matches against
//! [`Header::name`]; the same tree reaches nvcc as
//! `-Xcompiler=-iquote,…/kernels-cuda-new/csrc/src`, which is what the
//! archive's CMake passes (`crates/kernels-cuda/csrc/CMakeLists.txt:722`) and
//! what the offline typecheck passes. The same string is `KernelSig::file`,
//! so a row, a `#include` and an entry here cannot drift into three spellings
//! of one file.
//!
//! `csrc/shim` is a second `-iquote` beside it and must be, and the traffic
//! crosses in BOTH directions. Outward: two files in `csrc/src` reach the
//! shims by QUOTED include — `pie_fp8.cuh` says `#include "cuda_fp16.h"` and
//! `#include "cuda_fp8.h"`, `pie_half2.cuh` says the first. Inward:
//! `shim/cuda_fp16.h` and `shim/cuda_bf16.h` both open with
//! `#include "pie_device.cuh"`, which is back in `csrc/src`. Three directives
//! out, two in; every one of them used to resolve beside the includer.
//!
//! Costs nothing to get right and is silent to get wrong — in the outward
//! direction. With the shim moved and no `-iquote` naming its new home, a
//! quoted `"cuda_fp16.h"` falls through to the real toolkit header and
//! `__half` stops being `device::f16` — the exact 17,744-vs-15,088-byte
//! divergence measured below, and no diagnostic. The inward direction is the
//! kind one: there is no `pie_device.cuh` anywhere else, so a missing
//! `csrc/src` fails to resolve and says so. Three call sites pass both, and
//! they are the three places nvcc sees this tree at all:
//! `kernels-cuda/csrc/CMakeLists.txt`'s `target_compile_options` for
//! `pie_kernels_cuda`, `driver-cuda/build.rs`'s `cc::Build` target
//! (`pie_attn_flashinfer`), and `tests/device_typecheck_types.rs`'s
//! `compile`. It was four until `driver-cuda/csrc/vision/` was deleted and
//! `pie_vision_towers` with it: the three multimodal towers are Rust now, so
//! nvcc never sees this tree on their account.
//!
//! One shim-to-shim edge stays where it was: `shim/cuda_bf16.h` says
//! `#include "cuda_fp16.h"` and both are now in `shim/`, so it resolves
//! beside the includer with no flag at all. That is the whole reason `shim/`
//! was the first role to move — the group is nearly closed under its own
//! includes, and NVRTC, which has no `-iquote` and matches the literal name,
//! cannot tell the move happened.
//!
//! ## `-iquote`, and NOT `-I`, and the difference is silent
//!
//! An earlier version of this section said `-I`. It is wrong, and not as a
//! spelling: `-I` is the ANGLE-BRACKET search path, and the section above
//! just finished explaining that this tree holds fourteen shims wearing
//! NVIDIA's and the standard library's filenames — `cuda_fp16.h`,
//! `cuda_bf16.h`, `cuda_fp8.h`, `cuda_fp4.h`, `cooperative_groups.h`,
//! `cstdint`, `type_traits`, `bit`, `cuda.h`, `cuda_runtime.h`,
//! `boost/math/ccmath/fabs.hpp`, `cuda/{cmath,pipeline,std/limits}`. Putting
//! that directory on the angled path puts the impersonation ahead of the real
//! toolkit header UNDER NVCC,
//! where the real one exists and `__half` is not `device::f16`.
//!
//! Since the role cut this is narrower than it was, and deliberately: the
//! shims are no longer AT the root of a directory anybody has a reason to
//! `-I`. `-I …/csrc/src` reaches every kernel header and not one shim.
//! Reaching them wrongly now takes an `-I …/csrc/shim`, which is a line
//! somebody has to write on purpose.
//!
//! Measured here rather than reasoned about. One TU including
//! `pie_device.cuh` and `quant/dequant_wna16.cuh` — whose
//! `#include <cuda_fp16.h>` is the angled reach, and it is TRANSITIVE: the TU
//! itself includes nothing angled — launching `bf16_to_narrow<__half>`, nvcc
//! 13.0.88, `-std=c++20 -arch=sm_89 -c`, identical but for the spelling:
//!
//! ```text
//! -Xcompiler=-iquote,…   17,744 B   bf16_to_narrow<__half>
//! -I …                   15,088 B   bf16_to_narrow<pie_cuda_driver::kernels::device::f16>
//! ```
//!
//! **Both compiled without a diagnostic**, and `nm -C` gives two disjoint
//! symbol sets: the `<<<>>>` in that TU wrote `<__half>`, and the `-I` object
//! exports the OTHER instantiation. `kernels-cuda`'s
//! `examples/emit_device_typecheck.rs` records the same divergence at whole
//! fatbin scale (49,480 B vs 37,816 B, 31% smaller, still no diagnostic).
//!
//! That is this crate's problem and not only the archive's: the JIT finds a
//! kernel by naming its instantiation to `nvrtcAddNameExpression`, so an
//! archive built the `-I` way exports symbols the JIT names differently, and
//! the pair that is supposed to check one against the other agrees.
//!
//! ## What that means for a per-unit header set
//!
//! Which resolver a name reaches depends on the include-path SPELLING, not on
//! the set's contents — so [`crate::unit::Headers`] chooses the contents and
//! cannot choose this. Two consequences, before a second set is added:
//!
//! * under NVRTC there is only the one resolver: the carried set answers both
//!   spellings, which is exactly why the shims work at all. A carried name
//!   shadows unconditionally there, so the `-iquote`/`-I` question does not
//!   arise for [`crate::runtime::nvrtc::compile`] — it arises for every
//!   OFFLINE compile of the same headers, which is where the two halves are
//!   supposed to agree.
//! * a real toolkit header reached with angle brackets — `<crt/cuda_tile.h>`
//!   is the one waiting — is not a shim and is not carried here. It is
//!   answered by the compiler's own bundled headers, which is a statement
//!   about WHICH NVRTC and belongs in a unit's toolchain floor. That the
//!   floor and the header set are two fields rather than one is this
//!   distinction, and a second set that tried to carry `<crt/cuda_tile.h>`
//!   would be answering a version question with a file.

use std::ffi::CString;

/// One header, and the name an `#include` spells to reach it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// What `#include "…"` must say, relative to `csrc/src`.
    pub name: &'static str,
    /// The text, carried in the binary by [`include_str!`].
    pub text: &'static str,
}


/// The headers `csrc/` holds, generated by walking it.
///
/// Spelled out rather than listed by hand, because a list of sixty entries
/// that must be edited to add a file is a rule enforced by memory. See
/// `build.rs`'s `carried` module for the failure mode: a `.cuh` on disk and
/// absent from the array is not an error anywhere — it is an NVRTC *"could
/// not open source file"* on a machine with a GPU, at the first fire of
/// whatever unit needed it.
///
/// [`LIBRARY`] is `csrc/src` minus the internalised subtrees: every kernel
/// header this crate authored and the prelude they are written over. [`SHIM`]
/// is `csrc/shim`: the headers that wear NVIDIA's and the standard library's
/// filenames, because the source that reaches for those names is source we do
/// not own. [`UPSTREAM`] is `csrc/src/attn/flashinfer` and
/// `csrc/src/attn/xqa`: the patched FlashInfer and XQA closure, which is
/// 1.2 MB and goes only to the units that ask for it.
///
/// The third group used to be a third DIRECTORY, `csrc/vendor`. It is a
/// prefix now, because the rule that `csrc/` holds only device text left
/// those files nowhere else to be — internalised, not vendored. What did not
/// change is that they are still upstream's, still 1.2 MB, and still not
/// something a `norm` compile should carry.
mod carried {
    include!(concat!(env!("OUT_DIR"), "/carried.rs"));
}

pub use carried::{LIBRARY, SHIM, UPSTREAM};

/// What every compile in this crate resolves an `#include` against, unless the
/// unit says otherwise.
///
/// [`SHIM`] and [`LIBRARY`], and nothing else. Not [`UPSTREAM`]: NVRTC copies
/// every byte of every header it is handed, and a `norm` kernel has no
/// business paying for an attention library. A unit that needs it says so —
/// with [`crate::unit::Headers::LibraryAndUpstream`], which is the field that
/// used to be missing and the reason `tests/flashinfer_decode.rs` had to
/// reach past `nvrtc::compile` to compile such a unit at all.
///
/// This was a concatenation of two walks — `SHIMS` for `csrc/src` and
/// `LIBRARY` for the sibling tree's `*.cuh` — until the `.cuh` files moved
/// here and the two trees became one. Before that it was three headers
/// written out by hand: the prelude, `rope_device.cuh` and
/// `kv_paged_addr.cuh`, which was right while three was the number. It
/// stopped being right the moment the migration began extracting a `.cuh` per
/// `.cu`: a kernel header that includes a sibling is the normal case, not the
/// exception, and a hand-list is a set that is correct until someone forgets.
/// The generator walks the directory, so the set is the directory.
///
/// It is two walks again, and the second is `SHIM` — but the seam is now
/// where the ROLE changes rather than where the authorship does. Eight
/// impersonating headers used to sit under `csrc/src` and six more under the
/// vendor tree, doing one job from two directories, and the only thing the
/// split recorded was which of the two reasons a name had to be answered:
/// ours asked for it, or upstream's did. The SET is unchanged, file for file
/// and byte for byte; what changed is that a directory now answers *what is
/// this text for* instead of *who wrote it*.
///
/// It is three walks in the generator now and still two groups here, because
/// internalising the attention closure took its directory away without taking
/// away the reason it was separate. See [`ALL_HEADERS`].
const SHIMMED: [Header; SHIM.len() + LIBRARY.len()] =
    join::<{ SHIM.len() + LIBRARY.len() }>(SHIM, LIBRARY);

/// See [`SHIMMED`].
pub const DEVICE_HEADERS: &[Header] = &SHIMMED;

/// [`DEVICE_HEADERS`] plus [`UPSTREAM`] — what a unit compiling upstream
/// source resolves against.
///
/// A const-fn concatenation rather than a `Vec` behind a `OnceLock`, because
/// this must stay a `&'static [Header]` for every reader that already takes
/// one — `Unit::cache_key`, `as_nvrtc_arrays`, the tests — and neither
/// `concat` nor iterator chaining is const. [`crate::table`] builds `KERNELS`
/// the same way for the same reason.
pub const ALL_HEADERS: &[Header] =
    &join::<{ SHIM.len() + LIBRARY.len() + UPSTREAM.len() }>(&SHIMMED, UPSTREAM);

/// `[T] ++ [U] -> [T; N]` at compile time.
///
/// `Header` is `Copy`, which is what makes filling by index legal in a const
/// fn; it is `Copy` because it is two `&'static str` and copying it is copying
/// two pointers.
const fn join<const N: usize>(left: &[Header], right: &[Header]) -> [Header; N] {
    let mut out = [Header { name: "", text: "" }; N];
    let mut w = 0;
    let mut i = 0;
    while i < left.len() {
        out[w] = left[i];
        w += 1;
        i += 1;
    }
    let mut j = 0;
    while j < right.len() {
        out[w] = right[j];
        w += 1;
        j += 1;
    }
    out
}

/// The compilable unit sources, carried the same way their headers are.
///
/// A unit's root is not in [`DEVICE_HEADERS`] — nothing `#include`s it, it is
/// what `nvrtcCreateProgram` is handed as THE source — but it comes from the
/// same place and moves for the same reason, so it is spelled here rather
/// than beside the table that names its rows.
///
/// The module is `roots` and not `sources` because everything in this file is
/// a source; only these are handed to a compile as the thing being compiled.
pub mod roots {
    /// `norm`'s six AltUp auxiliary templates.
    pub const NORM_ALTUP_AUX: &str = include_str!("../csrc/src/norm/altup_aux.cuh");
    /// `norm`'s pointwise pair: `residual_add` and `scalar_mul`.
    pub const NORM_ELEMENTWISE: &str = include_str!("../csrc/src/norm/elementwise.cuh");
}

/// The header set as the two parallel arrays `nvrtcCreateProgram` wants,
/// texts first and names second, in table order.
///
/// Returned as owned `CString`s because NVRTC copies neither: the pointers
/// must outlive the call, and the caller holding the vectors is what makes
/// that visible rather than a comment.
///
/// # Errors
///
/// A header name or its text contains a NUL, which no source can.
pub fn as_nvrtc_arrays(headers: &[Header]) -> Result<(Vec<CString>, Vec<CString>), String> {
    let mut texts = Vec::with_capacity(headers.len());
    let mut names = Vec::with_capacity(headers.len());
    for header in headers {
        texts.push(
            CString::new(header.text)
                .map_err(|_| format!("header `{}` contains a NUL", header.name))?,
        );
        names.push(
            CString::new(header.name)
                .map_err(|_| format!("header name `{}` contains a NUL", header.name))?,
        );
    }
    Ok((texts, names))
}

/// Every carried header reachable from `root`, or the first `#include` the set
/// does not carry.
///
/// Breadth-first from the root, which NVRTC is handed as the source rather than
/// as a header — the only difference is its position in the call. `from` is the
/// name a diagnostic should blame, which is the unit's own name.
///
/// **`headers` is a parameter and that is the whole point of this function.**
/// It used to be [`DEVICE_HEADERS`], spelled inside the test that walked the
/// units, which asked one question of every unit: *does this resolve against
/// the set the crate ships?* The question that decides whether a fire works is
/// narrower — *does this resolve against the set THAT UNIT COMPILES WITH* —
/// and the two differ for the first unit that asks for
/// [`crate::unit::Headers::LibraryAndUpstream`]. A check against a set the
/// compile will not be handed is a check of the wrong thing, and it fails in
/// both directions: an upstream unit looks broken, and a library unit that
/// reaches an upstream header looks fine.
///
/// # Errors
///
/// The message names the unit, the include, and the header it was reached
/// from — which is what a migration needs, because the include is usually not
/// in the file the unit names.
pub fn reachable(from: &str, root: &str, headers: &[Header]) -> Result<Vec<&'static str>, String> {
    let mut seen: Vec<&'static str> = Vec::new();
    let mut queue: Vec<(&str, &str)> = vec![(from, root)];
    while let Some((at, text)) = queue.pop() {
        for included in quoted_includes(text) {
            let Some(header) = headers.iter().find(|h| h.name == included) else {
                return Err(format!(
                    "`{from}` reaches `{included}` from `{at}`, and the header set it \
                     compiles against does not carry it -- NVRTC resolves against the \
                     set and nothing else, so this compiles nowhere"
                ));
            };
            if !seen.contains(&header.name) {
                seen.push(header.name);
                queue.push((header.name, header.text));
            }
        }
    }
    Ok(seen)
}

/// FNV-1a 64 over every header's name and text, in table order.
///
/// **The cache key must span this.** `driver-cuda/src/program/cache.rs`
/// records the lesson in the past tense — a cubin keyed on less than what
/// produced it is served after the thing it was not keyed on changes — and
/// `driver-metal/src/program/cache.rs` keys on the RESOLVED text for the same
/// reason. With NVRTC resolving includes, the resolved text is the root plus
/// every header that can reach it, and this is the half that is not the root;
/// [`crate::unit::Unit::cache_key`] is where the two halves meet.
///
/// Content, not modification time: a rebuild that changes nothing must keep
/// the cache warm, and a header edited back to its old bytes must hit the
/// entry it had before.
#[must_use]
pub fn digest(headers: &[Header]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for header in headers {
        hash = fold(hash, header.name.as_bytes());
        // A separator, so `("ab", "c")` and `("a", "bc")` are different sets.
        hash = fold(hash, &[0]);
        hash = fold(hash, header.text.as_bytes());
        hash = fold(hash, &[0]);
    }
    hash
}

/// Every quoted `#include` in `source`, in order of appearance.
///
/// Column zero only, matching the Metal splicer's rule and for its reason: a
/// directive is a directive only at the start of a line, so the same
/// characters inside a string literal or a comment are left alone. Angle
/// includes are not reported — those name a compiler's own headers, and this
/// set does not carry any.
#[must_use]
pub fn quoted_includes(source: &str) -> Vec<&str> {
    source
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("#include")?;
            let rest = rest.strip_prefix(|c: char| c == ' ' || c == '\t')?;
            rest.trim_start().strip_prefix('"')?.split('"').next()
        })
        .collect()
}

/// FNV-1a 64 over bytes, the fold [`digest`] is built out of.
///
/// Crate-visible because a cache key spans more than the header set — a
/// unit's root text and the instantiations asked out of it move the cubin
/// too — and two spellings of one hash in one crate is how a key and the
/// thing it keys drift apart.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    fold(FNV_OFFSET_BASIS, bytes)
}

/// The algorithm's offset basis, and the same one
/// `driver::tensor_ir::fnv1a64` starts from.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// The algorithm's prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// One more chunk into a running FNV-1a, so a digest over several fields is
/// one hash rather than a hash of hashes.
fn fold(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unit::Headers;

    /// The check that makes the set authoritative: every `#include` REACHABLE
    /// FROM A UNIT names a header the set carries.
    ///
    /// Without this, adding a `#include` and forgetting the entry is an NVRTC
    /// error on a machine with a GPU — the slowest possible place to find out
    /// — and this file's whole claim is that the binary is self-sufficient.
    ///
    /// # Why reachability, and not "every carried header"
    ///
    /// An earlier version asserted over the whole set, which was right while
    /// the set was three headers written out by hand. It stopped being right
    /// when the generator began walking `kernels-cuda/csrc/src`: that tree
    /// held `.cuh` files belonging to the ahead-of-time build, and some of
    /// them included a host `.hpp` the JIT deliberately does not carry —
    /// `attn/attention_flashinfer_common.cuh` was the example, and it has
    /// since left `csrc/` for `kernels-cuda-new/spec/` because a file with no
    /// device text in it is not device text. The reason the check is scoped
    /// this way is unchanged even though its example is gone: the internalised
    /// FlashInfer and XQA closure under `csrc/src/attn/{flashinfer,xqa}` is
    /// carried for the units that ask, and a header no unit reaches is dead
    /// weight rather than a defect — NVRTC reads only what is included.
    ///
    /// What must hold is narrower and is the thing that actually breaks a
    /// fire: **from a unit's root, every include resolves, transitively.** A
    /// header that no unit can reach may say anything at all.
    ///
    /// # Against the set that unit compiles with, not the global one
    ///
    /// [`crate::unit::Unit::header_set`], not [`DEVICE_HEADERS`]. They are the
    /// same set for every unit declared today and they are not the same
    /// question: `nvrtc::compile` hands NVRTC the unit's own set, so a check
    /// against a different one proves nothing about the compile that will
    /// happen. The first unit to ask for the upstream closure is the first one
    /// this distinction saves, and it is the reason the check moved into
    /// [`reachable`] where the set is an argument.
    /// # It asserts its own denominator
    ///
    /// A loop over a set is a test whose strength is the set's size, and a
    /// walk that resolves nothing passes exactly as loudly as one that
    /// resolves everything. That is the shape five gates in this tree have
    /// already been found green-while-blind in — so the counts are checked
    /// here rather than assumed: every unit was visited, and includes were
    /// actually followed. Neither number is pinned to a constant, because the
    /// unit set grows on purpose; what is pinned is that they are not zero and
    /// that none of the walk was skipped.
    #[test]
    fn every_include_reachable_from_a_unit_resolves() {
        let mut visited = 0usize;
        let mut includes = 0usize;
        for unit in crate::unit::UNITS {
            match reachable(unit.name, unit.root, unit.header_set()) {
                Ok(reached) => {
                    visited += 1;
                    includes += reached.len();
                }
                Err(why) => panic!("{why}"),
            }
        }
        assert_eq!(visited, crate::unit::UNITS.len(), "every unit is walked, or this proves less");
        assert!(
            includes > 0,
            "{visited} units and not one include followed -- this test would pass against an \
             empty header set and a root that reaches nothing, which is what it exists to catch"
        );
    }

    /// The set a unit chooses decides what resolves — checked with a header
    /// only the internalised closure carries.
    ///
    /// The proof that the parameter is load-bearing rather than decorative:
    /// the same root resolves against one choice and not the other. Both
    /// halves matter. That `Library` REFUSES it is what keeps a `norm` unit
    /// from silently reaching into 1.2 MB of FlashInfer; that
    /// `LibraryAndUpstream` ACCEPTS it is the thing such a unit needs and
    /// could not previously state.
    ///
    /// Note what this still proves after internalisation: the two sets are
    /// now two halves of ONE directory, `csrc/src`, split by a prefix rather
    /// than by a walk root. If that split ever silently stopped happening,
    /// every name would be in both sets, the `find` below would return `None`
    /// and this test would fail on its own `expect` rather than passing
    /// vacuously.
    ///
    /// The upstream header is found rather than named, so this test does not
    /// have to be edited when the closure is repatched — the failure mode
    /// `rows_is_every_units_rows_in_order`'s comment describes.
    #[test]
    fn a_units_header_choice_decides_what_resolves() {
        let upstream = UPSTREAM
            .iter()
            .find(|v| !Headers::Library.set().iter().any(|l| l.name == v.name))
            .expect("the upstream closure carries a header the library does not");
        let root = format!("#include \"{}\"\n", upstream.name);

        let refused = reachable("a/unit", &root, Headers::Library.set())
            .expect_err("the library set does not carry an upstream header");
        assert!(refused.contains(upstream.name), "a refusal names the include: {refused}");

        let resolved = reachable("a/unit", &root, Headers::LibraryAndUpstream.set())
            .expect("the upstream set carries it");
        assert!(resolved.contains(&upstream.name), "and reports what it reached: {resolved:?}");
    }

    /// Reachability is transitive and reports the header it came THROUGH,
    /// which is what makes a missing include actionable.
    #[test]
    fn an_include_two_headers_deep_is_reached_and_blamed_precisely() {
        let carried = [
            Header { name: "one.cuh", text: "#include \"two.cuh\"\n" },
            Header { name: "two.cuh", text: "#include \"three.cuh\"\n" },
        ];
        let root = "#include \"one.cuh\"\n";

        let why = reachable("a/unit", root, &carried).expect_err("`three.cuh` is not carried");
        assert!(why.contains("`three.cuh`"), "the include that is missing: {why}");
        assert!(why.contains("from `two.cuh`"), "and the header that reaches it: {why}");
        assert!(why.contains("`a/unit`"), "and the unit that would not compile: {why}");

        let full = [
            carried[0],
            carried[1],
            Header { name: "three.cuh", text: "" },
        ];
        assert_eq!(
            reachable("a/unit", root, &full).expect("all three are carried"),
            ["one.cuh", "two.cuh", "three.cuh"]
        );
    }

    /// Every header `csrc/` carries is self-consistent.
    ///
    /// Narrower than the reachability check above and true of a bigger set
    /// than it used to be: `csrc/src` is authored here, so every include in
    /// it must resolve whether or not a unit happens to reach it today. That
    /// promise was made for eleven shims while the kernels lived in the
    /// sibling tree — which got no such promise, because a `.cuh` there is
    /// also the ahead-of-time build's and may say what nvcc can resolve. The
    /// files moved; the promise now covers all of them.
    ///
    /// # THE ONE EXEMPTION IS GONE, BECAUSE ITS SUBJECT LEFT `csrc/`
    ///
    /// This block used to open *"a `.hpp` and the `kernels.def` manifest are
    /// HOST text … `attn/attention_flashinfer_common.cuh` includes six of
    /// them … and it is a device header only by extension"*, and the test
    /// below skipped any quoted include ending in `.hpp` or `.def` to let
    /// that one file through. There was exactly one file in the exemption and
    /// it is no longer in this tree: it is host C++ with `__global__` = 0 and
    /// `__device__` = 0, it has **zero `#include` consumers anywhere in the
    /// workspace**, and `csrc/` holds device text only — so it moved to
    /// `kernels-cuda-new/spec/`, where `spec/README.md` states what it is and
    /// `tests/launch_rules.rs` goes on pinning the lines its twenty-four
    /// citations name.
    ///
    /// The skip went with it. **Measured before removing it: zero quoted
    /// `#include`s ending in `.hpp` or `.def` remain anywhere under `csrc/`.**
    /// (The one surviving `.hpp` spelling in the tree is
    /// `<boost/math/ccmath/fabs.hpp>` in the internalised `fp16.h`, which is
    /// angled — [`quoted_includes`] never saw it — and answered by a shim.)
    /// A rule with no exceptions is a rule; a rule with an exception whose
    /// subject has been deleted is a hole waiting for the next host header.
    ///
    /// So the rule checked is every include of DEVICE text, with nothing
    /// waived. A missing `.cuh` is a failure here, which is the case that
    /// costs a compile on a machine with a GPU.
    ///
    /// # Why [`ALL_HEADERS`] and not [`DEVICE_HEADERS`] or [`LIBRARY`]
    ///
    /// The role cut split one tree into two, and the includes cross the seam
    /// in both directions: `csrc/src/pie_fp8.cuh` reaches
    /// `shim/cuda_fp16.h`, and `shim/cuda_fp16.h` reaches back for
    /// `csrc/src/pie_device.cuh`. Walking [`LIBRARY`] alone would check the
    /// first direction and stop checking the second on the day the second
    /// became possible — a test that got quieter because the tree got more
    /// structure.
    ///
    /// [`DEVICE_HEADERS`] was the right denominator while the third group was
    /// `csrc/vendor` and this test's subject was *what this crate authored*.
    /// Internalising ended that distinction at the filesystem: there is no
    /// vendor directory, every carried file is under `csrc/`, and the fifty-
    /// eight relative directives the closure resolves among itself are now
    /// this crate's problem to keep resolving. So the denominator is the
    /// whole carried set.
    ///
    /// **Measured before widening: [`DEVICE_HEADERS`] leaves eleven quoted
    /// spellings unchecked, and [`ALL_HEADERS`] resolves all of them.** The
    /// eleven are the internalised closure's own cross-references — our
    /// `attn/*.cuh` reaching `attn/flashinfer/attention/*.cuh` and
    /// `attn/xqa/*.cuh`. Under the narrow set those eleven are not failures,
    /// they are simply not looked at, which is the shape of green-while-blind
    /// this file has already been found in five times.
    ///
    /// # The one thing this cannot check, and why it is angled
    ///
    /// `attn/attention_xqa_mha.cuh` includes `<attn/xqa/mha_sm90.cuh>` in
    /// angle brackets, on purpose: that file is deliberately not carried, and
    /// [`quoted_includes`] reads only `"…"`. The bracket style is how a
    /// directive says *do not hold me to this*.
    ///
    /// **Measured across every angled include under `csrc/`** — forty-two
    /// distinct spellings — **exactly two name something this set does not
    /// carry**, and neither is an accident: that one, and
    /// `<rng_contract.generated.h>` in `ptir/tier0.cuh`, which the build
    /// generates. Every other angled spelling is a system or NVIDIA name that
    /// [`SHIM`] answers. So the angle bracket is not a hole in this test; it
    /// is a two-entry list, and both entries are written down here.
    #[test]
    fn every_device_include_resolves() {
        for header in ALL_HEADERS {
            for included in quoted_includes(header.text) {
                assert!(
                    ALL_HEADERS.iter().any(|h| h.name == included),
                    "`{}` includes `{included}`, which the set does not carry",
                    header.name
                );
            }
        }
    }

    /// The set is what a compile is keyed on, so two different sets must not
    /// key the same. A digest that ignored the text would serve a stale cubin
    /// after a header edit — which is the exact failure
    /// `driver-cuda/src/program/cache.rs`'s header records having made once.
    #[test]
    fn the_digest_moves_when_any_header_does() {
        let base = digest(DEVICE_HEADERS);
        assert_eq!(base, digest(DEVICE_HEADERS), "and is stable");

        let edited = [Header {
            name: DEVICE_HEADERS[0].name,
            text: "// not what it was",
        }];
        assert_ne!(base, digest(&edited), "text is in the key");

        let renamed = [Header {
            name: "norm/somewhere_else.cuh",
            text: DEVICE_HEADERS[0].text,
        }];
        assert_ne!(base, digest(&renamed), "and so is the name it resolves by");

        // The separator earns its keep: without it these two sets are one
        // byte stream and one digest.
        let split = [
            Header { name: "a", text: "bc" },
            Header { name: "d", text: "e" },
        ];
        let joined = [Header { name: "ab", text: "c" }, Header { name: "d", text: "e" }];
        assert_ne!(digest(&split), digest(&joined));
    }

    #[test]
    fn only_column_zero_quoted_includes_are_directives() {
        let source = "\
#include \"a.cuh\"
  #include \"indented.cuh\"
#include <cuda_bf16.h>
const char* s = \"#include \\\"in_a_string.cuh\\\"\";
#include\t\"tabbed.cuh\"
";
        assert_eq!(quoted_includes(source), vec!["a.cuh", "tabbed.cuh"]);
    }

    /// The arrays handed to NVRTC are the table, in order and complete.
    #[test]
    fn the_nvrtc_arrays_are_the_table() {
        let (texts, names) = as_nvrtc_arrays(DEVICE_HEADERS).expect("no NULs in a source");
        assert_eq!(texts.len(), DEVICE_HEADERS.len());
        assert_eq!(names.len(), DEVICE_HEADERS.len());
        for (at, header) in DEVICE_HEADERS.iter().enumerate() {
            assert_eq!(names[at].to_str().unwrap(), header.name);
            assert_eq!(texts[at].to_str().unwrap(), header.text);
        }
    }
}
