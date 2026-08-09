//! The two control kernels: readiness before a pass, the commit bump after it.
//!
//! # Why these are compiled here rather than linked
//!
//! The host emitter produces one kind of kernel for CUDA — a fused region per
//! generated region — and emits no readiness kernel and no commit kernel,
//! because on CUDA those are *prebuilt*. The C++ driver's prebuilt copies live
//! in `driver-cuda/csrc/src/pipeline/channels.hpp`, a private header of a
//! crate this one is replacing. They are not in `libpie_kernels_cuda.a`: the
//! kernel table has no row for them, `kernels-cuda/csrc` includes
//! `channels.hpp` nowhere, and `driver-cuda-new --features bridge` therefore
//! links an archive containing neither symbol.
//!
//! So they have to come from somewhere, and there are two honest options: add
//! them to the kernels crate and build them with nvcc, or compile them here
//! through the NVRTC path this driver already has for every emitted region.
//!
//! This file takes the second, and the reason is the toolkit-free build. A
//! kernels-crate row would mean `driver-cuda-new` cannot run a PTIR program
//! without `bridge`, which needs nvcc *at build time* — and that build is
//! load-bearing for CI. Compiling forty lines through the NVRTC path already
//! in `super::nvrtc` costs one compile per process, is cached on disk beside
//! every other region, and needs no toolkit at build time at all.
//!
//! # These are ports and reproduce the C++ exactly
//!
//! Both bodies are transcriptions of `channels.hpp:93-149`, including the
//! `threadIdx.x != 0 || blockIdx.x != 0` guard that makes each a single-thread
//! kernel, and including the ORDER of the commit loops — puts before takes.
//! The order matters for a channel that is both taken and put in one pass (a
//! loop-carried ping-pong): publishing first and consuming second is what
//! leaves such a ring with the value the pass produced rather than with the
//! one it consumed.
//!
//! # The types are spelled in bare C
//!
//! `unsigned char` and `unsigned int`, not `std::uint8_t`/`std::uint32_t`.
//! NVRTC compiles these with **no include path at all** — the same condition
//! the emitted regions are compiled under — so `<cstdint>` cannot be found and
//! `std::` does not exist. The M1 runtime prologue solves the same problem the
//! same way, with its `m1_u8`/`m1_u32` typedefs, and says so in its own
//! header comment.

use std::sync::Arc;

use driver_pipeline::Failure;

use super::disk::{Disk, disk_key};
use super::module::Module;
use super::nvrtc::{self, FailureKind};

/// The ring's fixed row pitch in the `full` array.
///
/// `full` is two-dimensional and indexed `full[channel * MAX_RING + slot]`
/// with **this** stride whatever a channel's actual capacity is. Using `cap1`
/// as the pitch instead would be the natural-looking mistake and would make
/// every channel past the first index into its neighbour's flags.
pub const MAX_RING: u32 = 64;

/// The readiness kernel's entry point.
pub const READINESS_ENTRY: &str = "pie_ptir_stage_readiness";

/// The commit-bump kernel's entry point.
pub const COMMIT_ENTRY: &str = "pie_ptir_commit_bump";

/// Both kernels, in one translation unit.
///
/// One unit rather than two because they are compiled together, cached
/// together, and neither is ever wanted without the other: a pass that can
/// decide it may run and cannot then publish is not a pass.
///
/// `MAX_RING` is interpolated rather than written as a literal, so the Rust
/// constant above is the single definition. A driver whose host arithmetic
/// used 64 and whose kernel used 32 would read the right flags for channel 0
/// and the wrong ones for every channel after it.
fn source() -> String {
    format!(
        r#"
typedef unsigned char pie_u8;
typedef unsigned int  pie_u32;

#define PIE_MAX_RING {MAX_RING}u

// Stage readiness: AND this stage's requirement into the pass commit flag.
//
// `need_full` names the channels whose first op consumes -- their committed
// cell must be full. `need_empty` names the channels whose first op produces
// -- the ring must have room, which is the standard ring-not-full test
// `(tail + 1) % cap1 != head`, reserving one sentinel cell so a capacity-N
// channel holds at most N unconsumed items.
//
// A miss clears `pass_commit`, which does NOT abort the pass: the region
// kernels still run, over the same cells, and the commit bump then declines to
// publish. That is the dummy run, and it is what makes a blocked fire cost the
// same every time instead of branching on the device.
extern "C" __global__ void {READINESS_ENTRY}(
    const pie_u8*  full,
    const pie_u32* head,
    const pie_u32* tail,
    const pie_u32* cap1,
    const pie_u32* need_full_ch,  pie_u32 n_full,
    const pie_u32* need_empty_ch, pie_u32 n_empty,
    pie_u32* pass_commit) {{
  if (threadIdx.x != 0u || blockIdx.x != 0u) return;
  pie_u32 ok = 1u;
  for (pie_u32 i = 0u; i < n_full; ++i) {{
    const pie_u32 c = need_full_ch[i];
    if (!full[c * PIE_MAX_RING + head[c]]) ok = 0u;
  }}
  for (pie_u32 i = 0u; i < n_empty; ++i) {{
    const pie_u32 c = need_empty_ch[i];
    if (((tail[c] + 1u) % cap1[c]) == head[c]) ok = 0u;
  }}
  *pass_commit &= ok;
}}

// End-of-pass predicated commit bump.
//
// Iff `*pass_commit`: publish every put channel's pending cell by setting its
// full bit and advancing `tail`, then consume every taken channel's committed
// cell by clearing its full bit and advancing `head`.
//
// Puts run BEFORE takes, and the order is the contract rather than a
// preference: a channel that is both taken and put in one pass -- a
// loop-carried counter, the shape every decode loop has -- advances both, and
// publishing first is what leaves the ring holding what the pass produced.
extern "C" __global__ void {COMMIT_ENTRY}(
    pie_u8*  full,
    pie_u32* head,
    pie_u32* tail,
    const pie_u32* cap1,
    const pie_u32* taken_ch, pie_u32 n_taken,
    const pie_u32* put_ch,   pie_u32 n_put,
    const pie_u32* pass_commit) {{
  if (threadIdx.x != 0u || blockIdx.x != 0u) return;
  if (!*pass_commit) return;
  for (pie_u32 i = 0u; i < n_put; ++i) {{
    const pie_u32 c = put_ch[i];
    full[c * PIE_MAX_RING + tail[c]] = 1u;
    tail[c] = (tail[c] + 1u) % cap1[c];
  }}
  for (pie_u32 i = 0u; i < n_taken; ++i) {{
    const pie_u32 c = taken_ch[i];
    full[c * PIE_MAX_RING + head[c]] = 0u;
    head[c] = (head[c] + 1u) % cap1[c];
  }}
}}
"#
    )
}

/// The two control kernels, compiled and loaded.
#[derive(Debug, Clone)]
pub struct Control {
    readiness: Arc<Module>,
    commit: Arc<Module>,
}

impl Control {
    /// Compile both kernels for `architecture`, or take them off `disk`.
    ///
    /// Cached under a key of their own rather than a program's: they belong to
    /// no program, they are identical for every program on one device, and a
    /// key derived from a program would recompile them per program.
    ///
    /// # Errors
    ///
    /// [`Failure::Deterministic`] if the source above does not compile, which
    /// can only mean this file and the NVRTC in use disagree about the
    /// language; [`Failure::Retryable`] for anything the machine refused.
    pub fn compile(disk: &Disk, architecture: &str, identity: &str) -> Result<Self, Failure> {
        let source = source();
        let key = disk_key(identity, &source);
        // The pair is ONE cubin with two entry points, so one cache slot holds
        // it and the two modules below are two lookups into one image. The
        // region index is arbitrary and simply must not collide with a real
        // program's -- these share the directory, not the key space.
        let cubin = match disk.load(&key, CONTROL_REGION, READINESS_ENTRY) {
            Some(cubin) => cubin,
            None => {
                let cubin =
                    nvrtc::compile(&source, architecture).map_err(|error| match error.kind {
                        FailureKind::Deterministic => Failure::Deterministic {
                            reason: error.message,
                        },
                        FailureKind::Retryable => Failure::Retryable {
                            reason: error.message,
                        },
                    })?;
                disk.store(&key, CONTROL_REGION, READINESS_ENTRY, &cubin);
                cubin
            }
        };

        let load = |entry: &str| -> Result<Arc<Module>, Failure> {
            Module::load(&cubin, entry)
                .map(Arc::new)
                .map_err(|error| Failure::Retryable {
                    reason: format!("loading control kernel '{entry}': {error}"),
                })
        };
        Ok(Self {
            readiness: load(READINESS_ENTRY)?,
            commit: load(COMMIT_ENTRY)?,
        })
    }

    /// The readiness kernel.
    #[must_use]
    pub fn readiness(&self) -> &Module {
        &self.readiness
    }

    /// The commit-bump kernel.
    #[must_use]
    pub fn commit(&self) -> &Module {
        &self.commit
    }
}

/// The region index the control pair is cached under.
///
/// A real program's regions are numbered from zero and are dense, so a large
/// number cannot collide with one. It is not a magic constant so much as a
/// statement that these two kernels are not a region of anything.
const CONTROL_REGION: u32 = u32::MAX;

#[cfg(test)]
mod tests {
    use super::*;

    /// The stride the host indexes `full` with must be the one the kernel
    /// indexes it with. Interpolation is what guarantees it; this checks the
    /// interpolation actually happened rather than the literal surviving.
    #[test]
    fn the_ring_pitch_in_the_source_is_the_rust_constant() {
        let source = source();
        assert!(
            source.contains(&format!("#define PIE_MAX_RING {MAX_RING}u")),
            "the kernel must be built with the host's own ring pitch"
        );
        assert_eq!(MAX_RING, 64, "the C++ ring pitch is 64 and this is a port");
    }

    /// `%` is the modulo the ring arithmetic needs, and in a Rust `format!`
    /// string it is an ORDINARY CHARACTER. Only braces are special.
    ///
    /// This test was written expecting the printf rule — that `%%` escapes to
    /// `%` — and it caught the mistake on its first run: the doubled percent
    /// went through verbatim and would have reached NVRTC as `a %% b`, which
    /// is not an expression. It is kept, unchanged in intent, because the
    /// reflex it guards against is the one a reader of a template full of
    /// `{{` and `}}` is most likely to have.
    #[test]
    fn the_modulo_operators_are_single_percent_signs() {
        let source = source();
        assert_eq!(
            source.matches("% cap1[c]").count(),
            3,
            "three modulo operations over cap1: the ring-not-full test in \
             readiness, and the two cursor advances in commit"
        );
        assert!(
            !source.contains("%%"),
            "`%` is not escaped in a Rust format string; a doubled one reaches \
             NVRTC verbatim and `a %% b` is not an expression"
        );
    }

    /// The source must name no header and no `std::`, because NVRTC compiles
    /// it with no include path — the same condition the emitted regions are
    /// compiled under. A single `#include` here is a compile failure per
    /// process with a diagnostic three steps from its cause.
    #[test]
    fn the_source_reaches_for_no_header_it_cannot_have() {
        let source = source();
        assert!(!source.contains("#include"), "there is no include path");
        assert!(!source.contains("std::"), "there is no standard library");
    }

    /// Both entry points must be `extern "C"`: NVRTC mangles a C++ name, and
    /// the driver looks each up by the string above. A missing linkage
    /// specifier compiles, loads, and then cannot be found.
    #[test]
    fn both_entry_points_have_c_linkage_so_their_names_survive() {
        let source = source();
        for entry in [READINESS_ENTRY, COMMIT_ENTRY] {
            assert!(
                source.contains(&format!("extern \"C\" __global__ void {entry}(")),
                "{entry} must be declared with C linkage"
            );
        }
    }

    /// Puts before takes. A channel that is both taken and put in one pass is
    /// the shape of every decode loop, and the two orders leave different
    /// values in the ring.
    #[test]
    fn the_commit_publishes_before_it_consumes() {
        let source = source();
        let commit = source
            .split_once(COMMIT_ENTRY)
            .expect("the commit kernel is in the source")
            .1;
        let put_at = commit.find("put_ch[i]").expect("the put loop");
        let take_at = commit.find("taken_ch[i]").expect("the take loop");
        assert!(
            put_at < take_at,
            "publishing must precede consuming; the other order leaves a \
             loop-carried channel holding what the pass consumed rather than \
             what it produced"
        );
    }
}
