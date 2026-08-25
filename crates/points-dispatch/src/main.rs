//! **The dispatch generator, and the gate that says the files are current.**
//!
//! Each shader plane ships a `src/points_dispatch.rs` that is a FUNCTION of
//! two things it does not contain: the floor's point tables
//! (`kernels::points::*_POINTS`, emitted by `#[points]`) and that plane's
//! claim tables (`*_CLAIMS`, emitted by `#[claims]`). A generated file whose
//! generator is gone is just a file, and for one wave of work that is what
//! these three were.
//!
//! `--check` (the default) runs the four assertions the deleted test ran, on
//! every plane, and reports all of them rather than stopping at the first.
//! `--write` regenerates the files. Same code both ways, which is the only
//! arrangement in which "current" means anything.
//!
//! # `kernels-vulkan` IS HERE NOW, AND THE SENTENCE IT REPLACES SAID WHY
//!
//! This header used to read: *"`kernels-vulkan` is absent on purpose: it has
//! no `points_dispatch.rs`. Its claims are reached another way, and whether
//! that way is honest is a separate question this tool does not answer."*
//!
//! The other way was NO way. `kernels-vulkan`'s `#[claims]` blocks emit
//! their `*_CLAIMS` tables like every other plane's, and nothing read them:
//! `model_ir::kernels::point_claims` had no Vulkan arm, so `sweep::resolve`
//! could not bind a lane against them and no executor could route a statement
//! to one. Both halves of that are closed now — the plane has a `Backend` row
//! and this tool writes its dispatch — which is the same pair `Wgpu` landed
//! at P5b and `Metal` at P5a.
//!
//! [`vulkan`] states six of its fourteen families as an inline empty slice
//! rather than as a `*_CLAIMS` const, and that is the one place this tool
//! spells a claim table instead of naming one. It is not a preference: a
//! `#[claims]` block is what MAKES the const, and `kernels-vulkan` writes no
//! impl block at all for those six, so there is no const to name. See that
//! function's own note for what the difference costs.

// A COMMAND-LINE TOOL REPORTS ON THE TERMINAL, which is the one place the
// workspace's `print_stdout`/`print_stderr` lints have nothing to say about.
// Everything printed here is the answer, not a trace left behind.
#![allow(clippy::print_stderr, clippy::print_stdout)]

mod generator;

use generator::{Plane, Surface};
use kernels::points::{
    ATTENTION_POINTS, DIST_POINTS, GATE_POINTS, GEMM_POINTS, HC_POINTS, INDEX_POINTS,
    LAYOUT_POINTS, MLA_POINTS, MLP_POINTS, MOE_POINTS, NORM_POINTS, POOL_POINTS, ROPE_POINTS,
    SSM_POINTS,
};

/// `crates/`, which every plane's manifest directory hangs off.
fn crates() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .to_path_buf()
}

/// Where this tool names itself in the header it writes.
const SELF: &str = "crates/points-dispatch";

fn cuda() -> (Plane, Vec<Surface>) {
    let plane = Plane {
        krate: "kernels-cuda",
        generator: SELF,
        dispatch: crates().join("kernels-cuda/src/points_dispatch.rs"),
        prelude: "use crate::jit::Ctx;\nuse crate::jit::abi::bf16;",
        axes: &[("ScalarKind::Bf16", "bf16"), ("ScalarKind::F32", "f32")],
        canon: kernels_cuda::CANON,
    };
    let surfaces = vec![
        Surface::Family {
            trait_name: "Norm",
            points: NORM_POINTS,
            claims: kernels_cuda::norm::NORM_CLAIMS,
        },
        Surface::Family {
            trait_name: "Mlp",
            points: MLP_POINTS,
            claims: kernels_cuda::mlp::MLP_CLAIMS,
        },
        Surface::Family {
            trait_name: "Gemm",
            points: GEMM_POINTS,
            claims: kernels_cuda::gemm::GEMM_CLAIMS,
        },
        Surface::Family {
            trait_name: "Dist",
            points: DIST_POINTS,
            claims: kernels_cuda::dist::DIST_CLAIMS,
        },
        Surface::Family {
            trait_name: "Rope",
            points: ROPE_POINTS,
            claims: kernels_cuda::rope::ROPE_CLAIMS,
        },
        Surface::Family {
            trait_name: "Moe",
            points: MOE_POINTS,
            claims: kernels_cuda::moe::MOE_CLAIMS,
        },
        Surface::Family {
            trait_name: "Gate",
            points: GATE_POINTS,
            claims: kernels_cuda::mlp::GATE_CLAIMS,
        },
        Surface::Family {
            trait_name: "Layout",
            points: LAYOUT_POINTS,
            claims: kernels_cuda::layout::LAYOUT_CLAIMS,
        },
        Surface::Family {
            trait_name: "Ssm",
            points: SSM_POINTS,
            claims: kernels_cuda::ssm::SSM_CLAIMS,
        },
        Surface::Family {
            trait_name: "Attention",
            points: ATTENTION_POINTS,
            claims: kernels_cuda::attn::ATTENTION_CLAIMS,
        },
        Surface::Family {
            trait_name: "Mla",
            points: MLA_POINTS,
            claims: kernels_cuda::attn::MLA_CLAIMS,
        },
        Surface::Family {
            trait_name: "Index",
            points: INDEX_POINTS,
            claims: kernels_cuda::attn::INDEX_CLAIMS,
        },
        Surface::Family {
            trait_name: "Pool",
            points: POOL_POINTS,
            claims: kernels_cuda::attn::POOL_CLAIMS,
        },
        Surface::Family {
            trait_name: "Hc",
            points: HC_POINTS,
            claims: kernels_cuda::norm::HC_CLAIMS,
        },
        Surface::Tier2 {
            points: kernels_cuda::attn::TIER2_POINTS,
        },
    ];
    (plane, surfaces)
}

fn metal() -> (Plane, Vec<Surface>) {
    let plane = Plane {
        krate: "kernels-metal",
        generator: SELF,
        dispatch: crates().join("kernels-metal/src/points_dispatch.rs"),
        prelude: "use crate::points::bfloat;\nuse crate::plane::Ctx;",
        axes: &[("ScalarKind::Bf16", "bfloat"), ("ScalarKind::F32", "f32")],
        canon: kernels_metal::CANON,
    };
    let surfaces = vec![
        Surface::Family {
            trait_name: "Norm",
            points: NORM_POINTS,
            claims: kernels_metal::norm::NORM_CLAIMS,
        },
        Surface::Family {
            trait_name: "Mlp",
            points: MLP_POINTS,
            claims: kernels_metal::mlp::MLP_CLAIMS,
        },
        Surface::Family {
            trait_name: "Gemm",
            points: GEMM_POINTS,
            claims: kernels_metal::layout::GEMM_CLAIMS,
        },
        Surface::Family {
            trait_name: "Dist",
            points: DIST_POINTS,
            claims: kernels_metal::dist::DIST_CLAIMS,
        },
        Surface::Family {
            trait_name: "Rope",
            points: ROPE_POINTS,
            claims: kernels_metal::rope::ROPE_CLAIMS,
        },
        Surface::Family {
            trait_name: "Moe",
            points: MOE_POINTS,
            claims: kernels_metal::moe::MOE_CLAIMS,
        },
        Surface::Family {
            trait_name: "Gate",
            points: GATE_POINTS,
            claims: kernels_metal::attn::GATE_CLAIMS,
        },
        Surface::Family {
            trait_name: "Layout",
            points: LAYOUT_POINTS,
            claims: kernels_metal::layout::LAYOUT_CLAIMS,
        },
        Surface::Family {
            trait_name: "Ssm",
            points: SSM_POINTS,
            claims: kernels_metal::ssm::SSM_CLAIMS,
        },
        Surface::Family {
            trait_name: "Attention",
            points: ATTENTION_POINTS,
            claims: kernels_metal::attn::ATTENTION_CLAIMS,
        },
        Surface::Family {
            trait_name: "Mla",
            points: MLA_POINTS,
            claims: kernels_metal::attn::MLA_CLAIMS,
        },
        Surface::Family {
            trait_name: "Index",
            points: INDEX_POINTS,
            claims: kernels_metal::attn::INDEX_CLAIMS,
        },
        Surface::Family {
            trait_name: "Pool",
            points: POOL_POINTS,
            claims: kernels_metal::attn::POOL_CLAIMS,
        },
        Surface::Family {
            trait_name: "Hc",
            points: HC_POINTS,
            claims: kernels_metal::norm::HC_CLAIMS,
        },
    ];
    (plane, surfaces)
}

fn wgpu() -> (Plane, Vec<Surface>) {
    let plane = Plane {
        krate: "kernels-wgpu",
        generator: SELF,
        dispatch: crates().join("kernels-wgpu/src/points_dispatch.rs"),
        prelude: "use crate::points::bf16;\nuse crate::plane::Ctx;",
        axes: &[("ScalarKind::Bf16", "bf16"), ("ScalarKind::F32", "f32")],
        canon: &[],
    };
    let surfaces = vec![
        Surface::Family {
            trait_name: "Norm",
            points: NORM_POINTS,
            claims: kernels_wgpu::norm::NORM_CLAIMS,
        },
        Surface::Family {
            trait_name: "Mlp",
            points: MLP_POINTS,
            claims: kernels_wgpu::mlp::MLP_CLAIMS,
        },
        Surface::Family {
            trait_name: "Gemm",
            points: GEMM_POINTS,
            claims: kernels_wgpu::quant::GEMM_CLAIMS,
        },
        Surface::Family {
            trait_name: "Dist",
            points: DIST_POINTS,
            claims: kernels_wgpu::points::DIST_CLAIMS,
        },
        Surface::Family {
            trait_name: "Rope",
            points: ROPE_POINTS,
            claims: kernels_wgpu::rope::ROPE_CLAIMS,
        },
        Surface::Family {
            trait_name: "Moe",
            points: MOE_POINTS,
            claims: kernels_wgpu::moe::MOE_CLAIMS,
        },
        Surface::Family {
            trait_name: "Gate",
            points: GATE_POINTS,
            claims: kernels_wgpu::attn::GATE_CLAIMS,
        },
        Surface::Family {
            trait_name: "Layout",
            points: LAYOUT_POINTS,
            claims: kernels_wgpu::layout::LAYOUT_CLAIMS,
        },
        Surface::Family {
            trait_name: "Ssm",
            points: SSM_POINTS,
            claims: kernels_wgpu::ssm::SSM_CLAIMS,
        },
        Surface::Family {
            trait_name: "Attention",
            points: ATTENTION_POINTS,
            claims: kernels_wgpu::attn::ATTENTION_CLAIMS,
        },
        Surface::Family {
            trait_name: "Mla",
            points: MLA_POINTS,
            claims: kernels_wgpu::points::MLA_CLAIMS,
        },
        Surface::Family {
            trait_name: "Index",
            points: INDEX_POINTS,
            claims: kernels_wgpu::points::INDEX_CLAIMS,
        },
        Surface::Family {
            trait_name: "Pool",
            points: POOL_POINTS,
            claims: kernels_wgpu::points::POOL_CLAIMS,
        },
        Surface::Family {
            trait_name: "Hc",
            points: HC_POINTS,
            claims: kernels_wgpu::points::HC_CLAIMS,
        },
    ];
    (plane, surfaces)
}

/// The Vulkan plane: eight `#[claims]` blocks, and six families with no
/// block at all.
///
/// # THE SIX EMPTY SLICES ARE NOT `*_CLAIMS` CONSTS, AND THEY CANNOT BE
///
/// Every other plane's row here names a table its own crate emits, and the
/// generator's own doc says why that matters: *"a generator that named one of
/// them could only ever write for one plane"*. Seven rows below break the
/// pattern and spell `&[]` inline, because `#[claims]` is what MAKES a
/// `*_CLAIMS` const and `kernels-vulkan` states no impl block for `Dist`,
/// `Mla`, `Index`, `Pool` or `Hc`. There is no const to name.
///
/// `Gemm` AND `Ssm` LEFT THAT LIST in the same hour. `Ssm` names
/// `kernels_vulkan::ssm::SSM_CLAIMS`, five of the family's seven points --
/// the `kda` pair is not in the impl block, so it is generated as no arm and
/// refuses `unclaimed`, which is what the paragraph below says a hole costs
/// and is the honest reading here.
///
/// The other two shader planes DO write those blocks — empty, so the const is
/// an empty slice — and `model-ir`'s `METAL_CLAIMS` argues for exactly that:
/// *"a family a plane implements and claims nothing of is a measured backlog
/// while a family a plane does not implement at all is a hole in the table
/// where a measurement should be."* These seven are holes, and the rows below
/// say so rather than dropping the families, because
/// [`generator::every_family_the_floor_declares_has_a_surface`] is the check
/// that a family reaching no `Surface` is generated *as if it did not exist*
/// — and being generated away is strictly worse than being generated empty.
///
/// WHAT THE DEVIATION COSTS, precisely, AND WHAT IT ALREADY COST: this file
/// does not notice a family growing a block. `Gemm` grew one — the row below
/// names `kernels_vulkan::gemm::GEMM_CLAIMS` now — and nothing here failed
/// when it did: the arms would have been generated as no arms, and
/// `every_claim_has_an_arm` reads the same empty slice, so it would not have
/// noticed either. The row had to be repointed BY HAND, which is the cost
/// stated. Six empty `#[claims]` blocks in `kernels-vulkan` would close it for
/// the six that are left.
fn vulkan() -> (Plane, Vec<Surface>) {
    let plane = Plane {
        krate: "kernels-vulkan",
        generator: SELF,
        dispatch: crates().join("kernels-vulkan/src/points_dispatch.rs"),
        prelude: "use crate::points::bf16;\nuse crate::plane::Ctx;",
        axes: &[("ScalarKind::Bf16", "bf16"), ("ScalarKind::F32", "f32")],
        // NO `CANON` CONST ON THIS PLANE, exactly as `kernels-wgpu` states
        // none: the routine layer that carried canon rows is folded, and the
        // `#[claims]` blocks that replaced it answer by point.
        canon: &[],
    };
    let surfaces = vec![
        Surface::Family {
            trait_name: "Norm",
            points: NORM_POINTS,
            claims: kernels_vulkan::norm::NORM_CLAIMS,
        },
        Surface::Family {
            trait_name: "Mlp",
            points: MLP_POINTS,
            claims: kernels_vulkan::mlp::MLP_CLAIMS,
        },
        // NO LONGER ONE OF THE SEVEN. `kernels-vulkan` grew a `#[claims] impl
        // kernels::points::Gemm` — the day the doc above named — so there is a
        // const to point at and this row names it like every other plane's.
        // Six holes left, and each is still an inline `&[]`.
        Surface::Family {
            trait_name: "Gemm",
            points: GEMM_POINTS,
            claims: kernels_vulkan::gemm::GEMM_CLAIMS,
        },
        Surface::Family {
            trait_name: "Dist",
            points: DIST_POINTS,
            claims: &[],
        },
        Surface::Family {
            trait_name: "Rope",
            points: ROPE_POINTS,
            claims: kernels_vulkan::rope::ROPE_CLAIMS,
        },
        Surface::Family {
            trait_name: "Moe",
            points: MOE_POINTS,
            claims: kernels_vulkan::moe::MOE_CLAIMS,
        },
        // `GATE_CLAIMS` reads from `attn` for metal's reason: the impl lives
        // beside the one shader it fires, and on this plane that shader is
        // filed with the attention it gates.
        Surface::Family {
            trait_name: "Gate",
            points: GATE_POINTS,
            claims: kernels_vulkan::attn::GATE_CLAIMS,
        },
        Surface::Family {
            trait_name: "Layout",
            points: LAYOUT_POINTS,
            claims: kernels_vulkan::layout::LAYOUT_CLAIMS,
        },
        Surface::Family {
            trait_name: "Ssm",
            points: SSM_POINTS,
            claims: kernels_vulkan::ssm::SSM_CLAIMS,
        },
        Surface::Family {
            trait_name: "Attention",
            points: ATTENTION_POINTS,
            claims: kernels_vulkan::attn::ATTENTION_CLAIMS,
        },
        Surface::Family {
            trait_name: "Mla",
            points: MLA_POINTS,
            claims: &[],
        },
        Surface::Family {
            trait_name: "Index",
            points: INDEX_POINTS,
            claims: &[],
        },
        Surface::Family {
            trait_name: "Pool",
            points: POOL_POINTS,
            claims: &[],
        },
        Surface::Family {
            trait_name: "Hc",
            points: HC_POINTS,
            claims: &[],
        },
    ];
    (plane, surfaces)
}

/// Run one assertion and turn its panic into a line, so that a `--check`
/// reports every plane rather than the first one that is stale.
fn checked(what: &str, run: impl FnOnce() + std::panic::UnwindSafe) -> Option<String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(run);
    std::panic::set_hook(hook);
    out.err().map(|e| {
        let said = e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "a panic with no message".to_string());
        format!("{what}:\n{said}")
    })
}

fn main() -> std::process::ExitCode {
    let write = std::env::args().any(|a| a == "--write");
    let planes = [cuda(), metal(), wgpu(), vulkan()];

    if write {
        for (plane, surfaces) in &planes {
            let text = generator::generate(plane, surfaces);
            std::fs::write(&plane.dispatch, &text).expect("the dispatch file is writable");
            println!("{}: wrote {}", plane.krate, plane.dispatch.display());
        }
        println!("\npoints-dispatch: four planes written. `cargo fmt` is not run for you.");
        return std::process::ExitCode::SUCCESS;
    }

    let mut stale = Vec::new();
    for (plane, surfaces) in &planes {
        let mut said = Vec::new();
        said.extend(checked("points_dispatch_is_current", || {
            generator::points_dispatch_is_current(plane, surfaces);
        }));
        said.extend(checked("every_claim_has_an_arm", || {
            generator::every_claim_has_an_arm(plane, surfaces);
        }));
        said.extend(checked("every_canon_row_is_an_unclaimed_point", || {
            generator::every_canon_row_is_an_unclaimed_point(plane, surfaces);
        }));
        said.extend(checked(
            "every_family_the_floor_declares_has_a_surface",
            || {
                generator::every_family_the_floor_declares_has_a_surface(surfaces);
            },
        ));

        if said.is_empty() {
            println!("{}: current", plane.krate);
        } else {
            for s in &said {
                eprintln!("\n=== {} — {s}", plane.krate);
            }
            stale.push(plane.krate);
        }
    }

    if stale.is_empty() {
        println!("\npoints-dispatch: four planes current against their claim tables.");
        std::process::ExitCode::SUCCESS
    } else {
        eprintln!(
            "\npoints-dispatch: {stale:?} disagree with their claim tables. \
             Run `cargo run -p points-dispatch -- --write && cargo fmt` and \
             read the diff before you keep it."
        );
        std::process::ExitCode::FAILURE
    }
}
