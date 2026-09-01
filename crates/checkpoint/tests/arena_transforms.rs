//! What a backing that runs its own transforms is promised, and what it owes.
//!
//! `ArenaBacking::runs_named_kernels` / `run_tile_map` let a device do a
//! `TileMap` on operands that are already in its arena, instead of the host
//! staging them back, computing, and sending them across again. That is a
//! delegation with four moving parts and every one of them is checkable with
//! no GPU in the build:
//!
//! * the executor only OFFERS a transform whose operands it has resolved to
//!   arena spans, and offers nothing when the backing runs no kernels;
//! * a backing that is offered one is offered it because the *plan* named a
//!   kernel for it, so failing to run it fails the load — there is no decline;
//! * the operands it is handed describe the same transform the host path
//!   would have run; and
//! * the arena is released either way.
//!
//! The third is why this is a file rather than a unit test beside the trait. A
//! backing that runs nothing must leave the arena holding exactly what a
//! backing that was never asked leaves — which is what makes the delegation a
//! *route* rather than a second implementation that could disagree.
//!
//! A segmented backing STOOD BESIDE THEM — an arena spread over several
//! allocations, differentially compared against the flat one — and it went
//! with the executor::chunked module it was the only reader of. That module
//! was engine-metal's, moved here so the chunk arithmetic could be checked
//! with no GPU in the build; engine-metal came back at P5a with a baker
//! executor whose staging is its own, and took no loader dependency with it.
//! What was left was a public module read by one test of its own crate.

use std::borrow::Cow;

use checkpoint::file::{File, Metadata, RawTensor};
use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::error::Error;
use checkpoint::executor::Execution;
use checkpoint::executor::arena::{ArenaBacking, TileMapOp};
use checkpoint::executor::sink::MemorySink;
use checkpoint::plan::compile as compile_load_plan;
use checkpoint::plan::passes::tile::CUDA_CAST_FP32_TO_BF16;
use checkpoint::plan::{CUDA_TILE_MAP_MASK, LoadPlan, StorageInstr, StorageTarget, TileMapKind};
use checkpoint::types::{BackendKind, CheckpointFormat, DType, Encoding, FileId, TensorId};

// ── The fixture
//
// **IT WAS A DEQUANT AND A RE-ENCODE**, and §M-3 shut the encode door: no
// device mask carries one, so a contract that stated it no longer compiles
// for a CUDA target and the fixture would have tested nothing. What the file
// is about is unchanged, and it is restated over the one row the CUDA table
// still has — an F32 to BF16 cast — because the question was never which
// transform it is. It is WHERE THE OPERANDS ARE.
//
// So there are two casts, and they land on opposite sides of the line:
//
// * `v` casts the CHECKPOINT. Its bytes are on disk, not in the arena, so no
//   backing can be handed it and `lower_tile_map` names no row for it.
// * `w` casts `d`, which the contract publishes. Published means placed,
//   placed means arena-resident, and F32 to BF16 over a 2-D shape is the row
//   the CUDA table has. It is the offer.

fn raw(
    id: u32,
    name: &str,
    offset: u64,
    span_bytes: u64,
    shape: &[i64],
    dtype: DType,
) -> RawTensor {
    RawTensor {
        id: TensorId(id),
        name: name.to_string(),
        file_id: FileId(0),
        file_offset: offset,
        span_bytes,
        shape: shape.to_vec(),
        encoding: Encoding::Raw(dtype),
    }
}

fn metadata() -> Metadata {
    Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 256,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            raw(0, "w32", 0, 128, &[4, 8], DType::F32),
            raw(1, "v32", 128, 128, &[4, 8], DType::F32),
        ],
    }
}

fn contract() -> ModelContract {
    ModelContract {
        alignment: 256,
        tensors: vec![
            // The cast that reads the checkpoint, and is therefore not
            // delegable however good a row the table has for its dtypes.
            TensorContract::new(
                "v",
                Expr::src("v32").cast(Encoding::Raw(DType::Bf16)),
                vec![4, 8],
                Encoding::Raw(DType::Bf16),
            ),
            // The staged operand. PUBLISHED, not internal, and that is the
            // point of the fixture: a published tensor is placed in the
            // arena, so the cast below reads arena bytes and is a candidate
            // for delegation. An internal intermediate is a host scratch
            // buffer, and a transform reading one can never be offered to a
            // backing that cannot see it.
            TensorContract::new(
                "d",
                Expr::src("w32"),
                vec![4, 8],
                Encoding::Raw(DType::F32),
            ),
            TensorContract::new(
                "w",
                Expr::out("d").cast(Encoding::Raw(DType::Bf16)),
                vec![4, 8],
                Encoding::Raw(DType::Bf16),
            ),
        ],
        groups: Vec::new(),
    }
}

fn plan() -> LoadPlan {
    // A CUDA target, because the default one is `BackendKind::Unknown` and
    // names no kernel at all — and a named row is half of what this file is
    // about.
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: CUDA_TILE_MAP_MASK,
        ..StorageTarget::default()
    };
    compile_load_plan(&metadata(), &contract(), target).expect("the fixture must compile")
}

fn kinds(plan: &LoadPlan) -> Vec<TileMapKind> {
    plan.instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap { kind, .. } => Some(*kind),
            _ => None,
        })
        .collect()
}

// ── A backing that records what it is offered and runs none of it.

/// One offer, flattened to the facts it carries.
///
/// Recorded rather than the borrowed op, because the op borrows the
/// executor's plan and outlives nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Offer {
    kernel: String,
    has_dst_scales: bool,
    has_factors: bool,
    shape: Option<(u32, u32)>,
}

struct Recording {
    bytes: Vec<u8>,
    runs_kernels: bool,
    offers: Vec<Offer>,
    /// How many times the executor said the arena was finished.
    finished: usize,
    /// Whether the load finished, and why not. The delegation tests below turn
    /// on this: a backing that is offered an op it cannot run must fail the
    /// load rather than let the host quietly produce the bytes.
    outcome: Result<(), String>,
}

impl Recording {
    fn new(len: usize, runs_kernels: bool) -> Self {
        Self {
            bytes: vec![0; len],
            runs_kernels,
            offers: Vec::new(),
            finished: 0,
            outcome: Ok(()),
        }
    }
}

fn oob(what: &str) -> Error {
    Error::Contract(format!("recording arena {what} is out of bounds"))
}

impl ArenaBacking for Recording {
    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>, Error> {
        let end = offset.checked_add(len).ok_or_else(|| oob("read"))?;
        self.bytes
            .get(offset..end)
            .map(Cow::Borrowed)
            .ok_or_else(|| oob("read"))
    }

    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| oob("write"))?;
        self.bytes
            .get_mut(offset..end)
            .ok_or_else(|| oob("write"))?
            .copy_from_slice(bytes);
        Ok(())
    }

    fn fill(&mut self, offset: usize, len: usize, byte: u8) -> Result<(), Error> {
        let end = offset.checked_add(len).ok_or_else(|| oob("fill"))?;
        self.bytes
            .get_mut(offset..end)
            .ok_or_else(|| oob("fill"))?
            .fill(byte);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), Error> {
        self.finished += 1;
        Ok(())
    }

    fn runs_named_kernels(&self) -> bool {
        self.runs_kernels
    }

    fn run_tile_map(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        self.offers.push(Offer {
            kernel: op.kernel.to_string(),
            has_dst_scales: op.dst_scales.is_some(),
            has_factors: op.factors.is_some(),
            shape: op.shape,
        });
        // Records and refuses. There is no decline any more -- an op that gets
        // here is one the compiler named a row for, so a backing that cannot
        // run it is a backing that disagrees with the plan, and the load must
        // say so instead of finishing on the host at a fraction of the speed.
        Err(Error::Contract(format!(
            "the recording backing launches nothing, including `{}`",
            op.kernel
        )))
    }
}

/// The checkpoint the fixture names, on disk.
///
/// THE REASON THIS FUNCTION EXISTS. It did not, and the whole file was
/// vacuous: the execution died at its first read, so `offers` was empty in
/// every test and each one passed on nothing. `all()` over an empty vector is
/// true, `is_empty()` on an empty vector is true, an early `return` when the
/// offer is missing returns, and comparing the bytes of two runs that both
/// failed at the same instruction compares two identical prefixes. Five tests
/// asserting a delegation that never happened.
///
/// The bytes are raw at the offsets `metadata()` declares — the executor
/// reads `file_offset..file_offset + span_bytes` from the plan and does not
/// re-parse a safetensors header — so the payload is anything, as long as it
/// is there. `w32` is 128 bytes of F32 at 0 and `v32` is 128 more at 128.
/// The pattern is written rather than zeroed so `the_fixture_actually_runs`
/// can tell a completed load from an untouched arena.
fn checkpoint() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "pie_arena_transforms_{}_{}",
        std::process::id(),
        std::thread::current()
            .name()
            .unwrap_or("test")
            .replace("::", "_")
    ));
    std::fs::create_dir_all(&dir).expect("fixture dir");
    let mut bytes = vec![0u8; 256];
    for (i, byte) in bytes.iter_mut().enumerate() {
        // A small positive F32 in every lane: the exponent byte is the third
        // of each little-endian quad, so a pattern in the low bytes stays
        // finite whatever the mantissa says.
        *byte = if i % 4 == 3 { 0x3E } else { (i as u8).wrapping_mul(17) };
    }
    std::fs::write(dir.join("model.safetensors"), bytes).expect("fixture file");
    dir
}

/// Execute the fixture against a backing that does or does not run kernels.
fn run(runs_kernels: bool) -> Recording {
    let plan = plan();
    // The WHOLE allocation, staging included: the fixture's file-reading cast
    // stages its source, and the plan's offsets are all measured from one
    // base — which is the fact `the_executor_allocates_an_arena_its_own_plan_
    // fits_in` below is about.
    let len = usize::try_from(plan.memory.arena_bytes()).expect("fits");
    let mut arena = Recording::new(len.max(1), runs_kernels);
    let mut sink = MemorySink::default();
    let dir = checkpoint();
    arena.outcome = Execution::new(&plan, &dir)
        .arena(&mut arena)
        .sink(&mut sink)
        .run()
        .map(|_| ())
        .map_err(|err| err.to_string());
    std::fs::remove_dir_all(&dir).ok();
    arena
}

#[test]
fn the_fixture_produces_the_two_transforms_the_line_runs_between() {
    assert_eq!(
        kinds(&plan()),
        vec![TileMapKind::Cast, TileMapKind::Cast],
        "the delegation tests are only meaningful over these two"
    );
}

/// The fixture LOADS. Everything below reads from a completed execution.
///
/// First, because it is the assumption the rest of the file used to make and
/// did not check: the checkpoint was never written, every run died at its
/// first read, and five tests passed over an empty `offers`. A vacuous test
/// costs more than no test, because it also reports that the property holds.
#[test]
fn the_fixture_actually_runs() {
    let arena = run(false);
    assert_eq!(
        arena.outcome,
        Ok(()),
        "the host path must complete this fixture with no backing help; \
         every other test here reads the state it leaves"
    );
    assert!(
        arena.bytes.iter().any(|&b| b != 0),
        "a completed load writes the arena"
    );
}

#[test]
fn a_backing_that_runs_no_kernels_is_offered_nothing() {
    let arena = run(false);
    assert!(
        arena.offers.is_empty(),
        "host mode is the default, and the default must not reach the \
         backing: {:?}",
        arena.offers
    );
    assert_eq!(arena.outcome, Ok(()), "and the load still finishes");
}

/// Only an instruction the PLAN named a kernel for is offered.
///
/// This replaces `a_claim_is_per_kind_and_only_that_kind_is_offered`, which
/// asked whether the executor respected a per-kind capability mask. There is
/// no such mask: the compiler names a row per instruction with the tensor's
/// name in hand, and that answer is strictly better than a kind. So the
/// property is now about the plan rather than about the claim.
#[test]
fn only_an_instruction_the_plan_named_a_row_for_is_offered() {
    let arena = run(true);
    assert!(
        !arena.offers.is_empty(),
        "the fixture's staged cast is a row `cuda_kernel` names; if this is \
         empty the delegation stopped happening and the tests below are vacuous"
    );
    for offer in &arena.offers {
        assert!(
            named_kernels(&plan()).contains(&offer.kernel),
            "the backing was offered `{}`, which the plan names nowhere",
            offer.kernel
        );
    }
}

/// The one delegable instruction is the cast off the ARENA, and the cast off
/// the FILE is not.
///
/// Both halves matter. The first is offered because every operand is a span
/// of the arena and the compiler named a row for it. The second is not
/// offered because its input is the CHECKPOINT — bytes on disk, which a
/// backing handed nothing but offsets cannot read — and `lower_tile_map`
/// agrees, naming no row for it.
///
/// The exclusion is a property of WHERE THE OPERANDS ARE and not of which
/// transform it is, which is why both instructions here are the same kind
/// with the same dtypes and only one of them is an offer.
#[test]
fn the_offer_is_the_transform_whose_operands_are_all_in_the_arena() {
    let arena = run(true);
    let kernels: Vec<&str> = arena.offers.iter().map(|o| o.kernel.as_str()).collect();
    assert_eq!(
        kernels,
        vec![CUDA_CAST_FP32_TO_BF16],
        "exactly the cast that reads the published `d` and writes a placed \
         tensor"
    );
}

/// **AN `Encode` OFFER TEST STOOD HERE** — that a quantizing row carried the
/// second destination it writes, because a backing handed only the payload
/// span would write half the result and leave the scales zero. §M-3 shut the
/// door and there is no quantizing row left to offer, so what survives of it
/// is the half that is still checkable: an offer carries its shape, and it
/// carries no operand it does not read.
#[test]
fn an_offer_carries_its_shape_and_nothing_it_does_not_read() {
    let arena = run(true);
    let cast = arena
        .offers
        .iter()
        .find(|o| o.kernel == CUDA_CAST_FP32_TO_BF16)
        .expect("the fixture's staged cast is delegable and must be offered");
    assert!(
        !cast.has_dst_scales,
        "a cast publishes one tensor; a second destination is an operand it \
         would never write"
    );
    assert!(
        !cast.has_factors,
        "and it READS no factors; `factors` is the blocked-Scale operand, and \
         a row that reads what it should write is the confusion the shrunken \
         `TileMapOp` exists to prevent"
    );
    assert_eq!(
        cast.shape,
        Some((4, 8)),
        "the row takes a 2-D extent, read from the tensor's declaration and \
         not recomputed from the extent's dims"
    );
}

/// A backing that cannot run what it was offered FAILS the load.
///
/// This replaces `declining_leaves_the_host_to_produce_the_same_bytes`, and
/// the replacement is the point of the change rather than a casualty of it.
/// The old property was that a backing could hand an op back and the host
/// would quietly produce the bytes — correct, and unobservable: the load
/// finished, the bytes were right, and a transform the device was supposed to
/// run had gone to the host at a fraction of the speed. Nothing anywhere
/// reported it.
///
/// An op reaches a backing only because the compiler named a row for it. A
/// backing that then cannot run it is a backing that disagrees with the plan
/// it is executing, and that is drift between two halves of one tree.
#[test]
fn a_backing_that_cannot_run_its_offer_fails_the_load() {
    let refusing = run(true);
    assert!(
        !refusing.offers.is_empty(),
        "nothing was offered, so this test is about nothing"
    );
    let err = refusing
        .outcome
        .as_ref()
        .expect_err("a refused kernel must not be silently rerun on the host");
    assert!(
        err.contains("launches nothing"),
        "the backing's own refusal must reach the caller, not be replaced by \
         a later failure: {err}"
    );
}

/// The host still produces the whole load when no backing helps.
///
/// The property the whole design rests on: a backing is a ROUTE, not a second
/// implementation. `run(false)` is the reference — every transform on the host
/// path — and it must be a complete, correct load rather than the thing that
/// happens when delegation is unavailable.
#[test]
fn the_host_path_alone_completes_the_load() {
    let host_only = run(false);
    assert_eq!(host_only.outcome, Ok(()));
    let again = run(false);
    assert_eq!(
        host_only.bytes, again.bytes,
        "and it is deterministic, which is what makes it an oracle"
    );
}

/// Every kernel symbol the plan names, for the fixture's target.
fn named_kernels(plan: &LoadPlan) -> Vec<String> {
    plan.instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap { transform, .. } => transform.kernel.clone(),
            _ => None,
        })
        .collect()
}

// ── The arena is released either way

/// A load that completes says so, once.
///
/// `.wiki/fix/weight-loader.md` §8.1: release belongs to the seam, not to the
/// caller. Before it did, engine-cuda's own staging (deleted with the legacy load contract at R3) called
/// `arena.finish()?` itself after `run()?` returned — which is the same
/// sentence with a `?` in the middle of it, and the `?` is the bug. Every
/// caller had to remember, and on the path where it mattered most none of
/// them could: `run()` returning `Err` skipped the line.
#[test]
fn a_completed_load_finishes_the_arena_exactly_once() {
    let recording = run(false);
    assert_eq!(recording.outcome, Ok(()));
    assert_eq!(
        recording.finished, 1,
        "the executor owes the backing exactly one `finish` on the path where \
         the bytes are all there"
    );
}

/// A load that fails does not.
///
/// This is the half that reads backwards, so it is worth being explicit about
/// what it means. `finish` is "the arena is complete, publish it" — a failed
/// load has nothing to publish, and calling it would hand the engine a
/// half-written arena that reported success. What must happen instead is
/// RELEASE, and release is `Drop`, which is why §8.1 added one to `CudaArena`
/// rather than making the executor call `finish` on both paths.
///
/// The device half of that — that the freed allocation was not still in use —
/// is what §11 wants `compute-sanitizer` for, and it is the one check in this
/// file that needs a GPU. What is checkable here is the seam's side of the
/// contract: the executor does not claim completion for a load that failed.
#[test]
fn a_failed_load_does_not_finish_the_arena() {
    let refused = run(true);
    assert!(refused.outcome.is_err(), "the fixture's backing refuses");
    assert_eq!(
        refused.finished, 0,
        "a load that stopped part-way has no complete arena to hand over; \
         releasing it is `Drop`'s job, not `finish`'s"
    );
}

/// A plan whose arena is NOT all persistent.
///
/// A tensor cast straight off the FILE has to stage the source first, and
/// that is the region `persistent_bytes` and `arena_bytes()` disagree about.
/// One tensor rather than the fixture's three, so the two numbers differ by a
/// span this test can name.
fn scratch_plan() -> LoadPlan {
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "v",
            Expr::src("v32").cast(Encoding::Raw(DType::Bf16)),
            vec![4, 8],
            Encoding::Raw(DType::Bf16),
        )],
        groups: Vec::new(),
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: CUDA_TILE_MAP_MASK,
        ..StorageTarget::default()
    };
    compile_load_plan(&metadata(), &contract, target).expect("the fixture must compile")
}

/// The executor's own arena is big enough for the plan it is executing.
///
/// `Execution::run` with no `.arena(..)` allocates the backing itself, and
/// allocated `persistent_bytes` while `walk::run` requires `arena_bytes()` --
/// persistent PLUS staging, which the plan's offsets are all measured from
/// the same base as. So the default, no-argument way to execute a plan
/// refused any plan that staged anything, before reading a byte:
///
/// ```text
/// Contract("arena is 4352 bytes and the plan needs 20736")
/// ```
///
/// 206 lib tests and this file did not catch it because every one of them
/// either passes its own arena or executes a plan with no staging region.
/// It surfaced from `engine-cuda`'s `gpu_load_transforms`, two crates away,
/// where it read as a CUDA failure.
#[test]
fn the_executor_allocates_an_arena_its_own_plan_fits_in() {
    let plan = scratch_plan();

    // Without this the test is vacuous: a plan with no staging region cannot
    // tell the two sizes apart, and would pass against the bug.
    assert!(
        plan.memory.scratch_bytes > 0,
        "the fixture stages nothing, so it does not distinguish \
         persistent_bytes from arena_bytes() and proves nothing"
    );
    assert_eq!(
        plan.memory.arena_bytes(),
        plan.memory.persistent_bytes + plan.memory.scratch_bytes
    );

    let dir = checkpoint();
    let storage = Execution::new(&plan, &dir).run();
    std::fs::remove_dir_all(&dir).ok();
    let storage = storage.expect("the executor's own arena must fit its own plan");

    assert_eq!(
        storage.arena.len() as u64,
        plan.memory.arena_bytes(),
        "the returned arena is the whole allocation, staging included"
    );
}
