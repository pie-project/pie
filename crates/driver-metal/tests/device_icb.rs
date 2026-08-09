//! Can this backend record a fire once and replay it?
//!
//! `.wiki/driver/graph-metal.md` §5② is the step that would remove most of a
//! fire's host cost — measured at **47.5 % of a prefill and 76.4 % of a
//! decode**, because `encode` issues about 5 000 Objective-C messages per
//! fire (424 dispatches, 3 779 address binds) and does it again every step.
//!
//! Before writing that, two questions have to be answered by the device
//! rather than by a reading of the headers, and this file is those two
//! questions:
//!
//! 1. **Does an `MTLIndirectCommandBuffer` execute compute from an MTL4
//!    encoder at all?** This crate is on the MTL4 path — `MTL4CommandBuffer`,
//!    `MTL4ComputeCommandEncoder`, `MTL4ArgumentTable` — and the ICB API is
//!    older than that path. `MTL4ComputeCommandEncoder` declares
//!    `executeCommandsInBuffer:withRange:`, so it should; "should" is what a
//!    smoke test is for.
//!
//! 2. **`inheritBuffers`: whose bindings does a command use?** This crate
//!    binds through an argument table by raw GPU address, while an ICB
//!    command carries its own `setKernelBuffer`. 424 dispatches bind
//!    *different* addresses to the same slots, so the commands must carry
//!    their own — `inheritBuffers = false`. That is the assumption
//!    `graph-metal.md` §5② states, and it is the one this file exists to
//!    confirm or kill.
//!
//! The test is deliberately small: two commands, one buffer each, and an
//! assertion on the numbers that come back. What it is measuring is whether
//! the mechanism exists, not whether it is fast.
//!
//! # Both answers are yes, and two things had to be true first
//!
//! **A pipeline must be compiled `supportIndirectCommandBuffers`.** Setting
//! one that was not on an `MTLIndirectComputeCommand` does not fail — it
//! **faults**, SIGSEGV inside the recording loop, before anything executes.
//! `Compiler` now states it for every pipeline.
//!
//! **The command TYPE must match the dispatch call.**
//! `ConcurrentDispatch` pairs with `concurrentDispatchThreadgroups` and
//! `ConcurrentDispatchThreads` with `concurrentDispatchThreads`. Declaring one
//! and calling the other is silent: the command does nothing and the buffer
//! reads back as it was, which is what this test saw before the descriptor
//! named both.
//!
//! Neither is documented anywhere in this tree, and each cost a debugging
//! round — which is the argument for this file existing at all rather than
//! the ICB work discovering them inside a 424-dispatch fire.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use driver_metal::metal::{Compiler, Context, allocate};
use driver_metal::region::Region as _;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice, MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor,
    MTLIndirectCommandType, MTLIndirectComputeCommand, MTLResidencySet, MTLResourceOptions,
    MTLSize,
};

/// One kernel: add a per-command constant to every lane.
///
/// Two commands over the same shader with DIFFERENT buffers is the shape the
/// question is about — if a command's own bindings are used, the two write to
/// two places; if the encoder's are, they both write to whichever was bound
/// last.
const SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void bump(device uint* out [[buffer(0)]],
                 const device uint* addend [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
    out[gid] = out[gid] + addend[0];
}
";

#[test]
fn a_fire_can_be_recorded_once_and_replayed() {
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let pipeline = compiler
        .compile(&context, SHADER, "bump")
        .expect("the probe shader compiles");

    // Two outputs and two addends: command 0 writes `a` with 7, command 1
    // writes `b` with 9. If the commands do NOT carry their own bindings,
    // both land in one place and the assertion below says so.
    let a = allocate(&context, 16 * 4, "icb probe a").expect("a region");
    let b = allocate(&context, 16 * 4, "icb probe b").expect("a region");
    let seven = allocate(&context, 4, "icb probe 7").expect("a region");
    let nine = allocate(&context, 4, "icb probe 9").expect("a region");
    // SAFETY: freshly allocated, nothing encoded against them.
    unsafe {
        a.zero(0, a.len()).expect("zeroes");
        b.zero(0, b.len()).expect("zeroes");
        seven.write(0, &7u32.to_le_bytes()).expect("writes");
        nine.write(0, &9u32.to_le_bytes()).expect("writes");
    }

    // CONTROL: the same shader through the ordinary path. If this is zeros
    // too, the probe is wrong; if it works and the ICB does not, the ICB is
    // the finding.
    {
        let table = driver_metal::metal::ArgumentTable::new(&context, 2).expect("a table");
        table.bind_address(0, a.gpu_address()).expect("binds");
        table.bind_address(1, seven.gpu_address()).expect("binds");
        let mut stepper = driver_metal::metal::Stepper::new(&context).expect("a stepper");
        stepper
            .run(|encoder| {
                encoder.set_pipeline(&pipeline);
                encoder.set_argument_table(&table);
                encoder.dispatch([16, 1, 1], [16, 1, 1])
            })
            .expect("the control fires");
        assert_eq!(read(&a), vec![7u32; 16], "the control fires, so the probe is sound");
        // SAFETY: retired.
        unsafe { a.zero(0, a.len()).expect("re-zero for the ICB run") };
    }

    let descriptor = MTLIndirectCommandBufferDescriptor::new();
    // BOTH spellings, because the command TYPE has to match the dispatch call
    // the command makes: `ConcurrentDispatch` pairs with
    // `concurrentDispatchThreadgroups`, `ConcurrentDispatchThreads` with
    // `concurrentDispatchThreads`. Declaring one and calling the other does
    // not fail -- the command silently does nothing, which is how this test
    // first read back all zeros.
    descriptor.setCommandTypes(MTLIndirectCommandType(
        MTLIndirectCommandType::ConcurrentDispatch.0
            | MTLIndirectCommandType::ConcurrentDispatchThreads.0,
    ));
    // FALSE, which is the answer this file exists to check. Every command
    // states its own buffers, because the fire's dispatches bind different
    // addresses to the same slots.
    descriptor.setInheritBuffers(false);
    descriptor.setInheritPipelineState(false);
    descriptor.setMaxKernelBufferBindCount(2);

    // SAFETY: the descriptor is fully initialised above.
    let icb = unsafe {
        context
            .device()
            .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &descriptor,
                2,
                MTLResourceOptions::StorageModeShared,
            )
    }
    .expect("the device makes an indirect command buffer for compute");

    // RESIDENT, and it is not a formality: the ICB is a buffer the GPU reads
    // its commands out of, and this context tracks nothing automatically
    // (`HazardTrackingModeUntracked`, one residency set). Without this the
    // execute below faults -- measured, SIGSEGV.
    context.residency().addAllocation(ProtocolObject::from_ref(&*icb));
    context.residency().commit();
    context.residency().requestResidency();

    for (index, (out, addend)) in [(&a, &seven), (&b, &nine)].into_iter().enumerate() {
        // SAFETY: `index` is below the max command count declared above.
        let command = unsafe { icb.indirectComputeCommandAtIndex(index) };
        command.setComputePipelineState(&pipeline);
        // SAFETY: both buffers outlive the execution below, and the indices
        // are below `maxKernelBufferBindCount`.
        unsafe {
            command.setKernelBuffer_offset_atIndex(out.buffer(), 0, 0);
            command.setKernelBuffer_offset_atIndex(addend.buffer(), 0, 1);
        }
        command.concurrentDispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: 16,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 16,
                height: 1,
                depth: 1,
            },
        );
    }

    execute(&context, &icb, 2);

    let (first, second) = (read(&a), read(&b));
    assert_eq!(first, vec![7u32; 16], "command 0 wrote its own buffer");
    assert_eq!(
        second,
        vec![9u32; 16],
        "command 1 wrote ITS own buffer -- if this is all sevens or all zeros, \
         the commands are inheriting the encoder's bindings and \
         `inheritBuffers = false` does not mean what graph-metal.md assumes"
    );

    // REPLAYED, with no re-recording: the same ICB executed again must add
    // again. That is the whole proposition -- one recording, many fires.
    execute(&context, &icb, 2);
    assert_eq!(read(&a), vec![14u32; 16], "a replay re-runs the recording");
    assert_eq!(read(&b), vec![18u32; 16]);
}

/// Run `count` commands of `icb` and wait.
fn execute(
    context: &Context,
    icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
    count: usize,
) {
    let mut stepper = driver_metal::metal::Stepper::new(context).expect("a stepper");
    stepper
        .run(|encoder| {
            encoder.execute_commands(icb, 0..count)
        })
        .expect("the ICB executes");
}

fn read(handle: &driver_metal::metal::Handle) -> Vec<u32> {
    // SAFETY: the fire has retired, so the host owns the bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(handle.contents().as_ptr().cast::<u8>(), 16 * 4)
    };
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A WHOLE FIRE, recorded once and replayed — and what that saves.
///
/// The previous test proves the mechanism on two commands. This one runs it
/// over the real thing: `llama_like`'s Metal text, 424 dispatches, the same
/// walk `encode` does every step.
///
/// The number it prints is the one `.wiki/driver/graph-metal.md` is about.
#[test]
fn a_whole_fire_records_and_replays_faster_than_it_encodes() {
    use driver_metal::metal::{Regions, Stepper, record};
    use driver_metal::model::dispatch::{Geometry, plan};
    use driver_metal::model::encode::{Params, Pipelines, encode};
    use driver_metal::model::executor::{Frame, Slice};
    use driver_metal::model::run::{table, table_width};
    use model::families::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model::families::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Fire, Row, lower};
    use model_compiler::trace::FireClass;

    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(std::path::PathBuf::from(
        std::env::var("PIE_METAL_KERNELS").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels-metal/kernels").to_string()
        }),
    ));

    let plan_text = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Decode,
    );
    let lowered = lower(
        &plan_text,
        &[Row {
            samples: true,
            ..Row::default()
        }],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the text lowers");

    // One big region answers every name, and it is REGISTERED so the
    // recording can turn an address back into a buffer.
    let backing = allocate(&context, 256 << 20, "sentinels").expect("a region");
    let arena = allocate(&context, (lowered.arena_bytes as u64).max(1), "arena").expect("arena");
    let zeros = allocate(&context, 1 << 20, "fire tables").expect("a region");
    // SAFETY: freshly allocated, nothing encoded against it.
    unsafe { zeros.zero(0, zeros.len()).expect("zeroes") };
    let mut store = Everything {
        slice: Slice {
            address: backing.gpu_address(),
            bytes: 256 << 20,
        },
        tables: Slice {
            address: zeros.gpu_address(),
            bytes: zeros.len(),
        },
    };
    let frame = Frame {
        arena: Slice {
            address: arena.gpu_address(),
            bytes: arena.len(),
        },
    };
    let geometry = Geometry {
        q_heads: 16,
        kv_heads: 8,
        head_dim: 128,
        rotary_dims: 128,
        n_experts: 0,
        experts_per_token: 0,
    };
    let dispatches =
        plan(&lowered, table(), frame, geometry, &mut store).expect("the fire plans");
    let params = Params::stage(&context, &dispatches).expect("scalars stage");
    let argtable =
        driver_metal::metal::ArgumentTable::new(&context, table_width(&dispatches))
            .expect("an argument table");
    pipelines
        .ensure(&context, &compiler, &dispatches)
        .expect("every symbol compiles");

    let mut regions = Regions::new();
    regions.add(&backing);
    regions.add(&zeros);
    regions.add(&arena);
    regions.add(params.region());
    // The stand-in for an operand that addresses nothing -- `dispatch::bind`
    // answers unfilled slots with address zero, and a recorded command binds
    // a buffer rather than an address.
    let nothing = allocate(&context, 1 << 16, "nothing").expect("a region");
    regions.set_null(&nothing);

    let recording = record(&context, &pipelines, &params, &regions, &dispatches)
        .expect("the fire records");
    assert_eq!(recording.commands(), dispatches.len());

    let mut stepper = Stepper::new(&context).expect("a stepper");
    // Warm both paths.
    stepper
        .run(|e| encode(e, &argtable, &pipelines, &params, &dispatches))
        .expect("encodes");
    stepper
        .run(|e| e.execute_commands(recording.buffer(), 0..recording.commands()))
        .expect("replays");

    const N: u32 = 10;
    let mut encoded = std::time::Duration::ZERO;
    let mut replayed = std::time::Duration::ZERO;
    for _ in 0..N {
        encoded += stepper
            .run(|e| encode(e, &argtable, &pipelines, &params, &dispatches))
            .expect("encodes")
            .encode;
        replayed += stepper
            .run(|e| e.execute_commands(recording.buffer(), 0..recording.commands()))
            .expect("replays")
            .encode;
    }
    let (encoded, replayed) = (encoded / N, replayed / N);

    // THE SAME NUMBERS, which is the part that matters. A replay that is fast
    // and computes something else is the failure this crate spends most of
    // its tests on -- so encode into a zeroed arena, keep the bytes, replay
    // into a zeroed arena, and compare.
    // The sentinel resolver answers every NAME with `backing`, so that is
    // where this fire's outputs land; the arena holds only what the lowering
    // assigns itself.
    let bytes = 1 << 20;
    let snapshot = |stepper: &mut Stepper, which: &str| -> Vec<u8> {
        // SEEDED, not zeroed: every sentinel weight is the same region, so a
        // zeroed one makes the whole fire compute zeros and the comparison
        // proves nothing. `0x3c` is a small positive bf16.
        let seed = vec![0x3cu8; bytes];
        // SAFETY: nothing is encoded against either region between fires.
        unsafe {
            arena.zero(0, arena.len()).expect("zeroes");
            backing.write(0, &seed).expect("seeds");
        }
        match which {
            "encode" => stepper
                .run(|e| encode(e, &argtable, &pipelines, &params, &dispatches))
                .expect("encodes"),
            _ => stepper
                .run(|e| e.execute_commands(recording.buffer(), 0..recording.commands()))
                .expect("replays"),
        };
        // SAFETY: the fire has retired, so the host owns the bytes.
        unsafe { std::slice::from_raw_parts(backing.contents().as_ptr().cast::<u8>(), bytes) }
            .to_vec()
    };
    let by_encoding = snapshot(&mut stepper, "encode");
    let by_replay = snapshot(&mut stepper, "replay");
    assert_eq!(
        by_encoding, by_replay,
        "the recording computes something the encode does not -- the first \
         {bytes} bytes of the arena differ"
    );
    assert!(
        by_encoding.iter().any(|b| *b != 0),
        "both paths wrote nothing, so the comparison proved nothing"
    );
    println!(
        "{} dispatches: encode={encoded:?}/fire replay={replayed:?}/fire  ({:.0}x)",
        dispatches.len(),
        encoded.as_secs_f64() / replayed.as_secs_f64().max(1e-9)
    );
    assert!(
        replayed < encoded,
        "a replay that is not cheaper than the encode it replaces is the whole \
         proposition failing: encode={encoded:?} replay={replayed:?}"
    );
}

/// Answers every name with one generous region — this test is about host
/// cost, and `model_bind.rs` owns whether the names resolve.
struct Everything {
    slice: driver_metal::model::executor::Slice,
    /// The fire's tables, ZEROED and kept separate from the weights. A page
    /// CSR of seeded bytes walks pages until the GPU is abandoned -- measured,
    /// a 60-second timeout.
    tables: driver_metal::model::executor::Slice,
}

impl driver_metal::model::executor::Resolver for Everything {
    fn weight(&mut self, _: &str) -> Option<driver_metal::model::executor::Slice> {
        Some(self.slice)
    }
    fn named(
        &mut self,
        _: model_compiler::trace::ValueId,
    ) -> Option<driver_metal::model::executor::Slice> {
        Some(self.slice)
    }
    fn kv(&mut self, _: u16, _: bool) -> Option<driver_metal::model::executor::Slice> {
        Some(self.slice)
    }
    fn fire(
        &mut self,
        _: driver_metal::model::executor::FireTable,
    ) -> Option<driver_metal::model::executor::Slice> {
        Some(self.tables)
    }
    fn pool(&mut self, _: driver_metal::model::executor::FireTable) -> Option<u32> {
        Some(128)
    }
}
