//! §6.2 PTIR end-to-end smoke inferlet — the guest half of the greedy-argmax
//! wire test (pairs with the `cuda_ptir_e2e` harness).
//!
//! Exercises the full PTIR program path with **direct WIT bindings** (no
//! `inferlet/` convenience wrappers): build echo's canonical `greedy_argmax`
//! container (parametric in the live model's `output_vocab_size`, so it binds
//! against Qwen3-0.6B's vocab 151936), then `register-program →
//! pipeline.instantiate(seed = BOS on tok) → pipeline.channel(out) →
//! pipeline.submit → out.take()`. The pass embeds the seeded BOS, the model
//! forward fills `ws.logits`, and the PTIR epilogue argmaxes them into the
//! host-reader `out` channel — so `out.take()` yields the model's greedy next
//! token. The harness asserts it equals a plain greedy decode of the same model.

use inferlet::inference::{ForwardPass, InputBinding};
use inferlet::pie::inferlet::forward::{self, ChannelSeed};
use inferlet::sampling::{Graph, OutputKind};
use inferlet::working_set::KvWorkingSet;
use inferlet::{Result, model};

use pie_sampling_ir::ptir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use pie_sampling_ir::ptir::op::{IntrinsicId, Op};
use pie_sampling_ir::ptir::registry::{Port, Stage};
use pie_sampling_ir::types::{DType, Shape};

/// Dense channel indices (container declaration order below).
const TOK: u32 = 0; // seeded input token (embed_tokens source)
const OUT: u32 = 1; // host-reader: the argmax token lands here
const BOS: i32 = 1; // seed token embedded on the first fire

/// Build echo's `greedy_argmax` trace for `vocab`: `argmax(logits) → tok/out`.
/// Mirrors `interface/sampling-ir/tests/ptir_golden.rs::golden_greedy_argmax`,
/// parametric in the live model's logits width (echo's `bind` requires the
/// `Logits` shape's vocab dim to equal the model's `output_vocab_size`).
fn greedy_argmax_container(vocab: u32) -> Vec<u8> {
    // %0 logits[1,vocab] → %1 reshape[vocab] → %2 argmax → %3 reshape[1] →
    // put(tok) + put(out).
    let ops = vec![
        Op::IntrinsicVal { intr: IntrinsicId::Logits, shape: Shape::matrix(1, vocab), dtype: DType::F32 },
        Op::Reshape { value: 0, shape: Shape::vector(vocab) },
        Op::ReduceArgmax(1),
        Op::Reshape { value: 2, shape: Shape::vector(1) },
        Op::ChanPut { chan: TOK, value: 3 },
        Op::ChanPut { chan: OUT, value: 3 },
    ];
    let container = TraceContainer {
        names: vec![],
        externs: vec![],
        channels: vec![
            // 0 tok: seeded input token, device-private (embed source).
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(DType::I32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: true,
            },
            // 1 out: host-reader — the guest `take`s the produced token here.
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(DType::I32),
                capacity: 1,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: vec![PortBinding { port: Port::EmbedTokens, source: PortSource::Channel(TOK) }],
        stages: vec![StageProgram { stage: Stage::Epilogue, ops }],
    };
    container.encode()
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    // The live model's logits width — the vocab the container's `Logits` op must
    // declare (direct model WIT binding).
    let vocab = model::output_vocab_size();

    // register → (hash-deduped) validated program identity.
    let bytes = greedy_argmax_container(vocab);
    let program = ptir::register_program(&bytes).map_err(|e| format!("register-program: {e}"))?;

    // instantiate with the BOS seed on `tok` (i32 little-endian, D2 per-instance).
    let seed = ChannelSeed { channel: TOK, value: BOS.to_le_bytes().to_vec() };
    let pipeline =
        ptir::Pipeline::instantiate(program, &[seed]).map_err(|e| format!("instantiate: {e}"))?;

    // Grab the host endpoint on `out` (the reader), then fire.
    let out = pipeline.channel(OUT).map_err(|e| format!("channel({OUT}): {e}"))?;
    pipeline.submit().map_err(|e| format!("submit: {e}"))?;

    // Take the produced token (blocks until the fire commits the cell).
    let token_bytes = out.take().map_err(|e| format!("out.take: {e}"))?;
    if token_bytes.len() != 4 {
        return Err(format!("out.take: expected 4 bytes (i32), got {}", token_bytes.len()));
    }
    let ptir_token =
        i32::from_le_bytes([token_bytes[0], token_bytes[1], token_bytes[2], token_bytes[3]]) as u32;

    // Plain sampling-IR greedy reference (oracle): forward the SAME BOS through
    // the raw keep-core surface (KvWorkingSet + ForwardPass + an argmax sampling
    // program), one step. Proves the PTIR stage-runner's argmax == the
    // sampling-IR argmax over the same model logits. (Raw-WIT per In Gim's
    // directive — the facade is deleted; this is the keep-core one-step decode.)
    let g = Graph::new(vocab);
    let token_v = g.intrinsic_logits_dyn().argmax();
    g.output(&token_v, OutputKind::Token);
    let built = g.build().map_err(|e| format!("build plain argmax: {e:?}"))?;
    let plain_program =
        inferlet::emit::emit_program(&built.program).map_err(|e| format!("emit: {e}"))?;

    let kv = KvWorkingSet::new();
    kv.alloc(1).map_err(|e| format!("plain greedy alloc: {e}"))?;
    let pass = ForwardPass::new();
    pass.kv_working_set(&kv, 0, 0, 0, 0, 1, 0);
    pass.input_tokens(&[BOS as u32], &[0]);
    pass.sampler(&plain_program, vec![InputBinding::Logits(vec![0])]);
    pass.execute();
    let out = pass.output().await.map_err(|e| format!("plain greedy output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("plain greedy read: {e:?}"))?;
    let plain_token = if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        return Err(format!("plain greedy: short tensor ({} bytes)", bytes.len()));
    };

    // Print both for the harness to capture + diff (ptir == plain = full §6.2
    // correctness: the stage-runner argmax matches the proven sampling-IR argmax).
    println!("PTIR_GREEDY_E2E ptir={ptir_token} plain={plain_token}");
    Ok(format!("ptir={ptir_token} plain={plain_token}"))
}
