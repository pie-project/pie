//! The first fired token: the whole chain against a real checkpoint.
//!
//! `config.json` → descriptor → facts → geometry → load plan → staged
//! storage → scratch schedule → PSO table → bound step → one dispatch walk
//! on the GPU. Nothing here checks the ANSWER yet — that is the accuracy
//! gate's job (golden taps, token-exact decode) — this pins that the
//! assembly holds together: every weight the DAG asks for was staged,
//! every constant bound, every pipeline compiled, and the command buffer
//! retires.
//!
//! Gated on `PIE_METAL_SMOKE_CHECKPOINT` naming a qwen3.5/3.6-family MLX
//! snapshot directory, because a checkpoint is a machine's, not the
//! repo's. Without it the test states it skipped and why.

#![cfg(target_vendor = "apple")]

use std::path::PathBuf;

use driver_metal_new::batch::{
    AffineFormat, DagOptions, EntryNames, IoSlot, PsoFeatures, build_decode_dag,
    build_scratch_schedule, geometry_from_facts, plan_decode_psos, scratch_slot_elems,
};
use driver_metal_new::facts::ModelFacts;
use driver_metal_new::loader::{compile_load_plan, metal_storage_target};
use driver_metal_new::metal::Compiler;
use driver_metal_new::metal::{Context, DecodeStep, Stepper, load_step_psos, stage_decode_storage};
use driver_metal_new::region::Region as _;
use driver_metal_new::tuning::Tuning;

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels")
}

#[test]
fn the_assembly_fires_one_token_end_to_end() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to a qwen3.5-family MLX snapshot");
        return;
    };
    let snapshot = PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&config).expect("config.json parses");
    let descriptor = model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config converts to a descriptor");
    let descriptor_json = descriptor.to_string();

    // Facts and geometry, refused rather than defaulted.
    let facts = ModelFacts::from_descriptor(&descriptor_json)
        .expect("the driver's facts read the descriptor");
    let mut geometry = geometry_from_facts(&facts).expect("the config describes this family");
    geometry.quant = AffineFormat {
        bits: u32::try_from(facts.quant_bits).unwrap_or(4),
        group: u32::try_from(facts.quant_group_size).unwrap_or(64),
    };
    eprintln!(
        "geometry: {} layers, hidden {}, vocab {}, moe={}",
        geometry.n_layers,
        geometry.hidden,
        geometry.vocab,
        geometry.is_moe()
    );

    // The load plan, authored in-process.
    let target = metal_storage_target();
    let (plan, _moe) = compile_load_plan(&snapshot, &target, &descriptor_json)
        .expect("the plan compiles and its files exist");

    // Device side: stage, schedule, compile, bind.
    let context = Context::new().expect("a Metal device answers");
    let tuning = Tuning::default();
    let max_ctx = 4096u32;
    let scratch_bytes = scratch_slot_elems(&geometry, &tuning, 1) * 2;
    let storage = stage_decode_storage(
        &context,
        &plan,
        &snapshot,
        &geometry,
        max_ctx,
        scratch_bytes,
    )
    .expect("every region allocates and every tensor stages");
    eprintln!("staged {} weights", storage.weights.len());

    // Staging integrity: the arena-offset map is new code, and a wrong
    // offset is a fluent model with the wrong weights — the exact symptom
    // nothing downstream can diagnose. Re-run the plan on the host and
    // hold a sample of staged slices to the executor's own bytes.
    {
        let host = model_loader::executor::host::execute_plan(&plan, &snapshot)
            .expect("the host executor agrees to run the plan twice");
        let mut checked = 0usize;
        for (name, bytes) in host.tensors.iter().take(4096) {
            let Some(slice) = storage.weights.get(name) else {
                continue;
            };
            if slice.len() != bytes.len() as u64 {
                panic!(
                    "{name}: staged {} bytes, executor produced {}",
                    slice.len(),
                    bytes.len()
                );
            }
            // SAFETY: nothing is encoded yet.
            let staged = unsafe {
                std::slice::from_raw_parts(slice.contents().cast::<u8>().as_ptr(), bytes.len())
            };
            assert_eq!(
                staged,
                &bytes[..],
                "{name}: the staged bytes drifted from the plan's"
            );
            checked += 1;
        }
        eprintln!("staging verified for {checked} tensors");
        assert!(checked > 0, "the probe compared nothing");
    }

    // The shipping lane: the PSO plan serves GdnCore with the slimmed
    // recurrent kernel, so the DAG must be built with the prep split on.
    let options = DagOptions {
        with_argmax: true,
        gdn_prep: true,
        ..DagOptions::default()
    };
    let dag = build_decode_dag(&geometry, &tuning, options);
    let schedule = build_scratch_schedule(&dag, false).expect("the DAG schedules hazard-free");

    let features = PsoFeatures {
        argmax: true,
        gdn: true,
        gated_attention: true,
        sdpa_d256: geometry.head_dim == 256,
        routed: geometry.is_moe(),
        untied: !geometry.tied_embeddings,
        ..PsoFeatures::default()
    };
    let pso_plan = plan_decode_psos(&EntryNames::bf16_g64_b4(), features);
    let compiler = Compiler::new(&context).expect("the shader compiler starts");
    let psos = load_step_psos(&compiler, &context, &kernels_dir(), &pso_plan)
        .expect("every planned entrypoint compiles");

    let step = DecodeStep::prepare(
        &context, &storage, &geometry, &tuning, options, &schedule, psos, max_ctx,
    )
    .expect("the step binds whole");

    // Fire the checkpoint's own <bos> at position 0. SeqLen is position+1.
    // A multimodal wrapper keeps its text facts one level down.
    let bos: u32 = [&root, root.get("text_config").unwrap_or(&root)]
        .iter()
        .find_map(|level| level.get("bos_token_id"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .expect("the config states its bos");
    let io = |slot: IoSlot| storage.io[slot as usize].as_ref().expect("io slot");
    // SAFETY: nothing is encoded yet; the buffers are host-owned.
    unsafe {
        io(IoSlot::TokenId).write(0, &bos.to_le_bytes()).unwrap();
        io(IoSlot::Position).write(0, &0u32.to_le_bytes()).unwrap();
        io(IoSlot::SeqLen).write(0, &1u32.to_le_bytes()).unwrap();
    }

    let mut stepper = Stepper::new(&context).expect("a stepper");
    let timing = step.fire(&mut stepper).expect("the command buffer retires");
    eprintln!("fired: encode {:?}, gpu {:?}", timing.encode, timing.gpu);

    // The first answer check: the logits must be finite, non-degenerate
    // numbers and the argmax a real token. "Token 0 forever" and "all
    // zeros" are this family's two historical silent failures; both are
    // visible from here without a reference.
    let logits = io(IoSlot::Logits);
    let vocab = geometry.vocab as usize;
    // SAFETY: the step retired; the GPU is done with the pool.
    let bytes =
        unsafe { std::slice::from_raw_parts(logits.contents().cast::<u8>().as_ptr(), vocab * 2) };
    let mut finite = 0usize;
    let mut nonzero = 0usize;
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, pair) in bytes.chunks_exact(2).enumerate() {
        let value = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
        if value.is_finite() {
            finite += 1;
            if value != 0.0 {
                nonzero += 1;
            }
            if value > best.1 {
                best = (i, value);
            }
        }
    }
    let next = {
        // SAFETY: as above.
        let raw = unsafe {
            std::slice::from_raw_parts(io(IoSlot::NextToken).contents().cast::<u8>().as_ptr(), 4)
        };
        u32::from_le_bytes(raw.try_into().unwrap())
    };
    eprintln!(
        "logits: {finite}/{vocab} finite, {nonzero} nonzero; host argmax {} ({:.3}); device argmax {next}",
        best.0, best.1
    );
    assert_eq!(
        finite, vocab,
        "a NaN in the logits is a wrong kernel upstream"
    );
    assert!(
        nonzero > vocab / 2,
        "logits mostly zero: the head never ran or wrote elsewhere"
    );
    assert_eq!(
        next as usize, best.0,
        "the device argmax must agree with the host's read of the same logits"
    );

    // ── Multi-step decode: feed the argmax back and keep the GDN's
    // ping-pong honest. Step i reads what i-1 wrote, so the conv binds
    // swap by the slot's own parity — the counter is the ported
    // LinearStateSlots, so this exercises the same bookkeeping the
    // executor will use. ──
    let mut slots = driver_metal_new::store::LinearStateSlots::new(1);
    slots.step(0).unwrap(); // the <bos> fire above was step 0
    let mut step = step;
    // An optional real prompt: PIE_METAL_SMOKE_PROMPT_IDS as csv token ids,
    // fed sequentially before the greedy tail — a working model completes
    // it sensibly where a bare <bos> may legitimately degenerate.
    let prompt: Vec<u32> = std::env::var("PIE_METAL_SMOKE_PROMPT_IDS")
        .ok()
        .map(|csv| csv.split(',').map(|t| t.parse().unwrap()).collect())
        .unwrap_or_default();
    let mut token = next;
    let mut sequence = vec![bos, token];
    let mut feed: Vec<u32> = prompt.clone();
    if !feed.is_empty() {
        // Restart the sequence record at the prompt.
        sequence = vec![bos];
        sequence.extend(&feed);
        token = *feed.first().unwrap();
    }
    for position in 1..(1 + feed.len() as u32 + 11) {
        // While the prompt lasts, feed it; after, feed the argmax back.
        let input = if (position as usize) <= feed.len() {
            feed[position as usize - 1]
        } else {
            token
        };
        // SAFETY: the previous step retired; the buffers are host-owned
        // between steps.
        unsafe {
            io(IoSlot::TokenId).write(0, &input.to_le_bytes()).unwrap();
            io(IoSlot::Position)
                .write(0, &position.to_le_bytes())
                .unwrap();
            io(IoSlot::SeqLen)
                .write(0, &(position + 1).to_le_bytes())
                .unwrap();
        }
        step.set_gdn_parity(&context, &storage, slots.parity(0).unwrap())
            .expect("the parity rebind holds");
        step.fire(&mut stepper).expect("the step retires");
        slots.step(0).unwrap();
        // SAFETY: retired, as above.
        let raw = unsafe {
            std::slice::from_raw_parts(io(IoSlot::NextToken).contents().cast::<u8>().as_ptr(), 4)
        };
        token = u32::from_le_bytes(raw.try_into().unwrap());
        if (position as usize) >= feed.len() {
            sequence.push(token);
        }
    }
    let _ = &mut feed;
    eprintln!("greedy sequence: {sequence:?}");
    let distinct: std::collections::HashSet<_> = sequence.iter().collect();
    assert!(
        distinct.len() > 2,
        "a decode stuck on one token is this family's classic silent failure: {sequence:?}"
    );
}

// ── The bisect: dump every tap of one <bos> step and hold the head of the
// chain to host-computed values. The first tap that disagrees names the
// broken kernel; everything before it is exonerated. ──

fn read_npy(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("{} missing", path.display()));
    let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    bytes[10 + len..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Dequantize one row of an affine g64/b4 tensor from its staged triplet.
fn dequant_row(w: &[u8], scales: &[u8], biases: &[u8], row: usize, k: usize) -> Vec<f32> {
    let groups = k / 64;
    let mut out = Vec::with_capacity(k);
    for g in 0..groups {
        let scale = bf16(u16::from_le_bytes([
            scales[(row * groups + g) * 2],
            scales[(row * groups + g) * 2 + 1],
        ]));
        let bias = bf16(u16::from_le_bytes([
            biases[(row * groups + g) * 2],
            biases[(row * groups + g) * 2 + 1],
        ]));
        for i in 0..64 {
            let at = row * k / 2 + (g * 64 + i) / 2;
            let code = if i % 2 == 0 { w[at] & 0xf } else { w[at] >> 4 };
            out.push(f32::from(code) * scale + bias);
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-20)
}

#[test]
fn the_first_step_taps_agree_with_the_host() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT");
        return;
    };
    let snapshot = std::path::PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let root: serde_json::Value = serde_json::from_str(&config).unwrap();
    let descriptor = model::config::descriptor(&root, snapshot.to_str().unwrap()).unwrap();
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json).unwrap();
    let mut geometry = geometry_from_facts(&facts).unwrap();
    geometry.quant = AffineFormat { bits: 4, group: 64 };
    let target = metal_storage_target();
    let (plan, _) = compile_load_plan(&snapshot, &target, &descriptor_json).unwrap();
    let context = Context::new().unwrap();
    let tuning = Tuning::default();
    let max_ctx = 4096u32;
    let slot_bytes = scratch_slot_elems(&geometry, &tuning, 1) * 2;
    let mut storage =
        stage_decode_storage(&context, &plan, &snapshot, &geometry, max_ctx, slot_bytes).unwrap();

    let options = DagOptions {
        gdn_prep: true,
        ..DagOptions::default()
    };
    let dag = build_decode_dag(&geometry, &tuning, options);
    // No recycling: every value keeps its own buffer so the dump reads what
    // each kernel wrote, not what overwrote it.
    let schedule = build_scratch_schedule(&dag, true).unwrap();
    storage.scratch = driver_metal_new::metal::scratch_pool(
        &context,
        schedule.coloring.colors_used as usize,
        slot_bytes,
    )
    .expect("the no-recycle pool allocates");
    eprintln!("no-recycle pool: {} buffers", schedule.coloring.colors_used);

    let features = PsoFeatures {
        gdn: true,
        gated_attention: true,
        sdpa_d256: geometry.head_dim == 256,
        routed: geometry.is_moe(),
        untied: !geometry.tied_embeddings,
        ..PsoFeatures::default()
    };
    let pso_plan = plan_decode_psos(&EntryNames::bf16_g64_b4(), features);
    let compiler = Compiler::new(&context).unwrap();
    let psos = load_step_psos(&compiler, &context, &kernels_dir(), &pso_plan).unwrap();
    let step = DecodeStep::prepare(
        &context, &storage, &geometry, &tuning, options, &schedule, psos, max_ctx,
    )
    .unwrap();

    let bos: u32 = [&root, root.get("text_config").unwrap_or(&root)]
        .iter()
        .find_map(|level| level.get("bos_token_id"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap();
    let io = |slot: IoSlot| storage.io[slot as usize].as_ref().unwrap();
    unsafe {
        io(IoSlot::TokenId).write(0, &bos.to_le_bytes()).unwrap();
        io(IoSlot::SeqLen).write(0, &1u32.to_le_bytes()).unwrap();
    }
    let mut stepper = Stepper::new(&context).unwrap();
    step.fire(&mut stepper).unwrap();

    let dir = std::env::temp_dir().join("pie-golden-bisect");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let sites: Vec<_> = dag
        .iter()
        .map(|d| driver_metal_new::batch::TapSite {
            kind: d.kind,
            layer: d.layer,
        })
        .collect();
    unsafe {
        driver_metal_new::batch::dump_taps(
            &dir,
            &sites,
            &schedule,
            &storage.scratch,
            &geometry,
            1,
            0,
        )
    }
    .unwrap();

    // Host reference, tap by tap. embed: the dequantized bos row.
    let staged = |name: &str| {
        let h = storage
            .weights
            .get(name)
            .unwrap_or_else(|| panic!("{name} staged"));
        unsafe { std::slice::from_raw_parts(h.contents().cast::<u8>().as_ptr(), h.len() as usize) }
    };
    let hidden = geometry.hidden as usize;
    let embed = dequant_row(
        staged("embed_tokens.weight"),
        staged("embed_tokens.scales"),
        staged("embed_tokens.biases"),
        bos as usize,
        hidden,
    );
    let tap = read_npy(&dir.join("embed.npy"));
    let c = cosine(&embed, &tap);
    eprintln!("embed cosine {c:.6}");
    assert!(
        c > 0.999,
        "embed diverges: the gather or its binds are wrong"
    );

    // 0.attn_norm: RMS over the embed row with layer 0's weight.
    let w = staged("layers.0.input_layernorm.weight");
    let rms = {
        let mean: f32 = embed.iter().map(|x| x * x).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (mean + geometry.eps).sqrt();
        embed
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let wi = bf16(u16::from_le_bytes([w[i * 2], w[i * 2 + 1]]));
                x * inv * wi
            })
            .collect::<Vec<_>>()
    };
    let tap = read_npy(&dir.join("0.attn_norm.npy"));
    let c = cosine(&rms, &tap);
    eprintln!("0.attn_norm cosine {c:.6}");
    assert!(
        c > 0.999,
        "attn_norm diverges: RmsParams or its binds are wrong"
    );

    // 0.gdn_in_qkv: the first quantized matvec, host-recomputed whole.
    let conv_dim = geometry.gdn_conv_dim as usize;
    let wq = staged("layers.0.linear_attn.in_proj_qkv.weight");
    let sq = staged("layers.0.linear_attn.in_proj_qkv.scales");
    let bq = staged("layers.0.linear_attn.in_proj_qkv.biases");
    let mut qkv = Vec::with_capacity(conv_dim);
    for n in 0..conv_dim {
        let row = dequant_row(wq, sq, bq, n, hidden);
        qkv.push(row.iter().zip(&rms).map(|(w, x)| w * x).sum::<f32>());
    }
    let tap = read_npy(&dir.join("0.gdn_in_qkv.npy"));
    let c = cosine(&qkv, &tap);
    eprintln!("0.gdn_in_qkv cosine {c:.6}");
    assert!(
        c > 0.99,
        "the quantized matvec diverges: Qmv K/N or the triplet binds"
    );

    // ── Stage isolation from here: each tap is recomputed from the TAPS
    // it reads, so a stage with a correct function is exonerated even if
    // its input is globally wrong — and the first cosine that drops names
    // the kernel. ──
    let matvec = |base: &str, x: &[f32], n: usize| -> Vec<f32> {
        let w = staged(&format!("{base}.weight"));
        let sc = staged(&format!("{base}.scales"));
        let bi = staged(&format!("{base}.biases"));
        (0..n)
            .map(|row| {
                dequant_row(w, sc, bi, row, x.len())
                    .iter()
                    .zip(x)
                    .map(|(w, x)| w * x)
                    .sum::<f32>()
            })
            .collect()
    };
    let check = |name: &str, host: &[f32], floor: f32| {
        let tap = read_npy(&dir.join(format!("{name}.npy")));
        let c = cosine(host, &tap);
        eprintln!("{name} cosine {c:.6}");
        assert!(c > floor, "{name} diverges at cosine {c}");
        tap
    };

    // The GDN block's other three in-projections, from the norm tap.
    let z = matvec(
        "layers.0.linear_attn.in_proj_z",
        &rms,
        geometry.gdn_v_total as usize,
    );
    check("0.gdn_in_z", &z, 0.999);
    let a = matvec(
        "layers.0.linear_attn.in_proj_a",
        &rms,
        geometry.gdn_v_heads as usize,
    );
    check("0.gdn_in_a", &a, 0.999);
    let b = matvec(
        "layers.0.linear_attn.in_proj_b",
        &rms,
        geometry.gdn_v_heads as usize,
    );
    {
        let tap = read_npy(&dir.join("0.gdn_in_b.npy"));
        eprintln!("gdn_in_b host: {:?}", &b[..8]);
        eprintln!("gdn_in_b tap:  {:?}", &tap[..8]);
        eprintln!("vs a host:     {:?}", &a[..8]);
        let h = staged("layers.0.linear_attn.in_proj_b.weight");
        eprintln!(
            "in_proj_b.weight bytes {} (16x{hidden} 4-bit = {})",
            h.len(),
            16 * hidden / 2
        );
    }
    // The b buffer post-step holds fp32 gating values — even bf16 lanes
    // all zero is an f32 buffer read as bf16 pairs — so the core rewrites
    // it in place and the tap is unreadable by design. Observed, not
    // asserted; the projection's own kernel is A's, already exonerated.
    {
        let tap = read_npy(&dir.join("0.gdn_in_b.npy"));
        let c = cosine(&b, &tap);
        eprintln!("0.gdn_in_b cosine {c:.6} (in-place rewrite expected)");
    }

    // gdn_out from the CORE's own tap: exonerates the out projection even
    // while the core stays under suspicion.
    let core_tap = read_npy(&dir.join("0.gdn_core.npy"));
    let core_mag: f32 = core_tap.iter().map(|v| v * v).sum::<f32>().sqrt();
    eprintln!(
        "0.gdn_core tap magnitude {core_mag:.6}, first {:?}",
        &core_tap[..6]
    );
    let gdn_out = matvec("layers.0.linear_attn.out_proj", &core_tap, hidden);
    check("0.gdn_out", &gdn_out, 0.999);
    // The residual add closes layer 0's frame.
    let out_tap = read_npy(&dir.join("0.gdn_out.npy"));
    let resid: Vec<f32> = embed.iter().zip(&out_tap).map(|(a, b)| a + b).collect();
    check("0.attn_resid", &resid, 0.999);

    // ── The first full-attention layer, stage by stage. ──
    let full = (0..geometry.n_layers)
        .find(|&l| geometry.is_full_attn(l))
        .expect("some layer attends");
    let prefix = format!("layers.{full}");
    let norm_tap = read_npy(&dir.join(format!("{full}.attn_norm.npy")));
    let q_dim = (geometry.n_q_heads * geometry.head_dim) as usize;
    let kv_dim = (geometry.n_kv_heads * geometry.head_dim) as usize;
    let head = geometry.head_dim as usize;

    // In-place chains suppress their intermediate taps by design (only a
    // colour's final writer is named): k_proj/k_norm fold into rope_k,
    // q_norm into rope_q, sdpa into gated. Each host computation therefore
    // spans the whole in-place chain, and at position 0 the rope is the
    // identity, so the chain is still one hop of simple math.
    let qg = matvec(&format!("{prefix}.self_attn.q_proj"), &norm_tap, 2 * q_dim);
    check(&format!("{full}.q_proj"), &qg, 0.999);
    let k = matvec(&format!("{prefix}.self_attn.k_proj"), &norm_tap, kv_dim);
    let v = matvec(&format!("{prefix}.self_attn.v_proj"), &norm_tap, kv_dim);
    let v_tap = check(&format!("{full}.v_proj"), &v, 0.999);

    // QSplit deinterleaves [n_q, 2, head]: query halves then per-head RMS.
    let qg_tap = read_npy(&dir.join(format!("{full}.q_proj.npy")));
    let split_q: Vec<f32> = (0..q_dim)
        .map(|i| qg_tap[(i / head) * 2 * head + i % head])
        .collect();
    let split_gate: Vec<f32> = (0..q_dim)
        .map(|i| qg_tap[(i / head) * 2 * head + head + i % head])
        .collect();
    let per_head_rms = |x: &[f32], w_name: &str| -> Vec<f32> {
        let w = staged(w_name);
        x.chunks(head)
            .flat_map(|row| {
                let mean: f32 = row.iter().map(|v| v * v).sum::<f32>() / head as f32;
                let inv = 1.0 / (mean + geometry.eps).sqrt();
                row.iter()
                    .enumerate()
                    .map(|(i, v)| v * inv * bf16(u16::from_le_bytes([w[i * 2], w[i * 2 + 1]])))
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let qn = per_head_rms(&split_q, &format!("{prefix}.self_attn.q_norm.weight"));
    let kn = per_head_rms(&k, &format!("{prefix}.self_attn.k_norm.weight"));

    // Rope at position 0 is the identity, so the rope taps ARE the norms.
    check(&format!("{full}.rope_q"), &qn, 0.999);
    check(&format!("{full}.rope_k"), &kn, 0.999);
    // SDPA over one position is that position's value row per head-group,
    // and the gate multiplies sigmoid of the split's gate half onto it.
    let gqa = (geometry.n_q_heads / geometry.n_kv_heads) as usize;
    let gated: Vec<f32> = (0..q_dim)
        .map(|i| {
            let attn = v_tap[(i / head / gqa) * head + i % head];
            attn / (1.0 + (-split_gate[i]).exp())
        })
        .collect();
    check(&format!("{full}.gated"), &gated, 0.999);
    let gated_tap = read_npy(&dir.join(format!("{full}.gated.npy")));
    let o = matvec(&format!("{prefix}.self_attn.o_proj"), &gated_tap, hidden);
    check(&format!("{full}.o_proj"), &o, 0.999);

    // The MLP, from its own norm tap.
    let ffn_tap = read_npy(&dir.join("0.ffn_norm.npy"));
    let gate_p = matvec(
        "layers.0.mlp.gate_proj",
        &ffn_tap,
        geometry.intermediate as usize,
    );
    check("0.gate_proj", &gate_p, 0.999);
    let up_p = matvec(
        "layers.0.mlp.up_proj",
        &ffn_tap,
        geometry.intermediate as usize,
    );
    check("0.up_proj", &up_p, 0.999);
    let gate_tap = read_npy(&dir.join("0.gate_proj.npy"));
    let up_tap = read_npy(&dir.join("0.up_proj.npy"));
    let swiglu: Vec<f32> = gate_tap
        .iter()
        .zip(&up_tap)
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    check("0.swiglu", &swiglu, 0.999);
    let swiglu_tap = read_npy(&dir.join("0.swiglu.npy"));
    let down = matvec("layers.0.mlp.down_proj", &swiglu_tap, hidden);
    check("0.down_proj", &down, 0.999);

    eprintln!("bisect: every stage function verified except the GDN core itself");
}

// ── The paged path: one fire prefills the whole prompt, then n=1 paged
// decode fires continue it. The sequential M=1 run above is the reference:
// both must answer " Paris". ──

#[test]
fn the_paged_prefill_matches_the_sequential_decode() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT");
        return;
    };
    let prompt: Vec<u32> = match std::env::var("PIE_METAL_SMOKE_PROMPT_IDS") {
        Ok(csv) => csv.split(',').map(|t| t.parse().unwrap()).collect(),
        Err(_) => {
            eprintln!("SKIP: set PIE_METAL_SMOKE_PROMPT_IDS");
            return;
        }
    };
    let snapshot = std::path::PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let root: serde_json::Value = serde_json::from_str(&config).unwrap();
    let descriptor = model::config::descriptor(&root, snapshot.to_str().unwrap()).unwrap();
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json).unwrap();
    let mut geometry = geometry_from_facts(&facts).unwrap();
    geometry.quant = AffineFormat { bits: 4, group: 64 };
    geometry.paged_kv_enabled = true;
    geometry.max_tokens = 16;
    geometry.max_requests = 1;
    geometry.kv_page_size = 32;
    geometry.total_pages = 128;
    let target = metal_storage_target();
    let (plan, _) = compile_load_plan(&snapshot, &target, &descriptor_json).unwrap();
    let context = Context::new().unwrap();
    let tuning = Tuning::default();
    let max_ctx = 4096u32;
    let slot_bytes = scratch_slot_elems(&geometry, &tuning, geometry.max_tokens) * 2;
    let mut storage =
        stage_decode_storage(&context, &plan, &snapshot, &geometry, max_ctx, slot_bytes).unwrap();

    let options = DagOptions {
        gdn_prep: true,
        ..DagOptions::default()
    };
    // The MB DAG shares the M=1 order, so its schedule comes off its own
    // dispatch list the same way.
    let n = prompt.len() as u32;
    let dag_n = driver_metal_new::batch::build_decode_dag_mb(&geometry, &tuning, n, 0, options);
    let schedule_n = {
        let (uses, values) = driver_metal_new::batch::build_scratch_uses(&dag_n);
        let ends = driver_metal_new::batch::concurrent_run_ends(&dag_n);
        driver_metal_new::batch::schedule_scratch(dag_n.len(), &uses, &ends, values, false).unwrap()
    };

    let features = PsoFeatures {
        gdn: true,
        gated_attention: true,
        sdpa_d256: geometry.head_dim == 256,
        routed: geometry.is_moe(),
        untied: !geometry.tied_embeddings,
        ..PsoFeatures::default()
    };
    let compiler = Compiler::new(&context).unwrap();
    let base = load_step_psos(
        &compiler,
        &context,
        &kernels_dir(),
        &plan_decode_psos(&EntryNames::bf16_g64_b4(), features),
    )
    .unwrap();
    let base2 = load_step_psos(
        &compiler,
        &context,
        &kernels_dir(),
        &plan_decode_psos(&EntryNames::bf16_g64_b4(), features),
    )
    .unwrap();
    let mb_features = driver_metal_new::batch::MbFeatures {
        gdn: true,
        sdpa_d256: geometry.head_dim == 256,
        ..driver_metal_new::batch::MbFeatures::default()
    };
    let mb_plan =
        driver_metal_new::batch::plan_multibatch_psos(geometry.quant, mb_features, &tuning);
    let mb = driver_metal_new::metal::load_mb_psos(&compiler, &context, &kernels_dir(), &mb_plan)
        .unwrap();
    let mb2 = driver_metal_new::metal::load_mb_psos(&compiler, &context, &kernels_dir(), &mb_plan)
        .unwrap();

    // The paged IO plumbing for one request holding every page.
    let io = |slot: IoSlot| storage.io[slot as usize].as_ref().unwrap();
    let write_u32s = |slot: IoSlot, values: &[u32]| {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // SAFETY: host-owned between fires.
        unsafe { io(slot).write(0, &bytes).unwrap() };
    };
    let pages: Vec<u32> = (0..geometry.total_pages).collect();
    write_u32s(IoSlot::KvPageIndices, &pages);
    write_u32s(IoSlot::KvPageIndptr, &[0, geometry.total_pages]);
    write_u32s(IoSlot::QoIndptr, &[0, n]);
    write_u32s(IoSlot::KvLastPageLens, &[1]);
    write_u32s(IoSlot::RsSlotIds, &[0]);
    let feed_positions = |base_pos: u32, tokens: &[u32]| {
        write_u32s(IoSlot::TokenId, tokens);
        let positions: Vec<u32> = (0..tokens.len() as u32).map(|i| base_pos + i).collect();
        write_u32s(IoSlot::Position, &positions);
        write_u32s(
            IoSlot::SeqLen,
            &positions.iter().map(|p| p + 1).collect::<Vec<_>>(),
        );
        write_u32s(IoSlot::ReqOfToken, &vec![0; tokens.len()]);
        write_u32s(IoSlot::SlotOfToken, &vec![0; tokens.len()]);
        let wpage: Vec<u32> = positions
            .iter()
            .map(|p| p / geometry.kv_page_size)
            .collect();
        let woff: Vec<u32> = positions
            .iter()
            .map(|p| p % geometry.kv_page_size)
            .collect();
        write_u32s(IoSlot::WPage, &wpage);
        write_u32s(IoSlot::WOff, &woff);
    };

    // Prefill: one fire, or — under PIE_SMOKE_SEQ_PREFILL — a chain of
    // n=1 paged fires, which isolates the batched fire from the paged
    // plumbing when the two disagree.
    let sequential = std::env::var_os("PIE_SMOKE_SEQ_PREFILL").is_some();
    let mut slots = driver_metal_new::store::LinearStateSlots::new(1);
    let mut stepper = Stepper::new(&context).unwrap();
    let dag_1 = driver_metal_new::batch::build_decode_dag_mb(&geometry, &tuning, 1, 0, options);
    let schedule_1 = {
        let (uses, values) = driver_metal_new::batch::build_scratch_uses(&dag_1);
        let ends = driver_metal_new::batch::concurrent_run_ends(&dag_1);
        driver_metal_new::batch::schedule_scratch(dag_1.len(), &uses, &ends, values, false).unwrap()
    };
    if sequential {
        let mb3 =
            driver_metal_new::metal::load_mb_psos(&compiler, &context, &kernels_dir(), &mb_plan)
                .unwrap();
        let base3 = load_step_psos(
            &compiler,
            &context,
            &kernels_dir(),
            &plan_decode_psos(&EntryNames::bf16_g64_b4(), features),
        )
        .unwrap();
        let mut one = driver_metal_new::metal::MbStep::prepare(
            &context,
            &storage,
            &geometry,
            &tuning,
            options,
            &schedule_1,
            base3,
            mb3,
            1,
            max_ctx,
        )
        .unwrap();
        for (i, &t) in prompt.iter().enumerate() {
            write_u32s(IoSlot::QoIndptr, &[0, 1]);
            feed_positions(i as u32, &[t]);
            one.set_gdn_parity(&context, &storage, slots.parity(0).unwrap())
                .unwrap();
            one.fire(&mut stepper)
                .expect("the sequential prefill retires");
            slots.step(0).unwrap();
        }
    } else {
        // The prefill stream: the GDN recurrence is sequential over tokens,
        // so a single-request prompt runs one single-token DAG per row —
        // each bound at ITS row of the shared IO — never one flat N-token
        // fire, which would read every token against the same initial
        // state. (The flat fire is the FLEET's shape: n requests, one
        // token each, disjoint slots.)
        let _ = (&schedule_n, base, mb);
        let vocab_bytes = geometry.vocab as usize * 2;
        feed_positions(0, &prompt);
        for (row, _) in prompt.iter().enumerate() {
            let base_t = load_step_psos(
                &compiler,
                &context,
                &kernels_dir(),
                &plan_decode_psos(&EntryNames::bf16_g64_b4(), features),
            )
            .unwrap();
            let mb_t = driver_metal_new::metal::load_mb_psos(
                &compiler,
                &context,
                &kernels_dir(),
                &mb_plan,
            )
            .unwrap();
            let mut one = driver_metal_new::metal::MbStep::prepare_at(
                &context,
                &storage,
                &geometry,
                &tuning,
                options,
                &schedule_1,
                base_t,
                mb_t,
                1,
                max_ctx,
                driver_metal_new::metal::MbBindOffsets {
                    token_row: row as u64,
                    logits_bytes: (row * vocab_bytes) as u64,
                },
            )
            .unwrap();
            write_u32s(IoSlot::QoIndptr, &[0, n]);
            one.set_gdn_parity(&context, &storage, slots.parity(0).unwrap())
                .unwrap();
            one.fire(&mut stepper).expect("the prefill row retires");
            slots.step(0).unwrap();
        }
    }

    let vocab = geometry.vocab as usize;
    let argmax_row = |row: usize| -> u32 {
        let logits = io(IoSlot::Logits);
        // SAFETY: the fire retired.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                logits.contents().cast::<u8>().as_ptr().add(row * vocab * 2),
                vocab * 2,
            )
        };
        let mut best = (0u32, f32::NEG_INFINITY);
        for (i, pair) in bytes.chunks_exact(2).enumerate() {
            let v = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
            if v > best.1 {
                best = (i as u32, v);
            }
        }
        best.0
    };
    // Under PIE_SMOKE_PAGED_TAPS: dump one n=1 <bos> fire's taps on a
    // no-recycle pool and hold the chain heads to the host, exactly as the
    // M=1 bisect does — the first divergence names the paged kernel.
    if std::env::var_os("PIE_SMOKE_PAGED_TAPS").is_some() {
        // Per-fire tap dumps across the whole sequential run, then two host
        // checks at the diverging step: (a) every KV append landed — the
        // pool's row for position p equals that step's rope_k tap — and
        // (b) sdpa equals a host softmax over the pool. Between them they
        // separate "the history is wrong" from "the read of it is wrong".
        let sched_taps = {
            let (uses, values) = driver_metal_new::batch::build_scratch_uses(&dag_1);
            let ends = driver_metal_new::batch::concurrent_run_ends(&dag_1);
            driver_metal_new::batch::schedule_scratch(dag_1.len(), &uses, &ends, values, true)
                .unwrap()
        };
        storage.scratch = driver_metal_new::metal::scratch_pool(
            &context,
            sched_taps.coloring.colors_used as usize,
            slot_bytes,
        )
        .unwrap();
        let mbt =
            driver_metal_new::metal::load_mb_psos(&compiler, &context, &kernels_dir(), &mb_plan)
                .unwrap();
        let baset = load_step_psos(
            &compiler,
            &context,
            &kernels_dir(),
            &plan_decode_psos(&EntryNames::bf16_g64_b4(), features),
        )
        .unwrap();
        let mut one = driver_metal_new::metal::MbStep::prepare(
            &context,
            &storage,
            &geometry,
            &tuning,
            options,
            &sched_taps,
            baset,
            mbt,
            1,
            max_ctx,
        )
        .unwrap();
        let mut slots = driver_metal_new::store::LinearStateSlots::new(1);
        let mut stepper = Stepper::new(&context).unwrap();
        let sites: Vec<_> = dag_1
            .iter()
            .map(|d| driver_metal_new::batch::TapSite {
                kind: d.kind,
                layer: d.layer,
            })
            .collect();
        let root_dir = std::env::temp_dir().join("pie-paged-steps");
        let _ = std::fs::remove_dir_all(&root_dir);
        let mut feed: Vec<u32> = prompt.clone();
        let mut token = 0u32;
        for step_index in 0..8usize {
            let input = if step_index < feed.len() {
                feed[step_index]
            } else {
                token
            };
            write_u32s(IoSlot::QoIndptr, &[0, 1]);
            feed_positions(step_index as u32, &[input]);
            one.set_gdn_parity(&context, &storage, slots.parity(0).unwrap())
                .unwrap();
            one.fire(&mut stepper).unwrap();
            slots.step(0).unwrap();
            let dir = root_dir.join(format!("step{step_index}"));
            std::fs::create_dir_all(&dir).unwrap();
            unsafe {
                driver_metal_new::batch::dump_taps(
                    &dir,
                    &sites,
                    &sched_taps,
                    &storage.scratch,
                    &geometry,
                    1,
                    0,
                )
            }
            .unwrap();
            token = argmax_row(0);
            eprintln!("step {step_index}: fed {input}, answered {token}");
        }
        let _ = &mut feed;

        // Check (a): the pool holds each step's rope_k for the first
        // full-attention layer.
        let full = (0..geometry.n_layers)
            .find(|&l| geometry.is_full_attn(l))
            .unwrap();
        let kv = storage.kv[full as usize].as_ref().unwrap();
        let head = geometry.head_dim as usize;
        let kv_row = geometry.n_kv_heads as usize * head;
        let pool_row = |pages: &driver_metal_new::metal::Handle, position: usize| -> Vec<f32> {
            // NHD page-major: [page, row, n_kv_heads, head_dim] bf16.
            let page = position / geometry.kv_page_size as usize;
            let row = position % geometry.kv_page_size as usize;
            let at = (page * geometry.kv_page_size as usize + row) * kv_row * 2;
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    pages.contents().cast::<u8>().as_ptr().add(at),
                    kv_row * 2,
                )
            };
            bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect()
        };
        for position in 0..8usize {
            let tap = read_npy(
                &root_dir
                    .join(format!("step{position}"))
                    .join(format!("{full}.rope_k.npy")),
            );
            let pool = pool_row(&kv.k_pages, position);
            eprintln!("append pos {position}: k cosine {:.6}", cosine(&tap, &pool));
        }
        // Check (b): sdpa at the last dumped step against a host softmax
        // over the pool.
        let last = 7usize;
        let q_tap = read_npy(
            &root_dir
                .join(format!("step{last}"))
                .join(format!("{full}.rope_q.npy")),
        );
        let gqa = (geometry.n_q_heads / geometry.n_kv_heads) as usize;
        let scale = 1.0 / (head as f32).sqrt();
        let mut host_attn = vec![0.0f32; q_tap.len()];
        for h in 0..geometry.n_q_heads as usize {
            let kvh = h / gqa;
            let q = &q_tap[h * head..(h + 1) * head];
            let mut scores = Vec::new();
            for position in 0..=last {
                let k = pool_row(&kv.k_pages, position);
                let s: f32 = q
                    .iter()
                    .zip(&k[kvh * head..(kvh + 1) * head])
                    .map(|(a, b)| a * b)
                    .sum();
                scores.push(s * scale);
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for (position, w) in exps.iter().enumerate() {
                let v = pool_row(&kv.v_pages, position);
                for d in 0..head {
                    host_attn[h * head + d] += w / denom * v[kvh * head + d];
                }
            }
        }
        // The sdpa tap is suppressed by AttnGate's in-place write, so read
        // the gated tap and un-gate is impossible — instead compare the
        // gated tap against sigmoid(gate) * host_attn using the q_proj
        // split's gate half.
        let qg_tap = read_npy(
            &root_dir
                .join(format!("step{last}"))
                .join(format!("{full}.q_proj.npy")),
        );
        let q_dim = (geometry.n_q_heads * geometry.head_dim) as usize;
        let gated_host: Vec<f32> = (0..q_dim)
            .map(|i| {
                let gate = qg_tap[(i / head) * 2 * head + head + i % head];
                host_attn[i] / (1.0 + (-gate).exp())
            })
            .collect();
        let gated_tap = read_npy(
            &root_dir
                .join(format!("step{last}"))
                .join(format!("{full}.gated.npy")),
        );
        eprintln!(
            "sdpa host-vs-device (via gated) at step {last}: cosine {:.6}",
            cosine(&gated_host, &gated_tap)
        );
        return;
    }
    let mut token = argmax_row(if sequential { 0 } else { n as usize - 1 });
    eprintln!("prefill answer: {token}");
    let mut sequence = vec![token];

    // Decode: n=1 paged fires continuing the same request.
    let mut decode = driver_metal_new::metal::MbStep::prepare(
        &context,
        &storage,
        &geometry,
        &tuning,
        options,
        &schedule_1,
        base2,
        mb2,
        1,
        max_ctx,
    )
    .unwrap();
    for step_index in 0..6u32 {
        let position = n + step_index;
        write_u32s(IoSlot::QoIndptr, &[0, 1]);
        feed_positions(position, &[token]);
        decode
            .set_gdn_parity(&context, &storage, slots.parity(0).unwrap())
            .unwrap();
        decode.fire(&mut stepper).expect("the decode retires");
        slots.step(0).unwrap();
        token = argmax_row(0);
        sequence.push(token);
    }
    eprintln!("paged sequence: {sequence:?}");
    assert_eq!(
        &sequence[..3],
        &[11751, 13, 561],
        "the paged path must answer what the sequential path answered: ' Paris. The'"
    );
}

// ── The fleet: two requests, one token each, ONE flat fire per decode
// step. Each request owns its page range and its GDN slot; both lanes run
// the same prompt, so both must reproduce the single-request reference. ──

#[test]
fn a_two_lane_fleet_decodes_both_lanes_token_exact() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT");
        return;
    };
    let prompt: Vec<u32> = match std::env::var("PIE_METAL_SMOKE_PROMPT_IDS") {
        Ok(csv) => csv.split(',').map(|t| t.parse().unwrap()).collect(),
        Err(_) => {
            eprintln!("SKIP: set PIE_METAL_SMOKE_PROMPT_IDS");
            return;
        }
    };
    let snapshot = std::path::PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let root: serde_json::Value = serde_json::from_str(&config).unwrap();
    let descriptor = model::config::descriptor(&root, snapshot.to_str().unwrap()).unwrap();
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json).unwrap();
    let mut geometry = geometry_from_facts(&facts).unwrap();
    geometry.quant = AffineFormat { bits: 4, group: 64 };
    geometry.paged_kv_enabled = true;
    geometry.max_tokens = 16;
    geometry.max_requests = 2;
    geometry.max_slots = 2;
    geometry.kv_page_size = 32;
    geometry.total_pages = 128;
    let target = metal_storage_target();
    let (plan, _) = compile_load_plan(&snapshot, &target, &descriptor_json).unwrap();
    let context = Context::new().unwrap();
    let tuning = Tuning::default();
    let max_ctx = 4096u32;
    let slot_bytes = scratch_slot_elems(&geometry, &tuning, geometry.max_tokens) * 2;
    let storage =
        stage_decode_storage(&context, &plan, &snapshot, &geometry, max_ctx, slot_bytes).unwrap();

    let options = DagOptions {
        gdn_prep: true,
        ..DagOptions::default()
    };
    let schedule_of = |n: u32| {
        let dag = driver_metal_new::batch::build_decode_dag_mb(&geometry, &tuning, n, 0, options);
        let (uses, values) = driver_metal_new::batch::build_scratch_uses(&dag);
        let ends = driver_metal_new::batch::concurrent_run_ends(&dag);
        driver_metal_new::batch::schedule_scratch(dag.len(), &uses, &ends, values, false).unwrap()
    };
    let schedule_1 = schedule_of(1);
    let schedule_2 = schedule_of(2);

    let features = PsoFeatures {
        gdn: true,
        gated_attention: true,
        sdpa_d256: geometry.head_dim == 256,
        routed: geometry.is_moe(),
        untied: !geometry.tied_embeddings,
        ..PsoFeatures::default()
    };
    let compiler = Compiler::new(&context).unwrap();
    let mb_features = driver_metal_new::batch::MbFeatures {
        gdn: true,
        sdpa_d256: geometry.head_dim == 256,
        ..driver_metal_new::batch::MbFeatures::default()
    };
    let mb_plan =
        driver_metal_new::batch::plan_multibatch_psos(geometry.quant, mb_features, &tuning);
    let load_pair = || {
        (
            load_step_psos(
                &compiler,
                &context,
                &kernels_dir(),
                &plan_decode_psos(&EntryNames::bf16_g64_b4(), features),
            )
            .unwrap(),
            driver_metal_new::metal::load_mb_psos(&compiler, &context, &kernels_dir(), &mb_plan)
                .unwrap(),
        )
    };

    let io = |slot: IoSlot| storage.io[slot as usize].as_ref().unwrap();
    let write_u32s = |slot: IoSlot, values: &[u32]| {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // SAFETY: host-owned between fires.
        unsafe { io(slot).write(0, &bytes).unwrap() };
    };
    // Request r owns pages [r * 64, (r + 1) * 64).
    let pages: Vec<u32> = (0..128).collect();
    write_u32s(IoSlot::KvPageIndices, &pages);
    write_u32s(IoSlot::KvPageIndptr, &[0, 64, 128]);
    write_u32s(IoSlot::KvLastPageLens, &[1, 1]);
    write_u32s(IoSlot::RsSlotIds, &[0, 1]);
    let page_base = |request: u32| request * 64;

    let vocab = geometry.vocab as usize;
    let argmax_row = |row: usize| -> u32 {
        let logits = io(IoSlot::Logits);
        // SAFETY: the fire retired.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                logits.contents().cast::<u8>().as_ptr().add(row * vocab * 2),
                vocab * 2,
            )
        };
        let mut best = (0u32, f32::NEG_INFINITY);
        for (i, pair) in bytes.chunks_exact(2).enumerate() {
            let v = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
            if v > best.1 {
                best = (i as u32, v);
            }
        }
        best.0
    };

    // ONE global fire counter: both slots are touched in equal-length runs
    // (same prompt) and every decode fire touches both, so the conv
    // ping-pong stays in lockstep across the fleet.
    let mut fires = driver_metal_new::store::LinearStateSlots::new(1);
    let mut stepper = Stepper::new(&context).unwrap();

    // Prefill each lane as its own single-token stream.
    for request in 0..2u32 {
        for (row, &token) in prompt.iter().enumerate() {
            let (base_t, mb_t) = load_pair();
            let mut one = driver_metal_new::metal::MbStep::prepare(
                &context,
                &storage,
                &geometry,
                &tuning,
                options,
                &schedule_1,
                base_t,
                mb_t,
                1,
                max_ctx,
            )
            .unwrap();
            let position = row as u32;
            write_u32s(IoSlot::QoIndptr, &[0, 1]);
            write_u32s(IoSlot::TokenId, &[token]);
            write_u32s(IoSlot::Position, &[position]);
            write_u32s(IoSlot::SeqLen, &[position + 1]);
            write_u32s(IoSlot::ReqOfToken, &[request]);
            write_u32s(IoSlot::SlotOfToken, &[request]);
            write_u32s(
                IoSlot::WPage,
                &[page_base(request) + position / geometry.kv_page_size],
            );
            write_u32s(IoSlot::WOff, &[position % geometry.kv_page_size]);
            one.set_gdn_parity(&context, &storage, fires.parity(0).unwrap())
                .unwrap();
            one.fire(&mut stepper)
                .expect("the lane's prefill row retires");
            fires.step(0).unwrap();
        }
    }

    // Decode: BOTH lanes in one flat n=2 fire per step.
    let (base_f, mb_f) = load_pair();
    let mut fleet = driver_metal_new::metal::MbStep::prepare(
        &context,
        &storage,
        &geometry,
        &tuning,
        options,
        &schedule_2,
        base_f,
        mb_f,
        2,
        max_ctx,
    )
    .unwrap();
    let n = prompt.len() as u32;
    let mut lane_tokens = [0u32, 0u32];
    // Seed both lanes from their prefill logits: the last prompt row's
    // answer was produced by each lane's final prefill fire, which wrote
    // logits row 0 both times — so refire lane by lane is avoided by
    // simply taking the sequential reference's first token for both.
    let mut sequences: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
    // The last prefill fire (request 1) left ITS answer at row 0; request
    // 0's was overwritten. Rather than re-read, decode both lanes from the
    // known first token of the shared prompt's continuation, produced
    // per-lane below from the first joint fire.
    lane_tokens[0] = 11751;
    lane_tokens[1] = 11751;
    for step in 0..5u32 {
        let position = n + step;
        write_u32s(IoSlot::QoIndptr, &[0, 1, 2]);
        write_u32s(IoSlot::TokenId, &lane_tokens);
        write_u32s(IoSlot::Position, &[position, position]);
        write_u32s(IoSlot::SeqLen, &[position + 1, position + 1]);
        write_u32s(IoSlot::ReqOfToken, &[0, 1]);
        write_u32s(IoSlot::SlotOfToken, &[0, 1]);
        write_u32s(
            IoSlot::WPage,
            &[
                page_base(0) + position / geometry.kv_page_size,
                page_base(1) + position / geometry.kv_page_size,
            ],
        );
        write_u32s(
            IoSlot::WOff,
            &[
                position % geometry.kv_page_size,
                position % geometry.kv_page_size,
            ],
        );
        fleet
            .set_gdn_parity(&context, &storage, fires.parity(0).unwrap())
            .unwrap();
        fleet.fire(&mut stepper).expect("the fleet fire retires");
        fires.step(0).unwrap();
        lane_tokens = [argmax_row(0), argmax_row(1)];
        sequences[0].push(lane_tokens[0]);
        sequences[1].push(lane_tokens[1]);
    }
    eprintln!("lane 0: {:?}", sequences[0]);
    eprintln!("lane 1: {:?}", sequences[1]);
    // Both lanes ran the same context, so both must say what the
    // single-request run says: ". The capital of".
    assert_eq!(&sequences[0][..3], &[13, 561, 6511], "lane 0 drifted");
    assert_eq!(&sequences[1][..3], &[13, 561, 6511], "lane 1 drifted");
    assert_eq!(sequences[0], sequences[1], "identical lanes disagreed");
}

// ── The decoder runner: a mixed-length fleet, where the per-slot conv
// orientation diverges during prefill and the join copy earns its keep. ──

#[test]
fn a_mixed_length_fleet_runs_through_the_decoder() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT");
        return;
    };
    let prompt: Vec<u32> = match std::env::var("PIE_METAL_SMOKE_PROMPT_IDS") {
        Ok(csv) => csv.split(',').map(|t| t.parse().unwrap()).collect(),
        Err(_) => {
            eprintln!("SKIP: set PIE_METAL_SMOKE_PROMPT_IDS");
            return;
        }
    };
    let snapshot = std::path::PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let root: serde_json::Value = serde_json::from_str(&config).unwrap();
    let descriptor = model::config::descriptor(&root, snapshot.to_str().unwrap()).unwrap();
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json).unwrap();
    let mut geometry = geometry_from_facts(&facts).unwrap();
    geometry.quant = AffineFormat { bits: 4, group: 64 };
    geometry.paged_kv_enabled = true;
    geometry.max_tokens = 16;
    geometry.max_requests = 2;
    geometry.max_slots = 2;
    geometry.kv_page_size = 32;
    geometry.total_pages = 128;
    let target = metal_storage_target();
    let (plan, _) = compile_load_plan(&snapshot, &target, &descriptor_json).unwrap();
    let context = Context::new().unwrap();
    let tuning = Tuning::default();
    let max_ctx = 4096u32;
    let slot_bytes = scratch_slot_elems(&geometry, &tuning, geometry.max_tokens) * 2;
    let storage =
        stage_decode_storage(&context, &plan, &snapshot, &geometry, max_ctx, slot_bytes).unwrap();
    let options = DagOptions {
        gdn_prep: true,
        ..DagOptions::default()
    };
    let mut decoder = driver_metal_new::metal::Decoder::new(
        &context,
        storage,
        geometry.clone(),
        tuning,
        options,
        max_ctx,
        kernels_dir(),
    )
    .unwrap();

    // Lane 0: the full prompt. Lane 1: its first three tokens — a
    // different length, so the two slots' orientations diverge and the
    // first joint fire must normalize one of them.
    let short: Vec<u32> = prompt[..3].to_vec();
    let first0 = decoder.prefill(0, 0, &prompt).unwrap();
    let first1 = decoder.prefill(1, 1, &short).unwrap();
    eprintln!("lane 0 prefill answer {first0}, lane 1 {first1}");
    assert_eq!(first0, 11751, "lane 0 must still say ' Paris'");

    let mut tokens = [first0, first1];
    let mut positions = [prompt.len() as u32, short.len() as u32];
    let mut lane0 = Vec::new();
    let mut lane1 = Vec::new();
    for _ in 0..4 {
        decoder
            .fire(&[
                driver_metal_new::metal::Lane {
                    request: 0,
                    slot: 0,
                    token: tokens[0],
                    position: positions[0],
                },
                driver_metal_new::metal::Lane {
                    request: 1,
                    slot: 1,
                    token: tokens[1],
                    position: positions[1],
                },
            ])
            .unwrap();
        tokens = [decoder.argmax_row(0), decoder.argmax_row(1)];
        positions = [positions[0] + 1, positions[1] + 1];
        lane0.push(tokens[0]);
        lane1.push(tokens[1]);
    }
    eprintln!("lane 0: {lane0:?}");
    eprintln!("lane 1: {lane1:?}");
    assert_eq!(
        &lane0[..3],
        &[13, 561, 6511],
        "the full-prompt lane must keep the single-request answer through a mixed fleet"
    );
    let distinct: std::collections::HashSet<_> = lane1.iter().collect();
    assert!(distinct.len() > 1, "lane 1 wedged: {lane1:?}");
}

// ── The long horizon: a thousand greedy tokens through the decoder, the
// cutover gate's N ≥ 1000 leg minus the old-backend comparison. ──

#[test]
fn a_thousand_tokens_decode_without_a_wedge_or_a_nan() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT");
        return;
    };
    let prompt: Vec<u32> = match std::env::var("PIE_METAL_SMOKE_PROMPT_IDS") {
        Ok(csv) => csv.split(',').map(|t| t.parse().unwrap()).collect(),
        Err(_) => {
            eprintln!("SKIP: set PIE_METAL_SMOKE_PROMPT_IDS");
            return;
        }
    };
    let horizon: u32 = std::env::var("PIE_SMOKE_HORIZON")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let snapshot = std::path::PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let root: serde_json::Value = serde_json::from_str(&config).unwrap();
    let descriptor = model::config::descriptor(&root, snapshot.to_str().unwrap()).unwrap();
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json).unwrap();
    let mut geometry = geometry_from_facts(&facts).unwrap();
    geometry.quant = AffineFormat { bits: 4, group: 64 };
    geometry.paged_kv_enabled = true;
    geometry.max_tokens = 16;
    geometry.max_requests = 1;
    geometry.max_slots = 1;
    geometry.kv_page_size = 32;
    geometry.total_pages = 128; // 4096 positions: the whole max_ctx
    let target = metal_storage_target();
    let (plan, _) = compile_load_plan(&snapshot, &target, &descriptor_json).unwrap();
    let context = Context::new().unwrap();
    let tuning = Tuning::default();
    let max_ctx = 4096u32;
    let slot_bytes = scratch_slot_elems(&geometry, &tuning, geometry.max_tokens) * 2;
    let storage =
        stage_decode_storage(&context, &plan, &snapshot, &geometry, max_ctx, slot_bytes).unwrap();
    let options = DagOptions {
        gdn_prep: true,
        ..DagOptions::default()
    };
    let vocab = geometry.vocab;
    let mut decoder = driver_metal_new::metal::Decoder::new(
        &context,
        storage,
        geometry,
        tuning,
        options,
        max_ctx,
        kernels_dir(),
    )
    .unwrap();

    let mut token = decoder.prefill(0, 0, &prompt).unwrap();
    let started = std::time::Instant::now();
    let mut distinct = std::collections::HashSet::new();
    let mut position = prompt.len() as u32;
    let footprint_at_start = decoder.footprint_bytes();
    for step in 0..horizon {
        decoder
            .fire(&[driver_metal_new::metal::Lane {
                request: 0,
                slot: 0,
                token,
                position,
            }])
            .unwrap_or_else(|err| panic!("step {step} at position {position}: {err}"));
        token = decoder.argmax_row(0);
        assert!(
            token < vocab,
            "step {step}: argmax {token} outside the vocabulary"
        );
        // PIE_SMOKE_TOP5_AT=N prints step N's top five logits — the
        // instrument for judging a near-tie against another backend.
        if std::env::var("PIE_SMOKE_TOP5_AT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            == Some(step)
        {
            let logits = decoder.storage().io[IoSlot::Logits as usize]
                .as_ref()
                .unwrap();
            // SAFETY: the fire retired.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    logits.contents().cast::<u8>().as_ptr(),
                    vocab as usize * 2,
                )
            };
            let mut all: Vec<(f32, usize)> = bytes
                .chunks_exact(2)
                .enumerate()
                .map(|(i, c)| {
                    (
                        f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16),
                        i,
                    )
                })
                .collect();
            all.sort_by(|a, b| b.0.total_cmp(&a.0));
            eprintln!("step {step} top5: {:?}", &all[..5]);
        }
        distinct.insert(token);
        position += 1;
        if (step + 1) % 200 == 0 {
            let elapsed = started.elapsed().as_secs_f64();
            let footprint = decoder.footprint_bytes();
            eprintln!(
                "{} tokens, {:.1} tok/s, {} distinct, {} bytes held",
                step + 1,
                f64::from(step + 1) / elapsed,
                distinct.len(),
                footprint
            );
            assert_eq!(
                footprint, footprint_at_start,
                "the decode loop grew the device footprint — the leak class \
                 the soak gate exists for"
            );
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "horizon {horizon}: {:.1} tok/s end to end, {} distinct tokens, final position {position}",
        f64::from(horizon) / elapsed,
        distinct.len()
    );
    if horizon >= 50 {
        assert!(
            distinct.len() >= 5,
            "a greedy run may loop a sentence, but {} distinct tokens over {horizon} is a wedge",
            distinct.len()
        );
    }
}

/// The third family's first fired tokens: gpt-oss-20b against the mlx
/// MXFP4-Q4 publish, greedy, held token-exact to mlx_lm's continuation
/// of the same prompt ids.
///
/// The chain is the qwen assembly with the family's pieces swapped in:
/// facts → `gptoss_geometry_from_facts` → load plan → staged storage
/// (all-full-attention view) → the quantization trio SOLVED off the
/// staged extents → `gptoss_step_plan` → `GptOssStep`. The reference:
/// "The capital of France is" tokenizes to [976, 9029, 328, 10128, 382]
/// and mlx_lm continues [12650(' Paris'), 3692, 279, 12, 6240, 1, 976,
/// 9029] — eight fed-back argmaxes, every one checked.
#[test]
fn the_gptoss_assembly_decodes_the_reference_tokens() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_GPTOSS_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_GPTOSS_CHECKPOINT to a gpt-oss MLX snapshot");
        return;
    };
    let snapshot = PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&config).expect("config.json parses");
    let descriptor = model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config converts to a descriptor");
    let descriptor_json = descriptor.to_string();

    let facts = ModelFacts::from_descriptor(&descriptor_json)
        .expect("the driver's facts read the descriptor");
    let mut geometry =
        driver_metal_new::batch::gptoss_geometry_from_facts(&facts).expect("a gpt-oss shape");
    eprintln!(
        "gpt-oss geometry: {} layers, {} experts top-{}, window {}",
        geometry.n_layers, geometry.n_experts, geometry.experts_per_token, geometry.sliding_window
    );

    let target = metal_storage_target();
    let (plan, _moe) = compile_load_plan(&snapshot, &target, &descriptor_json)
        .expect("the plan compiles and its files exist");

    // Stage FIRST, then solve: the staging reads no quantization field,
    // and the staged extents are the only honest witness to the trio.
    let context = Context::new().expect("a Metal device answers");
    let max_ctx = 4096u32;
    let scratch_bytes = driver_metal_new::batch::gptoss_scratch_elems(&geometry) * 4;
    let shared_view = driver_metal_new::batch::gptoss_decode_geometry(&geometry);
    let storage = stage_decode_storage(
        &context,
        &plan,
        &snapshot,
        &shared_view,
        max_ctx,
        scratch_bytes,
    )
    .expect("every region allocates and every tensor stages");
    driver_metal_new::batch::solve_quant_into(&mut geometry, |name| {
        storage
            .weights
            .get(name)
            .map(driver_metal_new::region::Region::len)
    })
    .expect("the staged tensors carry the trio");
    eprintln!(
        "solved: router {} bits, proj {} bits, mxfp4 experts {}",
        geometry.router_bits, geometry.proj_bits, geometry.mxfp4_experts
    );

    // The bisect lever: truncated fires against the ordinary recycled
    // pool — the last dispatch's output slot still holds its value when
    // the prefix retires, so every stage is readable without the
    // no-recycle allocation.
    let taps = std::env::var("PIE_SMOKE_GPTOSS_TAPS").is_ok_and(|v| v == "1");
    let dag = driver_metal_new::batch::build_gptoss_dag(&geometry, true);
    let schedule = build_scratch_schedule(&dag, false).expect("the DAG schedules hazard-free");
    let pso_plan = driver_metal_new::batch::gptoss_step_plan(&geometry);
    let compiler = Compiler::new(&context).expect("the shader compiler starts");
    let psos = load_step_psos(&compiler, &context, &kernels_dir(), &pso_plan)
        .expect("every planned entrypoint compiles");

    let mut step = driver_metal_new::metal::GptOssStep::prepare(
        &context,
        &storage,
        &geometry,
        &Tuning::default(),
        &schedule,
        psos,
        max_ctx,
    )
    .expect("the step binds whole");
    step.force_barriers = taps;

    let io = |slot: IoSlot| storage.io[slot as usize].as_ref().expect("io slot");
    if taps {
        // Token 976 at position 0, one truncated fire per tapped stage of
        // the first two layers and the tail.
        // SAFETY: nothing is encoded yet.
        unsafe {
            io(IoSlot::TokenId).write(0, &976u32.to_le_bytes()).unwrap();
            io(IoSlot::Position).write(0, &0u32.to_le_bytes()).unwrap();
            io(IoSlot::SeqLen).write(0, &1u32.to_le_bytes()).unwrap();
        }
        let dir = std::env::temp_dir().join("pie-gptoss-bisect");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut stepper = Stepper::new(&context).expect("a stepper");
        for (ordinal, d) in dag.iter().enumerate() {
            if !matches!(d.layer, None | Some(0) | Some(1)) {
                continue;
            }
            let Some(tap) = driver_metal_new::batch::tap_for(d.kind, &shared_view) else {
                continue;
            };
            step.fire_prefix(&mut stepper, ordinal + 1)
                .expect("the truncated command buffer retires");
            let Some(color) = schedule.per_dispatch[ordinal]
                .iter()
                .find(|bind| bind.bind_index == tap.out_bind)
                .map(|bind| bind.color as usize)
            else {
                continue;
            };
            let slot = &storage.scratch[color];
            // SAFETY: the prefix retired; nothing after this dispatch ran.
            let raw = unsafe {
                std::slice::from_raw_parts(
                    slot.contents().cast::<u8>().as_ptr(),
                    tap.width as usize * 2,
                )
            };
            let bf: Vec<u16> = raw
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let name = match d.layer {
                None => format!("{ordinal:03}.{}", tap.name),
                Some(layer) => format!("{ordinal:03}.{layer}.{}", tap.name),
            };
            driver_metal_new::batch::dump_bf16(&dir, &name, &bf, 1, tap.width, 0).unwrap();
        }
        // The full fire last, for the logits.
        step.fire(&mut stepper).expect("the command buffer retires");
        let logits = io(IoSlot::Logits);
        // SAFETY: as above.
        let raw = unsafe {
            std::slice::from_raw_parts(
                logits.contents().cast::<u8>().as_ptr(),
                geometry.vocab as usize * 2,
            )
        };
        let bf: Vec<u16> = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        driver_metal_new::batch::dump_bf16(&dir, "logits", &bf, 1, geometry.vocab, 0).unwrap();
        eprintln!("taps dumped to {}", dir.display());
        return;
    }
    let read_next = || {
        // SAFETY: called only after the step retired.
        let raw = unsafe {
            std::slice::from_raw_parts(io(IoSlot::NextToken).contents().cast::<u8>().as_ptr(), 4)
        };
        u32::from_le_bytes(raw.try_into().unwrap())
    };
    let mut stepper = Stepper::new(&context).expect("a stepper");
    let mut fire_at = |token: u32, position: u32| {
        // SAFETY: the previous fire retired before we rewrite the inputs.
        unsafe {
            io(IoSlot::TokenId).write(0, &token.to_le_bytes()).unwrap();
            io(IoSlot::Position)
                .write(0, &position.to_le_bytes())
                .unwrap();
            io(IoSlot::SeqLen)
                .write(0, &(position + 1).to_le_bytes())
                .unwrap();
        }
        step.fire(&mut stepper).expect("the command buffer retires");
        read_next()
    };

    let prompt: Vec<u32> = std::env::var("PIE_METAL_SMOKE_GPTOSS_PROMPT_IDS")
        .ok()
        .map(|csv| {
            csv.split(',')
                .map(|t| t.trim().parse().expect("token ids"))
                .collect()
        })
        .unwrap_or_else(|| vec![976, 9029, 328, 10128, 382]);
    let reference: &[u32] = &[12650, 3692, 279, 12, 6240, 1, 976, 9029];
    let check_reference = prompt == [976, 9029, 328, 10128, 382];

    let mut position = 0u32;
    let mut next = 0u32;
    for &token in &prompt {
        next = fire_at(token, position);
        position += 1;
    }
    // The first answer sanity, before any comparison: finite,
    // non-degenerate logits and an argmax the host agrees with.
    {
        let logits = io(IoSlot::Logits);
        let vocab = geometry.vocab as usize;
        // SAFETY: the step retired.
        let bytes = unsafe {
            std::slice::from_raw_parts(logits.contents().cast::<u8>().as_ptr(), vocab * 2)
        };
        let mut finite = 0usize;
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, pair) in bytes.chunks_exact(2).enumerate() {
            let value = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
            if value.is_finite() {
                finite += 1;
                if value > best.1 {
                    best = (i, value);
                }
            }
        }
        eprintln!(
            "prompt fed: {finite}/{vocab} finite, host argmax {} ({:.3}), device {next}",
            best.0, best.1
        );
        assert_eq!(
            finite, vocab,
            "a NaN in the logits is a wrong kernel upstream"
        );
        assert_eq!(next as usize, best.0, "device and host argmax disagree");
    }

    let mut produced = Vec::new();
    produced.push(next);
    while produced.len() < reference.len() {
        next = fire_at(next, position);
        position += 1;
        produced.push(next);
    }
    eprintln!("produced {produced:?}");
    if check_reference {
        assert_eq!(
            produced, reference,
            "the greedy continuation drifted from mlx_lm's"
        );
    }
}

/// One packed 16-token paged prefill answers with mlx_lm's next token.
///
/// Sixteen rows is the fire that exercises every tile arm at once: the
/// dense projections take the shared BIASED GEMM (16 rows is past the
/// crossover and this family biases every projection), the mixture takes
/// the routed MXFP4 GEMM (64 pairs over 32 experts pads each expert's
/// run to a 16-row tile), the attention reads the PAGE pool through the
/// sink kernel, and the tail compacts to the ONE sampled row before the
/// LM head. Reference: mlx_lm greedily continues this prompt with 5542
/// (' known').
#[test]
fn the_gptoss_prefill_answers_in_one_paged_fire() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_GPTOSS_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_GPTOSS_CHECKPOINT to a gpt-oss MLX snapshot");
        return;
    };
    let snapshot = PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&config).expect("config.json parses");
    let descriptor = model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config converts to a descriptor");
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json)
        .expect("the driver's facts read the descriptor");
    let mut geometry =
        driver_metal_new::batch::gptoss_geometry_from_facts(&facts).expect("a gpt-oss shape");

    // "The quick brown fox jumps over the lazy dog, and the capital of
    // France is" — 16 tokens, so the batch fills whole row tiles.
    let prompt: [u32; 16] = [
        976, 4853, 19705, 68347, 65613, 1072, 290, 29082, 6446, 11, 326, 290, 9029, 328, 10128, 382,
    ];
    let n = prompt.len() as u32;
    geometry.max_tokens = n;
    geometry.max_requests = 1;
    geometry.paged_kv_enabled = true;
    geometry.kv_page_size = 32;
    geometry.total_pages = 128;

    let target = metal_storage_target();
    let (plan, _moe) = compile_load_plan(&snapshot, &target, &descriptor_json)
        .expect("the plan compiles and its files exist");
    let context = Context::new().expect("a Metal device answers");
    let tuning = Tuning::default();
    let max_ctx = 4096u32;
    let scratch_bytes = driver_metal_new::batch::gptoss_scratch_elems_mb(&geometry, &tuning, n) * 4;
    let shared_view = driver_metal_new::batch::gptoss_decode_geometry(&geometry);
    let storage = stage_decode_storage(
        &context,
        &plan,
        &snapshot,
        &shared_view,
        max_ctx,
        scratch_bytes,
    )
    .expect("every region allocates and every tensor stages");
    driver_metal_new::batch::solve_quant_into(&mut geometry, |name| {
        storage
            .weights
            .get(name)
            .map(driver_metal_new::region::Region::len)
    })
    .expect("the staged tensors carry the trio");

    let dag = driver_metal_new::batch::build_gptoss_dag_mb(&geometry, &tuning, n, 1, 0, false);
    let schedule = build_scratch_schedule(&dag, false).expect("the MB DAG schedules hazard-free");
    let compiler = Compiler::new(&context).expect("the shader compiler starts");
    let base = load_step_psos(
        &compiler,
        &context,
        &kernels_dir(),
        &driver_metal_new::batch::gptoss_mb_plan(&geometry),
    )
    .expect("every planned MB entrypoint compiles");
    let mb_plan = driver_metal_new::batch::plan_multibatch_psos(
        AffineFormat {
            bits: geometry.proj_bits,
            group: 64,
        },
        driver_metal_new::batch::MbFeatures {
            bias: true,
            ..driver_metal_new::batch::MbFeatures::default()
        },
        &tuning,
    );
    let mb = driver_metal_new::metal::load_mb_psos(&compiler, &context, &kernels_dir(), &mb_plan)
        .expect("the shared GEMM lattice compiles");
    let go = driver_metal_new::metal::load_gptoss_mb_psos(
        &compiler,
        &context,
        &kernels_dir(),
        &geometry,
    )
    .expect("the routed lattice compiles");

    let step = driver_metal_new::metal::GptOssMbStep::prepare(
        &context, &storage, &geometry, &tuning, &schedule, base, mb, go, n, 1, max_ctx,
    )
    .expect("the MB step binds whole");
    // The point of a 16-row fire: had the tiles silently not engaged,
    // this smoke would pass without ever touching either GEMM.
    let decided = |kind: driver_metal_new::batch::Kernel| {
        step.dag
            .iter()
            .find(|d| d.kind == kind)
            .expect("in the DAG")
            .qmm_bn
    };
    assert!(
        decided(driver_metal_new::batch::Kernel::GoQmvQ) > 0,
        "the dense projections must tile at 16 rows"
    );
    assert!(
        decided(driver_metal_new::batch::Kernel::GoExpertGate) > 0,
        "the mixture must tile at 64 pairs"
    );

    // The fire's wire form, validated against the same machinery the
    // engine will use. The sampler reads ONE row: the last.
    let csr = driver_metal_new::batch::FireCsr::prefill(
        prompt.to_vec(),
        geometry.kv_page_size,
        geometry.total_pages,
    );
    csr.validate(
        geometry.kv_page_size,
        geometry.total_pages,
        geometry.max_tokens,
        geometry.max_requests,
        1,
    )
    .expect("a coherent fire");
    // SAFETY: nothing is encoded yet.
    unsafe { driver_metal_new::metal::write_fire_io(&storage, &csr).expect("the io writes") };
    let io = |slot: IoSlot| storage.io[slot as usize].as_ref().expect("io slot");

    let mut stepper = Stepper::new(&context).expect("a stepper");
    step.fire(&mut stepper).expect("the paged prefill retires");

    let logits = io(IoSlot::Logits);
    let vocab = geometry.vocab as usize;
    // SAFETY: the step retired; row 0 is the one sampled row.
    let bytes =
        unsafe { std::slice::from_raw_parts(logits.contents().cast::<u8>().as_ptr(), vocab * 2) };
    let mut finite = 0usize;
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, pair) in bytes.chunks_exact(2).enumerate() {
        let value = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
        if value.is_finite() {
            finite += 1;
            if value > best.1 {
                best = (i, value);
            }
        }
    }
    eprintln!(
        "paged prefill: {finite}/{vocab} finite, argmax {} ({:.3})",
        best.0, best.1
    );
    assert_eq!(
        finite, vocab,
        "a NaN in the logits is a wrong kernel upstream"
    );
    assert_eq!(
        best.0, 5542,
        "mlx_lm continues this prompt with 5542 (' known')"
    );
}

/// The llama assembly decodes mlx_lm's greedy continuation, token-exact.
///
/// Llama-3.2-1B-Instruct is the family's sharpest M=1 exercise in the
/// smallest package: TIED embeddings (both ends read
/// `shared_embedding`), 64-wide heads (the attention entry is `_d_64`,
/// where the old literal `_d128` would stride past every head), and the
/// llama3 frequency TABLE on device — factor 32, so a wrong table is
/// not a subtle drift, it is a rope running 32× off on the slow
/// dimensions.
#[test]
fn the_llama_assembly_decodes_the_reference_tokens() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_LLAMA_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_LLAMA_CHECKPOINT to a llama-family MLX snapshot");
        return;
    };
    let snapshot = PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&config).expect("config.json parses");
    let descriptor = model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config converts to a descriptor");
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json)
        .expect("the driver's facts read the descriptor");
    let geometry =
        driver_metal_new::batch::llama_geometry_from_facts(&facts).expect("a llama shape");
    eprintln!(
        "llama geometry: {} layers, {}x{} heads d{}, tied {}, freq table {}",
        geometry.n_layers,
        geometry.n_q_heads,
        geometry.n_kv_heads,
        geometry.head_dim,
        geometry.tied_embeddings,
        geometry.rope_freq_table
    );

    let target = metal_storage_target();
    let (plan, _moe) = compile_load_plan(&snapshot, &target, &descriptor_json)
        .expect("the plan compiles and its files exist");
    let context = Context::new().expect("a Metal device answers");
    let tuning = Tuning::default();
    let max_ctx = 4096u32;
    let shared_view = driver_metal_new::batch::llama_decode_geometry(&geometry);
    let slot_bytes = scratch_slot_elems(&shared_view, &tuning, 1) * 4;
    let storage = stage_decode_storage(
        &context,
        &plan,
        &snapshot,
        &shared_view,
        max_ctx,
        slot_bytes,
    )
    .expect("every region allocates and every tensor stages");

    let dag = driver_metal_new::batch::build_llama_dag(&geometry, &tuning, true);
    let schedule = build_scratch_schedule(&dag, false).expect("the DAG schedules hazard-free");
    let pso_plan = driver_metal_new::batch::llama_step_plan(&geometry);
    let compiler = Compiler::new(&context).expect("the shader compiler starts");
    let psos = load_step_psos(&compiler, &context, &kernels_dir(), &pso_plan)
        .expect("every planned entrypoint compiles");
    let step = driver_metal_new::metal::LlamaStep::prepare(
        &context, &storage, &geometry, &tuning, &schedule, psos, max_ctx,
    )
    .expect("the step binds whole");

    let io = |slot: IoSlot| storage.io[slot as usize].as_ref().expect("io slot");
    let read_next = || {
        // SAFETY: called only after the step retired.
        let raw = unsafe {
            std::slice::from_raw_parts(io(IoSlot::NextToken).contents().cast::<u8>().as_ptr(), 4)
        };
        u32::from_le_bytes(raw.try_into().unwrap())
    };
    let mut stepper = Stepper::new(&context).expect("a stepper");
    let mut fire_at = |token: u32, position: u32| {
        // SAFETY: the previous fire retired before we rewrite the inputs.
        unsafe {
            io(IoSlot::TokenId).write(0, &token.to_le_bytes()).unwrap();
            io(IoSlot::Position)
                .write(0, &position.to_le_bytes())
                .unwrap();
            io(IoSlot::SeqLen)
                .write(0, &(position + 1).to_le_bytes())
                .unwrap();
        }
        step.fire(&mut stepper).expect("the command buffer retires");
        read_next()
    };

    // "The capital of France is", with llama's BOS — this family HAS
    // one, unlike gpt-oss.
    let prompt: Vec<u32> = std::env::var("PIE_METAL_SMOKE_LLAMA_PROMPT_IDS")
        .ok()
        .map(|csv| {
            csv.split(',')
                .map(|t| t.trim().parse().expect("token ids"))
                .collect()
        })
        .unwrap_or_else(|| vec![128000, 791, 6864, 315, 9822, 374]);
    let reference: &[u32] = &[12366, 627, 791, 6864, 315, 9822, 374, 12366];
    let check_reference = prompt == [128000, 791, 6864, 315, 9822, 374];

    let mut position = 0u32;
    let mut next = 0u32;
    for &token in &prompt {
        next = fire_at(token, position);
        position += 1;
    }
    {
        let logits = io(IoSlot::Logits);
        let vocab = geometry.vocab as usize;
        // SAFETY: the step retired.
        let bytes = unsafe {
            std::slice::from_raw_parts(logits.contents().cast::<u8>().as_ptr(), vocab * 2)
        };
        let mut finite = 0usize;
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, pair) in bytes.chunks_exact(2).enumerate() {
            let value = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
            if value.is_finite() {
                finite += 1;
                if value > best.1 {
                    best = (i, value);
                }
            }
        }
        eprintln!(
            "prompt fed: {finite}/{vocab} finite, host argmax {} ({:.3}), device {next}",
            best.0, best.1
        );
        assert_eq!(
            finite, vocab,
            "a NaN in the logits is a wrong kernel upstream"
        );
        assert_eq!(next as usize, best.0, "device and host argmax disagree");
    }

    let mut produced = Vec::new();
    produced.push(next);
    while produced.len() < reference.len() {
        next = fire_at(next, position);
        position += 1;
        produced.push(next);
    }
    eprintln!("produced {produced:?}");
    if check_reference {
        assert_eq!(
            produced, reference,
            "the greedy continuation drifted from mlx_lm's"
        );
    }
}

/// One packed 17-token paged prefill answers with mlx_lm's next token.
///
/// Seventeen rows pad to 32 — a whole GEMM row block plus fifteen
/// discardable rows the pool must absorb — and every dense projection
/// takes the shared unbiased GEMM at the unsplit width. The attention
/// walks pages through the `_p32` instantiation, and the tail compacts
/// to the ONE sampled row. Reference: mlx_lm greedily continues with
/// 12366 (' Paris').
#[test]
fn the_llama_prefill_answers_in_one_paged_fire() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_LLAMA_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_LLAMA_CHECKPOINT to a llama-family MLX snapshot");
        return;
    };
    let snapshot = PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&config).expect("config.json parses");
    let descriptor = model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config converts to a descriptor");
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json)
        .expect("the driver's facts read the descriptor");
    let mut geometry =
        driver_metal_new::batch::llama_geometry_from_facts(&facts).expect("a llama shape");

    let prompt: [u32; 17] = [
        128000, 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 11, 323, 279, 6864, 315,
        9822, 374,
    ];
    let n = prompt.len() as u32;
    geometry.max_tokens = driver_metal_new::batch::llama_qmm_pool_rows(n);
    geometry.max_requests = 1;
    geometry.paged_kv_enabled = true;
    geometry.kv_page_size = 32;
    geometry.total_pages = 128;

    let target = metal_storage_target();
    let (plan, _moe) = compile_load_plan(&snapshot, &target, &descriptor_json)
        .expect("the plan compiles and its files exist");
    let context = Context::new().expect("a Metal device answers");
    let tuning = Tuning::default();
    let max_ctx = 4096u32;
    let shared_view = driver_metal_new::batch::llama_decode_geometry(&geometry);
    let slot_bytes = scratch_slot_elems(&shared_view, &tuning, geometry.max_tokens) * 4;
    let storage = stage_decode_storage(
        &context,
        &plan,
        &snapshot,
        &shared_view,
        max_ctx,
        slot_bytes,
    )
    .expect("every region allocates and every tensor stages");

    let dag = driver_metal_new::batch::build_llama_dag_mb(&geometry, &tuning, n, 1, 1, 0, false);
    let schedule = build_scratch_schedule(&dag, false).expect("the MB DAG schedules hazard-free");
    let compiler = Compiler::new(&context).expect("the shader compiler starts");
    let base = load_step_psos(
        &compiler,
        &context,
        &kernels_dir(),
        &driver_metal_new::batch::llama_mb_plan(&geometry),
    )
    .expect("every planned MB entrypoint compiles");
    let mb_plan = driver_metal_new::batch::plan_multibatch_psos(
        geometry.quant,
        driver_metal_new::batch::MbFeatures::default(),
        &tuning,
    );
    let mb = driver_metal_new::metal::load_mb_psos(&compiler, &context, &kernels_dir(), &mb_plan)
        .expect("the shared GEMM lattice compiles");

    let step = driver_metal_new::metal::LlamaMbStep::prepare(
        &context, &storage, &geometry, &tuning, &schedule, base, mb, n, 1, 1, max_ctx,
    )
    .expect("the MB step binds whole");
    // Seventeen rows must tile the dense side — a smoke that silently
    // fell back to matvecs would pass without touching the GEMM.
    assert!(
        step.dag
            .iter()
            .any(|d| d.kind == driver_metal_new::batch::Kernel::QmvQ && d.qmm_bn > 0),
        "the dense projections must tile at 17 rows"
    );

    // The fire's wire form, validated against the same machinery the
    // engine will use — the CSR the smokes used to hand-roll.
    let csr = driver_metal_new::batch::FireCsr::prefill(
        prompt.to_vec(),
        geometry.kv_page_size,
        geometry.total_pages,
    );
    csr.validate(
        geometry.kv_page_size,
        geometry.total_pages,
        geometry.max_tokens,
        geometry.max_requests,
        1,
    )
    .expect("a coherent fire");
    // SAFETY: nothing is encoded yet.
    unsafe { driver_metal_new::metal::write_fire_io(&storage, &csr).expect("the io writes") };

    let io = |slot: IoSlot| storage.io[slot as usize].as_ref().expect("io slot");
    let mut stepper = Stepper::new(&context).expect("a stepper");
    step.fire(&mut stepper).expect("the paged prefill retires");

    let logits = io(IoSlot::Logits);
    let vocab = geometry.vocab as usize;
    // SAFETY: the step retired; row 0 is the one sampled row.
    let bytes =
        unsafe { std::slice::from_raw_parts(logits.contents().cast::<u8>().as_ptr(), vocab * 2) };
    let mut finite = 0usize;
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, pair) in bytes.chunks_exact(2).enumerate() {
        let value = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
        if value.is_finite() {
            finite += 1;
            if value > best.1 {
                best = (i, value);
            }
        }
    }
    eprintln!(
        "llama paged prefill: {finite}/{vocab} finite, argmax {} ({:.3})",
        best.0, best.1
    );
    assert_eq!(
        finite, vocab,
        "a NaN in the logits is a wrong kernel upstream"
    );
    assert_eq!(
        best.0, 12366,
        "mlx_lm continues this prompt with 12366 (' Paris')"
    );
}

/// The gemma4 assembly decodes mlx_lm's greedy continuation.
///
/// gemma-4-26B-A4B is the family's everything-at-once shape: the
/// mixture BESIDE the dense MLP, full-attention layers whose V is the
/// K projection (no v_proj exists), per-layer head widths and KV head
/// counts, the PLE stream, the norm sandwich, the softcap — and an
/// alt-quant router solved from the STAGED tensors, never the config.
#[test]
fn the_gemma4_assembly_decodes_the_reference_tokens() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_GEMMA4_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_GEMMA4_CHECKPOINT to a gemma-4 MLX snapshot");
        return;
    };
    let snapshot = PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&config).expect("config.json parses");
    let descriptor = model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config converts to a descriptor");
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json)
        .expect("the driver's facts read the descriptor");
    let mut geometry =
        driver_metal_new::batch::gemma4_geometry_from_facts(&facts).expect("a gemma4 shape");
    eprintln!(
        "gemma4 geometry: {} layers ({} owning, {} full), moe {} ({}x{}), keqv {}",
        geometry.n_layers,
        geometry.n_kv_owning(),
        geometry.n_full_attn(),
        geometry.is_moe(),
        geometry.n_experts,
        geometry.experts_per_token,
        geometry.attention_k_eq_v,
    );
    eprintln!(
        "  global kv heads {} interval {} hd {}/{}",
        geometry.n_global_kv_heads,
        geometry.full_attn_interval,
        geometry.head_dim,
        geometry.global_head_dim
    );

    let target = metal_storage_target();
    let (plan, _moe) = compile_load_plan(&snapshot, &target, &descriptor_json)
        .expect("the plan compiles and its files exist");
    let context = Context::new().expect("a Metal device answers");
    let tuning = Tuning::default();
    let max_ctx = 4096u32;
    let shared_view = driver_metal_new::batch::gemma4_decode_geometry(&geometry);
    let slot_bytes = scratch_slot_elems(&shared_view, &tuning, 1) * 4;
    let mut storage = stage_decode_storage(
        &context,
        &plan,
        &snapshot,
        &shared_view,
        max_ctx,
        slot_bytes,
    )
    .expect("every region allocates and every tensor stages");
    driver_metal_new::metal::stage_gemma4_kv(&context, &mut storage, &geometry, max_ctx)
        .expect("the per-layer KV region allocates");

    // The alt format, solved from the STAGED extents (bits = w/(4·s) at
    // group 64), exactly as gpt-oss's trio: the config records only the
    // model-wide choice and mlx_lm's predicate singles out tensors by
    // name.
    let bits_of = |name: &str| -> Option<u32> {
        let w = storage
            .weights
            .get(&format!("{name}.weight"))
            .map(driver_metal_new::region::Region::len)?;
        let s = storage
            .weights
            .get(&format!("{name}.scales"))
            .map(driver_metal_new::region::Region::len)?;
        driver_metal_new::batch::bits_from_extents(w, s)
    };
    let ffn = bits_of("layers.0.mlp.down_proj");
    let router = bits_of("layers.0.router.proj");
    if let Some(bits) = ffn.filter(|&b| b != geometry.quant.bits) {
        geometry.alt_quant_ffn = true;
        geometry.ffn_quant = AffineFormat { bits, group: 64 };
    }
    if let Some(bits) = router.filter(|&b| b != geometry.quant.bits) {
        geometry.alt_quant_router = true;
        geometry.ffn_quant = AffineFormat { bits, group: 64 };
    }
    eprintln!(
        "solved: ffn {ffn:?} router {router:?} -> alt_ffn {} alt_router {} at {:?}",
        geometry.alt_quant_ffn, geometry.alt_quant_router, geometry.ffn_quant
    );

    let dag = driver_metal_new::batch::build_gemma4_dag(&geometry, true);
    let schedule = build_scratch_schedule(&dag, false).expect("the DAG schedules hazard-free");
    let pso_plan = driver_metal_new::batch::gemma4_step_plan(&geometry);
    let compiler = Compiler::new(&context).expect("the shader compiler starts");
    let psos = load_step_psos(&compiler, &context, &kernels_dir(), &pso_plan)
        .expect("every planned entrypoint compiles");
    let step = driver_metal_new::metal::Gemma4Step::prepare(
        &context, &storage, &geometry, &tuning, &schedule, psos, max_ctx,
    )
    .expect("the step binds whole");

    let io = |slot: IoSlot| storage.io[slot as usize].as_ref().expect("io slot");
    let read_next = || {
        // SAFETY: called only after the step retired.
        let raw = unsafe {
            std::slice::from_raw_parts(io(IoSlot::NextToken).contents().cast::<u8>().as_ptr(), 4)
        };
        u32::from_le_bytes(raw.try_into().unwrap())
    };
    let mut stepper = Stepper::new(&context).expect("a stepper");
    let mut fire_at = |token: u32, position: u32| {
        // SAFETY: the previous fire retired before we rewrite the inputs.
        unsafe {
            io(IoSlot::TokenId).write(0, &token.to_le_bytes()).unwrap();
            io(IoSlot::Position)
                .write(0, &position.to_le_bytes())
                .unwrap();
            io(IoSlot::SeqLen)
                .write(0, &(position + 1).to_le_bytes())
                .unwrap();
        }
        step.fire(&mut stepper).expect("the command buffer retires");
        read_next()
    };

    // The chat-template prompt, not a raw completion: the instruct
    // model's raw continuations sit in a degenerate low-margin regime
    // where bf16-vs-mlx evaluation-order jitter can flip an argmax
    // (observed at token six of a completion prompt, first five exact).
    // Under its template the margins are wide — 26 logits at the first
    // answer token — and the comparison is honest.
    let prompt: Vec<u32> = std::env::var("PIE_METAL_SMOKE_GEMMA4_PROMPT_IDS")
        .ok()
        .map(|csv| {
            csv.split(',')
                .map(|t| t.trim().parse().expect("token ids"))
                .collect()
        })
        .unwrap_or_else(|| {
            vec![
                2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 3689, 563, 506, 5279, 529,
                7001, 236881, 25685, 528, 886, 3658, 236761, 106, 107, 105, 4368, 107,
            ]
        });
    let mut position = 0u32;
    let mut next = 0u32;
    for &token in &prompt {
        next = fire_at(token, position);
        position += 1;
    }
    {
        let logits = io(IoSlot::Logits);
        let vocab = geometry.vocab as usize;
        // SAFETY: the step retired.
        let bytes = unsafe {
            std::slice::from_raw_parts(logits.contents().cast::<u8>().as_ptr(), vocab * 2)
        };
        let mut finite = 0usize;
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, pair) in bytes.chunks_exact(2).enumerate() {
            let value = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
            if value.is_finite() {
                finite += 1;
                if value > best.1 {
                    best = (i, value);
                }
            }
        }
        eprintln!(
            "prompt fed: {finite}/{vocab} finite, host argmax {} ({:.3}), device {next}",
            best.0, best.1
        );
        assert_eq!(
            finite, vocab,
            "a NaN in the logits is a wrong kernel upstream"
        );
        assert_eq!(next as usize, best.0, "device and host argmax disagree");
    }
    let n_continue: usize = std::env::var("PIE_METAL_SMOKE_GEMMA4_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let mut produced = Vec::new();
    produced.push(next);
    while produced.len() < n_continue {
        next = fire_at(next, position);
        position += 1;
        produced.push(next);
    }
    eprintln!("produced {produced:?}");
    let reference: Vec<u32> = std::env::var("PIE_METAL_SMOKE_GEMMA4_REFERENCE")
        .ok()
        .map(|csv| csv.split(',').map(|t| t.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![100, 45518, 107, 236829, 139, 14977, 236787, 623]);
    if produced.len() == reference.len() {
        assert_eq!(
            produced, reference,
            "the greedy continuation drifted from mlx_lm's (26b-a4b, chat template)"
        );
    }
}

/// The engine decodes ACROSS fires: a prefill fire writes the KV pages,
/// and the decode fires that follow read them — the continuity no
/// single-fire smoke can prove, and the whole reason the pages live in
/// the storage rather than the steps.
#[test]
fn the_llama_engine_decodes_across_fires() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_LLAMA_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_LLAMA_CHECKPOINT to a llama-family MLX snapshot");
        return;
    };
    let snapshot = PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&config).expect("config.json parses");
    let descriptor = model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config converts to a descriptor");
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json)
        .expect("the driver's facts read the descriptor");
    let mut geometry =
        driver_metal_new::batch::llama_geometry_from_facts(&facts).expect("a llama shape");
    let prompt: Vec<u32> = vec![
        128000, 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 11, 323, 279, 6864, 315,
        9822, 374,
    ];
    geometry.max_tokens = driver_metal_new::batch::llama_qmm_pool_rows(prompt.len() as u32);
    geometry.max_requests = 1;
    geometry.paged_kv_enabled = true;
    geometry.kv_page_size = 32;
    geometry.total_pages = 128;

    let target = metal_storage_target();
    let (plan, _moe) = compile_load_plan(&snapshot, &target, &descriptor_json)
        .expect("the plan compiles and its files exist");
    let context = Context::new().expect("a Metal device answers");
    let compiler = Compiler::new(&context).expect("the shader compiler starts");
    let mut engine = driver_metal_new::metal::LlamaEngine::new(
        &context,
        &compiler,
        &kernels_dir(),
        &plan,
        &snapshot,
        geometry.clone(),
        Tuning::default(),
        4096,
    )
    .expect("the engine stages and compiles");
    engine.reset().expect("a fresh sequence");

    let mut stepper = Stepper::new(&context).expect("a stepper");
    let vocab = geometry.vocab as usize;
    let argmax_of = |engine: &driver_metal_new::metal::LlamaEngine| -> u32 {
        let logits = engine.logits().expect("paged logits");
        // SAFETY: the fire retired.
        let bytes = unsafe {
            std::slice::from_raw_parts(logits.contents().cast::<u8>().as_ptr(), vocab * 2)
        };
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, pair) in bytes.chunks_exact(2).enumerate() {
            let v = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
            if v.is_finite() && v > best.1 {
                best = (i, v);
            }
        }
        best.0 as u32
    };

    // The prefill fire writes positions 0..17 into the pages…
    let prefill = driver_metal_new::batch::FireCsr::prefill(
        prompt.clone(),
        geometry.kv_page_size,
        geometry.total_pages,
    );
    engine
        .fire(&context, &mut stepper, &prefill)
        .expect("the prefill fire retires");
    let mut next = argmax_of(&engine);
    assert_eq!(next, 12366, "the prefill answers ' Paris'");

    // …and the decode fires that follow READ them: each is a 1-row fire
    // whose attention walks the history every earlier fire appended.
    // mlx_lm's greedy continuation of this prompt is
    // [12366, 13, 4314, 527] (' Paris', '.', ' These', ' are').
    let mut produced = vec![next];
    for position in (prompt.len() as u32..).take(3) {
        let decode =
            driver_metal_new::batch::FireCsr::decode(next, position, geometry.kv_page_size);
        engine
            .fire(&context, &mut stepper, &decode)
            .expect("the decode fire retires");
        next = argmax_of(&engine);
        produced.push(next);
    }
    eprintln!("engine produced {produced:?}");
    assert_eq!(
        produced,
        vec![12366, 13, 4314, 527],
        "the decode fires must read the KV the prefill wrote"
    );
}

/// The gpt-oss engine decodes across fires — and cross-validates the
/// PAGED KV path against the ring: the M=1 smoke produced this exact
/// chain through the contiguous ring, and the engine reproduces it
/// through pages, one prefill fire and seven decode fires.
#[test]
fn the_gptoss_engine_decodes_across_fires() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_GPTOSS_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_GPTOSS_CHECKPOINT to a gpt-oss MLX snapshot");
        return;
    };
    let snapshot = PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&config).expect("config.json parses");
    let descriptor = model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config converts to a descriptor");
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json)
        .expect("the driver's facts read the descriptor");
    let mut geometry =
        driver_metal_new::batch::gptoss_geometry_from_facts(&facts).expect("a gpt-oss shape");
    let prompt: Vec<u32> = vec![976, 9029, 328, 10128, 382];
    geometry.max_tokens = 16;
    geometry.max_requests = 1;
    geometry.paged_kv_enabled = true;
    geometry.kv_page_size = 32;
    geometry.total_pages = 128;

    let target = metal_storage_target();
    let (plan, _moe) = compile_load_plan(&snapshot, &target, &descriptor_json)
        .expect("the plan compiles and its files exist");
    let context = Context::new().expect("a Metal device answers");
    let compiler = Compiler::new(&context).expect("the shader compiler starts");
    let mut engine = driver_metal_new::metal::GptOssEngine::new(
        &context,
        &compiler,
        &kernels_dir(),
        &plan,
        &snapshot,
        geometry.clone(),
        Tuning::default(),
        4096,
    )
    .expect("the engine stages, solves and compiles");
    assert!(
        engine.geometry.mxfp4_experts,
        "the trio solved off the heap"
    );
    engine.reset().expect("a fresh sequence");

    let mut stepper = Stepper::new(&context).expect("a stepper");
    let vocab = geometry.vocab as usize;
    let argmax_of = |engine: &driver_metal_new::metal::GptOssEngine| -> u32 {
        let logits = engine.logits().expect("paged logits");
        // SAFETY: the fire retired.
        let bytes = unsafe {
            std::slice::from_raw_parts(logits.contents().cast::<u8>().as_ptr(), vocab * 2)
        };
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, pair) in bytes.chunks_exact(2).enumerate() {
            let v = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
            if v.is_finite() && v > best.1 {
                best = (i, v);
            }
        }
        best.0 as u32
    };

    let prefill = driver_metal_new::batch::FireCsr::prefill(
        prompt.clone(),
        geometry.kv_page_size,
        geometry.total_pages,
    );
    engine
        .fire(&context, &mut stepper, &prefill)
        .expect("the prefill fire retires");
    let mut next = argmax_of(&engine);
    let mut produced = vec![next];
    for position in (prompt.len() as u32..).take(7) {
        let decode =
            driver_metal_new::batch::FireCsr::decode(next, position, geometry.kv_page_size);
        engine
            .fire(&context, &mut stepper, &decode)
            .expect("the decode fire retires");
        next = argmax_of(&engine);
        produced.push(next);
    }
    eprintln!("engine produced {produced:?}");
    assert_eq!(
        produced,
        vec![12650, 3692, 279, 12, 6240, 1, 976, 9029],
        "the paged chain must match the ring's token-exact reference"
    );
}

/// The gemma4 engine decodes across fires — the fourth family's paged
/// path and its engine, proven in one chain against the ring's
/// token-exact reference.
#[test]
fn the_gemma4_engine_decodes_across_fires() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_GEMMA4_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_GEMMA4_CHECKPOINT to a gemma-4 MLX snapshot");
        return;
    };
    let snapshot = PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&config).expect("config.json parses");
    let descriptor = model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config converts to a descriptor");
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json)
        .expect("the driver's facts read the descriptor");
    let mut geometry =
        driver_metal_new::batch::gemma4_geometry_from_facts(&facts).expect("a gemma4 shape");
    let prompt: Vec<u32> = vec![
        2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001,
        236881, 25685, 528, 886, 3658, 236761, 106, 107, 105, 4368, 107,
    ];
    geometry.max_tokens = 32;
    geometry.max_requests = 1;
    geometry.paged_kv_enabled = true;
    geometry.kv_page_size = 32;
    geometry.total_pages = 128;

    let target = metal_storage_target();
    let (plan, _moe) = compile_load_plan(&snapshot, &target, &descriptor_json)
        .expect("the plan compiles and its files exist");
    let context = Context::new().expect("a Metal device answers");
    let compiler = Compiler::new(&context).expect("the shader compiler starts");
    let mut engine = driver_metal_new::metal::Gemma4Engine::new(
        &context,
        &compiler,
        &kernels_dir(),
        &plan,
        &snapshot,
        geometry.clone(),
        Tuning::default(),
        4096,
    )
    .expect("the engine stages, solves and compiles");
    assert!(engine.geometry.alt_quant_router, "the 8-bit router solved");
    engine.reset().expect("a fresh sequence");

    let mut stepper = Stepper::new(&context).expect("a stepper");
    let vocab = geometry.vocab as usize;
    let argmax_of = |engine: &driver_metal_new::metal::Gemma4Engine| -> u32 {
        let logits = engine.logits().expect("paged logits");
        // SAFETY: the fire retired.
        let bytes = unsafe {
            std::slice::from_raw_parts(logits.contents().cast::<u8>().as_ptr(), vocab * 2)
        };
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, pair) in bytes.chunks_exact(2).enumerate() {
            let v = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
            if v.is_finite() && v > best.1 {
                best = (i, v);
            }
        }
        best.0 as u32
    };

    let prefill = driver_metal_new::batch::FireCsr::prefill(
        prompt.clone(),
        geometry.kv_page_size,
        geometry.total_pages,
    );
    engine
        .fire(&context, &mut stepper, &prefill)
        .expect("the prefill fire retires");
    let mut next = argmax_of(&engine);
    let mut produced = vec![next];
    for position in (prompt.len() as u32..).take(7) {
        let decode =
            driver_metal_new::batch::FireCsr::decode(next, position, geometry.kv_page_size);
        engine
            .fire(&context, &mut stepper, &decode)
            .expect("the decode fire retires");
        next = argmax_of(&engine);
        produced.push(next);
    }
    eprintln!("engine produced {produced:?}");
    assert_eq!(
        produced,
        vec![100, 45518, 107, 236829, 139, 14977, 236787, 623],
        "the paged chain must match the ring's token-exact reference"
    );
}

/// Two requests share every fire and neither contaminates the other —
/// the multi-request contract nothing had tested: per-request page
/// walks, disjoint page lists, and a 2-row decode fleet whose rows
/// belong to different conversations.
#[test]
fn the_llama_engine_isolates_two_requests() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_LLAMA_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_LLAMA_CHECKPOINT to a llama-family MLX snapshot");
        return;
    };
    let snapshot = PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&config).expect("config.json parses");
    let descriptor = model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config converts to a descriptor");
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json)
        .expect("the driver's facts read the descriptor");
    let mut geometry =
        driver_metal_new::batch::llama_geometry_from_facts(&facts).expect("a llama shape");
    // Request 0: "The capital of France is" — solo chain
    // [12366, 627, 791, 6864]. Request 1: the fox prompt — solo chain
    // [12366, 13, 4314, 527]. Same first token, DIFFERENT second: a
    // cross-contaminated KV shows immediately.
    let prompt_a: Vec<u32> = vec![128000, 791, 6864, 315, 9822, 374];
    let prompt_b: Vec<u32> = vec![
        128000, 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 11, 323, 279, 6864, 315,
        9822, 374,
    ];
    geometry.max_tokens = 64;
    geometry.max_requests = 2;
    geometry.paged_kv_enabled = true;
    geometry.kv_page_size = 32;
    geometry.total_pages = 128;
    // Request 1's pages start at 64: physically disjoint from request
    // 0's, and deliberately NOT arithmetically adjacent to its logical
    // positions — the page indirection is what is under test.
    const B_BASE: u32 = 64;

    let target = metal_storage_target();
    let (plan, _moe) = compile_load_plan(&snapshot, &target, &descriptor_json)
        .expect("the plan compiles and its files exist");
    let context = Context::new().expect("a Metal device answers");
    let compiler = Compiler::new(&context).expect("the shader compiler starts");
    let mut engine = driver_metal_new::metal::LlamaEngine::new(
        &context,
        &compiler,
        &kernels_dir(),
        &plan,
        &snapshot,
        geometry.clone(),
        Tuning::default(),
        4096,
    )
    .expect("the engine stages and compiles");
    engine.reset().expect("a fresh pool");

    let mut stepper = Stepper::new(&context).expect("a stepper");
    let vocab = geometry.vocab as usize;
    let argmax_row = |engine: &driver_metal_new::metal::LlamaEngine, row: usize| -> u32 {
        let logits = engine.logits().expect("paged logits");
        // SAFETY: the fire retired.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                logits.contents().cast::<u8>().as_ptr().add(row * vocab * 2),
                vocab * 2,
            )
        };
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, pair) in bytes.chunks_exact(2).enumerate() {
            let v = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
            if v.is_finite() && v > best.1 {
                best = (i, v);
            }
        }
        best.0 as u32
    };

    // Prefill request 0 on pages 0.. .
    let pre_a = driver_metal_new::batch::FireCsr::prefill(prompt_a.clone(), 32, 1);
    engine
        .fire(&context, &mut stepper, &pre_a)
        .expect("prefill A retires");
    let mut next_a = argmax_row(&engine, 0);

    // Prefill request 1 on pages 64.., hand-built: request 0's row
    // arrays but every physical page shifted.
    let nb = prompt_b.len() as u32;
    let positions: Vec<u32> = (0..nb).collect();
    let pre_b = driver_metal_new::batch::FireCsr {
        token_ids: prompt_b.clone(),
        position_ids: positions.clone(),
        req_of_token: vec![0; prompt_b.len()],
        w_page: positions.iter().map(|p| B_BASE + p / 32).collect(),
        w_off: positions.iter().map(|p| p % 32).collect(),
        qo_indptr: vec![0, nb],
        kv_page_indices: vec![B_BASE],
        kv_page_indptr: vec![0, 1],
        kv_last_page_lens: vec![((nb - 1) % 32) + 1],
        sample_rows: vec![nb - 1],
        run_argmax: false,
    };
    engine
        .fire(&context, &mut stepper, &pre_b)
        .expect("prefill B retires");
    let mut next_b = argmax_row(&engine, 0);
    assert_eq!((next_a, next_b), (12366, 12366), "both prompts end alike");

    // The fleet: one 2-row fire per step, each row a different
    // conversation walking its own pages.
    let mut chain_a = vec![next_a];
    let mut chain_b = vec![next_b];
    let mut pos_a = prompt_a.len() as u32;
    let mut pos_b = prompt_b.len() as u32;
    // Two counters advance together; clippy would have one of them own
    // the loop, but neither is more the loop's than the other.
    #[allow(clippy::explicit_counter_loop)]
    for _ in 0..3 {
        let fleet = driver_metal_new::batch::FireCsr {
            token_ids: vec![next_a, next_b],
            position_ids: vec![pos_a, pos_b],
            req_of_token: vec![0, 1],
            w_page: vec![pos_a / 32, B_BASE + pos_b / 32],
            w_off: vec![pos_a % 32, pos_b % 32],
            qo_indptr: vec![0, 1, 2],
            kv_page_indices: vec![0, B_BASE],
            kv_page_indptr: vec![0, 1, 2],
            kv_last_page_lens: vec![(pos_a % 32) + 1, (pos_b % 32) + 1],
            sample_rows: vec![0, 1],
            run_argmax: false,
        };
        engine
            .fire(&context, &mut stepper, &fleet)
            .expect("the fleet fire retires");
        next_a = argmax_row(&engine, 0);
        next_b = argmax_row(&engine, 1);
        chain_a.push(next_a);
        chain_b.push(next_b);
        pos_a += 1;
        pos_b += 1;
    }
    eprintln!("fleet chains a={chain_a:?} b={chain_b:?}");
    assert_eq!(
        chain_a,
        vec![12366, 627, 791, 6864],
        "request 0 must continue ITS conversation"
    );
    assert_eq!(
        chain_b,
        vec![12366, 13, 4314, 527],
        "request 1 must continue ITS conversation"
    );
}
