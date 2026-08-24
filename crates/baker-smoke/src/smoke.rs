//! The executor itself. `main.rs` is the two-arm shell; everything the
//! fire does lives here, and the module is gated on `_cuda` so a build
//! with no runtime version selected still WALKS -- `driver-cuda`'s
//! posture (`driver-cuda/src/lib.rs:19-21`: no `compile_error!` for a
//! featureless build, because nothing links `cudarc` and a consumer that
//! forgets is caught by an unresolved path), which is what keeps
//! `cargo check --workspace` able to sweep this member.


use core::ffi::c_void;
use std::collections::BTreeMap;

use kernels::bound::{Axis, Rides, Site};
use kernels::points::Form;
use kernels::raises::Struct;
use kernels::routine::{Cache, Const, In, InOut, Out, Refusal};
use kernels_cuda::attn::fa2::plan::{DecodePlanCache, Planned};
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::Bank as CudaBank;
use kernels_cuda::jit::abi::Planes;
use kernels_cuda::jit::abi::Tensor;
use kernels_cuda::views::{KvCache, PagedKvView, RecurrentState, RecurrentView};
use crate::marks::{Rect, rin, rio, rout, wconst};
use model::produce::Dtype;
use model::snapshot::Snapshot;
use model_compiler::program::{Call, Dt, Program, Slot};
use model_ir::plan::{CacheRow, Op, Plan, ValueId};

use crate::dev;
use crate::dev::Slab;

/// The KV page size, in tokens. NOT A KNOB, and copied rather than derived
/// for the reason `driver-cuda/src/boot.rs:23-25` gives: the paged-attention
/// kernels are compiled for 16, and the fire path restates the same 16 in
/// four places because it is a coupling and not a preference.
const PAGE_SIZE: i32 = 16;

/// The fa2 planner's two carves. `driver-cuda/src/fire/launch.rs:1494-1510`
/// takes `32 << 20` float and `16 << 20` int for a whole deployment; one
/// request needs a rounding error of that, and oversizing a slab that lives
/// for the process is cheaper than reasoning about the planner's carve.
const ATTN_FLOAT_BYTES: usize = 32 << 20;
const ATTN_INT_BYTES: usize = 16 << 20;

pub fn main() {
    match run() {
        Ok(()) => {}
        Err(e) => {
            eprintln!("\nREFUSED: {e}");
            std::process::exit(1);
        }
    }
}

struct Args {
    sku: String,
    cache: String,
    base: String,
    /// The tokens to consume, one FIRE EACH.
    ///
    /// This lane is the decode lane and nothing else: `qo_one` is what
    /// selected it, so every fire carries exactly one row. A prompt of seven
    /// is therefore seven fires that share the caches -- which is what
    /// autoregressive decoding IS, and what makes the last fire's logits
    /// comparable to a seven-token prefill's last row.
    prompt: Vec<i32>,
    top: usize,
    stop: Option<usize>,
    probes: Vec<usize>,
    trace: bool,
}

fn args() -> Args {
    let mut a = Args {
        sku: "qwen35-d0.8b-bf16-kv-bf16".to_string(),
        cache: "models--Qwen--Qwen3.5-0.8B".to_string(),
        base: "safetensors-bf16".to_string(),
        // A real id off `tokenizer.json`, not a magic constant: `--token`
        // and `--prompt` override it and the run prints the strings back.
        prompt: vec![785],
        top: 5,
        stop: None,
        probes: Vec::new(),
        trace: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut next = |what: &str| it.next().unwrap_or_else(|| panic!("{arg} wants {what}"));
        match arg.as_str() {
            "--sku" => a.sku = next("a catalog row"),
            "--cache" => a.cache = next("an hf cache dir"),
            "--base" => a.base = next("a checkpoint flavor"),
            "--token" => a.prompt = vec![next("a token id").parse().expect("a token id")],
            "--prompt" => {
                a.prompt = next("comma-separated token ids")
                    .split(',')
                    .map(|t| t.trim().parse().expect("a token id"))
                    .collect();
            }
            "--top" => a.top = next("a count").parse().expect("a count"),
            "--stop" => a.stop = Some(next("a step count").parse().expect("a step count")),
            "--probe" => a
                .probes
                .push(next("an op index").parse().expect("an op index")),
            "--trace" => a.trace = true,
            other => panic!("unknown flag `{other}`"),
        }
    }
    a
}

#[allow(clippy::too_many_lines)]
fn run() -> Result<(), String> {
    let args = args();

    // ── 1. The device. ──────────────────────────────────────────────────
    dev::bind(0)?;
    let stream = dev::stream()?;
    let cublas = dev::cublas(stream)?;
    // SAFETY: `stream` and `cublas` are live for the rest of `run`, and the
    // handle was bound to that stream three lines up -- which is exactly
    // what `Ctx::with_cublas` asks of its caller.
    let ctx = unsafe { Ctx::on(stream).with_cublas(cublas) };

    // ── 2. The plan, and the lane a one-token fire runs. ────────────────
    let trace = model::trace_of(&args.sku).ok_or_else(|| format!("`{}` is not a catalog row", args.sku))?;
    let plan = trace(model_dsl::Plane::Cuda);
    println!(
        "plan `{}` on {:?}\n  {} facts {:?}, {} params, {} caches, {} ops",
        plan.name,
        plan.plane,
        plan.facts.len(),
        plan.facts,
        plan.params.len(),
        plan.caches.len(),
        plan.ops.len()
    );

    let word = decode_word(&plan);
    let lanes = model_compiler::program::bound(&plan);
    let (at, program) = pick(&lanes, word)?;
    println!(
        "  decode word {word:#b} -> lane {at}: {} steps, row_pitch {} B, {} slots",
        program.steps.len(),
        program.row_pitch,
        program.slots.len()
    );
    assert!(
        program.words.contains(&word),
        "the chosen lane does not serve the one-token word"
    );

    // ── 3. The weights. ─────────────────────────────────────────────────
    let import = model::import_of(&args.sku, &args.base)
        .ok_or_else(|| format!("`{}` names no `{}` import", args.sku, args.base))?;
    let snap = Snapshot::open(&args.cache)
        .ok_or_else(|| format!("no safetensors snapshot under ~/.cache/huggingface/hub/{}", args.cache))?;
    println!(
        "  checkpoint {} ({} shard(s), {} tensors)",
        snap.dir.display(),
        snap.shards(),
        snap.len()
    );
    let t0 = std::time::Instant::now();
    let produced = model::produce::produce(&import, &|n| snap.read(n))
        .map_err(|e| format!("production refused: {e}"))?;
    let host_bytes: usize = produced.iter().map(|(_, t)| t.bytes.len()).sum();

    let mut banks: BTreeMap<String, Bank> = BTreeMap::new();
    let mut slabs: Vec<Slab> = Vec::new();
    for (name, t) in &produced {
        // THE UPLOAD HAS NO DECISION IN IT, which is what `baker_load.rs`
        // says and what makes this loop three lines: every produced tensor
        // is dense, row-major and canonical, so one contiguous H2D copy per
        // row and no restride, no repack, no cast.
        let slab = Slab::of(&t.bytes, stream)?;
        banks.insert(
            name.clone(),
            Bank {
                ptr: slab.ptr(),
                shape: t.shape.clone(),
                dtype: t.dtype,
                // The demand side's own column, carried across by name. A
                // produced row the plan binds no param for keeps an empty
                // repr, which is a refusal at a bank slot and no statement
                // can name it anyway.
                repr: plan
                    .params
                    .iter()
                    .find(|p| &p.name == name)
                    .map(|p| p.repr.clone())
                    .unwrap_or_default(),
            },
        );
        slabs.push(slab);
    }
    dev::sync(stream)?;
    println!(
        "  produced+uploaded {} tensors, {} in {:.1}s",
        produced.len(),
        human(host_bytes as u64),
        t0.elapsed().as_secs_f64()
    );

    // The join `baker_load` proves, restated here as a precondition: a
    // missing bank would be a null pointer at a `Const` slot and a fault
    // inside a kernel, which is the worst place to find out.
    let mut missing = Vec::new();
    for p in &plan.params {
        match banks.get(&p.name) {
            None => missing.push(p.name.clone()),
            Some(b) if b.shape != p.shape => missing.push(format!(
                "{} (plan wants {:?}, import produced {:?})",
                p.name, p.shape, b.shape
            )),
            Some(_) => {}
        }
    }
    if !missing.is_empty() {
        return Err(format!("{} param(s) the import does not satisfy: {missing:?}", missing.len()));
    }
    drop(produced);

    // ── 4. The arena, the caches, the runtime planes. ───────────────────
    let rows: i32 = 1;
    let arena = Slab::zeroed(program.row_pitch as usize * rows as usize, stream)?;

    let geom = Geometry::of(&plan, &program)?;
    println!(
        "  geometry: head_dim {}, kv_heads {}, q_heads {}{}",
        geom.head_dim,
        geom.kv_heads,
        geom.q_heads,
        // Printed only when the plan has one, for `lanes`' reason: a line
        // that said "conv 0x0" on every dense SKU would move that SKU's
        // output to report a number that is always the same absence.
        geom.recurrent.map_or(String::new(), |r| format!(
            ", k/v heads {}/{}, k/v dim {}/{}, conv {}x{}",
            r.k_h, r.v_h, r.k_d, r.v_d, r.conv_k, r.conv_dim
        )),
    );

    // Enough pages for the whole prompt, taken once: this smoke never
    // recycles a page, so `ceil(tokens / page_size)` is the whole pool.
    let tokens = i32::try_from(args.prompt.len()).map_err(|_| "a prompt longer than i32")?;
    let pages = (tokens + PAGE_SIZE - 1) / PAGE_SIZE;

    // The three runtime planes a one-row decode stages. `token_ids` and
    // `positions` are the fire's own data and are rewritten per fire;
    // `qo_indptr` is the request CSR (`[0, 1]`: one request, one token row)
    // and never moves. STAGED BEFORE THE POOLS, because a pool view carries
    // the fire's CSR and row validity (`PagedKvView::qo_indptr`) — the
    // driver builds its views the same way round, out of an `AttnCtx` the
    // fire assembled first.
    let ids = Slab::zeroed(4, stream)?;
    let positions = Slab::zeroed(4, stream)?;
    let qo_indptr = Slab::of(&u32s(&[0, 1]), stream)?;
    // One BYTE per row, all ones. `driver-cuda/src/fire/scratch.rs:261-276`
    // memsets exactly this; the routine declares it `In<Tensor<i32>>` and
    // casts to `*const u8` (`kernels-cuda/src/attn/mod.rs:2420`), so the
    // declared element is a fiction the DECLARATION carries and the buffer
    // must not.
    let row_valid = Slab::of(&[1u8], stream)?;

    let mut pools = Pools::new();
    pools.build(&plan, &geom, pages, qo_indptr.ptr(), row_valid.ptr(), stream)?;
    println!(
        "  caches: {} kv row(s) at {pages} x {PAGE_SIZE}-token page(s), {} state slab(s)",
        pools.kv.len(),
        pools.st.len()
    );

    let mut runtime: BTreeMap<String, Rect> = BTreeMap::new();
    runtime.insert("token_ids".into(), Rect { ptr: ids.ptr(), rows, width: 1, dt: Dt::I32 });
    runtime.insert("positions".into(), Rect { ptr: positions.ptr(), rows, width: 1, dt: Dt::I32 });
    // ROWS IS THE REQUEST COUNT on this one, not the buffer's length: the
    // appender reads `num_requests = qo_indptr.rows`
    // (`kernels-cuda/src/attn/mod.rs:2415`), which is what
    // `driver-cuda/src/bind/mod.rs:2206` puts there from `lowered.arg_rows`.
    runtime.insert("qo_indptr".into(), Rect { ptr: qo_indptr.ptr(), rows: 1, width: 2, dt: Dt::I32 });

    // ── 5. The fa2 workspaces. ─────────────────────────────────────────
    //
    // The gdn seam used to want a slab here too — three f32 planes
    // `ssm.gdn_prep`'s routine wrote and its statement never stated. Both
    // recurrence points are claim bodies now and stage those out of
    // `Ctx::scratch` beside every other plane they need (W10), so this
    // executor allocates nothing for them.
    let attn_float = Slab::zeroed(ATTN_FLOAT_BYTES, stream)?;
    let attn_int = Slab::zeroed(ATTN_INT_BYTES, stream)?;
    let mut decode_plan = Box::new(DecodePlanCache::new());

    let out = plan
        .seams
        .iter()
        .find(|s| s.seam == model_ir::seam::OUT.name)
        .and_then(|s| s.values.first().copied())
        .ok_or("the plan states no `out` seam")?;
    let vocab = vocabulary(&snap);

    // ── 6. The walk, once per prompt token. ─────────────────────────────
    let total = args.stop.unwrap_or(program.steps.len()).min(program.steps.len());
    println!(
        "\nfiring {total} of {} steps, {} time(s)",
        program.steps.len(),
        args.prompt.len()
    );
    let t0 = std::time::Instant::now();
    let mut last: Option<Vec<f32>> = None;
    for (position, token) in args.prompt.iter().copied().enumerate() {
        let position = i32::try_from(position).map_err(|_| "a position wider than i32")?;
        dev::upload(ids.ptr(), &token.to_le_bytes(), stream)?;
        dev::upload(positions.ptr(), &position.to_le_bytes(), stream)?;
        // The cache holds everything before this row, plus this row.
        let held = pools.hold(position + 1, stream)?;
        // REPLANNED PER FIRE, because the schedule bakes the page count:
        // `plan_decode` stamps `num_pages_in_batch` off the CSR it is
        // handed, and a fire that owns a second page is a different
        // schedule. The driver replans every fire for the same reason
        // (`driver-cuda/src/fire/launch.rs:1571-1666`).
        plan_decode(&mut decode_plan, &geom, held, &attn_float, &attn_int)?;
        if position == 0 {
            println!(
                "  fa2 decode plan: {} request(s), {} q / {} kv heads at {}, page {}, split_kv {}, padded batch {}",
                decode_plan.num_requests,
                decode_plan.num_q_heads,
                decode_plan.num_kv_heads,
                decode_plan.head_dim,
                decode_plan.page_size,
                decode_plan.plan_info.split_kv,
                decode_plan.plan_info.padded_batch_size
            );
        }

        let mut fire = Fire {
            plan: &plan,
            program,
            ctx: &ctx,
            stream,
            arena: arena.ptr(),
            rows,
            banks: &banks,
            runtime: &runtime,
            pools: &pools,
            geom: &geom,
            decode_plan: &*decode_plan,
            first_token: 0,
            qo_indptr: qo_indptr.ptr(),
            row_valid: row_valid.ptr(),
        };
        for (i, step) in program.steps.iter().take(total).enumerate() {
            let op = &plan.ops[step.op as usize];
            fire.step(step.op, &step.call).map_err(|e| {
                format!(
                    "token {position} step {i} (op {}, `{}`, layer {:?}) -> {e:?}",
                    step.op, op.kernel, op.layer
                )
            })?;
            if args.trace || args.probes.contains(&(step.op as usize)) {
                dev::sync(stream)?;
                if let Some(v) = op.outputs.first() {
                    let r = fire.rect(*v);
                    println!(
                        "  t{position} [{i:>3}] op {:>3} {:<28} {}",
                        step.op,
                        op.kernel,
                        digest(&r, stream)?
                    );
                } else {
                    println!(
                        "  t{position} [{i:>3}] op {:>3} {:<28} (an effect: no rectangle)",
                        step.op, op.kernel
                    );
                }
            }
        }
        dev::sync(stream)?;
        if total < program.steps.len() {
            println!("  t{position}: STOPPED EARLY at {total} steps -- no logits to read");
            continue;
        }
        let logits = fire.rect(out);
        let host = read_f32(&logits, stream)?;
        let finite = host.iter().filter(|v| v.is_finite()).count();
        if finite != host.len() {
            return Err(format!(
                "token {position}: {} non-finite logit(s)",
                host.len() - finite
            ));
        }
        let top = host
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("a non-empty vocabulary");
        println!(
            "  t{position} token {token:>6} {:<14} -> argmax {:>6} {:<14} at {:>8.4}",
            show(&vocab, token as usize),
            top.0,
            show(&vocab, top.0),
            top.1
        );
        last = Some(host);
    }
    println!(
        "  {} fire(s) x {total} steps retired in {:.3}s",
        args.prompt.len(),
        t0.elapsed().as_secs_f64()
    );

    // ── 7. The logits. ──────────────────────────────────────────────────
    let Some(host) = last else {
        println!("\nSTOPPED EARLY -- no logits to read");
        return Ok(());
    };
    let logits = Rect {
        ptr: core::ptr::null_mut(),
        rows,
        width: i32::try_from(host.len()).unwrap_or(i32::MAX),
        dt: Dt::Bf16,
    };
    println!(
        "\nlogits: value {out}, {} x {} x {:?} ({})",
        logits.rows,
        logits.width,
        logits.dt,
        human(logits.bytes() as u64)
    );

    let mut ranked: Vec<(usize, f32)> = host.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("\ntop-{} after {:?}:", args.top, args.prompt);
    for (rank, (id, v)) in ranked.iter().take(args.top).enumerate() {
        println!("  {rank}. {id:>7} {v:>10.4}  {}", show(&vocab, *id));
    }
    println!(
        "\nARGMAX {} ({}) at {:.4}",
        ranked[0].0,
        show(&vocab, ranked[0].0),
        ranked[0].1
    );
    if let Some(probe) = std::env::var_os("BAKER_PROBE_IDS") {
        let ids: Vec<usize> = probe
            .to_string_lossy()
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect();
        println!("probes:");
        for id in ids {
            println!("  {id:>7} {:>10.4}  {}", host[id], show(&vocab, id));
        }
    }
    Ok(())
}

// ── The plan side ───────────────────────────────────────────────────────

/// The fact word a ONE-TOKEN fire sets, computed off `plan.facts` rather
/// than assumed: bit `i` is `plan.facts[i]`, and `qo_one` is the only fact
/// a qwen trace declares. A SKU that declared more would set the ones a
/// one-token query implies and leave the rest clear, which is what this
/// match says and why an unknown fact is a refusal instead of a zero.
fn decode_word(plan: &Plan) -> u64 {
    let mut word = 0u64;
    for (bit, fact) in plan.facts.iter().enumerate() {
        let holds = match fact.as_str() {
            "qo_one" => true,
            other => panic!(
                "`{other}` is a fact this smoke does not know how to answer \
                 for a one-token fire; name it here or the lane is a guess"
            ),
        };
        if holds {
            word |= 1 << bit;
        }
    }
    word
}

fn pick(
    lanes: &[Result<Program, model_compiler::program::Refusal>],
    word: u64,
) -> Result<(usize, &Program), String> {
    let mut refused = Vec::new();
    for (at, lane) in lanes.iter().enumerate() {
        match lane {
            Ok(p) if p.words.contains(&word) => return Ok((at, p)),
            Ok(_) => {}
            Err(r) => {
                if r.words.contains(&word) {
                    return Err(format!("the one-token lane refuses: {r}"));
                }
                refused.push(r.to_string());
            }
        }
    }
    Err(format!(
        "no bound lane serves word {word:#b}; the refusing lanes said {refused:?}"
    ))
}

/// A weight on the device: an address, its shape, and the element the
/// CHECKPOINT stores it at.
///
/// The plan's `repr` column is not this. A model text declares qwen's
/// `a_log` and its gdn norm at the activation dtype and the checkpoint ships
/// both F32; `produce` reports the storage and `baker_load` prints the
/// disagreement as a note. The routines agree with the CHECKPOINT --
/// `qwen_gdn_post_conv_prep_bf16` declares `a_log: Const<Tensor<f32>>` --
/// so this table carries the storage dtype and the shim asserts against it.
struct Bank {
    ptr: *mut c_void,
    shape: Vec<u64>,
    dtype: Dtype,
    /// The plan's own `repr` column for this parameter, which the storage
    /// dtype above cannot stand in for and does not try to.
    ///
    /// A QUANTISED BANK'S FORM LIVES ONLY HERE. `mxfp4` codes and `e8m0`
    /// block exponents are both `U8` on disk, so `dtype` tells the two planes
    /// of one bank apart in neither direction; what says which is which is
    /// the name the model text declared them under and the repr it declared
    /// them at. `BoundOp::form` reads this and nothing else.
    repr: String,
}

/// The numbers the claim-only routines want and the statements do not carry.
///
/// EVERY ONE IS READ OFF THE PLAN, never off a config file: `head_dim` and
/// the four gdn head numbers are statement params, `kv_heads` divides the
/// cache row by `head_dim`, `q_heads` divides the decode statement's own
/// operand, and `conv_dim` is the conv statement's operand width. A number
/// this could not find is a refusal with the point named.
///
/// THE RECURRENT HALF IS OPTIONAL AND THE ATTENTION HALF IS NOT. Every SKU
/// this binary can fire has a paged attention; only the HYBRIDS have a
/// gated-delta mixer beside it, and a plan with no `ssm.gated_delta` used to
/// be refused here before a single step ran — which is what stood between
/// gpt-oss and a fire long after its last point was claimed. A number no
/// statement in this plan wants is not a missing measurement; it is a
/// question this plan does not ask, and `None` is the answer. The staging
/// arms that read one say so.
struct Geometry {
    head_dim: i32,
    kv_heads: i32,
    q_heads: i32,
    recurrent: Option<Recurrent>,
}

/// The gated-delta numbers, present exactly when the plan states a mixer.
#[derive(Clone, Copy)]
struct Recurrent {
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    conv_k: i32,
    conv_dim: i32,
}

impl Geometry {
    fn of(plan: &Plan, program: &Program) -> Result<Geometry, String> {
        let find = |kernel: &str| plan.ops.iter().find(|o| o.kernel == kernel);
        let width_of = |what: &str, id: u32| match program.slots[id as usize] {
            Slot::Arena { width, .. } => Ok(width),
            ref other => Err(format!("`{what}` lives at {other:?}")),
        };
        // THE DECODE STATEMENT, either spelling. A text that merges an
        // attention leg with its own sink states `decode_lse` — the same
        // reading with the per-row log-sum-exp kept — and its `head_dim` and
        // `q` sit at the same slots. gpt-oss states only that one.
        let decode = find("attention.decode")
            .or_else(|| find("attention.decode_lse"))
            .ok_or("the plan states no `attention.decode` and no `attention.decode_lse`")?;
        let head_dim = i32::try_from(decode.params[1]).map_err(|_| "a wide head_dim")?;
        // The decode statement's operand is the roped `q`, whose width is
        // `q_heads * head_dim`. `program.slots` is where that width lives.
        let q_width = width_of("the decode statement's q", decode.inputs[0])?;
        let kv_row = plan
            .caches
            .iter()
            .find_map(|c| match c {
                CacheRow::Kv { row, .. } => Some(row.clone()),
                CacheRow::State { .. } => None,
            })
            .ok_or("the plan declares no kv cache row")?;
        // `[2, kv_heads * head_dim]`: the k/v pair, then the plane's width.
        let kv_width = i32::try_from(kv_row[1]).map_err(|_| "a wide kv row")?;

        // BOTH OR NEITHER: a plan with a gated-delta step and no conv, or the
        // other way round, is a plan this executor would stage half of.
        let recurrent = match (find("ssm.gated_delta"), find("ssm.causal_conv1d")) {
            (Some(gd), Some(conv)) => Some(Recurrent {
                k_h: gd.params[0] as i32,
                v_h: gd.params[1] as i32,
                k_d: gd.params[2] as i32,
                v_d: gd.params[3] as i32,
                conv_k: conv.params[0] as i32,
                conv_dim: i32::try_from(width_of("`ssm.causal_conv1d`'s x", conv.inputs[0])?)
                    .map_err(|_| "a wide conv")?,
            }),
            (None, None) => None,
            _ => {
                return Err(
                    "the plan states one half of a gated-delta mixer and not the other".into(),
                );
            }
        };
        Ok(Geometry {
            head_dim,
            kv_heads: kv_width / head_dim,
            q_heads: i32::try_from(q_width).map_err(|_| "a wide q")? / head_dim,
            recurrent,
        })
    }
}

// ── The pools ───────────────────────────────────────────────────────────

/// One paged KV row and one recurrent slab pair per cache the plan
/// declares, sized for ONE request.
///
/// The views live on the HOST and stay put for the whole fire: a routine
/// taking `In<Struct<KvCache>>` dereferences the pointer on the host
/// (`kernels-cuda/src/attn/mod.rs:2406`) and reads the fields itself, so
/// what crosses to the device is the fields, never the struct.
struct Pools {
    kv: BTreeMap<String, PagedKvView>,
    st: BTreeMap<String, RecurrentView>,
    /// The two device arrays that move between fires: how many pages the
    /// request owns, and how full the last one is.
    page_indptr: *mut c_void,
    last_page_lens: *mut c_void,
    slabs: Vec<Slab>,
}

impl Pools {
    fn new() -> Pools {
        Pools {
            kv: BTreeMap::new(),
            st: BTreeMap::new(),
            page_indptr: core::ptr::null_mut(),
            last_page_lens: core::ptr::null_mut(),
            slabs: Vec::new(),
        }
    }

    /// The page table AFTER a fire that leaves `tokens` in the cache.
    ///
    /// `write_kv_to_pages` reads the append position out of exactly this:
    /// `total_kv_after = (pages - 1) * page_size + last_page_lens`, less the
    /// fire's own `new_tokens` (`attn/kv_paged.cuh:178-187`). So the caller
    /// states where the sequence ENDS and the kernel derives where this row
    /// goes -- which is why a decode step advances the table BEFORE it
    /// fires, not after.
    fn hold(&mut self, tokens: i32, stream: *mut c_void) -> Result<i32, String> {
        let pages = (tokens + PAGE_SIZE - 1) / PAGE_SIZE;
        let last = tokens - (pages - 1) * PAGE_SIZE;
        dev::upload(self.page_indptr, &u32s(&[0, pages.unsigned_abs()]), stream)?;
        dev::upload(self.last_page_lens, &u32s(&[last.unsigned_abs()]), stream)?;
        for v in self.kv.values_mut() {
            v.max_pages_per_request = pages;
            v.pages_in_batch = pages;
        }
        Ok(pages)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        &mut self,
        plan: &Plan,
        g: &Geometry,
        pages: i32,
        qo_indptr: *mut c_void,
        row_valid: *mut c_void,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // The single request's page table, shared by every layer:
        //
        //   page_indices [0 .. pages)  -- the request's pages, in order
        //   page_indptr  [0, held]     -- the CSR: request 0 owns them all
        //   last_page_lens [n]         -- how full the last page is
        //
        // The last two move per fire and [`Pools::hold`] is what moves them.
        // At the first fire they are `[0, 1]` and `[1]`: one page, one valid
        // token, so `abs_kv_pos = 0` -- page 0, offset 0, and the decode
        // reads `kv_len = 1`, the token it just wrote. That IS a fresh
        // conversation's first step.
        let indices: Vec<u32> = (0..pages.unsigned_abs()).collect();
        let page_indices = Slab::of(&u32s(&indices), stream)?;
        let page_indptr = Slab::of(&u32s(&[0, 1]), stream)?;
        let last_page_lens = Slab::of(&u32s(&[1]), stream)?;
        let write_page = Slab::of(&u32s(&[0]), stream)?;
        let write_offset = Slab::of(&u32s(&[0]), stream)?;
        self.page_indptr = page_indptr.ptr();
        self.last_page_lens = last_page_lens.ptr();
        // One request, in slot 0. `real_hybrid.rs:494` stages the same
        // one-element device array.
        let slot_ids = Slab::of(&0i32.to_le_bytes(), stream)?;

        let page_bytes =
            pages as usize * PAGE_SIZE as usize * g.kv_heads as usize * g.head_dim as usize * 2;
        for row in &plan.caches {
            match row {
                CacheRow::Kv { name, row } => {
                    assert_eq!(row[0], 2, "a kv row that is not a k/v pair");
                    let k = Slab::zeroed(page_bytes, stream)?;
                    let v = Slab::zeroed(page_bytes, stream)?;
                    self.kv.insert(
                        name.clone(),
                        PagedKvView {
                            keys: k.ptr().cast(),
                            values: v.ptr().cast(),
                            // Native bf16: the shadow planes ARE the planes,
                            // which is what `real_hybrid.rs:461-462` stages
                            // and `kv_cache_live.rs:293-302` does live.
                            bf16_keys: k.ptr().cast(),
                            bf16_values: v.ptr().cast(),
                            page_indices: page_indices.ptr().cast(),
                            page_indptr: page_indptr.ptr().cast(),
                            last_page_lens: last_page_lens.ptr().cast(),
                            key_scales: core::ptr::null(),
                            value_scales: core::ptr::null(),
                            write_page: write_page.ptr().cast(),
                            write_offset: write_offset.ptr().cast(),
                            page_size: PAGE_SIZE,
                            // NHD, in ELEMENTS, per
                            // `driver-cuda/src/bind/views.rs:330-347`: a page
                            // is `[page_size, kv_heads, head_dim]`, so a token
                            // step crosses every head and a head is `head_dim`.
                            seq_stride: i64::from(g.kv_heads) * i64::from(g.head_dim),
                            head_stride: i64::from(g.head_dim),
                            layout: 0,
                            storage_dtype: kernels_cuda::attn::KvDType::Bf16 as i32,
                            scheme_byte: kernels_cuda::attn::KvScheme::Native as i32,
                            native_bf16: true,
                            has_envelopes: false,
                            env_min: core::ptr::null(),
                            env_max: core::ptr::null(),
                            block_size: 0,
                            // The widest single request's page count, not the
                            // batch total (`driver-cuda/src/bind/mod.rs:966`).
                            max_pages_per_request: pages,
                            pages_in_batch: pages,
                            // The fire's own CSR and row validity, on the
                            // pool row: `driver-cuda/src/bind/views.rs`
                            // fills these from `AttnCtx` for the same
                            // reason — a `#[claims]` append body names ONE
                            // cache row and resolves its destination out of
                            // it. One request, one row, always valid.
                            qo_indptr: qo_indptr.cast(),
                            row_valid: row_valid.cast(),
                            requests: 1,
                        },
                    );
                    self.slabs.push(k);
                    self.slabs.push(v);
                }
                CacheRow::State { name, slab } => {
                    // A STATE ROW IS WHAT MAKES THE MIXER NUMBERS EXIST. The
                    // plan declares these rows exactly when it states a
                    // gated-delta statement, so `Geometry` measured them;
                    // a plan that declared one without stating the other
                    // would be sized from numbers nothing had read.
                    let g = g.recurrent.ok_or_else(|| {
                        format!(
                            "`{name}` is a recurrent slab and the plan states no gated-delta \
                             mixer to size it from"
                        )
                    })?;
                    let conv_elems = g.conv_k as usize * g.conv_dim as usize;
                    // TWO SLABS PER GDN LAYER AND ONE DECLARED ROW EACH, so
                    // the pair is joined by name: `conv.{l}` and `delta.{l}`
                    // are two `CacheRow::State`s and one `RecurrentView`.
                    // A fresh conversation's state is zeros, which is what
                    // `driver-cuda/src/fire/launch.rs:1046-1047` memsets and
                    // what `real_hybrid.rs:489-490` does per layer.
                    let (elems, stride) = if name.starts_with("conv.") {
                        // THE ONE SHAPE THE PLAN AND THE KERNEL DISAGREE ON.
                        // The text declares `[conv_dim, conv_kernel - 1]` --
                        // a rolling window holds `k-1` columns -- and
                        // `causal_conv1d_update_batched` indexes
                        // `state[k * C + c]` for `k in 0..K`
                        // (`kernels/ssm/causal_conv1d.cuh:372,397-411`),
                        // which is `K * C`. This allocates the KERNEL's
                        // extent, a strict superset, so nothing reads past
                        // the slab; the declared row is the seam and it is
                        // named here rather than papered over.
                        let want = g.conv_k as usize * g.conv_dim as usize;
                        assert_eq!(
                            slab.iter().product::<u64>() as usize,
                            (g.conv_k as usize - 1) * g.conv_dim as usize,
                            "`{name}` is not the declared `[conv_dim, k-1]` window"
                        );
                        (want, conv_elems)
                    } else {
                        // `[v_heads, k_dim, v_dim]`, and the kernel agrees:
                        // `state_base + slot * slot_stride + h * K_d * V_d`
                        // (`kernels/ssm/gated_delta_net.cuh:1290-1293`).
                        let want = slab.iter().product::<u64>() as usize;
                        assert_eq!(
                            want,
                            (g.v_h * g.k_d * g.v_d) as usize,
                            "`{name}` is not the declared `[v_heads, k_dim, v_dim]` slab"
                        );
                        (want, want)
                    };
                    // bf16 on both halves: the conv window is always u16
                    // (`recurrent_layout.rs:160-163`) and the recurrence this
                    // plan resolves to is
                    // `recurrent_gated_delta_step_batched_gqa_state_bf16`,
                    // whose kernel takes `__nv_bfloat16* state_base`
                    // (`gated_delta_net.cuh:1266`).
                    let s = Slab::zeroed(elems * 2, stream)?;
                    let view = if name.starts_with("conv.") {
                        RecurrentView {
                            slab: core::ptr::null_mut(),
                            slot_ids: slot_ids.ptr().cast(),
                            slot_stride_elems: 0,
                            slots: slot_ids.ptr().cast(),
                            state: core::ptr::null_mut(),
                            conv_state: s.ptr(),
                            new_conv_state: core::ptr::null_mut(),
                            conv_slab: s.ptr(),
                            conv_stride: stride as i64,
                        }
                    } else {
                        RecurrentView {
                            slab: s.ptr(),
                            slot_ids: slot_ids.ptr().cast(),
                            slot_stride_elems: stride as i64,
                            slots: slot_ids.ptr().cast(),
                            // `state` ALIASES `slab` on cuda and the swap
                            // plane stays null -- the double-buffered
                            // spelling is the shader planes'
                            // (`driver-cuda/src/bind/views.rs:395-402`).
                            state: s.ptr(),
                            conv_state: core::ptr::null_mut(),
                            new_conv_state: core::ptr::null_mut(),
                            conv_slab: core::ptr::null_mut(),
                            conv_stride: 0,
                        }
                    };
                    self.st.insert(name.clone(), view);
                    self.slabs.push(s);
                }
            }
        }
        self.slabs.push(page_indices);
        self.slabs.push(page_indptr);
        self.slabs.push(last_page_lens);
        self.slabs.push(write_page);
        self.slabs.push(write_offset);
        self.slabs.push(slot_ids);
        Ok(())
    }
}

/// Plan the decode schedule once, for the one request every attention layer
/// of this fire shares.
///
/// THE DRIVER'S OWN ARGUMENTS, and each of the three that are not obvious:
///
/// * `int_workspace` / `float_workspace` are stamped on the cache BEFORE
///   planning. `driver-cuda/src/bind/mod.rs:576-591` says why in as many
///   words: the dispatch reads `cache.int_workspace + int_base_bytes` to
///   upload the schedule and `cache.float_workspace` to fold a split, so an
///   unstamped cache dereferences null.
/// * `enable_cuda_graph = true`, matching `launch.rs:1573-1575` -- one
///   schedule serves every layer, so the padded batch size must not depend
///   on which layer is firing.
/// * `full_attention_variant = true` and `window_left = -1`. This text's
///   `attention.decode` states no window, and `launch.rs:1616-1626` records
///   that planning the WINDOWED schedule for a stack with no window made
///   `decode_arm` fall through to `DecodeArm::Window` and run the wrong
///   kernel, silently.
fn plan_decode(
    cache: &mut DecodePlanCache,
    g: &Geometry,
    pages: i32,
    float_ws: &Slab,
    int_ws: &Slab,
) -> Result<(), String> {
    cache.int_workspace = int_ws.ptr();
    cache.float_workspace = float_ws.ptr();
    cache.set_int_base(0);
    let device = kernels_cuda::attn::fa2::plan::plan_device();
    let max_grid = kernels_cuda::attn::fa2::plan::decode_max_grid_size(g.head_dim, g.q_heads, g.kv_heads);
    let planned = kernels_cuda::attn::fa2::plan::plan_decode(
        cache,
        &[0, pages.unsigned_abs()],
        1,
        g.q_heads,
        g.kv_heads,
        g.head_dim,
        PAGE_SIZE,
        kernels_cuda::attn::plan::Workspace::new(float_ws.bytes(), int_ws.bytes()),
        &device,
        max_grid,
        true,
        true,
        false,
        -1,
    );
    match planned {
        Planned::Full | Planned::StaticNonsplit => Ok(()),
        Planned::Declined(why) => Err(format!("the fa2 decode planner declined: {why}")),
    }
}

// ── The fire ────────────────────────────────────────────────────────────

struct Fire<'a> {
    plan: &'a Plan,
    program: &'a Program,
    ctx: &'a Ctx<'a>,
    stream: *mut c_void,
    arena: *mut c_void,
    rows: i32,
    banks: &'a BTreeMap<String, Bank>,
    runtime: &'a BTreeMap<String, Rect>,
    pools: &'a Pools,
    geom: &'a Geometry,
    decode_plan: *const DecodePlanCache,
    /// The fire's write origin, a SCALAR smuggled through the pointer
    /// channel: the appender reads `first_token.ptr as i32`
    /// (`kernels-cuda/src/attn/mod.rs:2423`) and the driver answers the same
    /// way (`driver-cuda/src/bind/views.rs:93`). Zero is a real origin, and
    /// the only one a fire with no peel split ever has.
    first_token: i32,
    qo_indptr: *mut c_void,
    row_valid: *mut c_void,
}

impl Fire<'_> {
    /// Where a value lives, chasing merges to the arm that survives.
    fn rect(&self, v: ValueId) -> Rect {
        match &self.program.slots[v as usize] {
            // THE ROW FACTOR MULTIPLIES THE FIRE'S ROWS, and that is the
            // whole of what a routed slot means here: a `FireTimes(k)` value
            // holds `k` rows per fire row, contiguous in its own column, so
            // the rectangle is `rows * k` rows of `width`. This binary fires
            // dense texts only, where every factor is one — carried rather
            // than assumed, because the day a routed text reaches it the
            // arithmetic is already right.
            Slot::Arena {
                offset,
                rows,
                width,
                dtype,
            } => Rect {
                ptr: unsafe { self.arena.cast::<u8>().add(*offset as usize).cast() },
                rows: self.rows
                    * i32::try_from(rows.factor()).expect("a row factor wider than i32"),
                width: i32::try_from(*width).expect("a rectangle wider than i32"),
                dt: *dtype,
            },
            Slot::Alias(to) => self.rect(*to),
            Slot::Runtime(name) => *self
                .runtime
                .get(name)
                .unwrap_or_else(|| panic!("this fire stages no runtime plane `{name}`")),
            Slot::Absent => panic!("value {v} is absent on this lane and a step reads it"),
        }
    }

    fn input(&self, op: &Op, at: usize) -> Rect {
        self.rect(op.inputs[at])
    }

    fn output(&self, op: &Op, at: usize) -> Rect {
        self.rect(op.outputs[at])
    }

    fn weight(&self, op: &Op, at: usize) -> &Bank {
        let name = &op.weights[at];
        self.banks
            .get(name)
            .unwrap_or_else(|| panic!("no bank named `{name}` is on the device"))
    }

    fn p32(op: &Op, at: usize) -> u32 {
        u32::try_from(op.params[at]).expect("a param wider than u32")
    }

    fn pf32(op: &Op, at: usize) -> f32 {
        f32::from_bits(Self::p32(op, at))
    }

    /// The result rectangle of an `InOut` point, with the operand's bytes
    /// already in it. See `dev::copy` for why the copy is not optional.
    fn inout(&self, from: Rect, to: Rect) -> Result<Rect, Refusal> {
        dev::copy(to.ptr, from.ptr.cast_const(), from.bytes(), self.stream)
            .map_err(|_| Refusal::Device { why: "the in-place operand could not be staged" })?;
        Ok(to)
    }

    fn step(&mut self, at: u32, call: &Call) -> Result<(), Refusal> {
        let op = &self.plan.ops[at as usize];
        match call {
            // THE GENERATED DISPATCH, and no shim beside it. Every arm this
            // used to write by hand is now emitted from the point's own slot
            // list into `kernels_cuda::points_dispatch`; what stays here is
            // the half a table cannot write -- the `BoundOp` impl below,
            // which says where THIS executor's rectangles live.
            Call::Point(point) => {
                // `ctx` is a shared reference and Copy: taking it out first
                // is what lets the bound statement borrow `self`.
                let ctx = self.ctx;
                let bound = Bound { fire: self, op, point };
                kernels_cuda::points_dispatch::dispatch(ctx, &bound)
            }
            Call::Symbol(symbol) => self.symbol(symbol, op),
            Call::Tier2(statement) => Err(Refusal::Absent {
                what: Box::leak(
                    format!("a tier-2 shim for `{statement}`; this SKU states none").into_boxed_str(),
                ),
            }),
        }
    }

    /// The pool row a statement names, as the `Cache` mark takes it.
    fn recurrent(&self, op: &Op) -> Result<Cache<Struct<RecurrentState>>, Refusal> {
        let name = op.cache.as_deref().ok_or(Refusal::Unstated {
            what: "the cache row this recurrent statement names",
        })?;
        let view = self.pools.st.get(name).ok_or(Refusal::Absent {
            what: "a recurrent slab for the row this statement names",
        })?;
        Ok(Cache { ptr: core::ptr::from_ref(view) })
    }

    fn pages(&self, op: &Op) -> Result<Cache<Struct<KvCache>>, Refusal> {
        let name = op.cache.as_deref().ok_or(Refusal::Unstated {
            what: "the kv row this attention statement names",
        })?;
        let view = self.pools.kv.get(name).ok_or(Refusal::Absent {
            what: "a kv page table for the row this statement names",
        })?;
        Ok(Cache { ptr: core::ptr::from_ref(view) })
    }

    // ── The staging shim: the routines that keep their own `canon`. ─────
    #[allow(clippy::too_many_lines)]
    fn symbol(&mut self, symbol: &str, op: &Op) -> Result<(), Refusal> {
        let ctx = self.ctx;
        let g = self.geom;
        match symbol {
            // The appender's three runtime planes. `first_token` is a scalar
            // in the pointer channel and `row_valid` is one BYTE per row --
            // both declared `In<Tensor<i32>>` and both read as something
            // else, which is the prefix-agreement the two legs of
            // `attn::write_kv_to_pages` are pinned to
            // (`kernels-cuda/src/attn/mod.rs:2384-2396`).
            "attn::write_kv_to_pages" => {
                let (k, v) = (self.input(op, 0), self.input(op, 1));
                let pages = self.pages(op)?.raised();
                kernels_cuda::attn::kv_paged::write_kv_to_pages_bf16(
                    ctx,
                    rin(k),
                    rin(v),
                    pages,
                    Const::new(g.kv_heads),
                    Const::new(g.head_dim),
                    In { ptr: self.first_token as usize as *const i32, rows: 0, width: 0 },
                    In { ptr: self.qo_indptr.cast(), rows: 1, width: 2 },
                    In { ptr: self.row_valid.cast(), rows: self.rows, width: 1 },
                )
            }

            // ONE ARM FOR BOTH DECODE SPELLINGS, because the second is the
            // first with one more result. `attention.decode_lse` keeps the
            // per-row log-sum-exp so a text can merge the leg with something
            // else — gpt-oss merges it with its attention SINK — and
            // `dispatch_attention_flashinfer_decode_lse` is literally
            // `dispatch_attention_flashinfer_decode` with `Some(lse)`
            // (`attn/fa2/mod.rs:1439-1461`). The staging is identical, so it
            // is written once and the `lse` slot is the only branch.
            "attn::dispatch_attention_flashinfer_decode"
            | "attn::dispatch_attention_flashinfer_decode_lse" => {
                let (q, o) = (self.input(op, 0), self.output(op, 0));
                let lse = (symbol == "attn::dispatch_attention_flashinfer_decode_lse")
                    .then(|| rout(self.output(op, 1)));
                let pages = self.pages(op)?.raised();
                // The statement's window param is `Option<u32>` flattened by
                // `Stmt::window`, which spells `None` as `0`
                // (`model-dsl/src/record.rs:133-135`). flashinfer spells the
                // same absence `-1`, and the driver passes `-1` for every
                // qwen fire (`driver-cuda/src/fire/launch.rs:3209`).
                //
                // A NON-ZERO WINDOW IS `w - 1`, and that is read off the two
                // kernels rather than assumed. flashinfer's window predicate
                // is `kv_idx + qo_len + window_left >= kv_len + qo_idx`
                // (`flashinfer/attention/variants.cuh:89`); a query at
                // absolute position `p = kv_len - qo_len + qo_idx` therefore
                // keeps `kv_idx >= p - window_left`, which is `window_left +
                // 1` keys counting itself. The naive paged kernel spells the
                // same thing arithmetically -- `kv < kv_lim - 1 -
                // window_left` is dropped
                // (`attn/attention_naive_paged.cuh:409`). A text's `window`
                // is the HF number and counts the query's own key: gemma's
                // 512 means `kv_idx > p - 512`, so `window_left = 511`.
                //
                // LEGACY IS OFF BY ONE HERE and the A/B could not see it:
                // `model-legacy/src/gemma_4/project.rs:353-361` passes
                // `sliding_window` itself as `window_left`, which shows one
                // key too many, and every gemma A/B in this tree prefills
                // seven tokens -- a prompt shorter than the window never
                // reaches the predicate.
                let window_left = match Self::p32(op, 0) {
                    0 => -1,
                    w => i32::try_from(w - 1).map_err(|_| Refusal::Wide {
                        what: "the sliding window this statement states",
                        at: i64::from(w),
                        max: i64::from(i32::MAX),
                    })?,
                };
                // The point declares no soft cap, so the statement states
                // none, and zero is what "no cap" spells at this routine
                // (`decode_arm`, `kernels-cuda/src/attn/fa2/mod.rs:1069`).
                let logits_soft_cap = 0.0f32;
                let sm_scale = Self::pf32(op, 2);
                kernels_cuda::attn::fa2::dispatch_attention_flashinfer_decode(
                    ctx,
                    rin(q),
                    In { ptr: self.decode_plan, rows: 0, width: 0 },
                    rout(o),
                    Const::new(window_left),
                    Const::new(logits_soft_cap),
                    Const::new(sm_scale),
                    pages,
                    // `None` where the statement declares no `lse` result:
                    // that lane states one attention leg and nothing merges
                    // partials across it.
                    lse,
                )
            }

            other => Err(Refusal::Absent {
                what: Box::leak(
                    format!("a staging shim for `{other}`; this executor states none").into_boxed_str(),
                ),
            }),
        }
    }

}

// ── One statement, bound ────────────────────────────────────────────────

/// This fire's answer to `kernels::bound::BoundOp`: the half of the point
/// path that a table CANNOT write.
///
/// `kernels_cuda::points_dispatch` is generated and says, for every point
/// the plane claims, which column each slot reads and what element the
/// axis rides. What it cannot say is where a column LIVES — that is the
/// executor's, and it is different for a driver with an arena and a pool
/// allocator than it is for this binary with one row and one page. Every
/// method below is the same lookup the retired hand shim opened each of its
/// arms with; the difference is that there are now twelve of them instead
/// of one per point.
struct Bound<'f, 'a> {
    fire: &'f Fire<'a>,
    op: &'f Op,
    point: &'f str,
}

/// What a rectangle the walk sized rides, as the floor names it.
fn axis(dt: Dt) -> Axis {
    match dt {
        Dt::Bf16 => Axis::Bf16,
        Dt::F32 => Axis::F32,
        Dt::I32 => Axis::I32,
        Dt::U32 => Axis::U32,
        Dt::U8 => Axis::U8,
    }
}

/// What a BANK rides, which is the checkpoint's storage axis and not the
/// plan's repr column. `None` for a dtype no point can be instantiated at,
/// which reads as a refusal rather than as a match.
fn bank_axis(d: Dtype) -> Option<Axis> {
    match d {
        Dtype::Bf16 => Some(Axis::Bf16),
        Dtype::F16 => Some(Axis::F16),
        Dtype::F32 => Some(Axis::F32),
        Dtype::I32 => Some(Axis::I32),
        Dtype::U32 => Some(Axis::U32),
        Dtype::U8 => Some(Axis::U8),
        _ => None,
    }
}

/// THE CHECK THE HAND SHIM MADE TWICE AND OWED EVERYWHERE ELSE.
///
/// A dispatch arm picks the element off ONE witness slot and asks every
/// other slot for the element its declaration pins. `norm.rmsnorm_gated`
/// states an f32 core and an f32 weight beside a bf16 gate, and reading a
/// bf16 rectangle as f32 is a reinterpretation, not a cast — it halves every
/// stride inside the kernel and returns a plausible wrong answer. One line,
/// once, for every slot of every point.
fn rides<T: Rides>(what: &'static str, have: Axis) -> Result<(), Refusal> {
    if T::AXIS == have {
        return Ok(());
    }
    Err(Refusal::Absent { what })
}

impl<'a> kernels::bound::BoundOp for Bound<'_, 'a> {
    type Plane = Ctx<'a>;

    fn point(&self) -> &str {
        self.point
    }

    fn dtype(&self, at: Site) -> Result<Axis, Refusal> {
        Ok(match at {
            Site::In(i) => axis(self.fire.input(self.op, i).dt),
            Site::Out(i) => axis(self.fire.output(self.op, i).dt),
            Site::Const(i) => {
                bank_axis(self.fire.weight(self.op, i).dtype).ok_or(Refusal::Absent {
                    what: "a bank at an element no point is instantiated at",
                })?
            }
        })
    }

    fn tin<T: Rides>(&self, at: usize) -> Result<In<Tensor<T>>, Refusal> {
        let r = self.fire.input(self.op, at);
        rides::<T>("an operand at an element the point does not state", axis(r.dt))?;
        Ok(rin(r))
    }

    fn tout<T: Rides>(&self, at: usize) -> Result<Out<Tensor<T>>, Refusal> {
        let r = self.fire.output(self.op, at);
        rides::<T>("a result at an element the point does not state", axis(r.dt))?;
        Ok(rout(r))
    }

    fn tinout<T: Rides>(&self, from: usize, to: usize) -> Result<InOut<Tensor<T>>, Refusal> {
        let r = self.fire.inout(self.fire.input(self.op, from), self.fire.output(self.op, to))?;
        rides::<T>("an in-place operand at an element the point does not state", axis(r.dt))?;
        Ok(rio(r))
    }

    fn tconst<T: Rides>(&self, at: usize) -> Result<Const<Tensor<T>>, Refusal> {
        let bank = self.fire.weight(self.op, at);
        let have = bank_axis(bank.dtype).ok_or(Refusal::Absent {
            what: "a bank at an element no point is instantiated at",
        })?;
        rides::<T>("a bank at an element the point does not state", have)?;
        Ok(wconst(bank.ptr))
    }

    fn form(&self, at: usize) -> Result<Form, Refusal> {
        match self.fire.weight(self.op, at).repr.as_str() {
            "mxfp4" => Ok(Form::Mxfp4),
            _ => Err(Refusal::Absent {
                what: "a bank at a repr no point is instantiated at",
            }),
        }
    }

    fn bank<R: kernels::points::Repr>(&self, at: usize) -> Result<Const<CudaBank<R>>, Refusal> {
        // THE PLANES ARE COLUMNS, and this is the only accessor that reads
        // more than one. The model text registered them in the repr's own
        // order — codes, then scales — and the DSL's `Stmt::bank` is what put
        // them in the statement that way, so this reads them positionally
        // exactly as every other accessor reads its column.
        let planes: Vec<&Bank> = (0..R::PLANES)
            .map(|p| self.fire.weight(self.op, at + p))
            .collect();
        let [codes, scales] = planes.as_slice() else {
            return Err(Refusal::Absent {
                what: "a bank whose repr stores a plane count this executor cannot bind",
            });
        };
        // Both planes are BYTES on every repr this executor binds, and a
        // plane that is not is a bank read as something it is not.
        for plane in [codes, scales] {
            if plane.dtype != Dtype::U8 {
                return Err(Refusal::Absent {
                    what: "a quantised bank plane stored at an element that is not `u8`",
                });
            }
        }
        Ok(Const::new(Planes {
            codes: codes.ptr.cast_const().cast::<u8>(),
            scales: scales.ptr.cast_const().cast::<u8>(),
        }))
    }

    fn recurrent(&self) -> Result<Cache<Struct<RecurrentState>>, Refusal> {
        self.fire.recurrent(self.op)
    }

    fn pages(&self) -> Result<Cache<Struct<KvCache>>, Refusal> {
        self.fire.pages(self.op)
    }

    fn u32(&self, at: usize) -> Result<u32, Refusal> {
        self.param(at).and_then(|w| {
            u32::try_from(w).map_err(|_| Refusal::Wide {
                what: "a statement param wider than u32",
                at: w.cast_signed(),
                max: i64::from(u32::MAX),
            })
        })
    }

    fn f32(&self, at: usize) -> Result<f32, Refusal> {
        self.u32(at).map(f32::from_bits)
    }

    fn bool(&self, at: usize) -> Result<bool, Refusal> {
        self.u32(at).map(|w| w != 0)
    }

    fn layer(&self) -> Result<u32, Refusal> {
        self.op.layer.ok_or(Refusal::Unstated {
            what: "the layer tag this statement is read at",
        })
    }
}

impl Bound<'_, '_> {
    /// `params[at]`, refused rather than panicked: a plan whose params run
    /// is shorter than the declaration's scalar slots is a lowering bug, and
    /// this is the fire that would report it.
    fn param(&self, at: usize) -> Result<u64, Refusal> {
        self.op.params.get(at).copied().ok_or(Refusal::Unstated {
            what: "a scalar the point declares and the statement does not carry",
        })
    }
}

// ── Reading back ────────────────────────────────────────────────────────

fn read_f32(r: &Rect, stream: *mut c_void) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0u8; r.bytes()];
    dev::download(&mut bytes, r.ptr.cast_const(), stream)?;
    Ok(match r.dt {
        Dt::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect(),
        Dt::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dt::I32 => bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        other => return Err(format!("no host reading for {other:?}")),
    })
}

/// One rectangle's fingerprint: enough to bisect a divergence against a
/// second implementation without shipping the whole row anywhere.
fn digest(r: &Rect, stream: *mut c_void) -> Result<String, String> {
    let v = read_f32(r, stream)?;
    let bad = v.iter().filter(|x| !x.is_finite()).count();
    let (mut lo, mut hi, mut sum, mut abs) = (f32::INFINITY, f32::NEG_INFINITY, 0f64, 0f64);
    for x in v.iter().filter(|x| x.is_finite()) {
        lo = lo.min(*x);
        hi = hi.max(*x);
        sum += f64::from(*x);
        abs += f64::from(x.abs());
    }
    let n = v.len().max(1) as f64;
    let head: Vec<String> = v.iter().take(4).map(|x| format!("{x:.5}")).collect();
    Ok(format!(
        "{}x{} {:?} mean {:+.5} |mean| {:.5} min {:+.5} max {:+.5} nonfinite {bad} head [{}]",
        r.rows,
        r.width,
        r.dt,
        sum / n,
        abs / n,
        lo,
        hi,
        head.join(", ")
    ))
}

/// The checkpoint's own `tokenizer.json`, id -> piece. No BPE decode: this
/// prints the vocabulary ENTRY, which is what identifies the token.
fn vocabulary(snap: &Snapshot) -> BTreeMap<usize, String> {
    let mut out = BTreeMap::new();
    let Ok(text) = std::fs::read_to_string(snap.dir.join("tokenizer.json")) else {
        return out;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
        return out;
    };
    if let Some(vocab) = json["model"]["vocab"].as_object() {
        for (piece, id) in vocab {
            if let Some(id) = id.as_u64() {
                out.insert(id as usize, piece.clone());
            }
        }
    }
    if let Some(added) = json["added_tokens"].as_array() {
        for t in added {
            if let (Some(id), Some(c)) = (t["id"].as_u64(), t["content"].as_str()) {
                out.insert(id as usize, c.to_string());
            }
        }
    }
    out
}

fn show(vocab: &BTreeMap<usize, String>, id: usize) -> String {
    vocab
        .get(&id)
        .map_or_else(|| "<unlisted>".to_string(), |p| format!("{p:?}"))
}

fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn human(n: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < U.len() {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.2} {}", U[i])
}
