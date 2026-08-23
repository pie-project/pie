//! The executor itself. `main.rs` is the two-arm shell; everything the
//! fire does lives here, and the module is gated on `_cuda` so a build
//! with no runtime version selected still WALKS -- `driver-cuda`'s
//! posture (`driver-cuda/src/lib.rs:19-21`: no `compile_error!` for a
//! featureless build, because nothing links `cudarc` and a consumer that
//! forgets is caught by an unresolved path), which is what keeps
//! `cargo check --workspace` able to sweep this member.


use core::ffi::c_void;
use std::collections::BTreeMap;

use kernels::points::{Gate, Gemm, Layout, Mlp, Norm, Rope, Ssm};
use kernels::raises::Struct;
use kernels::routine::{Cache, Const, In, Refusal};
use kernels_cuda::attn::fa2::plan::{DecodePlanCache, Planned};
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::bf16;
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
        "  geometry: head_dim {}, kv_heads {}, q_heads {}, k/v heads {}/{}, k/v dim {}/{}, conv {}x{}",
        geom.head_dim, geom.kv_heads, geom.q_heads, geom.k_h, geom.v_h, geom.k_d, geom.v_d,
        geom.conv_k, geom.conv_dim
    );

    // Enough pages for the whole prompt, taken once: this smoke never
    // recycles a page, so `ceil(tokens / page_size)` is the whole pool.
    let tokens = i32::try_from(args.prompt.len()).map_err(|_| "a prompt longer than i32")?;
    let pages = (tokens + PAGE_SIZE - 1) / PAGE_SIZE;
    let mut pools = Pools::new();
    pools.build(&plan, &geom, pages, stream)?;
    println!(
        "  caches: {} kv row(s) at {pages} x {PAGE_SIZE}-token page(s), {} state slab(s)",
        pools.kv.len(),
        pools.st.len()
    );

    // The three runtime planes a one-row decode stages. `token_ids` and
    // `positions` are the fire's own data and are rewritten per fire;
    // `qo_indptr` is the request CSR (`[0, 1]`: one request, one token row)
    // and never moves.
    let ids = Slab::zeroed(4, stream)?;
    let positions = Slab::zeroed(4, stream)?;
    let qo_indptr = Slab::of(&u32s(&[0, 1]), stream)?;
    // One BYTE per row, all ones. `driver-cuda/src/fire/scratch.rs:261-276`
    // memsets exactly this; the routine declares it `In<Tensor<i32>>` and
    // casts to `*const u8` (`kernels-cuda/src/attn/mod.rs:2420`), so the
    // declared element is a fiction the DECLARATION carries and the buffer
    // must not.
    let row_valid = Slab::of(&[1u8], stream)?;

    let mut runtime: BTreeMap<String, Rect> = BTreeMap::new();
    runtime.insert("token_ids".into(), Rect { ptr: ids.ptr(), rows, width: 1, dt: Dt::I32 });
    runtime.insert("positions".into(), Rect { ptr: positions.ptr(), rows, width: 1, dt: Dt::I32 });
    // ROWS IS THE REQUEST COUNT on this one, not the buffer's length: the
    // appender reads `num_requests = qo_indptr.rows`
    // (`kernels-cuda/src/attn/mod.rs:2415`), which is what
    // `driver-cuda/src/bind/mod.rs:2206` puts there from `lowered.arg_rows`.
    runtime.insert("qo_indptr".into(), Rect { ptr: qo_indptr.ptr(), rows: 1, width: 2, dt: Dt::I32 });

    // ── 5. The scratch the gdn seam needs, and the fa2 workspaces. ──────
    let scratch = Scratch::carve(&plan, program, &geom, rows, stream)?;
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
            scratch: &scratch,
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
}

/// The numbers the claim-only routines want and the statements do not carry.
///
/// EVERY ONE IS READ OFF THE PLAN, never off a config file: `head_dim` and
/// the four gdn head numbers are statement params, `kv_heads` divides the
/// cache row by `head_dim`, `q_heads` divides the decode statement's own
/// operand, and `conv_dim` is the conv statement's operand width. A number
/// this could not find is a refusal with the point named.
struct Geometry {
    head_dim: i32,
    kv_heads: i32,
    q_heads: i32,
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
        let decode = find("attention.decode").ok_or("the plan states no `attention.decode`")?;
        let head_dim = i32::try_from(decode.params[1]).map_err(|_| "a wide head_dim")?;
        // `attention.decode`'s operand is the roped `q`, whose width is
        // `q_heads * head_dim`. `program.slots` is where that width lives.
        let q_width = match program.slots[decode.inputs[0] as usize] {
            Slot::Arena { width, .. } => width,
            ref other => return Err(format!("`attention.decode`'s q lives at {other:?}")),
        };
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

        let gd = find("ssm.gated_delta").ok_or("the plan states no `ssm.gated_delta`")?;
        let conv = find("ssm.causal_conv1d").ok_or("the plan states no `ssm.causal_conv1d`")?;
        let conv_dim = match program.slots[conv.inputs[0] as usize] {
            Slot::Arena { width, .. } => i32::try_from(width).map_err(|_| "a wide conv")?,
            ref other => return Err(format!("`ssm.causal_conv1d`'s x lives at {other:?}")),
        };
        Ok(Geometry {
            head_dim,
            kv_heads: kv_width / head_dim,
            q_heads: i32::try_from(q_width).map_err(|_| "a wide q")? / head_dim,
            k_h: gd.params[0] as i32,
            v_h: gd.params[1] as i32,
            k_d: gd.params[2] as i32,
            v_d: gd.params[3] as i32,
            conv_k: conv.params[0] as i32,
            conv_dim,
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

    fn build(&mut self, plan: &Plan, g: &Geometry, pages: i32, stream: *mut c_void) -> Result<(), String> {
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
        let conv_elems = g.conv_k as usize * g.conv_dim as usize;
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
                        },
                    );
                    self.slabs.push(k);
                    self.slabs.push(v);
                }
                CacheRow::State { name, slab } => {
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

// ── The gdn seam's scratch ──────────────────────────────────────────────

/// The four extra f32 rows `ssm.gdn_prep` writes beside the `gates` its
/// statement states.
///
/// `kernels-cuda/src/ssm.rs:105-113` states this gap by name: the statement
/// carries ONE `[a | b]` operand and states ONE result, and
/// `qwen_gdn_post_conv_prep_bf16` writes FIVE rectangles. Two of the five --
/// `g_log` and `beta`, one f32 per value head each -- fit the stated `gates`
/// rectangle exactly: the width rule sizes it as `ba`'s row on f32, which is
/// `2 * v_heads`, so `[g_log | beta]` IS that row. The other three
/// (`q_norm_kh`, `k_norm_kh`, `v`, each `heads * dim` f32) have no rectangle
/// in the plan and are carved here, one column per statement, mirroring the
/// arena's own no-reuse rule.
struct Scratch {
    cuts: BTreeMap<u32, [Rect; 3]>,
    _slab: Slab,
}

impl Scratch {
    fn carve(
        plan: &Plan,
        program: &Program,
        g: &Geometry,
        rows: i32,
        stream: *mut c_void,
    ) -> Result<Scratch, String> {
        let widths = [g.k_h * g.k_d, g.k_h * g.k_d, g.v_h * g.v_d];
        let row: usize = widths.iter().map(|w| *w as usize * 4).sum();
        let preps: Vec<u32> = program
            .steps
            .iter()
            .filter(|s| plan.ops[s.op as usize].kernel == "ssm.gdn_prep")
            .map(|s| s.op)
            .collect();
        let slab = Slab::zeroed(row * preps.len().max(1), stream)?;
        let mut cuts = BTreeMap::new();
        for (i, op) in preps.iter().enumerate() {
            let mut at = i * row;
            let mut cut = [Rect { ptr: core::ptr::null_mut(), rows, width: 0, dt: Dt::F32 }; 3];
            for (j, w) in widths.iter().enumerate() {
                cut[j] = Rect {
                    ptr: unsafe { slab.ptr().cast::<u8>().add(at).cast() },
                    rows,
                    width: *w,
                    dt: Dt::F32,
                };
                at += *w as usize * 4;
            }
            cuts.insert(*op, cut);
        }
        Ok(Scratch { cuts, _slab: slab })
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
    scratch: &'a Scratch,
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
            Slot::Arena { offset, width, dtype } => Rect {
                ptr: unsafe { self.arena.cast::<u8>().add(*offset as usize).cast() },
                rows: self.rows,
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
            Call::Point(point) => self.point(point, op),
            Call::Symbol(symbol) => self.symbol(symbol, at, op),
            Call::Tier2(statement) => Err(Refusal::Absent {
                what: Box::leak(
                    format!("a tier-2 shim for `{statement}`; this SKU states none").into_boxed_str(),
                ),
            }),
        }
    }

    // ── The point shim: the plane's own claims, by declaration order. ───
    //
    // Every arm is the same three moves and nothing else: read the operands
    // and scalars off the statement in the order `model_dsl::kernels` records
    // them, wear them as the marks `kernels::points` declares, call the
    // method. The dtype comes off the SLOT -- there is no default and no
    // cast; a dtype with no arm is a refusal naming the point, which is what
    // a generated dispatch's `Elem^axes` match does for the axes a plane has
    // no instantiation for.
    #[allow(clippy::too_many_lines)]
    fn point(&mut self, point: &str, op: &Op) -> Result<(), Refusal> {
        let ctx = self.ctx;
        let unpointed = |dt: Dt| Refusal::Absent {
            what: Box::leak(format!("`{point}` at {dt:?}").into_boxed_str()),
        };
        match point {
            // The two norms and their OFFSET-BANK twins. The convention is
            // the checkpoint's, so it is the point that picks: `_plus_one`
            // scales by `1 + weight`.
            "norm.rmsnorm" | "norm.rmsnorm_plus_one" => {
                let (x, y, w) = (self.input(op, 0), self.output(op, 0), self.weight(op, 0).ptr);
                let eps = Self::pf32(op, 0);
                let plus = point == "norm.rmsnorm_plus_one";
                match (y.dt, plus) {
                    (Dt::Bf16, false) => ctx.rmsnorm::<bf16>(rin(x), wconst(w), eps, rout(y)),
                    (Dt::F32, false) => ctx.rmsnorm::<f32>(rin(x), wconst(w), eps, rout(y)),
                    (Dt::Bf16, true) => ctx.rmsnorm_plus_one::<bf16>(rin(x), wconst(w), eps, rout(y)),
                    (Dt::F32, true) => ctx.rmsnorm_plus_one::<f32>(rin(x), wconst(w), eps, rout(y)),
                    (other, _) => Err(unpointed(other)),
                }
            }
            "norm.rmsnorm_per_head" | "norm.rmsnorm_per_head_plus_one" => {
                let (x, y, w) = (self.input(op, 0), self.output(op, 0), self.weight(op, 0).ptr);
                let (head_dim, eps) = (Self::p32(op, 0), Self::pf32(op, 1));
                let plus = point == "norm.rmsnorm_per_head_plus_one";
                match (y.dt, plus) {
                    (Dt::Bf16, false) => {
                        ctx.rmsnorm_per_head::<bf16>(rin(x), wconst(w), head_dim, eps, rout(y))
                    }
                    (Dt::Bf16, true) => {
                        ctx.rmsnorm_per_head_plus_one::<bf16>(rin(x), wconst(w), head_dim, eps, rout(y))
                    }
                    (other, _) => Err(unpointed(other)),
                }
            }
            "norm.rmsnorm_gated" => {
                // The declaration SPELLS the core and the weight f32 and
                // quantifies only over the gate's element -- which is why
                // `program.rs`'s width rule sizes this result from `like(1)`.
                let (core, gate) = (self.input(op, 0), self.input(op, 1));
                let (y, w) = (self.output(op, 0), self.weight(op, 0));
                let (head_dim, eps) = (Self::p32(op, 0), Self::pf32(op, 1));
                if core.dt != Dt::F32 {
                    return Err(unpointed(core.dt));
                }
                if w.dtype != Dtype::F32 {
                    // The checkpoint ships qwen's gdn norm F32 and the
                    // declaration agrees; a bf16 bank here would be a silent
                    // halving of every stride inside the kernel.
                    return Err(Refusal::Absent { what: "a gated norm weight stored f32" });
                }
                match y.dt {
                    Dt::Bf16 => {
                        ctx.rmsnorm_gated::<bf16>(rin(core), rin(gate), wconst(w.ptr), head_dim, eps, rout(y))
                    }
                    other => Err(unpointed(other)),
                }
            }
            "norm.residual_add" => {
                // `(x: In, y: InOut)` -- the ONE point of the family whose
                // `InOut` is not the receiver, which `program.rs`'s width
                // table calls out and sizes from `like(1)`.
                let x = self.input(op, 0);
                let y = self.inout(self.input(op, 1), self.output(op, 0))?;
                match y.dt {
                    Dt::Bf16 => ctx.residual_add::<bf16>(rin(x), rio(y)),
                    Dt::F32 => ctx.residual_add::<f32>(rin(x), rio(y)),
                    other => Err(unpointed(other)),
                }
            }
            "gemm.matmul" | "gemm.lm_head" | "gemm.attention_landing" => {
                let (act, y, w) = (self.input(op, 0), self.output(op, 0), self.weight(op, 0).ptr);
                let layer = op.layer.unwrap_or(0);
                match y.dt {
                    Dt::Bf16 => match point {
                        "gemm.matmul" => ctx.matmul::<bf16>(rin(act), wconst(w), rout(y)),
                        "gemm.lm_head" => ctx.lm_head::<bf16>(rin(act), wconst(w), rout(y)),
                        _ => ctx.attention_landing::<bf16>(rin(act), wconst(w), layer, rout(y)),
                    },
                    other => Err(unpointed(other)),
                }
            }
            "mlp.swiglu" => {
                let (packed, y) = (self.input(op, 0), self.output(op, 0));
                let intermediate = Self::p32(op, 0);
                match y.dt {
                    Dt::Bf16 => ctx.swiglu::<bf16>(rin(packed), intermediate, rout(y)),
                    other => Err(unpointed(other)),
                }
            }
            "gate.sigmoid_mul" => {
                let gate = self.input(op, 1);
                let x = self.inout(self.input(op, 0), self.output(op, 0))?;
                match x.dt {
                    Dt::Bf16 => ctx.sigmoid_mul::<bf16>(rio(x), rin(gate)),
                    other => Err(unpointed(other)),
                }
            }
            "layout.split_q_gate" => {
                let packed = self.input(op, 0);
                let (q, gate) = (self.output(op, 0), self.output(op, 1));
                let head_dim = Self::p32(op, 0);
                match q.dt {
                    Dt::Bf16 => ctx.split_q_gate::<bf16>(rin(packed), head_dim, rout(q), rout(gate)),
                    other => Err(unpointed(other)),
                }
            }
            "layout.split_rows" => {
                let x = self.input(op, 0);
                let (left, right) = (self.output(op, 0), self.output(op, 1));
                let width = Self::p32(op, 0);
                match x.dt {
                    Dt::Bf16 => ctx.split_rows::<bf16>(rin(x), width, rout(left), rout(right)),
                    other => Err(unpointed(other)),
                }
            }
            "rope.partial" => {
                let pos = self.input(op, 2);
                let q = self.inout(self.input(op, 0), self.output(op, 0))?;
                let k = self.inout(self.input(op, 1), self.output(op, 1))?;
                let (rotary_dim, head_dim, theta) =
                    (Self::p32(op, 0), Self::p32(op, 1), Self::pf32(op, 2));
                match q.dt {
                    Dt::Bf16 => ctx.partial::<bf16>(rio(q), rio(k), rin(pos), rotary_dim, head_dim, theta),
                    other => Err(unpointed(other)),
                }
            }
            "ssm.causal_conv1d" => {
                let (x, y, w) = (self.input(op, 0), self.output(op, 0), self.weight(op, 0).ptr);
                let state = self.recurrent(op)?;
                let conv_width = Self::p32(op, 0);
                match y.dt {
                    Dt::Bf16 => ctx.causal_conv1d::<bf16>(rin(x), wconst(w), state, conv_width, rout(y)),
                    other => Err(unpointed(other)),
                }
            }
            other => Err(Refusal::Absent {
                what: Box::leak(
                    format!("a point shim for `{other}`; this executor states none").into_boxed_str(),
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

    fn pages(&self, op: &Op) -> Result<In<Struct<KvCache>>, Refusal> {
        let name = op.cache.as_deref().ok_or(Refusal::Unstated {
            what: "the kv row this attention statement names",
        })?;
        let view = self.pools.kv.get(name).ok_or(Refusal::Absent {
            what: "a kv page table for the row this statement names",
        })?;
        // A RAISE HAS NO SHAPE -- one object with one lifetime, not a
        // rectangle (`kernels/src/routine.rs:543-560`).
        Ok(In { ptr: core::ptr::from_ref(view), rows: 0, width: 0 })
    }

    // ── The staging shim: the routines that keep their own `canon`. ─────
    #[allow(clippy::too_many_lines)]
    fn symbol(&mut self, symbol: &str, at: u32, op: &Op) -> Result<(), Refusal> {
        let ctx = self.ctx;
        let g = self.geom;
        match symbol {
            // `layout.embed` stays off the floor because `embed_bf16` clamps
            // every id against the table's ROW count and a `Const` table is
            // an address with no rectangle -- a delegation could only invent
            // a bound (`kernels-cuda/src/layout.rs:100-107`). The row count
            // is right here in the plan's `params` column, which is what the
            // Load contract's parameter registration is FOR.
            "layout::embed_bf16" => {
                let (ids, y, table) = (self.input(op, 0), self.output(op, 0), self.weight(op, 0));
                let vocab = i32::try_from(table.shape[0]).map_err(|_| Refusal::Wide {
                    what: "the embedding table's row count",
                    at: table.shape[0].cast_signed(),
                    max: i64::from(i32::MAX),
                })?;
                kernels_cuda::layout::embed_bf16(ctx, wconst(table.ptr), rout(y), rin(ids), Const::new(vocab))
            }

            // FIVE RESULTS OUT OF ONE STATEMENT, and one operand the
            // statement does not carry. The missing operand is the
            // POST-CONVOLUTION qkv, and it is found through the plan's own
            // dataflow rather than guessed: the recurrence downstream takes
            // it as its first operand and takes this statement's result as
            // its third, so the `ssm.gated_delta` whose `inputs[2]` is this
            // op's output names the qkv in its `inputs[0]`. That join is the
            // text's own edge, read back off the plan.
            "ssm::qwen_gdn_post_conv_prep_bf16" => {
                let ba = self.input(op, 0);
                let recurrence = self.paired(at, op)?;
                let qkv = self.input(recurrence, 0);
                // `[b | a]`, in that order: the import packs
                // `[in_proj_b, in_proj_a]`
                // (`crates/model/src/qwen_3_5/import.rs:30-33`) and the legacy
                // text's `split_qwen_gdn_ba` returns `(b, a)`
                // (`model-legacy/src/qwen_3_5/forward/mod.rs:428`).
                let b = ba.column(0, g.v_h);
                let a = ba.column(g.v_h, g.v_h);
                let dt_bias = self.weight(op, 0);
                let a_log = self.weight(op, 1);
                if a_log.dtype != Dtype::F32 {
                    return Err(Refusal::Absent { what: "an `a_log` bank stored f32" });
                }
                let [q_norm, k_norm, v_f32] = self.scratch.cuts[&at];
                // The stated `gates` row IS `[g_log | beta]`: the width rule
                // sizes it as `ba`'s row on f32, which is `2 * v_heads`.
                let gates = self.output(op, 0);
                let g_log = gates.column(0, g.v_h);
                let beta = gates.column(g.v_h, g.v_h);
                kernels_cuda::driver_internal::qwen_gdn_post_conv_prep_bf16(
                    ctx,
                    rin(qkv),
                    rin(a),
                    rin(b),
                    wconst(a_log.ptr),
                    wconst(dt_bias.ptr),
                    rout(q_norm),
                    rout(k_norm),
                    rout(v_f32),
                    rout(g_log),
                    rout(beta),
                    Const::new(g.k_h),
                    Const::new(g.v_h),
                    Const::new(g.k_d),
                    Const::new(g.v_d),
                    Const::new(g.conv_dim),
                )
            }

            // THE SAME SEAM FROM THE OTHER SIDE. The recurrence takes the
            // prep's five f32 rows as five operands; the statement hands it
            // the packed `qkv`, the gate row `z`, and one fused `gates` that
            // stands for all five. This glue owns the real layout, so it
            // reads the three scratch columns and the two halves of `gates`
            // straight back off the statement that wrote them.
            //
            // `qkv` and `z` are the statement's own first two operands and
            // are NOT passed on: the recurrence reads the projections the
            // prep already normalised, and the gate `z` is the out-norm's,
            // spent by `norm.rmsnorm_gated` downstream.
            "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16" => {
                let prep = self.producer(op.inputs[2])?;
                let [q_norm, k_norm, v_f32] = *self.scratch.cuts.get(&prep).ok_or(Refusal::Unstated {
                    what: "the prep columns this recurrence reads",
                })?;
                let gates = self.input(op, 2);
                let g_log = gates.column(0, g.v_h);
                let beta = gates.column(g.v_h, g.v_h);
                let out = self.output(op, 0);
                let state = self.recurrent(op)?;
                kernels_cuda::ssm::recurrent_gated_delta_step_batched_gqa_state_bf16(
                    ctx,
                    rin(q_norm),
                    rin(k_norm),
                    rin(v_f32),
                    rin(g_log),
                    rin(beta),
                    rout(out),
                    Const::new(g.k_h),
                    Const::new(g.v_h),
                    Const::new(g.k_d),
                    Const::new(g.v_d),
                    // One request. The step form takes `r` as a scalar; only
                    // the chunked form reads it off a CSR's row count.
                    Const::new(1),
                    state.raised(),
                )
            }

            // The appender's three runtime planes. `first_token` is a scalar
            // in the pointer channel and `row_valid` is one BYTE per row --
            // both declared `In<Tensor<i32>>` and both read as something
            // else, which is the prefix-agreement the two legs of
            // `attn::write_kv_to_pages` are pinned to
            // (`kernels-cuda/src/attn/mod.rs:2384-2396`).
            "attn::write_kv_to_pages" => {
                let (k, v) = (self.input(op, 0), self.input(op, 1));
                let pages = self.pages(op)?;
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

            "attn::dispatch_attention_flashinfer_decode" => {
                let (q, o) = (self.input(op, 0), self.output(op, 0));
                let pages = self.pages(op)?;
                // The statement's window param is `Option<u32>` flattened by
                // `Stmt::window`, which spells `None` as `0`
                // (`model-dsl/src/record.rs:133-135`). flashinfer spells the
                // same absence `-1`, and the driver passes `-1` for every
                // qwen fire (`driver-cuda/src/fire/launch.rs:3209`). A
                // NON-ZERO window would need the `w` -> `window_left`
                // convention pinned, and no shipping text in this tree states
                // one -- so it refuses rather than guessing.
                let window_left = match Self::p32(op, 0) {
                    0 => -1,
                    _ => {
                        return Err(Refusal::Unstated {
                            what: "how a stated sliding window maps to flashinfer's `window_left`",
                        });
                    }
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
                    // No log-sum-exp: this lane states one attention leg and
                    // nothing merges partials across it.
                    None,
                )
            }

            other => Err(Refusal::Absent {
                what: Box::leak(
                    format!("a staging shim for `{other}`; this executor states none").into_boxed_str(),
                ),
            }),
        }
    }

    /// The `ssm.gated_delta` that consumes `prep`'s `gates`.
    fn paired(&self, prep: u32, op: &Op) -> Result<&Op, Refusal> {
        let gates = *op.outputs.first().ok_or(Refusal::Unstated {
            what: "the `gates` result `ssm.gdn_prep` states",
        })?;
        self.program
            .steps
            .iter()
            .map(|s| &self.plan.ops[s.op as usize])
            .find(|o| {
                o.kernel == "ssm.gated_delta" && o.inputs.get(2) == Some(&gates)
            })
            .ok_or(Refusal::Unstated {
                what: Box::leak(
                    format!("the recurrence that consumes op {prep}'s gates").into_boxed_str(),
                ),
            })
    }

    /// Which op states `v`.
    fn producer(&self, v: ValueId) -> Result<u32, Refusal> {
        match self.plan.values[v as usize] {
            model_ir::plan::ValueDef::Stmt(op) => Ok(op),
            _ => Err(Refusal::Unstated {
                what: "the statement that writes this recurrence's gates",
            }),
        }
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
