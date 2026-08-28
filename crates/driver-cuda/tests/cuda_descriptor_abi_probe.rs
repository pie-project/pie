//! **A PROBE, NOT A GATE (`palo cuda-abi` wave).** It asserts almost nothing
//! and prints a census; the deliverable is `.wiki/palo/cuda-abi.md`.
//!
//! The question: build log 10 ruled a per-fire rebind unreachable because
//! "rebinding needs a host-side map from graph node to kernel argument, and
//! the shell never sees one". A CAPTURED GRAPH IS THAT MAP. This probe
//! captures the same walk at several compositions, walks every kernel node
//! with `cuGraphKernelNodeGetParams` + `cuFuncGetParamInfo`
//! (`driver_cuda::device::nodes`), and asks of every component — three grid
//! axes, three block axes, shared memory, and every argument of the parameter
//! block — whether it is a constant, an affine function of the fire's window,
//! or neither.
//!
//! ```text
//! PIE_SMOKE_SNAPSHOT=... cargo test -p driver-cuda --features cuda-13 \
//!   --release --test cuda_descriptor_abi_probe -- --nocapture --test-threads=1
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use driver_cuda::device::nodes::{self, Walked};
use driver_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budgets;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";
const PROMPT: &str = "The capital of France is";

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
}

fn container(snapshot: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn word(query_len: u32) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    if !driver_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let checkpoint = snapshot()?;
    let container = container(&checkpoint)?;
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU");
    let plan = trace(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);
    let shell = Shell::load(Boot {
        plan,
        contract: &contract,
        checkpoint: &checkpoint,
        budgets: Budgets::new(4, 512),
        profile: None,
        page_size: 16,
        context: 1024,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::On,
    })
    .expect("the shell loads");
    Some((shell, tokenizer))
}

// ─────────────────────────────────────────────────────────────────────────
// The census

/// What one walked graph is, summarised.
fn census(label: &str, walked: &Walked) {
    let kernels = walked.nodes.iter().filter(|n| n.kernel()).count();
    let readable = walked
        .nodes
        .iter()
        .filter(|n| n.kernel() && n.opaque.is_none())
        .count();
    let components: usize = walked
        .nodes
        .iter()
        .filter(|n| n.kernel())
        .map(|n| 7 + n.params.iter().map(|p| p.bytes.len().div_ceil(8).max(1)).sum::<usize>())
        .sum();
    let mut kinds: BTreeMap<u32, usize> = BTreeMap::new();
    let mut opaque: BTreeMap<&'static str, usize> = BTreeMap::new();
    for node in &walked.nodes {
        *kinds.entry(node.kind).or_default() += 1;
        if node.kernel()
            && let Some(why) = node.opaque
        {
            *opaque.entry(why).or_default() += 1;
        }
    }
    let mut families: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut fam_components: BTreeMap<&'static str, usize> = BTreeMap::new();
    for node in walked.nodes.iter().filter(|n| n.kernel()) {
        *families.entry(family(&node.symbol)).or_default() += 1;
        *fam_components.entry(family(&node.symbol)).or_default() += 7 + node
            .params
            .iter()
            .map(|p| p.bytes.len().div_ceil(8).max(1))
            .sum::<usize>();
    }
    let widest = walked
        .nodes
        .iter()
        .filter(|n| n.kernel())
        .map(|n| n.params.len())
        .max()
        .unwrap_or(0);
    eprintln!(
        "\n-- {label}: {} nodes ({kernels} kernel, {} other), {} edges, \
         {readable} param blocks read, {components} components, widest arg list {widest}, \
         {} ambiguous order",
        walked.nodes.len(),
        walked.nodes.len() - kernels,
        walked.edges,
        walked.ambiguous,
    );
    let mut widths: BTreeMap<usize, usize> = BTreeMap::new();
    let mut blocks = 0usize;
    let mut block_bytes = 0usize;
    for node in walked.nodes.iter().filter(|n| n.kernel()) {
        for param in &node.params {
            *widths.entry(param.bytes.len()).or_default() += 1;
            if param.bytes.len() > 8 {
                blocks += 1;
                block_bytes += param.bytes.len();
            }
        }
    }
    eprintln!("   node kinds:      {kinds:?}");
    eprintln!(
        "   param widths:    {widths:?}  ({blocks} by-value blocks, {block_bytes} bytes)"
    );
    eprintln!("   nodes by plane:  {families:?}");
    eprintln!("   components:      {fam_components:?}");
    if !opaque.is_empty() {
        eprintln!("   OPAQUE:          {opaque:?}");
    }
}

/// Which plane a mangled symbol belongs to.
fn family(symbol: &str) -> &'static str {
    if symbol.contains("10flashinfer") {
        "flashinfer (ours, nvrtc)"
    } else if symbol.contains("3pie") {
        "pie (ours, nvrtc)"
    } else if symbol.contains("nccl") {
        "nccl"
    } else if symbol.contains("cutlass") {
        "cutlass (cublasLt)"
    } else if symbol.contains("cublas")
        || symbol.starts_with("nvjet")
        || symbol.contains("xmma")
        || symbol.starts_with("sm90_")
        || symbol.starts_with("ampere_")
    {
        "cublas(Lt)"
    } else if symbol.is_empty() {
        "unnamed"
    } else {
        "other"
    }
}

/// A component of a node: a grid/block axis, shared memory, or one argument.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Part {
    Grid(usize),
    Block(usize),
    Smem,
    /// The `w`-th aligned eight-byte word of parameter `a`. A scalar or a
    /// pointer is one word; a by-value block (`ArgValue::Bytes`, cutlass's
    /// `Params`, flashinfer's traits struct) is as many as it is wide, and
    /// fitting it word by word is what makes a library kernel's parameter
    /// block as derivable as one of ours.
    Arg(usize, usize),
}

impl core::fmt::Display for Part {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Part::Grid(a) => write!(f, "grid.{a}"),
            Part::Block(a) => write!(f, "block.{a}"),
            Part::Smem => write!(f, "smem"),
            Part::Arg(a, 0) => write!(f, "arg[{a}]"),
            Part::Arg(a, w) => write!(f, "arg[{a}].w{w}"),
        }
    }
}

fn value(node: &nodes::Node, part: Part) -> Option<u64> {
    match part {
        Part::Grid(a) => Some(u64::from(node.grid[a])),
        Part::Block(a) => Some(u64::from(node.block[a])),
        Part::Smem => Some(u64::from(node.smem)),
        Part::Arg(a, w) => {
            let param = node.params.get(a)?;
            let at = w * 8;
            if at >= param.bytes.len() {
                return None;
            }
            let mut cell = [0u8; 8];
            let take = (param.bytes.len() - at).min(8);
            cell[..take].copy_from_slice(&param.bytes[at..at + take]);
            Some(u64::from_le_bytes(cell))
        }
    }
}

fn parts(node: &nodes::Node) -> Vec<Part> {
    let mut all = vec![
        Part::Grid(0),
        Part::Grid(1),
        Part::Grid(2),
        Part::Block(0),
        Part::Block(1),
        Part::Block(2),
        Part::Smem,
    ];
    for (at, param) in node.params.iter().enumerate() {
        for w in 0..param.bytes.len().div_ceil(8).max(1) {
            all.push(Part::Arg(at, w));
        }
    }
    all
}

/// A device address, as far as this probe can tell.
fn pointerish(v: u64) -> bool {
    v >= 1 << 32
}

/// A readable stub of a mangled symbol.
fn short(symbol: &str) -> String {
    symbol.chars().take(64).collect()
}

#[derive(Default)]
struct Fit {
    constants: usize,
    constant_pointers: usize,
    affine: usize,
    affine_pointers: usize,
    unaffine: Vec<String>,
    unaffine_kinds: BTreeMap<String, usize>,
    moving_pointers: usize,
    moving_scalars: usize,
    arm_switches: usize,
    arm_symbols: BTreeMap<String, usize>,
    fitted_nodes: usize,
    /// Nodes with at least one component that moves - the ones a per-fire
    /// rebind would actually have to touch.
    moving_nodes: usize,
}

/// Fit `v = base + slope·x` (slope RATIONAL) on the first two points and
/// verify on every one after them.
///
/// The rational slope matters: `layout::embed`'s grid is `rows.div_ceil(2)`,
/// which moves by one for every two rows, and an integer-only fitter calls
/// that a break where there is none. What it CANNOT catch is a `div_ceil`
/// whose probe points are all congruent — hence the odd verification point
/// `.wiki/palo/icb.md` §9's fitter bug argues for.
fn fit(label: &str, axis: &str, points: &[(u64, Walked)]) -> Fit {
    let mut out = Fit::default();
    let first = &points[0].1;
    for (x, walked) in &points[1..] {
        if walked.nodes.len() != first.nodes.len() {
            eprintln!(
                "   !! {label}: {axis}={x} holds {} nodes against {} - topology moved, \
                 no law fitted",
                walked.nodes.len(),
                first.nodes.len()
            );
            return out;
        }
    }
    for at in 0..first.nodes.len() {
        let node = &first.nodes[at];
        if !node.kernel() {
            continue;
        }
        if points[1..]
            .iter()
            .any(|(_, w)| w.nodes[at].symbol != node.symbol)
        {
            out.arm_switches += 1;
            *out.arm_symbols
                .entry(format!("{} -> {}", family(&node.symbol), {
                    let other = points[1..]
                        .iter()
                        .map(|(_, w)| family(&w.nodes[at].symbol))
                        .find(|f| *f != family(&node.symbol))
                        .unwrap_or("?");
                    other
                }))
                .or_default() += 1;
            continue;
        }
        out.fitted_nodes += 1;
        let mut this_node_moves = false;
        for part in parts(node) {
            let Some(v0) = value(node, part) else { continue };
            let mut seen: Vec<(u64, u64)> = vec![(points[0].0, v0)];
            let mut missing = false;
            for (x, walked) in &points[1..] {
                match value(&walked.nodes[at], part) {
                    Some(v) => seen.push((*x, v)),
                    None => missing = true,
                }
            }
            if missing {
                continue;
            }
            if seen.iter().all(|(_, v)| *v == v0) {
                out.constants += 1;
                if pointerish(v0) {
                    out.constant_pointers += 1;
                }
                continue;
            }
            this_node_moves = true;
            if pointerish(v0) {
                out.moving_pointers += 1;
            } else {
                out.moving_scalars += 1;
            }
            let (x0, y0) = seen[0];
            let (x1, y1) = seen[1];
            let dx = i128::from(x1) - i128::from(x0);
            let dy = i128::from(y1) - i128::from(y0);
            let fits = dx != 0
                && seen.iter().all(|(x, y)| {
                    let rise = dy * (i128::from(*x) - i128::from(x0));
                    rise % dx == 0 && i128::from(y0) + rise / dx == i128::from(*y)
                });
            if fits {
                out.affine += 1;
                if pointerish(v0) {
                    out.affine_pointers += 1;
                }
            } else {
                *out.unaffine_kinds
                    .entry(format!("{} {part}", short(&node.symbol)))
                    .or_default() += 1;
                out.unaffine.push(format!(
                    "node {at} {} {part}: {:?}",
                    short(&node.symbol),
                    seen
                ));
            }
        }
        if this_node_moves {
            out.moving_nodes += 1;
        }
    }
    out
}

fn report(label: &str, f: &Fit) {
    eprintln!("\n   {label}");
    eprintln!(
        "      {} nodes fitted, {} ARM-SWITCHING nodes {:?}",
        f.fitted_nodes, f.arm_switches, f.arm_symbols
    );
    eprintln!(
        "      {} constants ({} pointers) | {} affine ({} pointers) | {} UNAFFINE",
        f.constants,
        f.constant_pointers,
        f.affine,
        f.affine_pointers,
        f.unaffine.len(),
    );
    eprintln!(
        "      movers: {} pointers, {} scalars, in {} of {} nodes",
        f.moving_pointers, f.moving_scalars, f.moving_nodes, f.fitted_nodes
    );
    if !f.unaffine_kinds.is_empty() {
        eprintln!("      unaffine, by (symbol, component):");
        let mut rows: Vec<(&String, &usize)> = f.unaffine_kinds.iter().collect();
        rows.sort_by(|a, b| b.1.cmp(a.1));
        for (what, count) in rows.iter().take(20) {
            eprintln!("        {count:>4}x {what}");
        }
        if rows.len() > 20 {
            eprintln!("        ... {} more distinct (symbol, component)", rows.len() - 20);
        }
    }
    for line in f.unaffine.iter().take(10) {
        eprintln!("      e.g. {line}");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The fires

/// Fire one composition until it captures, then hand back the graph the shell
/// kept for it.
fn capture(shell: &mut Shell, lanes: &[Lane<'_>], want: usize) -> Option<Walked> {
    for _ in 0..3 {
        shell.fire(lanes).expect("the probe fire fires");
    }
    let kept = shell.kept_graphs();
    if kept.len() <= want {
        eprintln!("   (nothing captured for this composition)");
        return None;
    }
    let (key, graph) = &kept[want];
    let walked = nodes::walk(graph).expect("the captured graph walks");
    eprintln!("   (key {key})");
    Some(walked)
}

/// **PROBE.** The whole census.
#[test]
fn what_a_captured_graph_says_about_its_own_arguments() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the cuda descriptor-abi probe") else {
        eprintln!("skipping the cuda descriptor-abi probe: no device or no checkpoint");
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let filler: Vec<u32> = core::iter::repeat_n(prompt[0], 32).collect();

    // -- Seat four slots.
    shell.set_mode(Graphs::Off);
    let mut fed = Vec::new();
    for slot in 0..4u32 {
        shell.open(slot).expect("a slot opens");
        let out = shell
            .fire(&[Lane {
                slot,
                word: word(prompt.len() as u32),
                tokens: &prompt,
            }])
            .expect("a seating prefill fires");
        fed.push(argmax(&out[0]));
    }

    // -- WARM AT THE CEILING BEFORE ANY CAPTURE. `Ctx::scratch` grows by
    //    free+malloc, so a slab that grows BETWEEN two captures moves its
    //    base address and every pointer into it - which reads as a law
    //    break that is really an allocator event. record.rs already states
    //    this doctrine for capture; the probe needs it for the FIT.
    for _ in 0..2 {
        shell.open(3).expect("slot 3 opens");
        shell
            .fire(&[
                Lane { slot: 0, word: word(1), tokens: core::slice::from_ref(&fed[0]) },
                Lane { slot: 1, word: word(1), tokens: core::slice::from_ref(&fed[1]) },
                Lane { slot: 2, word: word(1), tokens: core::slice::from_ref(&fed[2]) },
                Lane { slot: 3, word: word(32), tokens: &filler[..32] },
            ])
            .expect("the ceiling warm fires");
    }
    eprintln!("warmed at the ceiling (3 decode lanes + a 32-token prefill lane)");

    shell.keep_graphs(true);
    shell.set_mode(Graphs::On);

    // -- A: the all-decode topology.
    let mut decode: Vec<(u64, Walked)> = Vec::new();
    let mut decode1: Option<Walked> = None;
    for k in [1usize, 2, 3, 4] {
        let lanes: Vec<Lane<'_>> = (0..k)
            .map(|slot| Lane {
                slot: slot as u32,
                word: word(1),
                tokens: core::slice::from_ref(&fed[slot]),
            })
            .collect();
        let want = shell.kept_graphs().len();
        eprintln!("\n[A] all-decode, {k} lane(s)");
        if let Some(walked) = capture(&mut shell, &lanes, want) {
            census(&format!("A{k}: all-decode x{k}"), &walked);
            if k == 1 {
                decode1 = Some(walked);
            } else {
                decode.push((k as u64, walked));
            }
        }
    }

    // -- B: the mixed topology. 8 -> 16 fits; 24, 12 and 9 verify, and NINE
    //    is the one that breaks a `div_ceil` law the even points hide.
    let mut mixed: Vec<(u64, Walked)> = Vec::new();
    for tokens in [8usize, 16, 24, 12, 9] {
        shell.open(1).expect("slot 1 re-opens");
        let lanes = [
            Lane {
                slot: 0,
                word: word(1),
                tokens: core::slice::from_ref(&fed[0]),
            },
            Lane {
                slot: 1,
                word: word(tokens as u32),
                tokens: &filler[..tokens],
            },
        ];
        let want = shell.kept_graphs().len();
        eprintln!("\n[B] decode + prefill of {tokens}");
        if let Some(walked) = capture(&mut shell, &lanes, want) {
            census(&format!("B{tokens}: decode + prefill({tokens})"), &walked);
            mixed.push((tokens as u64, walked));
        }
    }

    // -- The law fits.
    let mut movers_hint = 0usize;
    eprintln!("\n==== LAW FIT ====");
    if decode.len() >= 3 {
        let f = fit("A", "decode lanes", &decode);
        report("A: all-decode, axis = decode lanes; fit 2->3, verified at 4", &f);
    }
    if mixed.len() >= 3 {
        let f = fit("B", "prefill tokens", &mixed);
        movers_hint = f.moving_nodes;
        report(
            "B: decode+prefill, axis = prefill tokens; fit 8->16, verified at 24, 12 and 9",
            &f,
        );
        // The same fit WITHOUT the odd point, to show what it would have missed.
        let even: Vec<(u64, Walked)> = mixed
            .iter()
            .filter(|(x, _)| x % 4 == 0)
            .map(|(x, w)| (*x, w.clone()))
            .collect();
        if even.len() >= 3 {
            let g = fit("B'", "prefill tokens", &even);
            eprintln!(
                "      (on the EVEN points alone: {} affine, {} unaffine - the difference \
                 is what a lattice-congruent probe set hides)",
                g.affine,
                g.unaffine.len()
            );
        }
    }

    // -- The structural difference: absent, not zero-grid.
    if let (Some(a), Some((_, b))) = (decode1.as_ref(), mixed.first()) {
        structural("all-decode x1", a, "decode+prefill(8)", b);
    }
    if let (Some((_, a)), Some((_, b))) = (decode.first(), mixed.first()) {
        structural("all-decode x2", a, "decode+prefill(8)", b);
    }

    // -- The rebind kill factors.
    let stats = shell.graph_stats();
    eprintln!(
        "\n==== CAPTURE COST ====\n   {} captures, {} replays, {} warming, {:.1} ms of capture \
         in all, {} execs resident",
        stats.captures, stats.replays, stats.warming, stats.capture_millis, stats.execs,
    );
    let movers = movers_hint;
    eprintln!("\n==== REBIND (cudaGraphExecKernelNodeSetParams) ====");
    if let Some((_, graph)) = shell.kept_graphs().last() {
        let exec = graph
            .instantiate(core::ptr::null_mut())
            .expect("the kept graph instantiates a second exec");
        let subset: Vec<usize> = (0..movers).collect();
        let probed = nodes::rebind(&exec, graph, &subset).expect("the rebind probe runs");
        eprintln!(
            "   identity pass: {} nodes in {:.1} us ({:.3} us/node)",
            probed.identity_nodes,
            probed.identity_us,
            probed.identity_us / probed.identity_nodes.max(1) as f64,
        );
        eprintln!(
            "   moving-nodes-only pass: {} nodes in {:.1} us",
            probed.subset_nodes, probed.subset_us,
        );
        eprintln!("   change the GRID:  {:?}", probed.grid);
        eprintln!("   change the SMEM:  {:?}", probed.smem);
        eprintln!("   change an ARG:    {:?}", probed.arg);
        eprintln!(
            "   change the FUNC:  {:?}  <- the arm switch\n      from {} ({} params)\n      to   {} ({} params)",
            probed.func,
            short(&probed.func_from.0),
            probed.func_from.1,
            short(&probed.func_to.0),
            probed.func_to.1,
        );
        eprintln!("   a NULL func:      {:?}  (the control)", probed.null_func);
        eprintln!(
            "   a ZERO grid:      {:?}  <- can an empty window's node be turned OFF?",
            probed.zero_grid
        );
        eprintln!("   a ONE-BLOCK grid: {:?}", probed.one_block);
        let (bytes, millis) =
            nodes::exec_footprint(graph, 8).expect("eight execs instantiate");
        eprintln!(
            "\n   one exec of {} nodes: {:.1} KiB of device memory, {:.2} ms to instantiate",
            graph.nodes(),
            bytes / 1024.0,
            millis,
        );
    }
}

/// How two compositions' graphs differ: which launches one holds and the
/// other does not.
fn structural(a_label: &str, a: &Walked, b_label: &str, b: &Walked) {
    let a_syms = tally(a);
    let b_syms = tally(b);
    let mut absent = 0usize;
    let mut extra = 0usize;
    for (sym, count) in &b_syms {
        absent += count.saturating_sub(a_syms.get(sym).copied().unwrap_or(0));
    }
    for (sym, count) in &a_syms {
        extra += count.saturating_sub(b_syms.get(sym).copied().unwrap_or(0));
    }
    eprintln!(
        "\n==== TOPOLOGY: {a_label} ({} nodes) vs {b_label} ({} nodes) ====",
        a.nodes.len(),
        b.nodes.len(),
    );
    eprintln!(
        "   {absent} launches {b_label} holds that {a_label} does NOT; {extra} the other way"
    );
    let mut only: Vec<(usize, String)> = b_syms
        .iter()
        .filter_map(|(s, c)| {
            let d = c.saturating_sub(a_syms.get(s).copied().unwrap_or(0));
            (d > 0).then(|| (d, short(s)))
        })
        .collect();
    only.sort_by(|x, y| y.0.cmp(&x.0));
    for (d, s) in only.iter().take(14) {
        eprintln!("      only in {b_label}: {d:>4}x {s}");
    }
    let mut back: Vec<(usize, String)> = a_syms
        .iter()
        .filter_map(|(s, c)| {
            let d = c.saturating_sub(b_syms.get(s).copied().unwrap_or(0));
            (d > 0).then(|| (d, short(s)))
        })
        .collect();
    back.sort_by(|x, y| y.0.cmp(&x.0));
    for (d, s) in back.iter().take(8) {
        eprintln!("      only in {a_label}: {d:>4}x {s}");
    }
}

fn tally(walked: &Walked) -> BTreeMap<String, usize> {
    let mut map = BTreeMap::new();
    for node in walked.nodes.iter().filter(|n| n.kernel()) {
        *map.entry(node.symbol.clone()).or_default() += 1;
    }
    map
}
