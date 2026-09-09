#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_metal::experts::{Attachments, Plan};
use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};
use model_ir::Trace;

const SKU: &str = "dsv4-flash-full-mtp-u4g64-u2g64-mxfp4-kv-bf16";

const REPO: &str = "models--mlx-community--DeepSeek-V4-Flash-2bit-DQ";

fn seats() -> u32 {
    std::env::var("PIE_U2_FULL_SEATS")
        .ok()
        .and_then(|it| it.parse().ok())
        .unwrap_or(40)
}

const PROMPT: &str = "The capital of France is";

const STEPS: usize = 16;

fn wired() -> Option<u64> {
    let said = std::process::Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8(said.stdout).ok()?;
    let mut page = 4096u64;
    let mut pages = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics: (page size of ")
            && let Some(n) = rest.split_whitespace().next()
        {
            page = n.parse().unwrap_or(page);
        }
        if let Some(rest) = line.strip_prefix("Pages wired down:") {
            pages = rest.trim().trim_end_matches('.').parse::<u64>().ok();
        }
    }
    Some(pages? * page)
}

fn swap() -> Option<u64> {
    let said = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("vm.swapusage")
        .output()
        .ok()?;
    let text = String::from_utf8(said.stdout).ok()?;
    let used = text.split("used = ").nth(1)?;
    let megabytes: f64 = used.trim_start().split('M').next()?.parse().ok()?;
    Some((megabytes * (1u64 << 20) as f64) as u64)
}

fn gib(bytes: Option<u64>) -> String {
    match bytes {
        Some(bytes) => format!("{:.3} GiB", bytes as f64 / (1u64 << 30) as f64),
        None => "unavailable".to_string(),
    }
}

fn delta(before: Option<u64>, after: Option<u64>) -> String {
    match (before, after) {
        (Some(before), Some(after)) => format!(
            "{:+.3} GiB",
            (after as f64 - before as f64) / (1u64 << 30) as f64
        ),
        _ => "unavailable".to_string(),
    }
}

fn artifact() -> Option<PathBuf> {
    let stamped = |path: &Path| {
        let stamp = checkpoint::file::serve::stamp_of(path).ok().flatten()?;
        (stamp.backend == "metal" && stamp.sku == SKU).then(|| path.to_path_buf())
    };
    if let Ok(stated) = std::env::var("PIE_METAL_FULL_ARTIFACT") {
        return stamped(Path::new(&stated));
    }
    let homes = [
        format!("{}/models", std::env::var("PIE_HOME").unwrap_or_default()),
        format!("{}/.pie/models", std::env::var("HOME").unwrap_or_default()),
    ];
    homes.iter().find_map(|home| {
        let mut found = walk(Path::new(home));
        found.sort();
        found.iter().find_map(|path| stamped(path))
    })
}

fn walk(at: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(at).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|it| it == "zt") {
            out.push(path);
        } else if path.is_dir() {
            out.extend(
                std::fs::read_dir(&path)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().is_some_and(|it| it == "zt")),
            );
        }
    }
    out
}

fn tokenizer_file() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_U2_FULL_SNAPSHOT") {
        let path = PathBuf::from(stated).join("tokenizer.json");
        return path.is_file().then_some(path);
    }
    let homes = [std::env::var("HOME").unwrap_or_default()];
    homes.iter().find_map(|home| {
        let snapshots = Path::new(home)
            .join(".cache/huggingface/hub")
            .join(REPO)
            .join("snapshots");
        let mut found: Vec<PathBuf> = std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path().join("tokenizer.json")))
            .filter(|path| path.is_file())
            .collect();
        found.sort();
        found
            .iter()
            .find(|path| {
                path.parent()
                    .and_then(Path::file_name)
                    .and_then(|it| it.to_str())
                    .is_some_and(|name| {
                        name.len() == 40 && name.chars().all(|c| c.is_ascii_hexdigit())
                    })
            })
            .or_else(|| found.first())
            .cloned()
    })
}

struct Read {
    trace: Trace,
    contract: checkpoint::contract::ModelContract,
    planes: Attachments,
}

fn read(artifact: &Path) -> Read {
    let trace = (models::sku(SKU)
        .expect("the catalog ships the full 2-bit row")
        .trace)(Platform::Metal);
    let source = ztensor_compat::index(artifact).expect("the artifact opens");
    let contract = checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Metal)
        .unwrap_or_else(|why| panic!("the artifact holds every plane of {SKU}: {why}"));
    drop(source);
    let planes = engine_metal::weights::attachments(&trace, &contract, artifact)
        .expect("the load plan pairs this artifact's quantized banks");
    Read {
        trace,
        contract,
        planes,
    }
}

fn seating(read: &Read, want: u32) -> Plan {
    let full = Plan::of(&read.trace, &read.planes, None)
        .expect("an uncapped plan is full residency")
        .device_demand();
    let plan_at = |budget: u64| Plan::of(&read.trace, &read.planes, Some(budget));
    let (mut lo, mut hi) = (0u64, full - 1);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match plan_at(mid) {
            Ok(plan) if plan.slots() >= want => hi = mid,
            _ => lo = mid + 1,
        }
    }
    let plan = plan_at(lo).unwrap_or_else(|why| {
        panic!("no budget under {full} seats {want} of this artifact's experts: {why}")
    });
    assert!(
        plan.slots() >= want,
        "the bisected budget seats what was asked"
    );
    assert!(
        plan.streams(),
        "a plan that holds nothing back is not a slab"
    );
    plan
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

fn finite_and_spread(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    let bad = logits.iter().position(|value| !value.is_finite());
    assert!(
        bad.is_none(),
        "{what} logit {} is {}, and a single NaN means the whole row is noise",
        bad.unwrap_or(0),
        logits[bad.unwrap_or(0)],
    );
    let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
    for value in logits {
        low = low.min(*value);
        high = high.max(*value);
    }
    assert!(
        high - low > 1.0,
        "{what} logits span only {} over {} classes, which is a row nothing chose \
         from rather than an answer",
        high - low,
        logits.len(),
    );
}

fn word(len: u32) -> u64 {
    models::deepseek_v4::forward::Facts::of(&Request::new(len, false)).word()
}

struct Run {
    tokens: Vec<u32>,
    load_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    warm: bool,
    windows: usize,
    kind: Option<&'static str>,
    source: Option<(u64, u64)>,
    motion: (u64, u64),
    slabs: usize,
    boot_wired: String,
    fire_wired: String,
    swap_after: Option<u64>,
}

fn run(what: &str, artifact: &Path, residency: Plan, prompt: &[u32]) -> Option<Run> {
    let read = read(artifact);
    let idle = wired();
    let at = Instant::now();
    let shell = Shell::load(Boot {
        voxels: None,
        trace: read.trace.clone(),
        contract: &read.contract,
        checkpoint: artifact,
        budget: Budget::new(4, 512),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        pages: (4) * (512) / (16),
        runahead: engine::runahead::Runahead::F1,
        residency,
    });
    let mut shell = match shell {
        Ok(shell) => shell,
        Err(engine_metal::Fault::Residency(said)) => {
            eprintln!("skipping: this machine cannot hold {what} — {said}");
            return None;
        }
        Err(other) => panic!("the {what} shell loads: {other}"),
    };
    let load_ms = at.elapsed().as_secs_f64() * 1000.0;
    let booted = wired();

    shell.open(0).expect("the slot opens");
    let at = Instant::now();
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .unwrap_or_else(|why| panic!("the {what} prefill fires: {why}"));
    let prefill_ms = at.elapsed().as_secs_f64() * 1000.0;
    let fired = wired();
    finite_and_spread(&prefill[0], what);
    let mut tokens = vec![argmax(&prefill[0])];

    let at = Instant::now();
    for step in 0..STEPS {
        let fed = [*tokens.last().expect("a step feeds the last token back")];
        let decode = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("{what} decode step {step} fires: {why}"));
        finite_and_spread(&decode[0], what);
        tokens.push(argmax(&decode[0]));
    }
    let decode_ms = at.elapsed().as_secs_f64() * 1000.0;

    Some(Run {
        tokens,
        load_ms,
        prefill_ms,
        decode_ms,
        warm: shell.weights_warm(),
        windows: shell.weight_windows(),
        kind: shell.expert_source_kind(),
        source: shell.expert_source(),
        motion: shell.expert_motion(),
        slabs: shell.expert_residency().len(),
        boot_wired: delta(idle, booted),
        fire_wired: delta(booted, fired),
        swap_after: swap(),
    })
}

#[test]
fn the_full_dsv4_artifact_loads_warm_streams_its_experts_and_answers_twice_the_same() {
    if !engine_metal::device::present() {
        eprintln!("skipping: this machine publishes no Metal device");
        return;
    }
    let Some(artifact) = artifact() else {
        eprintln!(
            "skipping: no `metal`-stamped {SKU} artifact found — import one with an \
             ENGINE-METAL-FEATURE binary (see this file's header; `-p pie`, never a \
             workspace build) and name it in PIE_METAL_FULL_ARTIFACT"
        );
        return;
    };
    let Some(vocabulary) = tokenizer_file() else {
        eprintln!(
            "skipping: no tokenizer.json beside a {REPO} snapshot — `--consume-source` \
             keeps it, so this is a cache that was cleared rather than consumed; name \
             the directory in PIE_U2_FULL_SNAPSHOT"
        );
        return;
    };
    let bytes = std::fs::metadata(&artifact).map(|it| it.len()).unwrap_or(0);
    eprintln!(
        "artifact {artifact:?} ({:.1} GiB)\ntokenizer {vocabulary:?}",
        bytes as f64 / (1u64 << 30) as f64
    );
    eprintln!("idle: wired {} swap {}", gib(wired()), gib(swap()));

    let tokenizer =
        tokenizer::Tokenizer::from_file(&vocabulary).expect("the checkpoint's tokenizer loads");
    let prompt = tokenizer.encode(PROMPT);
    assert!(
        !prompt.is_empty(),
        "the prompt encodes to at least one token"
    );

    let sized = read(&artifact);
    for want in [16u32, 24, 32, 40, 48, 64] {
        let plan = seating(&sized, want);
        eprintln!(
            "  {want:>3} seats -> {} slots, {:.2} GiB device demand",
            plan.slots(),
            plan.device_demand() as f64 / (1u64 << 30) as f64,
        );
    }
    let want = seats();
    let plan = seating(&sized, want);
    eprintln!(
        "firing with {} seats ({:.2} GiB device demand); a segment of {} token(s) can \
         route to at most {} distinct experts",
        plan.slots(),
        plan.device_demand() as f64 / (1u64 << 30) as f64,
        prompt.len(),
        prompt.len() * 6,
    );
    eprintln!(
        "the streamed plan seats {} experts across {} groups, {} bands; device demand {:.2} GiB",
        plan.slots(),
        plan.groups().len(),
        plan.bands().len(),
        plan.device_demand() as f64 / (1u64 << 30) as f64,
    );

    let ceiling = engine_metal::device::Context::bind()
        .expect("the device binds")
        .max_buffer();
    let least = bytes.div_ceil(ceiling);
    eprintln!(
        "maxBufferLength {ceiling} ({:.2} GiB) against an artifact of {bytes} ({:.2} \
         GiB): one `MTLBuffer` holds {:.1}% of this row, so the mapping binds as at \
         least {least} window(s) cut at the manifest's own blob boundaries",
        ceiling as f64 / (1u64 << 30) as f64,
        bytes as f64 / (1u64 << 30) as f64,
        100.0 * ceiling as f64 / bytes as f64,
    );

    let Some(first) = run("run-1", &artifact, plan.clone(), &prompt) else {
        return;
    };
    let Some(second) = run("run-2", &artifact, plan.clone(), &prompt) else {
        return;
    };

    for (what, at) in [("run-1", &first), ("run-2", &second)] {
        eprintln!(
            "{what}  load {:>8.0} ms  prefill {:>7.1} ms ({:.1} tok/s)  decode {:>7.1} ms \
             ({:.1} tok/s)\n       warm={} windows={} source={} {:?} motion={:?} \
             slabs={}  wired boot {} first fire {}  swap {}",
            at.load_ms,
            at.prefill_ms,
            prompt.len() as f64 / (at.prefill_ms / 1000.0),
            at.decode_ms,
            STEPS as f64 / (at.decode_ms / 1000.0),
            at.warm,
            at.windows,
            at.kind.unwrap_or("none"),
            at.source,
            at.motion,
            at.slabs,
            at.boot_wired,
            at.fire_wired,
            gib(at.swap_after),
        );
    }
    eprintln!("run-1 tokens {:?}", first.tokens);
    eprintln!("run-1 text   {:?}", tokenizer.decode(&first.tokens, false));

    assert!(
        first.warm,
        "the {SKU} artifact did not take the warm arm, so this load read 89.9 GiB into \
         a host store on a 32 GiB box rather than mapping it"
    );

    assert!(
        first.windows >= least as usize,
        "the {SKU} artifact bound as {} window(s) against a {ceiling}-byte          `maxBufferLength` and {bytes} bytes of file, which needs at least {least}",
        first.windows,
    );
    assert_eq!(
        first.windows, second.windows,
        "two loads of one artifact cut it into different numbers of windows"
    );

    assert!(
        first.slabs > 0,
        "a load with no slab is a full-residency load wearing the word, and full \
         residency of this bank does not fit this box"
    );
    assert_eq!(
        first.kind,
        Some("artifact"),
        "the seats came from {:?} rather than from the artifact's own mapping, which \
         means this load staged a second copy of the expert bank",
        first.kind,
    );

    assert_eq!(
        first.tokens, second.tokens,
        "two loads of one artifact answered differently, so something in the seat \
         bookkeeping is reading state a load does not own"
    );
    assert_eq!(
        first.tokens.len(),
        STEPS + 1,
        "the prefill's token and one per decode step"
    );
}
