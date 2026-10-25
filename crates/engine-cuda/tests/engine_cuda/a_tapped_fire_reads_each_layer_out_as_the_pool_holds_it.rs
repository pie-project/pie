#![cfg(feature = "device")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Lane, Seated, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{ParamSource, Trace};

const PAGE: u32 = 16;
const CONTEXT: u32 = 256;
const TOKENS: u32 = 64;

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn write_checkpoint(path: &Path, trace: &Trace) {
    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    for param in planes {
        let count: usize = param.shape.iter().product::<u64>() as usize;
        let norm = param.name.ends_with("norm");
        let mut bytes = Vec::with_capacity(count * 2);
        let mut seed = fnv(&param.name);
        for _ in 0..count {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value: f32 = if norm { 1.0 + 0.05 * unit } else { 0.08 * unit };
            bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        writer
            .add(
                param.name.as_str(),
                param.shape.clone(),
                ztensor::Leaf::BF16,
                &bytes,
            )
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

fn load(trace: &Trace, contract: &ModelContract, container: &Path) -> Shell {
    Shell::load(Boot {
        trace: trace.clone(),
        contract,
        checkpoint: container,
        budget: Budget::new(1, TOKENS),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: 1,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the micro trace loads")
}

/// A prefill's pages, tapped, come out a layer at a time in the layers'
/// order, and laid back out they are the bytes a move to the host tier
/// after the fire reads.
#[test]
fn a_tapped_fire_reads_each_layer_out_as_the_pool_holds_it() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let trace = poem_compiler::catalog::trace_of(
        "qwen3-a3b-micro",
        Dtype::Bf16,
        Dtype::Bf16,
        Platform::Cuda,
    );
    let dir = std::env::temp_dir().join(format!("pie-tap-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let _scratch = Scratch(dir.clone());
    let container = dir.join("micro.zt");
    write_checkpoint(&container, &trace);
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .expect("the fixture's planes are the plan's own");
    drop(source);

    let mut shell = load(&trace, &contract, &container);
    shell
        .seat_host_kv(None, 1, CONTEXT, 1)
        .expect("a host tier seats");
    let groups = shell.tap_groups().expect("a full-attention load taps");
    let told: Arc<Mutex<Vec<engine::Tapped>>> = Arc::default();
    shell.on_tapped({
        let told = told.clone();
        Arc::new(move |tapped| told.lock().unwrap().push(tapped))
    });
    shell.open(0).expect("slot 0 opens");

    let pages: Vec<u32> = (0..TOKENS / PAGE).collect();
    let slots: Vec<u32> = (0..TOKENS / PAGE).collect();
    let tap = engine::fire::Tap {
        tag: 7,
        pages: pages.clone(),
        slots,
    };
    let prompt: Vec<u32> = (0..TOKENS).map(|t| (t * 7 + 11) % 2048).collect();
    let lane = Lane {
        slot: 0,
        word: trace.facts.word(&Request::new(TOKENS, false)),
        tokens: &prompt,
    };
    let seated = Seated {
        pages: &pages,
        held: Some(0),
        tap: Some(&tap),
        ..Seated::of(lane)
    };
    shell
        .fire_media(&[seated], &[], &[])
        .expect("the tapped prefill fires");

    let started = std::time::Instant::now();
    while told.lock().unwrap().len() < groups.len() {
        assert!(
            started.elapsed().as_secs() < 10,
            "{} of {} groups told",
            told.lock().unwrap().len(),
            groups.len()
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let told = std::mem::take(&mut *told.lock().unwrap());
    assert_eq!(
        told.iter().map(|t| t.group).collect::<Vec<_>>(),
        (0..groups.len() as u32).collect::<Vec<_>>(),
        "a group a layer, in the layers' order"
    );

    let n = pages.len();
    let page_bytes: u64 = groups.iter().flatten().map(|(_, len)| len).sum();
    let mut laid = vec![0u8; n * page_bytes as usize];
    for tapped in &told {
        assert_eq!(tapped.tag, 7);
        let bytes = tapped.bytes.as_ref().expect("every group read out");
        let mut at = 0usize;
        for page in 0..n {
            for &(offset, len) in &groups[tapped.group as usize] {
                let to = page * page_bytes as usize + offset as usize;
                laid[to..to + len as usize].copy_from_slice(&bytes[at..at + len as usize]);
                at += len as usize;
            }
        }
        assert_eq!(
            at,
            bytes.len(),
            "group {} is its pages' spans",
            tapped.group
        );
    }

    let moved: Vec<u32> = (8..8 + n as u32).collect();
    shell
        .swap_kv(true, &pages, &moved, &[])
        .expect("the pages move to the host tier");
    let after = shell.read_host_kv(&moved).expect("the moved pages read");
    assert!(
        after.iter().any(|&byte| byte != 0),
        "the prefill wrote its pages"
    );
    assert!(
        laid == after,
        "the tapped layers are the pages the fire left"
    );
}
