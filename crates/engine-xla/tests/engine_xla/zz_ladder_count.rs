//! Scratch: the warm ladder's fires and programs for a deployment at a c256 load, over a dry device.

use std::path::Path;

use engine_xla::{Boot, DeviceBoot, Shell};
use poem::Platform;
use poem_compiler::Budget;

fn count(name: &str, lanes: u32, tokens: u32, context: u32, slots: u32) {
    let page = 16;
    let deployment = poem_compiler::catalog::deployment(name).expect("the deployment is listed");
    let trace = deployment.trace(Platform::Xla);
    let pages = std::env::var("ZZ_PAGES")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(u64::from(slots * context / page));
    let paging = engine_xla::store::kv::Paging::of(page, context, slots, pages)
        .expect("the paging lays out");
    let kv_budget = engine_xla::store::pool_demand(&trace, paging).expect("the pools size");
    let contract = checkpoint::contract::ModelContract {
        alignment: 1,
        tensors: Vec::new(),
        groups: Vec::new(),
    };
    let mut shell = Shell::load_dry(
        Boot {
            trace,
            contract: &contract,
            checkpoint: Path::new("/dev/null"),
            budget: Budget {
                max_lanes: lanes,
                max_tokens: tokens,
                buckets: Vec::new(),
                max_adapters: 0,
                readouts_per_lane: poem_compiler::arena::READOUTS_PER_LANE,
            },
            patches: None,
            page_size: page,
            context,
            slots,
            kv_budget: Some(kv_budget),
            memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
            device: &DeviceBoot::default(),
        },
        None,
        false,
    )
    .expect("the dry shell loads");
    let before = shell.device().dry_texts().len();
    let fired = shell.prewarm().expect("the ladder fires");
    let texts = shell.device().dry_texts();
    let new = &texts[before..];
    let mut sizes: Vec<usize> = new.iter().map(String::len).collect();
    sizes.sort_unstable();
    let uniq: std::collections::HashSet<&String> = new.iter().collect();
    if let Ok(dir) = std::env::var("ZZ_DUMP") {
        std::fs::create_dir_all(&dir).unwrap();
        for (i, t) in new.iter().enumerate() {
            std::fs::write(format!("{dir}/p{i:03}.mlir"), t).unwrap();
        }
    }
    eprintln!(
        "ZZ {name} lanes={lanes} tokens={tokens} ctx={context} slots={slots}: fires={fired} texts={} unique={} bytes p50={} max={} total_mib={}",
        new.len(),
        uniq.len(),
        sizes.get(sizes.len() / 2).copied().unwrap_or(0),
        sizes.last().copied().unwrap_or(0),
        sizes.iter().sum::<usize>() >> 20
    );
}

#[test]
fn zz_ladder_counts() {
    let name = std::env::var("ZZ_NAME").unwrap_or_else(|_| "qwen35-d0.8b-bf16-kv-bf16".into());
    let lanes = std::env::var("ZZ_LANES")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(256);
    let ctx = std::env::var("ZZ_CTX")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(2048);
    let slots = std::env::var("ZZ_SLOTS")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(512);
    count(&name, lanes, 2048, ctx, slots);
}

/// Scratch: compile time of a few fire shapes on a compiling dry device. ZZ_FIRE = comma list of
/// pN (one prompt of N rows), dL (L decode lanes at 300 rows), mLxN (L decodes beside an N-row prompt).
#[test]
fn zz_compile_shapes() {
    use engine_xla::{Lane, Seated};
    use poem::Request;
    let Ok(list) = std::env::var("ZZ_FIRE") else {
        return;
    };
    let name = std::env::var("ZZ_NAME").unwrap_or_else(|_| "qwen35-d0.8b-bf16-kv-bf16".into());
    let (page, context, slots) = (16u32, 2048u32, 512u32);
    let deployment = poem_compiler::catalog::deployment(&name).expect("the deployment is listed");
    let trace = deployment.trace(Platform::Xla);
    let paging =
        engine_xla::store::kv::Paging::of(page, context, slots, u64::from(slots * context / page))
            .unwrap();
    let kv_budget = engine_xla::store::pool_demand(&trace, paging).unwrap();
    let contract = checkpoint::contract::ModelContract {
        alignment: 1,
        tensors: Vec::new(),
        groups: Vec::new(),
    };
    let mut shell = Shell::load_dry(
        Boot {
            trace,
            contract: &contract,
            checkpoint: Path::new("/dev/null"),
            budget: Budget {
                max_lanes: 256,
                max_tokens: 2048,
                buckets: Vec::new(),
                max_adapters: 0,
                readouts_per_lane: poem_compiler::arena::READOUTS_PER_LANE,
            },
            patches: None,
            page_size: page,
            context,
            slots,
            kv_budget: Some(kv_budget),
            memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
            device: &DeviceBoot::default(),
        },
        None,
        std::env::var_os("ZZ_COMPILE").is_some(),
    )
    .expect("the dry shell loads");
    let facts = shell.trace().facts.clone();
    let pages: Vec<u32> = (0..context / page).collect();
    let tokens = vec![7u32; 2048];
    for item in list.split(',') {
        let (decodes, rows) = match item.as_bytes()[0] {
            b'p' => (0u32, item[1..].parse::<u32>().unwrap()),
            b'd' => (item[1..].parse::<u32>().unwrap(), 0),
            _ => {
                let (l, n) = item[1..].split_once('x').unwrap();
                (l.parse().unwrap(), n.parse().unwrap())
            }
        };
        for s in 0..decodes + 1 {
            shell.open(s % slots).unwrap();
        }
        let mut seated: Vec<Seated<'_>> = (0..decodes)
            .map(|lane| {
                let mut s = Seated::of(Lane {
                    slot: lane % slots,
                    word: facts.word(&Request::new(1, false)),
                    tokens: &tokens[..1],
                });
                s.pages = &pages[..301u32.div_ceil(page) as usize];
                s.held = Some(300);
                s
            })
            .collect();
        if rows > 0 {
            let mut s = Seated::of(Lane {
                slot: decodes % slots,
                word: facts.word(&Request::new(rows, false)),
                tokens: &tokens[..rows as usize],
            });
            s.pages = &pages[..rows.div_ceil(page) as usize];
            s.held = Some(0);
            seated.push(s);
        }
        let before = shell.device().dry_texts().len();
        let t = std::time::Instant::now();
        shell.fire_full(&seated).expect("the shape fires");
        let texts = shell.device().dry_texts();
        eprintln!(
            "ZZC {item}: {:.1} s, {} new texts, bytes {:?}",
            t.elapsed().as_secs_f64(),
            texts.len() - before,
            texts[before..].iter().map(String::len).collect::<Vec<_>>()
        );
    }
}
