//! The A/B gate: an eager walk and an indirect command buffer, byte for byte
//! (`.wiki/palo/icb.md` §6, §7 steps 3 and 4).
//!
//! **WHAT WOULD FAIL LOUDEST, WRITTEN FIRST.** The claim is that ONE indirect
//! command buffer, encoded once, serves every composition — so the gate is
//! two compositions through one buffer with no re-encode between them, and
//! their tokens against the tokens the ordinary encode path produces. Build
//! log 10's exec key was the per-class `(rows, lanes)` vector; here the two
//! compositions have different vectors and there is no key to consult.
//!
//! ```text
//! cargo test -p driver-metal --release --test indirect_ab -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use driver_metal::{Boot, Lane, Shell};
use model_compiler::Budgets;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";
const PROMPT: &str = "The capital of France is";

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn word(query_len: u32) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let snapshots =
            Path::new(home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
        std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .find(|path| path.join("tokenizer.json").exists())
    })
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

fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    if !driver_metal::device::present() {
        eprintln!("skipping {what}: this machine publishes no Metal device");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!("skipping {what}: no Qwen3.5-0.8B snapshot in the hugging face cache");
        return None;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU");
    let plan = trace(Platform::Metal);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);
    let shell = Shell::load(Boot {
        plan,
        contract: &contract,
        checkpoint: &checkpoint,
        budgets: Budgets::new(8, 256),
        profile: None,
        page_size: 16,
        context: 512,
        slots: 8,
    })
    .expect("the shell loads");
    Some((shell, tokenizer))
}

/// Greedy: the highest logit.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

#[test]
fn one_indirect_command_buffer_serves_two_compositions_and_both_say_what_the_eager_walk_says() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the indirect a/b gate") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);

    // THE TWO COMPOSITIONS. `A` is design §0's headline case — a decode lane
    // beside a prefill lane — and `B` is all-decode. Their `(rows, lanes)`
    // vectors differ in both classes, which is exactly the pair build log 10's
    // exec key would have keyed apart into two graphs.
    //
    // The buffer is built at A, because A holds every class: the walk skips a
    // zero-row region's nodes, so an all-decode build would hold no prefill
    // launch at all.
    let mut eager: Vec<Vec<u32>> = Vec::new();
    let mut indirect: Vec<Vec<u32>> = Vec::new();

    for pass in 0..2 {
        // A fresh pair of slots each pass, so the second pass is not
        // continuing the first pass's sequences.
        for slot in 0..4 {
            shell.open(slot).expect("the slot opens");
        }
        // Prefill both lanes so the decode fires have history.
        let seed: Vec<u32> = prompt.clone();
        let a = shell
            .fire(&[
                Lane { slot: 0, word: word(seed.len() as u32), tokens: &seed },
                Lane { slot: 1, word: word(seed.len() as u32), tokens: &seed },
            ])
            .expect("the seeding prefill fires");
        let next: Vec<u32> = a.iter().map(|row| argmax(row)).collect();

        if pass == 1 && shell.icb().is_none() {
            // Built at the MIXED composition — one prefill lane and one
            // decode lane — because that is the composition that holds every
            // class.
            let one = [next[0]];
            shell
                .build_icb(&[
                    Lane { slot: 0, word: word(seed.len() as u32), tokens: &seed },
                    Lane { slot: 1, word: word(1), tokens: &one },
                ])
                .expect("the indirect command buffer builds");
            let icb = shell.icb().expect("it is there now");
            eprintln!(
                "built {} slots, {} resident reservations, {:.1} KiB of scalar arena",
                icb.len(),
                icb.residents(),
                icb.constant_bytes() as f64 / 1024.0
            );
            // The build consumed the same rows the eager pass did; reopen so
            // the two passes see the same history.
            for slot in 0..4 {
                shell.open(slot).expect("the slot reopens");
            }
            let a = shell
                .fire(&[
                    Lane { slot: 0, word: word(seed.len() as u32), tokens: &seed },
                    Lane { slot: 1, word: word(seed.len() as u32), tokens: &seed },
                ])
                .expect("the seeding prefill fires again");
            let _ = a;
        }

        let fire = |shell: &mut Shell, lanes: &[Lane<'_>]| -> Vec<Vec<f32>> {
            if pass == 0 {
                shell.fire(lanes).expect("the eager fire")
            } else {
                shell.fire_indirect(lanes).expect("the indirect fire")
            }
        };

        // COMPOSITION A: a decode lane beside a prefill lane — design §0's
        // headline case, and the composition the buffer was built at.
        let one = [next[0]];
        let mixed = fire(
            &mut shell,
            &[
                Lane { slot: 2, word: word(seed.len() as u32), tokens: &seed },
                Lane { slot: 1, word: word(1), tokens: &one },
            ],
        );
        // COMPOSITION B: ALL DECODE. This is the gate `.wiki/palo/icb.md` §6
        // says to write first, because it is the one that would fail loudest:
        // the walk skips the prefill region entirely, so B dispatches strictly
        // fewer launches than the buffer holds, and the slots it does not want
        // have to be turned off rather than removed. Under build log 10's exec
        // key this was a second graph.
        let two = [argmax(&mixed[1]), next[1]];
        let decodes = fire(
            &mut shell,
            &[
                Lane { slot: 1, word: word(1), tokens: &two[..1] },
                Lane { slot: 0, word: word(1), tokens: &two[1..] },
            ],
        );
        // COMPOSITION C: back to mixed, from the same buffer — the slots B
        // turned off have to come back, which is the half of the mechanism a
        // one-way test would not reach.
        let three = [argmax(&decodes[0])];
        let other = fire(
            &mut shell,
            &[
                Lane { slot: 3, word: word(seed.len() as u32), tokens: &seed },
                Lane { slot: 1, word: word(1), tokens: &three },
            ],
        );

        let produced: Vec<Vec<u32>> = mixed
            .iter()
            .chain(decodes.iter())
            .chain(other.iter())
            .map(|row| vec![argmax(row)])
            .collect();
        if pass == 0 {
            eager = produced;
        } else {
            indirect = produced;
        }
    }

    eprintln!("eager    {eager:?}");
    eprintln!("indirect {indirect:?}");
    assert_eq!(
        eager, indirect,
        "one indirect command buffer, two compositions, and the tokens are the eager \
         walk's — or they are not"
    );
}

#[test]
fn ms_per_fire_eager_and_indirect() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the indirect perf gate") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    for slot in 0..4 {
        shell.open(slot).expect("the slot opens");
    }
    let a = shell
        .fire(&[
            Lane { slot: 0, word: word(prompt.len() as u32), tokens: &prompt },
            Lane { slot: 1, word: word(prompt.len() as u32), tokens: &prompt },
        ])
        .expect("the seeding prefill fires");
    let next: Vec<u32> = a.iter().map(|row| argmax(row)).collect();
    let one = [next[0]];
    let two = [next[0], next[1]];

    // The buffer is built at the MIXED composition, because that is the one
    // that holds every class.
    shell
        .build_icb(&[
            Lane { slot: 2, word: word(prompt.len() as u32), tokens: &prompt },
            Lane { slot: 1, word: word(1), tokens: &one },
        ])
        .expect("the indirect command buffer builds");
    eprintln!(
        "built {} slots",
        shell.icb().expect("built").len()
    );

    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        v[v.len() / 2]
    };

    // THE STEADY STATE, which is the case that matters: an all-decode fire,
    // repeated. Two lanes, one token each — what a serving loop does between
    // admissions.
    let mut eager_ms = Vec::new();
    for _ in 0..40 {
        let at = Instant::now();
        shell
            .fire(&[
                Lane { slot: 0, word: word(1), tokens: &two[..1] },
                Lane { slot: 1, word: word(1), tokens: &two[1..] },
            ])
            .expect("the eager decode");
        eager_ms.push(at.elapsed().as_secs_f64() * 1000.0);
    }
    let mut indirect_ms = Vec::new();
    for _ in 0..40 {
        let at = Instant::now();
        shell
            .fire_indirect(&[
                Lane { slot: 0, word: word(1), tokens: &two[..1] },
                Lane { slot: 1, word: word(1), tokens: &two[1..] },
            ])
            .expect("the indirect decode");
        indirect_ms.push(at.elapsed().as_secs_f64() * 1000.0);
    }
    let e = median(&mut eager_ms);
    let i = median(&mut indirect_ms);
    eprintln!(
        "all-decode ms/fire on {}: eager {e:.3}, indirect {i:.3} — {:+.1}% ({:+.3} ms); \
         last rebind moved {}",
        shell.device_name(),
        (i - e) / e * 100.0,
        i - e,
        shell.rebound()
    );

    // WHAT IS LEFT, AND IT IS THE NUMBER THAT SAYS WHERE THE NEXT STEP IS.
    // An indirect fire still WALKS: the recording that drives the rebind is
    // host arithmetic over the template, and only the encode is gone. Timing
    // the walk alone says how much a device-side rebind — `crate::abi`'s law
    // table read by a shader — would remove on top of what the buffer already
    // removed.
    let mut walk_ms = Vec::new();
    for _ in 0..40 {
        let at = Instant::now();
        shell
            .record(&[
                Lane { slot: 0, word: word(1), tokens: &two[..1] },
                Lane { slot: 1, word: word(1), tokens: &two[1..] },
            ])
            .expect("the walk records");
        walk_ms.push(at.elapsed().as_secs_f64() * 1000.0);
    }
    eprintln!(
        "the walk alone (no encode, no dispatch): {:.3} ms of the fire",
        median(&mut walk_ms)
    );

    // And the mixed composition, which is the one the buffer was built at and
    // therefore the one that turns nothing on or off.
    let mut eager_mixed = Vec::new();
    for _ in 0..24 {
        shell.open(2).expect("the slot reopens");
        let at = Instant::now();
        shell
            .fire(&[
                Lane { slot: 2, word: word(prompt.len() as u32), tokens: &prompt },
                Lane { slot: 1, word: word(1), tokens: &one },
            ])
            .expect("the eager fire");
        eager_mixed.push(at.elapsed().as_secs_f64() * 1000.0);
    }
    let mut indirect_mixed = Vec::new();
    for _ in 0..24 {
        shell.open(2).expect("the slot reopens");
        let at = Instant::now();
        shell
            .fire_indirect(&[
                Lane { slot: 2, word: word(prompt.len() as u32), tokens: &prompt },
                Lane { slot: 1, word: word(1), tokens: &one },
            ])
            .expect("the indirect fire");
        indirect_mixed.push(at.elapsed().as_secs_f64() * 1000.0);
    }
    let e = median(&mut eager_mixed);
    let i = median(&mut indirect_mixed);
    eprintln!(
        "mixed ms/fire: eager {e:.3}, indirect {i:.3} — {:+.1}% ({:+.3} ms); last rebind \
         moved {}",
        (i - e) / e * 100.0,
        i - e,
        shell.rebound()
    );
}
