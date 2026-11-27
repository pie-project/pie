//! A load that splits work with the Neural Engine walks both forms of a fire
//! it may share: the split one, and the GPU alone, which serves once the
//! split is retired. Neither meets a pipeline the walk did not build, and the
//! walk hands the Neural Engine nothing and stages no weights, so a walked
//! load's split fires land, to the bit, what a load never walked lands.
#![cfg(target_vendor = "apple")]

use poem_compiler::Budget;
use engine_metal::chip::{AneMlp, AneProjections};
use engine_metal::{Lane, Seated, Shell};

use super::a_scattered_pool_serves_the_same_logits::{Fixture, Load, Profiles, deployed_fixture, word};

/// A prompt of `rows` rows from an empty seat, its last row read out.
fn fire(shell: &mut Shell, fixture: &Fixture, rows: u32) -> Vec<f32> {
    let tokens: Vec<u32> = (0..rows).map(|at| (11 + at * 37) % 256).collect();
    let mut out = shell
        .fire_seated(&[Seated {
            held: Some(0),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, rows),
                tokens: &tokens,
            })
        }])
        .unwrap_or_else(|why| panic!("{rows} rows are served: {why}"));
    out.remove(0)
}

fn rms(got: &[f32], want: &[f32]) -> f32 {
    let sum = |of: &mut dyn Iterator<Item = f32>| of.map(|v| v * v).sum::<f32>();
    let off = sum(&mut got.iter().zip(want).map(|(a, b)| a - b));
    (off / sum(&mut want.iter().copied())).sqrt()
}

fn a_split_load(model: &str, mlp: bool, rows: &[u32]) {
    let fixture = deployed_fixture(model);
    let profiles = Profiles::new("walked-split");
    let load = || {
        Load {
            budget: Budget::new(2, 1024),
            context: 1024,
            slots: 2,
            profiles: Some(profiles.path()),
            ..Load::default()
        }
        .boot(&fixture)
        .and_then(|mut shell| shell.raise_whole().map(|()| shell))
        .unwrap_or_else(|why| panic!("the shell loads: {why}"))
    };
    let unwalked: Vec<Vec<f32>> = {
        let mut shell = load();
        let chip = shell.tuning().profile();
        let gpu = if mlp {
            matches!(chip.get::<AneMlp>(), AneMlp::Gpu)
        } else {
            matches!(chip.get::<AneProjections>(), AneProjections::Gpu)
        };
        if gpu {
            eprintln!("{model} not asked: this device's profile splits nothing");
            return;
        }
        shell.open(0).expect("a slot opens");
        rows.iter().map(|&n| fire(&mut shell, &fixture, n)).collect()
    };
    let mut shell = load();
    assert!(shell.neural_engine().is_some(), "{model} splits as the device profile states");
    shell.warm_pipelines().expect("the walk runs");
    shell.seal_pipelines().expect("the pipelines seal");
    let splits = |shell: &Shell| shell.neural_engine().expect("the split").splits();
    assert_eq!(splits(&shell), 0, "the walk hands nothing over");
    shell.open(0).expect("a slot opens");
    let built = shell.compiled();
    let shared: Vec<Vec<f32>> = rows.iter().map(|&n| fire(&mut shell, &fixture, n)).collect();
    for ((&n, got), want) in rows.iter().zip(&shared).zip(&unwalked) {
        assert!(got.iter().all(|v| v.is_finite()), "{model} at {n} rows is finite");
        assert_eq!(got, want, "{model} at {n} rows, walked and never walked");
    }
    assert_eq!(splits(&shell) > 0, mlp, "the MLPs are handed over where it has any");
    assert_eq!(shell.take_unbuilt_pipelines(), 0, "a shared fire met an unbuilt pipeline");
    shell
        .neural_engine()
        .expect("the split")
        .handoff()
        .fail("retired by the test");
    let handed = splits(&shell);
    for (&n, shared) in rows.iter().zip(&shared) {
        let alone = fire(&mut shell, &fixture, n);
        assert!(alone.iter().all(|v| v.is_finite()), "{model} at {n} rows alone is finite");
        eprintln!("{model} {n} rows, shared against alone: relative RMS {}", rms(shared, &alone));
        assert_ne!(*shared, alone, "{model} at {n} rows was shared");
    }
    assert_eq!(splits(&shell), handed, "a retired split hands nothing over");
    assert_eq!(shell.take_unbuilt_pipelines(), 0, "a retired fire met an unbuilt pipeline");
    assert_eq!(shell.compiled(), built, "a fire built a pipeline past the seal");
}

#[test]
fn a_walked_split_shares_with_the_neural_engine() {
    let _device = crate::device_alone();
    if !engine_metal::device::present() || kernels_metal::ane::available().is_err() {
        eprintln!("skipping: no Metal device or Neural Engine");
        return;
    }
    // Whole procedures; every count of rows past one the GPU runs alone (257
    // to 288); rows past one the Neural Engine's program pads (300, 800); a
    // tail past a later procedure (544).
    let tails: Vec<u32> = (256..=288).chain([300, 512, 544, 800]).collect();
    a_split_load("qwen35-d0.8b-u4g64-kv-bf16", true, &tails);
    a_split_load("gemma4-e4b-mini-l1-u4g64-kv-bf16", false, &[256, 272, 300, 512]);
}
