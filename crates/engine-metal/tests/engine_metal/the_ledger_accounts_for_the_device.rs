//! The ledger a shell reads off its owners (`Shell::ledger`) against what
//! the device holds for the process (`currentAllocatedSize`): after the load
//! and after fires, every device byte but the driver's own is entered.

use engine_metal::Lane;
use engine_metal::serve::Seated;

use super::a_scattered_pool_serves_the_same_logits::{argmax, fixture_of, load_with, word};

const CONTEXT: u32 = 256;

/// What the device holds beside the entries, in bytes: one 16 MiB placement
/// heap's worth, which the driver's own allocations stay well inside.
const SLACK: i128 = 16 << 20;

fn audited(shell: &engine_metal::Shell, when: &str) -> i128 {
    let audit = shell.audit();
    eprintln!(
        "{when}: {}; device {} MiB, gap {} KiB",
        shell.ledger(),
        audit.device >> 20,
        audit.gap() / 1024
    );
    audit.gap()
}

#[test]
fn the_ledger_accounts_for_the_device() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load_with(&fixture, CONTEXT);
    let mut gaps = vec![("after the load", audited(&shell, "after the load"))];

    let prompts: [Vec<u32>; 2] = [
        (0..150u32).map(|at| (at * 37 + 11) % 256).collect(),
        (0..90u32).map(|at| (at * 53 + 5) % 256).collect(),
    ];
    let mut held = [0u32; 2];
    let mut fed = prompts;
    let mut fire = |shell: &mut engine_metal::Shell, lanes: &[usize], fed: &mut [Vec<u32>; 2]| {
        let seated: Vec<Seated<'_>> = lanes
            .iter()
            .map(|&lane| Seated {
                pages: &[],
                held: Some(held[lane]),
                ..Seated::of(Lane {
                    slot: lane as u32,
                    word: word(&fixture.trace, fed[lane].len() as u32),
                    tokens: &fed[lane],
                })
            })
            .collect();
        let got = shell.fire_seated(&seated).expect("the fire lands");
        for (&lane, logits) in lanes.iter().zip(&got) {
            held[lane] += fed[lane].len() as u32;
            fed[lane] = vec![argmax(logits)];
        }
    };
    fire(&mut shell, &[0], &mut fed);
    fire(&mut shell, &[1], &mut fed);
    for _ in 0..8 {
        fire(&mut shell, &[0, 1], &mut fed);
    }
    gaps.push(("after fires", audited(&shell, "after fires")));

    for (when, gap) in gaps {
        assert!(
            gap.abs() <= SLACK,
            "{when}: the device holds {gap} bytes the ledger does not account for"
        );
    }
}
