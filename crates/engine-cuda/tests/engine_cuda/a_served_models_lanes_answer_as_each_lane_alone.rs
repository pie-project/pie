#![cfg(feature = "device")]
//! Contracts checked by hand on a served model, which the test names
//! through the environment (the shell itself reads none): its lanes served
//! together answer as each lane served alone.

/// Contract A at eight decode lanes: each lane's greedy decode served in
/// one fire of eight (the few-rows products at eight rows) against the same
/// lane served alone (one row, the gemv), token for token; a step where the
/// two pick apart is admitted only as a near-tie (the drift over the full
/// row diffuse, and the decision's gap within twice its largest per-logit
/// move). Run by hand on a served model:
/// `PIE_CONTRACT_ARTIFACT=<.zt> PIE_CONTRACT_DEPLOYMENT=<row>`.

mod eight_lanes {
    use std::path::Path;

    use engine_cuda::tune::{Shape, load, patches};

    const LANES: u32 = 8;
    const PROMPT: u32 = 128;
    const STEPS: usize = 32;

    fn argmax(row: &[f32]) -> usize {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(at, _)| at)
    }

    /// The decision's top-1 less its top-2.
    fn gap(row: &[f32]) -> f32 {
        let mut top = [f32::NEG_INFINITY; 2];
        for &v in row {
            if v > top[0] {
                top = [v, top[0]];
            } else if v > top[1] {
                top[1] = v;
            }
        }
        top[0] - top[1]
    }

    #[test]
    #[ignore = "a contract check run by hand on a served model"]
    fn eight_lanes_decode_as_each_lane_alone_up_to_near_ties() {
        let (Some(artifact), Some(deployment)) = (
            std::env::var_os("PIE_CONTRACT_ARTIFACT"),
            std::env::var("PIE_CONTRACT_DEPLOYMENT").ok(),
        ) else {
            eprintln!("skipping: PIE_CONTRACT_ARTIFACT / PIE_CONTRACT_DEPLOYMENT unset");
            return;
        };
        let slots = 2 * LANES;
        let (mut shell, trace) = load(
            Path::new(&artifact),
            &deployment,
            engine_cuda::Knobs::default(),
            |trace| Shape {
                slots,
                budget: poem_compiler::Budget::new(LANES, 0),
                patches: patches(trace, LANES),
                page_size: 16,
                context: 1024,
            },
        )
        .expect("the model loads");
        let word = |n: usize| trace.facts.word(&poem::Request::new(n as u32, false));
        let prompt = |lane: u32| -> Vec<u32> {
            (0..PROMPT)
                .map(|i| 1000 + (i * 7919 + lane * 104_729) % 20_000)
                .collect()
        };
        // Lane `l` runs in slot `l` with the others (batched) and in slot
        // `LANES + l` alone; both prefill alone, the same rows.
        let mut last = [[0u32; 2]; LANES as usize];
        for lane in 0..LANES {
            for (side, slot) in [(0, lane), (1, LANES + lane)] {
                shell.open(slot).expect("a slot opens");
                let tokens = prompt(lane);
                let rows = shell
                    .fire(&[engine_cuda::Lane {
                        slot,
                        word: word(tokens.len()),
                        tokens: &tokens,
                    }])
                    .expect("the prompt fires");
                last[lane as usize][side] = argmax(&rows[0]) as u32;
            }
        }
        let mut parted = vec![false; LANES as usize];
        let (mut same, mut ties) = (0usize, 0usize);
        for step in 0..STEPS {
            let next: Vec<[u32; 1]> = (0..LANES as usize).map(|l| [last[l][0]]).collect();
            let lanes: Vec<engine_cuda::Lane> = (0..LANES)
                .map(|l| engine_cuda::Lane {
                    slot: l,
                    word: word(1),
                    tokens: &next[l as usize],
                })
                .collect();
            let batched = shell.fire(&lanes).expect("the eight lanes fire");
            for lane in 0..LANES as usize {
                let token = [last[lane][1]];
                let alone = shell
                    .fire(&[engine_cuda::Lane {
                        slot: LANES + lane as u32,
                        word: word(1),
                        tokens: &token,
                    }])
                    .expect("the lane fires alone")
                    .swap_remove(0);
                let (a, b) = (&alone, &batched[lane]);
                let (ta, tb) = (argmax(a), argmax(b));
                if !parted[lane] && ta != tb {
                    let moved: Vec<f32> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
                    let drift = moved.iter().copied().fold(0.0f32, f32::max);
                    let elsewhere = moved
                        .iter()
                        .enumerate()
                        .filter(|(t, _)| *t != ta && *t != tb)
                        .map(|(_, d)| *d)
                        .fold(0.0f32, f32::max);
                    let diffuse = moved[ta] <= elsewhere && moved[tb] <= elsewhere;
                    let (ga, gb) = (gap(a), gap(b));
                    let tie = diffuse && ga.max(gb) <= 2.0 * drift;
                    println!(
                        "lane {lane} parts at step {step}: alone {ta} batched {tb}; gap {ga:.4} / \
                         {gb:.4}, max drift {drift:.4}, diffuse {diffuse}: {}",
                        if tie { "near-tie" } else { "NOT a near-tie" }
                    );
                    assert!(tie, "lane {lane} parts at step {step} on no near-tie");
                    parted[lane] = true;
                    ties += 1;
                } else if !parted[lane] {
                    same += 1;
                }
                // Side 0 is the batched slot's, side 1 the lone one's.
                last[lane] = [tb as u32, ta as u32];
            }
        }
        println!(
            "{deployment}: {LANES} lanes x {STEPS} steps: {same} steps the same token, {ties} \
             lane(s) parted on a near-tie"
        );
    }
}

/// A served model's logits, gathered at the readout rows, read back at each
/// lane's own row in a fire mixing a prompt between decodes: each lane's
/// last row is the one it answers alone. Run by hand on a served model:
/// `PIE_CONTRACT_ARTIFACT=<.zt> PIE_CONTRACT_DEPLOYMENT=<row>`.

mod mixed_readout {
    use std::path::Path;

    use engine_cuda::tune::{Shape, load, patches};

    #[test]
    #[ignore = "a readout check run by hand on a served model"]
    fn a_prompt_beside_three_decodes_reads_each_lanes_own_row() {
        let (Some(artifact), Some(deployment)) = (
            std::env::var_os("PIE_CONTRACT_ARTIFACT"),
            std::env::var("PIE_CONTRACT_DEPLOYMENT").ok(),
        ) else {
            eprintln!("skipping: PIE_CONTRACT_ARTIFACT / PIE_CONTRACT_DEPLOYMENT unset");
            return;
        };
        let (mut shell, trace) = load(
            Path::new(&artifact),
            &deployment,
            engine_cuda::Knobs::default(),
            |trace| Shape {
                slots: 8,
                budget: poem_compiler::Budget::new(4, 0),
                patches: patches(trace, 4),
                page_size: 16,
                context: 1024,
            },
        )
        .expect("the model loads");
        let word = |n: usize| trace.facts.word(&poem::Request::new(n as u32, false));
        let prompt = |salt: u32, n: u32| -> Vec<u32> {
            (0..n)
                .map(|i| 1000 + (i * 7919 + salt * 104_729) % 20_000)
                .collect()
        };
        for slot in 0..8 {
            shell.open(slot).expect("a slot opens");
        }
        // Decoding lanes in slots 0..3 (twins 4..7, but for the prompt's 3 / 7).
        for (slot, salt) in [(0, 0), (4, 0), (1, 1), (5, 1), (2, 2), (6, 2)] {
            let tokens = prompt(salt, 96);
            shell
                .fire(&[engine_cuda::Lane {
                    slot,
                    word: word(tokens.len()),
                    tokens: &tokens,
                }])
                .expect("a prompt fires");
        }
        let fresh = prompt(9, 40);
        let next = [[11u32], [12], [13]];
        let mixed = shell
            .fire(&[
                engine_cuda::Lane {
                    slot: 0,
                    word: word(1),
                    tokens: &next[0],
                },
                engine_cuda::Lane {
                    slot: 3,
                    word: word(fresh.len()),
                    tokens: &fresh,
                },
                engine_cuda::Lane {
                    slot: 1,
                    word: word(1),
                    tokens: &next[1],
                },
                engine_cuda::Lane {
                    slot: 2,
                    word: word(1),
                    tokens: &next[2],
                },
            ])
            .expect("the mixed fire serves");
        let alone: Vec<Vec<f32>> = [
            (4u32, &next[0][..]),
            (7, &fresh[..]),
            (5, &next[1][..]),
            (6, &next[2][..]),
        ]
        .iter()
        .map(|&(slot, tokens)| {
            shell
                .fire(&[engine_cuda::Lane {
                    slot,
                    word: word(tokens.len()),
                    tokens,
                }])
                .expect("a lane fires alone")
                .swap_remove(0)
        })
        .collect();
        for (lane, (a, b)) in alone.iter().zip(&mixed).enumerate() {
            let worst = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            let top = |row: &[f32]| {
                row.iter()
                    .enumerate()
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .map_or(0, |(at, _)| at)
            };
            println!(
                "{deployment}: lane {lane} worst {worst} top {} / {}",
                top(a),
                top(b)
            );
            assert!(
                worst < 0.5,
                "lane {lane}'s last row is not its own ({worst})"
            );
        }
    }
}
