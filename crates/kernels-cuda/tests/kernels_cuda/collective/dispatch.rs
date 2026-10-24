//! Expert-parallel dispatch and combine over ranks that share one device:
//! each rank a stream and its own buffers, as peers map one another's. The
//! rows each rank lands, their routes, weights, origins and per-slot counts
//! match the host's walk in source-rank then token order; the combine puts
//! every live route's output back at its token, and nothing else. Where the
//! process sees several devices the same walk runs with rank r on device
//! r % devices: real peers over the link, and past two ranks real and
//! virtual peers in one group.

use crate::common::{Alone, Gpu, device_count};
use dtype::Dtype::{Bf16, F32, I32};
use kernels_cuda::Tensor;
use kernels_cuda::collective::{
    EP_SIGNAL_BYTES, Ep, Landing, Placement, Tokens, combine, dispatch, ep_scratch_bytes,
};

/// What `routed` holds where no route wrote.
const UNTOUCHED: u16 = 0xbeef;

#[derive(Clone, Copy)]
struct Case {
    world: usize,
    tokens: usize,
    width: usize,
    out_width: usize,
    top_k: usize,
    experts: usize,
    /// Experts `0 .. extra` hold a second replica, at slot `experts + e`.
    extra: usize,
    weights: bool,
    /// `[live, first]`: the armed window, or every token live.
    window: Option<[u32; 2]>,
    /// `x`'s rows when fewer than the step's tokens (an eager walk's live
    /// rows under a step at its bucket), with no window armed.
    short: Option<usize>,
}

/// One rank's inputs for a round.
struct Data {
    x: Vec<u16>,
    routes: Vec<i32>,
    weights: Vec<f32>,
}

impl Case {
    fn slots(&self) -> usize {
        (self.experts + self.extra) / self.world
    }

    fn table(&self) -> Vec<i32> {
        (0..self.experts)
            .flat_map(|e| {
                let twin = e < self.extra;
                let other = if twin { (self.experts + e) as i32 } else { 0 };
                [1 + i32::from(twin), e as i32, other]
            })
            .collect()
    }

    fn live(&self, t: usize) -> bool {
        if let Some(rows) = self.short {
            return t < rows;
        }
        self.window
            .is_none_or(|[live, first]| t >= first as usize && t < (first + live) as usize)
    }

    /// Route j of rank r's token t: its physical slot, as the kernel picks
    /// the replica, or `None`.
    fn physical(&self, r: usize, routes: &[i32], t: usize, j: usize) -> Option<usize> {
        let e = usize::try_from(routes[t * self.top_k + j]).ok()?;
        let twin = e < self.extra;
        let pick = (r + t * self.top_k + j) % (1 + usize::from(twin));
        Some(if pick == 1 { self.experts + e } else { e })
    }

    fn data(&self, r: usize, round: usize) -> Data {
        let (t, k, h) = (self.tokens, self.top_k, self.width);
        let seed = r * 977 + round * 131;
        let x = (0..t * h)
            .map(|i| ((i * 7 + seed * 13) % 0x7fff) as u16)
            .collect();
        let routes = (0..t * k)
            .map(|i| {
                let (tok, j) = (i / k, i % k);
                if (tok + j + seed) % 11 == 10 {
                    -1
                } else {
                    ((tok * 5 + j * 7 + seed * 3 + tok / 3) % self.experts) as i32
                }
            })
            .collect();
        let weights = (0..t * k).map(|i| (i + seed) as f32 * 0.25).collect();
        Data { x, routes, weights }
    }
}

/// One rank's device buffers.
struct Held {
    tokens: Tokens,
    landing: Landing,
    counts: Tensor,
    routed: Tensor,
    y: Tensor,
    signals: u64,
    scratch: u64,
    window: u64,
    place: Placement,
}

/// What a slot's load holds before a dispatch adds its rows.
const LOADS_BEFORE: u32 = 5;

/// What a peer reads or writes sits in a mapping every device of the group
/// (`readers`) reaches; the rest is the rank's own.
fn hold(c: &Case, g: &mut Gpu, d: &Data, signals: u64, readers: &[i32]) -> Held {
    let (t, k, h, w) = (c.tokens, c.top_k, c.width, c.out_width);
    let (cap, slots) = (c.world * t, c.slots());
    let x = Tensor::new(g.up(&d.x), t as u32, h as u32, Bf16);
    let routes = Tensor::new(g.up(&d.routes), t as u32, k as u32, I32);
    let weights = c
        .weights
        .then(|| Tensor::new(g.up(&d.weights), t as u32, k as u32, F32));
    let mut peer = |bytes| g.shared(bytes, readers);
    let landing = Landing {
        rows: Tensor::new(peer(cap * h * 2), cap as u32, h as u32, Bf16),
        local: Tensor::new(peer(cap * k * 4), cap as u32, k as u32, I32),
        row_weights: c
            .weights
            .then(|| Tensor::new(peer(cap * k * 4), cap as u32, k as u32, F32)),
        origin: Tensor::new(peer(cap * 4), cap as u32, 1, I32),
    };
    let counts = Tensor::new(g.zeros(slots * 4), 1, slots as u32, I32);
    let routed = Tensor::new(
        g.up_shared(&vec![UNTOUCHED; t * k * w], readers),
        (t * k) as u32,
        w as u32,
        Bf16,
    );
    let y = Tensor::new(g.zeros(cap * k * w * 2), (cap * k) as u32, w as u32, Bf16);
    let window = g.up(&c.window.unwrap_or([0, 0]));
    let scratch = g.zeros(ep_scratch_bytes(t as u32, c.world as u32) as usize);
    let place = Placement {
        table: g.up(&c.table()),
        stride: 3,
        experts: c.experts as u32,
        slots: slots as u32,
        loads: g.up(&vec![LOADS_BEFORE; slots]),
    };
    Held {
        tokens: Tokens {
            x: Tensor {
                rows: c.short.map_or(x.rows, |r| r as u32),
                ..x
            },
            routes: Tensor {
                rows: c.short.map_or(routes.rows, |r| r as u32),
                ..routes
            },
            weights: weights.map(|w| Tensor {
                rows: c.short.map_or(w.rows, |r| r as u32),
                ..w
            }),
            step: t as u32,
        },
        landing,
        counts,
        routed,
        y,
        signals,
        scratch,
        window,
        place,
    }
}

fn ep_of(held: &[Held], rank: usize) -> Ep {
    let mut signals = [0; 8];
    for (s, h) in signals.iter_mut().zip(held) {
        *s = h.signals;
    }
    Ep {
        rank: rank as u32,
        world: held.len() as u32,
        signals,
        scratch: held[rank].scratch,
        scratch_tokens: held[rank].tokens.step,
    }
}

// The ranks' launches meet on the device: a call that waits on it from a
// test running beside them, landing between two ranks' launches, would wait
// on the first rank's spin while holding back the second's. Every check
// holds the devices alone (`open_ranks`).
fn fire_dispatch(c: &Case, gpus: &[Gpu], held: &[Held]) {
    let landing: Vec<_> = held.iter().map(|h| h.landing).collect();
    let ctxs: Vec<_> = gpus.iter().map(Gpu::ctx).collect();
    for (r, ctx) in ctxs.iter().enumerate() {
        gpus[r].bind();
        if c.window.is_some() {
            ctx.arm_stage(held[r].window);
        }
        let mut counts = held[r].counts;
        dispatch(
            ctx,
            &ep_of(held, r),
            held[r].place,
            held[r].tokens,
            &landing,
            &mut counts,
        )
        .expect("the dispatch fires");
    }
    gpus.iter().for_each(Gpu::sync);
}

fn fire_combine(gpus: &[Gpu], held: &[Held]) {
    let routed: Vec<_> = held.iter().map(|h| h.routed).collect();
    let ctxs: Vec<_> = gpus.iter().map(Gpu::ctx).collect();
    for (r, ctx) in ctxs.iter().enumerate() {
        gpus[r].bind();
        combine(ctx, &ep_of(held, r), &held[r].landing, held[r].y, &routed)
            .expect("the combine fires");
    }
    gpus.iter().for_each(Gpu::sync);
}

/// What the stand-in experts write for route j of the token that came from
/// `origin`, at local slot `slot`, element `w`.
fn pattern(origin: usize, j: usize, slot: usize, w: usize) -> u16 {
    ((origin * 31 + j * 7 + slot * 13 + w) % 0x7fff) as u16
}

fn check(c: &Case) {
    check_on(c, 1);
}

/// The group's ranks, rank r on device r % `devices`, with the devices to
/// themselves until the first of the three is dropped (it is the last).
fn open_ranks(world: usize, devices: i32) -> (Alone, Vec<Gpu>, Vec<i32>) {
    let alone = crate::common::alone();
    let gpus: Vec<_> = (0..world)
        .map(|r| Gpu::open_on(r as i32 % devices))
        .collect();
    let mut readers: Vec<_> = gpus.iter().map(Gpu::ordinal).collect();
    readers.sort_unstable();
    readers.dedup();
    // No module load may land between the ranks' launches: each device
    // loads them before any rank fires.
    for g in &gpus {
        g.bind();
        kernels_cuda::collective::load_ep().expect("the EP kernels load");
    }
    (alone, gpus, readers)
}

fn check_on(c: &Case, devices: i32) {
    let (world, t, k, h, w) = (c.world, c.tokens, c.top_k, c.width, c.out_width);
    let (slots, cap) = (c.slots(), world * t);
    let (_alone, mut gpus, readers) = open_ranks(world, devices);
    let signals: Vec<_> = gpus
        .iter_mut()
        .map(|g| g.shared(EP_SIGNAL_BYTES as usize, &readers))
        .collect();

    // Two rounds on the same signal blocks: the second meets the ranks on
    // counters the first left behind.
    for round in 0..2 {
        let data: Vec<_> = (0..world).map(|r| c.data(r, round)).collect();
        let held: Vec<_> = (0..world)
            .map(|r| hold(c, &mut gpus[r], &data[r], signals[r], &readers))
            .collect();
        fire_dispatch(c, &gpus, &held);

        for (d, g) in gpus.iter_mut().enumerate() {
            let at = format!("{world} ranks on {devices} devices, round {round}, rank {d}");
            let mut want = Vec::new();
            for (s, src) in data.iter().enumerate() {
                for tok in (0..t).filter(|&tok| c.live(tok)) {
                    let phys = |j| c.physical(s, &src.routes, tok, j);
                    if (0..k).any(|j| phys(j).is_some_and(|p| p / slots == d)) {
                        want.push((s, tok));
                    }
                }
            }
            let l = held[d].landing;
            let origin: Vec<i32> = g.down(l.origin.ptr, cap);
            let local: Vec<i32> = g.down(l.local.ptr, cap * k);
            let rows: Vec<u16> = g.down(l.rows.ptr, cap * h);
            let row_w: Vec<f32> = l
                .row_weights
                .map_or(Vec::new(), |rw| g.down(rw.ptr, cap * k));
            let counts: Vec<i32> = g.down(held[d].counts.ptr, slots);
            let mut tally = vec![0; slots];
            for i in 0..cap {
                let Some(&(s, tok)) = want.get(i) else {
                    assert_eq!(origin[i], -1, "{at}: padding row {i}'s origin");
                    assert!(
                        local[i * k..][..k].iter().all(|&v| v == -1),
                        "{at}: padding row {i}"
                    );
                    continue;
                };
                assert_eq!(origin[i], (s * t + tok) as i32, "{at}: row {i}'s origin");
                assert_eq!(
                    rows[i * h..][..h],
                    data[s].x[tok * h..][..h],
                    "{at}: row {i}"
                );
                for j in 0..k {
                    let here = c
                        .physical(s, &data[s].routes, tok, j)
                        .filter(|p| p / slots == d);
                    let slot = here.map_or(-1, |p| (p % slots) as i32);
                    assert_eq!(local[i * k + j], slot, "{at}: row {i} route {j}'s slot");
                    if let Some(p) = here {
                        tally[p % slots] += 1;
                    }
                    if c.weights {
                        let wt = if here.is_some() {
                            data[s].weights[tok * k + j]
                        } else {
                            0.0
                        };
                        assert_eq!(row_w[i * k + j], wt, "{at}: row {i} route {j}'s weight");
                    }
                }
            }
            assert_eq!(counts, tally, "{at}: rows per local slot");
            let loads: Vec<u32> = g.down(held[d].place.loads, slots);
            let summed: Vec<u32> = tally.iter().map(|&n| LOADS_BEFORE + n as u32).collect();
            assert_eq!(
                loads, summed,
                "{at}: the loads each slot's rows are added to"
            );

            // Stand-in experts: each landed route's output names its origin,
            // route and slot.
            let mut y = vec![0u16; cap * k * w];
            for i in 0..want.len() {
                for j in 0..k {
                    if local[i * k + j] >= 0 {
                        for e in 0..w {
                            y[(i * k + j) * w + e] =
                                pattern(origin[i] as usize, j, local[i * k + j] as usize, e);
                        }
                    }
                }
            }
            crate::common::host::upload_to(held[d].y.ptr, &y);
        }

        fire_combine(&gpus, &held);
        for (s, g) in gpus.iter().enumerate() {
            let routed: Vec<u16> = g.down(held[s].routed.ptr, t * k * w);
            for tok in 0..t {
                for j in 0..k {
                    let got = &routed[(tok * k + j) * w..][..w];
                    let at = format!(
                        "{world} ranks on {devices} devices, round {round}: rank {s} token {tok} route {j}"
                    );
                    match c
                        .physical(s, &data[s].routes, tok, j)
                        .filter(|_| c.live(tok))
                    {
                        Some(p) => {
                            let want: Vec<u16> = (0..w)
                                .map(|e| pattern(s * t + tok, j, p % slots, e))
                                .collect();
                            assert_eq!(got, want.as_slice(), "{at}");
                        }
                        None => assert!(got.iter().all(|&v| v == UNTOUCHED), "{at}: untouched"),
                    }
                }
            }
        }
    }
}

#[test]
fn every_rank_lands_its_routes_rows_in_rank_then_token_order() {
    let base = Case {
        world: 1,
        tokens: 37,
        width: 256,
        out_width: 128,
        top_k: 6,
        experts: 32,
        extra: 0,
        weights: true,
        window: None,
        short: None,
    };
    for world in [1, 2, 4, 8] {
        check(&Case { world, ..base });
    }
}

#[test]
fn replicas_a_window_and_short_rows_keep_the_rows_where_the_host_puts_them() {
    let base = Case {
        world: 4,
        tokens: 37,
        width: 64,
        out_width: 64,
        top_k: 8,
        experts: 32,
        extra: 8,
        weights: false,
        window: None,
        short: None,
    };
    check(&base);
    check(&Case {
        window: Some([20, 5]),
        ..base
    });
    check(&Case {
        short: Some(30),
        ..base
    });
    check(&Case {
        world: 8,
        weights: true,
        ..base
    });
}

#[test]
fn a_fire_past_one_count_block_scans_across_its_chunks() {
    check(&Case {
        world: 2,
        tokens: 1500,
        width: 64,
        out_width: 32,
        top_k: 4,
        experts: 16,
        extra: 0,
        weights: true,
        window: None,
        short: None,
    });
}

/// The devices to spread the ranks over, or `None` (with a note) on a
/// machine with one.
fn peer_devices() -> Option<i32> {
    let devices = device_count();
    if devices < 2 {
        eprintln!("one device: no real peers to check");
        return None;
    }
    Some(devices)
}

#[test]
fn real_peers_land_and_return_what_one_device_does() {
    let Some(devices) = peer_devices() else {
        return;
    };
    let base = Case {
        world: 2,
        tokens: 37,
        width: 256,
        out_width: 128,
        top_k: 6,
        experts: 32,
        extra: 0,
        weights: true,
        window: None,
        short: None,
    };
    for world in [2, 4, 8] {
        check_on(&Case { world, ..base }, devices);
    }
    let replicas = Case {
        world: 4,
        width: 64,
        out_width: 64,
        top_k: 8,
        extra: 8,
        weights: false,
        ..base
    };
    check_on(&replicas, devices);
    check_on(
        &Case {
            window: Some([20, 5]),
            ..replicas
        },
        devices,
    );
    check_on(
        &Case {
            world: 8,
            short: Some(30),
            weights: true,
            ..replicas
        },
        devices,
    );
    check_on(
        &Case {
            tokens: 1500,
            width: 64,
            out_width: 32,
            top_k: 4,
            experts: 16,
            ..base
        },
        devices,
    );
}

/// Dispatch's and combine's rate across real peers at two ranks, a rank a
/// device: the bytes one rank sends the other over the link per fire (its
/// rows there, and back each route's output), against the ~956 GB/s a
/// direction NV18 carries. Every rank fires the rounds back to back, so the
/// time is the slower rank's.
#[test]
#[ignore = "a rate bench: cargo test --release ... -- --ignored peer_rate --nocapture"]
fn peer_rate() {
    let Some(devices) = peer_devices() else {
        return;
    };
    let (reps, replays) = (5, 40);
    for width in [2048usize, 4096, 7168] {
        for tokens in [64usize, 256, 1024, 4096] {
            let c = Case {
                world: 2,
                tokens,
                width,
                out_width: width,
                top_k: 8,
                experts: 256,
                extra: 0,
                weights: true,
                window: None,
                short: None,
            };
            let (_alone, mut gpus, readers) = open_ranks(c.world, devices);
            let signals: Vec<_> = gpus
                .iter_mut()
                .map(|g| g.shared(EP_SIGNAL_BYTES as usize, &readers))
                .collect();
            let data: Vec<_> = (0..c.world).map(|r| c.data(r, 0)).collect();
            let held: Vec<_> = (0..c.world)
                .map(|r| hold(&c, &mut gpus[r], &data[r], signals[r], &readers))
                .collect();
            // Rank 0's tokens and routes that cross to rank 1.
            let slots = c.slots();
            let (mut rows, mut routes) = (0, 0);
            for tok in 0..tokens {
                let there = (0..c.top_k)
                    .filter(|&j| {
                        c.physical(0, &data[0].routes, tok, j)
                            .is_some_and(|p| p / slots == 1)
                    })
                    .count();
                rows += usize::from(there > 0);
                routes += there;
            }
            let landing: Vec<_> = held.iter().map(|h| h.landing).collect();
            let routed: Vec<_> = held.iter().map(|h| h.routed).collect();
            let ctxs: Vec<_> = gpus.iter().map(Gpu::ctx).collect();
            // Each rank's rounds in one graph, so the time is the devices'
            // and not the host's launches.
            let timed = |fire: &dyn Fn(usize)| {
                let graphs: Vec<_> = gpus
                    .iter()
                    .enumerate()
                    .map(|(r, g)| {
                        g.bind();
                        crate::common::captured(&ctxs[r], || (0..reps).for_each(|_| fire(r)))
                    })
                    .collect();
                let _capture = crate::common::capturing();
                let replay = || {
                    for (r, g) in gpus.iter().enumerate() {
                        g.bind();
                        crate::common::replay(&ctxs[r], graphs[r]);
                    }
                };
                replay();
                gpus.iter().for_each(Gpu::sync);
                // The fastest replay: the devices are shared, and another
                // process's kernels only ever add time.
                let best = (0..replays)
                    .map(|_| {
                        let start = std::time::Instant::now();
                        replay();
                        gpus.iter().for_each(Gpu::sync);
                        start.elapsed()
                    })
                    .min()
                    .expect("a replay");
                graphs.into_iter().for_each(crate::common::forget);
                best.as_secs_f64() * 1e6 / f64::from(reps)
            };
            let dispatch_us = timed(&|r| {
                let mut counts = held[r].counts;
                let ep = ep_of(&held, r);
                dispatch(
                    &ctxs[r],
                    &ep,
                    held[r].place,
                    held[r].tokens,
                    &landing,
                    &mut counts,
                )
                .expect("the dispatch fires");
            });
            let combine_us = timed(&|r| {
                let ep = ep_of(&held, r);
                combine(&ctxs[r], &ep, &held[r].landing, held[r].y, &routed)
                    .expect("the combine fires");
            });
            let gbs = |bytes: usize, us: f64| (bytes * width * 2) as f64 / us / 1e3;
            eprintln!(
                "hidden {width} tokens {tokens}: dispatch {dispatch_us:.1} us {:.0} GB/s, \
                 combine {combine_us:.1} us {:.0} GB/s",
                gbs(rows, dispatch_us),
                gbs(routes, combine_us),
            );
        }
    }
}
