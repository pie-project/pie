use super::*;
use std::collections::HashSet;

/// splitmix64: a seeded stream, so a failing case replays.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }
}

fn facts(total: u32, segment: u32, committed: u32, shapes: &[(ShapeKind, u32)]) -> SlotPoolFacts {
    facts_with(
        total,
        segment,
        committed,
        shapes,
        &[(16, 1), (256, 3), (4096, 6)],
    )
}

fn facts_with(
    total: u32,
    segment: u32,
    committed: u32,
    shapes: &[(ShapeKind, u32)],
    act_chunks: &[(u32, u32)],
) -> SlotPoolFacts {
    SlotPoolFacts {
        chunk_bytes: 1 << 20,
        total_chunks: total,
        segment_chunks: segment,
        committed_segments: committed,
        reserved_chunks: 1,
        shapes: shapes
            .iter()
            .map(|&(kind, slots_per_chunk)| SlotShape {
                kind,
                slot_bytes: (1 << 20) / u64::from(slots_per_chunk),
                slots_per_chunk,
            })
            .collect(),
        act_chunks: act_chunks.to_vec(),
        span_on_demand: false,
        act_span: None,
    }
}

fn cache(key: u8) -> Holder {
    Holder::CacheEntry([key; 32])
}

/// What the test believes each holder holds.
#[derive(Default)]
struct Model {
    held: HashMap<Holder, Vec<Slot>>,
    /// Let go, and the epoch it waits on.
    pending: HashMap<Slot, u64>,
    retired: u64,
    touched: HashMap<Slot, u64>,
}

impl Model {
    fn refs(&self, slot: Slot) -> (u32, u32) {
        let mut refs = (0, 0);
        for (holder, slots) in &self.held {
            let count = slots.iter().filter(|&&s| s == slot).count() as u32;
            if holder.is_live() {
                refs.0 += count;
            } else {
                refs.1 += count;
            }
        }
        refs
    }
}

fn property_run(seed: u64, shapes: &[(ShapeKind, u32)]) {
    let mut rng = Rng(seed);
    let mut pool = SlotPool::new(facts(96, 16, 3, shapes));
    let mut model = Model::default();
    let mut epoch = 1u64;
    let holders = |rng: &mut Rng| match rng.below(3) {
        0 => Holder::WorkingSet(rng.below(6)),
        1 => Holder::Snapshot(rng.below(3)),
        _ => cache(rng.below(4) as u8),
    };
    for step in 0..3000 {
        match rng.below(12) {
            0..=3 => {
                let holder = holders(&mut rng);
                let shape = rng.below(shapes.len() as u64) as ShapeId;
                let n = rng.below(12);
                let available = pool.available(shape);
                match pool.acquire(holder, shape, n) {
                    Ok(slots) => {
                        assert!(n <= available);
                        assert_eq!(slots.len() as u64, n);
                        let unique: HashSet<_> = slots.iter().collect();
                        assert_eq!(unique.len(), slots.len(), "a slot handed out twice");
                        for &slot in &slots {
                            assert_eq!(model.refs(slot), (0, 0), "{slot:?} handed out while held");
                            if let Some(&after) = model.pending.get(&slot) {
                                assert!(
                                    after <= model.retired,
                                    "{slot:?} reused before epoch {after}"
                                );
                                model.pending.remove(&slot);
                            }
                            model.touched.remove(&slot);
                        }
                        model.held.entry(holder).or_default().extend(slots);
                    }
                    Err(SlotError::Short { .. }) => assert!(n > available),
                    Err(other) => panic!("{other}"),
                }
            }
            4 => {
                // A holder takes a reference on another's slots.
                let from = holders(&mut rng);
                let to = holders(&mut rng);
                let mut slots: Vec<Slot> = model.held.get(&from).cloned().unwrap_or_default();
                slots.retain(|slot| !model.held.get(&to).is_some_and(|held| held.contains(slot)));
                slots.sort();
                slots.dedup();
                slots.truncate(rng.below(6) as usize);
                pool.share(to, &slots);
                model.held.entry(to).or_default().extend(slots);
            }
            5..=7 => {
                let holder = holders(&mut rng);
                let Some(held) = model.held.get_mut(&holder) else {
                    continue;
                };
                let count = rng.below(held.len() as u64 + 1) as usize;
                let gone: Vec<Slot> = held.drain(..count).collect();
                let at = epoch.saturating_sub(rng.below(3));
                pool.release(holder, &gone, at);
                for slot in gone {
                    if model.refs(slot) == (0, 0) {
                        let after = at.max(model.touched.get(&slot).copied().unwrap_or(0));
                        model.pending.insert(slot, after);
                    }
                }
            }
            8 => {
                // A fire touches some held slots.
                let mut slots: Vec<Slot> = model.held.values().flatten().copied().collect();
                slots.sort();
                slots.dedup();
                slots.retain(|_| rng.below(3) == 0);
                pool.touch(&slots, epoch);
                for slot in slots {
                    model.touched.insert(slot, epoch);
                }
                epoch += 1;
            }
            9 => {
                let through = epoch.saturating_sub(rng.below(4));
                pool.retire_through(through);
                model.retired = model.retired.max(through);
            }
            10 => {
                let rung = [0, 200, 1000, 5000][rng.below(4) as usize];
                if pool.set_rung(rung).is_err()
                    && let Ok(evacuation) = pool.plan_rung(rung)
                {
                    let refs: Vec<(Slot, u32)> = model
                        .held
                        .values()
                        .flatten()
                        .map(|&slot| (slot, pool.slot_ref(slot)))
                        .collect();
                    pool.apply_rung(&evacuation, epoch).unwrap();
                    for (slot, engine) in refs {
                        assert_eq!(pool.slot_ref(slot), engine);
                        assert!(pool.physical(slot.chunk).unwrap() < pool.state_limit());
                    }
                    epoch += 1;
                }
            }
            _ => {
                // Shrink or grow the pool.
                if rng.below(2) == 0 {
                    if let Some(segment) = pool.emptiest_segment()
                        && let Some(evacuation) = pool.plan_evacuation(segment)
                    {
                        let refs: Vec<(Slot, (u32, u32), u32)> = model
                            .held
                            .values()
                            .flatten()
                            .map(|&slot| (slot, pool.refs(slot), pool.slot_ref(slot)))
                            .collect();
                        pool.apply_evacuation(&evacuation, epoch).unwrap();
                        assert_eq!(pool.segments().used(segment), 0);
                        for (slot, refs, engine) in refs {
                            assert_eq!(pool.refs(slot), refs);
                            assert_eq!(pool.slot_ref(slot), engine);
                            let physical = pool.physical(slot.chunk).unwrap();
                            assert_ne!(pool.segments().of(physical), segment);
                        }
                        epoch += 1;
                    }
                    for segment in 0..pool.segments().count() {
                        let _ = pool.release_segment(segment);
                    }
                } else if let Some(&segment) = pool.commit_ahead(8 << 20).first() {
                    pool.commit_segment(segment).unwrap();
                }
            }
        }
        if step % 7 == 0 {
            pool.check();
        }
        // Refcount conservation, holder by holder and slot by slot.
        let mut total = 0;
        for (holder, slots) in &model.held {
            assert_eq!(pool.held_by(holder), slots.len() as u64, "{holder:?}");
            total += slots.len() as u64;
        }
        let pool_total: u64 = pool.holders.values().sum();
        assert_eq!(pool_total, total);
        if step % 50 == 0 {
            let mut all: Vec<Slot> = model.held.values().flatten().copied().collect();
            all.sort();
            all.dedup();
            for slot in all {
                assert_eq!(pool.refs(slot), model.refs(slot));
            }
        }
        assert!(
            pool.reserved_chunks() + pool.state_chunks() + pool.act_reserved()
                <= pool.committed_chunks()
        );
        assert!(pool.committed_chunks() <= pool.total_chunks());
    }
    // Let everything go: the pool comes back whole.
    for (holder, slots) in std::mem::take(&mut model.held) {
        pool.release(holder, &slots, epoch);
    }
    pool.retire_through(u64::MAX);
    assert_eq!(pool.state_chunks(), 0);
    assert_eq!(pool.demand(), Demand::default());
    pool.check();
}

#[test]
fn random_runs_keep_every_invariant() {
    for seed in 0..24 {
        property_run(
            seed,
            &[
                (ShapeKind::List, 4),
                (ShapeKind::Fixed, 1),
                (ShapeKind::List, 70),
            ],
        );
    }
}

/// Under park and restore churn, guest epilogues must not find "no
/// status at all" or rows read apart. A move's
/// destination is fenced with its source until the copy lands: a chunk
/// let go meanwhile is handed to no other owner the copy would then
/// overwrite.
#[test]
fn a_move_s_destination_is_handed_to_no_one_until_its_copy_lands() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 2)]));
    let holder = Holder::WorkingSet(1);
    let taken = pool.acquire_chunk_runs(holder, 0, 1, 6).unwrap();
    let low: Vec<Slot> = taken
        .iter()
        .copied()
        .filter(|slot| pool.chunk_table()[slot.chunk as usize] < 3)
        .collect();
    pool.release(holder, &low, 0);
    pool.retire_through(u64::MAX);
    let high = taken
        .iter()
        .map(|slot| slot.chunk)
        .max_by_key(|&chunk| pool.chunk_table()[chunk as usize])
        .unwrap();
    let moves = pool.plan_moves_of(&[high]);
    pool.apply_compaction(&moves, 5).unwrap();
    let to = moves[0].to;
    // Its owner lets it go before the copy lands.
    let rest: Vec<Slot> = taken
        .iter()
        .copied()
        .filter(|slot| slot.chunk == high)
        .collect();
    pool.release(holder, &rest, 0);
    assert!(
        !pool.open_chunks().contains(&to),
        "the copy's destination stays closed"
    );
    let from = moves[0].from;
    assert!(
        !pool.open_chunks().contains(&from),
        "the copy's source stays closed"
    );
    assert!(
        pool.plan_moves_of(&[high]).is_empty(),
        "a chunk mid-copy is moved by no one"
    );
    assert_eq!(pool.plan_compaction(2), None, "nor compacted over");
    let other = pool
        .acquire_chunk_runs(Holder::WorkingSet(2), 0, 1, 1)
        .unwrap();
    let got = pool.chunk_table()[other[0].chunk as usize];
    assert!(got != to && got != from, "neither end is handed out: {got}");
    pool.retire_through(5);
    assert!(
        pool.open_chunks().contains(&to),
        "the copy landed: open again"
    );
    pool.check();
}

/// The pool's table after its moves, one by one, is the engine's after
/// `ChunkTable::apply` of the same list: held chunks moved into holes
/// among free ones, free logicals swapped up in their place, and moves
/// whose destination is the physical another move of the list left (a
/// chain) or whose logical another moved onto (chunks moved into
/// scattered holes, then a segment release under the free ids swapped
/// up).
#[test]
fn the_pools_moves_leave_the_table_the_engines_apply_does() {
    let mut rng = 0x5eed_u64;
    let mut next = |below: u32| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        u32::try_from(rng % u64::from(below)).unwrap()
    };
    let mut applied = 0;
    for _ in 0..200 {
        let mut pool = SlotPool::new(facts_with(24, 4, 6, &[(ShapeKind::Fixed, 1)], &[]));
        let holder = Holder::WorkingSet(1);
        let held = pool.acquire(holder, 0, u64::from(next(16) + 1)).unwrap();
        // Scatter what is held: some given back.
        let back: Vec<Slot> = held.iter().copied().filter(|_| next(3) == 0).collect();
        pool.release(holder, &back, 0);
        pool.retire_through(u64::MAX);
        let mut engine = ::engine::chunk::ChunkTable::identity(pool.total_chunks());
        for _ in 0..4 {
            // Planned as the walk plans: held chunks to open ones, the
            // list applied by the pool move by move, by the engine
            // whole.
            let open: Vec<u32> = pool.open.iter_in(0, pool.total_chunks()).collect();
            let sources: Vec<u32> = (pool.reserved..pool.total_chunks())
                .filter(|&chunk| !pool.free.get(chunk) && !pool.fenced.get(chunk))
                .collect();
            let mut moves = Vec::new();
            let mut taken = BTreeSet::new();
            for &from in &sources {
                if next(2) == 0 || open.is_empty() {
                    continue;
                }
                let to = open[next(open.len() as u32) as usize];
                if !taken.insert(to) {
                    continue;
                }
                moves.push(ChunkMove {
                    logical: pool.physical_to_logical[from as usize],
                    from,
                    to,
                });
            }
            if moves.is_empty() || pool.apply_compaction(&moves, 1).is_err() {
                continue;
            }
            let evacuation = ::engine::chunk::Evacuation {
                moves: moves.iter().map(|step| (step.logical, step.to)).collect(),
            };
            engine.apply(&evacuation).unwrap();
            applied += 1;
            pool.retire_through(u64::MAX);
            for logical in 0..pool.total_chunks() {
                assert_eq!(
                    engine.physical(logical),
                    Some(pool.logical_to_physical[logical as usize]),
                    "logical chunk {logical} after {moves:?}"
                );
            }
            pool.check();
        }
        // Free chunks swapped among themselves (a move let go mid-way),
        // in chains: each the engine's swap too.
        let free: Vec<u32> = (0..pool.total_chunks())
            .filter(|&logical| {
                pool.chunks[logical as usize].shape == NO_SHAPE && logical >= pool.reserved
            })
            .collect();
        for _ in 0..8 {
            if free.len() < 2 {
                break;
            }
            let logical = free[next(free.len() as u32) as usize];
            let to = pool.logical_to_physical[free[next(free.len() as u32) as usize] as usize];
            let from = pool.logical_to_physical[logical as usize];
            pool.move_chunk(ChunkMove { logical, from, to });
            engine
                .apply(&::engine::chunk::Evacuation {
                    moves: vec![(logical, to)],
                })
                .unwrap();
        }
        for logical in 0..pool.total_chunks() {
            assert_eq!(
                engine.physical(logical),
                Some(pool.logical_to_physical[logical as usize])
            );
        }
    }
    assert!(applied > 100, "moves applied: {applied}");
}
