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

/// Test builds: a slot given back reaches the poison queue as its
/// physical byte range once its epoch retires, not before; a clone of
/// the pool (a placement tried on a copy) taps nothing.
#[cfg(feature = "test-slot-fill")]
#[test]
fn a_slot_given_back_is_queued_for_poison_by_its_physical_bytes() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 2)]));
    let queue = ::engine::chunk::PoisonQueue::default();
    pool.poison_into(queue.clone());
    let holder = Holder::WorkingSet(1);
    let slots = pool.acquire(holder, 0, 2).unwrap();
    let physical = pool.chunk_table()[slots[0].chunk as usize];
    let slot_bytes = (1u64 << 20) / 4;
    let mut copy = pool.clone();
    copy.release(holder, &slots, 0);
    copy.retire_through(u64::MAX);
    assert!(
        queue.lock().unwrap().is_empty(),
        "a clone's give-back is its own"
    );
    pool.touch(&slots, 5);
    pool.release(holder, &slots, 0);
    assert!(
        queue.lock().unwrap().is_empty(),
        "held for the fire that touched them"
    );
    pool.retire_through(5);
    let mut due = std::mem::take(&mut *queue.lock().unwrap());
    due.sort_unstable();
    let mut want: Vec<(u64, u64)> = slots
        .iter()
        .map(|slot| {
            (
                u64::from(physical) * (1 << 20) + u64::from(slot.index) * slot_bytes,
                slot_bytes,
            )
        })
        .collect();
    want.sort_unstable();
    assert_eq!(due, want);
}

/// "a guest item ... lies in a run of 2 chunk(s) ... and ends past a
/// gap": a raise moving a run's chunks one by one, each to the lowest
/// free chunk, splits it. A run moves whole, to a free
/// run of its length below, or the raise waits for one.
#[test]
fn a_raise_moves_a_run_whole_or_waits() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 2), (256, 6)]));
    let holder = Holder::WorkingSet(1);
    pool.set_rung(16).unwrap();
    let low = pool.acquire_chunk_runs(holder, 0, 1, 9).unwrap();
    let at = |pool: &SlotPool, slot: &Slot| pool.chunk_table()[slot.chunk as usize];
    let free_of = |pool: &mut SlotPool, chunks: &[u32]| {
        let slots: Vec<Slot> = low
            .iter()
            .copied()
            .filter(|slot| chunks.contains(&at(pool, slot)))
            .collect();
        pool.release(holder, &slots, 0);
        pool.retire_through(u64::MAX);
    };
    free_of(&mut pool, &[3, 5]);
    let run = pool.acquire_chunk_runs(holder, 0, 2, 1).unwrap();
    let firsts: BTreeSet<u32> = run.iter().map(|slot| at(&pool, slot)).collect();
    assert_eq!(
        firsts,
        BTreeSet::from([10, 11]),
        "the run lies in the span the raise takes"
    );
    assert!(
        matches!(pool.plan_rung(256), Err(SlotError::NoRoom { .. })),
        "free chunks 3 and 5 hold no run of two: never split, it waits"
    );
    free_of(&mut pool, &[6]);
    let evacuation = pool.plan_rung(256).expect("a free run of two below");
    let moved: BTreeSet<(u32, u32)> = evacuation
        .moves
        .iter()
        .map(|step| (step.from, step.to))
        .collect();
    assert!(
        moved.contains(&(10, 5)) && moved.contains(&(11, 6)),
        "whole: {moved:?}"
    );
    pool.apply_rung(&evacuation, 7).unwrap();
    pool.check();
}

/// A run with one chunk below the raise's span and
/// one in it moves whole (`moves_keeping_runs` under `plan_rung`): the
/// member below the limit goes with the one the span takes, never left
/// behind with its partner moved alone.
#[test]
fn a_raise_moves_a_run_straddling_its_span_whole() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 2), (256, 6)]));
    let holder = Holder::WorkingSet(1);
    pool.set_rung(16).unwrap();
    let low = pool.acquire_chunk_runs(holder, 0, 1, 8).unwrap();
    let at = |pool: &SlotPool, slot: &Slot| pool.chunk_table()[slot.chunk as usize];
    let run = pool.acquire_chunk_runs(holder, 0, 2, 1).unwrap();
    let firsts: BTreeSet<u32> = run.iter().map(|slot| at(&pool, slot)).collect();
    assert_eq!(
        firsts,
        BTreeSet::from([9, 10]),
        "the run straddles the span's limit"
    );
    let freed: Vec<Slot> = low
        .iter()
        .copied()
        .filter(|slot| [5, 6].contains(&at(&pool, slot)))
        .collect();
    pool.release(holder, &freed, 0);
    pool.retire_through(u64::MAX);
    let evacuation = pool.plan_rung(256).expect("a free run of two below");
    let moved: BTreeSet<(u32, u32)> = evacuation
        .moves
        .iter()
        .map(|step| (step.from, step.to))
        .collect();
    assert_eq!(
        moved,
        BTreeSet::from([(9, 5), (10, 6)]),
        "whole, both members"
    );
    pool.apply_rung(&evacuation, 7).unwrap();
    pool.check();
}

/// Metal pins its span's bottom at the
/// segment its widest carve with its guests' terms starts in
/// (`act_span_chunks`, 110 chunks: chunk 385), wider than the largest
/// rung's activation (65: segment 12, chunk 420). The pool's span
/// starts at the engine's: no state opens in 385..420 as a bottom
/// segment Metal never backs.
#[test]
fn the_span_starts_where_the_engine_pins_it() {
    let mut facts = facts_with(16, 4, 2, &[(ShapeKind::Fixed, 4)], &[(16, 2)]);
    facts.span_on_demand = true;
    let rungs = SlotPool::new(facts.clone());
    assert_eq!(
        rungs.top_segment(),
        3,
        "the rung's two chunks: the last segment"
    );
    facts.act_span = Some(6);
    let pinned = SlotPool::new(facts);
    assert_eq!(
        pinned.top_segment(),
        2,
        "six chunks from the end: segment 2 on"
    );
    pinned.check();
}

/// A beam-4 process's 72 runs of slots priced a whole chunk each where
/// its placement does not stand now would read past the pool, and
/// beams the budget serves be refused as never fitting. Placed alone, runs of one length
/// share chunks of their class: what held them once holds them again.
#[test]
fn small_slot_runs_placed_alone_share_chunks_of_their_class() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 2)]));
    let spc = pool.shape(0).unwrap().slots_per_chunk;
    let own = vec![OwnPiece::Slots { shape: 0, slots: 1 }; 72];
    let alone = pool.opens_alone(&own);
    assert_eq!(alone, 72u32.div_ceil(spc), "packed {spc} a chunk");
    let all = pool.free_chunks();
    let _held = pool
        .acquire_chunk_runs(Holder::WorkingSet(9), 0, 1, all)
        .unwrap();
    assert_eq!(pool.opens_beside(&own), None, "no room now");
    assert_eq!(pool.opens_alone(&own), alone, "placed alone, the same");
}

/// Whoever plans them, moves that would split a run are refused by
/// name: one of its chunks without the rest, or the rest not following
/// its first. A run moved whole is applied.
#[test]
fn moves_that_split_a_run_are_refused_whoever_planned_them() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 2)]));
    let holder = Holder::WorkingSet(1);
    let filler = pool.acquire_chunk_runs(holder, 0, 1, 4).unwrap();
    let run = pool.acquire_chunk_runs(holder, 0, 2, 1).unwrap();
    let at = |pool: &SlotPool, chunk: u32| pool.chunk_table()[chunk as usize];
    let low = filler
        .iter()
        .map(|slot| at(&pool, slot.chunk))
        .min()
        .unwrap();
    pool.release(holder, &filler, 0);
    pool.retire_through(u64::MAX);
    let mut chunks: Vec<u32> = run.iter().map(|slot| slot.chunk).collect();
    chunks.dedup();
    chunks.sort_by_key(|&chunk| at(&pool, chunk));
    let (head, second) = (chunks[0], chunks[1]);
    let (from, next) = (at(&pool, head), at(&pool, second));
    assert_eq!(next, from + 1);
    let alone = [ChunkMove {
        logical: second,
        from: next,
        to: low,
    }];
    assert!(matches!(
        pool.apply_compaction(&alone, 3),
        Err(SlotError::SplitsRun { .. })
    ));
    let apart = [
        ChunkMove {
            logical: head,
            from,
            to: low,
        },
        ChunkMove {
            logical: second,
            from: next,
            to: low + 2,
        },
    ];
    assert!(matches!(
        pool.apply_compaction(&apart, 3),
        Err(SlotError::SplitsRun { .. })
    ));
    let whole = [
        ChunkMove {
            logical: head,
            from,
            to: low,
        },
        ChunkMove {
            logical: second,
            from: next,
            to: low + 1,
        },
    ];
    pool.apply_compaction(&whole, 3).unwrap();
    pool.retire_through(u64::MAX);
    pool.check();
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

/// A chunk moved for a compaction and let go before the engine takes or
/// refuses the copy: taking the move back must not read its free id as
/// a shape ("index out of bounds" on SIGTERM). Taken back, a
/// chunk let go meanwhile returns to where the engine's table has it,
/// free, and the pool stays whole.
#[test]
fn a_move_taken_back_after_its_chunk_was_let_go_leaves_the_pool_whole() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 2)]));
    let holder = Holder::WorkingSet(1);
    let taken = pool.acquire_chunk_runs(holder, 0, 1, 6).unwrap();
    // Free the lowest chunks, then move the highest held one down.
    let low: Vec<Slot> = taken
        .iter()
        .copied()
        .filter(|slot| pool.chunk_table()[slot.chunk as usize] < 3)
        .collect();
    pool.release(holder, &low, 0);
    let high = taken
        .iter()
        .map(|slot| slot.chunk)
        .max_by_key(|&chunk| pool.chunk_table()[chunk as usize])
        .unwrap();
    let moves = pool.plan_moves_of(&[high]);
    assert_eq!(moves.len(), 1, "one move down");
    pool.apply_compaction(&moves, 5).unwrap();
    // Its process ends before the copy is taken or refused.
    let rest: Vec<Slot> = taken
        .iter()
        .copied()
        .filter(|slot| slot.chunk == high)
        .collect();
    pool.release(holder, &rest, 0);
    pool.retire_through(u64::MAX);
    pool.undo_compaction(&moves);
    pool.check();
    assert_eq!(
        pool.chunk_table()[high as usize],
        moves[0].from,
        "where the engine's table has it"
    );
}

/// An eviction's copy of state must not be refused "inside the
/// activation span's backed carve", failing the process. With the span backed on demand, a lower the engine
/// is not told leaves its carve as wide as before: the chunks it gave
/// back to the rung open for no state until the engine narrows.
#[test]
fn no_state_opens_where_the_engine_s_carve_still_lies() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut facts = facts_with(16, 4, 4, &fixed, &[(16, 2), (256, 6)]);
    facts.span_on_demand = true;
    let mut pool = SlotPool::new(facts);
    pool.set_rung(256).expect("the wide rung");
    let wide = pool.state_limit();
    pool.set_rung(16).expect("lowered, the engine not told");
    assert_eq!(
        pool.state_limit(),
        wide,
        "its carve still lies there: nothing lent"
    );
    pool.engine_narrowed();
    assert_eq!(pool.state_limit(), 16 - 2, "narrowed: the span is lent");
}

/// The planner must not read a run of four free (four free chunks,
/// adjacent) while the restore's unpark finds none, its serve having
/// taken its state first into the lowest free chunk.
/// One placement answers both: a restore's state taken as `acquire`
/// takes it opens the lowest free chunk, and the runs the store says
/// stand beside it are the runs left after that take.
#[test]
fn a_serve_s_state_is_placed_where_its_take_puts_it_before_the_runs() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 4)]));
    let holder = Holder::WorkingSet(1);
    let held = pool.acquire_chunk_runs(holder, 0, 1, 11).unwrap();
    let freed: Vec<Slot> = held
        .iter()
        .copied()
        .filter(|slot| (4..8).contains(&pool.physical(slot.chunk).unwrap()))
        .collect();
    pool.release(holder, &freed, 0);
    assert_eq!(
        pool.free_chunk_runs(4),
        1,
        "chunks 4..8 free: a run of four"
    );
    let take = [(0, 1)];
    assert_eq!(
        pool.opens_for(&take),
        Some(vec![4]),
        "the state opens the lowest free chunk"
    );
    let own = [OwnPiece::Take { shape: 0, slots: 1 }];
    assert!(
        !pool.chunk_runs_beside(&own, 4, 1),
        "after the serve's state, no run of four stands: the unpark's room"
    );
    assert!(pool.chunk_runs_beside(&own, 3, 1), "a run of three does");
    assert_eq!(pool.free_chunk_runs(4), 1, "read only");
}

/// A restore's own single-slot takes of one shape, its kv pages and its
/// unpark's, each read as finding the shape's partly filled slots,
/// would stand two free runs of three the takes then break
/// (`PendingBindShort`, one run left). Placed by the takes
/// themselves, the second take opens the chunk the first left full,
/// and one run of three is what stands, as the real takes find.
#[test]
fn a_placement_is_the_takes_own_and_counts_partly_filled_slots_once() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 4)]));
    let holder = Holder::WorkingSet(1);
    // Chunk 9 open with one slot free; 1..4 and 5..8 free (two runs
    // of three); the rest held (chunk 0 reserved, 12.. the activation).
    let held = pool.acquire_chunk_runs(holder, 0, 1, 11).unwrap();
    let at = |pool: &SlotPool, range: std::ops::RangeInclusive<u32>| -> Vec<Slot> {
        held.iter()
            .copied()
            .filter(|slot| range.contains(&pool.physical(slot.chunk).unwrap()))
            .collect()
    };
    let nine = at(&pool, 9..=9);
    pool.release(holder, &nine, 0);
    let _partial = pool.acquire(holder, 0, 3).unwrap();
    let runs: Vec<Slot> = at(&pool, 1..=3)
        .into_iter()
        .chain(at(&pool, 5..=7))
        .collect();
    pool.release(holder, &runs, 0);
    assert_eq!(
        pool.partial_free(0),
        1,
        "one slot free in the partly filled chunk"
    );
    assert_eq!(pool.free_chunk_runs(3), 2, "two free runs of three");
    let own = [
        OwnPiece::Take { shape: 0, slots: 1 },
        OwnPiece::Take { shape: 0, slots: 1 },
    ];
    let placement = pool.place(&own);
    assert!(placement.stands());
    assert_eq!(placement.opened.len(), 1, "the second take opens a chunk");
    assert_eq!(
        placement.free_runs(3),
        1,
        "one run of three stands after them"
    );
    assert!(
        !pool.chunk_runs_beside(&own, 3, 2),
        "the bind's two runs do not"
    );
    // The real takes find the same.
    pool.acquire(holder, 0, 1).unwrap();
    pool.acquire(holder, 0, 1).unwrap();
    assert_eq!(pool.free_chunk_runs(3), 1, "as the placement read");
    pool.check();
}

/// What a restore's parked pieces leave of the free runs, read where
/// the takes would put them: a run of slots goes back into a partly
/// filled chunk of its length (no free chunk taken), a whole-chunk
/// piece takes a free chunk; nothing is taken, and the real take puts
/// the run where the view did.
#[test]
fn a_parked_run_that_shares_a_partly_filled_chunk_leaves_the_free_runs_standing() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 4)]));
    let (holder, other) = (Holder::WorkingSet(1), Holder::WorkingSet(2));
    // Two runs of 2 share a chunk; one leaves: its hole is a run of 2
    // in a partly filled chunk.
    let kept = pool.acquire_run(other, 0, 2).unwrap();
    let gone = pool.acquire_run(holder, 0, 2).unwrap();
    assert_eq!(kept[0].chunk, gone[0].chunk, "one chunk");
    pool.release(holder, &gone, 0);
    let free = pool.free_chunk_runs(1);
    let snapshot = |pool: &SlotPool| {
        (
            pool.open_chunks(),
            pool.free_chunks(),
            pool.room_gives(),
            (1..=4)
                .map(|run| pool.free_chunk_runs(run))
                .collect::<Vec<_>>(),
        )
    };
    let before = snapshot(&pool);
    let own = [OwnPiece::Slots { shape: 0, slots: 2 }];
    assert!(
        pool.chunk_runs_beside(&own, 1, free),
        "the run goes back into the hole: every free chunk stands beside it"
    );
    assert!(!pool.chunk_runs_beside(&own, 1, free + 1));
    // A whole-chunk piece takes a free chunk.
    let whole = [OwnPiece::Chunks { run: 1, pieces: 1 }];
    assert!(!pool.chunk_runs_beside(&whole, 1, free));
    assert!(pool.chunk_runs_beside(&whole, 1, free - 1));
    assert_eq!(
        snapshot(&pool),
        before,
        "read only: the pool is the same after"
    );
    // And the real take puts it where the view did.
    let back = pool.acquire_run(holder, 0, 2).unwrap();
    assert_eq!(back[0].chunk, kept[0].chunk);
    assert_eq!(pool.free_chunk_runs(1), free);
}

/// Runs of adjacent chunks are taken inside one segment, lowest first,
/// never across a segment's end; a run past a segment is refused by
/// name, and too few free runs are refused naming what was free. An
/// evacuation moves a run whole onto adjacent free chunks of one lower
/// segment, its chunks in order.
/// An unpark's two runs of two beside a bind's run of three, the
/// two-stretches lowest. The placement is the takes' own: the runs of
/// three it leaves are those the real takes leave.
/// A device fault on an eviction's queued kv-to-disk copy: a segment a residency copy still reads is neither
/// evacuated nor released (unmapped) until the copy lands.
#[test]
fn a_segment_under_a_copy_is_neither_evacuated_nor_released() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 4)]));
    let holder = Holder::WorkingSet(1);
    let held = pool
        .acquire_chunk_runs(holder, 0, 1, pool.free_chunk_runs(1))
        .unwrap();
    // Segment 1 (chunks 4..8) keeps one held chunk, 5; the rest of the
    // pool below it free enough to take it.
    let (five, freed): (Vec<Slot>, Vec<Slot>) = held
        .iter()
        .copied()
        .partition(|slot| pool.physical(slot.chunk).unwrap() == 5);
    pool.release(holder, &freed, 0);
    assert!(
        pool.plan_evacuation(1).is_some(),
        "unfenced, it is evacuated"
    );
    pool.moving_out(&[5]);
    assert!(pool.plan_evacuation(1).is_none(), "under a copy, it is not");
    // Its chunk given back (the copy's commit) while the copy's fence
    // still stands: the empty segment is not released.
    pool.release(holder, &five, 0);
    assert!(
        pool.release_segment(1).is_err(),
        "not released under a copy"
    );
    pool.moved_out(&[5]);
    assert!(pool.release_segment(1).is_ok(), "released once it landed");
}

#[test]
fn a_placement_of_runs_leaves_what_the_real_takes_leave() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(32, 4, 8, &fixed, &[(16, 4)]));
    let holder = Holder::WorkingSet(1);
    let held = pool
        .acquire_chunk_runs(holder, 0, 1, pool.free_chunk_runs(1))
        .unwrap();
    let freed: Vec<Slot> = held
        .iter()
        .copied()
        .filter(
            |slot| matches!(pool.physical(slot.chunk).unwrap(), 1 | 2 | 5 | 6 | 8..=11 | 13..=15),
        )
        .collect();
    pool.release(holder, &freed, 0);
    let own = [OwnPiece::Chunks { run: 2, pieces: 2 }];
    let placement = pool.place(&own);
    assert!(placement.stands());
    let placed_runs = placement.free_runs(3);
    pool.acquire_chunk_runs(holder, 0, 2, 2).unwrap();
    assert_eq!(
        pool.free_chunk_runs(3),
        placed_runs,
        "what the real takes leave"
    );
    assert_eq!(
        placed_runs, 2,
        "the lowest two-stretches taken, both runs of three left"
    );
    pool.check();
}

/// A compaction never moves a chunk an eviction's copy still reads, and a
/// chunk held for a served restore is never its source or destination.
#[test]
fn a_compaction_moves_no_chunk_a_copy_reads_and_no_reserved_chunk() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let build = || {
        let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 4)]));
        let resident = Holder::WorkingSet(1);
        let singles: Vec<Vec<Slot>> = (0..11)
            .map(|_| pool.acquire_chunk_runs(resident, 0, 1, 1).unwrap())
            .collect();
        for slots in singles {
            if [2, 4, 6, 8, 10].contains(&pool.physical(slots[0].chunk).unwrap()) {
                pool.release(resident, &slots, 0);
            }
        }
        pool
    };
    // Unfenced, the cheapest window is 4..7, its chunk 5 moved.
    let moves = build().plan_compaction(3).expect("a run of three is made");
    assert_eq!(
        moves.iter().map(|step| step.from).collect::<Vec<_>>(),
        vec![5]
    );
    // An eviction's copy reads chunk 5: it is moved by no compaction.
    let mut pool = build();
    pool.moving_out(&[5]);
    let moves = pool.plan_compaction(3).unwrap_or_default();
    assert!(
        moves.iter().all(|step| step.from != 5 && step.to != 5),
        "{moves:?}"
    );
    // Chunks 2 and 4 held for a served restore: never a destination,
    // never inside a window cleared for another.
    let mut pool = build();
    let ticket = pool.reserve_chunks([2, 4]);
    let moves = pool.plan_compaction(3).unwrap_or_default();
    assert!(
        moves
            .iter()
            .all(|step| ![2, 4].contains(&step.to) && ![2, 4].contains(&step.from)),
        "{moves:?}"
    );
    assert_eq!(pool.reserved(ticket), vec![2, 4], "still held");
    pool.check();
}

#[test]
fn chunk_runs_lie_inside_a_segment_and_move_whole() {
    let fixed = [(ShapeKind::Fixed, 4)];
    // Four segments of 4 chunks, the top one under the activation.
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 4)]));
    let holder = Holder::WorkingSet(1);
    let physical = |pool: &SlotPool, slots: &[Slot]| -> Vec<u32> {
        let mut chunks: Vec<u32> = slots
            .iter()
            .map(|slot| pool.physical(slot.chunk).unwrap())
            .collect();
        chunks.dedup();
        chunks
    };
    // Segment 0 holds the null chunk: its run of 3 is 1..4.
    assert_eq!(pool.free_chunk_runs(3), 3);
    let first = pool.acquire_chunk_runs(holder, 0, 3, 1).unwrap();
    assert_eq!(physical(&pool, &first), [1, 2, 3]);
    assert_eq!(first.len(), 12, "every slot of each chunk");
    // A run of 2 then lies in segment 1, never across 3|4.
    let second = pool.acquire_chunk_runs(holder, 0, 2, 2).unwrap();
    assert_eq!(physical(&pool, &second), [4, 5, 6, 7]);
    assert_eq!(
        pool.acquire_chunk_runs(holder, 0, 5, 1),
        Err(SlotError::RunPastSegment { run: 5, segment: 4 })
    );
    assert_eq!(
        pool.acquire_chunk_runs(holder, 0, 3, 2),
        Err(SlotError::NoChunkRun {
            run: 3,
            pieces: 2,
            free_runs: 1,
            free_chunks: 4,
        })
    );
    // Segment 0 empties but for the null chunk; segment 2's run of 3
    // moves whole into 1..4.
    let third = pool.acquire_chunk_runs(holder, 0, 3, 1).unwrap();
    assert_eq!(physical(&pool, &third), [8, 9, 10]);
    pool.release(holder, &first, 0);
    let evacuation = pool.plan_evacuation(2).expect("the run fits segment 0");
    let moves: Vec<(u32, u32)> = evacuation
        .moves
        .iter()
        .map(|step| (step.from, step.to))
        .collect();
    assert_eq!(moves, [(8, 1), (9, 2), (10, 3)]);
    pool.apply_evacuation(&evacuation, 1).unwrap();
    assert_eq!(physical(&pool, &third), [1, 2, 3], "adjacent, in order");
    // Where no lower segment has the run free, the segment stays.
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 4)]));
    let run = pool.acquire_chunk_runs(holder, 0, 3, 1).unwrap();
    let singles = pool.acquire_chunk_runs(holder, 0, 1, 2).unwrap();
    assert_eq!(physical(&pool, &singles), [4, 5]);
    let far = pool.acquire_chunk_runs(holder, 0, 3, 1).unwrap();
    assert_eq!(physical(&pool, &far), [8, 9, 10]);
    pool.release(holder, &run, 0);
    // Segment 0 has 1..4 free again: segment 2's run fits there.
    assert!(pool.plan_evacuation(2).is_some());
    let taken = pool.acquire_chunk_runs(holder, 0, 1, 1).unwrap();
    assert_eq!(physical(&pool, &taken), [1]);
    assert!(
        pool.plan_evacuation(2).is_none(),
        "2..4 and 6..8 free, four chunks, and no run of 3 in a lower segment"
    );
}

/// A guest bind's run: free chunks enough for a run of 3 but none adjacent
/// (pages and slots of many owners interleave across chunks), nothing
/// left to evict. The cheapest window is cleared, one held chunk moved
/// out on its logical id, its chunk closed until the fire carrying the
/// copy retires; then the run stands and the bind takes it. Every
/// resident reads its bytes where its refs lead, as before the move.
/// Free chunks too few for the run whatever moves: no compaction.
/// A head short of a second run of its length, one standing and free
/// chunks enough elsewhere, is planned a compaction though a run
/// stands. One more run is cleared beside the one
/// standing, which neither its window nor its moves touch.
#[test]
fn a_second_run_is_compacted_beside_the_one_standing() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 4)]));
    let resident = Holder::WorkingSet(1);
    let singles: Vec<Vec<Slot>> = (0..11)
        .map(|_| pool.acquire_chunk_runs(resident, 0, 1, 1).unwrap())
        .collect();
    for slots in singles {
        if [1, 2, 5, 9].contains(&pool.physical(slots[0].chunk).unwrap()) {
            pool.release(resident, &slots, 0);
        }
    }
    assert_eq!(pool.free_chunk_runs(2), 1, "one run of two stands (1..3)");
    assert_eq!(pool.plan_compaction(2), None, "a run stands: none for one");
    let moves = pool
        .plan_compaction_for(2, 2)
        .expect("a second run is cleared");
    assert_eq!(moves.len(), 1);
    assert!(
        moves
            .iter()
            .all(|step| ![1, 2].contains(&step.from) && ![1, 2].contains(&step.to)),
        "the standing run untouched: {moves:?}"
    );
    pool.apply_compaction(&moves, 3).unwrap();
    pool.retire_through(3);
    assert_eq!(pool.free_chunk_runs(2), 2, "two runs stand");
    pool.check();
}

#[test]
fn scattered_free_chunks_are_compacted_into_a_run_and_residents_keep_their_bytes() {
    let fixed = [(ShapeKind::Fixed, 4)];
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 4)]));
    let resident = Holder::WorkingSet(1);
    let singles: Vec<Vec<Slot>> = (0..11)
        .map(|_| pool.acquire_chunk_runs(resident, 0, 1, 1).unwrap())
        .collect();
    // The engine's bytes, by physical chunk: each chunk's own mark.
    let mut bytes = vec![0u32; 16];
    for slots in &singles {
        let logical = slots[0].chunk;
        bytes[pool.physical(logical).unwrap() as usize] = 1000 + logical;
    }
    let marks: Vec<(Slot, u32)> = singles
        .iter()
        .flatten()
        .map(|&slot| (slot, 1000 + slot.chunk))
        .collect();
    // Free 2, 4, 6, 8, 10: five free, no three adjacent in a segment.
    let mut kept = Vec::new();
    for (i, slots) in singles.into_iter().enumerate() {
        let physical = pool.physical(slots[0].chunk).unwrap();
        if [2, 4, 6, 8, 10].contains(&physical) {
            pool.release(resident, &slots, 0);
        } else {
            kept.push((i, slots));
        }
    }
    assert_eq!(pool.free_chunk_runs(1), 5);
    assert_eq!(pool.free_chunk_runs(3), 0, "no run of 3 stands");
    assert!(
        matches!(
            pool.acquire_chunk_runs(Holder::WorkingSet(2), 0, 3, 1),
            Err(SlotError::NoChunkRun { .. })
        ),
        "the bind is short"
    );

    let moves = pool
        .plan_compaction(3)
        .expect("five free chunks stand a run of 3");
    assert_eq!(moves.len(), 1, "the cheapest window: one chunk moved");
    assert_eq!(
        (moves[0].from, moves[0].to),
        (5, 2),
        "4..7 cleared, its 5 to the lowest free"
    );
    pool.apply_compaction(&moves, 7).unwrap();
    // The engine's copy (`Engine::evacuate`): bytes follow the logical chunk.
    for step in &moves {
        bytes[step.to as usize] = bytes[step.from as usize];
    }
    assert_eq!(
        pool.free_chunk_runs(3),
        0,
        "closed until the copy's fire retires"
    );
    assert!(pool.compacting());
    assert_eq!(pool.plan_compaction(3), None, "one at a time");
    pool.retire_through(7);
    assert!(!pool.compacting());
    assert_eq!(pool.free_chunk_runs(3), 1, "the run stands");
    let bound = pool
        .acquire_chunk_runs(Holder::WorkingSet(2), 0, 3, 1)
        .expect("the bind takes it");
    let mut chunks: Vec<u32> = bound
        .iter()
        .map(|slot| pool.physical(slot.chunk).unwrap())
        .collect();
    chunks.dedup();
    assert_eq!(chunks, [4, 5, 6]);
    for (slot, mark) in &marks {
        if kept.iter().any(|(_, slots)| slots.contains(slot)) {
            assert_eq!(
                bytes[pool.physical(slot.chunk).unwrap() as usize],
                *mark,
                "a resident reads its own bytes through its unchanged ref"
            );
        }
    }
    assert_eq!(pool.plan_compaction(3), None, "a run stands: nothing to do");

    // Over-demand: two free chunks never make a run of 3.
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 4)]));
    let all: Vec<Vec<Slot>> = (0..11)
        .map(|_| pool.acquire_chunk_runs(resident, 0, 1, 1).unwrap())
        .collect();
    for slots in all {
        if [3, 9].contains(&pool.physical(slots[0].chunk).unwrap()) {
            pool.release(resident, &slots, 0);
        }
    }
    assert_eq!(
        pool.plan_compaction(3),
        None,
        "too few free chunks: it waits"
    );
}

/// Every return of slots tells the room signal once: a release whose
/// slots no fire still reads, at once; one a fire still reads, when that
/// fire's epoch retires, not before. What wakes a planner's waiter
/// whichever path gave the room back.
#[test]
fn slots_coming_free_tell_the_room_signal() {
    let mut pool = run_pool();
    let told = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let count = std::sync::Arc::clone(&told);
    pool.set_room_signal(
        RoomSignal(std::sync::Arc::new(move |_| {
            count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        })),
        RoomTier::Device,
        Some(0),
    );
    let holder = Holder::WorkingSet(1);
    let slots = pool.acquire(holder, 0, 10).unwrap();
    let told_now = || told.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(told_now(), 0, "taking tells nothing");
    pool.release(holder, &slots[..5], 0);
    assert_eq!(told_now(), 1, "freed at once: told once");
    pool.release(holder, &slots[5..], 7);
    assert_eq!(told_now(), 1, "waiting on a fire: not yet free");
    pool.retire_through(7);
    assert_eq!(told_now(), 2, "the fire retired: told");
    pool.retire_through(8);
    assert_eq!(told_now(), 2, "nothing more came free");
}

/// The run-class pool these tests take from: 1024-slot chunks of one
/// fixed shape, eleven of them free for state (the null chunk and the
/// top segment's activation aside).
fn run_pool() -> SlotPool {
    SlotPool::new(facts_with(
        16,
        4,
        4,
        &[(ShapeKind::Fixed, 1024)],
        &[(16, 4)],
    ))
}

/// A compaction moves a held chunk out of segment 1, swapping a free
/// logical id onto its physical there; once the move's fence lifts, a
/// take opens that free logical where it now lies, in segment 1. The
/// segment is no longer empty, whatever logical ids it holds: its
/// release is refused, by its physical chunks, never by logical ids
/// (a logical swapped onto a physical of the segment, taken there,
/// must not have the segment released under it).
#[test]
fn a_segment_holding_a_logical_swapped_onto_it_is_not_released() {
    let mut pool = SlotPool::new(facts_with(16, 4, 4, &[(ShapeKind::Fixed, 1)], &[]));
    let holder = Holder::WorkingSet(1);
    // Hold chunks 1..=7; segment 1 is 4..8.
    let held = pool.acquire(holder, 0, 7).unwrap();
    let in_segment: Vec<Slot> = held
        .iter()
        .copied()
        .filter(|slot| pool.logical_to_physical[slot.chunk as usize] >= 4)
        .collect();
    // Free chunks 8.. lie in segments 2 and 3: move segment 1's held
    // chunks there by a compaction, as a run's window would.
    let moves: Vec<ChunkMove> = in_segment
        .iter()
        .zip(8u32..)
        .map(|(slot, to)| ChunkMove {
            logical: slot.chunk,
            from: pool.logical_to_physical[slot.chunk as usize],
            to,
        })
        .collect();
    pool.apply_compaction(&moves, 1).unwrap();
    pool.retire_through(u64::MAX);
    assert_eq!(pool.segments.used(1), 0, "segment 1 emptied");
    // A take opens the lowest free chunk: one of segment 1's, its
    // logical a free id the moves swapped there.
    let taken = pool.acquire(Holder::WorkingSet(2), 0, 1).unwrap();
    let physical = pool.logical_to_physical[taken[0].chunk as usize];
    assert!(
        (4..8).contains(&physical),
        "taken where the swap left it: {physical}"
    );
    assert_eq!(pool.segments.used(1), 1);
    assert!(
        pool.release_segment(1).is_err(),
        "a segment holding a chunk at one of its physicals is not released"
    );
    pool.check();
}

/// Two runs of thirty asked where one free run of thirty stands among
/// scattered free chunks (a restore's guest state of two runs of 30,
/// one such run free of 110 free chunks): the placement
/// is short, as the takes are; with a second run opened, it stands.
#[test]
fn a_placement_of_two_runs_lays_each_disjointly() {
    let mut pool = SlotPool::new(facts_with(105, 35, 3, &[(ShapeKind::Fixed, 1)], &[]));
    let holder = Holder::WorkingSet(1);
    // Every chunk held, then segment 1 given back whole and every other
    // chunk of segment 2.
    let held = pool.acquire(holder, 0, 104).unwrap();
    let back: Vec<Slot> = held
        .iter()
        .copied()
        .filter(|slot| {
            let physical = pool.logical_to_physical[slot.chunk as usize];
            (35..70).contains(&physical) || (physical >= 70 && physical % 2 == 0)
        })
        .collect();
    pool.release(holder, &back, 0);
    pool.retire_through(u64::MAX);
    assert_eq!(pool.free_chunk_runs(30), 1);
    assert!(
        pool.free_chunks() >= 50,
        "free chunks enough: {}",
        pool.free_chunks()
    );
    let two = [OwnPiece::Chunks { run: 30, pieces: 2 }];
    assert!(
        !pool.place(&two).stands(),
        "one free run lays one piece, not two"
    );
    let one = [OwnPiece::Chunks { run: 30, pieces: 1 }];
    assert!(pool.place(&one).stands());
    // Nor two pieces asked apart, one after the other.
    let apart = [
        OwnPiece::Chunks { run: 30, pieces: 1 },
        OwnPiece::Chunks { run: 30, pieces: 1 },
    ];
    assert_eq!(pool.place(&apart).placed, 1);
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

/// A freed run's hole is one the next run of its length takes exactly:
/// runs lie only among runs of their own length, single slots never
/// among them.
#[test]
fn a_freed_run_is_reused_exactly_by_the_next_run_of_its_length() {
    let mut pool = run_pool();
    let holder = Holder::WorkingSet(1);
    let a = pool.acquire_run(holder, 0, 483).unwrap();
    let singles = pool.acquire(holder, 0, 10).unwrap();
    let b = pool.acquire_run(holder, 0, 483).unwrap();
    assert_eq!(
        a[0].chunk, b[0].chunk,
        "two runs of one length share a chunk"
    );
    assert!(
        singles.iter().all(|slot| slot.chunk != a[0].chunk),
        "no single among runs"
    );
    pool.release(holder, &a, 0);
    let again = pool.acquire_run(holder, 0, 483).unwrap();
    assert_eq!(
        again, a,
        "the next run of its length takes the hole exactly"
    );
    pool.check();
}

/// A chunk whose runs all leave is free whole again, for any class.
#[test]
fn a_chunk_whose_runs_all_leave_is_whole_again() {
    let mut pool = run_pool();
    let holder = Holder::WorkingSet(1);
    let free = pool.free_chunks();
    let a = pool.acquire_run(holder, 0, 483).unwrap();
    let b = pool.acquire_run(holder, 0, 483).unwrap();
    assert_eq!(pool.free_chunks(), free - 1, "both in one chunk");
    pool.release(holder, &a, 0);
    pool.release(holder, &b, 0);
    assert_eq!(pool.free_chunks(), free, "the chunk is whole again");
    let other = pool.acquire_run(holder, 0, 700).unwrap();
    assert_eq!(
        pool.free_chunks(),
        free - 1,
        "a run of another length opens it"
    );
    assert_eq!(other.len(), 700);
    pool.check();
}

/// AMD's 500-concurrent shape: single slots scattered beside guest
/// state's runs left no chunk holding a run of 483 free, though slots
/// were free in plenty. Single slots now never take a run chunk's
/// tail: with every other chunk filled by single slots and holes
/// scattered among them, the next run of 483 still lies in its own
/// class's chunk. Before, the single slots filled the fullest chunk
/// first, the run's, and the run was short.
#[test]
fn single_slots_never_fill_a_run_chunk_s_tail() {
    let mut pool = run_pool();
    let holder = Holder::WorkingSet(1);
    let first = pool.acquire_run(holder, 0, 483).unwrap();
    let room = u64::from(pool.free_chunks()) * 1024;
    let singles = pool.acquire(holder, 0, room).unwrap();
    assert!(singles.iter().all(|slot| slot.chunk != first[0].chunk));
    assert_eq!(pool.free_chunks(), 0, "every chunk taken");
    // Holes scattered among the single slots: no run of 483 there.
    let scattered: Vec<Slot> = singles.iter().step_by(2).copied().collect();
    pool.release(holder, &scattered, 0);
    let second = pool
        .acquire_run(holder, 0, 483)
        .expect("the run's own chunk holds it");
    assert_eq!(second[0].chunk, first[0].chunk);
    pool.check();
}
