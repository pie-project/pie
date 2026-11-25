use super::*;
use crate::inferlet::host::working_set::Cursor;

/// A lane's rows in a fire: its own, and what it carries in unfolded.
/// A host-placed lane carries the rows between its state and its first
/// row; a lane with no recurrent state, nothing.
#[test]
fn a_lane_counts_its_own_rows_and_the_rows_it_carries() {
    let host = lane_rows;
    assert_eq!(host(105, 113, Some(100)), FireRows { own: 8, carried: 5 });
    assert_eq!(host(100, 101, Some(100)), FireRows { own: 1, carried: 0 });
    assert_eq!(host(105, 113, None), FireRows { own: 8, carried: 0 });
    // A fire's are its lanes' summed.
    assert_eq!(
        FireRows::of([host(105, 113, Some(100)), host(102, 107, Some(100))]),
        FireRows {
            own: 13,
            carried: 7
        }
    );
}

/// A set whose prompt the host placed through slot 105, its state
/// folded through 100, with `reserved` slots granted.
fn decoding(reserved: u32) -> Cursor {
    Cursor {
        tail: Some(105),
        reserved,
        written: 105,
        ..Cursor::default()
    }
}

/// A lane the device places carries no more than stands unfolded in
/// its set: the rows between its state and the set's written end, and
/// every row of the fires staged and not yet settled. With `n` fires
/// of two rows staged since the last settle it asks for `5 + 2n`.
#[test]
fn a_device_lane_asks_for_what_stands_unfolded_in_its_set() {
    use crate::inferlet::host::working_set::StagedRows;
    let mut set = decoding(184);
    assert_eq!(
        device_lane_rows(2, &set, Some(100)),
        FireRows { own: 2, carried: 5 }
    );
    assert_eq!(device_lane_rows(2, &set, None).carried, 0);
    let mut fires = Vec::new();
    for n in 1..=10u32 {
        let fire = StagedRows::new(2);
        set.stage(fire.clone());
        fires.push(fire);
        let rows = device_lane_rows(2, &set, Some(100));
        assert_eq!(rows.carried, 5 + 2 * n);
        assert_eq!(rows.ask(), 5 + 2 * n);
    }
    // The first settles, its rows ended at 107 and one slot frozen:
    // its rows are counted where they landed, not as staged.
    fires[0].settle(Some(107));
    assert_eq!(set.standing(), (107, 18));
    assert_eq!(device_lane_rows(2, &set, Some(100)).carried, 25);
    assert_eq!(device_lane_rows(2, &set, Some(101)).carried, 24);
    // The second rewound and ended at 106: the rows through 107 are
    // still buffered, so the written end stays.
    fires[1].settle(Some(106));
    assert_eq!(set.standing(), (107, 16));
    // The rest settle; the set froze through 107: nothing is carried,
    // and the fire asks for its own rows.
    for fire in &fires[2..] {
        fire.settle(Some(107));
    }
    assert_eq!(
        device_lane_rows(2, &set, Some(107)),
        FireRows { own: 2, carried: 0 }
    );
    assert_eq!(device_lane_rows(2, &set, Some(107)).ask(), 2);
    // Staging takes in what settled.
    set.stage(StagedRows::new(2));
    assert_eq!((set.written, set.staged.len()), (107, 1));
    // A fire's rows are its lanes' summed, each lane of a set carrying
    // what stands unfolded in it.
    let lanes = [
        device_lane_rows(2, &set, Some(107)),
        device_lane_rows(3, &set, Some(107)),
    ];
    assert_eq!(FireRows::of(lanes), FireRows { own: 5, carried: 4 });
}

/// A fire that settles with no end reported (it failed, or its engine
/// tells none) wrote inside what its set reserved; a fork made while a
/// fire is in flight shares its rows, and counts them until it settles.
#[test]
fn a_fire_with_no_end_wrote_inside_what_was_reserved_and_a_fork_shares_it() {
    use crate::inferlet::host::working_set::StagedRows;
    let mut set = decoding(184);
    let fire = StagedRows::new(2);
    set.stage(fire.clone());
    let fork = set.clone();
    assert_eq!(fork.unfolded(100), 7);
    assert_eq!(fork.written_end(), 184, "a snapshot states no end it lacks");
    fire.settle(Some(107));
    assert_eq!((set.unfolded(100), fork.unfolded(100)), (7, 7));
    assert_eq!((set.standing(), fork.standing()), ((107, 0), (107, 0)));
    assert_eq!(fork.written_end(), 107);

    let fire = StagedFire {
        rows: StagedRows::new(2),
        tags: vec![7],
    };
    set.stage(fire.rows.clone());
    // Its lane reported, but the fire failed.
    let reported = std::collections::HashMap::from([(
        7,
        ::engine::KvWrites {
            tag: 7,
            end: Some(109),
            ..::engine::KvWrites::default()
        },
    )]);
    fire.settle(&reported, false);
    assert_eq!(set.standing(), (184, 0));

    let mut set = decoding(184);
    let fire = StagedFire {
        rows: StagedRows::new(2),
        tags: vec![7, 8],
    };
    set.stage(fire.rows.clone());
    // One of its two lanes reported no end.
    fire.settle(&reported, true);
    assert_eq!(set.standing(), (184, 0));

    let mut set = decoding(184);
    let fire = StagedFire {
        rows: StagedRows::new(2),
        tags: vec![7],
    };
    set.stage(fire.rows.clone());
    fire.settle(&reported, true);
    assert_eq!(set.standing(), (109, 0));

    // Dropped unsettled.
    let mut set = decoding(184);
    let fire = StagedFire {
        rows: StagedRows::new(2),
        tags: vec![7],
    };
    set.stage(fire.rows.clone());
    drop(fire);
    assert_eq!(set.standing(), (184, 0));
}

/// The bound is never below what the engine replays: over runs of
/// fires staged ahead, each landing wherever a lane may (at or past
/// the engine's folded length, no further than rows were written),
/// folding what its program says, and settling later, the rows
/// `buffered_reads` returns for the fire are no more than it stated
/// when it was staged.
#[test]
fn the_bound_is_never_below_what_the_engine_replays() {
    use crate::inferlet::host::working_set::StagedRows;
    struct Fire {
        own: u32,
        stated: u32,
        rows: Arc<StagedRows>,
        /// Once run: where its rows ended, and what it folded.
        ran: Option<(u32, u32)>,
    }
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut pick = |below: u32| {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((seed >> 33) % u64::from(below.max(1))) as u32
    };
    let mut replayed = 0u32;
    for depth in [1usize, 2, 10] {
        let mut set = decoding(4096);
        // The runtime's frozen length, the engine's folded length (past
        // it while a device-counted fold has not settled), and the
        // furthest slot written.
        let (mut frozen, mut folded, mut end) = (100u32, 100u32, 105u32);
        let mut fires: std::collections::VecDeque<Fire> = Default::default();
        for _ in 0..4000 {
            let running = fires.iter().position(|fire| fire.ran.is_none());
            match pick(3) {
                0 if fires.len() < depth => {
                    let own = 1 + pick(5);
                    let stated = device_lane_rows(own, &set, Some(frozen)).carried;
                    let rows = StagedRows::new(own);
                    set.stage(rows.clone());
                    fires.push_back(Fire {
                        own,
                        stated,
                        rows,
                        ran: None,
                    });
                }
                1 if running.is_some() => {
                    let fire = &mut fires[running.unwrap()];
                    // Anywhere from its state to the furthest row written:
                    // on, or rewound over rows it rejects.
                    let first = folded + pick(end - folded + 1);
                    let reads = ::engine::fire::buffered_reads(
                        None,
                        folded,
                        first,
                        ::engine::fire::Lineage::Set,
                    )
                    .unwrap();
                    let carried: u32 = reads.iter().map(|span| span.end - span.start).sum();
                    // A decode lane (one row) states none: its step
                    // replays them as data, beside its own row.
                    assert!(
                        carried <= fire.stated || fire.own == 1,
                        "replays {carried} rows, stated {}",
                        fire.stated
                    );
                    replayed = replayed.max(carried);
                    // Most fires fold nothing (their rows stand buffered);
                    // one in four folds any part of what stands.
                    let fold = match pick(4) {
                        0 => pick(first + fire.own - folded + 1),
                        _ => 0,
                    };
                    folded += fold;
                    end = end.max(first + fire.own);
                    fire.ran = Some((first + fire.own, fold));
                }
                _ => {
                    if fires.front().is_some_and(|fire| fire.ran.is_some()) {
                        let fire = fires.pop_front().unwrap();
                        let (ended, fold) = fire.ran.unwrap();
                        fire.rows.settle(Some(ended));
                        frozen += fold;
                    }
                }
            }
        }
    }
    assert!(replayed > 16, "the runs replay past what a rung of 16 held");
}

/// The rows a device-placed fire may carry raise the rung through the
/// ask every pass makes: the fire enqueues once the rung takes them.
#[tokio::test(flavor = "current_thread")]
async fn a_device_fire_asks_the_rung_for_the_rows_it_may_carry() {
    use crate::inferlet::host::working_set::StagedRows;
    const ENGINE: usize = 48;
    let stores = pool_of_one_row_rs_pages();
    let pid = uuid::Uuid::new_v4();
    crate::store::shared::set_fire_cap_for_test(ENGINE, 8);
    // Two rows of its own, behind ten fires of two rows still to
    // settle and five rows unfolded.
    let mut set = decoding(184);
    for _ in 0..10 {
        set.stage(StagedRows::new(2));
    }
    let rows = device_lane_rows(2, &set, Some(100)).ask();
    assert_eq!(rows, 25);
    let raise = tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 25);
        crate::store::shared::set_fire_cap_for_test(ENGINE, 32);
    });
    let ask = ask_rung(&stores, ENGINE, pid, pid, rows as usize, rows).expect("it asks");
    tokio::time::timeout(std::time::Duration::from_secs(5), ask.wait(rows))
        .await
        .expect("no indefinite wait")
        .expect("the rung takes it");
    raise.await.unwrap();
    assert!(crate::store::shared::fire_cap(ENGINE) >= 25);
    drop(ask);
    assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);
}

/// A fire carrying no more rows than its own asks its rung for what it
/// asked before: its own rows, in the pass's ask and in what the
/// scheduler steps and holds it by.
#[test]
fn a_fire_carrying_no_more_than_its_own_rows_asks_as_before() {
    const ENGINE: usize = 47;
    for carried in [0, 1, 7, 8] {
        assert_eq!(FireRows { own: 8, carried }.ask(), 8);
        let mut req = lane(100 + carried, 8);
        let plain = req.rung_rows();
        req.carried = carried;
        assert_eq!(req.rung_rows(), plain);
        assert_eq!(plain, 8);
    }
    // A decode of one row carrying none, or the one row before it: a
    // lone row the cap covers asks nothing.
    let stores = pool_of_one_row_rs_pages();
    let pid = uuid::Uuid::new_v4();
    crate::store::shared::set_fire_cap_for_test(ENGINE, 1);
    for carried in [0, 1] {
        let decode = FireRows { own: 1, carried };
        let rows = decode.ask();
        assert!(ask_rung(&stores, ENGINE, pid, pid, rows as usize, rows).is_none());
    }
    assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);
}

/// A fire carrying more rows than its own asks for those, as a fire
/// that wide does: its pass asks whatever the cap, and what the
/// scheduler holds it by is what it carries.
#[tokio::test(flavor = "current_thread")]
async fn a_fire_carrying_more_than_its_own_rows_asks_as_one_that_wide() {
    const ENGINE: usize = 46;
    let draft = FireRows {
        own: 5,
        carried: 40,
    };
    assert_eq!(draft.ask(), 40);
    let mut req = lane(140, 5);
    req.carried = 40;
    assert_eq!(req.rung_rows(), 40);
    // A lone row carrying a handful asks for them under a cap that
    // covers them too: the walker cuts nothing under the ask.
    let few = FireRows {
        own: 1,
        carried: 12,
    };
    let stores = pool_of_one_row_rs_pages();
    let pid = uuid::Uuid::new_v4();
    crate::store::shared::set_fire_cap_for_test(ENGINE, 16);
    let rows = few.ask();
    let ask = ask_rung(&stores, ENGINE, pid, pid, rows as usize, rows)
        .expect("a row carrying twelve asks as twelve rows do");
    assert_eq!(crate::store::shared::wanted_rows(ENGINE), 12);
    drop(ask);
    assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);
}

/// At the contract's floor (a rung stating nothing holds its tokens of
/// own rows and as many carried) no rung's replay scratch holds more
/// carried rows than the largest rung's tokens: a fire carrying more is
/// refused by name, naming the bound, before anything is submitted;
/// own rows past it are another refusal's (the capacity of a fire).
#[test]
fn a_fire_carrying_past_the_largest_rung_is_refused_by_name() {
    let (_, stores) = pool_with_rungs_of(vec![(256, 2), (2048, 8)]);
    let carrying = |carried| FireRows { own: 5, carried };
    assert_eq!(replay_rung(&stores, 56, carrying(0), 0, 1, false), Ok(5));
    assert_eq!(replay_rung(&stores, 56, carrying(2048), 2048, 1, false), Ok(2048));
    let host = replay_rung(&stores, 56, carrying(2049), 2049, 1, false).unwrap_err();
    assert!(
        host.contains("of 2048 tokens")
            && host.contains("2049 carried recurrent rows")
            && host.contains("fold what its lanes have written before they write more"),
        "{host}"
    );
    assert!(replay_rung(&stores, 56, FireRows { own: 4096, carried: 0 }, 0, 1, false).is_ok());
}

/// A pass cut into pieces: one folding all it writes folds piece by
/// piece, so only its first piece carries rows in; one folding nothing
/// carries its earlier pieces' rows into each later one.
#[test]
fn a_cut_pass_counts_what_each_piece_carries() {
    // Folded through 12, 8 rows written since, 30 more in three pieces.
    let full = lane(20, 30);
    let frozen = [Some(12)];
    let piece = |from, to| vec![Some((from, to))];
    let cuts = [piece(0, 10), piece(10, 20), piece(20, 30)];
    let rows = |folds_each: bool| -> Vec<FireRows> {
        cuts.iter()
            .enumerate()
            .map(|(at, cut)| piece_rows(&full, cut, &frozen, folds_each && at > 0))
            .collect()
    };
    let own = |carried| FireRows { own: 10, carried };
    assert_eq!(rows(true), [own(8), own(0), own(0)]);
    assert_eq!(rows(false), [own(8), own(18), own(28)]);
    assert_eq!(
        rows(false).iter().map(|rows| rows.ask()).max(),
        Some(28),
        "the widest ask of a pass folding nothing is what its last piece carries"
    );
    // The whole pass in one fire carries what its first piece does.
    assert_eq!(
        piece_rows(&full, &piece(0, 30), &frozen, false),
        FireRows {
            own: 30,
            carried: 8
        }
    );
    // A lane with no recurrent state carries nothing; one with no rows
    // in a piece counts nothing.
    assert_eq!(piece_rows(&full, &piece(10, 20), &[None], false), own(0));
    assert_eq!(
        piece_rows(&full, &[None], &frozen, false),
        FireRows::default()
    );
}

/// The rung a fire asks is the smallest whose stated replay scratch
/// holds it, each bound counted by the engine crate's functions:
///
/// - Metal-shaped (own T, carried T counted the `lane_carried` way, total
///   2T of scratch rows; an rs scratch laying every carried row): a w8
///   replay (8 decode lanes, 8 own rows, 1052 carried, a decode lane's
///   counting none toward carried) is served at the 1024 rung; a w11
///   replay of 929 composed own rows and 1144 carried (2073 scratch rows)
///   is refused by name at the top rung, its total binding, though its
///   lanes are few.
/// - CUDA-shaped (own T, carried the widest bucket's tokens, no total):
///   both are held to their own and carried rows alone.
/// - Each bound binds alone, the refusal naming it.
#[test]
fn a_fire_s_rung_holds_its_stated_replay_bounds() {
    use ::engine::RungReplay;
    let metal = |tokens: u32| RungReplay {
        own_max: Some(tokens),
        carried_max: Some(tokens),
        scratch: rows_bytes(2 * tokens),
    };
    let rungs = vec![(256, 2), (1024, 8)];
    let (_, stores) = pool_stating(rungs.clone(), vec![metal(256), metal(1024)], false);
    // Eight decode lanes, each replaying its share of 1052 rows.
    let w8 = FireRows { own: 8, carried: 0 };
    assert_eq!(replay_rung(&stores, 56, w8, 1052, 8, false), Ok(1024), "served at the 1024 rung");
    let w11 = FireRows { own: 929, carried: 0 };
    let refused = replay_rung(&stores, 56, w11, 1144, 11, false).unwrap_err();
    assert!(
        refused.contains("of 1024 tokens") && refused.contains("2077 bytes of rs scratch"),
        "{refused}"
    );
    // On a compact rs scratch a decode lane's rows take none of it.
    let (_, compact) = pool_stating(rungs.clone(), vec![metal(256), metal(1024)], true);
    assert_eq!(replay_rung(&compact, 56, w11, 1144, 11, true), Ok(929));

    // CUDA-shaped: carried held to the widest bucket's tokens, no total.
    let cuda = |tokens: u32| RungReplay {
        own_max: Some(tokens),
        carried_max: Some(1024),
        scratch: rows_bytes(0).unbound(),
    };
    let (_, cuda_stores) = pool_stating(rungs.clone(), vec![cuda(256), cuda(1024)], false);
    assert_eq!(replay_rung(&cuda_stores, 56, w8, 1052, 8, false), Ok(8));
    assert_eq!(replay_rung(&cuda_stores, 56, w11, 1144, 11, false), Ok(929), "no total bound: its own rows ask");

    // Each bound alone.
    let only = |bound: RungReplay| pool_stating(vec![(64, 1)], vec![bound], false).1;
    let none = RungReplay {
        own_max: None,
        carried_max: None,
        scratch: rows_bytes(0).unbound(),
    };
    let own = only(RungReplay {
        own_max: Some(64),
        ..none.clone()
    });
    assert!(replay_rung(&own, 56, FireRows { own: 65, carried: 0 }, 0, 1, false).is_ok(), "own past: capacity's");
    let carried = only(RungReplay {
        carried_max: Some(64),
        ..none.clone()
    });
    let why = replay_rung(&carried, 56, FireRows { own: 2, carried: 65 }, 65, 1, false).unwrap_err();
    assert!(why.contains("65 carried recurrent rows"), "{why}");
    let total = only(RungReplay {
        scratch: rows_bytes(64),
        ..none.clone()
    });
    let why = replay_rung(&total, 56, FireRows { own: 4, carried: 0 }, 61, 1, false).unwrap_err();
    assert!(why.contains("69 bytes of rs scratch"), "{why}");
    assert!(replay_rung(&total, 56, FireRows { own: 4, carried: 0 }, 60, 1, false).is_ok());
}

/// A host pass whose uncut own rows and carried rows together pass the
/// widest rung's replay scratch, but whose pieces each fit, is served in
/// pieces: the whole-pass check counts no own row, each piece its own
/// as it is cut. A piece that no cut brings under a rung is refused by
/// name.
#[test]
fn a_pass_too_wide_uncut_is_served_in_pieces_that_fit() {
    let metal = |tokens: u32| ::engine::RungReplay {
        own_max: Some(tokens),
        carried_max: Some(tokens),
        scratch: rows_bytes(2 * tokens),
    };
    let (_, stores) = pool_stating(vec![(256, 2), (1024, 8)], vec![metal(256), metal(1024)], false);
    // Folded through 100, 300 rows buffered since, 2000 more to write
    // folding all: uncut, 2000 own and 300 carried pass 2048.
    let full = lane(400, 2000);
    let frozen = [Some(100)];
    let whole = vec![Some((0, 2000))];
    assert!(
        replay_rung(
            &stores,
            56,
            piece_rows(&full, &whole, &frozen, false),
            piece_replayed(&full, &whole, &frozen, false),
            1,
            false
        )
        .is_ok(),
        "own rows past every rung are the cut's, not a refusal"
    );
    let cuts: Vec<Vec<Option<(u32, u32)>>> = host_cuts(&full, true, 512, 16, &[], 0)
        .into_iter()
        .map(|cut| vec![Some(cut)])
        .collect();
    assert!(cuts.len() > 1, "the pass is cut");
    assert_eq!(
        pieces_rung(&stores, 56, &full, &cuts, &frozen, true, &no_scratch()),
        Ok(512),
        "every piece fits: the widest asks its 512 rows, the 1024 rung, the first \
         carrying the 300 rows beside them"
    );
    // A piece of eleven lanes, 929 own rows, replaying 1144: refused by
    // name, whatever its lane count.
    let mut eleven = lane(1244, 84);
    for _ in 1..11 {
        let more = eleven.lanes[0].clone();
        eleven.lanes.push(more);
    }
    eleven.lanes[0].tokens = vec![0; 89];
    let frozen = [Some(1140); 11];
    let one = vec![(0..11).map(|at| Some((0, if at == 0 { 89 } else { 84 }))).collect()];
    let refused =
        pieces_rung(&stores, 56, &eleven, &one, &frozen, false, &no_scratch()).unwrap_err();
    assert!(refused.contains("of 1024 tokens"), "{refused}");
}

/// A pass of lanes alone on their sets, folding nothing, landing no
/// snapshot.
fn no_scratch() -> PassScratch<'static> {
    PassScratch {
        shared: Vec::new(),
        checkpoints: &[],
        snapshots: 0,
    }
}

/// Compactness is the fire's, by one function on both sides
/// ([`::engine::fire::scratch_compact`], each lane by
/// [`::engine::fire::lane_one_run`] from its verb's mask, folded index,
/// first row and lineage): a fire on a compact layout is compact unless
/// a lane of it lands an rs snapshot (an auto-checkpoint inside the
/// fire) or a lane's buffered rows are not one stretch (a beam's
/// ancestry among its siblings'). Not compact, its scratch counts every
/// carried row, a decode lane's included, and its rung is chosen so: a
/// 160-lane fire of 16 beam groups of 10 decode lanes, replaying 4200
/// rows, needs a rung whose total holds 4360, or is refused by name.
#[test]
fn a_fire_s_compactness_is_its_lanes_and_snapshots() {
    use ::engine::fire::{Lineage, lane_one_run, scratch_compact};
    assert!(scratch_compact(true, true, true, false));
    assert!(!scratch_compact(true, true, true, true), "a lane lands a snapshot");
    assert!(!scratch_compact(false, true, true, false), "a layout not compact");
    assert!(!scratch_compact(true, true, false, false), "a lane of more than one run");
    // A lane alone on its set replays one run ending at its first row.
    assert!(lane_one_run(None, 100, 104, Lineage::Set));
    assert!(lane_one_run(None, 104, 104, Lineage::Set), "nothing buffered");
    // A beam lane admitting slots 100..102 and 103..104 of 100..104:
    // two runs; admitting 102..104 alone: one run, ending at its row.
    let mask = |runs: Vec<u32>| ::engine::Masking::Extent(::engine::Mask::new(runs, 5));
    assert!(!lane_one_run(Some(&mask(vec![0, 2, 1, 1, 1])), 100, 104, Lineage::Admitted));
    assert!(lane_one_run(Some(&mask(vec![2, 3])), 100, 104, Lineage::Admitted));
    assert!(!lane_one_run(Some(&mask(vec![0, 2, 2, 1])), 100, 104, Lineage::Admitted), "not ending at its row");

    // The runtime's side, from what it puts in the fire: a lone lane
    // writing 102..106 from its state at 100, an auto-checkpoint at 104.
    let (_, stores) = pool_stating(vec![(64, 1)], Vec::new(), true);
    let full = lane_at(102, 4);
    let piece = vec![Some((0, 4))];
    let frozen = [Some(100)];
    assert!(no_scratch().compact(&stores, &full, &piece, &frozen, false), "compact");
    let checkpoints = [104u64];
    let landing = PassScratch {
        checkpoints: &checkpoints,
        snapshots: 1,
        ..no_scratch()
    };
    assert!(
        !landing.compact(&stores, &full, &piece, &frozen, false),
        "the snapshot it lands makes it not compact"
    );
    let mut beam = lane_at(102, 4);
    beam.lanes[0].lane.mask = Some(mask(vec![0, 1, 1, 1, 6]));
    let sharing = PassScratch {
        shared: vec![true],
        ..no_scratch()
    };
    assert!(
        !sharing.compact(&stores, &beam, &piece, &frozen, false),
        "a beam lane whose ancestry is two runs"
    );
    assert!(
        !sharing.compact(&stores, &full, &piece, &frozen, false),
        "a beam lane whose mask the device resolves"
    );
    // Mac's w4 lane: folded 34, its row at 38, sharing its set, its mask
    // admitting the one buffered slot 34 (a parent's), not ending at 38.
    let mut w4 = lane_at(38, 1);
    w4.lanes[0].lane.mask = Some(::engine::Masking::Extent(::engine::Mask::new(vec![0, 1, 3, 1], 5)));
    assert!(!sharing.compact(&stores, &w4, &[Some((0, 1))], &[Some(34)], false));

    // The 160-lane fire on a Metal-shaped load: not compact, it counts
    // 160 own and 4200 replayed rows in its scratch.
    let metal = |tokens: u32| ::engine::RungReplay {
        own_max: Some(tokens),
        carried_max: Some(tokens),
        scratch: rows_bytes(2 * tokens),
    };
    let (_, metal_stores) =
        pool_stating(vec![(1024, 4), (4096, 8)], vec![metal(1024), metal(4096)], true);
    let wide = FireRows { own: 160, carried: 0 };
    assert_eq!(replay_rung(&metal_stores, 56, wide, 4200, 160, false), Ok(4096), "a rung holding 4360");
    assert_eq!(replay_rung(&metal_stores, 56, wide, 4200, 160, true), Ok(160), "compact, it would not need it");
    let (_, narrow) = pool_stating(vec![(1024, 4)], vec![metal(1024)], true);
    let refused = replay_rung(&narrow, 56, wide, 4200, 160, false).unwrap_err();
    assert!(refused.contains("4364 bytes of rs scratch"), "{refused}");
}

/// Metal's rs scratch is bytes, not a row budget: a rung reserves each
/// scratch slot's rows at 2T and the lanes' work term at min(T, max
/// lanes); a fire of fewer lanes has the work term's slack for more
/// rows. A non-compact replay of 126 own and 1150 replayed rows fits
/// the 1024 rung at 60 and at 176 lanes, past 2T rows at 60 lanes too;
/// one whose bytes pass the reserve is refused by name.
#[test]
fn a_rung_s_scratch_is_its_bytes_over_its_lanes() {
    let pad = |bytes: u64| bytes.div_ceil(256) * 256;
    let metal = |tokens: u32| ::engine::RungReplay {
        own_max: Some(tokens),
        carried_max: Some(tokens),
        scratch: ::engine::ScratchBytes {
            slot_bytes: vec![1000],
            lane_bytes: 50_000,
            room_bytes: pad(2 * u64::from(tokens) * 1000) + pad(u64::from(tokens) * 50_000),
            pad: 256,
        },
    };
    let (_, stores) = pool_stating(vec![(1024, 8)], vec![metal(1024)], true);
    let fire = FireRows { own: 126, carried: 0 };
    assert!(replay_rung(&stores, 56, fire, 1150, 60, false).is_ok());
    assert!(replay_rung(&stores, 56, fire, 1150, 176, false).is_ok());
    assert!(
        replay_rung(&stores, 56, fire, 2300, 60, false).is_ok(),
        "past 2T rows, in the work term's slack"
    );
    let refused = replay_rung(&stores, 56, fire, 52_000, 176, false).unwrap_err();
    assert!(refused.contains("bytes of rs scratch"), "{refused}");
}

/// Measurements of Metal serving Qwen3.5-0.8B w8: its rs
/// scratch from `rs::Layout`, eight scratch slots of
/// [12288, 64, 4096, 12288, 128, 128, 8192, 8192] bytes a row, 1 MiB a
/// lane, each term rounded to 256; rung 256 reserves 291,667,968 bytes,
/// rung 512 583,335,936. Three non-compact replay fires fit the rung
/// they ran at, to the byte Metal counted; the fourth (256, 48 lanes,
/// 9040 scratch rows, 460,530,688 bytes) passes rung 256 and is placed
/// at 512, never refused.
#[test]
fn mac_s_qwen3_5_replay_fires_fit_their_rungs_by_their_bytes() {
    let scratch = |room_bytes| ::engine::ScratchBytes {
        slot_bytes: vec![12288, 64, 4096, 12288, 128, 128, 8192, 8192],
        lane_bytes: 262_144 * 4,
        room_bytes,
        pad: 256,
    };
    let rung = |tokens: u32, room| ::engine::RungReplay {
        own_max: Some(tokens),
        carried_max: Some(tokens),
        scratch: scratch(room),
    };
    let (r256, r512) = (rung(256, 291_667_968), rung(512, 583_335_936));
    let fire = |lanes: u32, rows: u32| ::engine::ReplayRows {
        own: lanes,
        carried: 0,
        scratch: rows,
        lanes,
    };
    assert_eq!(r512.scratch.bytes(1969, 121), 216_223_488);
    assert_eq!(r512.scratch.bytes(1640, 128), 208_634_368);
    assert_eq!(r256.scratch.bytes(5560, 32), 285_844_992);
    assert_eq!(r256.scratch.bytes(9040, 48), 460_530_688);
    assert!(::engine::fire::rung_holds(&r512, fire(121, 1969)).is_ok());
    assert!(::engine::fire::rung_holds(&r512, fire(128, 1640)).is_ok());
    assert!(::engine::fire::rung_holds(&r256, fire(32, 5560)).is_ok());
    let over = ::engine::fire::rung_holds(&r256, fire(48, 9040)).unwrap_err();
    assert_eq!(over.bound, ::engine::ReplayBound::Scratch);
    // The runtime places that fire at 512: 48 decode lanes replaying
    // 8992 rows, not compact.
    let (_, stores) = pool_stating(vec![(256, 4), (512, 8)], vec![r256, r512], true);
    let rows = FireRows { own: 48, carried: 0 };
    assert_eq!(replay_rung(&stores, 56, rows, 8992, 48, false), Ok(512));
}

fn lane_at(held: u32, rows: u32) -> crate::engine::FireRequest {
    lane(held, rows)
}

pub(super) fn pool_of_one_row_rs_pages() -> crate::store::registry::Stores {
    pool_with_rungs(Vec::new())
}

/// That pool with activation rungs `act_chunks` (tokens, chunks), its
/// span backed from the load: it starts at the largest.
pub(super) fn pool_with_rungs(act_chunks: Vec<(u32, u32)>) -> crate::store::registry::Stores {
    pool_with_rungs_of(act_chunks).1
}

/// [`pool_with_rungs`], with the model it is registered as.
pub(super) fn pool_with_rungs_of(
    act_chunks: Vec<(u32, u32)>,
) -> (usize, crate::store::registry::Stores) {
    pool_stating(act_chunks, Vec::new(), false)
}

/// [`pool_with_rungs_of`], each rung's replay scratch stated
/// (`act_replay`) and the load's rs scratch `rs_compact` or not.
/// A Metal-shaped test scratch: one byte a row, no lane term past its
/// 4-byte floor (`ScratchBytes::bytes`, Metal's formula), holding `rows`
/// scratch rows.
pub(super) fn rows_bytes(rows: u32) -> ::engine::ScratchBytes {
    ::engine::ScratchBytes {
        slot_bytes: vec![1],
        lane_bytes: 0,
        room_bytes: u64::from(rows) + 4,
        pad: 1,
    }
}

trait Unbound {
    fn unbound(self) -> ::engine::ScratchBytes;
}

impl Unbound for ::engine::ScratchBytes {
    /// No bytes binding.
    fn unbound(self) -> ::engine::ScratchBytes {
        ::engine::ScratchBytes {
            slot_bytes: Vec::new(),
            ..self
        }
    }
}

pub(super) fn pool_stating(
    act_chunks: Vec<(u32, u32)>,
    act_replay: Vec<::engine::RungReplay>,
    rs_compact: bool,
) -> (usize, crate::store::registry::Stores) {
    let act_replay = if act_replay.is_empty() {
        act_chunks
            .iter()
            .map(|&(tokens, _)| crate::store::shared::rows_only(tokens))
            .collect()
    } else {
        act_replay
    };
    use ::engine::chunk::{
        ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId, SlotKind, SlotRole,
        SlotShape,
    };
    let shape = |id, kind: SlotKind, role| SlotShape {
        id: ShapeId(id),
        kind,
        slot_bytes: 1 << 20,
        slots_per_chunk: 1,
        sets_per_working_set: kind.sets_per_working_set(),
        rows: Vec::new(),
        pinned: None,
        role,
    };
    let list = |tokens| SlotKind::List {
        tokens,
        rate: 1,
        window: None,
    };
    let facts = ChunkFacts {
        chunk_bytes: 1 << 20,
        total_chunks: 16,
        segment_chunks: 16,
        segments: SegmentFacts {
            committed: 1,
            max: 1,
            growth: Growth::Fixed,
        },
        reserved: vec![ReservedRun {
            what: Reserved::Null,
            first: 0,
            chunks: 1,
        }],
        shapes: vec![
            shape(0, list(16), SlotRole::State),
            // One recurrent row an rs page, as Qwen3.5's.
            shape(1, list(1), SlotRole::RsBuffer),
            shape(2, SlotKind::Fixed { pieces: 1 }, SlotRole::State),
        ],
        act_chunks,
        act_replay,
        rs_compact,
        rs_scratch_per_fire: true,
        ..ChunkFacts::default()
    };
    let pool = crate::store::shared::SharedPool::new(&facts, 1024).unwrap();
    let model = crate::store::registry::register_model_with_chunks(
        16,
        &[crate::store::registry::EngineStores {
            pools: crate::store::chunks::StorePools::of(&facts).unwrap(),
            max_context: 1024,
            shared: Some(Arc::new(Mutex::new(pool))),
            tiers: crate::store::shared::Tiers::default(),
        }],
    );
    (model, crate::store::registry::get(model, 0))
}

fn lane(held: u32, rows: u32) -> crate::engine::FireRequest {
    crate::engine::FireRequest {
        lanes: vec![
            ::engine::Lane {
                tokens: vec![0; rows as usize],
                kv: ::engine::KvDelta {
                    held,
                    ..Default::default()
                },
                ..::engine::Lane::default()
            }
            .into(),
        ],
        ..crate::engine::FireRequest::default()
    }
}

fn groups(plan: rs::RsPlan) -> Vec<rs::GroupPlan> {
    let rs::RsPlan::Groups(groups) = plan;
    groups
}

/// A lane staged to fold by the device's count asks for its report,
/// whichever pass stages it: the one whose rows the host places and the
/// one whose rows the device places. The fold's verb is the store's and
/// the tag is the fold's placing, so the verb alone carries none; a
/// fold the host counts asks for nothing.
#[test]
fn a_lane_the_device_folds_is_staged_asking_for_its_report() {
    use ::engine::fire::{FoldLen, RsVerb};
    let device = geometry::Fold::Device;
    let host = geometry::Fold::Host(vec![2]);
    for (device_places, fold) in [(false, &device), (true, &device), (true, &host)] {
        let what = format!("device_places {device_places}, {fold:?}");
        let stores = pool_of_one_row_rs_pages();
        let kv = stores.kv.lock().create_working_set();
        let ws = stores
            .rs
            .lock()
            .unwrap()
            .create_working_set(crate::store::rs::RsGeometry {
                state_size: 1 << 10,
                buffer_page_tokens: 1,
            });
        stores.rs.lock().unwrap().fold(ws, 10);
        let ids = [ws];
        let groups_of = lane_groups(&[kv]);
        let pass = PassLanes {
            places: vec![geometry::Place {
                folded: 10,
                folded_exact: true,
                tail: None,
                shared: false,
                spans: Vec::new(),
                // The device places the rows: its engine holds the lane
                // to what the set wrote.
                readable: 0,
                readable_known: false,
                set: 1,
            }],
            cursors: vec![Arc::new(Mutex::new(Cursor {
                reserved: 14,
                tail: None,
                ..Cursor::default()
            }))],
            kv: Vec::new(),
        };
        let mut req = lane(10, 4);
        let plan = if device_places {
            device_rs_plan(&stores, &ids, &groups_of, &pass, &[0, 4], fold, false)
        } else {
            host_rs_plan(&stores, &ids, &groups_of, &req, fold, &[], None, None)
        }
        .expect(&what);
        let prepared =
            rs::prepare_many(&mut stores.rs.lock().unwrap(), &ids, &plan, None).expect(&what);
        prepared.apply_to(&mut req);
        let counts = matches!(fold, geometry::Fold::Device);
        assert_eq!(
            matches!(
                req.lanes[0].rs,
                RsVerb::Buffer {
                    fold: FoldLen::Device,
                    ..
                }
            ),
            counts,
            "{what}"
        );
        assert_eq!(
            untagged_device_fold(&req),
            counts.then_some(0),
            "{what}: the verb alone asks for nothing"
        );
        let mut tags = Vec::new();
        let by = if device_places {
            let stated = device_fold(fold.of(0), 4, 4, false);
            device_fold_at(&pass, 0, stated, false, Some(ws), &mut req, &mut tags)
        } else {
            fold_at(
                &pass,
                &groups_of[0].1,
                &[geometry::Landing { start: 10, rows: 4 }],
                fold,
                u64::MAX,
                true,
                Some(ws),
                &mut req,
                &mut tags,
            )
        };
        // A count the host knows over a folded length it knows is a
        // slot; the device's count is placed when the fire settles.
        assert_eq!(matches!(by, Some(FoldBy::Settled { .. })), counts, "{what}");
        assert_eq!(req.lanes[0].reports != 0, counts, "{what}");
        assert_eq!(untagged_device_fold(&req), None, "{what}");
    }
}

/// A decode run ahead four fires on two recurrent sets, the host's read
/// of every output still to come: one pass stated its rows final (a
/// default pass: the SDK states it, CONTRACT-BC 25), the other did not
/// (a pass the program declared retractable). The final
/// lane folds its row at each fire and carries none into the next: its
/// verb is `Fold` and its step alone is classed folded. The lagged
/// lane folds nothing it has not read: it carries one more row each
/// fire and buffers. A step of both is buffered, each lane by its own
/// verb: the final lane's still `Fold`, carrying nothing.
#[test]
fn a_final_lane_folds_at_fire_and_a_lagged_one_beside_it_carries() {
    use ::engine::fire::RsVerb;
    use ::engine::shapes::{StepComposition, Treatment};
    let stores = pool_of_one_row_rs_pages();
    let ring = Arc::new(Mutex::new(crate::pipeline::channel::ChannelCell::new(
        vec![1],
        eta_ir::types::Dtype::I32,
        8,
    )));
    let lane_of = |final_rows: bool| {
        let kv = stores.kv.lock().create_working_set();
        let ws = stores
            .rs
            .lock()
            .unwrap()
            .create_working_set(crate::store::rs::RsGeometry {
                state_size: 1 << 10,
                buffer_page_tokens: 1,
            });
        // The prompt is folded through 10.
        stores.rs.lock().unwrap().fold(ws, 10);
        (final_rows, kv, ws, Arc::new(Mutex::new(Cursor::default())))
    };
    let lanes = [lane_of(true), lane_of(false)];
    for step in 0..4u32 {
        let start = 10 + step;
        let mut fired = Vec::new();
        for (final_rows, kv, ws, cursor) in &lanes {
            let ids = [*ws];
            let groups_of = lane_groups(&[*kv]);
            let frozen = stores.rs.lock().unwrap().frozen(*ws).unwrap();
            let pass = PassLanes {
                places: vec![geometry::Place {
                    folded: frozen,
                    folded_exact: true,
                    tail: None,
                    shared: false,
                    spans: Vec::new(),
                    readable: start,
                    readable_known: true,
                    set: 1,
                }],
                cursors: vec![cursor.clone()],
                kv: Vec::new(),
            };
            let landings = [geometry::Landing { start, rows: 1 }];
            // Every decode states the folding geometry: all it writes.
            let fold = pass_fold(
                geometry::Fold::Host(vec![u32::MAX]),
                true,
                true,
                *final_rows,
                &pass,
                &groups_of,
                &landings,
            );
            let mut req = lane(start, 1);
            let plan = host_rs_plan(&stores, &ids, &groups_of, &req, &fold, &[], None, None)
                .unwrap();
            let frozen_at = lanes_frozen(&stores, &ids, &groups_of, 1, false).unwrap();
            let counted = piece_rows(&req, &[Some((0, 1))], &frozen_at, false);
            let mut store = stores.rs.lock().unwrap();
            let prepared = rs::prepare_many(&mut store, &ids, &plan, None).unwrap();
            prepared.apply_to(&mut req);
            rs::settle(&mut store, prepared.txn);
            drop(store);
            // What the fire's submit records: a lagged run-ahead fire's
            // rows wait for the host to take its output.
            if !*final_rows {
                cursor.lock().unwrap().run_ahead(
                    start,
                    start + 1,
                    vec![crate::inferlet::host::working_set::HostRead {
                        cell: Arc::downgrade(&ring),
                        index: u64::from(step),
                    }],
                );
            }
            let replayed = req.lanes[0]
                .lane
                .replay_stated
                .map_or(0, |stated| stated.replayed);
            if *final_rows {
                assert_eq!(frozen_at, [Some(start)], "step {step}: folded through its first row");
                assert_eq!(counted.carried, 0, "step {step}");
                assert_eq!(req.lanes[0].lane.rs, RsVerb::Fold, "step {step}");
                assert_eq!(replayed, 0, "step {step}");
                assert_eq!(
                    stores.rs.lock().unwrap().frozen(*ws).unwrap(),
                    start + 1,
                    "step {step}: its row folded at fire"
                );
                let alone = StepComposition::of([&req.lanes[0].lane]);
                assert_eq!((alone.treatment, alone.carried), (Treatment::Folds, 0));
            } else {
                assert_eq!(frozen_at, [Some(10)], "step {step}: nothing read, nothing folded");
                assert_eq!(replayed, step, "step {step}: every row before it carried");
                assert!(matches!(req.lanes[0].lane.rs, RsVerb::Buffer { .. }), "step {step}");
                let alone = StepComposition::of([&req.lanes[0].lane]);
                assert_eq!(alone.treatment, Treatment::Buffers);
            }
            fired.push(req.lanes[0].lane.clone());
        }
        let mixed = StepComposition::of(&fired);
        assert_eq!(mixed.treatment, Treatment::Buffers, "step {step}: a mixed step buffers");
        assert_eq!(mixed.decode_lanes, 2);
    }
}

/// A prompt's prefill and its decodes, every pass stated final
/// (CONTRACT-BC 25), on a recurrent set: the prefill folds its rows at
/// fire, so the first decode, staged on the prompt's end, carries none
/// of them in. It folds its own row, its verb is `Fold` and its step is
/// the plain folded one, and so is every decode after it.
#[test]
fn a_final_prefill_leaves_its_first_decode_nothing_to_carry() {
    use ::engine::fire::RsVerb;
    use ::engine::shapes::{StepComposition, Treatment};
    const PROMPT: u32 = 10;
    let stores = pool_of_one_row_rs_pages();
    let kv = stores.kv.lock().create_working_set();
    let ws = stores
        .rs
        .lock()
        .unwrap()
        .create_working_set(crate::store::rs::RsGeometry {
            state_size: 1 << 10,
            buffer_page_tokens: 1,
        });
    let cursor = Arc::new(Mutex::new(Cursor::default()));
    let ids = [ws];
    let groups_of = lane_groups(&[kv]);
    // One fire of `rows` rows from `start`, the folding geometry, its
    // pass final; `run_ahead` where its tokens are the device's.
    let fire = |start: u32, rows: u32, run_ahead: bool| {
        let frozen = stores.rs.lock().unwrap().frozen(ws).unwrap();
        let pass = PassLanes {
            places: vec![geometry::Place {
                folded: frozen,
                folded_exact: true,
                tail: None,
                shared: false,
                spans: Vec::new(),
                readable: start,
                readable_known: true,
                set: 1,
            }],
            cursors: vec![cursor.clone()],
            kv: Vec::new(),
        };
        let landings = [geometry::Landing { start, rows }];
        let fold = pass_fold(
            geometry::Fold::Host(vec![u32::MAX]),
            true,
            run_ahead,
            true,
            &pass,
            &groups_of,
            &landings,
        );
        let mut req = lane(start, rows);
        let plan =
            host_rs_plan(&stores, &ids, &groups_of, &req, &fold, &[], None, None).unwrap();
        let frozen_at = lanes_frozen(&stores, &ids, &groups_of, 1, false).unwrap();
        let counted = piece_rows(&req, &[Some((0, rows))], &frozen_at, false);
        let mut store = stores.rs.lock().unwrap();
        let prepared = rs::prepare_many(&mut store, &ids, &plan, None).unwrap();
        prepared.apply_to(&mut req);
        rs::settle(&mut store, prepared.txn);
        (req, counted)
    };
    let (prefill, _) = fire(0, PROMPT, false);
    assert_eq!(
        stores.rs.lock().unwrap().frozen(ws).unwrap(),
        PROMPT,
        "the prefill folded every prompt row at fire"
    );
    assert_eq!(prefill.lanes[0].lane.rs, RsVerb::Fold);
    for step in 0..3u32 {
        let start = PROMPT + step;
        let (req, counted) = fire(start, 1, true);
        assert_eq!(counted.carried, 0, "decode {step} carries no row in");
        assert_eq!(req.lanes[0].lane.rs, RsVerb::Fold, "decode {step}");
        let replayed = req.lanes[0]
            .lane
            .replay_stated
            .map_or(0, |stated| stated.replayed);
        assert_eq!(replayed, 0, "decode {step}");
        let alone = StepComposition::of([&req.lanes[0].lane]);
        assert_eq!(
            (alone.treatment, alone.carried),
            (Treatment::Folds, 0),
            "decode {step}"
        );
        assert_eq!(stores.rs.lock().unwrap().frozen(ws).unwrap(), start + 1);
    }
}

/// MTP with k = 4 on a pool whose rs pages hold one row: the draft
/// writes a window of five rows unfolded, the next window carries those
/// five in and folds the accepted rows. Not one rs page: a lane's buffer
/// names several. No count of rows stops a lane that never folds: its
/// fire asks the rung for what it carries.
#[test]
fn mtp_k4_rides_one_row_rs_pages() {
    let stores = pool_of_one_row_rs_pages();
    let kv = stores.kv.lock().create_working_set();
    let ws = stores
        .rs
        .lock()
        .unwrap()
        .create_working_set(crate::store::rs::RsGeometry {
            state_size: 1 << 10,
            buffer_page_tokens: 1,
        });
    // The prompt is folded through 10.
    stores.rs.lock().unwrap().fold(ws, 10);
    let ids = [ws];
    let groups_of = lane_groups(&[kv]);
    let host = |held, rows, fold: Vec<u32>| {
        host_rs_plan(
            &stores,
            &ids,
            &groups_of,
            &lane(held, rows),
            &geometry::Fold::Host(fold),
            &[],
            None,
            None,
        )
    };

    // Draft: five rows (the truth and k = 4 drafts), nothing folded.
    let draft = groups(host(10, 5, vec![0]).expect("a draft of five rows"));
    assert_eq!(draft[0].writes, Some(10..15));
    assert_eq!(draft[0].fold, rs::GroupFold::None);

    // Verify: carries the draft's five in and folds the accepted three.
    let verify = groups(host(15, 5, vec![3]).expect("carries five in"));
    assert_eq!(verify[0].writes, Some(15..20));
    assert_eq!(verify[0].fold, rs::GroupFold::Host(3));
    stores.rs.lock().unwrap().fold(ws, 3);
    assert_eq!(stores.rs.lock().unwrap().frozen(ws).unwrap(), 13);

    // A lane that never folds piles up: past 13, carrying 100 in. Its
    // plan stands; the fire asks the rung for the hundred.
    let piled = groups(host(113, 5, vec![0]).expect("planned whatever it carries"));
    assert_eq!(piled[0].writes, Some(113..118));
    let frozen = lanes_frozen(&stores, &ids, &groups_of, 1, false).unwrap();
    assert_eq!(frozen, [Some(13)]);
    let counted = piece_rows(&lane(113, 5), &[Some((0, 5))], &frozen, false);
    assert_eq!(
        counted,
        FireRows {
            own: 5,
            carried: 100
        }
    );
    assert_eq!(counted.ask(), 100);
    // A host fold may leave any count behind it: the next fire asks.
    let left = groups(host(13, 100, vec![4]).expect("a fold of four of a hundred"));
    assert_eq!(left[0].fold, rs::GroupFold::Host(4));
    // A pass binding no state carries none.
    assert_eq!(
        lanes_frozen(&stores, &ids, &groups_of, 1, true).unwrap(),
        [None]
    );
    assert_eq!(
        lanes_frozen(&stores, &[], &groups_of, 1, false).unwrap(),
        [None]
    );

    // The device plan: the device picks where the rows start, inside
    // what the lane reserved.
    let device = |reserved: u32| {
        let cell = Arc::new(Mutex::new(Cursor {
            reserved,
            tail: None,
            ..Cursor::default()
        }));
        let lanes = PassLanes {
            places: vec![geometry::Place {
                folded: 13,
                folded_exact: true,
                tail: None,
                shared: false,
                spans: Vec::new(),
                // The device places the rows: its engine holds the lane
                // to what the set wrote.
                readable: 0,
                readable_known: false,
                set: 1,
            }],
            cursors: vec![cell],
            kv: Vec::new(),
        };
        device_rs_plan(
            &stores,
            &ids,
            &groups_of,
            &lanes,
            &[0, 5],
            &geometry::Fold::Device,
            false,
        )
    };
    let planned = groups(device(13 + 5 + 5).unwrap());
    assert_eq!(planned[0].writes, Some(13..23));
    assert_eq!(planned[0].fold, rs::GroupFold::Device);
    // The decode reserved its whole extent up front, as the MTP example:
    // what it reserved is buffered, and is no count of what it carries.
    let ahead = groups(device(13 + 64).expect("planned whatever it reserved"));
    assert_eq!(ahead[0].writes, Some(13..13 + 64));
}
