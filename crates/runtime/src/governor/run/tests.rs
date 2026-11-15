use super::*;

const MIB: u64 = 1 << 20;
const HARD: u64 = 1000 * MIB;
const FLOOR: u64 = 200 * MIB;

/// A pool that writes down what it is asked, and gives back what it is
/// told it has of each kind.
#[derive(Default)]
struct Fake {
    log: Vec<String>,
    held: u64,
    refused: bool,
    no_room: bool,
    /// What each kind has to give, in [`Give::ORDER`].
    has: [u64; 4],
}

impl Fake {
    fn new() -> Self {
        Self {
            held: HARD,
            ..Self::default()
        }
    }

    fn take(&mut self) -> Vec<String> {
        std::mem::take(&mut self.log)
    }
}

impl PoolTarget for Fake {
    fn segment_bytes(&self) -> u64 {
        MIB
    }
    fn floor(&self) -> u64 {
        FLOOR
    }
    fn hard_limit(&self) -> u64 {
        HARD
    }
    fn held(&self) -> u64 {
        self.held
    }
    fn growth(&self) -> Growth {
        Growth::Elastic
    }
    fn retarget(&mut self, segments: u64) -> Retarget {
        self.log.push(format!("retarget {segments}"));
        if self.no_room {
            Retarget::NoRoom {
                held: self.held / MIB,
            }
        } else {
            Retarget::Reached
        }
    }
    fn refused(&self) -> bool {
        self.refused
    }
    fn give(&mut self, what: Give, bytes: u64) -> u64 {
        let at = Give::ORDER.iter().position(|&w| w == what).unwrap();
        let given = self.has[at].min(bytes);
        self.has[at] -= given;
        self.log
            .push(format!("give {what:?} {} of {}", given / MIB, bytes / MIB));
        given
    }
    fn ticked(&mut self) {
        self.log.push("ticked".into());
    }
}

/// The segments a turn at `pressure` asks of a pool at `share`.
fn asked(share: f64, pressure: f64) -> Vec<String> {
    let mut pool = Fake::new();
    let mut steer = Steer::new(&pool, share);
    step(&mut steer, &mut pool, Some(pressure), false);
    pool.take()
}

#[test]
fn a_turn_moves_the_pool_to_the_floor_plus_what_yield_and_pressure_leave() {
    // The room above the floor is 800 MiB.
    for (share, pressure, segments) in [
        (0.3, 0.0, 1000),
        (0.3, 0.5, 880),
        (0.3, 1.0, 760),
        (1.0, 0.0, 1000),
        (1.0, 0.5, 600),
        (1.0, 1.0, 200),
    ] {
        assert_eq!(
            asked(share, pressure),
            [format!("retarget {segments}")],
            "yield {share}, pressure {pressure}"
        );
    }
    // At yield 0 the pool is not moved, whatever the pressure.
    for pressure in [0.0, 0.5, 1.0] {
        assert!(asked(0.0, pressure).is_empty(), "pressure {pressure}");
    }
}

#[test]
fn the_pool_fills_again_as_the_pressure_falls() {
    let mut pool = Fake::new();
    let mut steer = Steer::new(&pool, 1.0);
    let mut targets = Vec::new();
    for pressure in [1.0, 0.75, 0.25, 0.0] {
        targets.push(step(&mut steer, &mut pool, Some(pressure), false) / MIB);
    }
    assert_eq!(targets, [200, 400, 800, 1000]);
    assert_eq!(
        pool.take().last().map(String::as_str),
        Some("retarget 1000")
    );
}

#[test]
fn a_tick_turns_only_while_the_pressure_stands() {
    let mut pool = Fake::new();
    let mut steer = Steer::new(&pool, 1.0);
    step(&mut steer, &mut pool, Some(0.0), true);
    assert_eq!(pool.take(), ["ticked"], "calm: a tick only settles");
    step(&mut steer, &mut pool, Some(0.5), true);
    assert_eq!(
        pool.take(),
        ["retarget 600", "ticked"],
        "pressed: the floor is read again"
    );
}

#[test]
fn a_refused_commit_holds_the_target_at_what_the_pool_has() {
    let mut pool = Fake::new();
    pool.held = 300 * MIB;
    pool.refused = true;
    let mut steer = Steer::new(&pool, 1.0);
    assert_eq!(step(&mut steer, &mut pool, Some(0.0), true), 300 * MIB);
    assert_eq!(pool.take(), ["retarget 300", "ticked"]);
    pool.refused = false;
    assert_eq!(step(&mut steer, &mut pool, Some(0.0), false), HARD);
}

#[test]
fn a_shrink_with_no_room_gives_back_cache_then_weights_then_demotes_then_the_rung() {
    let mut pool = Fake::new();
    pool.no_room = true;
    // At yield 1 and pressure 0.75 the target is 400 MiB, 600 short of
    // what is held: 100 of cache, 50 of weights, 200 demoted, the rung the
    // rest.
    pool.has = [100 * MIB, 50 * MIB, 200 * MIB, 500 * MIB];
    let mut steer = Steer::new(&pool, 1.0);
    step(&mut steer, &mut pool, Some(0.75), false);
    assert_eq!(
        pool.take(),
        [
            "retarget 400",
            "give Cache 100 of 600",
            "give Weights 50 of 500",
            "give Demote 200 of 450",
            "give Rung 250 of 250",
        ]
    );
    // Covered by the cache alone, nothing else goes.
    pool.has = [700 * MIB, 50 * MIB, 200 * MIB, 500 * MIB];
    step(&mut steer, &mut pool, Some(0.75), false);
    assert_eq!(pool.take(), ["retarget 400", "give Cache 600 of 600"]);
}

#[test]
fn what_nothing_can_give_stays_where_it_is_and_nothing_is_raised() {
    let mut pool = Fake::new();
    pool.no_room = true;
    let mut steer = Steer::new(&pool, 1.0);
    assert_eq!(give_back(&mut pool, 300 * MIB), 0);
    assert_eq!(
        pool.take(),
        [
            "give Cache 0 of 300",
            "give Weights 0 of 300",
            "give Demote 0 of 300",
            "give Rung 0 of 300",
        ]
    );
    step(&mut steer, &mut pool, Some(1.0), false);
    assert_eq!(pool.held, HARD, "the pool keeps what it holds");
}

/// Today's dedicated default (`governor_backoff = 1.0`,
/// `governor_release_idle = false`) on a trace of ticks under every
/// pressure, refusals, a pool with no room and waiters between ticks,
/// recorded from the loop before this change at 70c315ff0: every turn left
/// the target at the ceiling, a tick only ticked the pool, a waiter's turn
/// called nothing, and nothing was signalled. At `yield = 0` the same trace
/// makes the same calls.
#[test]
fn at_yield_zero_the_loop_decides_what_the_dedicated_default_did() {
    // (a tick, the old reading as a pressure, a refusal, no room)
    let trace = [
        (true, 0.0, false, false),
        (true, 0.5, false, false),
        (true, 0.5, false, true),
        (true, 1.0, true, true),
        (false, 0.0, false, false),
        (true, 0.0, false, false),
        (false, 0.0, true, true),
        (true, 1.0, false, false),
    ];
    let recorded: [&[&str]; 8] = [
        &["ticked"],
        &["ticked"],
        &["ticked"],
        &["ticked"],
        &[],
        &["ticked"],
        &[],
        &["ticked"],
    ];
    let mut pool = Fake::new();
    let mut steer = Steer::new(&pool, 0.0);
    for ((ticked, pressure, refused, no_room), calls) in trace.into_iter().zip(recorded) {
        pool.refused = refused;
        pool.no_room = no_room;
        // The loop at yield 0 opens no reading; were one there, it would
        // change nothing.
        for reading in [None, Some(pressure)] {
            assert_eq!(step(&mut steer, &mut pool, reading, ticked), 1_048_576_000);
            assert_eq!(pool.take(), calls);
        }
    }
}
