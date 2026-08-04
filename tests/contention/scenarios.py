"""Contention scenario table for pie's residency planner (Project Rainer).

Every scenario drives a REAL driver (no mock device) through
`benches/pie_bench.py` and aims one specific planner endgame path. The point
of the suite is LIVENESS, not throughput: a scenario that fails requests
loudly is acceptable, a scenario that hangs, livelocks, or silently stops
exercising its target path is a defect.

Each scenario carries a `Contract` — the properties that must hold for the
run to count as passing. Contracts are deliberately loose on numbers
(throughput moves with hardware) and strict on liveness:

* ``max_wall_s``       the run must finish; exceeding it is a hang.
* ``min_completed``    the fleet must make progress.
* ``max_failed``       bounded starvation kills.
* ``accounted``        completed + failed == requests (nothing vanishes).
* ``require_counters`` planner counters that MUST be non-zero, i.e. proof the
  scenario still reaches the path it claims to test. This is what stops a
  scenario from rotting into a no-op when policy changes underneath it. It is
  evaluated across *all* attempts of a scenario, not per attempt: several
  targets are reached probabilistically, and a per-attempt check would turn a
  repeated scenario into a coin flip.
* ``max_counters``     counters that must stay bounded. This is how the
  host-swap eviction livelock is regression-tested: its rollback counter ran
  to 1.2M when the victim scan re-picked the same blocked process forever.
* ``forbid_log``       substrings that must not appear in the engine log
  (``[frame-stall]`` is the frame-seal deadlock's tell).

Counter names are keys of the ``[planner-trace]`` line that ``bootstrap.rs``
emits when ``PIE_CONTENTION_TRACE_MS`` is set.

``--swap-pool-size`` is a HOST-PINNED allocation (``cudaMallocHost`` in
``driver/cuda/src/store/swap_pool.cpp``, taken up front for every layer), so
it is not free headroom: pinned pages can be neither swapped nor reclaimed.
Size it from measured demand, not "generously". The scenarios here used to ask
for 8192-16384 pages against device pools of 12-256, which took the server to
32.3 GB RSS and pushed the host into reclaim -- ``/proc/pressure/memory``
``full avg10`` hit 66% during a run against 0.11% idle. The engine convoys
badly on that: a thread that stalls in the allocator while holding the global
KV store mutex freezes every lane behind it, and the KV lock trace
(``PIE_KV_LOCK_TRACE=1``) caught single ``create_working_set`` calls holding it
for 1.07s, 1.55s and 5.74s. Downstream that reads as 9-18 lanes falling silent
at once, ``[frame-stall]``, and ``submit deadline exceeded`` -- i.e. it looks
exactly like an engine scheduling bug. Every scenario that was flaky used
16384; none that used <= 512 ever was.

Measured peak use at 1024 is 12 pages (churn), 4 (churn_extreme), 108 (soak),
215 (mixed_head), so 1024 is still 4.7x the worst case. Right-sizing took the
suite from five flaky scenarios to 17/17, churn from 15-44s to 8.1s, and
churn_extreme from 10.8-33.6s to 11.1s on all four repeats.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Contract:
    max_wall_s: float
    min_completed: int = 1
    max_failed: int | None = None
    accounted: bool = True
    require_counters: tuple[str, ...] = ()
    max_counters: dict[str, int] = field(default_factory=dict)
    forbid_log: tuple[str, ...] = ("[frame-stall]",)


@dataclass(frozen=True)
class Scenario:
    name: str
    target: str
    args: list[str]
    contract: Contract
    env: dict[str, str] = field(default_factory=dict)
    # Prefix-sharing scenarios need one warmup request to publish the shared
    # prefix; without it every process arrives with a private copy of it and
    # the fleet starves on arrival instead of exercising the shared-page path.
    warmup: int = 0
    # A scenario whose point is a rare race needs repetition to mean
    # anything; one green run of a 10%-probability wedge proves little.
    repeat: int = 1


# Rollback ceiling is generous on purpose: a healthy engine rolls back a
# handful of evictions under thrash (13-28 observed on the tightest pool).
# The point is to catch a runaway, not to pin a number.
_ROLLBACK_CAP = 5_000

SCENARIOS: list[Scenario] = [
    # ---- reclaim rung disarmed: no host swap room at all -------------------
    Scenario(
        name="noswap_4x",
        target="Starved{NoSwapRoom}",
        args=["--total-pages", "128", "--swap-pool-size", "0",
              "--num-requests", "128", "--concurrency", "64",
              "--max-tokens", "128"],
        contract=Contract(max_wall_s=120,
                          require_counters=("starved",),
                          max_counters={"evict_rollbacks": _ROLLBACK_CAP}),
    ),
    Scenario(
        name="noswap_16x",
        target="Starved{NoSwapRoom}, extreme oversubscription",
        args=["--total-pages", "32", "--swap-pool-size", "0",
              "--num-requests", "128", "--concurrency", "64",
              "--max-tokens", "128"],
        contract=Contract(max_wall_s=120,
                          require_counters=("starved",),
                          max_counters={"evict_rollbacks": _ROLLBACK_CAP}),
    ),
    # ---- host swap present but tiny: the eviction-livelock repro -----------
    # Pre-fix this ran 152.7 s with evict_rollbacks climbing past 1.2M while
    # `serves` and `starved` stayed frozen: the victim scan re-picked the
    # same HostSwapFull process forever, and because `last_resort_evict` kept
    # reporting a dispatched eviction the starvation rung stayed disarmed.
    # The rollback cap IS the regression test.
    #
    # Sizing matters. The pool has to be tight enough that eviction is the
    # only way forward and the host pool small enough to fill, yet loose
    # enough that the fleet survives long enough to hit HostSwapFull many
    # times over — at 64/8/64 the fleet is killed by the starvation rung
    # before a single swap-out is refused and `swapfull` stays at 0.
    Scenario(
        name="tinyswap",
        target="host swap exhaustion / eviction livelock",
        args=["--total-pages", "128", "--swap-pool-size", "16",
              "--num-requests", "64", "--concurrency", "16",
              "--max-tokens", "128"],
        contract=Contract(max_wall_s=120, min_completed=48, max_failed=8,
                          require_counters=("swapfull", "evictions"),
                          max_counters={"evict_rollbacks": _ROLLBACK_CAP}),
        # Whether the host pool is actually refused a swap-out is a race:
        # measured 5 runs in 6. `require_counters` is folded across attempts,
        # so three tries make a false "never fired" ~0.5% likely.
        repeat=3,
    ),
    # Same path with the fleet four times wider than the pool can fund. This
    # used to require `starved`: most of the fleet was killed by the
    # starvation rung and the point was that the survivors still finished.
    # Making eviction a liveness rung (cebea17e4) retired that outcome on
    # purpose — an unfundable fleet now PARKS to host DRAM instead of having
    # requests destroyed — so the whole 256 completes with `starved` at 0
    # (measured: parks 1837, serves 1808, evictions 24, failed 0). Requiring
    # a kill here would pin a defect the planner deliberately removed, so the
    # contract asserts the good outcome instead: everyone finishes, and
    # eviction still runs against the rung with `evict_rollbacks` bounded.
    # `swapfull` is NOT required here: whether the host pool is actually
    # refused a swap-out is a race (observed 202 hits on one run, 0 on the
    # next), and a flaky assertion is worse than no assertion. `tinyswap`
    # above pins that path instead.
    Scenario(
        name="tinyswap_thrash",
        target="eviction parking an unfundable fleet instead of killing it",
        args=["--total-pages", "256", "--swap-pool-size", "32",
              "--num-requests", "256", "--concurrency", "64",
              "--max-tokens", "128"],
        contract=Contract(max_wall_s=120, min_completed=256, max_failed=0,
                          require_counters=("parks", "evictions"),
                          max_counters={"evict_rollbacks": _ROLLBACK_CAP}),
    ),
    # ---- every page shared: ReclaimQuote -> AllShared -> NoEligibleVictim --
    Scenario(
        name="allshared",
        target="Starved{NoEligibleVictim}",
        args=["--total-pages", "256", "--swap-pool-size", "1024",
              "--num-requests", "128", "--concurrency", "64",
              "--max-tokens", "64", "--shared-prefix-words", "900",
              "--no-unique-prompts"],
        contract=Contract(max_wall_s=180, min_completed=128, max_failed=0,
                          max_counters={"evict_rollbacks": _ROLLBACK_CAP}),
        warmup=1,
    ),
    # Guards the narrow-epoch side effect of a rejected frame-seal fix: dropping
    # idle members from the wait-set unconditionally (the old
    # `PIE_FRAME_REBIND_ESCAPE=1` mode, since deleted along with the escape
    # itself) served 12/128 here with 116 starvation kills. The seal now waits
    # for every member, which is what keeps this scenario dense.
    #
    # The fleet does NOT fit: the shared prefix is 61 pages, so a 256-page
    # pool funds FOUR processes at a time out of 64. The correct outcome is
    # that the other 60 PARK and are served in waves as holders retire — the
    # whole 128 completes in ~8.5 s. It is not "kill most of the fleet":
    # §20.17 traced the runs that killed 115 of 128 to a head-of-line wedge
    # in the planner (an uncoverable 61-page head hoarding the 12 free pages
    # that four 1-page asks needed), which is a defect, not the design. The
    # contract is therefore the full fleet, and it is a real regression test
    # for that rung — the bug reproduced roughly one run in three.
    Scenario(
        name="allshared_noswap",
        target="AllShared + no swap; head-of-line wedge under an unfundable fleet",
        args=["--total-pages", "256", "--swap-pool-size", "0",
              "--num-requests", "128", "--concurrency", "64",
              "--max-tokens", "64", "--shared-prefix-words", "900",
              "--no-unique-prompts"],
        contract=Contract(max_wall_s=120, min_completed=128, max_failed=0,
                          require_counters=("parks",),
                          max_counters={"evict_rollbacks": _ROLLBACK_CAP}),
        warmup=1,
        repeat=3,
    ),
    # ---- the destruction cascade: victims that free nothing ----------------
    #
    # PRIVATE (unshared) long prompts, no swap, and a fleet several times
    # larger than the pool can hold. Every process that gets a prefill in
    # holds its whole prompt privately; the rest park on their FIRST ask and
    # therefore hold NOTHING.
    #
    # That is the shape that exposed the cascade (§20.40). The starvation
    # rung picked "the youngest parked allocation" without asking whether
    # destroying it returns any pages, and the youngest are systematically
    # the ones parked on their first ask. Measured on a 3x-oversubscribed
    # pool: 752 wedge kills, 737 of them (98%) on `NoReclaim::HoldsNothing`,
    # freeing zero pages each, so the wedge survived every one of them and
    # the rung fired again on the next-youngest. 752 of 1024 requests were
    # destroyed to reclaim nothing, while the ~190 processes that actually
    # held the entire pool were never touched.
    #
    # The contract is therefore a BOUND ON KILLS, not zero: with no swap the
    # pool genuinely wedges and a holder must be destroyed to break it. What
    # must not come back is destroying the waiting queue. `starved` must stay
    # non-zero so the scenario cannot rot into a run that never reaches the
    # rung at all.
    Scenario(
        name="noswap_cascade",
        target="starvation victim selection: kills must free pages",
        args=["--total-pages", "512", "--swap-pool-size", "0",
              "--num-requests", "192", "--concurrency", "96",
              "--max-tokens", "64", "--shared-prefix-words", "600"],
        contract=Contract(max_wall_s=180, min_completed=160, max_failed=32,
                          require_counters=("parks", "starved")),
        repeat=2,
    ),
    # ---- head's own held pages plus its ask exceed the pool ----------------
    Scenario(
        name="hog",
        target="Hog",
        args=["--total-pages", "40", "--swap-pool-size", "1024",
              "--num-requests", "32", "--concurrency", "16",
              "--max-tokens", "512"],
        contract=Contract(max_wall_s=180, min_completed=32, max_failed=0),
    ),
    # ---- a single ask larger than the whole pool: fail loud, never hang ----
    Scenario(
        name="impossible",
        target="Impossible",
        args=["--total-pages", "16", "--swap-pool-size", "1024",
              "--num-requests", "16", "--concurrency", "8",
              "--max-tokens", "1024"],
        contract=Contract(max_wall_s=120, min_completed=0),
    ),
    # ---- arrival storms: churn, not pressure, is what hurts ----------------
    Scenario(
        name="churn",
        target="arrival storm / fleet turnover",
        args=["--total-pages", "192", "--swap-pool-size", "1024",
              "--num-requests", "2048", "--concurrency", "512",
              "--max-tokens", "32"],
        contract=Contract(max_wall_s=240, min_completed=2048, max_failed=0),
    ),
    # The rebind/frame-seal deadlock repro (`bind -> dispatch -> seal ->
    # bind`). Pre-fix this wedged permanently in 10-15% of runs, so a single
    # green run means nothing — repeat it. `[frame-stall]` in the log is the
    # direct tell, and the contract forbids it.
    Scenario(
        name="churn_extreme",
        target="extreme turnover / rebind-seal deadlock",
        args=["--total-pages", "96", "--swap-pool-size", "1024",
              "--num-requests", "4096", "--concurrency", "1024",
              "--max-tokens", "16", "--max-model-len", "1536"],
        contract=Contract(max_wall_s=300, min_completed=4096, max_failed=0),
        repeat=4,
    ),
    # ---- restore path: retries exhausted -----------------------------------
    Scenario(
        name="restore1",
        target="restore retry exhaustion",
        args=["--total-pages", "96", "--swap-pool-size", "512",
              "--num-requests", "256", "--concurrency", "128",
              "--max-tokens", "128"],
        env={"PIE_KV_RESTORE_RETRIES": "1"},
        contract=Contract(max_wall_s=180, min_completed=256, max_failed=0),
    ),
    # ---- pool holds ~one resident: every admission needs a full eviction ---
    Scenario(
        name="onefits",
        target="1-resident pool, maximum thrash",
        args=["--total-pages", "12", "--swap-pool-size", "1024",
              "--num-requests", "64", "--concurrency", "32",
              "--max-tokens", "128"],
        contract=Contract(max_wall_s=180, min_completed=64, max_failed=0,
                          require_counters=("evictions",),
                          max_counters={"evict_rollbacks": _ROLLBACK_CAP}),
    ),
    # ---- big FCFS head stuck behind small asks -----------------------------
    Scenario(
        name="mixed_head",
        target="head-of-line blocking + heterogeneity",
        args=["--total-pages", "128", "--swap-pool-size", "1024",
              "--num-requests", "256", "--concurrency", "128",
              "--max-tokens", "512", "--mixed-phase",
              "--mixed-long-prompt-words", "800",
              "--mixed-short-output", "8"],
        contract=Contract(max_wall_s=300, min_completed=256, max_failed=0),
    ),
    # ---- one request per forward: planner vs a serialized executor ---------
    Scenario(
        name="fwd1",
        target="forward serialization vs planner",
        args=["--total-pages", "128", "--swap-pool-size", "1024",
              "--num-requests", "128", "--concurrency", "64",
              "--max-tokens", "64", "--max-forward-requests", "1"],
        contract=Contract(max_wall_s=180, min_completed=128, max_failed=0),
    ),
    # ---- fleet far larger than the admission cap ---------------------------
    # A process registers with the planner at spawn but only reaches pooled
    # KV at execution admission, so the unadmitted remainder sits resident
    # holding zero pages. When the wedge predicate demanded that EVERY
    # resident process be parked, that remainder disarmed the starvation rung
    # forever. `--concurrency` sets the admission cap, so a request count far
    # above it keeps a permanent unadmitted tail.
    Scenario(
        name="admission_tail",
        target="unadmitted tail vs the starvation rung",
        args=["--total-pages", "64", "--swap-pool-size", "0",
              "--num-requests", "512", "--concurrency", "32",
              "--max-tokens", "128"],
        contract=Contract(max_wall_s=180,
                          max_counters={"evict_rollbacks": _ROLLBACK_CAP}),
    ),
    # ---- sustained pressure ------------------------------------------------
    Scenario(
        name="soak",
        target="sustained pressure soak",
        args=["--total-pages", "160", "--swap-pool-size", "1024",
              "--num-requests", "4096", "--concurrency", "256",
              "--max-tokens", "96"],
        contract=Contract(max_wall_s=420, min_completed=4096, max_failed=0,
                          max_counters={"evict_rollbacks": _ROLLBACK_CAP}),
    ),
]

BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}
