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
    # Same path with the fleet four times wider than the pool can fund. Most
    # of it is killed by the starvation rung, which is fine — the point is
    # that the survivors still finish and `evict_rollbacks` stays bounded
    # while eviction and the starvation rung run against each other.
    # `swapfull` is NOT required here: whether the host pool is actually
    # refused a swap-out before the rung kills the victim is a race
    # (observed 202 hits on one run, 0 on the next), and a flaky assertion
    # is worse than no assertion. `tinyswap` above pins that path instead.
    Scenario(
        name="tinyswap_thrash",
        target="eviction racing the starvation rung on an unfundable fleet",
        args=["--total-pages", "256", "--swap-pool-size", "32",
              "--num-requests", "256", "--concurrency", "64",
              "--max-tokens", "128"],
        contract=Contract(max_wall_s=120, min_completed=16,
                          require_counters=("starved", "evictions"),
                          max_counters={"evict_rollbacks": _ROLLBACK_CAP}),
    ),
    # ---- every page shared: ReclaimQuote -> AllShared -> NoEligibleVictim --
    Scenario(
        name="allshared",
        target="Starved{NoEligibleVictim}",
        args=["--total-pages", "256", "--swap-pool-size", "8192",
              "--num-requests", "128", "--concurrency", "64",
              "--max-tokens", "64", "--shared-prefix-words", "900",
              "--no-unique-prompts"],
        contract=Contract(max_wall_s=180, min_completed=128, max_failed=0,
                          max_counters={"evict_rollbacks": _ROLLBACK_CAP}),
        warmup=1,
    ),
    # Guards the narrow-epoch side effect of a rejected frame-seal fix: with
    # the rebind escape applied unconditionally (`PIE_FRAME_REBIND_ESCAPE=1`)
    # this served 12/128 with 116 starvation kills.
    #
    # The contract is deliberately liveness-only, not "serve the fleet
    # whole". The shared prefix is ~60 pages and each process adds ~5 more,
    # so a 64-wide fleet needs ~380 pages even when sharing perfectly — it
    # cannot fit in 256, and killing most of it is the CORRECT outcome (see
    # §20.5 on pie having no KV-aware admission control). Measured over four
    # runs per mode at this size: mode 1 completed 16-24, mode 2 completed
    # 31-59. The distributions touch, so a completion threshold here would
    # be a flaky gate rather than a regression test; §20.3's A/B harness is
    # what actually pins that difference. What must hold unconditionally is
    # that the run terminates, every request is accounted for, eviction does
    # not run away, and the pool is genuinely contended (`parks` > 0 — a good
    # run serves the fleet whole, so `starved` cannot be required here).
    Scenario(
        name="allshared_noswap",
        target="AllShared + no swap; starvation rung under an unfundable fleet",
        args=["--total-pages", "256", "--swap-pool-size", "0",
              "--num-requests", "128", "--concurrency", "64",
              "--max-tokens", "64", "--shared-prefix-words", "900",
              "--no-unique-prompts"],
        contract=Contract(max_wall_s=120, min_completed=8,
                          require_counters=("parks",),
                          max_counters={"evict_rollbacks": _ROLLBACK_CAP}),
        warmup=1,
    ),
    # ---- head's own held pages plus its ask exceed the pool ----------------
    Scenario(
        name="hog",
        target="Hog",
        args=["--total-pages", "40", "--swap-pool-size", "8192",
              "--num-requests", "32", "--concurrency", "16",
              "--max-tokens", "512"],
        contract=Contract(max_wall_s=180, min_completed=32, max_failed=0),
    ),
    # ---- a single ask larger than the whole pool: fail loud, never hang ----
    Scenario(
        name="impossible",
        target="Impossible",
        args=["--total-pages", "16", "--swap-pool-size", "8192",
              "--num-requests", "16", "--concurrency", "8",
              "--max-tokens", "1024"],
        contract=Contract(max_wall_s=120, min_completed=0),
    ),
    # ---- arrival storms: churn, not pressure, is what hurts ----------------
    Scenario(
        name="churn",
        target="arrival storm / fleet turnover",
        args=["--total-pages", "192", "--swap-pool-size", "16384",
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
        args=["--total-pages", "96", "--swap-pool-size", "16384",
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
        args=["--total-pages", "12", "--swap-pool-size", "8192",
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
        args=["--total-pages", "128", "--swap-pool-size", "16384",
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
        args=["--total-pages", "128", "--swap-pool-size", "16384",
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
        args=["--total-pages", "160", "--swap-pool-size", "16384",
              "--num-requests", "4096", "--concurrency", "256",
              "--max-tokens", "96"],
        contract=Contract(max_wall_s=420, min_completed=4096, max_failed=0,
                          max_counters={"evict_rollbacks": _ROLLBACK_CAP}),
    ),
]

BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}
