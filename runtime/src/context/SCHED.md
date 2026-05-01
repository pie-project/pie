# Credit-Market Scheduling via Knapsack Auction

A mechanism-design approach to GPU memory scheduling for inference serving.

---

## 1. Problem Statement

We allocate a perishable good — GPU KV-cache pages, renewed each batch step — among heterogeneous buyers (inference contexts) with private, time-varying valuations and a hard capacity constraint.

**Desiderata:**

- **Allocative efficiency.** Highest-value contexts are served each step.
- **Truthfulness.** No context gains by misreporting its value.
- **Conservation.** Credits are exactly preserved in the market wallet; no bankruptcy spiral. Compute billing is accounted separately in the token wallet.
- **Stability.** No evict→restore→re-evict thrashing.
- **Simplicity.** Minimal moving parts. Analytically tractable for formal results.

**Core claim.** A greedy knapsack auction with critical-value payments and endowment-proportional dividends, composed with Shapley cost-sharing for prefix colocation and gated on actual contention, achieves all five properties. Compute metering is a separate, disjoint concern (the token wallet).

---

## 2. Model

### 2.1 Entities

**Process.** Two wallets plus ownership: a credit balance (market) and a token budget (compute). Owns one or more contexts. Alive while it has at least one context.

**Context.** The schedulable unit. Has a page count `n_i`, a private value `v_i` (per page, per step), and a state: ADMITTED (pages on GPU, running) or EXCLUDED (pages off GPU, idle). Each context belongs to exactly one process and draws rent from its process's credit balance.

**Device.** A GPU with `C_d` page slots. Each device runs an independent auction. Multiple tiers (GPU, CPU cache) each run their own auction.

### 2.2 Two-Wallet Model

Each process holds **two independent wallets** that never exchange:

1. **Token wallet** (`tokens_remaining`, unit: tokens). Compute/billing
   currency. Monotonically destructive — debited by the number of tokens
   processed on every successful forward pass. Reaching zero blocks further
   passes. No market semantics.

2. **Credit wallet** (`balance`, unit: pages). Market/scheduling currency.
   Initialized to `endowment` at admission, then drifts via rent payments
   and dividends. Perfectly conserved: no source or sink other than the
   initial endowment.

Given a process admitted with token budget `T`:

```
tokens_remaining  =  T                              (compute wallet)
endowment         =  ⌈T / page_size⌉   pages        (credit wallet)
balance           =  endowment                      (credit wallet, initial)
```

Exhausting credits evicts the process from GPU but does not terminate it.
Exhausting tokens terminates forward progress but does not affect market
standing. The separation is deliberate: billing decisions (what to charge
per token) evolve independently of scheduling decisions (who runs when).

### 2.3 Invariant

At all times, within the population of admitted processes:

```
Σ_i balance_i  =  Σ_i endowment_i          (credit wallet, exactly)
tokens_remaining_i  monotone non-increasing (token wallet)
```

The credit-wallet identity is exact: rent payments and dividends are a
partition of the same revenue, so they sum to zero globally. No cumulative
make-cost term; no modulo caveats.

### 2.4 Unit of Endowment

Endowment is denominated in **KV pages** of long-run GPU residency.
Under contention, a process's steady-state held pages equals its
endowment share of capacity:

```
held_pages_i  =  (E_i / Σ_j E_j) × capacity     (at equilibrium)
```

So one endowment unit = one page of entitled long-run residency. This
anchors endowment to a physical resource and makes admission decidable
by a single integer compare (see §11).

---

## 3. Mechanism: Per-Step Knapsack Auction

Each batch step, each device runs the following procedure independently.

### 3.1 Bid Collection

Each context `i` on device `d` submits a bid `b_i` representing its value per page per step. The SDK provides a default (Section 5) but programs may override it.

### 3.2 Greedy Allocation

Sort contexts by `b_i` descending. Pack greedily until capacity is full:

```
ADMITTED ← ∅
used ← 0

for ctx in sorted(contexts, key=bid, descending):
    if used + ctx.pages ≤ C_d:
        ADMITTED ← ADMITTED ∪ {ctx}
        used ← used + ctx.pages
    else:
        marginal ← ctx
        break

EXCLUDED ← all contexts ∉ ADMITTED
```

At most one context is fractionally admissible (the marginal context). We round down: the marginal context is excluded. This wastes at most `max_i(n_i)` pages of capacity.

### 3.3 Critical-Value Payments

Each admitted context pays its **critical value** — the minimum bid at which it would still be admitted, given all other bids. This is the standard truthful payment rule for single-parameter domains (Lehmann, O'Callaghan, Shoham 2002).

For the greedy-by-density knapsack:

```
payment_i = critical_bid_i × n_i
```

where `critical_bid_i` is the highest bid at which context `i` would be displaced from the packing. For contexts well above the margin, this equals the marginal context's bid. For contexts near the margin, it may differ.

**Property:** Truthful bidding is a dominant strategy. Each context's payment depends only on others' bids, so no context can lower its payment by misreporting.

### 3.4 Endowment-Proportional Dividend

Total revenue `R = Σ_{i ∈ ADMITTED} payment_i` is redistributed to **all** contexts (admitted and excluded) proportional to their initial endowment:

```
dividend_i = (E_i / Σ_j E_j) × R
```

**Why endowment, not balance or per-context uniform?**

- **Per-context uniform** (`R/N`) is vulnerable to Sybil attack: a process splits into many tiny contexts to harvest more dividends.
- **Balance-proportional** creates intertemporal coupling: your bid affects the clearing price, which affects revenue, which affects dividends proportional to your balance. This breaks DSIC in the repeated game.
- **Endowment-proportional** is fixed at creation, independent of current behavior. Cannot be gamed by splitting (total endowment is conserved). Your dividend is independent of your bid. DSIC is preserved.

### 3.5 Net Balance Update

Each step, for each context owned by process `p`:

```
balance_p  ←  balance_p − payment_i + dividend_p

where:
    payment_i  = min(balance_p, critical_bid_i × n_i)   (clamped at balance)
    dividend_p = (E_p / Σ_q E_q) × R                    (endowment share)
    R          = Σ_i payment_i  (actually collected, not nominal)
```

The clamp prevents negative balances. Because `R` is summed from the
*actual* debits (not the uncapped nominal payments), dividends distribute
exactly what was collected — conservation holds even when a process defaults.

Token-wallet debits happen separately on successful forward pass completion:

```
tokens_remaining_p  ←  tokens_remaining_p − num_tokens_in_pass
```

### 3.6 Contention Gate

The auction charges rent **only when the device is contended**:

```
contended  ≡  device_full  ∨  ¬restore_queue.is_empty()  ∨  ¬alloc_queue.is_empty()

clearing_price_d  =  min(b_i : i ∈ GPU-resident on d)   if contended
                  =  0                                    otherwise
```

Without displacement pressure, the critical value is zero — no process
faces eviction, so no one owes rent. This also zeros the dividend pool,
so quiet devices do nothing. The market is dormant until demand exceeds
supply.

### 3.7 Anti-Thrashing

A context near the clearing price can oscillate between ADMITTED and
EXCLUDED across consecutive steps. Each transition has a physical cost
(GPU↔CPU page transfer). Three mechanisms prevent thrashing without
distorting bids:

1. **FCFS tiebreaker.** At equal bids, the eviction victim is the context
   whose owning process was spawned most recently. Incumbents have
   deterministic priority — no bid inflation needed.

2. **Admission check.** A suspended context is only restored when the
   device has enough free pages for working + replay + deferred ops.

3. **Default eviction.** Each tick, contexts whose process cannot afford
   full rent are flagged `defaulted`. Defaulted contexts are evicted first,
   regardless of bid. While suspended, the process accumulates dividends
   (while the device is contended); the context is restored only when the
   balance recovers enough to pay rent.

**Property:** Anti-thrashing is achieved without modifying the bid signal.
The auction uses raw bids; stability comes from deterministic tiebreaking
(FCFS) and automatic default enforcement.

---

## 4. Multi-Device Placement with Shapley Cost-Sharing

### 4.1 The Separation Principle

Two concerns, two mechanisms:

| Concern | Mechanism | Timescale |
|---------|-----------|-----------|
| **Placement**: which device holds a context? | Shapley cost-sharing (planner) | Periodic (every K steps) |
| **Admission**: does a context run this step? | Knapsack auction (market) | Every step |

Placement is a public-goods problem (shared prefixes are non-rival within a device). Markets fail on public goods. Admission is a private-goods problem (GPU pages are rival). Markets succeed here.

### 4.2 Effective Pages via Shapley Values

For tree-structured KV caches, each shared prefix segment `s` has a set of contexts sharing it. The Shapley value — the unique cost-sharing rule satisfying efficiency, symmetry, and additivity (Shapley 1953) — simplifies to equal-split per segment:

```
effective_pages(i, d) = unique_pages_i + Σ_s [pages_s / refcount_s(d)]
```

where `s` ranges over shared prefix segments on device `d`, and `refcount_s(d)` is the number of contexts sharing segment `s` on device `d`.

### 4.3 Placement Algorithm

For each unplaced or migration-candidate context:

```
assign context i to:

    argmin_d [ effective_pages(i, d) × price_prev(d)  +  migration_cost(i, d) ]

where:
    migration_cost(i, d) = transfer_pages(i) × TRANSFER_PENALTY   if d ≠ current_device(i)
                         = 0                                        if d = current_device(i)
```

`price_prev(d)` is the clearing price from the most recent auction on
device `d`. `TRANSFER_PENALTY` is a tunable physical-cost proxy (no longer
the credit-burning "make cost" of the prior design — the credit wallet is
conserved now). Migration only occurs when rent savings exceed transfer
cost.

### 4.4 Prefix Colocation Emerges

No explicit colocation rule is needed. Colocation reduces `effective_pages`, which reduces rent, which improves competitive position. Example:

```
Context A: 200 unique pages, 500-page shared prefix with B
Context B: 150 unique pages, 500-page shared prefix with A

Colocated on one device:
    effective_A = 200 + 500/2 = 450
    effective_B = 150 + 500/2 = 400
    total = 850

Split across devices:
    effective_A = 200 + 500/1 = 700
    effective_B = 150 + 500/1 = 650
    total = 1350
```

The placement algorithm discovers colocation because it minimizes cost.

### 4.5 Auction Uses Effective Pages

The per-device auction (Section 3) uses `effective_pages_i` in place of raw `n_i` for sorting, payment computation, and dividend impact. This is the interface between the two levels.

### 4.6 Revenue Pooling

Revenue is pooled globally across all devices. A context's dividend is `(E_i / Σ_j E_j) × R_global`. This creates a mild cross-subsidy (busy devices fund idle-device dividends) but decouples the program's income from its placement, which the program cannot control. The alternative — per-device pools — would make the dividend placement-dependent, coupling the two levels and complicating the theoretical story.

---

## 5. Optimal Bid

### 5.1 Total Rent Exposure

The bid is for rent — per page per step. A context generating tokens accumulates pages over time. If admitted continuously for `n` remaining steps with current page count `p` and page growth rate `g = 1/page_size` (one token per step), the **total future page-steps** (cumulative rent exposure) is:

```
S = Σ_{k=0}^{n-1} (p + g·k) = p·n + n(n−1) / (2·page_size)
```

This is quadratic in `n`. The naïve formula `p·n` ignores the triangular accumulation: each new page created at step `k` incurs rent for the remaining `n−k` steps.

### 5.2 Budget-Exhausting Bid

The bid should be the **maximum sustainable rent per page per step** — the
flat rate that exactly exhausts the credit wallet at the end of the horizon.

Over `n` steps while admitted, the balance evolves as:

```
B_final = B + n·d − r·S = 0
              ↑       ↑
           dividends  rent
```

where `d` is the per-step dividend (received regardless of admission status).
Make cost no longer appears — forward-pass compute is billed against the
token wallet, not the credit wallet. Solving for `r`:

```
bid = (B + n·d) / (p·n + n(n−1) / (2·page_size))
```

Factoring out `n`:

```
bid = (B/n + d) / (p + (n−1) / (2·page_size))
```

**Why this form:**

1. **Denominator** includes `(n−1)/(2·page_size)` — the quadratic rent
   exposure. For large `n` this dominates, preventing overbidding on the
   pages that will be created over time.
2. **Dividend is added**, not subtracted. Dividends are status-independent:
   a context earns the same dividend whether admitted or excluded. They
   replenish the budget, increasing affordable rent. There is no
   opportunity cost to subtract.

### 5.3 Stochastic Horizon

The remaining step count `n` is typically unknown. If `n` is a random variable with mean `μ` and variance `σ²`, take the expectation of the budget constraint. The only nonlinear term is `S`:

```
E[S] = p·μ + E[n(n−1)] / (2·page_size)
     = p·μ + (μ² + σ² − μ) / (2·page_size)
     ≈ p·μ + μ²(1 + cv²) / (2·page_size)
```

where `cv² = σ²/μ²` is the squared coefficient of variation. The bid becomes:

```
bid = (B/μ + d) / (p + μ(1 + cv²) / (2·page_size))
```

The `(1 + cv²)` multiplier captures uncertainty. Higher variance → more weight on expensive long-tail outcomes (quadratic rent growth) → more conservative bid.

| Distribution | cv² | Effect |
|---|---|---|
| Deterministic (`n` known) | 0 | Exact budget planning |
| Geometric (memoryless) | 1 | 2× more conservative (maximum entropy) |
| Heavy-tailed | ≫ 1 | Very conservative |

The **geometric distribution** (cv² = 1) is the maximum-entropy prior for positive integers with a given mean — making the fewest assumptions about the stopping time. It is minimax optimal: it minimizes worst-case regret against any true distribution.

### 5.4 Horizon Estimation

The bid is recomputed every step with the current balance, providing natural feedback: overbidding drains the balance and reduces future bids; underbidding accumulates dividend and raises future bids.

| Information available | μ | cv² | Rationale |
|---|---|---|---|
| `with_horizon(n)` | `n − generated` | 0 | Deterministic — program knows its output length |
| `with_max_tokens(n)` | `n − generated` | 1 | Hard cap known, stopping point within it unknown |
| Neither | `max(generated, 64)` | 1 | Lindy heuristic: E[remaining \| survived t] ≈ t |

The Lindy heuristic uses only observables with no tuning parameters. The floor of 64 prevents `μ = 0` at the start; after ~64 steps the estimate is driven by elapsed time.

### 5.5 Properties

- When `dividend ≈ 0` and the device is uncontended, `clearing_price = 0`
  (§3.6) and the bid is never collected against. Low-load behavior is
  effectively bid-free.
- The bid is always non-negative: `B/μ + d ≥ 0`. Contexts never prefer
  exclusion on bid-value grounds. Termination happens through the token
  wallet (tokens_remaining = 0) or through market eviction when the
  process is outbid.
- The balance feedback loop makes the strategy self-correcting regardless
  of horizon error.
- Truthful reporting of this value is a dominant strategy under
  critical-value payments (§10, Theorem 2).

---

## 6. Tiered Storage

When a context is excluded from GPU, its pages descend through a storage hierarchy:

| Tier | Medium | Restoration Cost | Capacity |
|------|--------|-----------------|----------|
| GPU | HBM | 0 (already there) | `C_d` pages |
| CPU | Host DRAM | ~0.16ms/page (PCIe) | `C_cpu` pages |
| Evicted | None | ~10ms/page (recompute) | Unlimited |

**Each tier runs a separate auction.** A context excluded from GPU may still win CPU cache if its bid exceeds the CPU clearing price. The CPU clearing price is naturally lower because holding a CPU-cached page is less valuable (it only saves restoration cost, not compute):

```
CPU bid_i = restore_cost_saved_i / effective_pages_i
          = (recompute_cost − transfer_cost) / effective_pages_i
```

This replaces the original design's `CPU_TIER_DISCOUNT` with a market-derived price.

---

## 7. Tool-Call Yielding

When a context enters an external wait (tool call, API call), its per-step compute value drops to zero. The decision is binary under critical-value payments:

- **Hold** (keep generation bid): pays `rent × pages` per step but resumes instantly.
- **Fold** (bid zero): pays nothing but risks restore-queue delay.

Under Vickrey payments, the process pays the clearing price regardless of its bid level. Bidding above the clearing price does not increase the payment — it only reduces eviction risk. Therefore, holding and folding have the same per-step cost for any bid above the margin.

### 7.1 Affordability Threshold

The process should hold iff its competitiveness surplus covers the idle rent. From the bid formula (§5.1):

```
generation_bid = (balance / remaining − dividend) / pages
```

Requiring that the process remain competitive after paying idle rent for `W` steps:

```
balance − rent × pages × W  >  (rent × pages + dividend) × remaining
```

Substituting `balance − (rent × pages + dividend) × remaining = remaining × pages × (bid − rent)`:

```
W  <  remaining × (bid − rent) / rent
```

Or equivalently:

```
W / remaining  <  (bid − rent) / rent
```

The right-hand side is the process's **competitiveness surplus** — how far above the clearing price its bid sits, as a fraction of the clearing price.

### 7.2 Properties

- **Endogenous threshold.** No tuning parameter. The hold/fold boundary is determined entirely by the process's market position.
- **Well above margin** (`bid = 5 × rent`): can idle for 4× its remaining horizon — effectively always holds.
- **Near margin** (`bid ≈ rent`): surplus is near zero — folds immediately.
- **Below margin** (`bid < rent`): already being evicted — fold is the only option.
- **No contention** (`rent = 0`): residency is free — always hold.

### 7.3 Role of Latency

The system exposes `latency(ctx)`, the EWA-smoothed per-tick wall-clock time of the device. This converts the expected wait duration (seconds) to market steps:

```
W = expected_wait_seconds / latency
```

Without this signal, the process would need a hardcoded step-duration assumption. With it, the threshold adapts to actual device speed.

### 7.4 Information Asymmetry

The process does not observe the restore-queue depth or expected restoration delay. This asymmetry is intentional: the hold/fold decision is framed as **budget affordability** rather than restoration cost estimation. The process asks "can I afford the insurance premium?" (observable) instead of "how expensive would eviction be?" (unobservable).

The bid is restored automatically when the wait completes.

---

## 8. Lifecycle

### 8.1 Admission

A new process arrives with token budget `T`. The system creates a wallet with endowment `E = ⌈T / page_size⌉` and a first context. The context enters the auction immediately. No admission gate — if the system is overloaded, the clearing price is high and a low-value newcomer is simply not admitted. If idle, the price is near zero and the newcomer runs for free.

### 8.2 Steady State

```
┌─────────────────────────────────────────────────────┐
│  BATCH STEP                                         │
│                                                     │
│  1. Contexts submit bids                            │
│     SDK default: (B/μ + d − 1/s) / (p + μ(1+cv²)/(2s))  │
│     or: program overrides with custom bid            │
│                                                     │
│  2. Placement check (every K steps)                 │
│     Shapley cost → assign to optimal device          │
│     Migrate if rent savings > transfer cost          │
│                                                     │
│  3. Auction clears (per device, per tier)            │
│     Greedy pack by bid descending                    │
│     Critical-value payments                          │
│     FCFS tiebreaker for incumbents                   │
│                                                     │
│  4. Payments and dividends                             │
│     Admitted: balance −= payment                     │
│     Everyone: balance += endowment-proportional share│
│                                                     │
│  5. Make cost for new pages                          │
│     balance −= new pages above high-water mark       │
│                                                     │
│  6. Page movement                                    │
│     Newly excluded: GPU → CPU cache or evict         │
│     Newly admitted: restore from CPU or recompute    │
│                                                     │
│  7. Forward pass                                     │
│     Admitted contexts run one decode step             │
│     Excluded contexts idle                           │
│                                                     │
│  8. Publish signals                                  │
│     price(device, tier), dividend(), balance()         │
└─────────────────────────────────────────────────────┘
```

### 8.3 Wallet Exhaustion

Two independent exhaustion paths, matching the two wallets (§2.2):

**Token wallet → termination.** `tokens_remaining` monotonically decreases
by one per successful token processed. Reaching zero blocks any further
forward pass (the `pin` step refuses with "token budget exhausted").
This is the hard compute bound. Lifetime in tokens is exactly `T`, the
admission-time budget.

**Credit wallet → eviction, not termination.** The credit balance drifts
under rent/dividend flow. Because the flow is zero-sum (§3.5), an excluded
process earns dividend and its balance strictly increases while others
pay rent. So `balance_i` never reaches zero in finite time while excluded
— this is the **no-bankruptcy guarantee**. A process with exhausted credits
simply loses GPU access until dividends restore it, without any risk of
accounting underflow.

A process terminates when:
1. Its token wallet hits zero, or
2. Computation completes naturally (EOS, max_tokens, guest exit).

Credit balance has no termination role. The guarantee that an exhausted-
credit process can always recover enough to contend again is structural,
not heuristic.

### 8.4 Multi-Context Processes

A process with multiple contexts (multi-turn agent, parallel generations) has each context bid independently from the shared wallet. High-value contexts win pages; low-value ones yield and earn dividend. The market performs optimal internal allocation without any process-level coordination.

### 8.5 Admission Gate

Admission is the one gate where the system can refuse service. Because
endowment is physical (§2.4) — one unit = one page of long-run residency
— the admission rule is an integer compare:

```
admit iff  Σ E_live + E_new  ≤  capacity × admission_oversubscription_factor
```

- `Σ E_live`: sum of endowments of currently admitted processes.
- `E_new`: endowment requested by the new process (`⌈T / page_size⌉`).
- `capacity`: total GPU pages across all devices.
- `admission_oversubscription_factor ∈ (0, ∞)`: a provider-chosen knob.

**Regimes:**

| factor | meaning |
|---|---|
| `1.0` | strict booking: every admitted process is guaranteed its full endowment at all times. No market contention arises. |
| `> 1.0` | overbook: provider sells more entitlement than physical capacity, betting on non-peak duty cycles. Market resolves transient crossings. |

This replaces the old unbounded admission (which let anyone in and let
the market sort it out, with no structural cap). It also gives the
provider a single knob to tune overall load rather than inferring it from
queueing metrics.

---

## 9. Guest API

```wit
price:        func() -> f64              // effective price for this context
rent:         func(ctx) -> f64           // clearing price on this context's device
dividend:     func() -> f64              // this context's dividend last step
balance:      func() -> f64              // current credit balance
latency:      func(ctx) -> f64           // EWA-smoothed tick latency (seconds)

bid:          func(value: f64)           // override SDK default
suspend:      func()                     // voluntary exclusion
```

The guest reads market signals (`price`, `rent`, `dividend`, `latency`), computes or overrides a bid, and the system handles allocation, placement, migration, and prefix optimization.

---

## 10. Theoretical Properties

### Theorem 1 (Efficiency)

The per-step allocation maximizes total declared value subject to the capacity constraint, up to an integrality gap of one marginal context.

*Proof sketch:* Greedy packing by value-per-page is optimal for the LP relaxation of the 0-1 knapsack. The LP relaxation is tight: at most one fractional variable. Rounding down loses at most one context.

### Theorem 2 (Truthfulness)

Under critical-value payments, truthful bidding is a dominant strategy in the stage game.

*Proof:* Standard result for greedy mechanisms in single-parameter domains. Each context's payment is determined by others' bids. A context cannot influence its payment by changing its own bid; it can only affect whether it is admitted. Since admission at the critical value is exactly the break-even point, bidding above or below true value cannot increase utility. (Lehmann, O'Callaghan, Shoham 2002.)

### Theorem 3 (Approximate Truthfulness in the Repeated Game)

In the T-step repeated game with N contexts, the mechanism is ε-DSIC with:

```
ε ≤ max_i(payment_i) / R  =  O(n_max / Σ_j n_j)
```

where `n_max` is the largest context's page count.

*Proof sketch:* A context's bid at step t affects revenue R_t, which affects dividends. But its share of the effect is bounded by its payment share, which is O(1/N) for contexts of similar size. As N grows, the incentive to deviate vanishes.

### Theorem 4 (Conservation)

Total credits satisfy `Σ_i balance_i(t) = Σ_i E_i` at all times
(for the currently admitted population). Rent revenue is exactly
redistributed. No process reaches zero balance in finite time while
excluded.

*Proof:* Actually-collected rent payments sum to `R`. Dividends sum to
`R` (they are a partition of the revenue). Net flow from rent is zero.
There is no other source or sink in the credit wallet (the token wallet
is a separate, disjoint accounting — see §2.2). An excluded context pays
no rent and receives positive dividend whenever the device is contended,
so its balance strictly increases while excluded.

### Theorem 5 (Efficient Placement)

Under Shapley cost-sharing, the cost-minimizing placement is the one that maximizes prefix sharing. Contexts sharing a common prefix are colocated in any cost-minimizing assignment.

*Proof sketch:* Shapley costs are additive over segments. Colocation reduces the per-context cost of each shared segment by a factor of 1/refcount. The placement minimizes Σ effective_pages × price, which is minimized when shared segments have maximum refcount — i.e., colocation.

### Theorem 6 (Composition)

The two-level mechanism (Shapley placement + knapsack auction) preserves all properties of the single-device auction. Placement affects effective_pages but not the auction's structure. Migration occurs only when rent savings exceed transfer cost (individual rationality of migration).

### Theorem 7 (Stability)

The clearing price is computed exactly each step (no dynamics, no
convergence). FCFS tiebreaking provides deterministic incumbent priority
at equal bids. The admission check prevents restoring contexts that would
immediately re-suspend. The system has no oscillatory modes.

*Proof:* The auction is a static optimization, not a dynamic process.
There is no feedback loop to oscillate. Deterministic tiebreaking
eliminates bid-level ambiguity in the evict/admit decision. The admission
check breaks the eviction↔restoration ping-pong by refusing restoration
until structural capacity exists.

---

## 11. Design Evolution

| Aspect | Original (tax) | v1 (unified credit) | Current (two-wallet) |
|--------|---------------|---------------------|----------------------|
| **Currencies** | 1 (credit) | 1 (credit, with make-cost destruction) | 2 (tokens + credits) |
| **Compute accounting** | Implicit in tax | Make cost drains market balance | Separate token wallet; no market coupling |
| **Conservation** | Tax in = interest out | `Σ B = Σ E − M(t)` | `Σ B = Σ E` exactly |
| **Congestion signal** | M/M/1 tax curve | Clearing price, always charged | Clearing price, **gated on contention** |
| **Idle-device rent** | Positive (tax still applied) | Positive (min bid charged) | Zero (critical value = 0) |
| **Endowment unit** | Abstract weight | Credits (pages of compute) | Pages of long-run GPU residency |
| **Admission** | Unbounded | Unbounded | `Σ E ≤ capacity × overbook_factor` |
| **Compute-or-wait** | Explicit MCC/MBW check | Negative bid = exclude | Always non-negative; budget gate is token-side |
| **Bid formula** | Prescribed | `(B/μ + d − 1/s) / (p + μ(1+cv²)/(2s))` | `(B/μ + d) / (p + μ(1+cv²)/(2s))` |
| **Anti-thrashing** | Utilization cap + budget | Make cost + FCFS + admission | FCFS + admission (make cost gone) |
| **Prefix sharing** | Tax split implicitly | Shapley cost-sharing | Shapley cost-sharing |
| **Truthfulness** | DSIC via convex tax | DSIC via critical-value payments | Unchanged |
| **Lifetime bound** | Implicit via credit drain | Credit drain (make cost) | Explicit via `tokens_remaining` |

---

