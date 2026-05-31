//! Speculative decoding / MTP — replaces the ~264 file-local helpers in
//! `driver/cuda/src/executor/executor.cpp` (`launch_mtp_argmax`,
//! `capture_mtp_chain_graph_exec`, the draft/verify/commit machinery).
//!
//! This is the highest-value extraction: in executor.cpp the spec-decode
//! state machine is interleaved with the plain forward path. Here it is a
//! self-contained Rust driver that sequences C++ draft/verify *bodies*
//! over the same coarse ABI, keeping the plain path (executor.rs) clean.
//!
//! The target shape is the frozen-verify + batched-repair design from the
//! current Pie MTP-uncap work — modelled explicitly as states rather than
//! emergent from helper-call ordering.
//!
//! # What this models vs. executor.cpp
//!
//! The C++ path runs, per fire, for each speculating request `r`:
//!
//!   1. **Draft** — `run_step_chained_system_drafter` / `draft_next`: the
//!      drafter proposes `n_d` tokens at positions
//!      `[first_draft_position .. first_draft_position + n_d)`.
//!   2. **Verify** — `expand_spec_batch` splices the drafts into the target
//!      forward (`spec_expansion.cpp`); the single frozen target forward
//!      samples one token per draft slot **plus a bonus** at the tail —
//!      `n_d + 1` verified tokens (`block[0..=n_d]`).
//!   3. **Accept** — executor.cpp:2746 counts the longest prefix where the
//!      verified token equals the drafted token; that count is `match`
//!      (= `accepted_len`). The emitted token list is `block[0..=match]`,
//!      i.e. `match` accepted drafts **+ 1** correction/bonus token (always
//!      one fresh token, so the engine never stalls).
//!   4. **CommitAdvance** — executor.cpp:2870 / 2990: re-run the *frozen*
//!      target over `[original_input | accepted_drafts]` (length
//!      `orig_n_in + accepted`) to advance the recurrent/SSM state and KV
//!      to exactly the confirmed prefix. The rejected suffix's state is
//!      never committed → no per-request snapshot buffer → no concurrency
//!      cap (the MTP-uncap win).
//!
//! Everything here is **pure host logic** — the acceptance math, position
//! bookkeeping, and phase sequencing. All device/FFI work (the actual
//! draft/verify/commit forwards) stays abstract behind [`Drafter`]; this
//! module only *decides* what to feed it and *interprets* what it returns.
//!
//! Simplifications relative to the C++ tangle (faithful where it matters,
//! dropped where it is device/perf plumbing):
//!   * CUDA-graph capture/caching (`MtpChainGraphKey`, `try_run_*_graph`)
//!     is a perf optimisation with no bearing on the state machine — out.
//!   * Tensor-parallel broadcast (`tp_broadcast_mtp_inputs`) lives in
//!     `tp.rs` — out.
//!   * The sampling-layout expansion (`expand_spec_batch`) is an executor
//!     concern; here we model only its *contract*: a verify produces
//!     `n_drafts + 1` token ids per request.
//!   * Adaptive draft-count (`mtp_desired_drafts`, gemma4_mtp.cpp:314) is
//!     modelled faithfully as a pure function ([`SpecConfig::next_drafts`]),
//!     since it is pure acceptance arithmetic.

/// A model that can propose draft tokens — the `Arch::drafter` opt-in.
///
/// Abstract on purpose: every method here ultimately drives a
/// `pie_body`-family call with the spec layout (phase 4), but the state
/// machine in this module is written and tested without ever touching a
/// device. Tests supply a fake.
pub trait Drafter {
    /// Upper bound on draft tokens proposed per request per round
    /// (`NativeSystemDrafter::max_drafts`). The engine never proposes more.
    fn max_drafts(&self) -> usize;

    /// Position passed to the first low-level draft step is
    /// `source_position + draft_position_offset`; later steps advance by
    /// one (`NativeSystemDrafter::draft_position_offset`, default 1).
    fn draft_position_offset(&self) -> u64 {
        1
    }

    /// Propose up to `n` draft token ids for a request whose last committed
    /// token is `last_token` at `last_position`. `n` is the engine-chosen
    /// budget (`<= max_drafts`); returning fewer than `n` is allowed and
    /// degrades gracefully (the verify simply has fewer slots).
    ///
    /// In the device path this is `draft_step` chained `n` times (or one
    /// `draft_next`); here it is the single point the fake overrides.
    fn propose(&mut self, request: RequestId, last_token: TokenId, last_position: u64, n: usize)
    -> Vec<TokenId>;

    /// The frozen target forward over a request's draft chain. Returns the
    /// `drafts.len() + 1` verified token ids (`block[0..=n_d]` in
    /// executor.cpp): one per draft slot plus the tail bonus token.
    ///
    /// This is the single most load-bearing abstraction: the *target's*
    /// opinion of what each position should hold, which the acceptance math
    /// ([`verify_outcome`]) diffs against the proposal.
    fn verify(&mut self, request: RequestId, prefix_token: TokenId, drafts: &[TokenId])
    -> Vec<TokenId>;

    /// Commit-advance: advance recurrent/KV state over the confirmed prefix
    /// `[input | accepted]` for one request. `commit_len` is
    /// `orig_input_len + accepted_len` (executor.cpp:2881 / 3018). Pure
    /// state plumbing on the device side; a no-op for the fake.
    fn commit_advance(&mut self, _request: RequestId, _commit: &CommitAdvance) {}
}

/// Request handle (the `request_index` / `source_row` pair carried through
/// the C++ `SystemSpecDraftRequest`). Opaque to this module.
pub type RequestId = u32;
/// A vocabulary token id (the C++ path uses `uint32_t` token ids).
pub type TokenId = u32;

// ---------------------------------------------------------------------------
// Explicit spec-decode states (vs. executor.cpp's implicit control flow).
// ---------------------------------------------------------------------------

/// Explicit spec-decode states. In executor.cpp these are *emergent* from
/// the order of helper calls within `handle_fire_batch`; here they are a
/// first-class cycle so the transitions are testable in isolation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Propose up to `max_drafts` tokens (the drafter forward).
    Draft,
    /// Single frozen target forward over the draft chain.
    Verify,
    /// Batched correction of the rejected suffix (here: take the target's
    /// bonus/correction token — the always-present `match + 1`th token).
    Repair,
    /// Advance recurrent state for the accepted prefix (frozen-verify +
    /// activation replay over `[input | accepted]`).
    CommitAdvance,
}

// ---------------------------------------------------------------------------
// Acceptance bookkeeping — pure functions over token ids.
// ---------------------------------------------------------------------------

/// Outcome of verifying one request's draft chain against the frozen
/// target — the pure core of executor.cpp:2746-2774.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyOutcome {
    /// Longest prefix length where `verified[k] == drafted[k]`
    /// (the C++ `match`). `0 <= accepted_len <= n_drafts`.
    pub accepted_len: usize,
    /// The always-present correction/bonus token: `verified[accepted_len]`.
    /// When every draft is accepted this is the genuine next token (the
    /// "bonus"); otherwise it is the target's correction at the first
    /// mismatch. Either way the request advances by at least one token.
    pub bonus_token: TokenId,
    /// `true` when the proposal was cut short — at least one drafted token
    /// was rejected (`accepted_len < n_drafts`). The rejected suffix's
    /// speculative state must not be committed; `Repair` substitutes the
    /// `bonus_token`. `false` means a full hit (all drafts accepted).
    pub needs_repair: bool,
}

impl VerifyOutcome {
    /// Total tokens emitted for the request this round: the accepted drafts
    /// plus the one bonus/correction token (executor.cpp's
    /// `block[0..match+1]`, length `match + 1`). Always `>= 1`.
    pub fn emitted_len(&self) -> usize {
        self.accepted_len + 1
    }
}

/// The full ordered list of tokens this request emits this round:
/// `[accepted drafts.., bonus_token]`. Mirrors the C++
/// `bucket.assign(block.begin(), block.begin() + match + 1)`.
pub fn accepted_tokens(drafted: &[TokenId], outcome: &VerifyOutcome) -> Vec<TokenId> {
    let mut out = Vec::with_capacity(outcome.emitted_len());
    out.extend_from_slice(&drafted[..outcome.accepted_len]);
    out.push(outcome.bonus_token);
    out
}

/// Compute the acceptance outcome for one request — the longest-correct-
/// prefix diff at the heart of frozen-verify.
///
/// `drafted` is the proposal (`n_d` tokens). `verified` is the target's
/// `n_d + 1` sampled tokens (`block[0..=n_d]`): one per draft slot plus the
/// tail bonus. Faithful to executor.cpp:2746:
///
/// ```text
/// int match = 0;
/// for (int k = 0; k < n_d; ++k) {
///     if (block[k] == draft[k]) match++; else break;
/// }
/// bucket = block[0 .. match+1];   // match accepted + 1 bonus/correction
/// ```
///
/// Edge cases:
///   * **All accepted** (`match == n_d`): `bonus_token = verified[n_d]`
///     (the genuine next token), `needs_repair = false`. The request
///     advances by `n_d + 1`.
///   * **Mismatch at k**: `accepted_len = k`, `bonus_token = verified[k]`
///     (the target's correction), `needs_repair = true`. Advances by
///     `k + 1`.
///   * **Empty drafts** (`n_d == 0`): degenerates to a single-token decode
///     — `accepted_len = 0`, `bonus_token = verified[0]`,
///     `needs_repair = false`. (`verified` must hold the one decode token.)
///
/// # Panics
/// If `verified.len() != drafted.len() + 1` — the target *always* produces
/// one token per draft slot plus the bonus; a mismatch is a logic bug in
/// the caller's verify layout, not a runtime input condition.
pub fn verify_outcome(drafted: &[TokenId], verified: &[TokenId]) -> VerifyOutcome {
    assert_eq!(
        verified.len(),
        drafted.len() + 1,
        "verify must produce n_drafts + 1 tokens (one per draft slot + bonus)"
    );
    let n_d = drafted.len();
    let mut accepted_len = 0usize;
    while accepted_len < n_d && verified[accepted_len] == drafted[accepted_len] {
        accepted_len += 1;
    }
    VerifyOutcome {
        accepted_len,
        bonus_token: verified[accepted_len],
        needs_repair: accepted_len < n_d,
    }
}

/// Per-request position/length bookkeeping for the commit-advance forward
/// (executor.cpp:2870-2898 / 3008-3019). Pure arithmetic; the device side
/// reads these to replay the recurrence over the confirmed prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitAdvance {
    /// Number of tokens to fold into committed state this round:
    /// `orig_input_len + accepted_len` (the C++ `commit_len`).
    pub commit_len: usize,
    /// Original (pre-draft) input length for the request — `orig_n_in`.
    pub orig_input_len: usize,
    /// Accepted draft count beyond the original input (`accepted`).
    pub accepted_len: usize,
    /// Position of the next token the request will produce after this
    /// commit — the new `source_position + 1` / `first_draft_position` for
    /// the following round's drafts.
    pub next_position: u64,
}

/// What one request did in a completed Draft→…→CommitAdvance round —
/// the host-visible result the executor turns into a response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepResult {
    pub request: RequestId,
    /// `[accepted drafts.., bonus]` — the tokens to emit (length `>= 1`).
    pub emitted: Vec<TokenId>,
    pub outcome: VerifyOutcome,
    pub commit: CommitAdvance,
    /// Drafts proposed last round (for adaptive draft-count next round).
    pub proposed_len: usize,
}

impl StepResult {
    /// The genuine new tokens accepted this round (always `>= 1`).
    pub fn num_accepted(&self) -> usize {
        self.outcome.emitted_len()
    }
}

// ---------------------------------------------------------------------------
// Adaptive draft count (gemma4_mtp.cpp:314 `mtp_desired_drafts`) — pure.
// ---------------------------------------------------------------------------

/// Static spec-decode policy knobs (the host-side `Gemma4MtpRuntimeConfig`
/// surface that affects the *decision*, not the device path).
#[derive(Clone, Copy, Debug)]
pub struct SpecConfig {
    /// Hard upper bound on drafts per round (`max_drafts`).
    pub max_drafts: usize,
    /// When `true`, the next round's draft budget adapts to the last
    /// round's acceptance (gemma4_mtp.cpp:319). When `false`, always
    /// `max_drafts`.
    pub adaptive: bool,
    /// Floor for the adaptive budget (`mtp_min_adaptive_drafts`).
    pub min_drafts: usize,
    /// Budget for a request with no acceptance history yet
    /// (`mtp_initial_adaptive_drafts`).
    pub initial_drafts: usize,
}

impl SpecConfig {
    /// Fixed-budget config: always propose `max_drafts`.
    pub fn fixed(max_drafts: usize) -> Self {
        Self {
            max_drafts,
            adaptive: false,
            min_drafts: max_drafts,
            initial_drafts: max_drafts,
        }
    }

    /// Choose next round's draft budget from the previous round's outcome.
    /// Faithful to `mtp_desired_drafts` (gemma4_mtp.cpp:314-329):
    ///
    /// ```text
    /// if (!adaptive)                  return max_drafts;
    /// if (max_drafts <= 1)            return max_drafts;
    /// if (no history)                 return initial_drafts (clamped);
    /// if (last_match >= last_n)       return clamp(last_n + 1, min, max);  // full hit → grow
    /// else                            return clamp(last_match + 1, min, max);
    /// ```
    ///
    /// `prev` is `None` for the request's first speculative round.
    pub fn next_drafts(&self, prev: Option<&VerifyOutcome>, prev_n_drafts: usize) -> usize {
        if !self.adaptive || self.max_drafts <= 1 {
            return self.max_drafts;
        }
        let Some(prev) = prev else {
            return self.initial_drafts.clamp(1, self.max_drafts);
        };
        if prev_n_drafts == 0 {
            return self.initial_drafts.clamp(1, self.max_drafts);
        }
        let desired = if prev.accepted_len >= prev_n_drafts {
            // Full hit last round → reach one further.
            prev_n_drafts + 1
        } else {
            // Mismatched at `accepted_len` → aim just past the hit prefix.
            prev.accepted_len + 1
        };
        desired.clamp(self.min_drafts, self.max_drafts)
    }
}

// ---------------------------------------------------------------------------
// The driving state machine.
// ---------------------------------------------------------------------------

/// Per-request state the engine threads across rounds. In executor.cpp this
/// is scattered across `mtp_base_rows`, `mtp_draft_positions`,
/// `spec_accepted_drafts`, and `SystemSpecDraftRequest`; here it is one
/// struct per active request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReqState {
    pub request: RequestId,
    /// The last *committed* token for this request — fed as the prefix to
    /// the next draft/verify (`accepted_token` in `SystemSpecDraftRequest`).
    pub last_token: TokenId,
    /// Position of `last_token` (`source_position`).
    pub last_position: u64,
    /// Original input length for the in-flight verify (`orig_n_in`). For a
    /// steady-state decode this is 1 (the single committed token re-fed).
    pub orig_input_len: usize,
    /// Last round's outcome + draft count, for adaptive budgeting.
    pub last_outcome: Option<VerifyOutcome>,
    pub last_n_drafts: usize,
}

impl ReqState {
    /// A fresh decode request: one committed `token` at `position`, no
    /// speculative history yet.
    pub fn new(request: RequestId, token: TokenId, position: u64) -> Self {
        Self {
            request,
            last_token: token,
            last_position: position,
            orig_input_len: 1,
            last_outcome: None,
            last_n_drafts: 0,
        }
    }
}

/// Transient per-round scratch captured between phases (Draft writes it,
/// Verify/Repair/CommitAdvance read it). Kept explicit rather than implied
/// by call ordering so each transition is a pure function of state + input.
#[derive(Clone, Debug, Default)]
struct RoundScratch {
    /// `request -> proposed draft tokens` for the in-flight round.
    drafts: Vec<(RequestId, Vec<TokenId>)>,
    /// `request -> verify outcome` filled by `Verify`.
    outcomes: Vec<(RequestId, VerifyOutcome)>,
}

/// Drives the Draft→Verify→Repair→CommitAdvance cycle over a batch of
/// speculating requests. Holds the explicit `Phase` and per-request state;
/// every transition is a method that consumes the current phase and the
/// device-supplied facts (via the [`Drafter`]) and yields the next phase.
///
/// Replaces the implicit ordering inside `handle_fire_batch`: there the
/// "phase" is wherever the instruction pointer happens to be among the 264
/// helpers; here it is data you can assert on.
pub struct SpecEngine {
    pub config: SpecConfig,
    pub phase: Phase,
    /// Active speculating requests. Empty ⇒ nothing to do.
    pub requests: Vec<ReqState>,
    scratch: RoundScratch,
    /// Results of the most recently completed round (cleared at `Draft`).
    results: Vec<StepResult>,
}

impl SpecEngine {
    pub fn new(config: SpecConfig) -> Self {
        Self {
            config,
            phase: Phase::Draft,
            requests: Vec::new(),
            scratch: RoundScratch::default(),
            results: Vec::new(),
        }
    }

    /// Convenience: a fixed-budget engine (no adaptive draft count).
    pub fn fixed(max_drafts: usize) -> Self {
        Self::new(SpecConfig::fixed(max_drafts))
    }

    /// Register a request to speculate on (steady-state decode entry).
    pub fn add_request(&mut self, req: ReqState) {
        self.requests.push(req);
    }

    /// The results of the last completed round (valid after a
    /// `CommitAdvance`, until the next `Draft` clears them).
    pub fn results(&self) -> &[StepResult] {
        &self.results
    }

    // -- Phase transitions --------------------------------------------------

    /// **Draft.** Propose up to the (possibly adaptive) budget of tokens per
    /// request. Empty/zero-budget proposals are fine — they degrade to a
    /// single-token decode in `Verify` (the `n_d == 0` branch).
    ///
    /// `Draft → Verify`.
    pub fn draft<D: Drafter>(&mut self, drafter: &mut D) {
        assert_eq!(self.phase, Phase::Draft, "draft() requires Phase::Draft");
        let max = self.config.max_drafts.min(drafter.max_drafts());
        self.results.clear();
        self.scratch = RoundScratch::default();
        for req in &self.requests {
            let budget = self
                .config
                .next_drafts(req.last_outcome.as_ref(), req.last_n_drafts)
                .min(max);
            let proposed = if budget == 0 {
                Vec::new()
            } else {
                drafter.propose(req.request, req.last_token, req.last_position, budget)
            };
            // The drafter may return fewer than asked; never more.
            let proposed = if proposed.len() > budget {
                proposed[..budget].to_vec()
            } else {
                proposed
            };
            self.scratch.drafts.push((req.request, proposed));
        }
        self.phase = Phase::Verify;
    }

    /// **Verify.** One frozen target forward per request over its draft
    /// chain; diff against the proposal to get the accepted prefix.
    ///
    /// `Verify → Repair`.
    pub fn verify<D: Drafter>(&mut self, drafter: &mut D) {
        assert_eq!(self.phase, Phase::Verify, "verify() requires Phase::Verify");
        // Index requests by id for prefix lookup without borrow conflicts.
        for (req_id, drafts) in &self.scratch.drafts {
            let prefix = self
                .requests
                .iter()
                .find(|r| r.request == *req_id)
                .map(|r| r.last_token)
                .expect("scratch request must be active");
            let verified = drafter.verify(*req_id, prefix, drafts);
            let outcome = verify_outcome(drafts, &verified);
            self.scratch.outcomes.push((*req_id, outcome));
        }
        self.phase = Phase::Repair;
    }

    /// **Repair.** Batched correction of the rejected suffix. In the
    /// frozen-verify design the "repair" is cheap: the verify already
    /// produced the target's correction token (`bonus_token`), so the
    /// rejected drafts are simply dropped and the correction appended. The
    /// emitted list becomes `[accepted.., bonus]`. No separate device
    /// forward is needed for correctness — the heavy lifting is the
    /// commit-advance that follows.
    ///
    /// `Repair → CommitAdvance`.
    pub fn repair(&mut self) {
        assert_eq!(self.phase, Phase::Repair, "repair() requires Phase::Repair");
        // Nothing device-side here in the frozen-verify model; the
        // correction token is already in hand. We just assert the outcomes
        // were computed. (Kept as an explicit phase to mirror the design's
        // four-state cycle and to host a real device call in phase 4.)
        debug_assert_eq!(
            self.scratch.outcomes.len(),
            self.scratch.drafts.len(),
            "every drafted request must have a verify outcome before repair"
        );
        self.phase = Phase::CommitAdvance;
    }

    /// **CommitAdvance.** Advance committed recurrent/KV state over the
    /// confirmed prefix `[input | accepted]` for each request, then roll
    /// per-request state forward for the next round. Produces the
    /// [`StepResult`] list.
    ///
    /// `CommitAdvance → Draft` (cycle complete).
    pub fn commit_advance<D: Drafter>(&mut self, drafter: &mut D) {
        assert_eq!(
            self.phase,
            Phase::CommitAdvance,
            "commit_advance() requires Phase::CommitAdvance"
        );
        // Drain scratch so we can mutate `self.requests` freely.
        let drafts = std::mem::take(&mut self.scratch.drafts);
        let outcomes = std::mem::take(&mut self.scratch.outcomes);

        for (req_id, drafted) in &drafts {
            let outcome = outcomes
                .iter()
                .find(|(id, _)| id == req_id)
                .map(|(_, o)| o.clone())
                .expect("repair must have produced an outcome");

            let emitted = accepted_tokens(drafted, &outcome);
            let bonus = outcome.bonus_token;

            // Find the request state to advance.
            let req = self
                .requests
                .iter_mut()
                .find(|r| r.request == *req_id)
                .expect("committing an active request");

            // commit_len = orig_n_in + accepted (executor.cpp:2881).
            let commit = CommitAdvance {
                commit_len: req.orig_input_len + outcome.accepted_len,
                orig_input_len: req.orig_input_len,
                accepted_len: outcome.accepted_len,
                // The request emitted `emitted_len` tokens ending at the
                // bonus; the next token sits one past the last emitted.
                next_position: req.last_position + outcome.emitted_len() as u64,
            };

            // Device-side state advance (no-op for the fake).
            drafter.commit_advance(*req_id, &commit);

            // Roll request state forward: the bonus is now the last
            // committed token; steady-state re-feeds exactly one token.
            req.last_token = bonus;
            req.last_position = commit.next_position;
            req.orig_input_len = 1;
            req.last_outcome = Some(outcome.clone());
            req.last_n_drafts = drafted.len();

            self.results.push(StepResult {
                request: *req_id,
                emitted,
                outcome,
                commit,
                proposed_len: drafted.len(),
            });
        }
        self.phase = Phase::Draft;
    }

    /// Run one full Draft→Verify→Repair→CommitAdvance round and return the
    /// per-request results. Convenience wrapper over the four transitions
    /// for callers that don't need to interleave device work between phases.
    pub fn step<D: Drafter>(&mut self, drafter: &mut D) -> Vec<StepResult> {
        self.draft(drafter);
        self.verify(drafter);
        self.repair();
        self.commit_advance(drafter);
        self.results.clone()
    }
}

// ===========================================================================
// Tests — pure transition logic + acceptance math. A fake Drafter stands in
// for the device path.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A scripted, device-free drafter. `proposals` gives the tokens the
    /// drafter will offer per request; `target` gives the frozen target's
    /// opinion at each absolute position (what `verify` "should" produce).
    struct FakeDrafter {
        max_drafts: usize,
        /// Per request: a queue of proposals (one Vec per round).
        proposals: HashMap<RequestId, Vec<Vec<TokenId>>>,
        /// Per request: the target's "true" token at each position index.
        /// `target[req][p]` = the token the target would sample for the
        /// slot whose *input* position is `p` (i.e. predicting `p`).
        target: HashMap<RequestId, Vec<TokenId>>,
        /// Records of commit_advance calls, for assertions.
        commits: Vec<(RequestId, CommitAdvance)>,
        /// Round cursor per request into `proposals`.
        round: HashMap<RequestId, usize>,
        /// Last prefix position seen per request (set in verify), so the
        /// target lookup knows where the chain starts.
        prefix_pos: HashMap<RequestId, u64>,
    }

    impl FakeDrafter {
        fn new(max_drafts: usize) -> Self {
            Self {
                max_drafts,
                proposals: HashMap::new(),
                target: HashMap::new(),
                commits: Vec::new(),
                round: HashMap::new(),
                prefix_pos: HashMap::new(),
            }
        }
    }

    impl Drafter for FakeDrafter {
        fn max_drafts(&self) -> usize {
            self.max_drafts
        }

        fn propose(
            &mut self,
            request: RequestId,
            _last_token: TokenId,
            last_position: u64,
            n: usize,
        ) -> Vec<TokenId> {
            self.prefix_pos.insert(request, last_position);
            let idx = *self.round.get(&request).unwrap_or(&0);
            let p = self
                .proposals
                .get(&request)
                .and_then(|rounds| rounds.get(idx))
                .cloned()
                .unwrap_or_default();
            // Advance the round cursor on propose (one propose per round).
            self.round.insert(request, idx + 1);
            // Honour the budget n.
            if p.len() > n { p[..n].to_vec() } else { p }
        }

        fn verify(
            &mut self,
            request: RequestId,
            _prefix_token: TokenId,
            drafts: &[TokenId],
        ) -> Vec<TokenId> {
            // The frozen target predicts the token at each successive
            // position starting from `prefix_pos + 1`. The verify produces
            // n_d + 1 tokens: positions [prefix+1 .. prefix+1 + n_d].
            let base = *self.prefix_pos.get(&request).unwrap_or(&0) + 1;
            let tgt = self.target.get(&request).cloned().unwrap_or_default();
            (0..=drafts.len())
                .map(|j| {
                    let pos = (base as usize) + j;
                    *tgt.get(pos).unwrap_or(&0)
                })
                .collect()
        }

        fn commit_advance(&mut self, request: RequestId, commit: &CommitAdvance) {
            self.commits.push((request, commit.clone()));
        }
    }

    // -- acceptance math ----------------------------------------------------

    #[test]
    fn outcome_all_accepted() {
        // 3 drafts, all match; bonus is the 4th verified token.
        let drafted = [10, 11, 12];
        let verified = [10, 11, 12, 13];
        let o = verify_outcome(&drafted, &verified);
        assert_eq!(o.accepted_len, 3);
        assert_eq!(o.bonus_token, 13);
        assert!(!o.needs_repair);
        assert_eq!(o.emitted_len(), 4);
        assert_eq!(accepted_tokens(&drafted, &o), vec![10, 11, 12, 13]);
    }

    #[test]
    fn outcome_mismatch_at_each_k() {
        // For each first-mismatch position k in 0..n, accepted_len == k,
        // bonus is the target's correction at k, needs_repair is true.
        let n = 5;
        for k in 0..n {
            let drafted: Vec<TokenId> = (0..n as TokenId).map(|i| 100 + i).collect();
            // verified agrees up to k, then diverges at k.
            let mut verified: Vec<TokenId> = drafted.clone();
            verified[k] = 999; // correction at first mismatch
            verified.push(7); // tail bonus slot (unused when mismatch < n)
            let o = verify_outcome(&drafted, &verified);
            assert_eq!(o.accepted_len, k, "k={k}");
            assert_eq!(o.bonus_token, 999, "k={k}");
            assert!(o.needs_repair, "k={k}");
            assert_eq!(o.emitted_len(), k + 1);
            let emitted = accepted_tokens(&drafted, &o);
            assert_eq!(emitted.len(), k + 1);
            assert_eq!(*emitted.last().unwrap(), 999);
            // accepted prefix is the matching drafts.
            assert_eq!(&emitted[..k], &drafted[..k]);
        }
    }

    #[test]
    fn outcome_mismatch_at_zero() {
        // First draft already wrong → accept nothing, take correction.
        let drafted = [42, 43];
        let verified = [7, 8, 9];
        let o = verify_outcome(&drafted, &verified);
        assert_eq!(o.accepted_len, 0);
        assert_eq!(o.bonus_token, 7);
        assert!(o.needs_repair);
        assert_eq!(accepted_tokens(&drafted, &o), vec![7]);
    }

    #[test]
    fn outcome_empty_drafts_is_single_decode() {
        // Zero drafts: degenerate single-token decode. verified has exactly
        // one token (the bonus), accepted_len 0, no repair.
        let drafted: [TokenId; 0] = [];
        let verified = [55];
        let o = verify_outcome(&drafted, &verified);
        assert_eq!(o.accepted_len, 0);
        assert_eq!(o.bonus_token, 55);
        assert!(!o.needs_repair);
        assert_eq!(o.emitted_len(), 1);
        assert_eq!(accepted_tokens(&drafted, &o), vec![55]);
    }

    #[test]
    #[should_panic(expected = "n_drafts + 1")]
    fn outcome_bad_verify_len_panics() {
        let drafted = [1, 2, 3];
        let verified = [1, 2, 3]; // missing the bonus slot
        let _ = verify_outcome(&drafted, &verified);
    }

    // -- adaptive draft count ----------------------------------------------

    #[test]
    fn next_drafts_fixed() {
        let c = SpecConfig::fixed(4);
        assert_eq!(c.next_drafts(None, 0), 4);
        let full = VerifyOutcome { accepted_len: 2, bonus_token: 0, needs_repair: true };
        assert_eq!(c.next_drafts(Some(&full), 4), 4);
    }

    #[test]
    fn next_drafts_adaptive() {
        let c = SpecConfig {
            max_drafts: 8,
            adaptive: true,
            min_drafts: 2,
            initial_drafts: 4,
        };
        // No history → initial.
        assert_eq!(c.next_drafts(None, 0), 4);
        // Full hit last round (match == n) → grow to n+1.
        let hit = VerifyOutcome { accepted_len: 4, bonus_token: 0, needs_repair: false };
        assert_eq!(c.next_drafts(Some(&hit), 4), 5);
        // Partial hit → aim just past the matched prefix (match+1), floored.
        let partial = VerifyOutcome { accepted_len: 5, bonus_token: 0, needs_repair: true };
        assert_eq!(c.next_drafts(Some(&partial), 8), 6);
        // Zero acceptance → floor at min_drafts (0+1 = 1, clamped up to 2).
        let zero = VerifyOutcome { accepted_len: 0, bonus_token: 0, needs_repair: true };
        assert_eq!(c.next_drafts(Some(&zero), 6), 2);
        // max_drafts <= 1 short-circuits regardless of adaptivity.
        let c1 = SpecConfig { max_drafts: 1, adaptive: true, min_drafts: 1, initial_drafts: 1 };
        assert_eq!(c1.next_drafts(Some(&hit), 4), 1);
    }

    // -- phase transitions --------------------------------------------------

    #[test]
    fn phase_cycle_order() {
        let mut eng = SpecEngine::fixed(2);
        eng.add_request(ReqState::new(0, 1, 0));
        let mut d = FakeDrafter::new(2);
        d.proposals.insert(0, vec![vec![5, 6]]);
        d.target.insert(0, vec![0, 5, 6, 7]); // pos1=5,pos2=6,pos3=7

        assert_eq!(eng.phase, Phase::Draft);
        eng.draft(&mut d);
        assert_eq!(eng.phase, Phase::Verify);
        eng.verify(&mut d);
        assert_eq!(eng.phase, Phase::Repair);
        eng.repair();
        assert_eq!(eng.phase, Phase::CommitAdvance);
        eng.commit_advance(&mut d);
        assert_eq!(eng.phase, Phase::Draft); // cycled back
    }

    #[test]
    #[should_panic(expected = "Phase::Draft")]
    fn phase_guard_rejects_out_of_order() {
        let mut eng = SpecEngine::fixed(2);
        let mut d = FakeDrafter::new(2);
        eng.phase = Phase::Verify;
        eng.draft(&mut d); // wrong phase
    }

    // -- full single-request cycle, all accepted ---------------------------

    #[test]
    fn step_all_accepted_advances_by_n_plus_1() {
        let mut eng = SpecEngine::fixed(3);
        eng.add_request(ReqState::new(7, 100, 10)); // last token 100 at pos 10
        let mut d = FakeDrafter::new(3);
        // Drafts at positions 11,12,13.
        d.proposals.insert(7, vec![vec![11, 12, 13]]);
        // Target agrees, with bonus 14 at pos 14. target[pos] indexed.
        d.target.insert(7, vec![0; 11].into_iter().chain([11, 12, 13, 14]).collect());

        let res = eng.step(&mut d);
        assert_eq!(res.len(), 1);
        let r = &res[0];
        assert_eq!(r.request, 7);
        assert_eq!(r.outcome.accepted_len, 3);
        assert!(!r.outcome.needs_repair);
        assert_eq!(r.emitted, vec![11, 12, 13, 14]); // 3 accepted + bonus
        assert_eq!(r.num_accepted(), 4);
        // commit_len = orig_n_in(1) + accepted(3) = 4
        assert_eq!(r.commit.commit_len, 4);
        // next position = 10 + 4
        assert_eq!(r.commit.next_position, 14);
        // request state rolled forward.
        let st = &eng.requests[0];
        assert_eq!(st.last_token, 14);
        assert_eq!(st.last_position, 14);
        assert_eq!(st.last_n_drafts, 3);
    }

    // -- full single-request cycle, mismatch at k --------------------------

    #[test]
    fn step_mismatch_advances_by_k_plus_1() {
        let mut eng = SpecEngine::fixed(4);
        eng.add_request(ReqState::new(3, 50, 20));
        let mut d = FakeDrafter::new(4);
        // 4 drafts at positions 21..24.
        d.proposals.insert(3, vec![vec![21, 22, 99, 24]]);
        // Target matches positions 21,22 then diverges (correction 23 at pos23).
        d.target
            .insert(3, vec![0; 21].into_iter().chain([21, 22, 23, 24, 25]).collect());

        let res = eng.step(&mut d);
        let r = &res[0];
        assert_eq!(r.outcome.accepted_len, 2); // 21,22 accepted; 99 != 23
        assert!(r.outcome.needs_repair);
        assert_eq!(r.outcome.bonus_token, 23); // correction
        assert_eq!(r.emitted, vec![21, 22, 23]); // 2 accepted + correction
        assert_eq!(r.commit.commit_len, 3); // orig 1 + accepted 2
        assert_eq!(r.commit.next_position, 23); // 20 + emitted_len(3)
        // committed token is the correction.
        assert_eq!(eng.requests[0].last_token, 23);
        assert_eq!(eng.requests[0].last_position, 23);
    }

    // -- empty draft fallback through the full cycle -----------------------

    #[test]
    fn step_empty_draft_degenerates_to_decode() {
        let mut eng = SpecEngine::fixed(3);
        eng.add_request(ReqState::new(1, 9, 5));
        let mut d = FakeDrafter::new(3);
        // Drafter offers nothing this round.
        d.proposals.insert(1, vec![vec![]]);
        d.target.insert(1, vec![0; 6].into_iter().chain([42]).collect()); // pos6 = 42

        let res = eng.step(&mut d);
        let r = &res[0];
        assert_eq!(r.proposed_len, 0);
        assert_eq!(r.outcome.accepted_len, 0);
        assert!(!r.outcome.needs_repair);
        assert_eq!(r.emitted, vec![42]); // single decoded token
        assert_eq!(r.commit.commit_len, 1); // just the input
        assert_eq!(r.commit.next_position, 6); // 5 + 1
        assert_eq!(eng.requests[0].last_token, 42);
    }

    // -- multi-step cycle exercising every transition, across rounds -------

    #[test]
    fn multi_step_cycle_all_phases() {
        let mut eng = SpecEngine::fixed(2);
        eng.add_request(ReqState::new(0, 1000, 0));
        let mut d = FakeDrafter::new(2);
        // Round 0: drafts [1,2] at pos 1,2; target agrees + bonus 3 → advance 3.
        // Round 1: drafts [4,9] at pos 4,5; target says pos4=4 (hit), pos5=5
        //          (miss 9) → accept 1, correction 5 → advance 2.
        // Round 2: empty drafts; last_position is 5 so verify predicts
        //          position 6 → single decode of tgt[6].
        d.proposals.insert(0, vec![vec![1, 2], vec![4, 9], vec![]]);
        // Absolute target tokens by position:
        // pos1=1 pos2=2 pos3=3  (round0)
        // pos4=4 pos5=5         (round1: predicts pos4 then pos5)
        // pos6=7                (round2)
        let mut tgt = vec![0u32; 8];
        tgt[1] = 1;
        tgt[2] = 2;
        tgt[3] = 3;
        tgt[4] = 4;
        tgt[5] = 5;
        tgt[6] = 7;
        d.target.insert(0, tgt);

        // Round 0 — explicit phase walk to exercise every transition.
        eng.draft(&mut d);
        assert_eq!(eng.phase, Phase::Verify);
        eng.verify(&mut d);
        assert_eq!(eng.phase, Phase::Repair);
        eng.repair();
        assert_eq!(eng.phase, Phase::CommitAdvance);
        eng.commit_advance(&mut d);
        assert_eq!(eng.phase, Phase::Draft);
        {
            let r = &eng.results()[0];
            assert_eq!(r.emitted, vec![1, 2, 3]); // all accepted + bonus
            assert!(!r.outcome.needs_repair);
            assert_eq!(eng.requests[0].last_position, 3);
            assert_eq!(eng.requests[0].last_token, 3);
        }

        // Round 1 — partial accept via step().
        let r1 = eng.step(&mut d);
        {
            let r = &r1[0];
            assert_eq!(r.outcome.accepted_len, 1); // 4 hit, 9 miss
            assert!(r.outcome.needs_repair);
            assert_eq!(r.emitted, vec![4, 5]); // 1 accepted + correction 5
            assert_eq!(eng.requests[0].last_position, 5); // 3 + 2
            assert_eq!(eng.requests[0].last_token, 5);
        }

        // Round 2 — empty-draft single decode via step().
        let r2 = eng.step(&mut d);
        {
            let r = &r2[0];
            assert_eq!(r.proposed_len, 0);
            assert_eq!(r.emitted, vec![7]);
            assert!(!r.outcome.needs_repair);
            assert_eq!(eng.requests[0].last_position, 6); // 5 + 1
            assert_eq!(eng.requests[0].last_token, 7);
        }

        // Three rounds, three commit_advance calls recorded device-side.
        assert_eq!(d.commits.len(), 3);
        assert_eq!(d.commits[0].1.commit_len, 3); // 1 + 2 accepted
        assert_eq!(d.commits[1].1.commit_len, 2); // 1 + 1 accepted
        assert_eq!(d.commits[2].1.commit_len, 1); // 1 + 0 accepted
    }

    // -- multi-request batch in one round ----------------------------------

    #[test]
    fn batch_multiple_requests_one_round() {
        let mut eng = SpecEngine::fixed(2);
        eng.add_request(ReqState::new(0, 1, 0));
        eng.add_request(ReqState::new(1, 1, 100));
        let mut d = FakeDrafter::new(2);
        d.proposals.insert(0, vec![vec![1, 2]]); // req0 all accepted
        d.proposals.insert(1, vec![vec![50, 51]]); // req1 first miss
        let mut t0 = vec![0u32; 4];
        t0[1] = 1;
        t0[2] = 2;
        t0[3] = 3;
        d.target.insert(0, t0);
        let mut t1 = vec![0u32; 104];
        t1[101] = 77; // req1 pos101 (predict after pos100) = 77 != 50
        d.target.insert(1, t1);

        let res = eng.step(&mut d);
        assert_eq!(res.len(), 2);
        let r0 = res.iter().find(|r| r.request == 0).unwrap();
        let r1 = res.iter().find(|r| r.request == 1).unwrap();
        assert_eq!(r0.emitted, vec![1, 2, 3]);
        assert!(!r0.outcome.needs_repair);
        assert_eq!(r1.outcome.accepted_len, 0);
        assert_eq!(r1.emitted, vec![77]); // immediate correction
        assert!(r1.outcome.needs_repair);
    }
}
