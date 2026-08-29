//! Remote executor partners: the registry and the admission accounting, and a
//! named hole where the transport was.
//!
//! # `palo B-remote`: what this module lost, and why it kept its shape
//!
//! Every verb that moved bytes lived on `engine_api::remote` —
//! `ExecutorRpcClient`, `ExecutorRequest`/`ExecutorResponse`, `PushKv`,
//! `InlineKvPayload`, `RemoteEncode`, `RemoteEmbeddings`, `RemoteMediaBlob`,
//! `ScratchGrant`, `REMOTE_WIRE_VERSION` — and the palo contract rewrite
//! deleted the module whole, on the ruling that remote is a property of a
//! transport and not an encoding of a contract (design §7, decision 19). So
//! the ~1,300 lines that framed those messages are gone: the inline KV push
//! and its host-side region import, the encode blob server and its TCP
//! listener, the surrogate-instance cache, the prefix-adoption planner, and
//! `try_prefill`.
//!
//! What is NOT gone is everything above the wire, because none of it was
//! about the wire:
//!
//! * the **registry** — which peers exist, in which role, behind which
//!   `EngineId`;
//! * the **admission accounting** — `max_outstanding`, the claim/release
//!   guard, the drain notification, power-of-two-choices selection;
//! * the **settings** and the **counters** an operator reads.
//!
//! A registered peer is an `EngineId` like any other, and the engine behind it
//! is `crate::engine::backend::remote::RemoteEngine`, which answers
//! [`Error::Unsupported`](engine_api::Error::Unsupported) to
//! every verb with the peer named. That is the shape the rewrite asks for: a
//! remote engine is *a `dyn Engine` whose envelope is the transport's*, and
//! until the transport exists the runtime can see the peer, refuse to use it,
//! and say why — rather than silently dropping the request, which is what a
//! stub that answered `Ok` would do.
//!
//! Each site the transport used to occupy carries a `palo B-remote:` marker
//! naming what the future envelope has to carry there.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

use anyhow::{Result, anyhow};

/// Which side of a disaggregated deployment a peer serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartnerRole {
    /// It runs prefills and hands back KV.
    Prefill,
    /// It runs a multimodal encoder and hands back embedding rows.
    Encode,
}

/// How a peer's bytes are meant to reach this node.
///
/// **A RUNTIME TYPE NOW, NOT A CONTRACT ONE.** It was
/// `engine_api::RemoteTransferKind`, sitting in the contract beside the tarpc
/// service it selected a codec for. Which wire a deployment runs on is a
/// deployment's decision and a transport's implementation; the runtime holds
/// it because the runtime is what an operator configures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TransferKind {
    /// The peer's pages travel inside the reply.
    #[default]
    Inline,
    /// The peer writes into this node's pool over RDMA.
    Nixl,
}

/// One registered peer.
///
/// The `client: Option<ExecutorRpcClient>` field is gone with the envelope
/// (see the module header); what is left is the identity a scheduler routes
/// on and the accounting an admission decision reads.
pub struct Partner {
    worker_id: u64,
    destination_worker_id: u64,
    engine_id: Option<usize>,
    role: PartnerRole,
    transfer: TransferKind,
    blob_host: RwLock<Option<IpAddr>>,
    max_outstanding: u32,
    outstanding: AtomicU32,
    available: AtomicBool,
    drained: tokio::sync::Notify,
}

impl Partner {
    /// Which worker this is.
    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    /// Which worker its answers are destined for.
    pub fn destination_worker_id(&self) -> u64 {
        self.destination_worker_id
    }

    /// Which registry slot its engine holds.
    ///
    /// # Panics
    ///
    /// For an encode partner, which has no engine of its own.
    pub fn engine_id(&self) -> usize {
        self.engine_id
            .expect("only prefill partners have remote engine slots")
    }

    /// Which side it serves.
    pub fn role(&self) -> PartnerRole {
        self.role
    }

    /// Which wire it was configured for.
    pub fn transfer_kind(&self) -> TransferKind {
        self.transfer
    }

    /// How many claims are outstanding against it.
    pub fn outstanding(&self) -> u32 {
        self.outstanding.load(Ordering::Relaxed)
    }

    /// Stop selecting it.
    pub fn mark_suspect(&self) {
        self.available.store(false, Ordering::Release);
    }

    /// Select it again.
    pub fn mark_available(&self) {
        self.available.store(true, Ordering::Release);
    }

    /// Where its large payloads would be fetched from.
    pub fn set_blob_host(&self, host: IpAddr) {
        *self.blob_host.write().unwrap() = Some(host);
    }

    /// Where its large payloads would be fetched from, if it stated one.
    pub fn blob_host(&self) -> Option<IpAddr> {
        *self.blob_host.read().unwrap()
    }

    /// Park until nothing is outstanding against it.
    pub async fn wait_drained(&self) {
        loop {
            if self.outstanding() == 0 {
                return;
            }
            let notified = self.drained.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.outstanding() == 0 {
                return;
            }
            notified.await;
        }
    }

    /// Pull this peer's KV pages into the home pool.
    ///
    /// # Errors
    ///
    /// Always, until the envelope exists.
    ///
    /// **`palo B-remote`**: the envelope must carry, for each direction it
    /// serves, either the page BYTES (the old `InlineKvPayload`: a page-stride
    /// count, the destination page ids echoed back, and one flat buffer the
    /// caller splits per region) or an RDMA registration
    /// ([`KvHandle`](engine_api::KvHandle) is serde and already says where a
    /// pool's regions are). It must also answer which one it used, because
    /// the caller's next act differs: an inline reply is imported region by
    /// region on the host; an RDMA push is already done when the reply
    /// arrives.
    pub async fn pull_kv(&self, src_page_ids: Vec<u32>, dst_page_ids: Vec<u32>) -> Result<()> {
        let _ = (src_page_ids, dst_page_ids);
        Err(self.no_transport("pull_kv"))
    }

    /// Run this peer's encoder over a media payload.
    ///
    /// # Errors
    ///
    /// Always, until the envelope exists.
    ///
    /// **`palo B-remote`**: [`MediaEncode`](engine_api::MediaEncode) is serde
    /// and carries its own bytes, so the message is that type and the answer
    /// is the same value with `output_rows` filled. What the envelope must
    /// add is a SIZE policy — an encode is megabytes, the old code stood up a
    /// TCP blob server above four of them, and a frame limit is the
    /// transport's business rather than the contract's.
    pub async fn encode(&self, plan: engine_api::MediaEncode) -> Result<engine_api::MediaEncode> {
        let _ = plan;
        Err(self.no_transport("encode"))
    }

    fn no_transport(&self, verb: &'static str) -> anyhow::Error {
        COUNTERS.remote_failure.fetch_add(1, Ordering::Relaxed);
        anyhow!(
            "partner {} cannot serve `{verb}`: the remote envelope \
             `engine_api::remote` carried was deleted by the palo contract \
             rewrite and its successor is palo B-remote",
            self.worker_id
        )
    }

    fn try_claim(self: &Arc<Self>) -> Option<PartnerGuard> {
        if self.transfer == TransferKind::Nixl && NIXL_QUARANTINED.load(Ordering::Acquire) {
            return None;
        }
        if !self.available.load(Ordering::Acquire) {
            return None;
        }
        let claimed = self
            .outstanding
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < self.max_outstanding).then_some(current + 1)
            })
            .is_ok();
        if !claimed {
            return None;
        }
        if !self.available.load(Ordering::Acquire)
            || (self.transfer == TransferKind::Nixl && NIXL_QUARANTINED.load(Ordering::Acquire))
        {
            let previous = self.outstanding.fetch_sub(1, Ordering::AcqRel);
            if previous == 1 {
                self.drained.notify_waiters();
            }
            return None;
        }
        Some(PartnerGuard {
            partner: Arc::clone(self),
        })
    }
}

/// One outstanding claim against a partner, released on drop.
pub struct PartnerGuard {
    partner: Arc<Partner>,
}

impl PartnerGuard {
    /// The peer this claim is against.
    pub fn partner(&self) -> &Arc<Partner> {
        &self.partner
    }
}

impl Drop for PartnerGuard {
    fn drop(&mut self) {
        let previous = self.partner.outstanding.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
        if previous == 1 {
            self.partner.drained.notify_waiters();
        }
    }
}

type PartnerKey = (PartnerRole, u64);

static PARTNERS: LazyLock<RwLock<HashMap<PartnerKey, Arc<Partner>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Register a peer, replacing (and suspecting) any earlier one in the same
/// role.
///
/// The `client: Option<ExecutorRpcClient>` argument is gone: there is no
/// client type to pass. Everything else is what it was.
pub fn register_partner(
    worker_id: u64,
    destination_worker_id: u64,
    engine_id: impl Into<Option<usize>>,
    role: PartnerRole,
    max_outstanding: u32,
    transfer: TransferKind,
) -> Arc<Partner> {
    let partner = Arc::new(Partner {
        worker_id,
        destination_worker_id,
        engine_id: engine_id.into(),
        role,
        transfer,
        blob_host: RwLock::new(None),
        max_outstanding: max_outstanding.max(1),
        outstanding: AtomicU32::new(0),
        available: AtomicBool::new(true),
        drained: tokio::sync::Notify::new(),
    });
    if let Some(previous) = PARTNERS
        .write()
        .unwrap()
        .insert((role, worker_id), Arc::clone(&partner))
    {
        previous.mark_suspect();
    }
    partner
}

/// Forget a peer.
pub fn remove_partner(worker_id: u64, role: PartnerRole) {
    if let Some(partner) = PARTNERS.write().unwrap().remove(&(role, worker_id)) {
        partner.mark_suspect();
    }
}

/// Close every surrogate instance an engine holds.
///
/// **`palo B-remote`**: a surrogate was a bound instance on the HOME engine
/// standing in for a peer's context extension, cached by
/// `(engine, program, home_instance)` — the prefix-adoption machinery. It
/// went with `try_prefill`; nothing holds a surrogate now, so this has
/// nothing to close. Kept as a door because the link layer calls it on
/// teardown and a peer's engine going away must remain a stated event.
pub fn close_engine_surrogates(engine_id: usize) {
    let _ = engine_id;
}

/// Note that a home instance is gone.
///
/// **`palo B-remote`**: as [`close_engine_surrogates`] — the home-state table
/// this walked existed to keep surrogates alive across a peer's answer.
pub(crate) fn close_home_instance(home_instance_id: u64) {
    let _ = home_instance_id;
}

/// Claim the least-loaded of two random available peers in `role`.
///
/// Power of two choices, unchanged: the selection policy is about load, and
/// load is not an encoding.
pub fn select_partner(role: PartnerRole) -> Option<PartnerGuard> {
    let candidates = PARTNERS
        .read()
        .unwrap()
        .values()
        .filter(|partner| {
            partner.role == role
                && partner.available.load(Ordering::Acquire)
                && partner.outstanding() < partner.max_outstanding
        })
        .cloned()
        .collect::<Vec<_>>();
    match candidates.len() {
        0 => None,
        1 => candidates[0].try_claim(),
        len => {
            let first = (next_random() % len as u64) as usize;
            let mut second = (next_random() % (len as u64 - 1)) as usize;
            if second >= first {
                second += 1;
            }
            let (a, b) = (&candidates[first], &candidates[second]);
            let preferred = if (a.outstanding(), a.engine_id) <= (b.outstanding(), b.engine_id) {
                [Arc::clone(a), Arc::clone(b)]
            } else {
                [Arc::clone(b), Arc::clone(a)]
            };
            let selected = preferred
                .into_iter()
                .find_map(|partner| partner.try_claim());
            selected.or_else(|| candidates.iter().find_map(|partner| partner.try_claim()))
        }
    }
}

fn next_random() -> u64 {
    static STATE: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);
    let mut value = STATE
        .fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Debug, Clone, Copy)]
struct OffloadSettings {
    enabled: bool,
    prefill_min_suffix_tokens: usize,
    encode_injection: bool,
    encode_hidden_size: u32,
}

static SETTINGS: LazyLock<RwLock<OffloadSettings>> = LazyLock::new(|| {
    RwLock::new(OffloadSettings {
        enabled: false,
        prefill_min_suffix_tokens: 0,
        encode_injection: false,
        encode_hidden_size: 0,
    })
});
static OFFLOAD_ENABLED: AtomicBool = AtomicBool::new(false);
static ENCODE_INJECTION_ENABLED: AtomicBool = AtomicBool::new(false);
static HOME_KV_HANDLE: LazyLock<RwLock<Option<engine_api::KvHandle>>> =
    LazyLock::new(|| RwLock::new(None));

/// Turn prefill offload on, and say how long a suffix has to be to qualify.
pub fn configure(enabled: bool, prefill_min_suffix_tokens: usize) {
    let mut settings = SETTINGS.write().unwrap();
    settings.enabled = enabled;
    settings.prefill_min_suffix_tokens = prefill_min_suffix_tokens;
    OFFLOAD_ENABLED.store(enabled, Ordering::Release);
}

/// Turn encode offload on, and say how wide an embedding row is.
pub fn configure_encode_injection(enabled: bool, hidden_size: u32) {
    let mut settings = SETTINGS.write().unwrap();
    settings.encode_injection = enabled;
    settings.encode_hidden_size = if enabled { hidden_size } else { 0 };
    ENCODE_INJECTION_ENABLED.store(enabled, Ordering::Release);
}

/// Publish this node's own KV pool address, so a peer can be told where to
/// write.
///
/// Still meaningful with no transport: [`KvHandle`](engine_api::KvHandle) is
/// a fact about this node's pool, and it is what a future envelope's hello
/// hands over.
pub fn set_home_kv_handle(handle: engine_api::KvHandle) {
    *HOME_KV_HANDLE.write().unwrap() = Some(handle);
}

/// This node's own KV pool address, if an engine exported one.
pub fn home_kv_handle() -> Option<engine_api::KvHandle> {
    HOME_KV_HANDLE.read().unwrap().clone()
}

/// What an operator reads to see where offload decisions went.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OffloadCounterSnapshot {
    /// No peer was available in the role.
    pub no_partner: u64,
    /// The suffix was shorter than the threshold.
    pub below_threshold: u64,
    /// The fire was not a plain contiguous append.
    pub noncanonical: u64,
    /// The fire touched recurrent state.
    pub recurrent_state: u64,
    /// The fire carried a guest-supplied mask.
    pub user_mask: u64,
    /// The fire touched a channel.
    pub channels: u64,
    /// The fire carried media.
    pub media: u64,
    /// The fire's shape was past a ceiling.
    pub shape: u64,
    /// The peer refused, or could not be reached.
    pub remote_failure: u64,
    /// The transfer failed after the peer answered.
    pub transfer_failure: u64,
    /// A prefix was adopted from a peer.
    pub adopted: u64,
    /// The NIXL path is quarantined after a failure.
    pub nixl_quarantined: bool,
}

#[derive(Default)]
struct OffloadCounters {
    no_partner: AtomicU64,
    below_threshold: AtomicU64,
    noncanonical: AtomicU64,
    recurrent_state: AtomicU64,
    user_mask: AtomicU64,
    channels: AtomicU64,
    media: AtomicU64,
    shape: AtomicU64,
    remote_failure: AtomicU64,
    transfer_failure: AtomicU64,
    adopted: AtomicU64,
}

static COUNTERS: LazyLock<OffloadCounters> = LazyLock::new(OffloadCounters::default);
static NIXL_QUARANTINED: AtomicBool = AtomicBool::new(false);

/// Read every counter at once.
pub fn counters() -> OffloadCounterSnapshot {
    let load = |value: &AtomicU64| value.load(Ordering::Relaxed);
    OffloadCounterSnapshot {
        no_partner: load(&COUNTERS.no_partner),
        below_threshold: load(&COUNTERS.below_threshold),
        noncanonical: load(&COUNTERS.noncanonical),
        recurrent_state: load(&COUNTERS.recurrent_state),
        user_mask: load(&COUNTERS.user_mask),
        channels: load(&COUNTERS.channels),
        media: load(&COUNTERS.media),
        shape: load(&COUNTERS.shape),
        remote_failure: load(&COUNTERS.remote_failure),
        transfer_failure: load(&COUNTERS.transfer_failure),
        adopted: load(&COUNTERS.adopted),
        nixl_quarantined: NIXL_QUARANTINED.load(Ordering::Acquire),
    }
}

/// Offer this fire's media to an encode partner, and inject the rows it
/// answers with.
///
/// Answers whether anything was injected — `false` every time, for now.
///
/// **`palo B-remote` AND `palo B-media`, IN ONE PLACE.** Two things stand
/// between this and working, and they are independent:
///
/// 1. the envelope (see [`Partner::encode`]);
/// 2. the fire's own seat for precomputed embedding rows. The wire plan
///    carried them inline — `embed_rows`, `embed_indptr`, `embed_shapes`,
///    `embed_anchor_rows`, `embed_block_indptr` — and the contract does not:
///    what a fire needs after an encode is ROWS IN THE ARENA, which is a seam
///    the shell resolves (design §9's export ops), not a payload the
///    submission carries. A `crate::engine::Media` struct held the payload
///    meanwhile and nothing ever wrote one, so alto E deleted it: the
///    payload arrives with the verb that produces it.
///
/// The check that the module is even configured for it is kept, so an
/// operator who turns encode injection on and sees nothing happen gets the
/// refusal from the partner rather than silence from here.
pub(crate) async fn try_encode(request: &mut crate::engine::FireRequest) -> bool {
    // The request is the seam the injected rows land on; nothing reads it
    // until there are rows to land.
    let _ = request;
    if !ENCODE_INJECTION_ENABLED.load(Ordering::Acquire) {
        return false;
    }
    let Some(guard) = select_partner(PartnerRole::Encode) else {
        COUNTERS.no_partner.fetch_add(1, Ordering::Relaxed);
        return false;
    };
    if let Err(error) = guard
        .partner()
        .encode(engine_api::MediaEncode::default())
        .await
    {
        tracing::warn!(%error, "encode offload declined");
    }
    false
}

/// Forget every peer.
///
/// The shutdown path's door, and the only way a test can start from a known
/// registry — which is why it stays `pub` rather than `#[cfg(test)]`: the
/// crate's public facade re-exports it (`crate::offload`), so a `cfg` would
/// break the export rather than document the intent.
pub fn clear_partners() {
    let partners: Vec<Arc<Partner>> = PARTNERS.write().unwrap().drain().map(|(_, p)| p).collect();
    for partner in partners {
        partner.mark_suspect();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, MutexGuard, PoisonError};

    use super::*;

    /// Serializes every test that reads or writes [`PARTNERS`].
    ///
    /// The registry is process-global and reachable only as a global: nothing
    /// in [`register_partner`], [`select_partner`] or [`remove_partner`] takes
    /// a registry to act on, because a node has exactly one set of peers and
    /// the runtime's public facade publishes those free functions as its whole
    /// interface to it. Threading a registry handle through them would be a
    /// production shape invented for the tests, so the honest fix is here: the
    /// tests take turns.
    ///
    /// Without a turn they wipe each other. Both start by calling
    /// [`clear_partners`], and `cargo test` runs them on separate threads by
    /// default, so under parallelism one test's `clear_partners` drains the
    /// peer the other just registered and its `select_partner` answers `None`
    /// — a flake that reported three times before it was read as a shared-state
    /// problem rather than a scheduling one.
    static REGISTRY: Mutex<()> = Mutex::new(());

    /// Take the registry for the duration of a test, empty.
    ///
    /// A test that fails while holding the lock poisons it, and a poisoned
    /// lock would turn one real failure into a cascade of unrelated ones. The
    /// state the poison warns about is exactly the state this drains on the
    /// way in, so [`PoisonError::into_inner`] is the correct reading and not a
    /// suppressed one.
    fn exclusive_registry() -> MutexGuard<'static, ()> {
        let guard = REGISTRY.lock().unwrap_or_else(PoisonError::into_inner);
        clear_partners();
        guard
    }

    #[test]
    fn a_claim_is_released_when_its_guard_drops() {
        let _registry = exclusive_registry();
        let partner = register_partner(1, 2, 7usize, PartnerRole::Prefill, 2, TransferKind::Inline);
        let first = select_partner(PartnerRole::Prefill).expect("a peer is available");
        assert_eq!(partner.outstanding(), 1);
        let second = select_partner(PartnerRole::Prefill).expect("two claims fit");
        assert_eq!(partner.outstanding(), 2);
        assert!(
            select_partner(PartnerRole::Prefill).is_none(),
            "a third claim is past max_outstanding"
        );
        drop(second);
        drop(first);
        assert_eq!(partner.outstanding(), 0);
        clear_partners();
    }

    /// Holds the registry across `await` points, which is sound here because
    /// `#[tokio::test]` drives the future to completion on this one thread: no
    /// other task can be scheduled onto the lock while it is parked.
    #[tokio::test]
    async fn every_transfer_verb_refuses_by_name_rather_than_answering_ok() {
        let _registry = exclusive_registry();
        let partner = register_partner(3, 4, 0usize, PartnerRole::Prefill, 1, TransferKind::Inline);
        let error = partner
            .pull_kv(vec![0], vec![1])
            .await
            .expect_err("there is no transport");
        assert!(
            format!("{error}").contains("palo B-remote"),
            "the refusal names the wave that owes the envelope: {error}"
        );
        let error = partner
            .encode(engine_api::MediaEncode::default())
            .await
            .expect_err("there is no transport");
        assert!(format!("{error}").contains("palo B-remote"), "{error}");
        clear_partners();
    }
}
