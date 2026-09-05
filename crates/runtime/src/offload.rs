//! Remote executor partners: the registry and the admission accounting, and a
//! named hole where the transport goes — remote is a property of a transport,
//! not an encoding of a contract, so nothing here moves bytes.
//!
//! What's kept is everything above the wire: the registry (which peers
//! exist, in which role, behind which `EngineId`), admission accounting
//! (`max_outstanding`, claim/release guard, drain notification,
//! power-of-two-choices selection), and the settings/counters an operator
//! reads. A registered peer's engine
//! (`crate::engine::backend::remote::RemoteEngine`) answers
//! [`Error::Unsupported`](engine::Error::Unsupported) to every verb, so the
//! runtime can see the peer, refuse to use it, and say why — rather than
//! silently dropping the request.

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

/// How a peer's bytes are meant to reach this node. A runtime type, not a
/// contract one: which wire a deployment runs on is the runtime's to
/// configure, not the contract's to encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TransferKind {
    /// The peer's pages travel inside the reply.
    #[default]
    Inline,
    /// The peer writes into this node's pool over RDMA.
    Nixl,
}

/// One registered peer: the identity a scheduler routes on and the
/// accounting an admission decision reads.
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
    /// Always, until the envelope exists: it must carry either page bytes
    /// or an RDMA registration
    /// ([`KvHandle`](engine::KvHandle)), and say which one it used, since
    /// the caller's next act differs between them.
    pub async fn pull_kv(&self, src_page_ids: Vec<u32>, dst_page_ids: Vec<u32>) -> Result<()> {
        let _ = (src_page_ids, dst_page_ids);
        Err(self.no_transport("pull_kv"))
    }

    /// Run this peer's encoder over a media payload.
    ///
    /// # Errors
    ///
    /// Always, until the envelope exists.
    /// [`MediaEncode`](engine::MediaEncode) already carries its own bytes;
    /// what the envelope must add is a size policy for large payloads.
    pub async fn encode(&self, plan: engine::MediaEncode) -> Result<engine::MediaEncode> {
        let _ = plan;
        Err(self.no_transport("encode"))
    }

    fn no_transport(&self, verb: &'static str) -> anyhow::Error {
        COUNTERS.remote_failure.fetch_add(1, Ordering::Relaxed);
        anyhow!(
            "partner {} cannot serve `{verb}`: remote executors are not \
             supported in this release",
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

/// Register a peer, replacing (and suspecting) any earlier one in the same role.
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

/// Close every surrogate instance an engine holds. Nothing holds a surrogate,
/// so this has nothing to close; it stays as a door the link layer calls on
/// teardown.
pub fn close_engine_surrogates(engine_id: usize) {
    let _ = engine_id;
}

/// Note that a home instance is gone. As with
/// [`close_engine_surrogates`], there is nothing to do while no surrogate
/// exists.
pub(crate) fn close_home_instance(home_instance_id: u64) {
    let _ = home_instance_id;
}

pub fn register_remote_store(
    model_idx: usize,
    engine_idx: usize,
    kv_page_size: u32,
    base_page: u32,
    num_kv_pages: usize,
) -> anyhow::Result<()> {
    crate::store::registry::register_engine_with_swap(
        model_idx,
        engine_idx,
        kv_page_size,
        base_page,
        num_kv_pages,
        0,
        0,
        0,
    )
}

pub fn unregister_remote_store(model_idx: usize, engine_idx: usize) -> anyhow::Result<()> {
    crate::store::registry::unregister_engine(model_idx, engine_idx)
}

/// Claim the least-loaded of two random available peers in `role`
/// (power-of-two-choices).
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
static HOME_KV_HANDLE: LazyLock<RwLock<Option<engine::KvHandle>>> =
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
/// write. Still meaningful with no transport: [`KvHandle`](engine::KvHandle)
/// is a fact about this node's pool.
pub fn set_home_kv_handle(handle: engine::KvHandle) {
    *HOME_KV_HANDLE.write().unwrap() = Some(handle);
}

/// This node's own KV pool address, if an engine exported one.
pub fn home_kv_handle() -> Option<engine::KvHandle> {
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
/// answers with. Answers whether anything was injected — `false` every
/// time, for now: two things are still missing and independent, the
/// envelope ([`Partner::encode`]) and the fire's own seat for precomputed
/// embedding rows (what a fire needs after an encode is rows in the arena,
/// a seam the shell resolves rather than a payload the submission carries).
/// The configured-check is kept so a caller gets the partner's refusal
/// rather than silence from here.
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
        .encode(engine::MediaEncode::default())
        .await
    {
        tracing::warn!(%error, "encode offload declined");
    }
    false
}

/// Forget every peer. The shutdown path's door, and the only way a test can
/// start from a known registry — `pub` rather than `#[cfg(test)]` since the
/// crate's public facade re-exports it.
pub fn clear_partners() {
    let partners: Vec<Arc<Partner>> = PARTNERS.write().unwrap().drain().map(|(_, p)| p).collect();
    for partner in partners {
        partner.mark_suspect();
    }
}

