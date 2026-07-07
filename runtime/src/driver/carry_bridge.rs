//! # X2/X3 (a) BRIDGE — the runtime carry-descriptor + in-flight close-gate
//!
//! The FFI-free heart of guru's ruled **(a) BRIDGE** shape: the runtime `enqueue`
//! no longer calls `pie_frame_carry` itself; it produces a typed
//! [`CarryDescriptor`] (stashed via the request wire) that the CUDA executor reads
//! at a2 fire-commit and hands to `pie_frame_carry` alongside the device
//! `fire_done` event + the real `committed_head`. This module owns the two
//! bravo-owned invariants of that bridge, kept **off the `driver-cuda` FFI** so
//! they are unit-tested standalone (the FFI path can't link without the CUDA lib):
//!
//! * [`CarryDescriptor`] — the **single typed marshal point** (guru's ABI rule): a
//!   `#[repr(C)]` struct whose first two words are a **version + size**, so an
//!   executor built against a different layout **loud-rejects** rather than
//!   misreading raw bytes. Carries the completion callback + its per-batch context,
//!   the pinned `word_index`, and the instance.
//! * [`InFlightTracker`] — the **close-gate** (B6 / §5.2 grace): a carry is
//!   in-flight from `enqueue` (descriptor stashed) until its completion callback
//!   fires. [`InFlightTracker::request_close`] must **defer** the frame region-free
//!   while any carry is in flight, so a close-during-in-flight never frees a frame
//!   a pending carry still writes; the free lands on the completing decrement that
//!   drains the last in-flight carry.
//!
//! Gated on `ptir` only (not `driver-cuda`), so both invariants compile + test
//! without the CUDA driver lib. `control_cuda` (the FFI side) consumes them.

use std::collections::HashMap;
use std::sync::Mutex;

use pie_driver_abi::ForwardRequest;

/// Current [`CarryDescriptor`] ABI version. Bump on ANY layout change; the executor
/// rejects a descriptor whose `(version, size)` doesn't match its compiled-in
/// expectation, so a stale executor never misreads a re-laid-out descriptor.
pub const CARRY_DESCRIPTOR_VERSION: u32 = 1;

/// The typed carry descriptor — the (a) BRIDGE payload. The runtime `enqueue`
/// builds one per fire and stashes it on the request wire; the CUDA executor reads
/// it at a2 fire-commit and calls `pie_frame_carry(instance, word_index,
/// committed_head, fire_done, done, user_data)` with it + its device-side values.
///
/// **ABI discipline (guru):** `version` + `size` LEAD the struct so a mismatched
/// executor loud-rejects (see [`CarryDescriptor::validate`]) instead of misreading.
/// `#[repr(C)]` fixes the field order; addresses are plain `u64` (fn-ptr /
/// `*mut c_void` cast at the FFI boundary in `control_cuda`), keeping this module
/// FFI-free + testable.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarryDescriptor {
    /// ABI version — MUST equal [`CARRY_DESCRIPTOR_VERSION`]. Leads the struct.
    pub version: u32,
    /// `size_of::<CarryDescriptor>()` in bytes. Follows `version`. The executor
    /// rejects if `(version, size)` != its compiled-in expectation.
    pub size: u32,
    /// The bound instance this carry drives.
    pub instance: u64,
    /// The pinned ring `word_index` the carrier release-stores the committed head
    /// into (`2*c` for channel `c`; word 0 = the single provisional channel head).
    pub word_index: u64,
    /// The runtime completion callback (`cuda_carry_done`) as a raw address — a
    /// stable static `extern "C" fn(*mut c_void)`. The executor forwards it to
    /// `pie_frame_carry`'s `done` param.
    pub done: u64,
    /// The boxed per-batch completion context (`CarryWake`) as a raw address.
    /// Reclaimed exactly once when `done` fires.
    pub user_data: u64,
}

impl CarryDescriptor {
    /// Build a current-version descriptor. `size` is set from the compiled struct,
    /// so a rebuild after a layout change auto-updates the guard.
    pub fn new(instance: u64, word_index: u64, done: u64, user_data: u64) -> CarryDescriptor {
        CarryDescriptor {
            version: CARRY_DESCRIPTOR_VERSION,
            size: core::mem::size_of::<CarryDescriptor>() as u32,
            instance,
            word_index,
            done,
            user_data,
        }
    }

    /// Loud-reject an unknown layout (guru's ABI rule). `Err` carries the observed
    /// vs expected `(version, size)` so a mismatch is diagnosable, never a silent
    /// misread. The executor calls this before trusting any descriptor bytes.
    pub fn validate(&self) -> Result<(), CarryDescriptorError> {
        let expected_size = core::mem::size_of::<CarryDescriptor>() as u32;
        if self.version != CARRY_DESCRIPTOR_VERSION || self.size != expected_size {
            return Err(CarryDescriptorError {
                observed_version: self.version,
                observed_size: self.size,
                expected_version: CARRY_DESCRIPTOR_VERSION,
                expected_size,
            });
        }
        Ok(())
    }

    /// Marshal to the wire bytes stashed on the request (the SoA descriptor blob).
    /// `#[repr(C)]` fixes the layout; the version/size lead makes it self-describing.
    pub fn to_bytes(&self) -> [u8; core::mem::size_of::<CarryDescriptor>()] {
        // SAFETY: `CarryDescriptor` is `#[repr(C)]`, `Copy`, and contains only POD
        // integer fields (no padding-sensitive reads — we write the whole struct).
        unsafe { core::mem::transmute_copy(self) }
    }

    /// Unmarshal from wire bytes, then [`validate`](Self::validate) — an unknown
    /// `(version, size)` is a loud `Err`, never a misread. `Err(None)` if the blob
    /// is too short to even hold the version/size guard.
    pub fn from_bytes(bytes: &[u8]) -> Result<CarryDescriptor, Option<CarryDescriptorError>> {
        if bytes.len() < core::mem::size_of::<CarryDescriptor>() {
            return Err(None);
        }
        // SAFETY: bounds-checked above; `CarryDescriptor` is `#[repr(C)]` POD.
        let desc: CarryDescriptor = unsafe {
            let mut raw = core::mem::MaybeUninit::<CarryDescriptor>::uninit();
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                raw.as_mut_ptr() as *mut u8,
                core::mem::size_of::<CarryDescriptor>(),
            );
            raw.assume_init()
        };
        desc.validate().map_err(Some)?;
        Ok(desc)
    }
}

/// Marshal one carry into the request's parallel SoA cols — the (a) BRIDGE wire the
/// CUDA executor reads at a2 fire-commit. This is the runtime→FR *populate* step:
/// `enqueue`'s per-fire carry is stashed here (as parallel-array columns, charlie's
/// enumerated shape) instead of the runtime calling `pie_frame_carry` directly.
///
/// * `user_data` — the boxed completion context (`CarryWake`) as a raw address;
///   `control_cuda` builds + leaks the box and passes its pointer here. The executor
///   hands it back to the once-registered `done` (`cuda_carry_done`) verbatim, which
///   reclaims it exactly once on commit.
/// * `word_index` — the pinned ring word the carrier release-stores the committed
///   head into (`2*c`; word 0 = the single provisional channel head).
/// * `instance` — the per-request bound instance (a2 batches R requests, each its
///   own bound instance — hence a parallel `carry_instance` col, NOT one per fire).
///
/// The ABI-version guard (`carry_abi_version = [CARRY_DESCRIPTOR_VERSION]`) is set on
/// the FIRST push so the executor validates ONCE before trusting any col
/// (`carry_abi_version[0] == CARRY_DESCRIPTOR_VERSION`, else loud-reject). Empty cols
/// (never pushed) are the behavior-neutral dormant state charlie's a2 loop skips
/// (`if !carry_user_ptr.is_empty()`). Pure data (FFI-free) so it is unit-tested
/// without the CUDA driver lib.
pub fn push_carry_request(
    req: &mut ForwardRequest,
    user_data: u64,
    word_index: u64,
    instance: u64,
) {
    if req.carry_abi_version.is_empty() {
        req.carry_abi_version.push(CARRY_DESCRIPTOR_VERSION);
    }
    req.carry_user_ptr.push(user_data);
    req.carry_word_index.push(word_index);
    req.carry_instance.push(instance);
}

/// A [`CarryDescriptor`] layout mismatch — the loud-reject payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarryDescriptorError {
    pub observed_version: u32,
    pub observed_size: u32,
    pub expected_version: u32,
    pub expected_size: u32,
}

impl std::fmt::Display for CarryDescriptorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "carry-descriptor ABI mismatch: observed (v{}, {}B), expected (v{}, {}B)",
            self.observed_version, self.observed_size, self.expected_version, self.expected_size
        )
    }
}

impl std::error::Error for CarryDescriptorError {}

/// What [`InFlightTracker::request_close`] tells the caller to do with the frame
/// region-free (B6 / §5.2 grace).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloseAction {
    /// No carry is in flight — free the instance's frame region now.
    FreeNow,
    /// A carry is still in flight — the free is DEFERRED; it lands on the
    /// completing decrement that drains the last in-flight carry
    /// ([`InFlightTracker::on_complete`] returns `true` then).
    Deferred,
}

/// Per-instance in-flight carry counter for the close-gate. A carry is in flight
/// from `enqueue` (descriptor stashed) until its completion callback fires; the
/// frame region must not be freed while any carry is in flight (a pending carry
/// still writes the frame/mirror/word). Freeing is deferred until the count drains.
///
/// **Callback-safe free:** the completion callback (`cuda_carry_done`) runs on the
/// CUDA host-func thread and must not call CUDA APIs, so it CANNOT `pie_frame_close`
/// (cudaFree) directly. Instead [`on_complete`](Self::on_complete) records the
/// drained instance on a `needs_free` list; a runtime thread later drains it via
/// [`reap`](Self::reap), which performs the actual free OFF the callback thread.
///
/// Kept FFI-free so the mandated close-during-in-flight test runs standalone.
#[derive(Default)]
pub struct InFlightTracker {
    inner: Mutex<HashMap<u64, InstanceInFlight>>,
    /// Instances whose deferred free is owed (drained-to-0 with a close pending) —
    /// pushed by the completion callback (no CUDA), performed later by [`reap`].
    ///
    /// [`reap`]: Self::reap
    needs_free: Mutex<Vec<u64>>,
}

#[derive(Clone, Copy, Default)]
struct InstanceInFlight {
    in_flight: u64,
    close_pending: bool,
}

impl InFlightTracker {
    pub fn new() -> InFlightTracker {
        InFlightTracker {
            inner: Mutex::new(HashMap::new()),
            needs_free: Mutex::new(Vec::new()),
        }
    }

    /// The process-global tracker — the singleton the free `extern "C"` completion
    /// callback (`cuda_carry_done`) reaches without a threaded handle, mirroring
    /// [`CompletionConsumer::global`](crate::driver::CompletionConsumer). Provisional
    /// for the single embedded driver.
    pub fn global() -> &'static InFlightTracker {
        static GLOBAL: std::sync::OnceLock<InFlightTracker> = std::sync::OnceLock::new();
        GLOBAL.get_or_init(InFlightTracker::new)
    }

    /// Register a freshly bound instance (in_flight = 0). Overwrites any stale entry
    /// for a reused id.
    pub fn on_bind(&self, instance: u64) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(instance, InstanceInFlight::default());
    }

    /// Count a carry as in-flight at `enqueue` (descriptor stashed). Lazily seeds
    /// the entry so an enqueue that races bind is still counted.
    pub fn on_enqueue(&self, instance: u64) {
        let mut m = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        m.entry(instance).or_default().in_flight += 1;
    }

    /// Drain one in-flight carry at its completion callback. Returns `true` iff this
    /// was the last in-flight carry AND a close was pending — i.e. a DEFERRED frame
    /// region-free is now owed. **Callback-safe:** it never frees here (no CUDA);
    /// the owed instance is queued for [`reap`](Self::reap) to free off-thread.
    pub fn on_complete(&self, instance: u64) -> bool {
        let owed = {
            let mut m = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let Some(st) = m.get_mut(&instance) else {
                return false; // unknown/closed — benign, no deferred free owed
            };
            if st.in_flight > 0 {
                st.in_flight -= 1;
            }
            if st.in_flight == 0 && st.close_pending {
                m.remove(&instance);
                true
            } else {
                false
            }
        };
        if owed {
            self.needs_free
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(instance);
        }
        owed
    }

    /// Request close of an instance (B6). If no carry is in flight → [`FreeNow`] and
    /// the entry is dropped (the caller frees now). Otherwise → [`Deferred`]: mark
    /// close-pending and keep the entry so the completing decrement queues the free.
    ///
    /// [`FreeNow`]: CloseAction::FreeNow
    /// [`Deferred`]: CloseAction::Deferred
    pub fn request_close(&self, instance: u64) -> CloseAction {
        let mut m = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match m.get_mut(&instance) {
            Some(st) if st.in_flight > 0 => {
                st.close_pending = true;
                CloseAction::Deferred
            }
            _ => {
                m.remove(&instance);
                CloseAction::FreeNow
            }
        }
    }

    /// Perform any deferred frees queued by [`on_complete`](Self::on_complete),
    /// off the callback thread — call `free_fn(instance)` (e.g. `pie_frame_close`)
    /// for each. Runtime threads call this at control-plane entry points (bind /
    /// enqueue / close) so a drained-during-in-flight close's free lands promptly.
    ///
    /// BOUNDED delayed-free (non-UAF, non-leak): a free queued by the completing
    /// carry only lands at the NEXT control-plane `reap`. If the runtime goes
    /// quiescent right after the last instance's close+drain (no further bind/
    /// enqueue/close), that instance's device frame + pinned regions stay allocated
    /// until the next op — or process exit. It's bounded (freed at the next op /
    /// shutdown) and never a UAF, so this is fine for the provisional path (the real
    /// §5.2 drain-before-free is the C++ carrier's `cudaStreamSynchronize` +
    /// charlie's executor-cut close). If prompt reclamation ever matters, call
    /// `reap` from an idle/shutdown hook too.
    pub fn reap<F: FnMut(u64)>(&self, mut free_fn: F) {
        let drained: Vec<u64> = {
            let mut q = self.needs_free.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *q)
        };
        for instance in drained {
            free_fn(instance);
        }
    }

    /// In-flight carry count for `instance` (0 if unknown). Introspection / tests.
    pub fn in_flight(&self, instance: u64) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&instance)
            .map(|s| s.in_flight)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_marshals_round_trip() {
        let d = CarryDescriptor::new(7, 2, 0xDEAD_BEEF, 0xCAFE_F00D);
        let bytes = d.to_bytes();
        let back = CarryDescriptor::from_bytes(&bytes).expect("valid round-trip");
        assert_eq!(d, back);
        assert_eq!(back.version, CARRY_DESCRIPTOR_VERSION);
        assert_eq!(back.size as usize, core::mem::size_of::<CarryDescriptor>());
        assert_eq!(back.instance, 7);
        assert_eq!(back.word_index, 2);
        assert_eq!(back.done, 0xDEAD_BEEF);
        assert_eq!(back.user_data, 0xCAFE_F00D);
    }

    #[test]
    fn descriptor_loud_rejects_unknown_version() {
        let mut d = CarryDescriptor::new(1, 0, 0, 0);
        d.version = CARRY_DESCRIPTOR_VERSION + 1;
        let err = d.validate().expect_err("wrong version must reject");
        assert_eq!(err.observed_version, CARRY_DESCRIPTOR_VERSION + 1);
        assert!(CarryDescriptor::from_bytes(&d.to_bytes()).is_err());
    }

    #[test]
    fn descriptor_loud_rejects_wrong_size() {
        let mut d = CarryDescriptor::new(1, 0, 0, 0);
        d.size = 4; // a stale/other layout
        assert!(d.validate().is_err());
    }

    #[test]
    fn descriptor_rejects_short_blob() {
        assert!(matches!(CarryDescriptor::from_bytes(&[0u8; 3]), Err(None)));
    }

    #[test]
    fn push_carry_request_fills_parallel_cols_and_sets_version_once() {
        let mut req = ForwardRequest::default();
        assert!(req.carry_user_ptr.is_empty());
        push_carry_request(&mut req, 0xAA, 0, 7);
        push_carry_request(&mut req, 0xBB, 2, 9);
        // Version guard set exactly ONCE (executor validates carry_abi_version[0]).
        assert_eq!(req.carry_abi_version, vec![CARRY_DESCRIPTOR_VERSION]);
        // Parallel SoA cols, R-aligned across all four arrays.
        assert_eq!(req.carry_user_ptr, vec![0xAA, 0xBB]);
        assert_eq!(req.carry_word_index, vec![0, 2]);
        assert_eq!(req.carry_instance, vec![7, 9]);
    }

    #[test]
    fn carry_cols_empty_by_default_is_dormant() {
        // The behavior-neutral state: no push ⇒ all cols empty ⇒ charlie's a2
        // R-carry loop skips (`if !carry_user_ptr.is_empty()`) ⇒ zero behavior change.
        let req = ForwardRequest::default();
        assert!(req.carry_abi_version.is_empty());
        assert!(req.carry_user_ptr.is_empty());
        assert!(req.carry_word_index.is_empty());
        assert!(req.carry_instance.is_empty());
    }

    /// THE mandated close-during-in-flight test (guru): a close requested while a
    /// carry is in flight must DEFER the frame region-free until the carry
    /// completes — never free a frame a pending carry still writes. And the free is
    /// callback-safe: the completion callback only QUEUES it; `reap` performs it.
    #[test]
    fn close_during_in_flight_defers_free_until_drain() {
        let t = InFlightTracker::new();
        t.on_bind(5);
        t.on_enqueue(5); // one carry in flight
        assert_eq!(t.in_flight(5), 1);

        // Close while in-flight > 0 → DEFERRED, not FreeNow. No free happens yet.
        assert_eq!(t.request_close(5), CloseAction::Deferred);
        let mut freed: Vec<u64> = Vec::new();
        t.reap(|id| freed.push(id));
        assert!(freed.is_empty(), "no free while a carry is still in flight");
        assert_eq!(t.in_flight(5), 1, "entry retained until drain");

        // The completing carry drains the last in-flight → owes the deferred free,
        // but does NOT free itself (callback-safe): it only queues it.
        assert!(t.on_complete(5), "last completion queues the deferred free");
        assert_eq!(t.in_flight(5), 0);

        // reap (off the callback thread) performs the actual free exactly once.
        t.reap(|id| freed.push(id));
        assert_eq!(freed, vec![5], "reap frees the drained instance once");
        // Entry dropped; a second complete is benign + queues nothing.
        assert!(!t.on_complete(5));
        freed.clear();
        t.reap(|id| freed.push(id));
        assert!(freed.is_empty());
    }

    #[test]
    fn close_with_no_in_flight_frees_now() {
        let t = InFlightTracker::new();
        t.on_bind(9);
        assert_eq!(t.request_close(9), CloseAction::FreeNow);
        // Unknown instance close is also FreeNow (benign).
        assert_eq!(t.request_close(999), CloseAction::FreeNow);
    }

    #[test]
    fn multiple_in_flight_defers_until_all_drain() {
        let t = InFlightTracker::new();
        t.on_bind(3);
        t.on_enqueue(3);
        t.on_enqueue(3); // two in flight
        assert_eq!(t.request_close(3), CloseAction::Deferred);
        assert!(!t.on_complete(3), "one still in flight → no free yet");
        assert!(t.on_complete(3), "last drains → deferred free queued");
        let mut freed: Vec<u64> = Vec::new();
        t.reap(|id| freed.push(id));
        assert_eq!(freed, vec![3]);
    }
}
