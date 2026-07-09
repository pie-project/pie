//! # X2 — CUDA control plane (Runtime–Driver Boundary, the real-device dual of X1)
//!
//! X1 proved the `register_program → bind_instance → enqueue → completion` shape
//! on [`MockControlPlane`](super::control::MockControlPlane) with host allocations
//! standing in for the device frame and the pinned mirror/word regions. X2 backs
//! the same [`ControlPlane`] interface with the real thing — a device frame
//! (`cudaMalloc`), a pinned host mirror, and pinned ring-index words — via the
//! `pie_frame_*` `extern "C"` surface in the CUDA driver ([`frame_carrier.cpp`]).
//!
//! ## Shape (builds on X1, no new trait)
//!
//! The device work lives behind the `pie_frame_register / bind / close` FFI in
//! the CUDA driver, declared here exactly like the `pie_pinned_*` / `pie_device_*`
//! tensor-I/O fast path — a direct call into `pie_driver_cuda_lib` that bypasses
//! the IPC/driver channel (B1/B2). [`bind_instance`](ControlPlane::bind_instance)
//! returns the device/pinned allocations as [`FrameAddresses`] plus the bind-fixed
//! [`WakeSlots`] (B15); [`enqueue`](ControlPlane::enqueue) re-backs the run-ahead
//! pacing [`Completion`] on the instance's fixed pacing slot + `word[0]`, targeting
//! the fire's monotonic sequence number. The driver advances `word[0]` and wakes
//! the pacing slot directly from its instance table on commit (boundary.md Phase 1);
//! steady-state fires allocate nothing.
//!
//! [`frame_carrier.cpp`]: driver/cuda/src/sampling_ir/frame_carrier.cpp
//!
//! Gated on `all(ptir, driver-cuda)`: off either flag the `pie_frame_*` symbols
//! are never referenced (no undefined-symbol link error) and only the mock exists.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};

use super::control::{
    BoundInstance, Completion, ControlPlane, EnqueueBatch, FrameAddresses, InstanceId, ProgramId,
    WakeSlots,
};
use crate::driver::waker::{ChannelWakers, WakerTable};

unsafe extern "C" {
    /// B4 — register a trace with the CUDA driver; returns a 1-based program
    /// handle (`0` = rejected). Implemented in `frame_carrier.cpp`.
    fn pie_frame_register(trace: *const u8, trace_len: usize) -> u64;
    /// B4/B5 — bind an instance; writes the device frame + pinned mirror + pinned
    /// word bases. Returns the 1-based instance id (`0` = unknown program).
    fn pie_frame_bind(
        program: u64,
        out_frame_base: *mut u64,
        out_mirror_base: *mut u64,
        out_word_base: *mut u64,
    ) -> u64;
    /// B6 — release an instance's frame/mirror/word regions (fail-loud on unknown).
    fn pie_frame_close(instance: u64);
}

/// Per-instance bind-fixed state (B15): the wake slots the driver wakes directly
/// + the shared pacing word[0] the [`Completion`] parks on + the monotonic fire
/// sequence. Allocated at bind, freed at close.
struct InstanceState {
    wakes: WakeSlots,
    /// The pacing counter word (committed fire count). The driver advances it on
    /// commit; the parked [`Completion`] polls it. Held as an `Arc` shared with
    /// each fire's completion.
    pacing_word: Arc<AtomicU64>,
    /// Next fire's monotonic target (1, 2, 3, …).
    fire_seq: u64,
}

/// **The CUDA control plane (X2).** The embedded-driver [`ControlPlane`] backed by
/// real device frames + pinned mirrors/words. Direct calls only — like
/// [`MockControlPlane`](super::control::MockControlPlane) it holds no channel and
/// no response slot; completion rides the X0 waker (the pacing slot the driver
/// wakes when `word[0]` advances).
pub struct CudaControlPlane {
    table: &'static WakerTable,
    /// Bind-fixed per-instance state (B15), keyed by instance id.
    instances: Mutex<HashMap<InstanceId, InstanceState>>,
}

impl Default for CudaControlPlane {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaControlPlane {
    pub fn new() -> CudaControlPlane {
        CudaControlPlane {
            table: WakerTable::global(),
            instances: Mutex::new(HashMap::new()),
        }
    }
}

impl ControlPlane for CudaControlPlane {
    fn register_program(&self, trace: &[u8]) -> Result<ProgramId> {
        // SAFETY: `trace` is a valid slice for the duration of the call; the driver
        // only reads it to compute the frame layout.
        let id = unsafe { pie_frame_register(trace.as_ptr(), trace.len()) };
        if id == 0 {
            return Err(anyhow!("register_program: CUDA driver rejected the trace"));
        }
        Ok(id)
    }

    fn bind_instance(&self, program: ProgramId, _bindings: &[u8]) -> Result<BoundInstance> {
        let (mut frame_base, mut mirror_base, mut word_base) = (0u64, 0u64, 0u64);
        // SAFETY: the three out-params are valid, distinct locals; the driver writes
        // each base once and returns 0 (leaving them untouched) on unknown program.
        let id = unsafe {
            pie_frame_bind(program, &mut frame_base, &mut mirror_base, &mut word_base)
        };
        if id == 0 {
            return Err(anyhow!("bind_instance: unknown program {program}"));
        }
        // Allocate the bind-fixed wake slots (B15): the pacing slot the driver
        // wakes when `word[0]` advances, plus one reader/writer pair per host-
        // visible channel. PROVISIONAL single channel (guru's frame-layout
        // reconcile fixes the real channel count); the slots are freed at close.
        let wakes = WakeSlots {
            pacing: self.table.alloc(),
            channels: vec![ChannelWakers::alloc(self.table)],
        };
        self.instances
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                id,
                InstanceState {
                    wakes: wakes.clone(),
                    pacing_word: Arc::new(AtomicU64::new(0)),
                    fire_seq: 0,
                },
            );
        Ok(BoundInstance {
            id,
            addresses: FrameAddresses { frame_base, mirror_base, word_base },
            wakes,
        })
    }

    fn close_instance(&self, id: InstanceId) -> Result<()> {
        // Free the bind-fixed wake slots (B15: freed only at close) before the
        // device frame regions go. A residual driver wake targeting a freed slot
        // is a harmless generation no-op (X0 B10).
        let state = self
            .instances
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id);
        if let Some(state) = state {
            self.table.free(state.wakes.pacing);
            for ch in &state.wakes.channels {
                ch.free(self.table);
            }
        }
        // SAFETY: `id` names a live instance; the driver fails loud on unknown/closed.
        unsafe { pie_frame_close(id) };
        Ok(())
    }

    fn enqueue(&self, batch: EnqueueBatch) -> Result<Completion> {
        let _ = &batch.descriptor;
        let mut instances = self.instances.lock().unwrap_or_else(|e| e.into_inner());
        let Some(state) = instances.get_mut(&batch.instance) else {
            return Err(anyhow!("enqueue: unknown instance {}", batch.instance));
        };
        // Re-back on the instance's FIXED pacing slot + shared pacing word[0]
        // (boundary.md Phase 1): each fire targets the next monotonic sequence
        // number; the driver advances the pacing word 1,2,3,… as fires commit and
        // wakes the pacing slot directly from its instance table. Steady state
        // allocates NO waker slot (the pacing slot is bind-fixed).
        state.fire_seq += 1;
        let target = state.fire_seq;
        let pacing = state.wakes.pacing;
        let word = Arc::clone(&state.pacing_word);
        Ok(Completion::parked_pacing(self.table, pacing, word, target))
    }
}
