//! Runtime-owned launch and direct-driver descriptors.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pie_driver_abi::{
    PIE_DRIVER_ABI_VERSION, PIE_MEMORY_DOMAIN_HOST_PINNED, PieBytes, PieChannelDesc,
    PieChannelEndpointBinding, PieChannelValueDesc, PieChannelValueDescSlice, PieCompletion,
    PieInstanceBinding, PieInstanceDesc, PieKvCopyDesc, PieKvMoveCell, PieKvMoveCellSlice,
    PieMaskWordsDesc, PieMemoryDomain, PiePoolRange, PiePoolRangeSlice, PiePoolResizeDesc,
    PieProgramDesc, PieStateCopyDesc, PieStateCopyRange, PieStateCopyRangeSlice, PieTerminalCell,
    PieTerminalCellPtrSlice, PieU8Slice, PieU32Slice, PieU64Slice,
};

use crate::ptir::PtirChannelValue;
use pie_grammar::brle::RunMask;

pub type ProgramId = u64;
pub type InstanceId = u64;

#[derive(Default, Debug, Clone, PartialEq)]
pub struct LaunchPlan {
    pub token_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub kv_page_indices: Vec<u32>,
    pub kv_page_indptr: Vec<u32>,
    pub kv_last_page_lens: Vec<u32>,
    pub qo_indptr: Vec<u32>,
    pub rs_slot_ids: Vec<u32>,
    pub rs_slot_flags: Vec<u8>,
    pub rs_fold_lens: Vec<u32>,
    pub rs_buffer_slot_ids: Vec<u32>,
    pub rs_buffer_slot_indptr: Vec<u32>,
    pub masks: Vec<RunMask>,
    pub mask_indptr: Vec<u32>,
    pub sampling_indices: Vec<u32>,
    pub sampling_indptr: Vec<u32>,
    pub context_ids: Vec<u64>,
    pub single_token_mode: bool,
    pub has_user_mask: bool,
    pub image_indptr: Vec<u32>,
    pub image_grids: Vec<u32>,
    pub image_anchor_positions: Vec<u32>,
    pub image_pixels: Vec<u8>,
    pub image_pixel_indptr: Vec<u32>,
    pub image_mrope_positions: Vec<u32>,
    pub image_mrope_indptr: Vec<u32>,
    pub image_patch_positions: Vec<u32>,
    pub image_anchor_rows: Vec<u32>,
    pub audio_features: Vec<u8>,
    pub audio_feature_indptr: Vec<u32>,
    pub audio_anchor_rows: Vec<u32>,
    pub audio_indptr: Vec<u32>,
    pub kv_len: Vec<u32>,
    pub kv_len_device: Vec<u64>,
    /// This fire's WorkingSet page translation: entry `i` = the PHYSICAL KV
    /// page id backing WorkingSet-relative index `i` (committed mapping
    /// overlaid with the prepared write targets). The driver maps channel-
    /// resolved `Pages`/`WSlot` references through it; empty = no
    /// WorkingSet-relative geometry in this fire.
    pub kv_translation: Vec<u32>,
}

pub const RS_FLAG_RESET: u8 = 1;
pub const RS_FLAG_FOLD: u8 = 2;

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct ProgramRegistration {
    pub program_hash: u64,
    pub canonical_bytes: Vec<u8>,
    pub sidecar_bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelRegistrationPlan {
    pub driver_id: usize,
    pub channel_id: u64,
    pub shape: Vec<u32>,
    pub dtype: u8,
    pub host_role: u8,
    pub seeded: bool,
    pub extern_dir: u8,
    pub capacity: u32,
    pub reader_wait_id: u64,
    pub writer_wait_id: u64,
    pub extern_name: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredChannel {
    pub driver_id: usize,
    pub binding: PieChannelEndpointBinding,
    pub reader_wait_id: u64,
    pub writer_wait_id: u64,
}

#[derive(Debug)]
pub struct ChannelEndpoint {
    registered: RegisteredChannel,
    closed: AtomicBool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelWaitError {
    Poisoned(u64),
    Closed,
}

impl std::fmt::Display for ChannelWaitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Poisoned(epoch) => write!(f, "channel is poisoned at epoch {epoch}"),
            Self::Closed => write!(f, "channel is closed"),
        }
    }
}

impl std::error::Error for ChannelWaitError {}

fn load_channel_word(word_base: u64, index: u32) -> u64 {
    unsafe { (&*((word_base as *const AtomicU64).add(index as usize))).load(Ordering::Acquire) }
}

impl ChannelEndpoint {
    pub fn new(registered: RegisteredChannel) -> Self {
        Self {
            registered,
            closed: AtomicBool::new(false),
        }
    }

    pub fn registered(&self) -> &RegisteredChannel {
        &self.registered
    }

    pub async fn wait_for_reader_change(&self, observed_tail: u64) -> Result<(), ChannelWaitError> {
        self.wait_for_word_change(
            self.registered.reader_wait_id,
            self.registered.binding.tail_word_index,
            observed_tail,
        )
        .await
    }

    pub async fn wait_for_writer_change(&self, observed_head: u64) -> Result<(), ChannelWaitError> {
        self.wait_for_word_change(
            self.registered.writer_wait_id,
            self.registered.binding.head_word_index,
            observed_head,
        )
        .await
    }

    async fn wait_for_word_change(
        &self,
        wait_id: u64,
        word_index: u32,
        observed: u64,
    ) -> Result<(), ChannelWaitError> {
        let binding = self.registered.binding;
        pie_waker::WaitFuture::new(pie_waker::WakerTable::global(), wait_id, move || {
            let poison = load_channel_word(binding.word_base, binding.poison_word_index);
            if poison != 0 {
                return pie_waker::Readiness::Ready(Err(ChannelWaitError::Poisoned(poison)));
            }
            if load_channel_word(binding.word_base, binding.closed_word_index) != 0 {
                return pie_waker::Readiness::Ready(Err(ChannelWaitError::Closed));
            }
            let current = load_channel_word(binding.word_base, word_index);
            if current > observed {
                pie_waker::Readiness::Ready(Ok(()))
            } else {
                pie_waker::Readiness::Pending {
                    observed_epoch: current,
                }
            }
        })
        .await
    }

    fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let table = pie_waker::WakerTable::global();
        let wait_ids = [
            self.registered.reader_wait_id,
            self.registered.writer_wait_id,
        ];
        if let Ok(handle) = crate::driver::scheduler_handle(self.registered.driver_id) {
            if let Err(error) = handle.close_channel(self.registered.binding.channel_id) {
                tracing::warn!(
                    channel_id = self.registered.binding.channel_id,
                    ?error,
                    "ordered channel close failed"
                );
            }
        }
        table.sweep(&wait_ids);
        for wait_id in wait_ids {
            table.deregister(wait_id);
            table.free(wait_id);
        }
    }
}

impl Drop for ChannelEndpoint {
    fn drop(&mut self) {
        self.close();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceBindingPlan {
    pub driver_id: usize,
    pub program_id: ProgramId,
    pub requested_instance_id: InstanceId,
    pub pacing_wait_id: u64,
    pub channel_ids: Vec<u64>,
    pub seed_values: Vec<PtirChannelValue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedInstanceBinding {
    pub instance_id: InstanceId,
}

#[derive(Debug)]
pub(crate) struct BoundWaitSlots {
    pacing_wait_id: u64,
    completion_wait_ids: Mutex<Vec<u64>>,
    close_requested: AtomicBool,
    freed: AtomicBool,
    active_leases: AtomicUsize,
}

impl BoundWaitSlots {
    fn new(pacing_wait_id: u64) -> Self {
        Self {
            pacing_wait_id,
            completion_wait_ids: Mutex::new(Vec::new()),
            close_requested: AtomicBool::new(false),
            freed: AtomicBool::new(false),
            active_leases: AtomicUsize::new(0),
        }
    }

    fn acquire_completion_lease(
        this: &Arc<Self>,
        completion_wait_id: u64,
    ) -> Arc<dyn crate::driver::completion::CompletionLease> {
        this.completion_wait_ids
            .lock()
            .unwrap()
            .push(completion_wait_id);
        this.active_leases.fetch_add(1, Ordering::AcqRel);
        Arc::new(BoundWaitLease {
            slots: Arc::clone(this),
            completion_wait_id,
        })
    }

    pub(crate) fn close(&self) {
        if !self.close_requested.swap(true, Ordering::AcqRel) {
            pie_waker::WakerTable::global().sweep(&self.wait_ids());
            let completion_wait_ids = self.completion_wait_ids.lock().unwrap().clone();
            pie_waker::WakerTable::global().sweep(&completion_wait_ids);
            self.maybe_finalize();
        }
    }

    fn release_completion_lease_for(&self, completion_wait_id: u64) {
        self.completion_wait_ids
            .lock()
            .unwrap()
            .retain(|&id| id != completion_wait_id);
        let prev = self.active_leases.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prev > 0);
        if prev == 1 {
            self.maybe_finalize();
        }
    }

    fn maybe_finalize(&self) {
        if !self.close_requested.load(Ordering::Acquire)
            || self.active_leases.load(Ordering::Acquire) != 0
            || self.freed.swap(true, Ordering::AcqRel)
        {
            return;
        }
        let table = pie_waker::WakerTable::global();
        for id in self.wait_ids() {
            table.deregister(id);
            table.free(id);
        }
    }

    fn wait_ids(&self) -> Vec<u64> {
        vec![self.pacing_wait_id]
    }

    fn is_closed(&self) -> bool {
        self.close_requested.load(Ordering::Acquire)
    }
}

impl crate::driver::completion::CompletionLease for BoundWaitLease {
    fn is_closed(&self) -> bool {
        self.slots.is_closed()
    }
}

#[derive(Debug)]
struct BoundWaitLease {
    slots: Arc<BoundWaitSlots>,
    completion_wait_id: u64,
}

impl Drop for BoundWaitLease {
    fn drop(&mut self) {
        self.slots
            .release_completion_lease_for(self.completion_wait_id);
    }
}

#[derive(Debug)]
pub struct BoundInstance {
    pub driver_id: usize,
    pub program_id: ProgramId,
    pub instance_id: InstanceId,
    pub binding: OwnedInstanceBinding,
    pub pacing_wait_id: u64,
    wait_slots: Arc<BoundWaitSlots>,
}

impl BoundInstance {
    pub fn new(
        driver_id: usize,
        program_id: ProgramId,
        binding: PieInstanceBinding,
        pacing_wait_id: u64,
    ) -> Self {
        let wait_slots = Arc::new(BoundWaitSlots::new(pacing_wait_id));
        Self {
            driver_id,
            program_id,
            instance_id: binding.instance_id,
            binding: OwnedInstanceBinding {
                instance_id: binding.instance_id,
            },
            pacing_wait_id,
            wait_slots,
        }
    }

    pub fn reserve_completion(&self) -> crate::driver::completion::InstanceCompletion {
        let wait_id = pie_waker::WakerTable::global().alloc();
        crate::driver::completion::InstanceCompletion::with_guard(
            wait_id,
            0,
            BoundWaitSlots::acquire_completion_lease(&self.wait_slots, wait_id),
        )
    }

    pub(crate) fn wait_slots(&self) -> Arc<BoundWaitSlots> {
        Arc::clone(&self.wait_slots)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LaunchSubmission {
    pub plan: LaunchPlan,
    pub instance_ids: Vec<u64>,
    pub terminal_cells: Vec<*mut PieTerminalCell>,
    /// Flattened per-instance WorkingSet page translations (see
    /// [`LaunchPlan::kv_translation`]) + their CSR partition.
    pub kv_translation: Vec<u32>,
    pub kv_translation_indptr: Vec<u32>,
    /// Program → wire-request attribution CSR (`instance_ids.len() + 1`
    /// entries): program `p` owns wire request rows
    /// `[row_indptr[p], row_indptr[p+1])`. Batched fires contribute one row
    /// each (a device-geometry fire's row is an empty placeholder the driver
    /// replaces with channel-resolved geometry); a prebuilt solo plan owns
    /// every row it shipped.
    pub program_row_indptr: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCopyPlan {
    pub src_domain: PieMemoryDomain,
    pub src_device_ordinal: u32,
    pub dst_domain: PieMemoryDomain,
    pub dst_device_ordinal: u32,
    pub src_page_ids: Vec<u32>,
    pub dst_page_ids: Vec<u32>,
    pub cells: Vec<PieKvMoveCell>,
}

impl Default for KvCopyPlan {
    fn default() -> Self {
        Self {
            src_domain: PIE_MEMORY_DOMAIN_HOST_PINNED,
            src_device_ordinal: 0,
            dst_domain: PIE_MEMORY_DOMAIN_HOST_PINNED,
            dst_device_ordinal: 0,
            src_page_ids: Vec::new(),
            dst_page_ids: Vec::new(),
            cells: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StateCopyPlan {
    pub slot_ranges: Vec<PieStateCopyRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PoolResizePlan {
    pub pool_id: u64,
    pub target_pages: u64,
    pub map_ranges: Vec<PiePoolRange>,
    pub unmap_ranges: Vec<PiePoolRange>,
}

fn bytes_slice(bytes: &[u8]) -> PieBytes {
    PieBytes {
        ptr: bytes.as_ptr(),
        len: bytes.len(),
    }
}
fn u8_slice(slice: &[u8]) -> PieU8Slice {
    PieU8Slice {
        ptr: slice.as_ptr(),
        len: slice.len(),
    }
}
fn u32_slice(slice: &[u32]) -> PieU32Slice {
    PieU32Slice {
        ptr: slice.as_ptr(),
        len: slice.len(),
    }
}
fn u64_slice(slice: &[u64]) -> PieU64Slice {
    PieU64Slice {
        ptr: slice.as_ptr(),
        len: slice.len(),
    }
}

#[derive(Default)]
struct MaskWordsStorage {
    request_indptr: Vec<u32>,
    word_indptr: Vec<u32>,
    words: Vec<u32>,
}

impl MaskWordsStorage {
    fn from_plan(plan: &LaunchPlan) -> Self {
        let request_count = plan.qo_indptr.len().checked_sub(1).unwrap_or_default() as u32;
        let request_indptr = if plan.mask_indptr.is_empty() {
            let mut indptr = Vec::with_capacity(request_count as usize + 1);
            indptr.resize(request_count as usize + 1, 0);
            indptr
        } else {
            plan.mask_indptr.clone()
        };

        let mut word_indptr = Vec::with_capacity(plan.masks.len() + 1);
        let mut words = Vec::new();
        word_indptr.push(0);
        for mask in &plan.masks {
            let bits = mask.to_vec();
            for chunk in bits.chunks(32) {
                let mut word = 0u32;
                for (bit, value) in chunk.iter().enumerate() {
                    if *value {
                        word |= 1u32 << bit;
                    }
                }
                words.push(word);
            }
            word_indptr.push(words.len() as u32);
        }
        Self {
            request_indptr,
            word_indptr,
            words,
        }
    }

    fn as_desc(&self) -> PieMaskWordsDesc {
        PieMaskWordsDesc {
            request_indptr: u32_slice(&self.request_indptr),
            word_indptr: u32_slice(&self.word_indptr),
            words: u32_slice(&self.words),
        }
    }
}

pub struct ProgramDescBorrow<'a> {
    _bytes: &'a [u8],
    _sidecar: &'a [u8],
    raw: PieProgramDesc,
}
impl<'a> ProgramDescBorrow<'a> {
    pub fn new(program: &'a ProgramRegistration) -> Self {
        Self {
            _bytes: &program.canonical_bytes,
            _sidecar: &program.sidecar_bytes,
            raw: PieProgramDesc {
                abi_version: PIE_DRIVER_ABI_VERSION,
                reserved0: 0,
                program_hash: program.program_hash,
                canonical_bytes: bytes_slice(&program.canonical_bytes),
                sidecar_bytes: bytes_slice(&program.sidecar_bytes),
            },
        }
    }
    pub fn as_raw(&self) -> &PieProgramDesc {
        &self.raw
    }
}

pub struct ChannelDescBorrow<'a> {
    _shape: &'a [u32],
    _extern_name: &'a [u8],
    raw: PieChannelDesc,
}

impl<'a> ChannelDescBorrow<'a> {
    pub fn new(plan: &'a ChannelRegistrationPlan) -> Self {
        Self {
            _shape: &plan.shape,
            _extern_name: &plan.extern_name,
            raw: PieChannelDesc {
                abi_version: PIE_DRIVER_ABI_VERSION,
                reserved0: 0,
                channel_id: plan.channel_id,
                shape: u32_slice(&plan.shape),
                dtype: plan.dtype,
                host_role: plan.host_role,
                seeded: u8::from(plan.seeded),
                extern_dir: plan.extern_dir,
                capacity: plan.capacity,
                reserved1: 0,
                reader_wait_id: plan.reader_wait_id,
                writer_wait_id: plan.writer_wait_id,
                extern_name: bytes_slice(&plan.extern_name),
            },
        }
    }

    pub fn as_raw(&self) -> &PieChannelDesc {
        &self.raw
    }
}

pub struct InstanceDescBorrow<'a> {
    _channel_ids: &'a [u64],
    _seed_values: Vec<PieChannelValueDesc>,
    raw: PieInstanceDesc,
}
impl<'a> InstanceDescBorrow<'a> {
    pub fn new(plan: &'a InstanceBindingPlan) -> Self {
        let seed_values: Vec<PieChannelValueDesc> = plan
            .seed_values
            .iter()
            .map(|value| PieChannelValueDesc {
                channel_id: value.channel,
                bytes: bytes_slice(&value.bytes),
            })
            .collect();
        let raw = PieInstanceDesc {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            program_id: plan.program_id,
            requested_instance_id: plan.requested_instance_id,
            pacing_wait_id: plan.pacing_wait_id,
            channel_ids: u64_slice(&plan.channel_ids),
            seed_values: PieChannelValueDescSlice {
                ptr: seed_values.as_ptr(),
                len: seed_values.len(),
            },
        };
        Self {
            _channel_ids: &plan.channel_ids,
            _seed_values: seed_values,
            raw,
        }
    }
    pub fn as_raw(&self) -> &PieInstanceDesc {
        &self.raw
    }
}

pub struct LaunchDescBorrow<'a> {
    _masks: MaskWordsStorage,
    raw: pie_driver_abi::PieLaunchDesc,
    _plan: &'a LaunchPlan,
}
impl<'a> LaunchDescBorrow<'a> {
    pub fn from_submission(submission: &'a LaunchSubmission) -> Self {
        let plan = &submission.plan;
        let masks = MaskWordsStorage::from_plan(plan);
        let raw = pie_driver_abi::PieLaunchDesc {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            instance_ids: u64_slice(&submission.instance_ids),
            terminal_cells: PieTerminalCellPtrSlice {
                ptr: submission.terminal_cells.as_ptr(),
                len: submission.terminal_cells.len(),
            },
            token_ids: u32_slice(&plan.token_ids),
            position_ids: u32_slice(&plan.position_ids),
            kv_page_indices: u32_slice(&plan.kv_page_indices),
            kv_page_indptr: u32_slice(&plan.kv_page_indptr),
            kv_last_page_lens: u32_slice(&plan.kv_last_page_lens),
            qo_indptr: u32_slice(&plan.qo_indptr),
            rs_slot_ids: u32_slice(&plan.rs_slot_ids),
            rs_slot_flags: u8_slice(&plan.rs_slot_flags),
            rs_fold_lens: u32_slice(&plan.rs_fold_lens),
            rs_buffer_slot_ids: u32_slice(&plan.rs_buffer_slot_ids),
            rs_buffer_slot_indptr: u32_slice(&plan.rs_buffer_slot_indptr),
            masks: masks.as_desc(),
            sampling_indices: u32_slice(&plan.sampling_indices),
            sampling_indptr: u32_slice(&plan.sampling_indptr),
            context_ids: u64_slice(&plan.context_ids),
            single_token_mode: u8::from(plan.single_token_mode),
            has_user_mask: u8::from(plan.has_user_mask),
            reserved_flags: [0; 6],
            image_indptr: u32_slice(&plan.image_indptr),
            image_grids: u32_slice(&plan.image_grids),
            image_anchor_positions: u32_slice(&plan.image_anchor_positions),
            image_pixels: bytes_slice(&plan.image_pixels),
            image_pixel_indptr: u32_slice(&plan.image_pixel_indptr),
            image_mrope_positions: u32_slice(&plan.image_mrope_positions),
            image_mrope_indptr: u32_slice(&plan.image_mrope_indptr),
            image_patch_positions: u32_slice(&plan.image_patch_positions),
            image_anchor_rows: u32_slice(&plan.image_anchor_rows),
            audio_features: bytes_slice(&plan.audio_features),
            audio_feature_indptr: u32_slice(&plan.audio_feature_indptr),
            audio_anchor_rows: u32_slice(&plan.audio_anchor_rows),
            audio_indptr: u32_slice(&plan.audio_indptr),
            kv_len: u32_slice(&plan.kv_len),
            kv_len_device: u64_slice(&plan.kv_len_device),
            kv_translation: u32_slice(&submission.kv_translation),
            kv_translation_indptr: u32_slice(&submission.kv_translation_indptr),
            ptir_program_row_indptr: u32_slice(&submission.program_row_indptr),
        };
        Self {
            _masks: masks,
            raw,
            _plan: plan,
        }
    }
    pub fn as_raw(&self) -> &pie_driver_abi::PieLaunchDesc {
        &self.raw
    }
}

pub struct KvCopyDescBorrow<'a> {
    raw: PieKvCopyDesc,
    _plan: &'a KvCopyPlan,
}
impl<'a> KvCopyDescBorrow<'a> {
    pub fn new(plan: &'a KvCopyPlan) -> Self {
        let raw = PieKvCopyDesc {
            abi_version: PIE_DRIVER_ABI_VERSION,
            src_domain: plan.src_domain,
            src_device_ordinal: plan.src_device_ordinal,
            dst_domain: plan.dst_domain,
            dst_device_ordinal: plan.dst_device_ordinal,
            reserved0: 0,
            src_page_ids: u32_slice(&plan.src_page_ids),
            dst_page_ids: u32_slice(&plan.dst_page_ids),
            cells: PieKvMoveCellSlice {
                ptr: plan.cells.as_ptr(),
                len: plan.cells.len(),
            },
        };
        Self { raw, _plan: plan }
    }
    pub fn as_raw(&self) -> &PieKvCopyDesc {
        &self.raw
    }
}

pub struct StateCopyDescBorrow<'a> {
    raw: PieStateCopyDesc,
    _plan: &'a StateCopyPlan,
}
impl<'a> StateCopyDescBorrow<'a> {
    pub fn new(plan: &'a StateCopyPlan) -> Self {
        Self {
            raw: PieStateCopyDesc {
                abi_version: PIE_DRIVER_ABI_VERSION,
                reserved0: 0,
                slot_ranges: PieStateCopyRangeSlice {
                    ptr: plan.slot_ranges.as_ptr(),
                    len: plan.slot_ranges.len(),
                },
            },
            _plan: plan,
        }
    }
    pub fn as_raw(&self) -> &PieStateCopyDesc {
        &self.raw
    }
}

pub struct PoolResizeDescBorrow<'a> {
    raw: PiePoolResizeDesc,
    _plan: &'a PoolResizePlan,
}
impl<'a> PoolResizeDescBorrow<'a> {
    pub fn new(plan: &'a PoolResizePlan) -> Self {
        Self {
            raw: PiePoolResizeDesc {
                abi_version: PIE_DRIVER_ABI_VERSION,
                reserved0: 0,
                pool_id: plan.pool_id,
                target_pages: plan.target_pages,
                map_ranges: PiePoolRangeSlice {
                    ptr: plan.map_ranges.as_ptr(),
                    len: plan.map_ranges.len(),
                },
                unmap_ranges: PiePoolRangeSlice {
                    ptr: plan.unmap_ranges.as_ptr(),
                    len: plan.unmap_ranges.len(),
                },
            },
            _plan: plan,
        }
    }
    pub fn as_raw(&self) -> &PiePoolResizeDesc {
        &self.raw
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerCompletion {
    pub completion: PieCompletion,
    pub target_epoch: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_endpoint() -> (ChannelEndpoint, Box<[u8]>, Box<[AtomicU64]>, u64, u64) {
        let mirror = vec![0; 8].into_boxed_slice();
        let words = (0..4)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let table = pie_waker::WakerTable::global();
        let reader_wait_id = table.alloc();
        let writer_wait_id = table.alloc();
        let endpoint = ChannelEndpoint::new(RegisteredChannel {
            driver_id: usize::MAX,
            binding: PieChannelEndpointBinding {
                channel_id: 1,
                mirror_base: mirror.as_ptr() as u64,
                word_base: words.as_ptr() as u64,
                mirror_bytes: mirror.len() as u64,
                word_bytes: (words.len() * std::mem::size_of::<AtomicU64>()) as u64,
                cell_bytes: 4,
                capacity: 1,
                head_word_index: 0,
                tail_word_index: 1,
                poison_word_index: 2,
                closed_word_index: 3,
            },
            reader_wait_id,
            writer_wait_id,
        });
        (endpoint, mirror, words, reader_wait_id, writer_wait_id)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn channel_waits_register_then_recheck_reader_and_writer_words() {
        let (endpoint, _mirror, words, reader_wait_id, writer_wait_id) = test_endpoint();
        let reader = endpoint.wait_for_reader_change(0);
        let publish_reader = async {
            tokio::task::yield_now().await;
            words[1].store(1, Ordering::Release);
            let _ = pie_waker::WakerTable::global().publish(reader_wait_id, 1);
        };
        let (result, ()) = tokio::join!(reader, publish_reader);
        result.unwrap();

        let writer = endpoint.wait_for_writer_change(0);
        let publish_writer = async {
            tokio::task::yield_now().await;
            words[0].store(1, Ordering::Release);
            let _ = pie_waker::WakerTable::global().publish(writer_wait_id, 1);
        };
        let (result, ()) = tokio::join!(writer, publish_writer);
        result.unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn channel_wait_surfaces_poison_after_wakeup() {
        let (endpoint, _mirror, words, reader_wait_id, _writer_wait_id) = test_endpoint();
        let reader = endpoint.wait_for_reader_change(0);
        let poison = async {
            tokio::task::yield_now().await;
            words[2].store(7, Ordering::Release);
            let _ = pie_waker::WakerTable::global().publish(reader_wait_id, 7);
        };
        let (result, ()) = tokio::join!(reader, poison);
        assert_eq!(result, Err(ChannelWaitError::Poisoned(7)));
    }
}
