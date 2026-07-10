//! Runtime-owned launch and direct-driver descriptors.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use pie_driver_abi::{
    PIE_DRIVER_ABI_VERSION, PIE_MEMORY_DOMAIN_HOST_PINNED, PieBytes, PieChannelValueDesc,
    PieChannelValueDescSlice, PieChannelWait, PieChannelWaitSlice, PieCompletion,
    PieInstanceBinding, PieInstanceDesc, PieKvCopyDesc, PieKvMoveCell, PieKvMoveCellSlice,
    PieMaskWordsDesc, PieMemoryDomain, PiePoolRange, PiePoolRangeSlice, PiePoolResizeDesc,
    PieProgramDesc, PieStateCopyDesc, PieStateCopyRange, PieStateCopyRangeSlice, PieU8Slice,
    PieU32Slice, PieU64Slice,
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
pub struct InstanceBindingPlan {
    pub driver_id: usize,
    pub program_id: ProgramId,
    pub requested_instance_id: InstanceId,
    pub pacing_wait_id: u64,
    pub channel_waits: Vec<PieChannelWait>,
    pub channel_ids: Vec<u64>,
    pub seed_values: Vec<PtirChannelValue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedChannelBinding {
    pub channel_id: u64,
    pub cell_bytes: u32,
    pub capacity: u32,
    pub mirror_offset: u64,
    pub head_word_index: u32,
    pub tail_word_index: u32,
    pub poison_word_index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedInstanceBinding {
    pub instance_id: InstanceId,
    pub frame_base: u64,
    pub mirror_base: u64,
    pub word_base: u64,
    pub channel_count: u32,
    pub word_count: u32,
    pub frame_bytes: u64,
    pub mirror_bytes: u64,
    pub word_bytes: u64,
    pub channels: Vec<OwnedChannelBinding>,
}

#[derive(Debug)]
pub(crate) struct BoundWaitSlots {
    pacing_wait_id: u64,
    channel_waits: Vec<PieChannelWait>,
    close_requested: AtomicBool,
    freed: AtomicBool,
    active_leases: AtomicUsize,
    close_mu: Mutex<()>,
    close_cv: Condvar,
}

impl BoundWaitSlots {
    fn new(pacing_wait_id: u64, channel_waits: Vec<PieChannelWait>) -> Self {
        Self {
            pacing_wait_id,
            channel_waits,
            close_requested: AtomicBool::new(false),
            freed: AtomicBool::new(false),
            active_leases: AtomicUsize::new(0),
            close_mu: Mutex::new(()),
            close_cv: Condvar::new(),
        }
    }

    fn acquire_completion_lease(
        this: &Arc<Self>,
    ) -> Arc<dyn crate::driver::completion::CompletionLease> {
        this.active_leases.fetch_add(1, Ordering::AcqRel);
        Arc::new(BoundWaitLease {
            slots: Arc::clone(this),
        })
    }

    pub(crate) fn close(&self) {
        if !self.close_requested.swap(true, Ordering::AcqRel) {
            pie_waker::WakerTable::global().sweep(&self.wait_ids());
            self.maybe_finalize();
        }
    }

    pub(crate) fn close_and_wait(&self) {
        self.close();
        let mut guard = self.close_mu.lock().unwrap();
        while !self.freed.load(Ordering::Acquire) {
            guard = self.close_cv.wait(guard).unwrap();
        }
    }

    fn release_completion_lease(&self) {
        let prev = self.active_leases.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prev > 0);
        if prev == 1 {
            self.maybe_finalize();
        }
        self.close_cv.notify_all();
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
        self.close_cv.notify_all();
    }

    fn wait_ids(&self) -> Vec<u64> {
        let mut ids = Vec::with_capacity(1 + self.channel_waits.len() * 2);
        ids.push(self.pacing_wait_id);
        for waits in &self.channel_waits {
            ids.push(waits.reader_wait_id);
            ids.push(waits.writer_wait_id);
        }
        ids
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
}

impl Drop for BoundWaitLease {
    fn drop(&mut self) {
        self.slots.release_completion_lease();
    }
}

#[derive(Debug)]
pub struct BoundInstance {
    pub driver_id: usize,
    pub program_id: ProgramId,
    pub instance_id: InstanceId,
    pub binding: OwnedInstanceBinding,
    pub pacing_wait_id: u64,
    pub channel_waits: Vec<PieChannelWait>,
    next_target_epoch: AtomicU64,
    wait_slots: Arc<BoundWaitSlots>,
}

impl BoundInstance {
    pub fn new(
        driver_id: usize,
        program_id: ProgramId,
        binding: PieInstanceBinding,
        pacing_wait_id: u64,
        channel_waits: Vec<PieChannelWait>,
    ) -> Self {
        let channels = if binding.channels.ptr.is_null() || binding.channels.len == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(binding.channels.ptr, binding.channels.len) }
                .iter()
                .map(|channel| OwnedChannelBinding {
                    channel_id: channel.channel_id,
                    cell_bytes: channel.cell_bytes,
                    capacity: channel.capacity,
                    mirror_offset: channel.mirror_offset,
                    head_word_index: channel.head_word_index,
                    tail_word_index: channel.tail_word_index,
                    poison_word_index: channel.poison_word_index,
                })
                .collect()
        };
        let wait_slots = Arc::new(BoundWaitSlots::new(pacing_wait_id, channel_waits.clone()));
        Self {
            driver_id,
            program_id,
            instance_id: binding.instance_id,
            binding: OwnedInstanceBinding {
                instance_id: binding.instance_id,
                frame_base: binding.frame_base,
                mirror_base: binding.mirror_base,
                word_base: binding.word_base,
                channel_count: binding.channel_count,
                word_count: binding.word_count,
                frame_bytes: binding.frame_bytes,
                mirror_bytes: binding.mirror_bytes,
                word_bytes: binding.word_bytes,
                channels,
            },
            pacing_wait_id,
            channel_waits,
            next_target_epoch: AtomicU64::new(1),
            wait_slots,
        }
    }

    pub fn reserve_completion(&self) -> crate::driver::completion::InstanceCompletion {
        let target_epoch = self.next_target_epoch.fetch_add(1, Ordering::Relaxed);
        crate::driver::completion::InstanceCompletion::with_guard(
            self.pacing_wait_id,
            target_epoch,
            BoundWaitSlots::acquire_completion_lease(&self.wait_slots),
        )
    }

    pub fn channel_bindings(&self) -> &[OwnedChannelBinding] {
        &self.binding.channels
    }

    pub(crate) fn wait_slots(&self) -> Arc<BoundWaitSlots> {
        Arc::clone(&self.wait_slots)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LaunchSubmission {
    pub plan: LaunchPlan,
    pub instance_ids: Vec<u64>,
    pub host_put_values: Vec<PtirChannelValue>,
    pub host_put_indptr: Vec<u32>,
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

pub struct InstanceDescBorrow<'a> {
    _channel_waits: &'a [PieChannelWait],
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
            channel_waits: PieChannelWaitSlice {
                ptr: plan.channel_waits.as_ptr(),
                len: plan.channel_waits.len(),
            },
            channel_ids: u64_slice(&plan.channel_ids),
            seed_values: PieChannelValueDescSlice {
                ptr: seed_values.as_ptr(),
                len: seed_values.len(),
            },
        };
        Self {
            _channel_waits: &plan.channel_waits,
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
    _host_put_values: Vec<PieChannelValueDesc>,
    _masks: MaskWordsStorage,
    raw: pie_driver_abi::PieLaunchDesc,
    _plan: &'a LaunchPlan,
}
impl<'a> LaunchDescBorrow<'a> {
    pub fn from_submission(submission: &'a LaunchSubmission) -> Self {
        let plan = &submission.plan;
        let host_put_values_raw: Vec<PieChannelValueDesc> = submission
            .host_put_values
            .iter()
            .map(|value| PieChannelValueDesc {
                channel_id: value.channel,
                bytes: bytes_slice(&value.bytes),
            })
            .collect();
        let masks = MaskWordsStorage::from_plan(plan);
        let raw = pie_driver_abi::PieLaunchDesc {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            instance_ids: u64_slice(&submission.instance_ids),
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
            ptir_host_put_values: PieChannelValueDescSlice {
                ptr: host_put_values_raw.as_ptr(),
                len: host_put_values_raw.len(),
            },
            host_put_indptr: u32_slice(&submission.host_put_indptr),
            kv_len: u32_slice(&plan.kv_len),
            kv_len_device: u64_slice(&plan.kv_len_device),
        };
        Self {
            _host_put_values: host_put_values_raw,
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
