//! Borrowed ABI marshalling: `*DescBorrow` types that borrow runtime-owned
//! plans and lay them out as `pie_driver_abi` descriptors for the lifetime of
//! a backend call. Most fields are pointer/length views; temporary backing
//! storage is allocated only where the wire layout requires packing.

use pie_driver_abi::{
    PIE_DRIVER_ABI_VERSION, PieBytes, PieChannelDesc, PieChannelValueDesc,
    PieChannelValueDescSlice, PieEncodeDesc, PieInstanceDesc, PieKvCopyDesc, PieKvMoveCellSlice,
    PieMaskWordsDesc, PieMutBytes, PiePoolRangeSlice, PiePoolResizeDesc, PieProgramDesc,
    PieStateCopyDesc, PieStateCopyRangeSlice, PieTerminalCellPtrSlice, PieU8Slice, PieU32MutSlice,
    PieU32Slice, PieU64Slice,
};

use super::command::{
    ChannelRegistrationPlan, KvCopyPlan, LaunchPlan, PoolResizePlan, ProgramRegistration,
    StateCopyPlan,
};
use super::instance::InstanceBindingPlan;
use super::submission::{FrameSubmission, StepSubmission};

fn bytes_slice(bytes: &[u8]) -> PieBytes {
    PieBytes {
        ptr: bytes.as_ptr(),
        len: bytes.len(),
    }
}
fn mut_bytes_slice(bytes: &mut [u8]) -> PieMutBytes {
    PieMutBytes {
        ptr: bytes.as_mut_ptr(),
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
fn u32_mut_slice(slice: &mut [u32]) -> PieU32MutSlice {
    PieU32MutSlice {
        ptr: slice.as_mut_ptr(),
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
            let word_count = pie_grammar::bitmask::bitmask_size(mask.len());
            let start = words.len();
            words.resize(start + word_count, 0);
            let mut run_start = 0usize;
            for (index, &run_len) in mask.runs.iter().enumerate() {
                let run_end = run_start.saturating_add(run_len as usize);
                if index % 2 == 1 {
                    for bit in run_start..run_end.min(mask.len()) {
                        pie_grammar::bitmask::set_bit(&mut words[start..], bit);
                    }
                }
                run_start = run_end;
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
            geometry_class: plan.geometry_class as u32,
            reserved1: 0,
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

fn step_desc<'a>(step: &'a StepSubmission, masks: &'a MaskWordsStorage) -> pie_driver_abi::PieStepDesc {
    let plan = &step.plan;
    pie_driver_abi::PieStepDesc {
        roster_rows: u32_slice(&step.roster_rows),
        sub_batch_indptr: u32_slice(&step.sub_batch_indptr),
        sub_batch_class: u32_slice(&step.sub_batch_class),
        terminal_cells: PieTerminalCellPtrSlice {
            ptr: step.terminal_cells.as_ptr(),
            len: step.terminal_cells.len(),
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
        reserved_flags: [0; 2],
        reserved0: 0,
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
        embed_rows: bytes_slice(&plan.embed_rows),
        embed_indptr: u32_slice(&plan.embed_indptr),
        embed_shapes: u32_slice(&plan.embed_shapes),
        embed_dtypes: u8_slice(&plan.embed_dtypes),
        embed_anchor_rows: u32_slice(&plan.embed_anchor_rows),
        embed_block_indptr: u32_slice(&plan.embed_block_indptr),
        kv_len: u32_slice(&plan.kv_len),
        kv_len_device: u64_slice(&plan.kv_len_device),
        ptir_program_row_indptr: u32_slice(&step.program_row_indptr),
        ptir_kv_write_lower_bounds: u64_slice(&plan.kv_write_lower_bounds),
        ptir_kv_write_upper_bounds: u64_slice(&plan.kv_write_upper_bounds),
        logical_fire_ids: u64_slice(&step.logical_fire_ids),
        channel_expected_head: u64_slice(&step.channel_expected_head),
        channel_expected_tail: u64_slice(&step.channel_expected_tail),
        channel_ticket_indptr: u32_slice(&step.channel_ticket_indptr),
    }
}

/// Borrowed v14 frame descriptor: owns the packed mask words and the
/// per-step descriptor array; every other field borrows the submission for
/// the lifetime of the backend call.
pub struct FrameDescBorrow<'a> {
    _masks: Vec<MaskWordsStorage>,
    _steps: Vec<pie_driver_abi::PieStepDesc>,
    raw: pie_driver_abi::PieFrameDesc,
    _submission: &'a FrameSubmission,
}
impl<'a> FrameDescBorrow<'a> {
    pub fn from_submission(submission: &'a FrameSubmission) -> Self {
        let masks: Vec<MaskWordsStorage> = submission
            .steps
            .iter()
            .map(|step| MaskWordsStorage::from_plan(&step.plan))
            .collect();
        let steps: Vec<pie_driver_abi::PieStepDesc> = submission
            .steps
            .iter()
            .zip(&masks)
            .map(|(step, masks)| step_desc(step, masks))
            .collect();
        let raw = pie_driver_abi::PieFrameDesc {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            instance_ids: u64_slice(&submission.instance_ids),
            kv_translation: u32_slice(&submission.kv_translation),
            kv_translation_indptr: u32_slice(&submission.kv_translation_indptr),
            required_kv_pages: submission.required_kv_pages,
            reserved1: 0,
            steps: pie_driver_abi::PieStepDescSlice {
                ptr: steps.as_ptr(),
                len: steps.len(),
            },
        };
        Self {
            _masks: masks,
            _steps: steps,
            raw,
            _submission: submission,
        }
    }
    pub fn as_raw(&self) -> &pie_driver_abi::PieFrameDesc {
        &self.raw
    }
}

pub struct EncodeDescBorrow<'a> {
    raw: PieEncodeDesc,
    _plan: &'a mut pie_driver_abi::MediaEncodePlan,
}

impl<'a> EncodeDescBorrow<'a> {
    pub fn new(plan: &'a mut pie_driver_abi::MediaEncodePlan) -> Self {
        let raw = PieEncodeDesc {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            image_grids: u32_slice(&plan.image_grids),
            image_pixels: bytes_slice(&plan.image_pixels),
            image_pixel_indptr: u32_slice(&plan.image_pixel_indptr),
            image_patch_positions: u32_slice(&plan.image_patch_positions),
            image_anchor_rows: u32_slice(&plan.image_anchor_rows),
            audio_features: bytes_slice(&plan.audio_features),
            audio_feature_indptr: u32_slice(&plan.audio_feature_indptr),
            audio_anchor_rows: u32_slice(&plan.audio_anchor_rows),
            output_rows: mut_bytes_slice(&mut plan.output_rows),
            output_row_indptr: u32_mut_slice(&mut plan.output_row_indptr),
        };
        Self { raw, _plan: plan }
    }

    pub fn as_raw(&self) -> &PieEncodeDesc {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_lowered_masks_keep_existing_launch_wire_layout() {
        let plan = LaunchPlan {
            qo_indptr: vec![0, 1, 2],
            masks: vec![
                crate::driver::command::EncodedMask::new(vec![0, 3, 1], 4),
                crate::driver::command::EncodedMask::new(vec![1, 2, 1], 4),
            ],
            mask_indptr: vec![0, 1, 2],
            has_user_mask: true,
            ..LaunchPlan::default()
        };

        let storage = MaskWordsStorage::from_plan(&plan);
        assert_eq!(storage.request_indptr, vec![0, 1, 2]);
        assert_eq!(storage.word_indptr, vec![0, 1, 2]);
        assert_eq!(storage.words, vec![0b0111, 0b0110]);
    }

    #[test]
    fn omitted_mask_serializes_as_empty_rows() {
        let plan = LaunchPlan {
            qo_indptr: vec![0, 1, 2],
            ..LaunchPlan::default()
        };
        let storage = MaskWordsStorage::from_plan(&plan);
        assert_eq!(storage.request_indptr, vec![0, 0, 0]);
        assert_eq!(storage.word_indptr, vec![0]);
        assert!(storage.words.is_empty());
    }
}
