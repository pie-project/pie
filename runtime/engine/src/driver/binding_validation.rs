use std::collections::HashSet;
use std::mem::{align_of, size_of};

use anyhow::{Result, anyhow, ensure};
use pie_driver_abi::{PieChannelBinding, PieInstanceBinding};

use super::frame::InstanceBindingPlan;

pub(crate) fn validate_instance_binding(
    binding: &PieInstanceBinding,
    plan: &InstanceBindingPlan,
) -> Result<()> {
    pie_driver_abi::validate_instance_binding(binding)
        .map_err(|err| anyhow!("invalid native instance binding: {err}"))?;
    ensure!(
        plan.channel_waits.len() == plan.channel_ids.len(),
        "bind plan has {} channel waits for {} channels",
        plan.channel_waits.len(),
        plan.channel_ids.len()
    );
    ensure!(
        binding.instance_id != 0,
        "native binding returned instance id 0"
    );
    if plan.requested_instance_id != 0 {
        ensure!(
            binding.instance_id == plan.requested_instance_id,
            "native binding returned instance {} for requested {}",
            binding.instance_id,
            plan.requested_instance_id
        );
    }
    ensure!(
        binding.channels.len == binding.channel_count as usize,
        "native binding channel_count {} != slice length {}",
        binding.channel_count,
        binding.channels.len
    );

    validate_region("frame", binding.frame_base, binding.frame_bytes, 1)?;
    validate_region("mirror", binding.mirror_base, binding.mirror_bytes, 1)?;
    validate_region(
        "word",
        binding.word_base,
        binding.word_bytes,
        align_of::<u64>() as u64,
    )?;
    let expected_word_bytes = u64::from(binding.word_count)
        .checked_mul(size_of::<u64>() as u64)
        .ok_or_else(|| anyhow!("native binding word byte count overflow"))?;
    ensure!(
        binding.word_bytes == expected_word_bytes,
        "native binding word_bytes {} != word_count {} × {}",
        binding.word_bytes,
        binding.word_count,
        size_of::<u64>()
    );

    if binding.channels.len == 0 {
        ensure!(
            binding.channel_count == 0,
            "native binding has no channel slice but nonzero channel_count"
        );
        return Ok(());
    }

    let ptr = binding.channels.ptr as usize;
    ensure!(
        ptr % align_of::<PieChannelBinding>() == 0,
        "native binding channel table is misaligned"
    );
    binding
        .channels
        .len
        .checked_mul(size_of::<PieChannelBinding>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| anyhow!("native binding channel table length overflow"))?;

    let expected_ids = plan.channel_ids.iter().copied().collect::<HashSet<_>>();
    ensure!(
        expected_ids.len() == plan.channel_ids.len(),
        "bind plan contains duplicate channel ids"
    );
    let channels =
        unsafe { std::slice::from_raw_parts(binding.channels.ptr, binding.channels.len) };
    let mut returned_ids = HashSet::with_capacity(channels.len());
    let mut word_indices = HashSet::with_capacity(channels.len() * 3);
    let mut mirror_ranges = Vec::with_capacity(channels.len());
    for channel in channels {
        ensure!(
            expected_ids.contains(&channel.channel_id),
            "native binding returned unknown channel {}",
            channel.channel_id
        );
        ensure!(
            returned_ids.insert(channel.channel_id),
            "native binding returned duplicate channel {}",
            channel.channel_id
        );
        ensure!(
            channel.cell_bytes != 0,
            "native binding channel {} has zero cell_bytes",
            channel.channel_id
        );
        let cap1 = channel
            .capacity
            .checked_add(1)
            .ok_or_else(|| anyhow!("channel {} capacity overflow", channel.channel_id))?;
        let span = u64::from(channel.cell_bytes)
            .checked_mul(u64::from(cap1))
            .ok_or_else(|| anyhow!("channel {} mirror span overflow", channel.channel_id))?;
        let end = channel
            .mirror_offset
            .checked_add(span)
            .ok_or_else(|| anyhow!("channel {} mirror end overflow", channel.channel_id))?;
        ensure!(
            end <= binding.mirror_bytes,
            "channel {} mirror range {}..{} exceeds mirror_bytes {}",
            channel.channel_id,
            channel.mirror_offset,
            end,
            binding.mirror_bytes
        );
        mirror_ranges.push((channel.mirror_offset, end, channel.channel_id));

        for (kind, index) in [
            ("head", channel.head_word_index),
            ("tail", channel.tail_word_index),
            ("poison", channel.poison_word_index),
        ] {
            ensure!(
                index < binding.word_count,
                "channel {} {kind} word index {} >= word_count {}",
                channel.channel_id,
                index,
                binding.word_count
            );
            ensure!(
                word_indices.insert(index),
                "native binding reuses word index {index}"
            );
        }
    }
    mirror_ranges.sort_unstable_by_key(|range| range.0);
    for pair in mirror_ranges.windows(2) {
        ensure!(
            pair[0].1 <= pair[1].0,
            "native binding mirror ranges for channels {} and {} overlap",
            pair[0].2,
            pair[1].2
        );
    }
    Ok(())
}

fn validate_region(name: &str, base: u64, bytes: u64, alignment: u64) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    ensure!(base != 0, "native binding {name} base is null");
    ensure!(
        base % alignment == 0,
        "native binding {name} base is not {alignment}-byte aligned"
    );
    let end = base
        .checked_add(bytes)
        .ok_or_else(|| anyhow!("native binding {name} extent overflow"))?;
    ensure!(
        end <= usize::MAX as u64,
        "native binding {name} extent exceeds address space"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_driver_abi::{PIE_DRIVER_ABI_VERSION, PieChannelBindingSlice, PieChannelWait};

    fn plan() -> InstanceBindingPlan {
        InstanceBindingPlan {
            driver_id: 0,
            program_id: 1,
            requested_instance_id: 7,
            pacing_wait_id: 11,
            channel_waits: vec![PieChannelWait::default()],
            channel_ids: vec![101],
            seed_values: Vec::new(),
        }
    }

    fn valid_binding(
        channels: &[PieChannelBinding],
        mirror: &mut [u8],
        words: &mut [u64],
    ) -> PieInstanceBinding {
        PieInstanceBinding {
            instance_id: 7,
            mirror_base: mirror.as_mut_ptr() as u64,
            word_base: words.as_mut_ptr() as u64,
            channel_count: channels.len() as u32,
            word_count: words.len() as u32,
            mirror_bytes: mirror.len() as u64,
            word_bytes: std::mem::size_of_val(words) as u64,
            channels: PieChannelBindingSlice {
                ptr: channels.as_ptr(),
                len: channels.len(),
            },
            ..PieInstanceBinding::default()
        }
    }

    fn channel() -> PieChannelBinding {
        PieChannelBinding {
            channel_id: 101,
            cell_bytes: 4,
            capacity: 1,
            mirror_offset: 0,
            head_word_index: 0,
            tail_word_index: 1,
            poison_word_index: 2,
            reserved: 0,
        }
    }

    #[test]
    fn accepts_valid_reader_layout() {
        assert_eq!(PIE_DRIVER_ABI_VERSION, 1);
        let channels = [channel()];
        let mut mirror = [0u8; 8];
        let mut words = [0u64; 3];
        validate_instance_binding(&valid_binding(&channels, &mut mirror, &mut words), &plan())
            .unwrap();
    }

    #[test]
    fn rejects_extent_index_and_identity_errors() {
        let mut mirror = [0u8; 8];
        let mut words = [0u64; 3];

        let mut bad = channel();
        bad.mirror_offset = 4;
        assert!(
            validate_instance_binding(&valid_binding(&[bad], &mut mirror, &mut words), &plan(),)
                .unwrap_err()
                .to_string()
                .contains("exceeds mirror_bytes")
        );

        let mut bad = channel();
        bad.poison_word_index = 3;
        assert!(
            validate_instance_binding(&valid_binding(&[bad], &mut mirror, &mut words), &plan(),)
                .unwrap_err()
                .to_string()
                .contains("word index")
        );

        let mut bad = channel();
        bad.channel_id = 999;
        assert!(
            validate_instance_binding(&valid_binding(&[bad], &mut mirror, &mut words), &plan(),)
                .unwrap_err()
                .to_string()
                .contains("unknown channel")
        );
    }

    #[test]
    fn rejects_overlapping_channels_and_bad_word_bytes() {
        let mut second = channel();
        second.channel_id = 102;
        second.head_word_index = 3;
        second.tail_word_index = 4;
        second.poison_word_index = 5;
        let mut expected_plan = plan();
        expected_plan.channel_ids.push(102);
        expected_plan.channel_waits.push(PieChannelWait::default());
        let mut mirror = [0u8; 16];
        let mut words = [0u64; 6];
        let channels = [channel(), second];
        let binding = valid_binding(&channels, &mut mirror, &mut words);
        assert!(
            validate_instance_binding(&binding, &expected_plan)
                .unwrap_err()
                .to_string()
                .contains("overlap")
        );

        let channels = [channel()];
        let mut binding = valid_binding(&channels, &mut mirror, &mut words[..3]);
        binding.word_bytes -= 1;
        assert!(
            validate_instance_binding(&binding, &plan())
                .unwrap_err()
                .to_string()
                .contains("word_bytes")
        );
    }
}
