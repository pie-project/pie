use std::mem::{align_of, offset_of, size_of};
use std::path::PathBuf;
use std::process::Command;

use pie_driver_abi::local::*;

macro_rules! assert_layout {
    ($ty:ty, $size:expr, $align:expr $(, $field:ident => $offset:expr )* $(,)?) => {{
        assert_eq!(size_of::<$ty>(), $size, "sizeof({})", stringify!($ty));
        assert_eq!(align_of::<$ty>(), $align, "alignof({})", stringify!($ty));
        $(
            assert_eq!(
                offset_of!($ty, $field),
                $offset,
                "offsetof({}, {})",
                stringify!($ty),
                stringify!($field)
            );
        )*
    }};
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn rust_layout_matches_committed_header_contract() {
    assert_layout!(PieBytes, 16, 8, ptr => 0, len => 8);
    assert_layout!(PieU8Slice, 16, 8, ptr => 0, len => 8);
    assert_layout!(PieU32Slice, 16, 8, ptr => 0, len => 8);
    assert_layout!(PieU64Slice, 16, 8, ptr => 0, len => 8);
    assert_layout!(PieTerminalCell, 8, 4, outcome => 0, reserved0 => 4);
    assert_layout!(PieTerminalCellPtrSlice, 16, 8, ptr => 0, len => 8);
    assert_layout!(
        PieChannelDesc,
        80,
        8,
        abi_version => 0,
        reserved0 => 4,
        channel_id => 8,
        shape => 16,
        dtype => 32,
        host_role => 33,
        seeded => 34,
        extern_dir => 35,
        capacity => 36,
        reserved1 => 40,
        reader_wait_id => 48,
        writer_wait_id => 56,
        extern_name => 64
    );
    assert_layout!(
        PieChannelEndpointBinding,
        64,
        8,
        channel_id => 0,
        mirror_base => 8,
        word_base => 16,
        mirror_bytes => 24,
        word_bytes => 32,
        cell_bytes => 40,
        capacity => 44,
        head_word_index => 48,
        tail_word_index => 52,
        poison_word_index => 56,
        closed_word_index => 60
    );
    assert_layout!(PieChannelValueDesc, 24, 8, channel_id => 0, bytes => 8);
    assert_layout!(PieChannelValueDescSlice, 16, 8, ptr => 0, len => 8);
    assert_layout!(
        PieMaskWordsDesc,
        48,
        8,
        request_indptr => 0,
        word_indptr => 16,
        words => 32
    );
    assert_layout!(
        PieKvMoveCell,
        16,
        4,
        dst_page_id => 0,
        dst_token_offset => 4,
        src_page_id => 8,
        src_token_offset => 12
    );
    assert_layout!(PieKvMoveCellSlice, 16, 8, ptr => 0, len => 8);
    assert_layout!(
        PieStateCopyRange,
        20,
        4,
        src_slot_id => 0,
        dst_slot_id => 4,
        src_token_offset => 8,
        dst_token_offset => 12,
        token_count => 16
    );
    assert_layout!(PieStateCopyRangeSlice, 16, 8, ptr => 0, len => 8);
    assert_layout!(PiePoolRange, 16, 8, page_index => 0, page_count => 8);
    assert_layout!(PiePoolRangeSlice, 16, 8, ptr => 0, len => 8);
    assert_layout!(
        PieRuntimeCallbacks,
        24,
        8,
        abi_version => 0,
        reserved0 => 4,
        ctx => 8,
        notify => 16
    );
    assert_layout!(
        PieCompletion,
        24,
        8,
        wait_id => 0,
        target_epoch => 8,
        terminal_cell => 16
    );
    assert_layout!(
        PieDriverCreateDesc,
        48,
        8,
        abi_version => 0,
        reserved0 => 4,
        config_bytes => 8,
        runtime => 24
    );
    assert_layout!(PieDriverCaps, 16, 8, json_bytes => 0, json_len => 8);
    assert_layout!(
        PieProgramDesc,
        48,
        8,
        abi_version => 0,
        reserved0 => 4,
        program_hash => 8,
        canonical_bytes => 16,
        sidecar_bytes => 32
    );
    assert_layout!(
        PieInstanceDesc,
        64,
        8,
        abi_version => 0,
        reserved0 => 4,
        program_id => 8,
        requested_instance_id => 16,
        pacing_wait_id => 24,
        channel_ids => 32,
        seed_values => 48
    );
    assert_layout!(PieInstanceBinding, 8, 8, instance_id => 0);
    assert_layout!(
        PieLaunchDesc,
        560,
        8,
        abi_version => 0,
        reserved0 => 4,
        instance_ids => 8,
        terminal_cells => 24,
        token_ids => 40,
        position_ids => 56,
        kv_page_indices => 72,
        kv_page_indptr => 88,
        kv_last_page_lens => 104,
        qo_indptr => 120,
        rs_slot_ids => 136,
        rs_slot_flags => 152,
        rs_fold_lens => 168,
        rs_buffer_slot_ids => 184,
        rs_buffer_slot_indptr => 200,
        masks => 216,
        sampling_indices => 264,
        sampling_indptr => 280,
        context_ids => 296,
        single_token_mode => 312,
        has_user_mask => 313,
        reserved_flags => 314,
        image_indptr => 320,
        image_grids => 336,
        image_anchor_positions => 352,
        image_pixels => 368,
        image_pixel_indptr => 384,
        image_mrope_positions => 400,
        image_mrope_indptr => 416,
        image_patch_positions => 432,
        image_anchor_rows => 448,
        audio_features => 464,
        audio_feature_indptr => 480,
        audio_anchor_rows => 496,
        audio_indptr => 512,
        kv_len => 528,
        kv_len_device => 544
    );
    assert_layout!(
        PieKvCopyDesc,
        72,
        8,
        abi_version => 0,
        src_domain => 4,
        src_device_ordinal => 8,
        dst_domain => 12,
        dst_device_ordinal => 16,
        reserved0 => 20,
        src_page_ids => 24,
        dst_page_ids => 40,
        cells => 56
    );
    assert_layout!(
        PieStateCopyDesc,
        24,
        8,
        abi_version => 0,
        reserved0 => 4,
        slot_ranges => 8
    );
    assert_layout!(
        PiePoolResizeDesc,
        56,
        8,
        abi_version => 0,
        reserved0 => 4,
        pool_id => 8,
        target_pages => 16,
        map_ranges => 24,
        unmap_ranges => 40
    );
}

fn run_compile(compiler: &str, standard: &str, source: &str, output: &str) {
    let manifest = manifest_dir();
    let support = manifest.join("tests").join("support");
    let include = manifest.join("include");
    let out_dir = manifest.join("target").join("header-layout-tests");
    std::fs::create_dir_all(&out_dir).expect("create header layout output dir");

    let status = Command::new(compiler)
        .arg(format!("-std={standard}"))
        .arg("-Werror")
        .arg("-pedantic")
        .arg("-I")
        .arg(&include)
        .arg("-I")
        .arg(&support)
        .arg("-c")
        .arg(support.join(source))
        .arg("-o")
        .arg(out_dir.join(output))
        .status()
        .unwrap_or_else(|e| panic!("spawn {compiler}: {e}"));

    assert!(status.success(), "{compiler} failed for {source}");
}

#[test]
fn c11_and_cpp20_layout_sources_compile() {
    run_compile("cc", "c11", "header_layout_c11.c", "header_layout_c11.o");
    run_compile(
        "c++",
        "c++20",
        "header_layout_cpp20.cpp",
        "header_layout_cpp20.o",
    );
}
