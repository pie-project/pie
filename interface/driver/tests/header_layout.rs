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
    assert_layout!(PieMutBytes, 16, 8, ptr => 0, len => 8);
    assert_layout!(PieU8Slice, 16, 8, ptr => 0, len => 8);
    assert_layout!(PieU32Slice, 16, 8, ptr => 0, len => 8);
    assert_layout!(PieU32MutSlice, 16, 8, ptr => 0, len => 8);
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
        PieModelLoadDesc,
        48,
        8,
        abi_version => 0,
        component => 4,
        compiler_version => 8,
        load_plan_bytes => 16,
        snapshot_dir => 32
    );
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
        72,
        8,
        abi_version => 0,
        reserved0 => 4,
        geometry_class => 8,
        reserved1 => 12,
        program_id => 16,
        requested_instance_id => 24,
        pacing_wait_id => 32,
        channel_ids => 40,
        seed_values => 56
    );
    assert_layout!(
        PieInstanceBinding,
        16,
        8,
        instance_id => 0,
        geometry_class => 8,
        reserved0 => 12
    );
    assert_layout!(
        PieStepDesc,
        792,
        8,
        roster_rows => 0,
        sub_batch_indptr => 16,
        sub_batch_class => 32,
        terminal_cells => 48,
        token_ids => 64,
        position_ids => 80,
        kv_page_indices => 96,
        kv_page_indptr => 112,
        kv_last_page_lens => 128,
        qo_indptr => 144,
        rs_slot_ids => 160,
        rs_slot_flags => 176,
        rs_fold_lens => 192,
        rs_buffer_slot_ids => 208,
        rs_buffer_slot_indptr => 224,
        masks => 240,
        sampling_indices => 288,
        sampling_indptr => 304,
        context_ids => 320,
        single_token_mode => 336,
        has_user_mask => 337,
        reserved_flags => 338,
        reserved0 => 340,
        image_indptr => 344,
        image_grids => 360,
        image_anchor_positions => 376,
        image_pixels => 392,
        image_pixel_indptr => 408,
        image_mrope_positions => 424,
        image_mrope_indptr => 440,
        image_patch_positions => 456,
        image_anchor_rows => 472,
        audio_features => 488,
        audio_feature_indptr => 504,
        audio_anchor_rows => 520,
        audio_indptr => 536,
        embed_rows => 552,
        embed_indptr => 568,
        embed_shapes => 584,
        embed_dtypes => 600,
        embed_anchor_rows => 616,
        embed_block_indptr => 632,
        kv_len => 648,
        kv_len_device => 664,
        ptir_program_row_indptr => 680,
        ptir_kv_write_lower_bounds => 696,
        ptir_kv_write_upper_bounds => 712,
        logical_fire_ids => 728,
        channel_expected_head => 744,
        channel_expected_tail => 760,
        channel_ticket_indptr => 776
    );
    assert_layout!(
        PieStepDescSlice,
        16,
        8,
        ptr => 0,
        len => 8
    );
    assert_layout!(
        PieFrameDesc,
        80,
        8,
        abi_version => 0,
        reserved0 => 4,
        instance_ids => 8,
        kv_translation => 24,
        kv_translation_indptr => 40,
        required_kv_pages => 56,
        reserved1 => 60,
        steps => 64
    );
    assert_layout!(
        PieEncodeDesc,
        168,
        8,
        abi_version => 0,
        reserved0 => 4,
        image_grids => 8,
        image_pixels => 24,
        image_pixel_indptr => 40,
        image_patch_positions => 56,
        image_anchor_rows => 72,
        audio_features => 88,
        audio_feature_indptr => 104,
        audio_anchor_rows => 120,
        output_rows => 136,
        output_row_indptr => 152
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
