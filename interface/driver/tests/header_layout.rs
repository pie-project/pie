//! The **layout contract** — one hand-maintained list, two derived artifacts.
//!
//! `interface/driver/include/pie_driver_abi.h` is generated from `local.rs` by
//! cbindgen, so it cannot drift. The contract that pins the *layout* used to be
//! hand-written twice — here and in `tests/support/layout_contract.inc` — and
//! the two halves once went stale together: they agreed with each other and
//! disagreed with reality.
//!
//! Now the `.inc` is generated from this file. The field list below is the
//! single hand-maintained copy; [`rust_layout_matches_committed_header_contract`]
//! regenerates the `.inc` from Rust's own `size_of`/`align_of`/`offset_of!` and
//! fails if the committed copy differs, and
//! [`c11_and_cpp20_layout_sources_compile`] then makes a C and a C++ compiler
//! check the header against it. Adding a struct or a field to `local.rs`
//! without listing it here is the one remaining gap, and it is a gap in
//! coverage rather than a silent disagreement.
//!
//! Run with `PIE_REGEN=1` to bless a deliberate layout change.

use std::fmt::Write as _;
use std::mem::{align_of, offset_of, size_of};
use std::path::PathBuf;
use std::process::Command;

use pie_driver_abi::local::*;

/// Emits one type's `PIE_LAYOUT` / `PIE_FIELD` lines from Rust's own layout.
macro_rules! contract {
    ($out:expr, $ty:ty $(, $field:ident )* $(,)?) => {{
        writeln!(
            $out,
            "PIE_LAYOUT({}, {}, {})",
            stringify!($ty),
            size_of::<$ty>(),
            align_of::<$ty>()
        )
        .unwrap();
        $(
            writeln!(
                $out,
                "PIE_FIELD({}, {}, {})",
                stringify!($ty),
                stringify!($field),
                offset_of!($ty, $field)
            )
            .unwrap();
        )*
        writeln!($out).unwrap();
    }};
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn contract_source() -> String {
    let mut out = String::new();
    contract!(out, PieBytes, ptr, len);
    contract!(out, PieMutBytes, ptr, len);
    contract!(out, PieU8Slice, ptr, len);
    contract!(out, PieU32Slice, ptr, len);
    contract!(out, PieU32MutSlice, ptr, len);
    contract!(out, PieU64Slice, ptr, len);
    contract!(out, PieTerminalCell, outcome, reserved0);
    contract!(out, PieTerminalCellPtrSlice, ptr, len);
    contract!(
        out,
        PieChannelDesc,
        abi_version,
        reserved0,
        channel_id,
        shape,
        dtype,
        host_role,
        seeded,
        extern_dir,
        capacity,
        reserved1,
        reader_wait_id,
        writer_wait_id,
        extern_name
    );
    contract!(
        out,
        PieChannelEndpointBinding,
        channel_id,
        mirror_base,
        word_base,
        mirror_bytes,
        word_bytes,
        cell_bytes,
        capacity,
        head_word_index,
        tail_word_index,
        poison_word_index,
        closed_word_index
    );
    contract!(out, PieChannelValueDesc, channel_id, bytes);
    contract!(out, PieChannelValueDescSlice, ptr, len);
    contract!(out, PieMaskWordsDesc, request_indptr, word_indptr, words);
    contract!(
        out,
        PieKvMoveCell,
        dst_page_id,
        dst_token_offset,
        src_page_id,
        src_token_offset
    );
    contract!(out, PieKvMoveCellSlice, ptr, len);
    contract!(
        out,
        PieStateCopyRange,
        src_slot_id,
        dst_slot_id,
        src_token_offset,
        dst_token_offset,
        token_count
    );
    contract!(out, PieStateCopyRangeSlice, ptr, len);
    contract!(out, PiePoolRange, page_index, page_count);
    contract!(out, PiePoolRangeSlice, ptr, len);
    contract!(
        out,
        PieRuntimeCallbacks,
        abi_version,
        reserved0,
        ctx,
        notify
    );
    contract!(out, PieCompletion, wait_id, target_epoch, terminal_cell);
    contract!(
        out,
        PieDriverCreateDesc,
        abi_version,
        reserved0,
        config_bytes,
        runtime
    );
    contract!(out, PieDriverCaps, json_bytes, json_len);
    contract!(
        out,
        PieModelLoadDesc,
        abi_version,
        component,
        mxfp4_moe,
        runtime_quant,
        snapshot_dir
    );
    contract!(
        out,
        PieEmittedKernel,
        kind,
        stage_index,
        region_index,
        reserved0,
        entry_name,
        source,
        error
    );
    contract!(out, PieEmittedKernelSlice, ptr, len);
    contract!(
        out,
        PieDirectArgmax,
        node,
        source_value,
        intrinsic,
        requires_single_row,
        reserved0
    );
    contract!(out, PieDirectArgmaxSlice, ptr, len);
    contract!(
        out,
        PieRegionAnalysis,
        stage_index,
        region_index,
        flags,
        reserved0,
        direct_argmax,
        skipped
    );
    contract!(out, PieRegionAnalysisSlice, ptr, len);
    contract!(
        out,
        PieProgramDesc,
        abi_version,
        reserved0,
        program_hash,
        emitter_version,
        reserved1,
        emitted_kernels,
        region_analysis,
        launch
    );
    contract!(
        out,
        PieInstanceDesc,
        abi_version,
        reserved0,
        geometry_class,
        reserved1,
        program_id,
        requested_instance_id,
        pacing_wait_id,
        channel_ids,
        seed_values
    );
    contract!(
        out,
        PieInstanceBinding,
        instance_id,
        geometry_class,
        reserved0
    );
    contract!(
        out,
        PieStepDesc,
        roster_rows,
        sub_batch_indptr,
        sub_batch_class,
        terminal_cells,
        token_ids,
        position_ids,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        qo_indptr,
        rs_slot_ids,
        rs_slot_flags,
        rs_fold_lens,
        rs_buffer_slot_ids,
        rs_buffer_slot_indptr,
        masks,
        sampling_indices,
        sampling_indptr,
        context_ids,
        single_token_mode,
        has_user_mask,
        reserved_flags,
        reserved0,
        image_indptr,
        image_grids,
        image_anchor_positions,
        image_pixels,
        image_pixel_indptr,
        image_mrope_positions,
        image_mrope_indptr,
        image_patch_positions,
        image_anchor_rows,
        audio_features,
        audio_feature_indptr,
        audio_anchor_rows,
        audio_indptr,
        embed_rows,
        embed_indptr,
        embed_shapes,
        embed_dtypes,
        embed_anchor_rows,
        embed_block_indptr,
        kv_len,
        kv_len_device,
        ptir_program_row_indptr,
        ptir_kv_write_lower_bounds,
        ptir_kv_write_upper_bounds,
        logical_fire_ids,
        channel_expected_head,
        channel_expected_tail,
        channel_ticket_indptr,
        planned_hook_free_prefix_rows,
        planned_unmasked_prefix_rows,
        planned_max_layers,
        planned_full_depth_rows
    );
    contract!(out, PieStepDescSlice, ptr, len);
    contract!(
        out,
        PieFrameDesc,
        abi_version,
        reserved0,
        instance_ids,
        kv_translation,
        kv_translation_indptr,
        required_kv_pages,
        reserved1,
        steps
    );
    contract!(
        out,
        PieEncodeDesc,
        abi_version,
        reserved0,
        image_grids,
        image_pixels,
        image_pixel_indptr,
        image_patch_positions,
        image_anchor_rows,
        audio_features,
        audio_feature_indptr,
        audio_anchor_rows,
        output_rows,
        output_row_indptr
    );
    contract!(
        out,
        PieKvCopyDesc,
        abi_version,
        src_domain,
        src_device_ordinal,
        dst_domain,
        dst_device_ordinal,
        reserved0,
        src_page_ids,
        dst_page_ids,
        cells
    );
    contract!(out, PieStateCopyDesc, abi_version, reserved0, slot_ranges);
    contract!(
        out,
        PiePoolResizeDesc,
        abi_version,
        reserved0,
        pool_id,
        target_pages,
        map_ranges,
        unmap_ranges
    );
    contract!(
        out,
        PieLaunchValue,
        id,
        source,
        dtype,
        intrinsic,
        reserved1,
        channel,
        literal_bits,
        reserved0,
        shape
    );
    contract!(out, PieLaunchValueSlice, ptr, len);
    contract!(
        out,
        PieLaunchOp,
        code,
        result_count,
        result_id,
        intrinsic,
        lit_dtype,
        dtype,
        pred_tag,
        rng_kind,
        reserved0,
        lit_bits,
        pred_payload,
        channel,
        name_index,
        imm,
        imm2,
        imm3,
        reserved1,
        args,
        shape
    );
    contract!(out, PieLaunchOpSlice, ptr, len);
    contract!(
        out,
        PieLaunchChannel,
        id,
        capacity,
        dtype,
        flags,
        extern_dir,
        readiness,
        reserved1,
        shape,
        extern_name
    );
    contract!(out, PieLaunchChannelSlice, ptr, len);
    contract!(
        out,
        PieLaunchPort,
        port,
        is_const,
        const_dtype,
        reserved0,
        channel,
        const_shape,
        const_data
    );
    contract!(out, PieLaunchPortSlice, ptr, len);
    contract!(out, PieLaunchPut, channel, value);
    contract!(out, PieLaunchPutSlice, ptr, len);
    contract!(
        out,
        PieLaunchRegion,
        kind,
        library,
        schedule,
        reserved0,
        reserved1,
        nodes,
        inputs,
        outputs,
        sinks
    );
    contract!(out, PieLaunchRegionSlice, ptr, len);
    contract!(
        out,
        PieLaunchPlanValue,
        dtype,
        reserved0,
        reserved1,
        extents,
        dims
    );
    contract!(out, PieLaunchPlanValueSlice, ptr, len);
    contract!(out, PieLaunchChannelRule, value, local);
    contract!(out, PieLaunchChannelRuleSlice, ptr, len);
    contract!(
        out,
        PieLaunchStagePlan,
        signature_hash,
        identity,
        flags,
        mtp_rows,
        ops,
        source_ops,
        source_op_counts,
        value_types,
        channel_bindings,
        names,
        singleton,
        fused,
        used_extents,
        channel_rules,
        error
    );
    contract!(out, PieLaunchStagePlanSlice, ptr, len);
    contract!(
        out,
        PieLaunchStage,
        kind,
        reserved0,
        reserved1,
        reserved2,
        ops,
        puts,
        takes,
        reads
    );
    contract!(out, PieLaunchStageSlice, ptr, len);
    contract!(out, PieBytesSlice, ptr, len);
    contract!(
        out,
        PieLaunchPackage,
        values,
        channels,
        ports,
        names,
        stages,
        plans
    );
    out
}

fn contract_path() -> PathBuf {
    manifest_dir()
        .join("tests")
        .join("support")
        .join("layout_contract.inc")
}

#[test]
fn rust_layout_matches_committed_header_contract() {
    let generated = contract_source();
    let path = contract_path();
    if std::env::var_os("PIE_REGEN").is_some() {
        std::fs::write(&path, &generated).expect("write layout contract");
        return;
    }
    let committed = std::fs::read_to_string(&path).expect("read layout contract");
    assert_eq!(
        committed,
        generated,
        "{} is stale; re-run with PIE_REGEN=1",
        path.display()
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
