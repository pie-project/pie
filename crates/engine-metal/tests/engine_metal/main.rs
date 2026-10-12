//! engine-metal's integration tests, one binary: each module was a test file.

mod a_buffered_fold_is_the_fold_it_replaces;
mod a_capped_fire_packs_no_more_windows_than_the_load_reserved;
mod a_column_cut_of_a_bank_multiplies_as_its_copy;
mod a_copied_window_is_one_encode_over_the_same_rows;
mod a_drafting_neighbour_leaves_a_plain_lane_alone;
mod a_family_is_read_against_its_reference;
mod a_fenced_frame_lands_both_sides;
mod a_forked_page_is_one_run_and_not_one_move_per_token;
mod a_ternary_decode_in_dot_answers_the_host_decoder;
mod a_tower_region_reads_the_patch_window;
mod an_apple10_gpu_reads_a_prompt_through_mpp;
mod every_catalog_deployment_dispatches;
mod mpp_attention_over_a_bf16_pool_matches_fp64;
mod the_block_drafter_loads_and_leaves_the_trunk_alone;
mod the_block_drafters_proposals_are_measured;
mod the_block_dyn_conv_matches_its_reference;
mod the_bonsai_27b_perplexity_matches_the_fork;
mod the_bonsai_27b_serves_and_matches_the_fork_oracle;
mod the_conditioning_ops_answer_their_host_reference;
mod the_folded_qmv_answers_the_reference;
mod the_full_two_bit_dsv4_serves_from_its_artifact;
mod the_gated_delta_scans_agree;
mod the_hadamard_transform_agrees;
mod the_kv_codec_round_trips;
mod the_kv_pool_grows_in_place;
mod the_kv_rotation_is_the_unrotated_forward;
mod the_kv_write_round_trips;
mod the_miniature_drafts;
mod the_mpp_prefill_matches_affine_dots;
mod the_mpp_routed_matmul_matches_affine_dots;
mod the_packed_kv_decode_reads;
mod the_packed_kv_page_migrates;
mod the_packed_kv_tiled_reads;
mod the_partial_rope_turns_only_the_rotary_pairs;
mod the_ptq1_0_gguf_passes_through_ingest_and_serves_the_host_decoder;
mod the_q8_kv_roundtrip;
mod the_ragged_attention_answers_its_host_reference;
mod the_rotated_packed_kv_serves;
mod the_selector_reads_out_the_reference_walk;
mod the_split_mlp_matches_the_gpu_alone;
mod the_tower_reads_pixels_as_the_host_cut_them;
mod the_voxel_axis_answers_its_host_reference;

/// The bytes a pool takes that holds `slots` sequences at `context`, the
/// `device_kv_budget` a test states for its shell.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
fn full_pool(trace: &poem_ir::Trace, page_size: u32, context: u32, slots: u32) -> Option<u64> {
    let pages = u64::from(slots * context.div_ceil(page_size));
    let paging = engine_metal::store::kv::Paging::of(page_size, context, slots, pages)
        .expect("a test's paging");
    Some(engine_metal::store::pool_bytes(trace, paging).expect("the pool sizes"))
}

/// Whether this process runs `test` itself: the first call re-runs the
/// binary on `test` alone, with `env` set, and asserts it passed, so a test
/// that states process-wide kernel tuning is not answered by a tuning an
/// earlier test already resolved. `true` in that child, `false` in the
/// parent.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
fn in_own_process(test: &str, env: &[(&str, &str)]) -> bool {
    const CHILD: &str = "PIE_TEST_OWN_PROCESS";
    if std::env::var(CHILD).as_deref() == Ok(test) {
        return true;
    }
    let status = std::process::Command::new(std::env::current_exe().expect("the test binary"))
        .args(["--exact", test, "--nocapture", "--test-threads=1"])
        .env(CHILD, test)
        .envs(env.iter().copied())
        .status()
        .expect("the test re-runs");
    assert!(status.success(), "{test}, run in its own process, failed");
    false
}
