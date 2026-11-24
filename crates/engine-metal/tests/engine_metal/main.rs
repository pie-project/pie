//! engine-metal's integration tests, one binary: each module was a test file.

mod common;
mod a_bank_past_2_gib_reads_its_last_rows;
mod a_beam_step_follows_the_spec;
mod a_bound_guest_leaves_the_span_as_it_was;
mod a_cold_registration_compiles_while_lanes_fire;
mod a_group_fold_is_the_sequential_fold;
mod a_carry_past_the_table_is_served;
mod a_lagged_decode_answers_the_unlagged;
mod a_lagged_window_fold_serves_the_folded_logits;
mod the_resident_probe_counts_the_cores;
mod a_mixed_fires_last_rows_are_each_lanes_own;
mod a_prefill_reads_no_kv_cell_past_its_keys;
mod a_prewarmed_program_registers_off_thread_without_compiling;
mod a_faulted_epilogue_fails_its_own_lanes_alone;
mod a_few_row_dense_matmul_reads_each_weight_once;
mod a_guest_over_two_lanes_reads_each_lanes_own_row;
mod a_guest_reads_only_its_own_lanes_rows;
mod a_buffered_fold_is_the_fold_it_replaces;
mod a_column_cut_of_a_bank_multiplies_as_its_copy;
mod a_copied_window_is_one_encode_over_the_same_rows;
mod a_dense_model_reads_no_routed_or_quantized_decision;
mod a_device_fold_lane_states_its_own_fold;
mod a_kv_only_lane_reports_its_device_fold;
mod a_masked_beam_beside_plain_decodes_answers_as_each_alone;
mod a_load_measuring_nothing_admits_its_own_pool;
mod a_load_walks_every_readout_tier_its_set_states;
mod a_load_walks_its_score_capturing_forms;
mod a_recurrent_load_walks_its_pipelines;
mod a_refused_lane_is_refused_alone;
mod a_device_placed_kv_write_lands_in_its_buffer;
mod a_fenced_frame_lands_both_sides;
mod a_fire_carrying_past_its_rungs_tokens_is_refused;
mod a_frames_steps_keep_their_rows_until_taken;
mod a_forked_lane_serves_the_unforked_logits;
mod a_gate_and_an_up_apart_land_what_they_land_packed;
mod a_lane_with_a_page_in_every_buffer_serves_the_seats_logits;
mod a_page_in_an_uncommitted_chunk_is_refused_by_name;
mod a_prompt_scan_lands_its_snapshots;
mod a_scattered_pool_serves_the_same_logits;
mod a_score_reader_holds_its_rows_through_a_park;
mod a_settled_fires_fold_is_told_before_it_returns;
mod a_take_leaves_passes_in_flight_flying;
mod a_stated_position_bounds_no_key;
mod a_stale_frozen_length_replays_from_where_the_record_stands;
mod a_stated_group_mask_reads_only_the_rows_it_admits;
mod a_ternary_decode_in_dot_answers_the_host_decoder;
mod a_tower_region_reads_the_patch_window;
mod a_verify_walk_is_the_committed_walk;
mod a_wide_row_reads_no_kv_cell_past_its_keys;
mod an_adapter_in_slots_is_one_landing_through_the_table;
mod an_airborne_pass_samples_its_own_row;
mod an_envelope_keeps_two_passes_in_flight;
mod an_idle_lane_reports_its_last_landing;
mod an_instance_outlives_its_failed_frame;
mod an_operand_past_an_ints_reach_reads_its_last_rows;
mod an_operator_declared_adapter_is_refused;
mod an_rs_page_past_the_first_buffer_holds_its_rows;
mod each_checked_in_profile_runs_its_familys_old_tuning;
mod every_catalog_deployment_dispatches;
mod every_unit_compiles;
mod mma_attention_at_128_matches_the_tiled_kernel_and_fp64;
mod mma_attention_matches_fp64;
mod mpp_attention_over_a_bf16_pool_matches_fp64;
mod the_block_dyn_conv_matches_its_reference;
mod the_conditioning_ops_answer_their_host_reference;
mod the_dense_mpp_matches_the_simdgroup_gemm;
mod the_folded_qmv_answers_the_reference;
mod the_gated_delta_scans_agree;
mod the_hadamard_transform_agrees;
mod the_kv_codec_round_trips;
mod the_kv_pool_grows_in_place;
mod the_kv_rotation_is_the_unrotated_forward;
mod the_kv_write_round_trips;
mod the_ledger_accounts_for_the_device;
mod the_mpp_prefill_matches_affine_dots;
mod the_mpp_routed_matmul_matches_affine_dots;
mod the_mpp_routed_mxfp4_matmul_matches_its_dots;
mod the_packed_kv_page_migrates;
mod the_packed_kv_reads;
mod the_partial_rope_turns_only_the_rotary_pairs;
mod the_prompt_conv_matches_a_running_reference;
mod the_ptq1_0_gguf_passes_through_ingest_and_serves_the_host_decoder;
mod a_q8_pool_takes_every_route_and_matches_fp64;
mod the_ragged_attention_answers_its_host_reference;
mod the_register_routed_mlp_answers_the_per_route_path;
mod the_register_rows_match_affine_dots;
mod the_rotated_packed_kv_serves;
mod the_rs_scratch_is_the_widest_layers;
mod the_score_rows_are_the_oracles;
mod the_selector_reads_out_the_reference_walk;
mod the_staged_rows_match_affine_dots;
mod the_tower_reads_pixels_as_the_host_cut_them;
mod the_voxel_axis_answers_its_host_reference;
mod the_weight_cache_serves_the_dense_logits;
mod two_models_exported_boot_a_clean_home_measuring_nothing;

/// The `device_kv_budget` a test states for its shell: none, so the chunk
/// pool spans the room. A sparse pool maps only the segments a fire touches
/// (a direct fire, its seats' runs), so the room costs address space alone.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
fn full_pool(trace: &poem_ir::Trace, page_size: u32, context: u32, slots: u32) -> Option<u64> {
    let _ = (trace, page_size, context, slots);
    None
}
mod an_unfolded_decode_is_the_folded_one;

/// The guest state places the runtime takes for instance `nth` of
/// `program` at `extents` (`engine::Engine::guest_state`), as a test lays
/// them past its tables and rings (8 chunks past the layout's first free
/// one): the pair's first, the same for every instance of the pair, then
/// each instance's own, apart. Each a run of `Instance` slots where the
/// state shares a chunk, else whole chunks. The room is committed, as the
/// runtime commits what a bind takes.
#[cfg(target_vendor = "apple")]
pub(crate) fn guest_places(
    shell: &mut engine_metal::Shell,
    program: engine::ProgramId,
    extents: &eta_exec::Extents,
    nth: u32,
) -> (engine::ScratchPlace, engine::ScratchPlace) {
    guest_places_of_pair(shell, program, extents, nth, 0)
}

/// [`guest_places`] for an instance of the `pair`-th pair a test binds: a
/// pair is a (program, extents), so instances of one program at other
/// extents are another pair and its state other bytes (the engine refuses
/// two pairs claiming one run). Pair 0 lies where [`guest_places`] lays it;
/// each later one past the room 16 instances' own state takes.
#[cfg(target_vendor = "apple")]
pub(crate) fn guest_places_of_pair(
    shell: &mut engine_metal::Shell,
    program: engine::ProgramId,
    extents: &eta_exec::Extents,
    nth: u32,
    pair: u32,
) -> (engine::ScratchPlace, engine::ScratchPlace) {
    let state = shell.guest_state(program, extents).expect("the program's guest state");
    let layout = shell.pool_layout();
    let instance = layout.instance.expect("an instance shape");
    let chunks_of = |bytes: u64| -> u32 {
        if bytes == 0 {
            0
        } else if state.run == 0 {
            1
        } else {
            let piece = u64::from(state.run) * layout.chunk_bytes;
            u32::try_from(bytes.div_ceil(piece)).expect("pieces") * state.run
        }
    };
    let place = |bytes: u64, chunk: u32| {
        if bytes == 0 {
            return engine::ScratchPlace::None;
        }
        if state.run == 0 {
            return engine::ScratchPlace::Run {
                first: chunk * instance.per_chunk,
                slots: u32::try_from(bytes.div_ceil(instance.slot_bytes)).expect("slots"),
            };
        }
        engine::ScratchPlace::Chunks {
            run: state.run,
            pieces: (0..chunks_of(bytes) / state.run).map(|k| chunk + k * state.run).collect(),
        }
    };
    let first_pair = layout.free_chunk() + 8;
    let own_at = first_pair + chunks_of(state.per_pair) + nth * chunks_of(state.instance_bytes());
    let pair_at = if pair == 0 {
        first_pair
    } else {
        first_pair
            + chunks_of(state.per_pair)
            + 16 * chunks_of(state.instance_bytes())
            + (pair - 1) * chunks_of(state.per_pair)
    };
    let places = (place(state.instance_bytes(), own_at), place(state.per_pair, pair_at));
    let end = (own_at + chunks_of(state.instance_bytes())).max(pair_at + chunks_of(state.per_pair));
    shell
        .commit_segments(end.div_ceil(layout.segment_chunks.max(1)))
        .expect("the guest state's segments commit");
    places
}
