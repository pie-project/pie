//! engine-metal's integration tests, one binary: each module was a test file.

mod common;
mod a_lagged_window_fold_serves_the_folded_logits;
mod a_guest_over_two_lanes_reads_each_lanes_own_row;
mod a_buffered_fold_is_the_fold_it_replaces;
mod a_column_cut_of_a_bank_multiplies_as_its_copy;
mod a_copied_window_is_one_encode_over_the_same_rows;
mod a_load_measuring_nothing_admits_its_own_pool;
mod a_load_walks_every_readout_tier_its_set_states;
mod a_refused_lane_is_refused_alone;
mod a_fenced_frame_lands_both_sides;
mod a_gate_and_an_up_apart_land_what_they_land_packed;
mod a_parked_lane_serves_the_never_parked_logits;
mod a_prompt_scan_lands_its_snapshots;
mod a_scattered_pool_serves_the_same_logits;
mod a_ternary_decode_in_dot_answers_the_host_decoder;
mod a_tower_region_reads_the_patch_window;
mod a_verify_walk_is_the_committed_walk;
mod an_airborne_pass_samples_its_own_row;
mod each_checked_in_profile_runs_its_familys_old_tuning;
mod every_catalog_deployment_dispatches;
mod every_unit_compiles;
mod mma_attention_matches_fp64;
mod mpp_attention_over_a_bf16_pool_matches_fp64;
mod the_block_dyn_conv_matches_its_reference;
mod the_conditioning_ops_answer_their_host_reference;
mod the_folded_qmv_answers_the_reference;
mod the_gated_delta_scans_agree;
mod the_hadamard_transform_agrees;
mod the_kv_codec_round_trips;
mod the_kv_pool_grows_in_place;
mod the_kv_rotation_is_the_unrotated_forward;
mod the_kv_write_round_trips;
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
    let pair_at = layout.free_chunk() + 8;
    let own_at = pair_at + chunks_of(state.per_pair) + nth * chunks_of(state.instance_bytes());
    let places = (place(state.instance_bytes(), own_at), place(state.per_pair, pair_at));
    let end = own_at + chunks_of(state.instance_bytes());
    shell
        .commit_segments(end.div_ceil(layout.segment_chunks.max(1)))
        .expect("the guest state's segments commit");
    places
}
