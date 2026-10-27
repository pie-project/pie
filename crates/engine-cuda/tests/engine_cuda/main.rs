//! engine-cuda's integration tests, one binary: each module was a test file.

// On ROCm the engine's kernels are kernels-rocm's, under kernels-cuda's name.
#[cfg(feature = "rocm")]
extern crate kernels_rocm as kernels_cuda;

// Strict kernels for the whole binary, before any test runs: an
// instantiation that will not compile fails its test instead of falling
// back unseen.
#[cfg(all(any(feature = "cuda", feature = "rocm"), target_os = "linux"))]
#[used]
#[allow(unsafe_code)]
#[unsafe(link_section = ".init_array")]
static STRICT_KERNELS: extern "C" fn() = {
    extern "C" fn on() {
        kernels_cuda::jit::strict_kernels();
    }
    on
};

mod bound;
#[cfg(feature = "device")]
mod common_dit;
#[cfg(feature = "device")]
mod common_encoder;
#[cfg(feature = "device")]
mod common_two_axis;
#[cfg(feature = "device")]
mod common_vae;

mod a_bank_says_which_site_it_corrects;
mod a_beam_reads_up_to_its_slot_what_its_mask_admits;
mod a_bidirectional_attention_adds_its_relative_bias_through_the_engine;
mod a_buffer_frees_after_another_context_closes;
mod a_buffered_fold_is_the_fold_it_replaces;
#[cfg(feature = "cuda")]
mod a_capture_survives_a_load_on_another_thread;
mod a_cell_table_reads_each_rows_patch_and_group;
mod a_channel_fed_voxel_input_lands_the_committed_cell;
mod a_context_input_merged_into_the_stream_lands_its_rows;
mod a_conv_decoder_fires_over_a_voxel_input;
mod a_declined_mega_moe_answers_what_its_legs_answer;
mod a_decode_envelope_packs_its_device_mask_into_each_lane;
mod a_dit_plan_loads_at_a_sixty_four_k_row_ceiling;
mod a_double_block_fires_two_streams_through_the_engine;
mod a_family_is_read_against_its_reference;
#[cfg(feature = "cuda")]
mod a_fired_router_and_mega_moe_answer_what_their_legs_answer;
mod a_fitted_pool_is_pledged_what_it_declared;
mod a_four_rank_group_fires_a_sharded_plan;
mod a_fragmented_window_gets_a_view_per_run;
mod a_fused_geglu_reads_its_affine_bank_as_planes;
#[cfg(feature = "cuda")]
mod a_fused_router_answers_what_its_product_and_pick_answer;
mod a_fused_swiglu_sizes_the_decoded_tile_to_its_packed_plane;
mod a_grouped_window_is_one_window_and_a_segment_list;
#[cfg(feature = "cuda")]
mod a_joined_projection_answers_what_its_legs_answer;
mod a_lane_sampler_draws_what_the_interpreter_draws;
mod a_missing_or_misshapen_input_feed_is_refused_by_name;
mod a_model_loaded_after_another_fits_as_in_a_fresh_process;
mod a_nucleus_cut_agrees_with_the_host_interpreter;
mod a_padded_row_writes_nothing;
mod a_prefill_past_the_gemvs_grid_is_served;
mod a_prologue_put_input_is_the_cell_the_forward_reads;
mod a_prologue_put_port_is_the_cell_the_fire_resolves;
mod a_regions_own_values_outlive_its_last_pass;
mod a_repacked_projection_serves_what_the_row_major_one_serves;
mod a_schedule_is_carved_where_it_is_launched;
#[cfg(feature = "cuda")]
mod a_seamed_fold_answers_its_fold_and_pre_mix;
mod a_second_fire_reads_the_input_cells_it_was_handed;
mod a_shared_adapter_is_one_slot_and_one_load;
mod a_shifted_body_admits_the_split_that_moved;
mod a_short_card_keeps_the_declared_context;
mod a_stored_k_quant_row_serves_as_stored;
mod a_synthesized_region_serves_the_logits_of_its_ops;
mod a_tapped_fire_reads_each_layer_out_as_the_pool_holds_it;
mod a_tower_region_reads_the_patch_window;
mod a_two_axis_plan_arms_its_token_bodies;
mod a_two_class_plan_reads_each_lane_back_from_its_own_seam;
mod a_verify_lands_what_the_point_rule_reads;
mod a_wide_class_attends_every_key_without_a_mask;
mod a_windowed_row_reads_what_the_full_row_reads;
mod an_adapter_is_not_a_lanes_scratch;
mod an_epilogue_agrees_with_the_host_interpreter;
mod an_exported_kv_stays_mapped;
mod an_mla_body_appends_only_its_window;
mod an_output_of_ids_reads_as_stored;
mod every_armed_body_answers_its_eager_walk;
mod every_catalog_deployment_dispatches;
mod every_diagnostic_is_a_word_the_boot_states;
#[cfg(feature = "device")]
mod every_op_agrees_with_the_host_interpreter;
#[cfg(feature = "device")]
mod every_suite_step_agrees_with_the_host_interpreter;
mod masked_axis;
mod no_shell_reads_the_environment;
mod routed_experts_stream;
mod the_flux_2_vae_answers_the_reference;
mod the_ltx_2_vae_answers_the_reference;
mod the_mini_dit_row_arms_its_bodies;
mod the_operators_fraction_sizes_the_pool;
mod the_wan_2_vae_answers_the_reference;
mod the_wan_2_vae_encodes_the_reference;
mod the_z_image_vae_answers_the_reference;
