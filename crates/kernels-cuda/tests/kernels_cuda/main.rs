//! kernels-cuda's integration tests, one binary: each module was a test file.
#![cfg(feature = "cuda")]

mod common;
mod a_body_captured_on_its_own_stream_takes_its_owners_tile;
mod a_qkv_split_reaches_every_row_of_a_tall_fire;
mod a_trimmed_pixel_shuffle_drops_its_anchor_frames;
mod deepseeks_attention_answers_the_reference;
mod every_dit_seat_entry_retires_the_staged_windows_padded_rows;
mod the_axial_rope_answers_the_one_axis_kernel_and_the_three_axis_reference;
mod the_bare_binary_and_activation_ops_answer_the_reference;
mod the_chunked_gated_delta_answers_the_serial_scan;
mod the_chunked_kda_folds_its_commit_and_replays_its_tail;
mod the_conv3d_answers_the_cpu_reference;
mod the_conv_update_and_the_row_cut_move_whole_vectors;
mod the_dflash2_dynamic_conv_answers_its_host_reference;
mod the_dspark_readout_points_answer_their_host_reference;
mod the_expanded_mla_prefill_lands_what_the_absorbed_one_does;
mod the_fused_gdn_step_answers_what_the_recurrence_says;
mod the_fused_norm_modulate_lands_what_the_two_launches_land;
mod the_fused_q_norm_rope_lands_the_pair;
mod the_gemms_answer_the_dequantized_dot;
mod the_group_norm_answers_the_cpu_reference;
mod the_hyper_connection_answers_the_host;
mod the_indexers_select_their_best_blocks;
mod the_inkling_points_answer_their_host_reference;
mod the_interleaved_mrope_turns_the_rotary_prefix_as_its_own_head;
mod the_mla_append_lands_each_request_in_its_own_pages;
mod the_mla_prefill_ada_arm_answers_the_reference;
mod the_mla_scores_land_their_log_sum_exp;
mod the_modulate_forms_answer_the_broadcast_and_the_per_token_reference;
mod the_moe_selects_answer_the_routed_dot;
mod the_norms_and_rotations_answer_the_host;
mod the_paged_attention_answers_the_reference;
mod the_partial_rope_turns_only_the_rotary_pairs;
mod the_quantizers_land_what_the_host_rounds;
mod the_ragged_arm_answers_the_reference;
mod the_relative_bucket_bias_table_reads_the_bucket_torch_lands;
mod the_routers_pick_the_best_experts;
mod the_row_tiled_causal_conv_answers_the_per_row_reference;
mod the_selected_attention_lands_alike_on_the_tensor_cores;
mod the_sinusoid_embedding_matches_the_diffusers_formula;
mod the_sparse_mla_lands_on_tcgen05;
mod the_spatial_attention_answers_the_cpu_reference;
mod the_spatial_attention_segments_by_frame;
mod the_tcgen05_gemms_answer_the_dequantized_dot;
mod the_voxel_reshapes_land_the_reference_ordering;
mod the_word_gather_lands_each_run_where_its_table_says;

// Strict kernels for the whole binary, before any test runs: an
// instantiation that will not compile fails its test instead of falling
// back unseen.
#[cfg(target_os = "linux")]
#[used]
#[unsafe(link_section = ".init_array")]
static STRICT_KERNELS: extern "C" fn() = {
    extern "C" fn on() {
        kernels_cuda::jit::strict_kernels();
    }
    on
};
