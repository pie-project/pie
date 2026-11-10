//! engine-xla's integration tests, one binary: each module was a test file.

mod common;
mod common_dit;

mod a_channel_fed_voxel_input_lands_the_committed_cell;
mod a_context_input_merged_into_the_stream_lands_its_rows;
mod a_conv_decoder_fires_over_a_voxel_input;
mod a_device_geometry_beam_answers_the_host_path;
mod a_double_block_fires_two_streams_through_the_engine;
mod a_prompt_length_reuses_its_programs;
mod an_epilogue_agrees_with_the_host_interpreter;
mod kernels_act;
mod kernels_attn;
mod kernels_collective;
mod kernels_decode;
mod kernels_hc;
mod kernels_index;
mod kernels_layout;
mod kernels_linear;
mod kernels_mla;
mod kernels_moe;
mod kernels_mosaic;
mod kernels_norm;
mod kernels_norm_fused;
mod kernels_pack;
mod kernels_ple;
mod kernels_pool;
mod kernels_quant;
mod kernels_replay;
mod kernels_rope;
mod kernels_spatial;
mod kernels_ssm;
mod nemotron_traces_on_a_dry_device;
mod pjrt_runs_a_built_module;
