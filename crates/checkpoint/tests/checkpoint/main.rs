//! checkpoint's integration tests, one binary: each module was a test file.

mod a_component_that_keeps_its_weights_a_folder_down_is_still_read;
mod a_constant_lands_as_the_bytes_it_carries;
mod a_diffusers_pipeline_reads_as_one_prefixed_name_space;
mod a_replicated_shard_lands_on_every_rank_of_its_group;
mod an_fp32_pipeline_component_lands_as_bf16_planes;
mod groups;
mod memory_mount;
mod quant_terms;
mod serving_artifact;
mod storage_compiler;
mod ztensor_convert;
mod ztensor_roundtrip;
