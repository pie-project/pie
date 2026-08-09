cd /root/Workspace/pie-new-driver
for f in write_kv_to_pages_bf16 write_kv_to_pages_devwin flashinfer_decode_bf16 flashinfer_prefill_sm90 \
  flashinfer_prefill_custom_planless attention_naive_paged_bf16 qkv_decode_fused_devwin \
  chunked_swiglu_strided gpt_oss_glu_strided sigmoid_scalar_gate_add sigmoid_scalar_gate_strided_add \
  token_batched_weighted_sum_aligned causal_conv1d_update causal_conv1d_prefill_single \
  gdn_step_single gdn_step_single_state_bf16 gdn_prefill_single gdn_prefill_single_state_bf16 \
  mxfp4_moe_gate_up_decode_grouped rope_yarn gemm_cublas residual_add_scale_rmsnorm add_bias_strided \
  embed_vocab_shard split_gate_up ; do
  echo "### fn $f"
  git grep -n -w -- "$f" -- crates ':!crates/kernels-cuda/csrc' ':!*.cu' ':!*.cuh' ':!*.hpp' ':!*.cpp' | grep -v "^crates/model-compiler/src/dsl.rs" | head -30
done
