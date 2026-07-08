from pathlib import Path


CUDA_SRC = Path(__file__).resolve().parents[1] / "src"


def test_prefill_only_fire_batch_does_not_copy_sampled_tokens():
    source = (CUDA_SRC / "executor" / "executor.cpp").read_text()

    guard = "const bool has_sampling = num_sampling > 0;"
    assert guard in source

    copy = "cudaMemcpyAsync(sampled_host, pi.sampled.data()"
    copy_index = source.index(copy)
    guard_index = source.rfind("if (has_sampling)", 0, copy_index)
    sync_index = source.rfind("cudaStreamSynchronize(cublas.stream())", 0, copy_index)

    assert guard_index > sync_index


def test_llama_like_model_honors_per_fire_emit_logits_flag():
    source = (CUDA_SRC / "model" / "llama_like_model.cpp").read_text()

    assert "LlamaLikeForwardCfg fwd = fwd_cfg_;" in source
    assert "fwd.emit_logits = in.emit_logits;" in source
    assert "weights_, hf_config_, fwd, plan_," in source
    assert "weights_, hf_config_, fwd_cfg_, plan_," not in source
