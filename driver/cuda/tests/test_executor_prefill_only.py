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


def test_forward_failures_do_not_return_zero_request_forward_response():
    executor_source = (CUDA_SRC / "executor" / "executor.cpp").read_text()
    service_source = (CUDA_SRC / "service" / "inproc_service.cpp").read_text()

    signature = "bool handle_fire_batch("
    assert signature in executor_source
    assert signature in (CUDA_SRC / "executor" / "executor.hpp").read_text()

    assert "return false;" in executor_source
    assert "out_resp = pie_driver::PieForwardResponseView{};" not in executor_source

    assert "const bool ok = handle_fire_batch(" in service_source
    assert "out.forward.num_requests == expected" in service_source
    assert "out.method = pie_driver::PIE_METHOD_HEALTH;" in service_source
    assert "out.status = -1;" in service_source
