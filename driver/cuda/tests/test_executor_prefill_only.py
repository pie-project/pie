from pathlib import Path


def test_prefill_only_fire_batch_does_not_copy_sampled_tokens():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "executor"
        / "executor.cpp"
    ).read_text()

    guard = "const bool has_sampling = num_sampling > 0;"
    assert guard in source

    copy = "cudaMemcpyAsync(sampled_host, pi.sampled.data()"
    copy_index = source.index(copy)
    guard_index = source.rfind("if (has_sampling)", 0, copy_index)
    sync_index = source.rfind("cudaStreamSynchronize(cublas.stream())", 0, copy_index)

    assert guard_index > sync_index
