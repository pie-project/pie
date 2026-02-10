import torch
from pie_backend.config import RuntimeConfig
from pie_backend.model.llama3 import ModelConfig as Llama3Config


def test_gpu_mem_utilization():
    # 1. Test RuntimeConfig factory and gpu_mem_utilization
    print("Testing RuntimeConfig.from_args with gpu_mem_utilization...")
    config = RuntimeConfig.from_args(
        model="meta-llama/Llama-3.2-1B-Instruct", gpu_mem_utilization=0.5, device="cpu"
    )
    assert (
        config.gpu_mem_utilization == 0.5
    ), f"Expected 0.5, got {config.gpu_mem_utilization}"
    # check default
    config_default = RuntimeConfig.from_args(
        model="meta-llama/Llama-3.2-1B-Instruct", device="cpu"
    )
    assert (
        config_default.gpu_mem_utilization == 0.9
    ), f"Expected default 0.9, got {config_default.gpu_mem_utilization}"

    print("RuntimeConfig tests passed.")

    # 2. Test ModelConfig.eval_max_num_kv_pages
    print("Testing Llama3Config.eval_max_num_kv_pages...")
    # Mock everything needed
    llama3_spec = {
        "num_layers": 16,
        "num_query_heads": 32,
        "num_key_value_heads": 8,
        "head_size": 64,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
        "rope": {
            "factor": 1.0,
            "high_frequency_factor": 1.0,
            "low_frequency_factor": 1.0,
            "theta": 500000.0,
        },
    }
    model_config = Llama3Config.from_dict(llama3_spec)

    # We need to mock get_available_memory or just run it and see if it runs
    # pie_backend.model.llama3 imports get_available_memory from ..utils
    # We can rely on it returning something valid for CPU.

    # Run eval
    pages = model_config.eval_max_num_kv_pages(config)
    print(f"Calculated pages (util=0.5): {pages}")
    assert pages > 0, "Should have calculated > 0 pages"

    pages_default = model_config.eval_max_num_kv_pages(config_default)
    print(f"Calculated pages (util=0.9): {pages_default}")

    # Since util 0.9 > 0.5, pages should be higher
    assert pages_default > pages, "Higher utilization should yield more pages"

    print("ModelConfig tests passed.")


if __name__ == "__main__":
    try:
        test_gpu_mem_utilization()
        print("ALL TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        exit(1)
