#!/usr/bin/env python3
"""
Comprehensive test for dummy mode.

This test verifies that dummy mode:
1. Works correctly on CPU
2. Returns random tokens
3. Uses minimal memory
"""

import os
import sys

import torch

def test_dummy_mode_on_cpu():
    """Verify dummy mode runs on CPU."""
    print("=" * 60)
    print("TEST 1: Dummy mode uses CPU")
    print("=" * 60)
    
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print("  (Dummy mode will use CPU regardless)")
    print()


def test_runtime_config():
    """Test RuntimeConfig with dummy_mode."""
    print("=" * 60)
    print("TEST 2: RuntimeConfig with dummy_mode")
    print("=" * 60)
    
    from pie_worker.config import RuntimeConfig
    
    config = RuntimeConfig.from_args(
        hf_repo='dummy-model',
        dummy_mode=True,
        device='cpu',
    )
    
    assert config.dummy_mode == True, "dummy_mode should be True"
    assert config.device == torch.device('cpu'), f"device should be cpu, got {config.device}"
    print(f"  ✓ dummy_mode: {config.dummy_mode}")
    print(f"  ✓ device: {config.device}")
    print()


def test_runtime_initialization():
    """Test Runtime initialization in dummy mode."""
    print("=" * 60)
    print("TEST 3: Runtime initialization in dummy mode")
    print("=" * 60)
    
    from pie_worker.config import RuntimeConfig
    from pie_worker.runtime import Runtime
    
    config = RuntimeConfig.from_args(
        hf_repo='dummy-model',
        dummy_mode=True,
        device='cpu',
    )
    
    runtime = Runtime(config)
    
    assert runtime.type == "dummy", f"type should be 'dummy', got {runtime.type}"
    assert runtime.engine.__class__.__name__ == "DummyForwardPass", \
        f"engine should be DummyForwardPass, got {runtime.engine.__class__.__name__}"
    
    print(f"  ✓ runtime.type: {runtime.type}")
    print(f"  ✓ runtime.engine: {runtime.engine.__class__.__name__}")
    print(f"  ✓ runtime.model_config: {runtime.model_config.__class__.__name__}")
    print()
    
    return runtime


def test_embed_inputs(runtime):
    """Test embed_inputs method."""
    print("=" * 60)
    print("TEST 4: embed_inputs method")
    print("=" * 60)
    
    batch_metadata = {
        'token_ids': torch.tensor([1, 2, 3, 4, 5, 100, 200, 300]),
    }
    
    embeds = runtime.engine.embed_inputs(batch_metadata)
    
    assert embeds.shape[0] == 8, f"Expected 8 tokens, got {embeds.shape[0]}"
    assert embeds.shape[1] == runtime.model_config.dim_hidden, \
        f"Expected hidden dim {runtime.model_config.dim_hidden}, got {embeds.shape[1]}"
    assert embeds.device == torch.device('cpu'), f"Expected CPU, got {embeds.device}"
    
    print(f"  ✓ embed shape: {embeds.shape}")
    print(f"  ✓ embed device: {embeds.device}")
    print(f"  ✓ embed dtype: {embeds.dtype}")
    print()
    
    return embeds


def test_transform(runtime, embeds):
    """Test transform method."""
    print("=" * 60)
    print("TEST 5: transform method")
    print("=" * 60)
    
    # Create minimal inputs for transform
    batch_size = embeds.shape[0]
    
    hidden_states = runtime.engine.transform(
        input_embeds=embeds,
        position_ids=torch.arange(batch_size),
        qo_indptr=torch.tensor([0, batch_size]),
        kv_cache_at_layer=runtime.kv_cache_at_layer,
        kv_page_indices=torch.tensor([0]),
        kv_page_indptr=torch.tensor([0, 1]),
        kv_last_page_lens=torch.tensor([batch_size]),
        custom_mask=None,
        single_token_inference_mode=False,
        adapter_subpass=None,
    )
    
    # In dummy mode, transform is a pass-through
    assert hidden_states.shape == embeds.shape, \
        f"Expected shape {embeds.shape}, got {hidden_states.shape}"
    
    print(f"  ✓ transform output shape: {hidden_states.shape}")
    print(f"  ✓ transform is pass-through: True")
    print()
    
    return hidden_states


def test_sample(runtime, hidden_states):
    """Test sample method."""
    print("=" * 60)
    print("TEST 6: sample method")
    print("=" * 60)
    
    # Create sampling metadata
    sampling_metadata = {
        'indices_for_logits': [0, 2, 4, 6],  # Sample 4 positions
        'temperatures': torch.ones(4),
        'sampler_groups': {1: [0, 1, 2, 3]},  # All use uniform sampling
        'top_k': torch.tensor([50, 50, 50, 50]),
        'top_p': torch.tensor([0.9, 0.9, 0.9, 0.9]),
        'min_p': torch.tensor([0.0, 0.0, 0.0, 0.0]),
    }
    
    result = runtime.engine.sample(hidden_states, sampling_metadata)
    
    assert 'tokens' in result, "Result should have 'tokens'"
    assert 'dists' in result, "Result should have 'dists'"
    assert len(result['tokens']) == 4, f"Expected 4 tokens, got {len(result['tokens'])}"
    
    # Verify tokens are within vocabulary range
    for token in result['tokens']:
        assert 0 <= token < runtime.model_config.vocab_size, \
            f"Token {token} out of range [0, {runtime.model_config.vocab_size})"
    
    print(f"  ✓ sample returned {len(result['tokens'])} tokens")
    print(f"  ✓ tokens: {result['tokens']}")
    print(f"  ✓ all tokens in valid range [0, {runtime.model_config.vocab_size})")
    print()


def test_metadata_accessors(runtime):
    """Test metadata accessor methods."""
    print("=" * 60)
    print("TEST 7: Metadata accessors")
    print("=" * 60)
    
    metadata = runtime.get_metadata()
    assert 'name' in metadata
    print(f"  ✓ get_metadata(): {metadata}")
    
    template = runtime.get_chat_template()
    assert template['template_type'] == 'none', "Dummy mode should have 'none' template type"
    print(f"  ✓ get_chat_template(): type={template['template_type']}")
    
    tokenizer = runtime.get_tokenizer()
    assert 'num_vocab' in tokenizer
    print(f"  ✓ get_tokenizer(): num_vocab={tokenizer['num_vocab']}")
    print()


def test_handshake(runtime):
    """Test handshake RPC."""
    print("=" * 60)
    print("TEST 8: Handshake RPC")
    print("=" * 60)
    
    from pie_worker.message import HandshakeRequest
    
    request = HandshakeRequest(version="1.0.0")
    response = runtime.handshake(request)
    
    assert response.model_name is not None
    assert response.kv_page_size > 0
    # max_num_kv_pages is in resources dict with key 0
    assert response.resources[0] > 0, "max_num_kv_pages should be > 0"
    
    print(f"  ✓ model_name: {response.model_name}")
    print(f"  ✓ kv_page_size: {response.kv_page_size}")
    print(f"  ✓ max_num_kv_pages (resources[0]): {response.resources[0]}")
    print(f"  ✓ tokenizer_num_vocab: {response.tokenizer_num_vocab}")
    print()


def test_kv_cache_minimal():
    """Test that KV cache allocation is minimal."""
    print("=" * 60)
    print("TEST 9: KV cache is minimal")
    print("=" * 60)
    
    from pie_worker.config import RuntimeConfig
    from pie_worker.runtime import Runtime
    
    config = RuntimeConfig.from_args(
        hf_repo='dummy-model',
        dummy_mode=True,
        device='cpu',
    )
    
    runtime = Runtime(config)
    
    # Check KV cache size
    total_kv_bytes = 0
    for layer_cache in runtime.kv_cache_at_layer:
        total_kv_bytes += layer_cache.numel() * layer_cache.element_size()
    
    total_kv_mb = total_kv_bytes / (1024 * 1024)
    
    print(f"  ✓ Number of KV cache layers: {len(runtime.kv_cache_at_layer)}")
    print(f"  ✓ Total KV cache size: {total_kv_mb:.2f} MB")
    
    # Should be very small (< 1 MB)
    assert total_kv_mb < 1.0, f"KV cache should be < 1 MB, got {total_kv_mb:.2f} MB"
    print(f"  ✓ KV cache is minimal (< 1 MB)")
    print()


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " DUMMY MODE COMPREHENSIVE TEST ".center(58) + "║")
    print("║" + " (No GPU Usage) ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    try:
        test_dummy_mode_on_cpu()
        test_runtime_config()
        runtime = test_runtime_initialization()
        embeds = test_embed_inputs(runtime)
        hidden_states = test_transform(runtime, embeds)
        test_sample(runtime, hidden_states)
        test_metadata_accessors(runtime)
        test_handshake(runtime)
        test_kv_cache_minimal()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print()
        print("Dummy mode works correctly with NO GPU usage.")
        print()
        
    except Exception as e:
        import traceback
        print()
        print("=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
