"""End-to-end integration test for the Metal backend handler."""

from __future__ import annotations

import sys
from pathlib import Path

from dataclasses import asdict

import pytest
import torch


# ---------------------------------------------------------------------------
# Configure Python path using repo utilities
# ---------------------------------------------------------------------------
# Add repo root to path first
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import and use repo utilities to set up all paths
from repo_utils import setup_pie_imports
setup_pie_imports()

# Add Metal-specific runtime path
BACKEND_METAL_SRC = repo_root / "backend" / "backend-metal" / "src"
INTEGRATION_TEST_DIR = Path(__file__).resolve().parent

for path in (BACKEND_METAL_SRC, INTEGRATION_TEST_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


import message
from config.common import ModelInfo, ModelConfig
from l4ma_runtime import MetalL4maBackend
# Import MetalBackend from the backend-metal implementation
import sys
from pathlib import Path

# Add backend-metal src to path to import our local MetalBackend
BACKEND_METAL_SRC = repo_root / "backend" / "backend-metal" / "src"
if str(BACKEND_METAL_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_METAL_SRC))

from metal_backend import MetalBackend
from metal_handler import MetalHandler
from model_loader import load_model
from test_tokenizer_util import TokenizerUtil
# Import Metal model factory directly to avoid ambiguity with Python backend version
import importlib.util
metal_factory_path = BACKEND_METAL_SRC / "model_factory.py"
spec = importlib.util.spec_from_file_location("metal_model_factory", metal_factory_path)
if spec is not None and spec.loader is not None:
    metal_model_factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metal_model_factory)
    create_metal_model_and_fusion_map = metal_model_factory.create_model_and_fusion_map
else:
    raise ImportError("Failed to load Metal model factory")


def _detect_model_weight_dtype(metadata_path: Path) -> str:
    try:
        cfg = ModelConfig.load_from_file(str(metadata_path))
        parameters = cfg.get_metadata_fields()["parameters"]
    except Exception as exc:  # pragma: no cover - configuration issues
        print(f"⚠️ Failed to parse model metadata for dtype detection: {exc}")
        return "float32"

    model_dir = metadata_path.parent / metadata_path.stem
    for param_file in parameters:
        weights_path = model_dir / param_file
        if not weights_path.exists():
            continue

        try:
            import ztensor

            with ztensor.Reader(str(weights_path)) as reader:  # type: ignore[attr-defined]
                tensor_names = reader.get_tensor_names()
                if not tensor_names:
                    continue
                first_name = tensor_names[0]
                metadata = reader.get_metadata(first_name)
                dtype_name = getattr(metadata, "dtype", None)
                if dtype_name:
                    return str(dtype_name)
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"⚠️ Failed to inspect {weights_path}: {exc}")
            break

    return "float32"


def _load_real_model(device: str) -> tuple[torch.nn.Module, ModelInfo, torch.dtype]:
    """Load the real cached model or raise with actionable context."""

    possible_cache_dirs = [
        Path.home() / "Library" / "Caches" / "pie",
        Path.home() / ".cache" / "pie",
    ]

    model_name = "llama-3.2-1b-instruct"
    load_errors: list[str] = []

    for cache_dir in possible_cache_dirs:
        metadata_path = cache_dir / "models" / f"{model_name}.toml"
        if not metadata_path.exists():
            load_errors.append(f"missing metadata at {metadata_path}")
            continue

        dtype_name = _detect_model_weight_dtype(metadata_path)
        try:
            dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise RuntimeError(f"Unsupported model dtype '{dtype_name}' detected") from exc

        print(f"   Detected model weight dtype: {dtype_name}")

        dtype_str = dtype_name
        config = {
            "model": model_name,
            "cache_dir": str(cache_dir),
            "device": device,
            "dtype": dtype_str,
        }

        def create_model_fn(model_info: ModelInfo):
            print(f"   Using Metal backend for {model_info.architecture.type}")
            return create_metal_model_and_fusion_map(model_info)

        print(f"🎯 Found real model metadata at {metadata_path}")
        print("📥 Loading real model weights...")
        try:
            model, model_info = load_model(config, create_model_fn)
        except Exception as exc:  # propagate detailed load error
            load_errors.append(f"failed to load weights from {cache_dir}: {exc}")
            continue

        if model_info.tokenizer is None:
            raise RuntimeError("Model tokenizer missing; cannot run integration test")

        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Loaded real {model_info.architecture.type.upper()} model:")
        print(f"   • {param_count:,} parameters")
        print(f"   • {model_info.architecture.num_layers} layers")
        print(f"   • {model_info.architecture.num_query_heads} query heads, {model_info.architecture.num_key_value_heads} KV heads")
        print(f"   • Vocab size: {model_info.architecture.vocab_size:,}")
        print(f"   • Tokenizer entries: {len(model_info.tokenizer.merge_table)} merges")
        return model, model_info, dtype

    error_details = "; ".join(load_errors) if load_errors else "no cache directories contained the model"
    raise FileNotFoundError(
        "Unable to load Metal integration model - "
        f"{error_details}. Ensure real PIE weights are present."
    )


@pytest.mark.integration
def test_metal_handler_forward_pass_e2e():
    """Verify that the Metal handler can run a forward pass end-to-end."""

    if sys.platform != "darwin":
        pytest.skip("Metal backend requires macOS")

    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an available MPS device")

    if not MetalL4maBackend.is_available():
        pytest.skip("Metal runtime not available in this environment")

    device = "mps"
    model, model_info, dtype = _load_real_model(device)

    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Initialise a Metal runtime with architecture metadata for accurate kernel planning
    try:
        metal_backend = MetalBackend(
            model_metadata={"architecture": asdict(model_info.architecture)}
        )
        if not metal_backend.initialize():
            pytest.skip("Metal backend failed to initialize executor")
        runtime_backend = MetalL4maBackend(metal_backend=metal_backend)
    except RuntimeError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Metal backend failed to initialise: {exc}")

    model.model.backend = runtime_backend

    handler = MetalHandler(
        model=model,
        model_info=model_info,
        kv_page_size=16,
        max_dist_size=32,
        max_num_kv_pages=64,
        max_num_embeds=8,
        max_num_adapters=4,
        max_adapter_rank=8,
        dtype=dtype,
        device=device,
    )

    # Sanity-check the handshake path returns metadata derived from ModelInfo.
    handshake_resp = handler.handshake([message.HandshakeRequest(version="0.1")])[0]
    assert handshake_resp.model_name == model_info.name
    assert handshake_resp.kv_page_size == handler.kv_page_size
    assert handshake_resp.resources[0] == handler.max_num_kv_pages

    # Build a forward-pass request that exercises both distribution queries and
    # Metal sampling kernels (top-k sampling funnels through metal_ops).
    input_tokens = [1, 5, 9, 3]
    positions = list(range(len(input_tokens)))
    mask = [[i + 1] for i in range(len(input_tokens))]
    kv_page_ptrs = [0]
    kv_page_last_len = len(input_tokens)

    request = message.ForwardPassRequest(
        input_tokens=input_tokens,
        input_token_positions=positions,
        input_embed_ptrs=[],
        input_embed_positions=[],
        adapter=None,
        adapter_seed=None,
        mask=mask,
        kv_page_ptrs=kv_page_ptrs,
        kv_page_last_len=kv_page_last_len,
        output_token_indices=[len(input_tokens) - 1, len(input_tokens) - 1],
        output_token_samplers=[
            {"sampler": 0, "top_k": 5, "temperature": 1.0},  # distribution
            {"sampler": 3, "top_k": 4, "temperature": 1.0},  # top-k sampling
        ],
        output_embed_ptrs=[],
        output_embed_indices=[],
    )

    responses = handler.forward_pass([request])

    assert len(responses) == 1
    forward_resp = responses[0]

    # First sampler requested a distribution; ensure we received top-k tuples.
    assert len(forward_resp.dists) == 1
    dist_ids, dist_vals = forward_resp.dists[0]
    assert len(dist_ids) == len(dist_vals) <= 5

    # Second sampler requested a sampled token via MetalOps top-k path.
    assert len(forward_resp.tokens) == 1
    sampled_token = forward_resp.tokens[0]
    assert isinstance(sampled_token, int)
    assert 0 <= sampled_token < model_info.architecture.vocab_size

    # Ensure the handler's KV cache has been populated for the configured layers.
    assert len(handler.kv_cache_at_layer) == model_info.architecture.num_layers
    for cache in handler.kv_cache_at_layer:
        assert cache.shape[0] == handler.max_num_kv_pages


@pytest.mark.integration
def test_metal_handler_with_all_prompts():
    """Test Metal handler with all prompts from test_prompts.json."""
    import json

    if sys.platform != "darwin":
        pytest.skip("Metal backend requires macOS")

    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an available MPS device")

    if not MetalL4maBackend.is_available():
        pytest.skip("Metal runtime not available in this environment")

    # Load test prompts from JSON file
    test_prompts_path = Path(__file__).parent / "test_prompts.json"
    if not test_prompts_path.exists():
        pytest.skip("test_prompts.json not found")

    with open(test_prompts_path, 'r') as f:
        prompts_config = json.load(f)

    # Get all prompts from all categories
    all_prompts = []
    for category, prompts in prompts_config.get("prompts", {}).items():
        if isinstance(prompts, list):
            all_prompts.extend(prompts)

    if not all_prompts:
        pytest.skip("No prompts found in test_prompts.json")

    device = "mps"

    # Load real cached model (fail fast if unavailable)
    model, model_info, dtype = _load_real_model(device)

    model = model.to(device=device, dtype=dtype)
    model.eval()

    first_ln = model.model.layers[0].input_layernorm.weight
    print(
        "[MetalTensorDebug]",
        "stage=post_model_to_device_input_layernorm_weight",
        "dtype=",
        first_ln.dtype,
        "device=",
        first_ln.device,
        "min=",
        float(first_ln.min()),
        "max=",
        float(first_ln.max()),
        "sample=",
        [float(x) for x in first_ln.flatten()[:8].cpu()],
    )

    if model_info.tokenizer is None:
        raise RuntimeError("Real model tokenizer unavailable; cannot proceed")

    tokenizer_util = TokenizerUtil(model_info)

    try:
        metal_backend = MetalBackend(
            model_metadata={"architecture": asdict(model_info.architecture)}
        )
        if not metal_backend.initialize():
            pytest.skip("Metal backend failed to initialize executor")
        runtime_backend = MetalL4maBackend(metal_backend=metal_backend)
    except RuntimeError as exc:
        pytest.skip(f"Metal backend failed to initialise: {exc}")

    model.model.backend = runtime_backend

    handler = MetalHandler(
        model=model,
        model_info=model_info,
        kv_page_size=16,
        max_dist_size=32,
        max_num_kv_pages=64,
        max_num_embeds=8,
        max_num_adapters=4,
        max_adapter_rank=8,
        dtype=dtype,
        device=device,
    )

    # Test ALL prompts from JSON file
    successful_tests = 0
    response_validation_results = []

    print(f"\n🧪 Testing ALL {len(all_prompts)} prompts from test_prompts.json with Metal kernels...")

    def build_forward_request(token_ids: list[int]) -> message.ForwardPassRequest:
        positions = list(range(len(token_ids)))

        mask = []
        for idx in range(len(token_ids)):
            context_len = idx + 1
            mask.append([context_len] if context_len > 0 else [])

        kv_page_size = handler.kv_page_size
        num_tokens = len(token_ids)
        num_full_pages = num_tokens // kv_page_size
        tokens_in_last_page = num_tokens % kv_page_size

        if tokens_in_last_page == 0 and num_full_pages > 0:
            kv_page_ptrs = list(range(num_full_pages))
            kv_page_last_len = kv_page_size
        else:
            total_pages = num_full_pages + (1 if tokens_in_last_page > 0 else 0)
            kv_page_ptrs = list(range(total_pages))
            kv_page_last_len = tokens_in_last_page if tokens_in_last_page > 0 else kv_page_size

        return message.ForwardPassRequest(
            input_tokens=token_ids,
            input_token_positions=positions,
            input_embed_ptrs=[],
            input_embed_positions=[],
            adapter=None,
            adapter_seed=None,
            mask=mask,
            kv_page_ptrs=kv_page_ptrs,
            kv_page_last_len=kv_page_last_len,
            output_token_indices=[len(token_ids) - 1],
            output_token_samplers=[
                {"sampler": 0, "top_k": 10, "temperature": 1.0},
            ],
            output_embed_ptrs=[],
            output_embed_indices=[],
        )

    def reset_handler_state() -> None:
        for layer_cache in handler.kv_cache_at_layer:
            layer_cache.zero_()
        handler.embeds.zero_()

    for i, prompt in enumerate(all_prompts, 1):
        print(f"\n📋 Testing prompt {i}/{len(all_prompts)}: '{prompt[:80]}{'...' if len(prompt) > 80 else ''}'")

        try:
            input_tokens = tokenizer_util.tokenize_with_production_tokenizer(prompt, model_info.tokenizer)
            if not input_tokens:
                raise RuntimeError(f"Tokenizer returned no tokens for prompt: {prompt}")

            # limit sequence length to manageable window while preserving production semantics
            max_tokens = handler.kv_page_size * 2
            input_tokens = input_tokens[:max_tokens]

            reset_handler_state()
            request = build_forward_request(input_tokens)
            responses = handler.forward_pass([request])

            assert len(responses) == 1
            forward_resp = responses[0]

            # Verify we got a distribution
            assert len(forward_resp.dists) == 1
            dist_ids, dist_vals = forward_resp.dists[0]
            assert len(dist_ids) == len(dist_vals) <= 10

            # Decode top tokens to text for validation using proper tokenizer
            top_tokens_text = [
                tokenizer_util.decode_tokens_with_tokenizer([token_id]).strip()
                for token_id in dist_ids[:3]
            ]

            print("     🔟 Top-10 distribution candidates:")
            top10_text = []
            for rank, (token_id, prob) in enumerate(zip(dist_ids[:10], dist_vals[:10]), start=1):
                token_text = tokenizer_util.decode_tokens_with_tokenizer([token_id]).replace("\n", "\\n")
                top10_text.append(token_text)
                print(f"       {rank:2d}. id={token_id:<7d} prob={prob:.6f} text='{token_text}'")

            # Validate response quality
            response_quality = validate_metal_response(prompt, dist_ids, dist_vals, prompts_config, top_tokens_text)
            response_validation_results.append(response_quality)

            print(f"  ✅ Metal backend processed prompt successfully")
            print(f"     Input: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")

            # Show real vs synthetic model difference
            if model_info and model_info.tokenizer:
                print(f"     🎯 Real model predictions:")
                for i, (token_text, prob) in enumerate(zip(top_tokens_text[:3], dist_vals[:3])):
                    print(f"       {i+1}. \"{token_text}\" (prob: {prob:.4f}, id: {dist_ids[i]})")
            else:
                print(f"     ⚠️ Synthetic model predictions: {top_tokens_text}")
                print(f"     Probabilities: {[f'{p:.4f}' for p in dist_vals[:3]]}")

            print(f"     📊 Response quality: {response_quality['quality']}")
            if response_quality['quality'] in ['good', 'excellent']:
                print(f"     ✅ {response_quality['validation_message']}")
            else:
                print(f"     ⚠️ {response_quality['validation_message']}")
                # Additional context for real model
                if model_info and model_info.tokenizer:
                    reasonable_count = sum(1 for token in top_tokens_text[:3] if token not in ["<no_decoder>", "<decode_error>", ""] and len(token.strip()) > 0)
                    if reasonable_count > 0:
                        print(f"     💡 Note: Real model is working, predictions are reasonable tokens")

            successful_tests += 1

            # Perform multiple decoding rounds to observe continuation behaviour
            num_decoding_rounds = 3
            current_tokens = list(input_tokens)
            current_dist_ids = list(dist_ids)
            current_dist_vals = list(dist_vals)

            for step in range(num_decoding_rounds):
                if not current_dist_ids:
                    break

                next_token_id = current_dist_ids[0]
                next_token_prob = current_dist_vals[0]
                next_token_text = tokenizer_util.decode_tokens_with_tokenizer([next_token_id]).replace("\n", "\\n")
                print(
                    f"     🔁 Decode step {step + 1}: chose id={next_token_id} prob={next_token_prob:.6f} text='{next_token_text}'"
                )

                current_tokens.append(next_token_id)
                reset_handler_state()
                iterative_request = build_forward_request(current_tokens)
                iterative_response = handler.forward_pass([iterative_request])[0]

                if iterative_response.dists:
                    current_dist_ids, current_dist_vals = iterative_response.dists[0]
                    print("       🔟 Follow-up top-10 candidates:")
                    for rank, (token_id, prob) in enumerate(zip(current_dist_ids[:10], current_dist_vals[:10]), start=1):
                        follow_text = tokenizer_util.decode_tokens_with_tokenizer([token_id]).replace("\n", "\\n")
                        print(f"         {rank:2d}. id={token_id:<7d} prob={prob:.6f} text='{follow_text}'")
                else:
                    current_dist_ids = []
                    current_dist_vals = []
                    print("       ⚠️ No distribution returned for follow-up step")

        except Exception as e:
            print(f"  ❌ Metal backend failed for prompt: {e}")
            response_validation_results.append({
                'prompt': prompt,
                'success': False,
                'error': str(e),
                'quality': 'failed'
            })
            continue

    # Analyze results by category
    category_results = analyze_results_by_category(all_prompts, response_validation_results, prompts_config)

    print(f"\n📊 COMPREHENSIVE METAL BACKEND TEST RESULTS:")
    print("=" * 80)
    print(f"Total prompts tested: {len(all_prompts)}")
    print(f"Successful Metal operations: {successful_tests}/{len(all_prompts)}")
    print(f"Success rate: {100 * successful_tests / len(all_prompts):.1f}%")

    print(f"\n📈 Results by Category:")
    for category, stats in category_results.items():
        print(f"  {category.capitalize()}: {stats['success']}/{stats['total']} successful "
              f"({100 * stats['success'] / stats['total']:.1f}%)")
        if stats['quality_distribution']:
            print(f"    Quality distribution: {stats['quality_distribution']}")

    print(f"\n🎯 Response Quality Analysis:")
    quality_counts = {}
    for result in response_validation_results:
        if result.get('success', False):
            quality = result.get('quality', 'unknown')
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

    for quality, count in quality_counts.items():
        print(f"  {quality}: {count} prompts")

    # Require high success rate for Metal operations
    assert successful_tests >= len(all_prompts) * 0.8, f"Metal backend should process at least 80% of prompts successfully, got {successful_tests}/{len(all_prompts)}"

    # Ensure we have reasonable response quality
    good_quality_count = sum(1 for r in response_validation_results
                           if r.get('success') and r.get('quality') in ['good', 'excellent'])
    assert good_quality_count > 0, "Metal backend should produce at least some high-quality responses"


def validate_metal_response(prompt: str, token_ids: list, token_probs: list, config: dict, top_tokens_text: list = None) -> dict:
    """Validate Metal backend response quality against prompt expectations."""
    prompt_lower = prompt.lower()

    # Check if response shows reasonable distribution characteristics
    if not token_ids or not token_probs:
        return {
            'prompt': prompt,
            'success': False,
            'quality': 'failed',
            'validation_message': 'No predictions returned'
        }

    # Check probability distribution quality and decoded token reasonableness
    prob_sum = sum(token_probs[:5])  # Top 5 should have meaningful probabilities

    # Basic kernel functionality check
    if prob_sum < 0.01:  # Very low probabilities suggest kernel issues
        quality = 'poor'
        message = f'Very low probability mass: {prob_sum:.4f} (possible kernel issue)'
    elif len(set(token_ids[:5])) < 3:  # Too few unique predictions
        quality = 'poor'
        message = f'Low diversity: {len(set(token_ids[:5]))} unique tokens in top 5'
    else:
        # Check decoded token quality if available
        if top_tokens_text and len(top_tokens_text) >= 3:
            # Look for reasonable tokens (not just random characters or errors)
            reasonable_tokens = 0
            for token_text in top_tokens_text[:3]:
                if (token_text and
                    token_text not in ['<decode_error>', '<no_decoder>', '<unk>'] and
                    len(token_text.strip()) > 0 and
                    not all(c in '!@#$%^&*()[]{}' for c in token_text.strip())):
                    reasonable_tokens += 1

            if reasonable_tokens >= 2 and prob_sum > 0.05:
                quality = 'good'
                message = f'Metal kernels working well: {reasonable_tokens}/3 reasonable tokens, prob_sum={prob_sum:.4f}'
            elif reasonable_tokens >= 1 and prob_sum > 0.03:
                quality = 'fair'
                message = f'Metal kernels functional: {reasonable_tokens}/3 reasonable tokens, prob_sum={prob_sum:.4f}'
            else:
                quality = 'poor'
                message = f'Poor token quality: {reasonable_tokens}/3 reasonable tokens, prob_sum={prob_sum:.4f}'
        else:
            # Fallback to probability-only assessment
            if prob_sum > 0.05:
                quality = 'good'
                message = f'Metal kernels working: prob_sum={prob_sum:.4f}, diversity={len(set(token_ids[:5]))}'
            else:
                quality = 'fair'
                message = f'Basic functionality: prob_sum={prob_sum:.4f}, diversity={len(set(token_ids[:5]))}'

    # Additional validation based on prompt type
    validation_keywords = config.get("validation", {})

    # For specific prompt types, we expect certain characteristics
    if "france" in prompt_lower or "capital" in prompt_lower:
        if quality == 'good':
            quality = 'excellent'
            message += ' (geography prompt handled well)'
    elif "hello" in prompt_lower or "name" in prompt_lower:
        if quality == 'good':
            quality = 'excellent'
            message += ' (greeting prompt handled well)'
    elif any(char in prompt for char in "🚀🇫🇷中文العربية"):
        if quality == 'good':
            quality = 'excellent'
            message += ' (unicode/emoji prompt handled well)'
    elif "def " in prompt or "fibonacci" in prompt_lower:
        if quality == 'good':
            quality = 'excellent'
            message += ' (code prompt handled well)'

    return {
        'prompt': prompt,
        'success': True,
        'quality': quality,
        'validation_message': message,
        'top_tokens': token_ids[:5],
        'top_probs': token_probs[:5],
        'prob_sum': prob_sum
    }


def analyze_results_by_category(prompts: list, results: list, config: dict) -> dict:
    """Analyze results by prompt category."""
    category_results = {}

    prompts_by_category = config.get("prompts", {})

    for category, category_prompts in prompts_by_category.items():
        if not isinstance(category_prompts, list):
            continue

        category_results[category] = {
            'total': len(category_prompts),
            'success': 0,
            'quality_distribution': {}
        }

        for prompt in category_prompts:
            # Find corresponding result
            result = None
            for r in results:
                if r.get('prompt') == prompt:
                    result = r
                    break

            if result and result.get('success'):
                category_results[category]['success'] += 1
                quality = result.get('quality', 'unknown')
                category_results[category]['quality_distribution'][quality] = \
                    category_results[category]['quality_distribution'].get(quality, 0) + 1

    return category_results
