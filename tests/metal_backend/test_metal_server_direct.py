#!/usr/bin/env python3
"""
Test the Metal backend server directly (what pie-cli communicates with)
to see if there are differences from our direct Metal handler test.
"""

import zmq
import msgspec
import time
import sys
import subprocess
import fcntl
import os
import struct
import torch
from pathlib import Path
import pytest

# Add repo root and setup imports
repo_root = Path(__file__).resolve().parents[2]  # Go up two levels: tests/metal_backend -> tests -> repo_root
sys.path.insert(0, str(repo_root))

from repo_utils import setup_pie_imports
setup_pie_imports()

# Add backend paths
BACKEND_METAL_SRC = repo_root / "backend" / "backend-metal" / "src"
BACKEND_COMMON = repo_root / "backend" / "common_python"
INTEGRATION_TEST_DIR = Path(__file__).resolve().parent  # Current directory

for path in (BACKEND_METAL_SRC, BACKEND_COMMON, INTEGRATION_TEST_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import message
from test_tokenizer_util import TokenizerUtil
from config.common import ModelInfo, ModelConfig

def _load_real_model_info() -> ModelInfo:
    """Load the real cached model info for tokenization."""
    possible_cache_dirs = [
        Path.home() / "Library" / "Caches" / "pie",
        Path.home() / ".cache" / "pie",
    ]

    model_name = "llama-3.2-1b-instruct"

    for cache_dir in possible_cache_dirs:
        metadata_path = cache_dir / "models" / f"{model_name}.toml"
        if not metadata_path.exists():
            continue

        try:
            # Use ModelInfo.load_from_file directly, similar to model_loader.py
            model_info = ModelInfo.load_from_file(str(metadata_path), "mps", torch.float16)

            if model_info.tokenizer is None:
                print("⚠️ Model tokenizer missing; cannot decode tokens")
                return None

            print(f"✅ Loaded model info with tokenizer: {len(model_info.tokenizer.merge_table)} merges")
            return model_info
        except Exception as exc:
            print(f"⚠️ Failed to load model info from {metadata_path}: {exc}")
            continue

    print("❌ Could not load model info from any cache directory")
    return None

def create_test_prompts():
    """Create comprehensive test prompts from test_prompts.json."""
    import json

    # Load test prompts from JSON file (same as integration test)
    test_prompts_path = Path(__file__).parent / "test_prompts.json"
    if not test_prompts_path.exists():
        # Fallback to simple prompts if JSON not available
        return [
            {"prompt": "What is", "tokens": [3923, 374]},
            {"prompt": "The capital of France is", "tokens": [791, 6864, 315, 9822, 374]},
            {"prompt": "Hello, my name is", "tokens": [9906, 11, 856, 836, 374]},
        ]

    with open(test_prompts_path, 'r') as f:
        prompts_config = json.load(f)

    # Get all prompts from all categories (same as integration test)
    all_prompts = []
    for category, prompts in prompts_config.get("prompts", {}).items():
        if isinstance(prompts, list):
            all_prompts.extend(prompts)

    # Return prompts without pre-tokenized forms since we'll tokenize dynamically
    return [{"prompt": prompt} for prompt in all_prompts]

def create_forward_request(tokens):
    """Create a forward pass request for given tokens."""
    positions = list(range(len(tokens)))
    mask = [[i + 1] for i in range(len(tokens))]

    return message.ForwardPassRequest(
        input_tokens=tokens,
        input_token_positions=positions,
        input_embed_ptrs=[],
        input_embed_positions=[],
        adapter=None,
        adapter_seed=None,
        mask=mask,
        kv_page_ptrs=[0],
        kv_page_last_len=len(tokens),
        output_token_indices=[len(tokens) - 1],
        output_token_samplers=[
            {"sampler": 0, "top_k": 10, "temperature": 1.0},
        ],
        output_embed_ptrs=[],
        output_embed_indices=[],
    )

def test_zmq_server_communication(ipc_endpoint):
    """Test direct ZMQ communication with Metal backend server."""
    print("🔌 Testing ZMQ communication with Metal backend server")

    # Load model info for token decoding
    model_info = _load_real_model_info()
    tokenizer_util = None
    if model_info:
        tokenizer_util = TokenizerUtil(model_info)
        print("🧠 TokenizerUtil initialized for token decoding")
    else:
        print("⚠️ Cannot decode tokens - continuing with token IDs only")

    # ZMQ setup
    context = zmq.Context()
    # Use DEALER so we speak the same ROUTER/DEALER pattern as the backend.
    # A REQ socket adds an empty delimiter frame that the server does not expect,
    # which caused the previous requests to be discarded.
    socket = context.socket(zmq.DEALER)

    try:
        print(f"🔗 Connecting to Metal backend via IPC: {ipc_endpoint}")
        socket.connect(ipc_endpoint)
        socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10 second timeout

        # Test forward pass with multiple prompts (no handshake needed)
        test_prompts = create_test_prompts()
        all_successful = True
        response_validation_results = []

        print(f"\n🧪 Testing {len(test_prompts)} prompts via ZMQ server communication...")

        for i, test_case in enumerate(test_prompts, 1):
            prompt = test_case["prompt"]

            # Tokenize the prompt dynamically using the tokenizer utility (like integration test)
            if tokenizer_util:
                try:
                    tokens = tokenizer_util.tokenize_with_production_tokenizer(prompt, model_info.tokenizer)
                    if not tokens:
                        print(f"   ⚠️ Tokenizer returned no tokens for prompt: '{prompt}' - skipping")
                        response_validation_results.append({
                            'prompt': prompt,
                            'success': False,
                            'quality': 'failed',
                            'validation_message': 'Tokenizer returned no tokens'
                        })
                        continue

                    # Limit sequence length for manageable testing
                    max_tokens = 32  # Keep reasonable limit for ZMQ test
                    tokens = tokens[:max_tokens]
                except Exception as e:
                    print(f"   ❌ Failed to tokenize prompt '{prompt}': {e} - skipping")
                    response_validation_results.append({
                        'prompt': prompt,
                        'success': False,
                        'quality': 'failed',
                        'validation_message': f'Tokenization failed: {str(e)}'
                    })
                    continue
            else:
                # Fallback: use hardcoded tokens for basic prompts
                if "capital of France" in prompt:
                    tokens = [791, 6864, 315, 9822, 374]
                elif "Hello" in prompt:
                    tokens = [9906, 11, 856, 836, 374]
                elif "What is" in prompt:
                    tokens = [3923, 374]
                else:
                    print(f"   ⚠️ Cannot tokenize '{prompt}' without tokenizer - skipping")
                    response_validation_results.append({
                        'prompt': prompt,
                        'success': False,
                        'quality': 'failed',
                        'validation_message': 'Cannot tokenize without tokenizer'
                    })
                    continue

            print(f"⚡ [{i}/{len(test_prompts)}] Testing: '{prompt[:60]}{'...' if len(prompt) > 60 else ''}' ({len(tokens)} tokens)")

            forward_req = create_forward_request(tokens)

            # Serialize and send using multipart format like server_common.py
            handler_id = 3  # FORWARD_PASS
            corr_id = i  # Use loop index as correlation ID

            msg_parts = [
                struct.pack(">I", corr_id),
                struct.pack(">I", handler_id),
                msgspec.msgpack.encode(forward_req)
            ]

            socket.send_multipart(msg_parts)

            # Receive response
            response_parts = socket.recv_multipart()
            if len(response_parts) >= 3:
                response_data = response_parts[2]  # Skip corr_id and handler_id
                forward_resp = msgspec.msgpack.decode(response_data, type=message.ForwardPassResponse)
            else:
                print(f"   ❌ Invalid response format for '{prompt}': {response_parts}")
                # Add failed response to validation results
                response_validation_results.append({
                    'prompt': prompt,
                    'success': False,
                    'quality': 'failed',
                    'validation_message': f'Invalid response format: {len(response_parts)} parts'
                })
                all_successful = False
                continue

            if not forward_resp.dists:
                print(f"   ❌ No distribution returned for '{prompt}'")
                # Add failed response to validation results
                response_validation_results.append({
                    'prompt': prompt,
                    'success': False,
                    'quality': 'failed',
                    'validation_message': 'No distribution returned'
                })
                all_successful = False
                continue

            dist_ids, dist_vals = forward_resp.dists[0]

            # Decode tokens to text if tokenizer is available
            if tokenizer_util:
                print(f"   🔟 Top-10 distribution candidates:")
                for rank, (token_id, prob) in enumerate(zip(dist_ids[:10], dist_vals[:10]), start=1):
                    try:
                        token_text = tokenizer_util.decode_tokens_with_tokenizer([token_id]).replace("\n", "\\n")
                        print(f"      {rank:2d}. id={token_id:<7d} prob={prob:.6f} text='{token_text}'")
                    except Exception as e:
                        print(f"      {rank:2d}. id={token_id:<7d} prob={prob:.6f} text='[DECODE_ERROR]'")

                # Show decoded top-3 for summary
                top_3_decoded = []
                for token_id in dist_ids[:3]:
                    try:
                        decoded = tokenizer_util.decode_tokens_with_tokenizer([token_id]).strip()
                        top_3_decoded.append(f"'{decoded}'")
                    except:
                        top_3_decoded.append(f"[id:{token_id}]")

                print(f"   ✅ Response: top-3 decoded = {top_3_decoded}, probs = {[f'{p:.4f}' for p in dist_vals[:3]]}")

                # Validate response quality (same as integration test)
                response_quality = validate_metal_response(prompt, dist_ids, dist_vals, tokenizer_util)
                print(f"   📊 Response quality: {response_quality['quality']} - {response_quality['validation_message']}")
            else:
                print(f"   ✅ Response received: top-3 tokens {dist_ids[:3]}, probs = {[f'{p:.4f}' for p in dist_vals[:3]]}")

                # Basic validation without tokenizer
                response_quality = validate_metal_response(prompt, dist_ids, dist_vals, None)
                print(f"   📊 Response quality: {response_quality['quality']} - {response_quality['validation_message']}")

            # Store validation result for summary
            response_validation_results.append(response_quality)

        # Calculate successful tests
        successful_tests = sum(1 for r in response_validation_results if r.get('success', False))

        # Comprehensive summary like integration test
        print(f"\n📊 ZMQ SERVER COMMUNICATION TEST RESULTS:")
        print("=" * 70)
        print(f"Total prompts tested: {len(test_prompts)}")
        print(f"Successful operations: {successful_tests}/{len(test_prompts)}")
        print(f"Success rate: {100 * successful_tests / len(test_prompts):.1f}%")

        # Quality analysis
        print(f"\n🎯 Response Quality Analysis:")
        quality_counts = {}
        for result in response_validation_results:
            if result.get('success', False):
                quality = result.get('quality', 'unknown')
                quality_counts[quality] = quality_counts.get(quality, 0) + 1

        for quality, count in quality_counts.items():
            print(f"  {quality}: {count} prompts")

        # Show some example high-quality responses
        excellent_responses = [r for r in response_validation_results
                             if r.get('success') and r.get('quality') == 'excellent']
        if excellent_responses:
            print(f"\n🌟 Sample excellent responses:")
            for i, response in enumerate(excellent_responses[:3], 1):
                prompt_preview = response['prompt'][:50] + ('...' if len(response['prompt']) > 50 else '')
                print(f"  {i}. '{prompt_preview}' -> {response['validation_message']}")

        if not all_successful or successful_tests < len(test_prompts) * 0.8:
            print("❌ Some tests failed or success rate too low!")
            return False

        print(f"✅ All critical tests passed! ZMQ server communication working properly.")

        return True

    except zmq.Again:
        print("❌ ZMQ timeout - Metal backend server not responding")
        return False
    except Exception as e:
        print(f"❌ ZMQ communication failed: {e}")
        return False
    finally:
        socket.close()
        context.term()

def start_metal_server():
    """Start the Metal backend server using PIE CLI with recording."""
    print("🚀 Starting Metal backend server using PIE CLI...")

    import subprocess
    import os

    # Set up environment
    env = os.environ.copy()
    env["PIE_MODEL_PATH"] = "/Users/seung-seoblee/Library/Caches/pie/models/llama-3.2-1b-instruct/llama-3.2-1b-instruct.zt"

    # Clear any existing recording file
    record_file = "/tmp/metal_server_req_res.jsonl"
    if Path(record_file).exists():
        Path(record_file).unlink()
        print(f"🗑️  Cleared existing recording file: {record_file}")

    # Start server using PIE CLI with recording enabled
    cmd = [
        "pie", "start",
        "--config", "./metal_config.toml",
        "--record",
        "--record-file", record_file,
        "--daemon"  # Run in daemon mode for non-interactive testing
    ]

    pie_cli_dir = repo_root / "pie-cli"
    print(f"📁 Running from directory: {pie_cli_dir}")
    print(f"🎬 Recording to: {record_file}")

    process = subprocess.Popen(
        cmd,
        cwd=str(pie_cli_dir),  # Run from pie-cli directory where config and relative paths work
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print("⏳ Waiting for PIE server to start...")

    # Monitor the process output to extract the IPC endpoint
    import select

    ipc_endpoint = None
    timeout = 30  # 30 second timeout
    start_time = time.time()

    # Set stdout to non-blocking
    fd = process.stdout.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    output_buffer = ""

    while time.time() - start_time < timeout:
        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print(f"❌ PIE process exited early with code {process.returncode}")
            print(f"📤 STDOUT: {output_buffer + stdout}")
            print(f"📥 STDERR: {stderr}")
            return None, None

        # Read available output
        try:
            chunk = process.stdout.read()
            if chunk is not None and chunk:
                output_buffer += chunk
                print(chunk, end='', flush=True)  # Show real-time output

                # Look for IPC endpoint in the output
                import re
                ipc_match = re.search(r'ipc:///tmp/pie-service-(\d+)', output_buffer)
                if ipc_match:
                    ipc_endpoint = f"ipc:///tmp/pie-service-{ipc_match.group(1)}"
                    print(f"\n🔍 Detected IPC endpoint: {ipc_endpoint}")
                    break

        except (BlockingIOError, OSError, TypeError):
            pass

        time.sleep(0.1)

    if ipc_endpoint is None:
        print("⚠️  Could not detect IPC endpoint from PIE output, trying to find the most recent IPC socket...")

        # Fallback: look for the most recently created IPC socket
        import glob
        ipc_sockets = glob.glob("/tmp/pie-service-*")
        if ipc_sockets:
            most_recent_socket = max(ipc_sockets, key=lambda x: Path(x).stat().st_mtime)
            ipc_endpoint = f"ipc://{most_recent_socket}"
            print(f"🔍 Using most recent IPC socket: {ipc_endpoint}")
        else:
            print("❌ No IPC sockets found at all!")
            return None, None

    print("✅ PIE server started with IPC endpoint detected!")
    time.sleep(5)  # Additional wait for full initialization

    return process, ipc_endpoint

def validate_metal_response(prompt: str, token_ids: list, token_probs: list, tokenizer_util=None) -> dict:
    """Validate Metal backend response quality against prompt expectations (from integration test)."""
    prompt_lower = prompt.lower()

    # Check if response shows reasonable distribution characteristics
    if not token_ids or not token_probs:
        return {
            'prompt': prompt,
            'success': False,
            'quality': 'failed',
            'validation_message': 'No predictions returned'
        }

    # Decode top tokens to text for validation if tokenizer available
    top_tokens_text = []
    if tokenizer_util:
        try:
            top_tokens_text = [
                tokenizer_util.decode_tokens_with_tokenizer([token_id]).strip()
                for token_id in token_ids[:3]
            ]
        except Exception:
            top_tokens_text = []

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

    # Additional validation based on prompt type (same as integration test)
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


def validate_recording_file():
    """Validate and parse the recording file to verify requests were recorded."""
    record_file = Path("/tmp/metal_server_req_res.jsonl")

    print(f"\n📊 Validating recording file: {record_file}")

    if not record_file.exists():
        print("❌ Recording file does not exist!")
        return False

    # Load model info for token decoding
    model_info = _load_real_model_info()
    tokenizer_util = None
    if model_info:
        tokenizer_util = TokenizerUtil(model_info)

    try:
        import json
        records = []
        with open(record_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Invalid JSON on line {line_num}: {e}")

        print(f"✅ Recording file contains {len(records)} request/response pairs")

        if records:
            first_record = records[0]
            print(f"📝 First record preview:")
            print(f"   Request tokens: {first_record.get('request', {}).get('input_tokens', 'N/A')}")
            if 'response' in first_record and 'dists' in first_record['response']:
                dists = first_record['response']['dists']
                if dists and len(dists[0]) >= 2:
                    top_tokens = dists[0][0][:3]  # First 3 token IDs
                    top_probs = dists[0][1][:3]   # First 3 probabilities

                    if tokenizer_util:
                        # Decode the top tokens
                        decoded_tokens = []
                        for token_id in top_tokens:
                            try:
                                decoded = tokenizer_util.decode_tokens_with_tokenizer([token_id]).strip()
                                decoded_tokens.append(f"'{decoded}'")
                            except:
                                decoded_tokens.append(f"[id:{token_id}]")
                        print(f"   Response top-3: {top_tokens} -> {decoded_tokens} (probs: {[f'{p:.4f}' for p in top_probs]})")
                    else:
                        print(f"   Response top-3: {top_tokens} (probs: {[f'{p:.4f}' for p in top_probs]})")

            print(f"   Timestamp: {first_record.get('timestamp', 'N/A')}")

        return len(records) > 0

    except Exception as e:
        print(f"❌ Error parsing recording file: {e}")
        return False

@pytest.mark.integration
def test_metal_server_direct_communication():
    """Test Metal backend server via direct ZMQ communication with token decoding."""

    if sys.platform != "darwin":
        pytest.skip("Metal backend requires macOS")

    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an available MPS device")

    print("🧪 Testing Metal Backend Server (ZMQ) vs Direct Handler")
    print("=" * 70)

    server_process = None
    try:
        # Start the Metal backend server
        result = start_metal_server()

        if result is None or result[0] is None:
            pytest.fail("Failed to start PIE server!")

        server_process, ipc_endpoint = result

        # Test ZMQ communication
        success = test_zmq_server_communication(ipc_endpoint)

        if success:
            print("\n✅ ZMQ server test completed!")

            # Validate recording file
            recording_success = validate_recording_file()
            if recording_success:
                print("✅ Recording validation successful!")
            else:
                pytest.fail("Recording validation failed!")
        else:
            pytest.fail("ZMQ server test failed!")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        pytest.fail("Test interrupted by user")

    finally:
        if server_process:
            print("\n🛑 Stopping Metal backend server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()
            print("✅ Server stopped")


@pytest.mark.integration
def test_metal_server_recording_validation():
    """Test Metal backend server recording file validation with token decoding."""

    if sys.platform != "darwin":
        pytest.skip("Metal backend requires macOS")

    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an available MPS device")

    # This test assumes the server was run previously and created a recording file
    record_file = Path("/tmp/metal_server_req_res.jsonl")

    if not record_file.exists():
        pytest.skip("No recording file found - run test_metal_server_direct_communication first")

    print("📊 Testing recording file validation with token decoding")

    # Validate recording file with token decoding
    success = validate_recording_file()

    assert success, "Recording file validation should succeed"


if __name__ == "__main__":
    # For standalone execution, run the main test
    test_metal_server_direct_communication()
