# pylint: disable-all
# type: ignore
# Ignoring checks for pylint and pyright since we are actively working on this file

import enum
import os
import random
import struct
import sys
import threading
import time
from pathlib import Path

import fire
import msgpack
import msgspec
import torch
import ztensor
import zmq
from platformdirs import user_cache_dir
from tqdm import tqdm
from websockets.sync.client import connect

from config.common import ModelInfo
from handler import Handler
from model.gptoss import GPTOSSForCausalLM, GPTOSSTensorLoader
from model.l4ma import L4maForCausalLM, L4maTensorLoader
from model.qwen3 import Qwen3ForCausalLM, Qwen3TensorLoader
from message import (
    EmbedImageRequest,
    ForwardPassRequest,
    HandshakeRequest,
    InitializeAdapterRequest,
    QueryRequest,
    UpdateAdapterRequest,
    HeartbeatRequest,
    UploadAdapterRequest,
    DownloadAdapterRequest,
)


class HandlerId(enum.Enum):
    HANDSHAKE = 0
    HEARTBEAT = 1
    QUERY = 2
    FORWARD_PASS = 3
    EMBED_IMAGE = 4
    INITIALIZE_ADAPTER = 5
    UPDATE_ADAPTER = 6
    UPLOAD_HANDLER = 7
    DOWNLOAD_HANDLER = 8


def main(
    model: str,
    host: str = "localhost",
    port: int = 10123,
    controller_host: str = "localhost",
    controller_port: int = 9123,
    auth_token: str = None,
    cache_dir: str = None,
    kv_page_size: int = 16,
    max_dist_size: int = 64,
    max_num_kv_pages: int = 1024,
    max_num_embeds: int = 128,
    max_num_adapters: int = 48,
    max_adapter_rank: int = 8,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
):
    """
    Runs the application with configuration provided as command-line arguments.

    Args:
        model: Name of the model to load (required).
        host: Hostname for the ZMQ service to bind to.
        port: Port for the ZMQ service to bind to.
        controller_host: Hostname of the controller to register with.
        controller_port: Port of the controller to register with.
        auth_token: Authentication token for connecting to the controller.
        cache_dir: Directory for model cache. Defaults to PIE_HOME env var,
                   then the platform-specific user cache dir.
        kv_page_size: The size of each page in the key-value cache.
        max_dist_size: Maximum distance for embeddings.
        max_num_kv_pages: Maximum number of pages in the key-value cache.
        max_num_embeds: Maximum number of embeddings to store.
        max_num_adapters: Maximum number of adapters that can be loaded.
        max_adapter_rank: Maximum rank for any loaded adapter.
        device: The device to run the model on (e.g., 'cuda:0' or 'cpu').
        dtype: The data type for model weights (e.g., 'bfloat16', 'float16').
    """
    # Resolve cache_dir using the precedence: CLI arg > Environment Var > Platform Default
    resolved_cache_dir = (
        cache_dir or os.environ.get("PIE_HOME") or str(Path(user_cache_dir("pie")))
    )

    # Create a config dictionary from function arguments for downstream use.
    # The 'locals()' function conveniently captures all arguments.
    config = locals()
    config["cache_dir"] = resolved_cache_dir  # Update with the resolved path

    print("--- Configuration ---")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("----------------------")

    model_instance, model_metadata = load_model(config)
    start_service(config, model_instance, model_metadata)


def load_model(config: dict):
    model_name = config["model"]
    cache_dir = config["cache_dir"]
    model_path = Path(cache_dir) / "models"
    metadata_path = model_path / f"{model_name}.toml"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    model_device = config["device"]
    model_dtype = getattr(torch, config["dtype"])
    model_info = ModelInfo.load_from_file(str(metadata_path), model_device, model_dtype)

    # Instantiate model and create tensor loader based on architecture
    match model_info.architecture.type.lower():
        case "qwen3":
            model = Qwen3ForCausalLM(model_info.architecture)
            tensor_loader = Qwen3TensorLoader(model)
        case "l4ma":
            model = L4maForCausalLM(model_info.architecture)
            tensor_loader = L4maTensorLoader(model)
        case "gptoss":
            model = GPTOSSForCausalLM(model_info.architecture)
            tensor_loader = GPTOSSTensorLoader(model)
        case _:
            raise ValueError(
                f"Unsupported architecture type: {model_info.architecture.type}"
            )

    # Prepare for loading
    model_state_keys = set(model.state_dict().keys())
    loaded_keys = set()

    # Track which checkpoint file each tensor comes from
    checkpoint_tensor_to_file = {}

    # First pass: scan all checkpoint files to build tensor-to-file mapping
    print("Scanning checkpoint files...")
    try:
        for param_file in model_info.parameters:
            weights_path = model_path / model_name / param_file
            with ztensor.Reader(str(weights_path)) as reader:
                tensor_names = reader.get_tensor_names()
                pbar_desc = (
                    f"Scanning {param_file[:30]}..."
                    if len(param_file) > 30
                    else f"Scanning {param_file}"
                )
                for name in tqdm(tensor_names, desc=pbar_desc, unit="tensors"):
                    checkpoint_tensor_to_file[name] = weights_path

        # Second pass: process all runtime tensors using the tensor loader
        print("\nProcessing runtime tensors...")

        for runtime_tensor_name in tqdm(
            model_state_keys, desc="Processing tensors", unit="tensors"
        ):
            if runtime_tensor_name in loaded_keys:
                continue  # Skip already loaded tensors

            # Query which checkpoint tensors are needed
            required_checkpoint_tensors = tensor_loader.query(runtime_tensor_name)

            # Load required checkpoint tensors on demand
            available_tensors = {}
            missing_tensors = []

            for checkpoint_name in required_checkpoint_tensors:
                if checkpoint_name not in checkpoint_tensor_to_file:
                    missing_tensors.append(checkpoint_name)
                    continue

                try:
                    checkpoint_file_path = checkpoint_tensor_to_file[checkpoint_name]
                    with ztensor.Reader(str(checkpoint_file_path)) as reader:
                        available_tensors[checkpoint_name] = reader.read_tensor(
                            checkpoint_name, to="torch"
                        )
                except Exception as e:
                    print(
                        f"    Error loading checkpoint tensor '{checkpoint_name}': {e}"
                    )
                    missing_tensors.append(checkpoint_name)

            if missing_tensors:
                # Weight tying for lm_head.weight is handled separately
                if runtime_tensor_name != "lm_head.weight":
                    print(
                        f"    Warning: Missing checkpoint tensors for '{runtime_tensor_name}': {missing_tensors}"
                    )
                continue

            # Load the runtime tensor using the tensor loader
            try:
                runtime_tensor = tensor_loader.load(
                    runtime_tensor_name, available_tensors
                )
                param = model.state_dict()[runtime_tensor_name]

                if runtime_tensor.shape != param.shape:
                    print(
                        f"    Warning: Shape mismatch for tensor '{runtime_tensor_name}'. "
                        f"Expected {param.shape}, got {runtime_tensor.shape}. Skipping."
                    )
                    continue

                with torch.no_grad():
                    param.copy_(runtime_tensor, non_blocking=True)
                loaded_keys.add(runtime_tensor_name)

            except Exception as e:
                print(f"    Error loading tensor '{runtime_tensor_name}': {e}")
                continue

        # Handle weight tying for lm_head
        if "lm_head.weight" in model_state_keys and "lm_head.weight" not in loaded_keys:
            model.state_dict()["lm_head.weight"].copy_(
                model.model.embed_tokens.weight, non_blocking=True
            )
            loaded_keys.add("lm_head.weight")

        missing_keys = model_state_keys - loaded_keys
        if missing_keys:
            print("\nWarning: Some model weights were not found in any parameter file:")
            for key in sorted(list(missing_keys)):
                print(f"  - {key}")
        else:
            print("\nSuccessfully loaded all expected model weights.")

        model.eval()
        return model, model_info

    except ztensor.ZTensorError as e:
        print(
            f"Fatal Error: Failed to read a ztensor file. Error: {e}", file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        print(
            f"An unexpected fatal error occurred during model loading: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def start_service(config, model, model_info):
    """
    Initializes and starts the service, including the ZMQ server and registration threads.
    """
    if config["controller_host"] in ["127.0.0.1", "localhost"]:
        unique_id = random.randint(1000, 9999)
        endpoint = f"ipc:///tmp/pie-service-{unique_id}"
        real_endpoint = endpoint
    else:
        endpoint = f"tcp://{config['host']}:{config['port']}"
        real_endpoint = f"tcp://*:{config['port']}"

    handler = Handler(
        model=model,
        model_info=model_info,
        kv_page_size=config["kv_page_size"],
        max_dist_size=config["max_dist_size"],
        max_num_kv_pages=config["max_num_kv_pages"],
        max_num_embeds=config["max_num_embeds"],
        max_num_adapters=config["max_num_adapters"],
        max_adapter_rank=config["max_adapter_rank"],
        dtype=getattr(torch, config["dtype"]),
        device=config["device"],
    )

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(real_endpoint)

    # Start ZMQ server and registration in daemon threads
    threading.Thread(target=run_zmq_server, args=(socket, handler), daemon=True).start()
    threading.Thread(target=register, args=(config, endpoint), daemon=True).start()

    # Keep main thread alive to handle shutdown
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        socket.close()
        context.term()
        print("Server shutdown complete.")


def register(config, endpoint):
    """
    Registers this service with the controller in a separate thread.
    """
    controller_addr = f"ws://{config['controller_host']}:{config['controller_port']}"
    try:
        # MODIFICATION 1: Added 'open_timeout=10' to attempt connection for 10 seconds.
        with connect(controller_addr, open_timeout=10) as websocket:
            # Authenticate with the controller
            websocket.send(
                msgpack.packb(
                    {
                        "type": "authenticate",
                        "corr_id": 0,
                        "token": config["auth_token"],
                    },
                    use_bin_type=True,
                )
            )
            auth_response = msgpack.unpackb(websocket.recv(), raw=False)
            if not auth_response.get("successful"):
                print(
                    f"Authentication failed: {auth_response.get('result', 'Unknown error')}"
                )
                # Use os._exit(1) to terminate the entire process immediately from a thread
                os._exit(1)

            # Register the service endpoint
            websocket.send(
                msgpack.packb(
                    {
                        "type": "attach_remote_service",
                        "corr_id": 0,
                        "endpoint": endpoint,
                        "service_name": config["model"],
                        "service_type": "model",
                    },
                    use_bin_type=True,
                )
            )
            reg_response = msgpack.unpackb(websocket.recv(), raw=False)
            if not reg_response.get("successful"):
                print(
                    f"Controller registration failed: {reg_response.get('result', 'Unknown error')}"
                )
                os._exit(1)

            print(f"Registered with controller at {controller_addr}")

    # MODIFICATION 2: Catch connection errors (including timeout) and exit the program.
    # The 'TimeoutError' is raised by 'open_timeout'.
    except (ConnectionRefusedError, TimeoutError) as e:
        print(
            f"Failed to connect to the controller at {controller_addr} within 10 seconds."
        )
        print(f"Error: {e}")
        print("Please ensure the controller is running and accessible. Terminating.")
        os._exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during registration: {e}. Terminating.")
        os._exit(1)


def run_zmq_server(socket, handler):
    """
    Runs the ZMQ server loop, listening for and processing client requests.
    Exits the program if a heartbeat is not received for 7 seconds or if any
    exception occurs.
    """
    # --- MODIFICATION: Set heartbeat timeout and initialize timer ---
    # Temporarily set to 60 seconds as a temporary fix to avoid the backend exiting
    # during FlashInfer JIT compilation.
    HEARTBEAT_TIMEOUT = 60  # seconds
    last_heartbeat_time = time.monotonic()

    # --- CORRECTION: Keys should be integer values of the enums ---
    DECODERS = {
        HandlerId.HANDSHAKE.value: msgspec.msgpack.Decoder(HandshakeRequest),
        HandlerId.HEARTBEAT.value: msgspec.msgpack.Decoder(HeartbeatRequest),
        HandlerId.QUERY.value: msgspec.msgpack.Decoder(QueryRequest),
        HandlerId.FORWARD_PASS.value: msgspec.msgpack.Decoder(ForwardPassRequest),
        HandlerId.EMBED_IMAGE.value: msgspec.msgpack.Decoder(EmbedImageRequest),
        HandlerId.INITIALIZE_ADAPTER.value: msgspec.msgpack.Decoder(
            InitializeAdapterRequest
        ),
        HandlerId.UPDATE_ADAPTER.value: msgspec.msgpack.Decoder(UpdateAdapterRequest),
        HandlerId.UPLOAD_HANDLER.value: msgspec.msgpack.Decoder(UploadAdapterRequest),
        HandlerId.DOWNLOAD_HANDLER.value: msgspec.msgpack.Decoder(
            DownloadAdapterRequest
        ),
    }
    MSGPACK_ENCODER = msgspec.msgpack.Encoder()

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    try:
        while True:
            # Check for heartbeat timeout before waiting for a message
            if time.monotonic() - last_heartbeat_time > HEARTBEAT_TIMEOUT:
                os._exit(1)  # Use os._exit for immediate termination from a thread

            # Poll for 1 second to remain responsive to the heartbeat check
            events = dict(poller.poll(timeout=1000))
            if socket in events:
                message = socket.recv_multipart()
            else:
                # Poller timed out, loop again to re-check the heartbeat timer
                continue

            if len(message) < 3:
                print(f"[!] Received invalid message: {message}")
                continue

            client_identity, corr_id_bytes, handler_id_bytes = message[:3]
            try:
                corr_id = struct.unpack(">I", corr_id_bytes)[0]
                handler_id = struct.unpack(">I", handler_id_bytes)[0]
                reqs = [DECODERS[handler_id].decode(m) for m in message[3:]]
            except (struct.error, KeyError, msgspec.DecodeError) as e:
                print(f"[!] Error decoding request header or payload: {e}")
                continue

            if not reqs:
                print(f"[!] Received empty request body")
                continue

            resps = []
            # The match statement correctly compares the integer handler_id with enum values
            match handler_id:
                case HandlerId.HANDSHAKE.value:
                    resps = handler.handshake(reqs)
                case HandlerId.HEARTBEAT.value:
                    # print(f"[*] Heartbeat received at {time.time()}")
                    last_heartbeat_time = time.monotonic()
                    resps = handler.heartbeat(reqs)
                case HandlerId.QUERY.value:
                    resps = handler.query(reqs)
                case HandlerId.FORWARD_PASS.value:
                    resps = handler.forward_pass(reqs)
                case HandlerId.EMBED_IMAGE.value:
                    handler.embed_image(reqs)
                case HandlerId.INITIALIZE_ADAPTER.value:
                    handler.initialize_adapter(reqs)
                case HandlerId.UPDATE_ADAPTER.value:
                    handler.update_adapter(reqs)
                case HandlerId.UPLOAD_HANDLER.value:
                    handler.upload_handler(reqs)
                case HandlerId.DOWNLOAD_HANDLER.value:
                    resps = handler.download_handler(reqs)
                case _:
                    print(f"[!] Unknown handler ID: {handler_id}", file=sys.stderr)

            if resps:
                response_msg = [client_identity, corr_id_bytes, handler_id_bytes] + [
                    MSGPACK_ENCODER.encode(r) for r in resps
                ]
                socket.send_multipart(response_msg)

    except Exception as e:
        print(
            f"\n[!!!] A fatal, unhandled error occurred in the ZMQ server loop: {e}",
        )
        import traceback

        traceback.print_exc()
        os._exit(1)


if __name__ == "__main__":
    # log_file = open("service.log", "w", buffering=1)

    # Redirect stdout and stderr to the log file
    # sys.stdout = log_file
    # sys.stderr = log_file
    # --------------------------
    fire.Fire(main)
