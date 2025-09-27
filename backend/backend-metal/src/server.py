#!/usr/bin/env python3
"""
Fixed Metal backend server for PIE CLI.
This solves the import path conflict by importing Metal modules before PIE setup.
"""

import sys
import os
from pathlib import Path

def main():
    # Get repo root (we're in backend/backend-metal/src, so go up 3 levels)
    repo_root = Path(__file__).resolve().parents[3]

    # Add Metal backend src directory FIRST, before any other imports
    metal_src = Path(__file__).resolve().parent  # We're already in the src directory
    sys.path.insert(0, str(metal_src))

    # Import Metal modules directly BEFORE setup_pie_imports
    try:
        from l4ma_runtime import MetalL4maBackend
        from metal_backend import MetalBackend
        from metal_handler import MetalHandler

        # Import the Metal model factory EXPLICITLY from the right location
        import importlib.util
        metal_factory_path = metal_src / "model_factory.py"
        spec = importlib.util.spec_from_file_location("metal_model_factory", metal_factory_path)
        if spec is not None and spec.loader is not None:
            metal_model_factory = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metal_model_factory)
            create_model_and_fusion_map = metal_model_factory.create_model_and_fusion_map
        else:
            raise ImportError("Failed to load Metal model factory")

    except Exception as e:
        print(f"❌ Failed to import Metal modules: {e}")
        sys.exit(1)

    # Now set up PIE imports for common functionality
    sys.path.insert(0, str(repo_root))
    from repo_utils import setup_pie_imports
    setup_pie_imports()

    # Import common PIE functionality
    from config.common import ModelInfo, ModelConfig
    from model_loader import load_model
    from server_common import start_service, build_config, print_config

    # Import debug registration
    from debug_server_common import register_with_debug

    import fire
    import torch
    from dataclasses import asdict

    def server_main(
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
        device: str = "mps",
        dtype: str = "bfloat16",
    ):
        print("🚀 Starting Metal Backend Server for PIE CLI")
        print(f"   Model: {model}")
        print(f"   Device: {device}")
        print(f"   Port: {port}")

        if not MetalL4maBackend.is_available():
            raise RuntimeError("Metal runtime not available")

        config = build_config(
            model=model,
            host=host,
            port=port,
            controller_host=controller_host,
            controller_port=controller_port,
            auth_token=auth_token,
            cache_dir=cache_dir,
            kv_page_size=kv_page_size,
            max_dist_size=max_dist_size,
            max_num_kv_pages=max_num_kv_pages,
            max_num_embeds=max_num_embeds,
            max_num_adapters=max_num_adapters,
            max_adapter_rank=max_adapter_rank,
            device=device,
            dtype=dtype,
        )

        print_config(config)

        model_instance, model_metadata = load_model(config, create_model_and_fusion_map)

        # Explicitly convert model to correct device and dtype
        # This ensures the entire model is properly converted, not just individual tensors
        target_dtype = getattr(torch, dtype)
        print(f"🔧 Converting model to device={device}, dtype={target_dtype}")

        # Log model state before conversion
        first_param = next(model_instance.parameters())
        print(f"   Before conversion: device={first_param.device}, dtype={first_param.dtype}")

        model_instance = model_instance.to(device=device, dtype=target_dtype)
        model_instance.eval()  # Ensure model is in eval mode

        # Log model state after conversion
        first_param = next(model_instance.parameters())
        print(f"   After conversion: device={first_param.device}, dtype={first_param.dtype}")

        # Enable controller registration when controller_host/port are provided
        # This allows the backend to work both standalone (for testing) and with PIE engine
        should_register = config.get("controller_host") is not None and config.get("controller_port") is not None

        start_service(
            config=config,
            handler_cls=MetalHandler,
            model=model_instance,
            model_info=model_metadata,
            register_with_controller=should_register,
        )

    # Run with fire
    fire.Fire(server_main)

if __name__ == "__main__":
    main()