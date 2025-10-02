# pylint: disable-all
# type: ignore
# Ignoring checks for pylint and pyright since we are actively working on this file

from typing import Optional
import fire
from common import (
    build_config,
    print_config,
    start_service,
)
from handler import Handler


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
    max_num_embeds: int = 128,
    max_batch_tokens: int = 10240,
    max_num_adapters: int = 48,
    max_adapter_rank: int = 8,
    max_num_kv_pages: Optional[int] = None,
    gpu_mem_headroom: Optional[float] = None,
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
        max_batch_tokens: Maximum number of tokens in a batch.
        max_num_adapters: Maximum number of adapters that can be loaded.
        max_adapter_rank: Maximum rank for any loaded adapter.
        device: The device to run the model on (e.g., 'cuda:0' or 'cpu').
        dtype: The data type for model weights (e.g., 'bfloat16', 'float16').
    """
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
        max_num_embeds=max_num_embeds,
        max_batch_tokens=max_batch_tokens,
        max_num_adapters=max_num_adapters,
        max_adapter_rank=max_adapter_rank,
        max_num_kv_pages=max_num_kv_pages,
        gpu_mem_headroom=gpu_mem_headroom,
        device=device,
        dtype=dtype,
    )

    print_config(config)

    start_service(
        config=config,
        handler_cls=Handler,
    )


if __name__ == "__main__":
    fire.Fire(main)
