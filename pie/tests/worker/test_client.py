"""
Test client for PIE backend server.

This module provides a mock client that connects to the server via ZMQ
and sends test requests to verify server functionality.
"""

from __future__ import annotations

import argparse
import struct
import time
from enum import IntEnum

import msgspec
import zmq

from src.message import (
    QueryRequest,
    QueryResponse,
    ForwardPassRequest,
    ForwardPassResponse,
)


class HandlerId(IntEnum):
    """Handler message types matching server.py."""

    QUERY = 2
    FORWARD_PASS = 3
    EMBED_IMAGE = 4
    INITIALIZE_ADAPTER = 5
    UPDATE_ADAPTER = 6
    UPLOAD_HANDLER = 7
    DOWNLOAD_HANDLER = 8


class TestClient:
    """Mock client for testing the PIE backend server."""

    socket: zmq.Socket
    context: zmq.Context
    encoder: msgspec.msgpack.Encoder
    corr_id: int

    def __init__(self, endpoint: str):
        """
        Initialize the test client and connect to the server.

        Args:
            endpoint: ZMQ endpoint to connect to (e.g., "ipc:///tmp/pie-model-service-1234567")
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.connect(endpoint)
        self.encoder = msgspec.msgpack.Encoder()
        self.corr_id = 0
        print(f"[TestClient] Connected to {endpoint}")

    def close(self):
        """Close the client connection."""
        self.socket.close()
        self.context.term()
        print("[TestClient] Connection closed")

    def _send_request(self, handler_id: HandlerId, requests: list) -> list[bytes]:
        """
        Send a request to the server and wait for response.

        Args:
            handler_id: The handler type to invoke
            requests: List of request objects to send

        Returns:
            List of raw response message parts (excluding headers)
        """
        self.corr_id += 1
        corr_id_bytes = struct.pack(">I", self.corr_id)
        handler_id_bytes = struct.pack(">I", handler_id)

        # Build message: [corr_id, handler_id, *encoded_requests]
        message_parts = [corr_id_bytes, handler_id_bytes]
        for req in requests:
            message_parts.append(self.encoder.encode(req))

        self.socket.send_multipart(message_parts)

        # Wait for response
        response = self.socket.recv_multipart()

        # Response format: [corr_id, handler_id, *encoded_responses]
        if len(response) < 2:
            raise RuntimeError(f"Invalid response: {response}")

        return response[2:]  # Skip corr_id and handler_id


    def send_query(self, query: str) -> list[QueryResponse]:
        """Send a query request."""
        req = QueryRequest(query=query)
        print(f"[TestClient] Sending query: {query[:50]}...")

        raw_responses = self._send_request(HandlerId.QUERY, [req])
        decoder = msgspec.msgpack.Decoder(QueryResponse)
        responses = [decoder.decode(r) for r in raw_responses]

        for resp in responses:
            print(f"[TestClient] Query response: {resp.value[:100]}...")
        return responses

    def send_forward_pass(
        self,
        input_tokens: list[int],
        input_token_positions: list[int] | None = None,
        mask: list[list[int]] | None = None,
        kv_page_ptrs: list[int] | None = None,
        kv_page_last_len: int = 0,
        output_token_indices: list[int] | None = None,
        output_token_samplers: list[dict] | None = None,
    ) -> list[ForwardPassResponse]:
        """Send a forward pass request."""
        if input_token_positions is None:
            input_token_positions = list(range(len(input_tokens)))
        if mask is None:
            # Default causal mask: each token attends to all previous tokens
            mask = [[i + 1] for i in range(len(input_tokens))]
        if kv_page_ptrs is None:
            kv_page_ptrs = [0]
        if output_token_indices is None:
            output_token_indices = [len(input_tokens) - 1]
        if output_token_samplers is None:
            output_token_samplers = [{"sampler": 1, "temperature": 1.0}]

        req = ForwardPassRequest(
            input_tokens=input_tokens,
            input_token_positions=input_token_positions,
            input_embed_ptrs=[],
            input_embed_positions=[],
            adapter=None,
            adapter_seed=None,
            mask=mask,
            kv_page_ptrs=kv_page_ptrs,
            kv_page_last_len=kv_page_last_len,
            output_token_indices=output_token_indices,
            output_token_samplers=output_token_samplers,
            output_embed_ptrs=[],
            output_embed_indices=[],
        )

        print(f"[TestClient] Sending forward pass with {len(input_tokens)} tokens")
        raw_responses = self._send_request(HandlerId.FORWARD_PASS, [req])
        decoder = msgspec.msgpack.Decoder(ForwardPassResponse)
        responses = [decoder.decode(r) for r in raw_responses]

        for resp in responses:
            print(f"[TestClient] Forward pass response: tokens={resp.tokens}")
        return responses


def run_embedded_tests(endpoint: str, startup_delay: float = 1.0):
    """
    Run tests embedded in the server process.

    This is called from the server when --test flag is passed.
    Waits for server startup, runs tests, then terminates the process.

    Args:
        endpoint: ZMQ endpoint to connect to
        startup_delay: Seconds to wait for server startup
    """
    import sys

    print(f"\n[TestClient] Waiting {startup_delay}s for server startup...")
    time.sleep(startup_delay)

    try:
        run_test_suite(endpoint)
    except Exception as e:
        print(f"\n[TestClient] Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n[TestClient] All tests complete. Shutting down...")
    import os

    os._exit(0)  # Force exit to terminate all threads


def run_test_suite(endpoint: str):
    """Run a test suite against the server."""
    client = TestClient(endpoint)

    try:
        print("\n" + "=" * 60)
        print("PIE Backend Test Suite")
        print("=" * 60)

        # Test 1: Query
        print("\n--- Test 1: Query ---")
        try:
            responses = client.send_query("Hello, world!")
            print(f"✓ Query successful, got {len(responses)} response(s)")
        except Exception as e:
            print(f"✗ Query failed: {e}")

        # Test 2: Forward Pass
        print("\n--- Test 2: Forward Pass ---")
        try:
            # Simple token sequence
            input_tokens = [1, 2, 3, 4, 5]
            responses = client.send_forward_pass(
                input_tokens=input_tokens,
                kv_page_ptrs=[0],
                kv_page_last_len=5,
            )
            print(f"✓ Forward pass successful, got {len(responses)} response(s)")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")

        print("\n" + "=" * 60)
        print("Test Suite Complete")
        print("=" * 60)

    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="PIE Backend Test Client")
    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="ZMQ endpoint to connect to (e.g., ipc:///tmp/pie-model-service-1234567)",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["query", "forward", "all"],
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()

    if args.test == "all":
        run_test_suite(args.endpoint)
    else:
        client = TestClient(args.endpoint)
        try:
            if args.test == "query":
                client.send_query("Test query from test client")
            elif args.test == "forward":
                client.send_forward_pass([1, 2, 3, 4, 5])
        finally:
            client.close()


if __name__ == "__main__":
    main()
