import sys
import torch
from pie_backend.engine import Engine
from pie_backend.config import RuntimeConfig
from pie_backend import message


def test_manual():
    print("Initializing Runtime...")
    try:
        config = RuntimeConfig.from_args(model="llama-3.2-1b-instruct")
        engine = Engine.load(config)
        print("Runtime initialized.")
    except Exception as e:
        print(f"Failed to initialize runtime: {e}")
        return

    # Create a dummy request
    # Single token input (BOS=128001 for Llama 3 usually, but 1 is fine for test)
    # Mask [1] means 1 True (attend to self)
    req = message.ForwardPassRequest(
        input_tokens=[1],
        input_token_positions=[0],
        input_embed_ptrs=[],
        input_embed_positions=[],
        adapter=None,
        adapter_seed=None,
        mask=[[1]],
        kv_page_ptrs=[0],
        kv_page_last_len=1,
        output_token_indices=[0],
        output_token_samplers=[
            {"sampler": 3, "top_k": 1, "temperature": 1.0}
        ],  # Top-K=1
        output_embed_ptrs=[],
        output_embed_indices=[],
    )

    print("Sending ForwardPassRequest...")
    try:
        responses = engine.forward_pass_handler([req])
        print(f"Received {len(responses)} responses.")

        if responses:
            resp = responses[0]
            print(f"Response tokens: {resp.tokens}")
            print(f"Response dists: {resp.dists}")

            if resp.tokens and len(resp.tokens) == 1:
                print("SUCCESS: Generated a token.")
            else:
                print("FAILURE: Did not generate a token.")
        else:
            print("FAILURE: No response received.")

    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_manual()
