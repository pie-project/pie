import unittest
import numpy as np
from pie_backend.batching import Batch
from pie_backend.backend import Backend, RuntimeConfig
from unittest.mock import MagicMock


class TestFlattenedMasks(unittest.TestCase):
    def test_flattened_mask_reconstruction(self):
        # 1. Setup Mock Runtime and Config
        config = RuntimeConfig.from_args(
            hf_repo="dummy", kv_page_size=16, activation_dtype="float16", device="cpu"
        )
        runtime = Runtime.__new__(Runtime)
        runtime.config = config
        runtime.adapters = {}

        # 2. Simulate Input Data
        # Scenario: 2 requests
        # Req 1: 2 tokens, context length 32 (2 KV pages)
        # Req 2: 1 token, context length 16 (1 KV page)

        # Construct BRLE masks
        # Mask 1 (Req 1, Token 1): [True]*33 (attend to 32 context + self)
        mask1_run = [33]
        # Mask 2 (Req 1, Token 2): [True]*34
        mask2_run = [34]
        # Mask 3 (Req 2, Token 1): [True]*17 (attend to 16 context + self)
        mask3_run = [17]

        flattened_masks = mask1_run + mask2_run + mask3_run

        # mask_indptr: points to start of each mask in flattened buffer
        # Tokens: Req1_T1 (len 1), Req1_T2 (len 1), Req2_T1 (len 1)
        # Offsets in flattened_masks: 0, 1, 2, 3
        mask_indptr = [0, 1, 2, 3]

        # Batch parameters
        # qo_indptr: [0, 2, 3] (Req 1 has 2 tokens, Req 2 has 1 token)
        qo_indptr = [0, 2, 3]

        # KV Page setup
        # Req 1: 2 pages (32 tokens) -> kv_page_indptr [0, 2], last_len 16
        # Req 2: 1 page (16 tokens) -> kv_page_indptr [2, 3], last_len 16
        kv_page_indptr = [0, 2, 3]
        kv_last_page_lens = [16, 16]

        # Construct args dictionary (simulating msgpack payload with SoA format)
        args = {
            "token_ids": [1, 2, 3],  # dummy
            "position_ids": [31, 31, 15],  # Use valid positions (0-indexed, < seq_len)
            "kv_page_indices": [0, 1, 2],  # dummy
            "kv_page_indptr": kv_page_indptr,
            "kv_last_page_lens": kv_last_page_lens,
            "qo_indptr": qo_indptr,
            "flattened_masks": [32, 32, 16],  # Full masks
            "mask_indptr": mask_indptr,
            "single_token_mode": False,
            "adapter_indices": [None, None],
            "adapter_seeds": [None, None],
            # SoA sampler fields (empty since no samplers in this test)
            "sampler_temperatures": [],
            "sampler_top_k": [],
            "sampler_top_p": [],
            "sampler_min_p": [],
            "sampler_types": [],
            "request_num_samplers": [0, 0],  # No samplers per request
            "flat_output_token_indices": [],
            "output_token_indptr": [0, 0, 0],  # No output indices per request
            "output_embed_ptrs": [[], []],
            "output_embed_indices": [[], []],
        }

        # 3. Execute Batch constructor
        batch = Batch(args, kv_page_size=16, max_dist_size=32000, adapters={})

        # 4. Verify attention masks
        # Expected: Flattened mask of size 80 (32+32+16)
        # Token 1 (Pos 30, valid 31): 31 True, 1 False (causal masking)
        # Token 2 (Pos 31, valid 32): 32 True
        # Token 3 (Pos 15, valid 16): 16 True
        # Total True: 31 + 32 + 16 = 79

        self.assertEqual(batch.attention_masks.size, 80)
        self.assertEqual(np.sum(batch.attention_masks), 79)
        self.assertFalse(
            batch.attention_masks[31], "Token 1 should not attend to Token 2"
        )

        print(
            "Verification Successful: Batched masks reconstructed correctly from flattened inputs."
        )


if __name__ == "__main__":
    unittest.main()
