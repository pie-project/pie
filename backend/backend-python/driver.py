"""TODO: Add module docstring."""

import time
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

from l4m_pb2 import (  # pylint: disable=no-name-in-module
    BatchAllocate,
    BatchDeallocate,
    BatchEmbedText,
    BatchMaskBlock,
    BatchCopyBlock,
    BatchDecodeTokenDistribution,
    BatchSampleTopKRequest,
    BatchSampleTopKResponse,
    ObjectKind,
    SampleTopKResponse,
    ForwardTextResponse,
    BatchFillBlock,
    Distribution,
    BatchSyncResponse, BatchForwardText, BatchForwardTextResponse,
    BatchDebugQueryRequest, BatchDebugQueryResponse, DebugQueryResponse
)

from l4m_vision_pb2 import BatchEmbedImage  # pylint: disable=no-name-in-module


@dataclass
class TextEmbed:
    """TODO: Add class docstring."""

    token_id: int
    position_id: int


@dataclass
class ImageEmbed:
    """TODO: Add class docstring."""

    vec_id: int
    position_id: tuple[int, int]


@dataclass
class Block:
    """TODO: Add class docstring."""

    position_ids: np.ndarray
    occupancy: np.ndarray


# union type
Embed = Union[TextEmbed, ImageEmbed]


class Driver:
    """TODO: Add class docstring."""

    embeds: dict[int, Embed]
    blocks: dict[int, Block]

    # dist_storage: VectorStorage

    def __init__(
            self,
            model,
            kv_page_size: int,
            dist_size: int,
            max_num_kv_pages: int,
            max_num_embeds: int,
            dtype: torch.dtype,
            device: str,
    ):
        """TODO: Add method docstring."""
        self.embeds = {}
        self.blocks = {}

        self.lm = model
        self.kv_page_size = kv_page_size
        self.dist_size = dist_size
        self.max_num_kv_pages = max_num_kv_pages
        self.max_num_embeds = max_num_embeds
        self.dtype = dtype
        self.device = device

        self.kv_cache_at_layer = [
            torch.randn(
                (
                    max_num_kv_pages,
                    2,
                    kv_page_size,
                    self.lm.config.num_key_value_heads,
                    self.lm.config.head_size,
                ),
                dtype=dtype,
                device=device,
            )
            for _ in range(self.lm.config.num_layers)
        ]

        self.embed_storage_p1 = torch.empty(
            (max_num_embeds, dist_size), device=device, dtype=dtype
        )

        self.embed_storage_p2 = torch.empty(
            (max_num_embeds, dist_size), device=device, dtype=torch.int32
        )

        # self.dist_storage = dist_storage

        self.inter_fill_time = time.time()

    def allocate(self, cmds: BatchAllocate):
        """TODO: Add method docstring."""
        # in current implementation, all allocations are already done in the constructor.
        # but in the future, we may want to allocate more blocks than the GPU capacity,
        # by offloading some of the blocks to the CPU memory.
        # This logic should handle that case.

        for cmd in cmds.items:
            if cmd.kind == ObjectKind.OBJECT_KIND_KV_BLOCK:
                for i in range(cmd.count):
                    self.blocks[cmd.object_id_offset + i] = Block(
                        position_ids=np.array([0] * self.kv_page_size),
                        occupancy=np.array([False] * self.kv_page_size),
                    )

            elif cmd.kind == ObjectKind.OBJECT_KIND_EMB:
                # do nothing. Embeds are allocated on the fly.
                pass
            elif cmd.kind == ObjectKind.OBJECT_KIND_DIST:
                # do nothing. Dists are allocated on the fly.
                pass

    def deallocate(self, cmds: BatchDeallocate):
        """TODO: Add method docstring."""
        # in current implementation, all allocations are already done in the constructor.
        # so we don't need to deallocate anything.

    def embed_text(self, cmds: BatchEmbedText):
        """TODO: Add method docstring."""
        # print(f"Debug - embed_text called with {len(cmds.items)} items")
        for _i, cmd in enumerate(cmds.items):
            self.embeds[cmd.embedding_id] = TextEmbed(
                token_id=cmd.token_id, position_id=cmd.position_id
            )

    def embed_image(self, cmds: BatchEmbedImage):
        """TODO: Add method docstring."""
        # unimplemented

    def mask_block(self, cmds: BatchMaskBlock):
        """TODO: Add method docstring."""
        for cmd in cmds.items:
            block = self.blocks[cmd.block_id]
            for i, m in enumerate(cmd.mask):
                block.occupancy[i] = m

    def copy_block(self, cmds: BatchCopyBlock):
        """TODO: Add method docstring."""
        for cmd in cmds.items:
            src_block = self.blocks[cmd.source_block_id]
            dst_block = self.blocks[cmd.destination_block_id]
            src_start = cmd.source_start
            dst_start = cmd.destination_start
            length = cmd.length

            dst_block.occupancy[dst_start: dst_start + length] = src_block.occupancy[
                src_start: src_start + length
            ]
            dst_block.position_ids[dst_start: dst_start + length] = (
                src_block.position_ids[src_start: src_start + length]
            )

            for kv_cache_layer in self.kv_cache_at_layer:
                kv_cache_layer[dst_block, :, dst_start: dst_start + length, :, :] = (
                    kv_cache_layer[src_block, :, src_start: src_start + length, :, :]
                )

    @torch.inference_mode()
    def decode_token_distribution(self, cmds: BatchDecodeTokenDistribution):
        """TODO: Add method docstring."""
        # start_time = time.time()

        # Make this more efficient by batching.
        # for i, cmd in enumerate(cmds.items):
        #     dist = torch.softmax(
        #         self.lm.lm_head(self.embed_storage.ptr[cmd.embedding_id]), dim=-1
        #     )
        #     self.dist_storage.ptr[cmd.distribution_id] = dist

        # torch.cuda.synchronize()
        # print(f"decode_token_distribution elapsed time {(time.time() - start_time) * 1000}ms")

    @torch.inference_mode()
    def sample_top_k_request(
            self, cmds: BatchSampleTopKRequest
    ) -> BatchSampleTopKResponse:
        """TODO: Add method docstring."""
        res = []
        for cmd in cmds.items:
            # Get the stored logits/probabilities and token IDs
            topk_probs = self.embed_storage_p1[cmd.distribution_id].tolist()
            topk_tokens = self.embed_storage_p2[cmd.distribution_id].tolist()
            #
            # # Limit to k requested tokens if needed
            # if cmd.k > 0 and cmd.k < len(topk_tokens):
            #     topk_tokens = topk_tokens[:cmd.k]
            #     topk_probs = topk_probs[:cmd.k]

            res.append(
                SampleTopKResponse(token_ids=topk_tokens, probabilities=topk_probs)
            )

        # torch.cuda.synchronize()
        # elapsed_time = (time.time() - start_time) * 1000
        # print(f"sample_top_k_request elapsed time {elapsed_time:.2f}ms")

        return BatchSampleTopKResponse(items=res)

    @torch.inference_mode()
    def fill_block(self, cmds: BatchFillBlock):
        """TODO: Add method docstring."""

        kv_page_indices = []
        kv_page_indptr = [0]
        kv_last_page_lens = []
        qo_indices = []
        qo_indptr = [0]
        custom_masks = []

        new_token_ids = []
        new_position_ids = []
        output_embed_postproc = []
        single_token_inference_mode = True

        for cmd in cmds.items:
            last_block_len = cmd.last_block_len  # change this name to "offset" later.

            ctx_block_ids = (
                cmd.context_block_ids
            )  # block == page. make names consistent later.
            input_embeds = cmd.input_embedding_ids
            output_embeds = cmd.output_embedding_ids

            kv_page_indices.extend(ctx_block_ids)
            kv_page_indptr.append(len(kv_page_indices))
            kv_last_page_lens.append(last_block_len)

            qo_indices.extend(input_embeds)
            qo_indptr.append(len(qo_indices))

            # let's compute the mask.

            inp_pos_ids = np.empty((len(input_embeds),), dtype=np.int32)
            inp_occupancy = np.zeros((len(input_embeds),), dtype=np.bool_)

            total_ctx_tokens = (
                    self.kv_page_size * (len(ctx_block_ids) - 1) + last_block_len
            )

            if len(input_embeds) > 1:
                single_token_inference_mode = False

            for i, input_embed in enumerate(input_embeds):

                token_offset = total_ctx_tokens - len(input_embeds) + i
                tgt_block_idx = token_offset // self.kv_page_size
                tgt_block_offset = token_offset % self.kv_page_size

                tgt_block_id = ctx_block_ids[tgt_block_idx]

                tgt_block = self.blocks[tgt_block_id]

                if input_embed in self.embeds:
                    embed = self.embeds[input_embed]
                    if isinstance(embed, TextEmbed):
                        new_token_ids.append(embed.token_id)
                        new_position_ids.append(embed.position_id)
                        inp_occupancy[i] = True
                        inp_pos_ids[i] = embed.position_id
                        tgt_block.occupancy[tgt_block_offset] = True
                        tgt_block.position_ids[tgt_block_offset] = embed.position_id
                else:
                    # should never happen, since the controller should have already
                    # checked that the input embeds are valid.
                    raise ValueError("Input embedding not found")

            for i, embed in enumerate(output_embeds):
                output_embed_postproc.append(
                    {
                        "idx": len(new_token_ids) - len(output_embeds) + i,
                        "vec_id": embed,
                    }
                )

            # Get the full sequence length (context + input tokens)
            total_sequence_length = total_ctx_tokens  # + len(input_embeds)

            # Get position IDs and occupancy for the entire sequence
            ctx_pos_ids = np.hstack(
                [self.blocks[ctx_id].position_ids for ctx_id in ctx_block_ids]
            )[
                :total_sequence_length
            ]  # int
            ctx_occupancy = np.hstack(
                [self.blocks[ctx_id].occupancy for ctx_id in ctx_block_ids]
            )[
                :total_sequence_length
            ]  # bool

            # print(f"ctx_pos_ids: {ctx_pos_ids}, ctx_occupancy: {ctx_occupancy}")

            # Build the causal and valid masks
            casual_mask = ctx_pos_ids[None, :] <= inp_pos_ids[:, None]
            valid_mask = np.logical_and(ctx_occupancy[None, :], inp_occupancy[:, None])
            mask = np.logical_and(casual_mask, valid_mask)

            mask_flat = mask.flatten()
            custom_masks.append(mask_flat)
            # print(mask)

        # concat all masks
        custom_mask = np.concatenate(custom_masks)

        # print all inputs
        # print('kv_page_indices', kv_page_indices)
        # print('kv_page_indptr', kv_page_indptr)
        # print('kv_last_page_lens', kv_last_page_lens)
        # print('qo_indptr', qo_indptr)
        # print('custom_mask', custom_mask)

        pt_new_token_ids = torch.as_tensor(
            new_token_ids, device=self.device, dtype=torch.int32
        )
        pt_new_position_ids = torch.as_tensor(
            new_position_ids, device=self.device, dtype=torch.int32
        )

        pt_kv_page_indices = torch.as_tensor(
            kv_page_indices, device=self.device, dtype=torch.int32
        )
        pt_kv_page_indptr = torch.as_tensor(
            kv_page_indptr, device=self.device, dtype=torch.int32
        )
        pt_kv_last_page_lens = torch.as_tensor(
            kv_last_page_lens, device=self.device, dtype=torch.int32
        )
        pt_qo_indptr = torch.as_tensor(qo_indptr, device=self.device, dtype=torch.int32)
        pt_custom_mask = torch.as_tensor(
            custom_mask, device=self.device, dtype=torch.bool
        )

        input_embeds = self.lm.model.embed_tokens(pt_new_token_ids)

        # torch.cuda.synchronize()
        # print(f"prepare time {(time.time() - start_time) * 1000}ms  ")
        # print('kv_page_indices', kv_page_indices)
        # print('kv_page_indptr', kv_page_indptr)
        # print('kv_last_page_lens', kv_last_page_lens)
        # print('qo_indptr', qo_indptr)
        # print('custom_mask', custom_masks)

        with torch.cuda.device(self.device):

            output_embeds = self.lm.model.forward(
                input_embeds=input_embeds,
                position_ids=pt_new_position_ids,
                kv_cache_at_layer=self.kv_cache_at_layer,
                kv_page_indices=pt_kv_page_indices,
                kv_page_indptr=pt_kv_page_indptr,
                kv_last_page_lens=pt_kv_last_page_lens,
                qo_indptr=pt_qo_indptr,
                custom_mask=pt_custom_mask,
                single_token_inference_mode=single_token_inference_mode,
            )
            # print(f"output_embeds mean: {output_embeds.mean().item()}")

            # precompute the dists
            logits = self.lm.lm_head(output_embeds)

            # torch.cuda.synchronize()
            # elapsed_time = (time.time() - start_time) * 1000
            # print(f"forward pass elapsed time {elapsed_time:.2f}ms size {output_embeds.shape}")
            # topk

            # print(f"logits mean: {logits.mean().item()}")
            probs = torch.softmax(logits, dim=-1)

            condensed = torch.topk(probs, k=self.dist_size, sorted=True)

        # print(logits.shape)
        # store the logits in the output embeds  -> replace with torch.scatter later
        for token_map in output_embed_postproc:
            vec_id = token_map["vec_id"]
            idx = token_map["idx"]
            self.embed_storage_p1[vec_id].copy_(
                condensed.values[idx], non_blocking=True
            )
            self.embed_storage_p2[vec_id].copy_(
                condensed.indices[idx], non_blocking=True
            )

        # torch.cuda.synchronize()
        # elapsed_time = (time.time() - start_time) * 1000
        # print(f"fill_block elapsed time {elapsed_time}ms size {output_embeds.shape}")

        self.inter_fill_time = time.time()

        return BatchSyncResponse()

    @torch.inference_mode()
    def forward_text(self, cmds: BatchForwardText):
        """TODO: Add method docstring."""

        start_time = time.time()
        kv_page_indices = []
        kv_page_indptr = [0]
        kv_page_last_lens = []
        qo_indices = []
        qo_indptr = [0]
        custom_masks = []

        new_token_ids = []
        new_position_ids = []

        all_output_indices = []
        all_output_indices_ptr = [0]
        single_token_inference_mode = True

        for cmd in cmds.items:

            kv_page_indices.extend(cmd.kv_page_ids)
            kv_page_indptr.append(len(kv_page_indices))
            kv_page_last_lens.append(cmd.kv_page_last_len)
            qo_indices.extend(cmd.token_ids)
            qo_indptr.append(len(qo_indices))

            num_total_tokens = self.kv_page_size * (len(cmd.kv_page_ids) - 1) + cmd.kv_page_last_len
            num_context_tokens = num_total_tokens - len(cmd.token_ids)

            output_indices = []
            for x in cmd.output_indices:
                output_indices.append(x + qo_indptr[-1] - len(cmd.token_ids))

            new_token_ids.extend(cmd.token_ids)
            new_position_ids.extend(cmd.position_ids)
            all_output_indices.extend(output_indices)
            all_output_indices_ptr.append(len(all_output_indices))

            if len(cmd.token_ids) > 1:
                single_token_inference_mode = False

            # --- New Mask Generation Logic ---
            if len(cmd.mask) != len(cmd.token_ids):
                raise ValueError(
                    f"Mismatch between number of masks ({len(cmd.mask)}) and "
                    f"input tokens ({len(cmd.token_ids)})."
                )

            cmd_mask = np.zeros((len(cmd.token_ids), num_total_tokens), dtype=np.bool_)
            for i, brle_buffer in enumerate(cmd.mask):
                # 1. Decode the provided BRLE mask for attention to the past context.
                decoded_mask = _decode_brle(brle_buffer.buffer)

                # The BRLE mask for query token `i` should cover the past context plus `i` previous new tokens.
                expected_len = num_context_tokens + i + 1
                if len(decoded_mask) != expected_len:
                    raise ValueError(
                        f"Decoded mask for token {i} has length {len(decoded_mask)}, but expected {expected_len}"
                    )

                cmd_mask[i, :expected_len] = decoded_mask

            mask_flat = cmd_mask.flatten()
            custom_masks.append(mask_flat)

        # concat all masks
        custom_mask = np.concatenate(custom_masks)

        # print all inputs
        # print('kv_page_indices', kv_page_indices)
        # print('kv_page_indptr', kv_page_indptr)
        # print('kv_last_page_lens', kv_last_page_lens)
        # print('qo_indptr', qo_indptr)
        # print('custom_mask', custom_mask)

        pt_new_token_ids = torch.as_tensor(
            new_token_ids, device=self.device, dtype=torch.int32
        )
        pt_new_position_ids = torch.as_tensor(
            new_position_ids, device=self.device, dtype=torch.int32
        )

        pt_kv_page_indices = torch.as_tensor(
            kv_page_indices, device=self.device, dtype=torch.int32
        )
        pt_kv_page_indptr = torch.as_tensor(
            kv_page_indptr, device=self.device, dtype=torch.int32
        )
        pt_kv_last_page_lens = torch.as_tensor(
            kv_page_last_lens, device=self.device, dtype=torch.int32
        )
        pt_qo_indptr = torch.as_tensor(qo_indptr, device=self.device, dtype=torch.int32)
        pt_custom_mask = torch.as_tensor(
            custom_mask, device=self.device, dtype=torch.bool
        )

        pt_output_indices = torch.as_tensor(
            all_output_indices, device=self.device, dtype=torch.int32)

        input_embeds = self.lm.model.embed_tokens(pt_new_token_ids)

        # torch.cuda.synchronize()
        # print(f"prepare time {(time.time() - start_time) * 1000}ms  ")
        # print('kv_page_indices', kv_page_indices)
        # print('kv_page_indptr', kv_page_indptr)
        # print('kv_page_last_lens', kv_page_last_lens)
        # print('qo_indptr', qo_indptr)
        # print('custom_mask', custom_masks)
        # print('output_indices', all_output_indices)
        # print('new_token_ids', new_token_ids)
        # print('new_position_ids', new_position_ids)

        responses = []

        with torch.cuda.device(self.device):

            output_embeds = self.lm.model.forward(
                input_embeds=input_embeds,
                position_ids=pt_new_position_ids,
                kv_cache_at_layer=self.kv_cache_at_layer,
                kv_page_indices=pt_kv_page_indices,
                kv_page_indptr=pt_kv_page_indptr,
                kv_last_page_lens=pt_kv_last_page_lens,
                qo_indptr=pt_qo_indptr,
                custom_mask=pt_custom_mask,
                single_token_inference_mode=single_token_inference_mode,
            )
            # print(f"output_embeds mean: {output_embeds.mean().item()}")

            # precompute the dists
            if len(all_output_indices) > 0:
                output_embeds = output_embeds[pt_output_indices]
                logits = self.lm.lm_head(output_embeds)
                probs = torch.softmax(logits, dim=-1)
                condensed = torch.topk(probs, k=self.dist_size, sorted=True)

                batch_ids = condensed.indices.tolist()
                batch_probs = condensed.values.tolist()

                # reshape them based on all_output_indices_ptr
                for i in range(len(all_output_indices_ptr) - 1):
                    start = all_output_indices_ptr[i]
                    end = all_output_indices_ptr[i + 1]
                    dists = []
                    for j in range(start, end):
                        dist = Distribution(
                            ids=batch_ids[j],
                            probs=batch_probs[j]
                        )
                        dists.append(dist)
                    res = ForwardTextResponse(
                        distributions=dists
                    )
                    responses.append(res)

        # torch.cuda.synchronize()
        # print(f"forward_text time {(time.time() - start_time) * 1000}ms  ")
        return BatchForwardTextResponse(
            items=responses
        )

    def debug_query_request(self, cmds: BatchDebugQueryRequest):

        responses = []
        for cmd in cmds.items:
            if cmd.query == "ping":
                response = DebugQueryResponse(
                    response="pong"
                )
                responses.append(response)

        return BatchDebugQueryResponse(
            items=responses
        )


def _decode_brle(brle_buffer: list[int]) -> np.ndarray:
    """
    Decodes a Binary Run-Length Encoded buffer into a boolean numpy array.
    The format assumes alternating runs of False and True, starting with False.
    """
    if not brle_buffer:
        return np.array([], dtype=bool)

    total_size = sum(brle_buffer)
    if total_size == 0:
        return np.array([], dtype=bool)

    decoded_array = np.empty(total_size, dtype=bool)
    current_pos = 0
    value = True  # originally False, but in attention masking, this is flipped to True
    for run_len in brle_buffer:
        if run_len > 0:
            # Assign the current boolean value to the slice of the array
            decoded_array[current_pos: current_pos + run_len] = value
        current_pos += run_len
        value = not value  # Flip value for the next run
    return decoded_array
