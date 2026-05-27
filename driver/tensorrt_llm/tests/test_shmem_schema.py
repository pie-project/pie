from __future__ import annotations

from types import SimpleNamespace

from pie_driver_tensorrt_llm._bridge.shmem_schema import _forward_to_dict


def test_forward_to_dict_preserves_context_ids():
    fr = SimpleNamespace(
        token_ids=[1],
        position_ids=[0],
        kv_page_indices=[11],
        kv_page_indptr=[0, 1],
        kv_last_page_lens=[1],
        qo_indptr=[0, 1],
        rs_slot_ids=[],
        rs_slot_flags=[],
        masks_len=0,
        masks_at=lambda _i: None,
        mask_indptr=[0],
        logit_masks_len=0,
        logit_masks_at=lambda _i: None,
        logit_mask_indptr=[0],
        sampling_indices=[],
        sampling_indptr=[0],
        samplers_len=0,
        samplers_at=lambda _i: None,
        sampler_indptr=[0],
        adapter_bindings_len=0,
        adapter_bindings_at=lambda _i: None,
        spec_token_ids=[],
        spec_position_ids=[],
        spec_indptr=[0],
        output_spec_flags=[False],
        context_ids=[1234567890123],
        single_token_mode=True,
        has_user_mask=False,
    )

    out = _forward_to_dict(fr, driver_id=7)

    assert out["context_ids"].tolist() == [1234567890123]
