from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from pie_driver_tensorrt_llm.engine import (
    TensorRTLLMEngine,
    _LookaheadBuffer,
    _ModelInfo,
    _PyExecutorSession,
    _validate_execution_mode,
)


@dataclass
class FakeSamplingParams:
    kwargs: dict

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))
        outputs = []
        for params in kwargs["sampling_params"]:
            max_tokens = int(params.kwargs.get("max_tokens", 1))
            outputs.append(
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            token_ids=list(range(42, 42 + max_tokens))
                        )
                    ]
                )
            )
        return outputs


class FakePyExecutorRequest:
    def __init__(self, request_id: int, tokens: list[int]):
        self.py_request_id = request_id
        self.request_id = request_id
        self.state = SimpleNamespace(name="GENERATION_IN_PROGRESS")
        self._tokens = tokens

    def get_tokens(self, _beam_idx: int):
        return self._tokens


class FakeResponseLock:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False


class FakePyExecutor:
    def __init__(self):
        self.terminated = []
        self.responses = {}
        self.response_cv = FakeResponseLock()
        self.resource_manager = SimpleNamespace(free_resources=lambda _request: None)

    def _terminate_request(self, request):
        self.terminated.append(int(request.py_request_id))


def _driver_config(**overrides):
    values = {
        "virtual_total_pages": 1024,
        "virtual_kv_page_size": 16,
        "max_batched_tokens": 64,
        "max_concurrent_requests": 8,
        "max_session_histories": 128,
        "max_history_tokens": None,
        "lookahead_tokens": 1,
        "enable_cache_salt": True,
        "execution_mode": "generate",
        "pyexecutor_max_tokens": 4096,
        "pyexecutor_worker_stop_timeout_s": 30.0,
        "pyexecutor_lookahead": False,
        "pyexecutor_lookahead_min_batch_size": None,
        "pyexecutor_direct_token_limit": None,
        "pyexecutor_speculative_lookahead": False,
        "max_seq_len": None,
        "max_batch_size": None,
        "max_num_tokens": None,
        "kv_cache_free_gpu_memory_fraction": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _engine(**driver_overrides):
    execution_mode = driver_overrides.get("execution_mode", "generate")
    return TensorRTLLMEngine(
        llm=FakeLLM(),
        sampling_params_cls=FakeSamplingParams,
        config=SimpleNamespace(total_pages=0, activation_dtype="bfloat16"),
        driver_config=_driver_config(**driver_overrides),
        model_info=_ModelInfo(
            "FakeForCausalLM",
            128,
            1024,
            eos_token_id=2,
            pad_token_id=0,
        ),
        snapshot_dir="/tmp/fake-snapshot",
        execution_mode=execution_mode,
    )


def _batch(**overrides):
    values = {
        "has_speculative_inputs": False,
        "adapter_subpass_needed": False,
        "sampling_masks": None,
        "logit_masks": None,
        "request_output_counts": [1],
        "qo_indptr": [0, 3],
        "token_ids": [10, 11, 12],
        "position_ids": [0, 1, 2],
        "context_ids": [7],
        "kv_page_indptr": [0, 0],
        "kv_page_indices": [],
        "sampler_types": [3],
        "indices_for_logits": [2],
        "temperatures": [0.7],
        "top_k_values": [0],
        "top_p_values": [0.9],
        "min_p_values": [0.0],
        "sampler_seeds_arr": [1234],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fire_batch_generates_one_token_from_replayed_prompt():
    engine = _engine()

    out = engine.fire_batch({"batch": _batch()}, {"batch": None})

    assert out["tokens"] == [42]
    assert engine._histories[7] == [10, 11, 12, 42]
    prompts, kwargs = engine.llm.calls[0]
    assert prompts == [[10, 11, 12]]
    assert kwargs["cache_salt"] == ["7"]
    params = kwargs["sampling_params"][0]
    assert params.kwargs == {
        "max_tokens": 1,
        "detokenize": False,
        "add_special_tokens": False,
        "ignore_eos": True,
        "temperature": 0.7,
        "end_id": 2,
        "pad_id": 0,
        "seed": 1234,
        "top_p": 0.9,
    }


def test_generated_tokens_keep_chained_replay_history_contiguous():
    engine = _engine()

    engine.fire_batch({"batch": _batch()}, {"batch": None})
    out = engine.fire_batch(
        {
            "batch": _batch(
                qo_indptr=[0, 1],
                token_ids=[99],
                position_ids=[4],
                indices_for_logits=[0],
            )
        },
        {"batch": None},
    )

    assert out["tokens"] == [42]
    assert engine._histories[7] == [10, 11, 12, 42, 99, 42]
    prompts, _ = engine.llm.calls[1]
    assert prompts == [[10, 11, 12, 42, 99]]


def test_deterministic_lookahead_drains_without_regenerating():
    engine = _engine(lookahead_tokens=4)

    out = engine.fire_batch(
        {
            "batch": _batch(
                temperatures=[0.0],
                top_p_values=[1.0],
                output_spec_flags=[True],
            )
        },
        {"batch": None},
    )
    assert out["tokens"] == [42]
    assert engine._histories[7] == [10, 11, 12, 42, 43, 44, 45]
    assert len(engine.llm.calls) == 1
    first_params = engine.llm.calls[0][1]["sampling_params"][0]
    assert first_params.kwargs["max_tokens"] == 4

    out = engine.fire_batch(
        {
            "batch": _batch(
                qo_indptr=[0, 1],
                token_ids=[42],
                position_ids=[3],
                indices_for_logits=[0],
                temperatures=[0.0],
                top_p_values=[1.0],
            )
        },
        {"batch": None},
    )
    assert out["tokens"] == [43]
    assert len(engine.llm.calls) == 1


def test_deterministic_lookahead_ignores_seed_changes():
    engine = _engine(lookahead_tokens=4)

    engine.fire_batch(
        {"batch": _batch(temperatures=[0.0], top_p_values=[1.0])},
        {"batch": None},
    )
    out = engine.fire_batch(
        {
            "batch": _batch(
                qo_indptr=[0, 1],
                token_ids=[42],
                position_ids=[3],
                indices_for_logits=[0],
                temperatures=[0.0],
                top_p_values=[1.0],
                sampler_seeds_arr=[999],
            )
        },
        {"batch": None},
    )

    assert out["tokens"] == [43]
    assert len(engine.llm.calls) == 1


def test_lookahead_invalidates_on_divergent_replay_token():
    engine = _engine(lookahead_tokens=4)

    engine.fire_batch(
        {"batch": _batch(temperatures=[0.0], top_p_values=[1.0])},
        {"batch": None},
    )
    out = engine.fire_batch(
        {
            "batch": _batch(
                qo_indptr=[0, 1],
                token_ids=[99],
                position_ids=[3],
                indices_for_logits=[0],
                temperatures=[0.0],
                top_p_values=[1.0],
            )
        },
        {"batch": None},
    )

    assert out["tokens"] == [42]
    assert engine._histories[7] == [10, 11, 12, 99, 42, 43, 44, 45]
    assert len(engine.llm.calls) == 2
    prompts, _ = engine.llm.calls[1]
    assert prompts == [[10, 11, 12, 99]]


def test_stochastic_sampling_disables_lookahead():
    engine = _engine(lookahead_tokens=4)

    engine.fire_batch({"batch": _batch(temperatures=[0.7])}, {"batch": None})

    params = engine.llm.calls[0][1]["sampling_params"][0]
    assert params.kwargs["max_tokens"] == 1


def test_lookahead_is_capped_to_stable_window():
    engine = _engine(lookahead_tokens=32)

    engine.fire_batch(
        {"batch": _batch(temperatures=[0.0], top_p_values=[1.0])},
        {"batch": None},
    )

    params = engine.llm.calls[0][1]["sampling_params"][0]
    assert params.kwargs["max_tokens"] == 16


def test_duplicate_session_in_one_batch_uses_prompt_snapshot():
    engine = _engine(lookahead_tokens=1)

    out = engine.fire_batch(
        {
            "batch": _batch(
                request_output_counts=[1, 1],
                qo_indptr=[0, 3, 4],
                token_ids=[10, 11, 12, 42],
                position_ids=[0, 1, 2, 3],
                context_ids=[7, 7],
                kv_page_indptr=[0, 0, 0],
                sampler_types=[3, 3],
                indices_for_logits=[2, 3],
                temperatures=[0.0, 0.0],
                top_k_values=[0, 0],
                top_p_values=[1.0, 1.0],
                min_p_values=[0.0, 0.0],
                sampler_seeds_arr=[0, 0],
            )
        },
        {"batch": None},
    )

    assert out["tokens"] == [42, 42]
    assert engine._histories[7] == [10, 11, 12, 42, 42]


def test_replay_history_gap_raises_instead_of_guessing_missing_prefix():
    engine = _engine()
    history = [10, 11]

    with pytest.raises(RuntimeError, match="replay history only contains 2 tokens"):
        engine._merge_tokens(history, [14], [4], session_id=7)

    assert history == [10, 11]


def test_trimmed_history_recovers_when_batch_carries_tail_token():
    engine = _engine(max_history_tokens=3)

    engine.fire_batch({"batch": _batch()}, {"batch": None})
    out = engine.fire_batch(
        {
            "batch": _batch(
                qo_indptr=[0, 1],
                token_ids=[42],
                position_ids=[3],
                indices_for_logits=[0],
            )
        },
        {"batch": None},
    )

    assert out["tokens"] == [42]


def test_special_sampler_types_fail_explicitly():
    engine = _engine()

    with pytest.raises(NotImplementedError, match="token-producing"):
        engine.fire_batch(
            {"batch": _batch(sampler_types=[7])},
            {"batch": None},
        )


def test_execution_mode_validation_is_explicit():
    assert _validate_execution_mode("generate") == "generate"
    assert _validate_execution_mode("pyexecutor") == "pyexecutor"
    with pytest.raises(ValueError, match="execution_mode"):
        _validate_execution_mode("worker")


def test_pyexecutor_capacity_evicts_inactive_sessions_before_new_wave():
    engine = _engine(max_batch_size=2)
    engine.pyexecutor = FakePyExecutor()
    engine._pyexecutor_sessions = {
        1: _PyExecutorSession(
            1, FakePyExecutorRequest(1, [10]), sampler_key=("sampler",)
        ),
        2: _PyExecutorSession(
            2, FakePyExecutorRequest(2, [20]), sampler_key=("sampler",)
        ),
    }

    engine._prepare_pyexecutor_capacity_for_wave(
        [
            (0, {"session_id": 2, "prompt": [20], "sampler_key": ("sampler",)}),
            (1, {"session_id": 3, "prompt": [30], "sampler_key": ("sampler",)}),
        ]
    )

    assert set(engine._pyexecutor_sessions) == {2}
    assert engine.pyexecutor.terminated == [1]


def test_drop_finished_pyexecutor_sessions_frees_tensorrt_resources():
    engine = _engine(max_batch_size=1)
    engine.pyexecutor = FakePyExecutor()
    engine._pyexecutor_sessions = {
        7: _PyExecutorSession(
            11, FakePyExecutorRequest(11, [10]), sampler_key=("sampler",)
        )
    }

    engine._drop_finished_pyexecutor_sessions([SimpleNamespace(py_request_id=11)])

    assert engine._pyexecutor_sessions == {}
    assert engine.pyexecutor.terminated == [11]


def test_history_eviction_drops_related_cached_state():
    engine = _engine(max_session_histories=1)
    engine._history_for(1).append(10)
    engine._lookahead[1] = object()
    engine._emitted_token_counts[1] = 3

    engine._history_for(2)

    assert 1 not in engine._histories
    assert 1 not in engine._lookahead
    assert 1 not in engine._emitted_token_counts
    assert 2 in engine._histories


def test_pyexecutor_speculative_lookahead_returns_direct_accepted_tokens():
    engine = _engine(
        execution_mode="pyexecutor",
        lookahead_tokens=4,
        pyexecutor_speculative_lookahead=True,
    )
    engine.pyexecutor = object()
    calls = []

    def fake_generate_many(work):
        calls.append(work)
        return [(out_idx, item, [42, 43, 44, 45]) for out_idx, item in work]

    engine._pyexecutor_generate_many = fake_generate_many

    out = engine.fire_batch(
        {
            "batch": _batch(
                temperatures=[0.0],
                top_p_values=[1.0],
                output_spec_flags=[True],
            )
        },
        {"batch": None},
    )
    assert out["tokens"] == [42]
    assert out["spec_accepted_tokens"] == [[42, 43, 44, 45]]
    assert engine._histories[7] == [10, 11, 12, 42, 43, 44, 45]
    assert len(calls) == 1
    assert calls[0][0][1]["max_tokens"] == 4
    assert engine.spec_step([(7, [42])]) == [[]]


def test_pyexecutor_direct_accepted_falls_back_when_limit_is_exhausted():
    engine = _engine(
        execution_mode="pyexecutor",
        lookahead_tokens=4,
        pyexecutor_direct_token_limit=3,
        pyexecutor_speculative_lookahead=True,
    )
    engine.pyexecutor = object()
    engine._emitted_token_counts[7] = 3
    calls = []

    def fake_generate_many(work):
        calls.append(
            [
                (out_idx, int(item["session_id"]), int(item["max_tokens"]))
                for out_idx, item in work
            ]
        )
        out = []
        for out_idx, item in work:
            first = 90 if int(item["session_id"]) == 7 else 42
            max_tokens = int(item["max_tokens"])
            out.append((out_idx, item, list(range(first, first + max_tokens))))
        return out

    engine._pyexecutor_generate_many = fake_generate_many

    out = engine.fire_batch(
        {
            "batch": _batch(
                request_output_counts=[1, 1],
                qo_indptr=[0, 3, 6],
                token_ids=[10, 11, 12, 20, 21, 22],
                position_ids=[0, 1, 2, 0, 1, 2],
                context_ids=[7, 8],
                kv_page_indptr=[0, 0, 0],
                sampler_types=[3, 3],
                indices_for_logits=[2, 5],
                temperatures=[0.0, 0.0],
                top_k_values=[0, 0],
                top_p_values=[1.0, 1.0],
                min_p_values=[0.0, 0.0],
                sampler_seeds_arr=[0, 0],
                output_spec_flags=[True, True],
            )
        },
        {"batch": None},
    )

    assert out["tokens"] == [90, 42]
    assert out["spec_accepted_tokens"] == [[90], [42, 43, 44]]
    assert calls == [[(0, 7, 4)], [(1, 8, 3)]]
    assert engine._emitted_token_counts[7] == 4
    assert engine._emitted_token_counts[8] == 3


def test_pyexecutor_direct_accepted_uses_buffered_token_before_generating():
    engine = _engine(
        execution_mode="pyexecutor",
        lookahead_tokens=4,
        pyexecutor_speculative_lookahead=True,
    )
    engine.pyexecutor = object()
    engine._histories[7] = [10, 11, 12, 42, 43, 44]
    engine._lookahead[7] = _LookaheadBuffer(
        base_pos=3,
        tokens=[42, 43, 44],
        next_idx=1,
        sampler_key=(3, 0.0, 0, 1.0, 0.0, 0, 2, 0),
    )

    def fail_generate_many(_work):
        raise AssertionError("buffered direct-accepted token should not regenerate")

    engine._pyexecutor_generate_many = fail_generate_many

    out = engine.fire_batch(
        {
            "batch": _batch(
                qo_indptr=[0, 1],
                token_ids=[42],
                position_ids=[3],
                indices_for_logits=[0],
                temperatures=[0.0],
                top_p_values=[1.0],
                output_spec_flags=[True],
            )
        },
        {"batch": None},
    )

    assert out["tokens"] == [43]
    assert out["spec_accepted_tokens"] == [[43]]
    assert engine._emitted_token_counts[7] == 1


def test_pyexecutor_duplicate_session_consumes_new_lookahead_in_order():
    engine = _engine(
        execution_mode="pyexecutor",
        lookahead_tokens=4,
        pyexecutor_lookahead=True,
    )
    engine.pyexecutor = object()
    calls = []

    def fake_generate_many(work):
        calls.append(work)
        return [(out_idx, item, [42, 43, 44, 45]) for out_idx, item in work]

    engine._pyexecutor_generate_many = fake_generate_many

    out = engine.fire_batch(
        {
            "batch": _batch(
                request_output_counts=[1, 1],
                qo_indptr=[0, 3, 4],
                token_ids=[10, 11, 12, 42],
                position_ids=[0, 1, 2, 3],
                context_ids=[7, 7],
                kv_page_indptr=[0, 0, 0],
                sampler_types=[3, 3],
                indices_for_logits=[2, 3],
                temperatures=[0.0, 0.0],
                top_k_values=[0, 0],
                top_p_values=[1.0, 1.0],
                min_p_values=[0.0, 0.0],
                sampler_seeds_arr=[0, 0],
            )
        },
        {"batch": None},
    )

    assert out["tokens"] == [42, 43]
    assert len(calls) == 1
    assert calls[0][0][1]["max_tokens"] == 4
