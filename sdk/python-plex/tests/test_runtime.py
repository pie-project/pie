import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from pie_plex import InvalidEvent, PolicyPackageError, Runtime


ROOT = Path(__file__).resolve().parents[3]
POLICY = Path(
    os.environ.get(
        "PLEX_TEST_POLICY",
        ROOT / "tests/policies/target/packages/plex_coordinated.plexpkg",
    )
)


def route_event(request_id="L", *, query=False):
    capabilities = {
        "queries": ["pie.cluster.capacity@1"] if query else [],
    }
    return {
        "api_version": "pie.plex.engine@1",
        "hook": "route",
        "context": {
            "request_id": request_id,
            "candidates": [
                {
                    "id": "node-a",
                    "facts": {
                        "queue_depth": 1,
                        "cached_tokens": 0,
                        "has_request_kv": False,
                    },
                }
            ],
            "context": {
                "model": "example-model",
                "capabilities": capabilities,
            },
        },
        "request_events": [
            {
                "op": "create",
                "request_id": request_id,
                "facts": {"generation_id": 0},
                "fields": {
                    "body": {"prompt": "hello"},
                    "metadata": {},
                },
            }
        ],
    }


def test_dictionary_round_trip_and_action_return():
    runtime = Runtime(
        str(POLICY),
        actions=["pie.kv.prefetch@1"],
    )
    outcome = runtime.invoke(route_event())
    assert outcome["status"] == "success"
    assert outcome["decision"] == {"order": [0]}
    assert outcome["request_fields"]["L"]["body"]["prompt"] == "hello|route"
    assert outcome["actions"] == [
        {
            "id": 0,
            "method": "pie.kv.prefetch@1",
            "args": {"request_id": "L", "target": "node-a"},
        }
    ]
    assert json.loads(runtime.invoke_json(json.dumps({
        "api_version": "pie.plex.engine@1",
        "hook": "evict",
        "context": {"resident": [], "bytes_needed": 1, "context": {}},
        "request_events": [],
    }))) == {
        "status": "success",
        "decision": {"selected": []},
        "request_fields": {},
        "actions": [],
    }


def test_query_callback_and_policy_visible_errors():
    runtime = Runtime(
        str(POLICY),
        query=lambda method, args: {
            "route_bias": 3.0,
            "method": method,
            "model": args["model"],
        },
    )
    assert runtime.invoke(route_event(query=True))["status"] == "success"

    def raises(_method, _args):
        raise RuntimeError("injected query failure")

    failed = Runtime(str(POLICY), query=raises).invoke(route_event("failed", query=True))
    assert failed["status"] == "fallback"
    assert failed["failure"]["kind"] == "query"
    assert "injected query failure" in failed["failure"]["message"]

    non_json = Runtime(str(POLICY), query=lambda _method, _args: {object()})
    outcome = non_json.invoke(route_event("non-json", query=True))
    assert outcome["status"] == "fallback"
    assert outcome["failure"]["kind"] == "query"


def test_same_runtime_recursive_query_is_rejected():
    holder = {}

    def recursive(_method, _args):
        return holder["runtime"].invoke({
            "api_version": "pie.plex.engine@1",
            "hook": "evict",
            "context": {"resident": [], "bytes_needed": 1, "context": {}},
            "request_events": [],
        })

    holder["runtime"] = Runtime(str(POLICY), query=recursive)
    outcome = holder["runtime"].invoke(route_event(query=True))
    assert outcome["status"] == "fallback"
    assert outcome["failure"]["kind"] == "query"
    assert "recursive invoke" in outcome["failure"]["message"]


def test_same_runtime_concurrent_invocations_are_serialized():
    entered = Event()
    release = Event()

    def query(_method, _args):
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=5)
        return {"route_bias": 0.0}

    runtime = Runtime(str(POLICY), query=query)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(runtime.invoke, route_event("first", query=True))
        assert entered.wait(timeout=5)
        second = executor.submit(runtime.invoke, route_event("second", query=True))
        release.set()
        outcomes = [first.result(timeout=5), second.result(timeout=5)]

    assert [outcome["status"] for outcome in outcomes] == ["success", "success"]


def test_invalid_events_raise_and_independent_runtimes_are_concurrent():
    with pytest.raises(PolicyPackageError):
        Runtime("/definitely/missing/policy.plexpkg")

    runtime = Runtime(str(POLICY))
    with pytest.raises(InvalidEvent):
        runtime.invoke({
            "api_version": "wrong",
            "hook": "route",
            "context": {},
            "request_events": [],
        })

    def invoke(index):
        return Runtime(str(POLICY)).invoke(route_event(f"L-{index}"))["status"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(invoke, range(8))) == ["success"] * 8
