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
        ROOT / "tests/policies/target/packages/plex_coordinated_v0_6.plexpkg",
    )
)


def request_ref(request_id):
    return {
        "request_id": request_id,
        "generation_id": 0,
        "group_id": f"group-{request_id}",
        "principal_id": "tenant",
    }


def meta(opportunity_id):
    return {
        "opportunity_id": opportunity_id,
        "snapshot": {"id": "host-filled", "revision": 0},
        "attempt": 0,
        "mechanics": [],
    }


def admit_event(request_id):
    group_id = f"group-{request_id}"
    return {
        "api_version": "pie.plex.engine@2",
        "operation": "admit",
        "context": {
            "meta": meta(f"admit-{request_id}"),
            "cause": "arrival",
            "candidates": [
                {
                    "request": request_ref(request_id),
                    "demand": [],
                    "facts": {"queue_depth": 0},
                }
            ],
            "capacity": {"max_accepted": 1, "limits": [], "facts": {}},
        },
        "lifecycle": [
            {
                "event": "create-group",
                "group_id": group_id,
                "principal_id": "tenant",
                "limits": {"max_members": 4, "max_scratch_bytes": 4096},
                "facts": {},
            },
            {
                "event": "create-request",
                "request_id": request_id,
                "principal_id": "tenant",
                "group_id": group_id,
                "fields": {},
                "facts": {},
            },
        ],
    }


def route_event(request_id, *, query=False):
    return {
        "api_version": "pie.plex.engine@2",
        "operation": "route",
        "context": {
            "meta": meta(f"route-{request_id}"),
            "cause": "admission",
            "requests": [
                {
                    "request": request_ref(request_id),
                    "facts": {"query": query},
                }
            ],
            "targets": [
                {
                    "target_id": "node-a",
                    "max_assignments": 1,
                    "capacity": [],
                    "revision": 1,
                    "facts": {},
                }
            ],
            "feasible_edges": [
                {
                    "request_index": 0,
                    "target_index": 0,
                    "demand": [],
                    "facts": {"queue_depth": 0},
                }
            ],
        },
        "lifecycle": [{"event": "admit-request", "request_id": request_id}],
    }


def cache_event(opportunity_id):
    return {
        "api_version": "pie.plex.engine@2",
        "operation": "cache",
        "context": {
            "meta": meta(opportunity_id),
            "cause": "pressure",
            "resident": [],
            "prospective": [],
            "capacity": {"max_bytes": 0, "fixed_bytes": 0, "facts": {}},
            "episode": None,
        },
    }


def bootstrap(runtime, request_id, *, query=False):
    admitted = runtime.invoke(admit_event(request_id))
    assert admitted["status"] == "success"
    return runtime.invoke(route_event(request_id, query=query))


def test_typed_round_trip_and_json_api():
    runtime = Runtime(str(POLICY))
    outcome = bootstrap(runtime, "A")
    assert outcome["status"] == "success"
    assert outcome["plan"]["plan"]["assignments"] == [
        {"request_index": 0, "edge_index": 0, "target_index": 0}
    ]
    cache = json.loads(runtime.invoke_json(json.dumps(cache_event("cache-A"))))
    assert cache["status"] == "success"
    assert cache["plan"]["plan"] == {"admissions": [], "reclaim": []}


def test_query_callback_and_policy_visible_errors():
    runtime = Runtime(
        str(POLICY),
        query=lambda method, args: {
            "queue_bias": 3,
            "method": method,
            "model": args["model"],
        },
    )
    assert bootstrap(runtime, "query", query=True)["status"] == "success"

    def raises(_method, _args):
        raise RuntimeError("injected query failure")

    failed = Runtime(str(POLICY), query=raises)
    failed.invoke(admit_event("failed"))
    outcome = failed.invoke(route_event("failed", query=True))
    assert outcome["status"] == "fallback"
    assert outcome["failure"]["kind"] == "query"
    assert "injected query failure" in outcome["failure"]["message"]

    non_json = Runtime(str(POLICY), query=lambda _method, _args: {object()})
    non_json.invoke(admit_event("non-json"))
    outcome = non_json.invoke(route_event("non-json", query=True))
    assert outcome["status"] == "fallback"
    assert outcome["failure"]["kind"] == "query"


def test_same_runtime_recursive_query_is_rejected():
    holder = {}

    def recursive(_method, _args):
        return holder["runtime"].invoke(cache_event("recursive-cache"))

    holder["runtime"] = Runtime(str(POLICY), query=recursive)
    holder["runtime"].invoke(admit_event("recursive"))
    outcome = holder["runtime"].invoke(route_event("recursive", query=True))
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
        return {"queue_bias": 0}

    runtime = Runtime(str(POLICY), query=query)
    runtime.invoke(admit_event("first"))
    runtime.invoke(admit_event("second"))
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
        runtime.invoke(
            {
                "api_version": "wrong",
                "operation": "route",
                "context": {},
            }
        )

    def invoke(index):
        return bootstrap(Runtime(str(POLICY)), f"request-{index}")["status"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(invoke, range(8))) == ["success"] * 8
