from pie_plex import Runtime
from pie_plex.adapters.sglang import PlexSchedulerAdapter as SGLangAdapter
from pie_plex.adapters.vllm import PlexSchedulerAdapter as VllmAdapter

from test_runtime import POLICY, route_event


class MockScheduler:
    def __init__(self):
        self.actions = []
        self.applied = None

    def plex_runnable(self):
        return [{"request_id": "L", "facts": {}, "max_token_budget": 8}]

    def plex_capacity(self):
        return {
            "max_selected": 1,
            "max_total_tokens": 8,
            "max_token_budget": 8,
        }

    def plex_context(self):
        return {"capabilities": {"token_budget": True}}

    def plex_request_events(self):
        return []

    def apply_plex_schedule(self, decision, request_fields):
        self.applied = (decision, request_fields)
        return decision

    def apply_plex_action(self, action):
        self.actions.append(action)

    def native_schedule(self):
        return "native"


def prepared_runtime():
    runtime = Runtime(str(POLICY))
    runtime.invoke(route_event())
    runtime.invoke({
        "api_version": "pie.plex.engine@1",
        "hook": "admit",
        "context": {
            "request_id": "L",
            "target": {
                "id": "node-a",
                "facts": {"queue_depth": 1},
            },
            "context": {},
        },
        "request_events": [],
    })
    return runtime


def test_vllm_and_sglang_mock_adapters_apply_identical_outcomes():
    reports = []
    for adapter_type in (VllmAdapter, SGLangAdapter):
        scheduler = MockScheduler()
        result = adapter_type(scheduler, prepared_runtime()).schedule()
        reports.append((result, scheduler.applied, scheduler.actions))
    assert reports[0] == reports[1]
    assert reports[0][0] == {
        "selected": [{"candidate_index": 0, "token_budget": 8}]
    }


def test_mock_adapters_select_native_fallback():
    class FallbackRuntime:
        def invoke(self, _event):
            return {"status": "fallback"}

    for adapter_type in (VllmAdapter, SGLangAdapter):
        assert adapter_type(MockScheduler(), FallbackRuntime()).schedule() == "native"
