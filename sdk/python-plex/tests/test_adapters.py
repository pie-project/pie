from pie_plex import Runtime
from pie_plex.adapters.sglang import PlexSchedulerAdapter as SGLangAdapter
from pie_plex.adapters.vllm import PlexSchedulerAdapter as VllmAdapter

from test_runtime import POLICY, admit_event, request_ref, route_event


class MockScheduler:
    def __init__(self):
        self.actions = []
        self.applied = None

    def plex_runnable(self):
        return [
            {
                "request": request_ref("L"),
                "facts": {},
                "max_token_budget": 8,
            }
        ]

    def plex_capacity(self):
        return {
            "max_selections": 1,
            "max_requests": 1,
            "max_total_tokens": 8,
            "facts": {},
        }

    def plex_opportunity_id(self):
        return "adapter-schedule"

    def plex_attempt(self):
        return 0

    def plex_schedule_cause(self):
        return "capacity-changed"

    def plex_lifecycle_events(self):
        return [{"event": "activate-request", "request_id": "L"}]

    def apply_plex_schedule(self, plan, state_update):
        self.applied = (plan, state_update)
        return plan

    def apply_plex_action(self, action):
        self.actions.append(action)

    def native_schedule(self):
        return "native"


def prepared_runtime():
    runtime = Runtime(str(POLICY))
    runtime.invoke(admit_event("L"))
    runtime.invoke(route_event("L"))
    return runtime


def test_vllm_and_sglang_mock_adapters_apply_identical_outcomes():
    reports = []
    for adapter_type in (VllmAdapter, SGLangAdapter):
        scheduler = MockScheduler()
        result = adapter_type(scheduler, prepared_runtime()).schedule()
        reports.append((result, scheduler.applied, scheduler.actions))
    assert reports[0] == reports[1]
    assert reports[0][0] == {
        "selections": [{"requests": [0], "token_budgets": [8]}]
    }


def test_mock_adapters_select_native_fallback():
    class FallbackRuntime:
        def invoke(self, _event):
            return {"status": "fallback"}

    for adapter_type in (VllmAdapter, SGLangAdapter):
        assert adapter_type(MockScheduler(), FallbackRuntime()).schedule() == "native"
