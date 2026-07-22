from .apply import schedule_outcome
from .snapshot import schedule_event


class PlexSchedulerAdapter:
    def __init__(self, scheduler, runtime):
        self.scheduler = scheduler
        self.runtime = runtime

    def schedule(self):
        outcome = self.runtime.invoke(schedule_event(self.scheduler))
        if outcome["status"] != "success":
            return self.scheduler.native_schedule()
        return schedule_outcome(self.scheduler, outcome)
