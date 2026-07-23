def schedule_event(scheduler):
    return {
        "api_version": "pie.plex.engine@2",
        "operation": "schedule",
        "context": {
            "meta": {
                "opportunity_id": scheduler.plex_opportunity_id(),
                "snapshot": {"id": "host-filled", "revision": 0},
                "attempt": scheduler.plex_attempt(),
                "mechanics": [],
            },
            "cause": scheduler.plex_schedule_cause(),
            "runnable": scheduler.plex_runnable(),
            "capacity": scheduler.plex_capacity(),
        },
        "lifecycle": scheduler.plex_lifecycle_events(),
    }
