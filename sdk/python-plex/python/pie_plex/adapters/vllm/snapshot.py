def schedule_event(scheduler):
    return {
        "api_version": "pie.plex.engine@1",
        "hook": "schedule",
        "context": {
            "runnable": scheduler.plex_runnable(),
            "capacity": scheduler.plex_capacity(),
            "context": scheduler.plex_context(),
        },
        "request_events": scheduler.plex_request_events(),
    }
