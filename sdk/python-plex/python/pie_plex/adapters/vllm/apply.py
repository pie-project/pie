def schedule_outcome(scheduler, outcome):
    result = scheduler.apply_plex_schedule(
        outcome["decision"],
        outcome["request_fields"],
    )
    for action in outcome["actions"]:
        scheduler.apply_plex_action(action)
    return result
