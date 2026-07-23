def schedule_outcome(scheduler, outcome):
    result = scheduler.apply_plex_schedule(
        outcome["plan"]["plan"],
        outcome["state_update"],
    )
    for action in outcome["actions"]:
        scheduler.apply_plex_action(action)
    return result
