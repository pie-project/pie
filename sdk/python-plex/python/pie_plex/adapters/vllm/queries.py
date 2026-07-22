def query_handler(scheduler):
    return lambda method, args: scheduler.plex_query(method, args)
