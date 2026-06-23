# controller

what it does:
1. Manages a registry of workers and gateways in the cluster,
2. Keep pusing workers (with long polling) about their "neighbors" that they should work with.
3. Keep getting reported load and status from workers.
4. Keep pushing gateways (with long polling) about the available workers and their status, so that gateways can make informed decisions about request routing.
5. Tracks the liveness and load of each worker / gateway.

structure:

1. the rpc server accepts rpc calls from workers and gateways.
  - register worker / register gateway
  - report status from workers
  - heartbeat from workers / gateways
  - watch (long polling) <- returns the controller's updates whenever there's one.

2. the worker registry. which is essnetially a map of workers Map<WorkerId, Worker>, the Worker contains its last heartbeat, misc data (e.g., number of inferlet it is serving, kv cache consumption, the model its serving, etc.)

3. the gateway registry. which is essnetially a map of gateways Map<GatewayId, Gateway>, the Gateway contains its last heartbeat, some stats.

4. the event system. which handles and dispatches events related to worker and gateway lifecycle. This basically links registry and RPC server. And resolves watch long polls.
