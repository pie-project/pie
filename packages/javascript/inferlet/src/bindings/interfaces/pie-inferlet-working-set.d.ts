/** @module Interface pie:inferlet/working-set@0.3.0 **/
export type Error = import('./pie-inferlet-types.js').Error;
export type Pipeline = import('./pie-inferlet-pipeline.js').Pipeline;
/**
 * Who a working set shares its frozen slots with. Each is within the
 * process's authenticated tenant, never a tenant the inferlet names:
 * `session` within its session (the one its launch named, else the
 * process alone); `tenant` within the tenant; `group(name)` within the
 * named group, which crosses tenants only where the operator's grant
 * covers this tenant, and is the tenant's own otherwise.
 */
export type ShareScope = ShareScopeSession | ShareScopeTenant | ShareScopeGroup;
export interface ShareScopeSession {
  tag: 'session',
}
export interface ShareScopeTenant {
  tag: 'tenant',
}
export interface ShareScopeGroup {
  tag: 'group',
  val: string,
}

export class WorkingSet {
  constructor()
  /**
  * Let passes write slots below `end`: needed only when a pass's slots
  * are computed on the device.
  */
  reserve(end: number): void;
  /**
  * The frozen length; while a device-counted fold is in flight, the
  * length before it.
  */
  folded(): number;
  /**
  * A child that shares the frozen part and gets a copy of the buffer,
  * ordered on `on`. It shares as its parent does.
  */
  fork(on: Pipeline): WorkingSet;
  /**
  * Publishes this set's frozen slots under `scope` and lets its passes
  * map, wait on, fetch and resume from what that scope holds. Unasked:
  * private. Asked before the set's first pass; refused after it.
  */
  share(scope: ShareScope): void;
}
