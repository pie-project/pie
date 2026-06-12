// Runtime functions — wraps pie:core/runtime WIT interface.

import * as _rt from 'pie:core/runtime';

/** Returns the runtime version string. */
export function version(): string {
  return _rt.version();
}

/** Returns a unique identifier for the running instance. */
export function instanceId(): string {
  return _rt.instanceId();
}

/** Returns the username of the user who invoked the inferlet. */
export function username(): string {
  return _rt.username();
}

/** Returns a list of all available model names. */
export function models(): string[] {
  return _rt.models();
}

/** Store or overwrite engine-lifetime metadata for this caller. */
export function metadataPut(namespace: string, key: string, value: Uint8Array): void {
  _rt.metadataPut(namespace, key, value);
}

/** Retrieve engine-lifetime metadata for this caller, if present. */
export function metadataGet(namespace: string, key: string): Uint8Array | undefined {
  return _rt.metadataGet(namespace, key);
}

/** Delete engine-lifetime metadata for this caller. */
export function metadataDelete(namespace: string, key: string): boolean {
  return _rt.metadataDelete(namespace, key);
}
