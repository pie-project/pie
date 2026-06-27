/** @module Interface pie:core/runtime **/
/**
 * Returns the runtime version string
 */
export function version(): string;
/**
 * Returns a unique identifier for the running instance
 */
export function instanceId(): string;
/**
 * Returns the username of the user who invoked the inferlet
 */
export function username(): string;
/**
 * Suspends the calling inferlet for the given duration. A host-provided
 * async timer: under component-model-async there is no guest-side
 * pollable->future bridge for wasi:clocks, so timing lives host-side.
 */
export function sleep(durationNs: bigint): Promise<void>;
