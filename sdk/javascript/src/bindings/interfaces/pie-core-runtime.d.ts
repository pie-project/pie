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
 * Get a list of all available model names
 */
export function models(): Array<string>;
export type Error = import('./pie-core-types.js').Error;
