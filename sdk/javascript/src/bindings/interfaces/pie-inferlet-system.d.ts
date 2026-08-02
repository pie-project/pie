/** @module Interface pie:inferlet/system@0.3.0 **/
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
