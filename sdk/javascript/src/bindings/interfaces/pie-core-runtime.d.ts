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
/**
 * Store or overwrite engine-lifetime inferlet metadata under a namespace.
 */
export function metadataPut(namespace: string, key: string, value: Blob): void;
/**
 * Retrieve engine-lifetime inferlet metadata from a namespace.
 */
export function metadataGet(namespace: string, key: string): Blob | undefined;
/**
 * Delete engine-lifetime inferlet metadata from a namespace.
 */
export function metadataDelete(namespace: string, key: string): boolean;
export type Blob = import('./pie-core-types.js').Blob;
export type Error = import('./pie-core-types.js').Error;
