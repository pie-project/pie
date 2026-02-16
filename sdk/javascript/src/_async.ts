// Internal async utilities for WASI pollable futures.

import { pollLoop } from './_poll_loop.js';

/** Minimal pollable interface (matches wasi:io/poll@0.2.4 Pollable). */
interface Pollable {
    ready(): boolean;
    block(): void;
}

/** Generic future interface for WASI async operations. */
export interface WasiFuture<T> {
    pollable(): Pollable;
    get(): T | undefined;
}

/**
 * Awaits a WASI future cooperatively.
 *
 * Registers the future's pollable with the PollLoop and yields
 * control, allowing other futures to make progress concurrently.
 *
 * @internal
 */
export async function awaitFuture<T>(future: WasiFuture<T>, errorMessage: string): Promise<T> {
    const pollable = future.pollable();
    await pollLoop.register(pollable);

    const result = future.get();
    if (result === undefined) {
        throw new Error(errorMessage);
    }
    return result;
}
