// Internal async utilities for WASI pollable futures.

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
 * Uses `pollable.block()` which componentize-js / StarlingMonkey handles
 * natively — it suspends the current execution context and resumes when
 * the pollable becomes ready, without conflicting with the JS event loop.
 *
 * @internal
 */
export async function awaitFuture<T>(future: WasiFuture<T>, errorMessage: string): Promise<T> {
    const pollable = future.pollable();
    while (!pollable.ready()) {
        pollable.block();
    }

    const result = future.get();
    if (result === undefined) {
        throw new Error(errorMessage);
    }
    return result;
}
