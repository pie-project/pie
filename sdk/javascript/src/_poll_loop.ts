// PollLoop — cooperative async scheduler backed by wasi:io/poll.
//
// Mirrors the Python SDK's poll_loop.py. Enables real concurrency:
// multiple WASI futures can be awaited concurrently via Promise.all().

import { poll, type Pollable } from 'wasi:io/poll@0.2.4';

interface Waker {
    pollable: Pollable;
    resolve: () => void;
}

/**
 * Cooperative poll loop for WASI futures.
 *
 * Registered pollables are multiplexed via `wasi:io/poll#poll`,
 * which blocks until at least one is ready. This enables true
 * concurrency within a single WASM component.
 *
 * @internal
 */
class PollLoop {
    private wakers: Waker[] = [];
    private microtasks: Array<() => void> = [];

    /** Register a pollable; returns a Promise that resolves when ready. */
    register(pollable: Pollable): Promise<void> {
        return new Promise<void>(resolve => {
            this.wakers.push({ pollable, resolve });
        });
    }

    /** Schedule a microtask for the next poll cycle. */
    schedule(cb: () => void): void {
        this.microtasks.push(cb);
    }

    /**
     * Drive the loop until the given promise settles.
     *
     * On each iteration:
     * 1. Drain queued microtasks (resolved promise callbacks).
     * 2. Collect all registered pollables and call `poll()`.
     * 3. Resolve wakers whose pollables are ready.
     * 4. Repeat until `task` settles.
     */
    runUntilComplete<T>(task: Promise<T>): T {
        let result: { value: T } | undefined;
        let error: { value: unknown } | undefined;

        task.then(
            v => { result = { value: v }; },
            e => { error = { value: e }; },
        );

        while (result === undefined && error === undefined) {
            // 1. Drain microtasks
            while (this.microtasks.length > 0) {
                const batch = this.microtasks;
                this.microtasks = [];
                for (const cb of batch) cb();
            }

            // Check again after draining
            if (result !== undefined || error !== undefined) break;

            // 2. Poll all registered pollables
            if (this.wakers.length > 0) {
                const pollables = this.wakers.map(w => w.pollable);
                const readyIndices = poll(pollables);
                const readySet = new Set<number>();
                for (const idx of readyIndices) readySet.add(idx);

                const remaining: Waker[] = [];
                for (let i = 0; i < this.wakers.length; i++) {
                    if (readySet.has(i)) {
                        this.wakers[i].resolve();
                    } else {
                        remaining.push(this.wakers[i]);
                    }
                }
                this.wakers = remaining;
            } else if (this.microtasks.length === 0) {
                // No wakers and no microtasks — deadlock guard
                break;
            }
        }

        if (error !== undefined) {
            throw error.value;
        }
        if (result !== undefined) {
            return result.value;
        }
        throw new Error('PollLoop: deadlock — no pending pollables or microtasks');
    }
}

/** Singleton poll loop instance. */
export const pollLoop = new PollLoop();
