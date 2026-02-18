// Adapter (LoRA) wrapper — wraps pie:core/adapter WIT resource.

import { Adapter as _Adapter } from 'pie:core/adapter';
import type { Model } from './model.js';
import { awaitFuture } from './_async.js';

/**
 * A LoRA adapter instance.
 *
 * Wraps the `pie:core/adapter.Adapter` WIT resource.
 */
export class Adapter {
    /** @internal */
    readonly _handle: _Adapter;

    private constructor(handle: _Adapter) {
        this._handle = handle;
    }

    /** Create a new adapter for a model with the given name. */
    static create(model: Model, name: string): Adapter {
        return new Adapter(_Adapter.create(model._handle, name));
    }

    /** Open an existing adapter by name. Returns `undefined` if not found. */
    static open(model: Model, name: string): Adapter | undefined {
        const handle = _Adapter.open(model._handle, name);
        return handle !== undefined ? new Adapter(handle) : undefined;
    }

    /** Destroy the adapter, releasing its resources. */
    destroy(): void {
        this._handle.destroy();
    }

    /** Fork this adapter with a new name. */
    fork(name: string): Adapter {
        return new Adapter(this._handle.fork(name));
    }

    /** Acquire an exclusive lock on the adapter. */
    async acquireLock(): Promise<boolean> {
        return awaitFuture(this._handle.acquireLock(), 'acquireLock() returned undefined');
    }

    /** Release the lock on the adapter. */
    releaseLock(): void {
        this._handle.releaseLock();
    }

    /** Load adapter weights from a file path. */
    load(path: string): void {
        this._handle.load(path);
    }

    /** Save adapter weights to a file path. */
    save(path: string): void {
        this._handle.save(path);
    }
}
