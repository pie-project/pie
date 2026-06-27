// Adapter (LoRA) wrapper — wraps pie:core/adapter WIT resource.

import { Adapter as _Adapter } from 'pie:core/adapter';

/**
 * A LoRA adapter instance.
 *
 * Wraps the `pie:core/adapter.Adapter` WIT resource. Implements
 * `Disposable` for use with `using`:
 *
 *     using adapter = Adapter.create("my-lora");
 *     adapter.load("/path/to/weights");
 *     // adapter.destroy() called automatically on scope exit
 */
export class Adapter implements Disposable {
    /** @internal */
    readonly _handle: _Adapter;

    private constructor(handle: _Adapter) {
        this._handle = handle;
    }

    /** Disposable protocol — calls `destroy()`. */
    [Symbol.dispose](): void {
        this.destroy();
    }

    /** Create a new adapter for the model with the given name. */
    static create(name: string): Adapter {
        return new Adapter(_Adapter.create(name));
    }

    /** Open an existing adapter by name. Returns `undefined` if not found. */
    static open(name: string): Adapter | undefined {
        const handle = _Adapter.open(name);
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

    /** Load adapter weights from a file path. */
    load(path: string): void {
        this._handle.load(path);
    }

    /** Save adapter weights to a file path. */
    save(path: string): void {
        this._handle.save(path);
    }
}
