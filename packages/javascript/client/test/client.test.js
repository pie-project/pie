import assert from 'node:assert/strict';
import test from 'node:test';

import msgpack from 'msgpack-lite';

import { PieClient, Process } from '../src/index.js';

test('client package import and construction work', () => {
    const client = new PieClient('ws://127.0.0.1:8080');
    assert.equal(client.serverUri, 'ws://127.0.0.1:8080/v1/ws');
});

test('pending requests are rejected and cleared', async () => {
    const client = new PieClient('ws://127.0.0.1:8080');
    const pending = new Promise((resolve, reject) => {
        client.pendingRequests.set(1, { resolve, reject });
    });

    client._rejectPendingRequests(new Error('WebSocket connection closed.'));

    await assert.rejects(pending, /WebSocket connection closed/);
    assert.equal(client.pendingRequests.size, 0);
});

test('close without an open websocket rejects pending requests', async () => {
    const client = new PieClient('ws://127.0.0.1:8080');
    const pending = new Promise((resolve, reject) => {
        client.pendingRequests.set(1, { resolve, reject });
    });

    await client.close();

    await assert.rejects(pending, /WebSocket connection closed/);
    assert.equal(client.pendingRequests.size, 0);
});

test('run leaves an input with no handle in it as it came', async () => {
    const client = new PieClient('ws://127.0.0.1:8080');
    const sent = [];
    client.launchProcess = async (...args) => sent.push(args);
    await client.run('p@1', [1, 2]);
    await client.run('p@1', 'hi');
    await client.run('p@1', null);
    await client.run('p@1', { a: 1 });
    assert.deepEqual(sent.map(([, input]) => input), [[1, 2], 'hi', null, { a: 1 }]);
});

test('run moves a handle in, and a handle yet to exist only back', async () => {
    const client = new PieClient('ws://127.0.0.1:8080');
    const sent = [];
    client.launchProcess = async (...args) => sent.push(args);
    client._handleOp = async () => ({ ok: true });
    const ws = await client.workingSet();
    const out = await client.frames();
    await client.run('p@1', { ws, out, n: 1 });
    const [, input, , , moved] = sent[0];
    assert.deepEqual(input, { n: 1, ws: { $handle: 'ws' } });
    assert.deepEqual(Object.keys(moved.handles), ['ws']);
    assert.deepEqual(Object.keys(moved.returns), ['ws', 'out']);
});

class Queue {
    constructor() { this.values = []; this.waiting = []; }
    put(value) { const w = this.waiting.shift(); if (w) w(value); else this.values.push(value); }
    get() { return this.values.length ? Promise.resolve(this.values.shift()) : new Promise((r) => this.waiting.push(r)); }
}

class FakeSocket {
    static OPEN = 1;
    constructor() {
        this.readyState = FakeSocket.OPEN;
        setTimeout(() => this.onopen());
    }
    close() {}
}

async function withProcesses(ids) {
    const client = new PieClient('ws://127.0.0.1:8080', { WebSocket: FakeSocket });
    await client.connect();
    const processes = ids.map((id) => {
        client.processEventQueues.set(id, new Queue());
        return new Process(client, id);
    });
    return { client, processes };
}

const frame = (message) => ({ data: new Blob([msgpack.encode(message)]) });

test('a message that fails to process fails its process alone', async () => {
    const { client, processes: [bad, good] } = await withProcesses(['bad', 'good']);
    await client.ws.onmessage(frame({
        type: 'file', process_id: 'bad', file_hash: 'h', chunk_index: 0, total_chunks: 1, chunk_data: null,
    }));
    await client.ws.onmessage(frame({ type: 'process_event', process_id: 'good', event: 'return', value: 'ok' }));

    assert.deepEqual(await good.recv(), { event: 'return', value: 'ok' });
    const failed = await bad.recv();
    assert.equal(failed.event, 'error');
    assert.match(failed.value, /could not process a `file` message/);
});

test('a closed connection fails every process on it, saying why', async () => {
    const { client, processes: [a, b] } = await withProcesses(['a', 'b']);
    client.ws.onclose({ code: 1009, reason: 'message too big' });
    for (const process of [a, b]) {
        const { event, value } = await process.recv();
        assert.equal(event, 'error');
        assert.match(value, /1009: message too big/);
    }
    assert.equal(client.processEventQueues.size, 0);
});
