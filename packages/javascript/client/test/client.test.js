import assert from 'node:assert/strict';
import test from 'node:test';

import { PieClient } from '../src/index.js';

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
