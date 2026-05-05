import assert from 'node:assert/strict';
import test from 'node:test';

import { PieClient } from '../src/index.js';

test('client package import and construction work', () => {
    const client = new PieClient('ws://127.0.0.1:8080');
    assert.equal(client.serverUri, 'ws://127.0.0.1:8080');
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
