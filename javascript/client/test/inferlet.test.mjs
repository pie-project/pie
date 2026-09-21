import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Inferlet, inferlet, withUpgradePath } from '../src/index.js';

test('a function becomes a module exporting it, named and versioned by its source', () => {
  const tokenCount = inferlet(async function tokenCount(input) {
    return pie.model.encode(input.prompt).length;
  }, { description: 'How many tokens.' });
  assert.ok(tokenCount instanceof Inferlet);
  assert.equal(tokenCount.name, 'tokenCount');
  assert.match(tokenCount.version, /^0\.\d+\.\d+$/);
  assert.equal(tokenCount.program, `tokenCount@${tokenCount.version}`);
  assert.match(tokenCount.source, /^export const main = async function tokenCount\(input\)/);
  const manifest = tokenCount.manifestToml();
  assert.match(manifest, /language = "javascript"/);
  assert.match(manifest, /entry = "main"/);
  assert.match(manifest, /description = "How many tokens."/);
});

test('the same source is the same version and an edit is a new one', () => {
  const a = inferlet((input) => input.x);
  const b = inferlet((input) => input.x);
  const c = inferlet((input) => input.x + 1);
  assert.equal(a.version, b.version);
  assert.notEqual(a.version, c.version);
});

test('a method shorthand is refused, an arrow and a function are taken', () => {
  assert.throws(() => inferlet({ main(input) { return input; } }.main), TypeError);
  assert.equal(inferlet(async (input) => input, { name: 'echo' }).name, 'echo');
  assert.equal(inferlet(function echo(input) { return input; }).name, 'echo');
});

test('a server URI with no path gets /v1/ws, one with a path keeps it', () => {
  assert.equal(withUpgradePath('ws://127.0.0.1:8080'), 'ws://127.0.0.1:8080/v1/ws');
  assert.equal(withUpgradePath('ws://127.0.0.1:8080/'), 'ws://127.0.0.1:8080/v1/ws');
  assert.equal(withUpgradePath('ws://127.0.0.1:8080/v1/ws'), 'ws://127.0.0.1:8080/v1/ws');
  assert.equal(withUpgradePath('wss://pie.example/edge'), 'wss://pie.example/edge');
  assert.equal(withUpgradePath('not a url'), 'not a url');
});
