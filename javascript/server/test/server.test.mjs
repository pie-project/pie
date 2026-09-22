import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Server, addonPath } from '../index.mjs';

test('the package loads, and says how to build the addon when it is missing', async () => {
  assert.equal(typeof Server.start, 'function');
  if (addonPath() === null) {
    await assert.rejects(Server.start('[server]\nport = 0\n'), /no native addon .*npm run build/);
  }
});

test('with the addon built, a bad config is refused before any engine starts', { skip: addonPath() === null }, async () => {
  await assert.rejects(Server.start('this is not toml'), /config/);
  await assert.rejects(Server.start({ server: { port: -1 } }), /config/);
});
