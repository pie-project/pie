import { test } from 'node:test';
import assert from 'node:assert/strict';
import { registeredLanguages } from '@pie-project/server/languages';
import '../index.mjs';

test('importing the package registers the python component', () => {
  const url = registeredLanguages().get('python');
  assert.ok(url instanceof URL && url.pathname.endsWith('/python.wasm'), String(url));
});
