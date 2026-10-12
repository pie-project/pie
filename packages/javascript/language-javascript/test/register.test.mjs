import { test } from 'node:test';
import assert from 'node:assert/strict';
import { registeredLanguages } from '@pie-project/server/languages';
import '../index.mjs';

test('importing the package registers the javascript component', () => {
  const url = registeredLanguages().get('javascript');
  assert.ok(url instanceof URL && url.pathname.endsWith('/javascript.wasm'), String(url));
});
