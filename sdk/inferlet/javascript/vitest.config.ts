// vitest config for the inferlet SDK's unit tests.
//
// The hand-written modules import WIT specifiers directly
// (`import * as _chat from 'pie:inferlet/chat'`). Under `tsc` those resolve
// through tsconfig's generated `paths` map to jco's `.d.ts` files, which carry
// types but no implementations; under `jco componentize` they resolve to the
// host. Neither is loadable by a plain Node ESM run, so vitest aliases them to
// the stubs in `src/__tests__/mocks/`.
//
// Adding an interface to the SDK means adding it here too — otherwise its
// tests fail to resolve, which is the intended failure mode rather than a
// silent skip.

import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

const mock = (name: string) =>
  fileURLToPath(new URL(`./src/__tests__/mocks/${name}.ts`, import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      'pie:inferlet/model': mock('model'),
      'pie:inferlet/tokenizer': mock('tokenizer'),
      'pie:inferlet/session': mock('session'),
      'pie:inferlet/chat': mock('chat'),
      'pie:inferlet/reasoning': mock('reasoning'),
    },
  },
  test: {
    include: ['src/__tests__/**/*.test.ts'],
    environment: 'node',
  },
});
