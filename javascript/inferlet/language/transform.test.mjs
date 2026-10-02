import { test } from 'node:test';
import assert from 'node:assert/strict';
import { describeError, evaluateModule } from './transform.js';

async function load(source, modules = {}) {
  const require = (spec) => {
    if (!(spec in modules)) throw new Error(`no module ${spec}`);
    return modules[spec];
  };
  return await evaluateModule(source, 'probe.js', require);
}

test('named, default and namespace imports bind from the module table', async () => {
  const exports = await load(
    `import { a, b as bee } from 'lib';\nimport * as ns from 'lib';\nimport d from 'lib';\n` +
    `export function main() { return [a, bee, ns.a, d]; }`,
    { lib: { a: 1, b: 2, default: 3 } },
  );
  assert.deepEqual(exports.main(), [1, 2, 1, 3]);
});

test('exports of every shape land on the exports object', async () => {
  const exports = await load(
    `export const x = 1, { y } = { y: 2 };\nfunction hidden() { return 3; }\nexport { hidden as shown };\n` +
    `export default async function (input) { return input.n + x; }\nexport class K {}\n`,
  );
  assert.equal(exports.x, 1);
  assert.equal(exports.y, 2);
  assert.equal(exports.shown(), 3);
  assert.equal(await exports.default({ n: 41 }), 42);
  assert.equal(typeof exports.K, 'function');
});

test('a top-level await and an anonymous default expression are fine', async () => {
  const exports = await load(`const v = await Promise.resolve(7);\nexport default v * 2`);
  assert.equal(exports.default, 14);
});

test('a stack trace names the program file and its own line numbers', async () => {
  const exports = await load(`import { a } from 'lib';\n\nexport function main() {\n  throw new Error('boom ' + a);\n}\n`, { lib: { a: 1 } });
  let described = '';
  try { exports.main(); } catch (e) { described = describeError(e); }
  assert.match(described, /^Error: boom 1/);
  assert.match(described, /probe\.js:4/);
});
