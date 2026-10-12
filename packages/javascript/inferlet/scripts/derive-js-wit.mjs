// Derive the JavaScript-facing WIT world from the runtime's canonical one.
//
// The runtime's `pie:inferlet` world is a component-model-async world: `run`
// is an `async func` export and the channel/session reads are `async func`
// imports. componentize-js (StarlingMonkey) can export an async function but
// "imported functions can only be synchronous pending component-model-level
// async support", and wasmtime types an `async func` import distinctly, so
// a JS component cannot link the async imports at all.
//
// The host therefore carries blocking twins (`channel.take-blocking`,
// `channel.read-blocking`, `session.receive-blocking`, ...), and a JS guest is
// built against THIS derived copy of the world: every `async func` import is
// dropped (the twins stay), `run` is exported as a plain `func`, and the
// wasi 0.3 imports (whose `stream`/`future` types the JS toolchain has no
// bindings for) are removed. Everything else is byte-for-byte the canonical
// text, so a JS component imports exactly the interfaces the host serves.
//
// Usage: node scripts/derive-js-wit.mjs [<src wit dir>] [<dst wit dir>]

import { cpSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = process.argv[2] ?? join(here, '..', '..', '..', '..', 'crates', 'inferlet', 'wit');
const dst = process.argv[3] ?? join(here, '..', 'wit');

export function deriveText(name, text) {
  if (name === 'world.wit') {
    return text
      .split('\n')
      .filter((line) => !/^\s*import wasi:(http|clocks|filesystem)\//.test(line))
      .join('\n');
  }
  if (name === 'run.wit') {
    return text.replace(/async func/g, 'func');
  }
  // Drop every `name: async func(...)...;` declaration (one line each in the
  // canonical text) and the doc comment block immediately above it.
  const lines = text.split('\n');
  const out = [];
  for (const line of lines) {
    if (/^\s*[a-z0-9-]+:\s*async func/.test(line)) {
      while (out.length && /^\s*\/\/\//.test(out[out.length - 1])) out.pop();
      continue;
    }
    out.push(line);
  }
  return out.join('\n');
}

rmSync(dst, { recursive: true, force: true });
mkdirSync(dst, { recursive: true });
for (const entry of readdirSync(src, { withFileTypes: true })) {
  if (entry.isFile() && entry.name.endsWith('.wit')) {
    writeFileSync(join(dst, entry.name), deriveText(entry.name, readFileSync(join(src, entry.name), 'utf8')));
  }
}
cpSync(join(src, 'deps'), join(dst, 'deps'), { recursive: true });
console.log(`derived JS world: ${src} -> ${dst}`);
