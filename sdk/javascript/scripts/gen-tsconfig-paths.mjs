// Rewrite tsconfig.json's `paths` from the generated bindings.
//
// Why this exists: `paths` maps a WIT import specifier (`pie:inferlet/chat`)
// onto the `.d.ts` jco emits for it. It used to be maintained by hand, and it
// drifted exactly the way the SDKs did -- it still listed `pie:core/inference`
// and `pie:instruct/chat` long after the WIT package was renamed to
// `pie:inferlet`, so every stale entry silently resolved and every new
// interface silently did not. `npm run generate-bindings` now runs this, so
// the map cannot outlive the world it describes.
//
// Source of truth is the world file jco writes (`src/bindings/inferlet.d.ts`),
// whose every line carries the exact specifier in a trailing comment:
//
//   export type * as PieInferletChat030 from './interfaces/pie-inferlet-chat.js'; // import pie:inferlet/chat@0.3.0
//
// Both the versioned and the unversioned specifier are mapped: WIT packages
// that carry a version can be imported either way, and which one a module
// spells is not something this script should get an opinion about.

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const world = readFileSync(join(root, 'src/bindings/inferlet.d.ts'), 'utf8');

const paths = {};
for (const line of world.split('\n')) {
  const m = line.match(/from '\.\/(interfaces\/[\w.-]+)\.js';\s*\/\/ (?:import|export) (\S+)/);
  if (!m) continue;
  const [, rel, spec] = m;
  const target = [`./src/bindings/${rel}.js`];
  paths[spec] = target;
  const unversioned = spec.replace(/@[\d.]+$/, '');
  if (unversioned !== spec) paths[unversioned] = target;
}

if (Object.keys(paths).length === 0) {
  throw new Error('no import specifiers found in src/bindings/inferlet.d.ts');
}

const tsconfigPath = join(root, 'tsconfig.json');
const tsconfig = JSON.parse(readFileSync(tsconfigPath, 'utf8'));
tsconfig.compilerOptions.paths = Object.fromEntries(
  Object.keys(paths).sort().map((k) => [k, paths[k]]),
);
writeFileSync(tsconfigPath, JSON.stringify(tsconfig, null, 2) + '\n');

console.log(`tsconfig.json: wrote ${Object.keys(paths).length} path mappings`);
