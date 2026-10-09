// Precompile a language component for the engine a tab runs inferlets on.
// A tab compiles the Python language component (37 MB) on its first Python
// inferlet, which takes seconds; a site can run this once at build time and
// serve the result beside the source, for `Server.installLanguage(language,
// source, { precompiled })`. The result fits only this exact build of
// pie_browser_bg.wasm, so run the copy that ships with the page.
//
//   node precompile.mjs python.wasm python.pulley.cwasm
//
// Needs no GPU: it builds the inferlet engine, not a server.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { initSync, pie_precompile_component } from "../pkg/pie_browser.js";

let ready = false;

/** The component's bytes, precompiled for this build's engine. */
export function precompileComponent(component) {
  if (!ready) {
    initSync({ module: readFileSync(new URL(__PIE_WASM__, import.meta.url)) });
    ready = true;
  }
  return pie_precompile_component(new Uint8Array(component));
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const [source, out] = process.argv.slice(2);
  if (!source || !out) {
    console.error("usage: node precompile.mjs <component.wasm> <out.cwasm>");
    process.exit(2);
  }
  const started = performance.now();
  const precompiled = precompileComponent(readFileSync(source));
  writeFileSync(out, precompiled);
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  console.log(`${out}: ${(precompiled.length / 1048576).toFixed(1)} MiB in ${seconds} s`);
}
