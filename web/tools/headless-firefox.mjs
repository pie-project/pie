// The Chrome runner's sibling for Firefox: same arguments, same exit contract
// (`window.__pie_done(status)`), with the prefs that turn on WebGPU and JSPI
// in Firefox. `node headless-firefox.mjs <root>... <path> [--timeout ms]`.
import { serve } from "./serve.mjs";

import { firefox } from "./browsers.mjs";

const args = process.argv.slice(2);
const timeoutIdx = args.indexOf("--timeout");
const timeout = timeoutIdx >= 0 ? Number(args.splice(timeoutIdx, 2)[1]) : 600_000;
const path = args.pop();
const roots = args;

const { server, port } = await serve(roots);
const browser = await firefox.launch({
  headless: true,
  firefoxUserPrefs: {
    "dom.webgpu.enabled": true,
    "dom.webgpu.workers.enabled": true,
    "gfx.webgpu.ignore-blocklist": true,
    "gfx.webrender.all": true,
    "javascript.options.wasm_js_promise_integration": true,
    "javascript.options.wasm_memory64": false,
  },
});
const page = await browser.newPage();
let done;
const finished = new Promise((r) => (done = r));
await page.exposeFunction("__pie_done", (status) => done(status));
page.on("console", (m) => console.log(m.text()));
page.on("pageerror", (e) => console.log(`pageerror: ${e.message}`));
page.on("crash", () => done(-99));
const caps = await page.evaluate(async () => {
  const out = { jspi: typeof WebAssembly.Suspending === "function" && typeof WebAssembly.promising === "function", webgpu: !!navigator.gpu };
  if (navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter();
      out.adapter = a ? (a.info?.description || a.info?.vendor || "adapter") : null;
      out.maxStorageBuffersPerShaderStage = a?.limits?.maxStorageBuffersPerShaderStage;
    } catch (e) {
      out.adapterError = String(e);
    }
  }
  return out;
});
console.log(`[firefox] ${navigator_version()} caps: ${JSON.stringify(caps)}`);
function navigator_version() {
  return browser.version();
}
await page.goto(`http://127.0.0.1:${port}/${path}`);
const status = await Promise.race([finished, new Promise((r) => setTimeout(() => r(-98), timeout))]);
console.log(`[headless] status ${status}`);
await browser.close();
server.close();
process.exit(status === 0 ? 0 : 1);
