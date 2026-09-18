// Open a page in headless Chromium with WebGPU + JSPI enabled, stream its
// console to stdout, and exit with the status the page reports through
// `window.__pie_done(status)`. `node headless.mjs <root>... <path> [--timeout ms]`.
import { serve } from "./serve.mjs";

import { launchChromium } from "./browsers.mjs";

const args = process.argv.slice(2);
const timeoutIdx = args.indexOf("--timeout");
const timeout = timeoutIdx >= 0 ? Number(args.splice(timeoutIdx, 2)[1]) : 600_000;
const path = args.pop();
const roots = args;

const { server, port } = await serve(roots);
const browser = await launchChromium();
const page = await browser.newPage();
let done;
const finished = new Promise((r) => (done = r));
await page.exposeFunction("__pie_done", (status) => done(status));
// The runtime logs from its worker (web/site/worker.mjs); Playwright routes a
// dedicated worker's console to the page's event, so nothing is relayed by
// hand. A worker that dies surfaces as `pie worker: …` from pie.mjs's onerror.
page.on("console", (m) => console.log(m.text()));
page.on("pageerror", (e) => {
  console.log(`pageerror: ${e.message}`);
});
page.on("crash", () => done(-99));
const url = `http://127.0.0.1:${port}/${path}`;
await page.goto(url);
const status = await Promise.race([
  finished,
  new Promise((r) => setTimeout(() => r(-98), timeout)),
]);
console.log(`[headless] status ${status}`);
await browser.close();
server.close();
process.exit(status === 0 ? 0 : 1);
