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
