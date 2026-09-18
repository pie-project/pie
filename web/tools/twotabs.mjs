// Two tabs of the same origin, each booting its own runtime on the same GPU
// at the same time — what a user who opens the page twice gets. Both must
// return the reference completion; the run reports each tab's timings.
// `node web/tools/twotabs.mjs web/site`
import { serve } from "./serve.mjs";

import { launchChromium } from "./browsers.mjs";

const root = process.argv[2] ?? "web/site";
const { server, port } = await serve([root]);
const browser = await launchChromium();
const context = await browser.newContext();

async function tab(name) {
  const page = await context.newPage();
  let done;
  const finished = new Promise((r) => (done = r));
  await page.exposeFunction("__pie_done", (status) => done(status));
  const lines = [];
  page.on("console", (m) => {
    const t = m.text();
    if (/booted|returned|process |failed|ERROR|wasm memory/.test(t)) lines.push(`[${name}] ${t.slice(0, 160)}`);
  });
  page.on("pageerror", (e) => lines.push(`[${name}] pageerror: ${e.message}`));
  const t0 = Date.now();
  await page.goto(`http://127.0.0.1:${port}/index.html?auto&max_tokens=16&runs=2`);
  const status = await Promise.race([finished, new Promise((r) => setTimeout(() => r(-98), 600_000))]);
  lines.push(`[${name}] status ${status} after ${Date.now() - t0} ms`);
  return { status, lines };
}

const [a, b] = await Promise.all([tab("tab A"), tab("tab B")]);
for (const l of [...a.lines, ...b.lines]) console.log(l);
console.log(`[twotabs] A=${a.status} B=${b.status}`);
await browser.close();
server.close();
process.exit(a.status === 0 && b.status === 0 ? 0 : 1);
