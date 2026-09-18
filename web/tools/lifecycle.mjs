// Page-lifecycle checks the headless runner cannot express in one page load:
// the same origin loaded twice (the second boot must come from the OPFS
// cache) and a tab that is hidden mid-run (Chrome throttles its timers).
// `node web/tools/lifecycle.mjs web/site`
import { serve } from "./serve.mjs";

import { launchChromium } from "./browsers.mjs";

const root = process.argv[2] ?? "web/site";
const { server, port } = await serve([root]);
const browser = await launchChromium();
// One persistent context, so OPFS survives between the two page loads.
const context = await browser.newContext();

async function runPhase(phase, extra = "") {
  const page = await context.newPage();
  let done;
  const finished = new Promise((r) => (done = r));
  await page.exposeFunction("__pie_done", (status) => done(status));
  // Emulate a hidden tab through the CDP; the page calls these around a run.
  const cdp = await context.newCDPSession(page);
  await page.exposeFunction("__pie_hide", async () => {
    await cdp.send("Emulation.setFocusEmulationEnabled", { enabled: false });
    await page.evaluate(() => {
      Object.defineProperty(document, "visibilityState", { value: "hidden", configurable: true });
      Object.defineProperty(document, "hidden", { value: true, configurable: true });
      document.dispatchEvent(new Event("visibilitychange"));
    });
  });
  await page.exposeFunction("__pie_show", async () => {
    await page.evaluate(() => {
      Object.defineProperty(document, "visibilityState", { value: "visible", configurable: true });
      Object.defineProperty(document, "hidden", { value: false, configurable: true });
      document.dispatchEvent(new Event("visibilitychange"));
    });
  });
  page.on("console", (m) => console.log(`[${phase}] ${m.text()}`));
  page.on("pageerror", (e) => console.log(`[${phase}] pageerror: ${e.message}`));
  await page.goto(`http://127.0.0.1:${port}/lifecycle.html?phase=${phase}${extra}`);
  const status = await Promise.race([finished, new Promise((r) => setTimeout(() => r(-98), 600_000))]);
  await page.close();
  return status;
}

const first = await runPhase("first");
const second = await runPhase("reload", "&hide");
console.log(`[lifecycle] first=${first} reload=${second}`);
await browser.close();
server.close();
process.exit(first === 0 && second === 0 ? 0 : 1);
