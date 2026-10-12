import { serve } from "./serve.mjs";

import { launchChromium } from "./browsers.mjs";

// `headless.mjs <root>... <page.html?...>... [--timeout ms]`: serves the roots
// and opens each page in a browser of its own, all at once; a page's console
// streams when it is the only one and prints whole when it is done otherwise.
// Exits non-zero when any page does not report status 0 within the timeout.
const args = process.argv.slice(2);
const timeoutIdx = args.indexOf("--timeout");
const timeout = timeoutIdx >= 0 ? Number(args.splice(timeoutIdx, 2)[1]) : 600_000;
const paths = args.filter((a) => a.includes(".html"));
const roots = args.filter((a) => !a.includes(".html"));

async function open(path, log) {
  const browser = await launchChromium();
  const page = await browser.newPage();
  let done;
  const finished = new Promise((r) => (done = r));
  await page.exposeFunction("__pie_done", (status) => done(status));
  page.on("console", (m) => log(m.text()));
  page.on("pageerror", (e) => log(`pageerror: ${e.message}`));
  page.on("crash", () => done(-99));
  await page.goto(`http://127.0.0.1:${port}/${path}`);
  const status = await Promise.race([finished, new Promise((r) => setTimeout(() => r(-98), timeout))]);
  log(`[headless] status ${status}`);
  await browser.close();
  return status;
}

const { server, port } = await serve(roots);
const statuses = await Promise.all(
  paths.map(async (path) => {
    if (paths.length === 1) return open(path, console.log);
    const lines = [];
    const status = await open(path, (line) => lines.push(line));
    console.log(`== ${path.split("?")[0]}\n${lines.join("\n")}`);
    return status;
  }),
);
server.close();
process.exit(statuses.every((s) => s === 0) ? 0 : 1);
