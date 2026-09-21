import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { PieClient } from "../../../javascript/client/src/index.js";

const require = createRequire(new URL("../../../javascript/client/package.json", import.meta.url));
const WsWebSocket = require("ws");
globalThis.WebSocket = class extends WsWebSocket {
  constructor(url) {
    super(url, { headers: { "x-pie-identity": "matrix" } });
  }
};

const uri = process.argv[2] ?? "ws://127.0.0.1:28417/v1/ws";
const only = process.argv[3];
const entryTimeoutMs = Number(process.env.PIE_MATRIX_ENTRY_TIMEOUT_MS ?? 60000);
const root = new URL("../../..", import.meta.url).pathname;
const list = JSON.parse(readFileSync(`${root}tests/browser/matrix.json`, "utf8")).filter((e) => !only || e.id === only);

const client = new PieClient(uri);
await client.connect();
let failures = 0;
let timeouts = 0;
for (const entry of list) {
  const t0 = performance.now();
  let status = "ok";
  let output = "";
  try {
    const file = entry.inferlet.replace(/-/g, "_");
    const manifest = `${root}examples/${entry.inferlet}/Pie.toml`;
    await client.installProgram(`${root}examples/target/wasm32-wasip2/release/${file}.wasm`, manifest, true);
    const toml = readFileSync(manifest, "utf8");
    const program = `${toml.match(/^name\s*=\s*"([^"]+)"/m)[1]}@${toml.match(/^version\s*=\s*"([^"]+)"/m)[1]}`;
    const proc = await client.launchProcess(program, entry.input);
    const streamed = [];
    let returned = null;
    const deadline = Date.now() + entryTimeoutMs;
    for (;;) {
      const left = deadline - Date.now();
      if (left <= 0) {
        status = `timeout: ${entryTimeoutMs} ms`;
        timeouts += 1;
        await proc.terminate?.().catch(() => {});
        break;
      }
      const next = await Promise.race([proc.recv(), new Promise((r) => setTimeout(() => r(null), left))]);
      if (next === null) continue;
      const { event, value } = next;
      if (event === "stdout" || event === "message") streamed.push(value);
      else if (event === "return") { returned = value; break; }
      else if (event === "error") { status = `error: ${String(value).slice(0, 200)}`; break; }
    }
    output = streamed.join("") + (returned === null ? "" : `⏎${returned}`);
  } catch (e) {
    status = `threw: ${String(e.message ?? e).slice(0, 200)}`;
  }
  if (status !== "ok") failures += 1;
  console.log(`MATRIX ${entry.id}\t${status}\t${JSON.stringify(output)}\t${(performance.now() - t0).toFixed(0)} ms`);
}
console.log(`RESULT ${JSON.stringify({ host: "native", entries: list.length, failures, timeouts })}`);
await client.close();
process.exit(timeouts ? 3 : failures ? 1 : 0);
