import { createRequire } from "node:module";
import { PieClient } from "../../../javascript/client/src/index.js";
import { runBench, runCorners } from "../../../tests/browser/bench.mjs";

const require = createRequire(new URL("../../../javascript/client/package.json", import.meta.url));
const WsWebSocket = require("ws");
globalThis.WebSocket = class extends WsWebSocket {
  constructor(url) {
    super(url, { headers: { "x-pie-identity": "bench" } });
  }
};

const uri = process.argv[2] ?? "ws://127.0.0.1:28417/v1/ws";
const what = process.argv[3] ?? "both";
const root = new URL("../../..", import.meta.url).pathname;

const client = new PieClient(uri);
await client.connect();
const program = await client.installProgram(`${root}examples/target/wasm32-wasip2/release/text_completion.wasm`, null, true);
const carried = await client.installProgram(`${root}examples/target/wasm32-wasip2/release/naive_baseline.wasm`, null, true);
const opts = {
  carried,
  makeClient: () => new PieClient(uri),
  fixture: process.env.PIE_BENCH_FIXTURE === "1",
};
const result = { host: "native", uri };
if (what !== "corners") result.bench = await runBench(client, program, console.log, opts);
if (what !== "bench") result.corners = await runCorners(client, program, console.log, opts);
await client.close();
console.log("RESULT " + JSON.stringify(result));
