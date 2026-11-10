// The corner checks against a native `pie serve`, over its websocket, from node (scripts/ci/lavapipe.sh runs them):
//   node tests/browser/tools/native.mjs [ws://127.0.0.1:28417/v1/ws]
import { createRequire } from "node:module";
import { PieClient } from "../../../packages/javascript/client/src/index.js";
import { runCorners } from "../corners.mjs";

const require = createRequire(new URL("../../../packages/javascript/client/package.json", import.meta.url));
const WsWebSocket = require("ws");
globalThis.WebSocket = class extends WsWebSocket {
  constructor(url) {
    super(url, { headers: { "x-pie-identity": "corners" } });
  }
};

const uri = process.argv[2] ?? "ws://127.0.0.1:28417/v1/ws";
const root = new URL("../../..", import.meta.url).pathname;

const client = new PieClient(uri);
await client.connect();
const program = await client.installProgram(`${root}examples/target/wasm32-wasip3/release/text_completion.wasm`, null, true);
const opts = {
  makeClient: () => new PieClient(uri),
  // a fixture model answers nonsense: the two-client check then only asks for two different answers
  fixture: process.env.PIE_CORNERS_FIXTURE === "1",
};
const result = { host: "native", uri, corners: await runCorners(client, program, console.log, opts) };
await client.close();
console.log("RESULT " + JSON.stringify(result));
