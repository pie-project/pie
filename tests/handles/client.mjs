// The JavaScript client against a real server: argv is a JSON object of what
// to run, and it prints what came back for e2e.py to compare.
import { readFileSync } from 'node:fs';
import { PieClient, HandleError } from '../../packages/javascript/client/src/index.js';

class Socket extends WebSocket {}

const { uri, prog, see, png, prompt } = JSON.parse(process.argv[2]);
const client = new PieClient(uri, { WebSocket: Socket, identity: 'default/bench' });
await client.connect();
const moved = (ws) => ({ handles: { ws }, returns: { ws } });
const turn = async (ws, tokens, n, drawn) => {
  const input = { ws: { $handle: 'ws' }, tokens, max_tokens: n, seed: 7, drawn };
  const proc = await client.launchProcess(prog, input, true, {}, moved(ws));
  for (;;) {
    const { event, value } = await proc.recv();
    if (event === 'return') return JSON.parse(value).tokens;
    if (event === 'error') throw new Error(value);
  }
};
const finished = async (proc) => {
  for (;;) {
    const { event, value } = await proc.recv();
    if (event === 'return') return JSON.parse(value);
    if (event === 'error') throw new Error(value);
  }
};

const out = {};
const ws = await client.workingSet();
const first = await turn(ws, prompt, 12, 0);
const fork = await ws.fork();
const rest = await turn(fork, [first[first.length - 1]], 20, 12);
out.tokens = [...first, ...rest];
out.stat = await ws.stat();
await ws.close();
await fork.close();
try {
  await ws.stat();
} catch (e) {
  out.closed = e instanceof HandleError && e.message;
}

const up = await client.frames(new Uint8Array(readFileSync(png)));
out.seen = await finished(await client.run(see, { image: up }));
const raw = await up.read('raw-rgb8');
out.raw = raw.length;
out.png = Array.from((await up.read('png')).slice(1, 4));
await up.close();
console.log(JSON.stringify(out));
process.exit(0);
