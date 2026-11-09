const SHORT = "The capital of France is";

function longPrompt(words) {
  const out = [];
  for (let i = 1; out.length < words; i++) out.push(`${i},`);
  return "Count: " + out.join(" ") + " and the next number is";
}

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

async function complete(client, program, input) {
  const t0 = now();
  const proc = await client.launchProcess(program, input);
  const tLaunch = now();
  for (;;) {
    const { event, value } = await proc.recv();
    if (event === "return") {
      return { ms: now() - t0, launchMs: tLaunch - t0, ret: JSON.parse(value) };
    }
    if (event === "error") throw new Error(value);
  }
}

export async function runCorners(client, program, log = () => {}, opts = {}) {
  const out = [];
  const check = async (name, fn, expect) => {
    try {
      const detail = await fn();
      const ok = expect ? expect(detail) : true;
      out.push({ name, ok, detail: typeof detail === "string" ? detail : JSON.stringify(detail) });
    } catch (e) {
      out.push({ name, ok: false, detail: `threw: ${e.message ?? e}` });
    }
    const last = out[out.length - 1];
    log(`${last.ok ? "ok  " : "FAIL"} ${name}: ${last.detail.slice(0, 160)}`);
  };

  await check("empty prompt", async () => (await complete(client, program, { prompt: "", max_tokens: 4 })).ret, (r) => r.count === 4);
  await check("unicode prompt", async () => (await complete(client, program, { prompt: "프랑스의 수도는", max_tokens: 8 })).ret, (r) => r.count === 8 && typeof r.text === "string");
  await check("max_tokens 0", async () => (await complete(client, program, { prompt: SHORT, max_tokens: 0 })).ret, (r) => r.count === 0);
  await check("long generation (256)", async () => (await complete(client, program, { prompt: SHORT, max_tokens: 256 })).ret, (r) => r.count === 256);
  await check("invalid input json → error event", async () => {
    const proc = await client.launchProcess(program, "not json at all");
    const { event, value } = await proc.recv();
    return `${event}: ${String(value).slice(0, 80)}`;
  }, (d) => d.startsWith("error"));
  await check("unknown program → launch refused", async () => {
    try {
      await client.launchProcess("no-such-program", { prompt: SHORT });
      return "launched?!";
    } catch (e) {
      return `refused: ${e.message.slice(0, 80)}`;
    }
  }, (d) => d.startsWith("refused"));
  await check("terminate mid-run, then run again", async () => {
    const proc = await client.launchProcess(program, { prompt: SHORT, max_tokens: 400 });
    await new Promise((r) => setTimeout(r, 150));
    await proc.terminate();
    let last = "none";
    for (let i = 0; i < 400; i++) {
      const { event, value } = await proc.recv();
      last = `${event}: ${String(value).slice(0, 60)}`;
      if (event === "return" || event === "error") break;
    }
    const again = await complete(client, program, { prompt: SHORT, max_tokens: 4 });
    return `after terminate: ${last}; next run ok with ${again.ret.count} tokens`;
  }, (d) => d.includes("next run ok with 4"));
  await check("more processes than lanes (6 at once)", async () => {
    const runs = await Promise.all(Array.from({ length: 6 }, (_, i) => complete(client, program, { prompt: `${SHORT} #${i}`, max_tokens: 8 })));
    return runs.map((r) => r.ret.count).join(",");
  }, (d) => d === "8,8,8,8,8,8");
  await check("prompt near max_context (750 words ≈ 3800 tokens)", async () => (await complete(client, program, { prompt: longPrompt(750), max_tokens: 2 })).ret, (r) => r.count === 2);
  await check("prompt past max_context (1200 words) → refused by name", async () => {
    try {
      await complete(client, program, { prompt: longPrompt(1200), max_tokens: 2 });
      return "accepted?!";
    } catch (e) {
      return e.message.includes("max_context") ? "refused, naming max_context" : `refused otherwise: ${e.message.slice(0, 100)}`;
    }
  }, (d) => d === "refused, naming max_context");
  if (opts.makeClient) {
    await check("two clients at once", async () => {
      const other = opts.makeClient();
      await other.connect();
      const [a, b] = await Promise.all([
        complete(client, program, { prompt: SHORT, max_tokens: 6 }),
        complete(other, program, { prompt: "The capital of Germany is", max_tokens: 6 }),
      ]);
      await other.close();
      return `${a.ret.text.trim()} | ${b.ret.text.trim()}`;
    }, (d) => {
      if (opts.fixture) {
        const [a, b] = d.split(" | ");
        return a.length > 0 && b.length > 0 && a !== b;
      }
      return d.includes("Paris") && d.includes("Berlin");
    });
  }
  return out;
}
