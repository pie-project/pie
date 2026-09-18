// The test pages' shared floor: the log, the query parameters, the runtime
// load, the inferlet install, a `check` that records a named verdict, and the
// exit the headless runner listens for (`window.__pie_done`).
import * as pie from "./pie.mjs";

export const DEFAULT_MODEL = "models/qwen--qwen3-5-0-8b.qwen35-d0-8b-u4g64-kv-bf16.wgpu.zt";
export const SHORT = "The capital of France is";

/** The page's log (`<pre id="out">` and the console), parameters and verdicts. */
export function page() {
  const out = document.getElementById("out");
  const log = (s) => { out.textContent += s + "\n"; console.log(s); };
  const params = new URLSearchParams(location.search);
  const modelUrl = params.get("model") ?? DEFAULT_MODEL;
  const results = [];
  const verdict = (name, ok, detail) => {
    results.push({ name, ok, detail: String(detail) });
    log(`${ok ? "ok  " : "FAIL"} ${name}: ${String(detail).slice(0, 180)}`);
    return ok;
  };
  /** Runs `fn`; `expect(detail)` judges what it returned (or what it threw, as "threw: …"). */
  const check = async (name, fn, expect) => {
    let detail;
    let ok;
    try {
      detail = await fn();
      ok = expect ? expect(detail) : true;
    } catch (e) {
      detail = `threw: ${e.message ?? e}`;
      ok = expect ? expect(detail) : false;
    }
    return verdict(name, ok, detail);
  };
  const finish = (status) => window.__pie_done?.(status);
  const fail = (e) => { log(`failed: ${e.stack ?? e}`); finish(1); };
  return { log, params, modelUrl, results, check, verdict, finish, fail };
}

/** Load the wasm: `?worker=0` keeps the runtime on this thread, `?log=` filters its console. */
export function loadRuntime(params) {
  return pie.load("./pkg/pie_web_bg.wasm", params.get("log") ?? undefined, { worker: params.get("worker") !== "0" });
}

/** The component and manifest of one of the inferlets beside the page (`text-completion`, `naive-baseline`, …). */
export async function inferletBytes(name) {
  const [wasm, manifest] = await Promise.all([
    fetch(`inferlets/${name.replace(/-/g, "_")}.wasm`).then((r) => r.arrayBuffer()),
    fetch(`inferlets/${name}.Pie.toml`).then((r) => r.text()),
  ]);
  return { bytes: new Uint8Array(wasm), manifest };
}

/** Install one of the inferlets: the program name, with the bytes and manifest for a re-install. */
export async function installInferlet(name) {
  const { bytes, manifest } = await inferletBytes(name);
  return { program: await pie.install(bytes, manifest), bytes, manifest };
}

/** One completion over a fresh session: the last event, its value, and how long it took. */
export async function complete(program, input) {
  const s = new pie.Session();
  const t = performance.now();
  try {
    const last = await s.run(program, JSON.stringify(input));
    return { event: last.event, value: last.value, ms: performance.now() - t };
  } finally {
    s.close();
  }
}
