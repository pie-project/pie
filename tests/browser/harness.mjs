import * as pie from "../../javascript/browser/dist/pie.mjs";

export const DEFAULT_MODEL = "models/qwen--qwen3-5-0-8b.qwen35-d0-8b-u4g64-kv-bf16.wgpu.zt";
export const SHORT = "The capital of France is";

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

export function loadRuntime(params) {
  return pie.load({ log: params.get("log") ?? undefined, worker: params.get("worker") !== "0" });
}

/** The `[runtime] language` a manifest declares, if it is a script. */
function manifestLanguage(manifest) {
  const m = /^\s*language\s*=\s*"([a-z]+)"/m.exec(manifest);
  return m ? m[1] : null;
}

const EXTENSION = { python: "py", javascript: "js" };

export async function inferletBytes(name) {
  const manifest = await fetch(`inferlets/${name}.Pie.toml`).then((r) => r.text());
  const language = manifestLanguage(manifest);
  const file = `inferlets/${name.replace(/-/g, "_")}.${language ? EXTENSION[language] : "wasm"}`;
  const bytes = new Uint8Array(await fetch(file).then((r) => r.arrayBuffer()));
  return { bytes, manifest, language };
}

const languages = new Map();

/** Install a language component once per page. */
export async function ensureLanguage(language) {
  if (!languages.has(language)) {
    languages.set(language, (async () => {
      const r = await fetch(`languages/${language}.wasm`);
      if (!r.ok) throw new Error(`no ${language} language component beside the page (languages/${language}.wasm); tools/inferlets.sh copies it from ~/.pie/languages`);
      await pie.installLanguage(language, new Uint8Array(await r.arrayBuffer()));
    })());
  }
  return languages.get(language);
}

export async function installInferlet(name) {
  const { bytes, manifest, language } = await inferletBytes(name);
  if (language) await ensureLanguage(language);
  return { program: await pie.install(bytes, manifest), bytes, manifest };
}

export const connect = pie.connect;

export async function run(client, program, input, onEvent) {
  const proc = await client.launchProcess(program, input);
  for (;;) {
    const m = await proc.recv();
    onEvent?.(m);
    if (m.event === "return" || m.event === "error") return m;
  }
}

export async function complete(program, input) {
  const client = await connect();
  const t = performance.now();
  try {
    const last = await run(client, program, input);
    return { event: last.event, value: last.value, ms: performance.now() - t };
  } finally {
    await client.close();
  }
}
