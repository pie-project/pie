import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const dir = process.argv[2] ?? new URL("./results", import.meta.url).pathname;
const columns = [];
for (const file of readdirSync(dir).filter((f) => f.endsWith(".log")).sort()) {
  const line = readFileSync(join(dir, file), "utf8").split("\n").find((l) => l.startsWith("RESULT "));
  if (!line) continue;
  const r = JSON.parse(line.slice(7));
  if (!r.bench) continue;
  columns.push({ name: file.replace(/\.log$/, ""), r });
}
if (!columns.length) {
  console.error("no RESULT lines found");
  process.exit(1);
}

const rows = [
  ["model", (r, name) => r.model ?? (name.includes("9b") ? "qwen35-d9b-u4g64" : name.includes("4b") ? "qwen35-d4b-u4g64" : name.includes("2b") ? "qwen35-d2b-u4g64" : "qwen35-d0.8b-u4g64")],
  ["boot (ms)", (r) => r.boot_ms],
  ["memory after boot (MiB)", (r) => r.memory_mib_after_boot],
  ["warm-up run (ms)", (r) => r.bench?.warm_ms],
  ["TTFT, short prompt (ms)", (r) => r.bench?.ttft_short_ms],
  ["decode (ms/token)", (r) => r.bench?.decode_ms_per_token],
  ["decode (tok/s, 1 lane)", (r) => r.bench?.decode_tok_s],
  ["TTFT, 200-word prompt (ms)", (r) => r.bench?.ttft_200w_ms],
  ["TTFT, 600-word prompt (ms)", (r) => r.bench?.ttft_600w_ms],
  ["2 concurrent, aggregate tok/s", (r) => r.bench?.conc2_tok_s],
  ["4 concurrent, aggregate tok/s", (r) => r.bench?.conc4_tok_s],
  ["8 concurrent, aggregate tok/s", (r) => r.bench?.conc8_tok_s],
  ["8 concurrent, latency (ms)", (r) => r.bench?.conc8_lat_ms],
  ["30 × 8 tokens, median (ms)", (r) => r.bench?.repeat30_median_ms],
  ["30 × 8 tokens, max (ms)", (r) => r.bench?.repeat30_max_ms],
  ["device-carried decode (ms/token)", (r) => r.bench?.carried_decode_ms_per_token],
  ["device-carried, 4 concurrent tok/s", (r) => r.bench?.carried_conc4_tok_s],
  ["memory after +200 runs (MiB)", (r) => r.bench?.memory_after_200_runs_mib],
  ["corner cases passed", (r) => (r.corners ? `${r.corners.filter((c) => c.ok).length}/${r.corners.length}` : undefined)],
];

const cell = (v) => (v === undefined || v === null ? "–" : String(v));
console.log(`| | ${columns.map((c) => c.name).join(" | ")} |`);
console.log(`|---|${columns.map(() => "---:").join("|")}|`);
for (const [label, get] of rows) {
  console.log(`| ${label} | ${columns.map((c) => cell(get(c.r, c.name))).join(" | ")} |`);
}
