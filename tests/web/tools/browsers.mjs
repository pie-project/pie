import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

function loadPlaywright() {
  const candidates = [process.env.PIE_PLAYWRIGHT, here, process.cwd()].filter(Boolean);
  for (const dir of candidates) {
    const pkg = join(dir, "node_modules", "playwright");
    if (existsSync(pkg)) return createRequire(join(dir, "package.json")).call(null, "playwright");
  }
  try {
    return createRequire(join(process.cwd(), "package.json")).call(null, "playwright");
  } catch {
    throw new Error(
      "playwright not found: set PIE_PLAYWRIGHT to a directory holding node_modules/playwright, " +
        "or run `npm install --prefix tests/web/tools playwright@1.63.0 && npx --prefix tests/web/tools playwright install chromium`",
    );
  }
}

export const playwright = loadPlaywright();
export const { chromium } = playwright;

export const chromeArgs = [
  "--enable-unsafe-webgpu",
  "--enable-features=Vulkan,WebGPU",
  "--use-angle=vulkan",
  "--use-vulkan=native",
  "--enable-dawn-features=allow_unsafe_apis",
  "--ignore-gpu-blocklist",
  "--js-flags=--experimental-wasm-jspi",
];

const extraArgs = (process.env.PIE_CHROME_ARGS ?? "").split(/\s+/).filter(Boolean);

export function launchChromium(extra = {}) {
  return chromium.launch({ headless: true, args: [...chromeArgs, ...extraArgs], ...extra });
}
