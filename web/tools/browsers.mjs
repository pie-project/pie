// Where the headless runners find Playwright and how they launch Chrome.
//
// Playwright is resolved from, in order: `$PIE_PLAYWRIGHT` (a directory that
// holds `node_modules/playwright`), `web/tools/node_modules` (after
// `npm install --prefix web/tools playwright@1.63.0`), then Node's normal
// resolution from the current directory. The browsers themselves come from
// `npx playwright install chromium firefox`.
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
        "or run `npm install --prefix web/tools playwright@1.63.0 && npx --prefix web/tools playwright install chromium`",
    );
  }
}

export const playwright = loadPlaywright();
export const { chromium, firefox } = playwright;

// Chrome with WebGPU on Vulkan and JSPI on, the way the port needs it.
export const chromeArgs = [
  "--enable-unsafe-webgpu",
  "--enable-features=Vulkan,WebGPU",
  "--use-angle=vulkan",
  "--use-vulkan=native",
  "--enable-dawn-features=allow_unsafe_apis",
  "--ignore-gpu-blocklist",
  "--js-flags=--experimental-wasm-jspi",
];

// `PIE_CHROME_ARGS="--use-webgpu-adapter=swiftshader --enable-unsafe-swiftshader"`
// appends flags: that pair runs WebGPU on SwiftShader, for a machine with no GPU.
const extraArgs = (process.env.PIE_CHROME_ARGS ?? "").split(/\s+/).filter(Boolean);

export function launchChromium(extra = {}) {
  return chromium.launch({ headless: true, args: [...chromeArgs, ...extraArgs], ...extra });
}
