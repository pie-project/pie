// Minimal static file server for the browser builds. `node serve.mjs <dir> [port]`.
// Sends the headers WebGPU/JSPI pages want and never caches, so a rebuild is
// one reload away.
import { createServer } from "node:http";
import { stat, readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".css": "text/css",
  ".zt": "application/octet-stream",
  ".toml": "text/plain; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

export function serve(roots, port = 0) {
  const list = Array.isArray(roots) ? roots : [roots];
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    let path = normalize(decodeURIComponent(url.pathname));
    if (path.endsWith("/")) path += "index.html";
    for (const root of list) {
      const file = join(root, path);
      if (!file.startsWith(root)) continue;
      let info;
      try {
        info = await stat(file);
      } catch {
        continue;
      }
      if (!info.isFile()) continue;
      const type = TYPES[extname(file)] ?? "application/octet-stream";
      const headers = {
        "content-type": type,
        "cache-control": "no-store",
        "cross-origin-opener-policy": "same-origin",
        "cross-origin-embedder-policy": "require-corp",
        "accept-ranges": "bytes",
      };
      const range = req.headers.range?.match(/^bytes=(\d+)-(\d*)$/);
      if (range) {
        const start = Number(range[1]);
        const end = range[2] ? Number(range[2]) : info.size - 1;
        const { createReadStream } = await import("node:fs");
        res.writeHead(206, {
          ...headers,
          "content-range": `bytes ${start}-${end}/${info.size}`,
          "content-length": end - start + 1,
        });
        createReadStream(file, { start, end }).pipe(res);
        return;
      }
      res.writeHead(200, { ...headers, "content-length": info.size });
      if (req.method === "HEAD") return res.end();
      const { createReadStream } = await import("node:fs");
      createReadStream(file).pipe(res);
      return;
    }
    res.writeHead(404).end("not found");
  });
  return new Promise((resolve) => {
    server.listen(port, "127.0.0.1", () => {
      resolve({ server, port: server.address().port });
    });
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const roots = process.argv.slice(2).filter((a) => !/^\d+$/.test(a));
  const port = Number(process.argv.slice(2).find((a) => /^\d+$/.test(a)) ?? 8765);
  const { port: p } = await serve(roots.length ? roots : [process.cwd()], port);
  console.log(`serving ${roots.join(", ")} at http://127.0.0.1:${p}/`);
}
