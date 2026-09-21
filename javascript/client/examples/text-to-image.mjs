/**
 * Connect to a serving pie, draw one picture, save the file it sends back.
 *
 * Run a pie that has a generative model bound, then:
 *
 *   pie --config ~/.pie/config.flux2.toml serve &
 *   node javascript/client/examples/text-to-image.mjs \
 *       --prompt "a red bicycle leaning on a blue wall" --out ./out
 *
 * The whole point of this file is the FILE event. `text-to-image` streams a
 * progress line or two and then sends its output as a file; what arrives is a
 * `ReceivedFile` -- a Buffer, with the name the inferlet suggested attached.
 * That name is the difference between writing `file-0000.bin` and writing
 * `image.latent.f32` (or `image.png`, on a model whose VAE decode reading the
 * guest can drive), so it is what this example is about.
 *
 * `text-to-image` must already be on the server: `pie inferlet install
 * <text_to_image.wasm>`, or `installProgram(wasm, manifest)` as below, or a
 * `pie run text-to-image` from a source checkout, which uploads it.
 */

import fs from 'node:fs';
import path from 'node:path';
import { PieClient } from '../src/index.js';

const INFERLET = 'text-to-image@0.1.0';

function flag(name, fallback) {
    const i = process.argv.indexOf(`--${name}`);
    return i === -1 ? fallback : process.argv[i + 1];
}

const uri = flag('uri', 'ws://127.0.0.1:8231/v1/ws');
const prompt = flag('prompt', 'a red bicycle leaning on a blue wall');
const outDir = flag('out', './out');
const steps = Number(flag('steps', 4));
const size = Number(flag('size', 1024));
const wasm = flag('wasm', null);
const manifest = flag('manifest', null);

const client = new PieClient(uri);
await client.connect();

// Only when pointed at a local build. A published inferlet is already there
// and this is skipped.
if (wasm && manifest) {
    await client.installProgram(wasm, manifest, true);
}

const process_ = await client.launchProcess(INFERLET, {
    prompt, steps, width: size, height: size, out: 'image',
});

fs.mkdirSync(outDir, { recursive: true });
let saved = 0;

for (;;) {
    const { event, value } = await process_.recv();

    if (event === 'file') {
        // `value` is a `ReceivedFile`: still a Buffer, plus `.name`.
        // `fileName()` sanitises what came off the wire -- a name is a
        // suggestion, not a path this program has to trust.
        const file = path.join(outDir, value.fileName('output.bin'));
        fs.writeFileSync(file, value);
        saved += 1;
        console.log(`wrote ${file} (${value.length.toLocaleString()} bytes)`);
    } else if (event === 'return') {
        // The guest's report: the geometry, the sigmas it stepped, and the
        // name it gave the file above.
        if (value.startsWith('{')) {
            const report = JSON.parse(value);
            console.log(
                `${report.width}x${report.height} in ${report.steps} steps, ` +
                `${report.rows} latent rows`,
            );
        }
        await client.close();
        process.exit(saved ? 0 : 1);
    } else if (event === 'error') {
        console.error(`the inferlet failed: ${value}`);
        await client.close();
        process.exit(1);
    } else {
        process.stdout.write(event === 'message' ? `${value}\n` : value);
    }
}
