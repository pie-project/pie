// Concurrency test — run two generate() tasks concurrently with Promise.all.
//
// If async concurrency works, we should see interleaved output from both
// contexts. If not, one context will finish completely before the other starts.

import {
    Model, Context, Sampler,
    session, runtime,
} from 'inferlet';

const model = Model.load(runtime.models()[0]);

// Create two separate contexts with different prompts
const ctx1 = Context.create(model);
ctx1.system('You are helpful.');
ctx1.user('Count from 1 to 5.');

const ctx2 = Context.create(model);
ctx2.system('You are helpful.');
ctx2.user('Name 3 colors.');

const log: string[] = [];

async function generate(ctx: Context, label: string): Promise<void> {
    log.push(`[${label}] START`);
    session.send(`[${label}] START`);

    let stepCount = 0;
    const stream = await ctx.generate({
        sampler: Sampler.topP(0.6, 0.95),
        maxTokens: 20,
        decode: { reasoning: true },
    });

    for await (const event of stream) {
        stepCount++;
        if (event.type === 'text' || event.type === 'thinking') {
            const msg = `[${label}] step=${stepCount} ${event.text}`;
            log.push(msg);
            session.send(msg);
        }
        if (event.type === 'done') break;
    }

    log.push(`[${label}] END`);
    session.send(`[${label}] END`);
}

// Run both concurrently 
session.send('[test] starting Promise.all');
await Promise.all([
    generate(ctx1, 'CTX1'),
    generate(ctx2, 'CTX2'),
]);
session.send('[test] Promise.all complete');

// Analyze results: check if interleaving happened
const labels = log.map(l => l.startsWith('[CTX1]') ? '1' : l.startsWith('[CTX2]') ? '2' : '_');
session.send(`[test] order: ${labels.join('')}`);

// Check for interleaving: if we see a pattern like 1..1..2..2 it's sequential
// If we see 1..2..1..2 it's concurrent
let switches = 0;
let last = '';
for (const l of labels) {
    if (l !== '_' && l !== last) {
        switches++;
        last = l;
    }
}
session.send(`[test] context switches: ${switches}`);
session.send(`[test] verdict: ${switches > 2 ? 'CONCURRENT' : 'SEQUENTIAL'}`);
