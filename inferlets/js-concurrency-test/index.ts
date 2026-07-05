// Concurrency test — run two generate() tasks concurrently with Promise.all.
//
// If async concurrency works, we should see interleaved output from both
// contexts. If not, one context will finish completely before the other starts.

import {
    Context, Sampler,
    chat,
    session,
} from 'inferlet';

export async function main(_input: Record<string, unknown>) {
    // Create two separate contexts with different prompts.
    const ctx1 = new Context();
    ctx1.system('You are helpful.');
    ctx1.user('Count from 1 to 5.');

    const ctx2 = new Context();
    ctx2.system('You are helpful.');
    ctx2.user('Name 3 colors.');

    const log: string[] = [];

    async function generate(ctx: Context, label: string): Promise<void> {
        log.push(`[${label}] START`);
        session.send(`[${label}] START`);

        const gen = ctx.generate(Sampler.topP(0.6, 0.95), { maxTokens: 20 });
        const dec = new chat.Decoder();

        let stepCount = 0;
        for await (const step of gen) {
            stepCount++;
            const out = await step.execute();
            const ev = dec.feed(out.tokens);
            if (ev.type === 'delta') {
                const msg = `[${label}] step=${stepCount} ${ev.text}`;
                log.push(msg);
                session.send(msg);
            } else if (ev.type === 'done') {
                break;
            }
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

    return '';
}
