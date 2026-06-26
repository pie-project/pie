// Simple text completion — JavaScript inferlet example.
//
// Demonstrates:
// - Using Context for chat-style prompt building
// - Manual streaming with `chat.Decoder` + optional `reasoning.Decoder`,
//   composed by hand (no implicit suppression).

import {
    Context, Sampler,
    chat, reasoning,
    session,
} from 'inferlet';

interface Input {
    prompt?: string;
    system?: string;
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
}

export async function main(input: Input) {
    using ctx = new Context();
    ctx.system(input.system ?? 'You are a helpful assistant.');
    ctx.user(input.prompt ?? 'What is the capital of France? Tell me a joke.');

    const sampler = Sampler.topP(
        input.temperature ?? 0.6,
        input.top_p ?? 0.95,
    );
    const gen = ctx.generate(sampler, { maxTokens: input.max_tokens ?? 256 });

    const chatDec = new chat.Decoder();
    const reasoningDec = new reasoning.Decoder();

    let output = '';
    for await (const step of gen) {
        const out = await step.execute();

        // Reasoning chunks (independent decoder; no implicit suppression).
        const rev = reasoningDec.feed(out.tokens);
        if (rev.type === 'delta') session.send(rev.text);

        // Visible chat text.
        const cev = chatDec.feed(out.tokens);
        if (cev.type === 'delta') {
            session.send(cev.text);
            output += cev.text;
        } else if (cev.type === 'done') {
            output = cev.text;
            break;
        }
    }

    // Mirror python-example: emit [done] unconditionally after the stream.
    // The chat decoder's 'done' event doesn't fire if the model produced
    // zero tokens (e.g. immediately hit max-tokens at zero).
    session.send('\n[done]');
    return output;
}
