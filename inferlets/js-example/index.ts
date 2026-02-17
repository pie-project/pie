// Simple text completion — JavaScript inferlet example.
//
// Demonstrates:
// - Loading a model
// - Using Context for chat-style prompt building
// - Streaming generation with EventStream + callback API

import {
    Model, Context, Sampler,
    session, runtime,
} from 'inferlet';

export async function main(args: string[]) {
    // Load model
    const model = Model.load(runtime.models()[0]);

    // Build context
    const ctx = Context.create(model);
    ctx.system('You are a helpful assistant.');
    ctx.user('Hello, world! Tell me a joke.');

    // Stream the response
    let output = '';
    const stream = await ctx.generate({
        sampler: Sampler.topP(0.6, 0.95),
        maxTokens: 256,
        decode: { reasoning: true },
    });

    await stream
        .on('thinking', text => session.send(text))
        .on('text', text => { session.send(text); output += text; })
        .on('done', () => session.send('\n[done]'))
        .run();

    return output;
}
