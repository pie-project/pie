// Grammar-constrained JSON generation — JavaScript inferlet example.
//
// Demonstrates:
// - `jsonSchema(...)` to constrain output to a JSON-schema-conforming string.
// - `Sampler.argmax()` for deterministic decoding (recommended for grammars).
// - `Generator.collectJson({ schema })` for one-shot constrained-and-parsed JSON.
// - Returning a structured object directly — the bakery wrapper and
//   `session.send` both auto-stringify, so no manual `JSON.stringify` is needed.
//
// Compare with the equivalent Rust inferlet `constrained-decoding` and
// the Python inferlet `python-constrained-decoding`.

import {
    Context, Sampler,
    session,
} from 'inferlet';

interface Person {
    name: string;
    age: number;
    email: string;
    skills: string[];
}

const PERSON_SCHEMA = JSON.stringify({
    type: 'object',
    properties: {
        name:   { type: 'string', minLength: 1 },
        age:    { type: 'integer', minimum: 0, maximum: 150 },
        email:  { type: 'string' },
        skills: { type: 'array', items: { type: 'string' }, minItems: 1 },
    },
    required: ['name', 'age', 'email', 'skills'],
});

interface Input {
    prompt?: string;
    max_tokens?: number;
}

export async function main(input: Input) {
    const prompt = input.prompt ??
        'Generate a profile for a fictional software engineer named Alice.';
    const maxTokens = input.max_tokens ?? 512;

    using ctx = new Context();
    ctx.system(
        'You are a helpful assistant that generates structured data. ' +
        'Output ONLY a raw JSON object — no markdown, no explanation.',
    ).user(prompt);

    // Grammar guarantees parseable JSON conforming to the schema. The
    // generic + `parse` hook (e.g., Zod) would give compile-time typing;
    // here we declare the type via the generic parameter.
    const person = await ctx
        .generate(Sampler.argmax(), { maxTokens })
        .collectJson<Person>({ schema: PERSON_SCHEMA });

    // Use the typed object — that's the point of collectJson.
    session.send(`Hello ${person.name}, age ${person.age}.\n`);
    session.send(`Skills: ${person.skills.join(', ')}\n`);
    session.send('[done]');

    // Return the object directly — the bakery wrapper serializes via JSON.stringify.
    return person;
}
