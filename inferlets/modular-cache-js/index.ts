// Modular-cache inferlet in TypeScript.
//
// This mirrors the Rust/Python versions:
// - Split prompt into reusable modules.
// - Sort by dependency.
// - Open longest saved prefix snapshot.
// - Append only missing modules.
// - Save each new prefix snapshot.

import { Context, Model, Sampler, runtime } from 'inferlet';

interface ModuleInput {
    id: string;
    role?: 'system' | 'user';
    text: string;
    deps?: string[];
}

interface Input {
    prompt?: string;
    max_tokens?: number;
    use_cache?: boolean;
    save_cache?: boolean;
    modules?: ModuleInput[];
}

function defaultModules(prompt: string): ModuleInput[] {
    return [
        {
            id: 'system/base',
            role: 'system',
            deps: [],
            text: 'You are a concise technical assistant.',
        },
        {
            id: 'style/simple',
            role: 'user',
            deps: ['system/base'],
            text: 'Explain simply first, then give implementation details.',
        },
        {
            id: 'context/pie',
            role: 'user',
            deps: ['style/simple'],
            text: 'Pie inferlets can control forward passes, KV cache snapshots, and generation loops.',
        },
        {
            id: 'task/current',
            role: 'user',
            deps: ['context/pie'],
            text: prompt,
        },
    ];
}

function stableHash(text: string): bigint {
    // FNV-1a 64-bit. Deterministic across runs.
    let h = 1469598103934665603n;
    for (const ch of new TextEncoder().encode(text)) {
        h ^= BigInt(ch);
        h = (h * 1099511628211n) & 0xffffffffffffffffn;
    }
    return h;
}

function prefixKey(modules: ModuleInput[]): string {
    const parts: string[] = ['modular-cache-v1'];
    for (const m of modules) {
        parts.push(m.id);
        parts.push(m.role ?? 'user');
        parts.push(m.text);
        for (const d of m.deps ?? []) parts.push(d);
    }
    return `modular-cache-js/${stableHash(parts.join('|')).toString(16).padStart(16, '0')}`;
}

function topoSort(modules: ModuleInput[]): ModuleInput[] {
    const byId = new Map<string, ModuleInput>();
    for (const m of modules) byId.set(m.id, m);

    const visiting = new Set<string>();
    const visited = new Set<string>();
    const ordered: ModuleInput[] = [];

    function visit(id: string) {
        if (visited.has(id)) return;
        if (visiting.has(id)) throw new Error(`cycle detected at module ${id}`);

        const m = byId.get(id);
        if (!m) throw new Error(`missing module ${id}`);

        visiting.add(id);
        for (const dep of m.deps ?? []) visit(dep);
        visiting.delete(id);

        visited.add(id);
        ordered.push(m);
    }

    for (const id of Array.from(byId.keys()).sort()) visit(id);
    return ordered;
}

function openLongestPrefix(model: Model, modules: ModuleInput[]): [Context | undefined, number] {
    for (let len = modules.length; len >= 1; --len) {
        const ctx = Context.open(model, prefixKey(modules.slice(0, len)));
        if (ctx) return [ctx, len];
    }
    return [undefined, 0];
}

export async function main(input: Input): Promise<string> {
    const model = Model.load(runtime.models()[0]);

    const prompt =
        input.prompt ?? 'Explain modular KV caching for LLM serving in simple terms.';
    const maxTokens = input.max_tokens ?? 256;
    const useCache = input.use_cache ?? true;
    const saveCache = input.save_cache ?? true;

    const modules = topoSort(input.modules ?? defaultModules(prompt));

    let ctx: Context;
    let resumeIndex = 0;

    if (useCache) {
        const [cached, len] = openLongestPrefix(model, modules);
        if (cached) {
            console.log(`cache_hit_modules=${len}`);
            ctx = cached.fork();
            resumeIndex = len;
        } else {
            console.log('cache_miss');
            ctx = new Context(model);
        }
    } else {
        ctx = new Context(model);
    }

    for (let i = resumeIndex; i < modules.length; ++i) {
        const m = modules[i];

        if ((m.role ?? 'user') === 'system') ctx.system(m.text);
        else ctx.user(m.text);

        await ctx.flush();

        if (saveCache) {
            const name = prefixKey(modules.slice(0, i + 1));
            ctx.save(name);
            console.log(`saved=${name}`);
        }
    }

    ctx.cue();

    return await ctx
        .generate(Sampler.argmax(), { maxTokens })
        .collectText();
}
