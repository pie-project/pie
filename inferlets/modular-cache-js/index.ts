// Modular-cache inferlet in TypeScript.
//
// This mirrors the Rust/Python versions:
// - Split prompt into reusable modules.
// - Sort by dependency (rejecting dup ids, missing deps, cycles).
// - Open the longest saved prefix snapshot.
// - Append only missing modules.
// - Save each new prefix snapshot so any stable prefix can be reused.
//
// Behavior:
//   first run               -> cache_miss
//   identical re-run        -> cache_hit_modules=N (full reuse)
//   only final task changed -> reuse the stable earlier prefix
//   use_cache=false         -> never open a saved snapshot
//   save_cache=false        -> never save new snapshots

import { Context, Sampler } from 'inferlet';

// Bump when the snapshot layout / key meaning changes.
const CACHE_SCHEMA = 'modular-cache-v1';
// Snapshot namespace — keeps JS snapshots distinct from the Rust / Python ports.
const CACHE_NS = 'modular-cache-js';
// Key field separator (ASCII unit separator — won't show up in prompt text).
const SEP = '\x1f';

type Role = 'system' | 'user';

interface ModuleInput {
    id: string;
    role?: Role;
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

// FNV-1a 64-bit. Deterministic across runs. Same algorithm as the Rust/Python ports.
function stableHash(text: string): bigint {
    let h = 0xcbf29ce484222325n;
    for (const ch of new TextEncoder().encode(text)) {
        h ^= BigInt(ch);
        h = (h * 0x100000001b3n) & 0xffffffffffffffffn;
    }
    return h;
}

// Deterministic snapshot name for a module prefix. Folds in the schema version
// plus every module's id, role, text, and deps, so any change anywhere in the
// prefix changes the key (invalidation).
function prefixKey(modules: ModuleInput[]): string {
    const parts: string[] = [CACHE_SCHEMA];
    for (const m of modules) {
        parts.push(m.id);
        parts.push(m.role ?? 'user');
        parts.push(m.text);
        for (const d of m.deps ?? []) parts.push(d);
    }
    return `${CACHE_NS}/${stableHash(parts.join(SEP)).toString(16).padStart(16, '0')}`;
}

// Dependency-order modules so deps come first. Rejects duplicate ids, missing
// deps, and cycles with clean errors.
function topoSort(modules: ModuleInput[]): ModuleInput[] {
    const byId = new Map<string, ModuleInput>();
    for (const m of modules) {
        if (!m.id || !m.id.trim()) throw new Error('module id must not be empty');
        const role = m.role ?? 'user';
        if (role !== 'system' && role !== 'user') {
            throw new Error(`unsupported role '${role}' on module ${m.id}`);
        }
        if (byId.has(m.id)) throw new Error(`duplicate module id: ${m.id}`);
        byId.set(m.id, m);
    }

    const visiting = new Set<string>();
    const visited = new Set<string>();
    const ordered: ModuleInput[] = [];

    function visit(id: string) {
        if (visited.has(id)) return;
        if (visiting.has(id)) throw new Error(`dependency cycle at module ${id}`);

        const m = byId.get(id);
        if (!m) throw new Error(`missing module ${id}`);

        visiting.add(id);
        for (const dep of m.deps ?? []) visit(dep);
        visiting.delete(id);

        visited.add(id);
        ordered.push(m);
    }

    // Sorted start ids => deterministic order independent of input order.
    for (const id of Array.from(byId.keys()).sort()) visit(id);
    return ordered;
}

// open() forks the snapshot (it stays immutable), so the context it returns is
// ours to append to. It *throws* on a missing snapshot rather than returning
// undefined, so a failed open just means a miss at that length.
function openLongestPrefix(modules: ModuleInput[]): [Context | undefined, number] {
    for (let len = modules.length; len >= 1; --len) {
        let ctx: Context | undefined;
        try {
            ctx = Context.open(prefixKey(modules.slice(0, len)));
        } catch {
            ctx = undefined;
        }
        if (ctx) return [ctx, len];
    }
    return [undefined, 0];
}

export async function main(input: Input): Promise<string> {
    const prompt =
        input.prompt ?? 'Explain modular KV caching for LLM serving in simple terms.';
    const maxTokens = input.max_tokens ?? 256;
    const useCache = input.use_cache ?? true;
    const saveCache = input.save_cache ?? true;

    const modules = topoSort(input.modules ?? defaultModules(prompt));

    console.log('--- modular-cache-js ---');
    console.log(`modules=${modules.length}`);
    console.log('order=' + modules.map((m) => m.id).join(' -> '));
    console.log(`use_cache=${useCache} save_cache=${saveCache}`);

    let ctx: Context;
    let resumeIndex = 0;

    if (useCache) {
        const [cached, len] = openLongestPrefix(modules);
        if (cached) {
            console.log(`cache_hit_modules=${len}`);
            ctx = cached;
            resumeIndex = len;
        } else {
            console.log('cache_miss');
            ctx = new Context();
        }
    } else {
        console.log('cache_miss (use_cache=false)');
        ctx = new Context();
    }

    for (let i = resumeIndex; i < modules.length; ++i) {
        const m = modules[i];

        if ((m.role ?? 'user') === 'system') ctx.system(m.text);
        else ctx.user(m.text);

        await ctx.flush();

        if (saveCache) {
            const name = prefixKey(modules.slice(0, i + 1));
            // best-effort: save() throws if an earlier run already saved this
            // exact prefix; a cache miss shouldn't abort generation.
            try {
                ctx.save(name);
                console.log(`saved=${name}`);
            } catch (e) {
                console.log(`save_skipped=${name} (${e})`);
            }
        }
    }

    ctx.cue();

    return await ctx
        .generate(Sampler.argmax(), { maxTokens })
        .collectText();
}
