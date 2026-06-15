// Hierarchical-attention inferlet in TypeScript.
//
// Mirrors the canonical Rust version (`hierarchical-attention-rust`). A long
// prompt is split into chunks; during a manual decode loop, each step builds a
// BRLE attention mask that keeps a *hierarchy* visible:
//
//   global instructions      (sink tokens at the start)
//     chunk summaries        (a header range per chunk)
//       selected chunk(s)    (the full body of the most relevant chunk)
//         recent window      (a sliding window at the end)
//
// How this differs from the existing attention examples:
//   attention-sink         = first sink tokens            + recent window
//   windowed-attention     = recent window only
//   hierarchical-attention = sink + ALL chunk summaries + selected full chunk + recent window
//
// MVP: "summaries" are the first N tokens of each chunk header, relevance is
// lexical overlap, and one shared mask is applied to every query token per pass.
// Demonstrates the programmable attention-mask path; no speedup is claimed.

import { Context, Model, Sampler, chat, runtime } from 'inferlet';

interface Input {
    prompt?: string;
    max_tokens?: number;
    chunk_size_words?: number;
    sink_tokens?: number;
    summary_tokens_per_chunk?: number;
    local_window_tokens?: number;
    selected_chunks?: number;
    selection_mode?: string;
}

interface Range {
    start: number;
    end: number;
}

// =============================================================================
// Pure helpers
// =============================================================================

// Split text into chunks of at most `chunkWords` words. An empty prompt yields
// a single empty chunk so range bookkeeping always has a target.
function splitWords(text: string, chunkWords: number): string[] {
    const words = text.split(/\s+/).filter(Boolean);
    if (words.length === 0) return [''];
    const n = Math.max(1, chunkWords);
    const chunks: string[] = [];
    for (let i = 0; i < words.length; i += n) {
        chunks.push(words.slice(i, i + n).join(' '));
    }
    return chunks;
}

// First `n` words of `text` — the stand-in "summary" for a chunk.
function summarizeWords(text: string, n: number): string {
    return text.split(/\s+/).filter(Boolean).slice(0, n).join(' ');
}

// Pick the `k` chunks with the highest lexical overlap with `query`. Overlap =
// count of query words (>3 chars, lowercased) appearing in the chunk. Ties
// break toward the earlier chunk. Returned indices are in positional order.
function selectRelevantChunks(chunks: string[], query: string, k: number): number[] {
    if (chunks.length === 0) return [];
    const q = new Set(
        query
            .split(/\s+/)
            .map((w) => w.toLowerCase())
            .filter((w) => w.length > 3),
    );
    const scored = chunks.map((chunk, i) => {
        let score = 0;
        for (const w of chunk.split(/\s+/)) {
            if (q.has(w.toLowerCase())) score++;
        }
        return { i, score };
    });
    // Sort by score desc, then index asc.
    scored.sort((a, b) => b.score - a.score || a.i - b.i);
    const take = Math.max(1, Math.min(k, chunks.length));
    return scored
        .slice(0, take)
        .map((s) => s.i)
        .sort((a, b) => a - b);
}

// Clip ranges to [0, total), drop empties, sort, and merge overlapping or
// touching ranges into a minimal disjoint set.
function mergeRanges(ranges: Range[], total: number): Range[] {
    const clipped = ranges
        .map((r) => ({
            start: Math.max(0, Math.min(total, r.start)),
            end: Math.max(0, Math.min(total, r.end)),
        }))
        .filter((r) => r.start < r.end)
        .sort((a, b) => a.start - b.start);

    const merged: Range[] = [];
    for (const r of clipped) {
        const last = merged[merged.length - 1];
        if (last && r.start <= last.end) {
            last.end = Math.max(last.end, r.end);
        } else {
            merged.push({ ...r });
        }
    }
    return merged;
}

// BRLE attention mask over [0, total) keeping `ranges`. BRLE alternates run
// lengths starting with a false run: [false, true, false, ...]. [0, total]
// means "all true". An empty / fully-clipped keep-set returns [0, total]
// (all-true / no restriction), never an all-false mask.
function buildBrleMask(total: number, ranges: Range[]): number[] {
    const merged = mergeRanges(ranges, total);
    if (merged.length === 0) return [0, total];

    const out: number[] = [];
    let cursor = 0;
    for (const r of merged) {
        out.push(r.start - cursor); // false run up to this range
        out.push(r.end - r.start); // true run covering this range
        cursor = r.end;
    }
    if (cursor < total) out.push(total - cursor); // trailing false run
    return out;
}

// Number of attended positions = sum of the true runs (odd indices).
function maskTrueCount(mask: number[]): number {
    let sum = 0;
    for (let i = 1; i < mask.length; i += 2) sum += mask[i];
    return sum;
}

function fmtRanges(ranges: Range[]): string {
    return '[' + ranges.map((r) => `[${r.start},${r.end})`).join(', ') + ']';
}

// =============================================================================
// Entry point
// =============================================================================

export async function main(input: Input): Promise<string> {
    const model = Model.load(runtime.models()[0]);
    const tokenizer = model.tokenizer();

    const prompt =
        input.prompt ??
        'Explain how LLM serving systems use KV cache, batching, scheduling, and ' +
            'attention masks. Include one practical example.';
    const maxTokens = input.max_tokens ?? 128;
    const chunkWords = Math.max(8, input.chunk_size_words ?? 80);
    const sinkTokens = input.sink_tokens ?? 64;
    const summaryTokensPerChunk = input.summary_tokens_per_chunk ?? 24;
    const localWindowTokens = input.local_window_tokens ?? 128;
    const selectedChunks = Math.max(1, input.selected_chunks ?? 1);
    const selectionMode = input.selection_mode ?? 'lexical';

    const chunks = splitWords(prompt, chunkWords);
    const selected = selectRelevantChunks(chunks, prompt, selectedChunks);

    const promptTokens: number[] = [];
    const summaryRanges: Range[] = [];
    const fullRanges: Range[] = [];

    promptTokens.push(
        ...Array.from(
            chat.system(
                model,
                'You are a concise assistant. Use the visible hierarchy: global ' +
                    'instructions, the chunk summaries, and the selected local chunk.',
            ),
        ),
    );

    chunks.forEach((chunk, i) => {
        const header = `Chunk ${i} summary: ${summarizeWords(chunk, 20)}\n`;
        const body = `Chunk ${i} full text:\n${chunk}\n`;

        const headerStart = promptTokens.length;
        promptTokens.push(...Array.from(chat.user(model, header)));
        const headerEnd = promptTokens.length;
        summaryRanges.push({
            start: headerStart,
            end: Math.min(headerEnd, headerStart + summaryTokensPerChunk),
        });

        const bodyStart = promptTokens.length;
        promptTokens.push(...Array.from(chat.user(model, body)));
        const bodyEnd = promptTokens.length;
        fullRanges.push({ start: bodyStart, end: bodyEnd });
    });

    promptTokens.push(
        ...Array.from(
            chat.user(
                model,
                'Answer the original request using the selected local chunk(s) and the ' +
                    'global chunk summaries.',
            ),
        ),
    );
    promptTokens.push(...Array.from(chat.cue(model)));

    console.log('--- hierarchical-attention-js ---');
    console.log(`chunks=${chunks.length}`);
    console.log(`selected_chunk=[${selected.join(', ')}] (mode=${selectionMode})`);
    console.log(`summary_ranges=${fmtRanges(summaryRanges)}`);
    console.log(`full_ranges=${fmtRanges(fullRanges)}`);

    const ctx = new Context(model);
    let pending = promptTokens;
    const generated: number[] = [];
    const stopTokens = new Set<number>(Array.from(chat.stopTokens(model)));
    let loggedMask = false;

    for (let i = 0; i < maxTokens; ++i) {
        if (pending.length === 0) break;

        const fwd = ctx.forward();
        const totalAfter = fwd.startPosition() + pending.length;

        const keep: Range[] = [];
        keep.push({ start: 0, end: Math.min(sinkTokens, totalAfter) }); // 1. sink
        keep.push(...summaryRanges); // 2. summaries
        for (const idx of selected) {
            if (idx < fullRanges.length) keep.push(fullRanges[idx]); // 3. selected bodies
        }
        keep.push({ start: Math.max(0, totalAfter - localWindowTokens), end: totalAfter }); // 4. window

        const mask = buildBrleMask(totalAfter, keep);

        if (!loggedMask) {
            console.log(`mask_true_tokens=${maskTrueCount(mask)} / total=${totalAfter}`);
            loggedMask = true;
        }

        fwd.input(new Uint32Array(pending));
        // One shared mask per query token in this pass (MVP simplification).
        fwd.attentionMask(pending.map(() => new Uint32Array(mask)));

        const h = fwd.sample([pending.length - 1], Sampler.argmax());
        const out = await fwd.execute();
        const token = out.token(h);

        if (token === undefined || stopTokens.has(token)) break;

        generated.push(token);
        pending = [token];
    }

    console.log(`generated_tokens=${generated.length}`);
    return tokenizer.decode(new Uint32Array(generated));
}
