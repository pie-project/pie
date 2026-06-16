// MCTS-reasoning inferlet in TypeScript.
//
// Mirrors the canonical Rust version (`mcts-rust`). Monte Carlo Tree Search
// budgets the model calls: each iteration walks the most promising path by UCB,
// expands it, simulates a candidate answer, scores it, and backs the score up
// the tree. Over many iterations visit counts concentrate on the branch that
// keeps scoring well.
//
// Four phases per iteration:
//   1. Selection       — descend from root by UCB to a not-fully-expanded node.
//   2. Expansion       — ask the model for the next reasoning step; attach a child.
//   3. Simulation      — from that child, generate a candidate final answer.
//   4. Eval + backprop — score 0-100, normalize to [0,1], add to every node on
//                        the path and bump visit counts.
//
// MVP scope: every phase runs in a fresh Context (clear control flow at the cost
// of recomputing shared prefixes). The pure search helpers (`ucb`, `parseScore`,
// `Tree`) are model-free.

import { Context, Model, Sampler, runtime } from 'inferlet';

interface Input {
    prompt?: string;
    max_iterations?: number;
    max_depth?: number;
    branch_factor?: number;
    rollout_tokens?: number;
    final_tokens?: number;
    exploration_constant?: number;
    show_trace?: boolean;
}

// =============================================================================
// Pure helpers
// =============================================================================

// UCB1 score for one child:
//   ucb = mean + c * sqrt( ln(parentVisits + 1) / (childVisits + 1) )
// An unvisited child returns +Infinity so it is always tried before any sibling
// is revisited.
function ucb(mean: number, childVisits: number, parentVisits: number, c: number): number {
    if (childVisits === 0) return Infinity;
    const exploration = Math.sqrt(Math.log(parentVisits + 1) / (childVisits + 1));
    return mean + c * exploration;
}

// Extract the first decimal number from arbitrary text, or undefined.
function firstNumber(text: string): number | undefined {
    let buf = '';
    for (const ch of text) {
        if ((ch >= '0' && ch <= '9') || (ch === '.' && !buf.includes('.'))) {
            buf += ch;
        } else if (buf) {
            break;
        }
    }
    if (!buf || buf === '.') return undefined;
    const v = Number(buf);
    return Number.isNaN(v) ? undefined : v;
}

// Normalize a model score string to [0,1]. Numbers > 1.0 are treated as a
// 0-100 score (divided by 100); a bare fraction <= 1.0 is taken as already
// normalized. Unparseable output falls back to a neutral 0.5.
function parseScore(text: string): number {
    const v = firstNumber(text);
    if (v === undefined) return 0.5;
    const normalized = v > 1.0 ? v / 100.0 : v;
    return Math.max(0.0, Math.min(1.0, normalized));
}

// =============================================================================
// Search tree (arena of node records)
// =============================================================================

interface Node {
    parent: number | null;
    children: number[];
    depth: number;
    action: string;
    visits: number;
    valueSum: number;
    terminal: boolean;
}

class Tree {
    nodes: Node[];

    constructor() {
        this.nodes = [
            { parent: null, children: [], depth: 0, action: '', visits: 0, valueSum: 0, terminal: false },
        ];
    }

    root(): number {
        return 0;
    }

    meanValue(node: number): number {
        const n = this.nodes[node];
        return n.visits === 0 ? 0 : n.valueSum / n.visits;
    }

    addChild(parent: number, action: string, maxDepth: number): number {
        const depth = this.nodes[parent].depth + 1;
        const id = this.nodes.length;
        this.nodes.push({
            parent,
            children: [],
            depth,
            action,
            visits: 0,
            valueSum: 0,
            terminal: depth >= maxDepth,
        });
        this.nodes[parent].children.push(id);
        return id;
    }

    isFullyExpanded(node: number, branchFactor: number): boolean {
        return this.nodes[node].children.length >= branchFactor;
    }

    bestUcbChild(node: number, c: number): number | undefined {
        const children = this.nodes[node].children;
        if (children.length === 0) return undefined;
        const parentVisits = this.nodes[node].visits;
        let best = children[0];
        let bestScore = -Infinity;
        for (const ch of children) {
            const s = ucb(this.meanValue(ch), this.nodes[ch].visits, parentVisits, c);
            if (s > bestScore) {
                bestScore = s;
                best = ch;
            }
        }
        return best;
    }

    // Selection: descend by UCB while fully expanded, non-terminal, has children.
    select(node: number, c: number, branchFactor: number): number {
        for (;;) {
            if (this.nodes[node].terminal) return node;
            if (!this.isFullyExpanded(node, branchFactor)) return node;
            const child = this.bestUcbChild(node, c);
            if (child === undefined) return node;
            node = child;
        }
    }

    // Backpropagation: add value to every node from `node` up to the root.
    backpropagate(node: number, value: number): void {
        let cur: number | null = node;
        while (cur !== null) {
            this.nodes[cur].visits += 1;
            this.nodes[cur].valueSum += value;
            cur = this.nodes[cur].parent;
        }
    }

    mostVisitedChild(node: number): number | undefined {
        const children = this.nodes[node].children;
        if (children.length === 0) return undefined;
        let best = children[0];
        for (const ch of children) {
            if (this.nodes[ch].visits > this.nodes[best].visits) best = ch;
        }
        return best;
    }

    pathActions(node: number): string[] {
        const out: string[] = [];
        let cur: number | null = node;
        while (cur !== null) {
            if (this.nodes[cur].action) out.push(this.nodes[cur].action);
            cur = this.nodes[cur].parent;
        }
        out.reverse();
        return out;
    }

    bestPath(): string[] {
        let node = this.root();
        for (;;) {
            const child = this.mostVisitedChild(node);
            if (child === undefined) break;
            node = child;
        }
        return this.pathActions(node);
    }
}

// =============================================================================
// Model-backed phases
// =============================================================================

function pathBlock(path: string[]): string {
    if (path.length === 0) return '(no steps yet)';
    return path.map((s, i) => `${i + 1}. ${s.trim()}`).join('\n');
}

async function expandStep(
    model: Model,
    problem: string,
    path: string[],
    rolloutTokens: number,
): Promise<string> {
    const ctx = new Context(model);
    ctx.system(
        'You extend a chain of reasoning. Given a problem and the steps so far, ' +
            'propose ONE concise next reasoning step. Output only that step.',
    );
    ctx.user(`Problem:\n${problem}\n\nReasoning so far:\n${pathBlock(path)}\n\nNext reasoning step:`);
    ctx.cue();
    const text = await ctx
        .generate(Sampler.topP(0.8, 0.95), { maxTokens: Math.max(16, Math.min(64, rolloutTokens)) })
        .collectText();
    return text.trim();
}

async function rollout(
    model: Model,
    problem: string,
    path: string[],
    rolloutTokens: number,
): Promise<string> {
    const ctx = new Context(model);
    ctx.system(
        'You finish a partial chain of reasoning. Continue from the steps given ' +
            'and produce a short, concrete candidate final answer.',
    );
    ctx.user(
        `Problem:\n${problem}\n\nReasoning so far:\n${pathBlock(path)}\n\nCandidate final answer:`,
    );
    ctx.cue();
    const text = await ctx
        .generate(Sampler.topP(0.7, 0.95), { maxTokens: rolloutTokens })
        .collectText();
    return text.trim();
}

async function evaluate(model: Model, problem: string, candidate: string): Promise<number> {
    const ctx = new Context(model);
    ctx.system(
        'You are a strict grader. Score the candidate answer from 0 to 100 for ' +
            'correctness, completeness, and reasoning quality. Return ONLY the number.',
    );
    ctx.user(`Problem:\n${problem}\n\nCandidate answer:\n${candidate}\n\nScore (0-100):`);
    ctx.cue();
    const text = await ctx.generate(Sampler.argmax(), { maxTokens: 8 }).collectText();
    return parseScore(text);
}

async function synthesize(
    model: Model,
    problem: string,
    bestPath: string[],
    finalTokens: number,
): Promise<string> {
    const ctx = new Context(model);
    ctx.system(
        'You write the final answer to a problem, guided by a vetted chain of ' +
            'reasoning. Be clear and correct.',
    );
    ctx.user(`Problem:\n${problem}\n\nBest reasoning path:\n${pathBlock(bestPath)}\n\nFinal answer:`);
    ctx.cue();
    const text = await ctx.generate(Sampler.argmax(), { maxTokens: finalTokens }).collectText();
    return text.trim();
}

// =============================================================================
// Entry point
// =============================================================================

export async function main(input: Input): Promise<string> {
    const model = Model.load(runtime.models()[0]);

    const prompt =
        input.prompt ?? 'A farmer has 17 sheep and all but 9 run away. How many are left?';
    const maxIterations = Math.max(1, input.max_iterations ?? 16);
    const maxDepth = Math.max(1, input.max_depth ?? 4);
    const branchFactor = Math.max(1, input.branch_factor ?? 3);
    const rolloutTokens = input.rollout_tokens ?? 128;
    const finalTokens = input.final_tokens ?? 256;
    const c = input.exploration_constant ?? 1.414;
    const showTrace = input.show_trace ?? true;

    console.log('--- mcts-js ---');
    console.log(
        `iterations=${maxIterations} max_depth=${maxDepth} branch_factor=${branchFactor} c=${c.toFixed(3)}`,
    );

    const tree = new Tree();
    let bestScore = 0;
    let bestCandidate = '';

    for (let it = 0; it < maxIterations; ++it) {
        // (a) Selection.
        const selected = tree.select(tree.root(), c, branchFactor);

        // (b) Expansion.
        const canExpand =
            !tree.nodes[selected].terminal &&
            tree.nodes[selected].depth < maxDepth &&
            !tree.isFullyExpanded(selected, branchFactor);

        let simNode: number;
        if (canExpand) {
            const path = tree.pathActions(selected);
            const action = await expandStep(model, prompt, path, rolloutTokens);
            simNode = tree.addChild(selected, action, maxDepth);
        } else {
            simNode = selected;
        }
        const expandedChildren = tree.nodes[selected].children.length;

        // (c) Simulation / rollout.
        const simPath = tree.pathActions(simNode);
        const candidate = await rollout(model, prompt, simPath, rolloutTokens);

        // (d) Evaluation.
        const value = await evaluate(model, prompt, candidate);

        // (e) Backpropagation.
        tree.backpropagate(simNode, value);

        if (value >= bestScore) {
            bestScore = value;
            bestCandidate = candidate;
        }

        console.log(
            `iteration=${it} selected_node=${selected} expanded_children=${expandedChildren} ` +
                `rollout_score=${value.toFixed(3)} best_score=${bestScore.toFixed(3)}`,
        );
    }

    const bestPath = tree.bestPath();
    let finalAnswer: string;
    if (bestPath.length === 0) {
        finalAnswer = bestCandidate || (await rollout(model, prompt, [], finalTokens));
    } else {
        finalAnswer = await synthesize(model, prompt, bestPath, finalTokens);
    }

    if (!showTrace) {
        return finalAnswer;
    }

    const lines = ['Final answer:', finalAnswer, '', 'Best reasoning path:'];
    if (bestPath.length === 0) {
        lines.push('(search produced no expanded steps)');
    } else {
        bestPath.forEach((step, i) => lines.push(`${i + 1}. ${step.trim()}`));
    }
    lines.push(
        '',
        'MCTS summary:',
        `iterations=${maxIterations}`,
        `nodes=${tree.nodes.length}`,
        `best_score=${bestScore.toFixed(3)}`,
    );
    return lines.join('\n');
}
