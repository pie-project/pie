"""MCTS-reasoning inferlet in Python.

Mirrors the canonical Rust version (`mcts-rust`). Monte Carlo Tree Search
budgets the model calls: each iteration walks the most promising path by UCB,
expands it, simulates a candidate answer, scores it, and backs the score up the
tree. Over many iterations visit counts concentrate on the branch that keeps
scoring well.

Four phases per iteration:

1. Selection      — descend from root by UCB to a not-fully-expanded node.
2. Expansion      — ask the model for the next reasoning step; attach a child.
3. Simulation     — from that child, generate a candidate final answer.
4. Eval + backprop— score 0-100, normalize to [0,1], add to every node on the
                    path and bump visit counts.

MVP scope: every phase runs in a fresh Context (clear control flow at the cost
of recomputing shared prefixes). The pure search helpers (`ucb`, `parse_score`,
`Tree`) are model-free and unit-tested in `test_mcts.py`.
"""

import math

from inferlet import Context, Sampler


# =============================================================================
# Pure helpers (unit-tested)
# =============================================================================


def ucb(mean_value: float, child_visits: int, parent_visits: int, c: float) -> float:
    """UCB1 score for one child::

        ucb = mean_value + c * sqrt( ln(parent_visits + 1) / (child_visits + 1) )

    An unvisited child returns +inf so it is always tried before any sibling is
    revisited (the "expand every action at least once" rule).
    """
    if child_visits == 0:
        return math.inf
    exploration = math.sqrt(math.log(parent_visits + 1) / (child_visits + 1))
    return mean_value + c * exploration


def first_number(text: str):
    """Extract the first decimal number from arbitrary text, or ``None``."""
    buf = ""
    for ch in text:
        if ch.isdigit() or (ch == "." and "." not in buf):
            buf += ch
        elif buf:
            break
    if not buf or buf == ".":
        return None
    try:
        return float(buf)
    except ValueError:
        return None


def parse_score(text: str) -> float:
    """Normalize a model score string into ``[0,1]``.

    The evaluation prompt asks for 0-100, but models are inconsistent ("85",
    "Score: 85/100", "0.85"). Grab the first number: ``> 1.0`` is treated as a
    0-100 score (divided by 100); a bare fraction ``<= 1.0`` is taken as already
    normalized. Unparseable output falls back to a neutral ``0.5``.
    """
    v = first_number(text)
    if v is None:
        return 0.5
    normalized = v / 100.0 if v > 1.0 else v
    return max(0.0, min(1.0, normalized))


# =============================================================================
# Search tree (arena of dict nodes)
# =============================================================================


class Tree:
    """MCTS search tree. Holds the node arena and all *pure* search logic
    (selection, UCB, expansion bookkeeping, backpropagation). Nothing here
    touches the model, so it is unit-testable on the host."""

    def __init__(self):
        # Node fields: parent, children, depth, action, visits, value_sum, terminal.
        self.nodes = [
            {
                "parent": None,
                "children": [],
                "depth": 0,
                "action": "",
                "visits": 0,
                "value_sum": 0.0,
                "terminal": False,
            }
        ]

    def root(self) -> int:
        return 0

    def mean_value(self, node: int) -> float:
        n = self.nodes[node]
        return 0.0 if n["visits"] == 0 else n["value_sum"] / n["visits"]

    def add_child(self, parent: int, action: str, max_depth: int) -> int:
        depth = self.nodes[parent]["depth"] + 1
        node_id = len(self.nodes)
        self.nodes.append(
            {
                "parent": parent,
                "children": [],
                "depth": depth,
                "action": action,
                "visits": 0,
                "value_sum": 0.0,
                "terminal": depth >= max_depth,
            }
        )
        self.nodes[parent]["children"].append(node_id)
        return node_id

    def is_fully_expanded(self, node: int, branch_factor: int) -> bool:
        return len(self.nodes[node]["children"]) >= branch_factor

    def best_ucb_child(self, node: int, c: float):
        children = self.nodes[node]["children"]
        if not children:
            return None
        parent_visits = self.nodes[node]["visits"]
        return max(
            children,
            key=lambda ch: ucb(self.mean_value(ch), self.nodes[ch]["visits"], parent_visits, c),
        )

    def select(self, node: int, c: float, branch_factor: int) -> int:
        """Descend by UCB while fully expanded, non-terminal, and has children."""
        while True:
            if self.nodes[node]["terminal"]:
                return node
            if not self.is_fully_expanded(node, branch_factor):
                return node
            child = self.best_ucb_child(node, c)
            if child is None:
                return node
            node = child

    def backpropagate(self, node: int, value: float) -> None:
        """Add ``value`` to every node from ``node`` up to the root."""
        cur = node
        while cur is not None:
            self.nodes[cur]["visits"] += 1
            self.nodes[cur]["value_sum"] += value
            cur = self.nodes[cur]["parent"]

    def most_visited_child(self, node: int):
        children = self.nodes[node]["children"]
        if not children:
            return None
        return max(children, key=lambda ch: self.nodes[ch]["visits"])

    def path_actions(self, node: int):
        out = []
        cur = node
        while cur is not None:
            action = self.nodes[cur]["action"]
            if action:
                out.append(action)
            cur = self.nodes[cur]["parent"]
        out.reverse()
        return out

    def best_path(self):
        node = self.root()
        while True:
            child = self.most_visited_child(node)
            if child is None:
                break
            node = child
        return self.path_actions(node)


# =============================================================================
# Model-backed phases
# =============================================================================


def path_block(path) -> str:
    if not path:
        return "(no steps yet)"
    return "\n".join(f"{i + 1}. {s.strip()}" for i, s in enumerate(path))


async def expand_step(problem, path, rollout_tokens) -> str:
    ctx = Context()
    ctx.system(
        "You extend a chain of reasoning. Given a problem and the steps so far, "
        "propose ONE concise next reasoning step. Output only that step."
    )
    ctx.user(f"Problem:\n{problem}\n\nReasoning so far:\n{path_block(path)}\n\nNext reasoning step:")
    ctx.cue()
    text = await ctx.generate(
        Sampler.top_p(0.8, 0.95), max_tokens=max(16, min(64, rollout_tokens))
    ).collect_text()
    return text.strip()


async def rollout(problem, path, rollout_tokens) -> str:
    ctx = Context()
    ctx.system(
        "You finish a partial chain of reasoning. Continue from the steps given "
        "and produce a short, concrete candidate final answer."
    )
    ctx.user(
        f"Problem:\n{problem}\n\nReasoning so far:\n{path_block(path)}\n\nCandidate final answer:"
    )
    ctx.cue()
    text = await ctx.generate(Sampler.top_p(0.7, 0.95), max_tokens=rollout_tokens).collect_text()
    return text.strip()


async def evaluate(problem, candidate) -> float:
    ctx = Context()
    ctx.system(
        "You are a strict grader. Score the candidate answer from 0 to 100 for "
        "correctness, completeness, and reasoning quality. Return ONLY the number."
    )
    ctx.user(f"Problem:\n{problem}\n\nCandidate answer:\n{candidate}\n\nScore (0-100):")
    ctx.cue()
    text = await ctx.generate(Sampler.argmax(), max_tokens=8).collect_text()
    return parse_score(text)


async def synthesize(problem, best_path, final_tokens) -> str:
    ctx = Context()
    ctx.system(
        "You write the final answer to a problem, guided by a vetted chain of "
        "reasoning. Be clear and correct."
    )
    ctx.user(f"Problem:\n{problem}\n\nBest reasoning path:\n{path_block(best_path)}\n\nFinal answer:")
    ctx.cue()
    text = await ctx.generate(Sampler.argmax(), max_tokens=final_tokens).collect_text()
    return text.strip()


# =============================================================================
# Entry point
# =============================================================================


async def main(input: dict) -> str:
    prompt = input.get(
        "prompt", "A farmer has 17 sheep and all but 9 run away. How many are left?"
    )
    max_iterations = max(1, int(input.get("max_iterations", 16)))
    max_depth = max(1, int(input.get("max_depth", 4)))
    branch_factor = max(1, int(input.get("branch_factor", 3)))
    rollout_tokens = int(input.get("rollout_tokens", 128))
    final_tokens = int(input.get("final_tokens", 256))
    c = float(input.get("exploration_constant", 1.414))
    show_trace = bool(input.get("show_trace", True))

    print("--- mcts-python ---")
    print(f"iterations={max_iterations} max_depth={max_depth} branch_factor={branch_factor} c={c:.3f}")

    tree = Tree()
    best_score = 0.0
    best_candidate = ""

    for it in range(max_iterations):
        # (a) Selection.
        selected = tree.select(tree.root(), c, branch_factor)

        # (b) Expansion.
        can_expand = (
            not tree.nodes[selected]["terminal"]
            and tree.nodes[selected]["depth"] < max_depth
            and not tree.is_fully_expanded(selected, branch_factor)
        )
        if can_expand:
            path = tree.path_actions(selected)
            action = await expand_step(prompt, path, rollout_tokens)
            sim_node = tree.add_child(selected, action, max_depth)
        else:
            sim_node = selected
        expanded_children = len(tree.nodes[selected]["children"])

        # (c) Simulation / rollout.
        sim_path = tree.path_actions(sim_node)
        candidate = await rollout(prompt, sim_path, rollout_tokens)

        # (d) Evaluation.
        value = await evaluate(prompt, candidate)

        # (e) Backpropagation.
        tree.backpropagate(sim_node, value)

        if value >= best_score:
            best_score = value
            best_candidate = candidate

        print(
            f"iteration={it} selected_node={selected} expanded_children={expanded_children} "
            f"rollout_score={value:.3f} best_score={best_score:.3f}"
        )

    best_path = tree.best_path()
    if not best_path:
        final_answer = best_candidate or await rollout(prompt, [], final_tokens)
    else:
        final_answer = await synthesize(prompt, best_path, final_tokens)

    if not show_trace:
        return final_answer

    lines = ["Final answer:", final_answer, "", "Best reasoning path:"]
    if not best_path:
        lines.append("(search produced no expanded steps)")
    else:
        lines.extend(f"{i + 1}. {step.strip()}" for i, step in enumerate(best_path))
    lines += [
        "",
        "MCTS summary:",
        f"iterations={max_iterations}",
        f"nodes={len(tree.nodes)}",
        f"best_score={best_score:.3f}",
    ]
    return "\n".join(lines)
