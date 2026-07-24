import unittest

import numpy as np
import torch

from gpu_lr1.kernels import (
    make_lr1_step_workspace,
    triton_lr1_step_fused,
    triton_lr1_step_split,
)
from gpu_lr1.lr1 import (
    ACTION_ACCEPT,
    CanonicalLR1Compiler,
    Grammar,
    LR1ConflictError,
    LR1StepStatus,
    RaggedLR1Stacks,
    pack_lr1_tables,
    select_and_step_lr1_cpu,
    step_lr1_terminals_cpu,
)
from gpu_lr1.lr1_workloads import (
    json_structure_grammar,
    reduction_chain_grammar,
    wide_choice_grammar,
)


def arithmetic_grammar() -> Grammar:
    return Grammar.from_rules(
        "arithmetic",
        "E",
        {
            "E": [("E", "+", "T"), ("T",)],
            "T": [("T", "*", "F"), ("F",)],
            "F": [("(", "E", ")"), ("id",)],
        },
    )


def balanced_grammar() -> Grammar:
    return Grammar.from_rules(
        "balanced",
        "S",
        {
            "S": [("(", "S", ")", "S"), ()],
        },
    )


class CanonicalLR1CompilerTest(unittest.TestCase):
    def test_compiles_left_recursive_expression_grammar(self) -> None:
        compiled = CanonicalLR1Compiler(arithmetic_grammar()).compile()

        self.assertTrue(compiled.accepts(["id"]))
        self.assertTrue(compiled.accepts(["id", "+", "id", "*", "id"]))
        self.assertTrue(compiled.accepts(["(", "id", "+", "id", ")"]))
        self.assertFalse(compiled.accepts(["id", "+"]))
        self.assertFalse(compiled.accepts(["(", "id"]))

    def test_compiles_epsilon_and_recursive_grammar(self) -> None:
        compiled = CanonicalLR1Compiler(balanced_grammar()).compile()

        self.assertTrue(compiled.accepts([]))
        self.assertTrue(compiled.accepts(["(", ")"]))
        self.assertTrue(compiled.accepts(["(", "(", ")", ")", "(", ")"]))
        self.assertFalse(compiled.accepts(["(", ")", ")"]))
        self.assertFalse(compiled.accepts([")", "("]))

    def test_rejects_non_lr1_conflict(self) -> None:
        grammar = Grammar.from_rules(
            "ambiguous",
            "S",
            {
                "S": [("S", "S"), ("a",)],
            },
        )
        with self.assertRaises(LR1ConflictError):
            CanonicalLR1Compiler(grammar).compile()

    def test_keeps_canonical_lr1_lookahead_states_separate(self) -> None:
        grammar = Grammar.from_rules(
            "lr1-not-lalr",
            "S",
            {
                "S": [
                    ("a", "A", "d"),
                    ("b", "B", "d"),
                    ("a", "B", "e"),
                    ("b", "A", "e"),
                ],
                "A": [("c",)],
                "B": [("c",)],
            },
        )
        compiled = CanonicalLR1Compiler(grammar).compile()

        for symbols in (
            ["a", "c", "d"],
            ["b", "c", "d"],
            ["a", "c", "e"],
            ["b", "c", "e"],
        ):
            self.assertTrue(compiled.accepts(symbols))
        self.assertFalse(compiled.accepts(["a", "c", "c"]))

    def test_json_structure_and_long_reduction_workloads(self) -> None:
        json_table = CanonicalLR1Compiler(json_structure_grammar()).compile()
        self.assertTrue(
            json_table.accepts(
                ["{", "STRING", ":", "[", "NUMBER", ",", "true", "]", "}"]
            )
        )
        self.assertFalse(
            json_table.accepts(["{", "STRING", ":", "NUMBER", ",", "}"])
        )

        chain = CanonicalLR1Compiler(reduction_chain_grammar(64)).compile()
        self.assertTrue(chain.accepts(["atom"]))

    def test_accept_action_is_only_on_eof(self) -> None:
        compiled = CanonicalLR1Compiler(arithmetic_grammar()).compile()
        accept_entries = np.flatnonzero(compiled.action_values == ACTION_ACCEPT)
        self.assertEqual(accept_entries.size, 1)
        self.assertEqual(
            int(compiled.action_symbols[accept_entries[0]]),
            compiled.eof_terminal,
        )


class PackedLR1TablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiled = [
            CanonicalLR1Compiler(arithmetic_grammar()).compile(),
            CanonicalLR1Compiler(balanced_grammar()).compile(),
        ]
        cls.tables = pack_lr1_tables(cls.compiled)

    def test_relocates_state_and_production_ids(self) -> None:
        tables = self.tables
        for grammar_id, compiled in enumerate(self.compiled):
            state_start = int(tables.state_offsets[grammar_id])
            state_end = int(tables.state_offsets[grammar_id + 1])
            production_start = int(tables.production_offsets[grammar_id])
            production_end = int(tables.production_offsets[grammar_id + 1])

            for state in range(state_start, state_end):
                start = int(tables.action_indptr[state])
                end = int(tables.action_indptr[state + 1])
                for action in tables.action_values[start:end]:
                    action = int(action)
                    if action > 0:
                        target = action - 1
                        self.assertGreaterEqual(target, state_start)
                        self.assertLess(target, state_end)
                    elif action != int(ACTION_ACCEPT):
                        production = -action - 1
                        self.assertGreaterEqual(production, production_start)
                        self.assertLess(production, production_end)

                start = int(tables.goto_indptr[state])
                end = int(tables.goto_indptr[state + 1])
                targets = tables.goto_targets[start:end]
                if targets.size:
                    self.assertGreaterEqual(int(targets.min()), state_start)
                    self.assertLess(int(targets.max()), state_end)

            self.assertEqual(
                int(tables.start_states[grammar_id]),
                state_start + compiled.start_state,
            )

    def test_ragged_cpu_batch_parses_heterogeneous_grammars(self) -> None:
        tables = self.tables
        stacks = RaggedLR1Stacks.initialize(
            tables.start_states,
            capacities=[16, 32],
        )
        terminal_ids = {
            name: index for index, name in enumerate(tables.terminal_names)
        }

        statuses, _ = step_lr1_terminals_cpu(
            tables,
            stacks,
            [terminal_ids["id"], terminal_ids["("]],
        )
        np.testing.assert_array_equal(
            statuses,
            [LR1StepStatus.SHIFTED, LR1StepStatus.SHIFTED],
        )

        statuses, reductions = step_lr1_terminals_cpu(
            tables,
            stacks,
            [tables.eof_terminals[0], terminal_ids[")"]],
        )
        np.testing.assert_array_equal(
            statuses,
            [LR1StepStatus.ACCEPTED, LR1StepStatus.SHIFTED],
        )
        self.assertGreater(int(reductions[0]), 0)

        statuses, reductions = step_lr1_terminals_cpu(
            tables,
            stacks,
            tables.eof_terminals,
        )
        np.testing.assert_array_equal(
            statuses,
            [LR1StepStatus.ACCEPTED, LR1StepStatus.ACCEPTED],
        )
        self.assertGreater(int(reductions[1]), 0)

    def test_select_and_step_uses_sparse_action_rows(self) -> None:
        tables = self.tables
        stacks = RaggedLR1Stacks.initialize(
            tables.start_states,
            capacities=16,
        )
        terminal_ids = {
            name: index for index, name in enumerate(tables.terminal_names)
        }
        logits = np.full(
            (2, tables.num_terminals),
            -100.0,
            dtype=np.float32,
        )
        logits[0, terminal_ids["id"]] = 10.0
        logits[1, terminal_ids["("]] = 10.0

        result = select_and_step_lr1_cpu(logits, tables, stacks)

        np.testing.assert_array_equal(
            result.terminals,
            [terminal_ids["id"], terminal_ids["("]],
        )
        np.testing.assert_array_equal(
            result.statuses,
            [LR1StepStatus.SHIFTED, LR1StepStatus.SHIFTED],
        )

    def test_reports_bounded_stack_overflow(self) -> None:
        tables = self.tables
        stacks = RaggedLR1Stacks.initialize(
            [tables.start_states[0]],
            capacities=1,
        )
        terminal_id = tables.terminal_names.index("id")

        statuses, reductions = step_lr1_terminals_cpu(
            tables,
            stacks,
            [terminal_id],
        )

        np.testing.assert_array_equal(statuses, [LR1StepStatus.OVERFLOW])
        np.testing.assert_array_equal(reductions, [0])
        np.testing.assert_array_equal(stacks.pointers, [1])

    def test_reports_reduction_limit(self) -> None:
        tables = self.tables
        stacks = RaggedLR1Stacks.initialize(
            [tables.start_states[1]],
            capacities=8,
        )

        statuses, reductions = step_lr1_terminals_cpu(
            tables,
            stacks,
            [tables.eof_terminals[1]],
            max_reductions=0,
        )

        np.testing.assert_array_equal(
            statuses,
            [LR1StepStatus.REDUCTION_LIMIT],
        )
        np.testing.assert_array_equal(reductions, [0])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class LR1KernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiled = [
            CanonicalLR1Compiler(arithmetic_grammar()).compile(),
            CanonicalLR1Compiler(balanced_grammar()).compile(),
        ]
        cls.tables = pack_lr1_tables(cls.compiled)
        cls.tensors = cls.tables.torch_tensors()
        cls.terminal_ids = {
            name: index for index, name in enumerate(cls.tables.terminal_names)
        }

    def test_fused_and_split_match_cpu_across_reduction_chains(self) -> None:
        grammar_ids = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32)
        starts = self.tables.start_states[grammar_ids]
        capacities = np.asarray([16, 24, 12, 32, 8, 20], dtype=np.int32)
        initial = RaggedLR1Stacks.initialize(starts, capacities)
        desired_steps = [
            ["id", "(", "id", "(", "id", "$"],
            ["+", ")", "*", ")", "$", "$"],
            ["id", "$", "id", "$", "$", "$"],
            ["$", "$", "$", "$", "$", "$"],
        ]

        cpu_stacks = initial.clone()
        fused_stacks = initial.torch_tensors()
        split_stacks = initial.torch_tensors()
        fused_workspace = make_lr1_step_workspace(len(grammar_ids))
        split_workspace = make_lr1_step_workspace(len(grammar_ids))

        for desired_names in desired_steps:
            desired = np.asarray(
                [self.terminal_ids[name] for name in desired_names],
                dtype=np.int32,
            )
            logits = np.full(
                (len(grammar_ids), self.tables.num_terminals),
                -20.0,
                dtype=np.float32,
            )
            logits[np.arange(len(grammar_ids)), desired] = 20.0
            cpu_result = select_and_step_lr1_cpu(
                logits,
                self.tables,
                cpu_stacks,
            )
            np.testing.assert_array_equal(cpu_result.terminals, desired)

            device_logits = torch.from_numpy(logits).cuda()
            fused = triton_lr1_step_fused(
                device_logits,
                self.tensors,
                fused_stacks,
                max_action_row_nnz=self.tables.max_action_row_nnz,
                max_goto_row_nnz=self.tables.max_goto_row_nnz,
                workspace=fused_workspace,
            )
            split = triton_lr1_step_split(
                device_logits,
                self.tensors,
                split_stacks,
                max_action_row_nnz=self.tables.max_action_row_nnz,
                max_goto_row_nnz=self.tables.max_goto_row_nnz,
                workspace=split_workspace,
            )
            torch.cuda.synchronize()

            for produced in (fused, split):
                np.testing.assert_array_equal(
                    produced[0].cpu().numpy(),
                    cpu_result.terminals,
                )
                np.testing.assert_array_equal(
                    produced[1].cpu().numpy(),
                    cpu_result.statuses,
                )
                np.testing.assert_array_equal(
                    produced[2].cpu().numpy(),
                    cpu_result.reductions,
                )
            for device_stacks in (fused_stacks, split_stacks):
                np.testing.assert_array_equal(
                    device_stacks.pointers.cpu().numpy(),
                    cpu_stacks.pointers,
                )
                np.testing.assert_array_equal(
                    device_stacks.values.cpu().numpy(),
                    cpu_stacks.values,
                )

    def test_gpu_reports_ragged_stack_overflow(self) -> None:
        initial = RaggedLR1Stacks.initialize(
            [self.tables.start_states[0], self.tables.start_states[1]],
            capacities=[1, 1],
        )
        logits = torch.full(
            (2, self.tables.num_terminals),
            -20.0,
            dtype=torch.float32,
            device="cuda",
        )
        logits[0, self.terminal_ids["id"]] = 20.0
        logits[1, self.terminal_ids["("]] = 20.0

        for step in (triton_lr1_step_fused, triton_lr1_step_split):
            stacks = initial.torch_tensors()
            _, statuses, reductions = step(
                logits,
                self.tensors,
                stacks,
                max_action_row_nnz=self.tables.max_action_row_nnz,
                max_goto_row_nnz=self.tables.max_goto_row_nnz,
            )
            torch.cuda.synchronize()
            np.testing.assert_array_equal(
                statuses.cpu().numpy(),
                [LR1StepStatus.OVERFLOW, LR1StepStatus.OVERFLOW],
            )
            np.testing.assert_array_equal(reductions.cpu().numpy(), [0, 0])
            np.testing.assert_array_equal(stacks.pointers.cpu().numpy(), [1, 1])

    def test_gpu_reduction_limit_matches_cpu(self) -> None:
        initial = RaggedLR1Stacks.initialize(
            [self.tables.start_states[1]],
            capacities=8,
        )
        logits = torch.full(
            (1, self.tables.num_terminals),
            -20.0,
            dtype=torch.float32,
            device="cuda",
        )
        logits[0, int(self.tables.eof_terminals[1])] = 20.0

        for step in (triton_lr1_step_fused, triton_lr1_step_split):
            stacks = initial.torch_tensors()
            _, statuses, reductions = step(
                logits,
                self.tensors,
                stacks,
                max_reductions=0,
            )
            torch.cuda.synchronize()
            np.testing.assert_array_equal(
                statuses.cpu().numpy(),
                [LR1StepStatus.REDUCTION_LIMIT],
            )
            np.testing.assert_array_equal(reductions.cpu().numpy(), [0])

    def test_wide_sparse_row_selection_and_bound_validation(self) -> None:
        compiled = CanonicalLR1Compiler(wide_choice_grammar(64)).compile()
        tables = pack_lr1_tables([compiled])
        tensors = tables.torch_tensors()
        initial = RaggedLR1Stacks.initialize(tables.start_states, capacities=4)
        terminal = tables.terminal_names.index("choice_64_63")
        logits = torch.zeros(
            (1, tables.num_terminals),
            dtype=torch.float32,
            device="cuda",
        )
        logits[0, terminal] = 10.0

        for step in (triton_lr1_step_fused, triton_lr1_step_split):
            stacks = initial.torch_tensors()
            selected, statuses, _ = step(logits, tensors, stacks)
            torch.cuda.synchronize()
            np.testing.assert_array_equal(selected.cpu().numpy(), [terminal])
            np.testing.assert_array_equal(
                statuses.cpu().numpy(),
                [LR1StepStatus.SHIFTED],
            )

            with self.assertRaisesRegex(ValueError, "ACTION"):
                step(
                    logits,
                    tensors,
                    initial.torch_tensors(),
                    max_action_row_nnz=8,
                )
            with self.assertRaisesRegex(ValueError, "terminals"):
                step(
                    logits[:, :-1],
                    tensors,
                    initial.torch_tensors(),
                )

    def test_tied_logits_select_lowest_terminal_id(self) -> None:
        compiled = CanonicalLR1Compiler(wide_choice_grammar(16)).compile()
        tables = pack_lr1_tables([compiled])
        tensors = tables.torch_tensors()
        logits = torch.zeros(
            (1, tables.num_terminals),
            dtype=torch.float32,
            device="cuda",
        )
        expected = int(tables.action_symbols[0])

        for step in (triton_lr1_step_fused, triton_lr1_step_split):
            stacks = RaggedLR1Stacks.initialize(
                tables.start_states,
                capacities=4,
            ).torch_tensors()
            selected, _, _ = step(logits, tensors, stacks)
            torch.cuda.synchronize()
            np.testing.assert_array_equal(selected.cpu().numpy(), [expected])


if __name__ == "__main__":
    unittest.main()
