import unittest

import numpy as np
import torch

from gpu_lr1.ragged_sampler import (
    MAX_SINGLE_TILE,
    RaggedSamplerTables,
    ragged_sample,
    ragged_sample_reference,
)


def build_rows(widths, vocab_size, seed):
    rng = np.random.default_rng(seed)
    indices = []
    indptr = [0]
    for width in widths:
        indices.append(
            np.sort(rng.choice(vocab_size, size=width, replace=False)).astype(
                np.int32
            )
        )
        indptr.append(indptr[-1] + width)
    return (
        np.asarray(indptr, dtype=np.int32),
        np.concatenate(indices).astype(np.int32),
    )


def build_batch(widths, batch, vocab_size, seed, *, top_k_high=64, top_p_low=0.4):
    rng = np.random.default_rng(seed + 1)
    indptr, indices = build_rows(widths, vocab_size, seed)
    return {
        "indptr": indptr,
        "indices": indices,
        "next_state": rng.integers(0, len(widths), size=indices.size).astype(
            np.int32
        ),
        "rows": rng.integers(0, len(widths), size=batch).astype(np.int32),
        "logits": rng.standard_normal((batch, vocab_size)).astype(np.float32),
        "temperature": rng.uniform(0.5, 1.5, batch).astype(np.float32),
        "top_k": rng.integers(1, top_k_high, batch).astype(np.int32),
        "top_p": rng.uniform(top_p_low, 1.0, batch).astype(np.float32),
        "uniform": rng.uniform(0.0, 1.0, batch).astype(np.float32),
    }


def to_cuda(case, *, with_next_state=True):
    tables = RaggedSamplerTables(
        torch.from_numpy(case["indptr"]).cuda(),
        torch.from_numpy(case["indices"]).cuda(),
        torch.from_numpy(case["next_state"]).cuda() if with_next_state else None,
    )
    return tables, {
        "rows": torch.from_numpy(case["rows"]).cuda(),
        "temperature": torch.from_numpy(case["temperature"]).cuda(),
        "top_k": torch.from_numpy(case["top_k"]).cuda(),
        "top_p": torch.from_numpy(case["top_p"]).cuda(),
        "uniform": torch.from_numpy(case["uniform"]).cuda(),
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RaggedSamplerTest(unittest.TestCase):
    def _assert_matches_reference(self, case, *, force_tiled):
        tables, args = to_cuda(case)
        logits = torch.from_numpy(case["logits"]).cuda()
        tokens, states = ragged_sample(
            logits,
            tables,
            args["rows"],
            temperature=args["temperature"],
            top_k=args["top_k"],
            top_p=args["top_p"],
            uniform=args["uniform"],
            force_tiled=force_tiled,
        )
        expected = ragged_sample_reference(
            case["logits"],
            case["indptr"],
            case["indices"],
            case["rows"],
            temperature=case["temperature"],
            top_k=case["top_k"],
            top_p=case["top_p"],
            uniform=case["uniform"],
        )
        np.testing.assert_array_equal(tokens.cpu().numpy(), expected)
        return tokens, states

    def test_single_tile_matches_reference(self) -> None:
        case = build_batch([1, 3, 17, 396, 1024], 64, 4096, seed=0)
        self._assert_matches_reference(case, force_tiled=False)

    def test_tiled_path_matches_reference(self) -> None:
        case = build_batch([1, 3, 17, 396, 1024], 64, 4096, seed=0)
        self._assert_matches_reference(case, force_tiled=True)

    def test_rows_wider_than_the_single_program_limit(self) -> None:
        """Rows beyond 32,768 tokens are the JSON string-body case."""
        case = build_batch([50_000, 147_144], 8, 151_669, seed=3)
        self.assertGreater(max(50_000, 147_144), MAX_SINGLE_TILE)
        self._assert_matches_reference(case, force_tiled=True)

    def test_mixed_narrow_and_wide_batch(self) -> None:
        case = build_batch([12, 396, 50_000, 147_144], 32, 151_669, seed=5)
        self._assert_matches_reference(case, force_tiled=True)

    def test_bucketed_dispatch_matches_uniform_dispatch(self) -> None:
        """A wide sequence must not force narrow ones onto the tiled path."""
        case = build_batch([12, 396, 147_144], 64, 151_669, seed=31)
        tables, args = to_cuda(case)
        logits = torch.from_numpy(case["logits"]).cuda()
        bucketed, _ = ragged_sample(
            logits, tables, args["rows"], bucket=True, **_params(args)
        )
        uniform_dispatch, _ = ragged_sample(
            logits, tables, args["rows"], bucket=False, **_params(args)
        )
        expected = ragged_sample_reference(
            case["logits"],
            case["indptr"],
            case["indices"],
            case["rows"],
            temperature=case["temperature"],
            top_k=case["top_k"],
            top_p=case["top_p"],
            uniform=case["uniform"],
        )
        np.testing.assert_array_equal(bucketed.cpu().numpy(), expected)
        np.testing.assert_array_equal(
            bucketed.cpu().numpy(), uniform_dispatch.cpu().numpy()
        )

    def test_next_state_follows_selected_token(self) -> None:
        case = build_batch([7, 33, 512], 48, 2048, seed=7)
        tokens, states = self._assert_matches_reference(case, force_tiled=False)
        tokens = tokens.cpu().numpy()
        states = states.cpu().numpy()
        for index, row in enumerate(case["rows"]):
            start = int(case["indptr"][row])
            end = int(case["indptr"][row + 1])
            offset = start + int(
                np.flatnonzero(case["indices"][start:end] == tokens[index])[0]
            )
            self.assertEqual(int(states[index]), int(case["next_state"][offset]))

    def test_wide_complement_bitset_matches_reference(self) -> None:
        """The wide bucket reads logits contiguously against a bitset."""
        case = build_batch([396, 147_144], 48, 151_669, seed=37)
        tables, args = to_cuda(case)
        tables.build_wide_bitsets(151_669)
        self.assertIsNotNone(tables.bitset)
        self.assertEqual(tables.bitset.shape[1], (151_669 + 31) // 32)
        tokens, states = ragged_sample(
            torch.from_numpy(case["logits"]).cuda(),
            tables,
            args["rows"],
            **_params(args),
        )
        expected = ragged_sample_reference(
            case["logits"],
            case["indptr"],
            case["indices"],
            case["rows"],
            temperature=case["temperature"],
            top_k=case["top_k"],
            top_p=case["top_p"],
            uniform=case["uniform"],
        )
        np.testing.assert_array_equal(tokens.cpu().numpy(), expected)

        chosen = tokens.cpu().numpy()
        produced = states.cpu().numpy()
        for index, row in enumerate(case["rows"]):
            start = int(case["indptr"][row])
            end = int(case["indptr"][row + 1])
            offset = start + int(
                np.flatnonzero(case["indices"][start:end] == chosen[index])[0]
            )
            self.assertEqual(
                int(produced[index]), int(case["next_state"][offset])
            )

    def test_wide_complement_never_violates_the_constraint(self) -> None:
        case = build_batch([147_144], 32, 151_669, seed=41)
        tables, args = to_cuda(case)
        tables.build_wide_bitsets(151_669)
        tokens, _ = ragged_sample(
            torch.from_numpy(case["logits"]).cuda(),
            tables,
            args["rows"],
            **_params(args),
        )
        allowed = set(case["indices"].tolist())
        for token in tokens.cpu().numpy():
            self.assertIn(int(token), allowed)

    def test_top_k_one_selects_argmax(self) -> None:
        case = build_batch([9, 129, 1024], 32, 4096, seed=11)
        case["top_k"] = np.ones_like(case["top_k"])
        tables, args = to_cuda(case)
        args["top_k"] = torch.from_numpy(case["top_k"]).cuda()
        tokens, _ = ragged_sample(
            torch.from_numpy(case["logits"]).cuda(),
            tables,
            args["rows"],
            temperature=args["temperature"],
            top_k=args["top_k"],
            top_p=args["top_p"],
            uniform=args["uniform"],
        )
        tokens = tokens.cpu().numpy()
        for index, row in enumerate(case["rows"]):
            start = int(case["indptr"][row])
            end = int(case["indptr"][row + 1])
            candidates = case["indices"][start:end]
            best = candidates[int(np.argmax(case["logits"][index, candidates]))]
            self.assertEqual(int(tokens[index]), int(best))

    def test_single_token_row_is_forced(self) -> None:
        case = build_batch([1], 16, 512, seed=13)
        tables, args = to_cuda(case)
        tokens, _ = ragged_sample(
            torch.from_numpy(case["logits"]).cuda(),
            tables,
            args["rows"],
            temperature=args["temperature"],
            top_k=args["top_k"],
            top_p=args["top_p"],
            uniform=args["uniform"],
        )
        forced = int(case["indices"][0])
        self.assertTrue(all(int(t) == forced for t in tokens.cpu().numpy()))

    def test_is_deterministic_for_the_same_uniform(self) -> None:
        case = build_batch([23, 777], 32, 4096, seed=17)
        tables, args = to_cuda(case)
        logits = torch.from_numpy(case["logits"]).cuda()
        first, _ = ragged_sample(logits, tables, args["rows"], **_params(args))
        second, _ = ragged_sample(logits, tables, args["rows"], **_params(args))
        np.testing.assert_array_equal(first.cpu().numpy(), second.cpu().numpy())

    def test_unconstrained_sampling_follows_the_softmax(self) -> None:
        """top_k and top_p disabled: empirical frequencies match softmax."""
        vocab_size = 256
        width = 8
        indptr, indices = build_rows([width], vocab_size, seed=19)
        rng = np.random.default_rng(23)
        logits = rng.standard_normal((1, vocab_size)).astype(np.float32)
        scores = logits[0, indices].astype(np.float64)
        expected = np.exp(scores - scores.max())
        expected /= expected.sum()

        draws = 20_000
        tables = RaggedSamplerTables(
            torch.from_numpy(indptr).cuda(), torch.from_numpy(indices).cuda()
        )
        tokens, _ = ragged_sample(
            torch.from_numpy(logits).cuda().expand(draws, vocab_size).contiguous(),
            tables,
            torch.zeros(draws, dtype=torch.int32, device="cuda"),
            temperature=torch.ones(draws, device="cuda"),
            top_k=torch.full((draws,), width, dtype=torch.int32, device="cuda"),
            top_p=torch.ones(draws, device="cuda"),
            uniform=torch.rand(draws, device="cuda"),
        )
        counts = np.bincount(
            np.searchsorted(indices, tokens.cpu().numpy()), minlength=width
        )
        observed = counts / draws
        np.testing.assert_allclose(observed, expected, atol=0.02)

    def test_rejects_mismatched_inputs(self) -> None:
        case = build_batch([16], 4, 512, seed=29)
        tables, args = to_cuda(case)
        logits = torch.from_numpy(case["logits"]).cuda()
        with self.assertRaises(ValueError):
            ragged_sample(
                logits,
                tables,
                args["rows"][:2],
                temperature=args["temperature"],
                top_k=args["top_k"],
                top_p=args["top_p"],
                uniform=args["uniform"],
            )
        with self.assertRaises(ValueError):
            ragged_sample(
                logits.cpu(),
                tables,
                args["rows"],
                temperature=args["temperature"],
                top_k=args["top_k"],
                top_p=args["top_p"],
                uniform=args["uniform"],
            )


def _params(args):
    return {
        "temperature": args["temperature"],
        "top_k": args["top_k"],
        "top_p": args["top_p"],
        "uniform": args["uniform"],
    }


if __name__ == "__main__":
    unittest.main()
