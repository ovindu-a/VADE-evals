"""CPU tests for methods/head_das.py.

The GPU-only parts (a real forward, real generation) are not covered here. What
IS covered is every place a silent-but-wrong outcome is possible: a patch that
is applied but carries no gradient trains fine and produces plausible numbers,
which is the same failure class ATTRIBUTE_HEAD_EXPERIMENTS.md's void-trace and
observer-return bugs belong to.
"""
import json
import os
import sys
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods import head_das                                    # noqa: E402
from methods.common.targets import MAX_ANSWER_TOKENS            # noqa: E402

VADE_ROOT = head_das.DEFAULT_VADE_ROOT
HAVE_VADE = os.path.exists(os.path.join(VADE_ROOT, "methods", "das", "intervention.py"))
HAVE_TUPLES = os.path.exists(os.path.join(VADE_ROOT, "data", "flags", "tuples", "language", "train.jsonl"))


@unittest.skipUnless(HAVE_VADE, "sibling VADE repo not present")
class TestInterventions(unittest.TestCase):
    def test_das_fixed_is_identity_when_source_equals_base(self):
        """Nothing to import -> nothing changes, whatever R is."""
        iv = head_das.make_intervention("das_fixed", 64, 8, VADE_ROOT)
        x = torch.randn(3, MAX_ANSWER_TOKENS, 64)
        torch.testing.assert_close(iv(x, x), x, atol=1e-5, rtol=1e-4)

    def test_das_fixed_leaves_the_orthogonal_complement_alone(self):
        """The whole point of a subspace patch: only K of D directions move."""
        iv = head_das.make_intervention("das_fixed", 32, 4, VADE_ROOT)
        base, src = torch.randn(2, 1, 32), torch.randn(2, 1, 32)
        out = iv(base, src)
        R = iv.proj.weight.detach()                 # [K, D], orthonormal rows
        delta = (out - base).squeeze(1)
        # delta must lie ENTIRELY inside the row space of R.
        residual = delta - (delta @ R.T) @ R
        self.assertLess(residual.abs().max().item(), 1e-4)
        self.assertGreater(delta.abs().max().item(), 1e-3, "patch did nothing at all")

    def test_das_fixed_k_is_capped_at_the_block_width(self):
        iv = head_das.make_intervention("das_fixed", 16, 999, VADE_ROOT)
        self.assertEqual(iv.subspace_dim, 16)

    def test_subspace_dim_zero_is_rejected(self):
        with self.assertRaises(AssertionError):
            head_das.make_intervention("das_fixed", 16, 0, VADE_ROOT)

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            head_das.make_intervention("nope", 16, 4, VADE_ROOT)

    def test_das_rotated_anneals_toward_a_sharper_mask(self):
        iv = head_das.make_intervention("das_rotated", 32, 0, VADE_ROOT)
        iv.set_temperature(50.0)
        warm = torch.sigmoid(iv.masks / iv.temperature)
        iv.set_temperature(0.1)
        cold = torch.sigmoid(iv.masks / iv.temperature)
        self.assertLess(abs(warm.mean() - 0.5), abs(cold.mean() - 0.5))

    def test_temperature_schedule_decreases(self):
        t = head_das.temperature_schedule_for("das_rotated", 20, VADE_ROOT)
        self.assertEqual(len(t), 20)
        self.assertGreater(float(t[0]), float(t[-1]))
        self.assertIsNone(head_das.temperature_schedule_for("das_fixed", 20, VADE_ROOT))

    def test_stats_report_the_trained_width(self):
        ivs = {21: head_das.make_intervention("das_fixed", 256, 8, VADE_ROOT)}
        s = head_das.intervention_stats("das_fixed", ivs)
        self.assertEqual(s[21]["subspace_dim"], 8)
        self.assertEqual(s[21]["dim"], 256)


@unittest.skipUnless(HAVE_VADE, "sibling VADE repo not present")
class TestGradientsReachTheIntervention(unittest.TestCase):
    """A patch applied with no gradient path trains silently and means nothing."""

    def _fake_forward(self, iv, cols, hidden=64):
        """Mimics intervened_logits' hook body on a plain tensor."""
        t = torch.randn(2, 5, hidden)
        z = torch.randn(2, MAX_ANSWER_TOKENS, len(cols))
        have = t[:, -MAX_ANSWER_TOKENS:, :].index_select(-1, cols)
        new = iv(have, z)
        patched = t.clone()
        patched[:, -MAX_ANSWER_TOKENS:, cols] = new
        return patched

    def test_gradient_flows_back_to_the_rotation(self):
        iv = head_das.make_intervention("das_fixed", 16, 4, VADE_ROOT)
        cols = torch.arange(16)
        self._fake_forward(iv, cols).sum().backward()
        grads = [p.grad for p in iv.parameters() if p.requires_grad]
        self.assertTrue(grads and all(g is not None for g in grads))
        self.assertGreater(sum(g.abs().sum() for g in grads), 0.0)

    def test_only_the_selected_columns_are_written(self):
        iv = head_das.make_intervention("das_fixed", 8, 4, VADE_ROOT)
        cols = torch.arange(8, 16)
        torch.manual_seed(0)
        t = torch.randn(2, 5, 64)
        z = torch.randn(2, MAX_ANSWER_TOKENS, 8)
        have = t[:, -MAX_ANSWER_TOKENS:, :].index_select(-1, cols)
        patched = t.clone()
        patched[:, -MAX_ANSWER_TOKENS:, cols] = iv(have, z)
        untouched = [c for c in range(64) if c not in set(cols.tolist())]
        torch.testing.assert_close(patched[:, :, untouched], t[:, :, untouched])
        # ... and the columns BEFORE the answer window are untouched too.
        torch.testing.assert_close(patched[:, :-MAX_ANSWER_TOKENS, :], t[:, :-MAX_ANSWER_TOKENS, :])

    def test_all_answer_columns_move_not_just_the_last(self):
        """The last-token-only bug this script exists to avoid."""
        iv = head_das.make_intervention("das_fixed", 16, 8, VADE_ROOT)
        cols = torch.arange(16)
        torch.manual_seed(1)
        t = torch.randn(2, 6, 16)
        z = torch.randn(2, MAX_ANSWER_TOKENS, 16)
        have = t[:, -MAX_ANSWER_TOKENS:, :].index_select(-1, cols)
        patched = t.clone()
        patched[:, -MAX_ANSWER_TOKENS:, cols] = iv(have, z)
        moved = (patched - t).abs().sum(dim=(0, 2))
        for j in range(MAX_ANSWER_TOKENS):
            self.assertGreater(moved[-MAX_ANSWER_TOKENS + j].item(), 1e-4,
                               f"answer column {j} was not patched")


class TestLoss(unittest.TestCase):
    def _batch(self, is_cause, toks, lens):
        return {"is_cause": torch.tensor(is_cause),
                "target_toks": torch.tensor(toks), "target_len": torch.tensor(lens)}

    def test_iso_weight_changes_the_loss_it_is_supposed_to(self):
        torch.manual_seed(0)
        logits = torch.randn(4, MAX_ANSWER_TOKENS, 50)
        batch = self._batch([True, True, False, False],
                            [[1, 2, 3]] * 4, [2, 2, 2, 2])
        a = head_das.weighted_ce(logits, batch, 1.0)
        b = head_das.weighted_ce(logits, batch, 3.0)
        self.assertNotAlmostEqual(float(a), float(b), places=4)

    def test_iso_weight_is_a_no_op_when_every_row_is_cause(self):
        torch.manual_seed(0)
        logits = torch.randn(3, MAX_ANSWER_TOKENS, 50)
        batch = self._batch([True] * 3, [[1, 2, 3]] * 3, [3, 3, 3])
        self.assertAlmostEqual(float(head_das.weighted_ce(logits, batch, 1.0)),
                               float(head_das.weighted_ce(logits, batch, 7.0)), places=5)

    def test_padding_positions_do_not_contribute(self):
        """A row whose gold is 1 token must not be scored on columns 1 and 2."""
        torch.manual_seed(0)
        logits = torch.randn(1, MAX_ANSWER_TOKENS, 50)
        short = self._batch([True], [[7, 9, 9]], [1])
        before = float(head_das.weighted_ce(logits, short, 1.0))
        logits2 = logits.clone()
        logits2[:, 1:, :] = torch.randn(1, MAX_ANSWER_TOKENS - 1, 50) * 10
        self.assertAlmostEqual(before, float(head_das.weighted_ce(logits2, short, 1.0)), places=5)


@unittest.skipUnless(HAVE_TUPLES, "VADE flags tuples not present")
class TestRows(unittest.TestCase):
    def test_train_and_eval_items_are_disjoint(self):
        tr = head_das.load_rows(VADE_ROOT, "flags", "language", "train", 0, 0)
        ev = head_das.load_rows(VADE_ROOT, "flags", "language", "test", 0, 1)
        tri = {c for r in tr for c in (r["base"], r["source"])}
        evi = {c for r in ev for c in (r["base"], r["source"])}
        self.assertEqual(tri & evi, set())

    def test_subsampling_keeps_both_pools(self):
        for n in (8, 50, 500):
            rows = head_das.load_rows(VADE_ROOT, "flags", "language", "train", n, 0)
            rules = {r["rule"] for r in rows}
            self.assertIn("match_source", rules, f"n={n} lost the cause pool")
            self.assertTrue(rules - {"match_source"}, f"n={n} lost the iso pool")

    def test_subsampling_is_deterministic_in_the_seed(self):
        a = head_das.load_rows(VADE_ROOT, "flags", "language", "train", 200, 3)
        b = head_das.load_rows(VADE_ROOT, "flags", "language", "train", 200, 3)
        c = head_das.load_rows(VADE_ROOT, "flags", "language", "train", 200, 4)
        self.assertEqual([r["row_index"] for r in a], [r["row_index"] for r in b])
        self.assertNotEqual([r["row_index"] for r in a], [r["row_index"] for r in c])

    def test_missing_attribute_fails_loudly(self):
        with self.assertRaises(AssertionError):
            head_das.load_rows(VADE_ROOT, "flags", "not_an_attribute", "train", 10, 0)


class TestHeadColumns(unittest.TestCase):
    def test_common10_widths_per_block(self):
        heads = head_das.parse_heads(head_das.COMMON10)
        widths = {b: len(head_das.head_columns(heads, b, 128)) for b in (21, 22, 23)}
        self.assertEqual(widths, {21: 256, 22: 512, 23: 512})
        self.assertEqual(sum(widths.values()), 10 * 128)

    def test_columns_are_the_right_slices(self):
        cols = head_das.head_columns([(21, 1), (21, 5)], 21, 128)
        self.assertEqual(cols.tolist(), list(range(128, 256)) + list(range(640, 768)))


if __name__ == "__main__":
    unittest.main()


class TestAccumGroup(unittest.TestCase):
    """The tail flush. A short final group divided by grad_accum_steps rather
    than by its own size scales that gradient down silently -- it trains, it
    converges, and the last step of every epoch just counts for less."""

    def _sizes(self, n, g):
        return [head_das.accum_group(i, n, g) for i in range(n)]

    def test_exact_multiple_has_no_short_group(self):
        self.assertTrue(all(sz == 4 for _, sz in self._sizes(12, 4)))

    def test_short_tail_reports_its_own_size(self):
        got = self._sizes(10, 4)
        self.assertEqual([sz for _, sz in got], [4] * 4 + [4] * 4 + [2, 2])
        self.assertEqual([k for k, _ in got], [0, 1, 2, 3, 0, 1, 2, 3, 0, 1])

    def test_every_batch_belongs_to_exactly_one_group(self):
        for n, g in [(1, 1), (1, 16), (7, 3), (10, 4), (100, 16), (1000, 16)]:
            flushes = [i for i in range(n) if head_das.accum_group(i, n, g)[0] + 1
                       == head_das.accum_group(i, n, g)[1]]
            counted = sum(head_das.accum_group(i, n, g)[1] for i in flushes)
            self.assertEqual(counted, n, f"n={n} g={g}: groups cover {counted} of {n} batches")

    def test_the_last_batch_always_flushes(self):
        for n, g in [(1, 16), (7, 3), (10, 4), (33, 16)]:
            k, sz = head_das.accum_group(n - 1, n, g)
            self.assertEqual(k + 1, sz, f"n={n} g={g}: final batch does not close its group")

    def test_fewer_batches_than_accum_is_one_group(self):
        got = self._sizes(3, 16)
        self.assertTrue(all(sz == 3 for _, sz in got))
        self.assertEqual([k for k, _ in got], [0, 1, 2])


class TestProgress(unittest.TestCase):
    def test_fmt_hms(self):
        self.assertEqual(head_das.fmt_hms(0), "0:00:00")
        self.assertEqual(head_das.fmt_hms(59), "0:00:59")
        self.assertEqual(head_das.fmt_hms(60), "0:01:00")
        self.assertEqual(head_das.fmt_hms(3599), "0:59:59")
        self.assertEqual(head_das.fmt_hms(3600), "1:00:00")
        self.assertEqual(head_das.fmt_hms(90061), "25:01:01")

    def test_fmt_hms_clamps_negative_and_floats(self):
        self.assertEqual(head_das.fmt_hms(-5), "0:00:00")
        self.assertEqual(head_das.fmt_hms(61.9), "0:01:01")

    def test_eta_extrapolates_linearly_from_the_mean_rate(self):
        import time
        t0 = time.time() - 100.0            # 100s spent on 25 of 100 units
        elapsed, eta, rate = head_das.progress(25, 100, t0)
        self.assertAlmostEqual(rate, 4.0, places=1)
        self.assertEqual(elapsed, "0:01:40")
        self.assertEqual(eta, "0:05:00")    # 75 remaining x 4s

    def test_eta_is_zero_at_completion(self):
        import time
        _, eta, _ = head_das.progress(50, 50, time.time() - 10.0)
        self.assertEqual(eta, "0:00:00")

    def test_progress_does_not_divide_by_zero_before_the_first_unit(self):
        import time
        elapsed, eta, rate = head_das.progress(0, 100, time.time())
        self.assertEqual(elapsed, "0:00:00")
        self.assertGreaterEqual(rate, 0.0)
