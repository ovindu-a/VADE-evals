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


class TestSliceExtra(unittest.TestCase):
    """pixel_values is patches-concatenated, not one row per image. Slicing it
    as if it were rows silently hands a batch the WRONG image's pixels."""

    def _extra(self, counts):
        thw = torch.tensor([[1, c, 1] for c in counts])          # prod == c
        px = torch.cat([torch.full((c, 4), float(i)) for i, c in enumerate(counts)])
        return {"pixel_values": px, "image_grid_thw": thw}

    def test_selects_the_right_patches_for_uneven_images(self):
        extra = self._extra([2, 5, 3])
        out = head_das._slice_extra(extra, 3, torch.tensor([2, 0]))
        self.assertEqual(out["pixel_values"].shape, (3 + 2, 4))
        self.assertEqual(out["pixel_values"][:, 0].tolist(), [2.0] * 3 + [0.0] * 2)
        self.assertEqual(out["image_grid_thw"].tolist(), [[1, 3, 1], [1, 2, 1]])

    def test_identity_selection_round_trips(self):
        extra = self._extra([3, 3, 3])
        out = head_das._slice_extra(extra, 3, torch.tensor([0, 1, 2]))
        torch.testing.assert_close(out["pixel_values"], extra["pixel_values"])

    def test_row_count_mismatch_fails_loudly(self):
        with self.assertRaises(AssertionError):
            head_das._slice_extra(self._extra([2, 2]), 3, torch.tensor([0]))


class TestDonorKey(unittest.TestCase):
    def test_key_is_the_donor_prompt_and_nothing_else(self):
        a = {"source": "FJ", "queried": "language", "template_id": "language_prefill_v1",
             "base": "YE", "rule": "match_source", "target_attribute": "language"}
        b = dict(a, base="AR", rule="match_base", target_attribute="capital")
        self.assertEqual(head_das.donor_key(a), head_das.donor_key(b),
                         "the base side must not affect the donor key")

    def test_queried_and_template_do_change_the_key(self):
        a = {"source": "FJ", "queried": "language", "template_id": "language_prefill_v1"}
        self.assertNotEqual(head_das.donor_key(a),
                            head_das.donor_key(dict(a, queried="capital")))
        self.assertNotEqual(head_das.donor_key(a),
                            head_das.donor_key(dict(a, template_id="language_prefill_v2")))


@unittest.skipUnless(HAVE_TUPLES, "VADE flags tuples not present")
class TestCacheSizing(unittest.TestCase):
    def test_distinct_donor_keys_are_far_fewer_than_rows(self):
        rows = head_das.load_rows(VADE_ROOT, "flags", "language", "train", 6000, 0)
        keys = {head_das.donor_key(r) for r in rows}
        self.assertLess(len(keys), len(rows) / 3, f"{len(keys)} keys for {len(rows)} rows")

    def test_prompt_cache_keys_are_bounded_by_items_plus_templates(self):
        rows = head_das.load_rows(VADE_ROOT, "flags", "language", "train", 6000, 0)
        items = {c for r in rows for c in (r["base"], r["source"])}
        tmpl = {(r["queried"], r["template_id"]) for r in rows}
        self.assertLess(len(items) + len(tmpl), 2 * len(rows) / 50)


class TestPromptCacheLogic(unittest.TestCase):
    """Exercises PromptCache against a stub processor -- no model, no images."""

    class _StubProc:
        class _Tok:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [ord(c) for c in text]}
        tokenizer = _Tok()

    def setUp(self):
        self.calls = []
        self.cache = head_das.PromptCache()

        def fake_build(processor, entity_dir, items, item, tmpl):
            self.calls.append((item, tmpl["question"]))
            self.cache.builds += 1
            return {"input_ids": torch.tensor([[1, 2, 3]]),
                    "pixel_values": torch.full((2, 4), float(ord(item[0]))),
                    "image_grid_thw": torch.tensor([[1, 2, 1]])}
        self.cache._build = fake_build

    def test_repeated_rows_build_once_per_item_and_template(self):
        tmpl = {"question": "q", "prefill": "The language is"}
        for _ in range(50):
            for item in ("AA", "BB", "CC"):
                self.cache.prompt(None, None, None, item, "language", "t1", tmpl)
        self.assertLessEqual(len(self.calls), 4, self.calls)
        self.assertEqual(len(self.cache.image), 3)
        self.assertEqual(len(self.cache.ids), 1)

    def test_each_item_keeps_its_own_pixels(self):
        tmpl = {"question": "q", "prefill": "p"}
        _, pa, _ = self.cache.prompt(None, None, None, "AA", "language", "t1", tmpl)
        _, pb, _ = self.cache.prompt(None, None, None, "BB", "language", "t1", tmpl)
        self.assertNotEqual(pa[0, 0].item(), pb[0, 0].item())

    def test_gold_ids_are_derived_once_per_prefill_label(self):
        tok = self._StubProc.tokenizer
        for _ in range(20):
            self.cache.gold_ids(tok, "The language is", "Spanish")
            self.cache.gold_ids(tok, "The language is", "French")
        self.assertEqual(len(self.cache.gold), 2)

    def test_a_template_whose_ids_differ_between_items_is_caught(self):
        """The fixed-canvas assumption, asserted rather than trusted."""
        tmpl = {"question": "q", "prefill": "p"}
        seq = iter([torch.tensor([[1, 2, 3]]), torch.tensor([[9, 9, 9, 9]])])

        def drifting_build(processor, entity_dir, items, item, tmpl):
            return {"input_ids": next(seq),
                    "pixel_values": torch.zeros(2, 4),
                    "image_grid_thw": torch.tensor([[1, 2, 1]])}
        self.cache._build = drifting_build
        self.cache.prompt(None, None, None, "AA", "language", "t1", tmpl)
        with self.assertRaises(AssertionError):
            self.cache.prompt(None, None, None, "BB", "language", "t1", tmpl)


@unittest.skipUnless(HAVE_VADE, "sibling VADE repo not present")
class TestSubspaceSpec(unittest.TestCase):
    """--subspace_dim: one value, per-block values, or 'full'."""

    BLOCKS = [21, 22, 23]
    WIDTHS = {21: 256, 22: 512, 23: 512}

    def test_a_single_value_applies_to_every_block(self):
        self.assertEqual(head_das.parse_subspace_spec("128", 3), [128, 128, 128])

    def test_per_block_values_keep_their_order(self):
        self.assertEqual(head_das.parse_subspace_spec("128,64,32", 3), [128, 64, 32])

    def test_full_means_each_blocks_own_width(self):
        d = head_das.resolve_subspace_dims("full", self.BLOCKS, self.WIDTHS)
        self.assertEqual(d, self.WIDTHS)

    def test_full_is_allowed_per_block_too(self):
        d = head_das.resolve_subspace_dims("full,64,64", self.BLOCKS, self.WIDTHS)
        self.assertEqual(d, {21: 256, 22: 64, 23: 64})

    def test_values_are_clamped_to_each_blocks_width(self):
        # the K=256 arm: block 21 silently becomes a full swap.
        d = head_das.resolve_subspace_dims("256", self.BLOCKS, self.WIDTHS)
        self.assertEqual(d, {21: 256, 22: 256, 23: 256})

    def test_resolution_pairs_values_with_SORTED_blocks(self):
        d = head_das.resolve_subspace_dims("8,16,32", [23, 21, 22], self.WIDTHS)
        self.assertEqual(d, {21: 8, 22: 16, 23: 32})

    def test_wrong_number_of_values_is_rejected(self):
        with self.assertRaises(AssertionError):
            head_das.parse_subspace_spec("8,16", 3)

    def test_non_numeric_is_rejected(self):
        with self.assertRaises(AssertionError):
            head_das.parse_subspace_spec("8,wide,16", 3)

    def test_zero_and_negative_are_rejected(self):
        for spec in ("0", "-4", "8,0,8"):
            with self.assertRaises(AssertionError):
                head_das.parse_subspace_spec(spec, 3)


class TestSubspaceDOF(unittest.TestCase):
    """k(dim-k), the Grassmannian dimension -- NOT the stored parameter count."""

    def test_dof_is_zero_at_full_width(self):
        self.assertEqual(head_das.subspace_dof(512, 512), 0)

    def test_dof_is_zero_when_k_exceeds_the_width(self):
        self.assertEqual(head_das.subspace_dof(999, 256), 0)

    def test_dof_is_below_the_stored_parameter_count(self):
        self.assertLess(head_das.subspace_dof(128, 512), 128 * 512)

    def test_dof_peaks_at_half_width(self):
        w = 512
        best = max(range(1, w + 1), key=lambda k: head_das.subspace_dof(k, w))
        self.assertEqual(best, w // 2)

    def test_k_and_its_complement_have_equal_dof(self):
        self.assertEqual(head_das.subspace_dof(8, 256), head_das.subspace_dof(248, 256))

    def test_zero_dof_block_really_is_the_full_swap(self):
        """The claim the guard rests on: at k == dim the output IS the source,
        for ANY weights, so training it is a no-op."""
        import torch
        iv = head_das.make_intervention("das_fixed", 32, 32, VADE_ROOT)
        base, src = torch.randn(4, 3, 32), torch.randn(4, 3, 32)
        self.assertTrue(torch.allclose(iv(base, src), src, atol=1e-4))
        iv(base, src).pow(2).sum().backward()
        g = max(p.grad.abs().max().item() for p in iv.parameters())
        self.assertLess(g, 1e-2)                       # gauge only

    def test_a_trainable_block_is_NOT_the_full_swap(self):
        import torch
        iv = head_das.make_intervention("das_fixed", 32, 8, VADE_ROOT)
        base, src = torch.randn(4, 3, 32), torch.randn(4, 3, 32)
        self.assertFalse(torch.allclose(iv(base, src), src, atol=1e-2))
        self.assertGreater(head_das.subspace_dof(8, 32), 0)


class TestBuildInterventions(unittest.TestCase):

    def test_each_block_gets_its_own_width(self):
        colmap = {21: list(range(256)), 22: list(range(512))}
        ivs = head_das.build_interventions("das_fixed", colmap, {21: 8, 22: 16}, VADE_ROOT, "cpu")
        self.assertEqual(ivs[21].proj.weight.shape, (8, 256))
        self.assertEqual(ivs[22].proj.weight.shape, (16, 512))

    def test_stats_report_dof_per_block(self):
        colmap = {21: list(range(256)), 22: list(range(512))}
        ivs = head_das.build_interventions("das_fixed", colmap, {21: 256, 22: 16}, VADE_ROOT, "cpu")
        s = head_das.intervention_stats("das_fixed", ivs)
        self.assertEqual(s[21]["dof"], 0)              # pinned to the full swap
        self.assertEqual(s[22]["dof"], 16 * (512 - 16))


class TestSparsityTerm(unittest.TestCase):
    class _Args:
        def __init__(self, **kw):
            self.l1_coef, self.mask_coef = 1e-3, 1e-3
            self.__dict__.update(kw)

    def test_das_fixed_has_no_term(self):
        ivs = {21: head_das.make_intervention("das_fixed", 32, 4, VADE_ROOT)}
        name, t, coef = head_das.sparsity_term("das_fixed", ivs, self._Args())
        self.assertIsNone(name)
        self.assertIsNone(t)

    def test_das_rotated_counts_soft_dimensions_and_carries_gradient(self):
        ivs = {b: head_das.make_intervention("das_rotated", 32, 0, VADE_ROOT) for b in (21, 22)}
        for iv in ivs.values():
            iv.set_temperature(50.0)
        name, t, coef = head_das.sparsity_term("das_rotated", ivs, self._Args())
        self.assertEqual(name, "maskK")
        self.assertAlmostEqual(float(t), 2 * 32 * 0.9526, places=1)   # sigmoid(150/50)
        self.assertTrue(t.requires_grad, "the penalty must reach the mask")
        t.backward()
        self.assertIsNotNone(ivs[21].masks.grad)

    def test_a_closed_mask_costs_near_zero(self):
        iv = head_das.make_intervention("das_rotated", 32, 0, VADE_ROOT)
        with torch.no_grad():
            iv.masks.fill_(-1e4)
        iv.set_temperature(1.0)
        _, t, _ = head_das.sparsity_term("das_rotated", {21: iv}, self._Args())
        self.assertLess(float(t), 1e-3)

    def test_zero_coefficient_disables_the_term(self):
        ivs = {21: head_das.make_intervention("das_rotated", 32, 0, VADE_ROOT)}
        name, t, _ = head_das.sparsity_term("das_rotated", ivs, self._Args(mask_coef=0.0))
        self.assertIsNone(name, "mask_coef=0 must remove the term, not scale it to zero")

    def test_dbm_uses_the_l1_of_the_raw_mask(self):
        from methods.dbm.intervention import l1_penalty
        ivs = {21: head_das.make_intervention("dbm", 32, 0, VADE_ROOT)}
        name, t, coef = head_das.sparsity_term("dbm", ivs, self._Args())
        self.assertEqual(name, "l1")
        self.assertAlmostEqual(float(t), float(l1_penalty(ivs[21])), places=5)
