import math
import unittest

import torch

from arc_rl.rl_qwen35_native_vllm import (
    build_coordinate_structure_advantages,
    coordinate_ranking_score,
    coordinate_structure_detail,
    group_normalized_sequence_advantages,
    pair_detail,
)
from arc_rl.rl_static_cache import (
    ArcCharacterTokenizer,
    RewardBatch,
    compute_clipped_policy_loss,
)


class PairDetailTests(unittest.TestCase):
    def test_counts_omissions_errors_and_duplicates(self):
        detail = pair_detail(
            "0-0:1 0-1:2 E",
            "0-0:1 0-0:1 2-2:3 E",
        )
        self.assertEqual(detail["exact"], 0)
        self.assertEqual(detail["good_pairs"], 2)
        self.assertEqual(detail["candidate_pairs"], 3)
        self.assertEqual(detail["covered_target_pairs"], 1)
        self.assertEqual(detail["omitted_unique_pairs"], 1)
        self.assertEqual(detail["incorrect_unique_pairs"], 1)
        self.assertEqual(detail["duplicate_pairs"], 1)
        self.assertAlmostEqual(detail["pair_precision"], 2 / 3)
        self.assertAlmostEqual(detail["pair_recall"], 1 / 2)


class CoordinateStructureTests(unittest.TestCase):
    TARGET = "0-0:1 0-1:2 1-0:3 1-1:4 E"

    def test_wrong_values_still_complete_structural_coverage(self):
        detail = coordinate_structure_detail(
            self.TARGET,
            "0-0:9 0-1:9 1-0:9 1-1:9",
            ended_with_eos=True,
        )
        self.assertEqual(detail.shown_coordinates, 4)
        self.assertEqual(detail.missing_coordinates, 0)
        self.assertTrue(detail.structurally_complete)

    def test_repeated_and_invalid_coordinates_do_not_fill_missing_slots(self):
        detail = coordinate_structure_detail(
            self.TARGET,
            "0-0:9 0-0:8 9-9:1",
            ended_with_eos=True,
        )
        self.assertEqual(detail.shown_coordinates, 1)
        self.assertEqual(detail.missing_coordinates, 3)
        self.assertEqual(detail.duplicate_coordinates, 1)
        self.assertEqual(detail.invalid_coordinates, 1)
        self.assertFalse(detail.structurally_complete)

    def test_half_coverage_gets_hard_negative_terminal_score(self):
        tokenizer = ArcCharacterTokenizer()
        predictions = [
            "0-0:9 0-1:9 1-0:9 1-1:9",
            "0-0:9 0-1:9",
        ]
        token_ids = [
            torch.tensor(tokenizer.encode(prediction + "E"))
            for prediction in predictions
        ]
        result = build_coordinate_structure_advantages(
            [self.TARGET, self.TARGET],
            predictions,
            token_ids,
            [True, True],
            group_size=2,
            width=max(ids.numel() for ids in token_ids),
            missing_penalty=8.0,
            duplicate_penalty=4.0,
            invalid_penalty=4.0,
            malformed_penalty=2.0,
            no_eos_penalty=2.0,
            device=torch.device("cpu"),
        )
        self.assertEqual(float(result.terminal_scores[0]), 2.0)
        self.assertEqual(float(result.terminal_scores[1]), -6.0)
        # The same normalized terminal return is distributed across the whole
        # rollout; it is never concentrated as a length-scaled EOS spike.
        self.assertTrue(
            torch.allclose(
                result.terminal[0, : token_ids[0].numel()],
                torch.ones(token_ids[0].numel()),
            )
        )
        self.assertTrue(
            torch.allclose(
                result.terminal[1, : token_ids[1].numel()],
                -torch.ones(token_ids[1].numel()),
            )
        )
        self.assertAlmostEqual(
            float(result.terminal[0].sum()), float(token_ids[0].numel())
        )
        self.assertAlmostEqual(
            float(result.terminal[1].sum()), -float(token_ids[1].numel())
        )

    def test_repeated_coordinate_gets_immediate_negative_advantage(self):
        tokenizer = ArcCharacterTokenizer()
        target = "0-0:1 0-1:2 E"
        predictions = ["0-0:9 0-1:9", "0-0:9 0-0:8"]
        token_ids = [
            torch.tensor(tokenizer.encode(prediction + "E"))
            for prediction in predictions
        ]
        result = build_coordinate_structure_advantages(
            [target, target],
            predictions,
            token_ids,
            [True, True],
            group_size=2,
            width=max(ids.numel() for ids in token_ids),
            missing_penalty=8.0,
            duplicate_penalty=4.0,
            invalid_penalty=4.0,
            malformed_penalty=2.0,
            no_eos_penalty=2.0,
            device=torch.device("cpu"),
        )
        active = result.coordinate[result.coordinate_mask]
        self.assertLess(float(active.min()), 0.0)
        self.assertGreater(float(active.max()), 0.0)

    def test_composite_ranking_favors_coverage_and_penalizes_bad_coordinates(self):
        clean = coordinate_structure_detail(
            self.TARGET,
            "0-0:9 0-1:9 1-0:9 1-1:9",
            ended_with_eos=True,
        )
        repeated = coordinate_structure_detail(
            self.TARGET,
            "0-0:1 0-0:1 9-9:1",
            ended_with_eos=True,
        )
        clean_score = coordinate_ranking_score(
            0.4,
            clean,
            pair_f1_weight=0.35,
            coverage_weight=0.65,
            duplicate_penalty=0.5,
            invalid_penalty=0.5,
        )
        repeated_score = coordinate_ranking_score(
            0.8,
            repeated,
            pair_f1_weight=0.35,
            coverage_weight=0.65,
            duplicate_penalty=0.5,
            invalid_penalty=0.5,
        )
        self.assertGreater(clean_score, repeated_score)
        self.assertAlmostEqual(clean_score, 0.79)


class GroupAdvantageTests(unittest.TestCase):
    def test_normalizes_each_sibling_group_and_masks_padding(self):
        scores = torch.tensor([0.0, 2.0, 10.0, 14.0])
        result = group_normalized_sequence_advantages(
            scores,
            [2, 3, 1, 2],
            group_size=2,
            width=3,
        )
        self.assertTrue(torch.allclose(result[0, :2], torch.tensor([-1.0, -1.0])))
        self.assertTrue(torch.allclose(result[1, :3], torch.tensor([1.0, 1.0, 1.0])))
        self.assertTrue(torch.allclose(result[2, :1], torch.tensor([-1.0])))
        self.assertTrue(torch.allclose(result[3, :2], torch.tensor([1.0, 1.0])))
        self.assertEqual(float(result[0, 2]), 0.0)
        self.assertEqual(float(result[2, 1]), 0.0)

    def test_extra_advantage_changes_gradient(self):
        current = torch.zeros((1, 2), requires_grad=True)
        old = torch.zeros((1, 2))
        mask = torch.ones((1, 2), dtype=torch.bool)
        zeros = torch.zeros((1, 2))
        rewards = RewardBatch(
            token_rewards=zeros,
            sequence_rewards=zeros,
            token_baselines=zeros,
            token_stds=zeros,
            sequence_baselines=zeros,
            sequence_std=zeros,
            mask=mask,
            sequence_mask=mask,
        )
        extra = torch.tensor([[1.0, -1.0]])
        loss, metrics = compute_clipped_policy_loss(
            current,
            old,
            rewards,
            pair_correctness_weight=0.0,
            extra_advantages=extra,
        )
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), 0.0)
        self.assertTrue(torch.allclose(current.grad, torch.tensor([[-0.5, 0.5]])))
        self.assertTrue(math.isclose(float(metrics["extra_advantage_abs_mean"]), 1.0))

    def test_sequence_reduction_weights_rollouts_equally(self):
        current = torch.zeros((2, 2), requires_grad=True)
        old = torch.zeros((2, 2))
        mask = torch.tensor([[True, False], [True, True]])
        zeros = torch.zeros((2, 2))
        rewards = RewardBatch(
            token_rewards=zeros,
            sequence_rewards=zeros,
            token_baselines=zeros,
            token_stds=zeros,
            sequence_baselines=zeros,
            sequence_std=zeros,
            mask=mask,
            sequence_mask=mask,
        )
        extra = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
        token_loss, _ = compute_clipped_policy_loss(
            current,
            old,
            rewards,
            pair_correctness_weight=0.0,
            extra_advantages=extra,
            loss_reduction="token",
        )
        sequence_loss, _ = compute_clipped_policy_loss(
            current,
            old,
            rewards,
            pair_correctness_weight=0.0,
            extra_advantages=extra,
            loss_reduction="sequence",
        )
        self.assertAlmostEqual(float(token_loss.detach()), -2 / 3)
        self.assertAlmostEqual(float(sequence_loss.detach()), -1.0)


if __name__ == "__main__":
    unittest.main()
