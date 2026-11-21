"""Deterministic checks for gait geometry, targets, and fitness penalties.

Run from Artificial-Intelligence-cw:
    python -m unittest discover -s test -p fitness_function_test.py
"""

import importlib
import unittest
from unittest.mock import patch

import numpy as np

from genetic_algorithm.chromosome import NUM_FRAMES
from genetic_algorithm.population import MATLAB_STANDING_POSE

fitness = importlib.import_module("genetic_algorithm.fitness_function")


class FootPositionTests(unittest.TestCase):
    def test_zero_angles_have_known_foot_coordinates(self):
        # All three segments point outward at the fixed 30-degree elevation.
        angles = np.deg2rad([45, 75, 105, 135, -135, -105, -75, -45])
        reach = 2.9 * np.cos(np.pi / 6)
        expected = np.column_stack(
            (
                (1.5 + reach) * np.cos(angles),
                (1.0 + reach) * np.sin(angles),
                np.full(8, 1.45),
            )
        )

        np.testing.assert_allclose(fitness.get_foot_positions([0] * 24), expected)

    def test_invalid_pose_shapes_are_rejected(self):
        for pose in (np.zeros(23), np.zeros(25), np.zeros((8, 3))):
            with self.subTest(shape=pose.shape):
                with self.assertRaisesRegex(ValueError, "24 joint angles"):
                    fitness.get_foot_positions(pose)

    def test_gait_preserves_frame_order(self):
        gait = np.tile(MATLAB_STANDING_POSE, (NUM_FRAMES, 1))
        gait[-1, 0] += 0.2
        feet = fitness.get_gait_foot_positions(gait.flatten())

        self.assertEqual(feet.shape, (NUM_FRAMES, 8, 3))
        for frame in (0, NUM_FRAMES - 1):
            np.testing.assert_allclose(
                feet[frame], fitness.get_foot_positions(gait[frame])
            )
        self.assertFalse(np.allclose(feet[0, 0], feet[-1, 0]))

    def test_ground_height_for_level_reference(self):
        self.assertAlmostEqual(fitness.get_ground_z(np.zeros(24)), 1.45)

    def test_ground_height_averages_variations_within_tolerance(self):
        feet = np.zeros((8, 3))
        feet[0, 2] = 5e-7
        with patch.object(fitness, "get_foot_positions", return_value=feet):
            self.assertAlmostEqual(fitness.get_ground_z(None), 5e-7 / 8)

    def test_nonlevel_reference_is_rejected(self):
        pose = MATLAB_STANDING_POSE.copy()
        pose[1] += 0.1
        with self.assertRaisesRegex(ValueError, "horizontal plane"):
            fitness.get_ground_z(pose)

    def test_nonfinite_ground_heights_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            feet = np.zeros((8, 3))
            feet[0, 2] = value
            with self.subTest(value=value):
                with patch.object(fitness, "get_foot_positions", return_value=feet):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        fitness.get_ground_z(None)


class PenaltyTests(unittest.TestCase):
    def test_penetration_only_counts_feet_below_ground(self):
        feet = np.zeros((2, 2, 3))
        feet[:, :, 2] = [[1, 2], [0, -1]]
        # Relative to ground=1, depths are 0, 0, 1, 2.
        self.assertEqual(fitness.ground_penetration_penalty(feet, 1), 1.25)

    def test_no_penetration_has_zero_penalty(self):
        feet = np.zeros((2, 8, 3))
        self.assertEqual(fitness.ground_penetration_penalty(feet, 0), 0)

    def test_constant_gait_is_smooth(self):
        gait = np.tile(MATLAB_STANDING_POSE, (NUM_FRAMES, 1))
        self.assertEqual(fitness.smoothness_penalty(gait), 0)

    def test_smoothness_includes_cycle_boundary(self):
        # Interior second differences vanish; the boundary values are +3 and -3.
        gait = np.tile(np.array([0.0, 1.0, 2.0])[:, None], (1, 24))
        self.assertEqual(fitness.smoothness_penalty(gait), 6)

    def test_support_penalty_ignores_swing_feet(self):
        feet = np.zeros((2, 2, 3))
        feet[:, :, 2] = [[3, 100], [-100, -1]]
        schedule = np.array([[True, False], [False, True]])
        self.assertEqual(fitness.support_height_penalty(feet, 1, schedule), 4)

    def test_horizontal_penalty_uses_only_x_coordinates(self):
        feet = np.full((2, 2, 3), 100.0)
        targets = np.array([[1, 2], [3, 4]])
        feet[:, :, 0] = targets + np.array([[0, 1], [-2, 3]])
        self.assertEqual(fitness.horizontal_movement_penalty(feet, targets), 3.5)
        feet[:, :, 0] = targets
        self.assertEqual(fitness.horizontal_movement_penalty(feet, targets), 0)


class SupportScheduleTests(unittest.TestCase):
    def test_default_shape_and_type(self):
        schedule = fitness.create_support_schedule()
        self.assertEqual(schedule.shape, (300, 8))
        self.assertEqual(schedule.dtype, np.dtype(bool))

    def test_leg_order_and_interval_boundaries(self):
        schedule = fitness.create_support_schedule(100)
        # Explicit intervals in L1, L2, L3, L4, R4, R3, R2, R1 order.
        intervals = [
            ((0, 67),),
            ((0, 20), (51, 100)),
            ((14, 89),),
            ((0, 44), (75, 100)),
            ((28, 93),),
            ((0, 36), (71, 100)),
            ((0, 70), (93, 100)),
            ((0, 12), (39, 100)),
        ]
        for leg, ranges in enumerate(intervals):
            expected = np.zeros(100, dtype=bool)
            for start, end in ranges:
                expected[start:end] = True
            np.testing.assert_array_equal(schedule[:, leg], expected)

    def test_too_few_frames_are_rejected(self):
        for frames in (-1, 0, 1):
            with self.subTest(frames=frames):
                with self.assertRaisesRegex(ValueError, "two frames"):
                    fitness.create_support_schedule(frames)


class MovementTargetTests(unittest.TestCase):
    def test_swing_target_wraps_across_cycle_boundary(self):
        schedule = np.array([[False], [True], [True], [False]])
        feet = np.zeros((4, 1, 3))
        # Two swing frames have sin(pi/4)^2 = sin(3*pi/4)^2 = 1/2.
        feet[:, 0, 2] = [2.1, 100, -100, 2.1]
        self.assertAlmostEqual(
            fitness.swing_clearance_penalty(feet, 2, schedule, clearance=0.2), 0
        )
        feet[[0, 3], 0, 2] = 2
        self.assertAlmostEqual(
            fitness.swing_clearance_penalty(feet, 2, schedule, clearance=0.2), 0.01
        )

    def test_all_support_has_no_swing_penalty(self):
        self.assertEqual(
            fitness.swing_clearance_penalty(
                np.zeros((4, 2, 3)), 1, np.ones((4, 2), dtype=bool)
            ),
            0,
        )

    def test_leg_without_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "support phase"):
            fitness.swing_clearance_penalty(
                np.zeros((4, 1, 3)), 0, np.zeros((4, 1), dtype=bool)
            )

    def test_horizontal_targets_wrap_and_follow_phase_direction(self):
        schedule = np.tile(np.array([True, False, False, True])[:, None], (1, 8))
        reference_x = fitness.get_foot_positions(MATLAB_STANDING_POSE)[:, 0]
        # Support runs frame 3 -> 0 (front to back), swing 1 -> 2 (back to front).
        expected_offsets = np.array([-0.06875, -0.06875, 0.06875, 0.06875])
        targets = fitness.create_horizontal_targets(schedule, step_length=0.2)
        np.testing.assert_allclose(targets, reference_x + expected_offsets[:, None])

    def test_zero_step_length_keeps_reference_x(self):
        schedule = fitness.create_support_schedule()
        reference_x = fitness.get_foot_positions(MATLAB_STANDING_POSE)[:, 0]
        np.testing.assert_allclose(
            fitness.create_horizontal_targets(schedule, step_length=0),
            np.tile(reference_x, (300, 1)),
        )

    def test_horizontal_targets_require_both_phases(self):
        for supporting in (True, False):
            with self.subTest(supporting=supporting):
                with self.assertRaisesRegex(ValueError, "support and swing"):
                    fitness.create_horizontal_targets(
                        np.full((4, 8), supporting, dtype=bool)
                    )


class CombinedFitnessTests(unittest.TestCase):
    def test_penalty_weights_and_score_direction(self):
        chromosome = np.tile(MATLAB_STANDING_POSE, NUM_FRAMES)
        # Each position penalty equals its squared tolerance.
        with (
            patch.object(
                fitness, "get_gait_foot_positions", return_value=np.zeros((300, 8, 3))
            ),
            patch.object(
                fitness, "support_height_penalty", return_value=0.03**2
            ) as support,
            patch.object(
                fitness, "swing_clearance_penalty", return_value=0.02**2
            ) as swing,
            patch.object(
                fitness, "horizontal_movement_penalty", return_value=0.05**2
            ) as horizontal,
            patch.object(fitness, "ground_penetration_penalty", return_value=0.01**2),
            patch.object(fitness, "smoothness_penalty", return_value=10),
            patch.object(
                fitness, "create_horizontal_targets", return_value=np.zeros((300, 8))
            ) as targets,
        ):
            self.assertAlmostEqual(fitness.fitness_function(chromosome, -1), -6)
            self.assertEqual(support.call_args.args[1], -1)
            self.assertEqual(swing.call_args.kwargs["clearance"], 0.15)
            self.assertEqual(targets.call_args.kwargs["step_length"], 0.8)
            support.return_value = 0
            swing.return_value = 0
            horizontal.return_value = 0
            self.assertAlmostEqual(fitness.fitness_function(chromosome, -1), -3)

    def test_standing_gait_has_finite_negative_fitness(self):
        ground_z = fitness.get_ground_z(MATLAB_STANDING_POSE)
        chromosome = np.tile(MATLAB_STANDING_POSE, NUM_FRAMES)
        score = fitness.fitness_function(chromosome, ground_z)
        self.assertIsInstance(score, float)
        self.assertTrue(np.isfinite(score))
        # Standing still misses the required swing clearance and horizontal path.
        self.assertLess(score, 0)


if __name__ == "__main__":
    unittest.main()
