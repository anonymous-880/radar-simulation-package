import unittest

import numpy as np

from radar_pipeline.radar_physics import (
    RadarPhysicsConfig,
    apply_radar_physics,
    approaching_radial_velocity,
)


class RadarPhysicsTests(unittest.TestCase):
    def test_static_target_and_stationary_sensor_have_zero_doppler(self):
        velocity = approaching_radial_velocity(
            sensor_velocity_world=np.zeros(3),
            target_velocity_world=np.zeros(3),
            target_position_world=np.array([20.0, 0.0, 0.0]),
            sensor_position_world=np.zeros(3),
        )
        self.assertAlmostEqual(velocity, 0.0)

    def test_target_approaching_sensor_is_positive(self):
        velocity = approaching_radial_velocity(
            sensor_velocity_world=np.zeros(3),
            target_velocity_world=np.array([-5.0, 0.0, 0.0]),
            target_position_world=np.array([20.0, 0.0, 0.0]),
            sensor_position_world=np.zeros(3),
        )
        self.assertAlmostEqual(velocity, 5.0)

    def test_sensor_motion_toward_static_target_is_positive(self):
        velocity = approaching_radial_velocity(
            sensor_velocity_world=np.array([3.0, 0.0, 0.0]),
            target_velocity_world=np.zeros(3),
            target_position_world=np.array([20.0, 0.0, 0.0]),
            sensor_position_world=np.zeros(3),
        )
        self.assertAlmostEqual(velocity, 3.0)

    def test_lateral_motion_has_zero_radial_doppler(self):
        velocity = approaching_radial_velocity(
            sensor_velocity_world=np.zeros(3),
            target_velocity_world=np.array([0.0, 5.0, 0.0]),
            target_position_world=np.array([20.0, 0.0, 0.0]),
            sensor_position_world=np.zeros(3),
        )
        self.assertAlmostEqual(velocity, 0.0)

    def test_disabled_physics_is_exact_copy(self):
        radar = np.array([[10.0, 0.1, -0.02, 3.0]], dtype=np.float32)
        result = apply_radar_physics(radar, RadarPhysicsConfig(enabled=False))
        np.testing.assert_array_equal(result, radar)
        self.assertIsNot(result, radar)

    def test_seeded_effects_are_reproducible(self):
        radar = np.tile(np.array([[10.0, 0.1, -0.02, 3.0]], dtype=np.float32), (20, 1))
        config = RadarPhysicsConfig(
            enabled=True,
            range_noise_std_m=0.2,
            azimuth_noise_std_rad=0.01,
            doppler_noise_std_mps=0.5,
            seed=17,
        )
        first = apply_radar_physics(radar, config)
        second = apply_radar_physics(radar, config)
        np.testing.assert_array_equal(first, second)

    def test_range_detection_probability_can_drop_all_far_returns(self):
        radar = np.array([[50.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        config = RadarPhysicsConfig(
            enabled=True,
            detection_probability_at_zero_m=1.0,
            detection_probability_at_max_range=0.0,
            detection_probability_range_m=50.0,
            seed=3,
        )
        result = apply_radar_physics(radar, config)
        self.assertEqual(result.shape, (0, 4))

    def test_multipath_and_clutter_add_returns(self):
        radar = np.array([[10.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        config = RadarPhysicsConfig(
            enabled=True,
            multipath_probability=1.0,
            multipath_extra_range_min_m=5.0,
            multipath_extra_range_max_m=5.0,
            clutter_per_detection=1.0,
            seed=11,
        )
        result = apply_radar_physics(radar, config)
        self.assertGreaterEqual(result.shape[0], 3)
        self.assertTrue(np.any(np.isclose(result[:, 0], 15.0)))


if __name__ == "__main__":
    unittest.main()
