import types

import numpy as np
import pytest

from carla_radar4d import (
    CubeGrid,
    RadarFrameProcessor,
    RadarPhysicsConfig,
    RadarSensorConfig,
    crop_to_roi,
    points_to_cube,
    points_to_ra_mask,
)
from carla_radar4d.geometry import (
    cartesian_to_spherical,
    ego_motion_doppler,
    spherical_to_cartesian,
)
from carla_radar4d.sensor import measurement_to_array

GRID = CubeGrid.radelft()


def legacy_params():
    """Axes exactly as written in radar_pipeline/data_preparation.get_default_params."""
    range_axis = np.arange(0.1004, 51.4242 + 0.1004, 0.1004)[10:-3]
    wx = np.linspace(-np.pi, np.pi, 256)[8:248]
    wz = np.linspace(-np.pi, np.pi, 128)[47:81]
    return {
        "range_axis": range_axis,
        "azimuth_axis": np.arcsin(wx / (2 * np.pi * 0.4972)),
        "elevation_axis": np.arcsin(wz / (2 * np.pi * 0.4972)),
        "vel_fft_size": 128,
        "vel_bin_size": 0.04607058455831936,
        "roi_spherical": np.asarray([0, -70, -15, 50, 70, 15]),
    }


def legacy_points_to_cube(radar_spher, params):
    """Verbatim copy of the pre-refactor loop implementation."""
    range_axis = params["range_axis"]
    azimuth_axis = params["azimuth_axis"]
    D = params["vel_fft_size"]
    R, A = len(range_axis), len(azimuth_axis)
    if radar_spher.size == 0:
        return np.zeros((2, D, R, A), dtype=np.float32)
    r, az, el, v = radar_spher.T
    r_idx = np.clip(np.searchsorted(range_axis, r, side="left"), 0, R - 1)
    a_idx = np.clip(np.searchsorted(azimuth_axis, az, side="left"), 0, A - 1)
    d_idx = np.round(v / params["vel_bin_size"] + D / 2.0).astype(int)
    d_idx = np.clip(d_idx, 0, D - 1)
    cube = np.zeros((2, D, R, A), dtype=np.float32)
    cube[0, d_idx, r_idx, a_idx] = 1.0
    el_sum = np.zeros((D, R, A), dtype=np.float32)
    el_count = np.zeros((D, R, A), dtype=np.float32)
    for di, ri, ai, eli in zip(d_idx, r_idx, a_idx, el):
        el_sum[di, ri, ai] += eli
        el_count[di, ri, ai] += 1.0
    nz = el_count > 0
    cube[1][nz] = el_sum[nz] / el_count[nz]
    return cube


def random_points(n, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.column_stack((
        rng.uniform(0.5, 60.0, n),
        rng.uniform(-1.4, 1.4, n),
        rng.uniform(-0.3, 0.3, n),
        rng.normal(0.0, 2.0, n),
    )).astype(np.float32)
    # Force voxel collisions so the elevation averaging is exercised.
    pts[n // 2:, :] = pts[: n - n // 2, :]
    pts[n // 2:, 2] += 0.01
    return pts


def test_grid_matches_legacy_params():
    params = legacy_params()
    for key in ("range_axis", "azimuth_axis", "elevation_axis"):
        np.testing.assert_array_equal(getattr(GRID, key), params[key])
    assert GRID.vel_fft_size == params["vel_fft_size"]
    assert GRID.vel_bin_size == params["vel_bin_size"]
    assert GRID.radar_cube_shape == (2, 128, 500, 240)
    assert GRID.target_cube_shape == (34, 500, 240)


def test_vectorized_cube_matches_legacy_loop():
    params = legacy_params()
    pts = crop_to_roi(random_points(2000), GRID)
    np.testing.assert_array_equal(points_to_cube(pts, GRID), legacy_points_to_cube(pts, params))


def test_cube_empty_input():
    cube = points_to_cube(np.empty((0, 4), np.float32), GRID)
    assert cube.shape == GRID.radar_cube_shape and not cube.any()


def test_crop_to_roi_bounds():
    pts = np.array([[10, 0, 0, 0], [55, 0, 0, 0], [10, np.deg2rad(80), 0, 0]], np.float32)
    assert len(crop_to_roi(pts, GRID)) == 1


def test_ra_mask_min_points():
    pts = np.array([[10.0, 0.0, 0.0, 0.0]] * 3 + [[20.0, 0.2, 0.0, 0.0]], np.float32)
    assert points_to_ra_mask(pts, GRID).sum() == 2
    assert points_to_ra_mask(pts, GRID, min_points=2).sum() == 1


def test_spherical_round_trip():
    sph = random_points(100)[:, :3]
    np.testing.assert_allclose(cartesian_to_spherical(spherical_to_cartesian(sph)), sph, atol=1e-4)


def test_ego_motion_doppler_matches_legacy_formula():
    xyz = spherical_to_cartesian(random_points(50))
    vel, yaw = np.array([3.0, -1.0, 0.0], np.float32), 30.0
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    world = (rot @ xyz.T).T
    u = world / np.maximum(np.linalg.norm(world, axis=1, keepdims=True), 1e-6)
    expected = np.sum(-vel[None, :] * u, axis=1)
    np.testing.assert_allclose(ego_motion_doppler(xyz, vel, yaw), expected, rtol=1e-6)


def translation(x=0.0, y=0.0, z=0.0):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (x, y, z)
    return m


def test_measurement_to_array_reorders_carla_layout():
    raw = np.array([[-1.5, 0.1, 0.02, 12.0]], np.float32)  # [vel, az, alt, depth]
    meas = types.SimpleNamespace(raw_data=raw.tobytes())
    expected = np.array([[12.0, 0.1, 0.02, -1.5]], np.float32)
    np.testing.assert_array_equal(measurement_to_array(meas), expected)


def test_native_mode_keeps_carla_velocity_and_sensor_frame():
    proc = RadarFrameProcessor(RadarSensorConfig(points_frame="sensor"), build_cube=False)
    native = np.array([[10.0, 0.0, 0.0, -4.0]], np.float32)  # approaching
    out = proc.process(native, translation(x=2.0), np.zeros(3), 0.0, timestamp=0.0)
    np.testing.assert_allclose(out.points, native)
    assert out.cube is None


def test_ego_frame_applies_mount_offset():
    proc = RadarFrameProcessor(RadarSensorConfig(points_frame="ego"))
    native = np.array([[10.0, 0.0, 0.0, 0.0]], np.float32)
    out = proc.process(native, translation(x=2.0, z=1.0), np.zeros(3), 0.0, timestamp=0.0)
    np.testing.assert_allclose(out.points[0, 0], np.hypot(12.0, 1.0), rtol=1e-6)
    assert out.cube.shape == GRID.radar_cube_shape and out.cube[0].sum() == 1


def test_ego_motion_mode_uses_finite_difference_velocity():
    proc = RadarFrameProcessor(RadarSensorConfig(doppler_mode="ego_motion"), build_cube=False)
    native = np.array([[10.0, 0.0, 0.0, 123.0]], np.float32)
    first = proc.process(native, np.eye(4), np.zeros(3), 0.0, timestamp=0.0)
    assert first.points[0, 3] == 0.0  # no previous pose yet
    second = proc.process(native, np.eye(4), np.array([1.0, 0, 0]), 0.0, timestamp=0.05)
    # Driving forward at 20 m/s toward a static point: range rate is -20.
    assert second.points[0, 3] == pytest.approx(-20.0, rel=1e-5)


def test_seeded_physics_reproducible_but_varies_per_frame():
    physics = RadarPhysicsConfig(enabled=True, range_noise_std_m=0.5, seed=7)
    native = np.tile(np.array([[20.0, 0.0, 0.0, 0.0]], np.float32), (10, 1))

    def run():
        proc = RadarFrameProcessor(RadarSensorConfig(points_frame="sensor"), physics, build_cube=False)
        return [proc.process(native, np.eye(4), np.zeros(3), 0.0, t).points for t in (0.0, 0.05)]

    a, b = run(), run()
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[0], a[1])


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        RadarFrameProcessor(RadarSensorConfig(doppler_mode="bogus"))
