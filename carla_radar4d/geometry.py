"""Coordinate conversions shared by the sensor, cube builder and scripts.

Spherical convention (CARLA radar): ``[range_m, azimuth_rad, elevation_rad]``
with x forward, y right, z up.
"""

import numpy as np


def spherical_to_cartesian(points: np.ndarray) -> np.ndarray:
    """(N, 3+) ``[r, az, el, ...]`` -> (N, 3) ``[x, y, z]``."""
    points = np.asarray(points)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    r, az, el = points[:, 0], points[:, 1], points[:, 2]
    return np.stack(
        [r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)],
        axis=1,
    ).astype(np.float32)


def cartesian_to_spherical(xyz: np.ndarray) -> np.ndarray:
    """(N, 3) ``[x, y, z]`` -> (N, 3) ``[r, az, el]``."""
    xyz = np.asarray(xyz)
    if xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.arcsin(np.clip(z / np.maximum(r, 1e-6), -1.0, 1.0))
    return np.stack([r, np.arctan2(y, x), el], axis=1).astype(np.float32)


def apply_transform(matrix: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform (e.g. ``carla.Transform.get_matrix()``)."""
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    return (xyz @ m[:3, :3].T + m[:3, 3]).astype(np.float32)


def yaw_rotation(yaw_deg: float) -> np.ndarray:
    """Rotation about z from a yaw angle in degrees (ego -> world, yaw only)."""
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def ego_motion_doppler(
    points_ego_xyz: np.ndarray, ego_velocity_world: np.ndarray, ego_yaw_deg: float
) -> np.ndarray:
    """Doppler from ego motion only (static-world assumption).

    Returns the range rate ``-v_ego . u`` (negative when approaching).

    Reproduces the velocity channel of the original training dataset.
    """
    points_ego_xyz = np.asarray(points_ego_xyz, dtype=np.float32)
    if points_ego_xyz.size == 0:
        return np.empty(0, dtype=np.float32)
    rays_world = points_ego_xyz @ yaw_rotation(ego_yaw_deg).T
    norms = np.maximum(np.linalg.norm(rays_world, axis=1, keepdims=True), 1e-6)
    unit = rays_world / norms
    ego_velocity_world = np.asarray(ego_velocity_world, dtype=np.float32)
    return np.sum(-ego_velocity_world[None, :] * unit, axis=1).astype(np.float32)
