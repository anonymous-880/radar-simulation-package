"""Spherical radar points -> 4D radar cube (Doppler x range x azimuth, + elevation)."""

import numpy as np

from .config import CubeGrid


def crop_to_roi(points: np.ndarray, grid: CubeGrid) -> np.ndarray:
    """Keep rows ``[r, az, el, ...]`` inside ``grid.roi_spherical``."""
    points = np.asarray(points)
    if points.size == 0:
        return points
    r_min, az_min, el_min, r_max, az_max, el_max = grid.roi_spherical
    r, az, el = points[:, 0], points[:, 1], points[:, 2]
    mask = (
        (r >= r_min) & (r <= r_max)
        & (az >= np.deg2rad(az_min)) & (az <= np.deg2rad(az_max))
        & (el >= np.deg2rad(el_min)) & (el <= np.deg2rad(el_max))
    )
    return points[mask]


def points_to_cube(points: np.ndarray, grid: CubeGrid) -> np.ndarray:
    """Bin (N, 4) ``[r, az, el, doppler]`` points into a (2, D, R, A) cube.

    Channel 0 is binary occupancy; channel 1 is the mean elevation (rad) of
    the points in each voxel.  Out-of-grid indices are clipped to the edges,
    so crop with :func:`crop_to_roi` first.
    """
    cube = np.zeros(grid.radar_cube_shape, dtype=np.float32)
    points = np.asarray(points)
    if points.size == 0:
        return cube

    _, D, R, A = grid.radar_cube_shape
    r_idx = np.clip(np.searchsorted(grid.range_axis, points[:, 0], side="left"), 0, R - 1)
    a_idx = np.clip(np.searchsorted(grid.azimuth_axis, points[:, 1], side="left"), 0, A - 1)
    d_idx = np.round(points[:, 3] / grid.vel_bin_size + D / 2.0).astype(int)
    d_idx = np.clip(d_idx, 0, D - 1)

    cube[0, d_idx, r_idx, a_idx] = 1.0

    el_sum = np.zeros((D, R, A), dtype=np.float32)
    el_count = np.zeros((D, R, A), dtype=np.float32)
    np.add.at(el_sum, (d_idx, r_idx, a_idx), points[:, 2].astype(np.float32))
    np.add.at(el_count, (d_idx, r_idx, a_idx), 1.0)
    nonzero = el_count > 0
    cube[1][nonzero] = el_sum[nonzero] / el_count[nonzero]
    return cube


def points_to_ra_mask(points: np.ndarray, grid: CubeGrid, min_points: int = 1) -> np.ndarray:
    """(R, A) boolean occupancy; points outside the ROI/grid are dropped."""
    R, A = len(grid.range_axis), len(grid.azimuth_axis)
    points = crop_to_roi(points, grid)
    if points.size == 0:
        return np.zeros((R, A), dtype=bool)
    r_idx = np.searchsorted(grid.range_axis, points[:, 0], side="left")
    a_idx = np.searchsorted(grid.azimuth_axis, points[:, 1], side="left")
    valid = (r_idx < R) & (a_idx < A)
    counts = np.zeros((R, A), dtype=np.int32)
    np.add.at(counts, (r_idx[valid], a_idx[valid]), 1)
    return counts >= min_points
