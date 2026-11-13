"""Single source of truth for sensor settings and the 4D radar cube grid."""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class MountTransform:
    """Sensor pose relative to its parent actor (CARLA units: m, deg)."""

    x: float = 2.0
    y: float = 0.0
    z: float = 1.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

    def to_carla(self):
        import carla

        return carla.Transform(
            carla.Location(x=self.x, y=self.y, z=self.z),
            carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll),
        )


@dataclass(frozen=True)
class RadarSensorConfig:
    """Attributes forwarded to CARLA's ``sensor.other.radar`` blueprint.

    ``doppler_mode`` selects the velocity channel:

    * ``"native"``: CARLA's per-detection relative radial velocity (includes
      target motion).
    * ``"ego_motion"``: velocity induced by ego motion only, as used to build
      the original training dataset.  Moving targets get static-world Doppler.

    Both are range rates: negative when the target is approaching.

    ``points_frame`` selects the origin of the spherical coordinates:
    ``"ego"`` (parent actor origin, matches the training dataset) or
    ``"sensor"`` (CARLA's native radar frame).
    """

    horizontal_fov: float = 70.0
    vertical_fov: float = 20.0
    range: float = 100.0
    points_per_second: int = 500_000
    sensor_tick: float = 0.0
    mount: MountTransform = field(default_factory=MountTransform)
    doppler_mode: str = "native"
    points_frame: str = "ego"

    def validate(self) -> None:
        if self.doppler_mode not in ("native", "ego_motion"):
            raise ValueError("doppler_mode must be 'native' or 'ego_motion'")
        if self.points_frame not in ("ego", "sensor"):
            raise ValueError("points_frame must be 'ego' or 'sensor'")

    def blueprint_attributes(self) -> Dict[str, str]:
        return {
            "horizontal_fov": str(self.horizontal_fov),
            "vertical_fov": str(self.vertical_fov),
            "range": str(self.range),
            "points_per_second": str(self.points_per_second),
            "sensor_tick": str(self.sensor_tick),
        }


def _radelft_range_axis() -> np.ndarray:
    range_cell_size = 0.1004
    max_range = 51.4242
    axis = np.arange(range_cell_size, max_range + range_cell_size, range_cell_size)
    return axis[10:-3]


def _fft_angle_axis(fft_size: int, start: int, stop: int) -> np.ndarray:
    w = np.linspace(-np.pi, np.pi, fft_size)[start:stop]
    return np.arcsin(w / (2 * np.pi * 0.4972))


@dataclass(frozen=True, eq=False)
class CubeGrid:
    """Axes of the (2, D, R, A) radar cube and the (E, R, A) target cube.

    ``roi_spherical`` is ``(r_min, az_min_deg, el_min_deg, r_max, az_max_deg,
    el_max_deg)``.
    """

    range_axis: np.ndarray
    azimuth_axis: np.ndarray
    elevation_axis: np.ndarray
    vel_fft_size: int
    vel_bin_size: float
    roi_spherical: Tuple[float, float, float, float, float, float]

    @classmethod
    def radelft(cls) -> "CubeGrid":
        """The RaDelft-style grid the published detector was trained on."""
        return cls(
            range_axis=_radelft_range_axis(),
            azimuth_axis=_fft_angle_axis(256, 8, 248),
            elevation_axis=_fft_angle_axis(128, 47, 81),
            vel_fft_size=128,
            vel_bin_size=0.04607058455831936,
            roi_spherical=(0.0, -70.0, -15.0, 50.0, 70.0, 15.0),
        )

    @classmethod
    def from_params(cls, params: dict) -> "CubeGrid":
        """Build from a legacy ``get_default_params()`` dictionary."""
        return cls(
            range_axis=np.asarray(params["range_axis"]),
            azimuth_axis=np.asarray(params["azimuth_axis"]),
            elevation_axis=np.asarray(params["elevation_axis"]),
            vel_fft_size=int(params["vel_fft_size"]),
            vel_bin_size=float(params["vel_bin_size"]),
            roi_spherical=tuple(float(v) for v in params["roi_spherical"]),
        )

    @property
    def radar_cube_shape(self) -> Tuple[int, int, int, int]:
        return (2, self.vel_fft_size, len(self.range_axis), len(self.azimuth_axis))

    @property
    def target_cube_shape(self) -> Tuple[int, int, int]:
        return (len(self.elevation_axis), len(self.range_axis), len(self.azimuth_axis))
