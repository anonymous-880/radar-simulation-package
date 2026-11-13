"""CARLA-attached 4D radar: native radar + physics effects + radar cube.

Example::

    from carla_radar4d import Radar4DSensor, RadarSensorConfig, RadarPhysicsConfig

    radar = Radar4DSensor.spawn(
        world, parent=ego,
        config=RadarSensorConfig(doppler_mode="native"),
        physics=RadarPhysicsConfig(enabled=True, range_noise_std_m=0.1, seed=0),
    )
    radar.listen(lambda frame: print(frame.frame, frame.points.shape))
    ...
    radar.destroy()
"""

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .config import CubeGrid, RadarSensorConfig
from .cube import crop_to_roi, points_to_cube
from .geometry import (
    apply_transform,
    cartesian_to_spherical,
    ego_motion_doppler,
    spherical_to_cartesian,
)
from .physics import RadarPhysicsConfig, apply_radar_physics

BLUEPRINT_ID = "sensor.other.radar"


@dataclass
class RadarFrame:
    """One processed radar measurement.

    ``points`` rows are ``[range_m, azimuth_rad, elevation_rad,
    radial_velocity_mps]`` (range rate: negative when approaching, same as
    CARLA and the training dataset), expressed in the
    frame chosen by ``RadarSensorConfig.points_frame``.  ``cube`` is the
    (2, D, R, A) radar cube, or ``None`` when cube building is disabled.
    """

    frame: int
    timestamp: float
    points: np.ndarray
    cube: Optional[np.ndarray] = None


def measurement_to_array(measurement) -> np.ndarray:
    """``carla.RadarMeasurement`` -> (N, 4) ``[depth, azimuth, altitude, velocity]``.

    CARLA packs each detection as ``[velocity, azimuth, altitude, depth]``.
    """
    raw = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    return raw[:, [3, 1, 2, 0]].copy()


class RadarFrameProcessor:
    """CARLA-independent processing core (unit-testable without a simulator)."""

    def __init__(
        self,
        config: Optional[RadarSensorConfig] = None,
        physics: Optional[RadarPhysicsConfig] = None,
        grid: Optional[CubeGrid] = None,
        build_cube: bool = True,
    ):
        self.config = config or RadarSensorConfig()
        self.config.validate()
        self.physics = physics
        if physics is not None:
            physics.validate()
        self.grid = grid or CubeGrid.radelft()
        self.build_cube = build_cube
        # One generator for the sensor's lifetime so a fixed seed gives a
        # reproducible *sequence* rather than identical noise every frame.
        self._rng = np.random.default_rng(physics.seed if physics else None)
        self._last_parent_pos: Optional[np.ndarray] = None
        self._last_time: Optional[float] = None

    def reset(self) -> None:
        self._last_parent_pos = None
        self._last_time = None

    def process(
        self,
        native: np.ndarray,
        sensor_to_parent: np.ndarray,
        parent_position_world: np.ndarray,
        parent_yaw_deg: float,
        timestamp: float,
        frame: int = 0,
    ) -> RadarFrame:
        """Process native rows ``[depth, azimuth, altitude, velocity]``.

        ``velocity`` is CARLA's native range rate (negative when approaching).
        ``sensor_to_parent`` is the 4x4 mount matrix.
        """
        native = np.asarray(native, dtype=np.float32).reshape(-1, 4)
        parent_position_world = np.asarray(parent_position_world, dtype=np.float32)

        ego_velocity = np.zeros(3, dtype=np.float32)
        if self._last_parent_pos is not None and self._last_time is not None:
            dt = max(timestamp - self._last_time, 1e-6)
            ego_velocity = (parent_position_world - self._last_parent_pos) / dt
        self._last_parent_pos = parent_position_world.copy()
        self._last_time = timestamp

        points = native.copy()
        if self.config.doppler_mode == "ego_motion":
            xyz_parent = apply_transform(sensor_to_parent, spherical_to_cartesian(native))
            points[:, 3] = ego_motion_doppler(xyz_parent, ego_velocity, parent_yaw_deg)

        if self.physics is not None:
            points = apply_radar_physics(points, self.physics, rng=self._rng)

        if self.config.points_frame == "ego" and len(points):
            xyz_parent = apply_transform(sensor_to_parent, spherical_to_cartesian(points))
            points = np.column_stack(
                (cartesian_to_spherical(xyz_parent), points[:, 3])
            ).astype(np.float32)

        cube = None
        if self.build_cube:
            cube = points_to_cube(crop_to_roi(points, self.grid), self.grid)
        return RadarFrame(frame=frame, timestamp=timestamp, points=points, cube=cube)


class Radar4DSensor:
    """A ``sensor.other.radar`` actor with 4D-radar post-processing attached."""

    def __init__(self, actor, parent, processor: RadarFrameProcessor):
        self.actor = actor
        self.parent = parent
        self.processor = processor
        self._sensor_to_parent = np.asarray(
            processor.config.mount.to_carla().get_matrix(), dtype=np.float32
        )
        self._lock = threading.Lock()
        self._latest: Optional[RadarFrame] = None

    @classmethod
    def spawn(
        cls,
        world,
        parent,
        config: Optional[RadarSensorConfig] = None,
        physics: Optional[RadarPhysicsConfig] = None,
        grid: Optional[CubeGrid] = None,
        build_cube: bool = True,
    ) -> "Radar4DSensor":
        processor = RadarFrameProcessor(config, physics, grid, build_cube)
        blueprint = world.get_blueprint_library().find(BLUEPRINT_ID)
        for name, value in processor.config.blueprint_attributes().items():
            if blueprint.has_attribute(name):
                blueprint.set_attribute(name, value)
        actor = world.spawn_actor(
            blueprint, processor.config.mount.to_carla(), attach_to=parent
        )
        return cls(actor, parent, processor)

    def _handle(self, measurement) -> RadarFrame:
        parent_tf = self.parent.get_transform()
        loc = parent_tf.location
        frame = self.processor.process(
            native=measurement_to_array(measurement),
            sensor_to_parent=self._sensor_to_parent,
            parent_position_world=np.array([loc.x, loc.y, loc.z], dtype=np.float32),
            parent_yaw_deg=parent_tf.rotation.yaw,
            timestamp=measurement.timestamp,
            frame=measurement.frame,
        )
        with self._lock:
            self._latest = frame
        return frame

    def listen(self, callback: Optional[Callable[[RadarFrame], None]] = None) -> None:
        """Start streaming; ``callback`` runs on CARLA's sensor thread."""

        def _on_measurement(measurement):
            frame = self._handle(measurement)
            if callback is not None:
                callback(frame)

        self.actor.listen(_on_measurement)

    @property
    def latest(self) -> Optional[RadarFrame]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        if self.actor is not None and self.actor.is_listening:
            self.actor.stop()

    def destroy(self) -> None:
        if self.actor is not None:
            self.stop()
            self.actor.destroy()
            self.actor = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()
