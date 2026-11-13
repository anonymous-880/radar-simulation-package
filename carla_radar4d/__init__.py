"""4D radar for CARLA: native radar + physics effects + range/Doppler/azimuth cube."""

from .config import CubeGrid, MountTransform, RadarSensorConfig
from .cube import crop_to_roi, points_to_cube, points_to_ra_mask
from .physics import RadarPhysicsConfig, apply_radar_physics, approaching_radial_velocity
from .sensor import Radar4DSensor, RadarFrame, RadarFrameProcessor

__version__ = "0.1.0"

__all__ = [
    "CubeGrid",
    "MountTransform",
    "RadarSensorConfig",
    "RadarPhysicsConfig",
    "Radar4DSensor",
    "RadarFrame",
    "RadarFrameProcessor",
    "apply_radar_physics",
    "approaching_radial_velocity",
    "crop_to_roi",
    "points_to_cube",
    "points_to_ra_mask",
]
