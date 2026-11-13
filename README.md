# radar-simulation-code

`carla_radar4d` is a 4D radar (range, azimuth, elevation, Doppler) sensor for
CARLA. It wraps CARLA's `sensor.other.radar`, can add configurable measurement
effects (noise, range-dependent detection loss, multipath ghosts, clutter), and
bins each measurement into the `(2, D, R, A)` radar cube used by the neural
detector in `radar_pipeline/`.

## Install

```bash
pip install -e .            # core: numpy only
pip install -e ".[dev]"     # + pytest
pip install -e ".[detector]"  # + torch stack for training / closed-loop scripts
```

The CARLA Python API must match your server version. Install it from the
simulator's `PythonAPI/carla/dist` wheel, or use `pip install -e ".[sim]"`
if a matching PyPI build exists.

The scripts in `radar_pipeline/` now import `carla_radar4d`, so install the
package (or set `PYTHONPATH` to the repo root) before running them.

## Usage

```python
import carla
from carla_radar4d import Radar4DSensor, RadarSensorConfig, RadarPhysicsConfig

client = carla.Client("127.0.0.1", 2002)
world = client.get_world()
ego = ...  # your vehicle actor

radar = Radar4DSensor.spawn(
    world,
    parent=ego,
    config=RadarSensorConfig(horizontal_fov=70, vertical_fov=20, range=100,
                             doppler_mode="native", points_frame="ego"),
    physics=RadarPhysicsConfig(enabled=True, range_noise_std_m=0.1,
                               clutter_per_detection=0.05, seed=0),
)
radar.listen(lambda f: print(f.frame, f.points.shape, f.cube.shape))
# ... world.tick() ...
radar.destroy()
```

Each `RadarFrame` carries:

- `points`: `(N, 4)` rows of `[range_m, azimuth_rad, elevation_rad, radial_velocity_mps]`
- `cube`: `(2, 128, 500, 240)`; channel 0 is occupancy, channel 1 is the mean elevation per voxel

### Conventions

- **Radial velocity is range rate: negative when the target is approaching.**
  CARLA's native radar and the training dataset both use this sign.
- `doppler_mode="native"` keeps CARLA's per-detection velocity, which includes
  target motion. `"ego_motion"` reproduces the training dataset, where Doppler
  comes from ego motion only and moving targets get static-world Doppler.
- `points_frame="ego"` expresses points relative to the parent actor's origin,
  as the dataset and the trained detector expect. `"sensor"` keeps CARLA's
  radar frame.
- `CubeGrid.radelft()` is the grid the detector was trained on. Pass a custom
  `CubeGrid` to change it.

To process radar data without a running simulator (for tests or offline
replay), use `RadarFrameProcessor.process(...)` directly.

## Tests

```bash
python -m pytest
```
## Generative AI Usage 

Generative AI was used to format this code. The output has been reviewed by the human author. 
