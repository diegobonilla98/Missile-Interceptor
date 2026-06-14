# Missile Interceptor

Simulation project for testing an interceptor guidance policy against procedurally generated missile trajectories in a 3D Three.js demo.

The recommended model is the trained ensemble used by the HTML demo. It blends the PPO neural policy with the XGBoost guidance teacher:

- PPO checkpoint: `https://huggingface.co/Boni98/MissileInterceptor/blob/main/best.pt`
- XGBoost teacher: `https://huggingface.co/Boni98/MissileInterceptor/blob/main/xgboost_guidance.joblib`

The code auto-downloads these files from `Boni98/MissileInterceptor` when model paths are not provided.

This is a simulation and visualization project, not an operational weapons system.

## What The Model Does

The simulator builds a noisy 3D missile path, spawns an interceptor near the protected target, and asks the guidance model for acceleration commands at each timestep. The default `ensemble` mode combines:

- an analytical guidance prior,
- the trained PPO residual policy from `best.pt`,
- the XGBoost teacher from `xgboost_guidance.joblib`.

The web demo exposes other modes for comparison, but `ensemble` is the trained model path to use.

## Best Checkpoint Metrics

Metrics are read from `checkpoints/projectile_intercept_20260430_134532/best.pt`.

| Metric | Value |
|---|---:|
| Best update | 90 |
| Environment steps | 17,694,720 |
| Architecture | Hybrid LSTM + Transformer |
| Hidden size | 512 |
| LSTM layers | 3 |
| Transformer layers | 3 |
| Attention heads | 8 |
| Evaluation episodes | 192 |
| Ensemble success rate | 100.00% |
| Ensemble mean minimum distance | 12.93 m |
| Policy + guidance success rate | 100.00% |
| XGBoost teacher success rate | 100.00% |
| Analytical expert success rate | 100.00% |
| Raw neural residual success rate | 2.08% |
| Training success rate at checkpoint | 100.00% |
| Intercept radius during eval | 15.88 m |
| Difficulty level | 1.5 |

The raw residual is intentionally weak by itself because the trained agent is configured as a residual controller on top of a guidance prior. Use `ensemble` unless you are doing ablation tests.

## Setup

Use the project conda environment:

```powershell
conda activate python311
cd E:\HugeProjects\TargetPrediction
python -m pip install -r requirements.txt
```

If you prefer the full Python path:

```powershell
cd E:\HugeProjects\TargetPrediction
C:\Users\diego\anaconda3\envs\python311\python.exe -m pip install -r requirements.txt
```

## Run The 3D Demo

```powershell
cd E:\HugeProjects\TargetPrediction
C:\Users\diego\anaconda3\envs\python311\python.exe server.py
```

Open:

```text
http://127.0.0.1:8765
```

The demo defaults to `Trained agent (ensemble)`. Click `LAUNCH` to run a simulation. Use `Policy`, `XGBoost`, and `Analytical PN guidance` only for comparisons.

Health and loaded-model metadata are available at:

```text
http://127.0.0.1:8765/api/health
```

## Use The Model From Python

You can omit both paths. The simulator will download `best.pt` and `xgboost_guidance.joblib` into `models/` if they are missing.

```python
from simulator import LaunchConfig, TrainedInterceptorSimulator


device = "cpu"
simulator = TrainedInterceptorSimulator(device=device)

launch = LaunchConfig(
    launch_azimuth_deg=30.0,
    launch_elevation_deg=55.0,
    thrust_speed=220.0,
    target_position=(1100.0, 400.0, 0.0),
    target_altitude=60.0,
    wind_strength_multiplier=1.0,
    jitter_strength_multiplier=1.0,
    noise_seed=7,
    launch_delay=1.0,
    interceptor_spawn_mode="near_target",
    policy_mode="ensemble",
)

result = simulator.simulate(launch)

print(result.outcome)
print(result.min_distance)
print(result.intercept_time)
```

To use explicit local files:

```python
from pathlib import Path

from simulator import LaunchConfig, TrainedInterceptorSimulator


checkpoint_path = Path("models/best.pt")
xgboost_path = Path("models/xgboost_guidance.joblib")
device = "cuda"

simulator = TrainedInterceptorSimulator(
    checkpoint_path=checkpoint_path,
    xgboost_path=xgboost_path,
    device=device,
)

launch = LaunchConfig(policy_mode="ensemble")
result = simulator.simulate(launch)
```

## API Example

With the server running:

```python
import json
from urllib.request import Request, urlopen


url = "http://127.0.0.1:8765/api/simulate"
payload = {
    "launch_position": [0.0, 0.0, 0.0],
    "launch_azimuth_deg": 30.0,
    "launch_elevation_deg": 55.0,
    "thrust_speed": 220.0,
    "target_position": [1100.0, 400.0, 0.0],
    "target_altitude": 60.0,
    "wind_strength_multiplier": 1.0,
    "jitter_strength_multiplier": 1.0,
    "noise_seed": 7,
    "launch_delay": 1.0,
    "interceptor_spawn_mode": "near_target",
    "policy_mode": "ensemble",
    "max_simulation_time": 22.0,
    "timestep": 0.04,
}
body = json.dumps(payload).encode("utf-8")
request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")

with urlopen(request, timeout=60) as response:
    data = json.loads(response.read().decode("utf-8"))

print(data["outcome"])
print(data["min_distance"])
print(data["intercept_time"])
```

## Model Files

Default local cache paths:

```text
models/best.pt
models/xgboost_guidance.joblib
```

The server first looks for the latest local checkpoint under `checkpoints/`. If none exists, it downloads `models/best.pt` from Hugging Face. Direct Python use with `TrainedInterceptorSimulator(device=device)` downloads both public assets automatically when missing.

## Important Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI app that serves the HTML demo and simulation API |
| `simulator.py` | Loads the trained model and runs one simulation |
| `model_assets.py` | Resolves and downloads Hugging Face model assets |
| `train.py` | PPO + XGBoost training pipeline |
| `xgboost_guidance.py` | XGBoost teacher training, save, load, and prediction helpers |
| `static/index.html` | Demo UI |
| `static/main.js` | Three.js visualization and API client |

## Policy Modes

| Mode | Meaning |
|---|---|
| `ensemble` | Recommended trained model: PPO guidance policy blended with XGBoost teacher |
| `policy` | PPO guidance policy without XGBoost blending |
| `xgboost` | XGBoost teacher only |
| `expert` | Analytical proportional-navigation-style guidance |
