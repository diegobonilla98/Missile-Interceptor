from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model_assets import resolve_checkpoint_path, resolve_xgboost_path
from simulator import LaunchConfig, TrainedInterceptorSimulator


checkpoint_directory = Path("checkpoints")
checkpoint_model_path = None
xgboost_model_path = None
device_name = "cuda" if torch.cuda.is_available() else "cpu"
host = "127.0.0.1"
port = 8765
static_directory = Path("static")


def find_latest_checkpoint() -> Path:
    if not checkpoint_directory.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_directory}")
    candidates: list[tuple[Path, int]] = []
    for run_dir in checkpoint_directory.iterdir():
        if not run_dir.is_dir():
            continue
        priorities = [
            (run_dir / "best.pt", 3),
            (run_dir / "latest.pt", 2),
            (run_dir / "final.pt", 1),
        ]
        for path, priority in priorities:
            if path.exists():
                candidates.append((path, priority))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints found inside {checkpoint_directory}")
    candidates.sort(key=lambda entry: (entry[0].parent.stat().st_mtime, entry[1]), reverse=True)
    return candidates[0][0]


def resolve_server_checkpoint() -> Path:
    if checkpoint_model_path is not None:
        return resolve_checkpoint_path(checkpoint_model_path)
    try:
        return find_latest_checkpoint()
    except FileNotFoundError:
        return resolve_checkpoint_path()


checkpoint_path = resolve_server_checkpoint()
xgboost_path = resolve_xgboost_path(xgboost_model_path)
simulator = TrainedInterceptorSimulator(
    checkpoint_path=checkpoint_path,
    device=device_name,
    xgboost_path=xgboost_path,
)
app = FastAPI(title="Missile Interceptor 3D")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class LaunchRequest(BaseModel):
    launch_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    launch_azimuth_deg: float = Field(30.0, ge=-360.0, le=360.0)
    launch_elevation_deg: float = Field(55.0, ge=10.0, le=85.0)
    thrust_speed: float = Field(220.0, ge=80.0, le=900.0)
    target_position: tuple[float, float, float] = (1100.0, 400.0, 0.0)
    target_altitude: float = Field(60.0, ge=0.0, le=400.0)
    wind_strength_multiplier: float = Field(1.0, ge=0.0, le=4.0)
    jitter_strength_multiplier: float = Field(1.0, ge=0.0, le=5.0)
    noise_seed: int = Field(7, ge=0, le=2**31 - 1)
    launch_delay: float = Field(1.0, ge=0.0, le=8.0)
    interceptor_position: Optional[tuple[float, float, float]] = None
    interceptor_launch_distance: float = Field(360.0, ge=120.0, le=900.0)
    interceptor_battery_radius: float = Field(320.0, ge=80.0, le=900.0)
    interceptor_battery_altitude_min: float = Field(25.0, ge=0.0, le=400.0)
    interceptor_battery_altitude_max: float = Field(80.0, ge=0.0, le=600.0)
    interceptor_initial_speed: float = Field(160.0, ge=0.0, le=400.0)
    interceptor_spawn_mode: str = Field("near_target", pattern="^(near_target|near_missile)$")
    max_simulation_time: float = Field(20.0, ge=4.0, le=45.0)
    timestep: float = Field(0.04, ge=0.01, le=0.1)
    policy_mode: str = Field("ensemble", pattern="^(ensemble|policy|xgboost|expert)$")


@app.get("/api/health")
def health() -> dict:
    metrics = simulator.checkpoint_metadata.get("metrics", {})
    return {
        "status": "ok",
        "device": device_name,
        "checkpoint": str(checkpoint_path),
        "xgboost_path": str(xgboost_path),
        "architecture": simulator.architecture,
        "checkpoint_update": simulator.checkpoint_metadata.get("update", 0),
        "checkpoint_global_step": simulator.checkpoint_metadata.get("global_step", 0),
        "xgboost_available": simulator.xgboost_teacher is not None,
        "xgboost_teacher_weight": simulator.xgboost_teacher_weight,
        "world_limit": simulator.env_config.world_limit,
        "intercept_radius": simulator.env_config.intercept_radius,
        "interceptor_max_accel": simulator.env_config.interceptor_max_accel,
        "wind_strength": simulator.env_config.wind_strength,
        "gravity": simulator.env_config.gravity,
        "eval_metrics": {
            "raw_success": metrics.get("eval_raw_success_rate"),
            "ensemble_success": metrics.get("eval_ensemble_success_rate"),
            "expert_success": metrics.get("eval_expert_success_rate"),
        },
    }


@app.post("/api/simulate")
def simulate(request: LaunchRequest) -> JSONResponse:
    launch = LaunchConfig(
        launch_position=tuple(request.launch_position),
        launch_azimuth_deg=request.launch_azimuth_deg,
        launch_elevation_deg=request.launch_elevation_deg,
        thrust_speed=request.thrust_speed,
        target_position=tuple(request.target_position),
        target_altitude=request.target_altitude,
        wind_strength_multiplier=request.wind_strength_multiplier,
        jitter_strength_multiplier=request.jitter_strength_multiplier,
        noise_seed=request.noise_seed,
        launch_delay=request.launch_delay,
        interceptor_position=tuple(request.interceptor_position) if request.interceptor_position is not None else None,
        interceptor_launch_distance=request.interceptor_launch_distance,
        interceptor_battery_radius=request.interceptor_battery_radius,
        interceptor_battery_altitude_min=request.interceptor_battery_altitude_min,
        interceptor_battery_altitude_max=request.interceptor_battery_altitude_max,
        interceptor_initial_speed=request.interceptor_initial_speed,
        interceptor_spawn_mode=request.interceptor_spawn_mode,
        max_simulation_time=request.max_simulation_time,
        timestep=request.timestep,
        policy_mode=request.policy_mode,
    )
    try:
        result = simulator.simulate(launch)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}")
    payload = {
        "outcome": result.outcome,
        "min_distance": result.min_distance,
        "intercept_time": result.intercept_time,
        "intercept_radius": result.intercept_radius,
        "target_position": list(result.target_position),
        "interceptor_launch_position": list(result.interceptor_launch_position),
        "missile_initial_velocity": list(result.missile_initial_velocity),
        "missile_path_points": [list(point) for point in result.missile_path_points],
        "missile_duration": result.missile_duration,
        "config": result.config,
        "frames": [
            {
                "t": frame.time,
                "mp": list(frame.missile_position),
                "mv": list(frame.missile_velocity),
                "ma": frame.missile_alive,
                "mt": frame.missile_thrusting,
                "ip": list(frame.interceptor_position),
                "iv": list(frame.interceptor_velocity),
                "ia": frame.interceptor_active,
                "ic": list(frame.interceptor_action),
                "wm": list(frame.wind_missile),
                "wi": list(frame.wind_interceptor),
                "d": frame.distance,
                "tau": frame.missile_tau,
            }
            for frame in result.frames
        ],
    }
    return JSONResponse(payload)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_directory / "index.html")


app.mount("/static", StaticFiles(directory=str(static_directory)), name="static")


if __name__ == "__main__":
    uvicorn.run("server:app", host=host, port=port, reload=False, log_level="info")
