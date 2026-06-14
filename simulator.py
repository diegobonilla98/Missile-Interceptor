from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from xgboost_guidance import load_xgboost_model, normalize_numpy_actions
from agent import HybridActorCritic, LSTMActorCritic
from environment import EnvironmentConfig, PerlinNoise3D
from model_assets import resolve_checkpoint_path, resolve_xgboost_path
from ppo import TorchRunningMeanStd


@dataclass
class LaunchConfig:
    launch_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    launch_azimuth_deg: float = 30.0
    launch_elevation_deg: float = 55.0
    thrust_speed: float = 220.0
    target_position: tuple[float, float, float] = (1100.0, 400.0, 0.0)
    target_altitude: float = 60.0
    wind_strength_multiplier: float = 1.0
    jitter_strength_multiplier: float = 1.0
    noise_seed: int = 7
    launch_delay: float = 1.0
    interceptor_position: tuple[float, float, float] | None = None
    interceptor_launch_distance: float = 360.0
    interceptor_battery_radius: float = 320.0
    interceptor_battery_altitude_min: float = 25.0
    interceptor_battery_altitude_max: float = 80.0
    interceptor_initial_speed: float = 160.0
    interceptor_spawn_mode: str = "near_target"
    max_simulation_time: float = 22.0
    timestep: float = 0.04
    policy_mode: str = "ensemble"


@dataclass
class SimulationFrame:
    time: float
    missile_position: tuple[float, float, float]
    missile_velocity: tuple[float, float, float]
    missile_alive: bool
    missile_thrusting: bool
    interceptor_position: tuple[float, float, float]
    interceptor_velocity: tuple[float, float, float]
    interceptor_active: bool
    interceptor_action: tuple[float, float, float]
    wind_missile: tuple[float, float, float]
    wind_interceptor: tuple[float, float, float]
    distance: float
    missile_tau: float


@dataclass
class SimulationResult:
    frames: list[SimulationFrame]
    outcome: str
    min_distance: float
    intercept_time: float
    intercept_radius: float
    target_position: tuple[float, float, float]
    interceptor_launch_position: tuple[float, float, float]
    missile_initial_velocity: tuple[float, float, float]
    missile_path_points: list[tuple[float, float, float]]
    missile_duration: float
    config: dict = field(default_factory=dict)


class TrainedInterceptorSimulator:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        xgboost_path: str | Path | None = None,
        auto_download: bool = True,
    ):
        self.device = torch.device(device)
        checkpoint_path = resolve_checkpoint_path(checkpoint_path, auto_download=auto_download)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        env_config_dict = checkpoint.get("environment_config", {})
        self.env_config = EnvironmentConfig(**env_config_dict)
        obs_rms_state = checkpoint.get("obs_rms", {})
        if "mean" in obs_rms_state:
            checkpoint_obs_dim = int(obs_rms_state["mean"].shape[-1])
        else:
            checkpoint_obs_dim = 41
        self.observation_dim = checkpoint_obs_dim
        self.action_dim = 3
        hidden_size = checkpoint.get("hidden_size", 256)
        lstm_layers = checkpoint.get("lstm_layers", 1)
        transformer_layers = checkpoint.get("transformer_layers", 2)
        attention_heads = checkpoint.get("attention_heads", 8)
        initial_log_std = checkpoint.get("policy_initial_log_std", -1.6)
        architecture = checkpoint.get("model_architecture", "lstm")
        self.architecture = architecture
        self.use_guidance_prior = bool(checkpoint.get("use_guidance_prior", False))
        self.residual_action_scale = float(checkpoint.get("residual_action_scale", 0.05))
        if architecture == "hybrid":
            self.model = HybridActorCritic(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                hidden_size=hidden_size,
                lstm_layers=lstm_layers,
                transformer_layers=transformer_layers,
                attention_heads=attention_heads,
                actor_hidden_size=hidden_size,
                critic_hidden_size=hidden_size,
                initial_log_std=initial_log_std,
            ).to(self.device)
        else:
            self.model = LSTMActorCritic(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                hidden_size=hidden_size,
                lstm_layers=lstm_layers,
                actor_hidden_size=hidden_size,
                critic_hidden_size=hidden_size,
                initial_log_std=initial_log_std,
            ).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.obs_rms = TorchRunningMeanStd((self.observation_dim,), device=self.device)
        self.obs_rms.load_state_dict(checkpoint["obs_rms"])
        self.checkpoint_metadata = {
            "update": checkpoint.get("update", 0),
            "global_step": checkpoint.get("global_step", 0),
            "architecture": architecture,
            "hidden_size": hidden_size,
            "lstm_layers": lstm_layers,
            "transformer_layers": transformer_layers,
            "attention_heads": attention_heads,
            "use_guidance_prior": self.use_guidance_prior,
            "residual_action_scale": self.residual_action_scale,
            "observation_dim": self.observation_dim,
            "metrics": checkpoint.get("metrics", {}),
        }
        self.xgboost_teacher = None
        self.xgboost_teacher_weight = float(checkpoint.get("xgboost_teacher_weight", 0.35))
        self.use_xgboost_teacher_default = bool(checkpoint.get("use_xgboost_teacher", False))
        if xgboost_path is not None or auto_download:
            xgboost_path = resolve_xgboost_path(xgboost_path, auto_download=auto_download)
            if xgboost_path.exists():
                self.xgboost_teacher, self.xgboost_metrics = load_xgboost_model(xgboost_path)
            else:
                self.xgboost_metrics = {}
        else:
            self.xgboost_metrics = {}

    def simulate(self, launch: LaunchConfig) -> SimulationResult:
        rng = np.random.default_rng(launch.noise_seed)
        noise = PerlinNoise3D(launch.noise_seed)
        env_config = self.env_config
        gravity_vector = np.array([0.0, 0.0, -env_config.gravity], dtype=np.float64)
        dt = float(launch.timestep)
        max_steps = int(launch.max_simulation_time / dt) + 4

        launch_position = np.asarray(launch.launch_position, dtype=np.float64).copy()
        target_position = np.asarray(launch.target_position, dtype=np.float64).copy()
        target_position[2] = max(target_position[2], float(launch.target_altitude))
        azimuth = np.deg2rad(launch.launch_azimuth_deg)
        elevation = np.deg2rad(launch.launch_elevation_deg)
        launch_direction = np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ], dtype=np.float64)
        thrust_speed = float(launch.thrust_speed)

        path = self._build_missile_path(
            launch_position=launch_position,
            target_position=target_position,
            launch_direction=launch_direction,
            thrust_speed=thrust_speed,
            noise_seed=launch.noise_seed,
        )
        missile_duration = path["missile_duration"]
        missile_position = self._missile_path_position(path, 0.0)
        missile_velocity = self._missile_path_derivative(path, 0.0) / missile_duration
        speed_norm = float(np.linalg.norm(missile_velocity))
        if speed_norm > 1e-6:
            missile_velocity = missile_velocity * (thrust_speed / max(speed_norm, 1.0))
        missile_initial_velocity = missile_velocity.copy()
        reference_position = missile_position.copy()

        sample_taus = np.linspace(0.0, 1.0, 64)
        missile_path_points = [tuple(self._missile_path_position(path, float(t)).tolist()) for t in sample_taus]

        interceptor_position = np.zeros(3, dtype=np.float64)
        interceptor_velocity = np.zeros(3, dtype=np.float64)
        interceptor_initialized = False
        interceptor_active = False

        if launch.interceptor_position is not None:
            interceptor_position = np.asarray(launch.interceptor_position, dtype=np.float64).copy()
            interceptor_position[2] = max(interceptor_position[2], 25.0)
            interceptor_velocity = np.zeros(3, dtype=np.float64)
            interceptor_initialized = True

        hidden_state = self.model.initial_state(1, self.device)
        episode_start = torch.ones(1, dtype=torch.float32, device=self.device)
        last_action = np.zeros(3, dtype=np.float64)

        time_value = 0.0
        elapsed = 0.0
        interceptor_episode_elapsed = 0.0

        wind_missile = self._wind(noise, missile_position, time_value, env_config) * float(launch.wind_strength_multiplier)
        wind_interceptor = np.zeros(3, dtype=np.float64)
        if interceptor_initialized:
            wind_interceptor = self._wind(noise, interceptor_position, time_value, env_config) * float(launch.wind_strength_multiplier)
        distance = float(np.linalg.norm(missile_position - interceptor_position)) if interceptor_initialized else float("inf")

        frames: list[SimulationFrame] = []
        outcome = "running"
        min_distance = float("inf")
        intercept_time = -1.0
        missile_alive = True
        recorded_launch_position = tuple(float(x) for x in interceptor_position) if interceptor_initialized else None

        frames.append(self._make_frame(
            time_value,
            missile_position,
            missile_velocity,
            True,
            True,
            interceptor_position,
            interceptor_velocity,
            False,
            last_action,
            wind_missile,
            wind_interceptor,
            distance if interceptor_initialized else 0.0,
            self._tau_from_elapsed(elapsed, missile_duration),
        ))

        intercept_radius = max(self.env_config.intercept_radius, 12.0)

        for _ in range(max_steps):
            elapsed_before = elapsed
            time_value += dt
            elapsed += dt

            if missile_alive:
                missile_position, missile_velocity, reference_position, wind_missile = self._step_missile(
                    path=path,
                    missile_position=missile_position,
                    missile_velocity=missile_velocity,
                    reference_position=reference_position,
                    elapsed_before=elapsed_before,
                    dt=dt,
                    noise=noise,
                    env_config=env_config,
                    wind_multiplier=float(launch.wind_strength_multiplier),
                    jitter_multiplier=float(launch.jitter_strength_multiplier),
                )
                missile_tau = self._tau_from_elapsed(elapsed, missile_duration)
                if missile_tau >= 1.0 or self._reached_target(missile_position, target_position, env_config.intercept_radius * 1.5):
                    missile_alive = False
                    if outcome == "running":
                        outcome = "missile_impact"
                if missile_position[2] <= 0.0 and missile_velocity[2] < 0.0:
                    missile_position[2] = 0.0
                    missile_alive = False
                    if outcome == "running":
                        outcome = "missile_impact"

            if missile_alive and not interceptor_initialized and elapsed >= launch.launch_delay:
                interceptor_position, interceptor_velocity = self._sample_interceptor_state(
                    rng=rng,
                    missile_position=missile_position,
                    target_position=target_position,
                    spawn_mode=launch.interceptor_spawn_mode,
                    launch_distance=float(launch.interceptor_launch_distance),
                    battery_radius=float(launch.interceptor_battery_radius),
                    altitude_min=float(launch.interceptor_battery_altitude_min),
                    altitude_max=float(launch.interceptor_battery_altitude_max),
                    initial_speed=float(launch.interceptor_initial_speed),
                    env_config=env_config,
                )
                interceptor_initialized = True
                interceptor_active = True
                interceptor_episode_elapsed = 0.0
                recorded_launch_position = tuple(float(x) for x in interceptor_position)
                hidden_state = self.model.initial_state(1, self.device)
                episode_start = torch.ones(1, dtype=torch.float32, device=self.device)
            elif interceptor_initialized:
                interceptor_episode_elapsed += dt

            if not interceptor_initialized:
                wind_interceptor = np.zeros(3, dtype=np.float64)
                distance = 0.0
            else:
                wind_interceptor = self._wind(noise, interceptor_position, time_value, env_config) * float(launch.wind_strength_multiplier)
                distance = float(np.linalg.norm(missile_position - interceptor_position))

            action = np.zeros(3, dtype=np.float64)
            if interceptor_active and missile_alive:
                missile_tau_full = self._tau_from_elapsed(elapsed, missile_duration)
                relative_position = missile_position - interceptor_position
                relative_velocity = missile_velocity - interceptor_velocity
                rel_distance = float(np.linalg.norm(relative_position))
                closing_speed = -float(np.dot(relative_position, relative_velocity)) / max(rel_distance, 1e-6)
                episode_fraction = float(np.clip(interceptor_episode_elapsed / env_config.max_episode_time, 0.0, 1.0))
                time_remaining = max(
                    0.0,
                    min(env_config.max_episode_time - interceptor_episode_elapsed, missile_duration - elapsed),
                )
                expert_action_hint = self._expert_action(
                    missile_position=missile_position,
                    missile_velocity=missile_velocity,
                    interceptor_position=interceptor_position,
                    interceptor_velocity=interceptor_velocity,
                )
                lookahead_seconds = float(getattr(env_config, "lookahead_seconds", 0.7))
                lookahead_missile = missile_position + missile_velocity * lookahead_seconds + 0.5 * gravity_vector * (lookahead_seconds ** 2)
                lookahead_interceptor = interceptor_position + interceptor_velocity * lookahead_seconds
                lookahead_relative = lookahead_missile - lookahead_interceptor
                time_to_go_clip = float(getattr(env_config, "time_to_go_clip", 4.0))
                time_to_go = float(np.clip(rel_distance / max(closing_speed + 50.0, 30.0), 0.05, time_to_go_clip))
                altitude_diff = float(missile_position[2] - interceptor_position[2])

                observation_parts = [
                    relative_position / env_config.distance_scale,
                    relative_velocity / env_config.velocity_scale,
                    missile_position / env_config.world_limit,
                    missile_velocity / env_config.velocity_scale,
                    interceptor_position / env_config.world_limit,
                    interceptor_velocity / env_config.velocity_scale,
                    np.array([rel_distance / env_config.distance_scale]),
                    np.array([closing_speed / env_config.velocity_scale]),
                    np.array([missile_tau_full]),
                    np.array([episode_fraction]),
                    np.array([time_remaining / env_config.max_episode_time]),
                    last_action,
                    wind_interceptor / max(env_config.wind_strength, 1e-8),
                    wind_missile / max(env_config.wind_strength, 1e-8),
                    np.array([env_config.gravity / env_config.acceleration_scale]),
                ]
                if self.observation_dim >= 41:
                    observation_parts.extend([
                        expert_action_hint,
                        lookahead_relative / env_config.distance_scale,
                        np.array([time_to_go / time_to_go_clip]),
                        np.array([altitude_diff / env_config.distance_scale]),
                    ])
                observation = np.concatenate(observation_parts).astype(np.float32)

                obs_tensor = torch.as_tensor(observation.reshape(1, -1), dtype=torch.float32, device=self.device)
                normalized_obs = self.obs_rms.normalize(obs_tensor)
                policy_mode = launch.policy_mode
                policy_action = np.zeros(3, dtype=np.float64)
                xgb_action: np.ndarray | None = None
                if policy_mode in {"ensemble", "policy"}:
                    with torch.no_grad():
                        residual_tensor, _, hidden_state = self.model.deterministic_action(
                            normalized_obs,
                            hidden_state=hidden_state,
                            episode_starts=episode_start,
                        )
                    episode_start[0] = 0.0
                    residual_action = residual_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64)
                    if self.use_guidance_prior:
                        policy_action = self._normalize_action(expert_action_hint + self.residual_action_scale * residual_action)
                    else:
                        policy_action = self._normalize_action(residual_action)

                if policy_mode in {"ensemble", "xgboost"} and self.xgboost_teacher is not None:
                    xgb_raw = self.xgboost_teacher.predict(observation.reshape(1, -1).astype(np.float32))[0]
                    xgb_action = normalize_numpy_actions(xgb_raw.reshape(1, -1))[0].astype(np.float64)

                if policy_mode == "expert":
                    action = expert_action_hint
                elif policy_mode == "xgboost" and xgb_action is not None:
                    action = xgb_action
                elif policy_mode == "ensemble" and xgb_action is not None:
                    blended = (1.0 - self.xgboost_teacher_weight) * policy_action + self.xgboost_teacher_weight * xgb_action
                    action = self._normalize_action(blended)
                else:
                    action = policy_action

                last_action = action.copy()

                thrust = action * env_config.interceptor_max_accel
                drag_interceptor = self._drag(interceptor_velocity, wind_interceptor, env_config.interceptor_drag)
                interceptor_acceleration = thrust + gravity_vector + drag_interceptor + wind_interceptor * 0.34
                interceptor_velocity = interceptor_velocity + interceptor_acceleration * dt
                interceptor_position = interceptor_position + interceptor_velocity * dt
                if interceptor_position[2] <= 0.0 and interceptor_velocity[2] < 0.0:
                    interceptor_position[2] = 0.0
                    interceptor_active = False
                    if outcome == "running":
                        outcome = "interceptor_crash"

            if interceptor_initialized:
                distance = float(np.linalg.norm(missile_position - interceptor_position))
                min_distance = min(min_distance, distance)

            frames.append(self._make_frame(
                time_value,
                missile_position,
                missile_velocity,
                missile_alive,
                missile_alive,
                interceptor_position,
                interceptor_velocity,
                interceptor_active,
                action,
                wind_missile,
                wind_interceptor,
                distance if interceptor_initialized else 0.0,
                self._tau_from_elapsed(elapsed, missile_duration),
            ))

            if missile_alive and interceptor_active and distance <= intercept_radius:
                outcome = "intercept"
                intercept_time = time_value
                break

            if elapsed >= launch.max_simulation_time:
                if outcome == "running":
                    outcome = "time_limit"
                break

            if not missile_alive:
                break

        return SimulationResult(
            frames=frames,
            outcome=outcome,
            min_distance=float(min_distance if np.isfinite(min_distance) else 0.0),
            intercept_time=float(intercept_time),
            intercept_radius=float(intercept_radius),
            target_position=tuple(float(x) for x in target_position),
            interceptor_launch_position=recorded_launch_position if recorded_launch_position is not None else (0.0, 0.0, 0.0),
            missile_initial_velocity=tuple(float(x) for x in missile_initial_velocity),
            missile_path_points=missile_path_points,
            missile_duration=float(missile_duration),
            config={
                "launch_position": list(map(float, launch.launch_position)),
                "target_position": list(map(float, launch.target_position)),
                "launch_azimuth_deg": launch.launch_azimuth_deg,
                "launch_elevation_deg": launch.launch_elevation_deg,
                "thrust_speed": launch.thrust_speed,
                "wind_strength_multiplier": launch.wind_strength_multiplier,
                "jitter_strength_multiplier": launch.jitter_strength_multiplier,
                "noise_seed": launch.noise_seed,
                "launch_delay": launch.launch_delay,
                "interceptor_spawn_mode": launch.interceptor_spawn_mode,
                "interceptor_battery_radius": launch.interceptor_battery_radius,
                "interceptor_battery_altitude_min": launch.interceptor_battery_altitude_min,
                "interceptor_battery_altitude_max": launch.interceptor_battery_altitude_max,
                "interceptor_launch_distance": launch.interceptor_launch_distance,
                "interceptor_initial_speed": launch.interceptor_initial_speed,
                "timestep": launch.timestep,
                "policy_mode": launch.policy_mode,
                "xgboost_teacher_weight": self.xgboost_teacher_weight,
                "xgboost_available": self.xgboost_teacher is not None,
                "architecture": self.architecture,
            },
        )

    def _build_missile_path(
        self,
        launch_position: np.ndarray,
        target_position: np.ndarray,
        launch_direction: np.ndarray,
        thrust_speed: float,
        noise_seed: int,
    ) -> dict:
        env_config = self.env_config
        rng = np.random.default_rng(noise_seed + 31415)
        chord = target_position - launch_position
        chord_distance = float(np.linalg.norm(chord))
        chord_distance = max(chord_distance, 50.0)
        chord_unit = chord / chord_distance
        ldir_norm = float(np.linalg.norm(launch_direction))
        if ldir_norm < 1e-6:
            launch_direction_unit = chord_unit.copy()
        else:
            launch_direction_unit = launch_direction / ldir_norm

        elevation_apex = max(launch_direction_unit[2], 0.05)
        ballistic_apex = max(120.0, min(env_config.world_limit * 0.55, 0.5 * elevation_apex * thrust_speed * thrust_speed / max(env_config.gravity, 1.0)))
        ballistic_apex = max(ballistic_apex, max(launch_position[2], target_position[2]) + 80.0)

        control_1 = launch_position + launch_direction_unit * (chord_distance * 0.42)
        control_1[2] = max(control_1[2], launch_position[2] + ballistic_apex * 0.45)
        approach_unit = chord_unit.copy()
        approach_unit[2] = -0.45
        approach_norm = float(np.linalg.norm(approach_unit))
        approach_unit = approach_unit / max(approach_norm, 1e-6)
        control_2 = target_position - approach_unit * (chord_distance * 0.32)
        control_2[2] = max(control_2[2], target_position[2] + ballistic_apex * 0.35)

        sample_taus = np.linspace(0.0, 1.0, 24)
        sampled_points = []
        for tau in sample_taus:
            point = (
                ((1.0 - tau) ** 3) * launch_position
                + 3.0 * ((1.0 - tau) ** 2) * tau * control_1
                + 3.0 * (1.0 - tau) * (tau ** 2) * control_2
                + (tau ** 3) * target_position
            )
            sampled_points.append(point)
        arc_length = 0.0
        for i in range(1, len(sampled_points)):
            arc_length += float(np.linalg.norm(sampled_points[i] - sampled_points[i - 1]))
        arc_length = max(arc_length, chord_distance * 1.05)

        missile_duration = arc_length / max(thrust_speed, 30.0)
        missile_duration = float(np.clip(missile_duration, env_config.missile_duration_min, env_config.missile_duration_max))

        u, v = self._orthonormal_basis(chord_unit)
        path = {
            "kind": "bezier",
            "start": launch_position.copy(),
            "end": target_position.copy(),
            "control_1": control_1,
            "control_2": control_2,
            "direction_unit": chord_unit,
            "u": u,
            "v": v,
            "missile_duration": missile_duration,
            "noise_amplitude": float(rng.uniform(env_config.path_noise_amplitude_min, env_config.path_noise_amplitude_max)) * 0.55,
            "noise_frequency": float(rng.uniform(2.0, 5.5)),
            "noise_offset": rng.uniform(-500.0, 500.0, size=3),
        }
        return path

    def _missile_path_position(self, path: dict, tau: float) -> np.ndarray:
        tau = float(np.clip(tau, 0.0, 1.0))
        start = path["start"]
        end = path["end"]
        control_1 = path["control_1"]
        control_2 = path["control_2"]
        base = (
            ((1.0 - tau) ** 3) * start
            + 3.0 * ((1.0 - tau) ** 2) * tau * control_1
            + 3.0 * (1.0 - tau) * (tau ** 2) * control_2
            + (tau ** 3) * end
        )
        u = path["u"]
        v = path["v"]
        direction_unit = path["direction_unit"]
        offset = path["noise_offset"]
        frequency = path["noise_frequency"]
        envelope = float(np.sin(np.pi * tau))
        noise_x = self._fractal_noise(np.array([offset[0] + tau * frequency, offset[1], offset[2]], dtype=np.float64))
        noise_y = self._fractal_noise(np.array([offset[0], offset[1] + tau * frequency, offset[2] + 19.7], dtype=np.float64))
        noise_z = self._fractal_noise(np.array([offset[0] + 41.3, offset[1], offset[2] + tau * frequency], dtype=np.float64))
        noise_offset_vec = path["noise_amplitude"] * envelope * (u * noise_x + v * noise_y + direction_unit * 0.25 * noise_z)
        return base + noise_offset_vec

    def _missile_path_derivative(self, path: dict, tau: float) -> np.ndarray:
        epsilon = 1e-3
        left_tau = max(0.0, tau - epsilon)
        right_tau = min(1.0, tau + epsilon)
        if right_tau == left_tau:
            return path["end"] - path["start"]
        return (self._missile_path_position(path, right_tau) - self._missile_path_position(path, left_tau)) / (right_tau - left_tau)

    def _step_missile(
        self,
        path: dict,
        missile_position: np.ndarray,
        missile_velocity: np.ndarray,
        reference_position: np.ndarray,
        elapsed_before: float,
        dt: float,
        noise: PerlinNoise3D,
        env_config: EnvironmentConfig,
        wind_multiplier: float,
        jitter_multiplier: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gravity_vector = np.array([0.0, 0.0, -env_config.gravity], dtype=np.float64)
        missile_duration = path["missile_duration"]
        tau = self._tau_from_elapsed(elapsed_before, missile_duration)
        target_tau = self._tau_from_elapsed(elapsed_before + dt, missile_duration)
        desired_position = self._missile_path_position(path, target_tau)
        desired_velocity = (desired_position - reference_position) / dt
        tracking_accel = (
            env_config.missile_tracking_kp * (desired_position - missile_position)
            + env_config.missile_tracking_kd * (desired_velocity - missile_velocity)
        )
        tracking_accel = self._limit_vector(tracking_accel, env_config.missile_max_accel)
        wind = self._wind(noise, missile_position, elapsed_before, env_config) * wind_multiplier
        drag = self._drag(missile_velocity, wind, env_config.missile_drag)
        jitter = self._jitter(noise, tau, env_config.missile_noise_accel) * jitter_multiplier
        acceleration = tracking_accel + gravity_vector + drag + wind * 0.22 + jitter
        new_velocity = missile_velocity + acceleration * dt
        new_position = missile_position + new_velocity * dt
        return new_position, new_velocity, desired_position, wind

    def _sample_interceptor_state(
        self,
        rng: np.random.Generator,
        missile_position: np.ndarray,
        target_position: np.ndarray,
        spawn_mode: str,
        launch_distance: float,
        battery_radius: float,
        altitude_min: float,
        altitude_max: float,
        initial_speed: float,
        env_config: EnvironmentConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        world_limit = env_config.world_limit
        if spawn_mode == "near_target":
            position = self._sample_near_target(
                rng=rng,
                target_position=target_position,
                battery_radius=battery_radius,
                altitude_min=altitude_min,
                altitude_max=altitude_max,
                world_limit=world_limit,
            )
        else:
            position = self._sample_near_missile(
                rng=rng,
                missile_position=missile_position,
                launch_distance=launch_distance,
                world_limit=world_limit,
            )
        aim = missile_position - position
        aim[2] = max(aim[2], abs(missile_position[2] - position[2]) * 0.4 + 80.0)
        aim_norm = max(float(np.linalg.norm(aim)), 1e-6)
        aim_unit = aim / aim_norm
        velocity = aim_unit * initial_speed + rng.normal(0.0, 3.0, size=3)
        return position.astype(np.float64), velocity.astype(np.float64)

    def _sample_near_target(
        self,
        rng: np.random.Generator,
        target_position: np.ndarray,
        battery_radius: float,
        altitude_min: float,
        altitude_max: float,
        world_limit: float,
    ) -> np.ndarray:
        for _ in range(64):
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            radius_jitter = float(rng.uniform(0.85, 1.15))
            radius = max(40.0, battery_radius * radius_jitter)
            altitude = float(rng.uniform(max(15.0, altitude_min), max(altitude_min + 1.0, altitude_max)))
            candidate = np.array(
                [
                    target_position[0] + np.cos(angle) * radius,
                    target_position[1] + np.sin(angle) * radius,
                    altitude,
                ],
                dtype=np.float64,
            )
            if abs(candidate[0]) > world_limit * 0.88:
                continue
            if abs(candidate[1]) > world_limit * 0.88:
                continue
            return candidate
        offset_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        return np.array(
            [
                float(np.clip(target_position[0] + np.cos(offset_angle) * battery_radius, -world_limit * 0.85, world_limit * 0.85)),
                float(np.clip(target_position[1] + np.sin(offset_angle) * battery_radius, -world_limit * 0.85, world_limit * 0.85)),
                float(np.clip((altitude_min + altitude_max) * 0.5, 15.0, world_limit * 0.85)),
            ],
            dtype=np.float64,
        )

    def _sample_near_missile(
        self,
        rng: np.random.Generator,
        missile_position: np.ndarray,
        launch_distance: float,
        world_limit: float,
    ) -> np.ndarray:
        for _ in range(64):
            direction = rng.normal(0.0, 1.0, size=3)
            direction[2] = float(rng.uniform(-0.35, 0.55))
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            direction = direction / norm
            candidate = missile_position - direction * launch_distance
            candidate[2] = max(candidate[2], 30.0)
            if abs(candidate[0]) > world_limit * 0.85:
                continue
            if abs(candidate[1]) > world_limit * 0.85:
                continue
            if candidate[2] > world_limit * 0.85:
                continue
            return candidate
        offset = np.array([0.0, -launch_distance, -120.0], dtype=np.float64)
        position = missile_position + offset
        position[0] = float(np.clip(position[0], -world_limit * 0.82, world_limit * 0.82))
        position[1] = float(np.clip(position[1], -world_limit * 0.82, world_limit * 0.82))
        position[2] = float(np.clip(position[2], 25.0, world_limit * 0.82))
        return position

    def _make_frame(
        self,
        time_value: float,
        missile_position: np.ndarray,
        missile_velocity: np.ndarray,
        missile_alive: bool,
        missile_thrusting: bool,
        interceptor_position: np.ndarray,
        interceptor_velocity: np.ndarray,
        interceptor_active: bool,
        action: np.ndarray,
        wind_missile: np.ndarray,
        wind_interceptor: np.ndarray,
        distance: float,
        missile_tau: float,
    ) -> SimulationFrame:
        return SimulationFrame(
            time=float(time_value),
            missile_position=tuple(float(x) for x in missile_position),
            missile_velocity=tuple(float(x) for x in missile_velocity),
            missile_alive=bool(missile_alive),
            missile_thrusting=bool(missile_thrusting),
            interceptor_position=tuple(float(x) for x in interceptor_position),
            interceptor_velocity=tuple(float(x) for x in interceptor_velocity),
            interceptor_active=bool(interceptor_active),
            interceptor_action=tuple(float(x) for x in action),
            wind_missile=tuple(float(x) for x in wind_missile),
            wind_interceptor=tuple(float(x) for x in wind_interceptor),
            distance=float(distance),
            missile_tau=float(missile_tau),
        )

    def _wind(self, noise: PerlinNoise3D, position: np.ndarray, time_value: float, env_config: EnvironmentConfig) -> np.ndarray:
        scale = env_config.wind_space_scale
        t = time_value * env_config.wind_time_scale
        x = position[0] * scale
        y = position[1] * scale
        z = position[2] * scale
        return env_config.wind_strength * np.array([
            noise.fractal(x + 0.0, y + 31.4, z + t),
            noise.fractal(x + 17.2, y + t, z + 11.1),
            noise.fractal(x + t, y + 5.7, z + 29.9),
        ], dtype=np.float64)

    def _drag(self, velocity: np.ndarray, wind: np.ndarray, coefficient: float) -> np.ndarray:
        relative = velocity - wind
        speed = float(np.linalg.norm(relative))
        return -coefficient * speed * relative

    def _jitter(self, noise: PerlinNoise3D, tau: float, amplitude: float) -> np.ndarray:
        return amplitude * np.array([
            noise.fractal(13.0 + tau * 7.0, 3.0, 9.0),
            noise.fractal(5.0, 17.0 + tau * 9.0, 11.0),
            noise.fractal(7.0, 19.0, 23.0 + tau * 8.0),
        ], dtype=np.float64)

    def _fractal_noise(self, coords: np.ndarray) -> float:
        rng_value = float(np.sin(coords[0] * 1.173 + coords[1] * 1.917 + coords[2] * 1.431))
        rng_value += 0.55 * float(np.sin(coords[0] * 2.311 - coords[1] * 1.227 + coords[2] * 2.719))
        rng_value += 0.30 * float(np.cos(coords[0] * 4.113 + coords[1] * 3.313 - coords[2] * 2.181))
        return float(np.clip(rng_value / 1.85, -1.0, 1.0))

    def _orthonormal_basis(self, direction_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        candidate = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(candidate, direction_unit))) > 0.92:
            candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(direction_unit, candidate)
        u = u / max(float(np.linalg.norm(u)), 1e-8)
        v = np.cross(direction_unit, u)
        v = v / max(float(np.linalg.norm(v)), 1e-8)
        return u, v

    def _limit_vector(self, vector: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= limit:
            return vector
        return vector / max(norm, 1e-8) * limit

    def _expert_action(
        self,
        missile_position: np.ndarray,
        missile_velocity: np.ndarray,
        interceptor_position: np.ndarray,
        interceptor_velocity: np.ndarray,
    ) -> np.ndarray:
        env_config = self.env_config
        gravity_vec = np.array([0.0, 0.0, -env_config.gravity], dtype=np.float64)
        relative_position = missile_position - interceptor_position
        relative_velocity = missile_velocity - interceptor_velocity
        distance = max(float(np.linalg.norm(relative_position)), 1.0)
        speed = max(float(np.linalg.norm(interceptor_velocity)), 20.0)
        time_to_go = float(np.clip(distance / (speed + 190.0), 0.18, 4.0))
        zero_effort_miss = relative_position + relative_velocity * time_to_go + 0.5 * gravity_vec * (time_to_go ** 2)
        desired_accel = 3.4 * zero_effort_miss / (time_to_go ** 2) + 0.8 * relative_velocity / time_to_go - gravity_vec
        action = desired_accel / env_config.interceptor_max_accel
        return self._normalize_action(action)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0)
        norm = float(np.linalg.norm(action))
        if norm > 1.0:
            action = action / norm
        return action

    def _tau_from_elapsed(self, elapsed: float, missile_duration: float) -> float:
        return float(np.clip(elapsed / max(missile_duration, 1e-6), 0.0, 1.0))

    def _reached_target(self, missile_position: np.ndarray, target_position: np.ndarray, threshold: float) -> bool:
        return float(np.linalg.norm(missile_position - target_position)) <= max(threshold, 6.0)
