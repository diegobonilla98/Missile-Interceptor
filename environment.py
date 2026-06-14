from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class EnvironmentConfig:
    dt: float = 0.04
    world_limit: float = 1800.0
    min_start_end_distance: float = 850.0
    max_start_end_distance: float = 2200.0
    missile_duration_min: float = 9.0
    missile_duration_max: float = 19.0
    missile_tracking_kp: float = 18.0
    missile_tracking_kd: float = 7.5
    missile_max_accel: float = 160.0
    missile_drag: float = 0.00045
    missile_noise_accel: float = 4.0
    interceptor_max_accel: float = 260.0
    interceptor_drag: float = 0.00065
    interceptor_initial_speed: float = 15.0
    interceptor_launch_distance_min: float = 220.0
    interceptor_launch_distance_max: float = 780.0
    gravity: float = 9.81
    wind_strength: float = 28.0
    wind_time_scale: float = 0.13
    wind_space_scale: float = 0.004
    intercept_radius: float = 18.0
    max_episode_time: float = 22.0
    launch_delay_min: float = 0.35
    launch_delay_max: float = 5.0
    path_noise_amplitude_min: float = 8.0
    path_noise_amplitude_max: float = 85.0
    path_lateral_amplitude_min: float = 40.0
    path_lateral_amplitude_max: float = 310.0
    distance_scale: float = 1200.0
    velocity_scale: float = 380.0
    acceleration_scale: float = 260.0
    lookahead_seconds: float = 0.7
    time_to_go_clip: float = 4.0
    proximity_sigma: float = 130.0
    proximity_weight: float = 6.0
    near_miss_sigma: float = 35.0
    near_miss_weight: float = 8.0
    progress_weight: float = 0.18
    progress_normalizer: float = 200.0
    closing_weight: float = 0.06
    closing_bias: float = 60.0
    alignment_weight: float = 0.18
    miss_distance_weight: float = 0.55
    control_penalty: float = 0.010
    path_length_penalty: float = 22.0
    energy_penalty: float = 1.6
    action_smoothness_penalty: float = 0.10
    distance_integral_penalty: float = 6.0
    late_success_penalty: float = 110.0
    efficient_success_bonus: float = 220.0
    time_penalty: float = 0.012
    success_reward: float = 1100.0
    miss_penalty: float = 320.0
    out_of_bounds_penalty: float = 460.0
    stagnation_window_steps: int = 20
    stagnation_penalty_weight: float = 0.04
    curriculum_start_intercept_radius: float = 110.0
    curriculum_start_launch_distance_min: float = 60.0
    curriculum_start_launch_distance_max: float = 280.0
    curriculum_start_wind_fraction: float = 0.10
    curriculum_start_path_noise_fraction: float = 0.10
    curriculum_start_lateral_fraction: float = 0.20
    curriculum_start_missile_speed_fraction: float = 0.55
    use_fast_vector_noise: bool = True


class PerlinNoise3D:
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(256)
        self.permutation = np.concatenate([permutation, permutation]).astype(np.int32)

    def fade(self, value: np.ndarray) -> np.ndarray:
        return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)

    def lerp(self, amount: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left + amount * (right - left)

    def grad(self, hashed: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        h = hashed & 15
        u = np.where(h < 8, x, y)
        v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
        return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)

    def noise(self, x: float, y: float, z: float) -> float:
        xf = np.floor(x).astype(np.int32) & 255
        yf = np.floor(y).astype(np.int32) & 255
        zf = np.floor(z).astype(np.int32) & 255

        x_rel = x - np.floor(x)
        y_rel = y - np.floor(y)
        z_rel = z - np.floor(z)

        u = self.fade(x_rel)
        v = self.fade(y_rel)
        w = self.fade(z_rel)

        p = self.permutation
        aaa = p[p[p[xf] + yf] + zf]
        aba = p[p[p[xf] + yf + 1] + zf]
        aab = p[p[p[xf] + yf] + zf + 1]
        abb = p[p[p[xf] + yf + 1] + zf + 1]
        baa = p[p[p[xf + 1] + yf] + zf]
        bba = p[p[p[xf + 1] + yf + 1] + zf]
        bab = p[p[p[xf + 1] + yf] + zf + 1]
        bbb = p[p[p[xf + 1] + yf + 1] + zf + 1]

        x1 = self.lerp(
            u,
            self.grad(aaa, x_rel, y_rel, z_rel),
            self.grad(baa, x_rel - 1.0, y_rel, z_rel),
        )
        x2 = self.lerp(
            u,
            self.grad(aba, x_rel, y_rel - 1.0, z_rel),
            self.grad(bba, x_rel - 1.0, y_rel - 1.0, z_rel),
        )
        y1 = self.lerp(v, x1, x2)

        x3 = self.lerp(
            u,
            self.grad(aab, x_rel, y_rel, z_rel - 1.0),
            self.grad(bab, x_rel - 1.0, y_rel, z_rel - 1.0),
        )
        x4 = self.lerp(
            u,
            self.grad(abb, x_rel, y_rel - 1.0, z_rel - 1.0),
            self.grad(bbb, x_rel - 1.0, y_rel - 1.0, z_rel - 1.0),
        )
        y2 = self.lerp(v, x3, x4)

        return float(np.clip(self.lerp(w, y1, y2), -1.0, 1.0))

    def fractal(self, x: float, y: float, z: float, octaves: int = 4) -> float:
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        norm = 0.0
        for _ in range(octaves):
            value += amplitude * self.noise(x * frequency, y * frequency, z * frequency)
            norm += amplitude
            amplitude *= 0.5
            frequency *= 2.03
        return value / max(norm, 1e-8)


class ProjectileInterceptEnv:
    action_dim = 3
    observation_dim = 41

    def __init__(self, config: EnvironmentConfig | None = None, seed: int | None = None, record_history: bool = True):
        self.config = config if config is not None else EnvironmentConfig()
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed if seed is not None else self.rng.integers(1, 2**31 - 1))
        self.noise = PerlinNoise3D(self.seed)
        self.record_history = record_history
        self.gravity_vector = np.array([0.0, 0.0, -self.config.gravity], dtype=np.float64)
        self.reset()

    def reset(self) -> np.ndarray:
        self.path = self._sample_path()
        self.missile_duration = float(self.rng.uniform(self.config.missile_duration_min, self.config.missile_duration_max))
        self.missile_elapsed = 0.0
        self.episode_elapsed = 0.0
        self.steps = 0
        self.done = False
        self.termination_reason = ""
        self.last_action = np.zeros(3, dtype=np.float64)
        self.min_distance = np.inf
        self.interceptor_path_length = 0.0
        self.action_energy = 0.0
        self.action_smoothness_energy = 0.0
        self.distance_integral = 0.0

        self.missile_position = self.path_position(0.0).copy()
        self.missile_velocity = self.path_derivative(0.0) / self.missile_duration

        launch_delay = float(self.rng.uniform(self.config.launch_delay_min, min(self.config.launch_delay_max, self.missile_duration * 0.55)))
        prelaunch_steps = max(1, int(launch_delay / self.config.dt))
        for _ in range(prelaunch_steps):
            self._advance_missile(self.config.dt)
        self.launch_delay = self.missile_elapsed
        self.launch_tau = self.missile_tau

        self.interceptor_position, self.interceptor_velocity = self._sample_interceptor_state()
        self.prev_distance = self._distance()
        self.min_distance = self.prev_distance
        self.initial_distance = self.prev_distance

        self.history = {
            "missile": [self.missile_position.copy()],
            "interceptor": [self.interceptor_position.copy()],
            "distance": [float(self.prev_distance)],
            "reward": [],
            "action": [],
            "path_tau": [float(self.missile_tau)],
        }

        return self._observation()

    @property
    def missile_tau(self) -> float:
        return float(np.clip(self.missile_elapsed / self.missile_duration, 0.0, 1.0))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)
        action_norm = np.linalg.norm(action)
        if action_norm > 1.0:
            action = action / action_norm

        previous_action = self.last_action.copy()
        previous_interceptor_position = self.interceptor_position.copy()
        self.last_action = action.copy()
        previous_distance = self.prev_distance
        previous_relative_position = self.missile_position - self.interceptor_position

        self._advance_missile(self.config.dt)
        self._advance_interceptor(action, self.config.dt)

        self.steps += 1
        self.episode_elapsed += self.config.dt

        distance = self._distance()
        segment_distance = self._segment_relative_distance(previous_relative_position, self.missile_position - self.interceptor_position)
        interceptor_step_distance = float(np.linalg.norm(self.interceptor_position - previous_interceptor_position))
        action_delta = action - previous_action
        self.interceptor_path_length += interceptor_step_distance
        self.action_energy += float(np.dot(action, action)) * self.config.dt
        self.action_smoothness_energy += float(np.dot(action_delta, action_delta))
        self.distance_integral += distance * self.config.dt
        self.prev_distance = distance
        self.min_distance = min(self.min_distance, segment_distance)

        reward = self._reward(previous_distance, distance, action, interceptor_step_distance, action_delta)
        done, terminal_reward, reason = self._termination(segment_distance)
        reward += terminal_reward
        self.done = done
        self.termination_reason = reason

        if self.record_history:
            self.history["missile"].append(self.missile_position.copy())
            self.history["interceptor"].append(self.interceptor_position.copy())
            self.history["distance"].append(float(distance))
            self.history["reward"].append(float(reward))
            self.history["action"].append(action.copy())
            self.history["path_tau"].append(float(self.missile_tau))

        info = {
            "distance": float(distance),
            "min_distance": float(self.min_distance),
            "missile_tau": float(self.missile_tau),
            "episode_elapsed": float(self.episode_elapsed),
            "success": bool(reason == "intercept"),
            "miss": bool(reason in {"missile_finished", "time_limit"}),
            "out_of_bounds": bool(reason == "out_of_bounds"),
            "termination_reason": reason,
            "path_kind": self.path["kind"],
            "launch_delay": float(self.launch_delay),
            "interceptor_path_length": float(self.interceptor_path_length),
            "action_energy": float(self.action_energy),
            "distance_integral": float(self.distance_integral),
        }
        return self._observation(), float(reward), bool(done), info

    def expert_action(self) -> np.ndarray:
        relative_position = self.missile_position - self.interceptor_position
        relative_velocity = self.missile_velocity - self.interceptor_velocity
        distance = max(float(np.linalg.norm(relative_position)), 1.0)
        speed = max(float(np.linalg.norm(self.interceptor_velocity)), 20.0)
        time_to_go = float(np.clip(distance / (speed + 190.0), 0.18, 4.0))
        zero_effort_miss = relative_position + relative_velocity * time_to_go + 0.5 * self.gravity_vector * (time_to_go**2)
        desired_accel = 3.4 * zero_effort_miss / (time_to_go**2) + 0.8 * relative_velocity / time_to_go - self.gravity_vector
        action = desired_accel / self.config.interceptor_max_accel
        action = np.clip(action, -1.0, 1.0)
        action_norm = np.linalg.norm(action)
        if action_norm > 1.0:
            action = action / action_norm
        return action.astype(np.float64)

    def path_position(self, tau: float) -> np.ndarray:
        tau = float(np.clip(tau, 0.0, 1.0))
        start = self.path["start"]
        end = self.path["end"]
        direction = end - start
        u = self.path["u"]
        v = self.path["v"]
        kind = self.path["kind"]

        if kind == "straight":
            base = start + direction * tau
        elif kind == "bezier":
            c1 = self.path["control_1"]
            c2 = self.path["control_2"]
            base = ((1.0 - tau) ** 3) * start + 3.0 * ((1.0 - tau) ** 2) * tau * c1 + 3.0 * (1.0 - tau) * (tau**2) * c2 + (tau**3) * end
        elif kind == "exponential":
            gain = self.path["gain"]
            eased = (np.exp(gain * tau) - 1.0) / (np.exp(gain) - 1.0)
            base = start + direction * eased
            base += u * self.path["amp_1"] * np.sin(np.pi * tau)
            base += v * self.path["amp_2"] * tau * (1.0 - tau) * 4.0
        elif kind == "s_curve":
            smooth = tau * tau * (3.0 - 2.0 * tau)
            base = start + direction * smooth
            base += u * self.path["amp_1"] * np.sin(np.pi * tau + self.path["phase"])
            base += v * self.path["amp_2"] * np.sin(2.0 * np.pi * tau + self.path["phase"]) * np.sin(np.pi * tau)
        else:
            turns = self.path["turns"]
            radius = self.path["amp_1"] * np.sin(np.pi * tau)
            angle = 2.0 * np.pi * turns * tau + self.path["phase"]
            base = start + direction * tau + radius * (u * np.cos(angle) + v * np.sin(angle))

        noise_x = self.noise.fractal(self.path["noise_offset"][0] + tau * self.path["noise_frequency"], self.path["noise_offset"][1], self.path["noise_offset"][2])
        noise_y = self.noise.fractal(self.path["noise_offset"][0], self.path["noise_offset"][1] + tau * self.path["noise_frequency"], self.path["noise_offset"][2] + 19.7)
        noise_z = self.noise.fractal(self.path["noise_offset"][0] + 41.3, self.path["noise_offset"][1], self.path["noise_offset"][2] + tau * self.path["noise_frequency"])
        envelope = np.sin(np.pi * tau)
        noise = self.path["noise_amplitude"] * envelope * (u * noise_x + v * noise_y + self.path["direction_unit"] * 0.25 * noise_z)
        return base + noise

    def path_derivative(self, tau: float) -> np.ndarray:
        epsilon = 1e-3
        left_tau = max(0.0, tau - epsilon)
        right_tau = min(1.0, tau + epsilon)
        if right_tau == left_tau:
            return self.path["end"] - self.path["start"]
        return (self.path_position(right_tau) - self.path_position(left_tau)) / (right_tau - left_tau)

    def plot_trajectory(self, save_path: str | Path | None = None, show: bool = True, include_reference: bool = True, title: str | None = None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        if include_reference:
            taus = np.linspace(0.0, 1.0, 320)
            reference = np.array([self.path_position(tau) for tau in taus])
            ax.plot(reference[:, 0], reference[:, 1], reference[:, 2], color="black", alpha=0.35, linewidth=1.4, label="noisy reference path")

        missile = np.array(self.history["missile"])
        interceptor = np.array(self.history["interceptor"])
        ax.plot(missile[:, 0], missile[:, 1], missile[:, 2], color="#d62728", linewidth=2.0, label="missile")
        ax.plot(interceptor[:, 0], interceptor[:, 1], interceptor[:, 2], color="#1f77b4", linewidth=2.0, label="interceptor")
        ax.scatter(missile[0, 0], missile[0, 1], missile[0, 2], color="#d62728", s=45)
        ax.scatter(interceptor[0, 0], interceptor[0, 1], interceptor[0, 2], color="#1f77b4", s=45)
        ax.scatter(missile[-1, 0], missile[-1, 1], missile[-1, 2], color="#d62728", marker="x", s=75)
        ax.scatter(interceptor[-1, 0], interceptor[-1, 1], interceptor[-1, 2], color="#1f77b4", marker="x", s=75)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title if title is not None else f"{self.path['kind']} path, min distance {self.min_distance:.2f}")
        ax.legend(loc="upper right")
        self._set_equal_axes(ax, np.concatenate([missile, interceptor], axis=0))
        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=160)
        if show:
            plt.show()
        return fig, ax

    def _sample_path(self) -> dict:
        start, end = self._sample_far_points()
        direction = end - start
        distance = np.linalg.norm(direction)
        direction_unit = direction / max(distance, 1e-8)
        u, v = self._orthonormal_basis(direction_unit)
        amp_min = self.config.path_lateral_amplitude_min
        amp_max = self.config.path_lateral_amplitude_max
        noise_min = self.config.path_noise_amplitude_min
        noise_max = self.config.path_noise_amplitude_max
        kind = str(self.rng.choice(["straight", "bezier", "exponential", "s_curve", "corkscrew"]))
        amp_1 = float(self.rng.uniform(amp_min, amp_max))
        amp_2 = float(self.rng.uniform(amp_min, amp_max))
        path = {
            "kind": kind,
            "start": start,
            "end": end,
            "direction_unit": direction_unit,
            "u": u,
            "v": v,
            "amp_1": amp_1,
            "amp_2": amp_2,
            "phase": float(self.rng.uniform(0.0, 2.0 * np.pi)),
            "gain": float(self.rng.uniform(1.15, 3.7)),
            "turns": float(self.rng.uniform(0.65, 2.35)),
            "noise_amplitude": float(self.rng.uniform(noise_min, noise_max)),
            "noise_frequency": float(self.rng.uniform(1.2, 7.5)),
            "noise_offset": self.rng.uniform(-500.0, 500.0, size=3),
        }
        path["control_1"] = start + direction * float(self.rng.uniform(0.18, 0.42)) + u * float(self.rng.uniform(-amp_1, amp_1)) + v * float(self.rng.uniform(-amp_2, amp_2))
        path["control_2"] = start + direction * float(self.rng.uniform(0.58, 0.88)) + u * float(self.rng.uniform(-amp_1, amp_1)) + v * float(self.rng.uniform(-amp_2, amp_2))
        return path

    def _sample_far_points(self) -> tuple[np.ndarray, np.ndarray]:
        limit = self.config.world_limit * 0.55
        altitude_min = 130.0
        altitude_max = self.config.world_limit * 0.62
        for _ in range(10_000):
            start = np.array([
                self.rng.uniform(-limit, limit),
                self.rng.uniform(-limit, limit),
                self.rng.uniform(altitude_min, altitude_max),
            ], dtype=np.float64)
            end = np.array([
                self.rng.uniform(-limit, limit),
                self.rng.uniform(-limit, limit),
                self.rng.uniform(altitude_min, altitude_max),
            ], dtype=np.float64)
            distance = np.linalg.norm(end - start)
            if self.config.min_start_end_distance <= distance <= self.config.max_start_end_distance:
                return start, end
        start = np.array([-limit, -limit * 0.35, altitude_max * 0.5], dtype=np.float64)
        end = np.array([limit, limit * 0.35, altitude_max * 0.7], dtype=np.float64)
        return start, end

    def _orthonormal_basis(self, direction_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        candidate = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(candidate, direction_unit))) > 0.92:
            candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(direction_unit, candidate)
        u = u / max(np.linalg.norm(u), 1e-8)
        v = np.cross(direction_unit, u)
        v = v / max(np.linalg.norm(v), 1e-8)
        return u, v

    def _sample_interceptor_state(self) -> tuple[np.ndarray, np.ndarray]:
        position = None
        for _ in range(128):
            rel_direction = self.rng.normal(0.0, 1.0, size=3)
            rel_direction[2] = self.rng.uniform(-0.35, 0.55)
            rel_direction = rel_direction / max(np.linalg.norm(rel_direction), 1e-8)
            distance = float(self.rng.uniform(self.config.interceptor_launch_distance_min, self.config.interceptor_launch_distance_max))
            candidate = self.missile_position - rel_direction * distance
            candidate[2] = max(15.0, candidate[2])
            if not self._out_of_bounds(candidate):
                position = candidate
                break
        if position is None:
            offset = np.array([0.0, -self.config.interceptor_launch_distance_min, -80.0], dtype=np.float64)
            position = self.missile_position + offset
            position[0] = np.clip(position[0], -self.config.world_limit * 0.82, self.config.world_limit * 0.82)
            position[1] = np.clip(position[1], -self.config.world_limit * 0.82, self.config.world_limit * 0.82)
            position[2] = np.clip(position[2], 20.0, self.config.world_limit * 0.82)
        initial_direction = self.missile_position - position
        initial_direction = initial_direction / max(np.linalg.norm(initial_direction), 1e-8)
        velocity = initial_direction * self.config.interceptor_initial_speed + self.rng.normal(0.0, 3.0, size=3)
        return position.astype(np.float64), velocity.astype(np.float64)

    def _advance_missile(self, dt: float) -> None:
        current_tau = self.missile_tau
        target_tau = min(1.0, current_tau + dt / self.missile_duration)
        desired_position = self.path_position(target_tau)
        desired_velocity = self.path_derivative(target_tau) / self.missile_duration
        tracking_accel = self.config.missile_tracking_kp * (desired_position - self.missile_position) + self.config.missile_tracking_kd * (desired_velocity - self.missile_velocity)
        tracking_accel = self._limit_vector(tracking_accel, self.config.missile_max_accel)
        wind = self._wind_acceleration(self.missile_position, getattr(self, "launch_delay", 0.0) + self.episode_elapsed)
        drag = self._drag_acceleration(self.missile_velocity, wind, self.config.missile_drag)
        noise_accel = self.config.missile_noise_accel * np.array([
            self.noise.fractal(13.0 + current_tau * 7.0, 3.0, 9.0),
            self.noise.fractal(5.0, 17.0 + current_tau * 9.0, 11.0),
            self.noise.fractal(7.0, 19.0, 23.0 + current_tau * 8.0),
        ])
        acceleration = tracking_accel + self.gravity_vector + drag + wind * 0.22 + noise_accel
        self.missile_velocity = self.missile_velocity + acceleration * dt
        self.missile_position = self.missile_position + self.missile_velocity * dt
        self.missile_elapsed = min(self.missile_duration, self.missile_elapsed + dt)

    def _advance_interceptor(self, action: np.ndarray, dt: float) -> None:
        thrust = action * self.config.interceptor_max_accel
        wind = self._wind_acceleration(self.interceptor_position, self.launch_delay + self.episode_elapsed)
        drag = self._drag_acceleration(self.interceptor_velocity, wind, self.config.interceptor_drag)
        acceleration = thrust + self.gravity_vector + drag + wind * 0.34
        self.interceptor_velocity = self.interceptor_velocity + acceleration * dt
        self.interceptor_position = self.interceptor_position + self.interceptor_velocity * dt

    def _wind_acceleration(self, position: np.ndarray, time_value: float) -> np.ndarray:
        scale = self.config.wind_space_scale
        t = time_value * self.config.wind_time_scale
        x = position[0] * scale
        y = position[1] * scale
        z = position[2] * scale
        wind = np.array([
            self.noise.fractal(x + 0.0, y + 31.4, z + t),
            self.noise.fractal(x + 17.2, y + t, z + 11.1),
            self.noise.fractal(x + t, y + 5.7, z + 29.9),
        ], dtype=np.float64)
        return wind * self.config.wind_strength

    def _drag_acceleration(self, velocity: np.ndarray, wind_acceleration: np.ndarray, coefficient: float) -> np.ndarray:
        relative_velocity = velocity - wind_acceleration
        speed = np.linalg.norm(relative_velocity)
        return -coefficient * speed * relative_velocity

    def _reward(self, previous_distance: float, distance: float, action: np.ndarray, interceptor_step_distance: float, action_delta: np.ndarray) -> float:
        relative_position = self.missile_position - self.interceptor_position
        relative_velocity = self.missile_velocity - self.interceptor_velocity
        closing_speed = -float(np.dot(relative_position, relative_velocity)) / max(distance, 1e-6)
        proximity = self.config.proximity_weight * float(np.exp(-(distance ** 2) / max(self.config.proximity_sigma ** 2, 1e-8)))
        near_miss = self.config.near_miss_weight * float(np.exp(-(distance ** 2) / max(self.config.near_miss_sigma ** 2, 1e-8)))
        progress = self.config.progress_weight * (previous_distance - distance) / max(self.config.progress_normalizer, 1.0)
        closing = self.config.closing_weight * (closing_speed - self.config.closing_bias) / max(self.config.velocity_scale, 1.0)
        rel_speed = max(float(np.linalg.norm(relative_velocity)), 1e-6)
        alignment_cos = -float(np.dot(relative_position, relative_velocity)) / (max(distance, 1e-6) * rel_speed)
        alignment_bonus = self.config.alignment_weight * max(0.0, alignment_cos)
        miss_distance = -self.config.miss_distance_weight * (distance / self.config.distance_scale)
        control = -self.config.control_penalty * float(np.dot(action, action))
        path_cost = -self.config.path_length_penalty * (interceptor_step_distance / self.config.distance_scale)
        energy_cost = -self.config.energy_penalty * float(np.dot(action, action)) * self.config.dt
        smoothness_cost = -self.config.action_smoothness_penalty * float(np.dot(action_delta, action_delta))
        distance_integral_cost = -self.config.distance_integral_penalty * (distance / self.config.distance_scale) * self.config.dt
        return (
            proximity
            + near_miss
            + progress
            + closing
            + alignment_bonus
            + miss_distance
            + control
            + path_cost
            + energy_cost
            + smoothness_cost
            + distance_integral_cost
            - self.config.time_penalty
        )

    def _termination(self, distance: float) -> tuple[bool, float, str]:
        if distance <= self.config.intercept_radius:
            efficiency = self.initial_distance / max(self.interceptor_path_length, self.initial_distance, 1.0)
            reward = self.config.success_reward + self.config.efficient_success_bonus * efficiency + 25.0 * (1.0 - self.missile_tau)
            reward -= self.config.late_success_penalty * (self.episode_elapsed / self.config.max_episode_time)
            return True, reward, "intercept"
        if self._out_of_bounds(self.interceptor_position) or self._out_of_bounds(self.missile_position):
            return True, -self.config.out_of_bounds_penalty, "out_of_bounds"
        if self.missile_tau >= 1.0:
            penalty = -self.config.miss_penalty - 80.0 * min(1.0, distance / self.config.distance_scale)
            return True, penalty, "missile_finished"
        if self.episode_elapsed >= self.config.max_episode_time:
            penalty = -self.config.miss_penalty - 80.0 * min(1.0, distance / self.config.distance_scale)
            return True, penalty, "time_limit"
        return False, 0.0, ""

    def _out_of_bounds(self, position: np.ndarray) -> bool:
        limit = self.config.world_limit
        return bool(np.any(np.abs(position[:2]) > limit) or position[2] < -120.0 or position[2] > limit)

    def _observation(self) -> np.ndarray:
        relative_position = self.missile_position - self.interceptor_position
        relative_velocity = self.missile_velocity - self.interceptor_velocity
        distance = np.linalg.norm(relative_position)
        closing_speed = -float(np.dot(relative_position, relative_velocity)) / max(distance, 1e-6)
        time_remaining = max(0.0, min(self.config.max_episode_time - self.episode_elapsed, self.missile_duration - self.missile_elapsed))
        wind_interceptor = self._wind_acceleration(self.interceptor_position, self.launch_delay + self.episode_elapsed)
        wind_missile = self._wind_acceleration(self.missile_position, self.launch_delay + self.episode_elapsed)
        expert_hint = self.expert_action()
        lookahead = self.config.lookahead_seconds
        lookahead_missile_position = self.missile_position + self.missile_velocity * lookahead + 0.5 * self.gravity_vector * (lookahead ** 2)
        lookahead_relative = lookahead_missile_position - (self.interceptor_position + self.interceptor_velocity * lookahead)
        time_to_go = float(np.clip(distance / max(closing_speed + 50.0, 30.0), 0.05, self.config.time_to_go_clip))
        altitude_diff = float(self.missile_position[2] - self.interceptor_position[2])

        observation = np.concatenate([
            relative_position / self.config.distance_scale,
            relative_velocity / self.config.velocity_scale,
            self.missile_position / self.config.world_limit,
            self.missile_velocity / self.config.velocity_scale,
            self.interceptor_position / self.config.world_limit,
            self.interceptor_velocity / self.config.velocity_scale,
            np.array([distance / self.config.distance_scale], dtype=np.float64),
            np.array([closing_speed / self.config.velocity_scale], dtype=np.float64),
            np.array([self.missile_tau], dtype=np.float64),
            np.array([self.episode_elapsed / self.config.max_episode_time], dtype=np.float64),
            np.array([time_remaining / self.config.max_episode_time], dtype=np.float64),
            self.last_action,
            wind_interceptor / max(self.config.wind_strength, 1e-8),
            wind_missile / max(self.config.wind_strength, 1e-8),
            np.array([self.config.gravity / self.config.acceleration_scale], dtype=np.float64),
            expert_hint,
            lookahead_relative / self.config.distance_scale,
            np.array([time_to_go / self.config.time_to_go_clip], dtype=np.float64),
            np.array([altitude_diff / self.config.distance_scale], dtype=np.float64),
        ])
        return observation.astype(np.float32)

    def _distance(self) -> float:
        return float(np.linalg.norm(self.missile_position - self.interceptor_position))

    def _segment_relative_distance(self, previous_relative_position: np.ndarray, current_relative_position: np.ndarray) -> float:
        delta = current_relative_position - previous_relative_position
        denom = float(np.dot(delta, delta))
        if denom <= 1e-12:
            return float(np.linalg.norm(current_relative_position))
        amount = float(np.clip(-np.dot(previous_relative_position, delta) / denom, 0.0, 1.0))
        closest = previous_relative_position + delta * amount
        return float(np.linalg.norm(closest))

    def _limit_vector(self, vector: np.ndarray, limit: float) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm <= limit:
            return vector
        return vector / max(norm, 1e-8) * limit

    def _set_equal_axes(self, ax, points: np.ndarray) -> None:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        centers = (mins + maxs) * 0.5
        radius = max(float(np.max(maxs - mins)) * 0.55, 1.0)
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(max(-50.0, centers[2] - radius), centers[2] + radius)


class TorchProjectileInterceptVecEnv:
    action_dim = 3
    observation_dim = 41

    def __init__(self, num_envs: int, config: EnvironmentConfig | None = None, seed: int = 0, device: torch.device | str = "cuda"):
        self.num_envs = int(num_envs)
        self.config = config if config is not None else EnvironmentConfig()
        self.device = torch.device(device)
        self.seed = int(seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)
        self.gravity_vector = torch.tensor([0.0, 0.0, -self.config.gravity], dtype=torch.float32, device=self.device)
        self.curriculum_progress = 1.0
        self.difficulty_level = 0.0
        self.current_intercept_radius = self.config.intercept_radius
        self.current_launch_distance_min = self.config.interceptor_launch_distance_min
        self.current_launch_distance_max = self.config.interceptor_launch_distance_max
        self.current_launch_delay_max = self.config.launch_delay_max
        self.current_wind_strength = self.config.wind_strength
        self.current_path_noise_fraction = 1.0
        self.current_lateral_fraction = 1.0
        self.current_interceptor_max_accel = self.config.interceptor_max_accel
        self.current_interceptor_initial_speed = self.config.interceptor_initial_speed
        self.current_missile_speed_fraction = 1.0
        self.current_missile_duration_min = self.config.missile_duration_min
        self.current_missile_duration_max = self.config.missile_duration_max
        self._allocate()
        self.reset()

    def set_curriculum(self, progress: float) -> None:
        progress = float(np.clip(progress, 0.0, 1.0))
        self.curriculum_progress = progress * progress * (3.0 - 2.0 * progress)
        self._update_runtime_parameters()

    def set_difficulty(self, level: float) -> None:
        self.difficulty_level = float(max(0.0, level))
        self._update_runtime_parameters()

    def _update_runtime_parameters(self) -> None:
        smooth = self.curriculum_progress
        level = self.difficulty_level
        base_radius = self.config.curriculum_start_intercept_radius + (self.config.intercept_radius - self.config.curriculum_start_intercept_radius) * smooth
        base_launch_min = self.config.curriculum_start_launch_distance_min + (self.config.interceptor_launch_distance_min - self.config.curriculum_start_launch_distance_min) * smooth
        base_launch_max = self.config.curriculum_start_launch_distance_max + (self.config.interceptor_launch_distance_max - self.config.curriculum_start_launch_distance_max) * smooth
        base_wind = self.config.wind_strength * (self.config.curriculum_start_wind_fraction + (1.0 - self.config.curriculum_start_wind_fraction) * smooth)
        base_noise = self.config.curriculum_start_path_noise_fraction + (1.0 - self.config.curriculum_start_path_noise_fraction) * smooth
        base_lateral = self.config.curriculum_start_lateral_fraction + (1.0 - self.config.curriculum_start_lateral_fraction) * smooth
        speed_fraction = self.config.curriculum_start_missile_speed_fraction + (1.0 - self.config.curriculum_start_missile_speed_fraction) * smooth
        self.current_intercept_radius = max(2.5, base_radius * (0.92**level))
        self.current_launch_distance_min = base_launch_min * (1.0 + 0.045 * level)
        self.current_launch_distance_max = max(self.current_launch_distance_min + 80.0, base_launch_max * (1.0 + 0.09 * level))
        self.current_launch_delay_max = self.config.launch_delay_max * (1.0 + 0.08 * level)
        self.current_wind_strength = base_wind * (1.0 + 0.13 * level)
        self.current_path_noise_fraction = min(2.4, base_noise * (1.0 + 0.10 * level))
        self.current_lateral_fraction = min(2.2, base_lateral * (1.0 + 0.09 * level))
        self.current_interceptor_max_accel = self.config.interceptor_max_accel * max(0.55, 1.0 - 0.030 * level)
        self.current_interceptor_initial_speed = self.config.interceptor_initial_speed * max(0.55, 1.0 - 0.025 * level)
        self.current_missile_speed_fraction = float(min(1.4, speed_fraction * (1.0 + 0.05 * level)))
        self.current_missile_duration_min = self.config.missile_duration_min / max(self.current_missile_speed_fraction, 0.1)
        self.current_missile_duration_max = self.config.missile_duration_max / max(self.current_missile_speed_fraction, 0.1)

    def reset(self, env_indices: torch.Tensor | np.ndarray | list[int] | None = None) -> torch.Tensor:
        if env_indices is None:
            indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            indices = torch.as_tensor(env_indices, dtype=torch.long, device=self.device)
            if indices.numel() == 0:
                return self._observation()

        n = indices.numel()
        start, end = self._sample_far_points(n)
        direction = end - start
        distance = torch.linalg.norm(direction, dim=1, keepdim=True).clamp_min(1e-8)
        direction_unit = direction / distance
        u, v = self._orthonormal_basis(direction_unit)
        kind = torch.randint(0, 5, (n,), device=self.device, generator=self.generator)

        amp_1 = self._uniform(self.config.path_lateral_amplitude_min, self.config.path_lateral_amplitude_max, (n,)) * self.current_lateral_fraction
        amp_2 = self._uniform(self.config.path_lateral_amplitude_min, self.config.path_lateral_amplitude_max, (n,)) * self.current_lateral_fraction
        phase = self._uniform(0.0, 2.0 * np.pi, (n,))
        gain = self._uniform(1.15, 3.7, (n,))
        turns = self._uniform(0.65, 2.35, (n,))
        noise_amplitude = self._uniform(self.config.path_noise_amplitude_min, self.config.path_noise_amplitude_max, (n,)) * self.current_path_noise_fraction
        noise_frequency = self._uniform(1.2, 7.5, (n,))
        noise_offset = self._uniform(-500.0, 500.0, (n, 3))
        control_1 = start + direction * self._uniform(0.18, 0.42, (n, 1)) + u * self._uniform(-1.0, 1.0, (n, 1)) * amp_1.unsqueeze(1) + v * self._uniform(-1.0, 1.0, (n, 1)) * amp_2.unsqueeze(1)
        control_2 = start + direction * self._uniform(0.58, 0.88, (n, 1)) + u * self._uniform(-1.0, 1.0, (n, 1)) * amp_1.unsqueeze(1) + v * self._uniform(-1.0, 1.0, (n, 1)) * amp_2.unsqueeze(1)

        self.path_kind[indices] = kind
        self.path_start[indices] = start
        self.path_end[indices] = end
        self.path_direction_unit[indices] = direction_unit
        self.path_u[indices] = u
        self.path_v[indices] = v
        self.path_amp_1[indices] = amp_1
        self.path_amp_2[indices] = amp_2
        self.path_phase[indices] = phase
        self.path_gain[indices] = gain
        self.path_turns[indices] = turns
        self.path_noise_amplitude[indices] = noise_amplitude
        self.path_noise_frequency[indices] = noise_frequency
        self.path_noise_offset[indices] = noise_offset
        self.path_control_1[indices] = control_1
        self.path_control_2[indices] = control_2

        missile_duration = self._uniform(float(self.current_missile_duration_min), float(self.current_missile_duration_max), (n,))
        launch_delay = self._uniform(self.config.launch_delay_min, self.current_launch_delay_max, (n,))
        launch_delay = torch.minimum(launch_delay, missile_duration * 0.55)
        launch_tau = torch.clamp(launch_delay / missile_duration, 0.0, 0.98)

        self.missile_duration[indices] = missile_duration
        self.launch_delay[indices] = launch_delay
        self.missile_elapsed[indices] = launch_delay
        self.episode_elapsed[indices] = 0.0
        self.steps[indices] = 0
        self.done[indices] = False
        self.last_action[indices] = 0.0
        self.episode_reward[indices] = 0.0
        self.episode_length[indices] = 0
        self.interceptor_path_length[indices] = 0.0
        self.action_energy[indices] = 0.0
        self.action_smoothness_energy[indices] = 0.0
        self.distance_integral[indices] = 0.0

        missile_position = self.path_position(launch_tau, indices)
        next_tau = torch.clamp(launch_tau + self.config.dt / missile_duration, 0.0, 1.0)
        missile_velocity = (self.path_position(next_tau, indices) - missile_position) / self.config.dt
        self.missile_position[indices] = missile_position
        self.missile_velocity[indices] = missile_velocity
        self.reference_position[indices] = missile_position

        interceptor_position, interceptor_velocity = self._sample_interceptor_state(indices)
        self.interceptor_position[indices] = interceptor_position
        self.interceptor_velocity[indices] = interceptor_velocity
        time_value = self.launch_delay[indices] + self.episode_elapsed[indices]
        self.last_missile_wind[indices] = self._wind_acceleration(self.missile_position[indices], time_value)
        self.last_interceptor_wind[indices] = self._wind_acceleration(self.interceptor_position[indices], time_value)
        distance_now = self._distance(indices)
        self.prev_distance[indices] = distance_now
        self.min_distance[indices] = distance_now
        self.initial_distance[indices] = distance_now
        return self._observation()

    @property
    def missile_tau(self) -> torch.Tensor:
        return torch.clamp(self.missile_elapsed / self.missile_duration.clamp_min(1e-8), 0.0, 1.0)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = torch.nan_to_num(actions.to(self.device, dtype=torch.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(actions, -1.0, 1.0)
        action_norm = torch.linalg.norm(actions, dim=1, keepdim=True).clamp_min(1.0)
        actions = actions / action_norm
        previous_distance = self.prev_distance.clone()
        previous_action = self.last_action.clone()
        previous_interceptor_position = self.interceptor_position.clone()
        previous_relative_position = self.missile_position - self.interceptor_position
        self.last_action = actions

        self._advance_missile(self.config.dt)
        self._advance_interceptor(actions, self.config.dt)

        self.steps += 1
        self.episode_elapsed += self.config.dt

        distance = self._distance()
        segment_distance = self._segment_relative_distance(previous_relative_position, self.missile_position - self.interceptor_position)
        interceptor_step_distance = torch.linalg.norm(self.interceptor_position - previous_interceptor_position, dim=1)
        action_delta = actions - previous_action
        self.interceptor_path_length += interceptor_step_distance
        self.action_energy += torch.sum(actions * actions, dim=1) * self.config.dt
        self.action_smoothness_energy += torch.sum(action_delta * action_delta, dim=1)
        self.distance_integral += distance * self.config.dt
        self.prev_distance = distance
        self.min_distance = torch.minimum(self.min_distance, segment_distance)
        reward = self._reward(previous_distance, distance, actions, interceptor_step_distance, action_delta)
        done, terminal_reward, success, miss, out_of_bounds = self._termination(segment_distance)
        reward = reward + terminal_reward
        self.done = done
        self.episode_reward += reward
        self.episode_length += 1

        terminal_reward_values = self.episode_reward[done].detach().clone()
        terminal_length_values = self.episode_length[done].detach().clone()
        terminal_success_values = success[done].detach().clone()
        terminal_min_distance_values = self.min_distance[done].detach().clone()
        terminal_path_length_values = self.interceptor_path_length[done].detach().clone()
        terminal_action_energy_values = self.action_energy[done].detach().clone()
        terminal_distance_integral_values = self.distance_integral[done].detach().clone()
        terminal_miss_values = miss[done].detach().clone()
        terminal_out_of_bounds_values = out_of_bounds[done].detach().clone()
        done_indices = torch.nonzero(done, as_tuple=False).flatten()
        if done_indices.numel() > 0:
            self.reset(done_indices)

        info = {
            "episode_rewards": terminal_reward_values,
            "episode_lengths": terminal_length_values,
            "successes": terminal_success_values,
            "min_distances": terminal_min_distance_values,
            "path_lengths": terminal_path_length_values,
            "action_energies": terminal_action_energy_values,
            "distance_integrals": terminal_distance_integral_values,
            "misses": terminal_miss_values,
            "out_of_bounds": terminal_out_of_bounds_values,
            "done_count": int(done_indices.numel()),
            "mean_distance": float(distance.mean().detach().cpu()),
            "intercept_radius": float(self.current_intercept_radius),
            "difficulty_level": float(self.difficulty_level),
            "launch_distance_max": float(self.current_launch_distance_max),
            "interceptor_max_accel": float(self.current_interceptor_max_accel),
        }
        return self._observation(), reward.detach(), done.float().detach(), info

    def expert_action(self) -> torch.Tensor:
        relative_position = self.missile_position - self.interceptor_position
        relative_velocity = self.missile_velocity - self.interceptor_velocity
        distance = torch.linalg.norm(relative_position, dim=1, keepdim=True).clamp_min(1.0)
        speed = torch.linalg.norm(self.interceptor_velocity, dim=1, keepdim=True).clamp_min(20.0)
        time_to_go = torch.clamp(distance / (speed + 190.0), 0.18, 4.0)
        zero_effort_miss = relative_position + relative_velocity * time_to_go + 0.5 * self.gravity_vector.unsqueeze(0) * time_to_go.square()
        desired_accel = 3.4 * zero_effort_miss / time_to_go.square() + 0.8 * relative_velocity / time_to_go - self.gravity_vector.unsqueeze(0)
        action = desired_accel / self.current_interceptor_max_accel
        action = torch.clamp(action, -1.0, 1.0)
        action_norm = torch.linalg.norm(action, dim=1, keepdim=True).clamp_min(1.0)
        return action / action_norm

    def path_position(self, tau: torch.Tensor, env_indices: torch.Tensor | None = None) -> torch.Tensor:
        if env_indices is None:
            env_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tau = torch.clamp(tau.to(self.device, dtype=torch.float32), 0.0, 1.0)
        start = self.path_start[env_indices]
        end = self.path_end[env_indices]
        direction = end - start
        u = self.path_u[env_indices]
        v = self.path_v[env_indices]
        direction_unit = self.path_direction_unit[env_indices]
        kind = self.path_kind[env_indices]
        amp_1 = self.path_amp_1[env_indices]
        amp_2 = self.path_amp_2[env_indices]
        phase = self.path_phase[env_indices]
        gain = self.path_gain[env_indices]
        turns = self.path_turns[env_indices]
        tau_col = tau.unsqueeze(1)

        straight = start + direction * tau_col
        one_minus_tau = 1.0 - tau_col
        bezier = one_minus_tau.pow(3) * start + 3.0 * one_minus_tau.pow(2) * tau_col * self.path_control_1[env_indices] + 3.0 * one_minus_tau * tau_col.pow(2) * self.path_control_2[env_indices] + tau_col.pow(3) * end
        eased = (torch.exp(gain * tau) - 1.0) / (torch.exp(gain) - 1.0).clamp_min(1e-8)
        exponential = start + direction * eased.unsqueeze(1)
        exponential = exponential + u * amp_1.unsqueeze(1) * torch.sin(torch.pi * tau).unsqueeze(1)
        exponential = exponential + v * amp_2.unsqueeze(1) * (tau * (1.0 - tau) * 4.0).unsqueeze(1)
        smooth = tau * tau * (3.0 - 2.0 * tau)
        s_curve = start + direction * smooth.unsqueeze(1)
        s_curve = s_curve + u * amp_1.unsqueeze(1) * torch.sin(torch.pi * tau + phase).unsqueeze(1)
        s_curve = s_curve + v * amp_2.unsqueeze(1) * (torch.sin(2.0 * torch.pi * tau + phase) * torch.sin(torch.pi * tau)).unsqueeze(1)
        radius = amp_1 * torch.sin(torch.pi * tau)
        angle = 2.0 * torch.pi * turns * tau + phase
        corkscrew = start + direction * tau_col + radius.unsqueeze(1) * (u * torch.cos(angle).unsqueeze(1) + v * torch.sin(angle).unsqueeze(1))

        base = torch.where((kind == 0).unsqueeze(1), straight, bezier)
        base = torch.where((kind == 2).unsqueeze(1), exponential, base)
        base = torch.where((kind == 3).unsqueeze(1), s_curve, base)
        base = torch.where((kind == 4).unsqueeze(1), corkscrew, base)

        offset = self.path_noise_offset[env_indices]
        frequency = self.path_noise_frequency[env_indices]
        coords_x = torch.stack([offset[:, 0] + tau * frequency, offset[:, 1], offset[:, 2]], dim=1)
        coords_y = torch.stack([offset[:, 0], offset[:, 1] + tau * frequency, offset[:, 2] + 19.7], dim=1)
        coords_z = torch.stack([offset[:, 0] + 41.3, offset[:, 1], offset[:, 2] + tau * frequency], dim=1)
        noise_x = self._fractal_value_noise(coords_x, 0)
        noise_y = self._fractal_value_noise(coords_y, 1)
        noise_z = self._fractal_value_noise(coords_z, 2)
        envelope = torch.sin(torch.pi * tau)
        noise = self.path_noise_amplitude[env_indices].unsqueeze(1) * envelope.unsqueeze(1) * (u * noise_x.unsqueeze(1) + v * noise_y.unsqueeze(1) + direction_unit * 0.25 * noise_z.unsqueeze(1))
        return base + noise

    def path_derivative(self, tau: torch.Tensor, env_indices: torch.Tensor | None = None) -> torch.Tensor:
        epsilon = 1e-3
        left_tau = torch.clamp(tau - epsilon, 0.0, 1.0)
        right_tau = torch.clamp(tau + epsilon, 0.0, 1.0)
        delta = (right_tau - left_tau).unsqueeze(1).clamp_min(1e-6)
        return (self.path_position(right_tau, env_indices) - self.path_position(left_tau, env_indices)) / delta

    def _allocate(self) -> None:
        n = self.num_envs
        self.path_kind = torch.zeros(n, dtype=torch.long, device=self.device)
        self.path_start = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.path_end = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.path_direction_unit = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.path_u = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.path_v = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.path_control_1 = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.path_control_2 = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.path_amp_1 = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.path_amp_2 = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.path_phase = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.path_gain = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.path_turns = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.path_noise_amplitude = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.path_noise_frequency = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.path_noise_offset = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.missile_duration = torch.ones(n, dtype=torch.float32, device=self.device)
        self.launch_delay = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.missile_elapsed = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.episode_elapsed = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.missile_position = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.missile_velocity = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.reference_position = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.interceptor_position = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.interceptor_velocity = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.last_missile_wind = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.last_interceptor_wind = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.prev_distance = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.min_distance = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.initial_distance = torch.ones(n, dtype=torch.float32, device=self.device)
        self.last_action = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self.episode_reward = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.episode_length = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.interceptor_path_length = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.action_energy = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.action_smoothness_energy = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.distance_integral = torch.zeros(n, dtype=torch.float32, device=self.device)

    def _sample_far_points(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        limit = self.config.world_limit * 0.55
        altitude_min = 130.0
        altitude_max = self.config.world_limit * 0.62
        start = self._sample_points(n, limit, altitude_min, altitude_max)
        end = self._sample_points(n, limit, altitude_min, altitude_max)
        for _ in range(96):
            distance = torch.linalg.norm(end - start, dim=1)
            valid = (distance >= self.config.min_start_end_distance) & (distance <= self.config.max_start_end_distance)
            if bool(valid.all()):
                break
            missing = torch.nonzero(~valid, as_tuple=False).flatten()
            start[missing] = self._sample_points(missing.numel(), limit, altitude_min, altitude_max)
            end[missing] = self._sample_points(missing.numel(), limit, altitude_min, altitude_max)
        return start, end

    def _sample_points(self, n: int, limit: float, altitude_min: float, altitude_max: float) -> torch.Tensor:
        xy = self._uniform(-limit, limit, (n, 2))
        z = self._uniform(altitude_min, altitude_max, (n, 1))
        return torch.cat([xy, z], dim=1)

    def _orthonormal_basis(self, direction_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = torch.zeros_like(direction_unit)
        candidate[:, 2] = 1.0
        alternate = torch.zeros_like(direction_unit)
        alternate[:, 1] = 1.0
        use_alternate = torch.abs((candidate * direction_unit).sum(dim=1, keepdim=True)) > 0.92
        candidate = torch.where(use_alternate, alternate, candidate)
        u = torch.linalg.cross(direction_unit, candidate, dim=1)
        u = u / torch.linalg.norm(u, dim=1, keepdim=True).clamp_min(1e-8)
        v = torch.linalg.cross(direction_unit, u, dim=1)
        v = v / torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-8)
        return u, v

    def _sample_interceptor_state(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = indices.numel()
        rel_direction = self._randn((n, 3))
        rel_direction[:, 2] = self._uniform(-0.35, 0.55, (n,))
        rel_direction = rel_direction / torch.linalg.norm(rel_direction, dim=1, keepdim=True).clamp_min(1e-8)
        distance = self._uniform(self.current_launch_distance_min, self.current_launch_distance_max, (n, 1))
        position = self.missile_position[indices] - rel_direction * distance
        position[:, 0] = torch.clamp(position[:, 0], -self.config.world_limit * 0.88, self.config.world_limit * 0.88)
        position[:, 1] = torch.clamp(position[:, 1], -self.config.world_limit * 0.88, self.config.world_limit * 0.88)
        position[:, 2] = torch.clamp(position[:, 2], 20.0, self.config.world_limit * 0.88)
        initial_direction = self.missile_position[indices] - position
        initial_direction = initial_direction / torch.linalg.norm(initial_direction, dim=1, keepdim=True).clamp_min(1e-8)
        velocity = initial_direction * self.current_interceptor_initial_speed + self._randn((n, 3)) * 3.0
        return position, velocity

    def _advance_missile(self, dt: float) -> None:
        tau = self.missile_tau
        target_tau = torch.clamp(tau + dt / self.missile_duration, 0.0, 1.0)
        desired_position = self.path_position(target_tau)
        desired_velocity = (desired_position - self.reference_position) / dt
        self.reference_position = desired_position
        tracking_accel = self.config.missile_tracking_kp * (desired_position - self.missile_position) + self.config.missile_tracking_kd * (desired_velocity - self.missile_velocity)
        tracking_accel = self._limit_vector(tracking_accel, self.config.missile_max_accel)
        wind = self._wind_acceleration(self.missile_position, self.launch_delay + self.episode_elapsed)
        self.last_missile_wind = wind
        drag = self._drag_acceleration(self.missile_velocity, wind, self.config.missile_drag)
        noise_coords = torch.stack([13.0 + tau * 7.0, 17.0 + tau * 9.0, 23.0 + tau * 8.0], dim=1)
        noise_accel = self.config.missile_noise_accel * torch.stack([
            self._fractal_value_noise(torch.stack([noise_coords[:, 0], torch.full_like(tau, 3.0), torch.full_like(tau, 9.0)], dim=1), 7),
            self._fractal_value_noise(torch.stack([torch.full_like(tau, 5.0), noise_coords[:, 1], torch.full_like(tau, 11.0)], dim=1), 8),
            self._fractal_value_noise(torch.stack([torch.full_like(tau, 7.0), torch.full_like(tau, 19.0), noise_coords[:, 2]], dim=1), 9),
        ], dim=1)
        acceleration = tracking_accel + self.gravity_vector.unsqueeze(0) + drag + wind * 0.22 + noise_accel
        self.missile_velocity = self.missile_velocity + acceleration * dt
        self.missile_position = self.missile_position + self.missile_velocity * dt
        self.missile_elapsed = torch.minimum(self.missile_duration, self.missile_elapsed + dt)

    def _advance_interceptor(self, action: torch.Tensor, dt: float) -> None:
        thrust = action * self.current_interceptor_max_accel
        wind = self._wind_acceleration(self.interceptor_position, self.launch_delay + self.episode_elapsed)
        self.last_interceptor_wind = wind
        drag = self._drag_acceleration(self.interceptor_velocity, wind, self.config.interceptor_drag)
        acceleration = thrust + self.gravity_vector.unsqueeze(0) + drag + wind * 0.34
        self.interceptor_velocity = self.interceptor_velocity + acceleration * dt
        self.interceptor_position = self.interceptor_position + self.interceptor_velocity * dt

    def _wind_acceleration(self, position: torch.Tensor, time_value: torch.Tensor) -> torch.Tensor:
        scale = self.config.wind_space_scale
        t = time_value * self.config.wind_time_scale
        coords_0 = torch.stack([position[:, 0] * scale, position[:, 1] * scale + 31.4, position[:, 2] * scale + t], dim=1)
        coords_1 = torch.stack([position[:, 0] * scale + 17.2, position[:, 1] * scale + t, position[:, 2] * scale + 11.1], dim=1)
        coords_2 = torch.stack([position[:, 0] * scale + t, position[:, 1] * scale + 5.7, position[:, 2] * scale + 29.9], dim=1)
        return torch.stack([
            self._fractal_value_noise(coords_0, 3),
            self._fractal_value_noise(coords_1, 4),
            self._fractal_value_noise(coords_2, 5),
        ], dim=1) * self.current_wind_strength

    def _drag_acceleration(self, velocity: torch.Tensor, wind_acceleration: torch.Tensor, coefficient: float) -> torch.Tensor:
        relative_velocity = velocity - wind_acceleration
        speed = torch.linalg.norm(relative_velocity, dim=1, keepdim=True)
        return -coefficient * speed * relative_velocity

    def _reward(self, previous_distance: torch.Tensor, distance: torch.Tensor, action: torch.Tensor, interceptor_step_distance: torch.Tensor, action_delta: torch.Tensor) -> torch.Tensor:
        relative_position = self.missile_position - self.interceptor_position
        relative_velocity = self.missile_velocity - self.interceptor_velocity
        closing_speed = -torch.sum(relative_position * relative_velocity, dim=1) / distance.clamp_min(1e-6)
        proximity = self.config.proximity_weight * torch.exp(-(distance ** 2) / max(self.config.proximity_sigma ** 2, 1e-8))
        near_miss = self.config.near_miss_weight * torch.exp(-(distance ** 2) / max(self.config.near_miss_sigma ** 2, 1e-8))
        progress = self.config.progress_weight * (previous_distance - distance) / max(self.config.progress_normalizer, 1.0)
        closing = self.config.closing_weight * (closing_speed - self.config.closing_bias) / max(self.config.velocity_scale, 1.0)
        rel_speed = torch.linalg.norm(relative_velocity, dim=1).clamp_min(1e-6)
        alignment_cos = -torch.sum(relative_position * relative_velocity, dim=1) / (distance.clamp_min(1e-6) * rel_speed)
        alignment_bonus = self.config.alignment_weight * torch.clamp(alignment_cos, min=0.0)
        miss_distance = -self.config.miss_distance_weight * (distance / self.config.distance_scale)
        control = -self.config.control_penalty * torch.sum(action * action, dim=1)
        path_cost = -self.config.path_length_penalty * (interceptor_step_distance / self.config.distance_scale)
        energy_cost = -self.config.energy_penalty * torch.sum(action * action, dim=1) * self.config.dt
        smoothness_cost = -self.config.action_smoothness_penalty * torch.sum(action_delta * action_delta, dim=1)
        distance_integral_cost = -self.config.distance_integral_penalty * (distance / self.config.distance_scale) * self.config.dt
        return (
            proximity
            + near_miss
            + progress
            + closing
            + alignment_bonus
            + miss_distance
            + control
            + path_cost
            + energy_cost
            + smoothness_cost
            + distance_integral_cost
            - self.config.time_penalty
        )

    def _termination(self, distance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        success = distance <= self.current_intercept_radius
        out_of_bounds = (~success) & (self._out_of_bounds(self.interceptor_position) | self._out_of_bounds(self.missile_position))
        missile_finished = (~success) & (~out_of_bounds) & (self.missile_tau >= 1.0)
        time_limit = (~success) & (~out_of_bounds) & (self.episode_elapsed >= self.config.max_episode_time)
        miss = missile_finished | time_limit
        done = success | out_of_bounds | miss
        terminal_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        efficiency = self.initial_distance / torch.maximum(torch.maximum(self.interceptor_path_length, self.initial_distance), torch.ones_like(self.initial_distance))
        success_reward = torch.full_like(terminal_reward, self.config.success_reward) + self.config.efficient_success_bonus * efficiency + 25.0 * (1.0 - self.missile_tau)
        success_reward = success_reward - self.config.late_success_penalty * (self.episode_elapsed / self.config.max_episode_time)
        terminal_reward = torch.where(success, success_reward, terminal_reward)
        terminal_reward = torch.where(out_of_bounds, torch.full_like(terminal_reward, -self.config.out_of_bounds_penalty), terminal_reward)
        miss_penalty = -self.config.miss_penalty - 80.0 * torch.clamp(distance / self.config.distance_scale, max=1.0)
        terminal_reward = torch.where(miss, miss_penalty, terminal_reward)
        return done, terminal_reward, success, miss, out_of_bounds

    def _out_of_bounds(self, position: torch.Tensor) -> torch.Tensor:
        limit = self.config.world_limit
        return (torch.abs(position[:, 0]) > limit) | (torch.abs(position[:, 1]) > limit) | (position[:, 2] < -120.0) | (position[:, 2] > limit)

    def _observation(self) -> torch.Tensor:
        relative_position = self.missile_position - self.interceptor_position
        relative_velocity = self.missile_velocity - self.interceptor_velocity
        distance = torch.linalg.norm(relative_position, dim=1, keepdim=True)
        closing_speed = -torch.sum(relative_position * relative_velocity, dim=1, keepdim=True) / distance.clamp_min(1e-6)
        time_remaining = torch.minimum(self.config.max_episode_time - self.episode_elapsed, self.missile_duration - self.missile_elapsed).clamp_min(0.0).unsqueeze(1)
        wind_interceptor = self.last_interceptor_wind
        wind_missile = self.last_missile_wind
        gravity = torch.full((self.num_envs, 1), self.config.gravity / self.config.acceleration_scale, dtype=torch.float32, device=self.device)
        expert_hint = self.expert_action()
        lookahead = self.config.lookahead_seconds
        lookahead_missile_position = self.missile_position + self.missile_velocity * lookahead + 0.5 * self.gravity_vector.unsqueeze(0) * (lookahead ** 2)
        lookahead_interceptor_position = self.interceptor_position + self.interceptor_velocity * lookahead
        lookahead_relative = lookahead_missile_position - lookahead_interceptor_position
        time_to_go = (distance / (closing_speed + 50.0).clamp_min(30.0)).clamp(0.05, self.config.time_to_go_clip)
        altitude_diff = (self.missile_position[:, 2:3] - self.interceptor_position[:, 2:3])
        return torch.cat([
            relative_position / self.config.distance_scale,
            relative_velocity / self.config.velocity_scale,
            self.missile_position / self.config.world_limit,
            self.missile_velocity / self.config.velocity_scale,
            self.interceptor_position / self.config.world_limit,
            self.interceptor_velocity / self.config.velocity_scale,
            distance / self.config.distance_scale,
            closing_speed / self.config.velocity_scale,
            self.missile_tau.unsqueeze(1),
            (self.episode_elapsed / self.config.max_episode_time).unsqueeze(1),
            time_remaining / self.config.max_episode_time,
            self.last_action,
            wind_interceptor / max(self.current_wind_strength, 1e-8),
            wind_missile / max(self.current_wind_strength, 1e-8),
            gravity,
            expert_hint,
            lookahead_relative / self.config.distance_scale,
            time_to_go / self.config.time_to_go_clip,
            altitude_diff / self.config.distance_scale,
        ], dim=1)

    def _distance(self, indices: torch.Tensor | None = None) -> torch.Tensor:
        if indices is None:
            return torch.linalg.norm(self.missile_position - self.interceptor_position, dim=1)
        return torch.linalg.norm(self.missile_position[indices] - self.interceptor_position[indices], dim=1)

    def _segment_relative_distance(self, previous_relative_position: torch.Tensor, current_relative_position: torch.Tensor) -> torch.Tensor:
        delta = current_relative_position - previous_relative_position
        denom = torch.sum(delta * delta, dim=1).clamp_min(1e-12)
        amount = torch.clamp(-torch.sum(previous_relative_position * delta, dim=1) / denom, 0.0, 1.0)
        closest = previous_relative_position + delta * amount.unsqueeze(1)
        return torch.linalg.norm(closest, dim=1)

    def _limit_vector(self, vector: torch.Tensor, limit: float) -> torch.Tensor:
        norm = torch.linalg.norm(vector, dim=1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(limit / norm, max=1.0)
        return vector * scale

    def _value_noise(self, coords: torch.Tensor, channel: int) -> torch.Tensor:
        base = torch.floor(coords)
        frac = coords - base
        fade = frac * frac * frac * (frac * (frac * 6.0 - 15.0) + 10.0)
        x0 = base[:, 0]
        y0 = base[:, 1]
        z0 = base[:, 2]
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        z1 = z0 + 1.0
        c000 = self._hash_lattice(x0, y0, z0, channel)
        c100 = self._hash_lattice(x1, y0, z0, channel)
        c010 = self._hash_lattice(x0, y1, z0, channel)
        c110 = self._hash_lattice(x1, y1, z0, channel)
        c001 = self._hash_lattice(x0, y0, z1, channel)
        c101 = self._hash_lattice(x1, y0, z1, channel)
        c011 = self._hash_lattice(x0, y1, z1, channel)
        c111 = self._hash_lattice(x1, y1, z1, channel)
        u = fade[:, 0]
        v = fade[:, 1]
        w = fade[:, 2]
        x00 = c000 + u * (c100 - c000)
        x10 = c010 + u * (c110 - c010)
        x01 = c001 + u * (c101 - c001)
        x11 = c011 + u * (c111 - c011)
        y0_value = x00 + v * (x10 - x00)
        y1_value = x01 + v * (x11 - x01)
        return y0_value + w * (y1_value - y0_value)

    def _fractal_value_noise(self, coords: torch.Tensor, channel: int, octaves: int = 3) -> torch.Tensor:
        if self.config.use_fast_vector_noise:
            phase = float(self.seed + channel * 101) * 0.0137
            x = coords[:, 0]
            y = coords[:, 1]
            z = coords[:, 2]
            channel_scale = float(channel + 1)
            value = 0.55 * torch.sin(x * (1.173 + channel_scale * 0.011) + y * (1.917 + channel_scale * 0.017) + z * (1.431 + channel_scale * 0.013) + phase)
            value = value + 0.30 * torch.sin(x * (2.311 + channel_scale * 0.019) - y * (1.227 + channel_scale * 0.023) + z * (2.719 + channel_scale * 0.007) + phase * 1.71)
            value = value + 0.15 * torch.cos(x * (4.113 + channel_scale * 0.005) + y * (3.313 + channel_scale * 0.009) - z * (2.181 + channel_scale * 0.015) + phase * 2.37)
            return torch.clamp(value, -1.0, 1.0)
        value = torch.zeros(coords.shape[0], dtype=torch.float32, device=self.device)
        amplitude = 1.0
        frequency = 1.0
        norm = 0.0
        for octave in range(octaves):
            value = value + amplitude * self._value_noise(coords * frequency, channel + octave * 17)
            norm += amplitude
            amplitude *= 0.5
            frequency *= 2.03
        return value / norm

    def _hash_lattice(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, channel: int) -> torch.Tensor:
        value = torch.sin(x * 12.9898 + y * 78.233 + z * 37.719 + float(self.seed + channel * 101) * 0.0137) * 43758.5453
        return (value - torch.floor(value)) * 2.0 - 1.0

    def _uniform(self, low: float, high: float, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, dtype=torch.float32, device=self.device, generator=self.generator) * (high - low) + low

    def _randn(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, dtype=torch.float32, device=self.device, generator=self.generator)
