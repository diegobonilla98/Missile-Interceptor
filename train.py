from collections import deque
from pathlib import Path
import csv
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from xgboost_guidance import collect_guidance_dataset, load_xgboost_model, predict_xgboost_tensor, save_xgboost_model, train_xgboost_guidance
from agent import HybridActorCritic, LSTMActorCritic
from environment import EnvironmentConfig, ProjectileInterceptEnv, TorchProjectileInterceptVecEnv
from ppo import PPOConfig, RecurrentPPO, TorchRolloutBuffer, TorchRunningMeanStd, linear_schedule


seed = 123
if seed < 0:
    seed = random.randint(0, 2**31 - 1)
device_name = "cuda" if torch.cuda.is_available() else "cpu"
num_envs = 1024 if torch.cuda.is_available() else 64
rollout_steps = 192
total_updates = 1500
learning_rate = 3.0e-4
min_learning_rate = 2.0e-5
warmup_updates = 25
lr_schedule = "cosine"
anneal_learning_rate = True
model_architecture = "hybrid"
hidden_size = 512
lstm_layers = 3
transformer_layers = 3
attention_heads = 8
policy_initial_log_std = -1.4
checkpoint_interval = 25
plot_interval = 25
evaluation_interval = 5
evaluation_episodes = 192 if torch.cuda.is_available() else 24
behavior_cloning_updates = 250 if torch.cuda.is_available() else 30
behavior_cloning_sequence_length = 32
behavior_cloning_learning_rate = 7.5e-4
bc_early_stopping = True
bc_early_stop_patience = 60
bc_early_stop_min_updates = 80
bc_early_stop_min_delta = 1.0e-5
use_guidance_prior = True
residual_action_scale = 0.30
use_xgboost_teacher = True
retrain_xgboost_teacher = True
xgboost_model_path = Path("models/xgboost_guidance.joblib")
xgboost_dataset_samples = 320_000 if torch.cuda.is_available() else 16_000
xgboost_dataset_batch_size = 8192 if torch.cuda.is_available() else 512
xgboost_n_estimators = 480
xgboost_max_depth = 8
xgboost_learning_rate = 0.035
xgboost_teacher_weight = 0.32
adaptive_curriculum = True
curriculum_initial_progress = 0.05
curriculum_threshold_lo = 0.85
curriculum_threshold_hi = 0.95
curriculum_progress_step_lo = 0.025
curriculum_progress_step_hi = 0.060
difficulty_level = 0.0
max_difficulty_level = 40.0
difficulty_step_lo = 0.5
difficulty_step_hi = 1.5
difficulty_success_threshold = 0.93
difficulty_adjust_interval = 5
early_stopping = True
early_stop_min_difficulty = 22.0
early_stop_min_guided_success = 0.985
early_stop_patience_evals = 4
plateau_patience_evals = 80
log_directory = Path("logs")
checkpoint_directory = Path("checkpoints")
plot_directory = Path("plots")
run_name = time.strftime("projectile_intercept_%Y%m%d_%H%M%S")

environment_config = EnvironmentConfig(
    interceptor_max_accel=900.0,
    interceptor_initial_speed=160.0,
    interceptor_launch_distance_max=450.0,
    launch_delay_max=1.6,
    proximity_sigma=140.0,
    proximity_weight=6.0,
    near_miss_sigma=32.0,
    near_miss_weight=10.0,
    progress_weight=0.20,
    progress_normalizer=200.0,
    closing_weight=0.06,
    closing_bias=60.0,
    alignment_weight=0.20,
    miss_distance_weight=0.40,
    control_penalty=0.008,
    path_length_penalty=22.0,
    energy_penalty=1.8,
    action_smoothness_penalty=0.10,
    distance_integral_penalty=7.5,
    late_success_penalty=140.0,
    efficient_success_bonus=240.0,
    time_penalty=0.008,
    success_reward=1200.0,
    miss_penalty=300.0,
    curriculum_start_intercept_radius=110.0,
    curriculum_start_launch_distance_min=60.0,
    curriculum_start_launch_distance_max=300.0,
    curriculum_start_wind_fraction=0.10,
    curriculum_start_path_noise_fraction=0.10,
    curriculum_start_lateral_fraction=0.20,
    curriculum_start_missile_speed_fraction=0.55,
)
ppo_config = PPOConfig(
    gamma=0.994,
    gae_lambda=0.95,
    clip_coef=0.16,
    value_clip_coef=0.18,
    entropy_coef=0.003,
    value_coef=0.55,
    max_grad_norm=0.8,
    update_epochs=4,
    num_minibatches=12,
    target_kl=0.025,
    normalize_advantages=True,
    clip_value_loss=True,
)


def configure_torch(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_rms: TorchRunningMeanStd,
    update: int,
    global_step: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "obs_rms": obs_rms.state_dict(),
        "update": update,
        "global_step": global_step,
        "metrics": metrics,
        "environment_config": environment_config.__dict__,
        "ppo_config": ppo_config.__dict__,
        "seed": seed,
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "model_architecture": model_architecture,
        "hidden_size": hidden_size,
        "lstm_layers": lstm_layers,
        "transformer_layers": transformer_layers,
        "attention_heads": attention_heads,
        "policy_initial_log_std": policy_initial_log_std,
        "use_guidance_prior": use_guidance_prior,
        "residual_action_scale": residual_action_scale,
        "adaptive_curriculum": adaptive_curriculum,
        "max_difficulty_level": max_difficulty_level,
        "difficulty_step_lo": difficulty_step_lo,
        "difficulty_step_hi": difficulty_step_hi,
        "curriculum_threshold_lo": curriculum_threshold_lo,
        "curriculum_threshold_hi": curriculum_threshold_hi,
        "use_xgboost_teacher": use_xgboost_teacher,
        "xgboost_model_path": str(xgboost_model_path),
        "xgboost_teacher_weight": xgboost_teacher_weight,
    }
    torch.save(checkpoint, path)


def write_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "update",
            "global_step",
            "learning_rate",
            "mean_reward",
            "mean_length",
            "success_rate",
            "rollout_hit_rate",
            "rollout_terminal_success_rate",
            "mean_min_distance",
            "eval_success_rate",
            "eval_mean_min_distance",
            "eval_guided_success_rate",
            "eval_guided_mean_min_distance",
            "eval_expert_success_rate",
            "eval_expert_mean_min_distance",
            "eval_xgboost_success_rate",
            "eval_xgboost_mean_min_distance",
            "eval_ensemble_success_rate",
            "eval_ensemble_mean_min_distance",
            "eval_raw_success_rate",
            "eval_raw_mean_min_distance",
            "mean_residual_norm",
            "mean_path_length",
            "mean_action_energy",
            "mean_distance_integral",
            "difficulty_level",
            "launch_distance_max",
            "interceptor_max_accel",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "grad_norm",
            "explained_variance",
            "intercept_radius",
        ])


def append_log(path: Path, row: dict) -> None:
    with path.open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            row["update"],
            row["global_step"],
            row["learning_rate"],
            row["mean_reward"],
            row["mean_length"],
            row["success_rate"],
            row["rollout_hit_rate"],
            row["rollout_terminal_success_rate"],
            row["mean_min_distance"],
            row["eval_success_rate"],
            row["eval_mean_min_distance"],
            row["eval_guided_success_rate"],
            row["eval_guided_mean_min_distance"],
            row["eval_expert_success_rate"],
            row["eval_expert_mean_min_distance"],
            row["eval_xgboost_success_rate"],
            row["eval_xgboost_mean_min_distance"],
            row["eval_ensemble_success_rate"],
            row["eval_ensemble_mean_min_distance"],
            row["eval_raw_success_rate"],
            row["eval_raw_mean_min_distance"],
            row["mean_residual_norm"],
            row["mean_path_length"],
            row["mean_action_energy"],
            row["mean_distance_integral"],
            row["difficulty_level"],
            row["launch_distance_max"],
            row["interceptor_max_accel"],
            row["policy_loss"],
            row["value_loss"],
            row["entropy"],
            row["approx_kl"],
            row["clip_fraction"],
            row["grad_norm"],
            row["explained_variance"],
            row["intercept_radius"],
        ])


def load_log(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)
    return {name: np.asarray(data[name]) for name in data.dtype.names}


def plot_training_summary(log_path: Path, output_path: Path) -> None:
    if not log_path.exists():
        return
    data = load_log(log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    axes[0, 0].plot(data["update"], data["mean_reward"])
    axes[0, 0].set_title("Episode reward")
    axes[0, 0].set_xlabel("update")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(data["update"], data["success_rate"], label="train")
    if "eval_guided_success_rate" in data:
        axes[0, 1].plot(data["update"], data["eval_guided_success_rate"], label="guided")
    if "eval_expert_success_rate" in data:
        axes[0, 1].plot(data["update"], data["eval_expert_success_rate"], label="expert")
    if "eval_xgboost_success_rate" in data:
        axes[0, 1].plot(data["update"], data["eval_xgboost_success_rate"], label="xgb")
    if "eval_ensemble_success_rate" in data:
        axes[0, 1].plot(data["update"], data["eval_ensemble_success_rate"], label="ensemble")
    if "eval_raw_success_rate" in data:
        axes[0, 1].plot(data["update"], data["eval_raw_success_rate"], label="raw")
    if "eval_guided_success_rate" not in data:
        axes[0, 1].plot(data["update"], data["eval_success_rate"], label="eval")
    axes[0, 1].set_title("Success rate")
    axes[0, 1].set_xlabel("update")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)
    axes[1, 0].plot(data["update"], data["mean_min_distance"], label="train")
    axes[1, 0].plot(data["update"], data["eval_mean_min_distance"], label="eval")
    axes[1, 0].set_title("Minimum distance")
    axes[1, 0].set_xlabel("update")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].plot(data["update"], data["policy_loss"], label="policy")
    axes[1, 1].plot(data["update"], data["value_loss"], label="value")
    axes[1, 1].plot(data["update"], data["entropy"], label="entropy")
    axes[1, 1].set_title("PPO losses")
    axes[1, 1].set_xlabel("update")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    efficiency_labels = 0
    if "mean_path_length" in data:
        axes[2, 0].plot(data["update"], data["mean_path_length"], label="path")
        efficiency_labels += 1
    if "mean_action_energy" in data:
        axes[2, 0].plot(data["update"], data["mean_action_energy"], label="energy")
        efficiency_labels += 1
    axes[2, 0].set_title("Efficiency costs")
    axes[2, 0].set_xlabel("update")
    if efficiency_labels > 0:
        axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.25)
    axes[2, 1].plot(data["update"], data["difficulty_level"], label="difficulty")
    axes[2, 1].plot(data["update"], data["intercept_radius"], label="radius")
    axes[2, 1].set_title("Task hardness")
    axes[2, 1].set_xlabel("update")
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def normalize_action_tensor(actions: torch.Tensor) -> torch.Tensor:
    actions = torch.clamp(actions, -1.0, 1.0)
    action_norm = torch.linalg.norm(actions, dim=1, keepdim=True).clamp_min(1.0)
    return actions / action_norm


def compute_learning_rate(update: int) -> float:
    if not anneal_learning_rate:
        return learning_rate
    if update <= warmup_updates:
        return learning_rate * (update / max(warmup_updates, 1))
    if lr_schedule == "cosine":
        progress = (update - warmup_updates) / max(total_updates - warmup_updates, 1)
        progress = max(0.0, min(1.0, progress))
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + (learning_rate - min_learning_rate) * cos_val
    return max(min_learning_rate, linear_schedule(learning_rate, update, total_updates))


def create_model(observation_dim: int, action_dim: int, device: torch.device) -> torch.nn.Module:
    if model_architecture == "hybrid":
        return HybridActorCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            transformer_layers=transformer_layers,
            attention_heads=attention_heads,
            actor_hidden_size=hidden_size,
            critic_hidden_size=hidden_size,
            initial_log_std=policy_initial_log_std,
        ).to(device)
    return LSTMActorCritic(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        actor_hidden_size=hidden_size,
        critic_hidden_size=hidden_size,
        initial_log_std=policy_initial_log_std,
    ).to(device)


def prepare_xgboost_teacher(device: torch.device):
    if not use_xgboost_teacher:
        return None, {}
    if xgboost_model_path.exists() and not retrain_xgboost_teacher:
        return load_xgboost_model(xgboost_model_path)
    difficulty_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    features, targets = collect_guidance_dataset(
        config=environment_config,
        samples=xgboost_dataset_samples,
        batch_size=xgboost_dataset_batch_size,
        device=device,
        seed=seed + 700_000,
        difficulty_levels=difficulty_values,
    )
    model, metrics = train_xgboost_guidance(
        features=features,
        targets=targets,
        seed=seed,
        n_estimators=xgboost_n_estimators,
        max_depth=xgboost_max_depth,
        learning_rate=xgboost_learning_rate,
    )
    save_xgboost_model(xgboost_model_path, model, metrics)
    return model, metrics


def blend_teacher_actions(expert_actions: torch.Tensor, xgboost_actions: torch.Tensor | None) -> torch.Tensor:
    if xgboost_actions is None:
        return expert_actions
    blended = (1.0 - xgboost_teacher_weight) * expert_actions + xgboost_teacher_weight * xgboost_actions
    return normalize_action_tensor(blended)


def guided_vector_action(env: TorchProjectileInterceptVecEnv, residual_actions: torch.Tensor) -> torch.Tensor:
    if not use_guidance_prior:
        return normalize_action_tensor(residual_actions)
    guidance_actions = env.expert_action()
    return normalize_action_tensor(guidance_actions + residual_action_scale * residual_actions)


def guided_scalar_action(env: ProjectileInterceptEnv, residual_action: np.ndarray) -> np.ndarray:
    residual_action = np.asarray(residual_action, dtype=np.float64)
    if use_guidance_prior:
        action = env.expert_action() + residual_action_scale * residual_action
    else:
        action = residual_action
    action = np.clip(action, -1.0, 1.0)
    action_norm = np.linalg.norm(action)
    if action_norm > 1.0:
        action = action / action_norm
    return action


def blend_numpy_actions(primary_actions: np.ndarray, xgboost_actions: np.ndarray | None) -> np.ndarray:
    primary_actions = np.asarray(primary_actions, dtype=np.float64)
    if xgboost_actions is None:
        action = primary_actions
    else:
        action = (1.0 - xgboost_teacher_weight) * primary_actions + xgboost_teacher_weight * np.asarray(xgboost_actions, dtype=np.float64)
    action = np.clip(action, -1.0, 1.0)
    action_norm = np.linalg.norm(action)
    if action_norm > 1.0:
        action = action / action_norm
    return action


def run_behavior_cloning(model: torch.nn.Module, obs_rms: TorchRunningMeanStd, device: torch.device, xgboost_teacher) -> None:
    if behavior_cloning_updates <= 0:
        return
    model.train()
    env = TorchProjectileInterceptVecEnv(num_envs=num_envs, config=environment_config, seed=seed + 100_000, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=behavior_cloning_learning_rate, eps=1e-5)
    env.set_curriculum(0.0)
    observations = env.reset()
    episode_starts = torch.ones(num_envs, dtype=torch.float32, device=device)
    progress = tqdm(range(1, behavior_cloning_updates + 1), dynamic_ncols=True, desc="BC")

    best_loss = float("inf")
    best_loss_smoothed = float("inf")
    plateau_count = 0
    smoothed_loss = None

    for update in progress:
        curriculum_progress = min(1.0, update / max(behavior_cloning_updates, 1))
        env.set_curriculum(curriculum_progress)
        sequence_observations = []
        sequence_actions = []
        sequence_starts = []
        for _ in range(behavior_cloning_sequence_length):
            obs_rms.update(observations)
            normalized_observations = obs_rms.normalize(observations)
            expert_actions = env.expert_action()
            if xgboost_teacher is not None:
                xgboost_actions = predict_xgboost_tensor(xgboost_teacher, observations, device)
            else:
                xgboost_actions = None
            teacher_actions = blend_teacher_actions(expert_actions, xgboost_actions)
            noise = torch.randn_like(expert_actions)
            if use_guidance_prior:
                target_actions = torch.zeros_like(expert_actions)
                noisy_actions = guided_vector_action(env, noise * 0.20)
            else:
                target_actions = teacher_actions
                noisy_actions = normalize_action_tensor(teacher_actions + noise * 0.05)
            sequence_observations.append(normalized_observations)
            sequence_actions.append(target_actions)
            sequence_starts.append(episode_starts)
            observations, _, dones, _ = env.step(noisy_actions)
            episode_starts = dones

        obs_sequence = torch.stack(sequence_observations, dim=0)
        action_sequence = torch.stack(sequence_actions, dim=0)
        start_sequence = torch.stack(sequence_starts, dim=0)
        hidden_state = model.initial_state(num_envs, device)
        latent, _ = model.forward_lstm(obs_sequence, hidden_state, start_sequence)
        predicted_actions = model.distribution(latent).mean
        if use_guidance_prior:
            bc_loss = F.mse_loss(predicted_actions, action_sequence)
        else:
            flat_predicted = predicted_actions.reshape(-1, env.action_dim)
            flat_target = action_sequence.reshape(-1, env.action_dim)
            cosine_loss = 1.0 - F.cosine_similarity(flat_predicted, flat_target, dim=1).mean()
            bc_loss = F.mse_loss(predicted_actions, action_sequence) + 0.35 * cosine_loss
        optimizer.zero_grad(set_to_none=True)
        bc_loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), ppo_config.max_grad_norm)
        optimizer.step()

        bc_loss_value = float(bc_loss.detach().cpu())
        smoothed_loss = bc_loss_value if smoothed_loss is None else 0.9 * smoothed_loss + 0.1 * bc_loss_value
        best_loss = min(best_loss, bc_loss_value)
        if smoothed_loss < best_loss_smoothed - bc_early_stop_min_delta:
            best_loss_smoothed = smoothed_loss
            plateau_count = 0
        else:
            plateau_count += 1
        progress.set_postfix({
            "loss": f"{bc_loss_value:.4f}",
            "smooth": f"{smoothed_loss:.4f}",
            "best": f"{best_loss:.4f}",
            "patience": str(plateau_count),
            "grad": f"{float(torch.as_tensor(grad_norm).detach().cpu()):.2f}",
        })
        if (
            bc_early_stopping
            and update >= bc_early_stop_min_updates
            and plateau_count >= bc_early_stop_patience
        ):
            progress.write(f"BC early stop at update {update} (smoothed loss {smoothed_loss:.5f}, best {best_loss_smoothed:.5f})")
            break


def evaluate(model: torch.nn.Module, obs_rms: TorchRunningMeanStd, device: torch.device, episodes: int, mode: str, current_difficulty_level: float, xgboost_teacher=None) -> dict:
    model.eval()
    if mode in {"xgboost", "ensemble"} and xgboost_teacher is None:
        model.train()
        return {
            "eval_success_rate": np.nan,
            "eval_mean_min_distance": np.nan,
            "eval_mean_reward": np.nan,
            "eval_mean_length": np.nan,
            "eval_mean_residual_norm": np.nan,
        }
    eval_envs = 512 if device.type == "cuda" else min(max(episodes, 1), 64)
    mode_offset = {"guided": 0, "expert": 10_000, "raw": 20_000, "xgboost": 30_000, "ensemble": 40_000}[mode]
    env = TorchProjectileInterceptVecEnv(num_envs=eval_envs, config=environment_config, seed=seed + 50_000 + mode_offset, device=device)
    env.set_curriculum(1.0)
    env.set_difficulty(current_difficulty_level)
    observations = env.reset()
    hidden_state = model.initial_state(eval_envs, device)
    episode_starts = torch.ones(eval_envs, dtype=torch.float32, device=device)
    successes = []
    min_distances = []
    rewards = []
    lengths = []
    residual_norms = []
    max_steps = int(environment_config.max_episode_time / environment_config.dt) + 64

    with torch.no_grad():
        while len(successes) < episodes:
            if mode == "expert":
                executed_actions = env.expert_action()
            elif mode == "xgboost":
                executed_actions = predict_xgboost_tensor(xgboost_teacher, observations, device)
            else:
                normalized_observations = obs_rms.normalize(observations)
                actions, _, hidden_state = model.deterministic_action(
                    normalized_observations,
                    hidden_state=hidden_state,
                    episode_starts=episode_starts,
                )
                residual_norms.append(float(torch.linalg.norm(actions, dim=1).mean().detach().cpu()))
                if mode == "guided":
                    executed_actions = guided_vector_action(env, actions)
                elif mode == "ensemble":
                    xgboost_actions = predict_xgboost_tensor(xgboost_teacher, observations, device)
                    executed_actions = blend_teacher_actions(normalize_action_tensor(actions), xgboost_actions)
                else:
                    executed_actions = normalize_action_tensor(actions)
            observations, _, dones, info = env.step(executed_actions)
            episode_starts = dones
            if info["done_count"] > 0:
                successes.extend(info["successes"].float().detach().cpu().tolist())
                min_distances.extend(info["min_distances"].detach().cpu().tolist())
                rewards.extend(info["episode_rewards"].detach().cpu().tolist())
                lengths.extend(info["episode_lengths"].detach().cpu().tolist())
            max_steps -= 1
            if max_steps <= 0:
                break

    model.train()
    successes = successes[:episodes]
    min_distances = min_distances[:episodes]
    rewards = rewards[:episodes]
    lengths = lengths[:episodes]
    return {
        "eval_success_rate": float(np.mean(successes)) if successes else 0.0,
        "eval_mean_min_distance": float(np.mean(min_distances)) if min_distances else np.nan,
        "eval_mean_reward": float(np.mean(rewards)) if rewards else np.nan,
        "eval_mean_length": float(np.mean(lengths)) if lengths else np.nan,
        "eval_mean_residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
    }


def evaluate_ablations(model: torch.nn.Module, obs_rms: TorchRunningMeanStd, device: torch.device, episodes: int, current_difficulty_level: float, xgboost_teacher=None) -> dict:
    guided = evaluate(model, obs_rms, device, episodes, "guided", current_difficulty_level, xgboost_teacher)
    expert = evaluate(model, obs_rms, device, episodes, "expert", current_difficulty_level, xgboost_teacher)
    raw = evaluate(model, obs_rms, device, episodes, "raw", current_difficulty_level, xgboost_teacher)
    xgboost = evaluate(model, obs_rms, device, episodes, "xgboost", current_difficulty_level, xgboost_teacher)
    ensemble = evaluate(model, obs_rms, device, episodes, "ensemble", current_difficulty_level, xgboost_teacher)
    primary = ensemble if np.isfinite(ensemble["eval_success_rate"]) else guided
    return {
        "eval_success_rate": primary["eval_success_rate"],
        "eval_mean_min_distance": primary["eval_mean_min_distance"],
        "eval_mean_reward": primary["eval_mean_reward"],
        "eval_mean_length": primary["eval_mean_length"],
        "eval_guided_success_rate": guided["eval_success_rate"],
        "eval_guided_mean_min_distance": guided["eval_mean_min_distance"],
        "eval_expert_success_rate": expert["eval_success_rate"],
        "eval_expert_mean_min_distance": expert["eval_mean_min_distance"],
        "eval_xgboost_success_rate": xgboost["eval_success_rate"],
        "eval_xgboost_mean_min_distance": xgboost["eval_mean_min_distance"],
        "eval_ensemble_success_rate": ensemble["eval_success_rate"],
        "eval_ensemble_mean_min_distance": ensemble["eval_mean_min_distance"],
        "eval_raw_success_rate": raw["eval_success_rate"],
        "eval_raw_mean_min_distance": raw["eval_mean_min_distance"],
        "eval_mean_residual_norm": raw["eval_mean_residual_norm"],
    }


def plot_deterministic_trajectory(model: torch.nn.Module, obs_rms: TorchRunningMeanStd, device: torch.device, output_path: Path, xgboost_teacher=None) -> None:
    model.eval()
    env = ProjectileInterceptEnv(config=environment_config, seed=seed + 90_000, record_history=True)
    observation = env.reset()
    hidden_state = model.initial_state(1, device)
    episode_start = torch.ones(1, dtype=torch.float32, device=device)
    done = False

    with torch.no_grad():
        while not done:
            obs_tensor = torch.as_tensor(observation.reshape(1, -1), dtype=torch.float32, device=device)
            normalized_observation = obs_rms.normalize(obs_tensor)
            action, _, hidden_state = model.deterministic_action(
                normalized_observation,
                hidden_state=hidden_state,
                episode_starts=episode_start,
            )
            policy_action = guided_scalar_action(env, action.squeeze(0).detach().cpu().numpy())
            if xgboost_teacher is not None:
                xgboost_action = xgboost_teacher.predict(observation.reshape(1, -1).astype(np.float32))[0]
                executed_action = blend_numpy_actions(policy_action, xgboost_action)
            else:
                executed_action = policy_action
            observation, _, done, _ = env.step(executed_action)
            episode_start[0] = float(done)

    env.plot_trajectory(save_path=output_path, show=False)
    plt.close("all")
    model.train()


def train() -> None:
    configure_torch(seed)
    device = torch.device(device_name)
    log_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    plot_directory.mkdir(parents=True, exist_ok=True)

    xgboost_teacher, xgboost_metrics = prepare_xgboost_teacher(device)
    if xgboost_metrics:
        print(f"XGBoost teacher: cosine={xgboost_metrics.get('xgboost_valid_cosine', np.nan):.4f}, mse={xgboost_metrics.get('xgboost_valid_mse', np.nan):.5f}")

    env = TorchProjectileInterceptVecEnv(num_envs=num_envs, config=environment_config, seed=seed, device=device)
    observation_dim = env.observation_dim
    action_dim = env.action_dim
    model = create_model(observation_dim, action_dim, device)
    obs_rms = TorchRunningMeanStd((observation_dim,), device=device)

    run_behavior_cloning(model, obs_rms, device, xgboost_teacher)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    ppo = RecurrentPPO(model, optimizer, ppo_config, device)
    buffer = TorchRolloutBuffer(rollout_steps, num_envs, observation_dim, action_dim, device)
    current_difficulty_level = difficulty_level
    curriculum_progress = curriculum_initial_progress
    env.set_curriculum(curriculum_progress)
    env.set_difficulty(current_difficulty_level)
    observations = env.reset()
    episode_starts = torch.ones(num_envs, dtype=torch.float32, device=device)
    hidden_state = model.initial_state(num_envs, device)
    early_stop_counter = 0
    plateau_counter = 0
    recent_rewards = deque(maxlen=1000)
    recent_lengths = deque(maxlen=1000)
    recent_successes = deque(maxlen=1000)
    recent_min_distances = deque(maxlen=1000)
    recent_path_lengths = deque(maxlen=1000)
    recent_action_energies = deque(maxlen=1000)
    recent_distance_integrals = deque(maxlen=1000)
    log_path = log_directory / f"{run_name}.csv"
    write_log_header(log_path)

    best_eval_success = -np.inf
    best_eval_distance = np.inf
    global_step = 0
    latest_row = {}
    last_eval_metrics = {
        "eval_success_rate": np.nan,
        "eval_mean_min_distance": np.nan,
        "eval_mean_reward": np.nan,
        "eval_mean_length": np.nan,
        "eval_guided_success_rate": np.nan,
        "eval_guided_mean_min_distance": np.nan,
        "eval_expert_success_rate": np.nan,
        "eval_expert_mean_min_distance": np.nan,
        "eval_xgboost_success_rate": np.nan,
        "eval_xgboost_mean_min_distance": np.nan,
        "eval_ensemble_success_rate": np.nan,
        "eval_ensemble_mean_min_distance": np.nan,
        "eval_raw_success_rate": np.nan,
        "eval_raw_mean_min_distance": np.nan,
        "eval_mean_residual_norm": np.nan,
    }
    progress = tqdm(range(1, total_updates + 1), dynamic_ncols=True, desc="PPO")

    for update in progress:
        env.set_curriculum(curriculum_progress)
        env.set_difficulty(current_difficulty_level)
        current_learning_rate = compute_learning_rate(update)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_learning_rate

        buffer.reset()
        rollout_initial_hidden = (hidden_state[0].detach().clone(), hidden_state[1].detach().clone())
        residual_norm_total = 0.0
        residual_norm_count = 0
        rollout_success_count = 0.0
        rollout_done_count = 0
        rollout_reward_total = 0.0
        rollout_reward_count = 0

        for _ in range(rollout_steps):
            obs_rms.update(observations)
            normalized_observations = obs_rms.normalize(observations)
            with torch.no_grad():
                residual_actions, log_probs, _, values, hidden_state = model.get_action_and_value(
                    normalized_observations,
                    hidden_state=hidden_state,
                    episode_starts=episode_starts,
                )

            executed_actions = guided_vector_action(env, residual_actions)
            residual_norm_total += float(torch.linalg.norm(residual_actions, dim=1).mean().detach().cpu())
            residual_norm_count += 1
            next_observations, rewards, dones, info = env.step(executed_actions)
            rollout_reward_total += float(rewards.mean().detach().cpu())
            rollout_reward_count += 1
            buffer.add(normalized_observations, residual_actions, log_probs, rewards, dones, values, episode_starts)
            observations = next_observations
            episode_starts = dones
            global_step += num_envs

            if info["done_count"] > 0:
                rollout_success_count += float(info["successes"].float().sum().detach().cpu())
                rollout_done_count += int(info["done_count"])
                recent_rewards.extend(info["episode_rewards"].detach().cpu().tolist())
                recent_lengths.extend(info["episode_lengths"].detach().cpu().tolist())
                recent_successes.extend(info["successes"].float().detach().cpu().tolist())
                recent_min_distances.extend(info["min_distances"].detach().cpu().tolist())
                recent_path_lengths.extend(info["path_lengths"].detach().cpu().tolist())
                recent_action_energies.extend(info["action_energies"].detach().cpu().tolist())
                recent_distance_integrals.extend(info["distance_integrals"].detach().cpu().tolist())

        obs_rms.update(observations)
        normalized_observations = obs_rms.normalize(observations)
        with torch.no_grad():
            _, _, _, last_values, _ = model.get_action_and_value(
                normalized_observations,
                hidden_state=hidden_state,
                episode_starts=episode_starts,
            )
        buffer.compute_returns_and_advantages(last_values, episode_starts, ppo_config.gamma, ppo_config.gae_lambda)
        metrics = ppo.update(buffer, rollout_initial_hidden)
        hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

        if update == 1 or update % evaluation_interval == 0:
            last_eval_metrics = evaluate_ablations(model, obs_rms, device, evaluation_episodes, current_difficulty_level, xgboost_teacher)
        eval_metrics = last_eval_metrics

        rollout_mean_step_reward = rollout_reward_total / max(rollout_reward_count, 1)
        mean_reward = float(np.mean(recent_rewards)) if recent_rewards else float(env.episode_reward.mean().detach().cpu())
        if not np.isfinite(mean_reward):
            mean_reward = rollout_mean_step_reward
        mean_length = float(np.mean(recent_lengths)) if recent_lengths else float(env.episode_length.float().mean().detach().cpu())
        success_rate = float(np.mean(recent_successes)) if recent_successes else 0.0
        mean_min_distance = float(np.mean(recent_min_distances)) if recent_min_distances else float(env.min_distance.mean().detach().cpu())
        mean_path_length = float(np.mean(recent_path_lengths)) if recent_path_lengths else float(env.interceptor_path_length.mean().detach().cpu())
        mean_action_energy = float(np.mean(recent_action_energies)) if recent_action_energies else float(env.action_energy.mean().detach().cpu())
        mean_distance_integral = float(np.mean(recent_distance_integrals)) if recent_distance_integrals else float(env.distance_integral.mean().detach().cpu())
        intercept_radius = float(env.current_intercept_radius)
        mean_residual_norm = residual_norm_total / max(residual_norm_count, 1)
        rollout_terminal_success_rate = rollout_success_count / max(rollout_done_count, 1)
        rollout_hit_rate = rollout_terminal_success_rate

        latest_row = {
            "update": update,
            "global_step": global_step,
            "learning_rate": current_learning_rate,
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "success_rate": success_rate,
            "rollout_hit_rate": rollout_hit_rate,
            "rollout_terminal_success_rate": rollout_terminal_success_rate,
            "mean_min_distance": mean_min_distance,
            "eval_success_rate": eval_metrics["eval_success_rate"],
            "eval_mean_min_distance": eval_metrics["eval_mean_min_distance"],
            "eval_guided_success_rate": eval_metrics["eval_guided_success_rate"],
            "eval_guided_mean_min_distance": eval_metrics["eval_guided_mean_min_distance"],
            "eval_expert_success_rate": eval_metrics["eval_expert_success_rate"],
            "eval_expert_mean_min_distance": eval_metrics["eval_expert_mean_min_distance"],
            "eval_xgboost_success_rate": eval_metrics["eval_xgboost_success_rate"],
            "eval_xgboost_mean_min_distance": eval_metrics["eval_xgboost_mean_min_distance"],
            "eval_ensemble_success_rate": eval_metrics["eval_ensemble_success_rate"],
            "eval_ensemble_mean_min_distance": eval_metrics["eval_ensemble_mean_min_distance"],
            "eval_raw_success_rate": eval_metrics["eval_raw_success_rate"],
            "eval_raw_mean_min_distance": eval_metrics["eval_raw_mean_min_distance"],
            "mean_residual_norm": mean_residual_norm,
            "mean_path_length": mean_path_length,
            "mean_action_energy": mean_action_energy,
            "mean_distance_integral": mean_distance_integral,
            "difficulty_level": current_difficulty_level,
            "launch_distance_max": float(env.current_launch_distance_max),
            "interceptor_max_accel": float(env.current_interceptor_max_accel),
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "approx_kl": metrics["approx_kl"],
            "clip_fraction": metrics["clip_fraction"],
            "grad_norm": metrics["grad_norm"],
            "explained_variance": metrics["explained_variance"],
            "intercept_radius": intercept_radius,
        }
        append_log(log_path, latest_row)

        progress.set_postfix({
            "rew": f"{mean_reward:.0f}" if np.isfinite(mean_reward) else "nan",
            "term": f"{rollout_terminal_success_rate:.2f}",
            "dist": f"{mean_min_distance:.1f}" if np.isfinite(mean_min_distance) else "nan",
            "guided": f"{eval_metrics['eval_guided_success_rate']:.2f}" if np.isfinite(eval_metrics["eval_guided_success_rate"]) else "nan",
            "raw": f"{eval_metrics['eval_raw_success_rate']:.2f}" if np.isfinite(eval_metrics["eval_raw_success_rate"]) else "nan",
            "ens": f"{eval_metrics['eval_ensemble_success_rate']:.2f}" if np.isfinite(eval_metrics["eval_ensemble_success_rate"]) else "nan",
            "exp": f"{eval_metrics['eval_expert_success_rate']:.2f}" if np.isfinite(eval_metrics["eval_expert_success_rate"]) else "nan",
            "path": f"{mean_path_length:.0f}" if np.isfinite(mean_path_length) else "nan",
            "resid": f"{mean_residual_norm:.2f}",
            "curr": f"{curriculum_progress:.2f}",
            "lvl": f"{current_difficulty_level:.1f}",
            "r": f"{intercept_radius:.1f}",
            "lr": f"{current_learning_rate:.2e}",
            "kl": f"{metrics['approx_kl']:.4f}",
            "es": f"{early_stop_counter}",
        })

        if update % checkpoint_interval == 0:
            save_checkpoint(checkpoint_directory / run_name / f"checkpoint_{update:05d}.pt", model, optimizer, obs_rms, update, global_step, latest_row)
            save_checkpoint(checkpoint_directory / run_name / "latest.pt", model, optimizer, obs_rms, update, global_step, latest_row)

        eval_success = eval_metrics["eval_success_rate"]
        eval_distance = eval_metrics["eval_mean_min_distance"]
        if np.isfinite(eval_success) and (eval_success > best_eval_success or (eval_success == best_eval_success and eval_distance < best_eval_distance)):
            best_eval_success = eval_success
            best_eval_distance = eval_distance
            save_checkpoint(checkpoint_directory / run_name / "best.pt", model, optimizer, obs_rms, update, global_step, latest_row)

        if update % plot_interval == 0:
            plot_training_summary(log_path, plot_directory / run_name / "training_summary.png")
            plot_deterministic_trajectory(model, obs_rms, device, plot_directory / run_name / "latest_eval_trajectory.png", xgboost_teacher)

        guided_metric = eval_metrics["eval_guided_success_rate"]
        if not np.isfinite(guided_metric):
            guided_metric = eval_metrics["eval_ensemble_success_rate"]
        if adaptive_curriculum and update % difficulty_adjust_interval == 0 and np.isfinite(guided_metric):
            if guided_metric >= curriculum_threshold_hi:
                progress_step = curriculum_progress_step_hi
                level_step = difficulty_step_hi
            elif guided_metric >= curriculum_threshold_lo:
                progress_step = curriculum_progress_step_lo
                level_step = difficulty_step_lo
            else:
                progress_step = 0.0
                level_step = 0.0
            if progress_step > 0.0:
                if curriculum_progress < 1.0:
                    new_progress = min(1.0, curriculum_progress + progress_step)
                    if new_progress > curriculum_progress:
                        curriculum_progress = new_progress
                        env.set_curriculum(curriculum_progress)
                        progress.write(f"curriculum -> {curriculum_progress:.3f} (guided {guided_metric:.3f})")
                elif current_difficulty_level < max_difficulty_level:
                    new_level = min(max_difficulty_level, current_difficulty_level + level_step)
                    if new_level > current_difficulty_level:
                        current_difficulty_level = new_level
                        env.set_difficulty(current_difficulty_level)
                        observations = env.reset()
                        episode_starts = torch.ones(num_envs, dtype=torch.float32, device=device)
                        hidden_state = model.initial_state(num_envs, device)
                        recent_rewards.clear()
                        recent_lengths.clear()
                        recent_successes.clear()
                        recent_min_distances.clear()
                        recent_path_lengths.clear()
                        recent_action_energies.clear()
                        recent_distance_integrals.clear()
                        progress.write(f"difficulty -> {current_difficulty_level:.2f} (guided {guided_metric:.3f})")
                        last_eval_metrics = evaluate_ablations(model, obs_rms, device, evaluation_episodes, current_difficulty_level, xgboost_teacher)

        if early_stopping and (update == 1 or update % evaluation_interval == 0):
            if np.isfinite(guided_metric) and guided_metric >= early_stop_min_guided_success:
                early_stop_counter += 1
            else:
                early_stop_counter = 0
            if np.isfinite(eval_success) and eval_success >= best_eval_success - 1e-6:
                plateau_counter = 0
            else:
                plateau_counter += 1
            if (
                early_stop_counter >= early_stop_patience_evals
                and current_difficulty_level >= early_stop_min_difficulty
                and curriculum_progress >= 1.0
            ):
                progress.write(
                    f"Early stopping: guided={guided_metric:.3f} >= {early_stop_min_guided_success:.3f} "
                    f"for {early_stop_counter} evals at difficulty {current_difficulty_level:.1f}"
                )
                break
            if plateau_counter >= plateau_patience_evals:
                progress.write(f"Plateau early stop: no improvement for {plateau_counter} evals (best={best_eval_success:.3f})")
                break

    save_checkpoint(checkpoint_directory / run_name / "final.pt", model, optimizer, obs_rms, total_updates, global_step, latest_row)
    plot_training_summary(log_path, plot_directory / run_name / "training_summary.png")


if __name__ == "__main__":
    train()
