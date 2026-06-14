from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import torch
from xgboost import XGBRegressor

from environment import EnvironmentConfig, TorchProjectileInterceptVecEnv


def set_xgboost_prediction_device(model: MultiOutputRegressor, device_name: str) -> None:
    if not hasattr(model, "estimators_"):
        return
    for estimator in model.estimators_:
        estimator.set_params(device=device_name)
        estimator.get_booster().set_param({"device": device_name})


def normalize_numpy_actions(actions: np.ndarray) -> np.ndarray:
    actions = np.clip(actions, -1.0, 1.0)
    norms = np.linalg.norm(actions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1.0)
    return actions / norms


def normalize_torch_actions(actions: torch.Tensor) -> torch.Tensor:
    actions = torch.clamp(actions, -1.0, 1.0)
    norms = torch.linalg.norm(actions, dim=1, keepdim=True).clamp_min(1.0)
    return actions / norms


def collect_guidance_dataset(
    config: EnvironmentConfig,
    samples: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    difficulty_levels: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    remaining = samples
    per_level = max(1, samples // max(len(difficulty_levels), 1))

    for level_index, difficulty_level_value in enumerate(difficulty_levels):
        env = TorchProjectileInterceptVecEnv(num_envs=batch_size, config=config, seed=seed + level_index * 10_003, device=device)
        env.set_curriculum(1.0)
        env.set_difficulty(difficulty_level_value)
        observations = env.reset()
        level_remaining = min(per_level, remaining)
        while level_remaining > 0:
            expert_actions = env.expert_action()
            take = min(level_remaining, batch_size)
            xs.append(observations[:take].detach().cpu().numpy())
            ys.append(expert_actions[:take].detach().cpu().numpy())
            exploration = normalize_torch_actions(expert_actions + 0.08 * torch.randn_like(expert_actions))
            observations, _, _, _ = env.step(exploration)
            level_remaining -= take
            remaining -= take
            if remaining <= 0:
                break
        if remaining <= 0:
            break

    x = np.concatenate(xs, axis=0).astype(np.float32)[:samples]
    y = normalize_numpy_actions(np.concatenate(ys, axis=0).astype(np.float32)[:samples])
    return x, y


def train_xgboost_guidance(
    features: np.ndarray,
    targets: np.ndarray,
    seed: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> tuple[MultiOutputRegressor, dict]:
    x_train, x_valid, y_train, y_valid = train_test_split(features, targets, test_size=0.12, random_state=seed)
    base_model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.92,
        colsample_bytree=0.92,
        min_child_weight=2.0,
        reg_lambda=1.25,
        reg_alpha=0.02,
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_jobs=-1,
        random_state=seed,
    )
    model = MultiOutputRegressor(base_model, n_jobs=1)
    model.fit(x_train, y_train)
    set_xgboost_prediction_device(model, "cpu")
    predicted = normalize_numpy_actions(model.predict(x_valid).astype(np.float32))
    metrics = {
        "xgboost_valid_mse": float(mean_squared_error(y_valid, predicted)),
        "xgboost_valid_cosine": float(np.mean(np.sum(y_valid * predicted, axis=1) / np.maximum(np.linalg.norm(y_valid, axis=1) * np.linalg.norm(predicted, axis=1), 1e-8))),
        "xgboost_train_samples": int(x_train.shape[0]),
        "xgboost_valid_samples": int(x_valid.shape[0]),
    }
    return model, metrics


def save_xgboost_model(path: Path, model: MultiOutputRegressor, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics}, path)


def load_xgboost_model(path: Path) -> tuple[MultiOutputRegressor, dict]:
    payload = joblib.load(path)
    model = payload["model"]
    set_xgboost_prediction_device(model, "cpu")
    return model, payload.get("metrics", {})


def predict_xgboost_tensor(model: MultiOutputRegressor, observations: torch.Tensor, device: torch.device) -> torch.Tensor:
    features = observations.detach().cpu().numpy().astype(np.float32)
    predictions = normalize_numpy_actions(model.predict(features).astype(np.float32))
    return torch.as_tensor(predictions, dtype=torch.float32, device=device)
