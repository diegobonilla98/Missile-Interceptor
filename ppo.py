from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_


@dataclass
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.18
    value_clip_coef: float = 0.18
    entropy_coef: float = 0.01
    value_coef: float = 0.55
    max_grad_norm: float = 0.8
    update_epochs: int = 5
    num_minibatches: int = 4
    target_kl: float = 0.035
    normalize_advantages: bool = True
    clip_value_loss: bool = True
    normalize_value_targets: bool = True


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...] | int, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        batch_mean = np.mean(values, axis=0)
        batch_var = np.var(values, axis=0)
        batch_count = values.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = float(total_count)

    def normalize(self, values: np.ndarray, clip: float = 8.0) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        normalized = (values - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class TorchRunningMeanStd:
    def __init__(self, shape: tuple[int, ...] | int, device: torch.device, epsilon: float = 1e-4):
        self.device = device
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(float(epsilon), dtype=torch.float32, device=device)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(self.device, dtype=torch.float32)
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = torch.tensor(values.shape[0], dtype=torch.float32, device=self.device)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: torch.Tensor) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.square() * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def normalize(self, values: torch.Tensor, clip: float = 8.0) -> torch.Tensor:
        values = values.to(self.device, dtype=torch.float32)
        normalized = (values - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(normalized, -clip, clip)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": float(self.count.detach().cpu()),
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = torch.as_tensor(state["mean"], dtype=torch.float32, device=self.device)
        self.var = torch.as_tensor(state["var"], dtype=torch.float32, device=self.device)
        self.count = torch.tensor(float(state["count"]), dtype=torch.float32, device=self.device)


class RolloutBuffer:
    def __init__(self, rollout_steps: int, num_envs: int, observation_dim: int, action_dim: int):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.rollout_steps, self.num_envs, self.observation_dim), dtype=np.float32)
        self.actions = np.zeros((self.rollout_steps, self.num_envs, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.advantages = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.position = 0

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        episode_starts: np.ndarray,
    ) -> None:
        self.observations[self.position] = observations
        self.actions[self.position] = actions
        self.log_probs[self.position] = log_probs
        self.rewards[self.position] = rewards
        self.dones[self.position] = dones
        self.values[self.position] = values
        self.episode_starts[self.position] = episode_starts
        self.position += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray, last_dones: np.ndarray, gamma: float, gae_lambda: float) -> None:
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_nonterminal = 1.0 - last_dones.astype(np.float32)
                next_values = last_values.astype(np.float32)
            else:
                next_nonterminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_values * next_nonterminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values

    def recurrent_minibatches(
        self,
        advantages: np.ndarray,
        initial_hidden_state: tuple[torch.Tensor, torch.Tensor],
        num_minibatches: int,
        device: torch.device,
    ):
        h, c = initial_hidden_state
        env_indices = np.random.permutation(self.num_envs)
        minibatch_envs = max(1, self.num_envs // num_minibatches)
        for start in range(0, self.num_envs, minibatch_envs):
            indices = env_indices[start : start + minibatch_envs]
            hidden_indices = torch.as_tensor(indices, dtype=torch.long, device=h.device)
            yield {
                "observations": torch.as_tensor(self.observations[:, indices], dtype=torch.float32, device=device),
                "actions": torch.as_tensor(self.actions[:, indices], dtype=torch.float32, device=device),
                "old_log_probs": torch.as_tensor(self.log_probs[:, indices], dtype=torch.float32, device=device),
                "advantages": torch.as_tensor(advantages[:, indices], dtype=torch.float32, device=device),
                "returns": torch.as_tensor(self.returns[:, indices], dtype=torch.float32, device=device),
                "old_values": torch.as_tensor(self.values[:, indices], dtype=torch.float32, device=device),
                "episode_starts": torch.as_tensor(self.episode_starts[:, indices], dtype=torch.float32, device=device),
                "hidden_state": (h.index_select(1, hidden_indices).contiguous().to(device), c.index_select(1, hidden_indices).contiguous().to(device)),
            }


class TorchRolloutBuffer:
    def __init__(self, rollout_steps: int, num_envs: int, observation_dim: int, action_dim: int, device: torch.device):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.observations = torch.zeros((self.rollout_steps, self.num_envs, self.observation_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.rollout_steps, self.num_envs, self.action_dim), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.episode_starts = torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((self.rollout_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.position = 0

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> None:
        self.observations[self.position].copy_(observations.detach())
        self.actions[self.position].copy_(actions.detach())
        self.log_probs[self.position].copy_(log_probs.detach())
        self.rewards[self.position].copy_(rewards.detach())
        self.dones[self.position].copy_(dones.detach())
        self.values[self.position].copy_(values.detach())
        self.episode_starts[self.position].copy_(episode_starts.detach())
        self.position += 1

    def compute_returns_and_advantages(self, last_values: torch.Tensor, last_dones: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_nonterminal = 1.0 - last_dones.float()
                next_values = last_values.float()
            else:
                next_nonterminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_values * next_nonterminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            self.advantages[step] = last_gae
        self.returns.copy_(self.advantages + self.values)

    def recurrent_minibatches(
        self,
        advantages: torch.Tensor,
        initial_hidden_state: tuple[torch.Tensor, torch.Tensor],
        num_minibatches: int,
        device: torch.device,
    ):
        h, c = initial_hidden_state
        env_indices = torch.randperm(self.num_envs, device=self.device)
        minibatch_envs = max(1, self.num_envs // num_minibatches)
        for start in range(0, self.num_envs, minibatch_envs):
            indices = env_indices[start : start + minibatch_envs]
            hidden_indices = indices.to(h.device)
            yield {
                "observations": self.observations.index_select(1, indices),
                "actions": self.actions.index_select(1, indices),
                "old_log_probs": self.log_probs.index_select(1, indices),
                "advantages": advantages.index_select(1, indices),
                "returns": self.returns.index_select(1, indices),
                "old_values": self.values.index_select(1, indices),
                "episode_starts": self.episode_starts.index_select(1, indices),
                "hidden_state": (h.index_select(1, hidden_indices).contiguous().to(device), c.index_select(1, hidden_indices).contiguous().to(device)),
            }


class RecurrentPPO:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: PPOConfig, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device

    def update(self, buffer: RolloutBuffer, initial_hidden_state: tuple[torch.Tensor, torch.Tensor]) -> dict:
        if isinstance(buffer.advantages, torch.Tensor):
            advantages = buffer.advantages.detach().clone()
        else:
            advantages = buffer.advantages.copy()
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if self.config.normalize_value_targets:
            if isinstance(buffer.returns, torch.Tensor):
                value_target_mean = buffer.returns.detach().mean()
                value_target_std = buffer.returns.detach().std(unbiased=False).clamp_min(1.0)
            else:
                returns_tensor = torch.as_tensor(buffer.returns, dtype=torch.float32, device=self.device)
                value_target_mean = returns_tensor.mean()
                value_target_std = returns_tensor.std(unbiased=False).clamp_min(1.0)
        else:
            value_target_mean = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            value_target_std = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_fraction = 0.0
        total_grad_norm = 0.0
        batches = 0
        stop_early = False

        for _ in range(self.config.update_epochs):
            for batch in buffer.recurrent_minibatches(advantages, initial_hidden_state, self.config.num_minibatches, self.device):
                _, new_log_probs, entropy, new_values, _ = self.model.get_action_and_value(
                    batch["observations"],
                    hidden_state=batch["hidden_state"],
                    episode_starts=batch["episode_starts"],
                    actions=batch["actions"],
                )

                log_ratio = new_log_probs - batch["old_log_probs"]
                ratio = torch.exp(log_ratio)
                policy_loss_unclipped = -batch["advantages"] * ratio
                policy_loss_clipped = -batch["advantages"] * torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef)
                policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

                scaled_new_values = (new_values - value_target_mean) / value_target_std
                scaled_old_values = (batch["old_values"] - value_target_mean) / value_target_std
                scaled_returns = (batch["returns"] - value_target_mean) / value_target_std

                if self.config.clip_value_loss:
                    value_pred_clipped = scaled_old_values + torch.clamp(scaled_new_values - scaled_old_values, -self.config.value_clip_coef, self.config.value_clip_coef)
                    value_loss_unclipped = torch.square(scaled_new_values - scaled_returns)
                    value_loss_clipped = torch.square(value_pred_clipped - scaled_returns)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * torch.square(scaled_new_values - scaled_returns).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()

                total_policy_loss += float(policy_loss.detach().cpu())
                total_value_loss += float(value_loss.detach().cpu())
                total_entropy += float(entropy_loss.detach().cpu())
                total_kl += float(approx_kl.detach().cpu())
                total_clip_fraction += float(clip_fraction.detach().cpu())
                total_grad_norm += float(torch.as_tensor(grad_norm).detach().cpu())
                batches += 1

                if self.config.target_kl is not None and float(approx_kl.detach().cpu()) > self.config.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        if isinstance(buffer.values, torch.Tensor):
            explained_variance = self.explained_variance_tensor(buffer.values.reshape(-1), buffer.returns.reshape(-1))
        else:
            explained_variance = self.explained_variance(buffer.values.reshape(-1), buffer.returns.reshape(-1))
        batches = max(1, batches)
        return {
            "policy_loss": total_policy_loss / batches,
            "value_loss": total_value_loss / batches,
            "entropy": total_entropy / batches,
            "approx_kl": total_kl / batches,
            "clip_fraction": total_clip_fraction / batches,
            "grad_norm": total_grad_norm / batches,
            "explained_variance": explained_variance,
            "batches": batches,
            "stopped_early": stop_early,
        }

    def explained_variance(self, values: np.ndarray, returns: np.ndarray) -> float:
        variance_returns = np.var(returns)
        if variance_returns < 1e-12:
            return 0.0
        return float(1.0 - np.var(returns - values) / variance_returns)

    def explained_variance_tensor(self, values: torch.Tensor, returns: torch.Tensor) -> float:
        variance_returns = torch.var(returns)
        if float(variance_returns.detach().cpu()) < 1e-12:
            return 0.0
        explained = 1.0 - torch.var(returns - values) / variance_returns
        return float(explained.detach().cpu())


def linear_schedule(initial_value: float, current_update: int, total_updates: int) -> float:
    fraction = 1.0 - (current_update - 1.0) / max(total_updates, 1)
    return float(fraction * initial_value)
