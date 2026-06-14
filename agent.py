import math

import torch
import torch.nn as nn
from torch.distributions import Normal


class LSTMActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        lstm_layers: int = 1,
        actor_hidden_size: int = 256,
        critic_hidden_size: int = 256,
        initial_log_std: float = -0.55,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, lstm_layers)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, actor_hidden_size),
            nn.Tanh(),
            nn.Linear(actor_hidden_size, actor_hidden_size),
            nn.Tanh(),
            nn.Linear(actor_hidden_size, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, critic_hidden_size),
            nn.Tanh(),
            nn.Linear(critic_hidden_size, critic_hidden_size),
            nn.Tanh(),
            nn.Linear(critic_hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))
        self.apply(self._orthogonal_init)
        nn.init.constant_(self.actor[-1].bias, 0.0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.critic[-1].bias, 0.0)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def initial_state(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward_lstm(
        self,
        observations: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None,
        episode_starts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        single_step = observations.dim() == 2
        if single_step:
            observations = observations.unsqueeze(0)

        sequence_length, batch_size, _ = observations.shape
        if hidden_state is None:
            hidden_state = self.initial_state(batch_size, observations.device)

        features = self.encoder(observations.reshape(sequence_length * batch_size, -1))
        features = features.reshape(sequence_length, batch_size, self.hidden_size)

        h, c = hidden_state
        outputs = []
        if episode_starts is None:
            lstm_output, (h, c) = self.lstm(features, (h, c))
            outputs = [lstm_output]
        else:
            episode_starts = episode_starts.float()
            if episode_starts.dim() == 1:
                episode_starts = episode_starts.unsqueeze(0)
            if sequence_length == 1:
                keep = (1.0 - episode_starts[0]).reshape(1, batch_size, 1)
                h = h * keep
                c = c * keep
                output, (h, c) = self.lstm(features, (h, c))
                outputs = [output]
            elif not bool(episode_starts.any().detach().cpu()):
                lstm_output, (h, c) = self.lstm(features, (h, c))
                outputs = [lstm_output]
            else:
                reset_steps = torch.nonzero(episode_starts[1:].any(dim=1), as_tuple=False).flatten() + 1
                boundaries = [0] + reset_steps.detach().cpu().tolist() + [sequence_length]
                for segment_start, segment_end in zip(boundaries[:-1], boundaries[1:]):
                    if segment_end <= segment_start:
                        continue
                    keep = (1.0 - episode_starts[segment_start]).reshape(1, batch_size, 1)
                    h = h * keep
                    c = c * keep
                    output, (h, c) = self.lstm(features[segment_start:segment_end], (h, c))
                    outputs.append(output)
            if not outputs:
                for step in range(sequence_length):
                    keep = (1.0 - episode_starts[step]).reshape(1, batch_size, 1)
                    h = h * keep
                    c = c * keep
                    output, (h, c) = self.lstm(features[step : step + 1], (h, c))
                    outputs.append(output)

        output = torch.cat(outputs, dim=0)
        if single_step:
            output = output.squeeze(0)
        return output, (h, c)

    def distribution(self, latent: torch.Tensor) -> Normal:
        mean = torch.tanh(self.actor(latent))
        log_std = torch.clamp(self.log_std, min=-5.0, max=1.2)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, latent: torch.Tensor) -> torch.Tensor:
        return self.critic(latent).squeeze(-1)

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        episode_starts: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        latent, new_hidden_state = self.forward_lstm(observations, hidden_state, episode_starts)
        dist = self.distribution(latent)
        if actions is None:
            if deterministic:
                actions = dist.mean
            else:
                actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.value(latent)
        return actions, log_prob, entropy, values, new_hidden_state

    def deterministic_action(
        self,
        observations: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        actions, _, _, values, new_hidden_state = self.get_action_and_value(
            observations,
            hidden_state=hidden_state,
            episode_starts=episode_starts,
            deterministic=True,
        )
        return actions, values, new_hidden_state

    def _orthogonal_init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            gain = math.sqrt(2.0)
            nn.init.orthogonal_(module.weight, gain=gain)
            nn.init.constant_(module.bias, 0.0)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.net(values)


class HybridActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_size: int = 384,
        lstm_layers: int = 2,
        transformer_layers: int = 2,
        attention_heads: int = 8,
        actor_hidden_size: int = 384,
        critic_hidden_size: int = 384,
        initial_log_std: float = -1.6,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        self.input_projection = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            ResidualMLPBlock(hidden_size, dropout),
            ResidualMLPBlock(hidden_size, dropout),
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, lstm_layers, dropout=dropout if lstm_layers > 1 else 0.0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.temporal_attention = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers, enable_nested_tensor=False)
        self.actor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, actor_hidden_size),
            nn.SiLU(),
            ResidualMLPBlock(actor_hidden_size, dropout),
            nn.Linear(actor_hidden_size, actor_hidden_size),
            nn.SiLU(),
            nn.Linear(actor_hidden_size, action_dim),
        )
        self.critic = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, critic_hidden_size),
            nn.SiLU(),
            ResidualMLPBlock(critic_hidden_size, dropout),
            nn.Linear(critic_hidden_size, critic_hidden_size),
            nn.SiLU(),
            nn.Linear(critic_hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))
        self.apply(self._orthogonal_init)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0.0)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0.0)

    def initial_state(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward_lstm(
        self,
        observations: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None,
        episode_starts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        single_step = observations.dim() == 2
        if single_step:
            observations = observations.unsqueeze(0)

        sequence_length, batch_size, _ = observations.shape
        if hidden_state is None:
            hidden_state = self.initial_state(batch_size, observations.device)

        features = self.input_projection(observations.reshape(sequence_length * batch_size, -1))
        features = features.reshape(sequence_length, batch_size, self.hidden_size)
        h, c = hidden_state
        outputs = []

        if episode_starts is None:
            lstm_output, (h, c) = self.lstm(features, (h, c))
            outputs = [lstm_output]
        else:
            episode_starts = episode_starts.float()
            if episode_starts.dim() == 1:
                episode_starts = episode_starts.unsqueeze(0)
            if sequence_length == 1:
                keep = (1.0 - episode_starts[0]).reshape(1, batch_size, 1)
                h = h * keep
                c = c * keep
                output, (h, c) = self.lstm(features, (h, c))
                outputs = [output]
            elif not bool(episode_starts.any().detach().cpu()):
                output, (h, c) = self.lstm(features, (h, c))
                outputs = [output]
            else:
                reset_steps = torch.nonzero(episode_starts[1:].any(dim=1), as_tuple=False).flatten() + 1
                boundaries = [0] + reset_steps.detach().cpu().tolist() + [sequence_length]
                for segment_start, segment_end in zip(boundaries[:-1], boundaries[1:]):
                    keep = (1.0 - episode_starts[segment_start]).reshape(1, batch_size, 1)
                    h = h * keep
                    c = c * keep
                    output, (h, c) = self.lstm(features[segment_start:segment_end], (h, c))
                    outputs.append(output)

        latent = torch.cat(outputs, dim=0)
        if sequence_length > 1:
            latent = self.temporal_attention(latent)
        if single_step:
            latent = latent.squeeze(0)
        return latent, (h, c)

    def distribution(self, latent: torch.Tensor) -> Normal:
        mean = torch.tanh(self.actor(latent))
        log_std = torch.clamp(self.log_std, min=-5.0, max=0.6)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, latent: torch.Tensor) -> torch.Tensor:
        return self.critic(latent).squeeze(-1)

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        episode_starts: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        latent, new_hidden_state = self.forward_lstm(observations, hidden_state, episode_starts)
        dist = self.distribution(latent)
        if actions is None:
            if deterministic:
                actions = dist.mean
            else:
                actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.value(latent)
        return actions, log_prob, entropy, values, new_hidden_state

    def deterministic_action(
        self,
        observations: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        actions, _, _, values, new_hidden_state = self.get_action_and_value(
            observations,
            hidden_state=hidden_state,
            episode_starts=episode_starts,
            deterministic=True,
        )
        return actions, values, new_hidden_state

    def _orthogonal_init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            nn.init.constant_(module.bias, 0.0)
