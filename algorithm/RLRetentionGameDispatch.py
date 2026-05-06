import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(1e-6, float(temperature))
    shifted = (logits - np.max(logits)) / temperature
    exp_values = np.exp(shifted)
    total = float(np.sum(exp_values))
    if total <= 0.0:
        return np.ones_like(exp_values) / max(1, exp_values.size)
    return exp_values / total


def _safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr))


def _euclidean(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    return float(math.hypot(lon1 - lon2, lat1 - lat2))


def _compute_service_ratio(
    assigned_tasks_by_region: Dict[int, int],
    total_tasks_by_region: Dict[int, int],
) -> Dict[int, float]:
    ratios: Dict[int, float] = {}
    for rid, total_tasks in total_tasks_by_region.items():
        if total_tasks <= 0:
            ratios[rid] = 1.0
        else:
            ratios[rid] = min(1.0, float(assigned_tasks_by_region.get(rid, 0)) / float(total_tasks))
    return ratios


def _compute_pairwise_unfairness(service_ratio: Dict[int, float]) -> Dict[int, float]:
    region_ids = sorted(service_ratio.keys())
    if len(region_ids) <= 1:
        return {rid: 0.0 for rid in region_ids}

    unfairness: Dict[int, float] = {}
    for rid in region_ids:
        disparity = 0.0
        for other in region_ids:
            if other == rid:
                continue
            disparity += abs(service_ratio[rid] - service_ratio[other])
        unfairness[rid] = disparity / max(1, len(region_ids) - 1)
    return unfairness


class RetentionPolicyNetwork(nn.Module):
    def __init__(self, feature_dim: int, action_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass
class RLRetentionBilateralState:
    region_ids: List[int]
    action_ratios: Tuple[float, ...] = (-0.30, -0.15, 0.0, 0.15, 0.30)
    learning_rate: float = 0.03
    temperature: float = 0.85
    exploration_prob: float = 0.12
    gamma: float = 0.0
    replay_capacity: int = 512
    dqn_batch_size: int = 32
    target_sync_interval: int = 32
    service_debt_decay: float = 0.85
    max_service_debt: float = 4.0
    move_history_size: int = 8
    random_seed: int = 42
    feature_dim: int = 13
    prediction_error_decay: float = 0.80
    prediction_bias_clip_ratio: float = 0.75
    affinity_decay: float = 0.92
    affinity_learning_rate: float = 0.12
    hidden_dim: int = 32
    device: Optional[str] = None
    prediction_bias_ema: Dict[int, float] = field(default_factory=dict)
    prediction_abs_error_ema: Dict[int, float] = field(default_factory=dict)
    q_networks: Dict[int, RetentionPolicyNetwork] = field(default_factory=dict)
    target_q_networks: Dict[int, RetentionPolicyNetwork] = field(default_factory=dict)
    q_optimizers: Dict[int, torch.optim.Optimizer] = field(default_factory=dict)
    replay_buffers: Dict[int, Deque[Dict[str, Any]]] = field(default_factory=dict)
    update_steps: Dict[int, int] = field(default_factory=dict)
    reward_baseline: Dict[int, float] = field(default_factory=dict)
    service_debt: Dict[int, float] = field(default_factory=dict)
    receiver_affinity: Dict[int, Dict[int, float]] = field(default_factory=dict)
    worker_move_slots: Dict[str, Deque[int]] = field(default_factory=dict)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)
        resolved_device = self.device or getattr(config, "TORCH_DEVICE_PREFERENCE", None)
        self.device = str(
            torch.device(resolved_device if resolved_device else ("cuda" if torch.cuda.is_available() else "cpu"))
        )
        action_count = len(self.action_ratios)
        for rid in self.region_ids:
            if rid not in self.q_networks:
                network = RetentionPolicyNetwork(
                    feature_dim=self.feature_dim,
                    action_count=action_count,
                    hidden_dim=max(8, int(self.hidden_dim)),
                ).to(self.device)
                target_network = RetentionPolicyNetwork(
                    feature_dim=self.feature_dim,
                    action_count=action_count,
                    hidden_dim=max(8, int(self.hidden_dim)),
                ).to(self.device)
                target_network.load_state_dict(network.state_dict())
                self.q_networks[rid] = network
                self.target_q_networks[rid] = target_network
            else:
                self.q_networks[rid] = self.q_networks[rid].to(self.device)
                self.target_q_networks[rid] = self.target_q_networks[rid].to(self.device)
            self.q_optimizers.setdefault(
                rid,
                torch.optim.Adam(self.q_networks[rid].parameters(), lr=self.learning_rate),
            )
            self.replay_buffers.setdefault(rid, deque(maxlen=max(32, int(self.replay_capacity))))
            self.update_steps.setdefault(rid, 0)
            self.reward_baseline.setdefault(rid, 0.0)
            self.service_debt.setdefault(rid, 0.0)
            self.prediction_bias_ema.setdefault(rid, 0.0)
            self.prediction_abs_error_ema.setdefault(rid, 0.0)
            affinity_row = self.receiver_affinity.setdefault(rid, {})
            for target_rid in self.region_ids:
                if target_rid == rid:
                    continue
                affinity_row.setdefault(target_rid, 0.0)

    def set_exploration_prob(self, exploration_prob: float) -> None:
        self.exploration_prob = float(np.clip(float(exploration_prob), 0.0, 1.0))

    def record_moves(self, slot_idx: int, moved_workers: Iterable[str]) -> None:
        for wid in moved_workers:
            history = self.worker_move_slots.setdefault(wid, deque(maxlen=self.move_history_size))
            history.append(int(slot_idx))

    def get_receiver_affinity(self, donor_region: int, receiver_region: int) -> float:
        if donor_region == receiver_region:
            return 0.0
        return float(self.receiver_affinity.get(donor_region, {}).get(receiver_region, 0.0))

    def update_receiver_affinity(
        self,
        moves: List[Dict[str, Any]],
        reward_by_region: Dict[int, float],
    ) -> None:
        decay = float(np.clip(self.affinity_decay, 0.0, 0.999))
        lr = max(0.0, float(self.affinity_learning_rate))
        for donor in self.region_ids:
            row = self.receiver_affinity.setdefault(donor, {})
            for receiver in self.region_ids:
                if receiver == donor:
                    continue
                row[receiver] = decay * float(row.get(receiver, 0.0))

        for move in moves:
            donor = int(move.get("from_region", -1))
            receiver = int(move.get("to_region", -1))
            if donor not in self.receiver_affinity or receiver == donor:
                continue
            receiver_reward = float(reward_by_region.get(receiver, 0.0))
            donor_reward = float(reward_by_region.get(donor, 0.0))
            reward_gap = receiver_reward - donor_reward
            signal = float(np.tanh(reward_gap / 50.0))
            row = self.receiver_affinity.setdefault(donor, {})
            row[receiver] = float(np.clip(row.get(receiver, 0.0) + lr * signal, -2.0, 2.0))

    def record_prediction_feedback(
        self,
        predicted_region_demand: Dict[int, int],
        actual_region_demand: Dict[int, int],
    ) -> None:
        decay = float(np.clip(self.prediction_error_decay, 0.0, 0.999))
        clip_ratio = max(0.0, float(self.prediction_bias_clip_ratio))
        for rid in self.region_ids:
            pred = float(predicted_region_demand.get(rid, 0.0))
            actual = float(actual_region_demand.get(rid, 0.0))
            err = pred - actual
            scale = max(1.0, pred, actual)
            clipped_err = float(np.clip(err, -clip_ratio * scale, clip_ratio * scale))
            prev_bias = float(self.prediction_bias_ema.get(rid, 0.0))
            prev_abs = float(self.prediction_abs_error_ema.get(rid, 0.0))
            self.prediction_bias_ema[rid] = decay * prev_bias + (1.0 - decay) * clipped_err
            self.prediction_abs_error_ema[rid] = decay * prev_abs + (1.0 - decay) * abs(clipped_err)

    def build_features(
        self,
        region_id: int,
        demand_profile: Dict[int, Dict[str, float]],
        available_workers: Dict[int, int],
        base_keep: int,
        shortage_workers: float,
        neighbor_backlog_pressure: float,
        max_tasks_per_worker: int,
    ) -> np.ndarray:
        profile = demand_profile[region_id]
        scale = max(
            1.0,
            float(profile.get("mu", 0.0) + profile.get("backlog", 0.0)),
            float(available_workers.get(region_id, 0) * max_tasks_per_worker),
            float(profile.get("effective_demand", 0.0)),
        )
        debt = float(self.service_debt.get(region_id, 0.0))
        bias_ema = float(profile.get("bias_ema", 0.0))
        abs_err_ema = float(profile.get("abs_err_ema", 0.0))
        features = np.asarray(
            [
                1.0,
                float(profile.get("mu", 0.0)) / scale,
                float(profile.get("sigma", 0.0)) / scale,
                float(profile.get("tail_gap", 0.0)) / scale,
                float(profile.get("burst_prob", 0.0)),
                float(profile.get("backlog", 0.0)) / scale,
                float(available_workers.get(region_id, 0) * max_tasks_per_worker) / scale,
                float(base_keep * max_tasks_per_worker) / scale,
                float(shortage_workers * max_tasks_per_worker) / scale,
                float(neighbor_backlog_pressure) / scale,
                bias_ema / scale,
                abs_err_ema / scale,
                debt / max(1.0, self.max_service_debt),
            ],
            dtype=np.float32,
        )
        return features

    def _q_values(self, region_id: int, features: np.ndarray) -> torch.Tensor:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.q_networks[region_id](feature_tensor).squeeze(0)

    def sample_action(self, region_id: int, features: np.ndarray) -> Tuple[int, float, np.ndarray]:
        with torch.no_grad():
            q_values = self._q_values(region_id, features)
            probs_tensor = torch.softmax(q_values / max(1e-6, float(self.temperature)), dim=-1)
        probs = probs_tensor.cpu().numpy().astype(np.float32)
        if self.rng.random() < self.exploration_prob:
            action_idx = int(self.rng.integers(0, len(self.action_ratios)))
        else:
            action_idx = int(torch.argmax(q_values).item())
        return action_idx, float(self.action_ratios[action_idx]), probs

    def _sync_target_network(self, region_id: int) -> None:
        self.target_q_networks[region_id].load_state_dict(self.q_networks[region_id].state_dict())

    def _train_region_q_network(self, region_id: int) -> None:
        replay_buffer = self.replay_buffers[region_id]
        batch_size = min(len(replay_buffer), max(1, int(self.dqn_batch_size)))
        if batch_size <= 0:
            return

        sample_indices = self.rng.choice(len(replay_buffer), size=batch_size, replace=False)
        batch = [replay_buffer[int(idx)] for idx in sample_indices]

        states = torch.as_tensor(np.stack([item["features"] for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([int(item["action_idx"]) for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([float(item["reward"]) for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([float(item.get("done", 1.0)) for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item["next_features"] for item in batch]), dtype=torch.float32, device=self.device)

        q_values = self.q_networks[region_id](states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            online_next_q = self.q_networks[region_id](next_states)
            next_actions = torch.argmax(online_next_q, dim=1)
            target_next_q = self.target_q_networks[region_id](next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + (1.0 - dones) * float(self.gamma) * target_next_q

        optimizer = self.q_optimizers[region_id]
        optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_values, targets)
        loss.backward()
        optimizer.step()

        self.update_steps[region_id] += 1
        if self.update_steps[region_id] % max(1, int(self.target_sync_interval)) == 0:
            self._sync_target_network(region_id)

    def update_policy(
        self,
        transitions: Dict[int, Dict[str, Any]],
        reward_by_region: Dict[int, float],
        total_tasks_by_region: Dict[int, int],
        assigned_tasks_by_region: Dict[int, int],
    ) -> None:
        service_ratio = _compute_service_ratio(assigned_tasks_by_region, total_tasks_by_region)
        for rid, total_tasks in total_tasks_by_region.items():
            prev_debt = float(self.service_debt.get(rid, 0.0))
            if total_tasks <= 0:
                self.service_debt[rid] = max(0.0, prev_debt * self.service_debt_decay)
                continue
            updated = prev_debt * self.service_debt_decay + max(0.0, 1.0 - service_ratio.get(rid, 1.0))
            self.service_debt[rid] = min(self.max_service_debt, updated)

        for rid, transition in transitions.items():
            reward = float(reward_by_region.get(rid, 0.0))
            baseline = float(self.reward_baseline.get(rid, 0.0))
            self.reward_baseline[rid] = 0.9 * baseline + 0.1 * reward

            features = np.asarray(transition["features"], dtype=np.float32)
            next_features = np.asarray(transition.get("next_features", features), dtype=np.float32)
            self.replay_buffers[rid].append(
                {
                    "features": features,
                    "action_idx": int(transition["action_idx"]),
                    "reward": reward,
                    "next_features": next_features,
                    "done": float(transition.get("done", 1.0)),
                }
            )
            self._train_region_q_network(rid)

    def offline_replay_train(
        self,
        epochs: int = 1,
        updates_per_region: int = 1,
    ) -> Dict[str, int]:
        epochs = max(0, int(epochs))
        updates_per_region = max(1, int(updates_per_region))
        optimization_steps = 0
        trained_regions = set()

        for _ in range(epochs):
            for rid in self.region_ids:
                if not self.replay_buffers.get(rid):
                    continue
                trained_regions.add(rid)
                for _ in range(updates_per_region):
                    self._train_region_q_network(rid)
                    optimization_steps += 1

        return {
            "epochs": epochs,
            "updates_per_region": updates_per_region,
            "trained_region_count": len(trained_regions),
            "optimization_steps": optimization_steps,
        }

    def imitation_update(
        self,
        region_id: int,
        features: np.ndarray,
        target_action_idx: int,
        strength: float = 1.0,
    ) -> None:
        optimizer = self.q_optimizers[region_id]
        optimizer.zero_grad()
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.q_networks[region_id](feature_tensor)
        target_tensor = torch.as_tensor([int(target_action_idx)], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, target_tensor) * float(strength)
        loss.backward()
        optimizer.step()
        self._sync_target_network(region_id)


@dataclass
class PlatformTaskFirstRLState:
    region_ids: List[int]
    action_profiles: Tuple[Tuple[float, float, float, float, float, float], ...] = (
        (1.30, 1.20, 0.95, 0.82, 1.18, 0.10),
        (1.15, 1.10, 1.00, 0.92, 1.08, 0.18),
        (1.00, 1.00, 1.08, 1.00, 0.98, 0.28),
        (0.92, 0.95, 1.20, 1.10, 0.88, 0.40),
    )
    learning_rate: float = 0.03
    temperature: float = 0.90
    exploration_prob: float = 0.10
    gamma: float = 0.6
    replay_capacity: int = 256
    dqn_batch_size: int = 32
    target_sync_interval: int = 32
    hidden_dim: int = 32
    device: Optional[str] = None
    random_seed: int = 7
    feature_dim: int = 9
    reward_baseline: float = 0.0
    completion_rate_ema: float = 0.0
    unfairness_ema: float = 0.0
    q_network: Optional[RetentionPolicyNetwork] = None
    target_q_network: Optional[RetentionPolicyNetwork] = None
    q_optimizer: Optional[torch.optim.Optimizer] = None
    replay_buffer: Deque[Dict[str, Any]] = field(default_factory=deque)
    update_steps: int = 0
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)
        resolved_device = self.device or getattr(config, "TORCH_DEVICE_PREFERENCE", None)
        self.device = str(
            torch.device(resolved_device if resolved_device else ("cuda" if torch.cuda.is_available() else "cpu"))
        )
        action_count = len(self.action_profiles)
        if self.q_network is None:
            self.q_network = RetentionPolicyNetwork(
                feature_dim=self.feature_dim,
                action_count=action_count,
                hidden_dim=max(8, int(self.hidden_dim)),
            ).to(self.device)
        else:
            self.q_network = self.q_network.to(self.device)
        if self.target_q_network is None:
            self.target_q_network = RetentionPolicyNetwork(
                feature_dim=self.feature_dim,
                action_count=action_count,
                hidden_dim=max(8, int(self.hidden_dim)),
            ).to(self.device)
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        else:
            self.target_q_network = self.target_q_network.to(self.device)
        if self.q_optimizer is None:
            self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        if not isinstance(self.replay_buffer, deque):
            self.replay_buffer = deque(self.replay_buffer, maxlen=max(32, int(self.replay_capacity)))
        else:
            self.replay_buffer = deque(self.replay_buffer, maxlen=max(32, int(self.replay_capacity)))

    def build_features(
        self,
        demand_profile: Dict[int, Dict[str, float]],
        available_workers: Dict[int, int],
        desired_workers: Dict[int, int],
        max_tasks_per_worker: int,
    ) -> np.ndarray:
        total_effective = float(sum(profile.get("effective_demand", 0.0) for profile in demand_profile.values()))
        total_backlog = float(sum(profile.get("backlog", 0.0) for profile in demand_profile.values()))
        mean_sigma = float(np.mean([profile.get("sigma", 0.0) for profile in demand_profile.values()])) if demand_profile else 0.0
        mean_bias = float(np.mean([profile.get("bias_ema", 0.0) for profile in demand_profile.values()])) if demand_profile else 0.0
        mean_abs_error = float(np.mean([profile.get("abs_err_ema", 0.0) for profile in demand_profile.values()])) if demand_profile else 0.0
        total_capacity = float(sum(int(available_workers.get(rid, 0)) * max_tasks_per_worker for rid in self.region_ids))
        total_shortage = float(sum(max(0, desired_workers.get(rid, 0) - int(available_workers.get(rid, 0))) for rid in self.region_ids))
        scale = max(1.0, total_effective + total_backlog, total_capacity)
        features = np.asarray(
            [
                1.0,
                total_effective / scale,
                total_backlog / scale,
                mean_sigma / scale,
                total_capacity / scale,
                total_shortage / max(1.0, total_capacity / max(1, max_tasks_per_worker)),
                mean_bias / scale,
                mean_abs_error / scale,
                0.0,
            ],
            dtype=np.float32,
        )
        return features

    def set_exploration_prob(self, exploration_prob: float) -> None:
        self.exploration_prob = float(np.clip(float(exploration_prob), 0.0, 1.0))

    def _q_values(self, features: np.ndarray) -> torch.Tensor:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.q_network(feature_tensor).squeeze(0)

    def sample_action(self, features: np.ndarray) -> Tuple[int, Tuple[float, float, float, float, float, float], np.ndarray]:
        with torch.no_grad():
            q_values = self._q_values(features)
            probs_tensor = torch.softmax(q_values / max(1e-6, float(self.temperature)), dim=-1)
        probs = probs_tensor.cpu().numpy().astype(np.float32)
        if self.rng.random() < self.exploration_prob:
            action_idx = int(self.rng.integers(0, len(self.action_profiles)))
        else:
            action_idx = int(torch.argmax(q_values).item())
        return action_idx, self.action_profiles[action_idx], probs

    def _sync_target_network(self) -> None:
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _train_q_network(self) -> None:
        batch_size = min(len(self.replay_buffer), max(1, int(self.dqn_batch_size)))
        if batch_size <= 0:
            return

        sample_indices = self.rng.choice(len(self.replay_buffer), size=batch_size, replace=False)
        batch = [self.replay_buffer[int(idx)] for idx in sample_indices]

        states = torch.as_tensor(np.stack([item["features"] for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([int(item["action_idx"]) for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([float(item["reward"]) for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([float(item.get("done", 1.0)) for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item["next_features"] for item in batch]), dtype=torch.float32, device=self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            online_next_q = self.q_network(next_states)
            next_actions = torch.argmax(online_next_q, dim=1)
            target_next_q = self.target_q_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + (1.0 - dones) * float(self.gamma) * target_next_q

        self.q_optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_values, targets)
        loss.backward()
        self.q_optimizer.step()

        self.update_steps += 1
        if self.update_steps % max(1, int(self.target_sync_interval)) == 0:
            self._sync_target_network()

    def update_policy(
        self,
        features: np.ndarray,
        action_idx: int,
        reward: float,
        completion_rate: float,
        unfairness: float,
        next_features: Optional[np.ndarray] = None,
        done: float = 1.0,
    ) -> None:
        baseline = float(self.reward_baseline)
        self.reward_baseline = 0.9 * baseline + 0.1 * float(reward)
        self.completion_rate_ema = 0.85 * float(self.completion_rate_ema) + 0.15 * float(completion_rate)
        self.unfairness_ema = 0.85 * float(self.unfairness_ema) + 0.15 * float(unfairness)
        next_features = np.asarray(next_features if next_features is not None else features, dtype=np.float32)
        self.replay_buffer.append(
            {
                "features": np.asarray(features, dtype=np.float32),
                "action_idx": int(action_idx),
                "reward": float(reward),
                "next_features": next_features,
                "done": float(done),
            }
        )
        self._train_q_network()

    def offline_replay_train(
        self,
        epochs: int = 1,
        updates_per_epoch: int = 1,
    ) -> Dict[str, int]:
        epochs = max(0, int(epochs))
        updates_per_epoch = max(1, int(updates_per_epoch))
        optimization_steps = 0

        if not self.replay_buffer:
            return {
                "epochs": epochs,
                "updates_per_epoch": updates_per_epoch,
                "optimization_steps": 0,
                "buffer_size": 0,
            }

        for _ in range(epochs):
            for _ in range(updates_per_epoch):
                self._train_q_network()
                optimization_steps += 1

        return {
            "epochs": epochs,
            "updates_per_epoch": updates_per_epoch,
            "optimization_steps": optimization_steps,
            "buffer_size": len(self.replay_buffer),
        }


def hydrate_platform_transition_with_next_state(
    state: PlatformTaskFirstRLState,
    transition: Dict[str, Any],
    available_workers: Dict[int, int],
    backlog_counts: Dict[int, int],
    max_tasks_per_worker: int,
    backlog_weight: float = 1.0,
    done: float = 0.0,
) -> Dict[str, Any]:
    if not transition:
        return transition

    demand_profile = {rid: dict(profile) for rid, profile in transition.get("demand_profile", {}).items()}
    desired_workers = dict(transition.get("desired_workers", {}))
    for rid, profile in demand_profile.items():
        backlog = float(max(0, int(backlog_counts.get(rid, 0))))
        profile["backlog"] = backlog
        profile["base_demand"] = float(profile.get("mu", 0.0)) + float(backlog_weight) * backlog
        profile["effective_demand"] = profile["base_demand"]

    transition["next_features"] = state.build_features(
        demand_profile=demand_profile,
        available_workers=available_workers,
        desired_workers=desired_workers,
        max_tasks_per_worker=max_tasks_per_worker,
    )
    transition["done"] = float(done)
    return transition


def _build_movable_workers(
    worker_sim,
    centers: Dict[int, Any],
    next_slot_start_seconds: float,
) -> Dict[int, List[str]]:
    movable_workers = {rid: [] for rid in centers.keys()}
    for wid, region_id in worker_sim.worker_center_map.items():
        if region_id not in centers or wid not in worker_sim.worker_positions:
            continue

        status = worker_sim.worker_status.get(wid, "idle")
        busy_until = worker_sim.worker_busy_until.get(wid, 0.0)
        if status == "en_route_to_task" and busy_until > next_slot_start_seconds:
            continue

        movable_workers[region_id].append(wid)
    return movable_workers


def _build_demand_profile(
    region_ids: List[int],
    predicted_demand: Dict[int, int],
    backlog_counts: Dict[int, int],
    state: RLRetentionBilateralState,
    predicted_distribution: Optional[Dict[int, Dict[str, float]]],
    backlog_weight: float,
    uncertainty_weight: float,
    quantile_weight: float,
    burst_weight: float,
    calibration_bias_weight: float,
    calibration_shrink_weight: float,
    calibration_sigma_boost: float,
    calibration_min_scale: float,
) -> Dict[int, Dict[str, float]]:
    demand_profile: Dict[int, Dict[str, float]] = {}
    for rid in region_ids:
        dist_profile = (predicted_distribution or {}).get(rid, {})
        raw_mu = float(max(0.0, dist_profile.get("mu", predicted_demand.get(rid, 0))))
        backlog = float(max(0.0, backlog_counts.get(rid, 0)))
        raw_sigma = float(max(0.0, dist_profile.get("sigma", math.sqrt(max(raw_mu, 1.0)) * 0.20)))
        raw_q90 = float(max(raw_mu, dist_profile.get("q90", raw_mu + raw_sigma)))
        raw_tail_gap = max(0.0, raw_q90 - raw_mu)
        burst = float(np.clip(dist_profile.get("burst_prob", min(1.0, raw_sigma / max(raw_mu + 1.0, 1.0))), 0.0, 1.0))
        debt = float(state.service_debt.get(rid, 0.0))
        bias_ema = float(state.prediction_bias_ema.get(rid, 0.0))
        abs_err_ema = float(state.prediction_abs_error_ema.get(rid, 0.0))
        hist_bias = float(dist_profile.get("hist_bias", 0.0))
        hist_abs_bias = float(dist_profile.get("hist_abs_bias", 0.0))
        bias_cap = max(8.0, 0.30 * max(1.0, raw_mu))
        clipped_hist_bias = float(np.clip(hist_bias, -bias_cap, bias_cap))
        clipped_online_bias = float(np.clip(bias_ema, -bias_cap, bias_cap))
        bias_agreement = 1.0 if clipped_hist_bias * clipped_online_bias > 0.0 else 0.0
        hist_bias_weight = 0.40 + 0.15 * bias_agreement
        online_bias_weight = 0.25 + 0.20 * bias_agreement
        combined_bias = hist_bias_weight * clipped_hist_bias + online_bias_weight * clipped_online_bias
        combined_bias = float(np.clip(combined_bias, -bias_cap, bias_cap))
        combined_abs_bias = min(
            bias_cap,
            0.60 * min(hist_abs_bias, bias_cap) + 0.40 * min(abs_err_ema, bias_cap),
        )
        relative_error = combined_abs_bias / max(1.0, raw_mu)

        corrected_mu = max(0.0, raw_mu)
        shrink_scale = 1.0
        mu = corrected_mu
        sigma = max(
            raw_sigma,
            raw_sigma * (1.0 + calibration_sigma_boost * relative_error),
            0.35 * combined_abs_bias,
        )
        q90 = max(mu, mu + raw_tail_gap)
        tail_gap = max(0.0, q90 - mu)
        base_demand = mu + backlog_weight * backlog
        underpredict_ratio = max(0.0, -combined_bias) / max(1.0, raw_mu)
        overpredict_ratio = max(0.0, combined_bias) / max(1.0, raw_mu)
        risk_scale = 0.0
        capped_uncertainty = 0.0
        effective = base_demand
        demand_profile[rid] = {
            "mu": mu,
            "sigma": sigma,
            "q90": q90,
            "tail_gap": tail_gap,
            "burst_prob": burst,
            "backlog": backlog,
            "debt": debt,
            "effective_demand": effective,
            "base_demand": base_demand,
            "risk_scale": risk_scale,
            "capped_uncertainty": capped_uncertainty,
            "raw_mu": raw_mu,
            "hist_bias": hist_bias,
            "hist_abs_bias": hist_abs_bias,
            "bias_cap": bias_cap,
            "clipped_hist_bias": clipped_hist_bias,
            "clipped_online_bias": clipped_online_bias,
            "bias_ema": bias_ema,
            "combined_bias": combined_bias,
            "abs_err_ema": abs_err_ema,
            "shrink_scale": shrink_scale,
        }
    return demand_profile


def _normalize_worker_targets(
    region_ids: List[int],
    demand_profile: Dict[int, Dict[str, float]],
    raw_desired_workers: Dict[int, int],
    total_available_workers: int,
) -> Dict[int, int]:
    total_available_workers = max(0, int(total_available_workers))
    total_raw_desired = int(sum(max(0, int(raw_desired_workers.get(rid, 0))) for rid in region_ids))
    if total_available_workers <= 0 or total_raw_desired <= total_available_workers:
        return {rid: max(0, int(raw_desired_workers.get(rid, 0))) for rid in region_ids}

    score_basis = {}
    for rid in region_ids:
        profile = demand_profile[rid]
        score_basis[rid] = max(
            1e-6,
            profile["effective_demand"] * (1.0 + 0.15 * profile["burst_prob"] + 0.10 * profile["debt"])
        )
    score_sum = float(sum(score_basis.values()))
    if score_sum <= 0.0:
        score_sum = float(len(region_ids))

    normalized_targets = {}
    fractional_targets = []
    allocated = 0
    for rid in region_ids:
        target_float = total_available_workers * score_basis[rid] / score_sum
        target_floor = int(math.floor(target_float))
        capped_floor = min(max(0, int(raw_desired_workers.get(rid, 0))), target_floor)
        normalized_targets[rid] = capped_floor
        allocated += capped_floor
        fractional_targets.append((target_float - target_floor, rid))

    remaining = max(0, total_available_workers - allocated)
    for _, rid in sorted(fractional_targets, reverse=True):
        if remaining <= 0:
            break
        raw_target = max(0, int(raw_desired_workers.get(rid, 0)))
        if normalized_targets[rid] >= raw_target:
            continue
        normalized_targets[rid] += 1
        remaining -= 1

    return normalized_targets


def _compute_task_max_abundance_controls(
    total_available_workers: int,
    total_desired_workers: int,
) -> Dict[str, float]:
    controls = {
        "abundance_ratio": 1.0,
        "abundance_strength": 0.0,
        "task_weight_scale": 1.0,
        "gap_weight_scale": 1.0,
        "keep_scale": 1.0,
        "need_scale": 1.0,
        "slot_start_blend_scale": 1.0,
        "move_share_scale": 1.0,
    }
    if not bool(getattr(config, "RBG_TASK_MAX_MODE", False)):
        return controls

    total_desired_workers = max(1, int(total_desired_workers))
    abundance_ratio = float(total_available_workers) / float(total_desired_workers)
    threshold = float(getattr(config, "RBG_TASK_MAX_ABUNDANCE_THRESHOLD", 1.05))
    ramp = max(0.05, float(getattr(config, "RBG_TASK_MAX_ABUNDANCE_RAMP", 0.30)))
    abundance_strength = float(np.clip((abundance_ratio - threshold) / ramp, 0.0, 1.0))

    controls["abundance_ratio"] = abundance_ratio
    controls["abundance_strength"] = abundance_strength
    if abundance_strength <= 0.0:
        return controls

    controls["task_weight_scale"] = 1.0 + abundance_strength * float(
        getattr(config, "RBG_TASK_MAX_TASK_WEIGHT_BOOST", 0.40)
    )
    controls["gap_weight_scale"] = 1.0 + abundance_strength * float(
        getattr(config, "RBG_TASK_MAX_GAP_WEIGHT_BOOST", 0.30)
    )
    controls["keep_scale"] = max(
        0.70,
        1.0 - abundance_strength * float(getattr(config, "RBG_TASK_MAX_KEEP_SCALE_DROP", 0.15)),
    )
    controls["need_scale"] = 1.0 + abundance_strength * float(
        getattr(config, "RBG_TASK_MAX_NEED_SCALE_BOOST", 0.18)
    )
    controls["slot_start_blend_scale"] = 1.0 + abundance_strength * float(
        getattr(config, "RBG_TASK_MAX_SLOT_BLEND_BOOST", 0.08)
    )
    controls["move_share_scale"] = 1.0 + abundance_strength * float(
        getattr(config, "RBG_TASK_MAX_MOVE_SHARE_BOOST", 0.20)
    )
    return controls


def hydrate_retention_transitions_with_next_state(
    state: RLRetentionBilateralState,
    transitions: Dict[int, Dict[str, Any]],
    demand_profile: Dict[int, Dict[str, float]],
    desired_workers: Dict[int, int],
    available_workers: Dict[int, int],
    backlog_counts: Dict[int, int],
    max_tasks_per_worker: int,
    min_buffer_workers: int = 1,
    backlog_weight: float = 1.0,
    done: float = 0.0,
) -> Dict[int, Dict[str, Any]]:
    if not transitions:
        return transitions

    total_backlog = float(sum(max(0, int(backlog_counts.get(rid, 0))) for rid in transitions.keys()))
    for rid, transition in transitions.items():
        profile = dict(demand_profile.get(rid, {}))
        backlog = float(max(0, int(backlog_counts.get(rid, 0))))
        profile["backlog"] = backlog
        profile["base_demand"] = float(profile.get("mu", 0.0)) + float(backlog_weight) * backlog
        profile["effective_demand"] = profile["base_demand"]

        idle_count = max(0, int(available_workers.get(rid, 0)))
        desired = max(0, int(desired_workers.get(rid, 0)))
        base_keep = min(
            idle_count,
            max(int(min_buffer_workers), int(math.ceil(desired))),
        )
        shortage_workers = max(0.0, float(desired - idle_count))
        neighbor_backlog_pressure = max(0.0, total_backlog - backlog)
        next_features = state.build_features(
            region_id=rid,
            demand_profile={rid: profile},
            available_workers=available_workers,
            base_keep=base_keep,
            shortage_workers=shortage_workers,
            neighbor_backlog_pressure=neighbor_backlog_pressure,
            max_tasks_per_worker=max_tasks_per_worker,
        )
        transition["next_features"] = next_features
        transition["done"] = float(done)

    return transitions


def _select_retained_and_releasable_workers(
    region_id: int,
    worker_ids: List[str],
    retain_count: int,
    shortage_receivers: List[int],
    receiver_need: Dict[int, int],
    demand_profile: Dict[int, Dict[str, float]],
    selection_action_ratio: float,
    worker_sim,
    centers: Dict[int, Any],
    state: RLRetentionBilateralState,
) -> Tuple[List[str], List[str]]:
    if not worker_ids:
        return [], []

    donor_center = centers[region_id]
    donor_profile = demand_profile.get(region_id, {})
    keep_bias = max(0.0, float(selection_action_ratio))
    lend_bias = max(0.0, -float(selection_action_ratio))
    donor_anchor = worker_sim.G.nodes[donor_center]
    donor_lon = donor_anchor.get("x", donor_anchor.get("lon"))
    donor_lat = donor_anchor.get("y", donor_anchor.get("lat"))
    local_need_pressure = (
        max(0.0, float(receiver_need.get(region_id, 0)))
        + 0.20 * max(0.0, float(donor_profile.get("backlog", 0.0)))
        + 0.10 * max(0.0, float(donor_profile.get("burst_prob", 0.0)))
    )
    donor_distance_order: List[Tuple[float, str]] = []
    for wid in worker_ids:
        _, worker_lon, worker_lat = worker_sim.worker_positions[wid]
        donor_distance = _euclidean(worker_lon, worker_lat, donor_lon, donor_lat)
        donor_distance_order.append((donor_distance, wid))
    donor_distance_order.sort()
    worker_local_rank = {wid: idx for idx, (_, wid) in enumerate(donor_distance_order)}
    local_quota = max(1, min(int(retain_count), len(worker_ids)))
    scored_workers: List[Tuple[float, str]] = []
    for wid in worker_ids:
        worker_node, worker_lon, worker_lat = worker_sim.worker_positions[wid]
        donor_distance = _euclidean(worker_lon, worker_lat, donor_lon, donor_lat)
        donor_proximity = 1.0 / (1.0 + donor_distance)
        local_rank = int(worker_local_rank.get(wid, len(worker_ids)))
        local_rank_factor = 1.0 if local_rank < local_quota else max(0.15, 1.0 - 0.12 * (local_rank - local_quota + 1))
        local_keep_value = (
            (1.0 + 0.60 * keep_bias)
            * (1.0 + 0.30 * local_need_pressure)
            * donor_proximity
            * local_rank_factor
        )

        best_lend_value = 0.0
        for receiver in shortage_receivers:
            if receiver == region_id:
                continue
            receiver_anchor = worker_sim.G.nodes[centers[receiver]]
            receiver_lon = receiver_anchor.get("x", receiver_anchor.get("lon"))
            receiver_lat = receiver_anchor.get("y", receiver_anchor.get("lat"))
            receiver_distance = _euclidean(worker_lon, worker_lat, receiver_lon, receiver_lat)
            receiver_profile = demand_profile.get(receiver, {})
            receiver_priority = (
                max(0.0, float(receiver_need.get(receiver, 0)))
                + 0.20 * max(0.0, float(receiver_profile.get("backlog", 0.0)))
                + 0.10 * max(0.0, float(receiver_profile.get("burst_prob", 0.0)))
                + 0.10 * max(0.0, state.get_receiver_affinity(region_id, receiver))
            )
            receiver_service_value = receiver_priority / (1.0 + receiver_distance)
            lend_value = max(
                0.0,
                receiver_service_value - 0.35 * donor_proximity,
            )
            best_lend_value = max(best_lend_value, lend_value)

        move_history = state.worker_move_slots.get(wid)
        recent_move_penalty = 0.15 * len(move_history) if move_history is not None else 0.0
        keep_score = (
            local_keep_value
            - (0.45 + 0.60 * lend_bias) * best_lend_value
            + 0.20 * recent_move_penalty
        )
        scored_workers.append((keep_score, wid))

    scored_workers.sort(reverse=True)
    retained = [wid for _, wid in scored_workers[:retain_count]]
    releasable = [wid for _, wid in scored_workers[retain_count:]]
    return retained, releasable


def _choose_worker_for_receiver(
    donor_region: int,
    receiver_region: int,
    releasable_workers: List[str],
    receiver_need: Dict[int, int],
    demand_profile: Dict[int, Dict[str, float]],
    donor_action_ratio: float,
    worker_sim,
    centers: Dict[int, Any],
    state: RLRetentionBilateralState,
    candidate_k: int,
) -> Optional[str]:
    if not releasable_workers:
        return None

    receiver_anchor = worker_sim.G.nodes[centers[receiver_region]]
    receiver_lon = receiver_anchor.get("x", receiver_anchor.get("lon"))
    receiver_lat = receiver_anchor.get("y", receiver_anchor.get("lat"))

    donor_anchor = worker_sim.G.nodes[centers[donor_region]]
    donor_lon = donor_anchor.get("x", donor_anchor.get("lon"))
    donor_lat = donor_anchor.get("y", donor_anchor.get("lat"))

    ranked: List[Tuple[float, str]] = []
    receiver_profile = demand_profile.get(receiver_region, {})
    keep_bias = max(0.0, float(donor_action_ratio))
    lend_bias = max(0.0, -float(donor_action_ratio))
    learned_receiver_affinity = state.get_receiver_affinity(donor_region, receiver_region)
    receiver_priority = (
        1.0
        + 0.20 * max(0.0, float(receiver_need.get(receiver_region, 0)))
        + 0.10 * max(0.0, float(receiver_profile.get("backlog", 0.0)))
        + 0.08 * max(0.0, float(receiver_profile.get("burst_prob", 0.0)))
        + 0.10 * max(0.0, learned_receiver_affinity)
    )
    for wid in releasable_workers:
        _, worker_lon, worker_lat = worker_sim.worker_positions[wid]
        receiver_distance = _euclidean(worker_lon, worker_lat, receiver_lon, receiver_lat)
        donor_distance = _euclidean(worker_lon, worker_lat, donor_lon, donor_lat)
        move_history = state.worker_move_slots.get(wid)
        move_penalty = 0.12 * len(move_history) if move_history is not None else 0.0
        score = (
            (receiver_distance / (receiver_priority * (1.0 + 0.50 * lend_bias)))
            - (0.20 + 0.20 * keep_bias) * donor_distance
            + move_penalty
        )
        ranked.append((score, wid))

    ranked.sort()
    top_candidates = ranked[:max(1, int(candidate_k))]
    return top_candidates[0][1] if top_candidates else None


def sample_platform_task_first_control(
    region_ids: List[int],
    predicted_demand: Dict[int, int],
    backlog_counts: Dict[int, int],
    available_workers: Dict[int, int],
    max_tasks_per_worker: int,
    retention_state: RLRetentionBilateralState,
    platform_state: PlatformTaskFirstRLState,
    predicted_distribution: Optional[Dict[int, Dict[str, float]]],
    backlog_weight: float,
    uncertainty_weight: float,
    quantile_weight: float,
    burst_weight: float,
    calibration_bias_weight: float,
    calibration_shrink_weight: float,
    calibration_sigma_boost: float,
    calibration_min_scale: float,
    base_platform_task_weight: float,
    base_platform_gap_weight: float,
    base_platform_release_credit_weight: float,
) -> Dict[str, Any]:
    demand_profile = _build_demand_profile(
        region_ids=region_ids,
        predicted_demand=predicted_demand,
        backlog_counts=backlog_counts,
        state=retention_state,
        predicted_distribution=predicted_distribution,
        backlog_weight=backlog_weight,
        uncertainty_weight=uncertainty_weight,
        quantile_weight=quantile_weight,
        burst_weight=burst_weight,
        calibration_bias_weight=calibration_bias_weight,
        calibration_shrink_weight=calibration_shrink_weight,
        calibration_sigma_boost=calibration_sigma_boost,
        calibration_min_scale=calibration_min_scale,
    )
    raw_desired_workers = {
        rid: int(math.ceil(demand_profile[rid]["effective_demand"] / max(1, int(max_tasks_per_worker))))
        for rid in region_ids
    }
    desired_workers = _normalize_worker_targets(
        region_ids=region_ids,
        demand_profile=demand_profile,
        raw_desired_workers=raw_desired_workers,
        total_available_workers=int(sum(max(0, int(available_workers.get(rid, 0))) for rid in region_ids)),
    )
    desired_workers = {
        rid: max(
            int(math.ceil(demand_profile[rid]["backlog"] / max(1, int(max_tasks_per_worker)))),
            desired_workers[rid] + max(
                0,
                int(math.ceil(max(0.0, -demand_profile[rid].get("combined_bias", 0.0)) / max(1, int(max_tasks_per_worker)) * 0.5)),
            ),
        )
        for rid in region_ids
    }
    abundance_controls = _compute_task_max_abundance_controls(
        total_available_workers=int(sum(max(0, int(available_workers.get(rid, 0))) for rid in region_ids)),
        total_desired_workers=int(sum(max(0, int(desired_workers.get(rid, 0))) for rid in region_ids)),
    )
    features = platform_state.build_features(
        demand_profile=demand_profile,
        available_workers=available_workers,
        desired_workers=desired_workers,
        max_tasks_per_worker=max_tasks_per_worker,
    )
    action_idx, action_profile, probs = platform_state.sample_action(features)
    task_scale, gap_scale, release_scale, keep_scale, need_scale, _fairness_weight = action_profile
    task_scale *= abundance_controls["task_weight_scale"]
    gap_scale *= abundance_controls["gap_weight_scale"]
    keep_scale *= abundance_controls["keep_scale"]
    need_scale *= abundance_controls["need_scale"]
    return {
        "features": features,
        "probs": probs,
        "action_idx": action_idx,
        "task_weight": float(base_platform_task_weight * task_scale),
        "gap_weight": float(base_platform_gap_weight * gap_scale),
        "release_credit_weight": float(base_platform_release_credit_weight * release_scale),
        "keep_scale": float(keep_scale),
        "need_scale": float(need_scale),
        "fairness_weight": 0.0,
        "desired_workers": desired_workers,
        "demand_profile": demand_profile,
        "move_share_scale": float(abundance_controls["move_share_scale"]),
        "slot_start_blend_scale": float(abundance_controls["slot_start_blend_scale"]),
        "abundance_ratio": float(abundance_controls["abundance_ratio"]),
        "abundance_strength": float(abundance_controls["abundance_strength"]),
    }


def rl_retention_bilateral_predispatch_workers(
    G: nx.Graph,
    worker_sim,
    centers: Dict[int, Any],
    predicted_demand: Dict[int, int],
    state: RLRetentionBilateralState,
    slot_idx: int,
    next_slot_start_seconds: float,
    predicted_distribution: Optional[Dict[int, Dict[str, float]]] = None,
    max_tasks_per_worker: int = 4,
    backlog_counts: Optional[Dict[int, int]] = None,
    backlog_weight: float = 1.0,
    uncertainty_weight: float = 0.45,
    quantile_weight: float = 0.55,
    burst_weight: float = 1.2,
    calibration_bias_weight: float = 0.60,
    calibration_shrink_weight: float = 0.55,
    calibration_sigma_boost: float = 0.75,
    calibration_min_scale: float = 0.55,
    platform_task_weight: float = 0.30,
    platform_gap_weight: float = 0.55,
    platform_release_credit_weight: float = 0.35,
    platform_fairness_weight: float = 0.0,
    platform_keep_scale: float = 1.0,
    platform_need_scale: float = 1.0,
    platform_move_share_scale: float = 1.0,
    platform_slot_start_blend_scale: float = 1.0,
    center_local_task_weight: float = 1.0,
    worker_completion_bonus: float = 0.20,
    worker_distance_penalty: float = 0.015,
    same_worker_chain_bonus: float = 0.08,
    min_buffer_workers: int = 1,
    reserve_ratio: float = 0.10,
    bid_shortage_weight: float = 0.90,
    bid_backlog_weight: float = 0.60,
    bid_debt_weight: float = 0.75,
    bid_burst_weight: float = 0.40,
    ask_shortage_weight: float = 0.85,
    ask_uncertainty_weight: float = 0.55,
    hoard_discount_weight: float = 0.40,
    move_cost_weight: float = 0.02,
    distance_penalty: float = 0.003,
    candidate_k: int = 12,
    edge_epsilon: float = 0.05,
    dispatch_phase: str = "slot_start",
    record_transition: bool = True,
) -> Dict[str, Any]:
    max_tasks_per_worker = max(1, int(max_tasks_per_worker))
    backlog_counts = backlog_counts or {}
    region_ids = sorted(centers.keys())
    movable_workers = _build_movable_workers(worker_sim, centers, next_slot_start_seconds)
    available_workers = {rid: len(movable_workers[rid]) for rid in region_ids}

    demand_profile = _build_demand_profile(
        region_ids=region_ids,
        predicted_demand=predicted_demand,
        backlog_counts=backlog_counts,
        state=state,
        predicted_distribution=predicted_distribution,
        backlog_weight=backlog_weight,
        uncertainty_weight=uncertainty_weight,
        quantile_weight=quantile_weight,
        burst_weight=burst_weight,
        calibration_bias_weight=calibration_bias_weight,
        calibration_shrink_weight=calibration_shrink_weight,
        calibration_sigma_boost=calibration_sigma_boost,
        calibration_min_scale=calibration_min_scale,
    )

    raw_desired_workers = {
        rid: int(math.ceil(demand_profile[rid]["effective_demand"] / max_tasks_per_worker))
        for rid in region_ids
    }
    allocation_target_workers = _normalize_worker_targets(
        region_ids=region_ids,
        demand_profile=demand_profile,
        raw_desired_workers=raw_desired_workers,
        total_available_workers=int(sum(available_workers.values())),
    )
    stabilized_desired_workers = {
        rid: max(
            int(math.ceil(demand_profile[rid]["backlog"] / max_tasks_per_worker)),
            allocation_target_workers[rid] + max(
                0,
                int(math.ceil(max(0.0, -demand_profile[rid].get("combined_bias", 0.0)) / max_tasks_per_worker * 0.5)),
            ),
        )
        for rid in region_ids
    }
    desired_workers = {
        rid: max(
            0,
            int(math.ceil(stabilized_desired_workers[rid] * max(0.65, float(platform_need_scale)))),
        )
        for rid in region_ids
    }
    phase = str(dispatch_phase).lower()
    slot_minutes = max(1e-6, float(getattr(config, 'EXPERIMENT_TIME_SLOT_MINUTES', 15)))
    retain_horizon_minutes = float(np.clip(
        getattr(config, 'RBG_REALTIME_RETAIN_HORIZON_MINUTES', 1.0),
        0.0,
        slot_minutes,
    ))
    backlog_guard_weight = max(0.0, float(getattr(config, 'RBG_RETAIN_BACKLOG_GUARD_WEIGHT', 1.0)))
    uncertainty_buffer_weight = max(0.0, float(getattr(config, 'RBG_RETAIN_UNCERTAINTY_BUFFER_WEIGHT', 0.50)))
    micro_guard_multiplier = max(1.0, float(getattr(config, 'RBG_MICRO_RETENTION_GUARD_MULTIPLIER', 1.15)))
    current_supply_anchor = {
        rid: max(0, int(available_workers.get(rid, 0)))
        for rid in region_ids
    }
    if phase == "slot_start":
        slot_start_blend = float(np.clip(
            getattr(config, 'RBG_SLOT_START_DEMAND_BLEND', 0.60) * max(0.85, float(platform_slot_start_blend_scale)),
            0.25,
            1.0,
        ))
        desired_workers = {
            rid: max(
                0,
                int(round(
                    current_supply_anchor[rid] * (1.0 - slot_start_blend)
                    + desired_workers[rid] * slot_start_blend
                )),
            )
            for rid in region_ids
        }
    shortage_guess = {
        rid: max(0.0, float(desired_workers[rid] - available_workers.get(rid, 0)))
        for rid in region_ids
    }

    transitions: Dict[int, Dict[str, Any]] = {}
    retained_workers_by_region: Dict[int, List[str]] = {rid: [] for rid in region_ids}
    releasable_workers_by_region: Dict[int, List[str]] = {rid: [] for rid in region_ids}
    retain_count_by_region: Dict[int, int] = {}
    safe_reserve_by_region: Dict[int, int] = {}
    hoard_penalty_by_region: Dict[int, float] = {}
    action_ratio_by_region: Dict[int, float] = {}
    action_index_by_region: Dict[int, int] = {}
    platform_reward_weight_by_region: Dict[int, float] = {}
    platform_release_credit_by_region: Dict[int, float] = {}

    avg_effective_demand = max(
        1.0,
        float(np.mean([demand_profile[rid]["effective_demand"] for rid in region_ids])) if region_ids else 1.0,
    )
    total_shortage_guess = max(1.0, float(sum(shortage_guess.values())))

    total_backlog = float(sum(max(0, backlog_counts.get(rid, 0)) for rid in region_ids))
    for rid in region_ids:
        idle_count = available_workers.get(rid, 0)
        profile = demand_profile[rid]
        base_keep = min(
            idle_count,
            max(
                min_buffer_workers,
                int(math.ceil(desired_workers[rid])),
            ),
        )
        keep_scale_effective = float(np.clip(float(platform_keep_scale), 0.85, 1.15))
        keep_scale_effective = float(np.clip(keep_scale_effective, 0.70, 1.20))
        base_keep = min(
            idle_count,
            max(
                min_buffer_workers,
                int(round(base_keep * keep_scale_effective)),
            ),
        )
        current_backlog_tasks = max(0.0, float(backlog_counts.get(rid, 0))) * backlog_guard_weight
        near_future_arrivals = max(
            0.0,
            float(predicted_demand.get(rid, 0)) * (retain_horizon_minutes / slot_minutes),
        )
        abs_err_ema = max(0.0, float(profile.get("abs_err_ema", 0.0)))
        underpredict_bias = max(0.0, -float(profile.get("combined_bias", 0.0)))
        guard_task_pressure = current_backlog_tasks + near_future_arrivals + uncertainty_buffer_weight * (
            abs_err_ema + underpredict_bias
        )
        if phase != "slot_start":
            guard_task_pressure *= micro_guard_multiplier
        guard_keep_floor = min(
            idle_count,
            max(
                min_buffer_workers,
                int(math.ceil(guard_task_pressure / max_tasks_per_worker)),
            ),
        )
        base_keep = max(base_keep, guard_keep_floor)
        safe_reserve_by_region[rid] = guard_keep_floor
        neighbor_backlog_pressure = max(0.0, total_backlog - float(backlog_counts.get(rid, 0)))
        features = state.build_features(
            region_id=rid,
            demand_profile=demand_profile,
            available_workers=available_workers,
            base_keep=base_keep,
            shortage_workers=shortage_guess[rid],
            neighbor_backlog_pressure=neighbor_backlog_pressure,
            max_tasks_per_worker=max_tasks_per_worker,
        )
        action_idx, action_ratio, probs = state.sample_action(rid, features)
        retain_anchor = max(base_keep, guard_keep_floor)
        retain_slack = max(0, idle_count - retain_anchor)
        retain_count = int(np.clip(
            retain_anchor + round(float(action_ratio) * max(1, retain_slack)),
            guard_keep_floor,
            idle_count,
        ))
        retain_count_by_region[rid] = retain_count
        action_ratio_by_region[rid] = action_ratio
        action_index_by_region[rid] = action_idx
        hoard_penalty_by_region[rid] = 0.0
        platform_reward_weight_by_region[rid] = float(np.clip(
            1.0
            + platform_task_weight * (demand_profile[rid]["effective_demand"] / avg_effective_demand - 1.0)
            + platform_gap_weight * (shortage_guess[rid] / total_shortage_guess),
            0.70,
            2.50,
        ))
        platform_release_credit_by_region[rid] = 0.0
        if record_transition:
            transitions[rid] = {
                "features": features,
                "probs": probs,
                "action_idx": action_idx,
                "retain_count": retain_count,
                "base_keep": base_keep,
                "guard_keep_floor": guard_keep_floor,
                "platform_reward_weight": platform_reward_weight_by_region[rid],
            }

    shortage_receivers = [rid for rid in region_ids if max(0, desired_workers[rid] - retain_count_by_region[rid]) > 0]
    for rid in region_ids:
        retained, releasable = _select_retained_and_releasable_workers(
            region_id=rid,
            worker_ids=movable_workers[rid],
            retain_count=retain_count_by_region[rid],
            shortage_receivers=shortage_receivers,
            receiver_need={
                region: max(0, desired_workers[region] - retain_count_by_region[region])
                for region in region_ids
            },
            demand_profile=demand_profile,
            selection_action_ratio=action_ratio_by_region[rid],
            worker_sim=worker_sim,
            centers=centers,
            state=state,
        )
        retained_workers_by_region[rid] = retained
        releasable_workers_by_region[rid] = releasable

    receiver_need = {
        rid: max(0, desired_workers[rid] - retain_count_by_region[rid])
        for rid in region_ids
    }
    donor_supply = {
        rid: len(releasable_workers_by_region[rid])
        for rid in region_ids
    }
    moves: List[Dict[str, Any]] = []
    move_cost_by_region = {rid: 0.0 for rid in region_ids}
    center_distance_cache: Dict[Tuple[Any, Any], float] = {}
    total_available_worker_count = int(sum(available_workers.values()))
    if phase == "micro":
        max_total_move_share = float(np.clip(
            getattr(config, 'RBG_MICRO_MAX_MOVE_SHARE', 0.80) * max(1.0, float(platform_move_share_scale)),
            0.05,
            1.0,
        ))
    else:
        max_total_move_share = float(np.clip(
            getattr(config, 'RBG_SLOT_START_MAX_MOVE_SHARE', 0.45) * max(1.0, float(platform_move_share_scale)),
            0.05,
            1.0,
        ))
    max_total_moves = max(1, int(math.ceil(total_available_worker_count * max_total_move_share)))

    def get_center_distance(donor_region: int, receiver_region: int) -> float:
        donor_node = centers[donor_region]
        receiver_node = centers[receiver_region]
        cache_key = (donor_node, receiver_node)
        if cache_key in center_distance_cache:
            return center_distance_cache[cache_key]
        try:
            distance = float(nx.shortest_path_length(G, source=donor_node, target=receiver_node, weight="length"))
        except nx.NetworkXNoPath:
            distance = float("inf")
        center_distance_cache[cache_key] = distance
        return distance

    while True:
        if len(moves) >= max_total_moves:
            break
        best_edge = None
        best_gain = edge_epsilon
        for donor in region_ids:
            if donor_supply[donor] <= 0:
                continue
            local_gap = max(0.0, float(desired_workers[donor] - retain_count_by_region[donor]))
            for receiver in region_ids:
                if receiver == donor or receiver_need[receiver] <= 0:
                    continue
                center_distance = get_center_distance(donor, receiver)
                if not np.isfinite(center_distance):
                    continue

                receiver_profile = demand_profile[receiver]
                donor_profile = demand_profile[donor]
                bid = (
                    center_local_task_weight * float(receiver_need[receiver])
                    + platform_reward_weight_by_region[receiver]
                    + bid_shortage_weight * float(receiver_need[receiver])
                    + bid_backlog_weight * receiver_profile["backlog"]
                    + state.get_receiver_affinity(donor, receiver)
                )
                ask = center_local_task_weight * local_gap + ask_shortage_weight * local_gap
                gain = bid - ask
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (donor, receiver, center_distance)

        if best_edge is None:
            break

        donor, receiver, center_distance = best_edge
        chosen_worker = _choose_worker_for_receiver(
            donor_region=donor,
            receiver_region=receiver,
            releasable_workers=releasable_workers_by_region[donor],
            receiver_need=receiver_need,
            demand_profile=demand_profile,
            donor_action_ratio=action_ratio_by_region.get(donor, 0.0),
            worker_sim=worker_sim,
            centers=centers,
            state=state,
            candidate_k=candidate_k,
        )
        if chosen_worker is None:
            donor_supply[donor] = 0
            continue

        releasable_workers_by_region[donor].remove(chosen_worker)
        donor_supply[donor] -= 1
        receiver_need[receiver] = max(0, receiver_need[receiver] - 1)
        move_cost_by_region[donor] += center_distance / 1000.0
        worker_sim.worker_center_map[chosen_worker] = receiver
        moves.append(
            {
                "wid": chosen_worker,
                "from_region": donor,
                "to_region": receiver,
                "distance_m": center_distance,
            }
        )

    if moves:
        state.record_moves(slot_idx=slot_idx, moved_workers=[move["wid"] for move in moves])

    diagnostics = {
        "retain_count": retain_count_by_region,
        "safe_reserve": safe_reserve_by_region,
        "receiver_need_after_trade": receiver_need,
        "donor_supply_after_trade": donor_supply,
        "hoard_penalty": hoard_penalty_by_region,
        "action_ratio": action_ratio_by_region,
        "action_index": action_index_by_region,
        "platform_reward_weight": platform_reward_weight_by_region,
        "platform_release_credit": platform_release_credit_by_region,
    }

    return {
        "moves": moves,
        "available_workers": available_workers,
        "desired_workers": desired_workers,
        "retain_count": retain_count_by_region,
        "raw_desired_workers": raw_desired_workers,
        "allocation_target_workers": allocation_target_workers,
        "retained_workers": retained_workers_by_region,
        "releasable_workers": releasable_workers_by_region,
        "demand_profile": demand_profile,
        "effective_demand": {rid: demand_profile[rid]["effective_demand"] for rid in region_ids},
        "hoard_penalty": hoard_penalty_by_region,
        "move_cost_by_region": move_cost_by_region,
        "transitions": transitions,
        "diagnostics": diagnostics,
        "stackelberg_control": {
            "region_priority_weight": platform_reward_weight_by_region,
            "worker_completion_bonus": float(worker_completion_bonus),
            "worker_distance_penalty": float(worker_distance_penalty),
            "same_worker_chain_bonus": float(same_worker_chain_bonus),
            "platform_release_credit": platform_release_credit_by_region,
        },
    }


def update_rl_retention_bilateral_state(
    state: RLRetentionBilateralState,
    transitions: Dict[int, Dict[str, Any]],
    assigned_tasks_by_region: Dict[int, int],
    total_tasks_by_region: Dict[int, int],
    hoard_penalty_by_region: Dict[int, float],
    move_cost_by_region: Dict[int, float],
    moves: Optional[List[Dict[str, Any]]] = None,
    hoard_penalty_weight: float = 0.02,
    move_cost_weight: float = 0.08,
    unfairness_weight: float = 1.0,
) -> Dict[int, float]:
    reward_by_region: Dict[int, float] = {}
    total_assigned = float(sum(assigned_tasks_by_region.get(rid, 0) for rid in state.region_ids))
    global_task_reward_weight = float(
        getattr(config, 'RBG_GLOBAL_TASK_REWARD_WEIGHT', 0.35)
    )
    for rid in state.region_ids:
        served = float(assigned_tasks_by_region.get(rid, 0))
        reward_by_region[rid] = served + global_task_reward_weight * total_assigned
    state.update_policy(
        transitions=transitions,
        reward_by_region=reward_by_region,
        total_tasks_by_region=total_tasks_by_region,
        assigned_tasks_by_region=assigned_tasks_by_region,
    )
    if moves:
        state.update_receiver_affinity(
            moves=moves,
            reward_by_region=reward_by_region,
        )
    return reward_by_region


def update_platform_task_first_state(
    state: PlatformTaskFirstRLState,
    transition: Dict[str, Any],
    assigned_tasks_by_region: Dict[int, int],
    total_tasks_by_region: Dict[int, int],
    fairness_secondary_weight: float = 0.20,
) -> Dict[str, float]:
    total_assigned = float(sum(assigned_tasks_by_region.values()))
    total_tasks = float(sum(total_tasks_by_region.values()))
    completion_rate = (total_assigned / total_tasks) if total_tasks > 0 else 0.0
    service_ratio = _compute_service_ratio(assigned_tasks_by_region, total_tasks_by_region)
    unfairness_by_region = _compute_pairwise_unfairness(service_ratio)
    mean_unfairness = float(np.mean(list(unfairness_by_region.values()))) if unfairness_by_region else 0.0
    reward = total_assigned
    state.update_policy(
        features=transition["features"],
        action_idx=int(transition["action_idx"]),
        reward=reward,
        next_features=transition.get("next_features"),
        done=float(transition.get("done", 1.0)),
        completion_rate=completion_rate,
        unfairness=mean_unfairness,
    )
    return {
        "platform_reward": reward,
        "completion_rate": completion_rate,
        "mean_unfairness": mean_unfairness,
    }


def offline_warm_start_retention_policy(
    state: RLRetentionBilateralState,
    historical_samples: List[Dict[str, Any]],
    max_tasks_per_worker: int = 4,
    min_buffer_workers: int = 1,
    reserve_ratio: float = 0.10,
    backlog_weight: float = 1.0,
    uncertainty_weight: float = 0.45,
    quantile_weight: float = 0.55,
    burst_weight: float = 1.2,
    epochs: int = 3,
) -> Dict[str, Any]:
    if not historical_samples:
        return {"sample_count": 0, "epochs": 0, "action_histogram": {}}

    action_histogram = {idx: 0 for idx in range(len(state.action_ratios))}
    max_tasks_per_worker = max(1, int(max_tasks_per_worker))
    epochs = max(1, int(epochs))

    for _ in range(epochs):
        state.service_debt = {rid: 0.0 for rid in state.region_ids}
        for sample in historical_samples:
            available_workers = {
                int(rid): int(count)
                for rid, count in sample.get("available_workers", {}).items()
            }
            backlog_counts = {
                int(rid): int(count)
                for rid, count in sample.get("backlog_counts", {}).items()
            }
            actual_counts = {
                int(rid): float(count)
                for rid, count in sample.get("actual_counts", {}).items()
            }
            sigma_map = {
                int(rid): float(value)
                for rid, value in sample.get("sigma_map", {}).items()
            }
            q90_map = {
                int(rid): float(value)
                for rid, value in sample.get("q90_map", {}).items()
            }
            burst_map = {
                int(rid): float(value)
                for rid, value in sample.get("burst_map", {}).items()
            }

            demand_profile: Dict[int, Dict[str, float]] = {}
            desired_workers: Dict[int, int] = {}
            base_keep_by_region: Dict[int, int] = {}

            for rid in state.region_ids:
                mu = max(0.0, actual_counts.get(rid, 0.0))
                backlog = max(0.0, float(backlog_counts.get(rid, 0)))
                sigma = max(0.0, sigma_map.get(rid, math.sqrt(max(mu, 1.0)) * 0.20))
                q90 = max(mu, q90_map.get(rid, mu + sigma))
                tail_gap = max(0.0, q90 - mu)
                burst = float(np.clip(burst_map.get(rid, 0.0), 0.0, 1.0))
                debt = float(state.service_debt.get(rid, 0.0))
                effective = (
                    mu
                    + backlog_weight * backlog
                    + uncertainty_weight * sigma
                    + quantile_weight * tail_gap
                    + burst_weight * burst * max(1.0, sigma)
                )
                demand_profile[rid] = {
                    "mu": mu,
                    "sigma": sigma,
                    "q90": q90,
                    "tail_gap": tail_gap,
                    "burst_prob": burst,
                    "backlog": backlog,
                    "debt": debt,
                    "effective_demand": effective,
                }
                desired_workers[rid] = int(math.ceil(effective / max_tasks_per_worker))
                idle_count = int(available_workers.get(rid, 0))
                base_keep_by_region[rid] = min(
                    idle_count,
                    max(
                        min_buffer_workers,
                        int(math.ceil(desired_workers[rid] * (1.0 + max(0.0, reserve_ratio)))),
                    ),
                )

            total_shortage = sum(
                max(0, desired_workers[rid] - base_keep_by_region[rid])
                for rid in state.region_ids
            )

            served_proxy_by_region: Dict[int, int] = {}
            total_tasks_by_region: Dict[int, int] = {}
            for rid in state.region_ids:
                idle_count = int(available_workers.get(rid, 0))
                base_keep = int(base_keep_by_region[rid])
                local_shortage = max(0, desired_workers[rid] - base_keep)
                other_shortage = max(0, total_shortage - local_shortage)
                neighbor_backlog_pressure = max(
                    0.0,
                    float(sum(backlog_counts.get(other, 0) for other in state.region_ids if other != rid))
                )

                features = state.build_features(
                    region_id=rid,
                    demand_profile=demand_profile,
                    available_workers=available_workers,
                    base_keep=base_keep,
                    shortage_workers=float(local_shortage),
                    neighbor_backlog_pressure=neighbor_backlog_pressure,
                    max_tasks_per_worker=max_tasks_per_worker,
                )

                target_ratio = 0.0
                if idle_count > 0:
                    shortage_ratio = float(local_shortage) / max(1.0, float(idle_count))
                    slack_workers = max(0, idle_count - base_keep)
                    if local_shortage > 0:
                        if (
                            shortage_ratio >= 0.25
                            or demand_profile[rid]["burst_prob"] >= 0.35
                            or demand_profile[rid]["debt"] >= 1.0
                        ):
                            target_ratio = 0.30
                        else:
                            target_ratio = 0.15
                    elif slack_workers > 0 and other_shortage > 0:
                        release_ratio = float(min(slack_workers, other_shortage)) / max(1.0, float(idle_count))
                        if release_ratio >= 0.25:
                            target_ratio = -0.30
                        else:
                            target_ratio = -0.15

                action_idx = int(np.argmin(np.abs(np.asarray(state.action_ratios) - target_ratio)))
                action_histogram[action_idx] += 1
                confidence = 1.0 + abs(target_ratio) * 2.0
                state.imitation_update(
                    region_id=rid,
                    features=features,
                    target_action_idx=action_idx,
                    strength=confidence,
                )

                retain_count = int(np.clip(
                    base_keep + round(float(state.action_ratios[action_idx]) * max(1, idle_count)),
                    0,
                    idle_count,
                ))
                served_proxy_by_region[rid] = min(
                    int(round(demand_profile[rid]["mu"] + demand_profile[rid]["backlog"])),
                    retain_count * max_tasks_per_worker,
                )
                total_tasks_by_region[rid] = max(
                    0,
                    int(round(demand_profile[rid]["mu"] + demand_profile[rid]["backlog"]))
                )

            service_ratio = _compute_service_ratio(served_proxy_by_region, total_tasks_by_region)
            for rid in state.region_ids:
                prev_debt = float(state.service_debt.get(rid, 0.0))
                total_tasks = total_tasks_by_region.get(rid, 0)
                if total_tasks <= 0:
                    state.service_debt[rid] = max(0.0, prev_debt * state.service_debt_decay)
                else:
                    updated = prev_debt * state.service_debt_decay + max(0.0, 1.0 - service_ratio.get(rid, 1.0))
                    state.service_debt[rid] = min(state.max_service_debt, updated)

    return {
        "sample_count": len(historical_samples),
        "epochs": epochs,
        "action_histogram": action_histogram,
    }
