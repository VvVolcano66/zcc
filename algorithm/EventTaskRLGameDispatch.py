from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Deque, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

import config
from algorithm.CenterPrepackedAssignment import center_prepacked_assignment_with_center_pickup
from algorithm.RLRetentionGameDispatch import (
    ContinuousSACAgent,
    PlatformTaskFirstRLState,
    RLRetentionBilateralState,
    sample_platform_task_first_control,
)


@dataclass
class CenterTaskAllocationRLState:
    """Per-center task portfolio policy for predicted tasks and live backlog."""

    region_ids: List[int]
    action_bounds: Tuple[Tuple[float, float], ...] = (
        (0.60, 1.60),
        (0.60, 1.60),
        (0.60, 1.60),
        (0.70, 1.50),
        (0.30, 1.50),
        (0.00, 1.00),
    )
    learning_rate: float = 0.03
    exploration_prob: float = 0.02
    gamma: float = 0.60
    replay_capacity: int = 512
    batch_size: int = 32
    hidden_dim: int = 32
    critic_learning_rate: Optional[float] = None
    alpha_learning_rate: float = 0.001
    sac_tau: float = 0.01
    sac_initial_alpha: float = 0.20
    sac_auto_entropy: bool = True
    sac_target_entropy_scale: float = 0.98
    random_seed: int = 42
    device: Optional[str] = None
    feature_dim: int = 10
    prediction_trust_decay: float = 0.80
    sac_agents: Dict[int, ContinuousSACAgent] = field(default_factory=dict)
    replay_buffers: Dict[int, Deque[Dict[str, Any]]] = field(default_factory=dict)
    update_steps: Dict[int, int] = field(default_factory=dict)
    prediction_trust: Dict[int, float] = field(default_factory=dict)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.action_bounds = tuple((float(low), float(high)) for low, high in self.action_bounds)
        self.rng = np.random.default_rng(self.random_seed)
        preferred_device = self.device or getattr(config, "TORCH_DEVICE_PREFERENCE", None)
        self.device = str(torch.device(preferred_device if preferred_device else ("cuda" if torch.cuda.is_available() else "cpu")))
        for rid in self.region_ids:
            self.sac_agents[rid] = ContinuousSACAgent(
                feature_dim=self.feature_dim,
                action_bounds=self.action_bounds,
                hidden_dim=self.hidden_dim,
                device=self.device,
                actor_learning_rate=self.learning_rate,
                critic_learning_rate=self.critic_learning_rate or self.learning_rate,
                alpha_learning_rate=self.alpha_learning_rate,
                gamma=self.gamma,
                tau=self.sac_tau,
                initial_alpha=self.sac_initial_alpha,
                auto_entropy=self.sac_auto_entropy,
                target_entropy_scale=self.sac_target_entropy_scale,
            )
            self.replay_buffers[rid] = deque(maxlen=max(32, int(self.replay_capacity)))
            self.update_steps[rid] = 0
            self.prediction_trust.setdefault(rid, 1.0)

    def action_profile(self, action: np.ndarray) -> Dict[str, Any]:
        return {
            "name": "continuous",
            "backlog_weight": float(action[0]),
            "predicted_weight": float(action[1]),
            "urgency_weight": float(action[2]),
            "candidate_scale": float(action[3]),
            "precommit_bonus": float(action[4]),
            "prediction_risk_penalty": float(action[5]),
        }

    def record_prediction_feedback(
        self,
        predicted_region_demand: Dict[int, int],
        actual_region_demand: Dict[int, int],
    ) -> None:
        decay = float(getattr(config, "EVENT_CENTER_PREDICTION_TRUST_DECAY", self.prediction_trust_decay))
        decay = min(0.99, max(0.0, decay))
        for rid in self.region_ids:
            predicted = max(0.0, float(predicted_region_demand.get(rid, 0)))
            actual = max(0.0, float(actual_region_demand.get(rid, 0)))
            error_ratio = abs(predicted - actual) / max(1.0, predicted, actual)
            instant_trust = max(0.0, 1.0 - error_ratio)
            old_trust = float(self.prediction_trust.get(rid, 1.0))
            self.prediction_trust[rid] = decay * old_trust + (1.0 - decay) * instant_trust

    def build_features(
        self,
        predicted_count: int,
        backlog_count: int,
        worker_count: int,
        predicted_distribution: Optional[Dict[str, float]],
    ) -> np.ndarray:
        distribution = predicted_distribution or {}
        capacity = max(1.0, float(worker_count * int(getattr(config, "MAX_TASKS_PER_WORKER", 4))))
        predicted = float(max(0, predicted_count))
        backlog = float(max(0, backlog_count))
        sigma = float(max(0.0, distribution.get("sigma", 0.0)))
        q90 = float(max(predicted, distribution.get("q90", predicted)))
        trust = float(max(0.0, min(1.0, distribution.get("trust", 1.0))))
        scale = max(1.0, capacity, predicted + backlog)
        return np.asarray(
            [
                1.0,
                predicted / scale,
                backlog / scale,
                capacity / scale,
                max(0.0, predicted + backlog - capacity) / scale,
                sigma / scale,
                max(0.0, q90 - predicted) / scale,
                backlog / max(1.0, predicted + backlog),
                min(1.0, capacity / max(1.0, predicted + backlog)),
                trust,
            ],
            dtype=np.float32,
        )

    def sample_action(self, region_id: int, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
        action, normalized_action = self.sac_agents[region_id].sample_action(
            features=features,
            exploration_noise=self.exploration_prob,
        )
        return action, self.action_profile(action), normalized_action

    def update_policy(
        self,
        transitions: Dict[int, Dict[str, Any]],
        assigned_tasks_by_region: Dict[int, int],
        total_tasks_by_region: Dict[int, int],
        remaining_tasks_by_region: Optional[Dict[int, int]] = None,
        actual_arrivals_by_region: Optional[Dict[int, int]] = None,
    ) -> Dict[int, float]:
        rewards: Dict[int, float] = {}
        remaining_tasks_by_region = remaining_tasks_by_region or {}
        total_assigned = float(sum(assigned_tasks_by_region.values()))
        for rid, transition in transitions.items():
            assigned = float(assigned_tasks_by_region.get(rid, 0))
            remaining = float(remaining_tasks_by_region.get(rid, 0))
            total = max(1.0, float(total_tasks_by_region.get(rid, 0)))
            predicted = float(transition.get("predicted_count", 0.0))
            actual = float((actual_arrivals_by_region or {}).get(rid, predicted))
            overprediction = max(0.0, predicted - actual)
            overprediction_penalty = float(getattr(config, "EVENT_CENTER_OVERPREDICTION_PENALTY", 0.35))
            reward = (
                assigned
                + 0.15 * total_assigned
                - 0.20 * remaining
                - 0.20 * max(0.0, remaining / total)
                - overprediction_penalty * overprediction
            )
            rewards[rid] = reward
            self.replay_buffers[rid].append(
                {
                    "features": np.asarray(transition["features"], dtype=np.float32),
                    "action": np.asarray(transition["action"], dtype=np.float32),
                    "reward": float(reward),
                    "next_features": np.asarray(transition.get("next_features", transition["features"]), dtype=np.float32),
                    "done": float(transition.get("done", 1.0)),
                }
            )
            self._train_region_sac(rid)
        return rewards

    def _train_region_sac(self, region_id: int) -> None:
        buffer = self.replay_buffers[region_id]
        if not buffer:
            return
        batch_size = min(len(buffer), max(1, int(self.batch_size)))
        indices = self.rng.choice(len(buffer), size=batch_size, replace=False)
        batch = [buffer[int(index)] for index in indices]
        self.sac_agents[region_id].train_batch(batch)
        self.update_steps[region_id] = self.sac_agents[region_id].update_steps

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
                    self._train_region_sac(rid)
                    optimization_steps += 1

        return {
            "epochs": epochs,
            "updates_per_region": updates_per_region,
            "trained_region_count": len(trained_regions),
            "optimization_steps": optimization_steps,
        }


def _build_center_task_plan(
    centers: Dict[int, Any],
    backlog_tasks: Dict[int, list],
    predicted_tasks: Dict[int, list],
    workers_per_center: Dict[int, list],
    action_profiles: Dict[int, Dict[str, Any]],
    prediction_trust: Dict[int, float],
    slot_start_seconds: float,
) -> Dict[int, list]:
    max_tasks = max(1, int(getattr(config, "MAX_TASKS_PER_WORKER", 4)))
    default_factor = max(1.0, float(getattr(config, "EVENT_PLAN_TASK_CANDIDATE_FACTOR", 3)))
    planning_tasks = {}
    for rid in centers:
        action = action_profiles[rid]
        backlog_ids = {str(task[1]) for task in backlog_tasks.get(rid, [])}
        candidates = list(backlog_tasks.get(rid, [])) + list(predicted_tasks.get(rid, []))
        scored = []
        for task in candidates:
            is_backlog = str(task[1]) in backlog_ids
            score_weight = _center_task_score(
                task=task,
                is_backlog=is_backlog,
                action=action,
                slot_start_seconds=slot_start_seconds,
                prediction_trust=prediction_trust.get(rid, 1.0),
            )
            scored.append((score_weight, task))
        scored.sort(key=lambda item: (-item[0], float(item[1][3]), str(item[1][1])))
        limit = max(
            len(backlog_tasks.get(rid, [])),
            int(math.ceil(len(workers_per_center.get(rid, [])) * max_tasks * default_factor * action["candidate_scale"])),
        )
        planning_tasks[rid] = [
            (task[0], task[1], float(task[2]) * score_weight, task[3], task[4])
            for score_weight, task in scored[:limit]
        ]
    return planning_tasks


def _build_precommit_worker_task_priority_map(
    planned_details: List[Dict[str, Any]],
    action_profiles: Dict[int, Dict[str, Any]],
    predicted_tasks: Dict[int, list],
    prediction_trust: Dict[int, float],
) -> Dict[str, float]:
    predicted_task_ids = {
        rid: {str(task[1]) for task in predicted_tasks.get(rid, [])}
        for rid in action_profiles
    }
    priority_map: Dict[str, float] = {}
    for detail in planned_details:
        rid = int(detail["region_id"])
        wid = str(detail["wid"])
        tid = str(detail["task_id"])
        action = action_profiles.get(rid, {})
        precommit_bonus = float(action.get("precommit_bonus", 0.90))
        if tid in predicted_task_ids.get(rid, set()):
            precommit_bonus *= max(0.0, min(1.0, float(prediction_trust.get(rid, 1.0))))
        priority_map[f"{wid}|{tid}"] = max(1.0, 1.0 + precommit_bonus)
    return priority_map


def _build_precommit_bundles_by_worker(
    planned_details: List[Dict[str, Any]],
    planning_task_by_id: Dict[str, tuple],
    predicted_tasks: Dict[int, list],
    worker_task_priority_map: Dict[str, float],
    slot_start_seconds: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    predicted_task_ids = {
        rid: {str(task[1]) for task in tasks}
        for rid, tasks in predicted_tasks.items()
    }
    details_by_worker: Dict[str, List[Dict[str, Any]]] = {}
    for detail in planned_details:
        details_by_worker.setdefault(str(detail["wid"]), []).append(detail)

    bundles_by_worker: Dict[str, Dict[str, Any]] = {}
    task_bundle_map: Dict[str, str] = {}
    for wid, details in details_by_worker.items():
        ordered_details = sorted(
            details,
            key=lambda item: (
                int(item.get("round_id", 0)),
                float(item.get("service_finish_time", item.get("finish_time", 0.0))),
                str(item.get("task_id")),
            ),
        )
        anchors = []
        for sequence_index, detail in enumerate(ordered_details):
            rid = int(detail["region_id"])
            task_id = str(detail["task_id"])
            if task_id not in predicted_task_ids.get(rid, set()):
                continue
            task = planning_task_by_id.get(task_id)
            if task is None:
                continue
            round_id = int(detail.get("round_id", 0))
            bundle_id = f"{wid}:{round_id}"
            task_bundle_map[task_id] = bundle_id
            anchors.append(
                {
                    "bundle_id": bundle_id,
                    "region_id": rid,
                    "task_id": task_id,
                    "node": task[0],
                    "reward": float(task[2]),
                    "expire_time": float(task[3]),
                    "release_time": float(task[4]) if len(task) > 4 else float(slot_start_seconds),
                    "sequence_index": sequence_index,
                    "round_id": round_id,
                    "priority_weight": float(worker_task_priority_map.get(f"{wid}|{task_id}", 1.0)),
                }
            )
        if anchors:
            bundles_by_worker[wid] = {
                "worker_id": wid,
                "anchors": anchors,
            }
    return bundles_by_worker, task_bundle_map


def _center_task_score(
    task: tuple,
    is_backlog: bool,
    action: Dict[str, Any],
    slot_start_seconds: float,
    prediction_trust: float,
) -> float:
    expiry = float(task[3])
    remaining = max(1.0, expiry - slot_start_seconds)
    urgency = 1.0 + 60.0 / remaining
    if is_backlog:
        source_weight = float(action["backlog_weight"])
    else:
        trust = max(0.0, min(1.0, float(prediction_trust)))
        source_weight = (
            float(action["predicted_weight"]) * trust
            - float(action["prediction_risk_penalty"]) * (1.0 - trust)
        )
    return max(0.05, source_weight + float(action["urgency_weight"]) * urgency)


def _event_precommit_worker_horizon(slot_start_seconds: float, slot_end_seconds: float) -> float:
    if not bool(getattr(config, "EVENT_PRECOMMIT_INCLUDE_FUTURE_WORKERS", True)):
        return float(slot_start_seconds)
    horizon_seconds = float(
        getattr(
            config,
            "EVENT_PRECOMMIT_FUTURE_WORKER_HORIZON_SECONDS",
            max(0.0, float(slot_end_seconds) - float(slot_start_seconds)),
        )
    )
    if horizon_seconds <= 0.0:
        return float(slot_end_seconds)
    return min(float(slot_end_seconds), float(slot_start_seconds) + horizon_seconds)


def _collect_workers(
    worker_sim,
    centers: Dict[int, Any],
    current_time: float,
    include_future: bool = False,
    future_until: Optional[float] = None,
) -> Dict[int, list]:
    workers = {}
    seen_worker_ids = set()
    for rid in sorted(centers):
        available = worker_sim.get_available_workers_with_center_info(rid, current_time=current_time)
        workers[rid] = [(item[0], item[1], item[2], item[3], centers[rid]) for item in available]
        seen_worker_ids.update(str(item[1]) for item in available)

    if not include_future:
        return workers

    horizon = float(future_until) if future_until is not None else float(current_time)
    if horizon <= float(current_time):
        return workers

    for wid, status in getattr(worker_sim, "worker_status", {}).items():
        wid = str(wid)
        if wid in seen_worker_ids or status != "en_route_to_task":
            continue
        ready_time = getattr(worker_sim, "worker_busy_until", {}).get(wid)
        if ready_time is None:
            continue
        ready_time = float(ready_time)
        if ready_time <= float(current_time) + 1e-6 or ready_time > horizon + 1e-6:
            continue
        rid = getattr(worker_sim, "worker_center_map", {}).get(wid)
        if rid not in centers:
            continue
        position = getattr(worker_sim, "worker_positions", {}).get(wid)
        if position is None:
            continue
        node, lon, lat = position
        workers[int(rid)].append((node, wid, lon, lat, centers[int(rid)], ready_time))
        seen_worker_ids.add(wid)
    return workers


def _resolve_event_precommit_service_horizon(
    planning_tasks: Dict[int, list],
    slot_start_seconds: float,
    slot_end_seconds: float,
) -> float:
    if not bool(getattr(config, "ONLINE_ALLOW_SERVICE_AFTER_BATCH_END", False)):
        return float(slot_end_seconds)
    max_expire = None
    for tasks in planning_tasks.values():
        for task in tasks:
            expire_time = float(task[3])
            max_expire = expire_time if max_expire is None else max(max_expire, expire_time)
    if max_expire is None:
        return float(slot_end_seconds)
    max_overtime = max(0.0, float(getattr(config, "ONLINE_DRAIN_MAX_SECONDS", 0.0)))
    capped_horizon = float(slot_end_seconds) + max_overtime if max_overtime > 0.0 else max_expire
    return max(float(slot_end_seconds), min(max_expire, capped_horizon))


def _run_virtual_center_assignment(
    G,
    centers: Dict[int, Any],
    workers_per_center: Dict[int, list],
    planning_tasks: Dict[int, list],
    slot_start_seconds: float,
    slot_end_seconds: float,
):
    planning_end_seconds = _resolve_event_precommit_service_horizon(
        planning_tasks=planning_tasks,
        slot_start_seconds=slot_start_seconds,
        slot_end_seconds=slot_end_seconds,
    )
    return center_prepacked_assignment_with_center_pickup(
        G=G,
        config=config,
        centers_dict=centers,
        workers_per_center=workers_per_center,
        tasks_per_center=planning_tasks,
        slot_start_seconds=slot_start_seconds,
        slot_end_seconds=planning_end_seconds,
        force_center_pickup_on_first_departure=True,
    )


def _platform_loan_surplus_workers(
    G,
    worker_sim,
    centers: Dict[int, Any],
    workers_per_center: Dict[int, list],
    preliminary_details: List[Dict[str, Any]],
    uncovered_tasks: Dict[int, int],
    platform_transition: Optional[Dict[str, Any]],
    max_tasks_per_worker: int,
    slot_start_seconds: Optional[float] = None,
    slot_end_seconds: Optional[float] = None,
) -> List[Dict[str, Any]]:
    used_workers = {str(detail["wid"]) for detail in preliminary_details}
    surplus = {
        rid: [str(worker[1]) for worker in workers_per_center[rid] if str(worker[1]) not in used_workers]
        for rid in centers
    }
    receiver_need = {
        rid: int(math.ceil(max(0, uncovered_tasks.get(rid, 0)) / max(1, max_tasks_per_worker)))
        for rid in centers
    }
    transition = platform_transition or {}
    task_weight = float(transition.get("task_weight", 1.0))
    gap_weight = float(transition.get("gap_weight", 1.0))
    move_scale = float(transition.get("move_share_scale", 1.0))
    distance_weight = float(getattr(config, "EVENT_PLATFORM_LOAN_DISTANCE_PENALTY", 0.35))
    post_service_surplus: Dict[int, List[Tuple[float, str]]] = {rid: [] for rid in centers}
    if (
        bool(getattr(config, "EVENT_PLATFORM_POST_SERVICE_LOAN_ENABLED", True))
        and slot_start_seconds is not None
        and slot_end_seconds is not None
    ):
        horizon_seconds = max(0.0, float(getattr(config, "EVENT_PLATFORM_POST_SERVICE_LOAN_HORIZON_SECONDS", 900.0)))
        horizon = min(float(slot_end_seconds), float(slot_start_seconds) + horizon_seconds)
        for wid, status in getattr(worker_sim, "worker_status", {}).items():
            wid = str(wid)
            if status != "en_route_to_task" or wid in used_workers:
                continue
            ready_time = getattr(worker_sim, "worker_busy_until", {}).get(wid)
            if ready_time is None:
                continue
            ready_time = float(ready_time)
            if ready_time <= float(slot_start_seconds) + 1e-6 or ready_time > horizon + 1e-6:
                continue
            donor = getattr(worker_sim, "worker_center_map", {}).get(wid)
            if donor not in centers:
                continue
            post_service_surplus[int(donor)].append((ready_time, wid))
        for workers in post_service_surplus.values():
            workers.sort(key=lambda item: (item[0], item[1]))

    immediate_share = max(0.0, float(getattr(config, "EVENT_PLATFORM_LOAN_MAX_SHARE", 0.45)))
    post_service_share = max(0.0, float(getattr(config, "EVENT_PLATFORM_POST_SERVICE_LOAN_MAX_SHARE", 0.20)))
    immediate_move_cap = sum(len(items) for items in surplus.values()) * immediate_share
    post_service_move_cap = sum(len(items) for items in post_service_surplus.values()) * post_service_share
    max_moves = int(math.ceil(
        (immediate_move_cap + post_service_move_cap) * max(0.25, move_scale)
    ))
    moves = []
    distance_cache: Dict[Tuple[int, int], float] = {}
    while len(moves) < max_moves:
        best = None
        best_score = 0.0
        for donor, worker_ids in surplus.items():
            if not worker_ids:
                continue
            for receiver, need in receiver_need.items():
                if donor == receiver or need <= 0:
                    continue
                key = (donor, receiver)
                if key not in distance_cache:
                    try:
                        distance_cache[key] = float(nx.shortest_path_length(G, centers[donor], centers[receiver], weight="length"))
                    except nx.NetworkXNoPath:
                        distance_cache[key] = float("inf")
                distance = distance_cache[key]
                if not np.isfinite(distance):
                    continue
                score = task_weight * uncovered_tasks[receiver] + gap_weight * need - distance_weight * distance / 1000.0
                if score > best_score:
                    best_score = score
                    best = (donor, receiver, distance)
        if best is None:
            break
        donor, receiver, distance = best
        wid = surplus[donor].pop(0)
        worker_sim.worker_center_map[wid] = receiver
        receiver_need[receiver] = max(0, receiver_need[receiver] - 1)
        moves.append({"wid": wid, "from_region": donor, "to_region": receiver, "distance_m": distance})

    min_gap = int(getattr(config, "EVENT_PLATFORM_POST_SERVICE_LOAN_MIN_UNCOVERED_GAP", 12))
    delay_penalty = float(getattr(config, "EVENT_PLATFORM_POST_SERVICE_LOAN_DELAY_PENALTY", 0.05))
    while len(moves) < max_moves and any(post_service_surplus.values()):
        best = None
        best_score = 0.0
        for donor, worker_items in post_service_surplus.items():
            if not worker_items:
                continue
            ready_time, wid = worker_items[0]
            donor_uncovered = int(uncovered_tasks.get(donor, 0))
            for receiver, need in receiver_need.items():
                if donor == receiver or need <= 0:
                    continue
                receiver_uncovered = int(uncovered_tasks.get(receiver, 0))
                uncovered_gap = receiver_uncovered - donor_uncovered
                if uncovered_gap < min_gap:
                    continue
                key = (donor, receiver)
                if key not in distance_cache:
                    try:
                        distance_cache[key] = float(nx.shortest_path_length(G, centers[donor], centers[receiver], weight="length"))
                    except nx.NetworkXNoPath:
                        distance_cache[key] = float("inf")
                distance = distance_cache[key]
                if not np.isfinite(distance):
                    continue
                delay_minutes = max(0.0, (ready_time - float(slot_start_seconds or ready_time)) / 60.0)
                score = (
                    task_weight * uncovered_gap
                    + gap_weight * need
                    - distance_weight * distance / 1000.0
                    - delay_penalty * delay_minutes
                )
                if score > best_score:
                    best_score = score
                    best = (donor, receiver, distance, ready_time, wid)
        if best is None:
            break
        donor, receiver, distance, ready_time, wid = best
        post_service_surplus[donor] = [
            item for item in post_service_surplus[donor] if item[1] != wid
        ]
        worker_sim.worker_center_map[wid] = receiver
        receiver_need[receiver] = max(0, receiver_need[receiver] - 1)
        moves.append(
            {
                "wid": wid,
                "from_region": donor,
                "to_region": receiver,
                "distance_m": distance,
                "post_service": True,
                "available_at": ready_time,
            }
        )
    return moves


def event_task_rl_game_predispatch_workers(
    G,
    worker_sim,
    centers: Dict[int, Any],
    predicted_tasks: Dict[int, list],
    backlog_tasks: Dict[int, list],
    predicted_distribution: Dict[int, Dict[str, float]],
    state: RLRetentionBilateralState,
    platform_state: Optional[PlatformTaskFirstRLState],
    center_task_state: CenterTaskAllocationRLState,
    slot_idx: int,
    slot_start_seconds: float,
    slot_end_seconds: float,
) -> Dict[str, Any]:
    """Center RL plans local tasks; platform RL loans only unused workers."""
    region_ids = sorted(centers)
    predicted_demand = {rid: len(predicted_tasks.get(rid, [])) for rid in region_ids}
    backlog_counts = {rid: len(backlog_tasks.get(rid, [])) for rid in region_ids}
    future_worker_horizon = _event_precommit_worker_horizon(slot_start_seconds, slot_end_seconds)
    immediate_workers_per_center = _collect_workers(worker_sim, centers, slot_start_seconds)
    workers_per_center = _collect_workers(
        worker_sim,
        centers,
        slot_start_seconds,
        include_future=True,
        future_until=future_worker_horizon,
    )
    available_workers = {rid: len(workers_per_center[rid]) for rid in region_ids}
    center_transitions = {}
    center_actions = {}
    for rid in region_ids:
        region_distribution = dict(predicted_distribution.get(rid, {}) or {})
        region_distribution["trust"] = center_task_state.prediction_trust.get(rid, 1.0)
        features = center_task_state.build_features(
            predicted_count=predicted_demand[rid],
            backlog_count=backlog_counts[rid],
            worker_count=available_workers[rid],
            predicted_distribution=region_distribution,
        )
        action_vector, action, normalized_action = center_task_state.sample_action(rid, features)
        center_actions[rid] = action
        center_transitions[rid] = {
            "features": features,
            "action": action_vector,
            "normalized_action": normalized_action,
            "predicted_count": predicted_demand[rid],
            "backlog_count": backlog_counts[rid],
        }

    planning_tasks = _build_center_task_plan(
        centers,
        backlog_tasks,
        predicted_tasks,
        workers_per_center,
        center_actions,
        center_task_state.prediction_trust,
        slot_start_seconds,
    )
    _, _, preliminary_details = _run_virtual_center_assignment(
        G, centers, workers_per_center, planning_tasks, slot_start_seconds, slot_end_seconds
    )
    preliminary_by_region = {rid: 0 for rid in region_ids}
    for detail in preliminary_details:
        preliminary_by_region[int(detail["region_id"])] += 1
    requested_task_count = {
        rid: len(backlog_tasks.get(rid, [])) + len(predicted_tasks.get(rid, []))
        for rid in region_ids
    }
    uncovered_preliminary = {
        rid: max(0, requested_task_count[rid] - preliminary_by_region[rid]) for rid in region_ids
    }

    platform_transition = None
    if platform_state is not None:
        platform_transition = sample_platform_task_first_control(
            region_ids=region_ids,
            predicted_demand={rid: len(planning_tasks[rid]) for rid in region_ids},
            backlog_counts=backlog_counts,
            available_workers=available_workers,
            max_tasks_per_worker=int(getattr(config, "MAX_TASKS_PER_WORKER", 4)),
            retention_state=state,
            platform_state=platform_state,
            predicted_distribution=predicted_distribution,
            backlog_weight=getattr(config, "UABG_BACKLOG_WEIGHT", 1.0),
            uncertainty_weight=getattr(config, "UABG_UNCERTAINTY_WEIGHT", 0.45),
            quantile_weight=getattr(config, "UABG_QUANTILE_WEIGHT", 0.55),
            burst_weight=getattr(config, "UABG_BURST_WEIGHT", 1.2),
            calibration_bias_weight=getattr(config, "RBG_PREDICTION_BIAS_WEIGHT", 0.60),
            calibration_shrink_weight=getattr(config, "RBG_PREDICTION_SHRINK_WEIGHT", 0.55),
            calibration_sigma_boost=getattr(config, "RBG_PREDICTION_SIGMA_BOOST", 0.75),
            calibration_min_scale=getattr(config, "RBG_PREDICTION_MIN_SCALE", 0.55),
            base_platform_task_weight=getattr(config, "RBG_PLATFORM_TASK_WEIGHT", 0.55),
            base_platform_gap_weight=getattr(config, "RBG_PLATFORM_GAP_WEIGHT", 0.85),
            base_platform_release_credit_weight=getattr(config, "RBG_PLATFORM_RELEASE_CREDIT_WEIGHT", 0.0),
        )
    moves = _platform_loan_surplus_workers(
        G, worker_sim, centers, immediate_workers_per_center, preliminary_details, uncovered_preliminary,
        platform_transition, int(getattr(config, "MAX_TASKS_PER_WORKER", 4)),
        slot_start_seconds=slot_start_seconds,
        slot_end_seconds=slot_end_seconds,
    )
    if platform_transition is not None:
        platform_transition["loan_count"] = len(moves)
        platform_transition["post_service_loan_count"] = sum(1 for move in moves if move.get("post_service"))
        platform_transition["loan_distance_km"] = sum(move["distance_m"] for move in moves) / 1000.0
    workers_after_loan = _collect_workers(
        worker_sim,
        centers,
        slot_start_seconds,
        include_future=True,
        future_until=future_worker_horizon,
    )
    planning_tasks = _build_center_task_plan(
        centers,
        backlog_tasks,
        predicted_tasks,
        workers_after_loan,
        center_actions,
        center_task_state.prediction_trust,
        slot_start_seconds,
    )
    planned_assignments, _, planned_details = _run_virtual_center_assignment(
        G, centers, workers_after_loan, planning_tasks, slot_start_seconds, slot_end_seconds
    )
    planned_by_worker: Dict[str, list] = {}
    planned_records_by_worker: Dict[str, list] = {}
    planned_by_region = {rid: 0 for rid in region_ids}
    planning_task_by_id = {
        str(task[1]): task
        for tasks in planning_tasks.values()
        for task in tasks
    }
    for detail in planned_details:
        rid = int(detail["region_id"])
        wid = str(detail["wid"])
        task_id = str(detail["task_id"])
        task = planning_task_by_id.get(task_id)
        planned_by_worker.setdefault(wid, []).append(task_id)
        if task is not None:
            planned_records_by_worker.setdefault(wid, []).append(
                {
                    "region_id": rid,
                    "task_id": task_id,
                    "node": task[0],
                    "reward": float(task[2]),
                    "expire_time": float(task[3]),
                    "release_time": float(task[4]) if len(task) > 4 else float(slot_start_seconds),
                }
            )
        planned_by_region[rid] += 1
    worker_task_priority_map = _build_precommit_worker_task_priority_map(
        planned_details=planned_details,
        action_profiles=center_actions,
        predicted_tasks=predicted_tasks,
        prediction_trust=center_task_state.prediction_trust,
    )
    precommit_bundles_by_worker, task_bundle_map = _build_precommit_bundles_by_worker(
        planned_details=planned_details,
        planning_task_by_id=planning_task_by_id,
        predicted_tasks=predicted_tasks,
        worker_task_priority_map=worker_task_priority_map,
        slot_start_seconds=slot_start_seconds,
    )
    uncovered_predicted = {
        rid: max(0, predicted_demand[rid] - planned_by_region[rid]) for rid in region_ids
    }
    region_priority = {
        rid: 1.0 + float(getattr(config, "EVENT_PLAN_SHORTAGE_PRIORITY_WEIGHT", 0.40))
        * uncovered_predicted[rid] / max(1.0, float(predicted_demand[rid]))
        for rid in region_ids
    }
    return {
        "moves": moves,
        "available_workers": available_workers,
        "desired_workers": {
            rid: int(math.ceil(requested_task_count[rid] / max(1, int(getattr(config, "MAX_TASKS_PER_WORKER", 4)))))
            for rid in region_ids
        },
        "retain_count": {
            rid: len(workers_after_loan[rid])
            for rid in region_ids
        },
        "hoard_penalty": {rid: 0.0 for rid in region_ids},
        "move_cost_by_region": {
            rid: sum(move["distance_m"] / 1000.0 for move in moves if move["from_region"] == rid)
            for rid in region_ids
        },
        "transitions": {},
        "center_transitions": center_transitions,
        "center_action_profile": center_actions,
        "demand_profile": predicted_distribution,
        "effective_demand": {rid: len(planning_tasks[rid]) for rid in region_ids},
        "center_prediction_trust": {rid: center_task_state.prediction_trust.get(rid, 1.0) for rid in region_ids},
        "stackelberg_control": {
            "region_priority_weight": region_priority,
            "event_center_task_action_profile": center_actions,
            "worker_task_priority_map": worker_task_priority_map,
            "precommit_task_records_by_worker": planned_records_by_worker,
            "precommit_bundles_by_worker": precommit_bundles_by_worker,
            "task_bundle_map": task_bundle_map,
            "worker_completion_bonus": getattr(config, "RBG_WORKER_COMPLETION_BONUS", 0.20),
            "worker_distance_penalty": getattr(config, "RBG_WORKER_DISTANCE_PENALTY", 0.0),
            "same_worker_chain_bonus": getattr(config, "RBG_WORKER_CHAIN_BONUS", 0.08),
            "same_bundle_bonus": getattr(config, "EVENT_PRECOMMIT_BUNDLE_SAME_WORKER_BONUS", 0.0),
            "locality_detour_penalty": getattr(config, "ROUTE_LOCALITY_DETOUR_PENALTY", 0.0),
        },
        "predicted_demand": predicted_demand,
        "predicted_tasks": predicted_tasks,
        "precommit_assignments": planned_assignments,
        "precommit_plan_by_worker": planned_by_worker,
        "precommit_task_records_by_worker": planned_records_by_worker,
        "precommit_bundles_by_worker": precommit_bundles_by_worker,
        "precommit_task_bundle_map": task_bundle_map,
        "precommit_worker_task_priority_map": worker_task_priority_map,
        "precommit_planned_by_region": planned_by_region,
        "precommit_uncovered_predicted": uncovered_predicted,
        "platform_transition": platform_transition,
    }


def update_center_task_allocation_state(
    state: CenterTaskAllocationRLState,
    transitions: Dict[int, Dict[str, Any]],
    assigned_tasks_by_region: Dict[int, int],
    total_tasks_by_region: Dict[int, int],
    remaining_tasks_by_region: Dict[int, int],
    actual_arrivals_by_region: Optional[Dict[int, int]] = None,
    predicted_demand_by_region: Optional[Dict[int, int]] = None,
) -> Dict[int, float]:
    if predicted_demand_by_region is not None and actual_arrivals_by_region is not None:
        state.record_prediction_feedback(
            predicted_region_demand=predicted_demand_by_region,
            actual_region_demand=actual_arrivals_by_region,
        )
    return state.update_policy(
        transitions=transitions,
        assigned_tasks_by_region=assigned_tasks_by_region,
        total_tasks_by_region=total_tasks_by_region,
        remaining_tasks_by_region=remaining_tasks_by_region,
        actual_arrivals_by_region=actual_arrivals_by_region,
    )
