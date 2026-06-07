import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np

import config
from algorithm.CenterPrepackedAssignment import center_prepacked_assignment_with_center_pickup
from algorithm.RLRetentionGameDispatch import rl_retention_bilateral_predispatch_workers


@dataclass
class LLMDispatchMemory:
    """Small online memory used by the deterministic trade executor."""

    region_ids: List[int]
    action_ratios: Tuple[float, ...] = (-0.30, -0.15, 0.0, 0.15, 0.30)
    prediction_error_decay: float = 0.80
    prediction_bias_clip_ratio: float = 0.75
    max_service_debt: float = 4.0
    move_history_size: int = 8
    prediction_bias_ema: Dict[int, float] = field(default_factory=dict)
    prediction_abs_error_ema: Dict[int, float] = field(default_factory=dict)
    service_debt: Dict[int, float] = field(default_factory=dict)
    receiver_affinity: Dict[int, Dict[int, float]] = field(default_factory=dict)
    worker_move_slots: Dict[str, deque] = field(default_factory=dict)
    _selected_ratios: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for rid in self.region_ids:
            self.prediction_bias_ema.setdefault(rid, 0.0)
            self.prediction_abs_error_ema.setdefault(rid, 0.0)
            self.service_debt.setdefault(rid, 0.0)
            self.receiver_affinity.setdefault(rid, {})

    def set_llm_ratios(self, ratios: Dict[int, float]) -> None:
        self._selected_ratios = {
            rid: min(self.action_ratios, key=lambda value: abs(value - float(ratios.get(rid, 0.0))))
            for rid in self.region_ids
        }

    def sample_action(self, region_id: int, features: np.ndarray):
        ratio = float(self._selected_ratios.get(region_id, 0.0))
        index = min(range(len(self.action_ratios)), key=lambda i: abs(self.action_ratios[i] - ratio))
        probs = np.zeros(len(self.action_ratios), dtype=np.float32)
        probs[index] = 1.0
        return index, float(self.action_ratios[index]), probs

    def record_moves(self, slot_idx: int, moved_workers: List[str]) -> None:
        for wid in moved_workers:
            history = self.worker_move_slots.setdefault(wid, deque(maxlen=self.move_history_size))
            history.append(int(slot_idx))

    def get_receiver_affinity(self, donor_region: int, receiver_region: int) -> float:
        return float(self.receiver_affinity.get(donor_region, {}).get(receiver_region, 0.0))

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
            scale = max(1.0, pred, actual)
            error = float(np.clip(pred - actual, -clip_ratio * scale, clip_ratio * scale))
            self.prediction_bias_ema[rid] = decay * self.prediction_bias_ema[rid] + (1.0 - decay) * error
            self.prediction_abs_error_ema[rid] = (
                decay * self.prediction_abs_error_ema[rid] + (1.0 - decay) * abs(error)
            )

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
        )
        return np.asarray(
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
                float(profile.get("bias_ema", 0.0)) / scale,
                float(profile.get("abs_err_ema", 0.0)) / scale,
                float(self.service_debt.get(region_id, 0.0)) / max(1.0, self.max_service_debt),
            ],
            dtype=np.float32,
        )


class LLMDispatchPolicyClient:
    """Call an OpenAI-compatible chat completions API for compact policy decisions."""

    def __init__(self) -> None:
        self.api_url = str(
            os.environ.get("LLM_DISPATCH_API_URL", getattr(config, "LLM_DISPATCH_API_URL", ""))
        ).strip()
        self.api_key = str(
            os.environ.get("LLM_DISPATCH_API_KEY", getattr(config, "LLM_DISPATCH_API_KEY", ""))
        ).strip()
        self.model = str(
            os.environ.get("LLM_DISPATCH_MODEL", getattr(config, "LLM_DISPATCH_MODEL", ""))
        ).strip()
        self.timeout = float(getattr(config, "LLM_DISPATCH_TIMEOUT_SECONDS", 45.0))

    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(content)

    def decide(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        mock_response = os.environ.get("LLM_DISPATCH_MOCK_RESPONSE_JSON", "").strip()
        if mock_response:
            return self._extract_json(mock_response)
        if not self.api_url or not self.api_key or not self.model:
            raise RuntimeError(
                "Event-LLM-Game requires LLM_DISPATCH_API_URL, LLM_DISPATCH_API_KEY, "
                "and LLM_DISPATCH_MODEL environment variables."
            )
        system_prompt = (
            "You control proactive courier relocation among fixed logistics centers. "
            "Maximize completed tasks while avoiding unnecessary transfers. "
            "Return only strict JSON with keys center_action_ratios, platform, reason. "
            "Each center ratio must be one of -0.30, -0.15, 0.0, 0.15, 0.30; "
            "negative releases more workers, positive retains more. "
            "platform must contain task_scale, gap_scale, keep_scale, need_scale, "
            "move_share_scale, slot_start_blend_scale, each between 0.70 and 1.30."
        )
        payload = {
            "model": self.model,
            "temperature": float(getattr(config, "LLM_DISPATCH_TEMPERATURE", 0.0)),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(situation, ensure_ascii=True)},
            ],
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        req = urlrequest.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM dispatch API call failed: {exc}") from exc
        try:
            return self._extract_json(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("LLM dispatch API response does not contain valid JSON policy output.") from exc


def _validate_decision(raw: Dict[str, Any], region_ids: List[int]) -> Dict[str, Any]:
    allowed_ratios = (-0.30, -0.15, 0.0, 0.15, 0.30)
    raw_ratios = raw.get("center_action_ratios", {})
    ratios = {}
    for rid in region_ids:
        value = raw_ratios.get(str(rid), raw_ratios.get(rid, 0.0))
        value = float(value)
        ratios[rid] = min(allowed_ratios, key=lambda candidate: abs(candidate - value))
    raw_platform = raw.get("platform", {})
    platform = {}
    for key in (
        "task_scale",
        "gap_scale",
        "keep_scale",
        "need_scale",
        "move_share_scale",
        "slot_start_blend_scale",
    ):
        platform[key] = float(np.clip(float(raw_platform.get(key, 1.0)), 0.70, 1.30))
    return {
        "center_action_ratios": ratios,
        "platform": platform,
        "reason": str(raw.get("reason", ""))[:240],
    }


def _build_situation(
    predicted_tasks: Dict[int, list],
    backlog_tasks: Dict[int, list],
    predicted_distribution: Dict[int, Dict[str, float]],
    available_workers: Dict[int, int],
    slot_start_seconds: float,
    slot_end_seconds: float,
) -> Dict[str, Any]:
    regions = []
    for rid in sorted(predicted_tasks.keys()):
        tasks = predicted_tasks.get(rid, [])
        profile = predicted_distribution.get(rid, {})
        regions.append(
            {
                "region_id": int(rid),
                "predicted_tasks": int(len(tasks)),
                "backlog_tasks": int(len(backlog_tasks.get(rid, []))),
                "available_workers": int(available_workers.get(rid, 0)),
                "sigma": round(float(profile.get("sigma", 0.0)), 3),
                "q90": round(float(profile.get("q90", len(tasks))), 3),
                "sample_release_seconds": [int(task[4]) for task in tasks[:3]],
            }
        )
    return {
        "slot_start_seconds": int(slot_start_seconds),
        "slot_end_seconds": int(slot_end_seconds),
        "max_tasks_per_worker": int(getattr(config, "MAX_TASKS_PER_WORKER", 4)),
        "regions": regions,
    }


def _merge_planning_tasks(
    centers: Dict[int, Any],
    backlog_tasks: Dict[int, list],
    predicted_tasks: Dict[int, list],
    workers_per_center: Dict[int, list],
) -> Dict[int, list]:
    factor = max(1, int(getattr(config, "EVENT_PLAN_TASK_CANDIDATE_FACTOR", 3)))
    capacity = max(1, int(getattr(config, "MAX_TASKS_PER_WORKER", 4)))
    planning_tasks = {}
    for rid in centers.keys():
        backlog = list(backlog_tasks.get(rid, []))
        future = sorted(predicted_tasks.get(rid, []), key=lambda task: (task[4], -task[2]))
        limit = max(len(backlog), len(workers_per_center.get(rid, [])) * capacity * factor)
        planning_tasks[rid] = backlog + future[: max(0, limit - len(backlog))]
    return planning_tasks


def event_task_llm_game_predispatch_workers(
    G,
    worker_sim,
    centers: Dict[int, Any],
    predicted_tasks: Dict[int, list],
    backlog_tasks: Dict[int, list],
    predicted_distribution: Dict[int, Dict[str, float]],
    memory: LLMDispatchMemory,
    llm_client: LLMDispatchPolicyClient,
    slot_idx: int,
    slot_start_seconds: float,
    slot_end_seconds: float,
) -> Dict[str, Any]:
    region_ids = sorted(centers.keys())
    predicted_demand = {rid: len(predicted_tasks.get(rid, [])) for rid in region_ids}
    available_workers = {
        rid: len(worker_sim.get_available_workers_with_center_info(rid, current_time=slot_start_seconds))
        for rid in region_ids
    }
    decision = _validate_decision(
        llm_client.decide(
            _build_situation(
                predicted_tasks=predicted_tasks,
                backlog_tasks=backlog_tasks,
                predicted_distribution=predicted_distribution,
                available_workers=available_workers,
                slot_start_seconds=slot_start_seconds,
                slot_end_seconds=slot_end_seconds,
            )
        ),
        region_ids,
    )
    memory.set_llm_ratios(decision["center_action_ratios"])
    platform = decision["platform"]
    result = rl_retention_bilateral_predispatch_workers(
        G=G,
        worker_sim=worker_sim,
        centers=centers,
        predicted_demand=predicted_demand,
        state=memory,
        slot_idx=slot_idx,
        next_slot_start_seconds=slot_start_seconds,
        predicted_distribution=predicted_distribution,
        max_tasks_per_worker=getattr(config, "MAX_TASKS_PER_WORKER", 4),
        backlog_counts={rid: len(backlog_tasks.get(rid, [])) for rid in region_ids},
        backlog_weight=getattr(config, "UABG_BACKLOG_WEIGHT", 1.0),
        uncertainty_weight=getattr(config, "UABG_UNCERTAINTY_WEIGHT", 0.45),
        quantile_weight=getattr(config, "UABG_QUANTILE_WEIGHT", 0.55),
        burst_weight=getattr(config, "UABG_BURST_WEIGHT", 1.2),
        calibration_bias_weight=getattr(config, "RBG_PREDICTION_BIAS_WEIGHT", 0.60),
        calibration_shrink_weight=getattr(config, "RBG_PREDICTION_SHRINK_WEIGHT", 0.55),
        calibration_sigma_boost=getattr(config, "RBG_PREDICTION_SIGMA_BOOST", 0.75),
        calibration_min_scale=getattr(config, "RBG_PREDICTION_MIN_SCALE", 0.55),
        platform_task_weight=getattr(config, "RBG_PLATFORM_TASK_WEIGHT", 0.55) * platform["task_scale"],
        platform_gap_weight=getattr(config, "RBG_PLATFORM_GAP_WEIGHT", 0.85) * platform["gap_scale"],
        platform_release_credit_weight=0.0,
        platform_keep_scale=platform["keep_scale"],
        platform_need_scale=platform["need_scale"],
        platform_move_share_scale=platform["move_share_scale"],
        platform_slot_start_blend_scale=platform["slot_start_blend_scale"],
        center_local_task_weight=getattr(config, "RBG_CENTER_LOCAL_TASK_WEIGHT", 1.0),
        worker_completion_bonus=getattr(config, "RBG_WORKER_COMPLETION_BONUS", 0.20),
        worker_distance_penalty=getattr(config, "RBG_WORKER_DISTANCE_PENALTY", 0.0),
        same_worker_chain_bonus=getattr(config, "RBG_WORKER_CHAIN_BONUS", 0.08),
        min_buffer_workers=getattr(config, "UABG_MIN_BUFFER_WORKERS", 1),
        reserve_ratio=getattr(config, "UABG_RESERVE_RATIO", 0.1),
        bid_shortage_weight=getattr(config, "UABG_BID_SHORTAGE_WEIGHT", 0.9),
        bid_backlog_weight=getattr(config, "UABG_BID_BACKLOG_WEIGHT", 0.45),
        bid_debt_weight=getattr(config, "UABG_BID_DEBT_WEIGHT", 0.85),
        bid_burst_weight=getattr(config, "UABG_BID_BURST_WEIGHT", 0.6),
        ask_shortage_weight=getattr(config, "UABG_ASK_SHORTAGE_WEIGHT", 0.85),
        ask_uncertainty_weight=getattr(config, "UABG_ASK_UNCERTAINTY_WEIGHT", 0.65),
        dispatch_phase="slot_start",
        hoard_discount_weight=getattr(config, "RBG_HOARD_DISCOUNT_WEIGHT", 0.40),
        move_cost_weight=getattr(config, "RBG_MOVE_COST_WEIGHT", 0.02),
        distance_penalty=getattr(config, "UABG_DISTANCE_PENALTY", 0.004),
        candidate_k=getattr(config, "UABG_CANDIDATE_K", 16),
        edge_epsilon=getattr(config, "UABG_EDGE_EPSILON", 0.05),
        record_transition=False,
    )
    workers_per_center = {}
    for rid in region_ids:
        workers = worker_sim.get_available_workers_with_center_info(rid, current_time=slot_start_seconds)
        workers_per_center[rid] = [(w[0], w[1], w[2], w[3], centers[rid]) for w in workers]
    planning_tasks = _merge_planning_tasks(centers, backlog_tasks, predicted_tasks, workers_per_center)
    planned_assignments, _, planned_details = center_prepacked_assignment_with_center_pickup(
        G=G,
        config=config,
        centers_dict=centers,
        workers_per_center=workers_per_center,
        tasks_per_center=planning_tasks,
        slot_start_seconds=slot_start_seconds,
        slot_end_seconds=slot_end_seconds,
        stackelberg_control=result.get("stackelberg_control", {}),
        force_center_pickup_on_first_departure=True,
    )
    planned_by_worker: Dict[str, list] = {}
    planned_by_region = {rid: 0 for rid in region_ids}
    for detail in planned_details:
        planned_by_worker.setdefault(detail["wid"], []).append(detail["task_id"])
        planned_by_region[int(detail["region_id"])] += 1
    uncovered = {
        rid: max(0, predicted_demand[rid] - planned_by_region.get(rid, 0))
        for rid in region_ids
    }
    priority = dict(result.get("stackelberg_control", {}).get("region_priority_weight", {}))
    shortage_weight = max(0.0, float(getattr(config, "EVENT_PLAN_SHORTAGE_PRIORITY_WEIGHT", 0.40)))
    for rid in region_ids:
        ratio = uncovered[rid] / max(1.0, float(predicted_demand[rid]))
        priority[rid] = float(priority.get(rid, 1.0)) * (1.0 + shortage_weight * ratio)
    result["stackelberg_control"]["region_priority_weight"] = priority
    result["predicted_demand"] = predicted_demand
    result["precommit_assignments"] = planned_assignments
    result["precommit_plan_by_worker"] = planned_by_worker
    result["precommit_planned_by_region"] = planned_by_region
    result["precommit_uncovered_predicted"] = uncovered
    result["llm_decision"] = decision
    return result
