from __future__ import annotations

import json
import math
import random
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

from environment import Observation, PrenatalEnvironment
from tasks import TASKS
from xai_reward_model import (
    SAFE_CONDITIONS,
    URGENCY_ORDER,
    choose_urgency,
    featurize,
    infer_reference_condition,
    latent_risk_scores,
)

CONDITION_TO_IDX = {name: idx for idx, name in enumerate(SAFE_CONDITIONS)}
IDX_TO_CONDITION = {idx: name for name, idx in CONDITION_TO_IDX.items()}
URGENCY_TO_IDX = {name: idx for idx, name in enumerate(URGENCY_ORDER)}
IDX_TO_URGENCY = {idx: name for name, idx in URGENCY_TO_IDX.items()}

RISK_FLAG_NAMES = [
    "DANGER_BP_CRITICAL",
    "HIGH_BP",
    "BP_RISING_TREND",
    "DANGER_LOW_KICKS",
    "HIGH_PREECLAMPSIA_SIGNAL",
    "DANGER_VISION_HEADACHE",
    "DANGER_BLEEDING",
    "ABDOMINAL_PAIN_SIGNAL",
    "LOW_NUTRITION",
    "LOW_KICK_AVG",
    "DIZZINESS_SIGNAL",
]

HISTORY_FLAG_NAMES = [
    "family_diabetes",
    "family_hypertension",
    "prev_preeclampsia",
    "prev_complication",
]

LATENT_RISK_NAMES = [
    "postpartum_hemorrhage",
    "maternal_infection",
    "dehydration",
    "intrahepatic_cholestasis",
    "placental_abruption",
    "maternal_exhaustion",
    "nutrition_deficit",
]

HANDCRAFTED_FEATURE_NAMES = [
    "abdominal_pain",
    "bleeding",
    "bp_rising",
    "breathlessness",
    "danger_bp",
    "danger_low_kicks",
    "dizziness",
    "good_kicks",
    "good_meals",
    "good_sleep",
    "headache_swelling",
    "high_bp",
    "high_meals",
    "history_complication",
    "history_diabetes",
    "history_htn",
    "history_preeclampsia",
    "late_pregnancy",
    "low_energy",
    "low_kick_avg",
    "low_nutrition",
    "maternal_strain",
    "no_flags",
    "no_history",
    "normal_bp",
    "poor_sleep",
    "trimester2_or_3",
    "trimester3",
    "very_low_meals",
    "vision_headache",
    "weeks_early_third",
    "weight_signal",
]


@dataclass
class DailySnapshot:
    bp_systolic: int
    bp_diastolic: int
    kick_count: int
    meals_count: int
    water_litres: float
    sleep_hours: float
    symptom_headache: bool
    symptom_blurred_vision: bool
    symptom_swelling: bool
    symptom_abdominal_pain: bool
    symptom_bleeding: bool
    symptom_dizziness: bool


@dataclass
class SeedCase:
    source_id: str
    source_type: str
    age: int
    region: str
    observation: Observation
    daily_sequence: List[DailySnapshot]


def trimester_for_weeks(weeks: int) -> int:
    if weeks <= 12:
        return 1
    if weeks <= 26:
        return 2
    return 3


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _randn(rng: random.Random, mean: float, std: float) -> float:
    return rng.gauss(mean, std)


def _bool(probability: float, rng: random.Random) -> bool:
    return rng.random() < probability


def _dense_latent_vector(obs: Observation) -> List[float]:
    latent = latent_risk_scores(featurize(obs))
    return [float(latent.get(name, 0.0)) for name in LATENT_RISK_NAMES]


def _risk_vector(names: Sequence[str], active: Sequence[str]) -> List[float]:
    active_set = set(active)
    return [1.0 if name in active_set else 0.0 for name in names]


def _continuous_vector(obs: Observation, age: int) -> List[float]:
    return [
        age / 45.0,
        obs.weeks_pregnant / 40.0,
        obs.trimester / 3.0,
        (obs.latest_weight_kg or 0.0) / 110.0,
        (obs.latest_energy or 0.0) / 10.0,
        (obs.latest_breathlessness or 0.0) / 10.0,
        obs.avg_meals / 5.0,
        obs.avg_sleep / 10.0,
        (obs.avg_kick_count or 0.0) / 12.0,
        obs.days_of_data / 3.0,
    ]


def _handcrafted_vector(obs: Observation) -> List[float]:
    features = featurize(obs)
    return [float(features[name]) for name in HANDCRAFTED_FEATURE_NAMES]


def _normalize_day(day: DailySnapshot) -> List[float]:
    return [
        day.bp_systolic / 180.0,
        day.bp_diastolic / 120.0,
        day.kick_count / 12.0,
        day.meals_count / 5.0,
        day.water_litres / 5.0,
        day.sleep_hours / 10.0,
        float(day.symptom_headache),
        float(day.symptom_blurred_vision),
        float(day.symptom_swelling),
        float(day.symptom_abdominal_pain),
        float(day.symptom_bleeding),
        float(day.symptom_dizziness),
    ]


def _observation_from_sequence(
    *,
    age: int,
    region: str,
    weeks_pregnant: int,
    history_flags: Sequence[str],
    daily_sequence: Sequence[DailySnapshot],
    latest_weight_kg: float,
    latest_energy: int,
    latest_breathlessness: int,
) -> Observation:
    latest = daily_sequence[-1]
    risk_flags: list[str] = []

    if latest.bp_systolic >= 160 or latest.bp_diastolic >= 110:
        risk_flags.append("DANGER_BP_CRITICAL")
    elif latest.bp_systolic >= 140 or latest.bp_diastolic >= 90:
        risk_flags.append("HIGH_BP")

    if latest.symptom_bleeding:
        risk_flags.append("DANGER_BLEEDING")
    if latest.symptom_blurred_vision and latest.symptom_headache:
        risk_flags.append("DANGER_VISION_HEADACHE")
    if latest.kick_count < 3:
        risk_flags.append("DANGER_LOW_KICKS")
    if latest.symptom_swelling and latest.symptom_headache:
        risk_flags.append("HIGH_PREECLAMPSIA_SIGNAL")
    if latest.symptom_dizziness:
        risk_flags.append("DIZZINESS_SIGNAL")
    if latest.symptom_abdominal_pain:
        risk_flags.append("ABDOMINAL_PAIN_SIGNAL")

    systolics = [item.bp_systolic for item in daily_sequence]
    bp_trend = "stable"
    if systolics[-1] > systolics[0] + 10:
        bp_trend = "rising"
        risk_flags.append("BP_RISING_TREND")
    elif systolics[-1] < systolics[0] - 10:
        bp_trend = "falling"

    avg_meals = sum(item.meals_count for item in daily_sequence) / len(daily_sequence)
    avg_sleep = sum(item.sleep_hours for item in daily_sequence) / len(daily_sequence)
    avg_kicks = sum(item.kick_count for item in daily_sequence) / len(daily_sequence)

    if avg_meals < 2:
        risk_flags.append("LOW_NUTRITION")
    if avg_kicks < 6:
        risk_flags.append("LOW_KICK_AVG")

    return Observation(
        user_id=0,
        weeks_pregnant=weeks_pregnant,
        trimester=trimester_for_weeks(weeks_pregnant),
        region=region,
        risk_flags=sorted(set(risk_flags)),
        bp_trend=bp_trend,
        avg_kick_count=avg_kicks,
        avg_meals=avg_meals,
        avg_sleep=avg_sleep,
        latest_weight_kg=latest_weight_kg,
        latest_energy=latest_energy,
        latest_breathlessness=latest_breathlessness,
        history_flags=list(history_flags),
        days_of_data=len(daily_sequence),
    )


def _task_daily_sequence(obs: Observation, rng: random.Random) -> List[DailySnapshot]:
    avg_kicks = obs.avg_kick_count or 8.0
    base_latest_bp_s = 118
    base_latest_bp_d = 76
    if "DANGER_BP_CRITICAL" in obs.risk_flags:
        base_latest_bp_s, base_latest_bp_d = 166, 112
    elif "HIGH_BP" in obs.risk_flags:
        base_latest_bp_s, base_latest_bp_d = 146, 94

    if obs.bp_trend == "rising":
        systolics = [base_latest_bp_s - 14, base_latest_bp_s - 6, base_latest_bp_s]
        diastolics = [base_latest_bp_d - 8, base_latest_bp_d - 4, base_latest_bp_d]
    elif obs.bp_trend == "falling":
        systolics = [base_latest_bp_s + 12, base_latest_bp_s + 5, base_latest_bp_s]
        diastolics = [base_latest_bp_d + 8, base_latest_bp_d + 3, base_latest_bp_d]
    else:
        systolics = [base_latest_bp_s - 2, base_latest_bp_s + 1, base_latest_bp_s]
        diastolics = [base_latest_bp_d - 1, base_latest_bp_d + 1, base_latest_bp_d]

    latest_kick = 2 if "DANGER_LOW_KICKS" in obs.risk_flags else max(1, int(round(avg_kicks)))
    if "LOW_KICK_AVG" in obs.risk_flags and "DANGER_LOW_KICKS" not in obs.risk_flags:
        kick_counts = [max(1, latest_kick + 2), max(1, latest_kick + 1), max(1, latest_kick)]
    else:
        kick_counts = [
            max(1, int(round(avg_kicks + rng.uniform(-1, 1)))),
            max(1, int(round(avg_kicks + rng.uniform(-1, 1)))),
            max(1, int(round(latest_kick))),
        ]

    symptoms = {
        "headache": "HIGH_PREECLAMPSIA_SIGNAL" in obs.risk_flags or "DANGER_VISION_HEADACHE" in obs.risk_flags,
        "blurred_vision": "DANGER_VISION_HEADACHE" in obs.risk_flags,
        "swelling": "HIGH_PREECLAMPSIA_SIGNAL" in obs.risk_flags,
        "abdominal_pain": "ABDOMINAL_PAIN_SIGNAL" in obs.risk_flags,
        "bleeding": "DANGER_BLEEDING" in obs.risk_flags,
        "dizziness": "DIZZINESS_SIGNAL" in obs.risk_flags,
    }

    meals = max(1, int(round(obs.avg_meals)))
    sleep = float(obs.avg_sleep)
    sequence: list[DailySnapshot] = []
    for idx in range(3):
        sequence.append(
            DailySnapshot(
                bp_systolic=int(_clamp(systolics[idx] + _randn(rng, 0, 2), 90, 180)),
                bp_diastolic=int(_clamp(diastolics[idx] + _randn(rng, 0, 2), 55, 120)),
                kick_count=int(_clamp(kick_counts[idx], 0, 12)),
                meals_count=int(_clamp(meals + round(_randn(rng, 0, 1)), 1, 5)),
                water_litres=float(_clamp(2.0 + _randn(rng, 0, 0.4), 0.8, 4.5)),
                sleep_hours=float(_clamp(sleep + _randn(rng, 0, 0.6), 2.5, 9.5)),
                symptom_headache=symptoms["headache"] and idx >= 1,
                symptom_blurred_vision=symptoms["blurred_vision"] and idx == 2,
                symptom_swelling=symptoms["swelling"] and idx >= 1,
                symptom_abdominal_pain=symptoms["abdominal_pain"] and idx == 2,
                symptom_bleeding=symptoms["bleeding"] and idx == 2,
                symptom_dizziness=symptoms["dizziness"] and idx >= 1,
            )
        )
    return sequence


def load_seed_cases(db_path: str | Path | None = None) -> list[SeedCase]:
    db_path = Path(db_path or "prenatal.db")
    seed_cases: list[SeedCase] = []

    if db_path.exists():
        env = PrenatalEnvironment()
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT id, age, region FROM user_profiles ORDER BY id"
        )
        users = cur.fetchall()
        for user in users:
            user_id = int(user["id"])
            prompt = env.reset(user_id)
            observation = prompt.observation
            cur.execute(
                """
                SELECT bp_systolic, bp_diastolic, kick_count, meals_count, water_litres, sleep_hours,
                       symptom_headache, symptom_blurred_vision, symptom_swelling,
                       symptom_abdominal_pain, symptom_bleeding, symptom_dizziness
                FROM daily_checkins
                WHERE user_id = ?
                ORDER BY created_at ASC
                LIMIT 3
                """,
                (user_id,),
            )
            rows = cur.fetchall()
            if len(rows) == 3:
                daily_sequence = [
                    DailySnapshot(
                        bp_systolic=int(row["bp_systolic"]),
                        bp_diastolic=int(row["bp_diastolic"]),
                        kick_count=int(row["kick_count"] if row["kick_count"] is not None else 0),
                        meals_count=int(row["meals_count"]),
                        water_litres=float(row["water_litres"]),
                        sleep_hours=float(row["sleep_hours"]),
                        symptom_headache=bool(row["symptom_headache"]),
                        symptom_blurred_vision=bool(row["symptom_blurred_vision"]),
                        symptom_swelling=bool(row["symptom_swelling"]),
                        symptom_abdominal_pain=bool(row["symptom_abdominal_pain"]),
                        symptom_bleeding=bool(row["symptom_bleeding"]),
                        symptom_dizziness=bool(row["symptom_dizziness"]),
                    )
                    for row in rows
                ]
                seed_cases.append(
                    SeedCase(
                        source_id=f"user_{user_id}",
                        source_type="database",
                        age=int(user["age"]),
                        region=str(user["region"]),
                        observation=observation,
                        daily_sequence=daily_sequence,
                    )
                )
        conn.close()

    rng = random.Random(19)
    for task in TASKS:
        obs = task["observation"]
        seed_cases.append(
            SeedCase(
                source_id=task["id"],
                source_type="task",
                age=24 + (len(seed_cases) % 8),
                region=obs.region,
                observation=obs,
                daily_sequence=_task_daily_sequence(obs, rng),
            )
        )

    if not seed_cases:
        raise RuntimeError("No MAAS seed cases available.")
    return seed_cases


def _condition_specific_adjustments(
    condition: str,
    *,
    weeks: int,
    history_flags: list[str],
    daily_sequence: list[DailySnapshot],
    weight_kg: float,
    energy: int,
    breathlessness: int,
    rng: random.Random,
) -> tuple[int, list[str], list[DailySnapshot], float, int, int]:
    seq = daily_sequence

    if condition == "preeclampsia":
        weeks = int(_clamp(weeks + rng.randint(0, 3), 24, 39))
        if "family_hypertension" not in history_flags and _bool(0.55, rng):
            history_flags.append("family_hypertension")
        base_s = int(_clamp(146 + rng.randint(-4, 16), 140, 175))
        base_d = int(_clamp(94 + rng.randint(-3, 16), 90, 118))
        seq[0].bp_systolic, seq[1].bp_systolic, seq[2].bp_systolic = base_s - 14, base_s - 7, base_s
        seq[0].bp_diastolic, seq[1].bp_diastolic, seq[2].bp_diastolic = base_d - 8, base_d - 4, base_d
        seq[1].symptom_headache = True
        seq[2].symptom_headache = True
        seq[1].symptom_swelling = True
        seq[2].symptom_swelling = True
        seq[2].symptom_blurred_vision = _bool(0.4, rng)
        energy = int(_clamp(energy + rng.randint(-3, 0), 2, 6))
        breathlessness = int(_clamp(breathlessness + rng.randint(0, 2), 3, 7))
    elif condition == "gestational_diabetes":
        weeks = int(_clamp(weeks + rng.randint(-1, 3), 22, 38))
        if "family_diabetes" not in history_flags:
            history_flags.append("family_diabetes")
        for day in seq:
            day.meals_count = int(_clamp(day.meals_count + rng.randint(1, 2), 3, 5))
            day.kick_count = int(_clamp(day.kick_count + rng.randint(-1, 1), 6, 12))
            day.bp_systolic = int(_clamp(day.bp_systolic + rng.randint(-2, 4), 108, 146))
            day.bp_diastolic = int(_clamp(day.bp_diastolic + rng.randint(-2, 3), 70, 94))
        weight_kg = float(_clamp(weight_kg + rng.uniform(5.0, 14.0), 64.0, 94.0))
        energy = int(_clamp(energy + rng.randint(-3, 0), 3, 6))
        breathlessness = int(_clamp(breathlessness + rng.randint(1, 3), 4, 8))
    elif condition == "anemia":
        weeks = int(_clamp(weeks + rng.randint(-2, 2), 16, 36))
        for day in seq:
            day.meals_count = int(_clamp(day.meals_count - rng.randint(1, 2), 1, 3))
            day.sleep_hours = float(_clamp(day.sleep_hours + rng.uniform(-1.0, 0.4), 3.0, 7.0))
            day.symptom_dizziness = True
            day.bp_systolic = int(_clamp(day.bp_systolic + rng.randint(-8, 3), 92, 132))
            day.bp_diastolic = int(_clamp(day.bp_diastolic + rng.randint(-6, 2), 58, 86))
        weight_kg = float(_clamp(weight_kg + rng.uniform(-6.0, 1.5), 41.0, 68.0))
        energy = int(_clamp(energy + rng.randint(-4, -1), 1, 4))
        breathlessness = int(_clamp(breathlessness + rng.randint(2, 4), 5, 9))
    elif condition == "preterm_risk":
        weeks = int(_clamp(weeks + rng.randint(0, 3), 27, 36))
        if "prev_complication" not in history_flags and _bool(0.6, rng):
            history_flags.append("prev_complication")
        seq[2].symptom_abdominal_pain = True
        seq[2].symptom_bleeding = True
        seq[1].symptom_abdominal_pain = _bool(0.7, rng)
        energy = int(_clamp(energy + rng.randint(-3, -1), 2, 5))
        breathlessness = int(_clamp(breathlessness + rng.randint(0, 2), 3, 7))
    elif condition == "fetal_distress":
        weeks = int(_clamp(weeks + rng.randint(0, 4), 28, 40))
        seq[0].kick_count = int(_clamp(seq[0].kick_count + rng.randint(-4, -1), 2, 8))
        seq[1].kick_count = int(_clamp(seq[1].kick_count + rng.randint(-5, -2), 1, 6))
        seq[2].kick_count = int(_clamp(seq[2].kick_count + rng.randint(-6, -3), 0, 4))
        energy = int(_clamp(energy + rng.randint(-2, 0), 3, 6))
        breathlessness = int(_clamp(breathlessness + rng.randint(0, 1), 3, 7))
    else:
        weeks = int(_clamp(weeks + rng.randint(-2, 2), 14, 32))
        history_flags = [flag for flag in history_flags if rng.random() < 0.2]
        for day in seq:
            day.bp_systolic = int(_clamp(day.bp_systolic + rng.randint(-6, 6), 106, 126))
            day.bp_diastolic = int(_clamp(day.bp_diastolic + rng.randint(-4, 4), 68, 82))
            day.kick_count = int(_clamp(day.kick_count + rng.randint(0, 3), 7, 12))
            day.meals_count = int(_clamp(day.meals_count + rng.randint(0, 1), 3, 5))
            day.sleep_hours = float(_clamp(day.sleep_hours + rng.uniform(0.0, 1.0), 6.0, 9.0))
            day.symptom_headache = False
            day.symptom_blurred_vision = False
            day.symptom_swelling = False
            day.symptom_abdominal_pain = False
            day.symptom_bleeding = False
            day.symptom_dizziness = False
        weight_kg = float(_clamp(weight_kg + rng.uniform(-2.0, 3.0), 48.0, 78.0))
        energy = int(_clamp(energy + rng.randint(1, 3), 6, 9))
        breathlessness = int(_clamp(breathlessness + rng.randint(-2, 0), 1, 4))

    return weeks, history_flags, seq, weight_kg, energy, breathlessness


def _mutate_seed_case(seed: SeedCase, rng: random.Random, target_condition: str | None = None) -> dict[str, Any]:
    obs = seed.observation
    weeks = int(_clamp(obs.weeks_pregnant + rng.randint(-2, 2), 14, 40))
    age = int(_clamp(seed.age + rng.randint(-2, 2), 18, 39))
    history_flags = list(obs.history_flags)

    daily_sequence = [
        DailySnapshot(
            bp_systolic=int(_clamp(day.bp_systolic + rng.randint(-5, 5), 90, 180)),
            bp_diastolic=int(_clamp(day.bp_diastolic + rng.randint(-4, 4), 55, 120)),
            kick_count=int(_clamp(day.kick_count + rng.randint(-2, 2), 0, 12)),
            meals_count=int(_clamp(day.meals_count + rng.randint(-1, 1), 1, 5)),
            water_litres=float(_clamp(day.water_litres + rng.uniform(-0.4, 0.4), 0.8, 5.0)),
            sleep_hours=float(_clamp(day.sleep_hours + rng.uniform(-0.8, 0.8), 2.5, 10.0)),
            symptom_headache=bool(day.symptom_headache),
            symptom_blurred_vision=bool(day.symptom_blurred_vision),
            symptom_swelling=bool(day.symptom_swelling),
            symptom_abdominal_pain=bool(day.symptom_abdominal_pain),
            symptom_bleeding=bool(day.symptom_bleeding),
            symptom_dizziness=bool(day.symptom_dizziness),
        )
        for day in seed.daily_sequence
    ]

    if target_condition is None:
        target_condition = infer_reference_condition(obs)

    weight_kg = float(_clamp((obs.latest_weight_kg or 58.0) + rng.uniform(-2.5, 2.5), 40.0, 95.0))
    energy = int(_clamp((obs.latest_energy or 6) + rng.randint(-1, 1), 1, 10))
    breathlessness = int(_clamp((obs.latest_breathlessness or 3) + rng.randint(-1, 1), 1, 10))

    weeks, history_flags, daily_sequence, weight_kg, energy, breathlessness = _condition_specific_adjustments(
        target_condition,
        weeks=weeks,
        history_flags=history_flags,
        daily_sequence=daily_sequence,
        weight_kg=weight_kg,
        energy=energy,
        breathlessness=breathlessness,
        rng=rng,
    )

    synthetic_obs = _observation_from_sequence(
        age=age,
        region=seed.region,
        weeks_pregnant=weeks,
        history_flags=history_flags,
        daily_sequence=daily_sequence,
        latest_weight_kg=weight_kg,
        latest_energy=energy,
        latest_breathlessness=breathlessness,
    )

    return {
        "age": age,
        "region": seed.region,
        "observation": synthetic_obs,
        "daily_sequence": daily_sequence,
    }


def synthesize_dataset(
    *,
    num_samples: int = 4096,
    rng_seed: int = 42,
    db_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(rng_seed)
    seeds = load_seed_cases(db_path=db_path)
    grouped: dict[str, list[SeedCase]] = {name: [] for name in SAFE_CONDITIONS}
    for seed in seeds:
        grouped[infer_reference_condition(seed.observation)].append(seed)

    condition_cycle = [name for name in SAFE_CONDITIONS if grouped[name]]
    samples: list[dict[str, Any]] = []
    regions = sorted({seed.region for seed in seeds} | {"Unknown"})
    region_to_idx = {name: idx for idx, name in enumerate(regions)}

    for sample_idx in range(num_samples):
        target_condition = condition_cycle[sample_idx % len(condition_cycle)]
        seed = rng.choice(grouped[target_condition])
        sample = None
        for _ in range(10):
            candidate = _mutate_seed_case(seed, rng, target_condition=target_condition)
            derived = infer_reference_condition(candidate["observation"])
            if derived == target_condition or rng.random() < 0.15:
                sample = candidate
                break
        if sample is None:
            sample = candidate

        obs = sample["observation"]
        reference_condition = infer_reference_condition(obs)
        reference_urgency = choose_urgency(reference_condition, featurize(obs))
        danger_target = float(reference_urgency == "go_to_hospital_today")
        samples.append(
            {
                "age": sample["age"],
                "region": sample["region"],
                "region_idx": region_to_idx.get(sample["region"], region_to_idx["Unknown"]),
                "observation": obs,
                "daily_sequence": sample["daily_sequence"],
                "condition_idx": CONDITION_TO_IDX[reference_condition],
                "urgency_idx": URGENCY_TO_IDX[reference_urgency],
                "danger_target": danger_target,
                "latent_target": _dense_latent_vector(obs),
            }
        )

    return samples, region_to_idx


class MAASDeepPolicyDataset(Dataset):
    def __init__(self, samples: Sequence[dict[str, Any]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.samples[index]
        obs = item["observation"]
        return {
            "day_seq": torch.tensor(
                [_normalize_day(day) for day in item["daily_sequence"]],
                dtype=torch.float32,
            ),
            "static_cont": torch.tensor(_continuous_vector(obs, item["age"]), dtype=torch.float32),
            "risk_flags": torch.tensor(_risk_vector(RISK_FLAG_NAMES, obs.risk_flags), dtype=torch.float32),
            "history_flags": torch.tensor(_risk_vector(HISTORY_FLAG_NAMES, obs.history_flags), dtype=torch.float32),
            "handcrafted": torch.tensor(_handcrafted_vector(obs), dtype=torch.float32),
            "region_idx": torch.tensor(item["region_idx"], dtype=torch.long),
            "condition_idx": torch.tensor(item["condition_idx"], dtype=torch.long),
            "urgency_idx": torch.tensor(item["urgency_idx"], dtype=torch.long),
            "danger_target": torch.tensor(item["danger_target"], dtype=torch.float32),
            "latent_target": torch.tensor(item["latent_target"], dtype=torch.float32),
        }


class GatedResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.ff(x)
        gated = self.gate(x) * residual
        return self.norm(x + self.dropout(gated))


class TemporalRelationalMAASModel(nn.Module):
    def __init__(
        self,
        *,
        num_regions: int,
        day_feature_dim: int,
        static_dim: int,
        risk_dim: int,
        history_dim: int,
        handcrafted_dim: int,
        hidden_size: int = 128,
        latent_dim: int = len(LATENT_RISK_NAMES),
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.region_embedding = nn.Embedding(num_regions, 24)
        self.day_projection = nn.Linear(day_feature_dim, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.positional = nn.Parameter(torch.zeros(1, 4, hidden_size))
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.10,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            ),
            num_layers=3,
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim + risk_dim + history_dim + handcrafted_dim + 24, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.10,
            batch_first=True,
        )
        self.fusion_projection = nn.Linear(hidden_size * 3, hidden_size)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.residual_stack = nn.ModuleList(
            [GatedResidualBlock(hidden_size, dropout=0.10) for _ in range(4)]
        )
        self.condition_head = nn.Linear(hidden_size, len(SAFE_CONDITIONS))
        self.urgency_head = nn.Linear(hidden_size, len(URGENCY_ORDER))
        self.danger_head = nn.Linear(hidden_size, 1)
        self.latent_head = nn.Linear(hidden_size, latent_dim)

    def forward(
        self,
        *,
        day_seq: torch.Tensor,
        static_cont: torch.Tensor,
        risk_flags: torch.Tensor,
        history_flags: torch.Tensor,
        handcrafted: torch.Tensor,
        region_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = day_seq.size(0)
        day_tokens = self.day_projection(day_seq)
        cls = self.cls_token.expand(batch_size, -1, -1)
        temporal_tokens = torch.cat([cls, day_tokens], dim=1)
        temporal_tokens = temporal_tokens + self.positional[:, : temporal_tokens.size(1)]
        temporal_tokens = self.temporal_encoder(temporal_tokens)
        temporal_summary = temporal_tokens[:, 0]

        static_features = torch.cat(
            [
                static_cont,
                risk_flags,
                history_flags,
                handcrafted,
                self.region_embedding(region_idx),
            ],
            dim=-1,
        )
        static_summary = self.static_encoder(static_features)

        attended, _ = self.cross_attention(
            static_summary.unsqueeze(1),
            temporal_tokens[:, 1:],
            temporal_tokens[:, 1:],
        )
        attended = attended.squeeze(1)

        fusion_input = torch.cat([temporal_summary, static_summary, attended], dim=-1)
        fused = self.fusion_projection(fusion_input)
        gate = self.fusion_gate(fusion_input)
        hidden = gate * fused + (1.0 - gate) * temporal_summary
        for block in self.residual_stack:
            hidden = block(hidden)

        return {
            "condition_logits": self.condition_head(hidden),
            "urgency_logits": self.urgency_head(hidden),
            "danger_logit": self.danger_head(hidden).squeeze(-1),
            "latent_scores": torch.sigmoid(self.latent_head(hidden)),
        }


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_model(
    *,
    num_samples: int = 4096,
    epochs: int = 12,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    db_path: str | Path | None = None,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    samples, region_to_idx = synthesize_dataset(
        num_samples=num_samples,
        rng_seed=seed,
        db_path=db_path,
    )
    rng = random.Random(seed)
    rng.shuffle(samples)
    split_idx = int(len(samples) * 0.85)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_dataset = MAASDeepPolicyDataset(train_samples)
    val_dataset = MAASDeepPolicyDataset(val_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = TemporalRelationalMAASModel(
        num_regions=len(region_to_idx),
        day_feature_dim=12,
        static_dim=10,
        risk_dim=len(RISK_FLAG_NAMES),
        history_dim=len(HISTORY_FLAG_NAMES),
        handcrafted_dim=len(HANDCRAFTED_FEATURE_NAMES),
    ).to(selected_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    mse = nn.SmoothL1Loss()

    history: list[dict[str, float]] = []
    best_state = None
    best_val_condition = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_condition_correct = 0
        train_urgency_correct = 0
        train_count = 0

        for batch in train_loader:
            batch = _batch_to_device(batch, selected_device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                day_seq=batch["day_seq"],
                static_cont=batch["static_cont"],
                risk_flags=batch["risk_flags"],
                history_flags=batch["history_flags"],
                handcrafted=batch["handcrafted"],
                region_idx=batch["region_idx"],
            )
            loss = (
                ce(outputs["condition_logits"], batch["condition_idx"])
                + 0.9 * ce(outputs["urgency_logits"], batch["urgency_idx"])
                + 0.5 * bce(outputs["danger_logit"], batch["danger_target"])
                + 0.25 * mse(outputs["latent_scores"], batch["latent_target"])
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size_now = batch["condition_idx"].size(0)
            train_count += batch_size_now
            train_loss += loss.item() * batch_size_now
            train_condition_correct += int(
                (outputs["condition_logits"].argmax(dim=-1) == batch["condition_idx"]).sum().item()
            )
            train_urgency_correct += int(
                (outputs["urgency_logits"].argmax(dim=-1) == batch["urgency_idx"]).sum().item()
            )

        model.eval()
        val_loss = 0.0
        val_condition_correct = 0
        val_urgency_correct = 0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = _batch_to_device(batch, selected_device)
                outputs = model(
                    day_seq=batch["day_seq"],
                    static_cont=batch["static_cont"],
                    risk_flags=batch["risk_flags"],
                    history_flags=batch["history_flags"],
                    handcrafted=batch["handcrafted"],
                    region_idx=batch["region_idx"],
                )
                loss = (
                    ce(outputs["condition_logits"], batch["condition_idx"])
                    + 0.9 * ce(outputs["urgency_logits"], batch["urgency_idx"])
                    + 0.5 * bce(outputs["danger_logit"], batch["danger_target"])
                    + 0.25 * mse(outputs["latent_scores"], batch["latent_target"])
                )
                batch_size_now = batch["condition_idx"].size(0)
                val_count += batch_size_now
                val_loss += loss.item() * batch_size_now
                val_condition_correct += int(
                    (outputs["condition_logits"].argmax(dim=-1) == batch["condition_idx"]).sum().item()
                )
                val_urgency_correct += int(
                    (outputs["urgency_logits"].argmax(dim=-1) == batch["urgency_idx"]).sum().item()
                )

        scheduler.step()
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss / max(train_count, 1),
            "val_loss": val_loss / max(val_count, 1),
            "train_condition_acc": train_condition_correct / max(train_count, 1),
            "val_condition_acc": val_condition_correct / max(val_count, 1),
            "train_urgency_acc": train_urgency_correct / max(train_count, 1),
            "val_urgency_acc": val_urgency_correct / max(val_count, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)
        if epoch_metrics["val_condition_acc"] > best_val_condition:
            best_val_condition = epoch_metrics["val_condition_acc"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model.cpu(),
        "history": history,
        "region_to_idx": region_to_idx,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "config": {
            "num_samples": num_samples,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
        },
    }


def save_checkpoint(
    *,
    model: TemporalRelationalMAASModel,
    region_to_idx: dict[str, int],
    output_dir: str | Path,
    config: dict[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "maas_deep_policy.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "region_to_idx": region_to_idx,
            "config": config,
        },
        checkpoint_path,
    )
    return checkpoint_path


def save_history(history: Sequence[dict[str, float]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(history), indent=2), encoding="utf-8")
    return output_path


def render_training_curve(history: Sequence[dict[str, float]], output_path: str | Path) -> Path:
    width = 1280
    height = 780
    margin = 70
    img = Image.new("RGB", (width, height), "#f7f5ef")
    draw = ImageDraw.Draw(img)

    plot_boxes = {
        "loss": (margin, 90, width // 2 - 30, height - 80),
        "accuracy": (width // 2 + 30, 90, width - margin, height - 80),
    }

    title = "MAAS Deep Policy Smoke Training"
    draw.text((margin, 24), title, fill="#17212b")
    draw.text((margin, 52), "Transformer + gated fusion over MAAS observation and daily history features", fill="#475569")

    def draw_axes(box: tuple[int, int, int, int], label: str) -> None:
        x0, y0, x1, y1 = box
        draw.rounded_rectangle(box, radius=18, outline="#d1d5db", fill="#ffffff", width=2)
        draw.line((x0 + 48, y1 - 42, x1 - 18, y1 - 42), fill="#94a3b8", width=2)
        draw.line((x0 + 48, y0 + 24, x0 + 48, y1 - 42), fill="#94a3b8", width=2)
        draw.text((x0 + 18, y0 + 10), label, fill="#0f172a")

    draw_axes(plot_boxes["loss"], "Loss")
    draw_axes(plot_boxes["accuracy"], "Validation Accuracy")

    epochs = [int(item["epoch"]) for item in history]
    max_epoch = max(epochs)
    loss_values = [item["train_loss"] for item in history] + [item["val_loss"] for item in history]
    max_loss = max(loss_values) * 1.05
    min_loss = min(loss_values) * 0.95
    if math.isclose(max_loss, min_loss):
        max_loss += 0.1

    def map_point(
        box: tuple[int, int, int, int],
        epoch: int,
        value: float,
        lo: float,
        hi: float,
    ) -> tuple[int, int]:
        x0, y0, x1, y1 = box
        usable_w = (x1 - x0) - 78
        usable_h = (y1 - y0) - 78
        px = x0 + 48 + int((epoch - 1) / max(max_epoch - 1, 1) * usable_w)
        py = y1 - 42 - int((value - lo) / max(hi - lo, 1e-6) * usable_h)
        return px, py

    def draw_series(
        box: tuple[int, int, int, int],
        values: Sequence[float],
        lo: float,
        hi: float,
        color: str,
        label: str,
        label_offset: int,
    ) -> None:
        points = [map_point(box, epochs[idx], values[idx], lo, hi) for idx in range(len(values))]
        for start, end in zip(points, points[1:]):
            draw.line((*start, *end), fill=color, width=4)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
        x0, y0, _, _ = box
        draw.rectangle((x0 + 210, y0 + 10 + label_offset, x0 + 228, y0 + 24 + label_offset), fill=color)
        draw.text((x0 + 236, y0 + 8 + label_offset), label, fill="#0f172a")

    draw_series(
        plot_boxes["loss"],
        [item["train_loss"] for item in history],
        min_loss,
        max_loss,
        "#d97706",
        "Train loss",
        0,
    )
    draw_series(
        plot_boxes["loss"],
        [item["val_loss"] for item in history],
        min_loss,
        max_loss,
        "#2563eb",
        "Val loss",
        20,
    )
    draw_series(
        plot_boxes["accuracy"],
        [item["val_condition_acc"] for item in history],
        0.0,
        1.0,
        "#16a34a",
        "Condition acc",
        0,
    )
    draw_series(
        plot_boxes["accuracy"],
        [item["val_urgency_acc"] for item in history],
        0.0,
        1.0,
        "#7c3aed",
        "Urgency acc",
        20,
    )

    for epoch in epochs:
        loss_x, loss_y = map_point(plot_boxes["loss"], epoch, min_loss, min_loss, max_loss)
        acc_x, acc_y = map_point(plot_boxes["accuracy"], epoch, 0.0, 0.0, 1.0)
        draw.text((loss_x - 4, plot_boxes["loss"][3] - 34), str(epoch), fill="#64748b")
        draw.text((acc_x - 4, plot_boxes["accuracy"][3] - 34), str(epoch), fill="#64748b")

    summary = history[-1]
    footer = (
        f"Final val condition acc: {summary['val_condition_acc']:.3f}    "
        f"Final val urgency acc: {summary['val_urgency_acc']:.3f}    "
        f"Final val loss: {summary['val_loss']:.3f}"
    )
    draw.text((margin, height - 42), footer, fill="#334155")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def _sample_to_model_inputs(
    age: int,
    region: str,
    observation: Observation,
    daily_sequence: Sequence[DailySnapshot],
    region_to_idx: dict[str, int],
) -> dict[str, torch.Tensor]:
    return {
        "day_seq": torch.tensor([_normalize_day(day) for day in daily_sequence], dtype=torch.float32).unsqueeze(0),
        "static_cont": torch.tensor(_continuous_vector(observation, age), dtype=torch.float32).unsqueeze(0),
        "risk_flags": torch.tensor(_risk_vector(RISK_FLAG_NAMES, observation.risk_flags), dtype=torch.float32).unsqueeze(0),
        "history_flags": torch.tensor(_risk_vector(HISTORY_FLAG_NAMES, observation.history_flags), dtype=torch.float32).unsqueeze(0),
        "handcrafted": torch.tensor(_handcrafted_vector(observation), dtype=torch.float32).unsqueeze(0),
        "region_idx": torch.tensor([region_to_idx.get(region, 0)], dtype=torch.long),
    }


def load_checkpoint(checkpoint_path: str | Path) -> tuple[TemporalRelationalMAASModel, dict[str, int], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    region_to_idx = dict(payload["region_to_idx"])
    model = TemporalRelationalMAASModel(
        num_regions=len(region_to_idx),
        day_feature_dim=12,
        static_dim=10,
        risk_dim=len(RISK_FLAG_NAMES),
        history_dim=len(HISTORY_FLAG_NAMES),
        handcrafted_dim=len(HANDCRAFTED_FEATURE_NAMES),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, region_to_idx, payload["config"]


def default_checkpoint_path() -> Path:
    return Path("trained_models/maas_deep_policy.pt")


@lru_cache(maxsize=4)
def _cached_checkpoint_bundle(checkpoint_path: str) -> tuple[TemporalRelationalMAASModel, dict[str, int], dict[str, Any]]:
    return load_checkpoint(checkpoint_path)


def predict_from_observation(
    *,
    model: TemporalRelationalMAASModel,
    region_to_idx: dict[str, int],
    age: int,
    region: str,
    observation: Observation,
    daily_sequence: Sequence[DailySnapshot],
) -> dict[str, Any]:
    with torch.no_grad():
        inputs = _sample_to_model_inputs(age, region, observation, daily_sequence, region_to_idx)
        outputs = model(**inputs)
        condition_probs = torch.softmax(outputs["condition_logits"], dim=-1).squeeze(0)
        urgency_probs = torch.softmax(outputs["urgency_logits"], dim=-1).squeeze(0)
        condition_idx = int(condition_probs.argmax().item())
        urgency_idx = int(urgency_probs.argmax().item())
        predicted_condition = IDX_TO_CONDITION[condition_idx]
        predicted_urgency = IDX_TO_URGENCY[urgency_idx]
        confidence = float(condition_probs[condition_idx].item())
        danger_probability = float(torch.sigmoid(outputs["danger_logit"]).item())
        latent_values = outputs["latent_scores"].squeeze(0).tolist()
        latent_named = {
            name: round(float(value), 4)
            for name, value in zip(LATENT_RISK_NAMES, latent_values)
        }
        active_features = featurize(observation)
        top_features = [
            name
            for name in HANDCRAFTED_FEATURE_NAMES
            if active_features[name] > 0
        ][:5]

    rationale_bits = [
        f"Predicted {predicted_condition} with {confidence:.2f} confidence",
        f"urgency={predicted_urgency}",
        f"danger_prob={danger_probability:.2f}",
    ]
    if top_features:
        rationale_bits.append("supporting features: " + ", ".join(top_features))
    dominant_latent = sorted(latent_named.items(), key=lambda item: item[1], reverse=True)[:2]
    rationale_bits.append(
        "latent risks: "
        + ", ".join(f"{name}={value:.2f}" for name, value in dominant_latent)
    )

    return {
        "condition": predicted_condition,
        "urgency": predicted_urgency,
        "rationale": "; ".join(rationale_bits),
        "confidence": confidence,
        "danger_probability": danger_probability,
        "latent_risks": latent_named,
    }


def predict_for_user_id(
    *,
    user_id: int,
    checkpoint_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_checkpoint = Path(checkpoint_path or default_checkpoint_path())
    model, region_to_idx, _ = _cached_checkpoint_bundle(str(resolved_checkpoint.resolve()))
    env = PrenatalEnvironment()
    prompt = env.reset(user_id)
    observation = prompt.observation

    conn = sqlite3.connect(str(Path(db_path or "prenatal.db")))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT age, region FROM user_profiles WHERE id = ?", (user_id,))
    user = cur.fetchone()
    cur.execute(
        """
        SELECT bp_systolic, bp_diastolic, kick_count, meals_count, water_litres, sleep_hours,
               symptom_headache, symptom_blurred_vision, symptom_swelling,
               symptom_abdominal_pain, symptom_bleeding, symptom_dizziness
        FROM daily_checkins
        WHERE user_id = ?
        ORDER BY created_at ASC
        LIMIT 3
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()

    daily_sequence = [
        DailySnapshot(
            bp_systolic=int(row["bp_systolic"]),
            bp_diastolic=int(row["bp_diastolic"]),
            kick_count=int(row["kick_count"] if row["kick_count"] is not None else 0),
            meals_count=int(row["meals_count"]),
            water_litres=float(row["water_litres"]),
            sleep_hours=float(row["sleep_hours"]),
            symptom_headache=bool(row["symptom_headache"]),
            symptom_blurred_vision=bool(row["symptom_blurred_vision"]),
            symptom_swelling=bool(row["symptom_swelling"]),
            symptom_abdominal_pain=bool(row["symptom_abdominal_pain"]),
            symptom_bleeding=bool(row["symptom_bleeding"]),
            symptom_dizziness=bool(row["symptom_dizziness"]),
        )
        for row in rows
    ]
    return predict_from_observation(
        model=model,
        region_to_idx=region_to_idx,
        age=int(user["age"]),
        region=str(user["region"]),
        observation=observation,
        daily_sequence=daily_sequence,
    )
