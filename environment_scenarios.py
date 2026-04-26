"""
Synthetic offline scenarios for the MAAS OpenEnv environment.

Constraints:
- Generated programmatically (no hardcoding 400+ entries).
- Deterministic across runs: random.seed(42).
- No imports from `environment.py` to avoid circular imports.
- Each scenario includes an `observation` dict matching (and supersetting) the
  Observation schema used by `environment.py`.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Literal, Optional, TypedDict


SAFE_CONDITIONS = (
    "preeclampsia",
    "gestational_diabetes",
    "anemia",
    "preterm_risk",
    "fetal_distress",
    "low_risk",
)

SAFE_URGENCIES = ("monitor_at_home", "visit_phc_this_week", "go_to_hospital_today")

Difficulty = Literal["easy", "medium", "hard", "expert"]
AccessTier = Literal["rural", "semi_urban", "urban"]


class Scenario(TypedDict):
    scenario_id: str
    description: str
    difficulty: Difficulty
    true_condition: str
    true_urgency: str
    observation: Dict[str, Any]


_REGIONAL_PROFILES: Dict[str, Dict[str, Any]] = {
    # Requested regional variants: tune meals/sleep/access and a bit of noise.
    "rural Rajasthan": {"access": "rural", "avg_meals": (1.6, 2.4), "avg_sleep": (4.0, 6.0)},
    "urban Karnataka": {"access": "urban", "avg_meals": (2.5, 3.2), "avg_sleep": (6.0, 8.0)},
    "tribal Jharkhand": {"access": "rural", "avg_meals": (1.4, 2.3), "avg_sleep": (4.0, 6.5)},
    "coastal Odisha": {"access": "semi_urban", "avg_meals": (2.0, 3.0), "avg_sleep": (5.5, 7.5)},
}


_SYMPTOMS = ("headache", "swelling", "dizziness", "bleeding", "abdominal_pain", "blurred_vision", "vomiting")


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _pick_region(rng: random.Random) -> str:
    return rng.choice(list(_REGIONAL_PROFILES.keys()))


def _access_tier_for_region(region: str) -> AccessTier:
    access = _REGIONAL_PROFILES.get(region, {}).get("access", "semi_urban")
    return access  # type: ignore[return-value]


def _regional_access_penalty(tier: AccessTier) -> float:
    return {"rural": 0.3, "semi_urban": 0.6, "urban": 1.0}[tier]


def _random_history_flags(rng: random.Random, *, base_condition: str) -> List[str]:
    flags: List[str] = []
    # Keep history noisy but plausible.
    if rng.random() < 0.25:
        flags.append("family_diabetes")
    if rng.random() < 0.25:
        flags.append("family_hypertension")
    if rng.random() < (0.35 if base_condition in ("preeclampsia", "preterm_risk") else 0.15):
        flags.append("prev_complication")
    if rng.random() < (0.25 if base_condition == "preeclampsia" else 0.08):
        flags.append("prev_preeclampsia")
    # Adversarial: misleading history.
    if rng.random() < 0.06:
        flags.append(rng.choice(["prev_gdm_proxy", "prev_anemia_proxy"]))
    return sorted(set(flags))


def _symptom_cluster(rng: random.Random, *, condition: str, difficulty: Difficulty) -> List[str]:
    # Default: sparse symptoms unless the condition demands them.
    p = {"easy": 0.25, "medium": 0.35, "hard": 0.45, "expert": 0.55}[difficulty]
    cluster: List[str] = []

    if condition == "preeclampsia":
        for s in ("headache", "swelling", "blurred_vision", "dizziness"):
            if rng.random() < p:
                cluster.append(s)
        if rng.random() < (0.20 if difficulty in ("hard", "expert") else 0.10):
            cluster.append("vomiting")
    elif condition == "fetal_distress":
        if rng.random() < 0.25:
            cluster.append("dizziness")
        if rng.random() < 0.15:
            cluster.append("vomiting")
    elif condition == "preterm_risk":
        for s in ("bleeding", "abdominal_pain", "dizziness"):
            if rng.random() < p:
                cluster.append(s)
    elif condition == "anemia":
        for s in ("dizziness",):
            if rng.random() < p:
                cluster.append(s)
        if rng.random() < 0.10:
            cluster.append("vomiting")
    elif condition == "gestational_diabetes":
        if rng.random() < 0.15:
            cluster.append("vomiting")
        if rng.random() < 0.12:
            cluster.append("dizziness")
    else:
        # low_risk: occasional benign symptoms.
        if rng.random() < 0.10:
            cluster.append(rng.choice(("dizziness", "vomiting")))

    # Expert/adversarial: add noisy symptom(s).
    if difficulty in ("hard", "expert") and rng.random() < 0.12:
        cluster.append(rng.choice(_SYMPTOMS))

    return sorted(set([s for s in cluster if s in _SYMPTOMS]))


def _bp_for_condition(rng: random.Random, *, condition: str, difficulty: Difficulty) -> tuple[Optional[int], Optional[int]]:
    # Some scenarios intentionally omit BP to test missing data.
    if difficulty in ("hard", "expert") and rng.random() < 0.10:
        return None, None

    if condition == "preeclampsia":
        severe = rng.random() < (0.55 if difficulty in ("hard", "expert") else 0.25)
        if severe:
            sys = rng.randint(160, 190)
            dia = rng.randint(110, 130)
        else:
            sys = rng.randint(140, 159)
            dia = rng.randint(90, 109)
        return sys, dia
    if condition == "gestational_diabetes":
        # GDM itself doesn't drive BP; keep mostly normal with mild elevations sometimes.
        sys = rng.randint(110, 135) + (5 if rng.random() < 0.12 else 0)
        dia = rng.randint(70, 88) + (3 if rng.random() < 0.10 else 0)
        return sys, dia
    if condition == "preterm_risk":
        sys = rng.randint(110, 140)
        dia = rng.randint(70, 90)
        return sys, dia
    if condition == "fetal_distress":
        sys = rng.randint(105, 145)
        dia = rng.randint(65, 92)
        return sys, dia
    if condition == "anemia":
        sys = rng.randint(95, 130)
        dia = rng.randint(55, 85)
        return sys, dia
    # low_risk
    sys = rng.randint(100, 125)
    dia = rng.randint(60, 80)
    return sys, dia


def _kicks_for_condition(rng: random.Random, *, condition: str, weeks: int, difficulty: Difficulty) -> tuple[Optional[float], Optional[int]]:
    # Some scenarios omit kicks (early pregnancy or missing data).
    if weeks < 20 or (difficulty in ("hard", "expert") and rng.random() < 0.08):
        return None, None

    if condition == "fetal_distress":
        absent = rng.random() < (0.30 if difficulty in ("hard", "expert") else 0.15)
        if absent:
            return 0.0, 0
        low = rng.random() < 0.65
        if low:
            avg = rng.uniform(1.0, 4.0)
            latest = rng.randint(0, 4)
        else:
            avg = rng.uniform(4.0, 7.0)
            latest = rng.randint(3, 7)
        return float(_clamp(avg, 0.0, 10.0)), int(_clamp(latest, 0, 12))

    # Other conditions: mostly normal.
    avg = rng.uniform(5.0, 9.0)
    latest = rng.randint(4, 10)
    # Adversarial noise: low kicks in non-fetal cases.
    if difficulty in ("hard", "expert") and rng.random() < 0.10 and condition != "low_risk":
        avg = rng.uniform(2.0, 5.0)
        latest = rng.randint(1, 5)
    return float(_clamp(avg, 0.0, 10.0)), int(_clamp(latest, 0, 12))


def _meals_sleep_energy_breath(rng: random.Random, *, condition: str, region: str, difficulty: Difficulty) -> tuple[float, float, int, int, Optional[float]]:
    rp = _REGIONAL_PROFILES.get(region, _REGIONAL_PROFILES["coastal Odisha"])
    meals = rng.uniform(*rp["avg_meals"])
    sleep = rng.uniform(*rp["avg_sleep"])

    # Condition-specific drift.
    if condition == "anemia":
        meals -= rng.uniform(0.2, 0.7)
        sleep -= rng.uniform(0.0, 0.8)
    if condition == "preeclampsia":
        sleep -= rng.uniform(0.0, 0.6)
    if condition == "preterm_risk":
        sleep -= rng.uniform(0.0, 0.6)

    # Adversarial missing/outliers.
    if difficulty == "expert" and rng.random() < 0.04:
        meals = rng.choice([0.8, 4.0])  # outlier under/over
    if difficulty == "expert" and rng.random() < 0.04:
        sleep = rng.choice([2.5, 9.5])

    meals = float(_clamp(meals, 0.5, 4.0))
    sleep = float(_clamp(sleep, 2.0, 10.0))

    # Energy and breathlessness proxies.
    energy = int(_clamp(rng.gauss(6.5, 1.8), 0.0, 10.0))
    breath = int(_clamp(rng.gauss(3.0, 2.0), 0.0, 10.0))

    if condition == "anemia":
        energy = int(_clamp(rng.gauss(3.5, 1.8), 0.0, 10.0))
        breath = int(_clamp(rng.gauss(6.5, 2.0), 0.0, 10.0))
        # Severe anemia proxy sometimes.
        if difficulty in ("hard", "expert") and rng.random() < 0.35:
            energy = int(_clamp(rng.gauss(1.8, 0.9), 0.0, 10.0))
            breath = int(_clamp(rng.gauss(8.2, 1.2), 0.0, 10.0))

    weight = float(_clamp(rng.gauss(58.0, 8.0), 38.0, 95.0))
    # Rapid weight gain proxy in preeclampsia harder cases.
    if condition == "preeclampsia" and difficulty in ("hard", "expert") and rng.random() < 0.18:
        weight = float(_clamp(weight + rng.uniform(5.0, 8.5), 38.0, 100.0))
    return meals, sleep, energy, breath, weight


def _bp_trend(rng: random.Random, *, condition: str, difficulty: Difficulty) -> str:
    if condition == "preeclampsia":
        return "rising" if rng.random() < (0.75 if difficulty in ("hard", "expert") else 0.55) else "stable"
    if rng.random() < 0.10:
        return "rising"
    return "stable"


def _danger_flags_for_obs(
    *,
    sys_bp: Optional[int],
    dia_bp: Optional[int],
    weeks: int,
    latest_kicks: Optional[int],
    symptoms: List[str],
    meals: float,
    sleep: float,
    energy: int,
    breath: int,
    weight: Optional[float],
) -> List[str]:
    flags: List[str] = []
    if sys_bp is not None and dia_bp is not None:
        if sys_bp >= 160 or dia_bp >= 110:
            flags.append("DANGER_BP_CRITICAL")
        elif sys_bp >= 140 or dia_bp >= 90:
            flags.append("WARN_BP_HIGH")
    if weeks >= 28 and latest_kicks is not None and latest_kicks <= 2:
        flags.append("DANGER_LOW_KICKS")
    if "bleeding" in symptoms:
        flags.append("DANGER_BLEEDING")
    if "blurred_vision" in symptoms and (sys_bp or 0) >= 140:
        flags.append("DANGER_VISION_CHANGES")

    # New flags requested.
    if "abdominal_pain" in symptoms:
        flags.append("ABDOMINAL_PAIN_SIGNAL")
    if "dizziness" in symptoms:
        flags.append("DIZZINESS_SIGNAL")
    if "vomiting" in symptoms:
        flags.append("VOMITING_SIGNAL")
    if len(symptoms) >= 3:
        flags.append("SYMPTOM_CLUSTER_HIGH")
    if weight is not None and weight >= 70 and (sys_bp or 0) >= 140 and "swelling" in symptoms:
        flags.append("RAPID_WEIGHT_GAIN")
    if breath >= 8 and energy <= 2 and meals <= 1.5:
        flags.append("SEVERE_ANEMIA_PROXY")
    if sleep < 4.0 and energy <= 3:
        flags.append("MATERNAL_EXHAUSTION")
    if weeks < 37 and "abdominal_pain" in symptoms and ("bleeding" in symptoms or (sys_bp or 0) >= 140):
        flags.append("PRETERM_CONTRACTION_PROXY")

    return sorted(set(flags))


def _composite_risk_score(flags: List[str]) -> float:
    # Simple 0..1 risk score weighted by danger-ness of flags.
    score = 0.0
    for f in flags:
        if f.startswith("DANGER_"):
            score += 0.22
        elif f.startswith("WARN_"):
            score += 0.10
        elif f.endswith("_PROXY") or f.endswith("_HIGH") or f.endswith("_SIGNAL"):
            score += 0.12
        else:
            score += 0.06
    return float(_clamp(score, 0.0, 1.0))


def _weeks_and_trimester(rng: random.Random, *, condition: str) -> tuple[int, int]:
    # Keep pregnancy weeks plausible per condition.
    if condition == "low_risk":
        weeks = rng.choice([rng.randint(8, 12), rng.randint(18, 24), rng.randint(28, 38)])
    elif condition == "fetal_distress":
        weeks = rng.randint(28, 40)
    elif condition == "preterm_risk":
        weeks = rng.randint(26, 36)
    elif condition == "gestational_diabetes":
        weeks = rng.randint(20, 38)
    elif condition == "preeclampsia":
        weeks = rng.randint(20, 40)
    else:  # anemia
        weeks = rng.randint(10, 38)
    trimester = 1 if weeks < 14 else 2 if weeks < 28 else 3
    return weeks, trimester


def _true_urgency(condition: str, *, flags: List[str], difficulty: Difficulty, access: AccessTier) -> str:
    # Default urgency by condition with safety overrides.
    if any(f.startswith("DANGER_") for f in flags) or "SEVERE_ANEMIA_PROXY" in flags:
        return "go_to_hospital_today"
    if condition in ("preeclampsia", "fetal_distress"):
        return "go_to_hospital_today" if difficulty in ("hard", "expert") else "visit_phc_this_week"
    if condition in ("gestational_diabetes", "anemia", "preterm_risk"):
        return "visit_phc_this_week"
    # low risk: consider access tier (rural gets slightly higher recommended follow-up).
    return "visit_phc_this_week" if access == "rural" and difficulty in ("hard", "expert") else "monitor_at_home"


def _make_scenario(rng: random.Random, *, scenario_id: str, condition: str, difficulty: Difficulty) -> Scenario:
    region = _pick_region(rng)
    access = _access_tier_for_region(region)
    weeks, trimester = _weeks_and_trimester(rng, condition=condition)
    symptoms = _symptom_cluster(rng, condition=condition, difficulty=difficulty)

    sys_bp, dia_bp = _bp_for_condition(rng, condition=condition, difficulty=difficulty)
    avg_kicks, latest_kicks = _kicks_for_condition(rng, condition=condition, weeks=weeks, difficulty=difficulty)
    meals, sleep, energy, breath, weight = _meals_sleep_energy_breath(
        rng, condition=condition, region=region, difficulty=difficulty
    )

    flags = _danger_flags_for_obs(
        sys_bp=sys_bp,
        dia_bp=dia_bp,
        weeks=weeks,
        latest_kicks=latest_kicks,
        symptoms=symptoms,
        meals=meals,
        sleep=sleep,
        energy=energy,
        breath=breath,
        weight=weight,
    )
    history = _random_history_flags(rng, base_condition=condition)
    days_of_data = rng.randint(1, 3) if difficulty in ("hard", "expert") else rng.randint(2, 3)
    urgency = _true_urgency(condition, flags=flags, difficulty=difficulty, access=access)

    obs: Dict[str, Any] = {
        "user_id": 0,
        "weeks_pregnant": weeks,
        "trimester": trimester,
        "region": region,
        "regional_access_tier": access,
        "risk_flags": flags,
        "composite_risk_score": _composite_risk_score(flags),
        "bp_trend": _bp_trend(rng, condition=condition, difficulty=difficulty),
        "avg_kick_count": avg_kicks,
        "avg_meals": meals,
        "avg_sleep": sleep,
        "latest_weight_kg": weight,
        "latest_energy": energy,
        "latest_breathlessness": breath,
        "history_flags": history,
        "days_of_data": days_of_data,
        # New fields requested.
        "symptom_cluster": symptoms,
        "bp_systolic_latest": sys_bp,
        "bp_diastolic_latest": dia_bp,
        # Existing environment fields for partial observability (default empty/off).
        "masked_signals": [],
        "episode_day_index": 1,
        "total_episode_days": 1,
        "belief_state": {},
        "available_signals": [],
        "withheld_signals": [],
        "signal_mask": {},
    }

    desc_bits = []
    if flags:
        desc_bits.append(", ".join(flags[:4]))
    if symptoms:
        desc_bits.append("sx=" + ",".join(symptoms[:4]))
    if condition == "gestational_diabetes" and ("family_diabetes" in history or "prev_gdm_proxy" in history):
        desc_bits.append("diabetes_history")
    if condition == "preterm_risk" and "bleeding" in symptoms:
        desc_bits.append("bleeding")
    if condition == "fetal_distress" and latest_kicks in (0, 1, 2):
        desc_bits.append("low_kicks")
    short = "; ".join(desc_bits) if desc_bits else "synthetic prenatal check-in"

    return {
        "scenario_id": scenario_id,
        "description": f"{condition} ({difficulty}): {short}",
        "difficulty": difficulty,
        "true_condition": condition,
        "true_urgency": urgency,
        "observation": obs,
    }


def _generate_registry(*, seed: int = 42, n: int = 420) -> List[Scenario]:
    rng = random.Random(seed)
    # Balance conditions and include an adversarial tail by difficulty.
    conditions = list(SAFE_CONDITIONS)
    difficulty_mix: List[Difficulty] = (
        ["easy"] * 120 + ["medium"] * 140 + ["hard"] * 110 + ["expert"] * 50
    )
    rng.shuffle(difficulty_mix)

    registry: List[Scenario] = []
    for idx in range(1, n + 1):
        scenario_id = f"env_{idx:03d}"
        difficulty = difficulty_mix[(idx - 1) % len(difficulty_mix)]
        # Condition selection: cycle + perturbation for coverage.
        base = conditions[(idx - 1) % len(conditions)]
        if difficulty in ("hard", "expert") and rng.random() < 0.18:
            base = rng.choice(conditions)  # mixed/noisy distribution
        scenario = _make_scenario(rng, scenario_id=scenario_id, condition=base, difficulty=difficulty)
        registry.append(scenario)

    # Safety: ensure uniqueness of ids.
    ids = {s["scenario_id"] for s in registry}
    if len(ids) != len(registry):
        raise RuntimeError("SCENARIO_REGISTRY contains duplicate scenario_id values")
    return registry


SCENARIO_REGISTRY: List[Scenario] = _generate_registry(seed=42, n=420)

