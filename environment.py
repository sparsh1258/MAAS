from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from pydantic import Field

from database import SessionLocal
from models import Checkin3Day, DailyCheckin, UserProfile
from xai_reward_model import (
    SAFE_CONDITIONS,
    URGENCY_ORDER,
    calculate_reward,
    choose_urgency,
    featurize,
    infer_reference_condition,
    latent_risk_scores,
    supporting_features,
)

OPENENV_IMPORT_ERROR: Optional[Exception] = None
try:
    from openenv_core import Environment as OpenEnvEnvironment
    OPENENV_AVAILABLE = True
except ImportError as primary_error:
    try:
        from openenv_core import OpenEnv as OpenEnvEnvironment  # type: ignore[attr-defined]
        OPENENV_AVAILABLE = True
    except ImportError as secondary_error:
        OPENENV_IMPORT_ERROR = secondary_error
        OPENENV_AVAILABLE = False

        class OpenEnvEnvironment:  # type: ignore[override]
            """Fallback shim when openenv-core is unavailable locally."""

            pass

CONDITION_URGENCY = {
    "preeclampsia": "go_to_hospital_today",
    "fetal_distress": "go_to_hospital_today",
    "gestational_diabetes": "visit_phc_this_week",
    "preterm_risk": "visit_phc_this_week",
    "anemia": "visit_phc_this_week",
    "low_risk": "monitor_at_home",
}

EPISODE_HIDEABLE_SIGNALS = [
    "risk_flags",
    "latest_blood_pressure",
    "latest_kick_count",
    "latest_symptoms",
    "bp_trend",
    "avg_kick_count",
    "avg_meals",
    "avg_sleep",
    "latest_weight_kg",
    "latest_energy",
    "latest_breathlessness",
]


def _default_signal_mask() -> dict[str, bool]:
    return {signal: True for signal in EPISODE_HIDEABLE_SIGNALS}


class Observation(BaseModel):
    user_id: int
    weeks_pregnant: int
    trimester: int
    region: str
    risk_flags: List[str]
    bp_trend: str
    avg_kick_count: Optional[float]
    avg_meals: Optional[float]
    avg_sleep: Optional[float]
    latest_weight_kg: Optional[float]
    latest_energy: Optional[int]
    latest_breathlessness: Optional[int]
    history_flags: List[str]
    days_of_data: int
    masked_signals: List[str] = []
    episode_day_index: int = 1
    total_episode_days: int = 1
    belief_state: dict[str, float] = Field(default_factory=dict)
    available_signals: List[str] = Field(default_factory=list)
    withheld_signals: List[str] = Field(default_factory=list)
    signal_mask: dict[str, bool] = Field(default_factory=_default_signal_mask)


class PromptObservation(BaseModel):
    observation: Observation
    text_observation: str
    system_prompt: str
    user_prompt: str
    response_format: str
    valid_conditions: List[str]
    valid_urgencies: List[str]


class StepResult(BaseModel):
    observation: Observation
    text_observation: str
    prompt: PromptObservation
    reward: float
    reward_components: dict
    done: bool
    predicted_condition: Optional[str]
    urgency: Optional[str]
    diet_advice: List[str]
    rationale: str
    reference_condition: Optional[str]
    reference_urgency: Optional[str]
    latent_risks: dict
    under_escalated: bool = False


class ActionModel(BaseModel):
    condition: Optional[str] = None
    urgency: Optional[str] = None
    rationale: Optional[str] = None
    action_type: Optional[str] = None
    signal_name: Optional[str] = None
    target: Optional[str] = Field(default=None, exclude=True)


LATENT_CONDITION_ADVICE = {
    "postpartum_hemorrhage": "Monitor bleeding volume closely and escalate immediately if it worsens.",
    "maternal_infection": "Review fever, discharge, and pain symptoms with urgent follow-up if they increase.",
    "dehydration": "Increase oral fluids and reassess dizziness and urine output.",
    "intrahepatic_cholestasis": "Review itching and late-pregnancy symptoms at the next clinical visit.",
    "placental_abruption": "Bleeding plus abdominal pain requires urgent clinical escalation.",
    "maternal_exhaustion": "Encourage rest, hydration, and close symptom monitoring.",
    "nutrition_deficit": "Improve meal regularity and iron-rich food intake.",
}

DIET_ADVICE = {
    "preeclampsia": [
        "Reduce salt completely and avoid pickles and papad.",
        "Eat banana and amla daily to support blood pressure management.",
        "Drink at least 8-10 glasses of water.",
        "Avoid fried and oily food.",
    ],
    "gestational_diabetes": [
        "Avoid rice and refined flour.",
        "Eat dal, vegetables, and roti in small portions.",
        "Take small meals 5-6 times a day.",
        "Prefer guava and jamun over sweeter fruits.",
    ],
    "anemia": [
        "Eat spinach, amaranth, or fenugreek daily.",
        "Consume jaggery with chickpeas for iron.",
        "Pair iron-rich foods with lemon for vitamin C.",
        "Avoid tea and coffee immediately after meals.",
    ],
    "preterm_risk": [
        "Increase protein intake with dal, eggs, or milk.",
        "Avoid lifting heavy weights.",
        "Take more rest with legs slightly elevated.",
        "Reduce stress and seek support.",
    ],
    "fetal_distress": [
        "Take something sweet and count baby kicks for one hour.",
        "Lie on your left side.",
        "Drink water and stay calm.",
        "If kicks do not increase, go to the hospital immediately.",
    ],
    "low_risk": [
        "Take a balanced diet with dal, vegetables, roti, and milk.",
        "Continue iron and folic acid tablets daily.",
        "Drink 8-10 glasses of water daily.",
        "Do light walking for 20-30 minutes.",
    ],
}


def _load_recent_data(user_id: int, days: int = 3):
    db = SessionLocal()
    try:
        user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")

        daily = (
            db.query(DailyCheckin)
            .filter(DailyCheckin.user_id == user_id)
            .order_by(DailyCheckin.created_at.desc())
            .limit(days)
            .all()
        )
        checkin3 = (
            db.query(Checkin3Day)
            .filter(Checkin3Day.user_id == user_id)
            .order_by(Checkin3Day.created_at.desc())
            .first()
        )
        return user, daily, checkin3
    finally:
        db.close()


def _build_observation(user: UserProfile, daily: list, checkin3: Optional[Checkin3Day]) -> Observation:
    risk_flags: List[str] = []
    history_flags: List[str] = []

    if user.history_diabetes:
        history_flags.append("family_diabetes")
    if user.history_hypertension:
        history_flags.append("family_hypertension")
    if user.history_preeclampsia:
        history_flags.append("prev_preeclampsia")
    if user.history_prev_comp:
        history_flags.append("prev_complication")

    if daily:
        latest = daily[0]
        if latest.bp_systolic >= 160 or latest.bp_diastolic >= 110:
            risk_flags.append("DANGER_BP_CRITICAL")
        elif latest.bp_systolic >= 140 or latest.bp_diastolic >= 90:
            risk_flags.append("HIGH_BP")
        if latest.symptom_bleeding:
            risk_flags.append("DANGER_BLEEDING")
        if latest.symptom_blurred_vision and latest.symptom_headache:
            risk_flags.append("DANGER_VISION_HEADACHE")
        if latest.kick_count is not None and latest.kick_count < 3:
            risk_flags.append("DANGER_LOW_KICKS")
        if latest.symptom_swelling and latest.symptom_headache:
            risk_flags.append("HIGH_PREECLAMPSIA_SIGNAL")
        if latest.symptom_dizziness:
            risk_flags.append("DIZZINESS_SIGNAL")
        if latest.symptom_abdominal_pain:
            risk_flags.append("ABDOMINAL_PAIN_SIGNAL")

    bp_trend = "stable"
    if len(daily) >= 2:
        systolics = [d.bp_systolic for d in daily]
        if systolics[0] > systolics[-1] + 10:
            bp_trend = "rising"
            risk_flags.append("BP_RISING_TREND")
        elif systolics[0] < systolics[-1] - 10:
            bp_trend = "falling"

    avg_meals = sum(d.meals_count for d in daily) / len(daily) if daily else 0.0
    avg_sleep = sum(d.sleep_hours for d in daily) / len(daily) if daily else 0.0
    kicks = [d.kick_count for d in daily if d.kick_count is not None]
    avg_kicks = sum(kicks) / len(kicks) if kicks else None

    if avg_meals < 2:
        risk_flags.append("LOW_NUTRITION")
    if avg_kicks is not None and avg_kicks < 6:
        risk_flags.append("LOW_KICK_AVG")
    masked_signals = []
    danger_present = any(f.startswith("DANGER") for f in risk_flags)
    if not danger_present:
        if random.random() < 0.35:
            avg_kicks = None
            masked_signals.append("kick_count")
        if random.random() < 0.35:
            masked_signals.append("energy_level")
        if random.random() < 0.25 and bp_trend == "stable":
            bp_trend = "unknown"
            masked_signals.append("bp_trend")

    return Observation(
        user_id=user.id,
        weeks_pregnant=user.weeks_pregnant,
        trimester=user.trimester,
        region=user.region,
        risk_flags=sorted(set(risk_flags)),
        bp_trend=bp_trend,
        avg_kick_count=avg_kicks,
        avg_meals=avg_meals,
        avg_sleep=avg_sleep,
        latest_weight_kg=checkin3.weight_kg if checkin3 else None,
        latest_energy=checkin3.energy_level if checkin3 else None,
        latest_breathlessness=checkin3.breathlessness if checkin3 else None,
        history_flags=history_flags,
        days_of_data=len(daily),
        masked_signals=masked_signals,
    )


def _build_belief_state(visible_daily_desc: list) -> dict[str, float]:
    latest_first = list(visible_daily_desc)
    total_days = len(latest_first)
    if total_days == 0:
        return {
            "visible_day_count": 0.0,
            "critical_bp_days": 0.0,
            "high_bp_days": 0.0,
            "low_kick_days": 0.0,
            "bleeding_days": 0.0,
            "abdominal_pain_days": 0.0,
            "dizziness_days": 0.0,
            "low_meal_days": 0.0,
        }

    return {
        "visible_day_count": float(total_days),
        "critical_bp_days": float(sum(1 for day in latest_first if day.bp_systolic >= 160 or day.bp_diastolic >= 110)),
        "high_bp_days": float(sum(1 for day in latest_first if day.bp_systolic >= 140 or day.bp_diastolic >= 90)),
        "low_kick_days": float(sum(1 for day in latest_first if day.kick_count is not None and day.kick_count < 3)),
        "bleeding_days": float(sum(1 for day in latest_first if day.symptom_bleeding)),
        "abdominal_pain_days": float(sum(1 for day in latest_first if day.symptom_abdominal_pain)),
        "dizziness_days": float(sum(1 for day in latest_first if day.symptom_dizziness)),
        "low_meal_days": float(sum(1 for day in latest_first if day.meals_count < 2)),
    }


def _annotate_episode_observation(
    observation: Observation,
    *,
    episode_day_index: int,
    total_episode_days: int,
    belief_state: dict[str, float],
) -> Observation:
    annotated = observation.model_copy(deep=True)
    annotated.episode_day_index = episode_day_index
    annotated.total_episode_days = total_episode_days
    annotated.belief_state = belief_state
    return annotated


def _mask_observation(observation: Observation, withheld_signals: List[str]) -> Observation:
    signal_mask = _default_signal_mask()
    for signal in withheld_signals:
        if signal in signal_mask:
            signal_mask[signal] = False

    masked = observation.model_copy(deep=True)
    if not signal_mask["risk_flags"]:
        masked.risk_flags = []
    if not signal_mask["bp_trend"]:
        masked.bp_trend = "unknown"
    if not signal_mask["avg_kick_count"]:
        masked.avg_kick_count = None
    if not signal_mask["avg_meals"]:
        masked.avg_meals = None
    if not signal_mask["avg_sleep"]:
        masked.avg_sleep = None
    if not signal_mask["latest_weight_kg"]:
        masked.latest_weight_kg = None
    if not signal_mask["latest_energy"]:
        masked.latest_energy = None
    if not signal_mask["latest_breathlessness"]:
        masked.latest_breathlessness = None

    masked.signal_mask = signal_mask
    masked.withheld_signals = sorted(withheld_signals)
    masked.available_signals = [
        signal_name for signal_name, is_visible in signal_mask.items() if is_visible
    ]
    return masked


def observation_to_prompt(obs: Observation, text_observation: str) -> PromptObservation:
    system_prompt = (
        "You are an obstetric triage agent operating inside OpenEnv. "
        "Return a JSON object with keys: action_type, signal_name, condition, urgency, rationale. "
        f"Valid conditions: {', '.join(SAFE_CONDITIONS)}. "
        f"Valid urgencies: {', '.join(URGENCY_ORDER)}. "
        "Always prioritize patient safety and do not under-escalate danger flags. "
        "Some clinically useful signals may be intentionally withheld each episode. "
        "Use action_type='request_signal' to reveal one withheld signal before a final diagnosis."
    )
    user_prompt = (
        text_observation
        + "\n\nSome signals may be withheld for this episode. "
        "Reason only from visible information and avoid hallucinating hidden values."
        + f"\nCurrently withheld signal names: {', '.join(obs.withheld_signals) if obs.withheld_signals else 'none'}."
        + "\nIf you need one more hidden signal, request it explicitly."
        + "\nIf you are ready to finish, use action_type='diagnose'."
    )
    response_format = json.dumps(
        {
            "action_type": "one of: assess, request_signal, diagnose",
            "signal_name": "withheld signal name when action_type=request_signal, otherwise null",
            "condition": "one of the valid condition labels when action_type=diagnose, otherwise null",
            "urgency": "one of the valid urgency labels when action_type=diagnose, otherwise null",
            "rationale": "short clinical explanation",
        },
        indent=2,
    )
    return PromptObservation(
        observation=obs,
        text_observation=text_observation,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=response_format,
        valid_conditions=SAFE_CONDITIONS,
        valid_urgencies=URGENCY_ORDER,
    )


def parse_llm_output(raw_output: str) -> ActionModel:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM output must be valid JSON.") from exc

    return ActionModel(
        condition=parsed.get("condition") or parsed.get("target"),
        urgency=parsed.get("urgency"),
        rationale=parsed.get("rationale"),
        action_type=parsed.get("action_type"),
        signal_name=parsed.get("signal_name"),
        target=parsed.get("target"),
    )


def _classify_condition(obs: Observation) -> tuple[str, str]:
    features = featurize(obs)
    condition = infer_reference_condition(obs)
    supporting = supporting_features(condition, features)
    rationale = f"Reward policy reference condition={condition}. Supporting features: {', '.join(supporting)}."
    return condition, rationale


class PrenatalEnvironment(OpenEnvEnvironment):
    """OpenEnv-friendly maternal-health environment with prompt translation."""

    env_name = "prenatal_health_openenv"
    env_version = "2.0.1"
    openenv_available = OPENENV_AVAILABLE

    def __init__(self):
        self.current_user_id: Optional[int] = None
        self.current_obs: Optional[Observation] = None
        self.current_full_obs: Optional[Observation] = None
        self.current_prompt: Optional[PromptObservation] = None
        self.current_text_observation: Optional[str] = None
        self.current_user: Optional[UserProfile] = None
        self.current_daily: list = []
        self.current_visible_daily: list = []
        self.current_episode_timeline: list = []
        self.current_visible_day_count: int = 0
        self.episode_withheld_signals: List[str] = []
        self.current_checkin3: Optional[Checkin3Day] = None
        self.episode_done: bool = False
        self.reference_condition: Optional[str] = None

    def _refresh_visible_prompt(self) -> None:
        if self.current_obs is None:
            return
        self.current_text_observation = self.get_text_observation(self.current_obs)
        self.current_prompt = observation_to_prompt(self.current_obs, self.current_text_observation)

    def _rebuild_visible_episode_state(self) -> None:
        if self.current_user is None:
            raise RuntimeError("No active user for episode state rebuild.")
        if not self.current_episode_timeline:
            raise RuntimeError("No episode timeline available for state rebuild.")

        visible_slice_asc = self.current_episode_timeline[: self.current_visible_day_count]
        visible_slice_desc = list(reversed(visible_slice_asc))
        self.current_visible_daily = visible_slice_desc
        visible_checkin3 = self.current_checkin3 if self.current_visible_day_count >= len(self.current_episode_timeline) else None
        visible_obs = _build_observation(self.current_user, visible_slice_desc, visible_checkin3)
        visible_obs = _annotate_episode_observation(
            visible_obs,
            episode_day_index=max(self.current_visible_day_count, 1),
            total_episode_days=len(self.current_episode_timeline),
            belief_state=_build_belief_state(visible_slice_desc),
        )
        self.current_obs = _mask_observation(visible_obs, self.episode_withheld_signals)
        self._refresh_visible_prompt()

    def get_text_observation(self, observation: Observation) -> str:
        signal_mask = observation.signal_mask or _default_signal_mask()

        def render_value(signal_name: str, value, *, none_label: str = "unknown", formatter=None) -> str:
            if not signal_mask.get(signal_name, True):
                return "withheld"
            if value is None:
                return none_label
            return formatter(value) if formatter else str(value)

        latest_daily = self.current_visible_daily[0] if self.current_visible_daily else None
        latest_bp = render_value(
            "latest_blood_pressure",
            latest_daily,
            formatter=lambda day: f"{day.bp_systolic}/{day.bp_diastolic} mmHg",
        )
        latest_kicks = render_value(
            "latest_kick_count",
            latest_daily.kick_count if latest_daily else None,
        )
        if signal_mask.get("latest_symptoms", True):
            latest_bleeding = "yes" if latest_daily and latest_daily.symptom_bleeding else "no"
            latest_headache = "yes" if latest_daily and latest_daily.symptom_headache else "no"
            latest_swelling = "yes" if latest_daily and latest_daily.symptom_swelling else "no"
            latest_vision = "yes" if latest_daily and latest_daily.symptom_blurred_vision else "no"
            latest_abdominal_pain = "yes" if latest_daily and latest_daily.symptom_abdominal_pain else "no"
            latest_dizziness = "yes" if latest_daily and latest_daily.symptom_dizziness else "no"
        else:
            latest_bleeding = "withheld"
            latest_headache = "withheld"
            latest_swelling = "withheld"
            latest_vision = "withheld"
            latest_abdominal_pain = "withheld"
            latest_dizziness = "withheld"

        profile_name = self.current_user.name if self.current_user else f"Patient {observation.user_id}"
        belief_text = ", ".join(
            f"{key}={int(value) if float(value).is_integer() else round(value, 2)}"
            for key, value in observation.belief_state.items()
        ) if observation.belief_state else "none"
        return (
            "Patient profile:\n"
            f"- Name: {profile_name}\n"
            f"- User ID: {observation.user_id}\n"
            f"- Region: {observation.region}\n"
            f"- Weeks pregnant: {observation.weeks_pregnant}\n"
            f"- Trimester: {observation.trimester}\n"
            f"- History flags: {', '.join(observation.history_flags) if observation.history_flags else 'none'}\n\n"
            "Episode progress:\n"
            f"- Visible day: {observation.episode_day_index}/{observation.total_episode_days}\n"
            f"- Carried belief state: {belief_text}\n\n"
            "Latest vitals and symptoms:\n"
            f"- Latest blood pressure: {latest_bp}\n"
            f"- Latest kick count: {latest_kicks}\n"
            f"- Bleeding: {latest_bleeding}\n"
            f"- Headache: {latest_headache}\n"
            f"- Swelling: {latest_swelling}\n"
            f"- Blurred vision: {latest_vision}\n"
            f"- Abdominal pain: {latest_abdominal_pain}\n"
            f"- Dizziness: {latest_dizziness}\n"
            f"- Latest weight (kg): {render_value('latest_weight_kg', observation.latest_weight_kg)}\n"
            f"- Latest energy (1-10): {render_value('latest_energy', observation.latest_energy)}\n"
            f"- Latest breathlessness (1-10): {render_value('latest_breathlessness', observation.latest_breathlessness)}\n\n"
            "3-day summary:\n"
            f"- Blood-pressure trend: {render_value('bp_trend', observation.bp_trend, none_label='unknown')}\n"
            f"- Average kick count: {render_value('avg_kick_count', observation.avg_kick_count)}\n"
            f"- Average meals per day: {render_value('avg_meals', observation.avg_meals, formatter=lambda value: f'{value:.2f}')}\n"
            f"- Average sleep hours: {render_value('avg_sleep', observation.avg_sleep, formatter=lambda value: f'{value:.2f}')}\n"
            f"- Risk flags: {render_value('risk_flags', observation.risk_flags, none_label='none', formatter=lambda flags: ', '.join(flags) if flags else 'none')}\n"
            f"- Days of data: {observation.days_of_data}\n"
            f"- Available signals: {', '.join(observation.available_signals) if observation.available_signals else 'all'}\n"
            f"- Withheld signals: {', '.join(observation.withheld_signals) if observation.withheld_signals else 'none'}"
        )

    def reset(self, user_id: int) -> PromptObservation:
        user, daily, checkin3 = _load_recent_data(user_id, days=3)
        self.current_user_id = user_id
        self.current_user = user
        self.current_daily = daily
        self.current_visible_daily = daily
        self.current_episode_timeline = list(reversed(daily)) if daily else []
        self.current_visible_day_count = 1 if self.current_episode_timeline else 0
        self.current_checkin3 = checkin3
        full_observation = _build_observation(user, daily, checkin3)
        full_observation = _annotate_episode_observation(
            full_observation,
            episode_day_index=max(len(self.current_episode_timeline), 1),
            total_episode_days=max(len(self.current_episode_timeline), 1),
            belief_state=_build_belief_state(daily),
        )
        withheld_signals = random.sample(
            EPISODE_HIDEABLE_SIGNALS,
            k=random.randint(1, 2),
        )
        self.current_full_obs = full_observation
        self.episode_withheld_signals = withheld_signals
        if self.current_episode_timeline:
            self._rebuild_visible_episode_state()
        else:
            self.current_obs = _mask_observation(full_observation, withheld_signals)
            self._refresh_visible_prompt()
        self.episode_done = False
        self.reference_condition = infer_reference_condition(self.current_full_obs)
        self._signals_requested = 0
        return self.current_prompt

    def step(self, action: ActionModel) -> StepResult:
        if self.episode_done:
            raise RuntimeError("Episode done. Call reset() first.")
        if self.current_obs is None or self.current_prompt is None:
            raise RuntimeError("No active episode. Call reset() first.")

        action_mode = action.action_type or "diagnose"
        chosen_condition = action.condition or action.target

        if action_mode == "assess":
            advanced = False
            if self.current_episode_timeline and self.current_visible_day_count < len(self.current_episode_timeline):
                self.current_visible_day_count += 1
                self._rebuild_visible_episode_state()
                advanced = True
            return StepResult(
                observation=self.current_obs,
                text_observation=self.current_text_observation or "",
                prompt=self.current_prompt,
                reward=0.0,
                reward_components={
                    "condition_score": 0.0,
                    "urgency_score": 0.0,
                    "under_escalation_penalty": 0.0,
                    "danger_override_penalty": 0.0,
                    "data_recency_bonus": 0.0,
                    "total_reward": 0.0,
                },
                done=False,
                predicted_condition=None,
                urgency=None,
                diet_advice=[],
                rationale=(
                    f"Assessment step requested; observation returned for reasoning. "
                    f"{'Episode advanced to the next visible day slice.' if advanced else 'No later day slice remains; reassess current belief.'}"
                ),
                reference_condition=None,
                reference_urgency=None,
                latent_risks={},
            )

        if action_mode == "request_signal":
            hidden_signals = list(self.current_obs.withheld_signals)
            if not hidden_signals:
                return StepResult(
                    observation=self.current_obs,
                    text_observation=self.current_text_observation or "",
                    prompt=self.current_prompt,
                    reward=-0.1,
                    reward_components={
                        "request_cost": -0.1,
                        "revealed_signal": None,
                        "remaining_hidden_signals": 0,
                        "total_reward": -0.1,
                    },
                    done=False,
                    predicted_condition=None,
                    urgency=None,
                    diet_advice=[],
                    rationale="No hidden signals remain. Proceed with assessment or diagnosis.",
                    reference_condition=None,
                    reference_urgency=None,
                    latent_risks={},
                )

            requested_signal = action.signal_name or hidden_signals[0]
            if requested_signal not in EPISODE_HIDEABLE_SIGNALS:
                return StepResult(
                    observation=self.current_obs,
                    text_observation=self.current_text_observation or "",
                    prompt=self.current_prompt,
                    reward=-0.2,
                    reward_components={
                        "request_cost": -0.2,
                        "revealed_signal": None,
                        "remaining_hidden_signals": len(hidden_signals),
                        "total_reward": -0.2,
                    },
                    done=False,
                    predicted_condition=None,
                    urgency=None,
                    diet_advice=[],
                    rationale=f"Signal '{requested_signal}' is not a valid requestable signal.",
                    reference_condition=None,
                    reference_urgency=None,
                    latent_risks={},
                )

            if requested_signal not in hidden_signals:
                return StepResult(
                    observation=self.current_obs,
                    text_observation=self.current_text_observation or "",
                    prompt=self.current_prompt,
                    reward=-0.1,
                    reward_components={
                        "request_cost": -0.1,
                        "revealed_signal": None,
                        "remaining_hidden_signals": len(hidden_signals),
                        "total_reward": -0.1,
                    },
                    done=False,
                    predicted_condition=None,
                    urgency=None,
                    diet_advice=[],
                    rationale=f"Signal '{requested_signal}' is already visible or was not withheld this episode.",
                    reference_condition=None,
                    reference_urgency=None,
                    latent_risks={},
                )

            remaining_hidden = [signal for signal in hidden_signals if signal != requested_signal]
            if self.current_full_obs is None:
                raise RuntimeError("Full observation missing for request_signal flow.")
            visible_slice_asc = self.current_episode_timeline[: self.current_visible_day_count]
            visible_slice_desc = list(reversed(visible_slice_asc))
            visible_checkin3 = self.current_checkin3 if self.current_visible_day_count >= len(self.current_episode_timeline) else None
            visible_obs = _build_observation(self.current_user, visible_slice_desc, visible_checkin3)
            visible_obs = _annotate_episode_observation(
                visible_obs,
                episode_day_index=max(self.current_visible_day_count, 1),
                total_episode_days=max(len(self.current_episode_timeline), 1),
                belief_state=_build_belief_state(visible_slice_desc),
            )
            self.episode_withheld_signals = remaining_hidden
            self.current_obs = _mask_observation(visible_obs, remaining_hidden)
            self._refresh_visible_prompt()
            return StepResult(
                observation=self.current_obs,
                text_observation=self.current_text_observation or "",
                prompt=self.current_prompt,
                reward=-0.25,
                reward_components={
                    "request_cost": -0.25,
                    "revealed_signal": requested_signal,
                    "remaining_hidden_signals": len(remaining_hidden),
                    "total_reward": -0.25,
                },
                done=False,
                predicted_condition=None,
                urgency=None,
                diet_advice=[],
                rationale=(
                    f"Signal '{requested_signal}' has been revealed. "
                    "Reassess the patient before issuing a final diagnosis."
                ),
                reference_condition=None,
                reference_urgency=None,
                latent_risks={},
            )

        if action_mode != "diagnose":
            raise ValueError(f"Unknown action_type: {action_mode}. Use 'assess', 'request_signal', or 'diagnose'.")
        if chosen_condition not in SAFE_CONDITIONS:
            raise ValueError(f"Unknown condition: {chosen_condition}. Valid: {SAFE_CONDITIONS}")
        if action.urgency not in URGENCY_ORDER:
            raise ValueError(f"Unknown urgency: {action.urgency}. Valid: {URGENCY_ORDER}")

        reference_observation = self.current_full_obs or self.current_obs
        breakdown = calculate_reward(chosen_condition, action.urgency, reference_observation)
        self.episode_done = True
        latent_advice = [
            LATENT_CONDITION_ADVICE[name]
            for name in breakdown.latent_risks
            if name in LATENT_CONDITION_ADVICE
        ]
        diet = DIET_ADVICE.get(chosen_condition, DIET_ADVICE["low_risk"]) + latent_advice[:2]

        return StepResult(
            observation=self.current_obs,
            text_observation=self.current_text_observation or "",
            prompt=self.current_prompt,
            reward=breakdown.reward,
            reward_components=breakdown.reward_components,
            done=True,
            predicted_condition=chosen_condition,
            urgency=action.urgency,
            diet_advice=diet,
            rationale=action.rationale or breakdown.rationale,
            reference_condition=breakdown.reference_condition,
            reference_urgency=breakdown.reference_urgency,
            latent_risks=breakdown.latent_risks,
            under_escalated=breakdown.under_escalated,
        )

    def state(self) -> dict:
        if self.current_obs is None or self.current_prompt is None:
            return {"status": "no_active_episode"}
        return {
            "user_id": self.current_user_id,
            "episode_done": self.episode_done,
            "observation": self.current_obs.model_dump(),
            "text_observation": self.current_text_observation,
            "prompt": self.current_prompt.model_dump(),
            "reference_condition": self.reference_condition if self.episode_done else None,
            "valid_actions": [
                {"action_type": "assess"},
                *[
                    {"action_type": "request_signal", "signal_name": signal_name}
                    for signal_name in self.current_obs.withheld_signals
                ],
                *[
                    {
                        "action_type": "diagnose",
                        "condition": condition,
                        "urgency": urgency,
                        "rationale": "clinical explanation",
                    }
                    for condition in SAFE_CONDITIONS
                    for urgency in URGENCY_ORDER
                ],
            ],
        }


@dataclass
class TrajectoryDay:
    bp_systolic: int
    bp_diastolic: int
    kick_count: int
    symptoms: dict[str, bool]
    energy_level: int
    meals_count: int
    sleep_hours: float


@dataclass
class PatientTrajectory:
    trajectory_id: str
    weeks_pregnant: int
    trimester: int
    region: str
    history_flags: list[str]
    target_condition: str
    target_urgency: str
    days: list[TrajectoryDay]

    def __repr__(self) -> str:
        day_summaries = []
        for index, day in enumerate(self.days, start=1):
            active_symptoms = [name for name, is_active in day.symptoms.items() if is_active]
            symptom_text = ", ".join(active_symptoms) if active_symptoms else "none"
            day_summaries.append(
                f"Day {index}: BP={day.bp_systolic}/{day.bp_diastolic}, "
                f"kicks={day.kick_count}, energy={day.energy_level}/10, "
                f"meals={day.meals_count}, sleep={day.sleep_hours}, symptoms={symptom_text}"
            )
        joined = " | ".join(day_summaries)
        return (
            f"PatientTrajectory<{self.trajectory_id}, target={self.target_condition}/{self.target_urgency}, "
            f"weeks={self.weeks_pregnant}, region={self.region}; {joined}>"
        )


def _symptoms(
    *,
    headache: bool = False,
    swelling: bool = False,
    bleeding: bool = False,
    blurred_vision: bool = False,
    breathlessness: bool = False,
    cramps: bool = False,
    dizziness: bool = False,
) -> dict[str, bool]:
    return {
        "headache": headache,
        "swelling": swelling,
        "bleeding": bleeding,
        "blurred_vision": blurred_vision,
        "breathlessness": breathlessness,
        "cramps": cramps,
        "dizziness": dizziness,
    }


MULTITURN_TRAJECTORIES: dict[str, PatientTrajectory] = {
    "traj_preeclampsia_slow": PatientTrajectory(
        trajectory_id="traj_preeclampsia_slow",
        weeks_pregnant=34,
        trimester=3,
        region="Bihar",
        history_flags=["family_hypertension"],
        target_condition="preeclampsia",
        target_urgency="go_to_hospital_today",
        days=[
            TrajectoryDay(138, 88, 8, _symptoms(), 5, 3, 6.5),
            TrajectoryDay(145, 95, 7, _symptoms(headache=True), 4, 3, 6.0),
            TrajectoryDay(162, 108, 6, _symptoms(headache=True, blurred_vision=True, swelling=True), 3, 2, 5.0),
        ],
    ),
    "traj_fetal_distress_sudden": PatientTrajectory(
        trajectory_id="traj_fetal_distress_sudden",
        weeks_pregnant=36,
        trimester=3,
        region="Odisha",
        history_flags=[],
        target_condition="fetal_distress",
        target_urgency="go_to_hospital_today",
        days=[
            TrajectoryDay(122, 80, 9, _symptoms(), 6, 3, 7.0),
            TrajectoryDay(124, 82, 8, _symptoms(), 6, 3, 7.0),
            TrajectoryDay(126, 82, 2, _symptoms(), 5, 3, 6.0),
        ],
    ),
    "traj_gestational_diabetes_noisy": PatientTrajectory(
        trajectory_id="traj_gestational_diabetes_noisy",
        weeks_pregnant=31,
        trimester=3,
        region="Madhya Pradesh",
        history_flags=["family_diabetes"],
        target_condition="gestational_diabetes",
        target_urgency="visit_phc_this_week",
        days=[
            TrajectoryDay(128, 84, 8, _symptoms(breathlessness=True), 4, 3, 6.0),
            TrajectoryDay(130, 86, 7, _symptoms(), 3, 3, 5.5),
            TrajectoryDay(132, 86, 8, _symptoms(breathlessness=True, dizziness=True), 4, 2, 5.0),
        ],
    ),
    "traj_anemia_gradual": PatientTrajectory(
        trajectory_id="traj_anemia_gradual",
        weeks_pregnant=27,
        trimester=3,
        region="Jharkhand",
        history_flags=[],
        target_condition="anemia",
        target_urgency="visit_phc_this_week",
        days=[
            TrajectoryDay(118, 76, 9, _symptoms(dizziness=True), 5, 2, 6.0),
            TrajectoryDay(120, 78, 9, _symptoms(dizziness=True, breathlessness=True), 4, 2, 5.5),
            TrajectoryDay(120, 78, 8, _symptoms(dizziness=True, breathlessness=True), 3, 1, 5.0),
        ],
    ),
    "traj_preterm_subtle": PatientTrajectory(
        trajectory_id="traj_preterm_subtle",
        weeks_pregnant=29,
        trimester=3,
        region="Chhattisgarh",
        history_flags=["prev_complication"],
        target_condition="preterm_risk",
        target_urgency="visit_phc_this_week",
        days=[
            TrajectoryDay(118, 76, 8, _symptoms(cramps=True), 5, 3, 6.5),
            TrajectoryDay(120, 78, 8, _symptoms(cramps=True), 5, 2, 6.0),
            TrajectoryDay(122, 80, 7, _symptoms(cramps=True, dizziness=True), 4, 2, 5.5),
        ],
    ),
    "traj_low_risk_reassuring": PatientTrajectory(
        trajectory_id="traj_low_risk_reassuring",
        weeks_pregnant=24,
        trimester=2,
        region="Rajasthan",
        history_flags=[],
        target_condition="low_risk",
        target_urgency="monitor_at_home",
        days=[
            TrajectoryDay(118, 76, 10, _symptoms(), 7, 3, 7.5),
            TrajectoryDay(120, 78, 10, _symptoms(), 7, 3, 7.0),
            TrajectoryDay(118, 76, 11, _symptoms(), 8, 3, 7.5),
        ],
    ),
    "traj_mixed_signals_hard": PatientTrajectory(
        trajectory_id="traj_mixed_signals_hard",
        weeks_pregnant=33,
        trimester=3,
        region="Uttar Pradesh",
        history_flags=["family_diabetes"],
        target_condition="fetal_distress",
        target_urgency="go_to_hospital_today",
        days=[
            TrajectoryDay(142, 92, 6, _symptoms(breathlessness=True), 4, 2, 5.5),
            TrajectoryDay(144, 94, 5, _symptoms(headache=True, breathlessness=True), 4, 2, 5.0),
            TrajectoryDay(146, 96, 2, _symptoms(headache=True, blurred_vision=True, breathlessness=True), 3, 2, 4.5),
        ],
    ),
    "traj_preeclampsia_fast": PatientTrajectory(
        trajectory_id="traj_preeclampsia_fast",
        weeks_pregnant=35,
        trimester=3,
        region="Maharashtra",
        history_flags=["prev_preeclampsia", "family_hypertension"],
        target_condition="preeclampsia",
        target_urgency="go_to_hospital_today",
        days=[
            TrajectoryDay(126, 82, 8, _symptoms(), 5, 3, 6.0),
            TrajectoryDay(160, 112, 6, _symptoms(headache=True, swelling=True, blurred_vision=True), 3, 2, 5.0),
            TrajectoryDay(166, 114, 5, _symptoms(headache=True, swelling=True, blurred_vision=True, breathlessness=True), 2, 2, 4.0),
        ],
    ),
}


class MultiTurnPrenatalEnvironment(OpenEnvEnvironment):
    """Three-day partially observable maternal triage environment."""

    env_name = "prenatal_health_multiturn_openenv"
    env_version = "1.0.0"
    max_steps = 8

    def __init__(self) -> None:
        self.current_trajectory: Optional[PatientTrajectory] = None
        self.current_day: int = 1
        self.cumulative_reward: float = 0.0
        self.step_count: int = 0
        self.done: bool = False
        self.step_logs: list[dict[str, Any]] = []
        self.bp_rechecks: set[int] = set()
        self.kick_requests: set[int] = set()
        self.last_prompt: Optional[PromptObservation] = None
        self.last_observation: Optional[Observation] = None
        self.last_text_observation: str = ""

    def _random_trajectory_id(self) -> str:
        return random.choice(list(MULTITURN_TRAJECTORIES.keys()))

    def reset(self, trajectory_id: Optional[str] = None) -> PromptObservation:
        chosen_id = trajectory_id or self._random_trajectory_id()
        if chosen_id not in MULTITURN_TRAJECTORIES:
            raise ValueError(f"Unknown trajectory_id: {chosen_id}")

        self.current_trajectory = MULTITURN_TRAJECTORIES[chosen_id]
        self.current_day = 1
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.done = False
        self.step_logs = []
        self.bp_rechecks = set()
        self.kick_requests = set()
        prompt = self._build_prompt()
        self.last_prompt = prompt
        return prompt

    def _current_day_state(self) -> TrajectoryDay:
        if self.current_trajectory is None:
            raise RuntimeError("No active trajectory. Call reset() first.")
        return self.current_trajectory.days[self.current_day - 1]

    def _visible_symptoms(self) -> dict[str, bool]:
        if self.current_day == 1:
            return {}
        return dict(self._current_day_state().symptoms)

    def _visible_history(self) -> list[str]:
        if self.current_day < 3 or self.current_trajectory is None:
            return []
        return list(self.current_trajectory.history_flags)

    def _bp_trend(self) -> str:
        if self.current_trajectory is None or self.current_day == 1:
            return "stable"
        visible_days = self.current_trajectory.days[: self.current_day]
        first = visible_days[0].bp_systolic
        latest = visible_days[-1].bp_systolic
        if latest >= first + 10:
            return "rising"
        if latest <= first - 10:
            return "falling"
        return "stable"

    def _risk_flags(self) -> list[str]:
        day = self._current_day_state()
        flags: list[str] = []
        if day.bp_systolic >= 160 or day.bp_diastolic >= 110:
            flags.append("DANGER_BP_CRITICAL")
        elif day.bp_systolic >= 140 or day.bp_diastolic >= 90:
            flags.append("HIGH_BP")

        if day.kick_count < 3:
            flags.append("DANGER_LOW_KICKS")
        elif day.kick_count < 6:
            flags.append("LOW_KICK_AVG")

        symptoms = self._visible_symptoms()
        if symptoms.get("bleeding"):
            flags.append("DANGER_BLEEDING")
        if symptoms.get("headache") and symptoms.get("swelling"):
            flags.append("HIGH_PREECLAMPSIA_SIGNAL")
        if symptoms.get("headache") and symptoms.get("blurred_vision"):
            flags.append("DANGER_VISION_HEADACHE")
        if symptoms.get("cramps"):
            flags.append("ABDOMINAL_PAIN_SIGNAL")
        if symptoms.get("dizziness"):
            flags.append("DIZZINESS_SIGNAL")
        if self.current_day >= 2 and self._bp_trend() == "rising":
            flags.append("BP_RISING_TREND")
        if self.current_day == 3:
            avg_meals = self._avg_meals()
            if avg_meals is not None and avg_meals < 2:
                flags.append("LOW_NUTRITION")
        return flags

    def _avg_kicks(self) -> Optional[float]:
        if self.current_trajectory is None:
            return None
        visible = self.current_trajectory.days[: self.current_day]
        return round(sum(day.kick_count for day in visible) / len(visible), 2)

    def _avg_meals(self) -> Optional[float]:
        if self.current_trajectory is None or self.current_day < 3:
            return None
        visible = self.current_trajectory.days[: self.current_day]
        return round(sum(day.meals_count for day in visible) / len(visible), 2)

    def _avg_sleep(self) -> Optional[float]:
        if self.current_trajectory is None or self.current_day < 3:
            return None
        visible = self.current_trajectory.days[: self.current_day]
        return round(sum(day.sleep_hours for day in visible) / len(visible), 2)

    def _build_observation(self) -> Observation:
        if self.current_trajectory is None:
            raise RuntimeError("No active trajectory. Call reset() first.")
        day = self._current_day_state()
        observation = Observation(
            user_id=0,
            weeks_pregnant=self.current_trajectory.weeks_pregnant,
            trimester=self.current_trajectory.trimester,
            region=self.current_trajectory.region,
            risk_flags=self._risk_flags(),
            bp_trend=self._bp_trend(),
            avg_kick_count=self._avg_kicks(),
            avg_meals=self._avg_meals(),
            avg_sleep=self._avg_sleep(),
            latest_weight_kg=None,
            latest_energy=day.energy_level if self.current_day == 3 else None,
            latest_breathlessness=8 if day.symptoms.get("breathlessness") and self.current_day == 3 else None,
            history_flags=self._visible_history(),
            days_of_data=self.current_day,
            episode_day_index=self.current_day,
            total_episode_days=len(self.current_trajectory.days),
            belief_state={
                "visible_day_count": float(self.current_day),
                "danger_flag_count": float(sum(1 for flag in self._risk_flags() if flag.startswith("DANGER"))),
            },
            available_signals=self._available_signal_names(),
            withheld_signals=self._withheld_signal_names(),
            signal_mask=_default_signal_mask(),
        )
        self.last_observation = observation
        return observation

    def _available_signal_names(self) -> list[str]:
        available = ["latest_blood_pressure", "latest_kick_count"]
        if self.current_day >= 2:
            available.append("latest_symptoms")
        if self.current_day >= 3:
            available.extend(["history_flags", "avg_meals", "avg_sleep", "latest_energy"])
        return available

    def _withheld_signal_names(self) -> list[str]:
        withheld = []
        if self.current_day == 1:
            withheld.extend(["latest_symptoms", "history_flags", "avg_meals", "avg_sleep", "latest_energy"])
        elif self.current_day == 2:
            withheld.extend(["history_flags", "avg_meals", "avg_sleep", "latest_energy"])
        return withheld

    def _build_text_observation(self) -> str:
        if self.current_trajectory is None:
            raise RuntimeError("No active trajectory. Call reset() first.")
        day = self._current_day_state()
        lines = [
            f"Trajectory: {self.current_trajectory.trajectory_id}",
            f"Day {self.current_day} of 3",
            f"Region: {self.current_trajectory.region}",
            f"Weeks pregnant: {self.current_trajectory.weeks_pregnant} (Trimester {self.current_trajectory.trimester})",
            "",
            "Visible information:",
            f"- Today's blood pressure: {day.bp_systolic}/{day.bp_diastolic} mmHg",
            f"- Today's kick count: {day.kick_count}",
            f"- BP trend across visible days: {self._bp_trend()}",
            f"- Visible risk flags: {', '.join(self._risk_flags()) if self._risk_flags() else 'none'}",
        ]
        if self.current_day >= 2:
            symptoms = self._visible_symptoms()
            symptom_text = ", ".join(f"{name}={value}" for name, value in symptoms.items())
            lines.append(f"- Symptom flags: {symptom_text or 'none'}")
        else:
            lines.append("- Symptom flags: hidden until day 2")

        if self.current_day >= 3:
            lines.extend(
                [
                    f"- History flags: {', '.join(self.current_trajectory.history_flags) if self.current_trajectory.history_flags else 'none'}",
                    f"- Average meals across visible days: {self._avg_meals()}",
                    f"- Average sleep across visible days: {self._avg_sleep()}",
                    f"- Current energy level: {day.energy_level}/10",
                ]
            )
        else:
            lines.append("- History flags: hidden until day 3")

        if self.current_day in self.bp_rechecks:
            lines.append(f"- BP recheck confirmed at {day.bp_systolic}/{day.bp_diastolic} mmHg")
        if self.current_day in self.kick_requests:
            lines.append(f"- Kick count rechecked at {day.kick_count}")
        self.last_text_observation = "\n".join(lines)
        return self.last_text_observation

    def _build_prompt(self) -> PromptObservation:
        observation = self._build_observation()
        text_observation = self._build_text_observation()
        system_prompt = (
            "You are MAAS, a maternal triage assistant operating in a multi-turn OpenEnv episode. "
            "You may gather more evidence over time before making a final diagnosis. "
            "Return only JSON with keys: action_type, target, urgency, rationale. "
            "Valid action_type values: request_bp_recheck, request_kick_count, advance_day, refer_to_phc, diagnose. "
            f"Valid conditions: {', '.join(SAFE_CONDITIONS)}. "
            f"Valid urgencies: {', '.join(URGENCY_ORDER)}. "
            "If danger signs appear, prioritize safety and do not under-escalate."
        )
        user_prompt = (
            text_observation
            + "\n\nAvailable actions now: "
            + ", ".join(action["action_type"] for action in self._valid_actions())
            + "\nReturn exactly one JSON action for the current step."
        )
        response_format = json.dumps(
            {
                "action_type": "request_bp_recheck | request_kick_count | advance_day | refer_to_phc | diagnose",
                "target": "condition label when diagnosing, otherwise null",
                "urgency": "urgency label when diagnosing, otherwise null",
                "rationale": "short clinical explanation",
            },
            indent=2,
        )
        prompt = PromptObservation(
            observation=observation,
            text_observation=text_observation,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=response_format,
            valid_conditions=SAFE_CONDITIONS,
            valid_urgencies=URGENCY_ORDER,
        )
        self.last_prompt = prompt
        return prompt

    def _valid_actions(self) -> list[dict[str, Any]]:
        if self.done:
            return []

        actions: list[dict[str, Any]] = []
        if self.current_day not in self.bp_rechecks:
            actions.append({"action_type": "request_bp_recheck"})
        if self.current_day not in self.kick_requests:
            actions.append({"action_type": "request_kick_count"})
        if self.current_day < 3:
            actions.append({"action_type": "advance_day"})
        actions.append({"action_type": "refer_to_phc"})
        actions.append({"action_type": "diagnose", "target": "<condition>", "urgency": "<urgency>"})
        return actions

    def _danger_present(self) -> bool:
        return any(flag.startswith("DANGER") for flag in self._risk_flags())

    def _log_step(self, action_dict: dict[str, Any], observation_revealed: str, step_reward: float, done: bool) -> dict[str, Any]:
        self.cumulative_reward += step_reward
        step_log = {
            "day": self.current_day,
            "action": action_dict,
            "observation_revealed": observation_revealed,
            "step_reward": round(step_reward, 4),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "done": done,
        }
        self.step_logs.append(step_log)
        return step_log

    def _normalize_final_reward(self, raw_reward: float) -> float:
        normalized = (raw_reward + 1.3) / 3.3
        return max(0.0, min(1.0, normalized))

    def _final_reward_components(self, condition: str, urgency: str) -> dict[str, Any]:
        if self.current_trajectory is None:
            raise RuntimeError("No active trajectory.")
        condition_score = 1.0 if condition == self.current_trajectory.target_condition else 0.0
        urgency_score = 0.5 if urgency == self.current_trajectory.target_urgency else 0.0
        danger_present = self._danger_present()
        if danger_present and urgency == "go_to_hospital_today":
            safety_score = 0.3
        elif danger_present and urgency != "go_to_hospital_today":
            safety_score = -1.0
        else:
            safety_score = 0.0
        efficiency_bonus = (
            0.2
            if condition_score == 1.0 and urgency_score == 0.5 and self.current_day <= 2
            else 0.0
        )
        over_escalation_penalty = (
            -0.3
            if self.current_trajectory.target_condition == "low_risk" and urgency == "go_to_hospital_today"
            else 0.0
        )
        raw_reward = (
            condition_score
            + urgency_score
            + safety_score
            + efficiency_bonus
            + over_escalation_penalty
        )
        return {
            "condition_score": condition_score,
            "urgency_score": urgency_score,
            "safety_score": safety_score,
            "efficiency_bonus": efficiency_bonus,
            "over_escalation_penalty": over_escalation_penalty,
            "raw_reward": raw_reward,
            "total_reward": self._normalize_final_reward(raw_reward),
        }

    def step(self, action: ActionModel | dict[str, Any]) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is complete. Call reset() first.")
        if self.current_trajectory is None:
            raise RuntimeError("No active trajectory. Call reset() first.")

        action_model = action if isinstance(action, ActionModel) else ActionModel(**action)
        action_type = action_model.action_type or "diagnose"
        action_dict = {
            "action_type": action_type,
            "target": action_model.condition or action_model.target,
            "urgency": action_model.urgency,
            "rationale": action_model.rationale,
        }

        self.step_count += 1
        if self.step_count > self.max_steps:
            self.done = True
            step_log = self._log_step(action_dict, "episode terminated: max steps reached", -0.2, True)
            prompt = self._build_prompt()
            return StepResult(
                observation=prompt.observation,
                text_observation=prompt.text_observation,
                prompt=prompt,
                reward=-0.2,
                reward_components={"reason": "max_steps_reached", "step_log": step_log},
                done=True,
                predicted_condition=None,
                urgency=None,
                diet_advice=[],
                rationale="Episode ended because max steps were exceeded.",
                reference_condition=self.current_trajectory.target_condition,
                reference_urgency=self.current_trajectory.target_urgency,
                latent_risks={},
            )

        if action_type == "request_bp_recheck":
            day = self._current_day_state()
            self.bp_rechecks.add(self.current_day)
            reward = -0.05
            revealed = f"BP recheck: {day.bp_systolic}/{day.bp_diastolic} mmHg"
            step_log = self._log_step(action_dict, revealed, reward, False)
            prompt = self._build_prompt()
            return StepResult(
                observation=prompt.observation,
                text_observation=prompt.text_observation,
                prompt=prompt,
                reward=reward,
                reward_components={"request_cost": reward, "step_log": step_log, "revealed_signal": "bp_recheck"},
                done=False,
                predicted_condition=None,
                urgency=None,
                diet_advice=[],
                rationale="Blood pressure was rechecked for the current day.",
                reference_condition=self.current_trajectory.target_condition,
                reference_urgency=self.current_trajectory.target_urgency,
                latent_risks={},
            )

        if action_type == "request_kick_count":
            day = self._current_day_state()
            self.kick_requests.add(self.current_day)
            reward = -0.05
            revealed = f"Kick count recheck: {day.kick_count}"
            step_log = self._log_step(action_dict, revealed, reward, False)
            prompt = self._build_prompt()
            return StepResult(
                observation=prompt.observation,
                text_observation=prompt.text_observation,
                prompt=prompt,
                reward=reward,
                reward_components={"request_cost": reward, "step_log": step_log, "revealed_signal": "kick_count"},
                done=False,
                predicted_condition=None,
                urgency=None,
                diet_advice=[],
                rationale="Kick count was rechecked for the current day.",
                reference_condition=self.current_trajectory.target_condition,
                reference_urgency=self.current_trajectory.target_urgency,
                latent_risks={},
            )

        if action_type == "advance_day":
            if self.current_day >= 3:
                reward = -0.1
                step_log = self._log_step(action_dict, "already at final day", reward, False)
                prompt = self._build_prompt()
                return StepResult(
                    observation=prompt.observation,
                    text_observation=prompt.text_observation,
                    prompt=prompt,
                    reward=reward,
                    reward_components={"advance_reward": reward, "step_log": step_log},
                    done=False,
                    predicted_condition=None,
                    urgency=None,
                    diet_advice=[],
                    rationale="Cannot advance beyond day 3.",
                    reference_condition=self.current_trajectory.target_condition,
                    reference_urgency=self.current_trajectory.target_urgency,
                    latent_risks={},
                )
            self.current_day += 1
            reward = 0.0
            step_log = self._log_step(action_dict, f"advanced to day {self.current_day}", reward, False)
            prompt = self._build_prompt()
            return StepResult(
                observation=prompt.observation,
                text_observation=prompt.text_observation,
                prompt=prompt,
                reward=reward,
                reward_components={"advance_reward": reward, "step_log": step_log},
                done=False,
                predicted_condition=None,
                urgency=None,
                diet_advice=[],
                rationale=f"Advanced to day {self.current_day}.",
                reference_condition=self.current_trajectory.target_condition,
                reference_urgency=self.current_trajectory.target_urgency,
                latent_risks={},
            )

        if action_type == "refer_to_phc":
            reference_urgency = self.current_trajectory.target_urgency
            reward = 0.3 if reference_urgency == "visit_phc_this_week" else -0.2
            step_log = self._log_step(action_dict, "PHC referral recorded", reward, False)
            prompt = self._build_prompt()
            return StepResult(
                observation=prompt.observation,
                text_observation=prompt.text_observation,
                prompt=prompt,
                reward=reward,
                reward_components={"intermediate_referral_reward": reward, "step_log": step_log},
                done=False,
                predicted_condition=None,
                urgency="visit_phc_this_week",
                diet_advice=[],
                rationale=(
                    "Referral timing was appropriate for the likely severity."
                    if reward > 0
                    else "Referral to PHC was too slow for a likely hospital-level emergency."
                ),
                reference_condition=self.current_trajectory.target_condition,
                reference_urgency=reference_urgency,
                latent_risks={},
            )

        if action_type != "diagnose":
            raise ValueError(f"Unknown action_type: {action_type}")

        target_condition = action_model.condition or action_model.target
        if target_condition not in SAFE_CONDITIONS:
            raise ValueError(f"Unknown condition: {target_condition}")
        if action_model.urgency not in URGENCY_ORDER:
            raise ValueError(f"Unknown urgency: {action_model.urgency}")

        components = self._final_reward_components(target_condition, action_model.urgency)
        final_reward = float(components["total_reward"])
        self.done = True
        step_log = self._log_step(action_dict, "final diagnosis submitted", final_reward, True)
        prompt = self._build_prompt()
        reward_components = dict(components)
        reward_components["step_log"] = step_log
        reward_components["episode_trace"] = list(self.step_logs)

        return StepResult(
            observation=prompt.observation,
            text_observation=prompt.text_observation,
            prompt=prompt,
            reward=final_reward,
            reward_components=reward_components,
            done=True,
            predicted_condition=target_condition,
            urgency=action_model.urgency,
            diet_advice=DIET_ADVICE.get(target_condition, []),
            rationale=action_model.rationale or "Final diagnosis submitted.",
            reference_condition=self.current_trajectory.target_condition,
            reference_urgency=self.current_trajectory.target_urgency,
            latent_risks={},
            under_escalated=self._danger_present() and action_model.urgency != "go_to_hospital_today",
        )

    def state(self) -> dict[str, Any]:
        prompt = self.last_prompt or self._build_prompt()
        return {
            "trajectory_id": self.current_trajectory.trajectory_id if self.current_trajectory else None,
            "current_day": self.current_day,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "done": self.done,
            "revealed_observation": prompt.observation.model_dump(),
            "text_observation": prompt.text_observation,
            "valid_actions": self._valid_actions(),
            "step_logs": list(self.step_logs),
        }
