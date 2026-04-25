from __future__ import annotations

import json
from typing import List, Optional
import random
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


class Observation(BaseModel):
    user_id: int
    weeks_pregnant: int
    trimester: int
    region: str
    risk_flags: List[str]
    bp_trend: str
    avg_kick_count: Optional[float]
    avg_meals: float
    avg_sleep: float
    latest_weight_kg: Optional[float]
    latest_energy: Optional[int]
    latest_breathlessness: Optional[int]
    history_flags: List[str]
    days_of_data: int
    masked_signals: List[str] = []


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


def observation_to_prompt(obs: Observation, text_observation: str) -> PromptObservation:
    system_prompt = (
        "You are an obstetric triage agent operating inside OpenEnv. "
        "Return a JSON object with keys: condition, urgency, rationale. "
        f"Valid conditions: {', '.join(SAFE_CONDITIONS)}. "
        f"Valid urgencies: {', '.join(URGENCY_ORDER)}. "
        "Always prioritize patient safety and do not under-escalate danger flags."
    )
    user_prompt = text_observation + "\n\nPredict the best condition label and urgency for this patient."
    response_format = json.dumps(
        {
            "condition": "one of the valid condition labels",
            "urgency": "one of the valid urgency labels",
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
        self.current_prompt: Optional[PromptObservation] = None
        self.current_text_observation: Optional[str] = None
        self.current_user: Optional[UserProfile] = None
        self.current_daily: list = []
        self.current_checkin3: Optional[Checkin3Day] = None
        self.episode_done: bool = False
        self.reference_condition: Optional[str] = None

    def get_text_observation(self, observation: Observation) -> str:
        latest_daily = self.current_daily[0] if self.current_daily else None
        latest_bp = (
            f"{latest_daily.bp_systolic}/{latest_daily.bp_diastolic} mmHg"
            if latest_daily
            else "unknown"
        )
        latest_kicks = latest_daily.kick_count if latest_daily and latest_daily.kick_count is not None else "unknown"
        latest_bleeding = "yes" if latest_daily and latest_daily.symptom_bleeding else "no"
        latest_headache = "yes" if latest_daily and latest_daily.symptom_headache else "no"
        latest_swelling = "yes" if latest_daily and latest_daily.symptom_swelling else "no"
        latest_vision = "yes" if latest_daily and latest_daily.symptom_blurred_vision else "no"
        latest_abdominal_pain = "yes" if latest_daily and latest_daily.symptom_abdominal_pain else "no"
        latest_dizziness = "yes" if latest_daily and latest_daily.symptom_dizziness else "no"

        profile_name = self.current_user.name if self.current_user else f"Patient {observation.user_id}"
        return (
            "Patient profile:\n"
            f"- Name: {profile_name}\n"
            f"- User ID: {observation.user_id}\n"
            f"- Region: {observation.region}\n"
            f"- Weeks pregnant: {observation.weeks_pregnant}\n"
            f"- Trimester: {observation.trimester}\n"
            f"- History flags: {', '.join(observation.history_flags) if observation.history_flags else 'none'}\n\n"
            "Latest vitals and symptoms:\n"
            f"- Latest blood pressure: {latest_bp}\n"
            f"- Latest kick count: {latest_kicks}\n"
            f"- Bleeding: {latest_bleeding}\n"
            f"- Headache: {latest_headache}\n"
            f"- Swelling: {latest_swelling}\n"
            f"- Blurred vision: {latest_vision}\n"
            f"- Abdominal pain: {latest_abdominal_pain}\n"
            f"- Dizziness: {latest_dizziness}\n"
            f"- Latest weight (kg): {observation.latest_weight_kg if observation.latest_weight_kg is not None else 'unknown'}\n"
            f"- Latest energy (1-10): {observation.latest_energy if observation.latest_energy is not None else 'unknown'}\n"
            f"- Latest breathlessness (1-10): {observation.latest_breathlessness if observation.latest_breathlessness is not None else 'unknown'}\n\n"
            "3-day summary:\n"
            f"- Blood-pressure trend: {observation.bp_trend}\n"
            f"- Average kick count: {observation.avg_kick_count if observation.avg_kick_count is not None else 'unknown'}\n"
            f"- Average meals per day: {observation.avg_meals:.2f}\n"
            f"- Average sleep hours: {observation.avg_sleep:.2f}\n"
            f"- Risk flags: {', '.join(observation.risk_flags) if observation.risk_flags else 'none'}\n"
            f"- Days of data: {observation.days_of_data}"
        )

    def reset(self, user_id: int) -> PromptObservation:
        user, daily, checkin3 = _load_recent_data(user_id, days=3)
        self.current_user_id = user_id
        self.current_user = user
        self.current_daily = daily
        self.current_checkin3 = checkin3
        self.current_obs = _build_observation(user, daily, checkin3)
        self.current_text_observation = self.get_text_observation(self.current_obs)
        self.current_prompt = observation_to_prompt(self.current_obs, self.current_text_observation)
        self.episode_done = False
        self.reference_condition = infer_reference_condition(self.current_obs)
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
                rationale="Assessment step requested; observation and prompt returned for further reasoning.",
                reference_condition=self.reference_condition,
                reference_urgency=None,
                latent_risks={},
            )

        if action_mode == "request_signal":
            signal = action.target

            if signal not in self.current_obs.masked_signals:
                return StepResult(
                    observation=self.current_obs,
                    text_observation=self.current_text_observation or "",
                    prompt=self.current_prompt,
                    reward=-2.0,
                    reward_components={
                        "condition_score": 0.0,
                        "urgency_score": 0.0,
                        "under_escalation_penalty": 0.0,
                        "danger_override_penalty": 0.0,
                        "data_recency_bonus": 0.0,
                        "total_reward": -2.0,
                    },
                    done=False,
                    predicted_condition=None,
                    urgency=None,
                    diet_advice=[],
                    rationale=f"Signal '{signal}' already visible or invalid. Penalty applied.",
                    reference_condition=self.reference_condition,
                    reference_urgency=None,
                    latent_risks={},
                )

            db = SessionLocal()
            try:
                daily = (
                    db.query(DailyCheckin)
                    .filter(DailyCheckin.user_id == self.current_user_id)
                    .order_by(DailyCheckin.created_at.desc())
                    .limit(3)
                    .all()
                )
                checkin3 = (
                    db.query(Checkin3Day)
                    .filter(Checkin3Day.user_id == self.current_user_id)
                    .order_by(Checkin3Day.created_at.desc())
                    .first()
                )
            finally:
                db.close()

            if signal == "kick_count":
                kicks = [d.kick_count for d in daily if d.kick_count is not None]
                self.current_obs.avg_kick_count = sum(kicks) / len(kicks) if kicks else None
            elif signal == "energy_level":
                self.current_obs.latest_energy = checkin3.energy_level if checkin3 else None
            elif signal == "bp_trend":
                if len(daily) >= 2:
                    systolics = [d.bp_systolic for d in daily]
                    self.current_obs.bp_trend = "rising" if systolics[0] > systolics[-1] + 10 else "stable"
                else:
                    self.current_obs.bp_trend = "stable"

            self.current_obs.masked_signals.remove(signal)
            self._signals_requested += 1

            return StepResult(
                observation=self.current_obs,
                text_observation=self.current_text_observation or "",
                prompt=self.current_prompt,
                reward=-0.5,
                reward_components={
                    "condition_score": 0.0,
                    "urgency_score": 0.0,
                    "under_escalation_penalty": 0.0,
                    "danger_override_penalty": 0.0,
                    "data_recency_bonus": 0.0,
                    "total_reward": -0.5,
                },
                done=False,
                predicted_condition=None,
                urgency=None,
                diet_advice=[],
                rationale=f"Revealed '{signal}'. Remaining masked: {self.current_obs.masked_signals}",
                reference_condition=self.reference_condition,
                reference_urgency=None,
                latent_risks={},
            )

        if action_mode != "diagnose":
            raise ValueError(f"Unknown action_type: {action_mode}. Use 'assess', 'request_signal', or 'diagnose'.")
        if chosen_condition not in SAFE_CONDITIONS:
            raise ValueError(f"Unknown condition: {chosen_condition}. Valid: {SAFE_CONDITIONS}")
        if action.urgency not in URGENCY_ORDER:
            raise ValueError(f"Unknown urgency: {action.urgency}. Valid: {URGENCY_ORDER}")

        breakdown = calculate_reward(chosen_condition, action.urgency, self.current_obs)
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
            "reference_condition": self.reference_condition,
            "valid_actions": [
                {"action_type": "assess"},
                *[
                    {"condition": condition, "urgency": urgency, "rationale": "clinical explanation"}
                    for condition in SAFE_CONDITIONS
                    for urgency in URGENCY_ORDER
                ],
            ],
        }
