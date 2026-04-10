from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from database import SessionLocal
from models import UserProfile, DailyCheckin, Checkin3Day
from rl_risk_model import RL_RISK_MODEL

# PYDANTIC MODELS — what the environment returns

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


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    predicted_condition: Optional[str]
    urgency: Optional[str]
    diet_advice: List[str]
    rationale: str


class ActionModel(BaseModel):
    action_type: str                # "assess" or "diagnose"
    target: Optional[str] = None   # condition name if diagnosing
    urgency: Optional[str] = None  # urgency level if diagnosing

# CONSTANTS

CONDITIONS = [
    "preeclampsia",
    "gestational_diabetes",
    "anemia",
    "preterm_risk",
    "fetal_distress",
    "low_risk",
]

URGENCY_LEVELS = [
    "monitor_at_home",
    "visit_phc_this_week",
    "go_to_hospital_today",
]

CONDITION_URGENCY = {
    "preeclampsia":         "go_to_hospital_today",
    "fetal_distress":       "go_to_hospital_today",
    "gestational_diabetes": "visit_phc_this_week",
    "preterm_risk":         "visit_phc_this_week",
    "anemia":               "visit_phc_this_week",
    "low_risk":             "monitor_at_home",
}

CONDITION_SEVERITY = {
    "preeclampsia":         10,
    "fetal_distress":       10,
    "preterm_risk":         7,
    "gestational_diabetes": 6,
    "anemia":               5,
    "low_risk":             0,
}

DIET_ADVICE = {
    "preeclampsia": [
        "Reduce salt completely — avoid pickles and papad",
        "Eat banana and amla daily — helps control blood pressure",
        "Drink plenty of water — at least 8–10 glasses",
        "Avoid fried and oily food",
    ],
    "gestational_diabetes": [
        "Avoid rice and refined flour (maida)",
        "Eat dal, vegetables, and roti in small portions",
        "Eat small meals 5–6 times a day",
        "For fruits, prefer guava and jamun — avoid banana",
    ],
    "anemia": [
        "Eat spinach, amaranth, or fenugreek daily",
        "Consume jaggery (gud) with chickpeas — increases iron",
        "Eat iron-rich foods with lemon (vitamin C helps absorption)",
        "Avoid tea and coffee immediately after meals",
    ],
    "preterm_risk": [
        "Increase protein intake — dal, eggs, milk daily",
        "Avoid lifting heavy weights",
        "Take more rest — lie down with legs slightly elevated",
        "Reduce stress — talk to someone",
    ],
    "fetal_distress": [
        "Eat something sweet — juice or jaggery — and count baby kicks for 1 hour",
        "Lie on your left side",
        "Drink water and stay calm",
        "If kicks don’t increase, go to the hospital immediately",
    ],
    "low_risk": [
        "Take a balanced diet — dal, vegetables, roti, milk",
        "Take iron and folic acid tablets daily",
        "Drink 8–10 glasses of water daily",
        "Do light walking — 20–30 minutes",
    ],
}

# HELPER: load recent data from DB

def _load_recent_data(user_id: int, days: int = 3):
    """Pull last N daily checkins and latest 3-day checkin from DB."""
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

# HELPER: build observation from real DB data

def _build_observation(user: UserProfile,
                        daily: list,
                        checkin3: Optional[Checkin3Day]) -> Observation:

    risk_flags = []
    history_flags = []

    # ── History flags from profile ────────────────────────
    if user.history_diabetes:
        history_flags.append("family_diabetes")
    if user.history_hypertension:
        history_flags.append("family_hypertension")
    if user.history_preeclampsia:
        history_flags.append("prev_preeclampsia")
    if user.history_prev_comp:
        history_flags.append("prev_complication")

    # ── Immediate danger flags from latest checkin ────────
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

    # ── BP trend across last 3 days ───────────────────────
    bp_trend = "stable"
    if len(daily) >= 2:
        systolics = [d.bp_systolic for d in daily]
        if systolics[0] > systolics[-1] + 10:
            bp_trend = "rising"
            risk_flags.append("BP_RISING_TREND")
        elif systolics[0] < systolics[-1] - 10:
            bp_trend = "falling"

    # ── Averages ──────────────────────────────────────────
    avg_meals = (
        sum(d.meals_count for d in daily) / len(daily) if daily else 0
    )
    avg_sleep = (
        sum(d.sleep_hours for d in daily) / len(daily) if daily else 0
    )

    kicks = [d.kick_count for d in daily if d.kick_count is not None]
    avg_kicks = sum(kicks) / len(kicks) if kicks else None

    if avg_meals < 2:
        risk_flags.append("LOW_NUTRITION")
    if avg_kicks is not None and avg_kicks < 6:
        risk_flags.append("LOW_KICK_AVG")

    return Observation(
        user_id=user.id,
        weeks_pregnant=user.weeks_pregnant,
        trimester=user.trimester,
        region=user.region,
        risk_flags=risk_flags,
        bp_trend=bp_trend,
        avg_kick_count=avg_kicks,
        avg_meals=avg_meals,
        avg_sleep=avg_sleep,
        latest_weight_kg=checkin3.weight_kg if checkin3 else None,
        latest_energy=checkin3.energy_level if checkin3 else None,
        latest_breathlessness=checkin3.breathlessness if checkin3 else None,
        history_flags=history_flags,
        days_of_data=len(daily),
    )

# HELPER: rule-based condition classifier

def _classify_condition(obs: Observation) -> tuple[str, str]:
    """
    Returns (predicted_condition, rationale) using an RL-style value policy.

    The policy estimates Q-values for the competition-safe condition labels and
    also tracks wider latent maternal risks. This keeps the OpenEnv contract
    intact while moving the diagnosis layer beyond a simple rule cascade.
    """
    policy_result = RL_RISK_MODEL.predict(obs)
    return policy_result.condition, policy_result.rationale

# REWARD FUNCTION

def _calculate_reward(predicted: str,
                       predicted_urgency: str,
                       true_condition: str,
                       days_of_data: int) -> float:

    reward = 0.0
    true_urgency = CONDITION_URGENCY[true_condition]
    severity = CONDITION_SEVERITY[true_condition]

    # Correct condition
    if predicted == true_condition:
        reward += 10.0
    else:
        reward -= 5.0

    # Correct urgency
    urgency_idx = URGENCY_LEVELS.index
    if predicted_urgency == true_urgency:
        reward += 5.0
    elif urgency_idx(predicted_urgency) > urgency_idx(true_urgency):
        reward -= 2.0          # over-escalated — cautious, acceptable
    else:
        reward -= severity      # under-escalated — dangerous

    # More data = better prediction, reward accordingly
    if days_of_data >= 3:
        reward += 2.0
    elif days_of_data == 2:
        reward += 1.0

    return reward

# MAIN ENVIRONMENT CLASS

class PrenatalEnvironment:
    """
    OpenEnv-compatible RL environment for Niva — maternal health AI.

    Connects directly to the live prenatal.db database.
    Each episode = one user's recent health data.
    The agent assesses real readings and makes a risk prediction.

    step() / reset() / state() follow the OpenEnv spec.
    """

    def __init__(self):
        self.current_user_id: Optional[int] = None
        self.current_obs: Optional[Observation] = None
        self.episode_done: bool = False
        self.true_condition: Optional[str] = None   # set by reset(), used for reward

    # ── RESET ─────────────────────────────────────────────
    def reset(self, user_id: int) -> Observation:
        """
        Start a new episode for a given user.
        Loads their profile + last 3 days of checkins from DB.
        """
        user, daily, checkin3 = _load_recent_data(user_id, days=3)

        self.current_user_id = user_id
        self.current_obs = _build_observation(user, daily, checkin3)
        self.episode_done = False

        # For reward calculation — classify true condition from data
        self.true_condition, _ = _classify_condition(self.current_obs)

        return self.current_obs

    # ── STEP ──────────────────────────────────────────────
    def step(self, action: ActionModel) -> StepResult:
        """
        Agent takes one action.

        action_type = "assess"   → just returns observation, no reward yet
        action_type = "diagnose" → agent commits to a condition + urgency,
                                   gets reward, episode ends
        """
        if self.episode_done:
            raise RuntimeError("Episode done. Call reset() first.")

        if self.current_obs is None:
            raise RuntimeError("No active episode. Call reset() first.")

        obs = self.current_obs

        # ── ASSESS: agent wants more info ─────────────────
        if action.action_type == "assess":
            return StepResult(
                observation=obs,
                reward=0.0,
                done=False,
                predicted_condition=None,
                urgency=None,
                diet_advice=[],
                rationale="Assessing — more data reviewed",
            )

        # ── DIAGNOSE: agent makes final call ──────────────
        elif action.action_type == "diagnose":

            if action.target not in CONDITIONS:
                raise ValueError(f"Unknown condition: {action.target}. Valid: {CONDITIONS}")
            if action.urgency not in URGENCY_LEVELS:
                raise ValueError(f"Unknown urgency: {action.urgency}. Valid: {URGENCY_LEVELS}")

            # Safety override — if DANGER flags exist, urgency must be hospital
            danger_flags = [f for f in obs.risk_flags if f.startswith("DANGER")]
            if danger_flags and action.urgency != "go_to_hospital_today":
                # Force correct urgency and penalize
                action.urgency = "go_to_hospital_today"
                reward = -5.0
                rationale = f"Safety override: {danger_flags[0]} detected. Urgency corrected to hospital."
            else:
                reward = _calculate_reward(
                    predicted=action.target,
                    predicted_urgency=action.urgency,
                    true_condition=self.true_condition,
                    days_of_data=obs.days_of_data,
                )
                RL_RISK_MODEL.update_from_reward(obs, action.target, reward)
                _, rationale = _classify_condition(obs)

            policy_snapshot = RL_RISK_MODEL.predict(obs)
            latent_advice = [LATENT_CONDITION_ADVICE[k] for k in policy_snapshot.latent_risks if k in LATENT_CONDITION_ADVICE]
            diet = DIET_ADVICE.get(action.target, DIET_ADVICE["low_risk"]) + latent_advice[:2]
            self.episode_done = True

            return StepResult(
                observation=obs,
                reward=reward,
                done=True,
                predicted_condition=action.target,
                urgency=action.urgency,
                diet_advice=diet,
                rationale=rationale,
            )

        else:
            raise ValueError(f"Unknown action_type: {action.action_type}. Use 'assess' or 'diagnose'.")

    # ── STATE ─────────────────────────────────────────────
    def state(self) -> dict:
        """Returns current environment state — OpenEnv spec."""
        if self.current_obs is None:
            return {"status": "no_active_episode"}

        return {
            "user_id": self.current_user_id,
            "episode_done": self.episode_done,
            "observation": self.current_obs.model_dump(),
            "true_condition": self.true_condition,
            "valid_actions": [
                {"action_type": "assess"},
                *[
                    {"action_type": "diagnose", "target": c, "urgency": u}
                    for c in CONDITIONS
                    for u in URGENCY_LEVELS
                ],
            ],
        }
