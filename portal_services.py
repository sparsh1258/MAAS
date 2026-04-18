from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from environment import CONDITION_URGENCY, PrenatalEnvironment, _classify_condition
from models import Checkin3Day, CoordinatorTask, DailyCheckin, PatientReview, UserProfile

URGENCY_ORDER = {
    "go_to_hospital_today": 0,
    "visit_phc_this_week": 1,
    "monitor_at_home": 2,
}

URGENCY_LABELS = {
    "go_to_hospital_today": "Go To Hospital Today",
    "visit_phc_this_week": "Visit PHC This Week",
    "monitor_at_home": "Monitor At Home",
}

FLAG_DISPLAY_ORDER = [
    "Preeclampsia",
    "GDM",
    "Anaemia",
    "Fetal Distress",
    "Preterm",
]

STATE_ALIASES = {
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "andhra pradesh": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chandigarh": "Chandigarh",
    "chhattisgarh": "Chhattisgarh",
    "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra & nagar haveli and daman & diu": "Dadra and Nagar Haveli and Daman and Diu",
    "delhi": "Delhi",
    "nct of delhi": "Delhi",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "jammu and kashmir": "Jammu and Kashmir",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "west bengal": "West Bengal",
}

INDIA_STATE_META = {
    "Jammu and Kashmir": {"short": "JK", "x": 120, "y": 44},
    "Ladakh": {"short": "LA", "x": 186, "y": 35},
    "Himachal Pradesh": {"short": "HP", "x": 156, "y": 86},
    "Punjab": {"short": "PB", "x": 112, "y": 100},
    "Chandigarh": {"short": "CH", "x": 128, "y": 111},
    "Haryana": {"short": "HR", "x": 124, "y": 125},
    "Delhi": {"short": "DL", "x": 145, "y": 132},
    "Uttarakhand": {"short": "UK", "x": 186, "y": 108},
    "Rajasthan": {"short": "RJ", "x": 94, "y": 171},
    "Uttar Pradesh": {"short": "UP", "x": 208, "y": 154},
    "Bihar": {"short": "BR", "x": 260, "y": 173},
    "Jharkhand": {"short": "JH", "x": 252, "y": 206},
    "West Bengal": {"short": "WB", "x": 304, "y": 196},
    "Sikkim": {"short": "SK", "x": 307, "y": 160},
    "Assam": {"short": "AS", "x": 344, "y": 155},
    "Arunachal Pradesh": {"short": "AR", "x": 386, "y": 121},
    "Nagaland": {"short": "NL", "x": 371, "y": 166},
    "Manipur": {"short": "MN", "x": 376, "y": 186},
    "Mizoram": {"short": "MZ", "x": 356, "y": 213},
    "Tripura": {"short": "TR", "x": 338, "y": 206},
    "Meghalaya": {"short": "ML", "x": 331, "y": 179},
    "Gujarat": {"short": "GJ", "x": 54, "y": 226},
    "Dadra and Nagar Haveli and Daman and Diu": {"short": "DN", "x": 74, "y": 258},
    "Madhya Pradesh": {"short": "MP", "x": 162, "y": 226},
    "Chhattisgarh": {"short": "CG", "x": 235, "y": 246},
    "Odisha": {"short": "OD", "x": 291, "y": 250},
    "Maharashtra": {"short": "MH", "x": 128, "y": 305},
    "Goa": {"short": "GA", "x": 110, "y": 355},
    "Telangana": {"short": "TS", "x": 222, "y": 318},
    "Andhra Pradesh": {"short": "AP", "x": 252, "y": 362},
    "Karnataka": {"short": "KA", "x": 168, "y": 384},
    "Tamil Nadu": {"short": "TN", "x": 223, "y": 458},
    "Kerala": {"short": "KL", "x": 171, "y": 472},
    "Puducherry": {"short": "PY", "x": 242, "y": 443},
    "Lakshadweep": {"short": "LD", "x": 62, "y": 430},
    "Andaman and Nicobar Islands": {"short": "AN", "x": 368, "y": 470},
}

INDIA_SILHOUETTE_PATH = (
    "M112 36 L150 52 L200 44 L238 64 L226 84 L260 102 L254 126 L274 138 "
    "L316 140 L358 152 L388 170 L380 188 L350 194 L330 220 L302 222 L290 246 "
    "L302 272 L286 292 L264 320 L250 360 L238 398 L226 438 L206 492 L170 484 "
    "L154 446 L142 402 L128 372 L104 344 L88 314 L70 286 L44 260 L48 214 "
    "L74 180 L70 136 L90 102 L96 72 Z"
)


def _ts(value: datetime | None) -> float:
    if value is None:
        return 0.0
    return value.timestamp()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _label(value: datetime | None) -> str:
    if not value:
        return "No data"
    return value.strftime("%d %b")


def _human_condition(name: str) -> str:
    return name.replace("_", " ").title()


def normalize_state_name(value: str | None) -> str:
    if not value:
        return "Unknown"
    cleaned = " ".join(value.strip().replace("&", "and").split())
    lowered = cleaned.lower()
    return STATE_ALIASES.get(lowered, cleaned.title())


def _severity_from_counts(critical: int, amber: int, total: int, escalated_recent: int) -> str:
    if critical > 0 or escalated_recent > 0:
        return "critical"
    if amber >= 2 or (total and amber / total >= 0.5):
        return "watch"
    if total > 0:
        return "stable"
    return "inactive"


def _risk_band_from_counts(critical: int, amber: int, safe: int, total: int, escalated_recent: int) -> str:
    if total == 0:
        return "gray"
    if critical > 0 or escalated_recent > 0:
        return "red"
    if amber >= 2 or (total and amber / total >= 0.6):
        return "orange"
    if amber > 0 or safe < total:
        return "yellow"
    return "green"


def _risk_band_label(risk_band: str) -> str:
    return {
        "red": "Red alert",
        "orange": "Orange watch",
        "yellow": "Yellow monitoring",
        "green": "Green stable",
        "gray": "No live data",
    }.get(risk_band, risk_band.title())


def _national_action_for_flag(flag: str) -> str:
    actions = {
        "Preeclampsia": "Deploy PHC and ANM blood-pressure screening plus same-day referral tracking.",
        "GDM": "Run low-cost diet counselling and fast-track PHC glucose review in high-burden blocks.",
        "Anaemia": "Scale iron-folic adherence drives, nutrition counselling, and community follow-up calls.",
        "Fetal Distress": "Push kick-count education, hydration reminders, and emergency transport readiness.",
        "Preterm": "Strengthen warning-sign counselling and early PHC review for bleeding or pain.",
    }
    return actions.get(flag, "Maintain antenatal outreach and district-level risk surveillance.")


def _prevention_methods(dominant_flags: list[str], severity: str) -> list[str]:
    methods: list[str] = []
    if "Preeclampsia" in dominant_flags:
        methods.append("Run low-cost BP checks through ANM/PHC follow-up and reduce salty packaged foods.")
        methods.append("Prepare a same-day transport plan for headache, swelling, blurred vision, or very high BP.")
    if "GDM" in dominant_flags:
        methods.append("Shift to smaller balanced meals, avoid sugary drinks, and use daily walking if clinically safe.")
        methods.append("Prioritize PHC glucose review and family-supported diet changes over expensive monitoring.")
    if "Anaemia" in dominant_flags:
        methods.append("Improve iron-folic tablet adherence and add low-cost iron foods like chana, jaggery, and greens.")
        methods.append("Pair meals with vitamin C sources like lemon or amla to improve iron absorption.")
    if "Fetal Distress" in dominant_flags:
        methods.append("Teach daily kick-count tracking and escalate immediately if movement drops.")
        methods.append("Encourage hydration, left-side rest, and a family emergency contact chain.")
    if "Preterm" in dominant_flags:
        methods.append("Reduce heavy physical work, increase rest, and educate on bleeding, leaking fluid, and pain warnings.")
        methods.append("Use early PHC follow-up rather than waiting for labor-like symptoms to worsen.")
    if not methods:
        methods.append("Continue routine PHC antenatal visits, hydration reminders, and family education on danger signs.")
    if severity == "critical":
        methods.append("Deploy phone-based follow-up within 24 hours for all high-risk households in this state cluster.")
    return methods[:4]


def _history_daily(db: Session, user_id: int, limit: int = 12) -> list[DailyCheckin]:
    return (
        db.query(DailyCheckin)
        .filter(DailyCheckin.user_id == user_id)
        .order_by(DailyCheckin.created_at.desc())
        .limit(limit)
        .all()
    )


def _history_3day(db: Session, user_id: int, limit: int = 8) -> list[Checkin3Day]:
    return (
        db.query(Checkin3Day)
        .filter(Checkin3Day.user_id == user_id)
        .order_by(Checkin3Day.created_at.desc())
        .limit(limit)
        .all()
    )


def _get_review(db: Session, patient_id: int) -> PatientReview | None:
    return db.query(PatientReview).filter(PatientReview.patient_id == patient_id).first()


def get_or_create_review(db: Session, patient_id: int) -> PatientReview:
    review = _get_review(db, patient_id)
    if review:
        return review
    review = PatientReview(patient_id=patient_id, reviewed=False, notes="")
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


def list_patient_tasks(db: Session, patient_id: int) -> list[CoordinatorTask]:
    return (
        db.query(CoordinatorTask)
        .filter(CoordinatorTask.patient_id == patient_id)
        .order_by(CoordinatorTask.created_at.desc())
        .all()
    )


def ensure_escalation_task(db: Session, patient_id: int) -> CoordinatorTask:
    task = (
        db.query(CoordinatorTask)
        .filter(
            CoordinatorTask.patient_id == patient_id,
            CoordinatorTask.task_type == "hospital_escalation",
            CoordinatorTask.status.in_(["open", "in_progress"]),
        )
        .order_by(CoordinatorTask.created_at.desc())
        .first()
    )
    if task:
        return task

    task = CoordinatorTask(
        patient_id=patient_id,
        task_type="hospital_escalation",
        title="Arrange same-day hospital escalation",
        details="Escalated by doctor portal. Coordinate transport, call family, and confirm arrival.",
        priority="critical",
        status="open",
        due_at=datetime.utcnow() + timedelta(hours=4),
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _condition_flags(profile: UserProfile, observation, predicted_condition: str) -> list[str]:
    flags: set[str] = set()
    risk_flags = set(observation.risk_flags)

    if predicted_condition == "preeclampsia" or {"HIGH_BP", "HIGH_PREECLAMPSIA_SIGNAL"} & risk_flags:
        flags.add("Preeclampsia")
    if predicted_condition == "gestational_diabetes" or (
        profile.history_diabetes and (observation.latest_energy or 0) <= 4 and (observation.latest_breathlessness or 0) >= 5
    ):
        flags.add("GDM")
    if predicted_condition == "anemia" or observation.avg_meals <= 2 or (observation.latest_energy or 10) <= 3:
        flags.add("Anaemia")
    if predicted_condition == "fetal_distress" or "LOW_KICK_AVG" in risk_flags or (observation.avg_kick_count or 10) < 3:
        flags.add("Fetal Distress")
    if predicted_condition == "preterm_risk" or "BLEEDING_PRESENT" in risk_flags:
        flags.add("Preterm")

    return [flag for flag in FLAG_DISPLAY_ORDER if flag in flags]


def _trend_arrow(daily: list[DailyCheckin], trends: list[Checkin3Day], urgency: str) -> str:
    if len(daily) >= 2:
        latest = daily[0]
        prev = daily[1]
        if latest.bp_systolic > prev.bp_systolic + 8 or (latest.kick_count or 0) < (prev.kick_count or 0) - 1:
            return "↑"
        if latest.bp_systolic < prev.bp_systolic - 8 or (latest.kick_count or 0) > (prev.kick_count or 0) + 1:
            return "↓"
    if len(trends) >= 2:
        latest_3 = trends[0]
        prev_3 = trends[1]
        if latest_3.energy_level < prev_3.energy_level - 1 or latest_3.breathlessness > prev_3.breathlessness + 1:
            return "↑"
        if latest_3.energy_level > prev_3.energy_level + 1 or latest_3.breathlessness < prev_3.breathlessness - 1:
            return "↓"
    return "↑" if urgency == "go_to_hospital_today" else "→"


def _build_observation(user_id: int):
    env = PrenatalEnvironment()
    return env.reset(user_id=user_id)


def build_patient_snapshot(db: Session, profile: UserProfile) -> dict[str, Any]:
    state_name = normalize_state_name(profile.region)
    daily = _history_daily(db, profile.id)
    trends = _history_3day(db, profile.id)
    review = _get_review(db, profile.id)
    observation = _build_observation(profile.id)
    predicted_condition, rationale = _classify_condition(observation)
    base_urgency = CONDITION_URGENCY[predicted_condition]
    effective_urgency = review.urgency_override if review and review.urgency_override else base_urgency
    last_checkin_at = max(
        [value for value in [daily[0].created_at if daily else None, trends[0].created_at if trends else None] if value],
        default=None,
    )
    recent_cutoff = datetime.utcnow() - timedelta(hours=24)
    escalated_recent = (
        effective_urgency == "go_to_hospital_today"
        and (
            (last_checkin_at and last_checkin_at >= recent_cutoff)
            or (review and review.escalated_at and review.escalated_at >= recent_cutoff)
        )
    )

    return {
        "patient_id": profile.id,
        "patient_name": profile.name,
        "region": state_name,
        "weeks_pregnant": profile.weeks_pregnant,
        "trimester": profile.trimester,
        "last_checkin_at": _iso(last_checkin_at),
        "risk_level": base_urgency,
        "effective_urgency": effective_urgency,
        "predicted_condition": predicted_condition,
        "predicted_condition_label": _human_condition(predicted_condition),
        "rationale": rationale,
        "condition_flags": _condition_flags(profile, observation, predicted_condition),
        "risk_flags": observation.risk_flags,
        "history_flags": observation.history_flags,
        "trend": _trend_arrow(daily, trends, effective_urgency),
        "reviewed": bool(review and review.reviewed),
        "reviewed_at": _iso(review.reviewed_at if review else None),
        "notes": review.notes if review and review.notes else "",
        "escalated_at": _iso(review.escalated_at if review else None),
        "escalated_recent": escalated_recent,
        "urgency_label": URGENCY_LABELS[effective_urgency],
    }


def list_patient_snapshots(
    db: Session,
    *,
    risk_level: str | None = None,
    condition_flag: str | None = None,
    last_checkin_date: date | None = None,
) -> list[dict[str, Any]]:
    profiles = db.query(UserProfile).order_by(UserProfile.name.asc()).all()
    items = [build_patient_snapshot(db, profile) for profile in profiles]

    if risk_level:
        items = [item for item in items if item["effective_urgency"] == risk_level]
    if condition_flag:
        items = [item for item in items if condition_flag in item["condition_flags"]]
    if last_checkin_date:
        items = [
            item
            for item in items
            if item["last_checkin_at"] and datetime.fromisoformat(item["last_checkin_at"]).date() == last_checkin_date
        ]

    items.sort(
        key=lambda item: (
            0 if item["escalated_recent"] else 1,
            URGENCY_ORDER.get(item["effective_urgency"], 99),
            -_ts(datetime.fromisoformat(item["last_checkin_at"])) if item["last_checkin_at"] else float("inf"),
            item["patient_name"].lower(),
        )
    )
    return items


def build_patient_detail(db: Session, patient_id: int) -> dict[str, Any]:
    profile = db.query(UserProfile).filter(UserProfile.id == patient_id).first()
    if not profile:
        raise ValueError("Patient not found")

    snapshot = build_patient_snapshot(db, profile)
    daily_desc = _history_daily(db, patient_id, limit=20)
    trend_desc = _history_3day(db, patient_id, limit=12)
    review = get_or_create_review(db, patient_id)
    tasks = list_patient_tasks(db, patient_id)

    daily = list(reversed(daily_desc))
    trends = list(reversed(trend_desc))

    bp_history = [
        {
            "label": _label(item.created_at),
            "timestamp": _iso(item.created_at),
            "systolic": item.bp_systolic,
            "diastolic": item.bp_diastolic,
        }
        for item in daily
    ]
    weight_history = [
        {
            "label": _label(item.created_at),
            "timestamp": _iso(item.created_at),
            "value": item.weight_kg,
        }
        for item in trends
    ]
    kick_history = [
        {
            "label": _label(item.created_at),
            "timestamp": _iso(item.created_at),
            "value": item.kick_count or 0,
        }
        for item in daily
    ]

    return {
        "profile": {
            "id": profile.id,
            "name": profile.name,
            "age": profile.age,
            "region": normalize_state_name(profile.region),
            "weeks_pregnant": profile.weeks_pregnant,
            "trimester": profile.trimester,
            "weight_kg": profile.weight_kg,
        },
        "summary": snapshot,
        "ai_summary": (
            f"{snapshot['predicted_condition_label']} suggested. {snapshot['rationale']} "
            f"Recommended urgency: {snapshot['urgency_label']}."
        ),
        "bp_history": bp_history,
        "weight_history": weight_history,
        "kick_history": kick_history,
        "notes": review.notes or "",
        "reviewed": review.reviewed,
        "reviewed_at": _iso(review.reviewed_at),
        "escalated_at": _iso(review.escalated_at),
        "tasks": [
            {
                "id": task.id,
                "task_type": task.task_type,
                "title": task.title,
                "details": task.details or "",
                "priority": task.priority,
                "status": task.status,
                "due_at": _iso(task.due_at),
                "completed_at": _iso(task.completed_at),
                "created_at": _iso(task.created_at),
            }
            for task in tasks
        ],
    }


def _build_state_analysis(patients: list[dict[str, Any]], selected_region: str | None = None) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for patient in patients:
        grouped[patient["region"]].append(patient)

    state_rows: list[dict[str, Any]] = []
    map_states: list[dict[str, Any]] = []
    national_flag_counter: Counter[str] = Counter()

    all_states = sorted(set(INDIA_STATE_META) | set(grouped))
    for state_name in all_states:
        items = grouped.get(state_name, [])
        flag_counter = Counter(flag for item in items for flag in item["condition_flags"])
        national_flag_counter.update(flag_counter)
        total = len(items)
        critical = sum(1 for item in items if item["effective_urgency"] == "go_to_hospital_today")
        amber = sum(1 for item in items if item["effective_urgency"] == "visit_phc_this_week")
        safe = sum(1 for item in items if item["effective_urgency"] == "monitor_at_home")
        escalated_recent = sum(1 for item in items if item["escalated_recent"])
        severity = _severity_from_counts(critical, amber, total, escalated_recent)
        risk_band = _risk_band_from_counts(critical, amber, safe, total, escalated_recent)
        attention_score = critical * 5 + amber * 2 + escalated_recent * 3 + total
        dominant_flags = [flag for flag, _ in flag_counter.most_common(3)]
        last_checkin = max(
            [datetime.fromisoformat(item["last_checkin_at"]) for item in items if item["last_checkin_at"]],
            default=None,
        )
        top_conditions = Counter(item["predicted_condition_label"] for item in items if item["predicted_condition_label"])

        row = {
            "state": state_name,
            "patient_count": total,
            "critical_count": critical,
            "watch_count": amber,
            "safe_count": safe,
            "recent_escalations": escalated_recent,
            "dominant_flags": dominant_flags,
            "top_condition": top_conditions.most_common(1)[0][0] if top_conditions else "No active pattern",
            "needs_immediate_attention": severity == "critical",
            "severity": severity,
            "risk_band": risk_band,
            "risk_band_label": _risk_band_label(risk_band),
            "attention_score": attention_score,
            "selected": state_name == selected_region,
            "last_checkin_at": _iso(last_checkin),
            "prevention_methods": _prevention_methods(dominant_flags, severity),
            "recommended_action": (
                "Immediate outreach and PHC/hospital coordination required."
                if severity == "critical"
                else "Targeted follow-up and low-cost prevention outreach recommended."
                if severity == "watch"
                else "Continue routine monitoring and community education."
                if total
                else "No live patient data yet."
            ),
        }

        if total:
            state_rows.append(row)

        meta = INDIA_STATE_META.get(state_name, {})
        map_states.append(
            {
                "state": state_name,
                "short": meta.get("short", state_name[:2].upper()),
                "x": meta.get("x"),
                "y": meta.get("y"),
                "patient_count": total,
                "critical_count": critical,
                "watch_count": amber,
                "safe_count": safe,
                "severity": severity,
                "risk_band": risk_band,
                "risk_band_label": _risk_band_label(risk_band),
                "selected": state_name == selected_region,
                "dominant_flags": dominant_flags,
                "recommended_action": row["recommended_action"],
            }
        )

    state_rows.sort(
        key=lambda row: (
            0 if row["needs_immediate_attention"] else 1,
            -row["attention_score"],
            row["state"],
        )
    )

    top_national_flags = [flag for flag, _ in national_flag_counter.most_common(3)]
    top_attention_states = state_rows[:5]
    ministry_actions = [_national_action_for_flag(flag) for flag in top_national_flags] or [
        "Keep district surveillance active and maintain routine ANC outreach."
    ]

    return {
        "national": {
            "states_with_data": len(state_rows),
            "critical_states": sum(1 for row in state_rows if row["severity"] == "critical"),
            "watch_states": sum(1 for row in state_rows if row["severity"] == "watch"),
            "red_states": sum(1 for row in state_rows if row["risk_band"] == "red"),
            "orange_states": sum(1 for row in state_rows if row["risk_band"] == "orange"),
            "yellow_states": sum(1 for row in state_rows if row["risk_band"] == "yellow"),
            "green_states": sum(1 for row in state_rows if row["risk_band"] == "green"),
            "same_day_cases": sum(row["critical_count"] for row in state_rows),
            "top_attention_states": [row["state"] for row in top_attention_states[:3]],
            "top_risk_drivers": top_national_flags,
            "recommended_focus": (
                "Prioritize rapid outreach in states with hospital-level urgency, then reinforce PHC and home prevention in watch states."
            ),
            "minister_briefing": (
                f"Immediate attention is required in {sum(1 for row in state_rows if row['risk_band'] == 'red')} "
                f"state clusters. The strongest national drivers are "
                f"{', '.join(top_national_flags) if top_national_flags else 'routine ANC follow-up'}."
            ),
            "ministry_actions": ministry_actions,
        },
        "india_map": {
            "path": INDIA_SILHOUETTE_PATH,
            "states": map_states,
        },
        "state_groups": state_rows,
    }


def build_coordinator_dashboard(
    db: Session,
    *,
    status: str | None = None,
    region: str | None = None,
) -> dict[str, Any]:
    all_patients = list_patient_snapshots(db)
    analytics = _build_state_analysis(all_patients, selected_region=region)
    patients = list(all_patients)
    if region:
        patients = [item for item in patients if item["region"] == region]

    tasks = db.query(CoordinatorTask).order_by(CoordinatorTask.created_at.desc()).all()
    if status:
        tasks = [task for task in tasks if task.status == status]

    open_tasks = [task for task in tasks if task.status in {"open", "in_progress"}]
    completed_today = [
        task for task in tasks if task.completed_at and task.completed_at.date() == datetime.utcnow().date()
    ]

    queue = []
    patient_lookup = {item["patient_id"]: item for item in patients}
    for item in patients:
        patient_tasks = [task for task in open_tasks if task.patient_id == item["patient_id"]]
        next_task = patient_tasks[0] if patient_tasks else None
        queue.append(
            {
                **item,
                "open_task_count": len(patient_tasks),
                "next_task_title": next_task.title if next_task else "No active task",
                "next_task_status": next_task.status if next_task else "none",
            }
        )

    queue.sort(
        key=lambda item: (
            URGENCY_ORDER.get(item["effective_urgency"], 99),
            0 if item["open_task_count"] else 1,
            -_ts(datetime.fromisoformat(item["last_checkin_at"])) if item["last_checkin_at"] else float("inf"),
        )
    )

    return {
        "summary": {
            "critical_today": sum(1 for item in queue if item["effective_urgency"] == "go_to_hospital_today"),
            "pending_tasks": len(open_tasks),
            "phc_visits_due": sum(
                1
                for task in open_tasks
                if task.task_type in {"phc_visit", "follow_up"} and (not region or patient_lookup.get(task.patient_id, {}).get("region") == region)
            ),
            "completed_today": len(completed_today),
        },
        "queue": queue,
        "tasks": [
            {
                "id": task.id,
                "patient_id": task.patient_id,
                "title": task.title,
                "task_type": task.task_type,
                "priority": task.priority,
                "status": task.status,
                "due_at": _iso(task.due_at),
                "completed_at": _iso(task.completed_at),
            }
            for task in tasks
            if not region or patient_lookup.get(task.patient_id, {}).get("region") == region
        ],
        "regions": sorted({item["region"] for item in all_patients}),
        "analytics": analytics,
    }
