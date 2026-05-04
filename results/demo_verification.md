# MAAS Demo Verification

Base URL: `https://sparsh122-maas-openenv.hf.space`
Endpoint status: `pass`
Reward schema current: `no`
Missing new reward fields: `adjusted_raw_reward, premature_diagnosis_penalty, trajectory_condition_score, trajectory_over_escalation_penalty, trajectory_under_escalation_penalty, trajectory_urgency_score`

| Check | Method | Status | Latency ms | Result |
|---|---|---:|---:|---|
| health | GET | 200 | 1458.5 | pass |
| reset | POST | 200 | 1337.4 | pass |
| step_advance_day2 | POST | 200 | 1284.3 | pass |
| step_advance_day3 | POST | 200 | 1529.8 | pass |
| step_diagnose | POST | 200 | 1366.1 | pass |
| state | GET | 200 | 1544.1 | pass |
| openenv_demo | GET | 200 | 1452.0 | pass |

## Raw Responses

### health

```json
{
  "status": "healthy"
}
```

### reset

```json
{
  "observation": {
    "user_id": 0,
    "weeks_pregnant": 34,
    "trimester": 3,
    "region": "Bihar",
    "regional_access_tier": "semi_urban",
    "risk_flags": [],
    "bp_trend": "stable",
    "avg_kick_count": 8.0,
    "avg_meals": null,
    "avg_sleep": null,
    "latest_weight_kg": null,
    "latest_energy": null,
    "latest_breathlessness": null,
    "symptom_cluster": [],
    "bp_systolic_latest": 138,
    "bp_diastolic_latest": 88,
    "composite_risk_score": 0.0,
    "history_flags": [],
    "days_of_data": 1,
    "masked_signals": [],
    "episode_day_index": 1,
    "total_episode_days": 3,
    "belief_state": {
      "visible_day_count": 1.0,
      "danger_flag_count": 0.0
    },
    "available_signals": [
      "latest_blood_pressure",
      "latest_kick_count"
    ],
    "withheld_signals": [
      "latest_symptoms",
      "history_flags",
      "avg_meals",
      "avg_sleep",
      "latest_energy"
    ],
    "signal_mask": {
      "risk_flags": true,
      "latest_blood_pressure": true,
      "latest_kick_count": true,
      "latest_symptoms": true,
      "symptom_cluster": true,
      "bp_trend": true,
      "avg_kick_count": true,
      "avg_meals": true,
      "avg_sleep": true,
      "latest_weight_kg": true,
      "latest_energy": true,
      "latest_breathlessness": true
    }
  },
  "text_observation": "Trajectory: traj_preeclampsia_slow\nDay 1 of 3\nRegion: Bihar\nWeeks pregnant: 34 (Trimester 3)\n\nVisible information:\n- Today's blood pressure: 138/88 mmHg\n- Today's kick count: 8\n- BP trend across visible days: stable\n- Visible risk flags: none\n- Symptom flags: hidden until day 2\n- History flags: hidden until day 3",
  "system_prompt": "You are MAAS, a maternal triage assistant operating in a multi-turn OpenEnv episode. You may gather more evidence over time before making a final diagnosis. Return only JSON with keys: action_type, target, urgency, rationale. Valid action_type values: request_bp_recheck, request_kick_count, advance_day, refer_to_phc, diagnose. Valid conditions: preeclampsia, gestational_diabetes, anemia, preterm_risk, fetal_distress, low_risk. Valid urgencies: monitor_at_home, visit_phc_this_week, go_to_hospital_today. If danger signs appear, prioritize safety and do not under-escalate.",
  "user_prompt": "Trajectory: traj_preeclampsia_slow\nDay 1 of 3\nRegion: Bihar\nWeeks pregnant: 34 (Trimester 3)\n\nVisible information:\n- Today's blood pressure: 138/88 mmHg\n- Today's kick count: 8\n- BP trend across visible days: stable\n- Visible risk flags: none\n- Symptom flags: hidden until day 2\n- History flags: hidden until day 3\n\nAvailable actions now: request_bp_recheck, request_kick_count, advance_day, refer_to_phc, diagnose\nReturn exactly one JSON action for the current step.",
  "response_format": "{\n  \"action_type\": \"request_bp_recheck | request_kick_count | advance_day | refer_to_phc | diagnose\",\n  \"target\": \"condition label when diagnosing, otherwise null\",\n  \"urgency\": \"urgency label when diagnosing, otherwise null\",\n  \"rationale\": \"short clinical explanation\"\n}",
  "valid_conditions": [
    "preeclampsia",
    "gestational_diabetes",
    "anemia",
    "preterm_risk",
    "fetal_distress",
    "low_risk"
  ],
  "valid_urgencies": [
    "monitor_at_home",
    "visit_phc_this_week",
    "go_to_hospital_today"
  ]
}
```

### step_advance_day2

```json
{
  "observation": {
    "user_id": 0,
    "weeks_pregnant": 34,
    "trimester": 3,
    "region": "Bihar",
    "regional_access_tier": "semi_urban",
    "risk_flags": [
      "HIGH_BP"
    ],
    "bp_trend": "stable",
    "avg_kick_count": 7.5,
    "avg_meals": null,
    "avg_sleep": null,
    "latest_weight_kg": null,
    "latest_energy": null,
    "latest_breathlessness": null,
    "symptom_cluster": [
      "headache"
    ],
    "bp_systolic_latest": 145,
    "bp_diastolic_latest": 95,
    "composite_risk_score": 0.1,
    "history_flags": [],
    "days_of_data": 2,
    "masked_signals": [],
    "episode_day_index": 2,
    "total_episode_days": 3,
    "belief_state": {
      "visible_day_count": 2.0,
      "danger_flag_count": 0.0
    },
    "available_signals": [
      "latest_blood_pressure",
      "latest_kick_count",
      "latest_symptoms"
    ],
    "withheld_signals": [
      "history_flags",
      "avg_meals",
      "avg_sleep",
      "latest_energy"
    ],
    "signal_mask": {
      "risk_flags": true,
      "latest_blood_pressure": true,
      "latest_kick_count": true,
      "latest_symptoms": true,
      "symptom_cluster": true,
      "bp_trend": true,
      "avg_kick_count": true,
      "avg_meals": true,
      "avg_sleep": true,
      "latest_weight_kg": true,
      "latest_energy": true,
      "latest_breathlessness": true
    }
  },
  "text_observation": "Trajectory: traj_preeclampsia_slow\nDay 2 of 3\nRegion: Bihar\nWeeks pregnant: 34 (Trimester 3)\n\nVisible information:\n- Today's blood pressure: 145/95 mmHg\n- Today's kick count: 7\n- BP trend across visible days: stable\n- Visible risk flags: HIGH_BP\n- Symptom flags: headache=True, swelling=False, bleeding=False, blurred_vision=False, breathlessness=False, cramps=False, dizziness=False\n- History flags: hidden until day 3",
  "prompt": {
    "observation": {
      "user_id": 0,
      "weeks_pregnant": 34,
      "trimester": 3,
      "region": "Bihar",
      "regional_access_tier": "semi_urban",
      "risk_flags": [
        "HIGH_BP"
      ],
      "bp_trend": "stable",
      "avg_kick_count": 7.5,
      "avg_meals": null,
      "avg_sleep": null,
      "latest_weight_kg": null,
      "latest_energy": null,
      "latest_breathlessness": null,
      "symptom_cluster": [
        "headache"
      ],
      "bp_systolic_latest": 145,
      "bp_diastolic_latest": 95,
      "composite_risk_score": 0.1,
      "history_flags": [],
      "days_of_data": 2,
      "masked_signals": [],
      "episode_day_index": 2,
      "total_episode_days": 3,
      "belief_state": {
        "visible_day_count": 2.0,
        "danger_flag_count": 0.0
      },
      "available_signals": [
        "latest_blood_pressure",
        "latest_kick_count",
        "latest_symptoms"
      ],
      "withheld_signals": [
        "history_flags",
        "avg_meals",
        "avg_sleep",
        "latest_energy"
      ],
      "signal_mask": {
        "risk_flags": true,
        "latest_blood_pressure": true,
        "latest_kick_count": true,
        "latest_symptoms": true,
        "symptom_cluster": true,
        "bp_trend": true,
        "avg_kick_count": true,
        "avg_meals": true,
        "avg_sleep": true,
        "latest_weight_kg": true,
        "latest_energy": true,
        "latest_breathlessness": true
      }
    },
    "text_observation": "Trajectory: traj_preeclampsia_slow\nDay 2 of 3\nRegion: Bihar\nWeeks pregnant: 34 (Trimester 3)\n\nVisible information:\n- Today's blood pressure: 145/95 mmHg\n- Today's kick count: 7\n- BP trend across visible days: stable\n- Visible risk flags: HIGH_BP\n- Symptom flags: headache=True, swelling=False, bleeding=False, blurred_vision=False, breathlessness=False, cramps=False, dizziness=False\n- History flags: hidden until day 3",
    "system_prompt": "You are MAAS, a maternal triage assistant operating in a multi-turn OpenEnv episode. You may gather more evidence over time before making a final diagnosis. Return only JSON with keys: actio
```

### step_advance_day3

```json
{
  "observation": {
    "user_id": 0,
    "weeks_pregnant": 34,
    "trimester": 3,
    "region": "Bihar",
    "regional_access_tier": "semi_urban",
    "risk_flags": [
      "DANGER_BP_CRITICAL",
      "HIGH_PREECLAMPSIA_SIGNAL",
      "DANGER_VISION_HEADACHE",
      "SYMPTOM_CLUSTER_HIGH",
      "BP_RISING_TREND"
    ],
    "bp_trend": "rising",
    "avg_kick_count": 7.0,
    "avg_meals": 2.67,
    "avg_sleep": 5.83,
    "latest_weight_kg": null,
    "latest_energy": 3,
    "latest_breathlessness": null,
    "symptom_cluster": [
      "blurred_vision",
      "headache",
      "swelling"
    ],
    "bp_systolic_latest": 162,
    "bp_diastolic_latest": 108,
    "composite_risk_score": 0.7200000000000001,
    "history_flags": [
      "family_hypertension"
    ],
    "days_of_data": 3,
    "masked_signals": [],
    "episode_day_index": 3,
    "total_episode_days": 3,
    "belief_state": {
      "visible_day_count": 3.0,
      "danger_flag_count": 2.0
    },
    "available_signals": [
      "latest_blood_pressure",
      "latest_kick_count",
      "latest_symptoms",
      "history_flags",
      "avg_meals",
      "avg_sleep",
      "latest_energy"
    ],
    "withheld_signals": [],
    "signal_mask": {
      "risk_flags": true,
      "latest_blood_pressure": true,
      "latest_kick_count": true,
      "latest_symptoms": true,
      "symptom_cluster": true,
      "bp_trend": true,
      "avg_kick_count": true,
      "avg_meals": true,
      "avg_sleep": true,
      "latest_weight_kg": true,
      "latest_energy": true,
      "latest_breathlessness": true
    }
  },
  "text_observation": "Trajectory: traj_preeclampsia_slow\nDay 3 of 3\nRegion: Bihar\nWeeks pregnant: 34 (Trimester 3)\n\nVisible information:\n- Today's blood pressure: 162/108 mmHg\n- Today's kick count: 6\n- BP trend across visible days: rising\n- Visible risk flags: DANGER_BP_CRITICAL, HIGH_PREECLAMPSIA_SIGNAL, DANGER_VISION_HEADACHE, SYMPTOM_CLUSTER_HIGH, BP_RISING_TREND\n- Symptom flags: headache=True, swelling=True, bleeding=False, blurred_vision=True, breathlessness=False, cramps=False, dizziness=False\n- History flags: family_hypertension\n- Average meals across visible days: 2.67\n- Average sleep across visible days: 5.83\n- Current energy level: 3/10",
  "prompt": {
    "observation": {
      "user_id": 0,
      "weeks_pregnant": 34,
      "trimester": 3,
      "region": "Bihar",
      "regional_access_tier": "semi_urban",
      "risk_flags": [
        "DANGER_BP_CRITICAL",
        "HIGH_PREECLAMPSIA_SIGNAL",
        "DANGER_VISION_HEADACHE",
        "SYMPTOM_CLUSTER_HIGH",
        "BP_RISING_TREND"
      ],
      "bp_trend": "rising",
      "avg_kick_count": 7.0,
      "avg_meals": 2.67,
      "avg_sleep": 5.83,
      "latest_weight_kg": null,
      "latest_energy": 3,
      "latest_breathlessness": null,
      "symptom_cluster": [
        "blurred_vision",
        "headache",
        "swelling"
      ],
      "bp_systolic_latest": 162,
      "bp_diastolic_latest": 108,
      "composite_risk_score": 0.7200000000000001,
      "history_flags": [
        "family_hypertension"
      ],
      "days_of_data": 3,
      "masked_signals": [],
      "episode_day_index": 3,
      "total_episode_days": 3,
      "belief_state": {
        "visible_day_count": 3.0,
        "danger_flag_count": 2.0
      },
      "available_signals": [
        "latest_blood_pressure",
        "latest_kick_count",
        "latest_symptoms",
        "history_flags",
        "avg_meals",
        "avg_sleep",
        "latest_energy"
      ],
      "withheld_signals": [],
      "signal_mask": {
        "risk_flags": true,
        "latest_blood_pressure": true,
        "latest_kick_count": true,
        "latest_symptoms": true,
        "symptom_cluster": true,
        "bp_trend": true,
        "avg_kick_count": true,
        "avg_meals": true,
        "avg_sleep": true,
        "latest_weight_kg": true,
        "latest_energy": true,
        "latest_breathlessness": true
      }
    },
    "tex
```

### step_diagnose

```json
{
  "observation": {
    "user_id": 0,
    "weeks_pregnant": 34,
    "trimester": 3,
    "region": "Bihar",
    "regional_access_tier": "semi_urban",
    "risk_flags": [
      "DANGER_BP_CRITICAL",
      "HIGH_PREECLAMPSIA_SIGNAL",
      "DANGER_VISION_HEADACHE",
      "SYMPTOM_CLUSTER_HIGH",
      "BP_RISING_TREND"
    ],
    "bp_trend": "rising",
    "avg_kick_count": 7.0,
    "avg_meals": 2.67,
    "avg_sleep": 5.83,
    "latest_weight_kg": null,
    "latest_energy": 3,
    "latest_breathlessness": null,
    "symptom_cluster": [
      "blurred_vision",
      "headache",
      "swelling"
    ],
    "bp_systolic_latest": 162,
    "bp_diastolic_latest": 108,
    "composite_risk_score": 0.7200000000000001,
    "history_flags": [
      "family_hypertension"
    ],
    "days_of_data": 3,
    "masked_signals": [],
    "episode_day_index": 3,
    "total_episode_days": 3,
    "belief_state": {
      "visible_day_count": 3.0,
      "danger_flag_count": 2.0
    },
    "available_signals": [
      "latest_blood_pressure",
      "latest_kick_count",
      "latest_symptoms",
      "history_flags",
      "avg_meals",
      "avg_sleep",
      "latest_energy"
    ],
    "withheld_signals": [],
    "signal_mask": {
      "risk_flags": true,
      "latest_blood_pressure": true,
      "latest_kick_count": true,
      "latest_symptoms": true,
      "symptom_cluster": true,
      "bp_trend": true,
      "avg_kick_count": true,
      "avg_meals": true,
      "avg_sleep": true,
      "latest_weight_kg": true,
      "latest_energy": true,
      "latest_breathlessness": true
    }
  },
  "text_observation": "Trajectory: traj_preeclampsia_slow\nDay 3 of 3\nRegion: Bihar\nWeeks pregnant: 34 (Trimester 3)\n\nVisible information:\n- Today's blood pressure: 162/108 mmHg\n- Today's kick count: 6\n- BP trend across visible days: rising\n- Visible risk flags: DANGER_BP_CRITICAL, HIGH_PREECLAMPSIA_SIGNAL, DANGER_VISION_HEADACHE, SYMPTOM_CLUSTER_HIGH, BP_RISING_TREND\n- Symptom flags: headache=True, swelling=True, bleeding=False, blurred_vision=True, breathlessness=False, cramps=False, dizziness=False\n- History flags: family_hypertension\n- Average meals across visible days: 2.67\n- Average sleep across visible days: 5.83\n- Current energy level: 3/10",
  "prompt": {
    "observation": {
      "user_id": 0,
      "weeks_pregnant": 34,
      "trimester": 3,
      "region": "Bihar",
      "regional_access_tier": "semi_urban",
      "risk_flags": [
        "DANGER_BP_CRITICAL",
        "HIGH_PREECLAMPSIA_SIGNAL",
        "DANGER_VISION_HEADACHE",
        "SYMPTOM_CLUSTER_HIGH",
        "BP_RISING_TREND"
      ],
      "bp_trend": "rising",
      "avg_kick_count": 7.0,
      "avg_meals": 2.67,
      "avg_sleep": 5.83,
      "latest_weight_kg": null,
      "latest_energy": 3,
      "latest_breathlessness": null,
      "symptom_cluster": [
        "blurred_vision",
        "headache",
        "swelling"
      ],
      "bp_systolic_latest": 162,
      "bp_diastolic_latest": 108,
      "composite_risk_score": 0.7200000000000001,
      "history_flags": [
        "family_hypertension"
      ],
      "days_of_data": 3,
      "masked_signals": [],
      "episode_day_index": 3,
      "total_episode_days": 3,
      "belief_state": {
        "visible_day_count": 3.0,
        "danger_flag_count": 2.0
      },
      "available_signals": [
        "latest_blood_pressure",
        "latest_kick_count",
        "latest_symptoms",
        "history_flags",
        "avg_meals",
        "avg_sleep",
        "latest_energy"
      ],
      "withheld_signals": [],
      "signal_mask": {
        "risk_flags": true,
        "latest_blood_pressure": true,
        "latest_kick_count": true,
        "latest_symptoms": true,
        "symptom_cluster": true,
        "bp_trend": true,
        "avg_kick_count": true,
        "avg_meals": true,
        "avg_sleep": true,
        "latest_weight_kg": true,
        "latest_energy": true,
        "latest_breathlessness": true
      }
    },
    "tex
```

### state

```json
{
  "trajectory_id": "traj_preeclampsia_slow",
  "current_day": 3,
  "cumulative_reward": 1.0,
  "done": true,
  "revealed_observation": {
    "user_id": 0,
    "weeks_pregnant": 34,
    "trimester": 3,
    "region": "Bihar",
    "regional_access_tier": "semi_urban",
    "risk_flags": [
      "DANGER_BP_CRITICAL",
      "HIGH_PREECLAMPSIA_SIGNAL",
      "DANGER_VISION_HEADACHE",
      "SYMPTOM_CLUSTER_HIGH",
      "BP_RISING_TREND"
    ],
    "bp_trend": "rising",
    "avg_kick_count": 7.0,
    "avg_meals": 2.67,
    "avg_sleep": 5.83,
    "latest_weight_kg": null,
    "latest_energy": 3,
    "latest_breathlessness": null,
    "symptom_cluster": [
      "blurred_vision",
      "headache",
      "swelling"
    ],
    "bp_systolic_latest": 162,
    "bp_diastolic_latest": 108,
    "composite_risk_score": 0.7200000000000001,
    "history_flags": [
      "family_hypertension"
    ],
    "days_of_data": 3,
    "masked_signals": [],
    "episode_day_index": 3,
    "total_episode_days": 3,
    "belief_state": {
      "visible_day_count": 3.0,
      "danger_flag_count": 2.0
    },
    "available_signals": [
      "latest_blood_pressure",
      "latest_kick_count",
      "latest_symptoms",
      "history_flags",
      "avg_meals",
      "avg_sleep",
      "latest_energy"
    ],
    "withheld_signals": [],
    "signal_mask": {
      "risk_flags": true,
      "latest_blood_pressure": true,
      "latest_kick_count": true,
      "latest_symptoms": true,
      "symptom_cluster": true,
      "bp_trend": true,
      "avg_kick_count": true,
      "avg_meals": true,
      "avg_sleep": true,
      "latest_weight_kg": true,
      "latest_energy": true,
      "latest_breathlessness": true
    }
  },
  "text_observation": "Trajectory: traj_preeclampsia_slow\nDay 3 of 3\nRegion: Bihar\nWeeks pregnant: 34 (Trimester 3)\n\nVisible information:\n- Today's blood pressure: 162/108 mmHg\n- Today's kick count: 6\n- BP trend across visible days: rising\n- Visible risk flags: DANGER_BP_CRITICAL, HIGH_PREECLAMPSIA_SIGNAL, DANGER_VISION_HEADACHE, SYMPTOM_CLUSTER_HIGH, BP_RISING_TREND\n- Symptom flags: headache=True, swelling=True, bleeding=False, blurred_vision=True, breathlessness=False, cramps=False, dizziness=False\n- History flags: family_hypertension\n- Average meals across visible days: 2.67\n- Average sleep across visible days: 5.83\n- Current energy level: 3/10",
  "valid_actions": [],
  "step_logs": [
    {
      "day": 2,
      "action": {
        "action_type": "advance_day",
        "target": null,
        "urgency": null,
        "rationale": "Verify day 2 transition."
      },
      "observation_revealed": "advanced to day 2",
      "step_reward": 0.0,
      "cumulative_reward": 0.0,
      "done": false
    },
    {
      "day": 3,
      "action": {
        "action_type": "advance_day",
        "target": null,
        "urgency": null,
        "rationale": "Verify day 3 transition."
      },
      "observation_revealed": "advanced to day 3",
      "step_reward": 0.0,
      "cumulative_reward": 0.0,
      "done": false
    },
    {
      "day": 3,
      "action": {
        "action_type": "diagnose",
        "target": "preeclampsia",
        "urgency": "go_to_hospital_today",
        "rationale": "Verify final diagnosis response."
      },
      "observation_revealed": "final diagnosis submitted",
      "step_reward": 1.0,
      "cumulative_reward": 1.0,
      "done": true
    }
  ]
}
```

### openenv_demo

```json
{
  "raw": "\n        <html>\n          <head>\n            <title>MAAS OpenEnv Demo</title>\n            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n            <style>\n              :root { color-scheme: light; }\n              body {\n                margin: 0;\n                font-family: \"Segoe UI\", Arial, sans-serif;\n                background: linear-gradient(180deg, #fff8f5 0%, #f8fafc 100%);\n                color: #1f2937;\n              }\n              .shell {\n                "
}
```
