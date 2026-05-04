# MAAS 90-Second Demo Script

## 0-10 seconds

Open the README and say:

"MAAS is a multi-turn OpenEnv environment for prenatal triage. The agent sees incomplete check-ins, chooses whether to gather evidence or escalate, then receives a safety-shaped reward."

## 10-30 seconds

Open `https://sparsh122-maas-openenv.hf.space/openenv-demo`.

Reset `traj_preeclampsia_slow`. Show day 1:

"On day 1, the case looks almost reassuring: borderline BP, normal fetal movement, symptoms hidden."

## 30-50 seconds

Click or submit `advance_day` twice.

"By day 2 and day 3, the hidden evidence appears: rising blood pressure, headache, swelling, blurred vision, and family hypertension."

## 50-70 seconds

Submit:

```json
{
  "action_type": "diagnose",
  "target": "preeclampsia",
  "urgency": "go_to_hospital_today",
  "rationale": "Critical BP plus headache and blurred vision require hospital escalation."
}
```

Say:

"The final result returns diagnosis, urgency, reward, and an episode trace so a judge can inspect exactly why the decision was scored."

## 70-90 seconds

Show `results/evaluation_report.md`.

"The current honest evidence is: acting on day 1 under-escalates most cases, while waiting for the full trajectory solves these checked-in cases. The GRPO pipeline has real training telemetry, but robust trained-model improvement still needs exact multi-turn before/after proof."
