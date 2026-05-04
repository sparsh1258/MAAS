# MAAS Multi-Turn Evaluation Report

This report is generated from `scripts/evaluate_multiturn.py`.
It is a deterministic benchmark over the checked-in three-day trajectories.

Safety note: these are synthetic benchmark metrics for a prototype triage environment, not clinical validation.

## Model Evaluation Status

- Deterministic baselines and oracle are always evaluated locally.
- Base/trained model rows are included only when `--base-model` and/or `--trained-model` are supplied and Hugging Face inference succeeds.
- Base model live evaluation: not run in this report.
- Trained model live evaluation: not run in this report.

## Summary

| Policy | Cases | Mean Reward | Condition Acc. | Urgency Acc. | Under-Esc. | Over-Esc. | JSON Valid | Mean Steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative_visible_baseline | 8 | 0.9551 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 3.00 |
| day1_fast_baseline | 8 | 0.4723 | 0.1250 | 0.1250 | 0.8750 | 0.0000 | 1.0000 | 1.00 |
| oracle | 8 | 0.9551 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 3.00 |

## Case Results

| Trajectory | Policy | Prediction | Reference | Reward | Under-Esc. | Steps |
|---|---|---|---|---:|---:|---:|
| traj_anemia_gradual | day1_fast_baseline | low_risk/monitor_at_home | anemia/visit_phc_this_week | 0.3969 | true | 1 |
| traj_anemia_gradual | conservative_visible_baseline | anemia/visit_phc_this_week | anemia/visit_phc_this_week | 1.0000 | false | 3 |
| traj_anemia_gradual | oracle | anemia/visit_phc_this_week | anemia/visit_phc_this_week | 1.0000 | false | 3 |
| traj_fetal_distress_sudden | day1_fast_baseline | low_risk/monitor_at_home | fetal_distress/go_to_hospital_today | 0.3969 | true | 1 |
| traj_fetal_distress_sudden | conservative_visible_baseline | fetal_distress/go_to_hospital_today | fetal_distress/go_to_hospital_today | 1.0000 | false | 3 |
| traj_fetal_distress_sudden | oracle | fetal_distress/go_to_hospital_today | fetal_distress/go_to_hospital_today | 1.0000 | false | 3 |
| traj_gestational_diabetes_noisy | day1_fast_baseline | low_risk/monitor_at_home | gestational_diabetes/visit_phc_this_week | 0.3969 | true | 1 |
| traj_gestational_diabetes_noisy | conservative_visible_baseline | gestational_diabetes/visit_phc_this_week | gestational_diabetes/visit_phc_this_week | 1.0000 | false | 3 |
| traj_gestational_diabetes_noisy | oracle | gestational_diabetes/visit_phc_this_week | gestational_diabetes/visit_phc_this_week | 1.0000 | false | 3 |
| traj_low_risk_reassuring | day1_fast_baseline | low_risk/monitor_at_home | low_risk/monitor_at_home | 1.0000 | false | 1 |
| traj_low_risk_reassuring | conservative_visible_baseline | low_risk/monitor_at_home | low_risk/monitor_at_home | 1.0000 | false | 3 |
| traj_low_risk_reassuring | oracle | low_risk/monitor_at_home | low_risk/monitor_at_home | 1.0000 | false | 3 |
| traj_mixed_signals_hard | day1_fast_baseline | low_risk/monitor_at_home | fetal_distress/go_to_hospital_today | 0.3969 | true | 1 |
| traj_mixed_signals_hard | conservative_visible_baseline | fetal_distress/go_to_hospital_today | fetal_distress/go_to_hospital_today | 0.6406 | false | 3 |
| traj_mixed_signals_hard | oracle | fetal_distress/go_to_hospital_today | fetal_distress/go_to_hospital_today | 0.6406 | false | 3 |
| traj_preeclampsia_fast | day1_fast_baseline | low_risk/monitor_at_home | preeclampsia/go_to_hospital_today | 0.3969 | true | 1 |
| traj_preeclampsia_fast | conservative_visible_baseline | preeclampsia/go_to_hospital_today | preeclampsia/go_to_hospital_today | 1.0000 | false | 3 |
| traj_preeclampsia_fast | oracle | preeclampsia/go_to_hospital_today | preeclampsia/go_to_hospital_today | 1.0000 | false | 3 |
| traj_preeclampsia_slow | day1_fast_baseline | low_risk/monitor_at_home | preeclampsia/go_to_hospital_today | 0.3969 | true | 1 |
| traj_preeclampsia_slow | conservative_visible_baseline | preeclampsia/go_to_hospital_today | preeclampsia/go_to_hospital_today | 1.0000 | false | 3 |
| traj_preeclampsia_slow | oracle | preeclampsia/go_to_hospital_today | preeclampsia/go_to_hospital_today | 1.0000 | false | 3 |
| traj_preterm_subtle | day1_fast_baseline | low_risk/monitor_at_home | preterm_risk/visit_phc_this_week | 0.3969 | true | 1 |
| traj_preterm_subtle | conservative_visible_baseline | preterm_risk/visit_phc_this_week | preterm_risk/visit_phc_this_week | 1.0000 | false | 3 |
| traj_preterm_subtle | oracle | preterm_risk/visit_phc_this_week | preterm_risk/visit_phc_this_week | 1.0000 | false | 3 |

## Honest Interpretation

- `oracle` is an upper-bound sanity check, not a deployable model.
- `conservative_visible_baseline` is a simple deterministic policy using visible signals after day 3.
- `day1_fast_baseline` shows how brittle one-shot triage is when it acts before evidence is revealed.
- Use `--base-model`, `--trained-model`, or `--predictions-jsonl` to evaluate actual model actions under the same table before claiming trained-model improvement.
