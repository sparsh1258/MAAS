from __future__ import annotations

import unittest

from environment import MULTITURN_TRAJECTORIES, MultiTurnPrenatalEnvironment
from xai_reward_model import calculate_reward


class MultiTurnEnvironmentTests(unittest.TestCase):
    def test_all_trajectories_reset(self) -> None:
        env = MultiTurnPrenatalEnvironment()
        for trajectory_id in MULTITURN_TRAJECTORIES:
            prompt = env.reset(trajectory_id)
            self.assertEqual(prompt.observation.episode_day_index, 1)
            self.assertFalse(env.done)
            self.assertIn("advance_day", {item["action_type"] for item in env.state()["valid_actions"]})

    def test_day_transition_and_final_diagnosis(self) -> None:
        env = MultiTurnPrenatalEnvironment()
        env.reset("traj_preeclampsia_slow")
        step = env.step({"action_type": "advance_day", "rationale": "Need symptom evidence."})
        self.assertFalse(step.done)
        self.assertEqual(step.observation.episode_day_index, 2)
        final = env.step(
            {
                "action_type": "diagnose",
                "target": "preeclampsia",
                "urgency": "go_to_hospital_today",
                "rationale": "Rising blood pressure and symptoms need hospital escalation.",
            }
        )
        self.assertTrue(final.done)
        self.assertGreaterEqual(final.reward, 0.0)
        self.assertLessEqual(final.reward, 1.0)
        self.assertEqual(final.reference_condition, "preeclampsia")
        self.assertFalse(final.under_escalated)

    def test_common_urgency_alias_is_normalized(self) -> None:
        env = MultiTurnPrenatalEnvironment()
        env.reset("traj_preeclampsia_fast")
        final = env.step(
            {
                "action_type": "diagnose",
                "target": "hypertension",
                "urgency": "urgent_care",
                "rationale": "Alias labels should normalize to supported labels.",
            }
        )
        self.assertEqual(final.predicted_condition, "preeclampsia")
        self.assertEqual(final.urgency, "go_to_hospital_today")

    def test_under_escalation_penalty_is_visible(self) -> None:
        env = MultiTurnPrenatalEnvironment()
        env.reset("traj_fetal_distress_sudden")
        env.step({"action_type": "advance_day", "rationale": "Collect day 2 evidence."})
        env.step({"action_type": "advance_day", "rationale": "Collect day 3 evidence."})
        final = env.step(
            {
                "action_type": "diagnose",
                "target": "low_risk",
                "urgency": "monitor_at_home",
                "rationale": "Unsafe false reassurance for a danger case.",
            }
        )
        self.assertTrue(final.under_escalated)
        self.assertLess(final.reward, 0.2)
        self.assertLess(final.reward_components["trajectory_under_escalation_penalty"], 0)

    def test_low_risk_case_penalizes_unnecessary_hospital_escalation(self) -> None:
        env = MultiTurnPrenatalEnvironment()
        env.reset("traj_low_risk_reassuring")
        env.step({"action_type": "advance_day", "rationale": "Collect day 2 evidence."})
        env.step({"action_type": "advance_day", "rationale": "Collect day 3 evidence."})
        final = env.step(
            {
                "action_type": "diagnose",
                "target": "low_risk",
                "urgency": "go_to_hospital_today",
                "rationale": "Overly cautious hospital referral on reassuring data.",
            }
        )
        self.assertLess(final.reward_components["trajectory_over_escalation_penalty"], 0)
        self.assertLess(final.reward, 0.8)

    def test_repeated_request_gets_extra_penalty(self) -> None:
        env = MultiTurnPrenatalEnvironment()
        env.reset("traj_preeclampsia_slow")
        first = env.step({"action_type": "request_bp_recheck", "rationale": "Confirm BP."})
        second = env.step({"action_type": "request_bp_recheck", "rationale": "Repeated BP check."})
        self.assertLess(second.reward, first.reward)
        self.assertLess(second.reward_components["repeated_request_penalty"], 0)

    def test_invalid_action_is_rejected(self) -> None:
        env = MultiTurnPrenatalEnvironment()
        env.reset("traj_low_risk_reassuring")
        with self.assertRaises(ValueError):
            env.step({"action_type": "invalid_jsonish_action", "rationale": "bad"})

    def test_reward_model_handles_invalid_labels_without_crashing(self) -> None:
        env = MultiTurnPrenatalEnvironment()
        prompt = env.reset("traj_low_risk_reassuring")
        reward = calculate_reward("not_a_condition", "not_an_urgency", prompt.observation)
        self.assertIn("invalid_condition_penalty", reward.reward_components)
        self.assertIn("invalid_urgency_penalty", reward.reward_components)


if __name__ == "__main__":
    unittest.main()
