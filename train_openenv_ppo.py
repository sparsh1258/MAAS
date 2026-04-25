"""
Multi-step OpenEnv -> LLM -> reward -> PPO loop for MAAS.

This is the primary hackathon training entrypoint. It connects directly to the
current MAAS environment and trains over the same multi-step contract used by
the environment and inference runtime:

    reset -> assess/request_signal/diagnose -> step -> reward -> PPO update

The loop is intentionally compact so it can run in Colab or Hugging Face Jobs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from environment import ActionModel, PrenatalEnvironment, PromptObservation, parse_llm_output
from xai_reward_model import choose_urgency, featurize, infer_reference_condition


@dataclass
class EpisodeSample:
    user_id: int


def format_chat_prompt(system_prompt: str, user_prompt: str, response_format: str) -> str:
    return (
        f"{system_prompt}\n\n"
        f"{user_prompt}\n\n"
        "Return JSON only in this format:\n"
        f"{response_format}\n"
    )


def _history_block(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    lines = [
        (
            f"- Turn {item['turn']}: action_type={item['action_type']}; "
            f"feedback={item['feedback']}; reward={item['reward']}"
        )
        for item in history
    ]
    return "Episode history so far:\n" + "\n".join(lines)


def _stage_instruction(prompt_obs: PromptObservation, turn_index: int, max_episode_steps: int) -> str:
    hidden_signals = prompt_obs.observation.withheld_signals
    if turn_index == 0:
        return "First summarize the visible evidence with action_type='assess'."
    if hidden_signals and turn_index == 1:
        return "Request one withheld signal with action_type='request_signal' before the final diagnosis."
    if turn_index >= max_episode_steps - 1:
        return "Provide the final diagnosis now with action_type='diagnose'."
    if hidden_signals:
        return "Reassess the current evidence. If still uncertain, you may request one more withheld signal or diagnose."
    return "No hidden signals remain. Provide the final diagnosis with action_type='diagnose'."


def build_prompt(
    prompt_obs: PromptObservation,
    history: list[dict[str, str]] | None = None,
    stage_instruction: str | None = None,
) -> str:
    sections = [prompt_obs.user_prompt]
    if history:
        sections.append(_history_block(history))
    if stage_instruction:
        sections.append("Current turn instruction:\n" + stage_instruction)
    user_prompt = "\n\n".join(section for section in sections if section)
    return format_chat_prompt(prompt_obs.system_prompt, user_prompt, prompt_obs.response_format)


def collect_prompt_dataset(user_ids: Iterable[int]) -> list[EpisodeSample]:
    return [EpisodeSample(user_id=user_id) for user_id in user_ids]


def _fallback_diagnose_action(prompt_obs: PromptObservation) -> ActionModel:
    observation = prompt_obs.observation
    condition = infer_reference_condition(observation)
    urgency = choose_urgency(condition, featurize(observation))
    return ActionModel(
        action_type="diagnose",
        condition=condition,
        urgency=urgency,
        rationale="Fallback final diagnosis from the deterministic safety policy.",
    )


def _fallback_action(prompt_obs: PromptObservation, turn_index: int, max_episode_steps: int) -> ActionModel:
    hidden_signals = prompt_obs.observation.withheld_signals
    if turn_index == 0:
        return ActionModel(
            action_type="assess",
            rationale="Fallback assessment step before the final diagnosis.",
        )
    if hidden_signals and turn_index < max_episode_steps - 1:
        return ActionModel(
            action_type="request_signal",
            signal_name=hidden_signals[0],
            rationale="Fallback signal request for one hidden clinical feature.",
        )
    return _fallback_diagnose_action(prompt_obs)


def _generate_response_tensor(
    trainer: PPOTrainer,
    query_tensor: torch.Tensor,
    generation_kwargs: dict,
) -> torch.Tensor:
    response_tensors = trainer.generate([query_tensor], **generation_kwargs)
    return response_tensors[0]


def rollout_episode(
    env: PrenatalEnvironment,
    trainer: PPOTrainer,
    tokenizer: AutoTokenizer,
    sample: EpisodeSample,
    generation_kwargs: dict,
    max_episode_steps: int,
):
    prompt_obs = env.reset(sample.user_id)
    history: list[dict[str, str]] = []
    query_tensors: list[torch.Tensor] = []
    response_tensors: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    action_trace: list[dict[str, str | float]] = []
    final_reward = -12.0

    for turn_index in range(max_episode_steps):
        stage_instruction = _stage_instruction(prompt_obs, turn_index, max_episode_steps)
        prompt_text = build_prompt(prompt_obs, history=history, stage_instruction=stage_instruction)
        query_tensor = tokenizer(prompt_text, return_tensors="pt").input_ids.squeeze(0).to(trainer.accelerator.device)
        response_tensor = _generate_response_tensor(trainer, query_tensor, generation_kwargs)
        decoded = tokenizer.decode(response_tensor, skip_special_tokens=True)

        try:
            action = parse_llm_output(decoded)
            step_result = env.step(action)
        except Exception:
            action = _fallback_action(prompt_obs, turn_index, max_episode_steps)
            step_result = env.step(action)

        reward_tensor = torch.tensor(step_result.reward, dtype=torch.float32, device=trainer.accelerator.device)
        query_tensors.append(query_tensor)
        response_tensors.append(response_tensor)
        rewards.append(reward_tensor)
        final_reward = float(step_result.reward)

        history.append(
            {
                "turn": str(turn_index + 1),
                "action_type": action.action_type or "diagnose",
                "feedback": step_result.rationale,
                "reward": f"{step_result.reward:.2f}",
            }
        )
        action_trace.append(
            {
                "turn": turn_index + 1,
                "action_type": action.action_type or "diagnose",
                "reward": round(step_result.reward, 4),
                "done": step_result.done,
            }
        )

        prompt_obs = step_result.prompt
        if step_result.done:
            break

    if query_tensors:
        trainer.step(query_tensors, response_tensors, rewards)

    return {
        "rewards": rewards,
        "final_reward": final_reward,
        "steps": len(action_trace),
        "trace": action_trace,
    }


def create_arg_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or "Train an LLM triage agent with PPO over the MAAS OpenEnv loop."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="./artifacts/niva-openenv-ppo")
    parser.add_argument("--user-ids", default="1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-episode-steps", type=int, default=4)
    return parser


def run_training(args) -> None:
    try:
        from trl import PPOConfig, PPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "This PPO script requires a TRL build that still exposes PPOConfig and PPOTrainer. "
            "If your local TRL is a newer GRPO-only build, use train_grpo.py or install a PPO-capable TRL version."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    config = PPOConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
        log_with=None,
    )
    trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    env = PrenatalEnvironment()
    user_ids = [int(value) for value in args.user_ids.split(",") if value.strip()]
    dataset = collect_prompt_dataset(user_ids)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for epoch in range(args.epochs):
        for start in range(0, len(dataset), args.batch_size):
            batch = dataset[start : start + args.batch_size]
            batch_rollouts = [
                rollout_episode(
                    env,
                    trainer,
                    tokenizer,
                    sample,
                    generation_kwargs,
                    args.max_episode_steps,
                )
                for sample in batch
            ]
            rewards = [rollout["final_reward"] for rollout in batch_rollouts]
            mean_reward = sum(rewards) / max(1, len(rewards))
            mean_steps = sum(rollout["steps"] for rollout in batch_rollouts) / max(1, len(batch_rollouts))
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "batch_start": start,
                        "mean_reward": round(mean_reward, 4),
                        "mean_steps": round(mean_steps, 2),
                        "sample_trace": batch_rollouts[0]["trace"] if batch_rollouts else [],
                    }
                )
            )

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def train(args) -> None:
    run_training(args)


if __name__ == "__main__":
    run_training(create_arg_parser().parse_args())
