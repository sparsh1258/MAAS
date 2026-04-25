"""
Minimal OpenEnv -> LLM -> reward -> PPO loop for Niva.

Designed for Colab / notebook execution with Hugging Face TRL or Unsloth
compatible models. This script keeps the loop small and hackathon-friendly:

1. Reset the environment to get a prompt observation.
2. Ask the LLM for a JSON diagnosis / urgency.
3. Score that output with the deterministic reward policy.
4. Update the policy with PPO.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer

from environment import PrenatalEnvironment, parse_llm_output


@dataclass
class EpisodeSample:
    user_id: int
    prompt: str


def format_chat_prompt(system_prompt: str, user_prompt: str, response_format: str) -> str:
    return (
        "<|system|>\n"
        f"{system_prompt}\n\n"
        "Return only JSON in this format:\n"
        f"{response_format}\n"
        "<|user|>\n"
        f"{user_prompt}\n"
        "<|assistant|>\n"
    )


def collect_prompt_dataset(env: PrenatalEnvironment, user_ids: Iterable[int]) -> List[EpisodeSample]:
    dataset: List[EpisodeSample] = []
    for user_id in user_ids:
        prompt_obs = env.reset(user_id)
        dataset.append(
            EpisodeSample(
                user_id=user_id,
                prompt=format_chat_prompt(
                    prompt_obs.system_prompt,
                    prompt_obs.user_prompt,
                    prompt_obs.response_format,
                ),
            )
        )
    return dataset


def rollout_batch(
    env: PrenatalEnvironment,
    trainer: PPOTrainer,
    tokenizer: AutoTokenizer,
    batch: List[EpisodeSample],
    generation_kwargs: dict,
):
    query_tensors = [tokenizer(sample.prompt, return_tensors="pt").input_ids.squeeze(0).to(trainer.accelerator.device) for sample in batch]
    response_tensors = trainer.generate(query_tensors, **generation_kwargs)

    decoded_queries = [tokenizer.decode(tensor, skip_special_tokens=True) for tensor in query_tensors]
    decoded_responses = []
    rewards = []

    for sample, response_tensor in zip(batch, response_tensors):
        decoded = tokenizer.decode(response_tensor, skip_special_tokens=True)
        decoded_responses.append(decoded)

        prompt_obs = env.reset(sample.user_id)
        try:
            action = parse_llm_output(decoded)
            step = env.step(action)
            rewards.append(torch.tensor(step.reward, dtype=torch.float32, device=trainer.accelerator.device))
        except Exception:
            rewards.append(torch.tensor(-12.0, dtype=torch.float32, device=trainer.accelerator.device))

    trainer.step(query_tensors, response_tensors, rewards)
    return decoded_queries, decoded_responses, rewards


def train(args):
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
    dataset = collect_prompt_dataset(env, user_ids)

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
            _, responses, rewards = rollout_batch(env, trainer, tokenizer, batch, generation_kwargs)
            mean_reward = sum(float(r.item()) for r in rewards) / max(1, len(rewards))
            print(json.dumps({"epoch": epoch, "batch_start": start, "mean_reward": mean_reward, "sample_response": responses[0][:240] if responses else ""}))

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LLM triage agent with PPO over the Niva OpenEnv prompt loop.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="./artifacts/niva-openenv-ppo")
    parser.add_argument("--user-ids", default="1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    train(parser.parse_args())
