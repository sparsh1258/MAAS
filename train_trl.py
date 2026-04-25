"""
Minimal PPO training script for Colab.

Expected usage in a notebook:

    !pip install -r requirements.txt
    !python train_trl.py --user-ids 1,2,3

This script:
1. Resets the prenatal OpenEnv-style environment.
2. Reads the text observation prompt.
3. Lets a small instruction model produce JSON with diagnosis + urgency.
4. Scores the output using xai_reward_model.calculate_reward().
5. Runs a PPO update step with Hugging Face TRL.
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
from xai_reward_model import calculate_reward


@dataclass
class PromptSample:
    user_id: int
    prompt_text: str


def build_prompt(text_observation: str) -> str:
    schema = {
        "condition": "preeclampsia|gestational_diabetes|anemia|preterm_risk|fetal_distress|low_risk",
        "urgency": "monitor_at_home|visit_phc_this_week|go_to_hospital_today",
        "rationale": "short explanation",
    }
    return (
        "You are a maternal-health triage assistant.\n"
        "Read the patient observation below and respond with JSON only.\n"
        f"JSON schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"{text_observation}\n"
    )


def build_dataset(env: PrenatalEnvironment, user_ids: Iterable[int]) -> List[PromptSample]:
    samples: List[PromptSample] = []
    for user_id in user_ids:
        prompt_obs = env.reset(user_id)
        samples.append(PromptSample(user_id=user_id, prompt_text=build_prompt(prompt_obs.text_observation)))
    return samples


def generate_batch(
    trainer: PPOTrainer,
    tokenizer: AutoTokenizer,
    prompts: List[PromptSample],
    generation_kwargs: dict,
):
    query_tensors = [
        tokenizer(prompt.prompt_text, return_tensors="pt").input_ids.squeeze(0).to(trainer.accelerator.device)
        for prompt in prompts
    ]
    response_tensors = trainer.generate(query_tensors, **generation_kwargs)
    responses = [tokenizer.decode(tensor, skip_special_tokens=True) for tensor in response_tensors]
    return query_tensors, response_tensors, responses


def build_model_kwargs(args) -> dict:
    kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if args.load_in_4bit:
        kwargs["load_in_4bit"] = True
    return kwargs


def run_training(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = build_model_kwargs(args)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    ppo_config = PPOConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
        log_with=None,
    )
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    env = PrenatalEnvironment()
    user_ids = [int(token.strip()) for token in args.user_ids.split(",") if token.strip()]
    dataset = build_dataset(env, user_ids)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for epoch in range(args.epochs):
        for start in range(0, len(dataset), args.batch_size):
            batch = dataset[start : start + args.batch_size]
            query_tensors, response_tensors, responses = generate_batch(
                trainer,
                tokenizer,
                batch,
                generation_kwargs,
            )

            rewards = []
            for sample, raw_response in zip(batch, responses):
                prompt_obs = env.reset(sample.user_id)
                try:
                    action = parse_llm_output(raw_response)
                    breakdown = calculate_reward(action.target, action.urgency, prompt_obs.observation)
                    rewards.append(torch.tensor(breakdown.reward, dtype=torch.float32, device=trainer.accelerator.device))
                except Exception:
                    rewards.append(torch.tensor(-20.0, dtype=torch.float32, device=trainer.accelerator.device))

            trainer.step(query_tensors, response_tensors, rewards)
            mean_reward = sum(float(reward.item()) for reward in rewards) / max(1, len(rewards))
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "batch_start": start,
                        "mean_reward": round(mean_reward, 4),
                        "sample_response": responses[0][:200] if responses else "",
                    }
                )
            )

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal TRL PPO script for the Niva prenatal OpenEnv environment.")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output-dir", default="./artifacts/niva-trl-ppo")
    parser.add_argument("--user-ids", default="1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--load-in-4bit", action="store_true")
    run_training(parser.parse_args())
