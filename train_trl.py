"""
Colab-friendly TRL PPO entrypoint for MAAS.

This module intentionally re-exports the primary multi-step PPO loop from
train_openenv_ppo.py so notebook users and existing scripts keep working while
the repo trains against the current MAAS environment contract.
"""

from train_openenv_ppo import build_prompt, create_arg_parser, run_training


if __name__ == "__main__":
    parser = create_arg_parser(
        "Colab-friendly TRL PPO script for the MAAS multi-step OpenEnv environment."
    )
    parser.set_defaults(output_dir="./artifacts/niva-trl-ppo")
    run_training(parser.parse_args())
