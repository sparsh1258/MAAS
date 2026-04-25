from __future__ import annotations

import json
from pathlib import Path


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()],
    }


NOTEBOOK = {
    "cells": [
        markdown_cell(
            """
            # MAAS Deep Policy Notebook

            This notebook trains a **MAAS-specific deep temporal policy model** over structured prenatal observations and short daily-history sequences.

            It is designed to be directly usable with the MAAS project:
            - it reads the MAAS `prenatal.db` when available
            - it uses the same observation semantics as `environment.py`
            - it predicts the same action schema used by the app: `condition`, `urgency`, `rationale`
            - it can export a checkpoint that can later be wired into the backend

            The model is intentionally more ambitious than the lightweight rule baselines:
            - Transformer encoder over 3-step maternal history
            - gated static/temporal feature fusion
            - multi-task heads for condition, urgency, danger escalation, and latent complication signals
            """
        ),
        markdown_cell(
            """
            ## 1. Install Dependencies

            If you run this in Colab or a fresh Jupyter environment, install the core packages first.
            """
        ),
        code_cell(
            """
            %pip install torch numpy pillow sqlalchemy pydantic fastapi -q
            print("Dependencies ready")
            """
        ),
        markdown_cell(
            """
            ## 2. Point Python at the MAAS repo
            """
        ),
        code_cell(
            """
            import sys
            from pathlib import Path

            REPO_ROOT = Path.cwd()
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))

            print("Repo root:", REPO_ROOT)
            print("prenatal.db exists:", (REPO_ROOT / "prenatal.db").exists())
            """
        ),
        markdown_cell(
            """
            ## 3. Import the MAAS deep policy stack
            """
        ),
        code_cell(
            """
            from maas_deep_policy import (
                load_seed_cases,
                predict_for_user_id,
                render_training_curve,
                save_checkpoint,
                save_history,
                train_model,
            )

            seed_cases = load_seed_cases(REPO_ROOT / "prenatal.db")
            print("Loaded seed cases:", len(seed_cases))
            print("Example sources:", [seed.source_id for seed in seed_cases[:5]])
            """
        ),
        markdown_cell(
            """
            ## 4. Configure a MAAS experiment

            Increase `num_samples` and `epochs` when you move to a GPU runtime.
            """
        ),
        code_cell(
            """
            EXPERIMENT = {
                "num_samples": 4096,
                "epochs": 14,
                "batch_size": 128,
                "learning_rate": 0.002,
                "weight_decay": 1e-4,
                "seed": 42,
                "db_path": str(REPO_ROOT / "prenatal.db"),
            }
            EXPERIMENT
            """
        ),
        markdown_cell(
            """
            ## 5. Train the deep policy model
            """
        ),
        code_cell(
            """
            artifacts = train_model(**EXPERIMENT)
            history = artifacts["history"]
            history[-1]
            """
        ),
        markdown_cell(
            """
            ## 6. Save checkpoint, metrics, and training-curve image
            """
        ),
        code_cell(
            """
            OUTPUT_DIR = REPO_ROOT / "artifacts" / "maas_deep_policy_notebook"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            checkpoint_path = save_checkpoint(
                model=artifacts["model"],
                region_to_idx=artifacts["region_to_idx"],
                output_dir=OUTPUT_DIR,
                config=artifacts["config"],
            )
            history_path = save_history(history, OUTPUT_DIR / "training_history.json")
            curve_path = render_training_curve(history, OUTPUT_DIR / "training_curve.png")

            print("Checkpoint:", checkpoint_path)
            print("History:", history_path)
            print("Curve:", curve_path)
            """
        ),
        markdown_cell(
            """
            ## 7. Display the training curve
            """
        ),
        code_cell(
            """
            from IPython.display import Image, display

            display(Image(filename=str(curve_path)))
            """
        ),
        markdown_cell(
            """
            ## 8. Run a real MAAS-style prediction

            This uses a real patient from `prenatal.db` and returns the same schema the MAAS backend expects.
            """
        ),
        code_cell(
            """
            prediction = predict_for_user_id(
                user_id=2,
                checkpoint_path=checkpoint_path,
                db_path=REPO_ROOT / "prenatal.db",
            )
            prediction
            """
        ),
        markdown_cell(
            """
            ## 9. What this model gives MAAS

            - A trainable structured policy instead of only rules
            - A checkpoint that can be loaded for backend inference
            - A richer representation over short maternal history sequences
            - A multi-head design that can support escalation confidence and latent-risk monitoring
            """
        ),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    output_path = Path("MAAS_Deep_Policy_Training.ipynb")
    output_path.write_text(json.dumps(NOTEBOOK, indent=2), encoding="utf-8")
    print(f"Wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
