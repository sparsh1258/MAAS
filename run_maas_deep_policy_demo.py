from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from maas_deep_policy import (
    predict_for_user_id,
    render_training_curve,
    save_checkpoint,
    save_history,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MAAS deep policy model and export a demo checkpoint."
    )
    parser.add_argument("--db-path", default="prenatal.db")
    parser.add_argument("--output-dir", default="artifacts/maas_deep_policy_demo_run")
    parser.add_argument("--publish-results-dir", default="results/maas_deep_policy_demo")
    parser.add_argument("--publish-model-path", default="trained_models/maas_deep_policy.pt")
    parser.add_argument("--num-samples", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--predict-users", default="1,2,3,4,5,6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    publish_results_dir = Path(args.publish_results_dir)
    publish_results_dir.mkdir(parents=True, exist_ok=True)
    publish_model_path = Path(args.publish_model_path)
    publish_model_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = train_model(
        num_samples=args.num_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        db_path=args.db_path,
        seed=args.seed,
    )

    checkpoint_path = save_checkpoint(
        model=artifacts["model"],
        region_to_idx=artifacts["region_to_idx"],
        output_dir=output_dir,
        config=artifacts["config"],
    )
    history_path = save_history(artifacts["history"], output_dir / "training_history.json")
    curve_path = render_training_curve(artifacts["history"], output_dir / "training_curve.png")

    predictions: dict[str, dict] = {}
    user_ids = [item.strip() for item in args.predict_users.split(",") if item.strip()]
    for user_id_text in user_ids:
        user_id = int(user_id_text)
        predictions[str(user_id)] = predict_for_user_id(
            user_id=user_id,
            checkpoint_path=checkpoint_path,
            db_path=args.db_path,
        )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "curve_path": str(curve_path),
        "final_epoch": artifacts["history"][-1],
        "predictions": predictions,
        "train_size": artifacts["train_size"],
        "val_size": artifacts["val_size"],
    }
    summary_path = output_dir / "demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    published_history_path = publish_results_dir / "training_history.json"
    published_curve_path = publish_results_dir / "training_curve.png"
    published_summary_path = publish_results_dir / "demo_summary.json"
    shutil.copy2(checkpoint_path, publish_model_path)
    shutil.copy2(history_path, published_history_path)
    shutil.copy2(curve_path, published_curve_path)

    published_summary = {
        **summary,
        "serving_checkpoint_path": str(publish_model_path),
        "published_history_path": str(published_history_path),
        "published_curve_path": str(published_curve_path),
        "published_summary_path": str(published_summary_path),
    }
    published_summary_path.write_text(
        json.dumps(published_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(published_summary, indent=2))
    print(f"\nSummary saved to {summary_path}")
    print(f"Published summary saved to {published_summary_path}")


if __name__ == "__main__":
    main()
