import argparse
import json
from pathlib import Path

from src.common import save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-metrics", default="metrics/base_metrics.json")
    parser.add_argument("--finetuned-metrics", default="metrics/finetuned_metrics.json")
    parser.add_argument("--output", default="metrics/comparison.json")
    return parser.parse_args()


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    base = load(args.base_metrics)
    finetuned = load(args.finetuned_metrics)

    base_test = base["final_test"]
    finetuned_test = finetuned["final_test"]

    comparison = {
        "base_model_test": base_test,
        "finetuned_model_test": finetuned_test,
        "delta_finetuned_minus_base": {
            key: finetuned_test[key] - base_test[key]
            for key in base_test
            if key in finetuned_test
        },
    }

    save_json(args.output, comparison)

    print(json.dumps(comparison, ensure_ascii=False, indent=2))

    delta_f1 = comparison["delta_finetuned_minus_base"].get("f1")
    delta_acc = comparison["delta_finetuned_minus_base"].get("accuracy")
    print("\nВывод:")
    if delta_f1 is not None and delta_f1 >= 0:
        print(f"После дообучения F1 вырос на {delta_f1:.4f}.")
    elif delta_f1 is not None:
        print(f"После дообучения F1 снизился на {abs(delta_f1):.4f}.")

    if delta_acc is not None and delta_acc >= 0:
        print(f"Accuracy выросла на {delta_acc:.4f}.")
    elif delta_acc is not None:
        print(f"Accuracy снизилась на {abs(delta_acc):.4f}.")


if __name__ == "__main__":
    main()
