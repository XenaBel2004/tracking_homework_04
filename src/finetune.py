import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_LOG_DIR,
    DEFAULT_METRICS_DIR,
    DEFAULT_MODEL_DIR,
    build_model,
    download_from_s3,
    evaluate,
    load_checkpoint,
    make_loaders,
    save_checkpoint,
    save_json,
    train_epoch,
    upload_to_s3,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-part", default="Train_2")
    parser.add_argument("--test-part", default="Test_2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--experiment-name", default="finetune")
    parser.add_argument("--input-model-path", default=f"{DEFAULT_MODEL_DIR}/base_model.pt")
    parser.add_argument("--output-model-path", default=f"{DEFAULT_MODEL_DIR}/finetuned_model.pt")
    parser.add_argument("--metrics-path", default=f"{DEFAULT_METRICS_DIR}/finetuned_metrics.json")
    parser.add_argument("--download-s3", action="store_true")
    parser.add_argument("--upload-s3", action="store_true")
    parser.add_argument("--input-s3-key", default="base_model.pt")
    parser.add_argument("--output-s3-key", default="finetuned_model.pt")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(DEFAULT_MODEL_DIR).mkdir(exist_ok=True)
    Path(DEFAULT_LOG_DIR).mkdir(exist_ok=True)
    Path(DEFAULT_METRICS_DIR).mkdir(exist_ok=True)

    if args.download_s3:
        download_from_s3(args.input_s3_key, args.input_model_path)

    params = vars(args).copy()
    params["device"] = str(device)

    writer = SummaryWriter(log_dir=f"{DEFAULT_LOG_DIR}/{args.experiment_name}")
    writer.add_text("params", json.dumps(params, ensure_ascii=False, indent=2))

    train_loader, test_loader = make_loaders(
        data_root=args.data_root,
        train_part=args.train_part,
        test_part=args.test_part,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(device)
    load_checkpoint(args.input_model_path, model, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    history = {"train": [], "test": []}
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        print(f"\nFinetune epoch {epoch}/{args.epochs}")
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device, desc="test_2")
        scheduler.step()

        history["train"].append(train_metrics)
        history["test"].append(test_metrics)

        for name, value in train_metrics.items():
            writer.add_scalar(f"finetune_train/{name}", value, epoch)
        for name, value in test_metrics.items():
            writer.add_scalar(f"finetune_test/{name}", value, epoch)

        print("finetune train:", train_metrics)
        print("finetune test:", test_metrics)

        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final_test_metrics = evaluate(model, test_loader, criterion, device, desc="final_test_2")
    writer.add_hparams(
        {"lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs},
        {
            "hparam/finetuned_test_loss": final_test_metrics["loss"],
            "hparam/finetuned_test_accuracy": final_test_metrics["accuracy"],
            "hparam/finetuned_test_f1": final_test_metrics["f1"],
        },
    )
    writer.close()

    output = {
        "params": params,
        "history": history,
        "final_test": final_test_metrics,
    }
    save_json(args.metrics_path, output)

    save_checkpoint(
        args.output_model_path,
        model,
        metadata={"params": params, "final_test": final_test_metrics},
    )

    if args.upload_s3:
        upload_to_s3(args.output_model_path, args.output_s3_key)

    print(f"Saved finetuned model to {args.output_model_path}")
    print(f"Saved finetuned metrics to {args.metrics_path}")


if __name__ == "__main__":
    main()
