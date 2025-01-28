import time
import source
from tools.cfg import py2cfg
import os
import torch
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)

    network = config.net

    params = 0
    for p in network.parameters():
        if p.requires_grad:
            params += p.numel()

    criterion = source.losses.CEWithLogitsLoss(weights=config.classes_wt)
    metric = source.metrics.IoU2()
    optimizer = torch.optim.Adam(network.parameters())
    network_fout = f"{config.name}_s0_{criterion.name}"

    OUT_DIR = config.OUT_DIR
    OUT_DIR += network_fout  # path to save prediction images
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Model output name  :", network_fout)
    print("Number of parameters: ", params)

    start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_score = 0
    train_hist = []
    valid_hist = []

    for epoch in range(config.n_epochs):
        print(f"\nEpoch: {epoch + 1}")

        logs_train = source.runner.train_epoch(
            model=network,
            optimizer=optimizer,
            criterion=criterion,
            metric=metric,
            dataloader=config.train_loader,
            device=device,
        )

        logs_valid = source.runner.valid_epoch(
            model=network,
            criterion=criterion,
            metric=metric,
            dataloader=config.val_loader,
            device=device,
        )

        train_hist.append(logs_train)
        valid_hist.append(logs_valid)

        score = logs_valid[metric.name]

        if max_score < score:
            max_score = score
            torch.save(
                network.state_dict(),
                os.path.join(config.WEIGHT_DIR, f"{network_fout}.pth"),
            )
            print("Model saved!")

    end = time.time()
    print("Processing time:", end - start)


if __name__ == "__main__":
    main()
