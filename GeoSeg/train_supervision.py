import time
import source
from tools.cfg import py2cfg
import os
import torch
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
import random


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    network = config.net

    params = 0
    for p in network.parameters():
        if p.requires_grad:
            params += p.numel()

    criterion = source.losses.CEWithLogitsLoss(weights=config.classes_wt)
    metric = source.metrics.IoU2()
    optimizer = torch.optim.Adam(network.parameters(), lr=config.lr)
    network_fout = f"{config.name}_s0_{criterion.name}"

    OUT_DIR = config.OUT_DIR
    OUT_DIR += network_fout  # path to save prediction images
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Model output name  :", network_fout)
    print("Number of parameters: ", params)

    if torch.cuda.device_count() > 1:
        print("Number of GPUs :", torch.cuda.device_count())
        network = torch.nn.DataParallel(network)
        optimizer = torch.optim.Adam(
            [dict(params=network.module.parameters(), lr=config.lr)]
        )

    start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_score = 0
    train_hist = []
    valid_hist = []

    for epoch in range(config.max_epoch):
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


"""     model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy='auto',
                         logger=logger)
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)
 """

if __name__ == "__main__":
    main()
