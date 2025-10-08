import time
import datetime as dt
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from tqdm import tqdm

from dataset import SegmentationDataset
from utils import dice_loss, plot_losses, load_encoder_weights
from models import EncoderSize
from models import Net


SEED = 777
np.random.seed(SEED)
torch.manual_seed(SEED)


def train_model(data_dir: str | Path, batch_size: int = 16):
    # dataset
    ds = SegmentationDataset(data_dir)
    print(f"\ntotal number of images: {len(ds)}")
    val_ratio = 0.1
    n_val = int(len(ds) * val_ratio)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    print(
        f"number of train images: {len(train_ds)}\n"
        f"number of validation images: {len(val_ds)}"
    )

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    model = Net(num_classes=1, fix_encoder=True).to(device)
    # load pretrained encoder weights
    load_encoder_weights(model, EncoderSize.SMALL, device)

    # hyper-params
    num_epochs = 40
    lr = 4e-4

    summary(model, input_size=(batch_size, 3, 512, 512))

    # dataloader
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2
    )
    # optim & loss
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=5, threshold=1e-4
    )

    exp_name = f"{model.name}_{dt.datetime.now():%Y.%m.%d_%H.%M.%S}"
    save_dir = Path("checkpoints").joinpath(exp_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    # training
    losses = []
    epoch_loss = 0
    val_losses = []
    best_val_loss = 9999
    best_epoch = 0
    old_lr = lr
    tic = time.perf_counter()
    print("\nstart training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        if scheduler.get_last_lr() != old_lr:
            print(f"lr: {scheduler.get_last_lr()}")
            old_lr = scheduler.get_last_lr()

        pbar_train = tqdm(
            total=n_train, desc=f"Epoch {epoch:02}/{num_epochs}", unit="img"
        )
        with pbar_train:
            for imgs, gt_masks, _ in train_loader:
                preds = model(imgs.to(device))
                loss = criterion(preds, gt_masks.to(device))
                loss += dice_loss(preds, gt_masks.to(device))
                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.item())
                pbar_train.set_postfix(loss=loss.item())
                pbar_train.update(imgs.shape[0])
            # epoch average loss
            epoch_loss = np.mean(losses[-len(train_loader):])
            pbar_train.set_postfix(loss=epoch_loss)

            # validation
            model.eval()
            pbar_val = tqdm(total=n_val, desc="Validation", unit="img", leave=False)
            with pbar_val:
                with torch.no_grad():
                    for imgs, gt_masks, _ in val_loader:
                        preds = model(imgs.to(device))
                        val_loss = criterion(preds, gt_masks.to(device))
                        val_loss += dice_loss(preds, gt_masks.to(device))
                        val_losses.append(val_loss.item())
                        pbar_val.set_postfix(loss=val_loss.item())
                        pbar_val.update(imgs.shape[0])
            # schedule lr based on average validation loss and save the best model
            epoch_val_loss = np.array(val_losses).reshape(
                -1, len(val_loader)).mean(axis=1)[-1]
            scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                torch.save(model.state_dict(), save_dir / "model_best.pth")
                best_epoch = epoch
                best_val_loss = epoch_val_loss
            pbar_train.set_postfix(train=epoch_loss, val=epoch_val_loss)
        # end of an epoch
        plot_losses(
            losses, len(train_loader),
            val_losses, len(val_loader),
            save_dir
        )
    # end of training
    toc = time.perf_counter()
    print(f"\ntraining was finished in {(toc - tic) / 60:.1f} minutes.")
    print(f"the best model was epoch {best_epoch}.\n")
    print(f"last lr: {scheduler.get_last_lr()}\n")

    torch.save(model.state_dict(), save_dir / "model_last.pth")

    plot_losses(
        losses, len(train_loader),
        val_losses, len(val_loader),
        save_dir
    )
    # save losses & log
    np.save(save_dir / "train_loss.npy", losses)
    np.save(save_dir / "val_loss.npy", val_losses)
    with open(save_dir / "log.txt", "w") as f:
        f.write(f"data_dir: {data_dir}\n")
        f.write(f"random seed: {SEED}\n")
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"learning_rate: {lr}\n")
        f.write("optim: AdamW (1e-3)\n")
        f.write(f"scheduler: {scheduler.state_dict()}\n")
        f.write(f"model: {model.name}\n")
        f.write("criterion: BCEWithLogitsLoss + DiceLoss\n")
        f.write(f"last epoch loss: {epoch_loss:.4f}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_loss: {best_val_loss:.4f}\n")
        f.write(f"\ntraining was finished in {(toc - tic) / 60:.1f} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="path to the train dataset directory."
    )
    parser.add_argument(
        "--bsize", type=int, default=16,
        help="batch size for training (default: 16)."
    )
    args = parser.parse_args()
    bsize = args.bsize
    data_dir = Path(args.dataset)

    train_model(
        data_dir=data_dir,
        batch_size=bsize
    )
