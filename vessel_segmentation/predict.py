import argparse
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from dataset import SegmentationDataset
from utils import dice_coeff
from models import Net


SEED = 777
np.random.seed(SEED)
torch.manual_seed(SEED)


def predict(data_dir: str | Path, model_path: str | Path):
    batch_size = 32

    # dataset
    ds = SegmentationDataset(data_dir)
    print(f"\ntotal number of images: {len(ds)}")

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(num_classes=1, fix_encoder=True).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"running on {device}")
    summary(model, input_size=(batch_size, 3, 512, 512))

    data_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    save_dir = Path(data_dir).joinpath(f"predictions_{model.name}")
    save_dir.mkdir(exist_ok=True)
    scores = []
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for b_idx, (imgs, gt_masks, indices) in pbar:
            preds = model(imgs.to(device))
            mask_preds = (F.sigmoid(preds) > 0.5)
            score = dice_coeff(mask_preds, gt_masks.to(device))
            scores.append(score.item())
            for i in range(len(imgs)):
                tifffile.imwrite(
                    save_dir.joinpath(f"{ds.image_files[indices[i]].stem}.tif"),
                    mask_preds[i].cpu().numpy().astype(np.uint8)
                )

    print(f"\naverage dice score: {np.array(scores).mean():.3f}\n")
    with open(Path(model_path).parent / "log.txt", "a") as f:
        f.write(f"\n\ntest average dice score: {np.array(scores).mean():.3f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="path to the test dataset directory."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="path to the model checkpoint."
    )

    args = parser.parse_args()

    predict(
        data_dir=args.dataset,
        model_path=args.model_path
    )
