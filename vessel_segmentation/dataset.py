from pathlib import Path

import tifffile
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as tv2


class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_dir: Path | str,
        is_inference: bool = False,
        resize_to: int | tuple | None = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.is_inference = is_inference

        if self.is_inference:
            self.image_dir = self.data_dir
        else:
            self.image_dir = self.data_dir.joinpath("images")
        assert self.image_dir.exists()

        self.label_dir = self.data_dir.joinpath("labels")
        self.has_label = self.label_dir.exists()

        self.image_files = list(self.image_dir.glob("*.tif")) + list(
            self.image_dir.glob("*.tiff"))
        self.image_files = sorted(self.image_files, key=lambda p: p.name)
        self.image_transform = tv2.Compose([
            tv2.ToDtype(torch.float32, scale=True),
            tv2.Normalize(
                # mean=[0.86734857, 0.68390552, 0.82139413],
                mean=[0.485, 0.456, 0.406],  # SAM2
                # std=[0.12822252, 0.21418721, 0.12574021]
                std=[0.229, 0.224, 0.225]  # SAM2
            )
        ])
        if resize_to is not None:
            self.image_transform.transforms.insert(  # type: ignore
                0, tv2.Resize(resize_to, antialias=True)
            )
        self.label_transform = tv2.Compose([
            tv2.ToDtype(torch.float32, scale=False),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        img_file = self.image_files[index]
        img = torch.from_numpy(tifffile.imread(img_file)).contiguous()
        if img.shape[-1] in [3, 4]:
            # make the image channel first
            img = img.permute(2, 0, 1)
        img = self.image_transform(img)

        label = torch.zeros(1)
        if self.has_label:
            label = torch.from_numpy(
                tifffile.imread(self.label_dir.joinpath(img_file.name))
            ).unsqueeze(0).contiguous()
            label = self.label_transform(label)

        return img, label, index


if __name__ == "__main__":
    ds = SegmentationDataset(
        "../data/vessel_dataset/train"
    )
    print(f"number of images: {len(ds)}")
    img, label, index = ds.__getitem__(5)
    print(img.shape, img.dtype, label.shape, label.dtype, index)
