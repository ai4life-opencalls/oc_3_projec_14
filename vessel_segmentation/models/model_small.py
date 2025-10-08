import torch

from .base import BaseModel
from .hiera_encoder import Encoder, EncoderSize
from .decoder import Decoder


class Net(BaseModel):
    def __init__(self, num_classes: int = 1, fix_encoder: bool = True):
        super().__init__()
        self.encoder = Encoder(EncoderSize.SMALL)
        self.decoder = Decoder(num_classes)
        self.fix_encoder = fix_encoder
        if fix_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.name = self.encoder.name + "_+_" + self.decoder.name

    def load_encoder_weights(self, state_dict: dict):
        # to match the keys
        keys = [f"encoder.{k}" for k in state_dict]
        state_dict = dict(zip(keys, state_dict.values()))
        self.encoder.load_state_dict(state_dict, strict=True)
        if self.fix_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features = features["backbone_fpn"]
        out = self.decoder(features[3], features[2], features[1], features[0])

        return out


if __name__ == "__main__":
    model = Net(num_classes=1, fix_encoder=True)
    print(model)
    print(model.name)
    print(f"Trainable parameters: {model.count_parameters() / 1e6:.2f}M")

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print(out.shape)
