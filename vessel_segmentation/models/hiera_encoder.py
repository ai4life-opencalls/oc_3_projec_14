from enum import Enum

import torch

from .base import BaseModel
from .sam2_src.hieradet import Hiera
from .sam2_src.position_encoding import PositionEmbeddingSine
from .sam2_src.image_encoder import FpnNeck, ImageEncoder


class EncoderSize(Enum):
    TINY = "tiny"
    SMALL = "small"
    BASE = "base+"
    LARGE = "large"


def get_config(model_size: EncoderSize) -> dict:
    # base+ config
    config = {
        "name": model_size.value,
        "embed_dim": 112,
        "num_heads": 2,
        "d_model": 256,
        "backbone_channel_list": [896, 448, 224, 112],
        "fpn_top_down_levels": [2, 3],
        "fpn_interp_model": "nearest",
        "stages": (2, 3, 16, 3),
        "global_att_blocks": (7, 10, 13),
        "window_pos_embed_bkg_spatial_size": (14, 14),
        "window_spec": (8, 4, 14, 7),
    }
    if model_size == EncoderSize.TINY:
        config.update({
            "embed_dim": 96,
            "num_heads": 1,
            "stages": (1, 2, 7, 2),
            "global_att_blocks": (5, 7, 9),
            "backbone_channel_list": [768, 384, 192, 96]
        })
    elif model_size == EncoderSize.SMALL:
        config.update({
            "embed_dim": 96,
            "num_heads": 1,
            "stages": (1, 2, 11, 2),
            "global_att_blocks": (7, 10, 13),
            "backbone_channel_list": [768, 384, 192, 96],
            "window_pos_embed_bkg_spatial_size": (7, 7),
        })
    elif model_size == EncoderSize.LARGE:
        config.update({
            "embed_dim": 144,
            "num_heads": 2,
            "stages": (2, 6, 36, 4),
            "global_att_blocks": (23, 33, 43),
            "window_spec": (8, 4, 16, 8),
            "backbone_channel_list": [1152, 576, 288, 144],
            "window_pos_embed_bkg_spatial_size": (7, 7),
        })
    return config


class Encoder(BaseModel):
    def __init__(self, model_size: EncoderSize = EncoderSize.SMALL):
        super().__init__()
        config = get_config(model_size)
        self.name = "hiera_" + config["name"]

        trunk = Hiera(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            stages=config["stages"],
            global_att_blocks=config["global_att_blocks"],
            window_pos_embed_bkg_spatial_size=config["window_pos_embed_bkg_spatial_size"],
            window_spec=config["window_spec"]
        )
        position_encoding = PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000
        )
        neck = FpnNeck(
            position_encoding=position_encoding,
            d_model=config["d_model"],
            backbone_channel_list=config["backbone_channel_list"],
            fpn_top_down_levels=config["fpn_top_down_levels"],
            fpn_interp_model=config["fpn_interp_model"]
        )
        self.encoder = ImageEncoder(
            trunk=trunk,
            neck=neck,
            scalp=0  # to keep the lowest resolution feature
        )

    def forward(self, sample: torch.Tensor) -> dict:
        # output = {
        #     "vision_features": tensor,
        #     "vision_pos_enc": list[tensor],
        #     "backbone_fpn": list[tensor],
        # }

        return self.encoder(sample)
