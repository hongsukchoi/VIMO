import os
import torch
from yacs.config import CfgNode as CN
from .hmr_vimo import HMR_VIMO


def get_default_config(cfg_path=None):
    if cfg_path is None:
        cfg_file = os.path.join(
            os.path.dirname(__file__),
            '../config_vimo.yaml'
        )
    else:
        cfg_file = cfg_path

    cfg = CN()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file)
    return cfg


def get_hmr_vimo(checkpoint=None, cfg_path=None, device='cuda'):
    cfg = get_default_config(cfg_path)
    cfg.device = device
    model = HMR_VIMO(cfg)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location='cpu')
        _ = model.load_state_dict(ckpt['model'], strict=False)

    model = model.to(device)
    _ = model.eval()

    return model

