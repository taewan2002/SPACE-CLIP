# utils/model_io.py

import os
import torch

def save_checkpoint(model, optimizer, epoch, filename, root="./checkpoints"):
    if not os.path.isdir(root):
        os.makedirs(root)

    fpath = os.path.join(root, filename)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
        fpath,
    )


def load_checkpoint(fpath, model, optimizer=None):
    ckpt = torch.load(fpath, map_location='cpu')
    ckpt_optimizer = ckpt.get("optimizer", None) if isinstance(ckpt, dict) else None
    if optimizer is None:
        optimizer = ckpt_optimizer
    elif ckpt_optimizer is not None:
        optimizer.load_state_dict(ckpt_optimizer)

    epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state[key.replace("module.", "", 1)] = value
        else:
            cleaned_state[key] = value

    model.load_state_dict(cleaned_state)
    return model, optimizer, epoch
