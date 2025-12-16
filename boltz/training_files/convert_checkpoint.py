#!/usr/bin/env python
"""
convert_checkpoint.py

This script loads your old Boltz checkpoint, filters out unexpected hyperparameters,
patches all vocabulary-dependent parameters (both 1D and 2D) using the new model’s state dict as a blueprint,
and then merges the patched parameters into a complete checkpoint dictionary.
This ensures that any new keys from your updated const.py are preserved.
The complete checkpoint (with extra keys preserved) is saved as a new file in the same weights folder.

Usage:
    python convert_checkpoint.py --old_ckpt /work/keshavsundar/env/boltz1x/weights/boltz1_conf.ckpt
"""

import argparse
import inspect
import math
import os
import torch
import torch.nn as nn
import omegaconf.dictconfig  # needed for safe globals
from dataclasses import dataclass, asdict

# Ensure we import our local `boltz` package (with sugar_trunk.py), not the installed one
import sys
local_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if local_src not in sys.path:
    sys.path.insert(0, local_src)

from boltz.model.model import Boltz1

# Allow safe global for DictConfig
torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])

@dataclass
class BoltzSteeringParams:
    """Steering parameters."""
    fk_steering: bool = True
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    guidance_update: bool = True
    num_gd_steps: int = 16

def filter_hparams(hparams):
    """
    Filters the hyperparameters so that only keys accepted by Boltz1.__init__ are retained.
    """
    init_params = inspect.signature(Boltz1.__init__).parameters
    valid_keys = set(init_params.keys())
    valid_keys.discard('self')
    filtered = {k: v for k, v in hparams.items() if k in valid_keys}
    return filtered

def patch_parameter(old_param, new_param, key):
    """
    Patches a parameter from the old model into the new parameter.
    Handles both 2D tensors (e.g. weight matrices) and 1D tensors (e.g. LayerNorm weights or biases).
    """
    if new_param.ndim == 2:
        out_features, old_dim = old_param.shape
        _, new_dim = new_param.shape
        new_param[:, :old_dim] = old_param
        if new_dim > old_dim:
            nn.init.kaiming_uniform_(new_param[:, old_dim:], a=math.sqrt(5))
        return new_param
    elif new_param.ndim == 1:
        old_dim = old_param.shape[0]
        new_dim = new_param.shape[0]
        new_param[:old_dim] = old_param
        if new_dim > old_dim:
            if key.endswith(".weight"):
                new_param[old_dim:] = 1.0
            elif key.endswith(".bias"):
                new_param[old_dim:] = 0.0
            else:
                new_param[old_dim:] = 0.0
        return new_param
    else:
        raise ValueError("Unsupported parameter dimension for key {}: {}".format(key, new_param.ndim))

def convert_checkpoint(old_ckpt_path, new_ckpt_path):
    # Load the old checkpoint using safe globals
    with torch.serialization.safe_globals([omegaconf.dictconfig.DictConfig]):
        ckpt = torch.load(old_ckpt_path, map_location="cpu", weights_only=False)

    # Extract hyperparameters saved by Lightning (under "hyper_parameters" or "hparams")
    hparams = ckpt.get("hyper_parameters", ckpt.get("hparams"))
    if hparams is None:
        raise ValueError("No hyperparameters found in the checkpoint; cannot instantiate new model.")
    
    # --- START OF FIX ---
    # Create a mutable copy of the hyperparameters to work with.
    candidate_hparams = dict(hparams)

    # Add default arguments required by the new model if they are not in the old checkpoint.
    # This prepares a "candidate" dictionary of parameters.
    if "glycan_bias_args" not in candidate_hparams:
        print("Adding default glycan_bias_args for the new, superior architecture.")
        candidate_hparams["glycan_bias_args"] = {
            "enabled": True,
            "params": {}  # This dictionary is now correctly empty
        }

    if "steering_args" not in candidate_hparams:
        print("Adding default steering_args to hyperparameters.")
        candidate_hparams["steering_args"] = asdict(BoltzSteeringParams())

    # Now, filter the candidate hyperparameters. This is the crucial step.
    # It ensures that only arguments accepted by the current Boltz1.__init__ are kept.
    # If the imported Boltz1 doesn't accept 'steering_args', it will be safely removed here.
    filtered_hparams = filter_hparams(candidate_hparams)
    # --- END OF FIX ---

    # Overwrite the checkpoint’s saved hyperparameters with the filtered ones
    # so that Lightning.load_from_checkpoint(...) only sees keys our __init__ expects.
    if "hyper_parameters" in ckpt:
        ckpt["hyper_parameters"] = filtered_hparams
    elif "hparams" in ckpt:
        ckpt["hparams"] = filtered_hparams

    # Instantiate a new model using the SAFE, filtered hyperparameters.
    # This will now succeed regardless of whether the imported Boltz1 accepts steering_args or not.
    new_model = Boltz1(**filtered_hparams)
    new_state_dict = new_model.state_dict()  # This is our blueprint, with new keys from const.py.

    # Get the old state dict (it might be nested under "state_dict")
    old_state_dict = ckpt.get("state_dict", ckpt)

    # For each key in the new model’s state dict, if it exists in the old checkpoint, patch or copy it.
    for key in new_state_dict:
        if key in old_state_dict:
            old_param = old_state_dict[key]
            new_param = new_state_dict[key]
            if old_param.shape != new_param.shape:
                print("Patching {}: old shape {}, new shape {}".format(key, old_param.shape, new_param.shape))
                patched = patch_parameter(old_param, new_param.clone(), key)
                new_state_dict[key] = patched
            else:
                new_state_dict[key] = old_param
        else:
            # The key is new (from updated const.py or architecture changes) – use the new model's initialization.
            print("New key {} not found in old checkpoint; using new initialization.".format(key))

    # Optionally, warn about keys in the old checkpoint that are not used by the new model.
    for key in old_state_dict:
        if key not in new_state_dict:
            print("Warning: key {} from old checkpoint is not used in the new model.".format(key))

    # Merge the patched new state dict back into the original checkpoint dictionary.
    if "state_dict" in ckpt:
        ckpt["state_dict"] = new_state_dict
    else:
        ckpt = new_state_dict

    # Save the complete (merged) checkpoint.
    torch.save(ckpt, new_ckpt_path)
    print("Converted checkpoint saved at:", new_ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Boltz checkpoint to support expanded token vocabulary."
    )
    parser.add_argument(
        "--old_ckpt",
        type=str,
        required=True,
        help="Path to the original checkpoint file (e.g. /work/keshavsundar/env/boltz_glycan/weights/boltz1_conf.ckpt)",
    )
    parser.add_argument(
        "--new_ckpt",
        type=str,
        default=None,
        help="Path to save the new checkpoint. If not provided, it will be saved in the same folder with '_converted' appended.",
    )
    args = parser.parse_args()

    old_ckpt_path = args.old_ckpt
    if args.new_ckpt is None:
        dirname, basename = os.path.split(old_ckpt_path)
        name, ext = os.path.splitext(basename)
        new_ckpt_path = os.path.join(dirname, "{}_converted{}".format(name, ext))
    else:
        new_ckpt_path = args.new_ckpt

    convert_checkpoint(old_ckpt_path, new_ckpt_path)
