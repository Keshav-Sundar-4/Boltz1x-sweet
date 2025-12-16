#!/usr/bin/env python3
import torch
import os
import sys
import copy
import omegaconf.dictconfig

# This is necessary to allow PyTorch to unpickle OmegaConf objects saved in the checkpoint.
torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])


def create_inference_checkpoint(base_ckpt, trained_ckpt, output_path):
    """
    Creates a checkpoint using the ORIGINAL method.
    This method is suitable for inference/pretrained loading, as it only updates the model weights.
    It is NOT suitable for resuming training because the optimizer state will be mismatched.
    """
    print("\n" + "-" * 80)
    print(f"METHOD 1: Creating INFERENCE-ONLY checkpoint -> {os.path.basename(output_path)}")
    print("-" * 80)
    
    # Use a deep copy to ensure the original objects are not modified
    base_copy = copy.deepcopy(base_ckpt)
    
    # Extract state_dicts
    base_sd = base_copy.get("state_dict", base_copy)
    trained_sd = trained_ckpt.get("state_dict", trained_ckpt)
    
    # Merge every trained parameter into the base checkpoint's state_dict
    updated_keys = 0
    for key, tensor in trained_sd.items():
        if key in base_sd:
            updated_keys += 1
        base_sd[key] = tensor.clone()
        
    print(f"✅ Merged {len(trained_sd)} parameter tensors from the trained checkpoint.")
    print(f"   > Overwrote {updated_keys} existing keys in the base state_dict.")

    # Place the updated state_dict back into the checkpoint structure
    if "state_dict" in base_copy:
        base_copy["state_dict"] = base_sd
    else:
        base_copy = base_sd

    # Save the new checkpoint
    torch.save(base_copy, output_path)
    print(f"💾 Saved new INFERENCE checkpoint successfully.")


def create_resume_checkpoint(base_ckpt, trained_ckpt, output_path):
    """
    Creates a consistent, RESUMABLE checkpoint using the NEW, corrected method.
    This method uses the trained checkpoint as the foundation, ensuring the model weights,
    optimizer state, and scheduler state are all consistent and from the same training run.
    """
    print("\n" + "-" * 80)
    print(f"METHOD 2: Creating RESUMABLE checkpoint -> {os.path.basename(output_path)}")
    print("-" * 80)
    
    # Use the TRAINED checkpoint as the foundation. This is the critical step.
    new_checkpoint = copy.deepcopy(trained_ckpt)
    print("[Step 1] Initialized from the TRAINED checkpoint to preserve optimizer/scheduler state.")

    # Safely merge state_dicts to include any potentially missing frozen weights
    base_sd = base_ckpt.get("state_dict", base_ckpt)
    trained_sd = trained_ckpt.get("state_dict", trained_ckpt)
    
    final_sd = trained_sd.copy()
    added_from_base = 0
    for key, tensor in base_sd.items():
        if key not in final_sd:
            final_sd[key] = tensor
            added_from_base += 1
            
    new_checkpoint["state_dict"] = final_sd
    print(f"[Step 2] Merged model weights. Final state_dict contains {len(final_sd)} tensors.")
    if added_from_base > 0:
        print(f"         > Transferred {added_from_base} missing parameter(s) from the base checkpoint.")

    # Final verification of critical keys
    print("[Step 3] Verifying final checkpoint for resume-critical keys...")
    all_keys_present = True
    for key in ["epoch", "global_step", "state_dict", "optimizer_states", "lr_schedulers"]:
        if key in new_checkpoint and new_checkpoint[key] is not None:
            print(f"  ✅ {key}: Found")
        else:
            print(f"  ❌ {key}: MISSING! This checkpoint may not be resumable.")
            all_keys_present = False
    
    if not all_keys_present:
        print("\nWARNING: One or more critical keys for resuming were missing. Proceed with caution.", file=sys.stderr)

    # Save the new checkpoint
    torch.save(new_checkpoint, output_path)
    print(f"💾 Saved new RESUMABLE checkpoint successfully.")


if __name__ == "__main__":
    # 1) Define all file paths
    base_ckpt_path = "/work/keshavsundar/env/boltz1x/weights/boltz1_conf_converted.ckpt"
    trained_ckpt_path = "/work/keshavsundar/work_sundar/glycan_test/checkpoints/last.ckpt"
    
    # Define the two distinct output paths
    out_ckpt_path_inference = "/work/keshavsundar/env/boltz1x/weights/boltz1_glycan.ckpt"
    out_ckpt_path_resume = "/work/keshavsundar/env/boltz1x/weights/boltz1_glycan_resume.ckpt"

    print("=" * 80)
    print("Starting checkpoint creation process for both Inference and Resume...")
    print("=" * 80)

    # Ensure output directory exists
    output_dir = os.path.dirname(out_ckpt_path_inference)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2) Load source checkpoints once
    print(f"Loading base checkpoint from:\n  {base_ckpt_path}")
    base_checkpoint = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    
    print(f"Loading trained checkpoint from:\n  {trained_ckpt_path}")
    trained_checkpoint = torch.load(trained_ckpt_path, map_location="cpu", weights_only=False)

    # 3) Create the inference-only checkpoint using the old method
    create_inference_checkpoint(base_checkpoint, trained_checkpoint, out_ckpt_path_inference)

    # 4) Create the resumable checkpoint using the new method
    create_resume_checkpoint(base_checkpoint, trained_checkpoint, out_ckpt_path_resume)

    print("\n" + "=" * 80)
    print("✅✅✅ ALL TASKS COMPLETE ✅✅✅")
    print(f"Inference Checkpoint saved to: {out_ckpt_path_inference}")
    print(f"Resume Checkpoint saved to:    {out_ckpt_path_resume}")
    print("=" * 80)
