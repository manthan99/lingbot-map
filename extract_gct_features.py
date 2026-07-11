import os
import zarr
import torch
import random
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import csv

# Lingbot imports
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.models.gct_stream import GCTStream

# Config
DATASET_ROOT = Path("/home/manthan/navigation_repos/less-is-more/data/dataset/grandtour")
MISSIONS_CSV = Path("/home/manthan/navigation_repos/less-is-more/limo/configs/dataset/missions_split.csv")
OUT_DIR = Path("/home/manthan/navigation_repos/less-is-more/data/dataset/gct_features")
OUT_CSV = Path("/home/manthan/navigation_repos/less-is-more/limo/configs/dataset/limo_gct_subset_v3.csv")

MODEL_PATH = "/home/manthan/lingbot-map/models/lingbot-map-long.pt"

# Sampling config
SUBSAMPLE_RATIO = 0.1
HISTORY_FRAMES = 16
STRIDE = 5

def get_missions():
    missions = []
    with MISSIONS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            missions.append(row["Timestamp"].strip())
    return missions

def setup_model(device):
    print("Loading GCTStream model...")
    model = GCTStream(
        img_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=False,
    )
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    
    # Cast to bfloat16 for inference efficiency
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)
        
    return model, dtype

def main():
    random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    missions = get_missions()
    all_samples = []
    
    # Cache image_id arrays in memory to avoid reading Zarr in inner loop
    image_id_cache = {}
    seen_images = set()
    
    print("Discovering dataset samples...")
    for mission in missions:
        mission_dir = DATASET_ROOT / mission
        if not mission_dir.exists():
            continue
            
        for path_type in ["teleop_paths", "geometric_paths"]:
            zarr_path = mission_dir / "data" / path_type
            if not zarr_path.exists():
                continue
                
            try:
                z = zarr.open_group(str(zarr_path), mode="r")
                if "image_id" in z:
                    # Read the entire array into memory at once
                    ids_array = z["image_id"][:]
                    image_id_cache[(mission, path_type)] = ids_array
                    
                    length = len(ids_array)
                    # Gather all valid indices, but only for unique images
                    for idx in range(length):
                        img_id = ids_array[idx]
                        if (mission, img_id) not in seen_images:
                            seen_images.add((mission, img_id))
                            all_samples.append({
                                "mission": mission,
                                "path_type": path_type,
                                "idx": idx,
                            })
            except Exception as e:
                print(f"Error reading {zarr_path}: {e}")

    print(f"Total samples found: {len(all_samples)}")
    
    # Subsample 1/10th
    num_sampled = int(len(all_samples) * SUBSAMPLE_RATIO)
    sampled_indices = random.sample(range(len(all_samples)), num_sampled)
    selected_samples = [all_samples[i] for i in sampled_indices]
    
    # (Removed 5-sample testing limit)
    
    # Setup PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dtype = setup_model(device)
    
    csv_records = []
    
    print("Extracting features...")
    for sample in tqdm(selected_samples):
        mission = sample["mission"]
        path_type = sample["path_type"]
        idx = sample["idx"]
        
        out_filename = f"{mission}_{sample['path_type']}_{idx}.pt"
        out_filepath = OUT_DIR / out_filename
        
        # Add to CSV directly if it already exists (resume capability)
        if out_filepath.exists():
            csv_records.append({
                "Mission": mission,
                "PathType": sample["path_type"],
                "ZarrIndex": idx,
                "FeaturePath": str(out_filepath)
            })
            continue
        
        # Build history indices
        history_indices = [max(0, idx - (HISTORY_FRAMES - 1 - i) * STRIDE) for i in range(HISTORY_FRAMES)]
        
        # Get cached IDs array
        ids_array = image_id_cache[(mission, path_type)]
        image_ids = [ids_array[hi] for hi in history_indices]
        
        # Build image paths
        image_paths = [str(DATASET_ROOT / mission / "images" / "hdr_front" / f"{img_id:06d}.jpeg") for img_id in image_ids]
        
        # Verify paths
        valid = True
        for p in image_paths:
            if not os.path.exists(p):
                valid = False
                break
        if not valid:
            continue
            
        # Load and preprocess
        # load_and_preprocess_images returns [S, 3, H, W]
        images = load_and_preprocess_images(
            image_paths,
            mode="crop",
            image_size=518,
            patch_size=14,
        ).unsqueeze(0).to(device) # -> [1, 16, 3, 518, 378]
        
        # Clean KV cache before processing isolated chunks
        model.clean_kv_cache()
        
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            # Process all 16 frames in one block
            aggregated_tokens_list, patch_start_idx = model._aggregate_features(
                images,
                num_frame_for_scale=16,
                num_frame_per_block=16,
                causal_inference=True
            )
            initial_embeds = model.aggregator._last_embeds
            
            # aggregated_tokens_list is a list of 4 tensors, usually shaped [B, S, N, D]
            # We only want the last frame's tokens: S = -1
            final_frame_tokens = []
            for tokens in aggregated_tokens_list:
                # tokens shape: [1, 16, N, D]
                # Slice the last frame -> [1, N, D], then squeeze batch
                final_frame_tokens.append(tokens[:, -1].squeeze(0).to(torch.float16).cpu())
            
            # --- Optimized: Extract DINOv2 features and raw image for the last frame ---
            # initial_embeds shape: [B*S, P, C] -> [16, 931, 2048]
            # Last frame is the last element in the sequence dimension (S=16)
            last_frame_embeds = initial_embeds[-1] # [P, C]
            
            # Slice only the patch tokens (after camera/register/scale tokens)
            dinov2_features = last_frame_embeds[patch_start_idx:].to(torch.float16).cpu()
            
            # Raw image (as uint8 tensor to save space)
            last_frame_raw = (images[0, -1] * 255).to(torch.uint8).cpu()
            # -------------------------------------------------------------------------
            
            # Save to disk
            out_filename = f"{mission}_{sample['path_type']}_{idx}.pt"
            out_filepath = OUT_DIR / out_filename
            torch.save({
                "multi_scale_maps": final_frame_tokens,
                "patch_start_idx": patch_start_idx,
                "raw_image": last_frame_raw,
                "dinov2_features": dinov2_features
            }, out_filepath)
            
            csv_records.append({
                "Mission": mission,
                "PathType": sample["path_type"],
                "ZarrIndex": idx,
                "FeaturePath": str(out_filepath)
            })

    # Save CSV
    df = pd.DataFrame(csv_records)
    df.to_csv(OUT_CSV, index=False)
    print(f"Successfully processed {len(csv_records)} samples. CSV saved to {OUT_CSV}")

if __name__ == "__main__":
    main()
