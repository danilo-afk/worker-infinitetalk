"""Network volume diagnostics for worker-ltx-video."""
import os
import logging

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {
    ".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".onnx",
}

MODEL_DIRS = [
    "checkpoints", "clip", "clip_vision", "controlnet",
    "embeddings", "loras", "upscale_models", "vae", "unet",
    "text_encoders", "LLM", "latent_upscale_models",
]


def is_network_volume_debug_enabled():
    return os.environ.get("NETWORK_VOLUME_DEBUG", "false").lower() == "true"


def _format_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def run_network_volume_diagnostics():
    print("=" * 60)
    print("worker-ltx-video - Network Volume Diagnostics")
    print("=" * 60)

    extra_paths = "/comfyui/extra_model_paths.yaml"
    if os.path.exists(extra_paths):
        print(f"  [OK] {extra_paths} exists")
    else:
        print(f"  [X] {extra_paths} NOT found")

    vol_path = "/runpod-volume"
    if os.path.ismount(vol_path) or os.path.isdir(vol_path):
        print(f"  [OK] {vol_path} exists")
    else:
        print(f"  [X] {vol_path} NOT found")
        return

    models_path = os.path.join(vol_path, "models")
    if os.path.isdir(models_path):
        print(f"  [OK] {models_path} exists")
    else:
        print(f"  [X] {models_path} NOT found")
        return

    for model_dir in MODEL_DIRS:
        dir_path = os.path.join(models_path, model_dir)
        if not os.path.isdir(dir_path):
            print(f"  [-] {model_dir}/: not present")
            continue

        files = []
        for root, _, filenames in os.walk(dir_path):
            for fn in filenames:
                fp = os.path.join(root, fn)
                ext = os.path.splitext(fn)[1].lower()
                if ext in VALID_EXTENSIONS:
                    size = os.path.getsize(fp)
                    rel = os.path.relpath(fp, dir_path)
                    files.append((rel, size))

        if files:
            print(f"  [OK] {model_dir}/: {len(files)} model(s)")
            for rel, size in files:
                print(f"       - {rel} ({_format_size(size)})")
        else:
            print(f"  [-] {model_dir}/: empty")

    print("=" * 60)
