import os
import sys

# 1. Register the global FFmpeg DLLs before ANY other imports
ffmpeg_path = r"C:\ffmpeg\bin"
if os.path.exists(ffmpeg_path):
    # This is the critical Windows-specific command
    os.add_dll_directory(ffmpeg_path)
    # Ensure subprocesses like MoviePy also see it
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

from audiox.interface.gradio import create_ui
import json 
import torch
from safetensors.torch import load_file

# Mapping for the UI to understand which config belongs to which weights
LOCAL_MODELS = {
    "AudioX": {
        "config": "model/config-audiox.json",
        "ckpt": "model/AudioX.safetensors"
    },
    "AudioX-MAF": {
        "config": "model/config-audiox-maf.json",
        "ckpt": "model/AudioX-MAF.safetensors"
    },
    "AudioX-MAF-MMDiT": {
        "config": "model/config-audiox-maf-mmdit.json",
        "ckpt": "model/AudioX-MAF-MMDiT.safetensors"
    }
}

def main(args):
    torch.manual_seed(42)

    # If no paths are provided via CLI, the UI will initialize with these defaults
    # but allow you to switch or browse within the interface.
    default_model = LOCAL_MODELS["AudioX-MAF-MMDiT"]
    
    model_config_path = args.model_config or default_model["config"]
    ckpt_path = args.ckpt_path or default_model["ckpt"]

    print(f"Launching Gradio with initial model: {ckpt_path}")

    interface = create_ui(
        model_config_path = model_config_path, 
        ckpt_path = ckpt_path, 
        pretrained_name = args.pretrained_name, 
        pretransform_ckpt_path = args.pretransform_ckpt_path or "model/VAE.safetensors",
        model_half = args.model_half
    )
    
    interface.queue()
    interface.launch(
        share=args.share, 
        auth=(args.username, args.password) if args.username is not None else None
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run AudioX Gradio Interface')
    parser.add_argument('--pretrained-name', type=str, help='Name of pretrained model', required=False)
    parser.add_argument('--model-config', type=str, help='Path to model config', required=False)
    parser.add_argument('--ckpt-path', type=str, help='Path to model checkpoint', required=False)
    parser.add_argument('--pretransform-ckpt-path', type=str, help='Path to VAE/Pretransform', required=False)
    parser.add_argument('--share', action='store_true', help='Create a public link', required=False)
    parser.add_argument('--username', type=str, help='Gradio username', required=False)
    parser.add_argument('--password', type=str, help='Gradio password', required=False)
    parser.add_argument('--model-half', action='store_true', help='Use half precision', required=False)
    
    args = parser.parse_args()
    main(args)