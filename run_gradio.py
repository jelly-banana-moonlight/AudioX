from audiox.interface.gradio import create_ui
import json 
import os
import subprocess
import torch

MODEL_CONFIGS = {
    "AudioX": {
        "repo": "HKUSTAudio/AudioX",
        "config_url": "https://huggingface.co/HKUSTAudio/AudioX/resolve/main/config.json",
        "ckpt_url": "https://huggingface.co/HKUSTAudio/AudioX/resolve/main/model.ckpt"
    },
    "AudioX-MAF": {
        "repo": "HKUSTAudio/AudioX-MAF",
        "config_url": "https://huggingface.co/HKUSTAudio/AudioX-MAF/resolve/main/config.json",
        "ckpt_url": "https://huggingface.co/HKUSTAudio/AudioX-MAF/resolve/main/model.ckpt",
        "syncformer_url": "https://huggingface.co/HKUSTAudio/AudioX-MAF/resolve/main/synchformer_state_dict.pth"
    },
    "AudioX-MAF-MMDiT": {
        "repo": "HKUSTAudio/AudioX-MAF-MMDiT",
        "config_url": "https://huggingface.co/HKUSTAudio/AudioX-MAF-MMDiT/resolve/main/config.json",
        "ckpt_url": "https://huggingface.co/HKUSTAudio/AudioX-MAF-MMDiT/resolve/main/model.ckpt",
        "syncformer_url": "https://huggingface.co/HKUSTAudio/AudioX-MAF-MMDiT/resolve/main/synchformer_state_dict.pth"
    }
}

def download_file(url, output_path):
    """Download a file using wget if it doesn't exist."""
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}, skipping download.")
        return True
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading {url} to {output_path}...")
    try:
        result = subprocess.run(
            ["wget", url, "-O", output_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Successfully downloaded {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: wget not found. Please install wget or download the files manually.")
        return False

def setup_model(model_name):
    """Setup model by downloading config and checkpoint if needed."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    model_dir = "model"
    config_path = os.path.join(model_dir, "config.json")
    ckpt_path = os.path.join(model_dir, "model.ckpt")
    
    # Download config if needed
    if not os.path.exists(config_path):
        if not download_file(config["config_url"], config_path):
            raise RuntimeError(f"Failed to download config for {model_name}")
    
    # Download checkpoint if needed
    if not os.path.exists(ckpt_path):
        if not download_file(config["ckpt_url"], ckpt_path):
            raise RuntimeError(f"Failed to download checkpoint for {model_name}")
    
    # Download syncformer if needed (for AudioX-MAF and AudioX-MAF-MMDiT)
    if model_name in ["AudioX-MAF", "AudioX-MAF-MMDiT"]:
        syncformer_path = os.path.join(model_dir, "synchformer_state_dict.pth")
        if not os.path.exists(syncformer_path):
            if "syncformer_url" in config:
                if not download_file(config["syncformer_url"], syncformer_path):
                    raise RuntimeError(f"Failed to download syncformer for {model_name}")
            else:
                raise RuntimeError(f"syncformer_url not found in config for {model_name}")
    
    return config_path, ckpt_path

def main(args):
    torch.manual_seed(42)

    # Handle --model argument
    if args.model is not None:
        if args.model_config is not None or args.ckpt_path is not None:
            print("Warning: --model is specified, ignoring --model-config and --ckpt-path")
        model_config_path, ckpt_path = setup_model(args.model)
    else:
        model_config_path = args.model_config
        ckpt_path = args.ckpt_path

    interface = create_ui(
        model_config_path = model_config_path, 
        ckpt_path=ckpt_path, 
        pretrained_name=args.pretrained_name, 
        pretransform_ckpt_path=args.pretransform_ckpt_path,
        model_half=args.model_half
    )
    interface.queue()
    interface.launch(share=args.share, auth=(args.username, args.password) if args.username is not None else None)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run gradio interface')
    parser.add_argument('--model', type=str, help='Predefined model name (AudioX, AudioX-MAF, AudioX-MAF-MMDiT)', required=False, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--pretrained-name', type=str, help='Name of pretrained model', required=False)
    parser.add_argument('--model-config', type=str, help='Path to model config', required=False)
    parser.add_argument('--ckpt-path', type=str, help='Path to model checkpoint', required=False)
    parser.add_argument('--pretransform-ckpt-path', type=str, help='Optional to model pretransform checkpoint', required=False)
    parser.add_argument('--share', action='store_true', help='Create a publicly shareable link', required=False)
    parser.add_argument('--username', type=str, help='Gradio username', required=False)
    parser.add_argument('--password', type=str, help='Gradio password', required=False)
    parser.add_argument('--model-half', action='store_true', help='Whether to use half precision', required=False)
    args = parser.parse_args()
    
    # Validate arguments
    if args.model is None and args.model_config is None and args.ckpt_path is None and args.pretrained_name is None:
        parser.error("Either --model, --pretrained-name, or both --model-config and --ckpt-path must be specified")
    
    if args.model_config is not None and args.ckpt_path is None:
        parser.error("--ckpt-path must be specified when --model-config is provided")
    
    if args.ckpt_path is not None and args.model_config is None:
        parser.error("--model-config must be specified when --ckpt-path is provided")
    
    main(args)