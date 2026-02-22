import gc
import platform
import os
import subprocess as sp
import numpy as np
import gradio as gr
import soundfile as sf
from scipy.io import wavfile
import json 
import torch
import torchaudio
import torchvision
import decord
from decord import VideoReader, cpu
import math
import einops
import torchvision.transforms as transforms
from PIL import Image

from aeiou.viz import audio_spectrogram_image
from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ..inference.generation import generate_diffusion_cond, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict
from ..inference.utils import prepare_audio
from ..training.utils import copy_state_dict
from ..data.utils import read_video, merge_video_audio

# --- GLOBAL STATE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_model = None
current_model_name = None
current_model_config = None
current_sample_rate = None
current_sample_size = None

# Local File Mapping based on your directory structure
LOCAL_MODELS = {
    "AudioX-MAF-MMDiT": {
        "config": "model/config-audiox-maf-mmdit.json",
        "ckpt": "model/AudioX-MAF-MMDiT.safetensors"
    },
    "AudioX-MAF": {
        "config": "model/config-audiox-maf.json",
        "ckpt": "model/AudioX-MAF.safetensors"
    },
    "AudioX": {
        "config": "model/config-audiox.json",
        "ckpt": "model/AudioX.safetensors"
    }
}

# --- MODEL LOADING LOGIC ---

def load_local_model(model_name, vae_path):
    global current_model, current_model_name, current_model_config, current_sample_rate, current_sample_size
    
    if model_name not in LOCAL_MODELS:
        return f"Error: {model_name} configuration not found."
    
    paths = LOCAL_MODELS[model_name]
    
    try:
        # Load Config
        with open(paths["config"], 'r') as f:
            config = json.load(f)
        
        # We manually handle VAE loading to avoid the '__init__' keyword argument error.
        # We temporarily remove the VAE path from config so create_model_from_config
        # doesn't try to load it using the old, broken keyword.
        vae_path_from_config = config.pop("pretransform_ckpt_path", None)
        
        print(f"Initializing {model_name} architecture...")
        model = create_model_from_config(config)
        
        # Load Main Model Weights
        print(f"Loading weights from {paths['ckpt']}...")
        model.load_state_dict(load_ckpt_state_dict(paths["ckpt"]), strict=False)

        # Manually Load VAE Weights
        # Use the path from the UI if provided, otherwise fall back to config path
        final_vae_path = vae_path if (vae_path and os.path.exists(vae_path)) else vae_path_from_config
        
        if final_vae_path and os.path.exists(final_vae_path):
            print(f"Loading VAE weights from {final_vae_path}...")
            vae_state = load_ckpt_state_dict(final_vae_path)
            model.pretransform.load_state_dict(vae_state, strict=False)

        model.to(device).eval().requires_grad_(False)
        
        # Update Global State
        current_model = model
        current_model_name = model_name
        current_model_config = config
        current_sample_rate = config["sample_rate"]
        current_sample_size = config["sample_size"]
        
        return f"✅ {model_name} loaded successfully on {device}."
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"❌ Error loading model: {str(e)}"

# --- GENERATION WRAPPER ---

def run_generation(prompt, video_file, steps, cfg_scale, seed, sampler):
    global current_model, current_sample_rate, current_sample_size
    
    if current_model is None:
        return None, "⚠️ Please load a model first using the settings on the left."

    # Clear cache for the 5090
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    # Setup sync features and video tensors
    sync_features = torch.zeros(1, 240, 768).to(device)
    target_fps = current_model_config.get("video_fps", 5)
    video_tensor = torch.zeros(int(target_fps * 10), 3, 224, 224) 
    
    if video_file:
        video_tensor = read_video(video_file.name, target_fps=target_fps)

    conditioning = [{
        "video_prompt": {"video_tensors": video_tensor.unsqueeze(0), "video_sync_frames": sync_features},        
        "text_prompt": prompt or "",
        "audio_prompt": torch.zeros((2, int(current_sample_rate * 10))).unsqueeze(0),
        "seconds_start": 0,
        "seconds_total": current_sample_size / current_sample_rate
    }]

    print(f"Generating with {current_model_name}...")
    output = generate_diffusion_cond(
        current_model, conditioning=conditioning, steps=int(steps), cfg_scale=cfg_scale, 
        sample_size=current_sample_size, sample_rate=current_sample_rate, 
        seed=int(seed) if seed != "-1" else np.random.randint(0, 100000), 
        device=device, sampler_type=sampler
    )

    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    
    os.makedirs("demo_result", exist_ok=True)
    out_path = "demo_result/result.wav"
    
    torchaudio.save(out_path, output.cpu(), current_sample_rate)
    
    return out_path, f"Generation complete using {current_model_name}."

# --- UI BUILDER ---

def create_ui(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False):
    with gr.Blocks(title="AudioX Local Controller") as ui:
        gr.Markdown("# 🎧 AudioX Local UI")
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### ⚙️ Model Settings")
                    model_drop = gr.Dropdown(choices=list(LOCAL_MODELS.keys()), value="AudioX-MAF-MMDiT", label="Model Selection")
                    vae_text = gr.Textbox(value="model/VAE.safetensors", label="VAE Path")
                    load_btn = gr.Button("🔄 Load / Switch Model", variant="primary")
                    load_status = gr.Label(label="System Status")
            
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 🪄 Generation")
                    prompt_text = gr.Textbox(label="Text Prompt", placeholder="Describe the sound you want to generate...")
                    video_up = gr.File(label="Optional Video Input")
                    
                    with gr.Row():
                        steps_slide = gr.Slider(1, 500, 100, step=1, label="Steps")
                        cfg_slide = gr.Slider(0, 20, 7, step=0.5, label="CFG Scale")
                    
                    seed_text = gr.Textbox(label="Seed (-1 for random)", value="-1")
                    sampler_drop = gr.Dropdown(["dpmpp-3m-sde", "k-heun", "k-lms"], value="dpmpp-3m-sde", label="Sampler")
                    
                    gen_btn = gr.Button("🔥 Generate Audio", variant="primary")
                    audio_out = gr.Audio(label="Resulting Audio")

        # UI Interactions
        load_btn.click(fn=load_local_model, inputs=[model_drop, vae_text], outputs=load_status)
        gen_btn.click(fn=run_generation, inputs=[prompt_text, video_up, steps_slide, cfg_slide, seed_text, sampler_drop], outputs=[audio_out, load_status])
        
    return ui