import torch
from safetensors.torch import load_file
from torch.nn.utils import remove_weight_norm

def load_ckpt_state_dict(ckpt_path):
    """
    Load the state dict from either a .safetensors or .ckpt/.pth file.
    Safetensors are loaded directly; legacy PyTorch files are loaded via torch.load.
    """
    if ckpt_path.endswith(".safetensors"):
        # Safetensors are usually flat dictionaries
        state_dict = load_file(ckpt_path, device="cpu")
    else:
        # Legacy .ckpt or .pth files usually have a "state_dict" key
        raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(raw_state, dict) and "state_dict" in raw_state:
            state_dict = raw_state["state_dict"]
        else:
            state_dict = raw_state
            
    return state_dict

def remove_weight_norm_from_model(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            try:
                remove_weight_norm(module)
            except Exception:
                pass 
    return model

# Sampling functions copied from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/utils/utils.py 
# under MIT license. License can be found in LICENSES/LICENSE_META.txt

def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    """torch.multinomial with arbitrary number of dimensions."""
    if num_samples == 1:
        q = torch.empty_like(input).exponential_(1, generator=generator)
        return torch.argmax(input / q, dim=-1, keepdim=True).to(torch.int64)

    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output

def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Sample next token from top K values."""
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs *= (probs >= min_value_top_k).float()
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs, num_samples=1)
    return next_token

def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample next token from top P probabilities."""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def next_power_of_two(n):
    return 2 ** (n - 1).bit_length()

def next_multiple_of_64(n):
    return ((n + 63) // 64) * 64