"""Helpers to read path and initialize models"""

import torch
from Demo3.config.configs import *

# --- HELPER FUNCTIONS ---
def resolve_path(path):
    if os.path.isabs(path): return path
    cwd_candidate = os.path.abspath(path)
    if os.path.exists(cwd_candidate): return cwd_candidate
    root_candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(root_candidate): return root_candidate
    return cwd_candidate

def choose_device(device_flag):
    if device_flag == 'cuda' or torch.cuda.is_available(): return torch.device('cuda')
    if device_flag == 'mps' or (
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()): return torch.device('mps')
    return torch.device('cpu')

def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    cleaned = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)

