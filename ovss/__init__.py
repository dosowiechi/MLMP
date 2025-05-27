import ovss.clip as clip
from ovss.clip import tokenize as clip_tokenize


def load_ovss(ovss_type, ovss_backbone, device='cpu'):
    """
    Load the OVSS model based on the specified type and backbone.

    Args:
        ovss_type: Type of the OVSS model.
        ovss_backbone: Backbone architecture of the OVSS model.
        device: Device to load the model on (e.g., 'cpu' or 'cuda').

    Returns:
        ovss_model: Loaded OVSS model.
    """
    if ovss_type == 'clip':
        arch = "vanilla"
        attn_strategy = "vanilla"
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'sclip':
        arch = "vanilla"
        attn_strategy = "csa"
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'naclip':
        arch = "reduced"
        attn_strategy = "naclip"
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    else:
        raise ValueError(f"Unsupported OVSS type: {ovss_type}")

    return ovss_model, tokenize
