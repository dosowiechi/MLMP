import time
import copy
from functools import lru_cache

import torch
import torch.nn as nn
import torch.optim as optim


from ovss import load_ovss

CLS_EOS_ID = 49407
REFERENCE_PROMPT = 'a photo of a {}'


class TPT(nn.Module):
    """
    Test-Time Prompt Tuning (TPT) for open-vocabulary semantic segmentation (OVSS) models.

    Inspired by the official TPT repository (https://github.com/azshue/TPT), this module
    injects a small set of learnable soft prompt tokens into the CLIP text encoder and
    adapts them per test image via entropy minimization, keeping the backbone frozen.
    """

    def __init__(self, ovss_type, ovss_backbone, classes, lr=5e-3, n_ctx=4, steps=1, 
                 runtime_calculation=False, device= "cuda",
                 ):
        """
        Initialize the TPT adaptation module.

        Args:
            ovss_type (str): Identifier for the OVSS model to load.
            ovss_backbone (str): Name of the backbone architecture.
            classes (List[str]): Ordered list of class names for segmentation.
            lr (float, optional): Learning rate for soft-prompt optimizer. Defaults to 5e-3.
            n_ctx (int, optional): Number of learnable prompt tokens. Defaults to 4.
            steps (int, optional): Adaptation steps per test sample. Defaults to 1.
            runtime_calculation (bool, optional): Record runtimes if True. Defaults to False.
            device (str, optional): Compute device ("cpu" or "cuda"). Defaults to "cuda".
        """
        super().__init__()

        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr

        if classes is not None:
            self.classes = classes
        else:
            raise Exception("Classes are required in the init")
        
        assert n_ctx == 4, "The default hand‑crafted prompt has exactly 4 tokens. If you change n_ctx, be sure the prompt template length matches."
        self.n_ctx = n_ctx

        self.steps = steps
        self.prompt = REFERENCE_PROMPT
        self.runtime = runtime_calculation
        self.device = device

        # ---------- OVSS Model ----------
        self.model, self.tokenize = load_ovss(self.ovss_type, self.ovss_backbone, device=self.device)

        # ---------- learnable soft tokens (initialised from "a photo of a") ----------
        with torch.no_grad():
            tmpl_ids = self.tokenize("a photo of a")[0]            # (77,)
            tmpl_ids = tmpl_ids.to(device)
            ctx_ids = tmpl_ids[1:1 + n_ctx]                        # remove <SOS>, take next 4 ids
            init_ctx = self.model.token_embedding(ctx_ids).clone()  # 4 × D (D = 512/768)
        self.ctx = nn.Parameter(init_ctx.to(device), requires_grad=True)
        self.D = self.ctx.shape[1]

        # ---------- optimiser ----------
        self.optimizer = optim.Adam([self.ctx], lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)

        # ---------- save pristine state for reset() ----------
        self._clean_state = {
            "model": copy.deepcopy(self.model.state_dict()),
            "ctx": self.ctx.data.clone(),
            "optim": copy.deepcopy(self.optimizer.state_dict()),
        }

        # ---------- pre‑tokenise every class once ----------
        self._token_ids = {c: self.tokenize(self.prompt.format(c))[0] for c in self.classes}

        # ---------- define variables to store adaptation and evaluation duration ----------
        if self.runtime:
            self.adapt_times = []
            self.eval_times = []


    # ==================================================================
    # PRIVATE HELPERS
    # ==================================================================
    @lru_cache(maxsize=None)
    def _sos_rest_eos(self, cls_name: str):
        """Return (<SOS> ids, rest ids, eos_position_in_original_sentence)."""
        ids = self._token_ids[cls_name].clone()
        eos_idx = (ids == CLS_EOS_ID).nonzero(as_tuple=True)[0].item()
        sos = ids[:1]       # keep the first token (SOS)
        rest = ids[1:]      # rest incl. <EOS> & <PAD>
        return sos, rest, eos_idx

    def _build_text_features(self):
        """Encode class names using current prompt ➜ (C, D_proj)."""
        C = len(self.classes)
        device = self.device

        sos_list, body_list, eos_positions = [], [], []
        for cls in self.classes:
            sos, body, eos_idx = self._sos_rest_eos(cls)
            # Drop the first n_ctx tokens of the body (they were replaced)
            body_trimmed = body[self.n_ctx:]
            sos_list.append(sos)
            body_list.append(body_trimmed)
            eos_positions.append(eos_idx)  # position unchanged because we replaced, not inserted

        ids_sos = torch.stack(sos_list).to(device)             # C × 1
        ids_body = torch.stack(body_list).to(device)           # C × (76‑4) = 72

        emb_sos = self.model.token_embedding(ids_sos)           # C × 1 × D
        emb_body = self.model.token_embedding(ids_body)         # C × 72 × D

        # Broadcast soft prompt to all classes
        ctx = self.ctx.unsqueeze(0).expand(C, -1, -1)          # C × 4 × D

        # Concatenate: seq length = 1 + 4 + 72 = 77
        x = torch.cat([emb_sos, ctx, emb_body], dim=1)         # C × 77 × D

        # Add positional encodings (77 tokens available)
        x = x + self.model.positional_embedding

        # Transformer expects (seq, batch, dim)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x.half())
        x = x.permute(1, 0, 2)                                # C × 77 × D

        eos_idx_tensor = torch.tensor(eos_positions, device=device)
        x = x[torch.arange(C, device=device), eos_idx_tensor]  # C × D
        x = self.model.ln_final(x)
        x = x @ self.model.text_projection
        x = x / x.norm(dim=-1, keepdim=True)
        return x


    # ==================================================================
    # PUBLIC API
    # ==================================================================
    def reset(self):
        """Restore backbone, prompt, and optimiser to their initial states."""
        self.model.load_state_dict(self._clean_state["model"], strict=True)
        self.ctx.data.copy_(self._clean_state["ctx"])
        self.optimizer.load_state_dict(self._clean_state["optim"])

    @torch.no_grad()
    def evaluate(self, images):
        """
        Forward pass without adaptation.

        Args:
            images (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Per-class logits of shape (batch_size, num_classes, H, W).

        """
        t0 = time.time()
        images = images.to(self.device)
        img_feat = self.model.encode_image(images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = self._build_text_features()           # C×D
        logits = self.model.logit_scale.exp() * img_feat @ txt_feat.T

        logits = logits[:, 1:]
        patch_size = self.model.visual.patch_size
        w, h = images[0].shape[-2] // patch_size, images[0].shape[-1] // patch_size
        b_dim = logits.shape[0]
        out_dim = logits.shape[-1]
        
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h) # (batch_size, #class, W, H)

        # interpolate to original image size
        logits = nn.functional.interpolate(logits, size=(images[0].shape[-2], images[0].shape[-1]), mode='bilinear', align_corners=False)

        if self.runtime:
            self.eval_times.append(time.time() - t0)

        return logits                   

    def adapt(self, images):
        """
        Forward pass with adaptation.

        Args:
            images (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).

        Returns:
            List[float]: Loss values recorded at each adaptation iteration.
        """
        
        self.reset()
        loss_report = []
        images = images.to(self.device)
        t0 = time.time()
        for _ in range(self.steps):
            img_feat = self.model.encode_image(images)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = self._build_text_features()
            logits = self.model.logit_scale.exp() * img_feat @ txt_feat.T



            logits = logits[:, 1:]
            patch_size = self.model.visual.patch_size
            w, h = images[0].shape[-2] // patch_size, images[0].shape[-1] // patch_size
            b_dim = logits.shape[0]
            out_dim = logits.shape[-1]
            
            logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h) # (batch_size, #class, W, H)



            loss = self.softmax_entropy(logits).mean()
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.runtime:
            self.adapt_times.append(time.time() - t0)

        return loss_report


    @staticmethod
    def softmax_entropy(x: torch.Tensor, dim=-3) -> torch.Tensor:
        """Entropy of softmax distribution from logits.
            x : torch.Tensor : logits of shape (#templates, batch_size, num_classes, H, W)
        """
        return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)
