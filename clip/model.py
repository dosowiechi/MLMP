from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.arch, self.attn_strategy, self.gaussian_std = None, None, 0
        self.addition_cache = dict()

    # nonly: Neighbourhood Only, kk: KK-Similarity, csa: SCLIP, vanilla: CLIP
    def set_params(self, arch, attn_strategy, gaussian_std):
        assert arch in ['reduced', 'vanilla']
        assert attn_strategy in ['naclip', 'nonly', 'kk', 'csa', 'vanilla']
        assert attn_strategy != 'csa' or arch == 'vanilla'
        assert gaussian_std > 0 or attn_strategy not in ['naclip', 'nonly']
        self.arch, self.attn_strategy, self.gaussian_std = arch, attn_strategy, gaussian_std

    def forward(self, x: torch.Tensor, output_layers=(-1,), out_type="mean", return_vanilla_cls=False, weights=None):
        B, nc, w, h = x.shape
        n_patches = (w // self.patch_size, h // self.patch_size)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        # convert negative indices to positive if applicable
        num_layers = len(self.transformer.resblocks)        
        output_layers = tuple(num_layers + idx if idx < 0 else idx for idx in output_layers)
        last_layer_idx = max(output_layers)

        out_features = [] 
        for idx, blk in enumerate(self.transformer.resblocks[:last_layer_idx+1]):
            blk = self.transformer.resblocks[idx]
            if idx != last_layer_idx:
                # we always use the vanilla attention for the intermediate layers beacuse in one experiment if we feed any other attention the performance of the next layers drops
                reduced = self.custom_attn("vanilla", blk.attn, blk.ln_1(x), n_patches) ## for the purpose of evaluation we can compute 2 types, one for feedforward and one for the evaluation
                x = x + reduced
                x = x + blk.mlp(blk.ln_2(x))

            else: 
                # Note that the att_strategy only will be applied to the last layer not intermediate layers
                reduced = self.custom_attn(self.attn_strategy, blk.attn, blk.ln_1(x), n_patches)
                final_x = x + reduced
                final_x = final_x + blk.mlp(blk.ln_2(final_x))
                if self.attn_strategy != 'vanilla' and return_vanilla_cls:
                    # with torch.no_grad():
                    vanilla_cls = blk(x)[0]

                    

            # Append the features based on self.arch for all output_layers
            if idx in output_layers:
                if self.arch == 'reduced':
                    out_features.append(reduced)
                elif self.arch == 'vanilla': #TODO: check if this is correct
                    out_features.append(final_x)
                else:
                    raise NotImplemented(f'arch {self.arch} is not implemented')
                
        if out_type=="mean": # simple mean
            if len(out_features) > 1:
                # Compute the final output as the average of the selected reduced features
                x = torch.mean(torch.stack(out_features), dim=0)
            else:
                x = out_features[0]

            # # replace vanilla cls token with the final_cls
            # if self.attn_strategy != 'vanilla' and return_vanilla_cls:
            #     x = torch.cat([vanilla_cls, x[1:]], dim=0)

            x = x.permute(1, 0, 2)  # LND -> NLD

            if return_vanilla_cls:
                return self.ln_post(x) @ self.proj, self.ln_post(vanilla_cls) @ self.proj
            else:
                return self.ln_post(x) @ self.proj
        
        elif out_type=="residual_mean":
            if len(out_features) > 1:
                # Compute the final output as the average of the selected reduced features
                x_last = out_features[-1]
                x_inter = torch.stack(out_features[:-1], dim=0)
                x_inter = torch.mean(x_inter, dim=0)
                x = (x_last + x_inter) / 2
            else:
                x = out_features[0]

            # # replace vanilla cls token with the final_cls
            # if self.attn_strategy != 'vanilla' and return_vanilla_cls:
            #     x = torch.cat([vanilla_cls, x[1:]], dim=0)

            x = x.permute(1, 0, 2)

            if return_vanilla_cls:  
                return self.ln_post(x) @ self.proj, self.ln_post(vanilla_cls) @ self.proj
            else:
                return self.ln_post(x) @ self.proj

        elif out_type=="all":
            out_features = [x.permute(1, 0, 2) for x in out_features]
            out_features = [self.ln_post(x) @ self.proj for x in out_features]
            # stack out_features on a new dim
            out_features = torch.stack(out_features, dim=0)
            if return_vanilla_cls:
                return out_features, self.ln_post(vanilla_cls) @ self.proj
            else:
                return out_features
        

        elif out_type=="weighted_mean":
            len(weights) == len(out_features)
            # Compute the final output as the weighted average of the selected reduced features
            x = torch.stack(out_features, dim=0)
            x = torch.sum(x * weights.unsqueeze(1).unsqueeze(-1), dim=0)
            x = x.permute(1, 0, 2)  # LND -> NLD


            return self.ln_post(x) @ self.proj

        elif out_type=="residual_weighted_mean":
            len(weights) == len(out_features) - 1
            # Compute the final output as the weighted average of the selected reduced features

            x_last = out_features[-1]

            x_inter = torch.stack(out_features[:-1], dim=0)
            x_inter = torch.sum(x_inter * weights.unsqueeze(1).unsqueeze(-1), dim=0)

            # mean
            x = (x_last + x_inter) / 2

            x = x.permute(1, 0, 2)  # LND -> NLD


            return self.ln_post(x) @ self.proj



    @staticmethod
    def gaussian_window(dim1, dim2, std=1.):
        constant = 1 / (std * math.sqrt(2))
        ks = list()
        for dim in [dim1, dim2]:
            start = -(dim - 1) / 2.0
            k = torch.linspace(start=start * constant,
                               end=(start + (dim - 1)) * constant,
                               steps=dim,
                               dtype=torch.float)
            ks.append(k)
        dist_square_to_mu = (torch.stack(torch.meshgrid(*ks, indexing='ij')) ** 2).sum(0)
        return torch.exp(-dist_square_to_mu)

    @staticmethod
    def get_attention_addition(dim1, dim2, window, adjust_for_cls=True):
        m = torch.einsum('ij,kl->ijkl', torch.eye(dim1), torch.eye(dim2))
        m = m.permute((0, 3, 1, 2)).contiguous()  # m[ijkl] = 1 iff (i, j) == (k, l)
        out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1), window.unsqueeze(0).unsqueeze(1), padding='same').squeeze(1)
        out = out.view(dim1 * dim2, dim1 * dim2)
        if adjust_for_cls:
            v_adjusted = torch.vstack([torch.zeros((1, dim1 * dim2)), out])
            out = torch.hstack([torch.zeros((dim1 * dim2 + 1, 1)), v_adjusted])
        return out

    def custom_attn(self, attn_strategy, attn_layer, x, n_patches, return_attn=False, with_attn=False):
        num_heads = attn_layer.num_heads
        num_tokens, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if attn_strategy in ['naclip', 'nonly']:
            addition = self.addition_cache.get(n_patches)
            if addition is None:
                window_size = [side * 2 - 1 for side in n_patches]
                window = VisionTransformer.gaussian_window(*window_size, std=self.gaussian_std)
                addition = VisionTransformer.get_attention_addition(*n_patches, window).unsqueeze(0).to(x.dtype).to(
                    x.device)
                self.addition_cache[n_patches] = addition

            if attn_strategy == 'naclip':
                attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale
                omega = addition
            elif attn_strategy == 'nonly':
                attn_weights = torch.zeros((num_heads, num_tokens, num_tokens)).to(x.dtype).to(x.device)
                omega = addition * (scale * torch.einsum('hop,hPO->hpP', q.norm(dim=2).unsqueeze(1),
                                                         k.norm(dim=2).unsqueeze(2)).mean().item())
            else:
                raise NotImplemented

            attn_weights += omega
            attn_weights = F.softmax(attn_weights, dim=-1)

        elif attn_strategy == 'csa':
            q_attn = torch.bmm(q, q.transpose(1, 2)) * scale
            k_attn = torch.bmm(k, k.transpose(1, 2)) * scale
            attn_weights = F.softmax(q_attn, dim=-1) + F.softmax(k_attn, dim=-1)
        elif attn_strategy == 'vanilla':
            attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif attn_strategy == 'kk':
            attn_weights = torch.bmm(k * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            raise NotImplemented(f'attn_strategy {self.attn_strategy} is not implemented')

        if return_attn:
            return attn_weights

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)

        if with_attn:
            return attn_output, attn_weights

        return attn_output

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.positional_embedding
        class_pos_embed = self.positional_embedding[[0]]
        patch_pos_embed = self.positional_embedding[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), mode='bicubic',
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), align_corners=False, recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.weights_track = []
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, output_layers=(-1,), return_vanilla_cls=False, out_type="mean", weights=None): 
        return self.visual(image.type(self.dtype), output_layers=output_layers, return_vanilla_cls=return_vanilla_cls, out_type=out_type, weights=weights)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text, text_ensemble=False, vision_outputs=(-1,), 
                return_vanilla_cls=False, interpolate=False, vision_out_type="mean",
                save_weights=False, K=3, topk_equal_weights=False, return_all_cls=False):
        
        logit_scale = self.logit_scale.exp()

        if text_ensemble:
            text_features = text
            # text_features = text_features.T
        else:
            text_features = self.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) #TODO: check if this is correct => in our other code it is 1

        
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(0) # (#templates, #classes, #features)

        # if return_vanilla_cls:
        #     image_features, vanilla_cls_features = self.encode_image(image, vision_outputs, return_vanilla_cls)
        # else:
        #     image_features = self.encode_image(image, vision_outputs, return_vanilla_cls)


        if vision_out_type == "mean":
            # image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="mean") # (batch_size, tokens, #features)
            if return_vanilla_cls:
                image_features, vanilla_cls_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="mean")
            else:
                image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="mean")
            # image_features = image_features[:, 1:]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
            # logits = logit_scale * image_features @ text_features.t() # (batch_size, tokens, #classes)
            logits = logit_scale * torch.einsum('bsd,tcd->tbsc', image_features, text_features) # (#templates, batch_size, tokens, #classes)

            # logits = logits[:, 1:] # (batch_size, tokens, #classes)
            logits = logits[:, :, 1:] # (#templates, batch_size, tokens, #classes)
            all_cls = logits[:, :, 0] # (#templates, batch_size, #classes)

            patch_size = self.visual.patch_size
            w, h = image[0].shape[-2] // patch_size, image[0].shape[-1] // patch_size
            temp_dim = logits.shape[0]
            b_dim = logits.shape[1]
            out_dim = logits.shape[-1]
            # logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h) # (batch_size, #class, W, H)
            logits = logits.permute(0, 1, 3, 2).reshape(logits.shape[0], logits.shape[1], out_dim, w, h) # (#templates, batch_size, #class, W, H)
            
            if interpolate:
                # Perform interpolation
                # logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False) # (batch_size, #classes, W, H)
                logits = logits.reshape(-1, out_dim, w, h)  # Flatten templates and batch dimensions for interpolation
                logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)  # (#templates*batch_size, #class, W', H')


                # Reshape back to include template and batch dimensions
                logits = logits.view(temp_dim, b_dim, out_dim, image.shape[-2], image.shape[-1])  # (#templates, batch_size, #class, W', H')

            if return_vanilla_cls:
                vanilla_cls_features = vanilla_cls_features / vanilla_cls_features.norm(dim=-1, keepdim=True)
                vanilla_cls_logits = logit_scale * torch.einsum('bd,tcd->tbc', vanilla_cls_features, text_features) # (#templates, batch_size, #classes)
                return logits, image_features, text_features, vanilla_cls_logits
            elif return_all_cls:
                return logits, image_features, text_features, all_cls
            
            return logits, image_features, text_features
        
        if vision_out_type == "residual_mean":
            image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="residual_mean") # (batch_size, tokens, #features)
            # image_features = image_features[:, 1:]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
            # logits = logit_scale * image_features @ text_features.t() # (batch_size, tokens, #classes)
            logits = logit_scale * torch.einsum('bsd,tcd->tbsc', image_features, text_features) # (#templates, batch_size, tokens, #classes)

            # logits = logits[:, 1:] # (batch_size, tokens, #classes)
            logits = logits[:, :, 1:] # (#templates, batch_size, tokens, #classes)

            patch_size = self.visual.patch_size
            w, h = image[0].shape[-2] // patch_size, image[0].shape[-1] // patch_size
            temp_dim = logits.shape[0]
            b_dim = logits.shape[1]
            out_dim = logits.shape[-1]
            # logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h) # (batch_size, #class, W, H)
            logits = logits.permute(0, 1, 3, 2).reshape(logits.shape[0], logits.shape[1], out_dim, w, h) # (#templates, batch_size, #class, W, H)
            
            if interpolate:
                # Perform interpolation
                # logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False) # (batch_size, #classes, W, H)
                logits = logits.reshape(-1, out_dim, w, h)  # Flatten templates and batch dimensions for interpolation
                logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)  # (#templates*batch_size, #class, W', H')


                # Reshape back to include template and batch dimensions
                logits = logits.view(temp_dim, b_dim, out_dim, image.shape[-2], image.shape[-1])  # (#templates, batch_size, #class, W', H')

            if return_vanilla_cls:
                vanilla_cls_features = vanilla_cls_features / vanilla_cls_features.norm(dim=-1, keepdim=True)
                vanilla_cls_logits = logit_scale * torch.einsum('bd,tcd->tbc', vanilla_cls_features, text_features) # (#templates, batch_size, #classes)
                return logits, image_features, text_features, vanilla_cls_logits
            
            return logits, image_features, text_features
        
        elif vision_out_type == "entropy_weighted":
            image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="all")
            # image_features = image_features[:, 1:]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # (#outlayers, batch_size, tokens, #features)
            # logits = logit_scale * image_features @ text_features.t() # (batch_size, tokens, #classes)
            logits = logit_scale * torch.einsum('obsd,tcd->otbsc', image_features, text_features) # (#outlayers, #templates, batch_size, tokens, #classes)

            # logits = logits[:, 1:] # (batch_size, tokens, #classes)
            # logits = logits[:, :, 1:] # (#templates, batch_size, tokens, #classes)
            logits = logits[:, :, :, 1:] # (#outlayers, #templates, batch_size, tokens, #classes)

            patch_size = self.visual.patch_size
            w, h = image[0].shape[-2] // patch_size, image[0].shape[-1] // patch_size
            layers_dim = logits.shape[0]
            temp_dim = logits.shape[1]
            b_dim = logits.shape[2]
            out_dim = logits.shape[-1]
            # logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h) # (batch_size, #class, W, H)
            logits = logits.permute(0, 1, 2, 4, 3).reshape(logits.shape[0], logits.shape[1], logits.shape[2], out_dim, w, h) # (#outlayers, #templates, batch_size, #class, W, H)
            
            # if interpolate:
            #     # Perform interpolation
            #     # logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False) # (batch_size, #classes, W, H)
            #     logits = logits.reshape(-1, out_dim, w, h)  # Flatten templates and batch dimensions for interpolation
            #     logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)  # (#outlayers, #templates*batch_size, #class, W', H')


            #     # Reshape back to include template and batch dimensions
            #     logits = logits.view(layers_dim, temp_dim, b_dim, out_dim, image.shape[-2], image.shape[-1])  # (#outlayers, #templates, batch_size, #class, W', H')

            # now we can calcualte the entropy weighted logits
            ent = -(logits.softmax(-3) * logits.log_softmax(-3)).sum(-3) # (#outlayers, #templates, batch_size, W', H')

            # mean over templates, W, H
            ent_weights = torch.mean(ent, dim=[1, 3, 4]) # (#outlayers, batch_size)

            # Invert entropy to prioritize confident layers
            ent_weights = -ent_weights  # Flip the relationship: lower entropy -> larger weight


            # softmax over outlayers
            ent_weights = F.softmax(ent_weights, dim=0) # (#outlayers, batch_size)

            # save the entropy weights to a list (detach them to avoid backpropagation)
            if save_weights: # just to be sure the weights are saved only during evaluation
                self.weights_track.append(ent_weights.detach().cpu().numpy()) ## (#outlayers, batch_size)

            # now recalcualte the logits based on the entropy weights
            image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="weighted_mean", weights=ent_weights) # (batch_size, tokens, #features)

            # image_features = image_features[:, 1:]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
            # logits = logit_scale * image_features @ text_features.t() # (batch_size, tokens, #classes)
            logits = logit_scale * torch.einsum('bsd,tcd->tbsc', image_features, text_features) # (#templates, batch_size, tokens, #classes)

            # logits = logits[:, 1:] # (batch_size, tokens, #classes)
            logits = logits[:, :, 1:] # (#templates, batch_size, tokens, #classes)

            patch_size = self.visual.patch_size
            w, h = image[0].shape[-2] // patch_size, image[0].shape[-1] // patch_size
            temp_dim = logits.shape[0]
            b_dim = logits.shape[1]
            out_dim = logits.shape[-1]
            # logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h) # (batch_size, #class, W, H)
            logits = logits.permute(0, 1, 3, 2).reshape(logits.shape[0], logits.shape[1], out_dim, w, h) # (#templates, batch_size, #class, W, H)
            
            if interpolate:
                # Perform interpolation
                # logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False) # (batch_size, #classes, W, H)
                logits = logits.reshape(-1, out_dim, w, h)  # Flatten templates and batch dimensions for interpolation
                logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)  # (#templates*batch_size, #class, W', H')


                # Reshape back to include template and batch dimensions
                logits = logits.view(temp_dim, b_dim, out_dim, image.shape[-2], image.shape[-1])  # (#templates, batch_size, #class, W', H')

            if return_vanilla_cls:
                vanilla_cls_features = vanilla_cls_features / vanilla_cls_features.norm(dim=-1, keepdim=True)
                vanilla_cls_logits = logit_scale * torch.einsum('bd,tcd->tbc', vanilla_cls_features, text_features) # (#templates, batch_size, #classes)
                return logits, image_features, text_features, vanilla_cls_logits
            
            return logits, image_features, text_features
            

        elif vision_out_type == "residual_entropy_weighted":
            assert len(vision_outputs) > 1, "The residual entropy weighted method requires more than one output layer"

            image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="all")
            # image_features = image_features[:, 1:]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # (#outlayers, batch_size, tokens, #features)
            # logits = logit_scale * image_features @ text_features.t() # (batch_size, tokens, #classes)
            logits = logit_scale * torch.einsum('obsd,tcd->otbsc', image_features, text_features) # (#outlayers, #templates, batch_size, tokens, #classes)

            # logits = logits[:, 1:] # (batch_size, tokens, #classes)
            # logits = logits[:, :, 1:] # (#templates, batch_size, tokens, #classes)
            logits = logits[:, :, :, 1:] # (#outlayers, #templates, batch_size, tokens, #classes)

            patch_size = self.visual.patch_size
            w, h = image[0].shape[-2] // patch_size, image[0].shape[-1] // patch_size
            layers_dim = logits.shape[0]
            temp_dim = logits.shape[1]
            b_dim = logits.shape[2]
            out_dim = logits.shape[-1]
            # logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h) # (batch_size, #class, W, H)
            logits = logits.permute(0, 1, 2, 4, 3).reshape(logits.shape[0], logits.shape[1], logits.shape[2], out_dim, w, h) # (#outlayers, #templates, batch_size, #class, W, H)
            
            # if interpolate:
            #     # Perform interpolation
            #     # logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False) # (batch_size, #classes, W, H)
            #     logits = logits.reshape(-1, out_dim, w, h)  # Flatten templates and batch dimensions for interpolation
            #     logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)  # (#outlayers, #templates*batch_size, #class, W', H')


            #     # Reshape back to include template and batch dimensions
            #     logits = logits.view(layers_dim, temp_dim, b_dim, out_dim, image.shape[-2], image.shape[-1])  # (#outlayers, #templates, batch_size, #class, W', H')

            # now we can calcualte the entropy weighted logits
            ent = -(logits.softmax(-3) * logits.log_softmax(-3)).sum(-3) # (#outlayers, #templates, batch_size, W', H')

            # mean over templates, W, H
            ent_weights = torch.mean(ent, dim=[1, 3, 4]) # (#outlayers, batch_size)

            # Invert entropy to prioritize confident layers
            ent_weights = -ent_weights  # Flip the relationship: lower entropy -> larger weight

            ## all except the last layer
            ent_weights = ent_weights[:-1] # (#outlayers-1, batch_size)


            # softmax over outlayers
            ent_weights = F.softmax(ent_weights, dim=0) # (#outlayers-1, batch_size)

            # save the entropy weights to a list (detach them to avoid backpropagation)
            if save_weights: # just to be sure the weights are saved only during evaluation
                self.weights_track.append(ent_weights.detach().cpu().numpy())

            # now recalcualte the logits based on the entropy weights
            image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="residual_weighted_mean", weights=ent_weights) # (batch_size, tokens, #features)

            # image_features = image_features[:, 1:]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
            # logits = logit_scale * image_features @ text_features.t() # (batch_size, tokens, #classes)
            logits = logit_scale * torch.einsum('bsd,tcd->tbsc', image_features, text_features) # (#templates, batch_size, tokens, #classes)

            # logits = logits[:, 1:] # (batch_size, tokens, #classes)
            logits = logits[:, :, 1:] # (#templates, batch_size, tokens, #classes)

            patch_size = self.visual.patch_size
            w, h = image[0].shape[-2] // patch_size, image[0].shape[-1] // patch_size
            temp_dim = logits.shape[0]
            b_dim = logits.shape[1]
            out_dim = logits.shape[-1]
            # logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h) # (batch_size, #class, W, H)
            logits = logits.permute(0, 1, 3, 2).reshape(logits.shape[0], logits.shape[1], out_dim, w, h) # (#templates, batch_size, #class, W, H)
            
            if interpolate:
                # Perform interpolation
                # logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False) # (batch_size, #classes, W, H)
                logits = logits.reshape(-1, out_dim, w, h)  # Flatten templates and batch dimensions for interpolation
                logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)  # (#templates*batch_size, #class, W', H')


                # Reshape back to include template and batch dimensions
                logits = logits.view(temp_dim, b_dim, out_dim, image.shape[-2], image.shape[-1])  # (#templates, batch_size, #class, W', H')

            if return_vanilla_cls:
                vanilla_cls_features = vanilla_cls_features / vanilla_cls_features.norm(dim=-1, keepdim=True)
                vanilla_cls_logits = logit_scale * torch.einsum('bd,tcd->tbc', vanilla_cls_features, text_features) # (#templates, batch_size, #classes)
                return logits, image_features, text_features, vanilla_cls_logits
            
            return logits, image_features, text_features
            



        elif vision_out_type == "all":
            if return_vanilla_cls:
                image_features, vanilla_cls_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="all")
            else:
                image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="all")
            # image_features = image_features[:, 1:]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # (#outlayers, batch_size, tokens, #features)
            # logits = logit_scale * image_features @ text_features.t() # (batch_size, tokens, #classes)
            logits = logit_scale * torch.einsum('obsd,tcd->otbsc', image_features, text_features) # (#outlayers, #templates, batch_size, tokens, #classes)

            # logits = logits[:, 1:] # (batch_size, tokens, #classes)
            # logits = logits[:, :, 1:] # (#templates, batch_size, tokens, #classes)
            logits = logits[:, :, :, 1:] # (#outlayers, #templates, batch_size, tokens, #classes)

            patch_size = self.visual.patch_size
            w, h = image[0].shape[-2] // patch_size, image[0].shape[-1] // patch_size
            layers_dim = logits.shape[0]
            temp_dim = logits.shape[1]
            b_dim = logits.shape[2]
            out_dim = logits.shape[-1]
            # logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h) # (batch_size, #class, W, H)
            logits = logits.permute(0, 1, 2, 4, 3).reshape(logits.shape[0], logits.shape[1], logits.shape[2], out_dim, w, h) # (#outlayers, #templates, batch_size, #class, W, H)

            if return_vanilla_cls:
                vanilla_cls_features = vanilla_cls_features / vanilla_cls_features.norm(dim=-1, keepdim=True)
                vanilla_cls_logits = logit_scale * torch.einsum('bd,tcd->tbc', vanilla_cls_features, text_features) # (#templates, batch_size, #classes)
                return logits, image_features, text_features, vanilla_cls_logits

            return logits, image_features, text_features

        elif vision_out_type == "mean_topk_entropy":
            image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="all")
            # image_features = image_features[:, 1:]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # (#outlayers, batch_size, tokens, #features)
            # logits = logit_scale * image_features @ text_features.t() # (batch_size, tokens, #classes)
            logits = logit_scale * torch.einsum('obsd,tcd->otbsc', image_features, text_features) # (#outlayers, #templates, batch_size, tokens, #classes)

            # logits = logits[:, 1:] # (batch_size, tokens, #classes)
            # logits = logits[:, :, 1:] # (#templates, batch_size, tokens, #classes)
            logits = logits[:, :, :, 1:] # (#outlayers, #templates, batch_size, tokens, #classes)

            patch_size = self.visual.patch_size
            w, h = image[0].shape[-2] // patch_size, image[0].shape[-1] // patch_size
            layers_dim = logits.shape[0]
            temp_dim = logits.shape[1]
            b_dim = logits.shape[2]
            out_dim = logits.shape[-1]
            # logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h) # (batch_size, #class, W, H)
            logits = logits.permute(0, 1, 2, 4, 3).reshape(logits.shape[0], logits.shape[1], logits.shape[2], out_dim, w, h) # (#outlayers, #templates, batch_size, #class, W, H)
            
            # if interpolate:
            #     # Perform interpolation
            #     # logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False) # (batch_size, #classes, W, H)
            #     logits = logits.reshape(-1, out_dim, w, h)  # Flatten templates and batch dimensions for interpolation
            #     logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)  # (#outlayers, #templates*batch_size, #class, W', H')


            #     # Reshape back to include template and batch dimensions
            #     logits = logits.view(layers_dim, temp_dim, b_dim, out_dim, image.shape[-2], image.shape[-1])  # (#outlayers, #templates, batch_size, #class, W', H')

            # now we can calcualte the entropy weighted logits
            ent = -(logits.softmax(-3) * logits.log_softmax(-3)).sum(-3) # (#outlayers, #templates, batch_size, W', H')

            # mean over templates, W, H
            ent_weights = torch.mean(ent, dim=[1, 3, 4]) # (#outlayers, batch_size)

            # Invert entropy to prioritize confident layers
            ent_weights = -ent_weights  # Flip the relationship: lower entropy -> larger weight


            # # softmax over outlayers
            # ent_weights = F.softmax(ent_weights, dim=0) # (#outlayers, batch_size)

            if topk_equal_weights:
                # Equal importance to top-K
                # Get the top-K indices along the outlayers dimension (dim=0)
                topk_indices = torch.topk(ent_weights, K, dim=0).indices  # Shape: (K, batch_size)

                # Create a mask to zero out non-top-K elements
                mask = torch.zeros_like(ent_weights, dtype=torch.bool)
                mask.scatter_(0, topk_indices, True)

                # Create equal weights for the top-K indices
                equal_weight = 1.0 / K
                ent_weights = torch.zeros_like(ent_weights)  # Initialize all weights to zero
                ent_weights[mask] = equal_weight  # Assign equal weight to top-K elements

            else:
                # Using top-K entropy weights with softmax
                # Get the top-K indices along the outlayers dimension (dim=0)
                topk_indices = torch.topk(ent_weights, K, dim=0).indices  # Shape: (K, batch_size)

                # Initialize mask with -inf
                masked_ent_weights = torch.full_like(ent_weights, float('-inf'))  # Set all to -inf

                # Scatter the original weights to the top-K indices
                masked_ent_weights.scatter_(0, topk_indices, ent_weights.gather(0, topk_indices))

                # Normalize the weights for the top-K layers using softmax
                ent_weights = F.softmax(masked_ent_weights, dim=0)  # Shape: (#outlayers, batch_size)




            # save the entropy weights to a list (detach them to avoid backpropagation)
            if save_weights: # just to be sure the weights are saved only during evaluation
                self.weights_track.append(ent_weights.detach().cpu().numpy()) ## (#outlayers, batch_size)

            # now recalcualte the logits based on the entropy weights
            image_features = self.encode_image(image, vision_outputs, return_vanilla_cls, out_type="weighted_mean", weights=ent_weights) # (batch_size, tokens, #features)

            # image_features = image_features[:, 1:]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
            # logits = logit_scale * image_features @ text_features.t() # (batch_size, tokens, #classes)
            logits = logit_scale * torch.einsum('bsd,tcd->tbsc', image_features, text_features) # (#templates, batch_size, tokens, #classes)

            # logits = logits[:, 1:] # (batch_size, tokens, #classes)
            logits = logits[:, :, 1:] # (#templates, batch_size, tokens, #classes)

            patch_size = self.visual.patch_size
            w, h = image[0].shape[-2] // patch_size, image[0].shape[-1] // patch_size
            temp_dim = logits.shape[0]
            b_dim = logits.shape[1]
            out_dim = logits.shape[-1]
            # logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h) # (batch_size, #class, W, H)
            logits = logits.permute(0, 1, 3, 2).reshape(logits.shape[0], logits.shape[1], out_dim, w, h) # (#templates, batch_size, #class, W, H)
            
            if interpolate:
                # Perform interpolation
                # logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False) # (batch_size, #classes, W, H)
                logits = logits.reshape(-1, out_dim, w, h)  # Flatten templates and batch dimensions for interpolation
                logits = nn.functional.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)  # (#templates*batch_size, #class, W', H')


                # Reshape back to include template and batch dimensions
                logits = logits.view(temp_dim, b_dim, out_dim, image.shape[-2], image.shape[-1])  # (#templates, batch_size, #class, W', H')

            if return_vanilla_cls:
                vanilla_cls_features = vanilla_cls_features / vanilla_cls_features.norm(dim=-1, keepdim=True)
                vanilla_cls_logits = logit_scale * torch.einsum('bd,tcd->tbc', vanilla_cls_features, text_features) # (#templates, batch_size, #classes)
                return logits, image_features, text_features, vanilla_cls_logits
            
            return logits, image_features, text_features
            

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
