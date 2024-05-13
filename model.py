# References:
    # https://github.com/berniwal/swin-transformer-pytorch/blob/master/swin_transformer_pytorch/swin_transformer.py
    # https://pajamacoder.tistory.com/18
    # https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import math
from torchvision.models import swin_t


class CyclicShift(nn.Module):
    def __init__(self, disp):
        super().__init__()

        self.disp = disp

    def forward(self, x):
        x = torch.roll(
            input=x, shifts=(self.disp, self.disp), dims=(2, 3),
        )
        return x


class MLP(nn.Module):
    def __init__(self, hidden_dim, exp_rate, drop_prob):
        super().__init__()

        mlp_dim = hidden_dim * exp_rate

        self.proj1 = nn.Linear(hidden_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.proj2 = nn.Linear(mlp_dim, hidden_dim)
        self.mlp_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.proj1(x)
        x = self.gelu(x)
        x = self.proj2(x)
        x = self.mlp_drop(x)
        return x


class WMSA(nn.Module):
    """
    "To make the window size (M;M) divisible by the feature map size of
    (h;w), bottom-right padding is employed on the feature map if needed."
    "To make the window size (M, M) divisible by the feature map size of (h, w),
    bottom-right padding is employed on the feature map if needed."
    """
    def get_rel_pos_bias(self, window_size):
        """
        "A relative position bias $B \in \mathbb{R}^{M^{2} \times M^{2}}$ to
        each head in computing similarity:
        $\text{Attention}(Q, K, V) = \text{SoftMax}(QK^{T}/\sqrt{d} + B)V$"
        "Since the relative position along each axis lies in the range
        $[-M + 1, M - 1]$, we parameterize a smaller-sized bias matrix
        $\hat{B} \in \mathbb{R}^{(2M - 1) \times (2M - 1)}$, and values in B are
        taken from $\hat{B}$."
        "The learnt relative position bias in pre-training can be also used to
        initialize a model for fine-tuning with a different window size through
        bi-cubic interpolation."
        """
        self.bias_mat = nn.Parameter(
            torch.randn((window_size * 2 - 1, window_size * 2 - 1))
        )
        pos = torch.tensor(
            [[x, y] for x in range(window_size) for y in range(window_size)],
        )
        rel_pos = pos[None, :] - pos[:, None]
        return self.bias_mat[rel_pos[..., 0], rel_pos[..., 1]]

    def __init__(self, window_size, hidden_dim, head_dim, drop_prob):
        super().__init__()

        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_heads = hidden_dim // head_dim

        self.window_to_seq = nn.Sequential(
            Rearrange(
                pattern="b c (M1 h) (M2 w) -> (b h w) (M1 M2) c",
                M1=window_size,
                M2=window_size,
            )
        )       
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_multi_heads = Rearrange(
            "b i (n h) -> b i n h", n=self.num_heads,
        )
        self.scale = hidden_dim ** (-0.5)
        self.rel_pos_bias = self.get_rel_pos_bias(window_size)
        self.attn_drop = nn.Dropout(drop_prob)
        self.to_one_head = Rearrange("b i n h -> b i (n h)")
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, mask=None):
        _, _, h, w = x.shape
        x = self.window_to_seq(x)

        q = self.q_proj(x) # "$hwC^{2}$"
        k = self.k_proj(x) # "$hwC^{2}$"
        v = self.v_proj(x) # "$hwC^{2}$"
        # "$(hw)^{2}C$" or "${M^{2}}^{2}(\frac{h}{M})(\frac{w}{M})C$"
        attn_score = torch.einsum(
            "binh,bjnh->bnij", self.to_multi_heads(q), self.to_multi_heads(k),
        ) * self.scale
        attn_score += einops.repeat(
            self.rel_pos_bias,
            pattern="h w -> b c h w",
            b=attn_score.size(0),
            c=attn_score.size(1),
        )
        if mask is not None:
            attn_score.masked_fill_(
                mask=einops.repeat(
                    mask, pattern="b i j -> b n i j", n=self.num_heads,
                ),
                value=-1e9,
            )
        attn_weight = F.softmax(attn_score, dim=-1)
        # "$(hw)^{2}C$" or "${M^{2}}^{2}(\frac{h}{M})(\frac{w}{M})C$"
        x = self.to_one_head(
            torch.einsum(
                "bnij,bjnh->binh",
                self.attn_drop(attn_weight),
                self.to_multi_heads(v),
            )
        )
        x = self.out_proj(x) # "$hwC^{2}$"
        x = einops.rearrange(
            x,
            pattern="(b h w) (M1 M2) c -> b c (M1 h) (M2 w)",
            M1=self.window_size,
            h=h // self.window_size,
            w=w // self.window_size,
        )
        return x, attn_weight

    def get_flops(self, seq_len):
        flops = 0
        flops += self.hidden_dim * self.hidden_dim * seq_len * 3
        flops += self.num_heads * seq_len * seq_len * self.head_dim
        flops += seq_len * self.num_heads * self.head_dim * seq_len
        flops += self.hidden_dim * self.hidden_dim * seq_len
        return flops


class SWMSA(nn.Module):
    """
    "The next module adopts a windowing configuration that is shifted from that
    of the preceding layer, by displacing the windows by
    $(\lfloor \frac{M}{2} \rfloor, \lfloor \frac{M}{2} \rfloor)$ pixels from the
    regularly partitioned windows."
    """
    def __init__(self, window_size,):
        super().__init__()
        
        self.cyc_shift = CyclicShift(disp=-window_size // 2)
        self.rev_cyc_shift = CyclicShift(disp=window_size // 2)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.cyc_shift(x)
        x = self.window_to_seq(x)
        x, _ = super().forward(q=x, k=x, v=x)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        window_size,
        hidden_dim,
        head_dim,
        exp_rate,
        attn_drop_prob,
        mlp_drop_prob,
        # res_drop_prob,
    ):
        super().__init__()

        self.window_size = window_size

        self.msa_ln = nn.LayerNorm(hidden_dim)
        self.msa = WMSA(
            window_size=window_size,
            hidden_dim=hidden_dim,
            head_dim=head_dim,
            drop_prob=attn_drop_prob,
        )
        self.mlp_ln = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim=hidden_dim,
            exp_rate=exp_rate,
            drop_prob=mlp_drop_prob,
        )

    def forward(self, x, mask=None):
        _, _, h, w = x.shape
        x = F.pad(
            x,
            pad=(
                0,
                math.ceil(w / self.window_size) * self.window_size - w,
                0,
                math.ceil(h / self.window_size) * self.window_size - h
            ),
            mode="constant",
            value=0,
        )

        skip = x
        x = x.permute((0, 2, 3, 1))
        x = self.msa_ln(x)
        x = x.permute((0, 3, 1, 2))
        x, _ = self.msa(x, mask=mask)
        x += skip

        skip = x
        x = x.permute((0, 2, 3, 1))
        x = self.mlp_ln(x)
        _, h, _, _ = x.shape
        x = einops.rearrange(x, pattern="b h w c -> b (h w) c")
        x = self.mlp(x)
        x = einops.rearrange(x, pattern="b (h w) c -> b c h w", h=h)
        x += skip
        return x


class PatchPartition(nn.Module):
    """
    "T). It first splits an input RGB image into non-overlapping
    patches by a patch splitting module, like ViT. Each patch is
    treated as a “token” and its feature is set as a concatenation
    of the raw pixel RGB values. In our implementation, we use
    a patch size of 4 4 and thus the feature dimension of each
    patch is 4   4   3 = 48.
    """
    def __init__(self, patch_size):
        super().__init__()

        in_chs = 3
        self.conv = nn.Conv2d(
            in_chs, patch_size ** 2 * in_chs, patch_size, patch_size, 0,
        )
    
    def forward(self, image):
        return self.conv(image)


class PatchMerging(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.conv = nn.Conv2d(hidden_dim, hidden_dim * 2, 2, 2, 0)
    
    def forward(self, x):
        return self.conv(x)


class SwinTransformer(nn.Module):
    """
    "A linear embedding layer is applied on this raw-valued feature to project
    it to an arbitrary dimension (denoted as $C$)."
    "The Transformer blocks maintain the number of tokens (H 4   W 4 ), and together with the linear embedding are referred to as 'Stage 1'".
    "The first patch merging layer concatenates the features of each group of 2   2 neighboring patches, and applies a linear layer on the 4C-dimensional concatenated features. This reduces the number of tokens by a multiple of 2 2 = 4 (2  downsampling of resolution), and the output dimension is set to 2C."
    "The window size is set to M = 7 by default. The query dimension of each head is d = 32, and the expansion layer of each MLP is   = 4"
    "The image classification is performed by applying a global average pooling
    layer on the output feature map of the last stage, followed by a linear
    classifier."
    """
    def __init__(
        self,
        num_classes=None,
        patch_size=4,
        hidden_dim=96, # $C$
        num_layers=(2, 2, 6, 2),
        window_size=7, # $M$
        head_dim=32,
        exp_rate=4,
        attn_drop_prob=0.1,
        mlp_drop_prob=0.1,
        # res_drop_prob=0.1,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.patch_part = PatchPartition(patch_size=patch_size)
        self.stage1 = nn.Sequential(
            nn.Conv2d(
                self.patch_part.conv.out_channels, hidden_dim, 1, 1, 0,
            ),
            *[
                SwinTransformerBlock(
                    window_size=window_size,
                    hidden_dim=hidden_dim,
                    head_dim=head_dim,
                    exp_rate=exp_rate,
                    attn_drop_prob=attn_drop_prob,
                    mlp_drop_prob=mlp_drop_prob,
                    # res_drop_prob=res_drop_prob,
                )
            ] * num_layers[0]
        )
        self.stage2 = nn.Sequential(
            PatchMerging(hidden_dim),
            *[
                SwinTransformerBlock(
                    window_size=window_size,
                    hidden_dim=hidden_dim * 2,
                    head_dim=head_dim,
                    exp_rate=exp_rate,
                    attn_drop_prob=attn_drop_prob,
                    mlp_drop_prob=mlp_drop_prob,
                    # res_drop_prob=res_drop_prob,
                )
            ] * num_layers[1]
        )
        self.stage3 = nn.Sequential(
            PatchMerging(hidden_dim * 2),
            *[
                SwinTransformerBlock(
                    window_size=window_size,
                    hidden_dim=hidden_dim * 4,
                    head_dim=head_dim,
                    exp_rate=exp_rate,
                    attn_drop_prob=attn_drop_prob,
                    mlp_drop_prob=mlp_drop_prob,
                    # res_drop_prob=res_drop_prob,
                )
            ] * num_layers[2]
        )
        self.stage4 = nn.Sequential(
            PatchMerging(hidden_dim * 4),
            *[
                SwinTransformerBlock(
                    window_size=window_size,
                    hidden_dim=hidden_dim * 8,
                    head_dim=head_dim,
                    exp_rate=exp_rate,
                    attn_drop_prob=attn_drop_prob,
                    mlp_drop_prob=mlp_drop_prob,
                    # res_drop_prob=res_drop_prob,
                )
            ] * num_layers[3]
        )
        if num_classes is not None:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(hidden_dim * 8, num_classes),
            )

    def forward(self, image):
        x = self.patch_part(image)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        if self.num_classes is not None:
            x = self.cls_head(x)
        return x


class SwinT(SwinTransformer):
    def __init__(self, num_classes=None):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=96,
            num_layers=(2, 2, 6, 2),
            head_dim=32,
            patch_size=4,
            window_size=7,
            exp_rate=4,
            attn_drop_prob=0.1,
            mlp_drop_prob=0.1,
            # res_drop_prob=0.1,
        )


class SwinS(SwinTransformer):
    def __init__(self, num_classes=None):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=96,
            num_layers=(2, 2, 18, 2),
            head_dim=32,
            patch_size=4,
            window_size=7,
            exp_rate=4,
            attn_drop_prob=0.1,
            mlp_drop_prob=0.1,
            # res_drop_prob=0.1,
        )


class SwinB(SwinTransformer):
    def __init__(self, num_classes=None):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=128,
            num_layers=(2, 2, 18, 2),
            head_dim=32,
            patch_size=4,
            window_size=7,
            exp_rate=4,
            attn_drop_prob=0.1,
            mlp_drop_prob=0.1,
            # res_drop_prob=0.1,
        )


class SwinL(SwinTransformer):
    def __init__(self, num_classes=None):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=192,
            num_layers=(2, 2, 18, 2),
            head_dim=32,
            patch_size=4,
            window_size=7,
            exp_rate=4,
            attn_drop_prob=0.1,
            mlp_drop_prob=0.1,
            # res_drop_prob=0.1,
        )


if __name__ == "__main__":
    model = SwinTransformer(num_classes=100)
    # model = SwinS()
    img_size = 223
    image = torch.randn((4, 3, img_size, img_size))
    model(image).shape
