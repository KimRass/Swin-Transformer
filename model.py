# Reference: https://github.com/berniwal/swin-transformer-pytorch/blob/master/swin_transformer_pytorch/swin_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange


class CyclicShift(nn.Module):
    def __init__(self, disp):
        super().__init__()

        self.disp = disp

    def forward(self, x):
        x = torch.roll(
            input=x, shifts=(self.disp, self.disp), dims=(2, 3),
        )
        return x


class ResidualConnection(nn.Module):
    def __init__(self, fn, hidden_dim, drop_prob):
        super().__init__()

        self.fn = fn
        self.res_drop = nn.Dropout(drop_prob)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, skip, **kwargs):
        x = self.fn(**kwargs)
        x = self.res_drop(x)
        x += skip
        x = self.norm(x)
        return x


class MLP(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, drop_prob):
        super().__init__()

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


class MSA(nn.Module):
    def __init__(self, hidden_dim, num_heads, drop_prob):
        super().__init__()
    
        self.num_heads = num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_multi_heads = Rearrange("b i (n h) -> b i n h", n=num_heads)
        self.scale = hidden_dim ** (-0.5)
        self.attn_drop = nn.Dropout(drop_prob)
        self.to_one_head = Rearrange("b i n h -> b i (n h)")
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, q, k, v, mask=None):
        q = self.q_proj(q) # "$hwC^{2}$"
        k = self.k_proj(k) # "$hwC^{2}$"
        v = self.v_proj(v) # "$hwC^{2}$"
        attn_score = torch.einsum(
            "binh,bjnh->bnij", self.to_multi_heads(q), self.to_multi_heads(k),
        ) * self.scale # "$(hw)^{2}C$"
        if mask is not None:
            attn_score.masked_fill_(
                mask=einops.repeat(
                    mask, pattern="b i j -> b n i j", n=self.num_heads,
                ),
                value=-1e9,
            )
        attn_weight = F.softmax(attn_score, dim=-1)
        x = self.to_one_head(
            torch.einsum(
                "bnij,bjnh->binh",
                self.attn_drop(attn_weight),
                self.to_multi_heads(v),
            )
        ) # "$(hw)^{2}C$"
        x = self.out_proj(x) # "$hwC^{2}$"
        return x, attn_weight
# (1, 56 ** 2, 768) / (64, 49, 768)
14 ** 2
7 ** 2 * 4

class WMSA(MSA):
    def __init__(self, window_size, hidden_dim, num_heads, drop_prob):
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            drop_prob=drop_prob,
        )

        self.spat_to_batch = Rearrange(
            pattern="b c (M1 h) (M2 w) -> (b h w) (M1 M2) c",
            M1=window_size,
            M2=window_size,
        )

    def forward(self, x):
        _, _, h, w = x.shape
        print(x.shape)
        x = self.spat_to_batch(x)
        print(x.shape)
        x, _ = super().forward(q=x, k=x, v=x)
        return einops.rearrange(
            x,
            pattern="(b h w) (M1 M2) c -> b c (M1 h) (M2 w)",
            M1=window_size,
            h=h // window_size,
            w=w // window_size,
        )


class SWMSA(nn.Module):
    """
    "the next module adopts a windowing configuration that is shifted from that of the preceding layer, by displacing the windows by (bM 2 c; bM 2 c) pixels from the regularly partitioned windows."
    """
    def __init__(self):
        super().__init__()


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        mlp_dim,
        attn_drop_prob,
        ff_drop_prob,
        res_drop_prob,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim

        self.self_attn = WMSA(
            hidden_dim=hidden_dim, num_heads=num_heads, drop_prob=attn_drop_prob,
        )
        self.feed_forward = MLP(
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            drop_prob=ff_drop_prob,
            activ="relu",
        )

        self.self_attn_res_conn = ResidualConnection(
            fn=lambda x, mask: self.self_attn(q=x, k=x, v=x, mask=mask)[0],
            hidden_dim=hidden_dim,
            drop_prob=attn_drop_prob,
        )
        self.ff_res_conn = ResidualConnection(
            fn=self.feed_forward,
            hidden_dim=hidden_dim,
            drop_prob=res_drop_prob,
        )

    def forward(self, x, mask=None):
        x = self.self_attn_res_conn(skip=x, x=x, mask=mask)
        return self.ff_res_conn(skip=x, x=x)


# class SwinTransformerBlock(nn.Module):
#     def __init__(self):
#         super().__init__()
#         patch_dim = 3

#         # self.ln1 = nn.LayerNorm(patch_dim)
#         # self.wmsa = WMSA()
#         # self.ln2 = nn.LayerNorm(hidden_dim)
#         # self.mlp = MLP(dim=)
#         # self.swmsa = SWMSA()
#         self.sequential1 = nn.Sequential(
#             nn.LayerNorm(patch_dim),
#             WMSA()
#         )
#         self.residual_connection1 = ResidualConnection(fn=self.sequential1)

#         self.sequential2 = nn.Sequential(
#             nn.LayerNorm(patch_dim),
#             SWMSA()
#         )
#         self.residual_connection2 = ResidualConnection(fn=self.sequential2)

#         self.sequential3 = nn.Sequential(
#             nn.LayerNorm(patch_dim),
#             MLP()
#         )
#         self.residual_connection3 = ResidualConnection(fn=self.sequential3)
    
#     def forward(self, x):
#         x = self.residual_connection1(x)
#         x = self.residual_connection3(x)
#         x = self.residual_connection2(x)
#         x = self.residual_connection3(x)
#         return x


class PatchPartition(nn.Module):
    """
    "T). It first splits an input RGB image into non-overlapping
    patches by a patch splitting module, like ViT. Each patch is
    treated as a “token” and its feature is set as a concatenation
    of the raw pixel RGB values. In our implementation, we use
    a patch size of 4 4 and thus the feature dimension of each
    patch is 4   4   3 = 48. A linear embedding layer is applied
    on this raw-valued feature to project it to an arbitrary
    dimension (denoted as C).
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
    def __init__(self, downscaling_factor, hidden_dim=96):
        super().__init__()

        self.downscaling_factor = downscaling_factor

        self.unfold = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(4 * hidden_dim, 2 * hidden_dim)
    
    def forward(self, x):
        # x = x
        # x.shape
        b, c, h, w = x.shape

        new_h = h // self.downscaling_factor
        new_w = w // self.downscaling_factor
        x = self.unfold(x)
        x = x.view((b, -1, new_h, new_w))
        x = x.permute((0, 2, 3, 1))
        x = self.linear(x)
        x = x.permute((0, 3, 1, 2))
        return x

    # def __init__(self, downscaling_factor, hidden_dim=96):
    #     super().__init__()

    # def forward(self, x):
    #     x = x
    #     downscaling_factor=2
        
    #     b, h, w, c = x.shape

    #     x = x.permute((0, 3, 1, 2))
        
    #     b, c, h, w = x.shape
    #     x.shape
    #     # (4, 96, 43, 70)

    #     new_h = h // downscaling_factor
    #     new_w = w // downscaling_factor
    #     unfold = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
    #     unfold(x).shape
    #     unfold(x).view((b, -1, new_h, new_w)).shape
    #     96*4

        
    #     1680 / hidden_dim / 4
    #     1680 / 96
    #     h
    #     new_h
    #     4 * new_h * new_w
        
    #     4*172*1680 / 2940
    #     unfold(x).view((b, new_h, new_w, -1))
    #     shape



class SwinTransformer(nn.Module):
    """
    "The Transformer blocks maintain the number of tokens (H 4   W 4 ), and together with the linear embedding are referred to as 'Stage 1'".
    "The first patch merging layer concatenates the features of each group of 2   2 neighboring patches, and applies a linear layer on the 4C-dimensional concatenated features. This reduces the number of tokens by a multiple of 2 2 = 4 (2  downsampling of resolution), and the output dimension is set to 2C."
    """
    # `window_size`: $M$
    # `hidden_dim`: $C$
    def __init__(
        self,
        num_classes,
        patch_size=4,
        window_size=7,
        hidden_dim=96,
        num_layers=(2, 2, 6, 2),
    ):
        super().__init__()

        # window_size=7
        # patch_size=4
        # hidden_dim=96
        self.patch_part = PatchPartition(patch_size=patch_size)
        lin_embed = nn.Conv2d(
            self.patch_part.conv.out_channels,
            hidden_dim,
            1,
            1,
            0,
        )
        self.stage1 = nn.ModuleList(
            [
                lin_embed,
            ]
        )
    

    def forward(self, image):
        x = self.patch_part(image)
        x = self.stage1(x)
        return x


if __name__ == "__main__":
    # model = SwinTransformer(num_classes=100)
    # image = torch.randn((4, 3, 512, 512))
    # model(image).shape
    # model
    
    window_size=7
    wmsa = WMSA(
        window_size=window_size,
        hidden_dim=768,
        num_heads=2,
        drop_prob=0.1,
    )
    x = torch.randn((1, 768, 56, 56)) # (4 * 8 * 8, 768, 7 * 7)
    wmsa(x).shape
    56 ** 2
