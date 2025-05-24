from functools import partial
import numpy as np
import torch
import torch.nn as nn
import random
from timm.models.vision_transformer import PatchEmbed, Block
import math
from util.pos_embed import get_2d_sincos_pos_embed

import models_mae


class MaskedAutoencoderViT(models_mae.MaskedAutoencoderViT):
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        H = W = int(L ** 0.5)
        # print("H", H)
        mask = torch.ones([N, L], device=x.device)
        mask = mask.reshape(N, H, W)
        crope_lenth = round(H * (1 - mask_ratio))
        assert crope_lenth % 2 == 0, 'errors'
        assert (H - crope_lenth) % 2 == 0, 'errors'
        start_idx = int((H - crope_lenth) / 2)  # 起始索引
        end_idx = start_idx + crope_lenth  # 结束索引
        mask[:, start_idx:end_idx, start_idx:end_idx] = 0
        mask = mask.reshape(N, L)
        # edge masking
        noise = mask
        len_keep = int(L - mask[0].sum())
        assert int(L - mask[0].sum()) == int((H * (1 - mask_ratio)) ** 2)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    # def forward_loss(self, imgs, pred, mask):
    #     pass


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks