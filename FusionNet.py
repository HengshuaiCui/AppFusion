import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from torch.nn import init, Sequential
import numpy as np
import kornia
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BasicConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        base_channel = 32
        self.avgpool = nn.AvgPool2d((2, 2))
        self.conv1_1_2 = BasicConv(1, base_channel, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv1_2_2 = BasicConv(base_channel, base_channel, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)

        self.conv2_1_2 = BasicConv(base_channel, base_channel * 2, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv2_2_2 = BasicConv(base_channel * 2, base_channel * 2, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)

        self.conv3_1_2 = BasicConv(base_channel * 2, base_channel * 4, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv3_2_2 = BasicConv(base_channel * 4, base_channel * 4, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv3_3_2 = BasicConv(base_channel * 4, base_channel * 4, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)


    def forward(self, x):
        x = self.conv1_1_2(x)
        x = self.conv1_2_2(x)
        s1 = x
        x = self.avgpool(x)
        x = self.conv2_1_2(x)
        x = self.conv2_2_2(x)
        s2 = x
        x = self.avgpool(x)
        x = self.conv3_1_2(x)
        x = self.conv3_2_2(x)
        x = self.conv3_3_2(x)
        s3 = x

        return s1, s2, s3

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        base_channel = 32
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1_2 = BasicConv(base_channel * 4, base_channel * 2, 1, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv2_2 = BasicConv(base_channel * 2, base_channel * 2, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)

        self.conv3_2 = BasicConv(base_channel * 4, base_channel, 1, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv4_2 = BasicConv(base_channel, base_channel, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)

        self.conv5_2 = BasicConv(base_channel * 2, base_channel, 1, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv6_2 = BasicConv(base_channel, base_channel, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv_out = BasicConv(base_channel, 1, 1, 1, activation=nn.LeakyReLU(0.2), use_bn=False)
        # self.conv7_2 = BasicConv(base_channel, base_channel // 2, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)

        # self.conv_out_base_1 = BasicConv(base_channel // 2, base_channel // 2, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        # self.conv_out_base_2 = BasicConv(base_channel // 2, base_channel // 2, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        # self.conv_out_base_3 = BasicConv(base_channel // 2, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

    def forward(self, s1, s2, s3):
        x = s3
        x = self.conv1_2(x)
        x = self.conv2_2(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s2], 1)
        x = self.conv3_2(x)
        x = self.conv4_2(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s1], 1)
        x = self.conv5_2(x)
        x = self.conv6_2(x)

        x = self.conv_out(x)

        # x = self.conv7_2(x)
        # x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # x = self.conv_out_base_1(x)
        # x = self.conv_out_base_2(x)
        # x = self.conv_out_base_3(x)

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        n_ctx = 16  # Number of context words
        ctx_init = ""  # Reset to empty, no initial context
        dtype = torch.float32
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization for context
            print("Initializing generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)  # Initialize without any specific context
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)  # Generic tokens "X" for each context word

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # Now, we don't need to tokenize class names or generate prompts based on them.
        # Just initialize the prompts with the generic context
        tokenized_prompt = clip.tokenize(prompt_prefix)
        if torch.cuda.is_available():
            tokenized_prompt = tokenized_prompt.to('cuda')  # Move tensor to GPU if available

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompt).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current prompt
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx: , :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.tokenized_prompt = tokenized_prompt  # torch.Tensor

    def forward(self, batch_size=1):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(1, -1, -1)  # (1, n_ctx, dim)

        prefix = self.token_prefix  # (1, 1, dim)
        suffix = self.token_suffix  # (1, *, dim)

        prompts = torch.cat([prefix, ctx, suffix], dim=1)  # (1, 77, dim)

        # 扩展到 batch 大小
        prompts = prompts.expand(batch_size, -1, -1)  # (B, 77, dim)

        if torch.cuda.is_available():
            prompts = prompts.to('cuda')

        return prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Cross_Fus(nn.Module):
    def __init__(self, visual_dim = 2048, text_dim = 1024, output_dim = 256, dropout = 0.1, nhead = 4):
        super().__init__()
        # ---------------#
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.nhead = nhead
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(self.visual_dim),
            nn.Linear(self.visual_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.output_dim),
        )
        self.cross_attn = Attention(dim=self.output_dim, num_heads=self.nhead, proj_drop=self.dropout)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.visual_dim)
        )
    def forward(self, text, visual):#6 1024 , 4 2048 7 7
        B, C, H, W = visual.shape
        visual_embeddings = visual.reshape(B, C, H * W).permute(0, 2, 1)
        text_embeddings = text.reshape(B, C, H * W).permute(0, 2, 1)
        visual_embeddings = self.visual_proj(visual_embeddings)
        text_embeddings = self.text_proj(text_embeddings)

        visual_embeddings = visual_embeddings + self.cross_attn(text_embeddings, visual_embeddings, visual_embeddings)
        visual_embeddings = self.out_proj(visual_embeddings)
        visual_embeddings = visual_embeddings.permute(0, 2, 1).reshape(B, C, H, W)
        # score_map = torch.einsum('bchw,bkc->bkhw', visual, text_embeddings)

        return visual_embeddings
class Bias_Residual_Guidance_(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(Bias_Residual_Guidance_, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, out_channels*2),
            nn.LeakyReLU(),
            nn.Linear(out_channels*2, out_channels*2)
        )

    def forward(self, x, text_embed):
        # print(text_embed.shape)
        text_embed = text_embed.unsqueeze(1)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x
class FFBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FFBlock, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channels*2, in_channels*2, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True),
            BasicConv(in_channels*2, out_channels, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True),
            BasicConv(out_channels, out_channels, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True),
        )

    def forward(self, en1, en2):
        cat = torch.cat([en1, en2], dim=1)
        fus = self.conv(cat)
        return fus
class FusionNet(nn.Module):
    def __init__(self,model_clip):
        super(FusionNet, self).__init__()
        self.model_clip = model_clip.float()
        self.model_clip.eval()
        self.dtype = torch.float32
        self.prompt_learner = PromptLearner(model_clip)
        self.tokenized_prompt = self.prompt_learner.tokenized_prompt
        self.text_encoder = TextEncoder(model_clip)
        self.logit_scale = model_clip.logit_scale
        self.brg = Bias_Residual_Guidance_(in_channels=1024, out_channels=128)

        self.cross_fus = Cross_Fus(visual_dim=128, text_dim=128, output_dim=128, dropout=0.1, nhead=4)

        self.encoder_ir = Encoder()
        self.encoder_rgb = Encoder()
        self.decoder = Decoder()
        self.conv = BasicConv(2, 3, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv1 = BasicConv(256, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.ffb1 = FFBlock(32, 32)
        self.ffb2 = FFBlock(64, 64)
        self.ffb3 = FFBlock(128, 128)
    def forward(self, input_ir, input_rgb):
        clip_fus =torch.cat([input_ir, input_rgb], 1)
        clip_fus = self.conv(clip_fus)
        fus_feas = self.get_img_feature(clip_fus)  # [0~4] 0：256 56 56， 2：512 28 28 3：1024 14 14 4：2048 7 7 [5] 2048 1 1
        fus_fea_list = list(fus_feas[0:4])
        # ------------ir and rgb prompt-------------------#
        prompts = self.prompt_learner(batch_size=input_ir.size(0))  # 6 77 512
        tokenized_prompt = self.tokenized_prompt
        text_features = self.text_encoder(prompts, tokenized_prompt)  # B N:4 1024
        fus_0 = fus_fea_list[0]#4, 256, 56, 56
        fus_0 = self.conv1(fus_0)#4, 128, 56, 56
        text_fus_0 = self.brg(fus_0, text_features)#
        ir_s1, ir_s2, ir_s3 = self.encoder_ir(input_ir)#4, 32, 224, 224-> 64, 112, 112-> 128, 56, 56
        rgb_s1, rgb_s2, rgb_s3 = self.encoder_rgb(input_rgb)
        fus_1 = self.ffb1(ir_s1, rgb_s1)
        fus_2 = self.ffb2(ir_s2, rgb_s2)
        fus_3 = self.ffb3(ir_s3, rgb_s3)
        cross_fus = self.cross_fus(text_fus_0, fus_3)
        res = self.decoder(fus_1, fus_2, cross_fus)
        return res

    @torch.no_grad()
    def get_img_feature(self, input_fus):
        fus_fea = self.model_clip.encode_image(input_fus.type(self.dtype))
        return fus_fea

