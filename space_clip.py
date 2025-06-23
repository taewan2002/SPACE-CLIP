# space_clip.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel
import math

class ResidualConvBlock(nn.Module):
    def __init__(self, num_features, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(num_features)

    def forward(self, x):
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        x += residual
        return self.relu(x)

class FiLMLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, features, gamma, beta):
        if features.ndim == 3 and gamma.ndim == 2:
            gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
        elif features.ndim == 4 and gamma.ndim == 2:
            gamma, beta = gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(self.upsample(x))

class MainDecoderBlock(nn.Module):
    def __init__(self, low_res_ch, skip_clip_ch, skip_early_refined_ch, out_ch, dropout=0.1, **kwargs):
        super().__init__()
        self.upsample_low = UpsampleBlock(low_res_ch, out_ch)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_clip_ch + skip_early_refined_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResidualConvBlock(out_ch, dropout)
        )

    def forward(self, low_res_feat, skip_feat_clip, skip_feat_early_refined):
        low_res_up = self.upsample_low(low_res_feat)
        
        # 스킵 연결들의 크기를 low_res_up에 맞춤
        target_size = low_res_up.shape[-2:]
        if skip_feat_clip.shape[-2:] != target_size:
            skip_feat_clip = F.interpolate(skip_feat_clip, size=target_size, mode='bilinear', align_corners=False)
        if skip_feat_early_refined.shape[-2:] != target_size:
            skip_feat_early_refined = F.interpolate(skip_feat_early_refined, size=target_size, mode='bilinear', align_corners=False)

        combined_feat = torch.cat([low_res_up, skip_feat_clip, skip_feat_early_refined], dim=1)
        
        return self.fusion_conv(combined_feat)


class StructuralPathwayBlock(nn.Module):
    def __init__(self, low_res_ch, skip_early_ch, out_ch, **kwargs):
        super().__init__()
        self.upsample_low = UpsampleBlock(low_res_ch, out_ch)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_early_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_res_feat, skip_feat_early):
        low_res_up = self.upsample_low(low_res_feat)
        
        target_size = low_res_up.shape[-2:]
        if skip_feat_early.shape[-2:] != target_size:
            skip_feat_early = F.interpolate(skip_feat_early, size=target_size, mode='bilinear', align_corners=False)
        # ----------------------------------------------------

        combined_feat = torch.cat([low_res_up, skip_feat_early], dim=1)
        return self.fusion_conv(combined_feat)

class SPACECLIP(nn.Module):
    def __init__(self, config_model):
        super().__init__()
        self.config = vars(config_model) if not isinstance(config_model, dict) else config_model
        
        self.use_film = self.config.get('use_film', True)
        self.use_structural_pathway = self.config.get('use_structural_pathway', True)
        
        print("--- SPACE-CLIP Configuration ---")
        print(f"Global Context (FiLM): {self.use_film}")
        print(f"Structural Refinement Pathway: {self.use_structural_pathway}")
        print("---------------------------------")

        # --- 1. Backbone ---
        clip_model_name = self.config.get('clip_model_name', "openai/clip-vit-base-patch16")
        self.clip_vision_model = CLIPVisionModel.from_pretrained(clip_model_name)
        if self.config.get('freeze_clip_vision', True):
            for param in self.clip_vision_model.parameters():
                param.requires_grad = False
        self.clip_vision_model.eval()
        self.clip_config = self.clip_vision_model.config
        self.clip_embed_dim = self.clip_config.hidden_size
        self.num_patches_sqrt = self.clip_config.image_size // self.clip_config.patch_size
        
        self.main_path_indices = self.config.get('main_path_indices', [12, 9, 6, 3])
        self.structural_path_indices = self.config.get('structural_path_indices', [2, 1, 0])

        # --- 2. FiLM Module ---
        if self.use_film:
            self.film_layer = FiLMLayer()
            film_param_hidden_dim = self.config.get('film_param_hidden_dim', 256)
            self.film_param_generators = nn.ModuleList([
                nn.Sequential(nn.Linear(self.clip_embed_dim, film_param_hidden_dim), nn.ReLU(inplace=True),
                              nn.Linear(film_param_hidden_dim, 2 * self.clip_embed_dim))
                for _ in self.main_path_indices
            ])

        # --- 3. 디코더 경로(Pathway) 구축 ---
        decoder_channels = self.config.get('decoder_channels', [256, 128, 64, 32])
        
        # 3.1 메인 디코더 경로
        self.main_path_projections = nn.ModuleList([
            nn.Conv2d(self.clip_embed_dim, ch, 1) for ch in decoder_channels
        ])
        self.main_decoder_blocks = nn.ModuleList()

        # 3.2 구조적 디코더 경로
        if self.use_structural_pathway:
            self.structural_path_projections = nn.ModuleList([
                nn.Conv2d(self.clip_embed_dim, ch, 1) for ch in decoder_channels[:len(self.structural_path_indices)]
            ])
            self.structural_decoder_blocks = nn.ModuleList()

        # 3.3 디코더 블록 생성
        for i in range(len(decoder_channels)):
            main_in_ch = decoder_channels[i-1] if i > 0 else decoder_channels[0]
            main_skip_ch = decoder_channels[i]
            
            if self.use_structural_pathway and i < len(self.structural_path_indices):
                struct_in_ch = decoder_channels[i-1] if i > 0 else self.clip_embed_dim
                struct_skip_ch = decoder_channels[i]
                self.structural_decoder_blocks.append(
                    StructuralPathwayBlock(struct_in_ch, struct_skip_ch, decoder_channels[i])
                )
                struct_refined_ch = decoder_channels[i]
            else:
                struct_refined_ch = 0
            
            self.main_decoder_blocks.append(
                MainDecoderBlock(main_in_ch, main_skip_ch, struct_refined_ch, decoder_channels[i])
            )

        # --- 4. 최종 깊이 예측 헤드 ---
        final_ch = decoder_channels[-1]
        self.depth_prediction_head = nn.Sequential(
            UpsampleBlock(final_ch, final_ch // 2),
            nn.Conv2d(final_ch // 2, 1, 3, padding=1),
            nn.ReLU()
        )

    def _reshape_patch_tokens(self, patch_tokens, B):
        return patch_tokens.permute(0, 2, 1).reshape(B, self.clip_embed_dim, self.num_patches_sqrt, self.num_patches_sqrt)

    def forward(self, pixel_values: torch.Tensor) -> tuple:
        B, C, H, W = pixel_values.shape

        with torch.no_grad():
            clip_out = self.clip_vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        
        pooler_out = clip_out.pooler_output
        hidden_states = clip_out.hidden_states

        # --- 1. 경로별 특징 추출 및 변조 ---
        main_feats = []
        for i, idx in enumerate(self.main_path_indices):
            patch_tokens = hidden_states[idx][:, 1:, :]
            if self.use_film:
                gamma, beta = torch.chunk(self.film_param_generators[i](pooler_out), 2, dim=-1)
                mod_patch_tokens = self.film_layer(patch_tokens, gamma, beta)
            else:
                mod_patch_tokens = patch_tokens
            feat_map = self._reshape_patch_tokens(mod_patch_tokens, B)
            main_feats.append(self.main_path_projections[i](feat_map))

        if self.use_structural_pathway:
            struct_feats = []
            for i, idx in enumerate(self.structural_path_indices):
                patch_tokens = hidden_states[idx][:, 1:, :]
                feat_map = self._reshape_patch_tokens(patch_tokens, B)
                struct_feats.append(self.structural_path_projections[i](feat_map))
        
        # --- 2. 디코더 실행 ---
        main_path_x = main_feats[0]
        struct_path_x = self._reshape_patch_tokens(hidden_states[self.structural_path_indices[0]][:,1:,:], B) if self.use_structural_pathway else None

        for i in range(len(self.main_decoder_blocks)):
            if self.use_structural_pathway and i < len(self.structural_decoder_blocks):
                struct_skip = struct_feats[i]
                struct_path_x = self.structural_decoder_blocks[i](struct_path_x, struct_skip)
                skip_struct_refined = struct_path_x
            else:
                B, _, H_main, W_main = main_path_x.shape
                target_h, target_w = (H_main * 2, W_main * 2) if i > 0 else (H_main, W_main)
                out_ch = self.main_decoder_blocks[i].fusion_conv[0].out_channels
                skip_struct_refined = torch.zeros(B, 0, target_h, target_w, device=pixel_values.device)

            skip_main = main_feats[i]
            main_path_x = self.main_decoder_blocks[i](main_path_x, skip_main, skip_struct_refined)

        # --- 3. 최종 깊이 예측 ---
        depth_map = self.depth_prediction_head(main_path_x)
        if depth_map.shape[-2:] != (H, W):
            depth_map = F.interpolate(depth_map, size=(H, W), mode='bilinear', align_corners=False)
        
        return None, depth_map