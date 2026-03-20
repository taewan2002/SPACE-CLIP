# space_clip.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, SiglipVisionModel
from typing import Optional, Tuple

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
        self.skip_clip_gate = nn.Sequential(
            nn.Conv2d(out_ch, skip_clip_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.skip_struct_gate = None
        if skip_early_refined_ch > 0:
            self.skip_struct_gate = nn.Sequential(
                nn.Conv2d(out_ch, skip_early_refined_ch, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

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

        clip_gate = self.skip_clip_gate(low_res_up)
        skip_feat_clip = skip_feat_clip * (1.0 + clip_gate)

        if self.skip_struct_gate is not None and skip_feat_early_refined.shape[1] > 0:
            struct_gate = self.skip_struct_gate(low_res_up)
            skip_feat_early_refined = skip_feat_early_refined * (1.0 + struct_gate)

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
        self.vision_backbone_type = self._resolve_backbone_type(clip_model_name)
        self.clip_vision_model = self._load_vision_backbone(clip_model_name)
        if self.config.get('freeze_clip_vision', True):
            for param in self.clip_vision_model.parameters():
                param.requires_grad = False
        self.clip_vision_model.eval()
        self.clip_config = self.clip_vision_model.config
        self.clip_embed_dim = self.clip_config.hidden_size
        self.patch_size = self.clip_config.patch_size
        self.vision_has_cls_token = self.vision_backbone_type == "clip"
        self.clip_interpolate_pos_encoding = self.config.get('clip_interpolate_pos_encoding', False)
        print(f"Vision Backbone: {self.vision_backbone_type}")
        print(f"CLIP Positional Interpolation: {self.clip_interpolate_pos_encoding}")
        
        self.main_path_indices = self.config.get('main_path_indices', [12, 9, 6, 3])
        self.structural_path_indices = self.config.get('structural_path_indices', [2, 1, 0])
        self.use_multiscale_supervision = self.config.get('use_multiscale_supervision', False)

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
        decoder_dropout = self.config.get('decoder_dropout', 0.1)
        
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
                MainDecoderBlock(
                    main_in_ch,
                    main_skip_ch,
                    struct_refined_ch,
                    decoder_channels[i],
                    dropout=decoder_dropout,
                )
            )

        # --- 4. 최종 깊이 예측 헤드 ---
        final_ch = decoder_channels[-1]
        self.depth_prediction_head = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(final_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_ch // 2, 1, 3, padding=1),
            nn.ReLU()
        )

        self.aux_depth_heads = nn.ModuleList()
        if self.use_multiscale_supervision and len(decoder_channels) > 1:
            for ch in decoder_channels[:-1]:
                mid_ch = max(ch // 2, 16)
                self.aux_depth_heads.append(
                    nn.Sequential(
                        nn.Conv2d(ch, mid_ch, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(mid_ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(mid_ch, 1, kernel_size=1),
                        nn.ReLU(),
                    )
                )

    def _resolve_backbone_type(self, model_name: str) -> str:
        explicit = str(self.config.get("vision_backbone_type", "auto")).strip().lower()
        if explicit in {"clip", "siglip"}:
            return explicit
        if "siglip" in model_name.lower():
            return "siglip"
        return "clip"

    def _load_vision_backbone(self, model_name: str):
        if self.vision_backbone_type == "siglip":
            return SiglipVisionModel.from_pretrained(model_name)
        return CLIPVisionModel.from_pretrained(model_name)

    def _extract_patch_tokens(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.vision_has_cls_token:
            return hidden_state[:, 1:, :]
        return hidden_state

    def _reshape_patch_tokens(self, patch_tokens, B, patch_grid_h, patch_grid_w):
        return patch_tokens.permute(0, 2, 1).reshape(B, self.clip_embed_dim, patch_grid_h, patch_grid_w)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        return_intermediates: bool = False,
    ) -> tuple:
        B, C, H, W = pixel_values.shape
        patch_grid_h = H // self.patch_size
        patch_grid_w = W // self.patch_size
        expected_hw = (self.clip_config.image_size, self.clip_config.image_size)

        if (H, W) != expected_hw and not self.clip_interpolate_pos_encoding:
            raise ValueError(
                "Non-224 CLIP input requires clip_interpolate_pos_encoding=true in the config. "
                f"Got input {(H, W)} with expected {expected_hw}."
            )

        with torch.no_grad():
            vision_kwargs = {
                "pixel_values": pixel_values,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if self.vision_backbone_type == "clip":
                vision_kwargs["interpolate_pos_encoding"] = self.clip_interpolate_pos_encoding
            try:
                clip_out = self.clip_vision_model(**vision_kwargs)
            except TypeError:
                if self.clip_interpolate_pos_encoding and (H, W) != expected_hw and self.vision_backbone_type == "clip":
                    raise RuntimeError(
                        "This transformers version does not support interpolate_pos_encoding for CLIPVisionModel. "
                        "Please upgrade transformers or disable use_image_orig_for_clip."
                    )
                vision_kwargs.pop("interpolate_pos_encoding", None)
                clip_out = self.clip_vision_model(**vision_kwargs)
        
        pooler_out = clip_out.pooler_output
        hidden_states = clip_out.hidden_states

        # --- 1. 경로별 특징 추출 및 변조 ---
        main_feats = []
        for i, idx in enumerate(self.main_path_indices):
            patch_tokens = self._extract_patch_tokens(hidden_states[idx])
            if self.use_film:
                gamma, beta = torch.chunk(self.film_param_generators[i](pooler_out), 2, dim=-1)
                mod_patch_tokens = self.film_layer(patch_tokens, gamma, beta)
            else:
                mod_patch_tokens = patch_tokens
            feat_map = self._reshape_patch_tokens(mod_patch_tokens, B, patch_grid_h, patch_grid_w)
            main_feats.append(self.main_path_projections[i](feat_map))

        struct_feats = []
        if self.use_structural_pathway:
            for i, idx in enumerate(self.structural_path_indices):
                patch_tokens = self._extract_patch_tokens(hidden_states[idx])
                feat_map = self._reshape_patch_tokens(patch_tokens, B, patch_grid_h, patch_grid_w)
                struct_feats.append(self.structural_path_projections[i](feat_map))
        
        # --- 2. 디코더 실행 ---
        main_path_x = main_feats[0]
        struct_path_x = self._reshape_patch_tokens(
            self._extract_patch_tokens(hidden_states[self.structural_path_indices[0]]), B, patch_grid_h, patch_grid_w
        ) if self.use_structural_pathway else None

        decoder_stage_features = []
        structural_decoder_features = []
        for i in range(len(self.main_decoder_blocks)):
            if self.use_structural_pathway and i < len(self.structural_decoder_blocks):
                struct_skip = struct_feats[i]
                struct_path_x = self.structural_decoder_blocks[i](struct_path_x, struct_skip)
                skip_struct_refined = struct_path_x
                structural_decoder_features.append(struct_path_x)
            else:
                B, _, H_main, W_main = main_path_x.shape
                target_h, target_w = (H_main * 2, W_main * 2) if i > 0 else (H_main, W_main)
                out_ch = self.main_decoder_blocks[i].fusion_conv[0].out_channels
                skip_struct_refined = torch.zeros(B, 0, target_h, target_w, device=pixel_values.device)

            skip_main = main_feats[i]
            main_path_x = self.main_decoder_blocks[i](main_path_x, skip_main, skip_struct_refined)
            decoder_stage_features.append(main_path_x)

        # --- 3. 최종 깊이 예측 ---
        depth_map = self.depth_prediction_head(main_path_x)
        target_size = output_size if output_size is not None else (H, W)
        if depth_map.shape[-2:] != target_size:
            depth_map = F.interpolate(depth_map, size=target_size, mode='bilinear', align_corners=False)

        aux_predictions = None
        if self.use_multiscale_supervision and len(self.aux_depth_heads) > 0:
            aux_predictions = []
            for aux_head, feat in zip(self.aux_depth_heads, decoder_stage_features[:-1]):
                aux_depth = aux_head(feat)
                if aux_depth.shape[-2:] != target_size:
                    aux_depth = F.interpolate(aux_depth, size=target_size, mode='bilinear', align_corners=False)
                aux_predictions.append(aux_depth)

        if not return_intermediates:
            return aux_predictions, depth_map

        intermediates = {
            'pooler_output': pooler_out,
            'semantic_projected': tuple(main_feats),
            'structural_projected': tuple(struct_feats),
            'semantic_decoder': tuple(decoder_stage_features),
            'structural_decoder': tuple(structural_decoder_features),
            'patch_grid_hw': (patch_grid_h, patch_grid_w),
            'semantic_layer_indices': tuple(self.main_path_indices),
            'structural_layer_indices': tuple(self.structural_path_indices),
        }
        return aux_predictions, depth_map, intermediates
