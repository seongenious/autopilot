"""BEV Transformer for multi-camera to BEV feature transformation.

This module transforms multi-camera multi-scale features into a unified
BEV (Bird's Eye View) representation using cross-attention mechanism.

Architecture:
    1. Multi-scale features from each camera are used as K, V
    2. BEV queries are generated from context summary + positional encoding
    3. Cross-attention transforms camera features to BEV space
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from autopilot.utils.registry import NECKS


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for spatial features."""

    def __init__(self, d_model: int, max_h: int = 200, max_w: int = 200) -> None:
        """Initialize 2D positional encoding.

        Args:
            d_model: Embedding dimension.
            max_h: Maximum height.
            max_w: Maximum width.
        """
        super().__init__()
        self.d_model = d_model

        # Create positional encoding
        pe = torch.zeros(d_model, max_h, max_w)
        d_model_half = d_model // 2

        # Height encoding
        h_pos = torch.arange(0, max_h).unsqueeze(1).float()
        div_term_h = torch.exp(
            torch.arange(0, d_model_half, 2).float() * (-math.log(10000.0) / d_model_half)
        )
        pe[0:d_model_half:2, :, :] = torch.sin(h_pos * div_term_h).T.unsqueeze(2).expand(-1, -1, max_w)
        pe[1:d_model_half:2, :, :] = torch.cos(h_pos * div_term_h).T.unsqueeze(2).expand(-1, -1, max_w)

        # Width encoding
        w_pos = torch.arange(0, max_w).unsqueeze(1).float()
        div_term_w = torch.exp(
            torch.arange(0, d_model_half, 2).float() * (-math.log(10000.0) / d_model_half)
        )
        pe[d_model_half::2, :, :] = torch.sin(w_pos * div_term_w).T.unsqueeze(1).expand(-1, max_h, -1)
        pe[d_model_half + 1::2, :, :] = torch.cos(w_pos * div_term_w).T.unsqueeze(1).expand(-1, max_h, -1)

        self.register_buffer('pe', pe)

    def forward(self, h: int, w: int) -> torch.Tensor:
        """Get positional encoding for given spatial size.

        Args:
            h: Height.
            w: Width.

        Returns:
            Positional encoding of shape (d_model, h, w).
        """
        return self.pe[:, :h, :w]


class BEVQueryGenerator(nn.Module):
    """Generate BEV queries from multi-scale multi-camera features.

    Creates BEV queries by:
    1. Average pooling multi-scale features to create context summary
    2. Adding positional encoding
    3. Passing through MLP to generate queries
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        bev_h: int = 20,
        bev_w: int = 80,
        num_cameras: int = 6,
    ) -> None:
        """Initialize BEV query generator.

        Args:
            in_channels: Input feature channels (from BiFPN).
            embed_dim: Output embedding dimension for BEV.
            bev_h: BEV grid height.
            bev_w: BEV grid width.
            num_cameras: Number of cameras.
        """
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cameras = num_cameras
        self.embed_dim = embed_dim

        # Context aggregation
        self.context_proj = nn.Conv2d(in_channels * num_cameras, embed_dim, 1)

        # Positional encoding for BEV grid
        self.bev_pos_enc = PositionalEncoding2D(embed_dim, bev_h, bev_w)

        # MLP for query generation
        self.query_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Learnable BEV embedding
        self.bev_embedding = nn.Parameter(torch.randn(1, embed_dim, bev_h, bev_w) * 0.02)

    def forward(
        self,
        multi_scale_features: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Generate BEV queries.

        Args:
            multi_scale_features: List of [num_cameras] lists of feature tensors.
                Each camera has multiple scale features.

        Returns:
            BEV queries of shape (B, embed_dim, bev_h, bev_w).
        """
        B = multi_scale_features[0][0].shape[0]
        device = multi_scale_features[0][0].device

        # Average pool all features to create context summary
        # First, collect and pool features from all cameras
        context_features = []
        for cam_features in multi_scale_features:
            cam_context = []
            for feat in cam_features:
                # Global average pooling
                pooled = F.adaptive_avg_pool2d(feat, (self.bev_h, self.bev_w))
                cam_context.append(pooled)
            # Average across scales
            cam_context = torch.stack(cam_context, dim=0).mean(dim=0)
            context_features.append(cam_context)

        # Concatenate all camera contexts (B, C*num_cam, H, W)
        context = torch.cat(context_features, dim=1)
        context = self.context_proj(context)  # (B, embed_dim, bev_h, bev_w)

        # Add positional encoding
        pos_enc = self.bev_pos_enc(self.bev_h, self.bev_w).unsqueeze(0).to(device)
        pos_enc = pos_enc.expand(B, -1, -1, -1)

        # Combine context with positional encoding
        # Flatten spatial dimensions for MLP
        context_flat = context.flatten(2).permute(0, 2, 1)  # (B, H*W, embed_dim)
        pos_flat = pos_enc.flatten(2).permute(0, 2, 1)  # (B, H*W, embed_dim)

        combined = torch.cat([context_flat, pos_flat], dim=-1)  # (B, H*W, embed_dim*2)
        queries = self.query_mlp(combined)  # (B, H*W, embed_dim)

        # Reshape back to spatial
        queries = queries.permute(0, 2, 1).view(B, self.embed_dim, self.bev_h, self.bev_w)

        # Add learnable embedding
        queries = queries + self.bev_embedding

        return queries


class MultiScaleDeformableAttention(nn.Module):
    """Multi-scale deformable attention for efficient cross-attention."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
    ) -> None:
        """Initialize multi-scale deformable attention.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_levels: Number of feature levels.
            num_points: Number of sampling points per head per level.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dim, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        nn.init.constant_(self.sampling_offsets.bias.data, 0.)
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with deformable attention.

        For simplicity, this implementation uses standard attention with
        learned sampling. For production, consider using CUDA-optimized
        deformable attention.

        Args:
            query: Query tensor (B, N_q, C).
            key: Key tensor (B, N_k, C).
            value: Value tensor (B, N_k, C).
            reference_points: Reference points for sampling (B, N_q, 2).
            spatial_shapes: Spatial shapes of each level.

        Returns:
            Output tensor (B, N_q, C).
        """
        B, N_q, C = query.shape
        N_k = key.shape[1]

        # Project values
        value = self.value_proj(value)

        # Compute attention weights
        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(
            B, N_q, self.num_heads, self.num_levels * self.num_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Simple attention (can be replaced with deformable attention)
        attn = torch.matmul(
            query.view(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3),
            key.view(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        ) / math.sqrt(C // self.num_heads)
        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(
            attn,
            value.view(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        )
        output = output.permute(0, 2, 1, 3).reshape(B, N_q, C)
        output = self.output_proj(output)

        return output


class BEVTransformerLayer(nn.Module):
    """Single BEV transformer layer with cross-attention and FFN."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        num_levels: int = 4,
    ) -> None:
        """Initialize BEV transformer layer.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            ffn_dim: FFN hidden dimension.
            dropout: Dropout rate.
            num_levels: Number of feature levels.
        """
        super().__init__()

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: BEV queries (B, H*W, C).
            key: Camera features as keys (B, N_cam * H * W, C).
            value: Camera features as values (B, N_cam * H * W, C).

        Returns:
            Updated BEV features (B, H*W, C).
        """
        # Cross-attention with camera features
        q = query
        attn_out, _ = self.cross_attn(q, key, value)
        query = query + self.dropout1(attn_out)
        query = self.norm1(query)

        # Self-attention on BEV
        q = query
        attn_out, _ = self.self_attn(q, q, q)
        query = query + self.dropout2(attn_out)
        query = self.norm2(query)

        # FFN
        query = query + self.ffn(query)
        query = self.norm3(query)

        return query


@NECKS.register_module()
class BEVTransformer(nn.Module):
    """BEV Transformer for multi-camera to BEV transformation.

    Transforms multi-scale features from multiple cameras into a unified
    BEV representation using transformer cross-attention.

    For memory efficiency, multi-scale features are downsampled to a fixed
    spatial size before attention computation.

    Args:
        in_channels: Input feature channels (from BiFPN).
        embed_dim: BEV embedding dimension.
        bev_h: BEV grid height.
        bev_w: BEV grid width.
        num_cameras: Number of cameras.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        ffn_dim: FFN hidden dimension.
        dropout: Dropout rate.
        num_levels: Number of multi-scale levels.
        kv_h: Height to downsample KV features to (for memory efficiency).
        kv_w: Width to downsample KV features to (for memory efficiency).

    Example:
        >>> bev_transformer = BEVTransformer(
        ...     in_channels=128,
        ...     embed_dim=256,
        ...     bev_h=20,
        ...     bev_w=80,
        ... )
        >>> # 6 cameras, 4 scale levels each
        >>> multi_cam_features = [
        ...     [torch.randn(2, 128, h, w) for h, w in [(120,160), (60,80), (30,40), (15,20)]]
        ...     for _ in range(6)
        ... ]
        >>> bev = bev_transformer(multi_cam_features)
        >>> print(bev.shape)  # (2, 256, 20, 80)
    """

    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 256,
        bev_h: int = 20,
        bev_w: int = 80,
        num_cameras: int = 6,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        num_levels: int = 4,
        kv_h: int = 10,
        kv_w: int = 15,
    ) -> None:
        """Initialize BEV Transformer."""
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cameras = num_cameras
        self.num_levels = num_levels
        self.kv_h = kv_h
        self.kv_w = kv_w

        # Project input features to embed_dim
        self.input_proj = nn.ModuleList([
            nn.Conv2d(in_channels, embed_dim, 1)
            for _ in range(num_levels)
        ])

        # BEV query generator
        self.query_generator = BEVQueryGenerator(
            in_channels=in_channels,
            embed_dim=embed_dim,
            bev_h=bev_h,
            bev_w=bev_w,
            num_cameras=num_cameras,
        )

        # Camera positional encoding (learnable)
        self.camera_embed = nn.Parameter(torch.randn(1, num_cameras, embed_dim) * 0.02)

        # Level positional encoding (learnable)
        self.level_embed = nn.Parameter(torch.randn(1, num_levels, embed_dim) * 0.02)

        # Spatial positional encoding for downsampled KV
        self.kv_pos_enc = PositionalEncoding2D(embed_dim, kv_h, kv_w)

        # Transformer layers
        self.layers = nn.ModuleList([
            BEVTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                num_levels=num_levels,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.output_norm = nn.BatchNorm2d(embed_dim)

    def forward(
        self,
        multi_cam_features: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            multi_cam_features: List of length num_cameras, each containing
                a list of num_levels feature tensors.
                Shape: [[cam0_lvl0, cam0_lvl1, ...], [cam1_lvl0, ...], ...]
                Each feature has shape (B, C, H, W).

        Returns:
            BEV features of shape (B, embed_dim, bev_h, bev_w).
        """
        B = multi_cam_features[0][0].shape[0]
        device = multi_cam_features[0][0].device

        # Generate BEV queries
        bev_queries = self.query_generator(multi_cam_features)  # (B, embed_dim, bev_h, bev_w)
        bev_queries = bev_queries.flatten(2).permute(0, 2, 1)  # (B, H*W, embed_dim)

        # Get spatial positional encoding for downsampled features
        kv_pos = self.kv_pos_enc(self.kv_h, self.kv_w).to(device)  # (embed_dim, kv_h, kv_w)
        kv_pos = kv_pos.flatten(1).permute(1, 0).unsqueeze(0)  # (1, kv_h*kv_w, embed_dim)

        # Prepare camera features as K, V with spatial downsampling
        all_features = []
        for cam_idx, cam_features in enumerate(multi_cam_features):
            cam_embed = self.camera_embed[:, cam_idx:cam_idx+1, :]  # (1, 1, embed_dim)
            for lvl_idx, feat in enumerate(cam_features):
                # Project to embed_dim
                feat = self.input_proj[lvl_idx](feat)  # (B, embed_dim, H, W)
                # Downsample to fixed size for memory efficiency
                feat = F.adaptive_avg_pool2d(feat, (self.kv_h, self.kv_w))  # (B, embed_dim, kv_h, kv_w)
                # Flatten and add camera + level embedding
                feat = feat.flatten(2).permute(0, 2, 1)  # (B, kv_h*kv_w, embed_dim)
                feat = feat + cam_embed + self.level_embed[:, lvl_idx:lvl_idx+1, :] + kv_pos
                all_features.append(feat)

        # Concatenate all camera features
        # Shape: (B, num_cam * num_lvl * kv_h * kv_w, embed_dim)
        kv_features = torch.cat(all_features, dim=1)

        # Apply transformer layers
        for layer in self.layers:
            bev_queries = layer(bev_queries, kv_features, kv_features)

        # Reshape to spatial
        bev_features = bev_queries.permute(0, 2, 1).view(
            B, self.embed_dim, self.bev_h, self.bev_w
        )

        # Output projection
        bev_features = self.output_proj(bev_features)
        bev_features = self.output_norm(bev_features)

        return bev_features

    @property
    def out_channels(self) -> int:
        """Return output channel dimension."""
        return self.embed_dim
