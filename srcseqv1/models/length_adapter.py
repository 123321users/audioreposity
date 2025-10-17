
# srcseqv1/models/length_adapter.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
import os
import math
import torch.nn.functional as F
logger = logging.getLogger(__name__)
class DynamicPositionalEncoding(nn.Module):
    """动态位置编码，支持不同长度序列的位置信息学习"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 3000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # 使用预计算的正弦/余弦位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        
        Returns:
            添加位置编码后的张量
        """
        seq_len = x.size(1)
        pos_encoding = self.pe[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        return self.dropout(x)

class MultiScaleDownsampler(nn.Module):
    """多尺度下采样模块，融合不同时间范围的信息"""
    def __init__(self, input_dim: int, output_dim: int, scales: List[int] = [2, 4, 8]):
        super().__init__()
        self.scales = scales
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 每个尺度的输出维度
        per_scale_dim = output_dim // len(scales)
        self.per_scale_dims = [per_scale_dim] * (len(scales) - 1) + [output_dim - per_scale_dim * (len(scales) - 1)]
        
        self.scale_convs = nn.ModuleList()
        for i, scale in enumerate(scales):
            # 每个尺度使用不同的卷积参数
            conv = nn.Conv1d(
                input_dim, 
                self.per_scale_dims[i], 
                kernel_size=scale*2-1,  # 核大小与下采样率相关
                stride=scale,
                padding=scale-1
            )
            self.scale_convs.append(conv)
        
        # 特征融合层
        self.fusion_norm = nn.LayerNorm(output_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, D, T]
        x_transposed = x.transpose(1, 2)
        
        scale_outputs = []
        target_length = None
        
        for i, conv in enumerate(self.scale_convs):
            out = conv(x_transposed)  # [B, per_scale_dim, T_new]
            if target_length is None:
                target_length = out.size(-1)
            elif out.size(-1) != target_length:
                # 对齐长度到最小的输出长度
                target_length = min(target_length, out.size(-1))
        
        # 截断到统一长度并转置回来
        for i, conv in enumerate(self.scale_convs):
            out = conv(x_transposed)
            out = out[:, :, :target_length]  # 截断
            out = out.transpose(1, 2)  # [B, T_new, per_scale_dim]
            scale_outputs.append(out)
        
        # 拼接多尺度特征
        fused_features = torch.cat(scale_outputs, dim=-1)  # [B, T_new, output_dim]
        
        # 门控融合
        gate = self.fusion_gate(fused_features)
        fused_features = gate * fused_features
        fused_features = self.fusion_norm(fused_features)
        
        return fused_features

class AdaptiveAttentionPooling(nn.Module):
    """自适应注意力池化，根据内容动态调整关注重点"""
    def __init__(self, input_dim: int, output_dim: int, num_queries: int, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_queries = num_queries
        
        # 可学习的查询向量
        self.queries = nn.Parameter(torch.randn(num_queries, output_dim) * 0.02)
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # 自适应查询生成器
        self.query_generator = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, num_queries * output_dim)
        )
        
        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 输出处理
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device
        
        # 投影输入
        x_proj = self.input_proj(x)  # [B, T, output_dim]
        
        # 生成自适应查询
        adaptive_queries = self.query_generator(x.transpose(1, 2))  # [B, num_queries * output_dim]
        adaptive_queries = adaptive_queries.view(B, self.num_queries, self.output_dim)
        
        # 与预训练查询融合
        base_queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        queries = 0.7 * base_queries + 0.3 * adaptive_queries
        
        # 交叉注意力池化
        key_padding_mask = ~attention_mask if attention_mask is not None else None
        
        pooled_output, _ = self.cross_attention(
            query=queries,
            key=x_proj,
            value=x_proj,
            key_padding_mask=key_padding_mask
        )
        
        # 残差连接和FFN
        pooled_output = self.output_norm(queries + pooled_output)
        ffn_output = self.output_ffn(pooled_output)
        final_output = self.output_norm(pooled_output + ffn_output)
        
        return final_output

class LengthAdapter(nn.Module):
    """层次化长度适配器：结合多尺度下采样和自适应注意力池化"""
    
    def __init__(self, input_dim: int, output_dim: int, adapter_config: Dict[str, Any]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = adapter_config
        
        # 获取配置参数
        self.num_queries = adapter_config.get('num_queries', 256)
        self.use_multiscale = adapter_config.get('use_multiscale_downsampling', False)
        self.use_attention_pooling = adapter_config.get('use_attention_pooling', False)
        self.num_transformer_layers = adapter_config.get('num_transformer_layers', 3)
        self.downsample_scales = adapter_config.get('downsample_scales', [2, 4, 8])
        
        logger.info(f"初始化层次化长度适配器:")
        logger.info(f"  - 输入维度: {input_dim}, 输出维度: {output_dim}")
        logger.info(f"  - 目标序列长度: {self.num_queries}")
        logger.info(f"  - 多尺度下采样: {self.use_multiscale}, 尺度: {self.downsample_scales}")
        logger.info(f"  - 注意力池化: {self.use_attention_pooling}")
        logger.info(f"  - Transformer层数: {self.num_transformer_layers}")
        
        # 构建网络层
        self._build_layers()
        
    def _build_layers(self):
        """构建网络层"""
        current_dim = self.input_dim
        
        # 第一阶段：多尺度下采样（可选）
        if self.use_multiscale:
            intermediate_dim = self.output_dim if not self.use_attention_pooling else self.output_dim
            self.multiscale_downsampler = MultiScaleDownsampler(
                input_dim=current_dim,
                output_dim=intermediate_dim,
                scales=self.downsample_scales
            )
            current_dim = intermediate_dim
        
        # 第二阶段：自适应注意力池化（可选）
        if self.use_attention_pooling:
            self.attention_pooler = AdaptiveAttentionPooling(
                input_dim=current_dim,
                output_dim=self.output_dim,
                num_queries=self.num_queries,
                num_heads=self.config.get('num_heads', 8)
            )
            current_dim = self.output_dim
        
        # 第三阶段：Transformer精化
        if self.num_transformer_layers > 0:
            self.pos_encoding = DynamicPositionalEncoding(
                d_model=self.output_dim,
                dropout=self.config.get('dropout', 0.1)
            )
            
            encoder_layer = TransformerEncoderLayer(
                d_model=self.output_dim,
                nhead=self.config.get('num_heads', 8),
                dim_feedforward=self.config.get('ffn_dim', self.output_dim * 4),
                dropout=self.config.get('dropout', 0.1),
                batch_first=True,
                norm_first=True  # Pre-norm for better training stability
            )
            
            self.transformer_encoder = TransformerEncoder(
                encoder_layer,
                num_layers=self.num_transformer_layers
            )
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.output_dim),
            nn.Dropout(self.config.get('dropout', 0.1))
        )
        
        # 如果既不用多尺度也不用注意力池化，则使用传统卷积
        if not self.use_multiscale and not self.use_attention_pooling:
            stride = max(1, self.config.get('fallback_stride', 4))
            self.fallback_conv = nn.Conv1d(
                self.input_dim, self.output_dim,
                kernel_size=7, stride=stride, padding=3
            )
            logger.info(f"  - 使用传统卷积作为后备方案 (stride={stride})")
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                audio_id: Optional[str] = None, epoch: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 输入检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("!!! LengthAdapter输入包含NaN或Inf !!!")
            
        B, T, D = x.shape
        device = x.device
        
        # 第一阶段：多尺度下采样
        if self.use_multiscale:
            x = self.multiscale_downsampler(x)
            logger.debug(f"多尺度下采样后形状: {x.shape}")
            
            # 更新attention mask
            if attention_mask is not None:
                new_length = x.size(1)
                # 简单的mask适配（可以优化为更精确的方法）
                ratio = new_length / attention_mask.size(1)
                if ratio <= 1.0:
                    # 下采样mask
                    indices = torch.linspace(0, attention_mask.size(1)-1, new_length, device=device).long()
                    attention_mask = attention_mask.gather(1, indices.unsqueeze(0).expand(B, -1))
                else:
                    # 上采样mask（通常不会发生）
                    attention_mask = F.interpolate(
                        attention_mask.unsqueeze(1).float(),
                        size=new_length,
                        mode='nearest'
                    ).squeeze(1).bool()
        
        # 第二阶段：注意力池化
        if self.use_attention_pooling:
            x = self.attention_pooler(x, attention_mask)
            attention_mask = torch.ones(B, self.num_queries, device=device, dtype=torch.bool)
            logger.debug(f"注意力池化后形状: {x.shape}")
        
        # 后备方案
        if not self.use_multiscale and not self.use_attention_pooling:
            x_conv = x.transpose(1, 2)  # [B, D, T]
            x_conv = self.fallback_conv(x_conv)
            x = x_conv.transpose(1, 2)  # [B, T', output_dim]
            
            if attention_mask is not None:
                new_T = x.size(1)
                stride = self.config.get('fallback_stride', 4)
                # 简化的mask下采样
                attention_mask = attention_mask[:, ::stride][:, :new_T]
        
        # 第三阶段：Transformer精化
        if self.num_transformer_layers > 0:
            x = self.pos_encoding(x)
            
            # 准备padding mask for transformer
            src_key_padding_mask = ~attention_mask if attention_mask is not None else None
            
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
            logger.debug(f"Transformer精化后形状: {x.shape}")
        
        # 输出投影
        x = self.output_projection(x)
        
        # 最终检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("!!! LengthAdapter输出包含NaN或Inf !!!")
        
        logger.debug(f"最终输出统计: 均值={x.mean().item():.4f}, 标准差={x.std().item():.4f}")
        
        return x, attention_mask

    def get_output_length(self, input_length: int) -> int:
        """计算给定输入长度的输出长度"""
        if self.use_attention_pooling:
            return self.num_queries
        
        if self.use_multiscale:
            # 使用最大下采样尺度估算
            max_scale = max(self.downsample_scales)
            return input_length // max_scale
        
        # 后备方案
        stride = self.config.get('fallback_stride', 4)
        return input_length // stride
