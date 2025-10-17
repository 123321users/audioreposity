
# srcseqv1/models/length_adapter.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
import os
import math

logger = logging.getLogger(__name__)





class LengthAdapter(nn.Module):
    """
    长度适配器模块，基于 Transformer 架构，用于调整序列长度。

    通过带步长的卷积进行下采样，然后接 Transformer 编码器层。
    """
    def __init__(self, input_dim: int, output_dim: int, adapter_config: Dict[str, Any]):
        """
        初始化长度适配器。

        Args:
            input_dim: 输入特征的维度 (音频编码器输出维度)。
            output_dim: 输出特征的维度 (文本解码器输入维度)。
            adapter_config: 配置字典，包含 Transformer 层和下采样参数。
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adapter_config = adapter_config

        num_layers = adapter_config.get('num_layers', 2)
        num_heads = adapter_config.get('num_heads', 8)
        ffn_dim = adapter_config.get('ffn_dim', input_dim * 4)
        self.dropout = adapter_config.get('dropout', 0.1)
        downsample_method = adapter_config.get('downsample_method', 'stride_conv')
        downsample_kernel_size = adapter_config.get('downsample_kernel_size', 3)
        downsample_stride = adapter_config.get('downsample_stride', 4)
        downsample_padding = adapter_config.get('downsample_padding', 1)

        # 1. 下采样层 (使用带步长的 Conv1D)
        if downsample_method == 'stride_conv':
            self.downsampler = nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=downsample_kernel_size,
                stride=downsample_stride,
                padding=downsample_padding
            )
            self.downsample_norm = nn.LayerNorm(output_dim)
            self.downsample_activation = nn.ReLU()
            logger.info(f"LengthAdapter: 使用带步长 Conv1D ({downsample_stride}) 进行下采样。")
        else:
            raise NotImplementedError(f"未实现的下采样方法: {downsample_method}")

        # 2. Transformer 编码器层
        encoder_layer = TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=self.dropout,
            batch_first=True,
            norm_first=False
        )
        self.transformer_layers = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        logger.info(f"LengthAdapter: 使用 {num_layers} 个 Transformer 编码器层进行适配。")

        # 位置编码
        max_estimated_time = 5000
        self.position_encoding = PositionalEncoding(self.output_dim, dropout=self.dropout, max_len=max_estimated_time)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        长度适配器的前向传播。

        Args:
            x: 来自音频编码器的输入张量，形状 [batch_size, Time, Dim]。
            attention_mask: 对应的 padding mask，形状 [batch_size, Time] (True 表示非填充部分)。

        Returns:
            适配后的张量，形状 [batch_size, New_Time, output_dim]。
            适配后的 padding mask，形状 [batch_size, New_Time] (True 表示非填充部分)。
        """
        
        logger.info(f"\n----------------------------进入 长度适配器-----------------------------")
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("!!! 严重错误: LengthAdapter 的输入包含 NaN 或 Inf !!!")
        # logger.info(f"来自音频编码器的音频特征: {x}, 形状为: {x.shape}) ---")
        # logger.info(f"输入注意力掩码：{attention_mask}，形状为：{attention_mask.shape}")
        # logger.debug(f"  LengthAdapter输入统计: 均值={x.mean().item():.4f}, 标准差={x.std().item():.4f}, 最大值={x.max().item():.4f}")


        batch_size, original_time, input_dim = x.shape

        if input_dim != self.input_dim:
            logger.warning(f"LengthAdapter: 输入维度不匹配。初始化期望 {self.input_dim}，实际接收 {input_dim}。这可能导致问题。")

        # 1. 下采样
        x = x.permute(0, 2, 1) # 将输入从 [B, Time, Dim] 转换为 [B, Dim, Time] 进行 Conv1D

        downsampled_x = self.downsampler(x) # 应用下采样卷积

        downsampled_x = downsampled_x.permute(0, 2, 1) # 转回 [B, Downsampled_Time, output_dim]

        if hasattr(self, 'downsample_norm'):
            downsampled_x = self.downsample_norm(downsampled_x)
        if hasattr(self, 'downsample_activation'):
             downsampled_x = self.downsample_activation(downsampled_x)

        new_time = downsampled_x.size(1)

        # 适配 attention mask
        adapted_attention_mask = None
        if attention_mask is not None:
            mask_for_conv = attention_mask.unsqueeze(1).float()

            mask_downsampler = nn.Conv1d(
                1, 1,
                kernel_size=self.downsampler.kernel_size,
                stride=self.downsampler.stride,
                padding=self.downsampler.padding,
                bias=False
            ).to(mask_for_conv.device)
            if mask_downsampler.weight.size(2) == self.downsampler.kernel_size[0]:
                 mask_downsampler.weight.data.fill_(1.0 / self.downsampler.kernel_size[0])
            else:
                 logger.warning(f"LengthAdapter mask_downsampler: 卷积核大小不匹配 downsampler ({mask_downsampler.weight.size(2)} vs {self.downsampler.kernel_size[0]}). 无法初始化权重。")

            if mask_for_conv.ndim == 3 and mask_for_conv.size(1) == 1:
                 adapted_mask_float = mask_downsampler(mask_for_conv).squeeze(1)
                 adapted_attention_mask = (adapted_mask_float > 1e-3)
            else:
                 logger.error(f"LengthAdapter mask_downsampler: 输入 mask 形状 {mask_for_conv.shape} 不符合 Conv1d 期望 [B, 1, T]。无法计算适配后的 mask。")
                 adapted_attention_mask = torch.zeros(batch_size, new_time, dtype=torch.bool, device=x.device)

            if logger.isEnabledFor(logging.DEBUG) or (logger.isEnabledFor(logging.INFO) and random.random() < 0.01):
                original_lengths = attention_mask.sum(dim=1)
                estimated_adapted_lengths = adapted_attention_mask.sum(dim=1)
                logger.debug(f"LengthAdapter: 原始时间步长 {original_time}, 适配后时间步长 {new_time}")
                logger.debug(f"LengthAdapter: 原始真实长度 (前 2 个样本): {original_lengths[:min(2, batch_size)].tolist()}")
                logger.debug(f"LengthAdapter: 适配后真实长度 (前 2 个样本): {estimated_adapted_lengths[:min(2, batch_size)].tolist()}")





        # adapted_attention_mask = None
        # if attention_mask is not None:
        #     # 导入 max_pool1d 函数
        #     from torch.nn.functional import max_pool1d
            
        #     # 将布尔掩码 [B, T] 转换为浮点型并增加通道维度 [B, 1, T]
        #     # 这是为了适应 max_pool1d 的输入格式
        #     mask_float = attention_mask.float().unsqueeze(1)
            
        #     # 使用与 self.downsampler 完全相同的参数进行最大池化。
        #     # 最大池化能正确地传播“只要窗口内有一个有效帧，输出就有效”的逻辑。
        #     # 这是一个无状态的函数调用，高效且准确。
        #     adapted_mask_float = max_pool1d(
        #         mask_float,
        #         kernel_size=self.downsampler.kernel_size[0],
        #         stride=self.downsampler.stride[0],
        #         padding=self.downsampler.padding[0],
        #         ceil_mode=False  # 确保与 Conv1d 的默认长度计算行为一致
        #     )
            
        #     # 将池化后的浮点型掩码 [B, 1, New_T] 转回布尔型 [B, New_T]
        #     # 经过最大池化后，结果只可能是 0 或 1，用 > 0.5 判断可以稳健地转为布尔型
        #     adapted_attention_mask = (adapted_mask_float.squeeze(1) > 0.5)

        #     # (可选) 调试日志，检查长度变化是否符合预期
        #     if logger.isEnabledFor(logging.DEBUG):
        #         original_lengths = attention_mask.sum(dim=1)
        #         adapted_lengths = adapted_attention_mask.sum(dim=1)
        #         logger.debug(f"LengthAdapter Masking: 原始真实长度 (前2个): {original_lengths[:2].tolist()}")
        #         logger.debug(f"LengthAdapter Masking: 适配后真实长度 (前2个): {adapted_lengths[:2].tolist()}")

        # 2. 通过 Transformer 编码器层
        downsampled_x_pos_encoded = self.position_encoding(downsampled_x)

        transformer_padding_mask = ~adapted_attention_mask if adapted_attention_mask is not None else None
        # logger.debug(f"LengthAdapter: 适配后注意力掩码: {transformer_padding_mask}，形状为{transformer_padding_mask.shape}")


        adapted_output = self.transformer_layers(
            downsampled_x_pos_encoded,
            src_key_padding_mask=transformer_padding_mask
        )

        # if logger.isEnabledFor(logging.DEBUG):
        #      logger.debug(f"LengthAdapter: 适配后输出形状: {adapted_output.shape}, mask 形状: {adapted_attention_mask.shape if adapted_attention_mask is not None else 'None'}")
        # logger.info(f" 长度适配器输出adapted_attention_mask是：形状: {adapted_attention_mask.shape}")
        # logger.info(f" 长度适配器adapted_output输出形状: {adapted_output.shape}")
        logger.info(f"*********************************退出 长度适配器*************************************\n")

        return adapted_output, adapted_attention_mask

class PositionalEncoding(nn.Module):
     """
     位置编码模块，为文本序列添加位置信息。(在这里用于 LengthAdapter 输出的序列)
     """
     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

     def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)






# class AdapterBlock(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim, num_heads, dropout=dropout, batch_first=True
#         )
#         self.sa_norm = nn.LayerNorm(embed_dim)
#         self.sa_dropout = nn.Dropout(dropout)

#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim, num_heads, dropout=dropout, batch_first=True
#         )
#         self.ca_norm = nn.LayerNorm(embed_dim)
#         self.ca_dropout = nn.Dropout(dropout)

#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, ffn_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(ffn_dim, embed_dim)
#         )
#         self.ffn_norm = nn.LayerNorm(embed_dim)
#         self.ffn_dropout = nn.Dropout(dropout)

#     def forward(
#         self,
#         query: torch.Tensor,
#         key_value: torch.Tensor,
#         key_padding_mask: Optional[torch.Tensor] = None,
#         need_weights: bool = False
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
#         sa_output, _ = self.self_attn(query, query, query)
#         query = self.sa_norm(query + self.sa_dropout(sa_output))


#         ca_output, ca_weights = self.cross_attn(
#             query=query, 
#             key=key_value, 
#             value=key_value, 
#             key_padding_mask=key_padding_mask,
#             need_weights=need_weights,
#             average_attn_weights=False 
#         )
#         query = self.ca_norm(query + self.ca_dropout(ca_output))
        
#         ffn_output = self.ffn(query)
#         query = self.ffn_norm(query + self.ffn_dropout(ffn_output))
        
#         return query, ca_weights


# class LengthAdapter(nn.Module):
#     def __init__(self, input_dim: int, output_dim: int, adapter_config: Dict[str, Any]):
#         super().__init__()
        
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.num_queries = adapter_config.get("num_queries", 256)
#         num_layers = adapter_config.get("num_layers", 4)
        
#         logger.info(f"长度适配器 (最终版): 初始化 {num_layers} 个迭代式 AdapterBlock。")
#         logger.info(f"  - 输入维度: {input_dim}, 输出维度: {output_dim}")
#         logger.info(f"  - 输出固定长度 (查询数): {self.num_queries}")

#         self.queries = nn.Parameter(torch.randn(self.num_queries, output_dim))
#         self.input_projection = nn.Linear(input_dim, output_dim)
        
#         self.adapter_layers = nn.ModuleList([
#             AdapterBlock(
#                 embed_dim=output_dim,
#                 num_heads=adapter_config.get("num_heads", 8),
#                 ffn_dim=adapter_config.get("ffn_dim", output_dim * 4),
#                 dropout=adapter_config.get("dropout", 0.1)
#             ) for _ in range(num_layers)
#         ])
        
#         self.query_pos_encoder = nn.Parameter(torch.randn(1, self.num_queries, output_dim) * 0.02)
#         logger.info("  - 已为查询向量添加可学习的位置编码。")
#         logger.info("  - 注意力可视化功能已启用。")


#     def forward(
#         self, 
#         x: torch.Tensor, 
#         attention_mask: Optional[torch.Tensor] = None,
#         audio_id: Optional[str] = None, 
#         epoch: Optional[int] = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         logger.info(f"\n----------------------------进入 长度适配器-----------------------------")
        

#         if torch.isnan(x).any() or torch.isinf(x).any():
#             logger.error("!!! 严重错误: LengthAdapter 的输入包含 NaN 或 Inf !!!")
#         logger.info(f"来自音频编码器的输入张量: x, 形状为: {x.shape}) ---")
#         logger.info(f"输入注意力掩码：attention_mask，形状为：{attention_mask.shape}")
#         logger.debug(f"  LengthAdapter输入统计: 均值={x.mean().item():.4f}, 标准差={x.std().item():.4f}, 最大值={x.max().item():.4f}")

#         batch_size, T_encoder, input_dim = x.shape

#         if input_dim != self.input_dim:
#             logger.warning(f"LengthAdapter: 输入维度不匹配。初始化期望 {self.input_dim}，实际接收 {input_dim}。这可能导致问题。")
#         device = x.device

#         projected_x = self.input_projection(x)
#         queries = self.queries.unsqueeze(0).repeat(batch_size, 1, 1)
#         queries = queries + self.query_pos_encoder

#         if attention_mask is not None:
#             cross_attn_mask = ~(attention_mask.bool())
#         else:
#             cross_attn_mask = None

#         for i, layer in enumerate(self.adapter_layers):
#             is_last_layer = (i == len(self.adapter_layers) - 1)
            
#             queries, weights = layer(
#                 query=queries, 
#                 key_value=projected_x, 
#                 key_padding_mask=cross_attn_mask,
#                 need_weights=is_last_layer 
#             )
            
#             if is_last_layer and weights is not None:
#                 if audio_id and epoch is not None and audio_id == "BAC009S0240W0216" and epoch in [1, 5, 10, 20, 50, 100]:
#                     logger.info(f"--- 触发注意力可视化 (Epoch: {epoch}, Audio ID: {audio_id}) ---")
#                     self._visualize_attention(weights, audio_id, epoch, layer_index=i)

#         adapted_output = queries
#         adapted_attention_mask = torch.ones(batch_size, self.num_queries, device=device, dtype=torch.bool)

#         logger.info(f" 长度适配器输出adapted_attention_mask是：形状: {adapted_attention_mask.shape}")
#         logger.info(f" 长度适配器adapted_output输出形状: {adapted_output.shape}")
#         logger.info(f"*********************************退出 长度适配器*************************************\n")
        
#         return adapted_output, adapted_attention_mask

#     def _visualize_attention(self, attn_weights: torch.Tensor, audio_id: str, epoch: int, layer_index: int, head: int = 0):
#         """
#         将注意力权重矩阵绘制成热力图并保存。
#         """

#         import matplotlib.pyplot as plt
#         import numpy as np

#         w_tensor = attn_weights.squeeze(0).detach().cpu()
        

#         if w_tensor.ndim != 3:
#             logger.warning(f"无法绘制注意力图谱：期望得到3维的注意力权重张量 (heads, query, key)，但实际得到 {w_tensor.ndim} 维，形状为 {w_tensor.shape}。跳过绘图。")
#             return
            
#         w = w_tensor.numpy()


#         m_mean = w.mean(axis=0)
        

#         if m_mean.ndim != 2:
#             logger.warning(f"无法绘制注意力图谱：平均后的权重矩阵不是2维的，形状为 {m_mean.shape}。跳过绘图。")
#             return

#         plt.figure(figsize=(20, 8))
#         plt.imshow(m_mean, aspect='auto', cmap='viridis', interpolation='none')
#         plt.colorbar(label='Attention Weight')
#         plt.xlabel('Audio Feature Timesteps (from Encoder)')
#         plt.ylabel('Query Vectors (Adapter Output Timesteps)')
#         plt.title(f'Cross-Attention Map (Mean over Heads)\nAudio: {audio_id}, Epoch: {epoch}, Adapter Layer: {layer_index}')

#         output_dir = 'debug_attention'
#         os.makedirs(output_dir, exist_ok=True)
        
#         save_path = os.path.join(output_dir, f'attn_map_epoch_{epoch}_id_{audio_id}_layer_{layer_index}.png')
#         plt.savefig(save_path)
#         plt.close()
#         logger.info(f"注意力图谱已保存至: {save_path}")


