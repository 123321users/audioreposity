
# src/models/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math # 导入 math，即使不直接使用 PositionalEncoding，其他地方可能用到
from typing import Optional, Tuple, Dict, Any, List
import logging # 导入 logging

# 导入 Hugging Face Transformers 库，用于加载预训练模型
from transformers import AutoModel, AutoConfig

# 获取logger实例，用于日志记录
logger = logging.getLogger(__name__)



# src/models/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math # 导入 math，即使不直接使用 PositionalEncoding，其他地方可能用到
from typing import Optional, Tuple, Dict, Any, List
import logging # 导入 logging

# 导入 Hugging Face Transformers 库，用于加载预训练模型
from transformers import AutoModel, AutoConfig

# 获取logger实例，用于日志记录
logger = logging.getLogger(__name__)



class Wav2Vec2DownsamplingCalculator:
    """
    Wav2Vec2模型的精确下采样计算器
    
    处理不同Wav2Vec2模型变体的下采样机制：
    - wav2vec2-base/large: 7层卷积，总下采样率~320
    - wav2vec2-large-xlsr: 类似结构
    - 其他变体: 根据config自动检测
    """
    
    @staticmethod
    def get_conv_layers_config(model_config):
        """获取卷积层配置"""
        if hasattr(model_config, 'conv_kernel') and hasattr(model_config, 'conv_stride'):
            # 标准Wav2Vec2配置
            return list(zip(model_config.conv_kernel, model_config.conv_stride))
        else:
            # 默认wav2vec2-base配置
            default_config = [
                (10, 5),   # Layer 0: kernel=10, stride=5
                (3, 2),    # Layer 1: kernel=3, stride=2  
                (3, 2),    # Layer 2: kernel=3, stride=2
                (3, 2),    # Layer 3: kernel=3, stride=2
                (3, 2),    # Layer 4: kernel=3, stride=2
                (2, 2),    # Layer 5: kernel=2, stride=2
                (2, 2),    # Layer 6: kernel=2, stride=2
            ]
            logger.warning("使用默认Wav2Vec2卷积配置，可能不准确")
            return default_config
    
    @staticmethod
    def calculate_output_length(input_length: int, conv_layers_config: List[Tuple[int, int]]) -> int:
        """
        根据卷积层配置精确计算输出长度
        
        Args:
            input_length: 输入序列长度
            conv_layers_config: [(kernel_size, stride), ...] 列表
            
        Returns:
            输出序列长度
        """
        current_length = input_length
        
        for kernel_size, stride in conv_layers_config:
            # 卷积输出长度公式: floor((input_length - kernel_size) / stride) + 1
            # 但Wav2Vec2使用了padding，所以公式稍有不同
            # 实际使用: floor((input_length + 2*padding - kernel_size) / stride) + 1
            # Wav2Vec2的padding策略通常是: padding = kernel_size // 2
            
            padding = kernel_size // 2
            current_length = (current_length + 2 * padding - kernel_size) // stride + 1
            
        return max(1, current_length)  # 确保至少有1个输出
    
    @staticmethod
    def calculate_downsample_ratio(input_length: int, output_length: int) -> float:
        """计算实际的下采样比例"""
        if output_length == 0:
            return float('inf')
        return input_length / output_length
    
    @classmethod
    def get_precise_output_length_and_ratio(cls, model_config, input_length: int) -> Tuple[int, float]:
        """
        获取精确的输出长度和下采样比例
        
        Args:
            model_config: Wav2Vec2模型配置
            input_length: 输入音频长度
            
        Returns:
            (output_length, downsample_ratio)
        """
        conv_layers = cls.get_conv_layers_config(model_config)
        output_length = cls.calculate_output_length(input_length, conv_layers)
        downsample_ratio = cls.calculate_downsample_ratio(input_length, output_length)
        
        return output_length, downsample_ratio



class AudioEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        encoder_config = config['model']['encoder']
        self.debug_specific_audio_ids = config.get('debug_specific_audio_ids', [])

        pretrained_model_name = encoder_config.get('pretrained_model_name', 'jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
        logger.info(f"音频编码器: 正在使用预训练模型: {pretrained_model_name}")

        try:
            # 同时加载模型和配置
            self.model_config = AutoConfig.from_pretrained(pretrained_model_name)
            self.pretrained_encoder = AutoModel.from_pretrained(pretrained_model_name)
            self.hidden_size = self.pretrained_encoder.config.hidden_size
            
            # 初始化下采样计算器
            self.downsampling_calculator = Wav2Vec2DownsamplingCalculator()
            
            logger.info(f"音频编码器: 已加载预训练模型 '{pretrained_model_name}'。输出维度: {self.hidden_size}")
            
            # 记录卷积层配置信息
            conv_layers = self.downsampling_calculator.get_conv_layers_config(self.model_config)
            logger.info(f"音频编码器: 检测到卷积层配置: {conv_layers}")
            
            # 计算理论下采样率（使用标准输入长度）
            test_input_length = 16000  # 1秒的16kHz音频
            theoretical_output_length, theoretical_ratio = self.downsampling_calculator.get_precise_output_length_and_ratio(
                self.model_config, test_input_length
            )
            logger.info(f"音频编码器: 理论下采样率约为 {theoretical_ratio:.1f} (1秒音频 -> {theoretical_output_length}特征)")

            # 冻结参数设置
            for param in self.pretrained_encoder.parameters():
                param.requires_grad = False
            
            if not encoder_config.get('freeze_feature_extractor', True):
                if hasattr(self.pretrained_encoder, 'feature_extractor'):
                    logger.info("音频编码器: 正在解冻特征提取层...")
                    for param in self.pretrained_encoder.feature_extractor.parameters():
                        param.requires_grad = True
                else:
                    logger.warning("音频编码器: 模型没有 'feature_extractor' 属性，无法解冻。")

            if not encoder_config.get('freeze_encoder', False):
                if hasattr(self.pretrained_encoder, 'encoder'):
                    logger.info("音频编码器: 正在解冻主编码器层...")
                    for param in self.pretrained_encoder.encoder.parameters():
                        param.requires_grad = True
                else:
                    logger.warning("音频编码器: 模型没有 'encoder' 属性，无法解冻。")
            
        except Exception as e:
            logger.error(f"音频编码器: 加载预训练模型 '{pretrained_model_name}' 失败: {e}", exc_info=True)
            raise

    def _calculate_precise_output_attention_mask(
        self, 
        input_attention_mask: torch.Tensor, 
        encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        精确计算输出注意力掩码
        
        Args:
            input_attention_mask: 输入掩码 [batch_size, input_length]
            encoder_output: 编码器输出 [batch_size, output_length, hidden_size]
            
        Returns:
            output_attention_mask: 输出掩码 [batch_size, output_length]
        """
        batch_size, input_length = input_attention_mask.shape
        output_length = encoder_output.size(1)
        device = input_attention_mask.device
        
        # 计算每个样本的实际输入长度
        input_lengths = input_attention_mask.sum(dim=1)  # [batch_size]
        
        # 为每个样本精确计算输出长度
        output_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            actual_input_length = input_lengths[i].item()
            if actual_input_length > 0:
                # 使用精确的下采样计算器
                predicted_output_length, _ = self.downsampling_calculator.get_precise_output_length_and_ratio(
                    self.model_config, actual_input_length
                )
                # 限制在实际输出长度范围内
                output_lengths[i] = min(predicted_output_length, output_length)
            else:
                output_lengths[i] = 0
        
        # 创建输出掩码
        output_attention_mask = torch.zeros(batch_size, output_length, dtype=torch.bool, device=device)
        for i in range(batch_size):
            if output_lengths[i] > 0:
                output_attention_mask[i, :output_lengths[i]] = True
        
        return output_attention_mask

    def _validate_and_adjust_output_mask(
        self,
        predicted_mask: torch.Tensor,
        encoder_output: torch.Tensor,
        input_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        验证和调整输出掩码，确保其合理性
        
        Args:
            predicted_mask: 预测的输出掩码
            encoder_output: 编码器实际输出
            input_attention_mask: 输入掩码
            
        Returns:
            调整后的输出掩码
        """
        batch_size, output_length = predicted_mask.shape
        device = predicted_mask.device
        
        # 获取输入有效长度
        input_lengths = input_attention_mask.sum(dim=1)
        predicted_output_lengths = predicted_mask.sum(dim=1)
        
        adjusted_mask = predicted_mask.clone()
        
        for i in range(batch_size):
            input_len = input_lengths[i].item()
            predicted_len = predicted_output_lengths[i].item()
            
            # 检查是否存在异常情况
            if input_len == 0:
                # 输入为空，输出也应为空
                adjusted_mask[i, :] = False
            elif predicted_len == 0 and input_len > 0:
                # 输入非空但预测输出为空，给予最小长度
                adjusted_mask[i, 0] = True
                logger.warning(f"样本{i}: 输入长度{input_len}但预测输出为0，调整为最小长度1")
            elif predicted_len > output_length:
                # 预测长度超出实际输出长度，截断
                adjusted_mask[i, :] = True
                logger.warning(f"样本{i}: 预测输出长度{predicted_len}超出实际长度{output_length}，已截断")
        
        return adjusted_mask

    def forward(
        self, 
        audio_features: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        audio_ids: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        音频编码器前向传播
        
        Args:
            audio_features: 音频特征 [batch_size, sequence_length]
            attention_mask: 注意力掩码 [batch_size, sequence_length]  
            audio_ids: 音频ID列表，用于调试
            
        Returns:
            encoder_output: 编码器输出 [batch_size, output_sequence_length, hidden_size]
            output_attention_mask: 输出注意力掩码 [batch_size, output_sequence_length]
        """
        logger.info(f"\n----------------------------进入 编码器-----------------------------")

        batch_size = audio_features.size(0)
        input_length = audio_features.size(1)
        
        # 判断当前批次是否为调试目标
        current_batch_is_debug_target = False
        if audio_ids and self.debug_specific_audio_ids:
            current_batch_is_debug_target = any(str(aid) in self.debug_specific_audio_ids for aid in audio_ids)

        if logger.isEnabledFor(logging.DEBUG) or current_batch_is_debug_target:
            # logger.info(f"--- 音频编码器前向传播开始 (批次大小: {batch_size}, 设备: {audio_features.device}) ---")
            # logger.info(f"  输入音频特征形状 (波形): {audio_features.shape}")
            if attention_mask is not None:
                logger.info(f"  输入attention_mask形状: {attention_mask.shape}")
                input_lengths = attention_mask.sum(dim=1)
                logger.info(f"  各样本输入长度: {input_lengths.tolist()}")

        # 预测输出长度（用于调试和验证）
        if attention_mask is not None and (logger.isEnabledFor(logging.DEBUG) or current_batch_is_debug_target):
            input_lengths = attention_mask.sum(dim=1)
            for i in range(min(batch_size, 3)):  # 只显示前3个样本的预测
                actual_input_len = input_lengths[i].item()
                if actual_input_len > 0:
                    predicted_output_len, downsample_ratio = self.downsampling_calculator.get_precise_output_length_and_ratio(
                        self.model_config, actual_input_len
                    )
                    aid_str = str(audio_ids[i]) if audio_ids and i < len(audio_ids) else 'N/A'
                    logger.debug(f"  样本[{i}] (ID: {aid_str}): 输入长度{actual_input_len} -> 预测输出{predicted_output_len} (比例:{downsample_ratio:.1f})")
        
        try:
            # 保存原始训练状态
            is_training_orig = self.pretrained_encoder.training
            if self.training:
                self.pretrained_encoder.train()
            else:
                self.pretrained_encoder.eval()
            
            # 前向传播
            outputs = self.pretrained_encoder(
                input_values=audio_features,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            encoder_output = outputs.last_hidden_state
            
            # 恢复原始训练状态
            self.pretrained_encoder.train(is_training_orig)
            
            # 精确计算输出注意力掩码
            output_attention_mask = None
            if attention_mask is not None:
                output_attention_mask = self._calculate_precise_output_attention_mask(
                    attention_mask, encoder_output
                )
                
                # 验证和调整掩码
                output_attention_mask = self._validate_and_adjust_output_mask(
                    output_attention_mask, encoder_output, attention_mask
                )
            else:
                # 如果没有输入掩码，创建全True的输出掩码
                batch_size, output_seq_len = encoder_output.size(0), encoder_output.size(1)
                output_attention_mask = torch.ones(
                    batch_size, output_seq_len, dtype=torch.bool, device=encoder_output.device
                )
        
        except Exception as e:
            logger.error(f"音频编码器前向传播失败: {e}", exc_info=True)
            
            # 生成安全的备用输出
            if attention_mask is not None:
                # 根据输入长度估算输出长度
                avg_input_length = attention_mask.sum(dim=1).float().mean().item()
                estimated_output_length = max(1, int(avg_input_length / 320))  # 使用近似下采样率
            else:
                estimated_output_length = max(1, input_length // 320)
            
            dummy_output = torch.zeros(
                batch_size, estimated_output_length, self.hidden_size, 
                device=audio_features.device
            )
            dummy_mask = torch.zeros(
                batch_size, estimated_output_length, dtype=torch.bool, 
                device=audio_features.device
            )
            
            logger.warning(f"因发生错误，音频编码器返回了虚拟的零输出(长度:{estimated_output_length})和掩码。")
            return dummy_output, dummy_mask

        # 详细的输出日志
        if logger.isEnabledFor(logging.DEBUG) or current_batch_is_debug_target:
            logger.debug(f"--- 音频编码器前向传播结束 ---")
            logger.info(f" 编码器输出形状: {encoder_output.shape}")
            logger.info(f" 输出attention_mask形状: {output_attention_mask.shape}")
            
            # 验证预测准确性
            actual_output_lengths = output_attention_mask.sum(dim=1)
            input_lengths = attention_mask.sum(dim=1) if attention_mask is not None else [input_length] * batch_size
            
            for b in range(batch_size):
                aid_str = str(audio_ids[b]) if audio_ids and b < len(audio_ids) else 'N/A'
                if logger.isEnabledFor(logging.DEBUG) or (aid_str in self.debug_specific_audio_ids):
                    input_len = input_lengths[b].item() if hasattr(input_lengths[b], 'item') else input_lengths[b]
                    actual_output_len = actual_output_lengths[b].item()
                    actual_ratio = input_len / actual_output_len if actual_output_len > 0 else float('inf')
                    
                    logger.debug(f"  样本 [{b}] (ID: {aid_str}):")
                    logger.debug(f"    输入长度: {input_len}")
                    logger.debug(f"    实际输出长度: {actual_output_len}")
                    logger.debug(f"    实际下采样比例: {actual_ratio:.2f}")

        logger.info(f"*********************************退出 编码器*************************************\n")
        return encoder_output, output_attention_mask


def test_downsampling_calculator():
    """测试下采样计算器的准确性"""
    print("=== 测试Wav2Vec2下采样计算器 ===")
    
    # 创建测试配置
    from transformers import Wav2Vec2Config
    config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base')
    
    calculator = Wav2Vec2DownsamplingCalculator()
    
    # 测试不同输入长度
    test_lengths = [16000, 32000, 48000, 8000, 24000]  # 不同秒数的16kHz音频
    
    print(f"模型配置: {calculator.get_conv_layers_config(config)}")
    print("\n输入长度 -> 预测输出长度 (下采样比例)")
    print("-" * 50)
    
    for input_len in test_lengths:
        output_len, ratio = calculator.get_precise_output_length_and_ratio(config, input_len)
        seconds = input_len / 16000
        print(f"{input_len:6d} ({seconds:4.1f}s) -> {output_len:4d} ({ratio:6.1f}x)")


if __name__ == "__main__":
    test_downsampling_calculator()



