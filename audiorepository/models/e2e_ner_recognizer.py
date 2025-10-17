# srcseqv1/models/e2e_ner_recognizer.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

from transformers import T5Tokenizer

from srcseqv1.models.encoder import AudioEncoder
from srcseqv1.models.length_adapter import LengthAdapter
from srcseqv1.models.text_decoder import TextDecoder

# 获取日志实例
logger = logging.getLogger(__name__)



class E2ENamedEntityRecognizer(nn.Module):
    def __init__(self, config: Dict[str, Any], tokenizer: T5Tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.debug_specific_audio_ids = config.get('debug_specific_audio_ids', [])

        self.audio_encoder = AudioEncoder(config)
        logger.info("音频编码器初始化完成。")

        # 检查是否需要解冻并微调编码器
        if not config['model']['encoder'].get('freeze_encoder', False):
            logger.info("确保音频编码器的主干部分是可训练的...")
            # 遍历音频编码器的所有参数
            for name, param in self.audio_encoder.named_parameters():
                # 除了特征提取器部分，其他参数都设置为需要计算梯度（即可训练）
                if 'feature_extractor' not in name:
                    param.requires_grad = True

        # 获取编码器和解码器的隐藏层维度
        encoder_hidden_size = self.audio_encoder.hidden_size
        decoder_hidden_size = config['model']['decoder']['hidden_size']
        length_adapter_config = config['model']['length_adapter']

        # 实例化长度适配器
        self.length_adapter = LengthAdapter(
            input_dim=encoder_hidden_size,
            output_dim=decoder_hidden_size,
            adapter_config=length_adapter_config
        )
        logger.info("长度适配器初始化完成。")

        # 检查分词器是否包含所有必需的NER特殊词元
        ner_tokens = ['<PER_START>', '<PER_END>', '<LOC_START>', '<LOC_END>', '<ORG_START>', '<ORG_END>']
        vocab = tokenizer.get_vocab()
        missing_tokens = [t for t in ner_tokens if t not in vocab]
        if missing_tokens:
            # 如果有缺失，记录致命错误并中断
            logger.error(f"致命错误：分词器词汇表中缺少必要的NER特殊词元：{missing_tokens}。请在开始训练前将它们添加进去。")

        # 实例化文本解码器
        self.text_decoder = TextDecoder(config, tokenizer)

        # 调整T5模型的词嵌入层大小，以匹配可能已添加了新词元的分词器
        self.text_decoder.t5_model.resize_token_embeddings(len(self.tokenizer))
        logger.info(f"已将T5模型的词嵌入层调整为与分词器大小一致：{len(self.tokenizer)}")

        logger.info("文本解码器初始化完成。")

    def forward(
        self,
        audio_features: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        audio_ids: Optional[List[Any]] = None,
        current_epoch: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        current_batch_is_debug_target = False
        # 检查当前批次中是否有需要特别调试的音频ID
        if audio_ids and self.debug_specific_audio_ids:
            current_batch_is_debug_target = any(str(aid) in self.debug_specific_audio_ids for aid in audio_ids)

        # 根据是训练还是评估模式，打印不同的前向传播开始信息
        if self.training and logger.isEnabledFor(logging.INFO):
            logger.info(f"--- 端到端前向传播开始 (训练, 批量: {audio_features.size(0)}) ---")
        elif not self.training and logger.isEnabledFor(logging.INFO):
            logger.info(f"--- 端到端前向传播开始 (评估, 批量: {audio_features.size(0)}) ---")

        # 获取单个音频ID用于调试
        single_audio_id = audio_ids[0] if audio_ids else None

        # 步骤1：通过音频编码器
        encoder_output, encoder_mask_updated = self.audio_encoder(audio_features, encoder_attention_mask, audio_ids)
        if logger.isEnabledFor(logging.DEBUG) or current_batch_is_debug_target:
            logger.debug(f"经过编码器后: 输出形状={encoder_output.shape}, 掩码形状={encoder_mask_updated.shape if encoder_mask_updated is not None else '无'}")
        # print("从编码器中的输出：",encoder_output, encoder_mask_updated)
        # 步骤2：通过长度适配器
        adapted_output, adapted_attention_mask = self.length_adapter(
            encoder_output, 
            encoder_mask_updated#, 
            # audio_id=single_audio_id,
            # epoch=current_epoch
        )

        if logger.isEnabledFor(logging.DEBUG) or current_batch_is_debug_target:
            logger.debug(f"经过适配器后: 输出形状={adapted_output.shape}, 掩码形状={adapted_attention_mask.shape if adapted_attention_mask is not None else '无'}")

        # 将适配器输出的布尔掩码转换为Hugging Face模型兼容的整型掩码
        adapted_attention_mask_hf = None
        if adapted_attention_mask is not None:
            adapted_attention_mask_hf = adapted_attention_mask.to(dtype=torch.long)
            if logger.isEnabledFor(logging.DEBUG) or current_batch_is_debug_target:
                logger.debug(f"为Hugging Face转换适配器掩码: 数据类型={adapted_attention_mask_hf.dtype}, "f"有效比例={(adapted_attention_mask_hf.float().mean().item()):.3f}")

        # 在训练时，于特定轮次打印适配器输出的统计信息
        if self.training and current_epoch in [1, 10, 20, 50]:
            logger.info(f"轮次 {current_epoch} - 适配器输出 均值: {adapted_output.mean().item()}, "f"标准差: {adapted_output.std().item()}")

        # 步骤3：通过文本解码器进行预测
        predictions = self.text_decoder(
            encoder_output=adapted_output,
            encoder_attention_mask=adapted_attention_mask_hf,
            labels=labels,
            audio_ids=audio_ids,
            current_epoch=current_epoch
        )

        # 在调试模式下，打印预测结果和真实目标的形状
        if self.training and (logger.isEnabledFor(logging.DEBUG) or current_batch_is_debug_target):
            if isinstance(predictions, tuple) and len(predictions) == 2:
                logger.debug(f"预测logits形状: {predictions[0].shape}")
                logger.debug(f"真实目标形状: {predictions[1].shape}")

        return predictions

