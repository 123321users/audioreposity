
# srcseqv1/models/text_decoder.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
import random
logger = logging.getLogger(__name__)


class TextDecoder(nn.Module):
    def __init__(self, config: Dict[str, Any], tokenizer: T5Tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.debug_specific_audio_ids = config.get('debug_specific_audio_ids', [])

        decoder_config_params = config['model']['decoder']
        pretrained_decoder_name = decoder_config_params.get('pretrained_model_name', 'Langboat/mengzi-t5-base')

        self.t5_model = T5ForConditionalGeneration.from_pretrained(pretrained_decoder_name)

        self.sos_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        if any(id is None for id in [self.eos_token_id, self.pad_token_id]):
            raise ValueError("文本解码器从 tokenizer 中缺少必要的特殊 token ID (EOS, PAD)。")

        self.max_output_length = decoder_config_params.get('max_output_length', 256)
        self.generation_params = decoder_config_params.get('generation_params', {})
        self.beam_size = int(self.generation_params.get('beam_size', 1))#解码策略从“贪心搜索”升级为了“集束搜索”，宽度为 5。这意味着在每一步，模型都会保留 5 条最有可能的路径进行探索。
        self.temperature = float(self.generation_params.get('temperature', 0.0))

        self.no_repeat_ngram_size = int(self.generation_params.get('no_repeat_ngram_size', 0))#可以防止模型陷入简单的重复循环（例如生成“权权权权权”）。

        self.repetition_penalty = float(self.generation_params.get('repetition_penalty', 1.0))
        self.max_new_tokens = self.generation_params.get('max_new_tokens', 100)
        self.ban_extra_ids = bool(self.generation_params.get('ban_extra_ids', True))

    def _build_bad_words_ids(self) -> Optional[List[List[int]]]:
        if not self.ban_extra_ids:
            return None

        bad_tokens = [f"<extra_id_{i}>" for i in range(100)]
        bad_ids = []
        for tok in bad_tokens:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != self.tokenizer.unk_token_id and tid >= 0:
                bad_ids.append([tid])

        return bad_ids if bad_ids else None

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        audio_ids: Optional[List[Any]] = None,
        current_epoch: Optional[int] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        encoder_outputs_for_t5 = BaseModelOutput(last_hidden_state=encoder_output)

        logger.info(f"\n----------------------------进入 解码器-----------------------------")
        

        

        current_batch_is_debug_target = False
        if audio_ids and self.debug_specific_audio_ids:
            current_batch_is_debug_target = any(str(aid) in self.debug_specific_audio_ids for aid in audio_ids)

        if labels is not None:
            labels_for_loss = labels.clone()
            labels_for_loss[labels_for_loss == self.pad_token_id] = -100 #使用 -100 作为忽略索引: 它将labels中所有pad_token_id的位置都替换为-100。

            if logger.isEnabledFor(logging.DEBUG) or current_batch_is_debug_target:
                mode = "训练" if self.training else "验证损失计算"
                # logger.debug(f"  文本解码器 ({mode}): 解码器输入encoder_output数据: {encoder_output},形状为：{encoder_output.shape}")
                # logger.debug(f" 文本解码器 ({mode})： encoder_attention_mask数据: {encoder_attention_mask}，形状为：{encoder_attention_mask.shape}")
                # logger.debug(f" 文本解码器 ({mode}): labels数据: {labels}，标签形状: {labels}")


            # 直接将处理后的 labels 传给 t5_model: T5模型（以及BERT、GPT等所有Hugging Face模型）的内部实现是：如果labels参数被提供了，模型会自动地将其向右平移一位来创建decoder_input_ids，并使用原始的labels（内部处理了-100）来计算交叉熵损失。
            outputs = self.t5_model(
                encoder_outputs=encoder_outputs_for_t5,
                attention_mask=encoder_attention_mask,
                labels=labels_for_loss,
                return_dict=True
            )
            logits = outputs.logits
            # logger.info(f"  logits修改前数据: {logits},形状为：{logits.shape}")
            # logger.info(f"  ground_truth_target (原始labels): {labels},形状为：{labels.shape}")
            # ground_truth_target = labels[:, 1:].contiguous()
            # aligned_logits = logits[:, :-1, :].contiguous()


            # logger.info(f"  ground_truth_target修改后数据: {ground_truth_target},形状为：{ground_truth_target.shape}")

            # logger.info(f"  logits修改后数据: {aligned_logits},形状为：{aligned_logits.shape}")

            if self.training and (current_batch_is_debug_target or (random.random() < 0.01)):
                sample_idx = 0
                pred_ids = torch.argmax(logits, dim=-1)
                predicted_text = self.tokenizer.decode(pred_ids[sample_idx], skip_special_tokens=False)
                gt_text = self.tokenizer.decode(labels[sample_idx], skip_special_tokens=False)
                audio_id_str = audio_ids[sample_idx] if audio_ids else "N/A"
                # logger.debug(f"--- 损失对齐检查 (ID: {audio_id_str}) ---")
                # logger.debug(f"  预测序列: {predicted_text}")
                # logger.debug(f"  真实目标: {gt_text}")

            logger.info(f" 解码器返回的数据: logits, labels")

            logger.info(f"*********************************退出 解码器*************************************\n")

            return  logits, labels

        if logger.isEnabledFor(logging.DEBUG) or current_batch_is_debug_target:
            logger.debug(f"  文本解码器 (推理): 解码器输入encoder_output数据: {encoder_output},形状为：{encoder_output.shape}")


        do_sample = (self.temperature is not None) and (self.temperature > 0.0)

        bad_words_ids = self._build_bad_words_ids()

        generation_kwargs: Dict[str, Any] = {
            "encoder_outputs": encoder_outputs_for_t5,
            "attention_mask": encoder_attention_mask,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "decoder_start_token_id": self.sos_token_id,
        }

        if self.max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = int(self.max_new_tokens)
        else:
            generation_kwargs["max_length"] = int(self.max_output_length)

        if self.no_repeat_ngram_size and self.no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = int(self.no_repeat_ngram_size)
        if self.repetition_penalty and abs(self.repetition_penalty - 1.0) > 1e-6:
            generation_kwargs["repetition_penalty"] = float(self.repetition_penalty)
        if bad_words_ids:
            generation_kwargs["bad_words_ids"] = bad_words_ids
        # 束搜索
        if self.beam_size > 1:
            generation_kwargs.update(dict(
                num_beams=int(self.beam_size),
                early_stopping=True,
                do_sample=False
            ))
        else:
            # （贪心/采样
            generation_kwargs.update(dict(
                num_beams=1,
                do_sample=do_sample
            ))
            if do_sample:
                generation_kwargs["temperature"] = float(self.temperature)

        generated_sequences = self.t5_model.generate(** generation_kwargs)

        if logger.isEnabledFor(logging.DEBUG) or current_batch_is_debug_target:
            logger.debug(f"  文本解码器 (推理): 生成的序列形状: {generated_sequences.shape}")

        logger.info(f"*********************************退出 解码器*************************************\n")

        return generated_sequences








            