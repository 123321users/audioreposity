# srcseqv1/trainv2-gemini.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import yaml
import os
import tqdm
import numpy as np
import math
import sys
import logging
import signal
from datetime import datetime
from collections import defaultdict # 导入 defaultdict 以便更好地管理 history
import random
from torch.cuda.amp import autocast, GradScaler
from transformers import T5Tokenizer
# 获取当前时间，用于日志文件名
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"training_{current_time_str}.log"
log_dir = "logs" # 日志文件夹，与 srcseqv1 同级别

# 确保 logs 文件夹存在
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filepath = os.path.join(log_dir, log_filename)

# 配置根 logger
# 清除 root logger 的所有现有 handler，避免重复输出
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 创建文件处理器
file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG) # 文件处理器记录所有 DEBUG 及以上信息
logging.root.addHandler(file_handler) # 将文件处理器添加到 root logger

# 获取当前模块的 logger 实例
logger = logging.getLogger(__name__)
# 初始设置为 INFO，后续会根据 config 调整 root logger 的级别


# 添加 srcseqv1 目录到 sys.path，确保可以导入自定义模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入数据集和模型
from srcseqv1.dataset import AudioRelationDataset, collate_fn
from srcseqv1.models.e2e_ner_recognizer import E2ENamedEntityRecognizer

# 导入 NER 文本工具和评估指标
from srcseqv1.utils.ner_text_utils import (
    init_tokenizer, 
    get_vocab_info_from_tokenizer, 
    decode_ner_predictions, 
    parse_tagged_text_to_entities
)
from srcseqv1.utils.ner_metrics import compute_ner_metrics, evaluate_from_files # 确保导入 evaluate_from_files

import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union, Tuple

# 添加早停类
class EarlyStopping:
    """早停类，用于监控验证指标，在指标不再改善时停止训练"""
    def __init__(self, patience=5, min_delta=0, mode='min', verbose=True):
        """
        初始化早停类
        
        参数:
            patience: 容忍验证指标不改善的epoch数
            min_delta: 最小变化量，小于此值视为没有改善
            mode: 'min'表示指标越小越好，'max'表示指标越大越好
            verbose: 是否打印早停相关信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, epoch, val_score):
        """
        检查是否应该早停
        
        参数:
            epoch: 当前epoch
            val_score: 当前验证指标值
            
        返回:
            early_stop: 是否应该早停
        """
        score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif (self.mode == 'min' and score >= self.best_score + self.min_delta) or \
             (self.mode == 'max' and score <= self.best_score - self.min_delta):
            self.counter += 1
            if self.verbose:
                logger.info(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f'触发早停! 最佳性能出现在epoch {self.best_epoch+1}')
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
        return self.early_stop

def load_config(config_path: str) -> Dict[str, Any]:
    """从 YAML 文件加载配置。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], output_dir: str):
    """将配置保存到输出目录的 YAML 文件中。"""
    config_save_path = os.path.join(output_dir, 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, indent=2)
    logger.info(f"配置已保存到 {config_save_path}")



def plot_history(history: Dict[str, List[float]], output_dir: str):
    """绘制训练历史曲线，并将数据保存到 JSON。"""
    if not history or 'train_loss' not in history or not history['train_loss']:
        logger.warning("没有足够的训练历史数据用于绘图")
        return
        
    epochs = range(1, len(history['train_loss']) + 1) # 从1开始的 epoch

    # 将历史数据保存到 JSON 文件
    history_data = {}
    for key, value in history.items():
        if isinstance(value, list) and value:  # 确保值是非空列表
            history_data[key] = value
    
    history_data_path = os.path.join(output_dir, "training_history.json")
    try:
        with open(history_data_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=4)
        logger.info(f"保存训练历史数据到 {history_data_path}")
    except IOError as e:
        logger.error(f"保存训练历史数据到 JSON 文件失败: {e}")
        
    # 设置 Matplotlib 使用支持中文的字体
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history['train_loss'], label='trainloss', marker='o')
    plt.plot(epochs, history['val_loss'], label='devloss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('train and dev loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    logger.info(f"保存损失曲线到 {loss_plot_path}")

    # 绘制 F1、精确率和召回率曲线
    if all(metric in history for metric in ['f1', 'precision', 'recall']):
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, history['f1'], label='F1 Score', marker='o', color='green')
        plt.plot(epochs, history['precision'], label='Precision', marker='s', color='blue')
        plt.plot(epochs, history['recall'], label='Recall', marker='^', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('dev meters')
        plt.legend()
        plt.grid(True)
        metrics_plot_path = os.path.join(output_dir, "metrics_curve.png")
        plt.savefig(metrics_plot_path, dpi=300)
        plt.close()
        logger.info(f"保存指标曲线到 {metrics_plot_path}")
    
    # 绘制学习率曲线（如果存在）
    if 'learning_rate' in history and history['learning_rate']:
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, history['learning_rate'], label='Learning Rate', marker='o', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('learning rate line')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # 对学习率使用对数尺度
        lr_plot_path = os.path.join(output_dir, "lr_curve.png")
        plt.savefig(lr_plot_path, dpi=300)
        plt.close()
        logger.info(f"保存学习率曲线到 {lr_plot_path}")
        
    # 创建训练进度总结图
    plt.figure(figsize=(15, 10))
    
    # 第一个子图：损失
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='训练损失', marker='o')
    plt.plot(epochs, history['val_loss'], label='验证损失', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('损失曲线')
    plt.legend()
    plt.grid(True)
    
    # 第二个子图：F1、精确率和召回率
    if all(metric in history for metric in ['f1', 'precision', 'recall']):
        plt.subplot(2, 2, 2)
        plt.plot(epochs, history['f1'], label='F1', marker='o', color='green')
        plt.plot(epochs, history['precision'], label='Precision', marker='s', color='blue')
        plt.plot(epochs, history['recall'], label='Recall', marker='^', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('NER指标')
        plt.legend()
        plt.grid(True)
    
    # 第三个子图：学习率（如果存在）
    if 'learning_rate' in history and history['learning_rate']:
        plt.subplot(2, 2, 3)
        plt.plot(epochs, history['learning_rate'], label='Learning Rate', marker='o', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('学习率变化')
        plt.grid(True)
        plt.yscale('log')
    
    # 调整布局并保存
    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, "training_summary.png")
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    logger.info(f"保存训练总结图表到 {summary_plot_path}")




def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    current_epoch: int,
    log_interval: int,
    tokenizer: T5Tokenizer
) -> float:
    model.train()
    total_loss = 0
    progress_bar = tqdm.tqdm(dataloader, desc=f"第 {current_epoch} 轮训练 (过拟合测试)", leave=False)

    for batch_idx, batch_data in enumerate(progress_bar):
        if batch_data is None:
            logger.warning("训练期间跳过一个空批次。")
            continue

        audio_features = batch_data['audio_features'].to(device)
        audio_padding_mask = batch_data['audio_padding_mask'].to(device)
        targets_sequence = batch_data['targets_sequence'].to(device)
        
        optimizer.zero_grad()

        predicted_logits, ground_truth_target = model(
            audio_features=audio_features,
            encoder_attention_mask=audio_padding_mask,
            labels=targets_sequence,
            current_epoch=current_epoch
        )

        vocab_size = predicted_logits.size(-1)
        predicted_logits_flat = predicted_logits.view(-1, vocab_size)
        ground_truth_target_flat = ground_truth_target.view(-1)
        
        loss = criterion(predicted_logits_flat, ground_truth_target_flat)
        
        loss.backward()


        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


        # 检查解码器
        decoder_grad = model.text_decoder.t5_model.lm_head.weight.grad
        if decoder_grad is not None:
            logger.debug(f"  TextDecoder lm_head grad norm: {decoder_grad.norm().item():.6f}")
        else:
            logger.error("  !!! TextDecoder lm_head 没有梯度 !!!")

        # 检查编码器（这部分逻辑不变）
        last_encoder_layer_name_part = f'encoder.layers.{model.audio_encoder.pretrained_encoder.config.num_hidden_layers - 1}'
        last_layer_param_to_check = None
        last_layer_param_name = ""
        for name, param in model.audio_encoder.named_parameters():
            if last_encoder_layer_name_part in name and param.requires_grad:
                last_layer_param_to_check = param
                last_layer_param_name = name
                break

        if last_layer_param_to_check is not None:
            encoder_grad = last_layer_param_to_check.grad
            if encoder_grad is not None:
                logger.debug(f"  AudioEncoder last layer param ('{last_layer_param_name}') grad norm: {encoder_grad.norm().item():.6f}")
            else:
                logger.error(f"  !!! AudioEncoder 最后一层 ('{last_layer_param_name}') 没有梯度 !!!")
        else:
            logger.warning("  无法在 AudioEncoder 中定位到最后一个 Transformer 层的参数进行梯度检查。")




        optimizer.step()
        
       

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        with torch.no_grad():
            predicted_token_ids = torch.argmax(predicted_logits, dim=-1)
            
            predicted_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=False)
            
            ground_truth_decoded = tokenizer.decode(targets_sequence[0], skip_special_tokens=False)

            logger.info(f"--- 过拟合检查 (轮 {current_epoch}) ---")
            logger.info(f"  损失: {loss.item():.6f}")
            logger.info(f"  真实目标 (解码后): {ground_truth_decoded.replace('<pad>', '').strip()}")
            logger.info(f"  预测序列 (解码后): {predicted_text.replace('<pad>', '').strip()}")
        
    avg_loss = total_loss / len(dataloader) if dataloader else 0.0
    logger.info(f"第 {current_epoch} 轮平均训练损失: {avg_loss:.4f}")
    return avg_loss



def validate_epoch(
    model: nn.Module,
    val_dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    tokenizer: T5Tokenizer,
    text_vocab: Dict[str, Any],
    epoch: int,
    output_dir: str,
    config: Dict[str, Any],
    debug_mode: bool = False
) -> Tuple[float, str]:
    logger.info(f"--- 开始第 {epoch+1} 轮验证 ---")
    model.eval()
    total_val_loss = 0.0
    all_predictions_data: List[Dict[str, Any]] = []

    debug_specific_audio_ids = config.get('debug_specific_audio_ids', [])
    val_loop = tqdm.tqdm(val_dataloader, leave=False, desc=f"第 {epoch+1} 轮验证")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loop):
            if batch_data is None:
                logger.warning(f"验证批次 {batch_idx+1} 为空，跳过。")
                continue

            audio_features = batch_data['audio_features'].to(device)
            audio_padding_mask = batch_data['audio_padding_mask'].to(device)
            targets_sequence = batch_data['targets_sequence'].to(device)
            audio_ids = batch_data['audio_ids']
            original_tagged_texts = batch_data['original_tagged_text']
            # 1. 获取真实推理结果 (自回归生成) - 用于JSON文件
            generated_sequences = model(
                audio_features=audio_features,
                encoder_attention_mask=audio_padding_mask,
                audio_ids=audio_ids,
                current_epoch=epoch
            )
            # 2. 获取教师强制下的结果 - 用于计算验证损失和日志对比
            logits, targets_for_loss = model(
                audio_features=audio_features,
                encoder_attention_mask=audio_padding_mask,
                labels=targets_sequence,
                audio_ids=audio_ids,
                current_epoch=epoch
            )

            # 计算验证损失 (这部分逻辑不变)
            # pad_id = text_vocab.get('pad_token_id')
            # active_loss = (targets_for_loss.reshape(-1) != pad_id)
            # if active_loss.any():
            #     loss = loss_fn(
            #         logits.reshape(-1, logits.size(-1))[active_loss],
            #         targets_for_loss.reshape(-1)[active_loss]
            #     )
            # else:
            #     loss = torch.tensor(0.0, device=device)


            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                targets_for_loss.reshape(-1)
            )

            total_val_loss += loss.item()
            val_loop.set_postfix(loss=loss.item())

            # ----- START: 新增的日志输出部分 -----
            # (只对每个批次的第一个样本进行日志记录，以避免刷屏)
            if batch_idx == 0:
                logger.info(f"--- 验证集样本对比 (轮 {epoch+1}) ---")
                
                # 解码真实目标
                ground_truth_decoded = tokenizer.decode(targets_sequence[0], skip_special_tokens=False)
                logger.info(f"  [真实目标]       : {ground_truth_decoded.replace('<pad>', '').strip()}")

                # 解码教师强制下的预测 (开卷考试)
                teacher_forced_ids = torch.argmax(logits, dim=-1)
                teacher_forced_text = tokenizer.decode(teacher_forced_ids[0], skip_special_tokens=False)
                logger.info(f"  [预测-教师强制]   : {teacher_forced_text.replace('<pad>', '').strip()}")

                # 解码自回归生成的结果 (闭卷考试 - 这是写入JSON文件的内容)
                generated_text = tokenizer.decode(generated_sequences[0], skip_special_tokens=True) # 使用skip_special_tokens=True更干净
                logger.info(f"  [预测-自回归生成] : {generated_text.strip()}")
            # ----- END: 新增的日志输出部分 -----



            decoded_predictions_batch = decode_ner_predictions(
                generated_sequences,
                tokenizer,
                debug_specific_audio_ids=debug_specific_audio_ids,
                audio_ids_for_batch=audio_ids
            )

            ground_truth_entities_batch = [parse_tagged_text_to_entities(txt) for txt in original_tagged_texts]

            for i in range(len(audio_ids)):
                all_predictions_data.append({
                    "audio_id": audio_ids[i],
                    "ground_truth_text": original_tagged_texts[i],
                    "predicted_text": decoded_predictions_batch[i].get('predicted_text', ''),
                    "ground_truth_entities": ground_truth_entities_batch[i],
                    "predicted_entities": decoded_predictions_batch[i].get('data_annotation', [])
                })

    avg_val_loss = total_val_loss / len(val_dataloader) if val_dataloader else 0.0
    logger.info(f"第 {epoch+1} 轮验证损失: {avg_val_loss:.4f}")

    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    output_filename = f"val_predictions_epoch_{epoch+1}.json"
    output_filepath = os.path.join(predictions_dir, output_filename)

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_predictions_data, f, ensure_ascii=False, indent=4)
        logger.info(f"验证预测结果已保存至 {output_filepath}")
    except Exception as e:
        logger.error(f"保存验证预测结果失败: {e}", exc_info=True)

    logger.info(f"--- 完成第 {epoch+1} 轮验证 ---")
    return avg_val_loss, output_filepath





import re
def main():
    config_path = 'srcseqv1/config/config.yaml'
    config = load_config(config_path)

    # --- 1. Setup Output Directory and Resume Logic ---
    output_dir = ""
    start_epoch = 0
    checkpoint_to_load = None

    if config['training'].get('resume_training', False):
        run_id = config['training'].get('resume_from_run')
        if not run_id:
            logger.error("'resume_training' is true, but 'resume_from_run' is not specified.")
            return
        
        output_dir = os.path.join(config['training'].get('output_dir', 'outputs'), run_id)
        if not os.path.isdir(output_dir):
            logger.error(f"Cannot resume. The specified run directory does not exist: {output_dir}")
            return
        
        checkpoint_to_load = config['training'].get('checkpoint_path')
        if not checkpoint_to_load or not os.path.exists(checkpoint_to_load):
            logger.error(f"Cannot resume. Checkpoint path '{checkpoint_to_load}' is invalid or does not exist.")
            return

        # Automatically determine start_epoch from the checkpoint filename
        try:
            epoch_str = re.search(r'epoch(\d+)', os.path.basename(checkpoint_to_load)).group(1)
            start_epoch = int(epoch_str)
            logger.info(f"Resuming training from run '{run_id}'")
            logger.info(f"Loading checkpoint: {checkpoint_to_load}")
            logger.info(f"Will start next epoch at: {start_epoch + 1}")
        except (AttributeError, ValueError):
            start_epoch = config['training'].get('start_epoch', 0)
            logger.warning(f"Could not parse epoch from filename. Using 'start_epoch' from config: {start_epoch}. Training will start at epoch {start_epoch + 1}.")
            
    else:
        # Create a new run directory
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(config['training'].get('output_dir', 'outputs'), f"run_{run_id}")
        logger.info(f"开始新的训练: {run_id}")

    # Create all necessary subdirectories
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Save the config file for this run
    save_config(config, output_dir)

    log_level = config['logging']['level'].upper()
    logging.root.setLevel(getattr(logging, log_level, logging.INFO))
    logger.info(f"日志级别已设置为 {log_level}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # --- 2. Tokenizer and Datasets ---
    decoder_config = config['model']['decoder']
    pretrained_decoder_name = decoder_config.get('pretrained_model_name', 'Langboat/mengzi-t5-base')
    tokenizer = init_tokenizer(pretrained_decoder_name)
    text_vocab = get_vocab_info_from_tokenizer(tokenizer)
    
    tokenizer_save_path = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_save_path)
    logger.info(f"Tokenizer 配置已保存至 {tokenizer_save_path}")

    train_dataset = AudioRelationDataset(config, tokenizer, config['data']['train_txt'], 'train')
    val_dataset = AudioRelationDataset(config, tokenizer, config['data']['val_txt'], 'dev')
    logger.info("训练集和验证集：",train_dataset,val_dataset)
    if not train_dataset or len(train_dataset) == 0:
        logger.error("训练数据集加载失败或为空，退出程序。")
        return
    if not val_dataset or len(val_dataset) == 0:
        logger.error("验证数据集加载失败或为空，退出程序。")
        return
    debug_train_subset_size = config['training'].get('debug_train_subset_size', 0)
    debug_val_subset_size = config['training'].get('debug_val_subset_size', 0)


    is_overfit_test = (debug_train_subset_size == 1 and debug_val_subset_size == 1)

    if is_overfit_test:
        logger.warning("--- 正在进入单样本过拟合测试模式 ---")
        
        random.seed(config.get('seed', 42))
        single_sample_data = random.sample(train_dataset.data, 1)
        
        train_dataset.data = single_sample_data
        val_dataset.data = single_sample_data
        
        logger.warning(f"  训练集和验证集都将使用同一个样本 (ID: {single_sample_data[0]['audio_id']})")
        
        val_dataset.audio_dir = os.path.join(val_dataset.audio_root_dir, 'train')
        val_dataset.audio_subdir_name = 'train'
        logger.warning(f"  已强制验证集从 '{val_dataset.audio_dir}' 目录加载音频。")
    elif debug_train_subset_size > 0 or debug_val_subset_size > 0:
        if debug_train_subset_size > 0 and debug_train_subset_size < len(train_dataset.data):
            random.seed(config.get('seed', 42))
            train_dataset.data = random.sample(train_dataset.data, debug_train_subset_size)
            logger.warning(f"已启用 DEBUG 模式，训练集裁剪至 {len(train_dataset)} 个样本。")
        
        if debug_val_subset_size > 0 and debug_val_subset_size < len(val_dataset.data):
            random.seed(config.get('seed', 42) + 1)
            val_dataset.data = random.sample(val_dataset.data, debug_val_subset_size)
            logger.warning(f"已启用 DEBUG 模式，验证集裁剪至 {len(val_dataset)} 个样本。")
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # Handle debug subset logic (your original code was fine here)
    # debug_train_subset_size = config['training'].get('debug_train_subset_size', 0)
    # if debug_train_subset_size > 0:
    #     train_dataset.data = random.sample(train_dataset.data, min(debug_train_subset_size, len(train_dataset.data)))
    #     logger.warning(f"DEBUG MODE: Training set trimmed to {len(train_dataset.data)} samples.")
    
    # debug_val_subset_size = config['training'].get('debug_val_subset_size', 0)
    # if debug_val_subset_size > 0:
    #     val_dataset.data = random.sample(val_dataset.data, min(debug_val_subset_size, len(val_dataset.data)))
    #     logger.warning(f"DEBUG MODE: Validation set trimmed to {len(val_dataset.data)} samples.")

    # logger.info(f"Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'], collate_fn=collate_fn, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'], collate_fn=collate_fn, pin_memory=True)

    # --- 3. Model, Optimizer, and Scheduler Initialization ---
    model = E2ENamedEntityRecognizer(config, tokenizer).to(device)
    logger.info("模型初始化完成。")
    if checkpoint_to_load:
        model.load_state_dict(torch.load(checkpoint_to_load, map_location=device))
        logger.info("成功从检查点加载模型权重")
    
    logger.info(f"模型总参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M, 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=text_vocab.get('pad_token_id'))
    logger.info(f"损失函数: CrossEntropyLoss (忽略 PAD token ID: {text_vocab.get('pad_token_id')})")
    
    optimizer_config = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config['learning_rate'], weight_decay=optimizer_config.get('weight_decay', 0.01))
    
    scheduler = None
    scheduler_config = config['training'].get('scheduler')
    if scheduler_config and scheduler_config['use']:
        scheduler_name = scheduler_config['name'].lower()
        if scheduler_name == 'reducelronplateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 5),
                verbose=True,
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
            logger.info(f"学习率调度器: ReduceLROnPlateau (factor={scheduler_config.get('factor', 0.1)}, "
                        f"patience={scheduler_config.get('patience', 5)}, "
                        f"min_lr={scheduler_config.get('min_lr', 1e-6)})")
        elif scheduler_name == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 5),
                gamma=scheduler_config.get('gamma', 0.5)
            )
            logger.info(f"学习率调度器: StepLR (step_size={scheduler_config.get('step_size', 5)}, "
                        f"gamma={scheduler_config.get('gamma', 0.5)})")
        else:
            logger.warning(f"不支持的学习率调度器: {scheduler_name}")
        
    early_stopping_config = config['training'].get('early_stopping', {})
    early_stopping_enabled = early_stopping_config.get('enabled', False)
    early_stopping_patience = early_stopping_config.get('patience', 10)
    early_stopping_min_delta = early_stopping_config.get('min_delta', 0.0001)
    early_stopping_metric = early_stopping_config.get('metric', 'val_loss')  
    early_stopping_mode = 'min' if early_stopping_metric == 'val_loss' else 'max'
    
    if early_stopping_enabled:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode=early_stopping_mode,
            verbose=True
        )
        logger.info(f"早停功能已启用 (patience={early_stopping_patience}, min_delta={early_stopping_min_delta}, metric={early_stopping_metric})")
    else:
        logger.info("早停功能已禁用")

    # --- 4. Training Loop Setup ---
    best_val_loss = float('inf')
    best_f1 = -1.0 # 用于保存最佳 F1 分数
    history = defaultdict(list) # 使用 defaultdict 更好地管理历史记录
    
    # Pointers to the three checkpoints we want to preserve
    best_loss_checkpoint_path = None
    best_f1_checkpoint_path = None
    latest_checkpoint_path = None
    
    # If resuming, load previous history and find existing best checkpoints
    history_path = os.path.join(output_dir, "training_history.json")
    if config['training'].get('resume_training', False):
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f: 
                history_data = json.load(f)
                history.update(history_data)
                best_val_loss = min(history.get('val_loss', [float('inf')]))
                best_f1 = max(history.get('f1', [-1.0]))
                logger.info(f"加载以前训练历史数据，最低的损失: {best_val_loss:.4f}, 最好的F1: {best_f1:.4f}")

        # Scan for existing best checkpoints to preserve them
        for filename in os.listdir(checkpoints_dir):
            if filename.startswith("best_loss_model"):
                best_loss_checkpoint_path = os.path.join(checkpoints_dir, filename)
            elif filename.startswith("best_f1_model"):
                best_f1_checkpoint_path = os.path.join(checkpoints_dir, filename)
        logger.info(f"Found existing best loss model: {os.path.basename(best_loss_checkpoint_path) if best_loss_checkpoint_path else 'None'}")
        logger.info(f"Found existing best F1 model: {os.path.basename(best_f1_checkpoint_path) if best_f1_checkpoint_path else 'None'}")


    trainepoch_filepath = os.path.join(output_dir, "trainepoch.txt")
    
    # 定义信号处理函数，在程序终止时绘制图表
    def signal_handler(sig, frame):
        logger.info("检测到终止信号，正在保存当前训练状态...")
        if history and history['train_loss']:
            logger.info("绘制训练历史图表...")
            plot_history(history, output_dir)
            logger.info(f"图表已保存到 {output_dir}")
        else:
            logger.warning("没有足够的训练历史数据用于绘图")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # 终止信号


    logger.info("开始训练...")

    
    # Open log file in append mode if resuming, otherwise write mode
    file_mode = 'a' if start_epoch > 0 else 'w'
    with open(trainepoch_filepath, file_mode, encoding='utf-8') as f_trainepoch:
        try:
            # 仅在创建新文件时写入标题
            if file_mode == 'w':
                f_trainepoch.write(f"{'Epoch':<5} | {'Loss':<10} | {'Val Loss':<10} | {'P':<8} | {'R':<8} | {'F1':<8} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Best F1':<8} | {'LR':<10}\n")
            
            total_epochs = config['training']['epochs']
            for epoch in range(start_epoch, total_epochs):
                epoch_num = epoch + 1
                logger.info(f"\n===== Epoch {epoch_num}/{total_epochs} =====")
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"当前学习率: {current_lr:.6f}")

                train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch_num, config['training']['log_every'], tokenizer)
                val_loss, predictions_output_path = validate_epoch(model, val_dataloader, criterion, device, tokenizer, text_vocab, epoch, output_dir, config)
                
                # --- 计算指标 ---
                ner_metrics = {}

                try:
                    ner_metrics = evaluate_from_files(
                            ground_truth_filepath=config['data']['val_txt'], # 验证集真实标签文件
                            predictions_json_filepath=predictions_output_path,
                            text_vocab=text_vocab,
                            debug_specific_audio_ids=config['inference']['debug_specific_audio_ids']
                        )
                except Exception as e:
                    logger.error(f"在 Epoch {epoch_num} 计算 NER 指标时发生错误: {e}", exc_info=True)
                    ner_metrics = {'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_tp': 0, 'overall_fp': 0, 'overall_fn': 0}

                current_f1 = ner_metrics.get('overall_f1', 0.0)
                logger.info(f"Epoch {epoch_num}验证指标 -> Loss: {val_loss:.4f}, Precision: {ner_metrics.get('overall_precision', 0.0):.4f}, Recall: {ner_metrics.get('overall_recall', 0.0):.4f}, F1: {current_f1:.4f}")
                
                # --- 更新 History ---
                history['epoch'].append(epoch + 1)
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['precision'].append(ner_metrics.get('overall_precision', 0.0))
                history['recall'].append(ner_metrics.get('overall_recall', 0.0))
                history['f1'].append(ner_metrics.get('overall_f1', 0.0))
                history['learning_rate'].append(current_lr)  # 记录当前学习率

                # --- 检查点保存、最佳指标更新和清理 ---
                latest_checkpoint_path = os.path.join(checkpoints_dir, f"latest_model_epoch{epoch_num}.pth")
                torch.save(model.state_dict(), latest_checkpoint_path)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_loss_checkpoint_path = os.path.join(checkpoints_dir, f"best_loss_model_epoch{epoch_num}_loss{val_loss:.4f}.pth")
                    torch.save(model.state_dict(), best_loss_checkpoint_path)
                    logger.info(f"New best validation loss! Saved model to {os.path.basename(best_loss_checkpoint_path)}")

                # 在写入日志前更新 best_f1
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_f1_checkpoint_path = os.path.join(checkpoints_dir, f"best_f1_model_epoch{epoch_num}_f1{current_f1:.4f}.pth")
                    torch.save(model.state_dict(), best_f1_checkpoint_path)
                    logger.info(f"New best F1 score! Saved model to {os.path.basename(best_f1_checkpoint_path)}")

                preserved_files = set(filter(None, [best_loss_checkpoint_path, best_f1_checkpoint_path, latest_checkpoint_path]))
                for filename in os.listdir(checkpoints_dir):
                    file_path = os.path.join(checkpoints_dir, filename)
                    if file_path not in preserved_files and filename.endswith(".pth"):
                        try:
                            os.remove(file_path)
                            logger.debug(f"移除旧的检查点: {filename}")
                        except OSError as e:
                            logger.error(f"移除旧的检查点出错： {filename}: {e}")
                
                # --- 将指标写入 trainepoch.txt (使用您要求的格式) ---
                summary_line = (
                    f"{epoch_num:<5} | {train_loss:<10.4f} | {val_loss:<10.4f} | "
                    f"{ner_metrics.get('overall_precision', 0.0):<8.4f} | "
                    f"{ner_metrics.get('overall_recall', 0.0):<8.4f} | "
                    f"{current_f1:<8.4f} | "
                    f"{ner_metrics.get('overall_tp', 0):<6} | "
                    f"{ner_metrics.get('overall_fp', 0):<6} | "
                    f"{ner_metrics.get('overall_fn', 0):<6} | "
                    f"{best_f1:<8.4f} | {current_lr:<10.9f}\n"
                )
                f_trainepoch.write(summary_line)
                f_trainepoch.flush()

    

                # 学习率调度 (在每个 epoch 训练和验证后进行)
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if val_dataloader and not math.isnan(val_loss):
                            scheduler.step(val_loss)
                        else:
                            logger.warning("跳过 ReduceLROnPlateau.step() 调用，因为没有有效的验证数据或验证损失。")
                    else:
                        scheduler.step()
                        logger.debug(f"Epoch {epoch+1}: scheduler.step() called successfully.")



                if early_stopping_enabled:
                    # 根据配置的指标选择用于早停的值
                    early_stop_value = val_loss if early_stopping_metric == 'val_loss' else ner_metrics.get('overall_f1', 0.0)
                    
                    if early_stopper(epoch, early_stop_value):
                        logger.info(f"触发早停! 在 {early_stopper.best_epoch+1} 轮达到最佳性能，当前为 {epoch+1} 轮")
                        logger.info(f"最佳{early_stopping_metric}值: {early_stopper.best_score:.6f}")
                        # 在早停触发时保存训练历史图表
                        if history and history['train_loss']:
                            logger.info("由于早停触发，绘制训练历史图表...")
                            plot_history(history, output_dir)
                        break
        except KeyboardInterrupt:
            logger.info("检测到键盘中断，正在保存当前训练状态...")
            # 不退出程序，允许执行后续清理操作
    logger.info("训练完成，绘制历史数据...")
    if history['train_loss']:
        history_path = os.path.join(output_dir, "training_history.json")
        plot_history(history, output_dir)
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(dict(history), f, ensure_ascii=False, indent=4)
        logger.info(f"Final training history saved to {history_path}")

    logger.info("Training complete.")




if __name__ == '__main__':
    main()







# def main():
#     config_path = 'srcseqv1/config/config.yaml'
#     config = load_config(config_path)

#     # 设置输出目录
#     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = f"outputs/run_{run_id}"
#     checkpoints_dir = os.path.join(output_dir, "checkpoints")
#     predictions_dir = os.path.join(output_dir, "predictions")
#     os.makedirs(checkpoints_dir, exist_ok=True)
#     os.makedirs(predictions_dir, exist_ok=True)

#     config['run_id'] = run_id
#     save_config(config, output_dir)

#     log_level = config['logging']['level'].upper()
#     logging.root.setLevel(getattr(logging, log_level, logging.INFO))
#     logger.info(f"日志级别已设置为 {log_level}")


#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"使用设备: {device}")


#     decoder_config = config['model']['decoder']
#     pretrained_decoder_name = decoder_config.get('pretrained_model_name', 'Langboat/mengzi-t5-base')
    
#     tokenizer = init_tokenizer(pretrained_decoder_name)

#     text_vocab = get_vocab_info_from_tokenizer(tokenizer)
    
#     tokenizer_save_path = os.path.join(output_dir, "tokenizer")
#     tokenizer.save_pretrained(tokenizer_save_path)
#     logger.info(f"Tokenizer 配置已保存至 {tokenizer_save_path}")

#     train_dataset = AudioRelationDataset(config, tokenizer, config['data']['train_txt'], 'train')
#     val_dataset = AudioRelationDataset(config, tokenizer, config['data']['val_txt'], 'dev')
#     logger.info("训练集和验证集：",train_dataset,val_dataset)
#     if not train_dataset or len(train_dataset) == 0:
#         logger.error("训练数据集加载失败或为空，退出程序。")
#         return
#     if not val_dataset or len(val_dataset) == 0:
#         logger.error("验证数据集加载失败或为空，退出程序。")
#         return


#     debug_train_subset_size = config['training'].get('debug_train_subset_size', 0)
#     debug_val_subset_size = config['training'].get('debug_val_subset_size', 0)


#     is_overfit_test = (debug_train_subset_size == 1 and debug_val_subset_size == 1)

#     if is_overfit_test:
#         logger.warning("--- 正在进入单样本过拟合测试模式 ---")
        
#         random.seed(config.get('seed', 42))
#         single_sample_data = random.sample(train_dataset.data, 1)
        
#         train_dataset.data = single_sample_data
#         val_dataset.data = single_sample_data
        
#         logger.warning(f"  训练集和验证集都将使用同一个样本 (ID: {single_sample_data[0]['audio_id']})")
        
#         val_dataset.audio_dir = os.path.join(val_dataset.audio_root_dir, 'train')
#         val_dataset.audio_subdir_name = 'train'
#         logger.warning(f"  已强制验证集从 '{val_dataset.audio_dir}' 目录加载音频。")
#     elif debug_train_subset_size > 0 or debug_val_subset_size > 0:
#         if debug_train_subset_size > 0 and debug_train_subset_size < len(train_dataset.data):
#             random.seed(config.get('seed', 42))
#             train_dataset.data = random.sample(train_dataset.data, debug_train_subset_size)
#             logger.warning(f"已启用 DEBUG 模式，训练集裁剪至 {len(train_dataset)} 个样本。")
        
#         if debug_val_subset_size > 0 and debug_val_subset_size < len(val_dataset.data):
#             random.seed(config.get('seed', 42) + 1)
#             val_dataset.data = random.sample(val_dataset.data, debug_val_subset_size)
#             logger.warning(f"已启用 DEBUG 模式，验证集裁剪至 {len(val_dataset)} 个样本。")

#     # # 对训练集进行裁剪
#     # if debug_train_subset_size > 0 and debug_train_subset_size < len(train_dataset.data): # 确保裁剪大小小于实际数据量
#     #     random.seed(config.get('seed', 42)) # 确保随机性可复现
#     #     train_dataset.data = random.sample(train_dataset.data, debug_train_subset_size)
#     #     logger.warning(f"已启用 DEBUG 模式，训练集裁剪至 {len(train_dataset)} 个样本。")
#     # elif debug_train_subset_size > 0 and debug_train_subset_size >= len(train_dataset.data) and len(train_dataset.data) > 0:
#     #     logger.warning(f"训练集调试子集大小 ({debug_train_subset_size}) 大于或等于原始训练集大小 ({len(train_dataset.data)})。将使用全部训练集。")


#     # # 对验证集进行裁剪
#     # if debug_val_subset_size > 0 and debug_val_subset_size < len(val_dataset.data): # 确保裁剪大小小于实际数据量
#     #     random.seed(config.get('seed', 42) + 1) # 使用不同的种子或固定种子+偏移，确保与训练集不同的样本
#     #     val_dataset.data = random.sample(val_dataset.data, debug_val_subset_size)
#     #     logger.warning(f"已启用 DEBUG 模式，验证集裁剪至 {len(val_dataset)} 个样本。")
#     # elif debug_val_subset_size > 0 and debug_val_subset_size >= len(val_dataset.data) and len(val_dataset.data) > 0:
#     #     logger.warning(f"验证集调试子集大小 ({debug_val_subset_size}) 大于或等于原始验证集大小 ({len(val_dataset.data)})。将使用全部验证集。")

#     logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")


#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=config['training']['batch_size'],
#         shuffle=True,
#         num_workers=config['data']['num_workers'],
#         collate_fn=collate_fn,
#         pin_memory=True
#     )
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=config['training']['batch_size'],
#         shuffle=False,
#         num_workers=config['data']['num_workers'],
#         collate_fn=collate_fn,
#         pin_memory=True
#     )

#     model = E2ENamedEntityRecognizer(config, tokenizer).to(device)
#     logger.info("模型初始化完成。")


#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     logger.info(f"--- 模型参数统计 ---")
#     logger.info(f"  总参数量: {total_params / 1e6:.2f} M")
#     logger.info(f"  可训练参数量: {trainable_params / 1e6:.2f} M")

#     logger.debug("--- 可训练参数名称 ---")
#     # for name, param in model.named_parameters():
#     #     if param.requires_grad:
#     #         logger.debug(name)

#     criterion = nn.CrossEntropyLoss(ignore_index=text_vocab.get('pad_token_id'))
#     logger.info(f"损失函数: CrossEntropyLoss (忽略 PAD token ID: {text_vocab.get('pad_token_id')})")

#     # 优化器
#     optimizer_config = config['training']
#     optimizer_name = optimizer_config['optimizer'].lower()
#     base_learning_rate = optimizer_config['learning_rate']
#     weight_decay = optimizer_config.get('weight_decay', 0.01)


#     # decoder_lr = base_learning_rate / 10.0 # 例如 5e-6
#     decoder_lr_factor = 0.1 # Decoder的学习率是基础学习率的1/10
#     decoder_lr = base_learning_rate * decoder_lr_factor 
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in model.named_parameters() if "text_decoder" not in n],
#             "lr": base_learning_rate,
#             "name": "AudioEncoder_and_LengthAdapter" # 用于日志
#         },
#         {
#             "params": [p for n, p in model.named_parameters() if "text_decoder" in n],
#             "lr": decoder_lr,
#             "name": "TextDecoder" # 用于日志
#         },
#     ]

#     if optimizer_name == 'adamw':
#         optimizer = torch.optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay)
#     elif optimizer_name == 'adam':
#         optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=base_learning_rate, weight_decay=weight_decay)



#     else:
#         logger.error(f"不支持的优化器: {optimizer_name}")
#         return
#     logger.info("已启用差分学习率:")
#     # for param_group in optimizer.param_groups:
#     #     logger.info(f"  - 参数组: {param_group['name']}, 学习率: {param_group['lr']},权重衰减: {weight_decay}")

    

#     scheduler = None
#     scheduler_config = config['training'].get('scheduler')
#     if scheduler_config and scheduler_config['use']:
#         scheduler_name = scheduler_config['name'].lower()
#         if scheduler_name == 'reducelronplateau':
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer,
#                 mode='min',
#                 factor=scheduler_config.get('factor', 0.1),
#                 patience=scheduler_config.get('patience', 5),
#                 verbose=True,
#                 min_lr=scheduler_config.get('min_lr', 1e-6)
#             )
#             logger.info(f"学习率调度器: ReduceLROnPlateau (factor={scheduler_config.get('factor', 0.1)}, "
#                         f"patience={scheduler_config.get('patience', 5)}, "
#                         f"min_lr={scheduler_config.get('min_lr', 1e-6)})")
#         elif scheduler_name == 'steplr':
#             scheduler = torch.optim.lr_scheduler.StepLR(
#                 optimizer,
#                 step_size=scheduler_config.get('step_size', 5),
#                 gamma=scheduler_config.get('gamma', 0.5)
#             )
#             logger.info(f"学习率调度器: StepLR (step_size={scheduler_config.get('step_size', 5)}, "
#                         f"gamma={scheduler_config.get('gamma', 0.5)})")
#         else:
#             logger.warning(f"不支持的学习率调度器: {scheduler_name}")


#     early_stopping_config = config['training'].get('early_stopping', {})
#     early_stopping_enabled = early_stopping_config.get('enabled', False)
#     early_stopping_patience = early_stopping_config.get('patience', 10)
#     early_stopping_min_delta = early_stopping_config.get('min_delta', 0.0001)
#     early_stopping_metric = early_stopping_config.get('metric', 'val_loss')  
#     early_stopping_mode = 'min' if early_stopping_metric == 'val_loss' else 'max'
    
#     if early_stopping_enabled:
#         early_stopper = EarlyStopping(
#             patience=early_stopping_patience,
#             min_delta=early_stopping_min_delta,
#             mode=early_stopping_mode,
#             verbose=True
#         )
#         logger.info(f"早停功能已启用 (patience={early_stopping_patience}, min_delta={early_stopping_min_delta}, metric={early_stopping_metric})")
#     else:
#         logger.info("早停功能已禁用")

#     # 训练循环
#     best_val_loss = float('inf')
#     best_f1 = -1.0
#     history = defaultdict(list)

#     # --- 新增：用于跟踪需要保留的检查点路径 ---
#     best_loss_checkpoint_path = None
#     best_f1_checkpoint_path = None
#     latest_checkpoint_path = None
    
#     # 定义 trainepoch.txt 的路径
#     trainepoch_filepath = os.path.join(output_dir, "trainepoch.txt")
    
#     # 定义信号处理函数，在程序终止时绘制图表
#     def signal_handler(sig, frame):
#         logger.info("检测到终止信号，正在保存当前训练状态...")
#         if history and history['train_loss']:
#             logger.info("绘制训练历史图表...")
#             plot_history(history, output_dir)
#             logger.info(f"图表已保存到 {output_dir}")
#         else:
#             logger.warning("没有足够的训练历史数据用于绘图")
#         sys.exit(0)
    
#     # 注册信号处理器
#     signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
#     signal.signal(signal.SIGTERM, signal_handler) # 终止信号

#     logger.info("开始训练...")
#     # 打开 trainepoch.txt 文件，用于追加写入
#     with open(trainepoch_filepath, 'w', encoding='utf-8') as f_trainepoch:
#         f_trainepoch.write("Epoch Summary:\n")
#         f_trainepoch.write(f"{'Epoch':<5} | {'Loss':<10} | {'Val Loss':<10} | {'P':<8} | {'R':<8} | {'F1':<8} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Best F1':<8} | {'LR':<10}\n")     
#         try:
#             for epoch in range(config['training']['epochs']):
#                 logger.info(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
#                 current_lr = optimizer.param_groups[0]['lr']
#                 logger.info(f"当前学习率: {current_lr:.6f}")

#                 train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch + 1, config['training']['log_every'], tokenizer )
#                 val_loss, predictions_output_path = validate_epoch(
#                     model, val_dataloader, criterion, device,
#                     tokenizer, text_vocab,
#                     epoch, output_dir, config,
#                     debug_mode=(logging.root.level == logging.DEBUG)
#                 )
                
#                 ner_metrics = {}
#                 try:
#                     ner_metrics = evaluate_from_files(
#                         ground_truth_filepath=config['data']['val_txt'],
#                         predictions_json_filepath=predictions_output_path,
#                         text_vocab=text_vocab,
#                         debug_specific_audio_ids=config['inference']['debug_specific_audio_ids']
#                     )
#                     current_f1 = ner_metrics.get('overall_f1', 0.0)
#                     logger.info(f"Epoch {epoch+1} NER 指标: Precision: {ner_metrics['overall_precision']:.4f}, Recall: {ner_metrics['overall_recall']:.4f}, F1: {current_f1:.4f}")

#                 except Exception as e:
#                     logger.error(f"在 Epoch {epoch+1} 计算 NER 指标时发生错误: {e}", exc_info=True)
#                     ner_metrics = {'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_tp': 0, 'overall_fp': 0, 'overall_fn': 0}
#                     current_f1 = 0.0

#                 # 更新最佳 F1，但不在这里保存模型，统一在后面处理
#                 if current_f1 > best_f1:
#                     best_f1 = current_f1

#                 # 将指标写入 trainepoch.txt 和 history
#                 epoch_summary_line = (
#                     f"{epoch+1:<5} | {train_loss:<10.4f} | {val_loss:<10.4f} | "
#                     f"{ner_metrics.get('overall_precision', 0.0):<8.4f} | "
#                     f"{ner_metrics.get('overall_recall', 0.0):<8.4f} | "
#                     f"{current_f1:<8.4f} | "
#                     f"{ner_metrics.get('overall_tp', 0):<6} | "
#                     f"{ner_metrics.get('overall_fp', 0):<6} | "
#                     f"{ner_metrics.get('overall_fn', 0):<6} | "
#                     f"{best_f1:<8.4f} | {current_lr:<10.9f}\n"
#                 )
#                 f_trainepoch.write(epoch_summary_line)
#                 f_trainepoch.flush()
#                 logger.info(f"已将 epoch {epoch+1} 统计信息写入 {trainepoch_filepath}")
                
#                 history['epoch'].append(epoch + 1)
#                 history['train_loss'].append(train_loss)
#                 history['val_loss'].append(val_loss)
#                 history['precision'].append(ner_metrics.get('overall_precision', 0.0))
#                 history['recall'].append(ner_metrics.get('overall_recall', 0.0))
#                 history['f1'].append(current_f1)
#                 history['learning_rate'].append(current_lr)

#                 # --- START: 全新的检查点保存和清理逻辑 ---
#                 checkpoints_dir = os.path.join(output_dir, "checkpoints")

#                 # 1. 检查并保存“最佳损失”模型
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     checkpoint_path = os.path.join(checkpoints_dir, f"best_loss_model_epoch{epoch+1}_loss{val_loss:.4f}.pth")
#                     torch.save(model.state_dict(), checkpoint_path)
#                     logger.info(f"发现新的最低验证损失，保存检查点到 {checkpoint_path}")
#                     best_loss_checkpoint_path = checkpoint_path

#                 # 2. 检查并保存“最佳F1”模型
#                 if current_f1 == best_f1 and current_f1 > 0: # 只有当F1是历史最佳时才保存
#                     checkpoint_path = os.path.join(checkpoints_dir, f"best_f1_model_epoch{epoch+1}_f1{current_f1:.4f}.pth")
#                     torch.save(model.state_dict(), checkpoint_path)
#                     logger.info(f"发现新的最高F1分数，保存检查点到 {checkpoint_path}")
#                     best_f1_checkpoint_path = checkpoint_path

#                 # 3. 保存“最新”模型
#                 latest_checkpoint_path = os.path.join(checkpoints_dir, f"latest_model_epoch{epoch+1}.pth")
#                 torch.save(model.state_dict(), latest_checkpoint_path)
#                 logger.info(f"保存最新轮次检查点到 {latest_checkpoint_path}")

#                 # 4. 清理旧的检查点
#                 preserved_files = set(filter(None, [best_loss_checkpoint_path, best_f1_checkpoint_path, latest_checkpoint_path]))
                
#                 for filename in os.listdir(checkpoints_dir):
#                     file_path = os.path.join(checkpoints_dir, filename)
#                     if file_path not in preserved_files and filename.endswith(".pth"):
#                         try:
#                             os.remove(file_path)
#                             logger.info(f"删除旧检查点: {filename}")
#                         except OSError as e:
#                             logger.error(f"删除检查点 {filename} 失败: {e}")
#                 # --- END: 全新的检查点保存和清理逻辑 ---

#                 # 学习率调度
#                 if scheduler is not None:
#                      if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                           if val_dataloader and not math.isnan(val_loss):
#                                scheduler.step(val_loss)
#                           else:
#                                logger.warning("跳过 ReduceLROnPlateau.step() 调用，因为没有有效的验证数据或验证损失。")
#                      else:
#                           scheduler.step()
#                           logger.debug(f"Epoch {epoch+1}: scheduler.step() called successfully.")
                
#                 # 应用早停
#                 if early_stopping_enabled:
#                     early_stop_value = val_loss if early_stopping_metric == 'val_loss' else current_f1
                    
#                     if early_stopper(epoch, early_stop_value):
#                         logger.info(f"触发早停! 在 {early_stopper.best_epoch+1} 轮达到最佳性能，当前为 {epoch+1} 轮")
#                         logger.info(f"最佳{early_stopping_metric}值: {early_stopper.best_score:.6f}")
#                         if history and history['train_loss']:
#                             logger.info("由于早停触发，绘制训练历史图表...")
#                             plot_history(history, output_dir)
#                         break
#         except KeyboardInterrupt:
#             logger.info("检测到键盘中断，正在保存当前训练状态...")
#             if history and history['train_loss']:
#                 logger.info("绘制训练历史图表...")
#                 plot_history(history, output_dir)
#                 logger.info(f"图表已保存到 {output_dir}")
#             else:
#                 logger.warning("没有足够的训练历史数据用于绘图")

#     # 训练循环结束后，绘制训练历史
#     if history and history['train_loss']:
#         plot_history(history, output_dir)

#     logger.info("训练完成。")


















