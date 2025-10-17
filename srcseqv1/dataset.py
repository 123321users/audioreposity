# srcseqv1/dataset.py

import os
import json
import random
from typing import Dict, List, Tuple, Union, Any, Optional
import logging
import torch.nn.functional as F
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import T5Tokenizer
import sys
# 添加 srcseqv1 目录到 sys.path，确保可以导入自定义模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from srcseqv1.utils.ner_text_utils import  convert_tagged_text_to_token_ids


logger = logging.getLogger(__name__)


def collate_fn(batch: List[Dict[str, Any]]):
    filtered_batch = []
    for item in batch:
        if item is None or item.get("error", False):
            logger.warning("collate_fn: Skipping a None or errored sample.")
            continue
        if 'audio_features' not in item:
            logger.error(f"collate_fn: Sample (ID: {item.get('audio_id', 'Unknown')}) is missing 'audio_features'. Skipping.")
            continue
        filtered_batch.append(item)

    batch = filtered_batch
    if not batch:
        logger.warning("collate_fn received an empty batch after filtering. Returning None.")
        return None

    waveforms_list = [item['audio_features'] for item in batch]
    waveform_lengths = torch.tensor([wf.size(-1) for wf in waveforms_list if wf.numel() > 0], dtype=torch.long)
    max_audio_len = max(waveform_lengths).item() if waveform_lengths.numel() > 0 else 0
    
    padded_audio_features = []
    audio_padding_mask = []

    for waveform in waveforms_list:
        padding_len = max_audio_len - waveform.size(0)
        padded_audio = F.pad(waveform, (0, padding_len), "constant", 0)
        mask = torch.ones_like(waveform, dtype=torch.bool)
        padded_mask = F.pad(mask, (0, padding_len), "constant", False)
        padded_audio_features.append(padded_audio)
        audio_padding_mask.append(padded_mask)

    audio_features_batch = torch.stack(padded_audio_features)
    audio_padding_mask_batch = torch.stack(audio_padding_mask)

    targets_sequence_list = [item.get('targets_sequence') for item in batch if item.get('targets_sequence') is not None]

    
    final_targets_sequence = None
    if targets_sequence_list:
        max_target_len = max(target.size(0) for target in targets_sequence_list if target.numel() > 0)
        
        pad_token_id = int(targets_sequence_list[0][-1].item()) if targets_sequence_list[0].numel() > 0 else 0


        padded_targets_sequence = []
        for target in targets_sequence_list:
            target_padding_len = max_target_len - target.size(0)
            padded_target = F.pad(target, (0, target_padding_len), "constant", pad_token_id)
            padded_targets_sequence.append(padded_target)
        
        if padded_targets_sequence:
            final_targets_sequence = torch.stack(padded_targets_sequence)

    audio_ids = [item['audio_id'] for item in batch]
    original_tagged_texts = [item.get('original_tagged_text') for item in batch]
    indices = [item.get('index') for item in batch if item.get('index') is not None]

    batch_data = {
        'audio_features': audio_features_batch,
        'audio_padding_mask': audio_padding_mask_batch,
        'audio_ids': audio_ids,
        'original_tagged_text': original_tagged_texts,
    }
    if final_targets_sequence is not None:
        batch_data['targets_sequence'] = final_targets_sequence
    if indices:
        batch_data['indices'] = torch.tensor(indices, dtype=torch.long)
    logger.info(f"从collate_fn返回的数据为：{batch_data}")
    return batch_data


class AudioRelationDataset(Dataset):
    def __init__(self,
                 config: Dict[str, Any],
                 tokenizer: T5Tokenizer,
                 txt_path: str,
                 audio_subdir_name: str,
                 is_test: bool = False
                 ):
        logger.info(f"加载数据集中: {txt_path}")
        self.data = []
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        self.data.append({
                            'audio_id': parts[0],
                            'tagged_text': parts[1]
                        })
                    else:
                        logger.warning(f"跳过格式不正确的行: {line}")
        except FileNotFoundError:
             logger.error(f"数据集文件未找到: {txt_path}")
             raise

        self.config = config
        # print("配置文件为：",config)
        self.is_test = is_test
        self.tokenizer = tokenizer

        self.audio_root_dir = config['data']['audio_root_dir']
        self.audio_subdir_name = audio_subdir_name
        self.audio_dir = os.path.join(self.audio_root_dir, self.audio_subdir_name)
        
        self.sample_rate = config['data']['sample_rate']
        self.max_output_length = config['data'].get('max_output_length', 20)
        
        
        self.waveform_augment_prob = config['data'].get('waveform_augment_prob', 0.0)

        self.augment = None


        logger.info(f"数据集加载完成。样本数量: {len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if len(self.data) == 1:
            item_for_log = self.data[0]
            audio_id_for_log = item_for_log.get('audio_id', 'Unknown')
            dataset_name = self.audio_subdir_name.upper()
            
            logger.info(f"--- [__getitem__] {dataset_name} 数据集正在加载样本 ---")
            logger.info(f"  ID: {audio_id_for_log}, Index: {idx}")
        
        item = self.data[idx]
        audio_id = item.get('audio_id', f'unknown_audio_{idx}')
        original_tagged_text = item.get('tagged_text', '')
        
        if audio_id == "BAC009S0240W0216":
            logger.critical(f"--- 数据加载验证 (ID: {audio_id}) ---")
            logger.critical(f"  从标注文件加载的原始文本是: '{original_tagged_text}'")
        
        try:
            audio_path = os.path.join(self.audio_dir, f"{audio_id}.wav")

            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return None

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != self.sample_rate:
                # print("sample_rate执行t----------------------------")
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            if self.augment and waveform.dtype.is_floating_point:
                # print("augment----------------------------")
                waveform = self.augment(waveform)

            if waveform.size(0) > 1:
                # print(" waveform.size(0)----------------------------")
                waveform = torch.mean(waveform, dim=0)
            else:
                # print(" waveform.size(0)不执行----------------------------")
                waveform = waveform.squeeze(0)
            # 1. 计算均值 (去除直流偏移)
            mean = torch.mean(waveform)

            # 2. 计算标准差 (衡量整体“音量”)
            std = torch.std(waveform)

            # 3. 防止除以零的保护措施
            # 如果音频完全是静音，std会是0，直接相除会导致NaN
            if std > 1e-6:
                # 4. 执行归一化：(原始值 - 均值) / 标准差
                # 这就是标准的 Z-score Normalization
                waveform = (waveform - mean) / std
            else:
                # 如果是静音，只做去均值操作
                waveform = waveform - mean
            #  {'audio_features': tensor([-0.0104, -0.0172, -0.0151,  ...,  0.0077,  0.0089,  0.0115]),
            targets_sequence = None
            if not self.is_test:
                 targets_sequence = convert_tagged_text_to_token_ids(
                     original_tagged_text,
                     self.tokenizer,
                     self.max_output_length
                 )
                 
                 if random.random() < 0.001:
                     logger.debug(f"--- 样本调试 (ID: {audio_id}) ---")
                     logger.debug(f"  原始带标签文本: {original_tagged_text}")
                     decoded_tokens = self.tokenizer.decode(targets_sequence, skip_special_tokens=False)
                     logger.debug(f"  转换后的目标序列 (解码后): {decoded_tokens}")
                     logger.debug(f"  转换后的目标序列 (原始 ID): {targets_sequence.tolist()}")
                 
                 if targets_sequence is None or targets_sequence.numel() == 0:
                      logger.error(f"Sample {idx} (audio_id: {audio_id}) target sequence processing failed or is empty.")
                      return None

            sample_data = {
                'audio_features': waveform,
                'audio_id': audio_id,
                'original_tagged_text': original_tagged_text,
                'index': idx,
            }
            if targets_sequence is not None:
                sample_data['targets_sequence'] = targets_sequence
            logger.info(f"从AudioRelationDataset返回的数据为：{sample_data}")
            return sample_data

        except Exception as e:
            logger.error(f"Failed to load sample {idx} (audio_id: {audio_id}): {e}", exc_info=True)
            return {"audio_id": audio_id, "error": True}

# ==============================================================================
#  单独测试模块 - 将此部分代码添加到 srcseqv1/dataset.py 文件末尾
# ==============================================================================

if __name__ == '__main__':
    import shutil
    from torch.utils.data import DataLoader
    # 为了让测试脚本能够找到 ner_text_utils，需要确保项目根目录在PYTHONPATH中
    # 或者像 train.py 一样动态添加路径，但在这里我们假设您从根目录运行
    import sys
    # 添加 srcseqv1 目录到 sys.path，确保可以导入自定义模块
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from srcseqv1.utils.ner_text_utils import  convert_tagged_text_to_token_ids
    from srcseqv1.utils.ner_text_utils import init_tokenizer


    # 1. 设置基本的日志记录器
    # --------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger.info("--- 开始对 dataset.py 进行单元测试 ---")

    # 2. 准备依赖项 (使用真实音频和一个临时的、小型的标注文件)
    # --------------------------------------------------------------------
    # 初始化一个临时目录，用于存放测试用的标注文件
    temp_test_dir = "temp_test_data"
    os.makedirs(temp_test_dir, exist_ok=True)
    temp_txt_path = os.path.join(temp_test_dir, "train.txt")
    
    # --- 指向你的真实数据 ---
    real_data_root = "data/raw/audio"

    if not os.path.isdir(real_data_root):
        logger.error(f"真实数据根目录未找到: {real_data_root}")
        exit(1)

    # --- 指定用于测试的真实音频文件ID (请确保它们存在于 data/raw/audio/train 目录下) ---
    test_audio_id_1 = "BAC009S0240W0216"
    test_audio_id_2 = "BAC009S0002W0127" # 您这里用了两个相同的ID，测试依然可以工作

    real_audio_dir_train = os.path.join(real_data_root, "train")
    if not os.path.exists(os.path.join(real_audio_dir_train, f"{test_audio_id_1}.wav")):
         logger.error(f"测试音频 {test_audio_id_1}.wav 未在 {real_audio_dir_train} 中找到!")
         exit(1)
    if not os.path.exists(os.path.join(real_audio_dir_train, f"{test_audio_id_2}.wav")):
         logger.error(f"测试音频 {test_audio_id_2}.wav 未在 {real_audio_dir_train} 中找到!")
         exit(1)
    
    logger.info(f"将使用真实音频进行测试，音频来自: {real_audio_dir_train}")

    # 写入一个临时的、只包含这两条记录的标注文件
    with open(temp_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{test_audio_id_1} 这是<PER_START>张三<PER_END>的第一个音频文件\n")
        f.write(f"{test_audio_id_2} 第二个文件来自<LOC_START>北京<LOC_END>。\n")

    # 创建一个模拟的配置文件，但 audio_root_dir 指向你的真实数据目录
    mock_config = {
        'data': {
            'audio_root_dir': real_data_root,
            'sample_rate': 16000,
            'num_workers': 0,
            'waveform_augment_prob': 0.0,
            'max_output_length': 40
        }  
    }
    
    try:
        tokenizer_name = 'hugging/mengzi-t5-base' # 使用您日志中显示的路径
        test_tokenizer = init_tokenizer(tokenizer_name)
        logger.info(f"成功初始化Tokenizer: {tokenizer_name}")
    except Exception as e:
        logger.error(f"初始化Tokenizer失败: {e}")
        shutil.rmtree(temp_test_dir)
        exit(1)

    # 3. 实例化和测试 AudioRelationDataset
    # --------------------------------------------------------------------
    try:
        logger.info("\n--- 测试 AudioRelationDataset 实例化 ---")
        dataset = AudioRelationDataset(
            config=mock_config,
            tokenizer=test_tokenizer,
            txt_path=temp_txt_path,
            audio_subdir_name="train", # 指定加载 train 子目录下的音频
            is_test=False
        )

        logger.info(f"数据集大小 (测试 __len__): {len(dataset)}")
        assert len(dataset) == 2, "数据集大小不正确！"
        logger.info("✅ __len__ 测试通过。")

        logger.info("\n--- 测试 AudioRelationDataset[0] (__getitem__) ---")
        single_sample = dataset[0]
        print("样本0：",single_sample)
        expected_keys = ['audio_features', 'audio_id', 'original_tagged_text', 'index', 'targets_sequence']
        assert all(key in single_sample for key in expected_keys), "样本字典缺少必要的键！"
        logger.info(f"样本键: {list(single_sample.keys())}")
        logger.info("✅ 样本结构测试通过。")
        
        # --- 错误1 修复 ---
        # 将断言中的 'sample1' 修改为我们期望的真实音频ID
        assert single_sample['audio_id'] == test_audio_id_1, f"audio_id 不匹配！期望得到 {test_audio_id_1}，实际得到 {single_sample['audio_id']}"
        logger.info("✅ audio_id 测试通过。")
        
        assert isinstance(single_sample['audio_features'], torch.Tensor)
        assert single_sample['audio_features'].ndim == 1
        assert isinstance(single_sample['targets_sequence'], torch.Tensor)
        
        decoded_text = test_tokenizer.decode(single_sample['targets_sequence'], skip_special_tokens=False)
        logger.info(f"解码后的目标文本: {decoded_text.replace(test_tokenizer.pad_token, '')}")
        assert '<PER_START>' in decoded_text and '<PER_END>' in decoded_text, "解码文本中缺少NER标签！"
        logger.info("✅ __getitem__ 内容和形状测试通过。")

    except Exception as e:
        logger.error(f"测试 AudioRelationDataset 时发生错误: {e}", exc_info=True)
    
    # 4. 实例化和测试 DataLoader (间接测试 collate_fn)
    # --------------------------------------------------------------------
    logger.info("\n--- 测试 DataLoader 和 collate_fn ---")
    try:
        data_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn
        )

        batch_data = next(iter(data_loader))
        logger.info(f"批次数据键: {list(batch_data.keys())}")
        
        audio_batch_shape = batch_data['audio_features'].shape
        mask_batch_shape = batch_data['audio_padding_mask'].shape
        target_batch_shape = batch_data['targets_sequence'].shape
        
        logger.info(f"音频批次形状: {audio_batch_shape}")
        logger.info(f"掩码批次形状: {mask_batch_shape}")
        logger.info(f"目标批次形状: {target_batch_shape}")

        assert audio_batch_shape[0] == 2, "音频批次的batch_size不为2！"
        # --- 错误2 修复 ---
        # 移除对音频填充长度的硬编码断言，因为它取决于真实音频的长度
        # 我们可以检查更通用的属性
        assert audio_batch_shape[1] > 0, "音频批次长度为0！"
        assert mask_batch_shape == audio_batch_shape, "掩码批次形状与音频不匹配！"
        assert target_batch_shape[0] == 2, "目标批次的batch_size不为2！"

        logger.info("✅ DataLoader 和 collate_fn 形状测试通过。")

    except Exception as e:
        logger.error(f"测试 DataLoader 时发生错误: {e}", exc_info=True)

    # 5. 清理临时文件
    # --------------------------------------------------------------------
    finally:
        if os.path.exists(temp_test_dir):
            shutil.rmtree(temp_test_dir)
            logger.info(f"\n已清理并删除临时测试目录: {temp_test_dir}")
        
        logger.info("--- dataset.py 单元测试结束 ---")



