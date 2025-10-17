# src/utils/ner_text_utils.py

import torch
import numpy as np
from collections import defaultdict
import logging
import re # Import regex for parsing tagged text

from typing import List, Dict, Set, Union, Tuple, Any, Optional
from datetime import datetime
from transformers import T5Tokenizer


# 获取当前模块的 logger 实例
logger = logging.getLogger(__name__)
# 确保这个 logger 实例的级别设置为 DEBUG，以便在 trainv2-gemini.py 中设置 root logger 后能够输出


# 修正：根据实际数据格式定义NER标记模式
# 这个模式用于从原始标注文本中识别旧式标签（如 [人名], (地名), <组织名>）
# 也用于 parse_tagged_text_to_entities 兼容旧格式
OLD_NER_TAG_PATTERNS = {
    'PER': r'\[(.*?)\]',  # 人名: [实体] - 移除了多余的转义
    'LOC': r'\((.*?)\)',  # 地名: (实体) - 移除了多余的转义
    'ORG': r'<(.*?)>'     # 组织名: <实体>
}


# 将内部 NER 类型名称映射到用于构建词汇表的特殊 token 名称
# 也是模型期望生成的新式标签
NER_SPECIAL_TOKEN_MAP = {
    'PER': ('<PER_START>', '<PER_END>'),
    'LOC': ('<LOC_START>', '<LOC_END>'),
    'ORG': ('<ORG_START>', '<ORG_END>')
}


def get_special_tokens() -> List[str]:
    tokens = []
    for start, end in NER_SPECIAL_TOKEN_MAP.values():
        tokens.append(start)
        tokens.append(end)
    return sorted(list(set(tokens)))

def init_tokenizer(pretrained_model_name: str) -> T5Tokenizer:
    logger.info(f"正在从 '{pretrained_model_name}' 加载 Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)
    
    special_tokens_to_add = get_special_tokens()
    
    special_tokens_dict = {'additional_special_tokens': special_tokens_to_add}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    if num_added_toks > 0:
        logger.info(f"已向 Tokenizer 添加 {num_added_toks} 个新的特殊 NER token。")
    logger.info(f"Tokenizer 初始化完成。词汇表大小: {len(tokenizer)}")
    
    return tokenizer

def get_vocab_info_from_tokenizer(tokenizer: T5Tokenizer) -> Dict[str, Any]:
    ner_token_ids = {}
    for ner_type, (start_tag, end_tag) in NER_SPECIAL_TOKEN_MAP.items():
        start_id = tokenizer.convert_tokens_to_ids(start_tag)
        end_id = tokenizer.convert_tokens_to_ids(end_tag)
        ner_token_ids[f'{ner_type.lower()}_start_token_id'] = start_id
        ner_token_ids[f'{ner_type.lower()}_end_token_id'] = end_id

    vocab_info = {
        'token_to_id': tokenizer.get_vocab(),
        'id_to_text_token': {v: k for k, v in tokenizer.get_vocab().items()},
        'text_vocab_size': len(tokenizer),
        'sos_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'unk_token_id': tokenizer.unk_token_id,
        **ner_token_ids
    }
    return vocab_info

def convert_tagged_text_to_token_ids(
    tagged_text: str,
    tokenizer: T5Tokenizer,
    max_output_length: int
) -> Optional[torch.Tensor]:
    special_tokens = get_special_tokens()
    pattern = f"({'|'.join(re.escape(token) for token in special_tokens)})"
    
    parts = re.split(pattern, tagged_text)
    
    token_ids = []
    for part in parts:
        if not part:
            continue
        if part in special_tokens:
            part_ids = tokenizer.convert_tokens_to_ids([part])
        else:
            part_ids = tokenizer(part, add_special_tokens=False).input_ids
        token_ids.extend(part_ids)
        
    token_ids.append(tokenizer.eos_token_id)

    if len(token_ids) > max_output_length:
        token_ids = token_ids[:max_output_length - 1] + [tokenizer.eos_token_id]
    else:
        padding_needed = max_output_length - len(token_ids)
        token_ids.extend([tokenizer.pad_token_id] * padding_needed)

    return torch.tensor(token_ids, dtype=torch.long)

def parse_tagged_text_to_entities(tagged_text: str) -> List[Dict[str, str]]:
    entities = []
    
    if not any(tag in tagged_text for tag in get_special_tokens()):
        for ner_type, pattern_str in OLD_NER_TAG_PATTERNS.items():
            for match in re.finditer(pattern_str, tagged_text):
                entity_text = match.group(1).strip()
                if entity_text:
                    entities.append({'text': entity_text, 'type': ner_type})

    for ner_type, (start_tag, end_tag) in NER_SPECIAL_TOKEN_MAP.items():
        escaped_start_tag = re.escape(start_tag)
        escaped_end_tag = re.escape(end_tag)
        pattern = f"{escaped_start_tag}(.*?){escaped_end_tag}"
        for match in re.finditer(pattern, tagged_text, re.DOTALL):
            entity_text = match.group(1).strip().replace(' ', '')
            if entity_text:
                entities.append({'text': entity_text, 'type': ner_type})
            
    unique_entities_tuples = set()
    final_extracted_entities = []
    for ent in entities:
        ent_tuple = (ent['text'], ent['type'])
        if ent_tuple not in unique_entities_tuples:
            unique_entities_tuples.add(ent_tuple)
            final_extracted_entities.append(ent)
            
    return final_extracted_entities



def decode_ner_predictions(
    batch_token_ids: torch.Tensor,
    tokenizer: T5Tokenizer,
    debug_specific_audio_ids: Optional[List[str]] = None,
    audio_ids_for_batch: Optional[List[Any]] = None
) -> List[Dict[str, Any]]:
    """
    重写的NER预测解码函数 (最终修复版)。
    正确处理特殊Token，并返回带标签文本、纯净文本和结构化实体。
    """
    decoded_results = []
    
    # 1. 解码时保留所有Token，包括特殊Token
    predicted_texts_raw = tokenizer.batch_decode(
        batch_token_ids,
        skip_special_tokens=False,  # <-- 关键修复：设置为 False
        clean_up_tokenization_spaces=True
    )

    # 2. 定义我们想要手动移除的控制Token，而不是NER标签
    control_tokens_to_remove = {tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token}
    control_tokens_to_remove.discard(None) # 安全地移除None（如果bos_token不存在）

    # 3. 遍历每个原始解码文本，进行精细处理
    for i, raw_text in enumerate(predicted_texts_raw):
        audio_id = audio_ids_for_batch[i] if audio_ids_for_batch and i < len(audio_ids_for_batch) else f"unknown_{i}"
        
        # --- A. 手动清理控制Token，得到带NER标签的文本 ---
        tagged_text = raw_text
        for token in control_tokens_to_remove:
            tagged_text = tagged_text.replace(token, "")
        tagged_text = tagged_text.strip()
        
        # --- B. 生成纯净文本 (predicted_text) ---
        all_ner_tags = get_special_tokens()
        all_tags_pattern = r'(' + '|'.join(re.escape(tag) for tag in all_ner_tags) + r')'
        predicted_text = re.sub(all_tags_pattern, '', tagged_text).strip()
        predicted_text = ' '.join(predicted_text.split())

        # --- C. 严格解析实体 (data_annotation) ---
        entities = _parse_entities_strictly(tagged_text)
        unique_entities = _deduplicate_entities(entities)

        # 4. 组装最终结果
        decoded_results.append({
            "audio_id": audio_id,
            "predicted_tagged_text": tagged_text, # 带NER标签的文本
            "predicted_text": predicted_text,           # 不带标签的纯净文本
            "data_annotation": unique_entities          # 严格解析出的实体
        })
        
    return decoded_results

def _parse_entities_strictly(tagged_text: str) -> List[Dict[str, str]]:
    """一个只查找完整、配对的新格式标签的辅助函数。"""
    entities = []
    for ner_type, (start_tag, end_tag) in NER_SPECIAL_TOKEN_MAP.items():
        escaped_start_tag = re.escape(start_tag)
        escaped_end_tag = re.escape(end_tag)
        pattern = f"{escaped_start_tag}(.*?){escaped_end_tag}"
        for match in re.finditer(pattern, tagged_text, re.DOTALL):
            entity_text = match.group(1).strip()
            if entity_text:
                entities.append({'text': entity_text, 'type': ner_type})
    return entities




def _extract_entity_tokens(seq_ids, start_pos, entity_type, ner_tag_id_map):
    """提取实体token序列"""
    entity_tokens = []
    i = start_pos + 1  # 跳过开始标签
    
    while i < len(seq_ids):
        token_id = seq_ids[i]
        
        if token_id in ner_tag_id_map:
            tag_type, tag_pos = ner_tag_id_map[token_id]
            if tag_pos == 'end' and tag_type == entity_type:
                # 找到匹配的结束标签
                return entity_tokens, i
            elif tag_pos == 'start':
                # 遇到新的开始标签，当前实体未正确闭合
                return entity_tokens, i - 1
        else:
            entity_tokens.append(token_id)
        
        i += 1
    
    # 到达序列末尾，实体未闭合
    return entity_tokens, -1

def _remove_repetitive_sequences(token_ids, tokenizer, max_repeat=3):
    """智能移除重复序列"""
    if len(token_ids) < 10:
        return token_ids
    
    # 检测重复模式
    for pattern_len in range(1, min(20, len(token_ids) // 4)):
        pattern = token_ids[:pattern_len]
        repeat_count = 1
        
        i = pattern_len
        while i + pattern_len <= len(token_ids):
            if token_ids[i:i+pattern_len] == pattern:
                repeat_count += 1
                i += pattern_len
            else:
                break
        
        if repeat_count >= max_repeat:
            # 发现重复模式，截断
            return token_ids[:i]
    
    return token_ids

def _deduplicate_entities(entities):
    """去重实体"""
    unique_entities = []
    seen = set()
    
    for entity in entities:
        key = (entity['text'], entity['type'])
        if key not in seen:
            unique_entities.append(entity)
            seen.add(key)
    
    return unique_entities



# ==============================================================================
#  单独测试模块 - 将此部分代码添加到 ner_text_utils.py 文件末尾
# ==============================================================================

if __name__ == '__main__':
    # 0. 设置一个基本的日志记录器，以便在控制台看到测试过程中的详细输出
    # --------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger.info("--- 开始对 ner_text_utils.py 进行单元测试 ---")

    # ==============================================================================
    #  Part 1: 测试 parse_tagged_text_to_entities 函数
    # ==============================================================================
    print("\n" + "="*20 + " 测试 parse_tagged_text_to_entities 函数" + "="*20)
    
    # 定义一系列测试用例，覆盖不同情况
    test_cases_parse = [
        # (输入字符串, 期望输出的实体列表)
        ("没有任何实体。", []),
        ("我叫<PER_START>张三<PER_END>。", [{'text': '张三', 'type': 'PER'}]),
        ("他住在<LOC_START> 北京 <LOC_END>。", [{'text': '北京', 'type': 'LOC'}]),
        ("我为<ORG_START> 谷歌 <ORG_END>工作。", [{'text': '谷歌', 'type': 'ORG'}]),
        ("混合实体：<PER_START>姚明<PER_END>来自<LOC_START>上海<LOC_END>，曾在<ORG_START>火箭队<ORG_END>效力。", 
         [{'text': '姚明', 'type': 'PER'}, {'text': '上海', 'type': 'LOC'}, {'text': '火箭队', 'type': 'ORG'}]),
        ("旧格式兼容：[李华]住在(东京)。",  [{'text': '李华', 'type': 'PER'}, {'text': '东京', 'type': 'LOC'}]),
        ("重复实体<PER_START>王五<PER_END>和<PER_START>王五<PER_END>应该被去重。", [{'text': '王五', 'type': 'PER'}]),
        ("空实体<PER_START> <PER_END>不应被提取。", []),
        ("混合新旧格式：<PER_START>小明<PER_END>在<LOC_START>广州<LOC_END>。", [{'text': '小明', 'type': 'PER'}, {'text': '广州', 'type': 'LOC'}]),
        ("包含空格的实体名<PER_START> 乔治 R R 马丁 <PER_END>", [{'text': '乔治RR马丁', 'type': 'PER'}]) # .replace(' ', '')
    ]

    # 运行测试
    all_passed_parse = True
    for i, (text, expected_entities) in enumerate(test_cases_parse):
        # 将期望输出和实际输出都转换为元组集合，以便进行与顺序无关的比较
        expected_set = {tuple(sorted(d.items())) for d in expected_entities}
        
        actual_entities = parse_tagged_text_to_entities(text)
        actual_set = {tuple(sorted(d.items())) for d in actual_entities}

        if actual_set == expected_set:
            print(f"  ✅ Parse Test Case {i+1} PASSED.")
        else:
            print(f"  ❌ Parse Test Case {i+1} FAILED.")
            print(f"     Input:    '{text}'")
            print(f"     Expected: {expected_entities}")
            print(f"     Got:      {actual_entities}")
            all_passed_parse = False
    
    if all_passed_parse:
        print("--- ✅ 所有 parse_tagged_text_to_entities 测试通过! ---")
    else:
        print("--- ❌部分 parse_tagged_text_to_entities 测试失败! ---")


    # ==============================================================================
    #  Part 2: 测试 decode_ner_predictions 函数
    # ==============================================================================
    print("\n" + "="*20 + " 测试 decode_ner_predictions " + "="*20)

    # 1. 准备环境：初始化一个包含自定义NER标签的Tokenizer
    try:
        # 使用一个常见的中文T5模型名称，与项目保持一致
        tokenizer_name = 'hugging/mengzi-t5-base'
        test_tokenizer = init_tokenizer(tokenizer_name)
    except Exception as e:
        print(f"无法初始化Tokenizer: {e}。将跳过 decode_ner_predictions 的测试。")
        test_tokenizer = None

    if test_tokenizer:
        # 2. 准备测试数据：创建模拟的模型输出 (一个批次的Token ID)
        
        # 辅助函数：将一个Token ID列表的批次填充为规整的PyTorch张量
        def create_padded_batch(sequences: List[List[int]], pad_value: int) -> torch.Tensor:
            max_len = max(len(s) for s in sequences if s)
            padded_sequences = [s + [pad_value] * (max_len - len(s)) for s in sequences]
            return torch.tensor(padded_sequences, dtype=torch.long)

        # 用例1: 干净、标准的序列
        text1 = "你好<PER_START>李雷<PER_END>。"
        ids1 = convert_tagged_text_to_token_ids(text1, test_tokenizer, max_output_length=50).tolist()
        expected1 = {
            "audio_id": "test1",
            "predicted_tagged_text": "你好 <PER_START> 李雷 <PER_END> 。",
            "predicted_text": "你好 李雷 。",
            "data_annotation": [{'text': '李雷', 'type': 'PER'}]
        }

        # 用例2: 包含重复序列的序列 (测试_remove_repetitive_sequences)
        # 手动构造重复序列
        part_ids = test_tokenizer("在<LOC_START>上海<LOC_END>", add_special_tokens=False).input_ids
        ids2 = part_ids * 4 + [test_tokenizer.eos_token_id] # 重复4次
        # print("ids1",ids2)
        expected2 = {
            "audio_id": "test2",
            "predicted_tagged_text": "在 <LOC_START> 上海 <LOC_END> 在 <LOC_START> 上海 <LOC_END> 在 <LOC_START> 上海 <LOC_END> 在 <LOC_START> 上海 <LOC_END>",
            "predicted_text": "在 上海 在 上海 在 上海 在 上海",
            "data_annotation": [{'text': '上海', 'type': 'LOC'}] # 实体会被去重
        }

        # 用例3: 包含未闭合标签的序列
        text3 = "这是<PER_START>韩梅梅" # 没有 </PER_END>
        ids3 = test_tokenizer(text3, add_special_tokens=False).input_ids + [test_tokenizer.eos_token_id]
        expected3 = {
            "audio_id": "test3",
            "predicted_tagged_text": "这是 <PER_START> 韩梅梅",
            "predicted_text": "这是 韩梅梅",
            "data_annotation": [] # 严格模式下，不应提取任何实体
        }
        
        # 用例4: 一个空序列 (只有控制ID)
        ids4 = [test_tokenizer.pad_token_id, test_tokenizer.eos_token_id]
        expected4 = {
             "audio_id": "test4",
             "predicted_text": "",
             "data_annotation": []
        }

        # 3. 组装批次并运行函数
        batch_ids_list = [ids1, ids2, ids3, ids4]
        print("batch_ids_list:",batch_ids_list)

# batch_ids_list: [[9363, 32133, 269, 1098, 32132, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [8, 32129, 629, 32128, 8, 32129, 629, 32128, 8, 32129, 629, 32128, 8, 32129, 629, 32128, 1], [315, 32133, 1941, 1333, 1333, 1], [0, 1]]


        batch_tensor = create_padded_batch(batch_ids_list, test_tokenizer.pad_token_id)
        mock_audio_ids = ["test1", "test2", "test3", "test4"]
        print("batch_tensor",batch_tensor)
        actual_results = decode_ner_predictions(batch_tensor, test_tokenizer, audio_ids_for_batch=mock_audio_ids)

        # 4. 验证结果
        expected_results = [expected1, expected2, expected3, expected4]
        all_passed_decode = True
        for i, (actual, expected) in enumerate(zip(actual_results, expected_results)):
            # 同样使用集合进行实体列表的比较
            actual_entities_set = {tuple(sorted(d.items())) for d in actual['data_annotation']}
            expected_entities_set = {tuple(sorted(d.items())) for d in expected['data_annotation']}

            if (actual['audio_id'] == expected['audio_id'] and 
                actual['predicted_text'] == expected['predicted_text'] and 
                actual_entities_set == expected_entities_set):
                print(f"  ✅ Decode Test Case {i+1} ('{expected['audio_id']}') PASSED.")
            else:
                print(f"  ❌ Decode Test Case {i+1} ('{expected['audio_id']}') FAILED.")
                print(f"     Expected: {expected}")
                print(f"     Got:      {actual}")
                all_passed_decode = False
        
        if all_passed_decode:
            print("--- ✅ 所有 decode_ner_predictions 测试通过! ---")
        else:
            print("--- ❌ 部分 decode_ner_predictions 测试失败! ---")

    logger.info("--- ner_text_utils.py 单元测试结束 ---")




