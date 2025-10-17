# src/utils/ner_metrics.py

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from collections import defaultdict

# 导入 ner_text_utils 中的解析函数
from srcseqv1.utils.ner_text_utils import parse_tagged_text_to_entities, decode_ner_predictions # 导入 decode_ner_predictions

logger = logging.getLogger(__name__)





def find_matching_entities(predicted_entities: List[Dict[str, str]], true_entities: List[Dict[str, str]]) -> Tuple[int, int, int]:
    predicted_set = set((ent['text'], ent['type']) for ent in predicted_entities if isinstance(ent, dict) and 'text' in ent and 'type' in ent)
    true_set = set((ent['text'], ent['type']) for ent in true_entities if isinstance(ent, dict) and 'text' in ent and 'type' in ent)

    tp = len(predicted_set.intersection(true_set))
    fp = len(predicted_set) - tp
    fn = len(true_set) - tp

    return tp, fp, fn

def compute_ner_metrics(
    decoded_predicted_results: List[Dict[str, Any]],
    decoded_true_results: List[Dict[str, Any]],
    logger: logging.Logger
) -> Dict[str, float]:
    true_data_map = {item['audio_id']: item['data_annotation'] for item in decoded_true_results}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred_item in decoded_predicted_results:
        audio_id = pred_item['audio_id']
        predicted_entities = pred_item['data_annotation']
        
        true_entities = true_data_map.get(audio_id, [])

        tp, fp, fn = find_matching_entities(predicted_entities, true_entities)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    logger.info(f"NER Metrics: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    logger.info(f"NER Metrics: Precision={overall_precision:.4f}, Recall={overall_recall:.4f}, F1-Score={overall_f1:.4f}")

    return {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'overall_tp': total_tp,
        'overall_fp': total_fp,
        'overall_fn': total_fn
    }

def evaluate_from_files(
    ground_truth_filepath: str,
    predictions_json_filepath: str,
    text_vocab: Dict[str, Any],
    debug_specific_audio_ids: Optional[List[str]] = None
) -> Dict[str, float]:
    true_data = defaultdict(list)
    try:
        with open(ground_truth_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    audio_id_str = parts[0]
                    tagged_text = parts[1]
                    entities = parse_tagged_text_to_entities(tagged_text)
                    true_data[audio_id_str] = entities
                else:
                    logger.warning(f"跳过格式不正确的真实标注行: {line}")
    except FileNotFoundError:
        logger.error(f"真实标注 TXT 文件未找到: {ground_truth_filepath}")
        return {'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_tp': 0, 'overall_fp': 0, 'overall_fn': 0}
    except Exception as e:
        logger.error(f"读取或解析真实标注文件 {ground_truth_filepath} 失败: {e}", exc_info=True)
        return {'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_tp': 0, 'overall_fp': 0, 'overall_fn': 0}

    predicted_data = []
    try:
        with open(predictions_json_filepath, 'r', encoding='utf-8') as f:
            predicted_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"预测结果 JSON 文件未找到: {predictions_json_filepath}")
        return {'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_tp': 0, 'overall_fp': 0, 'overall_fn': 0}
    except json.JSONDecodeError as e:
        logger.error(f"解析预测结果 JSON 文件 {predictions_json_filepath} 失败: {e}", exc_info=True)
        return {'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_tp': 0, 'overall_fp': 0, 'overall_fn': 0}
    except Exception as e:
        logger.error(f"读取或解析预测结果文件 {predictions_json_filepath} 失败: {e}", exc_info=True)
        return {'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_tp': 0, 'overall_fp': 0, 'overall_fn': 0}

    decoded_true_results = []
    for audio_id, entities in true_data.items():
        decoded_true_results.append({
            'audio_id': audio_id,
            'data_annotation': entities
        })

    decoded_predicted_results = []
    for item in predicted_data:
        audio_id = item.get('audio_id')
        predicted_entities = item.get('predicted_entities', [])

        if audio_id is None:
            logger.warning(f"预测结果中缺少 'audio_id' 键，跳过该条目: {item}")
            continue
        
        decoded_predicted_results.append({
            'audio_id': audio_id,
            'data_annotation': predicted_entities
        })

    metrics = compute_ner_metrics(
        decoded_predicted_results=decoded_predicted_results,
        decoded_true_results=decoded_true_results,
        logger=logger
    )

    return metrics































# def find_matching_relations(predicted_relations: List[List[str]], true_relations: List[List[str]]) -> Tuple[int, int, int]:
#     """
#     比较预测的三元组列表和真实的标注三元组列表，计算匹配、误报和漏报数量。

#     Args:
#         predicted_relations: 预测的关系三元组列表，格式为 List[[head, relation, tail]]。
#         true_relations: 真实的标注关系三元组列表，格式为 List[[head, relation, tail]]。

#     Returns:
#         一个包含 (真阳性TP, 假阳性FP, 假阴性FN) 的元组。
#     """
#     predicted_set = set(tuple(r) for r in predicted_relations if isinstance(r, list) and len(r) == 3)
#     true_set = set(tuple(r) for r in true_relations if isinstance(r, list) and len(r) == 3)

#     # 计算真阳性 (TP): 预测正确且实际存在的关系
#     tp = len(predicted_set.intersection(true_set))

#     # 计算假阳性 (FP): 预测存在但实际不存在的关系
#     fp = len(predicted_set.difference(true_set))

#     # 计算假阴性 (FN): 实际存在但未被预测到的关系
#     fn = len(true_set.difference(predicted_set))

#     return tp, fp, fn


# def compute_ner_metrics(
#     decoded_predicted_results: List[Dict[str, Any]],
#     decoded_true_results: List[Dict[str, Any]],
#     debug_specific_audio_ids: Optional[List[str]] = None
# ) -> Dict[str, float]:
#     """
#     计算命名实体识别 (NER) 的精确率 (Precision)、召回率 (Recall) 和 F1 分数。

#     Args:
#         decoded_predicted_results: 包含模型预测结果的列表，每个字典包含 'audio_id' 和 'data_annotation' (预测实体列表)。
#                                    预测实体列表的格式: [{'text': '实体文本', 'type': '实体类型'}, ...]
#         decoded_true_results: 包含真实标注结果的列表，每个字典包含 'audio_id' 和 'data_annotation' (真实实体列表)。
#                               真实实体列表的格式: [{'text': '实体文本', 'type': '实体类型'}, ...]
#         debug_specific_audio_ids: 一个可选的音频 ID 列表，用于在调试模式下输出特定样本的详细评估信息。

#     Returns:
#         包含整体 Precision, Recall, F1 分数以及 TP, FP, FN 的字典。
#     """
#     overall_tp = 0
#     overall_fp = 0
#     overall_fn = 0

#     # 将真实标签结果转换为以 audio_id 为键的字典，方便查找
#     true_results_map = {item['audio_id']: item['data_annotation'] for item in decoded_true_results}

#     for pred_item in decoded_predicted_results:
#         audio_id = pred_item['audio_id']
#         predicted_entities = pred_item['data_annotation']

#         true_entities = true_results_map.get(audio_id, [])  # 获取对应 audio_id 的真实实体

#         # 调试日志
#         is_debug_target_sample = (debug_specific_audio_ids and audio_id in debug_specific_audio_ids)
#         if logger.isEnabledFor(logging.DEBUG) or is_debug_target_sample:
#             logger.debug(f"\n--- Evaluating Audio ID: {audio_id} ---")
#             logger.debug(f"  Ground Truth Entities: {true_entities}")
#             logger.debug(f"  Predicted Entities: {predicted_entities}")

#         # 将实体列表转换为可哈希的集合，以便进行集合操作（严格匹配：文本和类型都匹配）
#         # 确保转换为元组，方便集合操作。使用 (text, type) 形式来比较实体。
#         predicted_set_for_comparison = set((ent['text'], ent['type']) for ent in predicted_entities)
#         true_set_for_comparison = set((ent['text'], ent['type']) for ent in true_entities)

#         if logger.isEnabledFor(logging.DEBUG) or is_debug_target_sample:
#             logger.debug(f"  Predicted Set (for comparison): {predicted_set_for_comparison}")
#             logger.debug(f"  True Set (for comparison): {true_set_for_comparison}")


#         tp_current = len(predicted_set_for_comparison.intersection(true_set_for_comparison))
#         fp_current = len(predicted_set_for_comparison.difference(true_set_for_comparison))
#         fn_current = len(true_set_for_comparison.difference(predicted_set_for_comparison))

#         overall_tp += tp_current
#         overall_fp += fp_current
#         overall_fn += fn_current

#         if logger.isEnabledFor(logging.DEBUG) or is_debug_target_sample:
#             logger.debug(f"  Current TP: {tp_current}, FP: {fp_current}, FN: {fn_current}")

#     # 计算总体的 Precision, Recall, F1
#     precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
#     recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
#     f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

#     return {
#         'overall_precision': precision,
#         'overall_recall': recall,
#         'overall_f1': f1,
#         'overall_tp': overall_tp,
#         'overall_fp': overall_fp,
#         'overall_fn': overall_fn
#     }


# def evaluate_from_files(
#     ground_truth_filepath: str,
#     predictions_json_filepath: str,
#     text_vocab: Dict[str, Any],
#     debug_specific_audio_ids: Optional[List[str]] = None
# ) -> Dict[str, float]:
#     """
#     从文件读取预测结果和真实标注，并计算 NER 评估指标。

#     Args:
#         predictions_json_filepath: 包含模型预测结果的 JSON 文件路径。
#                                    JSON 格式: [{'audio_id': '...', 'predicted_text': '...', 'predicted_entities': [...]}, ...]
#         ground_truth_filepath: 包含真实标注文本的 TXT 文件路径。
#                            TXT 格式: "audio_id 原始带标签文本"
#         debug_specific_audio_ids: 一个可选的音频 ID 列表，用于在调试模式下输出特定样本的详细评估信息。

#     Returns:
#         包含 Precision, Recall, F1 分数以及 TP, FP, FN 的字典。
#     """
#     # 1. 读取真实标注数据
#     true_data = defaultdict(list)
#     try:
#         with open(ground_truth_filepath, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 parts = line.split(' ', 1)
#                 if len(parts) == 2:
#                     audio_id = parts[0]
#                     tagged_text = parts[1]
#                     # 使用 parse_tagged_text_to_entities 从原始带标签文本中提取实体
#                     entities = parse_tagged_text_to_entities(
#                         tagged_text,
#                         audio_id=audio_id, # 传递 audio_id 用于调试日志
#                         debug_specific_audio_ids=debug_specific_audio_ids
#                     )
#                     true_data[audio_id] = entities
#                 else:
#                     logger.warning(f"跳过格式不正确的真实标注行: {line}")
#     except FileNotFoundError:
#         logger.error(f"真实标注 TXT 文件未找到: {ground_truth_filepath}")
#         return {'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_tp': 0, 'overall_fp': 0, 'overall_fn': 0}
#     except Exception as e:
#         logger.error(f"读取或解析真实标注文件 {ground_truth_filepath} 失败: {e}", exc_info=True)
#         return {'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_tp': 0, 'overall_fp': 0, 'overall_fn': 0}


#     # 2. 读取预测结果
#     predicted_data = []
#     try:
#         with open(predictions_json_filepath, 'r', encoding='utf-8') as f:
#             predicted_data = json.load(f)
#     except FileNotFoundError:
#         logger.error(f"预测结果 JSON 文件未找到: {predictions_json_filepath}")
#         return {'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_tp': 0, 'overall_fp': 0, 'overall_fn': 0}
#     except json.JSONDecodeError as e:
#         logger.error(f"解析预测结果 JSON 文件 {predictions_json_filepath} 失败: {e}", exc_info=True)
#         return {'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_tp': 0, 'overall_fp': 0, 'overall_fn': 0}
#     except Exception as e:
#         logger.error(f"读取或解析预测结果文件 {predictions_json_filepath} 失败: {e}", exc_info=True)
#         return {'overall_precision': 0.0, 'overall_recall': 0.0, 'overall_f1': 0.0, 'overall_tp': 0, 'overall_fp': 0, 'overall_fn': 0}


#     # 3. 准备 compute_ner_metrics 函数所需的输入格式
#     # decoded_true_results 保持不变
#     decoded_true_results = []
#     for audio_id, entities in true_data.items():
#         decoded_true_results.append({
#             'audio_id': audio_id,
#             'data_annotation': entities
#         })

#     # predicted_data_for_evaluation 转换为正确的格式
#     # 注意：我们假设 predictions_json_filepath 中的每个条目已经包含了 'predicted_entities'
#     # 如果不包含，这里需要调用 decode_ner_predictions 来生成 'predicted_entities'
#     decoded_predicted_results = []
#     for item in predicted_data:
#         audio_id = item.get('audio_id')
#         predicted_text = item.get('predicted_text')
#         predicted_entities = item.get('predicted_entities') # 直接使用 JSON 中已有的 predicted_entities

#         if audio_id is None:
#             logger.warning(f"预测结果中缺少 'audio_id' 键，跳过该条目: {item}")
#             continue
        
#         if predicted_entities is None:
#             # 如果 predicted_entities 不存在，但有 predicted_text，则尝试从 predicted_text 中解析
#             if predicted_text is not None:
#                 logger.warning(f"预测结果中 'predicted_entities' 为空或缺失，尝试从 'predicted_text' 解析 audio_id: {audio_id}")
#                 predicted_entities = decode_ner_predictions(
#                     [predicted_text], # decode_ner_predictions 期望列表输入
#                     text_vocab=None, # 在此函数中不需要 text_vocab，因为它直接操作文本
#                     audio_ids=[audio_id], # 传递 audio_id 用于调试日志
#                     debug_specific_audio_ids=debug_specific_audio_ids
#                 )[0]['data_annotation'] # 获取解析后的实体列表
#             else:
#                 logger.warning(f"预测结果中 'predicted_entities' 和 'predicted_text' 都缺失，跳过 audio_id: {audio_id}")
#                 predicted_entities = [] # 确保为列表，避免后续错误

#         decoded_predicted_results.append({
#             'audio_id': audio_id,
#             'data_annotation': predicted_entities
#         })


#     # 4. 调用 compute_ner_metrics 计算指标
#     metrics = compute_ner_metrics(
#         decoded_predicted_results=decoded_predicted_results,
#         decoded_true_results=decoded_true_results,
#         debug_specific_audio_ids=debug_specific_audio_ids
#     )
#     print("指标为：",metrics)

#     return metrics








    