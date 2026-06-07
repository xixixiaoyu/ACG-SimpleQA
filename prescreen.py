#!/usr/bin/env python3
"""
预筛脚本：用 V4 Flash 快速跑全量 4235 题，
选出它答错 / 不知道的题目，作为 V4 Pro 评测用的困难子集。
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

# ---------- 配置 ----------
API_URL = 'https://opencode.ai/zen/go/v1'
API_KEY = os.getenv('OPENCODE_API_KEY', 'sk-u6EN2W6TWpbEql89F3f0B6UO1tKnDASphEwScUV8oODvOJmmHtQdA68ydIvdsYG4')

DATASET_PATH = './acg_simpleqa.jsonl'
OUTPUT_PATH = './filtered_questions.jsonl'
TARGET_COUNT = 500

SYSTEM_PROMPT = """你是一个知识渊博的 ACG（动画、漫画、游戏）领域专家。请只输出答案本身，不要任何解释、分析或额外文字。如果你不知道答案，请只输出"我不知道"三个字，不要加任何其他内容。"""


def load_dataset(path: str) -> list[dict]:
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def is_correct(predicted: str, gold: str) -> bool:
    """简单判断：预测答案是否包含标准答案。"""
    predicted = predicted.strip().lower()
    gold = gold.strip().lower()
    if not predicted:
        return False
    if predicted == '我不知道':
        return False
    # 检查是否包含 gold answer
    if gold in predicted:
        return True
    # 检查 predicted 是否是 gold 的子串（针对很短的预测）
    if len(predicted) >= 2 and predicted in gold:
        return True
    return False


def ask_v4_flash(client: OpenAI, question: str) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model='deepseek-v4-flash',
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': question},
                ],
                temperature=0.0,
                max_tokens=256,
                extra_body={'thinking': {'type': 'disabled'}},
            )
            content = resp.choices[0].message.content
            return content.strip() if content else ''
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
    return ''


def main():
    examples = load_dataset(DATASET_PATH)
    total = len(examples)
    print(f'加载 {total} 条样本，开始预筛...')

    client = OpenAI(base_url=API_URL, api_key=API_KEY, timeout=60)
    
    results = []
    correct_count = 0
    wrong_count = 0
    idk_count = 0
    error_count = 0

    def process(ex):
        question = ex['question']
        gold = ex['answer']
        pred = ask_v4_flash(client, question)
        correct = is_correct(pred, gold)
        is_idk = pred == '我不知道'
        return {**ex, 'v4flash_pred': pred, 'v4flash_correct': correct, 'v4flash_idk': is_idk}

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process, ex): i for i, ex in enumerate(examples)}
        
        with tqdm(total=total, desc='V4 Flash 预筛', unit='题') as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    result = {**examples[futures[future]], 'v4flash_pred': '[ERROR]', 'v4flash_correct': False, 'v4flash_idk': False}
                    error_count += 1

                results.append((futures[future], result))
                
                if result['v4flash_correct']:
                    correct_count += 1
                elif result['v4flash_idk']:
                    idk_count += 1
                else:
                    wrong_count += 1

                pbar.set_postfix({'对': correct_count, '错': wrong_count, '不知': idk_count, 'err': error_count})
                pbar.update(1)

    # 排序回原始顺序
    results.sort(key=lambda x: x[0])
    all_results = [r[1] for r in results]

    print(f'\n预筛完成：共 {total} 题')
    print(f'  V4 Flash 正确: {correct_count} ({correct_count/total:.1%})')
    print(f'  V4 Flash 错误: {wrong_count} ({wrong_count/total:.1%})')
    print(f'  V4 Flash 不知道: {idk_count} ({idk_count/total:.1%})')

    # 过滤：保留错误 + 不知道的
    hard = [r for r in all_results if not r['v4flash_correct']]
    wrong_only = [r for r in hard if not r['v4flash_idk']]
    idk_only = [r for r in hard if r['v4flash_idk']]

    print(f'  困难题目（错+不知）: {len(hard)}')
    print(f'    其中答错: {len(wrong_only)}')
    print(f'    其中不知道: {len(idk_only)}')

    # 从困难题目中均匀采样
    target = min(TARGET_COUNT, len(hard))
    # 按比例从 wrong 和 idk 中采样
    wrong_sample_count = min(len(wrong_only), int(target * len(wrong_only) / len(hard)))
    idk_sample_count = target - wrong_sample_count
    if idk_sample_count > len(idk_only):
        idk_sample_count = len(idk_only)
        wrong_sample_count = target - idk_sample_count

    import random
    random.seed(42)
    sampled = random.sample(wrong_only, wrong_sample_count) + random.sample(idk_only, idk_sample_count)
    random.shuffle(sampled)

    # 保存为 JSONL（只保留原始字段）
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in sampled:
            f.write(json.dumps({
                'category': item['category'],
                'question': item['question'],
                'answer': item['answer'],
                'urls': item.get('urls', []),
            }, ensure_ascii=False) + '\n')

    print(f'\n已保存 {len(sampled)} 条筛选后题目到: {OUTPUT_PATH}')
    print(f'  答错类: {wrong_sample_count}, 不知道类: {idk_sample_count}')


if __name__ == '__main__':
    main()
