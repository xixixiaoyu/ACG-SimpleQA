#!/usr/bin/env python3
"""
流式预筛：V4 Flash 逐个答题，答错或不知道的收集起来，凑满 500 条即停。
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

API_URL = 'https://opencode.ai/zen/go/v1'
API_KEY = os.getenv('OPENCODE_API_KEY', 'sk-u6EN2W6TWpbEql89F3f0B6UO1tKnDASphEwScUV8oODvOJmmHtQdA68ydIvdsYG4')
DATASET_PATH = './acg_simpleqa.jsonl'
OUTPUT_PATH = './filtered_500.jsonl'
TARGET = 500

SYSTEM_PROMPT = """你是一个知识渊博的 ACG（动画、漫画、游戏）领域专家。请只输出答案本身，不要任何解释、分析或额外文字。如果你不知道答案，请只输出"我不知道"三个字，不要加任何其他内容。"""


def is_wrong_or_idk(predicted: str, gold: str) -> str:
    """判断是否答错或不知道。返回 'correct' / 'wrong' / 'idk'"""
    predicted = predicted.strip()
    gold_str = gold.strip()
    if not predicted or predicted == '我不知道':
        return 'idk'
    if gold_str.lower() in predicted.lower():
        return 'correct'
    if len(predicted) >= 2 and predicted.lower() in gold_str.lower():
        return 'correct'
    return 'wrong'


def ask(client: OpenAI, question: str) -> str:
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
    all_questions = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_questions.append(json.loads(line))

    # 随机打乱，避免前 500 全是同一类别
    random.seed(42)
    random.shuffle(all_questions)

    print(f'数据集 {len(all_questions)} 题，目标收集 {TARGET} 条困难题')
    print('流式处理中...')

    client = OpenAI(base_url=API_URL, api_key=API_KEY, timeout=60)

    collected = []
    stats = {'correct': 0, 'wrong': 0, 'idk': 0}
    lock = Lock()

    def process(ex):
        nonlocal collected
        pred = ask(client, ex['question'])
        result = is_wrong_or_idk(pred, ex['answer'])
        with lock:
            stats[result] += 1
            if result in ('wrong', 'idk') and len(collected) < TARGET:
                collected.append({
                    'category': ex['category'],
                    'question': ex['question'],
                    'answer': ex['answer'],
                    'urls': ex.get('urls', []),
                })
        return result, len(collected)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process, ex): i for i, ex in enumerate(all_questions)}

        with tqdm(total=len(all_questions), desc='筛选中', unit='题') as pbar:
            for future in as_completed(futures):
                result, collected_count = future.result()
                pbar.set_postfix({
                    '已收集': collected_count,
                    '对': stats['correct'],
                    '错': stats['wrong'],
                    '不知': stats['idk'],
                })
                pbar.update(1)

                if collected_count >= TARGET:
                    # 取消剩余任务
                    for f in futures:
                        f.cancel()
                    break

    # 截断到 500
    collected = collected[:TARGET]

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in collected:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'\n完成！测试了 {stats["correct"] + stats["wrong"] + stats["idk"]} 题')
    print(f'  正确: {stats["correct"]}  答错: {stats["wrong"]}  不知道: {stats["idk"]}')
    print(f'  已保存 {len(collected)} 条到: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
