#!/bin/bash
# ============================================================
# ACG-SimpleQA 评测配置
# 复制此文件为 config.sh，修改对应值，然后 source config.sh
# ============================================================

# ---------- 待测模型（Sampler） ----------
# 你的待测模型的 OpenAI 兼容 API 地址
export SAMPLER_API_URL="http://localhost:8000/v1"
# 待测模型的 API Key（无鉴权时可填 placeholder）
export SAMPLER_API_KEY="not-needed"
# 待测模型名称（传给 API 的 model 参数）
export SAMPLER_MODEL="your-model-name"

# ---------- 评判模型（Grader） ----------
# 评判模型的 OpenAI 兼容 API 地址
export GRADER_API_URL="https://api.openai.com/v1"
# 评判模型的 API Key
export GRADER_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# 评判模型名称
export GRADER_MODEL="gpt-4o"

# ---------- 评测参数 ----------
# 评测样本数（留空则评测全部 4235 条）
export EVAL_LIMIT=""
# 并行线程数（默认 10）
export EVAL_THREADS="10"
# 数据集路径
export DATASET_PATH="./acg_simpleqa.jsonl"
# 结果输出路径
export OUTPUT_PATH="./results.json"
