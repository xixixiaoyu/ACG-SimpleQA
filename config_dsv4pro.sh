#!/bin/bash
# ACG-SimpleQA 评测配置 - DeepSeek V4 Pro

export SAMPLER_API_URL="https://opencode.ai/zen/go/v1"
export SAMPLER_API_KEY="sk-u6EN2W6TWpbEql89F3f0B6UO1tKnDASphEwScUV8oODvOJmmHtQdA68ydIvdsYG4"
export SAMPLER_MODEL="deepseek-v4-pro"

export GRADER_API_URL="https://opencode.ai/zen/go/v1"
export GRADER_API_KEY="sk-u6EN2W6TWpbEql89F3f0B6UO1tKnDASphEwScUV8oODvOJmmHtQdA68ydIvdsYG4"
export GRADER_MODEL="deepseek-v4-flash"

export DATASET_PATH="./filtered_500.jsonl"
export OUTPUT_PATH="./results_dsv4pro.json"
export EVAL_THREADS="20"
export EVAL_LIMIT=""
