#!/bin/bash
# ACG-SimpleQA 评测配置 - DeepSeek V4 Pro (sampler) + V4 Flash (grader)

export SAMPLER_API_URL="https://openrouter.ai/api/v1"
export SAMPLER_API_KEY="your-openrouter-api-key"
export SAMPLER_MODEL="google/gemini-3.5-flash"

export GRADER_API_URL="https://opencode.ai/zen/go/v1"
export GRADER_API_KEY="your-grader-api-key"
export GRADER_MODEL="deepseek-v4-flash"

export DATASET_PATH="./filtered_500.jsonl"
export OUTPUT_PATH="./results_gemini35flash.json"
export EVAL_THREADS="20"
export EVAL_LIMIT=""
