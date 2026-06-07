#!/bin/bash
# GLM 5.1 x 300题

export SAMPLER_API_URL="https://opencode.ai/zen/go/v1"
export SAMPLER_API_KEY="sk-u6EN2W6TWpbEql89F3f0B6UO1tKnDASphEwScUV8oODvOJmmHtQdA68ydIvdsYG4"
export SAMPLER_MODEL="glm-5.1"

export GRADER_API_URL="https://opencode.ai/zen/go/v1"
export GRADER_API_KEY="sk-u6EN2W6TWpbEql89F3f0B6UO1tKnDASphEwScUV8oODvOJmmHtQdA68ydIvdsYG4"
export GRADER_MODEL="deepseek-v4-flash"

export DATASET_PATH="./selected_300_clean.jsonl"
export OUTPUT_PATH="./results_300_glm51.json"
export EVAL_THREADS="20"
export EVAL_LIMIT=""
