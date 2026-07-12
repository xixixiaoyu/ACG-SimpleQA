# ACG SimpleQA

基于 OpenAI SimpleQA 范式，评测大模型在 ACG（动画、漫画、游戏）领域的事实性知识。

## 数据

- `acg_simpleqa.jsonl` — 全量题库（4,234 条）
- `selected_300_clean.jsonl` — 精选 300 题（清洗版）

## 评测方式

### API 评测（`acg_simpleqa_eval.py`）

适用于任何 OpenAI 兼容 API。采样模型答题，评分模型判对错。

```bash
cp config.example.sh config.sh   # 填入 API 配置
source config.sh && python3 acg_simpleqa_eval.py
```

### CLI 评测（`eval_cli.py`）

绕过 API 兼容问题，直接用 opencode CLI 调模型。

```bash
python3 eval_cli.py opencode-go/qwen3.7-max
```

## 评测结果

### 500 题精选

| 模型 | 正确率 | 文件 |
|---|---|---|
| DeepSeek V4 Pro | 27.2% | `results_dsv4pro_500.json` |
| Gemini 3 Flash (preview) | 39.0% | `results_gemini3flash_500.json` |
| Gemini 3.5 Flash | 41.2% | `results_gemini35flash_500.json` |

### 300 题精选

| 模型 | 正确率 | 文件 |
|---|---|---|
| DeepSeek V4 Pro | 57.0% | `results_300_dsv4pro.json` |
| Kimi K2.6 | 51.3% | `results_300_kimi-k2.6.json` |
| Qwen 3.7 Max | 45.0% | `results_300_qwen3.7-max.json` |
| GLM 5.1 | 30.0% | `results_300_glm51.json` |
