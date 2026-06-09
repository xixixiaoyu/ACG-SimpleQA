#!/usr/bin/env python3
"""用 opencode CLI 跑评测，绕过 API 兼容问题"""
import json, subprocess, sys, os

MODEL = sys.argv[1] if len(sys.argv) > 1 else "opencode-go/qwen3.7-max"
DATASET = "selected_300_clean.jsonl"
OUTPUT = f"results_300_{MODEL.split('/')[-1]}.json"

PROMPT = "你是一个ACG领域专家。只输出答案，不要解释。不知道就说'我不知道'。"

def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def ask(question):
    try:
        result = subprocess.run(
            ["opencode", "run", "-m", MODEL, "--print-logs"],
            input=f"{PROMPT}\n\n问题：{question}",
            capture_output=True, text=True, timeout=120, cwd="/Users/yunmu/Desktop/acg"
        )
        # 提取实际输出（跳过 INFO 行和空行）
        lines = result.stdout.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('INFO') and not line.startswith('>'):
                return line
        return result.stdout.strip()
    except Exception as e:
        return f"[ERROR] {e}"

def grade(pred, gold):
    pred = pred.strip().lower()
    gold = gold.strip().lower()
    # 用 DeepSeek V4 Flash 做 grader，用 API 调
    if not pred or pred == "我不知道":
        return {"grade": "NOT_ATTEMPTED", "is_correct": False, "raw": pred}
    from openai import OpenAI
    client = OpenAI(base_url="https://opencode.ai/zen/go/v1",
                    api_key="sk-u6EN2W6TWpbEql89F3f0B6UO1tKnDASphEwScUV8oODvOJmmHtQdA68ydIvdsYG4")
    grader_prompt = f"""判断以下回答是否正确，只输出 A (正确), B (错误), C (未尝试)。
问题：{question}
标准答案：{gold}
预测答案：{pred}
你的判断（只输出字母）："""
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role":"user","content":grader_prompt}],
                max_tokens=10, temperature=0,
                extra_body={"thinking":{"type":"disabled"}},
            )
            raw = r.choices[0].message.content.strip()
            if "A" in raw: return {"grade": "CORRECT", "is_correct": True, "raw": raw}
            if "B" in raw: return {"grade": "INCORRECT", "is_correct": False, "raw": raw}
            return {"grade": "NOT_ATTEMPTED", "is_correct": False, "raw": raw}
        except:
            pass
    return {"grade": "NOT_ATTEMPTED", "is_correct": False, "raw": "[GRADER_ERROR]"}

if __name__ == "__main__":
    items = load(DATASET)
    print(f"模型: {MODEL}, 题目: {len(items)} 条")
    results = []
    correct = wrong = na = 0
    for i, item in enumerate(items):
        q = item["question"]
        gold = item["answer"]
        pred = ask(q)
        g = grade(pred, gold)
        if g["is_correct"]: correct += 1
        elif g["grade"] == "INCORRECT": wrong += 1
        else: na += 1
        results.append({
            "category": item["category"],
            "question": q,
            "gold_answer": gold,
            "predicted_answer": pred,
            "grader_raw": g["raw"],
            "grade": g["grade"],
            "is_correct": g["is_correct"],
        })
        print(f"[{i+1}/{len(items)}] C:{correct} I:{wrong} NA:{na}  {q[:40]}...", flush=True)

    out = {
        "config": {"model": MODEL},
        "metrics": {
            "total": len(items), "correct": correct, "incorrect": wrong, "not_attempted": na,
            "correct_rate": round(correct/len(items),4),
        },
        "details": results,
    }
    with open(OUTPUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n=== {MODEL} ===")
    print(f"正确: {correct}, 错误: {wrong}, NA: {na}, 分: {correct/len(items):.4f}")
    print(f"结果: {OUTPUT}")
