# -*- coding: utf-8 -*-
"""不需要模型的快速檢查：模擬一個「只會照抄記憶」的理想讀者。

對每題，看記憶文字（截到最後 300 字）中出現了哪些選項的內容：
  只出現一個 → 選它；出現多個 → 算錯半；都沒出現 → 視為亂猜（0.25）。
用來快速比較記憶框架的版本，最終仍以 eval_agent.py 的真實分數為準。

  python proxy_eval.py data/val [my_agent:MyAgent]
"""
import importlib
import json
import re
import sys
import types
from collections import defaultdict

sys.modules.setdefault("torch", types.SimpleNamespace(no_grad=lambda: (lambda f: f)))  # 不需要真的 torch

ABSTAIN = "你沒有提過"
PREFIX = ("現在當", "愛吃", "通勤", "住", "當", "念", "叫", "對")


def parts(opt):
    out = []
    for p in re.split(r"[，；]", opt):
        for pre in PREFIX:
            if p.startswith(pre) and len(p) > len(pre):
                p = p[len(pre):]
                break
        out.append(p.replace("過敏", "") or p)
    return out


def judge(q, ctx):
    hit = [L for L, o in q["options"].items()
           if (o == ABSTAIN and ABSTAIN in ctx) or (o != ABSTAIN and all(x in ctx for x in parts(o)))]
    if len(hit) == 1:
        return float(hit[0] == q["answer"]), hit
    if not hit:
        return 0.25, hit
    return (0.5 if q["answer"] in hit else 0.0), hit


def main():
    data_dir = sys.argv[1]
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    data_dir = args[0]
    spec = args[1] if len(args) > 1 else "my_agent:MyAgent"
    mod, cls = spec.split(":")
    agent = getattr(importlib.import_module(mod), cls)(None, None, "cpu")
    acc, present, examples = defaultdict(list), defaultdict(list), defaultdict(list)
    for line in open(f"{data_dir}/memory.jsonl", encoding="utf-8"):
        p = json.loads(line)
        agent.reset()
        for s in p["sessions"]:
            agent.new_session(s["date"])
            for t in s["turns"]:
                agent.chat(t)
        snap = agent.save_snapshot()
        for q in p["questions"]:
            agent.load_snapshot(snap)
            ctx = (agent.prepare(q["question"]) or "")[-300:]
            sc, hit = judge(q, ctx)
            acc[q["type"]].append(sc)
            present[q["type"]].append(q["answer"] in hit)
            if sc < 1 and len(examples[q["type"]]) < 4:
                examples[q["type"]].append((q["question"], q["options"][q["answer"]], ctx.replace("\n", " | ")))
    types_ = ["single", "update", "multi", "abstain", "distractor", "tool"]
    per = {t: sum(acc[t]) / len(acc[t]) for t in types_}
    print(f"proxy macro = {sum(per.values()) / 6:.3f}")
    for t in types_:
        print(f"  {t:<11} proxy {per[t]:.3f}   正確答案出現在記憶中 {sum(present[t]) / len(present[t]):.3f}")
    if "-v" in sys.argv:
        for t in types_:
            for e in examples[t]:
                print(f"[{t}] {e[0]} 答案={e[1]}  記憶={e[2]}")


if __name__ == "__main__":
    main()
