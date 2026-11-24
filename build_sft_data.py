# -*- coding: utf-8 -*-
"""產生 SFT 訓練資料（工具呼叫 + 讀記憶），格式與 eval_agent.py 評估時完全相同。

工具呼叫（tool）：
  prompt = BaseAgent.next_action 的訊息（同樣的 system、tools 放進 chat template）
  target = <tool_call>{...}</tool_call>，或不該呼叫工具時的一句簡短回應
讀記憶（read）：
  prompt = eval_agent.build_prompt（記憶文字由 MyAgent.prepare() 實際產生）
  四個選項都存下，訓練時讓正確選項的平均機率高於其他選項
資料只來自 train 與自製的 A、C 套（val、B 套模擬測試、D 套盲測都不使用）。

  python build_sft_data.py --model_dir <minimind-3> --out data/sft/sft.jsonl \
      --sources data/train data/train_extraA
"""
import argparse
import json
import os
import random

from transformers import AutoTokenizer

from agent_base import BaseAgent
from eval_agent import build_prompt

SYSTEM = "你是一個會記住使用者資訊的助理。"   # 與 BaseAgent.next_action 相同
REPLIES = ["好的，我知道了。", "了解！", "聽起來不錯。", "嗯嗯，我懂。", "謝謝你跟我分享。", "原來如此。", "好喔！"]


def tool_call_text(name, args):
    """MiniMind 原生格式，與 tool_spec.json 的 call_format 相同。"""
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"


def value_pool():
    """換值增強用的詞庫：公開模板與自製 A、C 套的所有值（不含 B、D 套），再加上隨機組字。"""
    import importlib.util
    mods = []
    for path in ["templates_public.py", "templates/extra_A.py", "templates/extra_C.py"]:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            mods.append(m)
    words, chars = set(), set()
    for m in mods:
        for d in list(getattr(m, "ATTRS", {}).values()) + list(getattr(m, "EXTRA", {}).values()) \
                + list(getattr(m, "NEW_ATTRS", {}).values()):
            for v in d.get("pool", []):
                words.add(v)
                chars.update(c for c in v if "\u4e00" <= c <= "\u9fff")
    return sorted(words), sorted(chars)


def augment_value(it, rng, pool):
    """把 memory_save 的值換成別的詞（一半用詞庫、一半隨機組字），讓模型學會「照抄原句」而不是背值。"""
    exp = it["expected"]
    v = exp.get("args", {}).get("value")
    last = it["context"][-1]
    if not isinstance(v, str) or not v or last.count(v) != 1:
        return None
    words, chars = pool
    new = rng.choice(words) if rng.random() < 0.5 else "".join(rng.choice(chars) for _ in range(rng.randint(2, 4)))
    if new == v or new in last:
        return None
    it = json.loads(json.dumps(it, ensure_ascii=False))
    it["context"][-1] = last.replace(v, new)
    it["expected"]["args"]["value"] = new
    it["id"] += "_aug"
    return it


def tool_examples(path, tok, rng, value_aug=0.0, pool=None):
    tools = BaseAgent.tools()
    for line in open(path, encoding="utf-8"):
        it = json.loads(line)
        items = [it]
        if value_aug > 0 and it["kind"] in ("save", "update") and rng.random() < value_aug:
            aug = augment_value(it, rng, pool)
            if aug:
                items.append(aug)
        for it in items:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(it["context"])}]
            prompt = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
            exp = it["expected"]
            target = tool_call_text(exp["tool"], exp.get("args", {})) if exp["action"] == "call" else rng.choice(REPLIES)
            yield dict(task="tool", id=it["id"], kind=it["kind"], prompt=prompt, target=target)


def read_examples(path, tok, agent, rng, n_shuffle):
    letters = "ABCD"
    for line in open(path, encoding="utf-8"):
        p = json.loads(line)
        agent.reset()
        for s in p["sessions"]:
            agent.new_session(s["date"])
            for t in s["turns"]:
                agent.chat(t)
        snap = agent.save_snapshot()
        for q in p["questions"]:
            agent.load_snapshot(snap)
            context = (agent.prepare(q["question"]) or "")[-300:]
            opts = [q["options"][L] for L in letters]
            gold = opts[letters.index(q["answer"])]
            for k in range(n_shuffle):                       # 第 0 次用原始順序，其餘打亂選項順序
                order = opts[:] if k == 0 else rng.sample(opts, 4)
                qq = dict(question=q["question"], options=dict(zip(letters, order)))
                yield dict(task="read", id=f"{q['qid']}_{k}", type=q["type"],
                           prompt=build_prompt(tok, context, qq, True),
                           options=order, answer=order.index(gold))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True, help="用它的 tokenizer / chat template（minimind-3）")
    ap.add_argument("--sources", nargs="+", default=["data/train", "data/train_extraA"])
    ap.add_argument("--out", default="data/sft/sft.jsonl")
    ap.add_argument("--n_shuffle", type=int, default=2, help="讀記憶題每題產生幾種選項順序")
    ap.add_argument("--no_read", action="store_true", help="只產生工具呼叫資料（消融用）")
    ap.add_argument("--no_tool", action="store_true", help="只產生讀記憶資料（消融用）")
    ap.add_argument("--context_mode", default=None,
                    help="MyAgent 記憶文字的組法（retrieve / fill / fill_hint / chrono），預設用 my_agent.py 的設定")
    ap.add_argument("--value_aug", type=float, default=0.0,
                    help="memory_save 題目有多少比例額外產生一筆「換值」的增強資料")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu",
                    help="embedding 模型使用的裝置")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    agent = None
    if not args.no_read:
        from my_agent import MyAgent
        if args.context_mode:
            MyAgent.context_mode = args.context_mode
        agent = MyAgent(None, tok, args.device)
        print("context_mode =", MyAgent.context_mode)
    pool = value_pool() if args.value_aug > 0 else None
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    counts = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for src in args.sources:
            gens = []
            if not args.no_tool:
                gens.append(tool_examples(os.path.join(src, "tool.jsonl"), tok, rng, args.value_aug, pool))
            if not args.no_read:
                gens.append(read_examples(os.path.join(src, "memory.jsonl"), tok, agent, rng, args.n_shuffle))
            for g in gens:
                for ex in g:
                    ex["source"] = os.path.basename(src)
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    key = (ex["source"], ex["task"])
                    counts[key] = counts.get(key, 0) + 1
    for (src, task), n in sorted(counts.items()):
        print(f"  {src:<14} {task:<5} {n:>6} 筆")
    print(f"寫入 {args.out}")


if __name__ == "__main__":
    main()
