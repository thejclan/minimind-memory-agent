# -*- coding: utf-8 -*-
"""HW2 SFT：同時訓練「工具呼叫」與「讀記憶」，從 minimind-3 出發，輸出 Hugging Face 格式模型。

loss：
  tool  一般 SFT：只對助理輸出（<tool_call>…</tool_call> 或簡短回應）計算 cross-entropy
  read  依評分方式設計：每個選項算「平均每字 log-likelihood」（與 eval_agent.score_options 相同），
        四個選項做 softmax，對正確選項取 cross-entropy；再加上正確選項的一般 LM loss（權重 lm_weight）

  python train_sft.py --base_model ~/hw1/minimind-3 --data data/sft/sft.jsonl --out_dir models/sft_v1
"""
import argparse
import json
import math
import os
import random
import shutil
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def turn_end(tok):
    """chat template 在助理回覆後加上的結尾文字（例如 <|im_end|>\\n），訓練目標要包含它，模型才會停。"""
    mark = "⁣答覆⁣"
    text = tok.apply_chat_template([{"role": "user", "content": "x"}, {"role": "assistant", "content": mark}],
                                   tokenize=False)
    return text.split(mark, 1)[1]


def pad(seqs, pad_id):
    n = max(map(len, seqs))
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return ids, att


def tool_batch(model, tok, exs, end, dev):
    seqs, labels = [], []
    for e in exs:
        p = tok(e["prompt"], add_special_tokens=False).input_ids
        t = tok(e["target"] + end, add_special_tokens=False).input_ids
        seqs.append(p + t)
        labels.append([-100] * len(p) + t)
    ids, att = pad(seqs, tok.pad_token_id or 0)
    lab, _ = pad(labels, -100)
    lab[att == 0] = -100
    ids, att, lab = ids.to(dev), att.to(dev), lab.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
        return model(input_ids=ids, attention_mask=att, labels=lab).loss


def read_batch(model, tok, exs, dev, lm_weight):
    """每題 4 個選項各一條序列；算每個選項的平均每字 log-likelihood。"""
    seqs, spans, nchar = [], [], []
    for e in exs:
        p = tok(e["prompt"], add_special_tokens=False).input_ids
        for o in e["options"]:
            oi = tok(o, add_special_tokens=False).input_ids
            seqs.append(p + oi)
            spans.append((len(p), len(p) + len(oi)))
            nchar.append(max(len(o), 1))
    ids, att = pad(seqs, tok.pad_token_id or 0)
    ids, att = ids.to(dev), att.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
        logits = model(input_ids=ids, attention_mask=att).logits
    logp = F.log_softmax(logits.float(), -1)
    tot = []
    for r, (a, b) in enumerate(spans):
        pos = torch.arange(a - 1, b - 1, device=dev)
        tot.append(logp[r, pos, ids[r, a:b]].sum())
    tot = torch.stack(tot)
    score = (tot / torch.tensor(nchar, device=dev, dtype=torch.float)).view(len(exs), 4)
    gold = torch.tensor([e["answer"] for e in exs], device=dev)
    rank = F.cross_entropy(score, gold)
    gold_tok = torch.tensor([b - a for a, b in spans], device=dev, dtype=torch.float).view(len(exs), 4)
    lm = -(tot.view(len(exs), 4).gather(1, gold[:, None]) / gold_tok.gather(1, gold[:, None])).mean()
    acc = (score.argmax(1) == gold).float().mean()
    return rank + lm_weight * lm, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--data", default="data/sft/sft.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tasks", nargs="+", default=["tool", "read"])
    ap.add_argument("--sources", nargs="*", default=None, help="只用這些來源（例如 train），預設全部")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--tool_bs", type=int, default=16)
    ap.add_argument("--read_bs", type=int, default=4, help="每批幾題（每題 4 條序列）")
    ap.add_argument("--lm_weight", type=float, default=0.5)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 筆（除錯用）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise SystemExit(f"輸出資料夾已存在，為避免覆蓋而停止：{out_dir}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = args.device

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    rows = [r for r in rows if r["task"] in args.tasks and (not args.sources or r["source"] in args.sources)]
    if args.limit:
        rows = random.sample(rows, min(args.limit, len(rows)))
    tok = AutoTokenizer.from_pretrained(args.base_model)
    end = turn_end(tok)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32).to(dev)
    model.train()
    by_task = {t: [r for r in rows if r["task"] == t] for t in args.tasks}
    print("訓練資料：" + "、".join(f"{t} {len(v)} 筆" for t, v in by_task.items()), "｜回覆結尾：", repr(end))
    ex = next((r for r in rows if r["task"] == "tool"), None)
    if ex:
        print("---- 工具呼叫範例（模型看到的結尾 + 要學的輸出）----")
        print(ex["prompt"][-200:] + "【" + ex["target"] + end + "】")

    def batches(rng):
        out = []
        for t, rs in by_task.items():
            rs = rs[:]
            rng.shuffle(rs)
            bs = args.tool_bs if t == "tool" else args.read_bs
            out += [(t, rs[i:i + bs]) for i in range(0, len(rs), bs)]
        rng.shuffle(out)
        return out

    rng = random.Random(args.seed)
    n_steps = len(batches(random.Random(0))) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(n_steps * args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (n_steps - s) / max(1, n_steps - warm))))
    step, t0, log = 0, time.time(), {"tool": [], "read": [], "read_acc": []}
    history = []
    for epoch in range(1, args.epochs + 1):
        for task, exs in batches(rng):
            if task == "tool":
                loss = tool_batch(model, tok, exs, end, dev)
            else:
                loss, acc = read_batch(model, tok, exs, dev, args.lm_weight)
                log["read_acc"].append(acc.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            log[task].append(loss.item())
            step += 1
            if step % 100 == 0 or step == n_steps:
                avg = lambda v: sum(v[-100:]) / max(len(v[-100:]), 1)
                msg = (f"epoch {epoch} step {step}/{n_steps} tool_loss {avg(log['tool']):.4f} "
                       f"read_loss {avg(log['read']):.4f} read_train_acc {avg(log['read_acc']):.3f} "
                       f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f}min")
                print(msg, flush=True)
                history.append(msg)

    model.save_pretrained(out_dir)
    for f in os.listdir(args.base_model):          # tokenizer、chat template 等沿用原模型
        src = os.path.join(args.base_model, f)
        if os.path.isfile(src) and not f.endswith((".safetensors", ".bin")) and f not in ("config.json",):
            shutil.copy2(src, os.path.join(out_dir, f))
    minutes = round((time.time() - t0) / 60, 1)
    with open(os.path.join(out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), n_rows={t: len(v) for t, v in by_task.items()}, n_steps=n_steps,
                       minutes=minutes, history=history), f, ensure_ascii=False, indent=2)
    n = sum(p.numel() for p in AutoModelForCausalLM.from_pretrained(out_dir).parameters())
    print(f"完成：{out_dir}（重新載入成功，參數量 {n / 1e6:.2f}M，訓練 {minutes} 分鐘）")


if __name__ == "__main__":
    main()
