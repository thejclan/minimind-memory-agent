#!/usr/bin/env bash
# run.sh 跑完後執行（在 code/ 內）：報告用的消融實驗、TMMLU+ 退化檢查、繳交前的檢查。約 20 分鐘。
# 結果放在 ../logs/eval/，摘要寫在 ../logs/post_eval.log
set -uo pipefail
cd "$(dirname "$0")"
MM3=${MM3:-$HOME/hw1/minimind-3}
HW1=${HW1:-$HOME/hw1}
OUT=../logs/eval
mkdir -p "$OUT"

evaluate() {  # evaluate <模型> <agent> <資料集> <名稱>
  local out="$OUT/$4"
  [ -f "$out/summary.json" ] && { echo "[skip] $4"; return; }
  python eval_agent.py --model_dir "$1" --agent "$2" --data_dir "data/$3" --out "$out" > "$OUT/$4.log" 2>&1
  echo "$4: $(grep -E 'Leaderboard' "$OUT/$4.log")"
  grep -E '^  (single|update|multi|abstain|distractor|tool) |^Tool:|^Time' "$OUT/$4.log"
}

echo "== 1. 最終模型 + 最終 agent（run.sh 已跑過，搬到 logs）"
for d in val blind; do
  [ -d "results/reproduced_$d" ] && [ ! -d "$OUT/final_myagent_$d" ] && mv "results/reproduced_$d" "$OUT/final_myagent_$d"
done
rmdir results 2>/dev/null || true
for d in val blind_test; do
  n=final_myagent_${d%_test}
  evaluate ../model my_agent:MyAgent "$d" "$n"
done

echo "== 2. 消融：沒有 SFT（minimind-3 + 最終 agent）"
evaluate "$MM3" my_agent:MyAgent val        mm3_myagent_val
evaluate "$MM3" my_agent:MyAgent blind_test mm3_myagent_blind

echo "== 3. 消融：最終模型 + 參考 agent"
for a in NoMemoryAgent RecentContextAgent KeywordRetrievalAgent; do
  evaluate ../model agent_base:$a val        final_${a}_val
  evaluate ../model agent_base:$a blind_test final_${a}_blind
done

echo "== 4. baseline：minimind-3 + 參考 agent（D 套；val 的結果在 Task 1 已跑過）"
for a in NoMemoryAgent RecentContextAgent KeywordRetrievalAgent; do
  evaluate "$MM3" agent_base:$a blind_test mm3_${a}_blind
done

echo "== 5. TMMLU+ 退化檢查"
for d in holdout holdout_shuffled; do
  out="$OUT/tmmlu_final_$d.json"
  [ -f "$out" ] || python "$HW1/eval_tmmluplus.py" eval --model_path ../model --data "$HW1/data/val_$d.jsonl" \
      --output "$out" > "$OUT/tmmlu_final_$d.log" 2>&1
  echo "tmmlu final $d: $(grep -E 'MACRO|micro' "$OUT/tmmlu_final_$d.log" | tr '\n' ' ')"
done

echo "== 6. 繳交前檢查"
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("../model", trust_remote_code=True)
n = sum(p.numel() for p in m.parameters())
print(f"model 載入成功，參數量 {n/1e6:.2f}M（上限 128M）：{'OK' if n <= 128e6 else 'FAIL'}")
from transformers import AutoModel
emb = AutoModel.from_pretrained("embedder")
k = sum(p.numel() for p in emb.parameters())
print(f"embedder 參數量 {k/1e6:.2f}M（上限 30M）：{'OK' if k <= 30e6 else 'FAIL'}")
PY
# 助教原版（hw2_student_release.zip）的 md5
declare -A ORIG=([eval_agent.py]=f5afbbf9c2e379b7c3176ce3393f681a [agent_base.py]=f30c36594bbc89b26f9f75bb403458f4 [tool_spec.json]=e27ad548d0baf2e076323770153717ef)
for f in "${!ORIG[@]}"; do
  [ "$(md5sum "$f" | cut -d' ' -f1)" = "${ORIG[$f]}" ] && echo "$f 與助教原版相同：OK" || echo "$f 與助教原版不同：FAIL"
done
echo "== 完成 $(date +%H:%M)"
