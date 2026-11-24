#!/usr/bin/env bash
# 第三輪：用第二輪最好的 chrono 組法，訓練資料再加入 B 套，訓練最終模型；
# 再用從沒看過的 D 套盲測，比較 sft_v1（第一版）、sft_r2_chrono（不含 B）與最終模型。
# 在 ~/hw2 執行，約 50 分鐘。已經存在的結果會跳過。
set -uo pipefail
cd "$(dirname "$0")"
MM3=${MM3:-$HOME/hw1/minimind-3}
HW1=${HW1:-$HOME/hw1}
mkdir -p logs results models data/sft

evaluate() {  # evaluate <模型資料夾> <agent> <context_mode> <資料集> <結果名稱>
  local out="results/$5"
  [ -f "$out/summary.json" ] && { echo "[skip] $out"; return; }
  HW2_CONTEXT_MODE=$3 python eval_agent.py --model_dir "$1" --agent "$2" --data_dir "data/$4" --out "$out" \
    > "logs/eval_$5.log" 2>&1
  echo "$5: $(grep -E 'Leaderboard' "logs/eval_$5.log")"
  grep -E '^  (single|update|multi|abstain|distractor|tool) |^Tool:' "logs/eval_$5.log"
}

data=data/sft/sft_final.jsonl
echo "== 1. 產生 SFT 資料（train + A + B + C） $(date +%H:%M)"
[ -f "$data" ] || python build_sft_data.py --model_dir "$MM3" --out "$data" \
    --sources data/train data/train_extraA data/pseudo_test data/train_extraC \
    --context_mode chrono --value_aug 0.5 2>&1 | grep -v Warning | tee logs/build_sft_final.log

echo "== 2. 訓練最終模型 $(date +%H:%M)"
[ -d models/sft_final ] || python -u train_sft.py --base_model "$MM3" --data "$data" --out_dir models/sft_final \
    2>&1 | tee logs/sft_final.log | grep -E "訓練資料|完成"

echo "== 3. D 套盲測 $(date +%H:%M)"
evaluate models/sft_v1        my_agent_v2:MyAgent retrieve blind_test sft_v1_myagent_blind
evaluate models/sft_r2_chrono my_agent:MyAgent    chrono   blind_test sft_r2_chrono_blind
evaluate models/sft_final     my_agent:MyAgent    chrono   blind_test sft_final_blind

echo "== 4. 最終模型的 val 與 TMMLU+ 退化檢查 $(date +%H:%M)"
evaluate models/sft_final my_agent:MyAgent chrono val sft_final_val
for d in holdout holdout_shuffled; do
  out="results/tmmlu_sft_final_$d.json"
  [ -f "$out" ] && { echo "[skip] $out"; continue; }
  python "$HW1/eval_tmmluplus.py" eval --model_path models/sft_final --data "$HW1/data/val_$d.jsonl" --output "$out" \
    > "logs/tmmlu_sft_final_$d.log" 2>&1
  echo "tmmlu sft_final $d: $(grep -E 'MACRO|micro' "logs/tmmlu_sft_final_$d.log" | tr '\n' ' ')"
done
echo "== 完成 $(date +%H:%M)"
