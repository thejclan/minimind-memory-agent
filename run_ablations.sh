#!/usr/bin/env bash
# HW2 消融實驗與 TMMLU+ 退化評估（在 ~/hw2 執行，約 1 小時）。
# 已經存在的結果會跳過，中斷後重跑只會補做沒完成的部分。
set -uo pipefail
cd "$(dirname "$0")"
MM3=${MM3:-$HOME/hw1/minimind-3}
HW1=${HW1:-$HOME/hw1}
mkdir -p logs results models

train() {  # train <名稱> [train_sft.py 的額外參數...]
  local name=$1; shift
  [ -d "models/$name" ] && { echo "[skip] models/$name"; return; }
  python -u train_sft.py --base_model "$MM3" --data data/sft/sft.jsonl --out_dir "models/$name" "$@" \
    2>&1 | tee "logs/$name.log" | grep -E "訓練資料|完成"
}

evaluate() {  # evaluate <模型資料夾> <agent> <資料集> <結果名稱>
  local out="results/$4"
  [ -f "$out/summary.json" ] && { echo "[skip] $out"; return; }
  python eval_agent.py --model_dir "$1" --agent "$2" --data_dir "data/$3" --out "$out" \
    > "logs/eval_$4.log" 2>&1
  echo "$4: $(grep -E 'Leaderboard' "logs/eval_$4.log")"
}

tmmlu() {  # tmmlu <模型資料夾> <結果名稱>
  for d in holdout holdout_shuffled; do
    local out="results/tmmlu_$2_$d.json"
    [ -f "$out" ] && { echo "[skip] $out"; continue; }
    python "$HW1/eval_tmmluplus.py" eval --model_path "$1" --data "$HW1/data/val_$d.jsonl" --output "$out" \
      > "logs/tmmlu_$2_$d.log" 2>&1
    echo "tmmlu $2 $d: $(grep -E 'MACRO|micro' "logs/tmmlu_$2_$d.log" | tr '\n' ' ')"
  done
}

echo "== 1. 訓練消融模型"
train sft_tool_only   --tasks tool                       # 只訓練工具呼叫
train sft_read_only   --tasks read                       # 只訓練讀記憶
train sft_public_only --sources train                    # 只用公開資料（不加自製 A 套）

echo "== 2. 評估（val 與模擬測試集）"
for data in val pseudo_test; do
  tag=$([ "$data" = val ] && echo val || echo pseudo)
  # 沒有 SFT：baseline 與我們的 agent
  evaluate "$MM3" agent_base:KeywordRetrievalAgent "$data" "mm3_keyword_$tag"
  evaluate "$MM3" my_agent:MyAgent                 "$data" "mm3_myagent_$tag"
  # 有 SFT：框架關掉（NoMemory / Keyword）vs 我們的 agent
  evaluate models/sft_v1 agent_base:NoMemoryAgent          "$data" "sft_v1_nomemory_$tag"
  evaluate models/sft_v1 agent_base:KeywordRetrievalAgent  "$data" "sft_v1_keyword_$tag"
  evaluate models/sft_v1 my_agent:MyAgent                  "$data" "sft_v1_myagent_$tag"
  # SFT 的消融
  for m in sft_tool_only sft_read_only sft_public_only; do
    evaluate "models/$m" my_agent:MyAgent "$data" "${m}_myagent_$tag"
  done
done

echo "== 3. TMMLU+ 退化評估（HW1 的保留集）"
tmmlu "$MM3"         minimind3
tmmlu models/sft_v1  sft_v1

echo "== 完成"
