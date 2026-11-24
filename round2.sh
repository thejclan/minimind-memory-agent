#!/usr/bin/env bash
# 第二輪改進：比較三種記憶文字組法（fill / fill_hint / chrono），每種各訓練一個模型。
# 在 ~/hw2 執行，約 1.5～2 小時。已經存在的結果會跳過，中斷後重跑只會補做沒完成的部分。
# 只用 val 與模擬測試集（B 套）比較；盲測集（D 套）留到最後才看。
set -uo pipefail
cd "$(dirname "$0")"
MM3=${MM3:-$HOME/hw1/minimind-3}
MODES=${MODES:-"fill chrono fill_hint"}
mkdir -p logs results models data/sft

echo "== 0. 產生 C 套訓練資料與 D 套盲測資料 $(date +%H:%M)"
bash make_extra_data.sh
md5sum data/train_extraC/*.jsonl data/blind_test/*.jsonl

evaluate() {  # evaluate <模型資料夾> <agent> <context_mode> <資料集> <結果名稱>
  local out="results/$5"
  [ -f "$out/summary.json" ] && { echo "[skip] $out"; return; }
  HW2_CONTEXT_MODE=$3 python eval_agent.py --model_dir "$1" --agent "$2" --data_dir "data/$4" --out "$out" \
    > "logs/eval_$5.log" 2>&1
  echo "$5: $(grep -E 'Leaderboard' "logs/eval_$5.log")"
}

echo "== 1. 舊模型 sft_v1 + 新規則（retrieve 組法，只差工具參數修正與房租規則） $(date +%H:%M)"
evaluate models/sft_v1 my_agent:MyAgent retrieve pseudo_test sft_v1_rules3_pseudo

for mode in $MODES; do
  name=sft_r2_$mode
  echo "== 2. [$mode] 產生 SFT 資料 $(date +%H:%M)"
  data=data/sft/sft_r2_$mode.jsonl
  [ -f "$data" ] || python build_sft_data.py --model_dir "$MM3" --out "$data" \
      --sources data/train data/train_extraA data/train_extraC \
      --context_mode "$mode" --value_aug 0.5 2>&1 | grep -v Warning | tee "logs/build_$name.log"
  echo "== 3. [$mode] 訓練 $(date +%H:%M)"
  if [ ! -d "models/$name" ]; then
    python -u train_sft.py --base_model "$MM3" --data "$data" --out_dir "models/$name" \
      2>&1 | tee "logs/$name.log" | grep -E "訓練資料|完成"
  fi
  echo "== 4. [$mode] 評估 $(date +%H:%M)"
  evaluate "models/$name" my_agent:MyAgent "$mode" pseudo_test "${name}_pseudo"
  evaluate "models/$name" my_agent:MyAgent "$mode" val "${name}_val"
done
echo "== 完成 $(date +%H:%M)"
