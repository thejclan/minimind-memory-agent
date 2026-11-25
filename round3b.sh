#!/usr/bin/env bash
# 補跑 sft_v1 的 D 套盲測，並用修正後的規則（計算題、工具參數）重新評估 sft_final。約 15 分鐘。
set -uo pipefail
cd "$(dirname "$0")"
evaluate() {  # evaluate <模型資料夾> <agent> <context_mode> <資料集> <結果名稱>
  local out="results/$5"
  [ -f "$out/summary.json" ] && { echo "[skip] $out"; return; }
  HW2_CONTEXT_MODE=$3 python eval_agent.py --model_dir "$1" --agent "$2" --data_dir "data/$4" --out "$out" \
    > "logs/eval_$5.log" 2>&1
  echo "$5: $(grep -E 'Leaderboard' "logs/eval_$5.log")"
  grep -E '^  (single|update|multi|abstain|distractor|tool) |^Tool:|^Time' "logs/eval_$5.log"
}
evaluate models/sft_v1    my_agent_v2:MyAgent retrieve blind_test  sft_v1_myagent_blind
evaluate models/sft_final my_agent:MyAgent    chrono   blind_test  sft_final_rules4_blind
evaluate models/sft_final my_agent:MyAgent    chrono   pseudo_test sft_final_rules4_pseudo
evaluate models/sft_final my_agent:MyAgent    chrono   val         sft_final_rules4_val
echo "== 完成 $(date +%H:%M)"
