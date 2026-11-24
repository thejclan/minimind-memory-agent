#!/usr/bin/env bash
# 用自製模板（templates/extra_*.py）搭配助教的 gen_hw2_data.py 產生額外資料。
# gen_hw2_data.py 的 --split test 會讀取名為 templates_heldout 的模組，
# 這裡把自製模板複製成該名稱，放在暫存資料夾並加入 PYTHONPATH。已存在的資料夾會跳過。
set -euo pipefail
cd "$(dirname "$0")"
gen() {  # gen <模板檔> <seed> <人物數> <工具題數> <輸出資料夾>
  [ -f "$5/memory.jsonl" ] && { echo "[skip] $5"; return; }
  local tmp; tmp=$(mktemp -d)
  cp "$1" "$tmp/templates_heldout.py"
  PYTHONPATH="$tmp" python gen_hw2_data.py --split test --seed "$2" --n_personas "$3" --n_tool "$4" --out "$5"
  rm -rf "$tmp"
}
gen templates/extra_A.py 11 300 3000 data/train_extraA   # SFT 訓練資料（A 套）
gen templates/extra_C.py 31 300 3000 data/train_extraC   # SFT 訓練資料（C 套，第二輪加入）
gen templates/extra_B.py 21 100 300  data/pseudo_test    # 模擬測試集（B 套，用來比較與挑選版本）
gen templates/extra_D.py 41 100 300  data/blind_test     # 盲測集（D 套，只在最後看一次）
