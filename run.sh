#!/usr/bin/env bash
# 重現最終模型（在 code/ 內執行）：bash run.sh
# 流程：起始模型 minimind-3 → 產生自製模板資料 → 組 SFT 資料 → SFT 訓練 → 評估。
# RTX 4070 Ti SUPER 16GB 約 35 分鐘（訓練約 28 分鐘）。
# 可用環境變數覆寫：MM3（minimind-3 位置）、OUT（輸出模型位置）、REGEN=1（刪掉附帶的自製資料，從模板重新產生）
set -euo pipefail
cd "$(dirname "$0")"
MM3=${MM3:-minimind-3}
OUT=${OUT:-../model_reproduced}

echo "== 0. 起始模型與 embedding 模型"
if [ ! -f "$MM3/config.json" ]; then
  git lfs install && git clone https://huggingface.co/jingyaogong/minimind-3 "$MM3"
fi
if [ ! -f embedder/config.json ]; then        # 繳交檔已附上，這步通常會跳過
  hf download BAAI/bge-small-zh-v1.5 --local-dir embedder
fi

echo "== 1. 自製模板資料（A、C 訓練用；B 模擬測試，最終也加入訓練；D 盲測，不訓練）"
if [ "${REGEN:-0}" = 1 ]; then
  rm -rf data/train_extraA data/train_extraC data/pseudo_test data/blind_test
fi
bash make_extra_data.sh
(cd data && md5sum -c MD5SUMS)                # 確認資料與我們訓練時完全相同

echo "== 2. 組 SFT 資料（工具呼叫 + 讀記憶）"
mkdir -p data/sft
python build_sft_data.py --model_dir "$MM3" --out data/sft/sft_final.jsonl \
  --sources data/train data/train_extraA data/pseudo_test data/train_extraC \
  --context_mode chrono --value_aug 0.5 --n_shuffle 2 --seed 42

echo "== 3. SFT 訓練（2 epoch、lr 2e-5、seed 42）"
python train_sft.py --base_model "$MM3" --data data/sft/sft_final.jsonl --out_dir "$OUT" \
  --epochs 2 --lr 2e-5 --tool_bs 16 --read_bs 4 --lm_weight 0.5 --seed 42

echo "== 4. 評估（val 與 D 套盲測）"
python eval_agent.py --model_dir "$OUT" --agent my_agent:MyAgent --data_dir data/val --out results/reproduced_val
python eval_agent.py --model_dir "$OUT" --agent my_agent:MyAgent --data_dir data/blind_test --out results/reproduced_blind
