#!/bin/bash

DATA_DIR=fairseq/data-bin/ted_bel_spm8000/eng_bel/
MODEL_DIR=/content/drive/My\ Drive/MNLP/assignment2/ted_bel_spm8000/eng_bel/
mkdir -pv "$MODEL_DIR"

# train the model
CUDA_VISIBLE_DEVICE=xx fairseq-train \
	$DATA_DIR \
	--arch transformer_iwslt_de_en \
	--max-epoch 80 \
        --distributed-world-size 1 \
	--share-all-embeddings \
	--no-epoch-checkpoints \
	--dropout 0.3 \
	--optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	--warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4  \
	--criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	--max-tokens 4500 \
	--update-freq 2 \
	--seed 2 \
  	--save-dir "$MODEL_DIR" \
	--log-interval 100 >> "$MODEL_DIR"/train.log 2>&1

# translate the valid and test set
CUDA_VISIBLE_DEVICE=xx fairseq-generate $DATA_DIR \
          --gen-subset test \
          --path "$MODEL_DIR"/checkpoint_best.pt \
          --batch-size 32 \
	  --tokenizer moses \
          --remove-bpe sentencepiece \
	  --scoring sacrebleu \
          --beam 5   > "$MODEL_DIR"/test_b5.log


CUDA_VISIBLE_DEVICE=xx fairseq-generate $DATA_DIR \
          --gen-subset valid \
          --path "$MODEL_DIR"/checkpoint_best.pt \
          --batch-size 32 \
	  --tokenizer moses \
          --remove-bpe sentencepiece \
	  --scoring sacrebleu \
          --beam 5   > "$MODEL_DIR"/valid_b5.log


