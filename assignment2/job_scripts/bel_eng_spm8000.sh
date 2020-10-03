#!/bin/bash

DATA_DIR=fairseq/data-bin/ted_bel_spm8000/bel_eng/
MODEL_DIR1=/content/drive/My\ Drive/MNLP/assignment2/ted_bel_spm8000/bel_eng/
mkdir -pv "$MODEL_DIR1"

# change the cuda_visible_device to the GPU device number you are using
# train the model
CUDA_VISIBLE_DEVICE=0 fairseq-train \
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
  	--save-dir "$MODEL_DIR1" \
	--log-interval 100 >> "$MODEL_DIR1"/train.log 2>&1

# translate the valid and test set
CUDA_VISIBLE_DEVICE=0  fairseq-generate $DATA_DIR \
          --gen-subset test \
          --path "$MODEL_DIR1"/checkpoint_best.pt \
          --batch-size 32 \
	  --tokenizer moses \
          --remove-bpe sentencepiece \
	  --scoring sacrebleu \
          --beam 5   > "$MODEL_DIR1"/test_b5.log


CUDA_VISIBLE_DEVICE=0 fairseq-generate $DATA_DIR \
          --gen-subset valid \
          --path "$MODEL_DIR1"/checkpoint_best.pt \
          --batch-size 32 \
	  --tokenizer moses \
          --remove-bpe sentencepiece \
	  --scoring sacrebleu \
          --beam 5   > "$MODEL_DIR1"/valid_b5.log


