#!/usr/bin/env bash

# Evaluate the final model on zero-shot Hellaswag
python eval/lm_eval_harness.py \
  --checkpoint_dir out/tinyllama/final \
  --eval_tasks "[hellaswag]" \
  --precision 16-true \
  --num_fewshot 0 \
  --save_filepath "tinyllama_results.json"