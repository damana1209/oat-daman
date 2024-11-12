export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/u/darora1/miniconda3/envs/oat/lib/"

python -m oat.experiment.main \
    --gpus 2 \
    --collocate \
    --dap-algo DPO \
    --beta 2 \
    --reward-oracle pairrm \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --output_key pythia-1b-reference \
    --sync-params-every 1 \
    --rollout-batch-size-per-device 2 \
    --exp-method EnnBAITS \
    --pi-buffer-maxlen-per-device 64 \
    --train-batch-size-per-device 8 \
    --num-samples 10 \
    --max-eval 10 \
    --best-of-n-exploration \
    --use-wb \
    --max-train 10 \
    --wb-run-name 1b_pairrm_dpo_online


# rollout-batch-size-per-device should be 64
# best-of-n-exploration compares BoN and best_running_response