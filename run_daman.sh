export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/u/darora1/miniconda3/envs/oat/lib/"

# python -m oat.experiment.main \
#     --gpus 2 \
#     --dap-algo DPO \
#     --beta 2 \
#     --reward-oracle pairrm \
#     --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
#     --prompt-data lkevinzc/tldr-with-sft-reference \
#     --output_key pythia-1b-reference \
#     --sync-params-every 1 \
#     --rollout-batch-size 10 \
#     --rollout-batch-size-per-device 2 \
#     --exp-method EnnBAITS \
#     --pi-buffer-maxlen-per-device 64 \
#     --train-batch-size-per-device 2 \
#     --num-samples 10 \
#     --max-eval 100 \
#     --best-of-n-exploration \
#     --use-wb \
#     --max-train 512 \
#     --wb-run-name 1b_pairrm_dpo_online \
#     --temperature 1


python -m oat.experiment.main \
    --gpus 2 \
    --dap-algo DPO \
    --beta 2 \
    --reward-oracle pairrm \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --output_key pythia-1b-reference \
    --sync-params-every 1 \
    --rollout-batch-size-per-device 64 \
    --exp-method EnnBAITS \
    --pi-buffer-maxlen-per-device 64 \
    --train-batch-size-per-device 2 \
    --num-samples 10 \
    --best-of-n-exploration \
    --use-wb \
    --wb-run-name 1b_pairrm_dpo_daman_online \
    --temperature 1