export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/u/darora1/miniconda3/envs/oat/lib/"

MOSEC_LOG_LEVEL=debug python -m oat.oracles.remote.server --cuda-devices 0 > /dev/null &
sleep 30
python -m oat.experiment.main \
    --gpus 2 \
    --dap-algo DPO \
    --beta 0.1 \
    --preference-oracle remote \
    --remote-rm-url http://0.0.0.0:8000 \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --output_key pythia-1b-reference \
    --sync-params-every 1 \
    --rollout-batch-size-per-device 64 \
    --pi-buffer-maxlen-per-device 64 \
    --train-batch-size-per-device 8 \
    --learn-rm \
    --exp-method EnnBAITS \
    --num_samples 10 \
    --use-wb \
    --wb-run-name 1b_skywork-8b_dpo_sea