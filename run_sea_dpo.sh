python -m oat.experiment.main \
-   --gpus 2 \
+   --gpus 4 \
    --dap-algo SimPO \
    --beta 2 \
    --reward-oracle pairrm \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --output_key pythia-1b-reference \
    --sync-params-every 1 \
-   --rollout-batch-size-per-device 64 \
-   --pi-buffer-maxlen-per-device 64 \
-   --train-batch-size-per-device 8 \
+   --rollout-batch-size-per-device 32 \
+   --pi-buffer-maxlen-per-device 32 \
+   --train-batch-size-per-device 1 \
+   --learn-rm \
+   --exp-method EnnBAITS \
+   --num_samples 10 \
    --use-wb \
-   --wb-run-name 1b_pairrm_simpo_online
+   --wb-run-name 1b_pairrm_simpo_sea