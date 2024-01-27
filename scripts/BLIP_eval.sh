python -m torch.distributed.run --nproc_per_node=2 \
    train_BLIP_retrieval.py \
    --config ./configs/blip_retrieval-eval.yaml \
    --output_dir /path/to/output \
    --evaluate