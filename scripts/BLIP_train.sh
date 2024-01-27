python -m torch.distributed.run --nproc_per_node=4 \
    train_BLIP_retrieval.py \
    --config configs/retrieval_blip.yaml \
    --output_dir /path/to/output