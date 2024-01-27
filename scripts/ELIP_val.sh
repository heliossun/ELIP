python -m torch.distributed.run --nproc_per_node=1 \
    train_retrieval.py \
    --config configs/retrieval_flickr_elip.yaml \
    --output_dir /path/to/output \
    --evaluate