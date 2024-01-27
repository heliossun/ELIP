python -m torch.distributed.run --nproc_per_node=4 \
    train_retrieval.py \
    --config configs/retrieval_coco_finetune-noEV.yaml \
    --output_dir /path/to/output \
    --seed 255