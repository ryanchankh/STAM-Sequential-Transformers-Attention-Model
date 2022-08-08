python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
        --other_lr=1e-6 \
        --critic_lr=1e-3 \
        --seed=0 \
        --batch_size=1024 \
        --batch_size_val=625 \
        --maxT=21 \
        --stepT=3 \
        --num_workers=8 \
        --model_name='STAM_deit_small_patch16_224' \
        --teacher_name='DeiT_distill_small_patch16_224' \
        --pin_mem=True \
        --smoothing=False \
        --drop=0.0 \
        --drop_path=0.1 \
        --checkpoint_epoch=None \
        --output_dir='imagenet' \
        --epochs=200 \
        --dataset='imagenet' \
        --clip_grad=None \
        --training_mode=True \
        --world_size=4 \
        --distributed=True \
        --pretrained=True \
        --mixup=False \
        --erasing=False \
        --input_size=224 \
        --min_lr=1e-6 \
        --weight_decay=0.05 \
        --dist_eval=False \
        --sync_bn=True \
        --loc_tau=4 \
        --mlp_layers=4 \
        --mlp_hidden_dim=2048 \
        --dist_temp=1 \
        --dist_type=None \
