# train 224
python train_ddp.py \
    --data-dir ../datasets/xgaze_224 \
    --backbone swin_large_patch4_window7_224 \
    --epochs 50 \
    --batch-size 128 \
    --input-size 224 \
    --splits train_eval_test_split.json \
    --use-tensorboard \
    --fresh-per-iter 1 \
    --optimizer adamw \
    --lr 0.0001 \
    --weight-decay 0.05 \
    --scheduler step \
    --step-size 30 \
    --lr-decay-factor 0.1 \
    --ckpt-root ./runs \
    --checkname ab_flip_224 \
    --loss l1 \
    --amp \
    --eval \
    --eval-dist \
    --eval-batch-size 256 \
    # --is-flip \
    # --debug \
    # --use-warmup \
    # --warmup-epochs 5 \


# train 384
python train_ddp.py \
    --data-dir ../datasets/xgaze_224 \
    --backbone swin_large_patch4_window12_384 \
    --epochs 50 \
    --batch-size 32 \
    --input-size 384 \
    --splits train_eval_test_split.json \
    --use-tensorboard \
    --fresh-per-iter 1 \
    --optimizer adamw \
    --lr 0.000025 \
    --weight-decay 0.05 \
    --scheduler step \
    --step-size 30 \
    --lr-decay-factor 0.1 \
    --ckpt-root ./runs \
    --checkname ab_flip_384 \
    --loss l1 \
    --amp \
    --eval \
    --eval-dist \
    --eval-batch-size 64 \