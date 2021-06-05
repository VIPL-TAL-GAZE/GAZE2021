# train 224
python train_ddp.py \
    --data-dir ../datasets/xgaze_224 \
    --backbone hrnet_w64 \
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
    --is-flip \
    # --debug \
    # --use-warmup \
    # --warmup-epochs 5 \


# train 448
python train_ddp.py \
    --data-dir ../datasets/xgaze_224 \
    --backbone hrnet_w64 \
    --epochs 50 \
    --batch-size 32 \
    --input-size 448 \
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
    --checkname ab_flip_448 \
    --loss l1 \
    --amp \
    --eval \
    --eval-dist \
    --eval-batch-size 64 \
    --is-flip \
    # --debug \
    # --use-warmup \
    # --warmup-epochs 5 \


# train 640
python train_ddp.py \
    --data-dir ../datasets/xgaze_224 \
    --backbone hrnet_w64 \
    --epochs 30 \
    --batch-size 24 \
    --input-size 640 \
    --splits train_eval_test_split.json \
    --use-tensorboard \
    --fresh-per-iter 1 \
    --optimizer adamw \
    --lr 0.000025 \
    --weight-decay 0.05 \
    --scheduler step \
    --step-size 20 \
    --lr-decay-factor 0.1 \
    --ckpt-root ./runs \
    --checkname ab_flip_640 \
    --loss l1 \
    --amp \
    --eval \
    --eval-dist \
    --eval-batch-size 48 \
    --is-flip \
    # --debug \
    # --use-warmup \
    # --warmup-epochs 5 \


python train_ddp.py \
    --data-dir ../datasets/xgaze_224 \
    --backbone hrnet_w64 \
    --epochs 30 \
    --batch-size 16 \
    --input-size 768 \
    --splits train_eval_test_split.json \
    --use-tensorboard \
    --fresh-per-iter 1 \
    --optimizer adamw \
    --lr 0.000025 \
    --weight-decay 0.05 \
    --scheduler step \
    --step-size 20 \
    --lr-decay-factor 0.1 \
    --ckpt-root ./runs \
    --checkname ab_flip_768 \
    --loss l1 \
    --amp \
    --eval \
    --eval-dist \
    --eval-batch-size 32 \
    --is-flip \


python train_ddp.py \
    --data-dir ../datasets/xgaze_224 \
    --backbone hrnet_w64 \
    --epochs 30 \
    --batch-size 8 \
    --input-size 896 \
    --splits train_eval_test_split.json \
    --use-tensorboard \
    --fresh-per-iter 1 \
    --optimizer adamw \
    --lr 0.000025 \
    --weight-decay 0.05 \
    --scheduler step \
    --step-size 20 \
    --lr-decay-factor 0.1 \
    --ckpt-root ./runs \
    --checkname ab_flip_896 \
    --loss l1 \
    --amp \
    --eval \
    --eval-dist \
    --eval-batch-size 16 \
    --is-flip \