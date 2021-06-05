python train_ddp.py \
    --data-dir ../datasets/xgaze_224 \
    --backbone botnet \
    --epochs 40 \
    --batch-size 24 \
    --input-size 224 \
    --splits train_eval_test_split.json \
    --use-tensorboard \
    --fresh-per-iter 1 \
    --optimizer adamw \
    --lr 0.0001 \
    --weight-decay 0.05 \
    --scheduler exp \
    --lr-decay-factor 0.1 \
    --ckpt-root ./runs \
    --checkname botnet_224 \
    --loss angular \
    --eval \
    --eval-dist \
    --eval-batch-size 24 \