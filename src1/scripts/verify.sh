# verify hrnet_w64_640, eval error 3.706, submit 3.22
python verify.py \
    --data-dir ../datasets/xgaze_224 \
    --backbone hrnet_w64 \
    --mode test \
    --input-size 640 \
    --splits train_eval_test_split.json \
    --batch-size 512 \
    --workers 4 \
    --checkpoint ../checkpoints/hrnet_w64/flip_640/epoch_12.pth.tar


# verify hrnet_w64_768, eval error 3.705, submit 3.37
python verify.py \
    --data-dir ../datasets/xgaze_224 \
    --backbone hrnet_w64 \
    --mode test \
    --input-size 768 \
    --splits train_eval_test_split.json \
    --batch-size 512 \
    --workers 4 \
    --checkpoint ../checkpoints/hrnet_w64/flip_768/epoch_22.pth.tar


# verify hrnet_w64_896, eval error 3.782, submit 3.38
python verify.py \
    --data-dir ../datasets/xgaze_224 \
    --backbone hrnet_w64 \
    --mode test \
    --input-size 896 \
    --splits train_eval_test_split.json \
    --batch-size 512 \
    --workers 4 \
    --checkpoint ../checkpoints/hrnet_w64/flip_896/epoch_29.pth.tar


# verify botnet, eval error 4.47, submit 3.84
python verify.py \
    --data-dir ../datasets/xgaze_224 \
    --backbone botnet \
    --mode test \
    --input-size 224 \
    --splits train_eval_test_split.json \
    --batch-size 256 \
    --workers 4 \
    --checkpoint ../checkpoints/botnet/epoch_26.pth.tar