# submit 3.34
python ./verify.py \
    --data_dir ../datasets/xgaze_224 \
    --batch-size 128 \
    --checkpoint ../checkpoints/resnest269e_448/best_net.pth.tar

# submit 3.42
python verify.py \
    --data_dir ../datasets/xgaze_224 \
    --batch-size 128 \
    --checkpoint ../checkpoints/itrackerMHA448/epoch_12.pth.tar
