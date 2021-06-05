# Code for #1 submission in ETH-XGaze Competition 

[Challenge](https://competitions.codalab.org/competitions/28930)

Official implementation of ETH-XGaze Competition #1 Solution.

## Requirements

- Python == 3.8

  Please refer to requirements.txt.

- [timm](https://github.com/rwightman/pytorch-image-models)

  Pytorch image models in timm are used for our code.

- [apex](https://github.com/NVIDIA/apex)

  A NVIDIA-maintained utilities to streamline mixed precision and distributed training in Pytorch. Apex is used for training HRNet models in our checkpoints.   You can also train our HRNet models without apex by decreasing batch size.

  To install apex:

  > git clone https://www.github.com/nvidia/apex
  >
  > cd apex
  >
  > python setup.py install

## File Structure 

Please download our pretrained models in [baiduyun](https://pan.baidu.com/s/1GLQqDQzvfYP8frG6ahZEYw)，the password is **eo7p**.

Pretrained models in gaze2021/checkpoints/* should should be placed into **checkpoints** folder, and gaze2021/datasets/*  should be placed into **datasets** folder.

The training data xgaze_224 should be put in **datasets** folder .Or change **data_dir** argument in config for the data path.

The final structure would look like:

```
├── datasets			
│   ├── xgaze_224
│   │   ├── train
│   │   │   ├── subject0000.h5
│   │   │   ├── ...
│   │   ├── test
│   │   │   ├── ...
│   ├── xgaze_landmarks
│   ├── train_eval_test_split.json
├── checkpoints
│   │   ├── botnet
│   │   ├── ...
├── src1
├── src2
├── requirements.txt
├── README.md
```

## Train & Verify

Code for training HRNet and BoTNet  is in src1 folder.  To train HRNet as an example:

```
cd src1
./scripts/train_hrnet.sh
```

Code for training ResNeSt and iTracker-MHSA is in src2 folder.  To train ResNeSt as an example:

```
cd src2
./train.sh
```

To verify all results we submitted to the leaderboard, please implement verify.sh in src1 and src2.



