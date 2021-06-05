from functools import partial
import timm


__all__ = [
    'efficientnet_b0',
    # 'efficientnet_b1',
    # 'efficientnet_b1_pruned',
    # 'efficientnet_b2',
    # 'efficientnet_b2_pruned',
    # 'efficientnet_b2a',
    # 'efficientnet_b3',
    # 'efficientnet_b3_pruned',
    # 'efficientnet_b3a',
    # 'efficientnet_b4',
    # 'efficientnet_b5',
    # 'efficientnet_b6',
    # 'efficientnet_b7',
    'efficientnet_b8',
    # 'efficientnet_cc_b0_4e',
    # 'efficientnet_cc_b0_8e',
    # 'efficientnet_cc_b1_8e',
    'efficientnet_el',
    # 'efficientnet_em',
    # 'efficientnet_es',
    # 'efficientnet_l2',
    # 'efficientnet_lite0',
    # 'efficientnet_lite1',
    # 'efficientnet_lite2',
    # 'efficientnet_lite3',
    # 'efficientnet_lite4'
]


# 1280
efficientnet_b0 = partial(
    timm.create_model,
    model_name='efficientnet_b0',
    num_classes=0,
    global_pool='avg'
)


# 2816
efficientnet_b8 = partial(
    timm.create_model,
    model_name='efficientnet_b8',
    num_classes=0,
    global_pool='avg'
)


# 1536
efficientnet_el = partial(
    timm.create_model,
    model_name='efficientnet_el',
    num_classes=0,
    global_pool='avg',
)


if __name__ == '__main__':
    import torch
    model = efficientnet_el(pretrained=False)
    img = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        feat = model(img)

    print(feat.shape)