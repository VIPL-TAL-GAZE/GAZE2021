from functools import partial
import timm


__all__ = [
    'mobilenetv2_100',
    'mobilenetv2_140',
    'mobilenetv3_large_075',
    'mobilenetv3_large_100',
    'mobilenetv3_rw'
]


mobilenetv2_100 = partial(
    timm.create_model,
    model_name='mobilenetv2_100',
    num_classes=0
)


mobilenetv2_140 = partial(
    timm.create_model,
    model_name='mobilenetv2_140',
    num_classes=0
)


mobilenetv3_large_075 = partial(
    timm.create_model,
    model_name='mobilenetv3_large_075',
    num_classes=0
)


mobilenetv3_large_100 = partial(
    timm.create_model,
    model_name='mobilenetv3_large_100',
    num_classes=0
)


mobilenetv3_rw = partial(
    timm.create_model,
    model_name='mobilenetv3_rw',
    num_classes=0
)