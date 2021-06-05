from functools import partial
import timm

__all__ = [
    "swin_small_patch4_window7_224",
    "swin_large_patch4_window7_224",
]

swin_small_patch4_window7_224 = partial(
    timm.create_model,
    model_name="swin_small_patch4_window7_224",
    num_classes=0
)

swin_large_patch4_window7_224 = partial(
    timm.create_model,
    model_name="swin_large_patch4_window7_224",
    num_classes=0
)


if __name__ == '__main__':
    model_names = timm.list_models("mobi*")
    print('\n'.join(model_names))
    model = timm.create_model('mobilenetv3_rw', num_classes=0)
    print(model)
# print(timm.__version__)
# models_names = timm.list_models("swin*")
# print('\n'.join(models_names))
#
# model = timm.create_model('swin_small_patch4_window7_224',
#                           num_classes=0, drop_rate=0.5)
#
# print(model)
