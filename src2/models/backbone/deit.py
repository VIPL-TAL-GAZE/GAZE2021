from functools import partial
import timm


# out 768
deit_base = partial(
    timm.create_model,
    model_name="vit_deit_base_patch16_224",
    num_classes=0
)


# out 384
deit_small = partial(
    timm.create_model,
    model_name="vit_deit_small_patch16_224",
    num_classes=0
)

# out 192
deit_tiny = partial(
    timm.create_model,
    model_name="vit_deit_tiny_patch16_224",
    num_classes=0
)

# model_names = timm.list_models("*deit*")
# print('\n'.join(model_names))

# vit_deit_base_distilled_patch16_224
# vit_deit_base_distilled_patch16_384
# vit_deit_base_patch16_224
# vit_deit_base_patch16_384
# vit_deit_small_distilled_patch16_224
# vit_deit_small_patch16_224
# vit_deit_tiny_distilled_patch16_224
# vit_deit_tiny_patch16_224

# model = timm.create_model('vit_deit_tiny_patch16_224', num_classes=0)
# print(model)