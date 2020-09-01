import tensorflow as tf
from tensorflow.contrib import slim
from frontends import xception
import os 


def build_frontend(inputs, frontend, is_training=True, pretrained_dir="models"):
    if frontend == 'ResNet50':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_50(inputs, is_training=is_training, scope='resnet_v2_50')
            frontend_scope='resnet_v2_50'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'resnet_v2_50.ckpt'), var_list=slim.get_model_variables('resnet_v2_50'), ignore_missing_vars=True)
    elif frontend == 'ResNet101':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_101(inputs, is_training=is_training, scope='resnet_v2_101')
            frontend_scope='resnet_v2_101'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'resnet_v2_101.ckpt'), var_list=slim.get_model_variables('resnet_v2_101'), ignore_missing_vars=True)
    elif frontend == 'xception':
        with slim.arg_scope(xception.xception_arg_scope()):
            logits, end_points = xception.xception39(inputs, is_training=is_training, scope='xception39')
            frontend_scope='xception39'
            init_fn = None
    else:
        raise ValueError("Unsupported fronetnd model '%s'. This function only supports ResNet50, ResNet101, ResNet152, and MobileNetV2, xception" % (frontend))

    return logits, end_points, frontend_scope, init_fn 