import sys, os
import tensorflow as tf
import subprocess

sys.path.append("models")
from models.BiSeNet import build_bisenet
from models.DepthwiseBiseNet import build_bisenet2
from models.DepthwiseAAFF import build_bisenet3
from models.DepthwiseAAFF2 import build_bisenet4

SUPPORTED_MODELS = ["BiSeNet", "DepthwiseBiseNet", "DepthwiseAAFF", "DepthwiseAAFF2"]

SUPPORTED_FRONTENDS = ["ResNet50", "ResNet101", "xception"]

def download_checkpoints(model_name):
    subprocess.check_output(["python", "utils/get_pretrained_checkpoints.py", "--model=" + model_name])


def build_model(model_name, net_input, num_classes, crop_width, crop_height, frontend="xception", is_training=True):
	# Get the selected model. 
	# Some of them require pre-trained ResNet

	print("Preparing the model ...")

	if model_name not in SUPPORTED_MODELS:
		raise ValueError("The model you selected is not supported. The following models are currently supported: {0}".format(SUPPORTED_MODELS))

	if frontend not in SUPPORTED_FRONTENDS:
		raise ValueError("The frontend you selected is not supported. The following models are currently supported: {0}".format(SUPPORTED_FRONTENDS))
    # download the frontend model by 'download_checkpoints'
	if "ResNet50" == frontend and not os.path.isfile("models/resnet_v2_50.ckpt"):
	    download_checkpoints("ResNet50")
	if "ResNet101" == frontend and not os.path.isfile("models/resnet_v2_101.ckpt"):
	    download_checkpoints("ResNet101")
	if "xception" == frontend and not os.path.isfile("models/resnet_v2_50.ckpt"):
	    download_checkpoints("xception")       

	network = None
	init_fn = None
   # BiSeNet requires pre-trained ResNet weights
	if model_name == "BiSeNet":
		network, init_fn = build_bisenet(net_input, preset_model = model_name, frontend=frontend, num_classes=num_classes, is_training=is_training)
	elif model_name == "DepthwiseBiseNet":
		network, init_fn = build_bisenet2(net_input, preset_model = model_name, frontend=frontend, num_classes=num_classes, is_training=is_training)
	elif model_name == "DepthwiseAAFF":
		network, init_fn = build_bisenet3(net_input, preset_model = model_name, frontend=frontend, num_classes=num_classes, is_training=is_training)
	elif model_name == "DepthwiseAAFF2":
		network, init_fn = build_bisenet4(net_input, preset_model = model_name, frontend=frontend, num_classes=num_classes, is_training=is_training)
	else:
	    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

	return network, init_fn