import tensorflow as tf
from tensorflow.contrib import slim
from builders import frontend_builder
import numpy as np
import os, sys

def Upsampling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale, tf.shape(inputs)[2]*scale])

def ConvBlock(inputs, n_filters, kernel_size=[3, 3], strides=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = slim.conv2d(inputs, n_filters, kernel_size, stride=[strides, strides], activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net

# introduce the attention mechanism
def AttentionRefinementModule(inputs, n_filters):
    inputs = slim.conv2d(inputs, n_filters, [3, 3], activation_fn=None)
    inputs = tf.nn.relu(slim.batch_norm(inputs, fused=True))

    # Global average pooling 
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = slim.batch_norm(net, fused=True)
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)
    return net

# fusion all the paths in the final stage
def FeatureFusionModule(input_1, input_2, n_filters):
    inputs = tf.concat([input_1, input_2], axis=-1)
    inputs = ConvBlock(inputs, n_filters=n_filters, kernel_size=[3, 3])

    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    net = tf.add(inputs, net)

    return net

# build the bisenet network, end_points is the output features from xception
def build_bisenet(inputs, num_classes, preset_model='BiSeNet', frontend="xception", weight_decay=1e-5, is_training=True, pretrained_dir="models"):
    """
    Builds the BiSeNet model. 

    Arguments:
      inputs: The input tensor=
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes

    Returns:
      BiSeNet model
    """

    ### The spatial path
    ### The number of feature maps for each convolution is not specified in the paper
    ### It was chosen here to be equal to the number of feature maps of a classification
    ### model at each corresponding stage 
    spatial_net = ConvBlock(inputs, n_filters=64, kernel_size=[3, 3], strides=2)
    spatial_net = ConvBlock(spatial_net, n_filters=128, kernel_size=[3, 3], strides=2)
    spatial_net = ConvBlock(spatial_net, n_filters=256, kernel_size=[3, 3], strides=2)


    ### Context path
    logits, end_points, frontend_scope, init_fn  = frontend_builder.build_frontend(inputs, frontend, pretrained_dir=pretrained_dir, is_training=is_training)
    # global pool in order to get highest receptive field
    size = tf.shape(end_points['pool5'])[1:3]
    global_channels = tf.reduce_mean(end_points['pool5'], [1, 2], keep_dims=True)
    global_channels = slim.conv2d(global_channels, 128, 1, [1, 1], activation_fn=None)
    global_channels = tf.nn.relu(slim.batch_norm(global_channels, fused=True))
    global_channels = tf.image.resize_bilinear(global_channels, size=size)

    net_5 = AttentionRefinementModule(end_points['pool5'], n_filters=128)

    net_5_scaled = tf.add(global_channels, net_5)
    net_5 = Upsampling(net_5, scale=2)
    net_5 = ConvBlock(net_5, n_filters=128, kernel_size=[3, 3]) 
    
    net_4 = AttentionRefinementModule(end_points['pool4'], n_filters=128)
    net_4 = tf.add(net_4, net_5)
    net_4 = Upsampling(net_4, scale=2)
    net_4 = ConvBlock(net_4, n_filters=128, kernel_size=[3, 3])
    

    context_net = net_4
    
    ### Combining the paths
    net = FeatureFusionModule(input_1=spatial_net, input_2=context_net, n_filters=256)
    net = ConvBlock(net, n_filters=64, kernel_size=[3, 3])

    ### Final upscaling and finish
    net = Upsampling(net, scale=2)
    net = slim.conv2d(net, 64, [3, 3], rate=2, activation_fn=tf.nn.relu, biases_initializer=None,
                                      normalizer_fn=slim.batch_norm)
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
    net = Upsampling(net, 4)
    

    return net, init_fn

