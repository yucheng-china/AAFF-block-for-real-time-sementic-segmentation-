import tensorflow as tf
from tensorflow.contrib import slim
from builders import frontend_builder
import numpy as np
import os, sys

# Use bilinear interpolation to adjust images to a fixed size.
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

# ARM to introduce the attention mechanism
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

# feature fusion in the final stage
def FeatureFusionModule(input_1, input_2, input_3, n_filters):
    inputs = tf.concat([input_1, input_2], axis=-1)
    inputs = tf.concat([inputs, input_3], axis=-1)
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

# key block in this project (AAFF)
def AttentionAndFeatureFussion(input_1, input_2, n_filters):
    net = ConvBlock(input_1, n_filters, kernel_size=[3, 3], strides=2)
    
    net_global = tf.reduce_mean(net, [1, 2], keep_dims=True)
    net_global = slim.conv2d(net_global, n_filters, kernel_size=[1, 1])
    net_global = slim.batch_norm(net_global, fused=True)
    net_global = tf.sigmoid(net_global)
    net_attention = tf.multiply(net_global, net)
    
    net = tf.concat([net_attention, input_2], axis=-1)
    net = ConvBlock(net, n_filters, kernel_size=[1, 1], strides=1)
    return net

# build the depth-wise AAFF network
def build_bisenet3(inputs, num_classes, preset_model='DepthwiseAAFF', frontend="xception", weight_decay=1e-5, is_training=True, pretrained_dir="models"):

    initializer = slim.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
    
    ### The spatial path
    ### The number of feature maps for each convolution is not specified in the paper
    ### It was chosen here to be equal to the number of feature maps of a classification
    ### model at each corresponding stage 

    # depth-wise convolution
    point_filter1 = tf.get_variable(name="point_filter1", shape=(1, 1, 64, 128), initializer=initializer)
    point_filter2 = tf.get_variable(name="point_filter2", shape=(1, 1, 128, 256), initializer=initializer)
    filter1 = tf.get_variable(name="filter1", shape=(3, 3, 64, 1), initializer=initializer)
    filter2 = tf.get_variable(name="filter2", shape=(3, 3, 128, 1), initializer=initializer)
    # spatial path
    spatial_net = ConvBlock(inputs, n_filters=64, kernel_size=[3, 3], strides=2)
    spatial_net = tf.nn.separable_conv2d(input=spatial_net, depthwise_filter=filter1, pointwise_filter=point_filter1, strides=[1,2,2,1], rate=[1,1], padding='SAME')
    spatial_net = tf.nn.separable_conv2d(input=spatial_net, depthwise_filter=filter2, pointwise_filter=point_filter2, strides=[1,2,2,1], rate=[1,1], padding='SAME')
    spatial_net = ConvBlock(spatial_net, n_filters=32, kernel_size=[1, 1])

    # Context path
    logits, end_points, frontend_scope, init_fn  = frontend_builder.build_frontend(inputs, frontend, pretrained_dir=pretrained_dir, is_training=is_training)
    
    size = tf.shape(end_points['pool5'])[1:3]

    net_1 = AttentionAndFeatureFussion(end_points['pool3'], end_points['pool4'], 64)
    net_2 = AttentionAndFeatureFussion(net_1, end_points['pool5'], 128)
    net_2 = Upsampling(net_2, scale=2)
    net_1_2 = tf.concat([net_1, net_2], axis=-1)
    net_1_2 = Upsampling(net_1_2, scale=2)
    net_1_2_3 = tf.concat([net_1_2, end_points['pool3']], axis=-1)
    net_1_2_3 = ConvBlock(net_1_2_3, n_filters=128, kernel_size=[1, 1], strides=1)
    context_path_left = AttentionRefinementModule(net_1_2_3, n_filters=128)
    
    net_3 = AttentionAndFeatureFussion(end_points['pool3'], end_points['pool4'], 64)
    net_4 = AttentionAndFeatureFussion(net_3, end_points['pool5'], 128)
    net_4 = Upsampling(net_4, scale=2)
    net_3_4 = tf.concat([net_3, net_4], axis=-1)
    net_3_4 = Upsampling(net_3_4, scale=2)
    net_3_4_5 = tf.concat([net_3_4, end_points['pool3']], axis=-1)
    net_3_4_5 = ConvBlock(net_3_4_5, n_filters=128, kernel_size=[1, 1], strides=1)
    context_path_right = AttentionRefinementModule(net_3_4_5, n_filters=128)
    
    
    ### Combining the paths
    net = FeatureFusionModule(input_1=context_path_left, input_2=context_path_right, input_3=spatial_net, n_filters=256)
    net = ConvBlock(net, n_filters=64, kernel_size=[3, 3])

    ### Final upscaling and finish # Upsampling + dilation or only Upsampling
    net = Upsampling(net, scale=2)
    net = slim.conv2d(net, 64, [3, 3], rate=2, activation_fn=tf.nn.relu, biases_initializer=None,
                                      normalizer_fn=slim.batch_norm)
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
    net = Upsampling(net, 4)

    return net, init_fn