import tensorflow as tf
from tensorflow.contrib import slim
from builders import frontend_builder
import numpy as np
import os, sys

# Use bilinear interpolation to adjust images to a fixed size
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
# ARM for attention mechanism
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

# build the depth-wise bisenet network
def build_bisenet2(inputs, num_classes, preset_model='DepthwiseBiseNet', frontend="xception", weight_decay=1e-5, is_training=True, pretrained_dir="models"):

    ### The spatial path
    ### The number of feature maps for each convolution is not specified in the paper
    ### It was chosen here to be equal to the number of feature maps of a classification
    ### model at each corresponding stage 

    # depth-wise convolution
    point_filter1 = tf.get_variable(name="point_filter1", shape=(1, 1, 64, 128), initializer=tf.contrib.layers.xavier_initializer())
    point_filter2 = tf.get_variable(name="point_filter2", shape=(1, 1, 128, 256), initializer=tf.contrib.layers.xavier_initializer())
    filter1 = tf.get_variable(name="filter1", shape=(3, 3, 64, 1), initializer=tf.contrib.layers.xavier_initializer())
    filter2 = tf.get_variable(name="filter2", shape=(3, 3, 128, 1), initializer=tf.contrib.layers.xavier_initializer())
        
    spatial_net = ConvBlock(inputs, n_filters=64, kernel_size=[3, 3], strides=2)
    spatial_net = tf.nn.separable_conv2d(input=spatial_net, depthwise_filter=filter1, pointwise_filter=point_filter1, strides=[1,2,2,1], rate=[1,1], padding='SAME')
    spatial_net = tf.nn.separable_conv2d(input=spatial_net, depthwise_filter=filter2, pointwise_filter=point_filter2, strides=[1,2,2,1], rate=[1,1], padding='SAME')
    spatial_net = ConvBlock(spatial_net, n_filters=32, kernel_size=[1, 1])

    ### Context path
    logits, end_points, frontend_scope, init_fn  = frontend_builder.build_frontend(inputs, frontend, pretrained_dir=pretrained_dir, is_training=is_training)
    
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
