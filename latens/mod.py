import tensorflow as tf
from tensorflow.keras import layers
from shutil import rmtree
from latens.utils import dat
import os
import numpy as np
import logging

logger = logging.getLogger('latens')

class ConvAutoEncoder(tf.keras.Model):
  def __init__(self, image_shape, num_components,
               level_filters=[32,32,16],
               level_depth=1,
               dense_nodes=[1024],
               l2_reg=None,
               rep_activation=tf.nn.sigmoid,
               dropout=0.1,
               **kwargs):
    """Create a convolutional autoencoder similar to the UNet architecture.

    :param level_filters: number of filters to use at each level. Default is
    [16, 32].
    :param level_depth: how many convolutional layers to pass the
    image through at each level. Default is 3.
    :param dense_nodes: number of nodes in fully
    :param rep_activation: activation function to use for the representational
    layer. Default is sigmoid. 
    :param dropout: dropout rate to use after pooling and deconv layers. Default
    is 0.1. 0 specifies no dropout.
    """
    super().__init__(name='conv_auto_encoder')
    self.image_shape = tuple(np.atleast_3d(image_shape))
    self.num_components = num_components
    
    self.level_filters = level_filters
    self.level_depth = level_depth
    self.dense_nodes = dense_nodes
    self._rep_activation = rep_activation
    self._dropout_rate = dropout

    if l2_reg is None:
      self.regularizer = None
    else:
      self.regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg)     


    scale_factor = 2**(len(level_filters) - 1)
    assert(image_shape[0] % scale_factor == 0
           and image_shape[1] % scale_factor == 0)
    self._unflat_shape = (image_shape[0] // scale_factor,
                          image_shape[1] // scale_factor,
                          level_filters[-1])

    self.encoding_layers = self.create_encoding_layers()
    self.decoding_layers = self.create_decoding_layers()

    # self.flatten_l = layers.Flatten(input_shape=self.image_shape)
    # self.hidden_l = layers.Dense(1000, activation=tf.nn.sigmoid)
    # self.output_l = layers.Dense(784, activation=tf.nn.sigmoid)
    # self.reshape_l = layers.Reshape((28,28,1))

  def call(self, inputs, training=False):
    embedding = self.encode(inputs, training=training)
    return self.decode(embedding, training=training)
    # flat = self.flatten_l(inputs)
    # # hidden = self.hidden_l(flat)
    # outputs = self.output_l(flat)
    # image = self.reshape_l(outputs)
    # return image

  def encode(self, inputs, training=False):
    for i, layer in enumerate(self.encoding_layers):
      input_shape = inputs.shape
      inputs = layer(inputs)
      output_shape = inputs.shape
      logger.debug(f"{input_shape} -> {output_shape}:{layer.name}")
    return inputs

  def decode(self, inputs, training=False):
    for layer in self.decoding_layers:
      input_shape = inputs.shape
      inputs = layer(inputs)
      output_shape = inputs.shape
      logger.debug(f"{input_shape} -> {output_shape}:{layer.name}")
    return inputs

  def create_encoding_layers(self):
    encoding_layers = []

    for i, filters in enumerate(self.level_filters):
      if i > 0:
        encoding_layers += self.maxpool()
      for _ in range(self.level_depth):
        input_shape = None if len(encoding_layers) == 0 else self.image_shape
        encoding_layers += self.conv(filters, input_shape=input_shape)
    
    encoding_layers.append(layers.Flatten())
      
    for nodes in self.dense_nodes:
      encoding_layers += self.dense(nodes)

    encoding_layers += self.dense(
      self.num_components,
      activation=self._rep_activation)

    return encoding_layers

  def create_decoding_layers(self):
    decoding_layers = []
    for nodes in reversed(self.dense_nodes):
      decoding_layers += self.dense(nodes)

    decoding_layers += self.dense(np.product(self._unflat_shape))
    decoding_layers.append(layers.Reshape(self._unflat_shape))

    for i, filters in enumerate(reversed(self.level_filters)):
      if i > 0:
        decoding_layers += self.conv_transpose(filters)
      for _ in range(self.level_depth):
        decoding_layers += self.conv(filters)

    decoding_layers.append(layers.Conv2D(
      1, [1,1],
      activation=tf.nn.sigmoid,
      kernel_regularizer=self.regularizer))

    return decoding_layers

  def conv(self, filters, input_shape=None, activation=tf.nn.relu):
    conv_layers = []
    if input_shape is None:
      conv_layers.append(layers.Conv2D(
        filters, [3,3],
        activation=activation,
        padding='same',
        kernel_regularizer=self.regularizer))
    else:
      conv_layers.append(layers.Conv2D(
        filters, [3,3],
        activation=activation,
        padding='same',
        kernel_regularizer=self.regularizer,
        input_shape=input_shape))
    conv_layers.append(layers.BatchNormalization())
    return conv_layers

  def maxpool(self):
    pool_layers = []
    pool_layers.append(layers.MaxPool2D())
    # pool_layers.append(layers.Dropout(self._dropout_rate))
    return pool_layers
    
  def dense(self, nodes, activation=tf.nn.relu):
    return [layers.Dense(
      nodes,
      activation=activation,
      kernel_regularizer=self.regularizer)]

  def conv_transpose(self, filters, activation=tf.nn.relu):
    deconv_layers = []
    deconv_layers.append(layers.Conv2DTranspose(
      filters, [2,2],
      strides=[2,2],
      padding='same',
      activation=activation,
      kernel_regularizer=self.regularizer))
    # deconv_layers.append(layers.Dropout(self._dropout_rate))
    return deconv_layers
