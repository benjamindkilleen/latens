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
               level_filters=[16,32,64],
               level_depth=2,
               dense_nodes=[1024,1024],
               l2_reg=None,
               activation=tf.nn.sigmoid,
               dropout=0.1,
               **kwargs):
    """Create a convolutional autoencoder similar to the UNet architecture.

    :param level_filters: number of filters to use at each level. Default is
    [16, 32].
    :param level_depth: how many convolutional layers to pass the
    image through at each level. Default is 3.
    :param dense_nodes: number of nodes in fully
    :param activation: activation function to use for the representational
    layer. Default is sigmoid. 
    :param dropout: dropout rate to use after the representation layer. Default
    is 0.1. 0 specifies no dropout.
    """
    super().__init__(**kwargs)
    self.image_shape = tuple(np.atleast_3d(image_shape))
    self.num_components = num_components
    
    self.level_filters = level_filters
    self.level_depth = level_depth
    self.dense_nodes = dense_nodes
    self.activation = activation

    if l2_reg is None:
      self.regularizer = None
    else:
      self.regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg)     


    scale_factor = 2**(len(level_filters) - 1)
    self._unflat_shape = (image_shape[0] // scale_factor,
                          image_shape[1] // scale_factor,
                          level_filters[-1])
    logger.debug(f"unflat_shape: {self._unflat_shape}")

    self.encoding_layers = self.create_encoding_layers()
    self.dropout = layers.Dropout(dropout)
    self.decoding_layers = self.create_decoding_layers()

  def create_encoding_layers(self):
    encoding_layers = []

    for i, filters in enumerate(self.level_filters):
      if i > 0:
        encoding_layers.append(layers.MaxPool2D())
      for _ in range(self.level_depth):
        if len(encoding_layers) == 0:
          encoding_layers.append(layers.Conv2D(
            filters, [3,3], activation=tf.nn.relu,
            padding='same',
            kernel_regularizer=self.regularizer,
            input_shape=self.image_shape))
        else:
          encoding_layers.append(layers.Conv2D(
            filters, [3,3], activation=tf.nn.relu,
            padding='same',
            kernel_regularizer=self.regularizer))
        # TODO: specify random normal init
        encoding_layers.append(layers.BatchNormalization())
    
    encoding_layers.append(layers.Flatten())
      
    for nodes in self.dense_nodes:
      encoding_layers.append(layers.Dense(
        nodes, activation=tf.nn.relu,
        kernel_regularizer=self.regularizer))

    encoding_layers.append(layers.Dense(
      self.num_components,
      activation=self.activation,
      kernel_regularizer=self.regularizer))

    return encoding_layers      # TODO: add dropout layers?

  def create_decoding_layers(self):
    decoding_layers = []
    for nodes in reversed(self.dense_nodes):
      decoding_layers.append(layers.Dense(
        nodes, activation=tf.nn.relu,
        kernel_regularizer=self.regularizer))

    decoding_layers.append(layers.Dense(np.product(self._unflat_shape)))
    decoding_layers.append(layers.Reshape(self._unflat_shape))

    for i, filters in enumerate(reversed(self.level_filters)):
      if i > 0:
        decoding_layers.append(layers.Conv2DTranspose(
          filters, (2,2),
          strides=(2,2),
          padding='same',
          activation=tf.nn.relu,
          kernel_regularizer=self.regularizer))
      for _ in range(self.level_depth):
        decoding_layers.append(layers.Conv2D(
          filters, [3,3],
          activation=tf.nn.relu,
          padding='same',
          kernel_regularizer=self.regularizer))
        decoding_layers.append(layers.BatchNormalization())

    decoding_layers.append(layers.Conv2D(
      1, [1,1],
      kernel_regularizer=self.regularizer))

    return decoding_layers

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

  def call(self, inputs, training=False):
    embedding = self.encode(inputs, training=training)
    if training:
      embedding = self.dropout(embedding)
    return self.decode(embedding, training=training)
