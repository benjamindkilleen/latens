"""Wrappers around functional Keras models."""

import tensorflow as tf
from tensorflow import keras
from shutil import rmtree
from latens.utils import dat, vis, act
import os
import numpy as np
import logging

logger = logging.getLogger('latens')

def model_files(model_dir):
  """Return the names of the weights and architecture files

  :param model_dir: directory of the model
  :returns: weights path, model path
  :rtype: 

  """
  if model_dir is None:
    return None, None

  return (os.path.join(model_dir, 'weights.h5'),
          os.path.join(model_dir, 'model.h5'))

def load(model_dir):
  """Load the model from model_dir and compile it if necessary.

  :param model_dir: 
  :returns: 
  :rtype: 

  """
  if model_dir is None:
    return None
  weights_path, model_path = model_files(model_dir)
  with open(model_path, 'r') as f:
    model = keras.models.model_from_json(f.read())
  model.load_weights(weights_path)

class Model(tf.keras.Model):
  def __init__(self, model_dir=None, overwrite=False, **kwargs):
    super().__init__(**kwargs)
    self.model_dir = model_dir
    self.model_path = (None if model_dir is None
                       else os.path.join(model_dir, 'model.h5'))
    self.overwrite = overwrite
    
  def save(self):
    """Save the model to self.model_dir, depending on self.overwrite."""
    if self.model_dir is None:
      logger.warning(f"called 'save' with no model_dir")
      return

    if not os.path.exists(self.model_dir):
      os.mkdir(self.model_dir)
      logger.info(f"created model dir at {self.model_dir}")
    if os.path.exists(self.model_path) and not self.overwrite:
      logger.info(f"can't save model at {self.model_dir} without overwriting")
      if input("overwrite? [y/n]: ") != 'y':
        return
    self.save_weights(self.model_path)
    logger.info(f"saved model to {self.model_path}")
    
  def load(self):
    # TODO: figure out how to properly run model before-hand to load weights well.
    if (self.model_path is not None
        and os.path.exists(self.model_path)):
      self.load_weights(self.model_path)
      logger.info(f"loaded weights from {self.model_path}")
    else:
      logger.warning(f"failed to load weights from {self.model_path}")

class AutoEncoder(Model):
  def __init__(self, image_shape, num_components, **kwargs):
    """A superclass for autoencoders.

    :param image_shape: 
    :param num_components: 
    :returns: 
    :rtype: 

    """
    super().__init__(**kwargs)
    self.image_shape = image_shape
    self.num_components = num_components
    
  def call(self, inputs, training=True):
    embedding = self.encode(inputs, training=training)
    reconstruction = self.decode(embedding, training=training)
    # if tf.executing_eagerly():
    #   vis.show_image(inputs[0], reconstruction[0])
    return reconstruction

  def encode(self, inputs, training=False):
    for layer in self.encoding_layers:
      if (not training) and 'dropout' in layer.name:
        continue
      input_shape = inputs.shape
      inputs = layer(inputs)
      output_shape = inputs.shape
      logger.debug(f"{input_shape} -> {output_shape}:{layer.name}")
    return inputs

  def decode(self, inputs, training=True):
    for layer in self.decoding_layers:
      if (not training) and 'dropout' in layer.name:
        continue
      input_shape = inputs.shape
      inputs = layer(inputs)
      output_shape = inputs.shape
      logger.debug(f"{input_shape} -> {output_shape}:{layer.name}")
    return inputs

  def add_layer(self, *layers):
    for layer in layers:
      setattr(self, layer.name + '_layer', layer)

class ConvAutoEncoder(AutoEncoder):
  def __init__(self, image_shape, num_components,
               level_filters=[64,32,32], # [64,64,32],
               level_depth=2,
               dense_nodes=[1024], # [1024,1024],
               l2_reg=None,
               rep_activation=act.clu,
               dropout=0.1,
               **kwargs):
    """Create a convolutional autoencoder.

    :param image_shape: shape of the inputs
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
    super().__init__(image_shape, num_components, **kwargs)
    self.level_filters = level_filters
    self.level_depth = level_depth
    self.dense_nodes = dense_nodes
    self._rep_activation = rep_activation
    self._dropout_rate = dropout

    if l2_reg is None:
      self.regularizer = None
    else:
      self.regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg)     

    if len(level_filters) == 0:
      scale_factor = 1
    else:
      scale_factor = 2**(len(level_filters) - 1)
    assert(self.image_shape[0] % scale_factor == 0
           and self.image_shape[1] % scale_factor == 0)
    self._unflat_shape = (self.image_shape[0] // scale_factor,
                          self.image_shape[1] // scale_factor,
                          1 if len(level_filters) == 0 else level_filters[-1])

    self.encoding_layers = self.create_encoding_layers()
    self.decoding_layers = self.create_decoding_layers()

  def create_encoding_layers(self):
    layers = []
    layers.append(keras.layers.InputLayer(self.image_shape))

    for i, filters in enumerate(self.level_filters):
      if i > 0:
        layers += self.maxpool()
      for _ in range(self.level_depth):
        layers += self.conv(filters)
    
    layers.append(keras.layers.Flatten())
      
    for nodes in self.dense_nodes:
      layers += self.dense(nodes)

    layers += self.dense(
      self.num_components,
      activation=self._rep_activation)

    self.add_layer(*layers)
    return layers

  def create_decoding_layers(self):
    layers = []
    for nodes in reversed(self.dense_nodes):
      layers += self.dense(nodes)

    layers += self.dense(np.product(self._unflat_shape))
    layers.append(keras.layers.Reshape(self._unflat_shape))

    for i, filters in enumerate(reversed(self.level_filters)):
      if i > 0:
        layers += self.conv_transpose(filters)
      for _ in range(self.level_depth):
        layers += self.conv(filters)

    # TODO: use clu on last layer
    layers += self.conv(1, activation=act.clu, normalize=False)

    self.add_layer(*layers)
    return layers
  
  def conv(self, filters,
           activation=keras.activations.relu, normalize=False):
    layers = []
    layers.append(keras.layers.Conv2D(
      filters, (3,3),
      activation=activation,
      padding='same',
      kernel_initializer='glorot_normal'))
    if normalize:
      layers.append(keras.layers.BatchNormalization())
    return layers

  def maxpool(self):
    layers = []
    layers.append(keras.layers.MaxPool2D())
    layers.append(keras.layers.Dropout(self._dropout_rate))
    return layers

  def upsample(self):
    return [keras.layers.UpSampling2D()]
  
  def dense(self, nodes, activation='relu', normalize=False):
    layers = []
    layers.append(keras.layers.Dense(
      nodes, activation=activation,
      kernel_regularizer=self.regularizer))
    if normalize:
      layers.append(keras.layers.BatchNormalization())
    return layers

  def conv_transpose(self, filters, activation=keras.activations.relu):
    layers = []
    layers.append(keras.layers.Conv2DTranspose(
      filters, (2,2),
      strides=(2,2),
      padding='same',
      activation=activation,
      kernel_regularizer=self.regularizer))
    layers.append(keras.layers.Dropout(self._dropout_rate))
    return layers
    
class ShallowAutoEncoder(AutoEncoder):
  def __init__(self, *args, **kwargs):
    """A very shallow autoencoder with just one hidden layer.

    """
    super().__init__(*args, name='shallow_auto_encoder', **kwargs)

    self.encoding_layers = self.create_encoding_layers()
    self.decoding_layers = self.create_decoding_layers()
  
  def create_encoding_layers(self):
    layers = []
    layers.append(keras.layers.Flatten(input_shape=self.image_shape))
    layers.append(keras.layers.Dense(self.num_components,
                                     activation=keras.activations.relu))
    self.add_layer(*layers)
    return layers

  def create_decoding_layers(self):
    layers = []
    layers.append(keras.layers.Dense(784, activation=keras.activations.relu))
    layers.append(keras.layers.Reshape(self.image_shape))
    self.add_layer(*layers)
    return layers


    
