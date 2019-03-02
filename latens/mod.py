"""Wrappers around functional Keras models."""

import tensorflow as tf
from tensorflow import keras
from shutil import rmtree
from latens.utils import dat, vis, act
import os
import numpy as np
from glob import glob
import logging

logger = logging.getLogger('latens')

class Model():
  def __init__(self, model_dir=None, dropout=0.2):
    """Create a sequential model in self.model.

    In general, these layers should include an input layer. Furthermore,
    super().__init__() should usually be called at the end of the subclass's
    __init__, after it has initialized variables necessary for
    self.create_layers().

    :param input_shape: 
    :returns: 
    :rtype: 

    """
    self.model_dir = model_dir
    self.checkpoint_path = (None if model_dir is None else
                            os.path.join(model_dir, "cp-{epoch:04d}.hdf5"))
    
    self.dropout = dropout
    self.model = keras.models.Sequential()
    self.layers = self.create_layers()
    for layer in self.layers:
      self.model.add(layer)
      logger.debug(f"layer:{layer.input_shape} -> {layer.output_shape}:{layer.name}")


  def create_layers(self):
    """Implemented by subclasses.

    Should return a list of keras layers to add to the model. Should only be
    called once. Can return a '_layers' attribute that is already instantiated.

    """
    raise NotImplementedError

  def compile(self, *args, **kwargs):
    raise NotImplementedError

  @property
  def callbacks(self):
    callbacks = []
    if self.checkpoint_path is not None:
      callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        self.checkpoint_path, verbose=1, save_weights_only=True,
        period=1))
    return callbacks
    
  def fit(self, *args, **kwargs):
    kwargs['callbacks'] = self.callbacks
    return self.model.fit(*args, **kwargs)

  def predict(self, *args, **kwargs):
    return self.model.predict(*args, **kwargs)

  def evaluate(self, *args, **kwargs):
    return self.model.evaluate(*args, **kwargs)

  def save(self, *args, **kwargs):
    return self.model.save(*args, **kwargs)

  # load the model from a most recent checkpoint
  def load(self):
    if self.model_dir is None:
      logger.warning(f"failed to load weights; no `model_dir` set")
      return
    
    latest = tf.train.latest_checkpoint(self.model_dir)
    self.model.load_weights(latest)
    logger.info(f"restored model from {latest}")

  def create_maxpool(self):
    layers = []
    layers.append(keras.layers.MaxPool2D())
    layers.append(keras.layers.Dropout(self.dropout))
    return layers

  def create_conv(self, filters, activation='relu', normalize=False):
    layers = []
    layers.append(keras.layers.Conv2D(
      filters, (3,3),
      activation=activation,
      padding='same',
      kernel_initializer='glorot_normal'))
    return layers

  def create_dense(self, nodes, activation='relu', normalize=False):
    layers = []
    layers.append(keras.layers.Dense(
      nodes, activation=activation))
    return layers

  def create_conv_transpose(self, filters, activation='relu'):
    layers = []
    layers.append(keras.layers.Conv2DTranspose(
      filters, (2,2),
      strides=(2,2),
      padding='same',
      activation=activation))
    layers.append(keras.layers.Dropout(self.dropout))
    return layers
  

class Classifier(Model):
  def __init__(self, input_shape, num_classes,
               output_activation='softmax',
               **kwargs):
    self.input_shape = input_shape
    self.num_classes = num_classes
    self.output_activation = output_activation
    super().__init__(**kwargs)

  def compile(self, learning_rate=0.1, **kwargs):
    kwargs['optimizer'] = kwargs.get(
      'optimizer', tf.train.AdadeltaOptimizer(learning_rate))
    kwargs['loss'] = kwargs.get('loss', 'categorical_crossentropy')
    kwargs['metrics'] = kwargs.get('metrics', ['accuracy'])
    self.model.compile(**kwargs)

class ConvClassifier(Classifier):
  def __init__(self, input_shape, num_classes,
               level_filters=[64,32,32],
               level_depth=2,
               dense_nodes=[1024],
               **kwargs):
    """Create a classifier.

    :param input_shape: 
    :param num_classes: 
    :param level_filters: 
    :param level_depth: 
    :param dense_nodes: 
    :param dropout: 
    :returns: 
    :rtype: 

    """
    self.level_filters = level_filters
    self.level_depth = level_depth
    self.dense_nodes = dense_nodes
    super().__init__(input_shape, num_classes, **kwargs)
    
  def create_layers(self):
    layers = []
    layers.append(keras.layers.InputLayer(self.input_shape))

    for i, filters in enumerate(self.level_filters):
      if i > 0:
        layers += self.create_maxpool()
      for _ in range(self.level_depth):
        layers += self.create_conv(filters)

    layers.append(keras.layers.Flatten())

    for nodes in self.dense_nodes:
      layers += self.create_dense(nodes)

    layers += self.create_dense(
      self.num_classes,
      activation = self.output_activation)

    return layers
  

class AutoEncoder(Model):
  def __init__(self, input_shape, num_components,
               rep_activation=act.clu,
               **kwargs):
    """Create an autoencoder as a keras functional model.

    :param input_shape: 
    :param num_components: 
    :param rep_activation: 
    :returns: 
    :rtype: 

    """

    self.input_shape = input_shape
    self.num_components = num_components
    self.rep_activation = rep_activation
    super().__init__(**kwargs)

    self.encoder = self.create_encoder()
    self.decoder = self.create_decoder()

  def create_encoder(self):
    encoder = keras.models.Sequential()
    for layer in self.encoding_layers:
      encoder.add(layer)
    encoder.compile(
      optimizer=tf.train.AdadeltaOptimizer(),
      loss='mse',
      metrics=['mae'])
    return encoder

  def create_decoder(self):
    decoder = keras.models.Sequential()
    for layer in self.decoding_layers:
      decoder.add(layer)
    decoder.compile(
      optimizer=tf.train.AdadeltaOptimizer(),
      loss='mse',
      metrics=['mae'])
    return decoder

  def compile(self, learning_rate=0.1, **kwargs):
    kwargs['optimizer'] = kwargs.get(
      'optimizer', tf.train.AdadeltaOptimizer(learning_rate))
    kwargs['loss'] = kwargs.get('loss', 'mse')
    kwargs['metrics'] = kwargs.get('metrics', ['mae'])
    self.model.compile(**kwargs)

  def create_layers(self):
    self.encoding_layers = self.create_encoding_layers()
    self.decoding_layers = self.create_decoding_layers()
    return self.encoding_layers + self.decoding_layers

  def create_encoding_layers(self):
    raise NotImplementedError("subclasses implement create_encoding_layers()")

  def create_decoding_layers(self):
    raise NotImplementedError("subclasses implement create_decoding_layers()")

  def encode(self, *args, **kwargs):
    return self.encoder.predict(*args, **kwargs)

  def decode(self, *args, **kwargs):
    return self.decoder.predict(*args, **kwargs)

  def decode_generator(self, *args, **kwargs):
    return self.decoder.predict_generator(*args, **kwargs)

  
class ConvAutoEncoder(AutoEncoder):
  def __init__(self, input_shape, num_components,
               level_filters=[64,32,32],
               level_depth=2,
               dense_nodes=[1024],
               rep_reg=None,
               **kwargs):
    """Create a convolutional autoencoder.

    :param input_shape: shape of the inputs
    :param num_components: number of components in the representation
    :param level_filters: number of filters to use at each level. Default is
    [64,32,32].
    :param level_depth: how many convolutional layers to pass the
    image through at each level. Default is 3.
    :param dense_nodes: number of nodes in fully
    :param rep_reg: regularization factor for the representation layer
    (notimplemented)

    """
    self.level_filters = level_filters
    self.level_depth = level_depth
    self.dense_nodes = dense_nodes

    if len(level_filters) == 0:
      scale_factor = 1
    else:
      scale_factor = 2**(len(level_filters) - 1)
    assert(input_shape[0] % scale_factor == 0
           and input_shape[1] % scale_factor == 0)
    self._unflat_shape = (input_shape[0] // scale_factor,
                          input_shape[1] // scale_factor,
                          1 if len(level_filters) == 0 else level_filters[-1])
    
    super().__init__(input_shape, num_components, **kwargs)

  def create_encoding_layers(self):
    layers = []
    layers.append(keras.layers.InputLayer(self.input_shape))

    for i, filters in enumerate(self.level_filters):
      if i > 0:
        layers += self.create_maxpool()
      for _ in range(self.level_depth):
        layers += self.create_conv(filters)
    
    layers.append(keras.layers.Flatten())
      
    for nodes in self.dense_nodes:
      layers += self.create_dense(nodes)

    layers += self.create_dense(
      self.num_components,
      activation=self.rep_activation)
    # TODO: add activation regularizer to representation level

    return layers

  
  def create_decoding_layers(self):
    layers = []
    
    for nodes in reversed(self.dense_nodes):
      layers += self.create_dense(nodes)

    layers += self.create_dense(np.product(self._unflat_shape))
    layers.append(keras.layers.Reshape(self._unflat_shape))

    for i, filters in enumerate(reversed(self.level_filters)):
      if i > 0:
        layers += self.create_conv_transpose(filters)
      for _ in range(self.level_depth):
        layers += self.create_conv(filters)

    layers += self.create_conv(1, activation=act.clu)
    return layers

    
