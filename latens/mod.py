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

class Model(tf.keras.Model):
  def __init__(self, image_shape, model_dir=None, overwrite=False,
               batch_size=1, **kwargs):
    super().__init__(**kwargs)
    self.image_shape = image_shape
    self.batch_size = batch_size
    self.model_dir = model_dir
    self.model_path = (None if model_dir is None
                       else os.path.join(model_dir, 'model.h5'))
    self.overwrite = overwrite
    self.dummy_set = dat.DummyInput(image_shape, batch_size=batch_size)
    
  def save(self, epoch=None, **kwargs):
    """Save the model to self.model_dir, depending on self.overwrite."""
    if self.model_dir is None:
      logger.warning(f"called 'save' with no model_dir")
      return

    if epoch is None:
      model_path = self.model_path
    else:
      model_path = os.path.join(self.model_dir, f'model_e={epoch}.h5')

    if not os.path.exists(self.model_dir):
      os.mkdir(self.model_dir)
      logger.info(f"created model dir at {self.model_dir}")
    if epoch is None and os.path.exists(model_path) and not self.overwrite:
      logger.info(f"can't save model at {self.model_dir} without overwriting")
      if input("overwrite? [y/n]: ") != 'y':
        logger.info("skipping overwrite")
        return
    self.save_weights(model_path)
    logger.info(f"saved model to {model_path}")

  def load(self, recent=False):
    """Load weights for the dataset from its model dir, if possible.
    
    Because of keras weirdness, runs a single training step to determine the
    network topology (doesn't actually train, since weights are loaded after).
    
    :param recent: prefer the most recent epoch weights over a completed file.

    """
    if self.model_path is None:
      logger.warning(f"failed to load weights from {self.model_path}")
      return

    if recent or not os.path.exists(self.model_path):
      model_paths = sorted(glob(os.path.join(self.model_dir, 'model_e=*.h5')))
      if len(model_paths) == 0:
        logger.warning(f"No epoch data, failed to load weights.")
        return
      model_path = model_paths[-1]
    else:
      model_path = self.model_path
    
    self.fit(self.dummy_set.self_supervised, epochs=1, steps_per_epoch=1,
             verbose=0)
    self.load_weights(model_path)
    logger.info(f"loaded weights from {model_path}")

    
class AutoEncoder(Model):
  def __init__(self, image_shape, num_components, **kwargs):
    """A superclass for autoencoders. Subclasses must define the attributes
    'encoding_layers' and 'decoding_layers'.

    :param image_shape: 
    :param num_components: 
    :returns: 
    :rtype: 

    """
    super().__init__(image_shape, **kwargs)
    self.num_components = num_components

  def compile(self, learning_rate=0.1, **kwargs):
    kwargs['optimizer'] = kwargs.get(
      'optimizer', tf.train.AdadeltaOptimizer(learning_rate))
    kwargs['loss'] = kwargs.get('loss', tf.losses.mean_squared_error)
    kwargs['metrics'] = kwargs.get('metrics', ['mae'])
    super().compile(**kwargs)
    
  def call(self, inputs, training=False):
    embedding = self.encode(inputs, training=training)
    reconstruction = self.decode(embedding, training=training)
    return reconstruction

  def encode(self, inputs, training=False):
    for layer in self.encoding_layers:
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

  def add_layer(self, *layers):
    for layer in layers:
      setattr(self, layer.name + '_layer', layer)

  def create_encoding_layers(self):
    raise NotImplementedError

  def create_decoding_layers(self):
    raise NotImplementedError
  
class ConvAutoEncoder(AutoEncoder):
  def __init__(self, image_shape, num_components,
               level_filters=[64,32,32],
               level_depth=2,
               dense_nodes=[1024],
               l2_reg=None,
               rep_activation=act.clu,
               rep_dropout=0.1,
               dropout=0.2,
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
    self._dropout = dropout
    self._rep_dropout = rep_dropout

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
    self.add_layer(*self.encoding_layers)
    self.add_layer(*self.decoding_layers)

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

    return layers

  def create_decoding_layers(self):
    layers = []
    # layers.append(keras.layers.Dropout(self._rep_dropout))
    
    for nodes in reversed(self.dense_nodes):
      layers += self.dense(nodes)

    layers += self.dense(np.product(self._unflat_shape))
    layers.append(keras.layers.Reshape(self._unflat_shape))

    for i, filters in enumerate(reversed(self.level_filters)):
      if i > 0:
        layers += self.conv_transpose(filters)
      for _ in range(self.level_depth):
        layers += self.conv(filters)

    layers += self.conv(1, activation=act.clu, normalize=False)

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
    layers.append(keras.layers.Dropout(self._dropout))
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
    layers.append(keras.layers.Dropout(self._dropout))
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

  
class ConvEmbedder(ConvAutoEncoder):
  def __init__(self, autoencoder, **kwargs):
    """An embedder is created from an autoencoder. 

    It shouldn't be trained, really, but borrows the weights (and parameters)
    from the autoencoder used initializes it.

    TODO: currently only works in eager mode. Figure out why.

    """
    
    super().__init__(autoencoder.image_shape,
                     autoencoder.num_components,
                     batch_size=autoencoder.batch_size,
                     **kwargs)

    self.encoding_layers = autoencoder.create_encoding_layers()
    self.add_layer(*self.encoding_layers)

    self.compile()
    self.fit(self.dummy_set.embed(self.num_components), epochs=1,
             steps_per_epoch=1, verbose=0)
    for layer, other_layer in zip(self.encoding_layers,
                                  autoencoder.encoding_layers):
      layer.set_weights(other_layer.get_weights())

  def call(self, inputs, training=False):
    embedding = self.encode(inputs, training=training)
    return embedding

class SModel():
  def __init__(self, model_dir=None, overwrite=False, batch_size=1):
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
    
    self.overwrite = overwrite
    self.batch_size = batch_size

    self.model = keras.models.Sequential()
    self.layers = self.create_layers()
    for layer in self.layers:
      self.model.add(layer)

  def create_layers(self):
    """Implemented by subclasses.

    Should return a list of keras layers to add to the model.

    """
    raise NotImplementedError

  def compile(self, *args, **kwargs):
    self.model.compile(*args, **kwargs)

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

  # TODO: make a function to wrap around keras.load_model
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
  

class Classifier(SModel):
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
               dropout=0.2,
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
    self.dropout = dropout
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
  
