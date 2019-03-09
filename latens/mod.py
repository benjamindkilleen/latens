"""Wrappers around functional Keras models."""

import tensorflow as tf
from tensorflow import keras
from shutil import rmtree
from latens import lay
from latens.utils import dat, vis, act
import os
import numpy as np
from glob import glob
import logging

logger = logging.getLogger('latens')

def log_model(model):
  logger.info(f'model: {model.name}')
  log_layers(model.layers)

def log_layers(layers):
  for layer in layers:
    logger.info(
      f"layer:{layer.input_shape} -> {layer.output_shape}:{layer.name}")
      
class Model():
  def __init__(self, model, model_dir=None, tensorboard=None):
    """Wrapper around a keras.Model for saving, loading, and running.

    Subclasses can overwrite the loss() method to create their own loss.

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
    self.tensorboard = tensorboard
    
    self.model = model
    log_model(self.model)

  def compile(self, *args, **kwargs):
    raise NotImplementedError

  @property
  def callbacks(self):
    callbacks = []
    if self.checkpoint_path is not None:
      callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        self.checkpoint_path, verbose=1, save_weights_only=True,
        period=1))
    if self.tensorboard:
      # Need to have an actual director in which to store the logs.
      raise NotImplementedError
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
    if latest is None:
      logger.info(f"no checkpoint found in {self.model_dir}")
    else:
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
  
class SequentialModel(Model):
  def __init__(self, **kwargs):
    """FIXME! briefly describe function

    :returns: 
    :rtype: 

    """
    model = keras.models.Sequential(layers=self.create_layers())
    super().__init__(model, **kwargs)

  def create_layers(self):
    """Implemented by subclasses.

    Should return a list of keras layers to add to the model. Should only be
    called once.

    """
    raise NotImplementedError

  
class Classifier(SequentialModel):
  def __init__(self, input_shape, num_classes,
               output_activation='softmax',
               dropout=0.2,
               **kwargs):
    self.input_shape = input_shape
    self.num_classes = num_classes
    self.output_activation = output_activation
    self.dropout = dropout
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
  def __init__(self, input_shape, latent_dim,
               num_components=None,
               rep_activation=None,
               dropout=0.2,
               **kwargs):
    """Create an autoencoder using two Sequential models.

    Creates two models sequential models from the layers generated by
    self.create_encoding_layers() and self.create_decoding_layers(), which it
    then strings together. Stores the tensor at the representation
    layer as self.representation, which subclasses can use in a custom loss
    function.

    To create a custom loss, subclasses can overwrite the 'self.loss()' method,
    as in superclasses.

    Optionally, subclasses can override self.create_encoder and
    self.create_decoder, which currently just use the sequential models
    above. This may be desired, for instance, with variational autoencoders,
    where self.representation really has 2*latent_dim components, half of
    which should be discarded for a strict encoding.

    Subclasses should implement create_encoding_layers() and
    create_decoding_layers(). These can include an InputLayer or not.

    TODO: create the ability to copy weights into a classifier with the same
    architecture.

    :param input_shape: shape of the input
    :param latent_dim: dimension of the latent space
    :param num_components: number of components in representation, default uses
    latent_dim
    :param rep_activation: activation at the representation_layer
    :param dropout: 
    :returns: 
    :rtype: 

    """
    self.input_shape = input_shape
    self.latent_dim = latent_dim
    self.num_components = latent_dim if num_components is None else num_components
    self.rep_activation = rep_activation
    self.dropout = dropout

    # creating the actual model
    self.encoding_layers = self.create_encoding_layers()
    self.decoding_layers = self.create_decoding_layers()
    encoder = self._create_encoder()
    decoder = self._create_decoder()

    inputs = keras.layers.Input(input_shape)
    self.representation = encoder(inputs)
    outputs = decoder(self.representation)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    super().__init__(model, **kwargs)

    # for encoding/decoding after the fact
    self.encoder = self.create_encoder()
    self.decoder = self.create_decoder()
    
  def create_encoding_layers(self):
    raise NotImplementedError("subclasses implement create_encoding_layers()")

  def create_decoding_layers(self):
    raise NotImplementedError("subclasses implement create_decoding_layers()")
    
  def _create_encoder(self):
    return keras.models.Sequential(layers=self.encoding_layers)

  def _create_decoder(self):
    return keras.models.Sequential(layers=self.decoding_layers)

  def create_encoder(self):
    return self._create_encoder()

  def create_decoder(self):
    return self._create_decoder()

  @property
  def loss(self):
    return 'mse'
  
  def compile(self, learning_rate=0.1, **kwargs):
    kwargs['optimizer'] = kwargs.get(
      'optimizer', tf.train.AdadeltaOptimizer(learning_rate))
    kwargs['loss'] = kwargs.get('loss', self.loss)
    kwargs['metrics'] = kwargs.get('metrics', ['mae'])
    self.model.compile(**kwargs)

    # compile encoders and decoders, config doesn't matter
    kwargs['optimizer'] = 'adadelta'
    kwargs['loss'] = kwargs.get('loss', 'mse')
    self.decoder.compile(**kwargs)
    self.encoder.compile(**kwargs)

  def encode(self, *args, **kwargs):
    enc = self.encoder.predict(*args, **kwargs)
    return enc[:,:self.latent_dim]

  def encode_generator(self, *args, **kwargs):
    return self.encoder.predict_generator(*args, **kwargs)

  def decode(self, *args, **kwargs):
    return self.decoder.predict(*args, **kwargs)

  def decode_generator(self, *args, **kwargs):
    return self.decoder.predict_generator(*args, **kwargs)

  
class ConvAutoEncoder(AutoEncoder):
  def __init__(self, input_shape, latent_dim,
               level_filters=[64,32,32],
               level_depth=2,
               dense_nodes=[1024],
               rep_reg=None,
               **kwargs):
    """Create a convolutional autoencoder.

    :param input_shape: shape of the inputs
    :param latent_dim: dimension of the representation
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
    
    super().__init__(input_shape, latent_dim, **kwargs)

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

class ConvVariationalAutoEncoder(ConvAutoEncoder):
  """Create a convolutional variational autoencoder. 

  Uses the same layer setup as a ConvAutoEncoder, with a different sampling
  strategy and loss. 
  """
  def __init__(self, input_shape, latent_dim,
               epsilon_std=1.0,
               **kwargs):
    self.epsilon_std = epsilon_std
    super().__init__(input_shape, latent_dim, num_components=2*latent_dim,
                     **kwargs)
    
  def create_decoding_layers(self):
    layers = []
    layers.append(lay.Sampling(epsilon_std=self.epsilon_std))
    layers += super().create_decoding_layers()
    return layers

  @property
  def loss(self):
    # assumes representation is batched
    z_mean = self.representation[:,:self.latent_dim]
    z_log_std = self.representation[:,self.latent_dim:]
    def loss_function(inputs, outputs):
      cross_entropy = tf.reduce_sum(
        tf.keras.backend.binary_crossentropy(inputs, outputs))
      kl_batch = -0.5 * tf.reduce_sum(1 + z_log_std
                                      - tf.square(z_mean)
                                      - tf.exp(z_log_std), axis=-1)
      kl_div = tf.reduce_mean(kl_batch)
      return cross_entropy + kl_div
    return loss_function


