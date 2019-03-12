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

  def errors(self, dataset, steps, batch_size):
    """Return an array giving the everage absolute error for every example in the
    dataset.

    :param dataset: tf dataset
    :param steps: 
    :param batch_size:

    """
    predictions = self.predict(dataset, steps=steps)

    get_next = dataset.make_one_shot_iterator().get_next()
    errors = np.ones(steps*batch_size)
    with tf.Session() as sess:
      for b in range(0, steps, batch_size):
        X,Y = sess.run(get_next)
        Y_pred = predictions[b:b+batch_size]
        Y = Y.reshape(Y.shape[0], -1)
        Y_pred = Y_pred.reshape(Y_pred.shape[0], -1)
        errors[b:b+batch_size] = np.mean(np.abs(Y - Y_pred), axis=-1)
    return errors

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
  def __init__(self, layers=None, **kwargs):
    """Create a Sequential model.

    :param layers: if not provided, uses self.create_layers()
    :returns: 
    :rtype: 

    """
    if layers is None:
      layers = self.create_layers()
    model = keras.models.Sequential(layers=layers)
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
    """Create a classifier using a sequential model.

    :param input_shape: 
    :param num_classes: 
    :param output_activation: 
    :param dropout: 
    :returns: 
    :rtype: 

    """
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

  @staticmethod
  def from_autoencoder(autoencoder, num_classes, **kwargs):
    """Create a classifier from the encoding layers of an AE.

    classifier needs to be compiled after initialization

    :param autoencoder: 
    :param num_classes: 
    :returns: 
    :rtype: 

    """
    layers = autoencoder.encoding_layers
    for layer in layers:
      layer.trainable = False
    layers[-1] = keras.layers.Dense(num_classes, activation='softmax')
    classifier = Classifier(autoencoder.input_shape, num_classes,
                            layers=layers, **kwargs)
    return classifier

  def incorrect(self, dataset, steps, batch_size):
    """Return a "sampling" as in sam.py given indices of incorrect examples."""
    predictions = self.predict(dataset, steps=steps)

    get_next = dataset.make_one_shot_iterator().get_next()
    incorrect = np.zeros(steps*batch_size, dtype=np.int64)
    with tf.Session() as sess:
      for b in range(0, steps, batch_size):
        X,Y = sess.run(get_next)
        Y_pred = predictions[b:b+batch_size]
        Y = np.argmax(Y, axis=1)
        Y_pred = np.argmax(Y_pred, axis=1)
        incorrect[b:b+batch_size] = np.not_equal(Y, Y_pred).astype(np.int64)
    return incorrect

  @staticmethod
  def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.sum(targets*np.log(predictions+1e-9), axis=1) / predictions.shape[0]
  
  def losses(self, dataset, steps, batch_size):
    """Return a "sampling" as in sam.py given indices of incorrect examples."""
    predictions = self.predict(dataset, steps=steps)

    get_next = dataset.make_one_shot_iterator().get_next()
    incorrect = np.zeros(steps*batch_size, dtype=np.int64)
    with tf.Session() as sess:
      for b in range(0, steps, batch_size):
        X,Y = sess.run(get_next)
        Y_pred = predictions[b:b+batch_size]
        incorrect[b:b+batch_size] = Classifier.cross_entropy(Y_pred, Y)
    return incorrect

  
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
               level_depth=3,
               dense_nodes=[1024, 1024],
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

    layers += self.create_conv(1, activation=None)
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

  def compute_vae_kl(self, z_mean, z_log_std):
    kl_batch = -0.5 * tf.reduce_sum(1 + z_log_std
                                      - tf.square(z_mean)
                                      - tf.exp(z_log_std), axis=-1)
    return tf.reduce_mean(kl_batch)
  
  @property
  def loss(self):
    z_mean = self.representation[:,:self.latent_dim]
    z_log_std = self.representation[:,self.latent_dim:]
    def loss_function(inputs, outputs):
      recon_loss = tf.reduce_mean(
        tf.keras.backend.binary_crossentropy(inputs, outputs))

      kl_div = self.compute_vae_kl(z_mean, z_log_std)
      return recon_loss + kl_div
    return loss_function

  
class StudentAutoEncoder(ConvAutoEncoder):
  """Create a convolutional variational autoencoder. 

  Uses the same layer setup as a ConvAutoEncoder, with a different sampling
  strategy and loss.

  For now, calculate P and Q on the fly for each batch. For this, feed in a
  "self_supervised" style dataset.

  TODO:
  Use the self_supervised_distributed attribute of a dat.Data to calculate each
  batch's P distribution in a prefetched way. In principle, much larger batch
  sizes should work better for this process. examples must therefore take the
  shape: (x, (x,p))

  t-SNE implementation borrows details from a variety of sources, including
  https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/

  """
  def __init__(self, input_shape, latent_dim, batch_size,
               perplexity=30.0, kl_multiplier=1e4,
               **kwargs):
    """

    :param input_shape: shape of input images
    :param latent_dim: number of latent dimensions
    :param batch_size: size of each batch, needed for loss function
    :param perplexity: desired perplexity of distribution
    :param kl_multiplier: weights the KL divergence factor
    :returns: 
    :rtype: 

    """
    self.batch_size = batch_size
    self.perplexity = perplexity
    self.kl_multiplier = kl_multiplier
    super().__init__(input_shape, latent_dim, **kwargs)

  @staticmethod
  def squared_distances(X):
    """Compute pairwise euclidean distances"""
    sum_X_sqr = tf.reduce_sum(tf.square(X), axis=1)
    return (sum_X_sqr - 2*tf.matmul(X, X, transpose_b=True) +
            tf.transpose(sum_X_sqr))
  
  @staticmethod
  def softmax(X, diag_zero=True):
    """Take softmax of each row of 2D tensor X."""
    exp_x = tf.exp(X - tf.reduce_max(X, axis=1, keepdims=True))
    if diag_zero:
      mask = tf.constant(1, exp_x.dtype) - tf.eye(tf.shape(exp_x)[0], dtype=exp_x.dtype)
      exp_x = exp_x * mask
    exp_x = exp_x + tf.constant(1e-8, exp_x.dtype)
    return exp_x / tf.reduce_sum(exp_x, axis=1, keepdims=True)

  @staticmethod
  def probability_matrix(distances, sigmas=None):
    """Compute the (row-based) probability matrix over distances."""
    if sigmas is None:
      return StudentAutoEncoder.softmax(distances)
    else:
      den = 2*tf.reshape(tf.square(sigmas), (-1,1))
      return StudentAutoEncoder.softmax(distances / den)

  @staticmethod
  def binary_search(func, target, n=1, tolerance=1e-5, max_iter=50, 
                    lower=1e-10, upper=1000.):
    """Perform a binary search over input values to `func` in tf.

    Can perform this search over a vector of targets, as long as func is a
    vector function.

    :param func: tf function to evaluate
    :param target: target value we want the function to output
    :param n: dimensionality of search, default is 1
    :param tolerance: "close enough" threshold
    :param max_iter: number of iterations over
    :param lower: lower bound to search
    :param upper: upper bound to search

    """
    target = tf.constant(target, dtype=tf.float32)
    tolerance = tf.constant(tolerance, tf.float32)
    lower = tf.constant(lower, tf.float32, (n,))
    upper = tf.constant(upper, tf.float32, (n,))
    two = tf.constant(2, dtype=tf.float32)
    guess = (lower + upper) / two
    close_guess = tf.constant(False, tf.bool, (n,))

    def cond(x, close, low, up):
      return tf.reduce_all(close)
    
    def body(x, close, low, up):
      """single iteration of the search"""
      val = func(x)
      close = tf.abs(val - target) <= tolerance
      which = val > target
      low = tf.where(close, low, tf.where(which, low, x))
      up = tf.where(close, up, tf.where(which, x, up))
      new_x = (low + up) / two
      x = tf.where(close, x, new_x)
      return x, close, low, up
    
    out = tf.while_loop(
      cond, body, (guess, close_guess, lower, upper),
      maximum_iterations=max_iter,
      back_prop=True) # change?
    return out[0]

  @staticmethod
  def log_2(x):
    two = tf.constant(2, x.dtype)
    return tf.log(x) / tf.log(two)
  
  @staticmethod
  def calculate_perplexity(prob_matrix):
    """Calculate perplexity for each row in P"""
    entropy = -tf.reduce_sum(
      prob_matrix * StudentAutoEncoder.log_2(prob_matrix), axis=1)
    two = tf.constant(2, prob_matrix.dtype)
    return tf.pow(two, entropy)

  @staticmethod
  def perplexity(distances, sigmas):
    """Wrapper function computing perplexity of each row in distances."""
    return StudentAutoEncoder.calculate_perplexity(
      StudentAutoEncoder.probability_matrix(distances, sigmas=sigmas))

  @staticmethod
  def calculate_sigmas(distances, desired_perplexity):
    """Calculate the sigmas for each row of `distances`"""
    func = lambda sigmas : StudentAutoEncoder.perplexity(distances, sigmas)
    return StudentAutoEncoder.binary_search(func, desired_perplexity,
                                            n=distances.shape[0])

  @staticmethod
  def joint_probability_matrix(prob_matrix):
    """Calculater the joint probability matrix form `prob_matrix`"""
    den = tf.constant(2, tf.float32) * tf.cast(tf.shape(prob_matrix)[0], tf.float32)
    return (prob_matrix + tf.transpose(prob_matrix)) / den
  
  def calculate_P(self, X):
    n = self.batch_size
    X = tf.reshape(X, (n, -1))
    distances = - StudentAutoEncoder.squared_distances(X) # negative distances
    sigmas = StudentAutoEncoder.calculate_sigmas(distances, self.perplexity)
    prob_matrix = StudentAutoEncoder.probability_matrix(distances, sigmas=sigmas)
    return StudentAutoEncoder.joint_probability_matrix(prob_matrix)
  
  def calculate_Q(self, Z):
    distances = StudentAutoEncoder.squared_distances(Z)
    inv_distances = tf.reciprocal(tf.ones_like(distances) + distances)
    mask = (tf.constant(1, inv_distances.dtype) -
            tf.eye(tf.shape(inv_distances)[0], dtype=inv_distances.dtype))
    inv_distances = mask * inv_distances
    return inv_distances / tf.reduce_sum(inv_distances)

  def compute_student_kl(self, X, Z):
    eps = tf.constant(1e-20, tf.float32, (self.batch_size, self.batch_size))
    P = self.calculate_P(X)
    P = tf.where(P < eps, eps, P)
    
    Q = self.calculate_Q(Z)
    Q = tf.where(Q < eps, eps, Q)
    
    return tf.reduce_sum(P * tf.log(P / Q))
  
  @property
  def loss(self):
    representation = self.representation
    kl_multiplier = tf.constant(self.kl_multiplier, tf.float32)
    def loss_function(inputs, outputs):
      recon_loss = tf.reduce_mean(
        tf.keras.backend.binary_crossentropy(inputs, outputs))
      kl_div = kl_multiplier * self.compute_student_kl(inputs, representation)
      loss = recon_loss + kl_div
      ops = []
      ops.append(tf.print('recon_loss:', recon_loss))
      ops.append(tf.print('kl_div:', kl_div))
      ops.append(tf.print('loss:', recon_loss + kl_div))
      # with tf.control_dependencies(ops):
      #   loss = tf.identity(loss)
      return loss
    return loss_function


class VariationalStudentAutoEncoder(
    ConvVariationalAutoEncoder,
    StudentAutoEncoder):
  
  def __init__(self, input_shape, latent_dim, batch_size,
               perplexity=30.0, epsilon_std=1.0,
               student_kl_multiplier=1e4,
               vae_kl_multiplier=1,
               **kwargs):
    """Instantiated same way StudentAutoEncoder is."""
    self.batch_size = batch_size
    self.perplexity = perplexity
    self.epsilon_std = epsilon_std
    self.student_kl_multiplier = student_kl_multiplier
    self.vae_kl_multiplier = vae_kl_multiplier
    ConvAutoEncoder.__init__(self, input_shape, latent_dim,
                             num_components=2*latent_dim, **kwargs)
    
  @property
  def loss(self):
    z_mean = self.representation[:,:self.latent_dim]
    z_log_std = self.representation[:,self.latent_dim:]
    student_kl_multiplier = tf.constant(self.student_kl_multiplier, tf.float32)
    vae_kl_multiplier = tf.constant(self.vae_kl_multiplier, tf.float32)
    def loss_function(inputs, outputs):
      recon_loss = tf.reduce_mean(
        tf.keras.backend.binary_crossentropy(inputs, outputs))
      vae_kl_div = vae_kl_multiplier * self.compute_vae_kl(z_mean, z_log_std)
      student_kl_div = self.compute_student_kl(inputs, z_mean)
      loss = recon_loss + student_kl_div + vae_kl_div
      ops = []
      ops.append(tf.print('recon_loss:', recon_loss))
      ops.append(tf.print('student_kl_div:', student_kl_div))
      ops.append(tf.print('vae_kl_div:', vae_kl_div))
      ops.append(tf.print('loss:', loss))
      # with tf.control_dependencies(ops):
      #   loss = tf.identity(loss)
      return loss
    return loss_function
