import tensorflow as tf
from shutil import rmtree
from latens.utils import dat
import os
import numpy as np
import logging

logger = logging.getLogger('latens')

class Model:
  """A Model has an `encode` method."""
  def __init__(self, image_shape, **kwargs):
    self.image_shape = image_shape
    learning_rate = kwargs.get('learning_rate', 0.01)
    momentum = kwargs.get('momentum', 0.9)
    self.model_dir = kwargs.get('model_dir')
    
    l2_reg = kwargs.get('l2_reg')
    if l2_reg is None:
      self.regularizer = None
    else:
      self.regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg) 

    feature_columns = [tf.feature_column.numeric_column(
      'image', shape=self.image_shape, dtype=tf.float32)]
    self._params = {'feature_columns' : feature_columns}


  def create(self, training=True):
    """Create the model function for an estimator.

    TODO: determine how much of this can be superclassed
    """
    raise NotImplementedError

  
  def infer(self, image, **kwargs):
    """Run inference on a single batch.

    :param image: the tensor containing the batch

    """
    raise NotImplementedError

  def train(self, record_name, eval_data_input=None, overwrite=False,
            num_epochs=1, eval_secs=600, save_steps=100, log_steps=5,
            cores=None):
    """Train the model with train_data_input.

    :param train_data_input: DataInput subclass to use for training, usually a
    TrainDataInput.
    :param eval_data_input: DataInput subclass to use for evaluation. If None, does no
    evaluation.
    :param overwrite: overwrite the existing model
    :param num_epochs: number fo epochs to train for
    :param eval_secs: run evaluation every `eval_secs`
    :param save_steps: save the model every `save_steps`
    :param log_steps: log the progress every `log_steps` steps
    :param cores: parellelize over `cores` cores. Default (None) does no
    parallelization.

    """

    assert eval_secs >= 0

    if overwrite and self.model_dir is None:
      logger.warning("FAIL to overwrite; model_dir is None")

    if (overwrite and self.model_dir is not None
        and os.path.exists(self.model_dir)):
      rmtree(self.model_dir)
      
    if (overwrite and self.model_dir is not None
        and not os.path.exists(self.model_dir)):
      os.mkdir(self.model_dir)

    # Configure session
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    run_config = tf.estimator.RunConfig(model_dir=self.model_dir,
                                        save_checkpoints_steps=save_steps,
                                        log_step_count_steps=log_steps)
    

    # Train the model. (Might take a while.)
    model = tf.estimator.Estimator(model_fn=self.create(training=True),
                                   model_dir=self.model_dir,
                                   params=self._params,
                                   config=run_config)

    def input_fn():
      dataset = dat.load_dataset(record_name)
      return (dataset
              .shuffle(10000)
              .repeat(num_epochs)
              .batch(4)
              .prefetch(4)
              .make_one_shot_iterator()
              .get_next())
    
    if eval_data_input is None:
      logger.info("train...")
      model.train(input_fn=input_fn)
    else:
      logger.info("train and evaluate...")
      train_spec = tf.estimator.TrainSpec(
        input_fn=train_data_input(num_epochs=num_epochs))
      eval_spec = tf.estimator.EvalSpec(input_fn=eval_data_input(),
                                        throttle_secs=eval_secs)
      tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

  def predict(self, data_input):
    """Return the estimator's predictions on data_input.

    :param data_input: DataInput object
    :returns: 
    :rtype: 

    """
    
    if self.model_dir is None:
      logger.warning("prediction FAILED (no model_dir)")
      return None
  
    model = tf.estimator.Estimator(model_fn=self.create(training=False),
                                   model_dir=self.model_dir,
                                   params=self.params)

    predictions = model.predict(input_fn=data_input())
    return predictions

    
class AutoEncoder(Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_components = kwargs.get('num_components', 10)

  def create(self, training=True):
    """Create the model function for an estimator.

    :param training: Whether this is for training.
    :returns: model function

    """

    def model_function(features, labels, mode, params):
      image = tf.reshape(features, [-1] + self.image_shape)
      reconstruction = self.infer(image, training=training)

      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
          mode=mode, predictions={'image' : image,
                                  'reconstruction' : reconstruction})

      # Calculate loss
      loss = tf.losses.mean_squared_error(image, reconstruction)

      # Return an optimizer, if mode is TRAIN
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        # TODO: allow choice between these
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss=cross_entropy,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=cross_entropy, 
                                          train_op=train_op)

      assert mode == tf.estimator.ModeKeys.EVAL
      eval_metrics = {'loss' : loss}
      return tf.estimator.EstimatorSpec(mode=mode,
                                        loss=cross_entropy,
                                        eval_metric_ops=eval_metrics)

    return model_function
    
  def infer(self, image, **kwargs):
    """Run the encoder and decoder on the batch 'image'."""
    encoding = self.encode(image, **kwargs)
    return self.decode(encoding, **kwargs)
  
  def encode(self, image, **kwargs):
    raise NotImplementedError

  def decode(self, image, **kwargs):
    raise NotImplementedError

  
class ConvAutoEncoder(AutoEncoder):
  def __init__(self, *args, **kwargs):
    """Create a convolutional autoencoder similar to the UNet architecture.

    :param level_filters: number of filters to use at each level. Default is
    [16, 32].
    :param level_depth: how many convolutional layers to pass the
    image through at each level. Default is 3.
    :param fc_nodes: number of nodes in fully 

    """
    super().__init__(*args, **kwargs)
    self.level_filters = kwargs.get('level_filters', [32, 64])
    self.level_depth = kwargs.get('level_depth', 4)
    self.fc_nodes = kwargs.get('fc_nodes', [1024, 1024])
    
  def encode(self, inputs, **kwargs):
    for filters in self.level_filters:
      inputs = self.down_level(inputs, filters, **kwargs)

    self.unflat_shape = inputs.shape
    inputs = tf.layers.flatten(inputs)

    for nodes in self.fc_nodes:
      inputs = self.fc(inputs, nodes, **kwargs)

    return self.fc(inputs, self.num_components, fc_activation=tf.nn.sigmoid,
                   **kwargs)
      
  def decode(self, inputs, unflat_shape, **kwargs):
    for nodes in reversed(self.fc_nodes):
      inputs = self.fc(inputs, nodes, **kwargs)

    inputs = tf.reshape(inputs, self._unflat_shape)

    for filters in reversed(self.level_filters):
      inputs = self.up_level(inputs, filters, **kwargs)

    inputs = self.conv(inputs, filters=1, conv_kernel_size=[1,1],
                       conv_activation=None)
    return inputs

  
  def down_level(self, inputs, filters, **kwargs):
    """Apply level_depth convolutional layers, then a 2x2 pooling layer."""
    for _ in range(self.level_depth):
      inputs = self.conv(inputs, filters, **kwargs)
    return self.pool(inputs)
  
  def up_level(self, inputs, filters, **kwargs):
    """Apply a deconv layer, then level_depth convolutional layers."""
    inputs = self.deconv(inputs, filters, **kwargs)
    for _ in range(self.level_depth):
      inputs = self.conv(outputs, filters, **kwargs)
    return inputs
      
  def conv(self, inputs, filters, conv_kernel_size=[3,3], conv_activation=tf.nn.relu,
           conv_padding='same', training=True, **kwargs):    
    """Perform a convolutional layer."""
    stddev = np.sqrt(2 / (np.prod(conv_kernel_size) * filters))
    initializer = tf.initializers.random_normal(stddev=stddev)

    output = tf.keras.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=conv_kernel_size,
      padding=conv_padding,
      activation=conv_activation,
      kernel_initializer=initializer,
      kernel_regularizer=self.regularizer)

    # normalize the weights in the kernel
    output = tf.keras.layers.batch_normalization(
      inputs=output,
      axis=-1,
      momentum=0.9,
      epsilon=0.001,
      center=True,
      scale=True,
      training=training)

    return output

  def pool(self, inputs, **kwargs):
    """Apply 2x2 maxpooling."""
    return tf.keras.layers.max_pooling2d(
      inputs=inputs, size=[2, 2], strides=2)
  
  def deconv(inputs, filters, deconv_padding='same', **kwargs):
    """Perform "de-convolution" or "up-conv" to the inputs, increasing shape."""

    stddev = np.sqrt(2 / (2*2*filters))
    initializer = tf.initializers.random_normal(stddev=stddev)
    
    output = tf.layers.conv2d_transpose(
      inputs=inputs,
      filters=filters,
      strides=[2, 2],
      kernel_size=[2, 2],
      padding=deconv_padding,
      activation=tf.nn.relu,
      kernel_initializer=initializer,
      kernel_regularizer=self.regularizer)

    return output

  def fc(inputs, nodes, fc_activation=tf.nn.relu, training=True, **kwargs):

    stddev = np.sqrt(2 / nodes)
    initializer = tf.initializers.random_normal(stddev=stddev)

    outputs = tf.layers.dense(
      inputs,
      units,
      activation=fc_activation,
      kernel_initializer=initializer,
      kernel_regularizer=self.regularizer,
      trainable=training)
    
    return outputs


