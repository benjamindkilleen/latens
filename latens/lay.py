"""Inspired by various sources, including:
https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/
https://blog.keras.io/building-autoencoders-in-keras.html

"""

import tensorflow as tf
from tensorflow import keras

class Sampling(keras.layers.Layer):
  """Given the means and log variance in a single tensor, sample from the
  corresponding distribution.

  Considers the first half of an input tensor to be the mean, second half to be
  the log variance

  """    
  def call(self, inputs):
    latent_dim = tf.shape(inputs)[1] // 2
    z_mean = self.representation[:,:latent_dim]
    z_log_std = self.representation[:,latent_dim:]
    epsilon = tf.random_normal(
      shape=(self.batch_size, self.num_components),
      mean=0.0, std=self.epsilon_std,
      dtype=z_mean.dtype)
    return z_mean + epsilon * tf.exp(z_log_sigma)
  
  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[1] // 2)
  
class TakeMean(keras.layers.Layer):
  def call(self, inputs):
    latent_dim = tf.shape(inputs)[1] // 2
    return inputs[:,:latent_dim]
  
  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[1] // 2)
  
