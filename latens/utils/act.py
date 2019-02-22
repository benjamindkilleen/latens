"""Custom activations for the middle layer."""

import tensorflow as tf


def clu(x):
  """Implement a "clipped linear unit."

  Implement the function f(x) = 0 for x < 0, f(x) = x for 0 <= x <= 1, 
  and f(x) = 1 for x > 1.

  :param x: a tensor
  :returns: clipped tensor f(x)

  """
  
  return tf.clip_by_value(x, tf.constant(0, dtype=x.dtype),
                          tf.constant(1, dtype=x.dtype), name='clu')
