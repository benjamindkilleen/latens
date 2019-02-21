"""Miscellaneous util functions mainly used in latens.py for various commands.
"""

import os
import tensorflow as tf
import logging
logger = logging.getLogger('latens')


def create_callbacks(args):
  """Create callbacks based on parsed arguments.
  """
  callbacks = []
  if args.tensorboard:
    tensorboard = tf.keras.callbacks.TensorBoard(
      log_dir=os.path.join(args.model_dir[0], 'logs'))
    callbacks.append(tensorboard)
  return callbacks
