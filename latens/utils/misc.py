"""Miscellaneous util functions mainly used in latens.py for various commands.
"""

import os
import tensorflow as tf
from tensorflow import keras
import logging
logger = logging.getLogger('latens')

# TODO: create a callback subclass that saves the model ever
class SaveCallback(tf.keras.callbacks.Callback):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def on_epoch_end(self, epoch, *args, **kwargs):
    self.model.save(epoch)

def create_callbacks(args, model):
  """Create callbacks based on parsed arguments.
  """
  callbacks = []
  if args.model_dir[0] is not None:
    callbacks.append(SaveCallback(model))
  if args.tensorboard:
    tensorboard = tf.keras.callbacks.TensorBoard(
      log_dir=os.path.join(args.model_dir[0], 'logs'))
    callbacks.append(tensorboard)
  return callbacks
