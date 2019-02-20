#!/usr/local/bin/python3

"""Main script for running latens.

"""

import logging
logger = logging.getLogger('latens')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s:latens:%(message)s'))
logger.addHandler(handler)

import sys
import os
import argparse
from glob import glob
import tensorflow as tf
from latens.utils import docs, dat, vis
from latens import mod

if sys.version_info < (3,6):
  logger.error(f"Use python{3.6} or higher.")
  exit()

def cmd_debug(args):
  """Run the debug command."""
  data = dat.Data(args.input)
  train_set, dev_set, eval_set = data.split(
    50000,10000,10000, types=[dat.TrainDataInput, dat.DataInput, dat.DataInput])
  logger.debug(f"Output shapes: {train_set._dataset.output_shapes}")
  logger.debug(f"Output types: {train_set._dataset.output_types}")


def cmd_convert(args):
  """Convert the dataset in args.input[0] to tfrecord and store in the same
  directory as a .tfrecord file."""
  dat.convert_from_npz(args.input[0])

def cmd_autoencoder(args):
  """Run training for the autoencoder."""
  data = dat.DataInput(args.input, num_parallel_calls=args.cores[0],
                       batch_size=args.batch_size[0])
  train_set, validation_set = data.split(
    *args.splits, types=[dat.TrainDataInput, dat.DataInput])[:2]
  
  logger.debug(f"train_set: {type(train_set)}")
  
  model = mod.ConvAutoEncoder(
    args.image_shape,
    args.num_components[0],
    l2_reg=args.l2_reg[0],
    activation=args.activation[0],
    dropout=args.dropout[0])

  if (not args.overwrite
      and args.model_dir[0] is not None
      and os.path.exists(args.model_dir[0])
      and len(glob(os.path.join(args.model_dir[0], 'model*'))) > 0):
    logger.info(f"loading weights from {args.model_dir[0]}")
    model.load_weights(os.path.join(args.model_dir[0], 'model'))
    
  def loss(Y, Y_hat):
    """For debugging."""
    if args.eager:
      y = Y[0]
      y_hat = Y_hat[0]
      logger.debug(f"y: {y.shape}")
      logger.debug(f"y_hat: {y_hat.shape}")
      vis.show_image(y, y_hat)
    return tf.losses.mean_squared_error(Y, Y_hat)
    
  # TODO: allow customize
  model.compile(
    optimizer=tf.train.AdadeltaOptimizer(args.learning_rate[0]),
    loss=tf.losses.mean_squared_error,
    metrics=['mae'])

  model.fit(
    train_set.self_supervised,
    epochs=args.epochs[0],
    steps_per_epoch=args.splits[0] // args.batch_size[0],
    validation_data=validation_set.self_supervised,
    verbose=args.verbose[0])

  if args.model_dir[0] is not None:
    if not os.path.exists(args.model_dir[0]):
      os.mkdir(args.model_dir[0])
    model.save_weights(os.path.join(args.model_dir[0], 'model'))
  
def cmd_reconstruct(args):
  """Run reconstruction."""
  data = dat.Data(args.input, num_parallel_calls=args.cores[0],
                  batch_size=args.batch_size[0])
  test_set = data.split(
    *args.splits, types=[None, None, dat.DataInput])[2]

  model = mod.ConvAutoEncoder(
    args.image_shape,
    args.num_components[0],
    l2_reg=args.l2_reg[0],
    activation=args.activation[0],
    dropout=args.dropout[0])

  assert args.model_dir[0] is not None and os.path.exists(args.model_dir[0]) > 0
  model.load_weights(os.path.join(args.model_dir[0], 'model'))

  model.compile(
    optimizer=tf.train.AdadeltaOptimizer(args.learning_rate[0]),
    loss=tf.losses.mean_squared_error,
    metrics=['mae'])

  reconstructions = model.predict(test_set.self_supervised, steps=1)
  for example, reconstruction in zip(test_set, reconstructions):
    image, label = example
    vis.show_image(image, reconstruction)

  
def main():
  parser = argparse.ArgumentParser(description=docs.description)
  parser.add_argument('command', choices=docs.command_choices,
                      help=docs.command_help)
  parser.add_argument('--input', '-i', nargs='+', required=True,
                      help=docs.input_help)
  parser.add_argument('--output', '-o', nargs=1,
                      default=['show'],
                      help=docs.output_help)
  parser.add_argument('--model-dir', '-m', nargs=1,
                      default=[None],
                      help=docs.model_dir_help)
  parser.add_argument('--epochs', '-e', nargs=1,
                      default=[1], type=int,
                      help=docs.epochs_help)
  parser.add_argument('--overwrite', '-f', action='store_true',
                      help=docs.overwrite_help)
  parser.add_argument('--image-shape', '--shape', '-s', nargs=3,
                      type=int, default=[28, 28, 1],
                      help=docs.image_shape_help)
  parser.add_argument('--l2-reg', nargs=1,
                      default=[None], type=float,
                      help=docs.l2_reg_help)
  parser.add_argument('--num-components', '--components', '-n', nargs=1,
                      default=[128], type=int,
                      help=docs.num_components_help)
  eval_time = parser.add_mutually_exclusive_group()
  eval_time.add_argument('--eval-secs', nargs=1,
                         default=[1200], type=int,
                         help=docs.eval_secs_help)
  eval_time.add_argument('--eval-mins', nargs=1,
                         default=[None], type=int,
                         help=docs.eval_mins_help)
  parser.add_argument('--splits', nargs='+',
                      default=[50000,10000,10000],
                      type=int,
                      help=docs.splits_help)
  parser.add_argument('--cores', '--num-parallel-calls', nargs=1,
                      default=[-1], type=int,
                      help=docs.cores_help)
  parser.add_argument('--batch-size', '-b', nargs=1,
                      default=[128], type=int,
                      help=docs.batch_size_help)
  parser.add_argument('--dropout', nargs=1,
                      default=[0.01], type=float,
                      help=docs.dropout_help)
  parser.add_argument('--activation', nargs=1,
                      choices=docs.activation_choices,
                      default=['sigmoid'],
                      help=docs.activation_help)
  parser.add_argument('--learning-rate', nargs=1,
                      default=[0.001],
                      help=docs.learning_rate_help)
  parser.add_argument('--momentum', nargs=1,
                      default=[0.9],
                      help=docs.momentum_help)
  parser.add_argument('--eager', action='store_true',
                      help=docs.eager_help)
  parser.add_argument('--verbose', '-v', nargs=1,
                      default=[1], type=int,
                      help=docs.verbose_help)

  args = parser.parse_args()

  if args.eager:
    tf.enable_eager_execution()

  if args.cores[0] == -1:
    args.cores[0] = os.cpu_count()
  if args.eval_mins[0] is not None:
      args.eval_secs[0] = args.eval_mins[0] * 60
  args.activation[0] = docs.activation_choices[args.activation[0]]

  if args.command == 'debug':
    cmd_debug(args)
  elif args.command == 'convert':
    cmd_convert(args)
  elif args.command == 'autoencoder':
    cmd_autoencoder(args)
  elif args.command == 'reconstruct':
    cmd_reconstruct(args)
  else:
    RuntimeError()


if __name__ == "__main__":
  main()
