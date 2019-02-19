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

def cmd_train(args):
  """Run training."""
  data = dat.DataInput(args.input, num_parallel_calls=args.cores[0],
                       batch_size=args.batch_size[0])
  train_set, validation_set = data.split(
    *args.splits, types=[dat.TrainDataInput, dat.DataInput])[:2]

  model = mod.ConvAutoEncoder(
    args.image_shape,
    args.num_components[0],
    l2_reg=args.l2_reg[0],
    activation=args.activation[0],
    dropout=args.dropout[0])

  if args.model_path[0] is not None and len(glob(args.model_path[0] + "*")) > 0:
    model.load_weights(args.model_path[0])

  # TODO: allow customize
  model.compile(
    optimizer=tf.train.MomentumOptimizer(args.learning_rate[0],
                                         args.momentum[0]),
    loss=tf.losses.mean_squared_error,
    metrics=['mae'])

  model.fit(
    train_set.self_supervised,
    batch_size=args.batch_size[0],
    epochs=args.epochs[0],
    steps_per_epoch=args.splits[0],
    validation_data=validation_set.self_supervised)

  if args.model_path[0] is not None:
    # TODO: add overwrite protection
    model.save_weights(args.model_path)
  
def cmd_predict(args):
  """Run prection."""
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

  assert args.model_path[0] is not None and len(glob(args.model_path[0] + "*")) > 0
  model.load_weights(args.model_path[0])

  predictions = model.predict_generator(test_set.self_supervised)
  for example, prediction in zip(predictions, test_set):
    image, label = example
    vis.show(image, prediction)

  raise NotImplementedError

  
def main():
  parser = argparse.ArgumentParser(description=docs.description)
  parser.add_argument('command', choices=docs.command_choices,
                      help=docs.command_help)
  parser.add_argument('--input', '-i', nargs='+', required=True,
                      help=docs.input_help)
  parser.add_argument('--output', '-o', nargs=1,
                      default=['show'],
                      help=docs.output_help)
  parser.add_argument('--model-path', '-m', nargs=1,
                      default=[None],
                      help=docs.model_path_help)
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
  parser.add_argument('--num-components', '--classes', '-c', nargs=1,
                      default=[10], type=int,
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
                      default=[16], type=int,
                      help=docs.batch_size_help)
  parser.add_argument('--dropout', nargs=1,
                      default=[0.1], type=float,
                      help=docs.dropout_help)
  parser.add_argument('--activation', nargs=1,
                      choices=docs.activation_choices,
                      default=['sigmoid'],
                      help=docs.activation_help)
  parser.add_argument('--learning-rate', nargs=1,
                      default=[0.01],
                      help=docs.learning_rate_help)
  parser.add_argument('--momentum', nargs=1,
                      default=[0.9],
                      help=docs.learning_rate_help)
  
  args = parser.parse_args()

  if args.cores[0] == -1:
    args.cores[0] = os.cpu_count()
  if args.eval_mins[0] is not None:
      args.eval_secs[0] = args.eval_mins[0] * 60
  args.activation[0] = docs.activation_choices[args.activation[0]]

  if args.command == 'debug':
    cmd_debug(args)
  elif args.command == 'convert':
    cmd_convert(args)
  elif args.command == 'train':
    cmd_train(args)
  elif args.command == 'predict':
    cmd_predict(args)
  else:
    RuntimeError()


if __name__ == "__main__":
  main()
