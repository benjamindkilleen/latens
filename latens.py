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
from latens.utils import docs, dat
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
  train_set = dat.TrainDataInput(args.input, num_parallel_calls=args.cores[0])
  # train_set, dev_set, eval_set = data.split(
  #   *args.splits, types=[dat.TrainDataInput, dat.DataInput, dat.DataInput])

  model = mod.ConvAutoEncoder(
    args.image_shape,
    num_components=args.num_components[0],
    model_dir=args.model_dir[0],
    l2_reg=args.l2_reg[0])

  model.train(
    args.input,
    overwrite=args.overwrite,
    num_epochs=args.epochs[0],
    eval_secs=args.eval_secs[0])

def cmd_predict(args):
  """Run prection."""
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
  parser.add_argument('--model-dir', '-m', nargs=1,
                      default=[None],
                      help=docs.model_dir_help)
  parser.add_argument('--epochs', '-e', nargs=1,
                      default=[-1], type=int,
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
                      default=[2], type=int,
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
  
  args = parser.parse_args()

  if args.cores[0] == -1:
    args.cores[0] = os.cpu_count()
  if args.eval_mins[0] is not None:
      args.eval_secs[0] = args.eval_mins[0] * 60

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
