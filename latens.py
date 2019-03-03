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
from latens.utils import docs, dat, vis, misc
from latens import mod, sam
from shutil import rmtree
from time import time
import numpy as np
import matplotlib.pyplot as plt

if sys.version_info < (3,6):
  logger.error(f"Use python{3.6} or higher.")
  exit()

logger.info(f"tensorflow {tf.__version__}, keras {tf.keras.__version__}")

def cmd_debug(args):
  """Run the debug command."""

  if type(args.sample[0]) == str:
    sampling = np.load(args.sample[0])
  else:
    points = np.load(args.input[0])
    sampler = args.sample[0](num_examples=args.num_examples[0])
    sampling = sampler(points)
    logger.debug(f"sampling: {sampling}")
    logger.debug(f"drew {np.sum(sampling)} new points")
    if args.output[0] is not 'show':
      np.save(args.output[0], sampling)
    exit()

  data = dat.Data(args.input, num_parallel_calls=args.cores[0],
                  batch_size=args.batch_size[0],
                  num_classes=args.num_classes[0],
                  num_components=args.num_components[0])
  train_set, tune_set, test_set = data.split(
    *args.splits, types=[dat.TrainDataInput, dat.DataInput, dat.DataInput])
  
  sampled_train_set = train_set.sample(sampling)
  sampled_size = np.sum(sampling)
  
  classifier = mod.ConvClassifier(
    args.image_shape,
    args.num_classes[0],
    model_dir=args.model_dir[0],
    dropout=args.dropout[0])

  classifier.compile(args.learning_rate[0])

  if not args.overwrite:
    classifier.load()

  classifier.fit(
    sampled_train_set.labeled,
    epochs=args.epochs[0],
    steps_per_epoch=sampled_size // args.batch_size[0] + 1,
    validation_data=tune_set.labeled,
    validation_steps=args.splits[1] // args.batch_size[0],
    verbose=args.keras_verbose[0])    

  loss, accuracy = classifier.evaluate(
    test_set.labeled,
    steps = args.splits[2] // args.batch_size[0])
  logger.info(f'test accuracy: {accuracy:.03f}')
  
  
def cmd_convert(args):
  """Convert the dataset in args.input[0] to tfrecord and store in the same
  directory as a .tfrecord file."""
  dat.convert_from_npz(args.input[0])


def cmd_autoencoder(args):
  """Run training for the autoencoder."""
  data = dat.DataInput(args.input, num_parallel_calls=args.cores[0],
                       batch_size=args.batch_size[0],
                       num_components=args.num_components[0])
  train_set, tune_set = data.split(
    *args.splits, types=[dat.TrainDataInput, dat.DataInput])[:2]
  
  model = mod.ConvAutoEncoder(
    args.image_shape,
    args.num_components[0],
    model_dir=args.model_dir[0],
    rep_activation=args.rep_activation[0],
    dropout=args.dropout[0])
      
  model.compile(learning_rate=args.learning_rate[0])

  if not args.overwrite:
    model.load()

  model.fit(
    train_set.self_supervised,
    epochs=args.epochs[0],
    steps_per_epoch=args.splits[0] // args.batch_size[0],
    validation_data=tune_set.self_supervised,
    validation_steps=args.splits[1] // args.batch_size[0],
    verbose=args.keras_verbose[0])

def cmd_reconstruct(args):
  """Run reconstruction."""
  data = dat.Data(args.input, num_parallel_calls=args.cores[0],
                  batch_size=args.batch_size[0],
                  num_components=args.num_components[0],
                  num_classes=args.num_classes[0])
  train_set, tune_set, test_set = data.split(
    *args.splits, types=[dat.TrainDataInput, dat.DataInput, dat.DataInput])

  model = mod.ConvAutoEncoder(
    args.image_shape,
    args.num_components[0],
    model_dir=args.model_dir[0],
    rep_activation=args.rep_activation[0],
    dropout=args.dropout[0])
  
  model.compile(learning_rate=args.learning_rate[0])

  model.load()

  reconstructions = model.predict(test_set.self_supervised, steps=1,
                                  verbose=1)
  if tf.executing_eagerly():
    for (original, _), reconstruction in zip(test_set, reconstructions):
      vis.show_image(original, reconstruction)
  else:
    for i in range(reconstructions.shape[0]):
      vis.show_image(reconstructions[i])

def cmd_encode(args):
  """Encodes the training set."""
  data = dat.Data(args.input, num_parallel_calls=args.cores[0],
                  num_components=args.num_components[0],
                  num_classes=args.num_classes[0],
                  batch_size=args.batch_size[0])
  train_set, tune_set, test_set = data.split(
    *args.splits, types=[dat.TrainDataInput, dat.DataInput, dat.DataInput])

  model = mod.ConvAutoEncoder(
    args.image_shape,
    args.num_components[0],
    model_dir=args.model_dir[0],
    rep_activation=args.rep_activation[0],
    dropout=args.dropout[0])
      
  model.compile(learning_rate=args.learning_rate[0])

  model.load()

  encodings = model.encode(
    train_set.encode(args.num_components[0]),
    steps=args.splits[0] // args.batch_size[0] + 1,
    verbose=1)[:args.splits[0]]
  
  logger.debug(f"encodings:\n{encodings}")
  if args.output[0] is not None:
    if args.output[0] == 'show':
      vis.show_encodings(encodings)
    filename, ext = os.path.splitext(args.output[0])
    if ext == '.npy':
      np.save(args.output[0], encodings)
      logger.info(f"saved encodings to '{args.output[0]}'")

def cmd_decode(args):
  """Decode a numpy array of encodings from args.input and show."""
  encodings = np.load(args.input[0])
  logger.info(f"loaded encodings from '{args.input[0]}'")

  model = mod.ConvAutoEncoder(
    args.image_shape,
    args.num_components[0],
    model_dir=args.model_dir[0],
    rep_activation=args.rep_activation[0],
    dropout=args.dropout[0])
      
  model.compile(learning_rate=args.learning_rate[0])

  model.load()
  
  reconstructions = model.decode(encodings[:args.batch_size[0]], verbose=1,
                                 batch_size=args.batch_size[0])
  if tf.executing_eagerly():
    for (original, _), reconstruction in zip(test_set, reconstructions):
      vis.show_image(original, reconstruction)
  else:
    for i in range(reconstructions.shape[0]):
      vis.show_image(reconstructions[i])

def cmd_visualize(args):
  """Visualize the decodings that the model makes."""
  model = mod.ConvAutoEncoder(
    args.image_shape,
    args.num_components[0],
    model_dir=args.model_dir[0],
    rep_activation=args.rep_activation[0],
    dropout=args.dropout[0])
  model.compile(learning_rate=args.learning_rate[0])
  model.load()

  rows = args.num_components[0]
  cols = 20
  points = 0.3 * np.ones((rows, cols, args.num_components[0]), dtype=np.float32)
  for i in range(points.shape[0]):
    points[i,:,i] = np.linspace(0, 1.0, num=cols, dtype=np.float32)

  images = model.decode(points.reshape(-1, args.num_components[0]), verbose=1,
                        batch_size=args.batch_size[0])

  if args.output[0] == 'show':
    vis.show_image(*images, columns=cols)
  else:
    vis.plot_image(*images, columns=cols)
    plt.savefig(args.output[0])

def cmd_classifier(args):
  """Run training from scratch for a classifier."""
  raise NotImplementedError
  
def main():
  parser = argparse.ArgumentParser(description=docs.description)
  parser.add_argument('command', choices=docs.command_choices,
                      help=docs.command_help)
  parser.add_argument('--input', '-i', nargs='+',
                      default=[None],
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
                      default=[64], type=int,
                      help=docs.batch_size_help)
  parser.add_argument('--dropout', nargs=1,
                      default=[0.2], type=float,
                      help=docs.dropout_help)
  parser.add_argument('--rep-activation', nargs=1,
                      choices=docs.rep_activation_choices,
                      default=['clu'],
                      help=docs.rep_activation_help)
  parser.add_argument('--learning-rate', '-l', nargs=1,
                      default=[0.1], type=float,
                      help=docs.learning_rate_help)
  parser.add_argument('--momentum', nargs=1,
                      default=[0.9], type=float,
                      help=docs.momentum_help)
  parser.add_argument('--eager', action='store_true',
                      help=docs.eager_help)
  parser.add_argument('--keras-verbose', nargs=1,
                      default=[1], type=int,
                      help=docs.keras_verbose_help)
  parser.add_argument('--verbose', '-v', nargs=1,
                      default=[2], type=int,
                      help=docs.verbose_help)
  parser.add_argument('--tensorboard', action='store_true',
                      help=docs.tensorboard_help)
  parser.add_argument('--load', action='store_true',
                      help=docs.load_help)
  parser.add_argument('--num-classes', nargs=1,
                      default=[10], type=int,
                      help=docs.num_classes_help)
  parser.add_argument('--num-examples', nargs=1,
                      default=[1000], type=float,
                      help=docs.num_examples_help)
  parser.add_argument('--sample', nargs=1,
                      default=['random'],
                      help=docs.sample_help)

  args = parser.parse_args()

  if args.verbose[0] == 0:
    logger.setLevel(logging.WARNING)
  elif args.verbose[0] == 1:
    logger.setLevel(logging.INFO)
  else:
    logger.setLevel(logging.DEBUG)

  if args.eager: # or args.command == 'reconstruct':
    tf.enable_eager_execution()

  if args.cores[0] == -1:
    args.cores[0] = os.cpu_count()
  if args.eval_mins[0] is not None:
    args.eval_secs[0] = args.eval_mins[0] * 60

  # Take care of mappings
  args.rep_activation[0] = docs.rep_activation_choices[args.rep_activation[0]]
  if args.sample[0] in docs.sampler_choices:
    # otherwise, sampling is a numpy file containing the sampling
    args.sample[0] = docs.sampler_choices[args.sample[0]]
    

  if args.input[0] is None and args.command != 'visualize':
    logger.warning("no input provided")

  if args.command == 'debug':
    cmd_debug(args)
  elif args.command == 'convert':
    cmd_convert(args)
  elif args.command == 'autoencoder':
    cmd_autoencoder(args)
  elif args.command == 'reconstruct':
    cmd_reconstruct(args)
  elif args.command == 'encode':
    cmd_encode(args)
  elif args.command == 'decode':
    cmd_decode(args)
  elif args.command == 'visualize':
    cmd_visualize(args)
  elif args.command == 'classifier':
    cmd_classifier(args)
  else:
    RuntimeError()


if __name__ == "__main__":
  main()
