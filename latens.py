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
from latens import mod, sam
from shutil import rmtree
from time import time
import numpy as np
import matplotlib.pyplot as plt

if sys.version_info < (3,6):
  logger.error(f"Use python{3.6} or higher.")
  exit()

logger.info(f"tensorflow {tf.__version__}, keras {tf.keras.__version__}")


class Latens:
  def __init__(self, args):
    """Bag of state for the latens run."""

    # general
    self.autoencoder_type = args.autoencoder_type[0]
    self.cores = args.cores[0]
    if self.cores == -1:
      self.cores = os.cpu_count()
    self.verbose = args.verbose[0]
    self.keras_verbose = args.keras_verbose[0]
    self.tensorboard = args.tensorboard

    # num_somethings
    self.latent_dim = args.latent_dim[0]
    if self.autoencoder_type == 'vae':
      self.num_components = 2*self.latent_dim
    else:
      self.num_components = self.latent_dim
    self.num_classes = args.num_classes[0]
    self.epochs = args.epochs[0]
    self.batch_size = args.batch_size[0]
    
    # sampling, etc
    self.sample_size = args.sample_size[0]
    self.sample = args.sample[0]
    self.sampler_type = docs.sample_choices[self.sample]

    # dataset sizes
    self.splits = args.splits
    self.train_size = args.splits[0]
    self.tune_size = args.splits[1]
    self.test_size = args.splits[2]

    # number of steps for different iterations
    self.epoch_multiplier = args.epoch_multiplier[0]
    self.train_steps = int(np.ceil(
      self.epoch_multiplier * self.train_size / self.batch_size))
    self.single_train_steps = int(np.ceil(self.train_size / self.batch_size))
    self.tune_steps = int(np.ceil(self.tune_size / self.batch_size))
    self.test_steps = int(np.ceil(self.test_size / self.batch_size))
    self.sample_steps = int(np.ceil(
      self.epoch_multiplier * self.sample_size / self.batch_size))

    # model arguments
    self.learning_rate = args.learning_rate[0]
    self.overwrite = args.overwrite
    self.rep_activation = docs.rep_activation_choices[args.rep_activation[0]]
    self.image_shape = args.image_shape
    self.dropout = args.dropout[0]

    # the full classifier
    self.full_classifier_dir = 'models/full_classifier'
    if not os.path.exists(self.full_classifier_dir):
      os.mkdir(self.full_classifier_dir)
    
    # model dir and dependent data
    self.model_root = args.model_root[0]
    if not os.path.exists(self.model_root):
      os.mkdir(self.model_root)
    self.autoencoder_dir = os.path.join(self.model_root, 'autoencoder')
    self.classifier_dir = os.path.join(self.model_root, 'classifier')
    self.encodings_path = os.path.join(self.model_root, 'encodings.npy')
    self.sampling_path = os.path.join(
      self.model_root, f'{self.sample}_sampling_{self.sample_size}.npy')
    self.sample_path = os.path.join(
      self.model_root, f'{self.sample}_sample_{self.sample_size}.tfrecord')
    self.cluster_labels_path = os.path.join(
      self.model_root, f'{self.sample}_cluster_labels_{self.sample_size}.npy')

    # figure paths
    self.encodings_fig_path = os.path.join(
      self.model_root, f'encodings.pdf')
    self.clustered_encodings_fig_path = os.path.join(
      self.model_root, f'{self.sample}_clustered_encodings.pdf')
    self.sampling_fig_path = os.path.join(
      self.model_root, f'{self.sample}_sampling_{self.sample_size}.pdf')
    self.sampling_distribution_fig_path = os.path.join(
      self.model_root, f'{self.sample}_sampling_distribution_{self.sample_size}.pdf')

    
    # input tfrecord prefix and its derivatices
    self.input_prefix, _ = os.path.splitext(args.input[0])
    self.npz_path = self.input_prefix + '.npz'
    self.data_path = self.input_prefix + '.tfrecord'

    # output files, mainly for visualization
    self.show = args.show
    self.output = args.output[0]

  def load_train(self):
    data = np.load(self.npz_path)
    return data['data'][:self.train_size] / 255., data['labels'][:self.train_size]
    
  def make_data(self, training=True):
    """Make the train, test, and split sets.

    :returns: train, test, and split sets
    :rtype: dat.TrainDataInput, dat.DataInput, dat.DataInput

    """
    if training:
      types = [dat.TrainDataInput, dat.DataInput, dat.DataInput]
    else:
      types = [dat.DataInput, dat.DataInput, dat.DataInput]
    data = dat.Data(self.data_path,
                    num_parallel_calls=self.cores,
                    batch_size=self.batch_size,
                    num_classes=self.num_classes,
                    num_components=self.num_components)
    return data.split(
    *self.splits, types=types)
  
  def make_sample_data(self):
    return dat.TrainDataInput(
      self.sample_path,
      num_parallel_calls=self.cores,
      batch_size=self.batch_size,
      num_classes=self.num_classes,
      num_components=self.num_components)
  
  def make_autoencoder(self):
    if self.autoencoder_type == 'student':
      model = mod.StudentAutoEncoder(
        self.image_shape,
        self.latent_dim,
        self.batch_size,
        model_dir=self.autoencoder_dir,
        rep_activation=self.rep_activation,
        dropout=self.dropout,
        tensorboard=self.tensorboard)
    elif self.autoencoder_type == 'vaestudent':
      model = mod.VariationalStudentAutoEncoder(
        self.image_shape,
        self.latent_dim,
        self.batch_size,
        model_dir=self.autoencoder_dir,
        rep_activation=self.rep_activation,
        dropout=self.dropout,
        tensorboard=self.tensorboard)
    elif self.autoencoder_type == 'vae':
      model = mod.ConvVariationalAutoEncoder(
        self.image_shape,
        self.latent_dim,
        model_dir=self.autoencoder_dir,
        rep_activation=self.rep_activation,
        dropout=self.dropout,
        tensorboard=self.tensorboard)
    elif self.autoencoder_type == 'conv':      
      model = mod.ConvAutoEncoder(
        self.image_shape,
        self.latent_dim,
        model_dir=self.autoencoder_dir,
        rep_activation=self.rep_activation,
        dropout=self.dropout,
        tensorboard=self.tensorboard)
    else:
      raise RuntimeError

    model.compile(learning_rate=self.learning_rate)
    return model

  def classifier_from_autoencoder(self, autoencoder):
    model = mod.Classifier.from_autoencoder(
      autoencoder, self.num_classes,
      model_dir=self.classifier_dir,
      tensorboard=self.tensorboard)
    
    model.compile(learning_rate=self.learning_rate)
    return model

  def make_full_classifier(self):
    model = mod.SimpleClassifier(
      self.image_shape,
      self.num_classes,
      model_dir=self.full_classifier_dir,
      dropout=self.dropout,
      tensorboard=self.tensorboard)

    model.compile(learning_rate=self.learning_rate)
    return model
    
    
def cmd_debug(lat):
  """Run the debug command."""
  if type(args.sample[0]) == str:
    sample = np.load(args.sample[0])
  else:
    points = np.load(args.input[0])
    sampler = args.sample[0](sample_size=args.sample_size[0])
    sample = sampler(points)
    logger.debug(f"sample: {sample}")
    logger.debug(f"drew {np.sum(sample)} new points")
    if args.output[0] is not 'show':
      np.save(args.output[0], sample)
    exit()

  data = dat.Data(args.input, num_parallel_calls=args.cores[0],
                  batch_size=args.batch_size[0],
                  num_classes=args.num_classes[0],
                  num_components=args.num_components[0])
  train_set, tune_set, test_set = data.split(
    *args.splits, types=[dat.TrainDataInput, dat.DataInput, dat.DataInput])
  
  sample_train_set = train_set.sample(sample)
  sample_size = np.sum(sample)
  
  classifier = mod.ConvClassifier(
    args.image_shape,
    args.num_classes[0],
    model_dir=args.model_dir[0],
    dropout=args.dropout[0])

  classifier.compile(args.learning_rate[0])

  if not args.overwrite:
    classifier.load()

  classifier.fit(
    sample_train_set.labeled,
    epochs=args.epochs[0],
    steps_per_epoch=sample_size // args.batch_size[0] + 1,
    validation_data=tune_set.labeled,
    validation_steps=args.splits[1] // args.batch_size[0],
    verbose=args.keras_verbose[0])    

  loss, accuracy = classifier.evaluate(
    test_set.labeled,
    steps = args.splits[2] // args.batch_size[0])
  logger.info(f'test accuracy: {accuracy:.03f}')
  
  
def cmd_convert(lat):
  """Convert the dataset in args.input[0] to tfrecord and store in the same
  directory as a .tfrecord file."""
  dat.convert_from_npz(lat.npz_path)
  
  
def cmd_autoencoder(lat):
  """Run training for the autoencoder."""
  train_set, tune_set, test_set = lat.make_data()

  model = lat.make_autoencoder()
  if not lat.overwrite:
    model.load()

  model.fit(
    train_set.self_supervised,
    epochs=lat.epochs,
    steps_per_epoch=lat.train_steps,
    validation_data=tune_set.self_supervised,
    validation_steps=lat.tune_steps,
    verbose=lat.keras_verbose)


def cmd_reconstruct(lat):
  """Run reconstruction."""
  train_set, tune_set, test_set = lat.make_data(training=False)

  model = lat.make_autoencoder()
  model.load()

  reconstructions = model.predict(test_set.self_supervised, steps=1, verbose=1)
  get_next = test_set.get_next
  with tf.Session() as sess:
    for i, reconstruction in enumerate(reconstructions):
      image, label = sess.run(get_next)
      vis.plot_image(image, reconstruction)
      if lat.show:
        plt.show()
      else:
        plt.savefig(os.path.join(lat.model_root, f'reconstruction_{i}.pdf'))
        if i > 5:
          break
        
def cmd_encode(lat):
  """Encodes the training set."""
  train_set, tune_set, test_set = lat.make_data(training=False)

  model = lat.make_autoencoder()
  model.load()

  encodings = model.encode(
    train_set.encoded,
    steps=lat.single_train_steps,
    verbose=1)[:lat.train_size]
  
  logger.debug(f"encodings:\n{encodings}")
  
  if lat.show:
    vis.show_encodings(encodings)
    
  np.save(lat.encodings_path, encodings)
  logger.info(f"saved encodings to '{lat.encodings_path}'")

def cmd_decode(lat):
  """Decode a numpy array of encodings from args.input and show."""
  encodings = np.load(lat.encodings_path)
  logger.info(f"loaded encodings from '{lat.encodings_path}'")

  model = lat.make_autoencoder()
  model.load()
  
  reconstructions = model.decode(encodings[:lat.batch_size], verbose=1,
                                 batch_size=lat.batch_size)
  if tf.executing_eagerly():
    for (original, _), reconstruction in zip(test_set, reconstructions):
      vis.show_image(original, reconstruction)
  else:
    for i in range(reconstructions.shape[0]):
      vis.show_image(reconstructions[i])

def cmd_sample(lat):
  """Run sampling on the encoding (assumed to exist) and store in a new tfrecord
  file."""
  train_set, tune_set, test_set = lat.make_data(training=False)

  if lat.sample == 'error':
    model = lat.make_autoencoder()
    model.load()
    encodings = model.errors(train_set.self_supervised, lat.train_steps,
                             lat.batch_size)[:lat.train_size]
  elif lat.sample in ['classifier-error', 'classifier-losses',
                      'classifier-incorrect']:
    model = lat.make_full_classifier()
    model.load()
    if lat.sample == 'classifier-error':
      encodings = model.errors(train_set.labeled, lat.train_steps,
                               lat.batch_size)[:lat.train_size]
    elif lat.sample == 'classifier-losse':
      encodings = model.losses(train_set.labeled, lat.train_steps,
                               lat.batch_size)[:lat.train_size]
    elif lat.sample == 'classifier-incorrect':
      encodings = model.incorrect(train_set.labeled, lat.train_steps,
                                  lat.batch_size)[:lat.train_size]    
  else:
    encodings = np.load(lat.encodings_path)
    
  sampler = lat.sampler_type(sample_size=lat.sample_size)
  if issubclass(lat.sampler_type, sam.ClusterSampler):
    sampler.n_clusters = lat.num_classes
  sampling = sampler(encodings)
  logger.debug(f"sampling: {sampling.shape}, {np.sum(sampling)}")

  np.save(lat.sampling_path, sampling)
  if issubclass(lat.sampler_type, sam.ClusterSampler):
    np.save(lat.cluster_labels_path, sampler.cluster_labels)

  sample_set = train_set.sample(sampling)
  sample_set.save(lat.sample_path)

  
def cmd_classifier(lat):
  """Run training from scratch for a classifier, using ."""
  train_set, tune_set, test_set = lat.make_data()
  sample_set = lat.make_sample_data()

  autoencoder = lat.make_autoencoder()
  
  classifier = lat.classifier_from_autoencoder(autoencoder)
  if not lat.overwrite:
    classifier.load()

  classifier.fit(
    sample_set.labeled,
    epochs=lat.epochs,
    steps_per_epoch=lat.sample_steps,
    validation_data=tune_set.labeled,
    validation_steps=lat.tune_steps,
    verbose=lat.keras_verbose)

  loss, accuracy = classifier.evaluate(
    test_set.labeled,
    steps=lat.test_steps)
  logger.info(f'test accuracy: {100*accuracy:.01f}%')

def cmd_full_classifier(lat):
  """Run training for a classifier on the full dataset."""
  train_set, tune_set, test_set = lat.make_data()
  classifier = lat.make_full_classifier()
  if not lat.overwrite:
    classifier.load()

  classifier.fit(
    train_set.labeled,
    epochs=lat.epochs,
    steps_per_epoch=lat.train_steps,
    validation_data=tune_set.labeled,
    validation_steps=lat.tune_steps,
    verbose=lat.keras_verbose)

  loss, accuracy = classifier.evaluate(
    test_set.labeled,
    steps=lat.test_steps)
  logger.info(f'test accuracy: {100*accuracy:.01f}%')
  

def cmd_visualize(lat):
  """Visualize the decodings that the model makes."""
  images, labels = lat.load_train()

  if os.path.exists(lat.encodings_path):
    logger.info("Plotting encoding...")
    encodings = np.load(lat.encodings_path)
    logger.debug(f"encodings: {encodings}")
    if False and lat.latent_dim == 3:
      vis.plot_encodings_3d(encodings, labels=labels)
    else:
      vis.plot_encodings(encodings, labels=labels)
    if not lat.show:
      plt.savefig(lat.encodings_fig_path)    

  if (os.path.exists(lat.encodings_path) and
      os.path.exists(lat.cluster_labels_path)):
    logger.info("Plotting clusters...")
    encodings = np.load(lat.encodings_path)
    cluster_labels = np.load(lat.cluster_labels_path)
    if False and lat.latent_dim == 3:
      vis.plot_encodings_3d(encodings, labels=cluster_labels)
    else:
      vis.plot_encodings(encodings, labels=cluster_labels)
    plt.title("Clustered Encodings")
    if not lat.show:
      plt.savefig(lat.clustered_encodings_fig_path)
      
  if (os.path.exists(lat.encodings_path) and
      os.path.exists(lat.sampling_path)):
    logger.info("Plotting sampling...")
    encodings = np.load(lat.encodings_path)
    sampling = np.load(lat.sampling_path)
    vis.plot_sampled_encodings(encodings, sampling, labels=labels)
    if not lat.show:
      plt.savefig(lat.sampling_fig_path)
    vis.plot_sampling_distribution(sampling, labels)
    if not lat.show:
      plt.savefig(lat.sampling_distribution_fig_path)
    
                  
  if lat.show:
    plt.show()

    
def main():
  parser = argparse.ArgumentParser(description=docs.description)
  parser.add_argument('command', choices=docs.command_choices,
                      help=docs.command_help)
  parser.add_argument('--input', '-i', nargs=1, 
                      default=['data/mnist/mnist'],
                      help=docs.input_help)
  parser.add_argument('--output', '-o', nargs=1,
                      default=['docs'],
                      help=docs.output_help)
  parser.add_argument('--show', action='store_true',
                      help=docs.show_help)
  parser.add_argument('--model-root', '--model-dir', '-m', nargs=1,
                      default=['models/tmp'], help=docs.model_dir_help)
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
  parser.add_argument('--latent-dim', '-L', nargs=1,
                      default=[2], type=int,
                      help=docs.latent_dim_help)
  parser.add_argument('--splits', nargs=3,
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
                      default=['None'],
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
  parser.add_argument('--sample-size', nargs=1,
                      default=[1000], type=int,
                      help=docs.sample_size_help)
  parser.add_argument('--sample', nargs=1, choices=docs.sample_choices,
                      default=['error'],
                      help=docs.sample_help)
  parser.add_argument('--epoch-multiplier', '--mult', nargs=1,
                      default=[1], type=int,
                      help=docs.epoch_multiplier_help)
  parser.add_argument('--autoencoder-type', '--ae', nargs=1,
                      default=['conv'],
                      choices=docs.autoencoder_type_choices,
                      help=docs.autoencoder_type_help)

  args = parser.parse_args()

  # Must be handled on initialization
  if args.verbose[0] == 0:
    logger.setLevel(logging.WARNING)
  elif args.verbose[0] == 1:
    logger.setLevel(logging.INFO)
  else:
    logger.setLevel(logging.DEBUG)

  if args.eager: # or args.command == 'reconstruct':
    tf.enable_eager_execution()

  lat = Latens(args)

  if args.command == 'convert':
    cmd_convert(lat)
  elif args.command == 'autoencoder':
    cmd_autoencoder(lat)
  elif args.command == 'encode':
    cmd_encode(lat)
  elif args.command == 'sample':
    cmd_sample(lat)
  elif args.command == 'classifier':
    cmd_classifier(lat)
  elif args.command == 'full-classifier':
    cmd_full_classifier(lat)
  elif args.command == 'reconstruct':
    cmd_reconstruct(lat)
  elif args.command == 'decode':
    cmd_decode(lat)
  elif args.command == 'visualize':
    cmd_visualize(lat)
  elif args.command == 'debug':
    cmd_debug(lat)
  else:
    RuntimeError()


if __name__ == "__main__":
  main()
