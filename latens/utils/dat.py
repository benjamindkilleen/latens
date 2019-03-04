import tensorflow as tf
import numpy as np
import os
import logging

logger = logging.getLogger('latens')

def _bytes_feature(value):
  # Helper function for writing a string to a tfrecord
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  # Helper function for writing an array to a tfrecord
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _mnist_proto_from_example(example):
  """Create a tf proto from the image, label pair.

  Stores data in the desired format. Images are float32 in [0,1] and labels are
  int64.

  :param example: tuple containing image and label, as numpy arrays.

  """
  image, label = example
  image = np.atleast_3d(image)
  if image.dtype in [np.float32, np.float64]:
    image = image.astype(np.float32)
  elif image.dtype in [np.uint8, np.int32, np.int64]:
    image = image.astype(np.float32) / 255.
  else:
    raise NotImplementedError(f"Image has type {image.dtype}")

  image_string = image.tostring()
  image_shape = np.array(image.shape, dtype=np.int64)

  feature = {
    'image' : _bytes_feature(image_string),
    'image_shape' : _int64_feature(image_shape),
    'label' : _int64_feature([label])}

  features = tf.train.Features(feature=feature)
  example = tf.train.Example(features=features)
  return example.SerializeToString()

def _mnist_proto_from_tensors(image, label):
  raise NotImplementedError

def _mnist_example_from_proto(proto):
  """Convert a serialized example to an mnist tensor example."""

  features = tf.parse_single_example(
    proto,
    # Defaults are not specified since both keys are required.
    features={
      'image': tf.FixedLenFeature([], tf.string),
      'image_shape': tf.FixedLenFeature([3], tf.int64),
      'label' : tf.FixedLenFeature([1], tf.int64),
    })

  image = tf.decode_raw(features['image'], tf.float32)
  image_shape = (28,28,1)       # TODO: fix
  image = tf.reshape(image, image_shape, name='reshape_image_string')
  label = features['label']

  return image, label

def _dataset_from_tfrecord(data, **kwargs):
  raise NotImplementedError

def _dataset_from_arrays(features, labels, **kwargs):
  """Create an in-memory dataset.

  :param features: array of features
  :param labels: array of labels

  """
  assert features.shape[0] == labels.shape[0]
  
  features_placeholder = tf.placeholder(features.dtype, features.shape)
  labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
  return tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

def _dataset_from_npy(features_file, labels_file, **kwargs):
  """Create a tf.data.Dataset from data in numpy files.

  Assumes the data is large enough to fit in memory.
  TODO: allow otherwise.

  :param features_file: name of the .npy data file
  :param labels_file: name of the .npy labels file
  :returns: tf.data.Dataset
  """
  features = np.load(features_file)
  labels = np.load(labels_file)
  return _dataset_from_arrays(features, labels, **kwargs)

def convert_from_npz(fname, features_key='data', labels_key='labels',
                     proto_from_example=_mnist_proto_from_example, **kwargs):
  """Convert a dataset from an npz file containing data and labels to a tfrecord.

  :param fname: .npz file
  :param features_key: name of the key containing data
  :param labels_key: name of the key containing labels
  :param proto_from_example: function that encodes an (image,label) pair
  """
  
  with np.load(fname) as data:
    features = data[features_key]
    labels = data[labels_key]

  output_fname = os.path.splitext(fname)[0] + '.tfrecord'
  logger.info(f"Writing tfrecord to {output_fname}...")
  writer = tf.python_io.TFRecordWriter(output_fname)
  for example in zip(features, labels):
    writer.write(proto_from_example(example))

  writer.close()

def load_dataset(record_name,
                 parse_entry=_mnist_example_from_proto,
                 num_parallel_calls=None,
                 **kwargs):
  """Load the record_name as a tf.data.Dataset"""
  dataset = tf.data.TFRecordDataset(record_name)
  return dataset.map(parse_entry, num_parallel_calls=num_parallel_calls)

def save_dataset(record_name, dataset,
                 proto_from_example=_mnist_proto_from_example,
                 num_parallel_calls=None):
  """Save the dataset in tfrecord format

  :param record_name: filename to save to
  :param dataset: tf.data.Dataset to save
  :param proto_from_example: 
  :param num_parallel_calls: 
  :returns: 
  :rtype: 

  """
  next_example = dataset.make_one_shot_iterator().get_next()

  logger.info(f"writing dataset to {record_name}...")
  writer = tf.python_io.TFRecordWriter(record_name)
  with tf.Session() as sess:
    example = sess.run(next_example)
    writer.write(proto_from_example(example))

  writer.close()

  
class Data(object):
  def __init__(self, data, **kwargs):
    """Holds data. Subclass for data input or augmentation.

    :param data: tf.data.Dataset OR .tfrecord filename(s) (as a list) OR tuple
    of .npy file namess containing examples and labels OR .npz file containing
    data and labels.

    """

    if issubclass(type(data), Data):
      self._fname = data._dataset
    elif issubclass(type(data), tf.data.Dataset):
      self._dataset = data
    elif type(data) in [str, list, tuple]:
      # Loading tfrecord files
      self._dataset = load_dataset(data, **kwargs)
    else:
      raise ValueError(f"unrecognized data '{data}'")

    self.num_parallel_calls = kwargs.get('num_parallel_calls')
    self.num_components = kwargs.get('num_components', 10)
    self.num_classes = kwargs.get('num_classes', 10)
    self._kwargs = kwargs

  def save(self, record_name):
    save_dataset(record_name, self._dataset,
                 num_parallel_calls=self.num_parallel_calls)
    
  def __iter__(self):
    return self._dataset.__iter__()

  def postprocess(self, dataset):
    return dataset
    
  @property
  def dataset(self):
    return self.postprocess(self._dataset)

  @property
  def labeled(self):
    return self.postprocess(self._dataset.map(
      lambda x,y : (x, tf.reshape(tf.one_hot(y, self.num_classes), (-1,))),
      num_parallel_calls=self.num_parallel_calls))
  
  @property
  def encoded(self):
    return self.postprocess(self._dataset.map(
      lambda x,y : (x, tf.ones(self.num_components, dtype=x.dtype)),
      num_parallel_calls=self.num_parallel_calls))

  @property
  def self_supervised_encoded(self):
    return self.postprocess(self._dataset.map(
      lambda x,y : (x, (x, tf.ones(self.num_components, dtype=x.dtype))),
      num_parallel_calls=self.num_parallel_calls))
  
  def encode(self, n):
    return self.postprocess(self._dataset.map(
      lambda x,y : (x, tf.ones(n, x.dtype)),
      num_parallel_calls=self.num_parallel_calls))
  
  @property
  def supervised(self):
    return self.dataset

  @property
  def self_supervised(self):
    return self.postprocess(
      self._dataset.map(lambda x,y : (x,tf.identity(x)),
                        num_parallel_calls=self.num_parallel_calls))
        
  def split(self, *splits, types=None):
    """Split the dataset into different sets.

    Pass in the number of examples for each split. Returns a new Data object
    with datasets of the corresponding number of examples.

    :param types: optional list of Data subclasses to instantiate the splits
    :returns: list of Data objects with datasets of the corresponding sizes.
    :rtype: [Data]

    """
    
    datas = []
    for i, n in enumerate(splits):
      dataset = self._dataset.skip(sum(splits[:i])).take(n)
      if types is None or i >= len(types):
        datas.append(type(self)(dataset, **self._kwargs))
      elif types[i] is None:
        datas.append(None)
      else:
        datas.append(types[i](dataset, **self._kwargs))
      
    return datas

  def sample(self, sampling):
    """Draw a sampling from the dataset, returning a new dataset of the same type.

    :param sampling: 1-D boolean array, same size as dataset.
    :returns: new dataset with examples selected by sampling.
    :rtype: a Data subclass, same as self

    """
    sampling = tf.constant(sampling)
    dataset = self._dataset.apply(tf.data.experimental.enumerate_dataset())
    def map_func(idx, example):
      return tf.data.Dataset.from_tensors(example).repeat(sampling[idx])
    dataset = dataset.flat_map(map_func)
    return type(self)(dataset, **self._kwargs)

  
class DataInput(Data):
  def __init__(self, *args, **kwargs):
    """Used to feed a dataset into a model using the input_function attribute.

    This baseclass is good for eval and dev sets.

    """
    super().__init__(*args, **kwargs)
    self._prefetch_buffer_size = kwargs.get('prefetch_buffer_size', 1)
    self._batch_size = kwargs.get('batch_size', 1)

  def postprocess(self, dataset):
    return (dataset.repeat(-1)
            .batch(self._batch_size)
            .prefetch(self._prefetch_buffer_size))

  
class TrainDataInput(DataInput):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._num_shuffle = kwargs.get('num_shuffle', 100000)

  def postprocess(self, dataset):
    # TODO: allow for augmentation?
    return (dataset.shuffle(self._num_shuffle)
            .repeat(-1)
            .batch(self._batch_size)
            .prefetch(self._prefetch_buffer_size))

  
class DummyInput(DataInput):
  def __init__(self, image_shape, **kwargs):
    """Creates a dummy set to pass through before loading a keras model.

    :param image_shape: shape of the input image.
    :param batch_size: should always be provided, if model is batched.

    """
    tensors = (tf.ones(image_shape, dtype=tf.float32),
               tf.ones(1, dtype=tf.float32))
    dummyset = tf.data.Dataset.from_tensors(tensors)
    super().__init__(dummyset, **kwargs)
