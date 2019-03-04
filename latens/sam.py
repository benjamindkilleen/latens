"""Sample a set of points in a low-dimensional embedding, according to some
distribution. Points must fit in memory.

"""

import numpy as np
import scipy
import logging

logger = logging.getLogger('latens')

class Sampler:
  def __init__(self, sample_size=None):
    """A callable that returns an integer array indexing the points given to it.

    :param sample_size: number of examples that this sampler takes. Not used by
    every Sampler, but included for consistency. Can be a fraction, in which
    case len(points) * sample_size are taken from points. Can be None, in
    which case all points are taken.
    :returns: 
    :rtype: 

    """
    self.sample_size = sample_size

  def get_sample_size(self, N):
    """Get the number of examples to take from a set of points.

    :param N: number of original points
    :returns: sample_size
    :rtype: 

    """
    if self.sample_size is None:
      return int(N)
    elif self.sample_size >= 1:
      return int(self.sample_size)
    elif 0 <= self.sample_size < 1:
      return int(np.floor(self.sample_size * N))
    else:
      raise ValueError(f'unrecognized sample_size: {self.sample_size}')
    
  def __call__(self, *args, **kwargs):
    """Wrapper around self.sample()"""
    return self.sample(*args, **kwargs)

  def sample(self, points):
    """Select some subset of points.

    Returns an array indexing into points. Specifies number of times to repeat
    each example (could be interpreted as a boolean array).

    :param points: points to sample
    :returns: integer or boolean array for indexing points, i.e. a "sampling"
    :rtype: 1-D array

    """
    return np.ones(points.shape[0], dtype=np.int64)


class RandomSampler(Sampler):
  def sample(self, points):
    N = points.shape[0]
    n = self.get_sample_size(N)
    perm = np.random.permutation(N)
    sampling = np.zeros(N, dtype=np.int64)
    sampling[perm[:n]] = 1
    return sampling

  
class UniformSampler(Sampler):
  def __init__(self, low=0.0, high=1.0, threshold=0.2,
               metric='euclidean', **kwargs):
    """Samples uniformly over the space covered by points within [low,hi].

    :param low: ignored
    :param high: ignored
    :param threshold: distance threshold at which to resample.
    :param metric: passed to scipy.spatial.distance.cdist

    """
    self.low = low
    self.high = high
    self.threshold = threshold
    self.metric = metric
    super().__init__(**kwargs)
    
  def sample(self, points):
    """Sample according to uniform sampling.

    For each randomly sampled point, finds the closest point from points,
    according the given distance metric. For each of these steps, calculates nxN
    pairwise distances, pretty inefficiently.
    
    Draws points in the uniform distribution in each component x from
    `[min(x) - thresh, max(x) + thresh]`

    TODO: use an Approximate Nearest Neighbor algorithm instead
    
    :param points: points to sample
    :returns: integer array for indexing points, i.e. a "sampling"
    :rtype: 1-D integer array

    """
    
    N = points.shape[0]
    n = self.get_sample_size(N) # number of examples to add to sampling
    sampling = np.zeros(N, dtype=np.int64) # starts with zero examples
    
    while n > 0:
      logger.debug(f"sampler: drawing {n} points")
      draws = np.random.uniform(self.low, self.high, size=(n, points.shape[1]))
      distances = scipy.spatial.distance.cdist(
        draws, points, metric=self.metric) # (n,N) array of distances

      # index of closest point to each new draw
      closest_indices = np.argmin(distances, axis=1) 
      
      # distance to closest point for each new draw
      closest_distances = np.min(distances, axis=1) # len == n
      # logger.debug(f"closest distances: {closest_distances}")
      
      # indices to include in the sampling (with possible repetitions)
      indices = closest_indices[closest_distances <= self.threshold]
      for idx in indices:
        sampling[idx] += 1
      
      n -= indices.shape[0]
      
    return sampling
