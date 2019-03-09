"""Utils for visualizing artifice output. (Mostly for testing).
"""

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import logging


logger = logging.getLogger('artifice')

max_plot = 5000

def plot_image(*images, columns=10, ticks=False):
  columns = min(columns, len(images))
  rows = max(1, len(images) // columns)
  fig, axes = plt.subplots(rows,columns, squeeze=False,
                           figsize=(columns/2, rows/2))
  for i, image in enumerate(images):
    ax = axes[i // columns, i % columns]
    im = ax.imshow(np.squeeze(image), cmap='magma', vmin=0., vmax=1.)
  if not ticks:
    for ax in axes.ravel():
      ax.axis('off')
      ax.set_aspect('equal')
  fig.subplots_adjust(wspace=0, hspace=0)  
  
def plot_encodings(encodings, labels=None, num_classes=10):
  xs = encodings[:,0]
  ys = encodings[:,1]
  plt.figure()
  if labels is None:
    plt.plot(xs, ys, 'b,')
  else:
    ls = labels
    for i in range(num_classes):
      plt.plot(xs[ls == i], ys[ls == i], f'C{i},', label=str(i))
    plt.legend()
  plt.title("Latent Space Encodings")

def plot_sampled_encodings(encodings, sampling, labels=None, num_classes=10):
  xs = encodings[:,0]
  ys = encodings[:,1]
  unsampled = sampling == 0
  sampled = sampling > 0
  plt.figure()
  plt.plot(xs[unsampled], ys[unsampled], c='gray', marker=',', linestyle='')
  if labels is None:
    plt.plot(xs[sampled], ys[sampled], 'b.', markersize=1)
  else:
    for i in range(num_classes):
      which = np.logical_and(sampled, labels == i)
      plt.plot(xs[which], ys[which], f'C{i}.', markersize=2,
               label=str(i))
    plt.legend()
  plt.title("Latent Space Sampling")

def plot_sampling_distribution(sampling, labels, num_classes=10):
  plt.figure()
  counts = np.zeros(num_classes, dtype=np.int64)
  for i in range(counts.shape[0]):
    counts[i] += np.sum(sampling[labels == i])

  ticks = [i for i in range(num_classes)]
  tick_labels = [str(i) for i in range(num_classes)]
  plt.bar(ticks, counts, color='gray')
  plt.xlabel("Class Label")
  plt.ylabel("Count")
  plt.xticks(ticks, tick_labels)
  plt.title("Class Sampling")
  
def plot_encodings_3d(encodings, labels=None, num_classes=10):
  encodings = encodings[:max_plot]
  xs = encodings[:,0]
  ys = encodings[:,1]
  zs = encodings[:,2]
  plt.figure()
  ax = plt.axes(projection='3d')
  if labels is None:
    ax.scatter3D(xs, ys, zs)
  else:
    ls = labels[:max_plot]
    for i in range(num_classes):
      ax.scatter3D(xs[ls == i], ys[ls == i], zs[ls==i], c=f'C{i}',
                   marker='.')
  plt.title("Latent Space Encodings")

def plot_sampled_encodings_3d(encodings, sampling, labels=None, num_classes=10):
  encodings = encodings[:max_plot]
  sampling = sampling[:max_plot]
  xs = encodings[:,0]
  ys = encodings[:,1]
  zs = encodings[:,2]
  unsampled = sampling == 0
  sampled = sampling > 0
  plt.figure()
  ax = plt.axes(projection='3d')
  ax.plot3D(xs[unsampled], ys[unsampled], zs[unsampled],
               'gray')
  if labels is None:
    ax.scatter3D(xs[sampled], ys[sampled], zs[sampled], c='b', marker='o')
  else:
    labels = labels[:max_plot]
    for i in range(num_classes):
      which = np.logical_and(sampled, labels == i)
      ax.scatter3D(xs[which], ys[which], zs[which], c=f'C{i}',
                   marker='o')
  plt.title("Latent Space Sampling")
  
def show_image(*images, **kwargs):
  plot_image(*images, **kwargs)
  plt.show()

def show_encodings(encodings, **kwargs):
  plot_encodings(encodings, **kwargs)
  plt.show()


