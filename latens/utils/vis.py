"""Utils for visualizing artifice output. (Mostly for testing).
"""

import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger('artifice')

max_plot = 10000

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
  ls = labels[:N]
  plt.figure()
  if labels is None:
    plt.plot(xs, ys, 'b,')
  else:
    for i in range(num_classes):
      plt.plot(xs[ls == i], ys[ls == i], f'C{i}.',
               label=str(i))
  plt.title("Latent Space Encodings")

def plot_sampled_encodings(encodings, sampling, labels=None, num_classes=10):
  xs = encodings[:,0]
  ys = encodings[:,1]
  unsampled = sampling == 0
  sampled = sampling > 0
  plt.figure()
  plt.plot(xs[unsampled], ys[unsampled], c='gray', marker=',', linestyle='')
  if labels is None:
    plt.plot(xs[sampled], ys[sampled], 'b.')
  else:
    for i in range(num_classes):
      which = np.logical_and(sampled, labels == i)
      plt.plot(xs[which], ys[which], f'C{i}.',
               label=str(i))
    plt.legend()
  plt.title("Latent Space Sampling")
  
def show_image(*images, **kwargs):
  plot_image(*images, **kwargs)
  plt.show()

def show_encodings(encodings, **kwargs):
  plot_encodings(encodings, **kwargs)
  plt.show()


