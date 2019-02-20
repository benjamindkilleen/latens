"""Utils for visualizing artifice output. (Mostly for testing).
"""

import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger('artifice')

def plot_image(*images):
  fig, axes = plt.subplots(1,len(images), squeeze=False)
  for i, image in enumerate(images):
    im = axes[0,i].imshow(np.squeeze(image), cmap='magma', vmin=0., vmax=1.)
  plt.colorbar(im, ax=axes.flat)
  
def show_image(*images):
  plot_image(*images)
  plt.show()
