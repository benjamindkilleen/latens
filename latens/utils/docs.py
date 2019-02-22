from tensorflow.keras import activations
from latens.utils import act

"""Documentation strings for latens.
"""

description = """\
Latens. An Unsupervised Learning approach to active learning.
"""

command_choices = ['debug', 'autoencoder', 'reconstruct', 'convert']
command_help = "latens command to run."

input_help = """Name or names of data files."""

l2_reg_help = """L2 regularization factor. Default is None."""

output_help = """Command output. Default is 'show'."""

model_dir_help = """Directory to save the model in. Default (None) doesn't save
the model. Required for prediction."""

overwrite_help = """overwrite model if it exists"""

image_shape_help = """Shape of the image. Must be 3D. Grayscale uses 1 for last
dimension. Default is '28 28 1' for mnist."""

epochs_help = """Number of training epochs. Default is 1."""

num_components_help = """Number of components in the low-dimensional
representation. Default is 10."""

eval_secs_help = """Evaluate model every EVAL_SECS during training. Default is
1200."""

eval_mins_help = """See EVAL_SECS. Default is 20 minutes (1200 seconds)"""

splits_help = """Number of examples to use for splits on the dataset. Must
define three splits for training, validation, and test sets. Default is '50000
10000 10000' for MNIST."""

cores_help = """Number of CPU cores to use when parallelizing. Default, -1,
parallelizes across all visible processors."""

batch_size_help = """Batch size for training. Default is 16."""

dropout_help = """Dropout rate for the representational layer. Default is 0.1"""

activation_choices = {'sigmoid' : activations.sigmoid,
                      'relu' : activations.relu,
                      'clipped_relu' : act.clu}

activation_help = """Activation function to use at the representational
layer. Default is relu."""

learning_rate_help = """Learning rate for training. Default is 0.01"""

momentum_help = """Momentum for the momentum optimizer. Default is 0.9"""

eager_help = """Perform eager execution. Useful for debugging."""

keras_verbose_help = """Verbosity to use for model.fit(). 0 for no logging, 1
for progress bar, 2 for info at each epoch. Default is 1."""

verbose_help = """latens logging verbosity. 0 is for WARNING and above. 1
(default) for INFO. 2 for DEBUG."""

tensorboard_help = """Write tensorboard logs to MODEL_DIR/logs."""

