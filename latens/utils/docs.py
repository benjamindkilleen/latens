"""Documentation strings for latens.
"""

description = """\
Latens. An Unsupervised Learning approach to active learning.
"""

command_choices = ['debug', 'train', 'predict', 'convert']
command_help = "latens command to run."

input_help = """Name or names of data files."""

l2_reg_help = """L2 regularization factor. Default is None."""

output_help = """Command output. Default is 'show'."""

model_dir_help = """Model directory. Default (None) doesn't save the
model. Required for prediction."""

overwrite_help = """overwrite model if it exists"""

image_shape_help = """Shape of the image. Must be 3D. Grayscale uses 1 for last
dimension. Default is '28 28 1' for mnist."""

epochs_help = """Number of training epochs. Default is -1, repeats indefinitely."""

num_components_help = """Number of components in the low-dimensional
representation. Default is 10."""

eval_secs_help = """Evaluate model every EVAL_SECS during training. Default is
1200."""

eval_mins_help = """See EVAL_SECS. Default is 20 minutes (1200 seconds)"""

splits_help = """Number of examples to use for splits on the dataset. Must
define three splits for training, dev, and test sets. Default is '50000 10000
10000' for MNIST."""

cores_help = """Number of CPU cores to use when parallelizing. Default, -1,
parallelizes across all visible processors."""
