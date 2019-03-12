from tensorflow.keras import activations
from latens.utils import act
from latens import sam

"""Documentation strings for latens.
"""

description = """\
Latens. An Unsupervised Learning approach to active learning.
"""

command_choices = ['convert', 'autoencoder', 'encode', 'sample', 'classifier',
                   'full-classifier', 'reconstruct', 'decode', 'visualize', 'debug']
command_help = """latens command to run. The core functionality depends on
commands: 'convert', 'autoencoder', 'encode', 'sample', and 'classifier', in
that order."""

input_help = """Input dataset file or prefix. Prefix is used to store other
files."""

l2_reg_help = """L2 regularization factor. Default is None."""

output_help = """Command output. Usually a directory to store plots and logs in."""

model_dir_help = """Directory to save other model dirs in. Default (None) doesn't save
the model. Should be provided for prediction, otherwise initialized weights are
used."""

overwrite_help = """overwrite model if it exists"""

image_shape_help = """Shape of the image. Must be 3D. Grayscale uses 1 for last
dimension. Default is '28 28 1' for mnist."""

epochs_help = """Number of training epochs. Default is 1."""

latent_dim_help = """Number of latent dimensions. Default is 2."""

eval_secs_help = """Evaluate model every EVAL_SECS during training. Default is
1200."""

eval_mins_help = """See EVAL_SECS. Default is 20 minutes (1200 seconds)"""

splits_help = """Number of examples to use for the train, tune, and test sets,
respectively. Default is '50000 10000 10000' for MNIST."""

cores_help = """Number of CPU cores to use when parallelizing. Default, -1,
parallelizes across all visible processors."""

batch_size_help = """Batch size for training. Default is 16."""

dropout_help = """Dropout rate for the representational layer. Default is 0.1"""

rep_activation_choices = {'none' : None,
                          'None' : None,
                          'sigmoid' : activations.sigmoid,
                          'relu' : activations.relu,
                          'clu' : act.clu,
                          'softmax' : activations.softmax}
rep_activation_help = """Activation function to use at the representational
layer. Default is 'relu'."""

learning_rate_help = """Learning rate for training. Default is 0.01"""

momentum_help = """Momentum for the momentum optimizer. Default is 0.9"""

eager_help = """Perform eager execution. Useful for debugging."""

keras_verbose_help = """Verbosity to use for model.fit(). 0 for no logging, 1
for progress bar, 2 for info at each epoch. Default is 1."""

verbose_help = """latens logging verbosity. 0 is for WARNING and above. 1
(default) for INFO. 2 for DEBUG."""

tensorboard_help = """Write tensorboard logs to MODEL_DIR/logs. (NOT IMPLEMENTED)"""

load_help = """Prefer most recent epoch file rather than finished weights."""

num_classes_help = """Specify number of classes for classifier. Default is 10."""

sample_size_help = """Number of examples for a sampler to draw. Default is 1000."""

sample_choices = {'random' : sam.RandomSampler,
                  'normal' : sam.NormalSampler,
                  'multi-normal' : sam.MultivariateNormalSampler,
                  'uniform' : sam.UniformSampler,
                  'uniform-cluster' : sam.UniformClusterSampler,
                  'normal-cluster' : sam.NormalClusterSampler,
                  'multi-normal-cluster' : sam.MultivariateNormalClusterSampler,
                  'error' : sam.MaximizingSampler,
                  'classifier-error' : sam.MaximizingSampler,
                  'classifier-loss' : sam.MaximizingSampler,
                  'classifier-incorrect' : sam.IdentitySampler}
sample_help = """Type of sampling to use. Default is 'multi-normal'."""

epoch_multiplier_help = """Runs the dataset 'mult' times per epoch. Total of
EPOCHS * MULT epochs will be run."""

show_help = """Show outputs to screen. Used mainly for visualizations."""

autoencoder_type_choices = ['conv', 'vae', 'student', 'vaestudent']
autoencoder_type_help = "Type of autoencoder to use. Default is 'conv'"
