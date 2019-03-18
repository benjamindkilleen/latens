# Latens

An Unsupervised Learning approach to active learning.

Here:
![Overview](https://github.com/bendkill/latens/blob/master/docs/query_selection.pdf "Overview")

## Dependencies:
Latens uses Python 3.6 or higher; see [here](https://www.python.org/downloads/)
for recent downloads, or install from brew. Additionally, it relies on
Tensorflow 1.12.0 or higher, which can be found
[here](https://www.tensorflow.org/install/pip).

## Installation

After cloning this repository, add its root to the `PYTHON_PATH` by running
```
export PYTHONPATH=PATH_TO_LATENS:$PYTHONPATH
```
where `PATH_TO_LATENS` is replaced with the path to the root directory (where
this file is located). You can add the same line to your `~/.bash_profile` or
equivalent config file to make the change permanent.

## Usage

`latens` can be easily imported, once it has been added to the `PYTHONPATH`, but
to run directly, the `latens.py` script contains the main functionality. Run
```
python latens.py -h
```
to see the available options.

## Data

Getting started can be tricky because of the data format that `latens` expects
data to be in. It was originally developed with the MNIST dataset. Similar
datasets should also be compatible.

Before it can begin training, `latens` requires data to be stored in a
`.tfrecord` format. TFRecords are not very well standardized, so we provide the
`convert` command, which *should* format data as expected. Store images and
labels in a single `.npz` file with keywords "data" and "labels"
respectively. `data/mnist.npz` is provided for reference. Once this is done, run
```
python latens.py convert -i data/mnist.npz
```
to create a `.tfrecord` file in the same directory.

