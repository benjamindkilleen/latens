# Latens

An Unsupervised Learning approach to active learning.

## Dependencies:
Latens uses Python 3.6 or higher; see [here](https://www.python.org/downloads/)
for recent downloads, or install from brew. Additionally, it relies on
Tensorflow 1.9.0, which can be found
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

Latens can be easily imported, once it has been added to the `PYTHONPATH`, but
to run directly, the `latens.py` script contains the main functionality. Run
```
python latens.py -h
```
to see the available options.
