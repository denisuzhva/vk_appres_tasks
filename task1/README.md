# Task 1

## Problem statement
Implement an approach described in the paper ["Speaker Recognition from Raw Waveform with SincNet"](https://arxiv.org/abs/1808.00158).
Set up experiments and compare the results with those proposed and illustrated in the article.

## Paper overview
SincNet is a one-dimensional convolutional neural network (CNN) used for speaker recognition.
Compared to a classical 1-D CNN, SincNet differs in its first layer: convolutional kernels are forced to only sample values from the [sinc function](https://en.wikipedia.org/wiki/Sinc_function), thus such layer is called SincConv.
Such approach may allow to extract more meaningful features than by learning filters with arbitrary values.
More in-depth motivation is outlined in the [report for the Task](report.ipynb).

## Implementation

Required packages:

* Python 3
* PyTorch
* NumPy
* Pandas
* [SoundFile](https://pypi.org/project/SoundFile/)

The SincConv layer with the network architecture are implemented in [model.py](model.py).
Note that the implementation allows the SincConv layer to accept an input with multiple channels, in contrast to the single-channel version proposed in the [original paper](https://arxiv.org/abs/1808.00158).
A model can be fed to the trainer ```train_model()``` function in [trainer.py](trainer.py), which also handles dataset split into train and validation parts.
A dataset is loaded by a corresponding loader in [dataloader.py](dataloader.py).
In [main.py](main.py), one can schedule training tasks for various configs defined in [cfg/](cfg/).
Overall pipeline is outlined in [nn_pipeline.pdf](nn_pipeline.pdf):

![](nn_pipeline.png)

## Data preparation

For the series of experiments described in the [report](report.ipynb), the [TIMIT dataset](https://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3) is exploited.
This dataset is composed of 6300 spoken sentences stored in `.WAV` files, which correspond to 630 various speakers (10 sentences per speaker).
The speakers can also separated by their gender into two classes.
With that being said, [timit_make_labels.py](timit_make_labels.py) allows to prepare a dataframe with label assignments for each sentence.
Two labeling schemes are available: one obtains 630 classes, while the other extracts two gender classes.
In this way, the same TIMIT dataset can be used both for binary and multi-class speaker identification tasks.
