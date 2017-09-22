# AutoSleepScorer
An attempt to create a robust sleep scorer using Convolutional Neural Networks with Long Short-Term Memory. This package aims at researchers trying to find an open-source solution for automatic sleep stage classification of human PSG recordings.

![Sample Hypnogram](https://github.com/skjerns/AutoSleepScorer/blob/master/figures/hypno.png?raw=true)
## Introduction
In this project a Convolutional Neural Network with Long Short-Term Memory is used for the detection of sleep stages. This approach has the advantage that it can automatically detect and extract features from the raw EEG, EMG and EOG signal (see [here](https://github.com/skjerns/AutoSleepScorer/blob/master/figures/features.png?raw=true) for example features that are learned by the network). The network was trained on several different public and private datasets to ensure a good generalizability (CCSHS, EDFx, EMSA). The network architecture can be found [here](https://github.com/skjerns/AutoSleepScorer/blob/master/figures/architecture.png?raw=true).

Currently the classifier reaches the state-of-the-art of automatic sleep stage classification while obtaining a similar performance to a human scorer. A link to the accompanying publication will be provided soon.

**Classifier performance (5-fold cross validation)**
Data Set| Accuracy | F1-score
------------ | -------------|-------------
[CCSHS](https://sleepdata.org/datasets/ccshs) | 89% | 81%
[EDFx](https://physionet.nlm.nih.gov/pn4/sleep-edfx/) | 87% | 80%
[EMSA](https://www.ncbi.nlm.nih.gov/pubmed/28594100) | 87% | 77%
||
[General Inter Rater Reliability](https://www.ncbi.nlm.nih.gov/pubmed/19250176) | ~80-82% | ?
## Usage

The Classifier expects a PSG recording with at least one EEG, EMG and EOG present. Theoretically all data formats supported by [MNE](https://martinos.org/mne/stable/python_reference.html#reading-raw-data) can be used. All data will be resampled to 100 Hz.

Documentation and command line instructions are coming soon. For now please see [Quickstart](https://github.com/skjerns/AutoSleepScorer#quickstart).

## Installation
The AutoSleepScorer is currently running with Python 3 using Keras with Tensorflow and has been tested on Windows 10. Due to Pythons multi-platform support it should run other OS as well. The easiest way to get started is using Anaconda, a Python package manager.

### 1. Install Anaconda
Download and install Anaconda with Python 3.6 64 bit from https://www.anaconda.com/download/#download

If you already have a working Python 3.x environment you can skip this step.

### 2. Install Tensorflow
Open a command line and install tensorflow via `pip install tensorflow`

If you wish to have GPU support with a GeForce graphics card and have installed CUDA you can use `pip install tensorflow-gpu`. Running calculations on the GPU accelerates them significantly.
Installation of CUDA can be a bit tricky and is described here: https://www.tensorflow.org/install/

### 3. Install AutoSleepScorer
Clone and install this repository via pip:
`pip install git+https://github.com/skjerns/AutoSleepScorer`

If you get an error that`git` is not installed, you can install it using the command line `conda install git`

## Quickstart

Open a python console for instance using Anaconda.

**Minimal example**

For quick classification

```Python
from sleepscorer import Scorer
file = "eeg_filename" #link to the EEG header or EEG file
scorer = Scorer([file], hypnograms=True)
scorer.run()
```

**Extended example**

First we download a sample file from the EDFx database

```Python
from sleepscorer import tools
# download sample EEG file from the EDFx database
tools.download('https://physionet.nlm.nih.gov/pn4/sleep-edfx/SC4001E0-PSG.edf', 'sample-psg.edf')
# download corresponding hypnogram for comparrison of classification
tools.download('https://pastebin.com/raw/jbzz16wP', 'sample-psg.groundtruth.csv') 
```
Now we can start the Scorer using a list of EEG files.
Instead of an EEG-filename we can also set advanced options using a `SleepData` object
```Python
# create a SleepData object 
from sleepscorer import Scorer, SleepData
file = SleepData('sample-psg.edf', start = 2880000, stop = 5400000, 
							  channels={'EEG':'EEG Fpz-Cz', 'EMG':'EMG submental', 
                              			'EOG':'EOG horizontal'}, preload=False)
# Create and run Scorer
scorer = Scorer([file], hypnograms=True, demo=True)
scorer.run()
tools.show_sample_hypnogram('sample-psg.groundtruth.csv', start=960, stop=1800)
```
The predictions will now be saved as `sample-psg.edf.csv`, where each row corresponds to one epoch.
