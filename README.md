# AutoSleepScorer
An attempt to create a robust sleep scorer using Convolutional Neural Networks with Long Short-Term Memory.

The sleep stage scoring (also: sleep staging) process is most of the time still performed manually by a trained technician and very tedious and time-consuming. Although many papers have been published which propose automatic sleep stage detection systems none of them could assert themselves as an industry standard. Main reasons for that are the lacking proof of reliability as well as the high costs of the software. Up to this day no open-source implementation or package is available for the researchers or clinicians. In this project a Convolutional Neural Network with Long Short-Term Memory is used for the detection of sleep stages. This approach has the advantage that it can automatically detect and extract features from the raw EEG signal. The network will be trained on several different public and private datasets to ensure a good generalizability. The final goal of this project will be an open-source package with an easy-to-use interface aimed at researchers and clinicians which can be used either in a computer-aided diagnosis or fully automated manner.

## Installation
The AutoSleepScorer is currently running with Python 3 using Keras with Tensorflow and has been tested on Windows 10. Due to Pythons multi-platform support it should run other OS as well. The easiest way to get started is using Anaconda, a Python package manager.

### 1. Install Anaconda
Download and install Anaconda with Python 3.6 64 bit from https://www.anaconda.com/download/#download

If you already have a working Python 3.x environment you can skip this step.

### 2. Install Tensorflow
Open a command line and install tensorflow via `pip install tensorflow`

If you wish to have GPU support with a GeForce graphics card and have installed CUDA you can use `pip install tensorflow-gpu`. Running calculations on the GPU accelerates them significantly.
Installation of CUDA can be a bit tricky and described here: https://www.tensorflow.org/install/

### 3. Install AutoSleepScorer
Clone and install this repository via pip:
`pip install git+https://github.com/skjerns/AutoSleepScorer`

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
