# AutoSleepScorer
An attempt to create a robust sleep scorer.

## Installation

### 1. Install Anaconda
Download and install Anaconda with Python 3.6 64 bit from https://www.anaconda.com/download/#download

If you already have a working Python 3.x environment you can skip this step.

### 2. Install Tensorflow
Open a command line and install tensorflow via `pip install tensorflow`
If you wish to have GPU support and have installed CUDA you can use `pip install tensorflow-gpu`

### 3. Install AutoSleepScorer
Clone and install this repository via pip:
`pip install git+https://github.com/skjerns/AutoSleepScorer`

## Quickstart

Open a python console.

To classify a sample EEG file

```Python
# download sample EEG file
tools.download('https://physionet.nlm.nih.gov/pn4/sleep-edfx/SC4001E0-PSG.edf', 'sample-psg.edf')
file = os.path.join(os.getcwd(), 'sample-psg.edf')
# create and run scorer
scorer = sleepscorer.AutoSleepScorer([file], hypnograms=True)
scorer.run()
```