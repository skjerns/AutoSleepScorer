# AutoSleepScorer
An attempt to create a robust sleep scorer.

## Installation
The AutoSleepScorer is currently running with Python 3 using Keras with Tensorflow and has been tested on Windows.

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

**Minimal example**
For quick classification

```Python
from AutoSleepScorer import Scorer
scorer = Scorer([eeg_filename], hypnograms=True)
scorer.run()
```

**Extended example**
First download a sample file from the EDFx database

```Python
from AutoSleepScorer import Scorer, SleepData
# download sample EEG file from the EDFx database
tools.download('https://physionet.nlm.nih.gov/pn4/sleep-edfx/SC4001E0-PSG.edf', 'sample-psg.edf')
# download corresponding hypnogram for comparrison
tools.download('https://pastebin.com/raw/jbzz16wP', 'sample-psg.hypnogram.csv') 
```
Instead of a filename we can also set advanced options using a `SleepData` object
```Python
# create a SleepData object 
file = sleeploader.SleepData('sample-psg.edf', start = 2880000, stop = 5400000, 
							  channels={'EEG':'EEG Fpz-Cz', 'EMG':'EMG submental', 
                              			'EOG':'EOG horizontal'}, preload=False)
# Create and run Scorer
scorer = sleepscorer.Scorer([file], hypnograms=True)
scorer.run()
tools.show_sample_hypnogram('sample-psg.hypnogram.csv', start=960, stop=1800)  
```