from setuptools import setup

setup(name='sleepscorer',
      version='0.231',
      description='An Automatic Sleep Stage Classification Package',
      url='http://github.com/skjerns/AutoSleepScorer',
      author='skjerns',
      author_email='sikern@uos.de',
      license='See Github',
      packages=['sleepscorer'],
      install_requires=['keras==2.0.6', 'tensorflow==1.3.0','h5py==2.7.0', 'mne', 'tqdm', 'matplotlib'],
      zip_safe=False)
