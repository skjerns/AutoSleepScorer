from setuptools import setup

setup(name='sleepscorer',
      version='0.232',
      description='An Automatic Sleep Stage Classification Package',
      url='http://github.com/skjerns/AutoSleepScorer',
      author='skjerns',
      author_email='sikern@uos.de',
      license='See Github',
      packages=['sleepscorer'],
      install_requires=['keras','mne', 'tqdm', 'sklearn', 'matplotlib'],
      zip_safe=False)