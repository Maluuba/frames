#!/usr/bin/env python

from setuptools import setup
reqs = [
    "numpy",
    "click",
    "tabulate",
    "nose",
    "colorama",
]

setup(name='Frames',
      version='1.0',
      description='SRED Reporting and Experiment Management Utility',
      author='Microsoft Maluuba',
      url='http://github.com/Maluuba/frames',
      packages=['frames'],
      scripts=['bin/frametracking-tagger', 'bin/frametracking-evaluate'],
      install_requires=reqs)
