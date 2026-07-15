#!/usr/bin/env python

from setuptools import setup

with open('requirements.txt', 'rt') as f:
    install_requires = [line.strip() for line in f if line.strip()]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pathfinder',
      description='Joint multimodal decomposition',
      author='Saad Jbabdi',
      author_email='<saad.jbabdi@ndcn.ox.ac.uk>',
      packages=['pathfinder',],
      install_requires=install_requires,
      python_requires='>=3.10',
      )

