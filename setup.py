#!/usr/bin/env python

from setuptools import setup

with open('requirements.txt', 'rt') as f:
    install_requires = [line.strip() for line in f if line.strip()]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pathfinder',
      version='0.1.0',
      description='Joint multimodal decomposition',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Akina Ying-Qiu Zheng and Saad Jbabdi',
      author_email='<saad.jbabdi@ndcn.ox.ac.uk>',
      license='MIT',
      packages=['pathfinder',],
      install_requires=install_requires,
      python_requires='>=3.10',
      )

