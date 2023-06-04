from __future__ import absolute_import
import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cellulus", 
    version="0.0.1",
    author="TODO",
    author_email="TODO",
    description="cellulus provides unsupervised segmentation of objects in microscopy images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/funkelab/cellulus/",
    project_urls={
        "Bug Tracker": "https://github.com/funkelab/cellulus/issues",
    },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
          'matplotlib',
          'numpy',
          'scipy',
          "tifffile",
          "numba",
          "tqdm",
          "jupyter",
          "scikit-image",
          "pytest",
          "imagecodecs",
          "zarr"
        ]
)
