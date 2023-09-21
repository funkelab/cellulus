<h2 align="center">Cellulus</h2>

- **[Introduction](#introduction)**
- **[Dependencies](#dependencies)**
- **[Getting Started](#getting-started)**
- **[Issues](#issues)**


### Introduction
This repository hosts the version of the code used for the **[preprint]()** titled **Unsupervised Learning of *Object-Centric Embeddings* for Cell Instance Segmentation in Microscopy Images**. 

We refer to the proposed techniques described in the preprint as **Cellulus** - Cellulus is a deep learning based method which can be used to obtain instance-segmentation of objects in microscopy images in an unsupervised fashion i.e. requiring no ground truth labels during training. 

### Dependencies 

One could execute these lines of code to create a new environment and install dependencies. 

If you would like to run Cellulus, using GPU:

```
conda create -n cellulus python
conda activate cellulus
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
git clone https://github.com/funkelab/cellulus.git
cd cellulus
pip install -e .
```

If you would like to run Cellulus, using CPU or the MPS framework:

```
conda create -n cellulus python
conda activate cellulus
pip install torch torchvision
git clone https://github.com/funkelab/cellulus.git
cd cellulus
pip install -e .
```

### Getting Started

Try out the `tutorial.ipynb` notebook for 2D images. 


### Issues

If you encounter any problems, please **[file an issue](https://github.com/funkelab/cellulus/issues)** along with a description.





