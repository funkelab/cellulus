<h2 align="center">Cellulus</h2>

- **[Introduction](#introduction)**
- **[Dependencies](#dependencies)**
- **[Getting Started](#getting-started)**
- **[Citation](#citation)**
- **[Issues](#issues)**


### Introduction
This repository hosts the version of the code used for the **[preprint](https://arxiv.org/pdf/2310.08501.pdf)** titled **Unsupervised Learning of *Object-Centric Embeddings* for Cell Instance Segmentation in Microscopy Images**. This work was accepted to the International Conference for Computer Vision (ICCV), 2023.

We refer to the proposed techniques described in the preprint as **Cellulus** - Cellulus is a deep learning based method which can be used to obtain instance-segmentation of objects in microscopy images in an unsupervised fashion i.e. requiring no ground truth labels during training. 

### Dependencies 

One could execute these lines of code to create a new environment and install dependencies. 

If you would like to run Cellulus, using GPU:

```
conda create -y -n cellulus python==3.9
conda activate cellulus
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
git clone https://github.com/funkelab/cellulus.git
cd cellulus
pip install -e .
```

If you would like to run Cellulus, using the CPU or the MPS framework:

```
conda create -y -n cellulus python==3.9
conda activate cellulus
pip install torch torchvision
git clone https://github.com/funkelab/cellulus.git
cd cellulus
pip install -e .
```

### Getting Started

Try out `2D Example` available **[here](https://funkelab.github.io/cellulus)**. 

### Citation

If you find our work useful in your research, please consider citing:

```bibtex
@misc{wolf2023unsupervised,
      title={Unsupervised Learning of Object-Centric Embeddings for Cell Instance Segmentation in Microscopy Images}, 
      author={Steffen Wolf and Manan Lalit and Henry Westmacott and Katie McDole and Jan Funke},
      year={2023},
      eprint={2310.08501},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Issues

If you encounter any problems, please **[file an issue](https://github.com/funkelab/cellulus/issues)** along with a description.



