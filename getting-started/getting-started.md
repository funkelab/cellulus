## Using the terminal

If you would like to run `cellulus` , using GPU:

```
conda create -y -n cellulus python==3.9
conda activate cellulus
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
git clone https://github.com/funkelab/cellulus.git
cd cellulus
pip install -e .
```

If you would like to run `cellulus`, using the CPU or the MPS framework:

```
conda create -y -n cellulus python==3.9
conda activate cellulus
pip install torch torchvision
git clone https://github.com/funkelab/cellulus.git
cd cellulus
pip install -e .
```

## Using the `napari` plugin




