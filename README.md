
# Unsupervised Learning of Object-Centric Embeddings for Cell Instance Segmentation in Microscopy Images

*Algorithm for unsupervised cell instance segmentation.* We present a self-supervised learning method for object-centric embeddings (OCEs) which embed image patches such that the spatial offsets between patches cropped from the same object are preserved. Those learnt embeddings can be used to delineate individual objects and thus obtain instance segmentations. The method relies on the assumptions (commonly found in microscopy images) that objects have a similar appearance and are randomly distributed in the image. If ground-truth annotations are available, this method serves as an excellent starting point for supervised training, reducing the required amount of ground-truth needed

![](.assets/autospem.webp)

## Requirements and Setup

Install the required packages with conda
```
conda create --name autospem --file environment.yml
```

## Train Spatial Instance Embedding Networks


```
python colocseg/train_ssl.py --shape 252 252 --in_channels 2 --out_channels 2 --dspath <path to tissuenet files> --initial_lr 4e-05 --output_shape 236 236 --positive_radius 10 --regularization 1e-05 --check_val_every_n_epoch 10 --limit_val_batches 256 --max_epochs 50 --temperature 10 --lr_milestones 20 30 --batch_size 8 --loader_workers 8 --gpu 1
```

## Infer Mean and Std of Spatial Embeddings

```
python colocseg/infer_spatial_embeddings.py <path_to_model>/model.torch output.zarr spatial_embedding <path_to_tissuenet>/tissuenet_v1.0_test.npz 102 raw 2 32 transpose
```

## Infer Segmentation from Spatial Embedding

```
python colocseg/infer_pseudo_gt_from_mean_std.py output.zarr <path_to_tissuenet>/tissuenet_v1.0_test.npz spatial_embedding meanshift_segmentation 0 0.21
```
## Postprocess Embeddings (Shrinking Instances by Fixed Distance)

```
python scripts/postprocess.py output.zarr meanshift_segmentation
```



## External Datasets

Models were trained on cell segmentation datasets that are part of the [tissuenet dataset](https://datasets.deepcell.org/) and the [cell tracking challenge datasets](http://celltrackingchallenge.net/2d-datasets/)

## 3D Segmentation

![](.assets/3dcellulus.webp)
> Fully unsupervised 3D segmentation with no prior training

