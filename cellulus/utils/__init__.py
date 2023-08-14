from typing import List

from cellulus.utils.logger import Logger


def get_logger(keys: List[str], title: str) -> Logger:
    return Logger(keys, title)


def tiff_2_zarr(path_to_tiffs, output_path, path_to_labels = None, downsample_factor=1,):
  im_list = os.listdir(path_to_tiffs)
  if path_to_labels != None:
    label_list = os.listdir(path_to_labels)
  x_max = 0
  y_max = 0
  for file in im_list:
      im = Image.open(path_to_tiffs+'/'+file)
      array = np.array(im)
      x_max = max(x_max,array.shape[0])
      y_max = max(y_max,array.shape[1])

  x_max = np.floor(x_max/downsample_factor)
  y_max = np.floor(y_max/downsample_factor)

  root = zarr.open(output_path, mode='w')
  z_img = root.zeros('img', shape=[len(im_list),1,x_max,y_max], chunks=(1, 1, 200, 200), dtype='f4')
  z_img.attrs["axis_names"] = ["s", "c","y","x"]
  z_img.attrs["resolution"] = [1,1,1]
  if path_to_labels != None:
    z_labels = root.zeros('labels', shape=[len(label_list),1,x_max,y_max], chunks=(1, 1, 200, 200), dtype='f4')
    z_labels.attrs["axis_names"] = ["s", "c","y","x"]
    z_labels.attrs["resolution"] = [1,1,1]

  for i in range(len(im_list)):
    tif_im = Image.open(path_to_tiffs+'/'+im_list[i])
    img = np.array(tif_im)
    z_img[i,0,:,:] = resize(img,(x_max,y_max))
    if path_to_labels != None:
      tif_label = Image.open(path_to_labels+'/'+label_list[i])
      np_label = np.array(tif_label)
      z_labels[i,0,:,:] = resize(np_label,(x_max,y_max))