from skimage.io import imsave
import io

import matplotlib
matplotlib.use('agg')
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
import numpy as np

from colocseg.utils import label2color

def vis_anchor_embedding(embedding, patch_coords, img, grad=None, output_file=None):
    # patch_coords.shape = (num_patches, 2)

    if img is not None:
      if img.shape[0] not in [3]:
        plt.imshow(img[0], cmap='magma', interpolation='nearest')
      else:
        plt.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')

    if isinstance(embedding, list):
      for e in embedding:
        plt.quiver(patch_coords[:, 0],
           patch_coords[:, 1],
           e["embedding"][:, 0],
           e["embedding"][:, 1], 
           angles='xy',
           scale_units='xy',
           scale=1., color=e["color"])
    else:
      plt.quiver(patch_coords[:, 0],
                 patch_coords[:, 1],
                 embedding[:, 0],
                 embedding[:, 1], 
                 angles='xy',
                 scale_units='xy',
                 scale=1., color='#8fffdd')

    if grad is not None:
        plt.quiver(patch_coords[:, 0],
                   patch_coords[:, 1],
                   (10 * grad[:, 0]) / (grad[:, :2].max() + 1e-9),
                   (10 * grad[:, 1]) / (grad[:, :2].max() + 1e-9),
                   angles='xy',
                   scale_units='xy',
                   scale=1.,
                   color='#ff8fa0')

    plt.axis('off')

    if output_file is not None:
      if isinstance(output_file, (list, tuple)):
        for of in output_file:
          plt.savefig(of, dpi=300, bbox_inches='tight')
      else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    plt.cla()
    plt.clf()
    plt.close()
    # return buf
