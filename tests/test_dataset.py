import unittest
import numpy as np


class DatamodulesTest(unittest.TestCase):

    def test_initialization(self):
        from colocseg.datasets import PatchedDataset
        from torch.utils.data.dataset import Dataset

        class NoiseDs(Dataset):

            def __getitem__(self, key):
                return np.random.rand(1, 128, 128)

            def __len__(self):
                return 10

        ds = PatchedDataset(NoiseDs(),
                            output_shape=(100, 100),
                            positive_radius=15,
                            density=0.1)

        anchor_samples, refernce_samples = ds.sample_coordinates()

        # make sure all coordinates are in the expected range
        assert((refernce_samples >= 0).all())
        assert((refernce_samples < 128).all())
        assert((anchor_samples >= 15).all())
        assert((refernce_samples < 128 - 15).all())

    def test_supervised_anchors(self):
        from colocseg.datasets import SupervisedCoordinateDataset
        from torch.utils.data.dataset import Dataset
        from skimage.filters import gaussian
        from skimage.measure import label
        from skimage import data
        class CoinDs(Dataset):

            def __getitem__(self, key):
                seg = label((gaussian(data.coins(), sigma=3)>0.4))
                return data.coins()[None], seg

            def __len__(self):
                return 10

        ds = SupervisedCoordinateDataset(CoinDs(),
                            (200, 200),
                            2000,
                            min_size=20,
                            return_segmentation=True)

        raw, anc, ref, gt = ds[0]
        
        # uncomment to inspect results visually
        # import matplotlib.pyplot as plt
        # plt.imshow(gt)
        # for i in range(0, len(anc)):
        #     plt.plot((anc[i, 0], ref[i, 0]), (anc[i, 1], ref[i, 1]), 'ro-')
        # plt.show()



if __name__ == '__main__':
    # unittest.main()
    DatamodulesTest().test_supervised_anchors()
