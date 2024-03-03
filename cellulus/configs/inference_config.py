from typing import List

import attrs
from attrs.validators import in_, instance_of

from .dataset_config import DatasetConfig
from .utils import to_config


@attrs.define
class InferenceConfig:
    """Inference configuration.

    Parameters
    ----------

        dataset_config:

            Configuration object for the data to predict and segment.

        prediction_dataset_config:

            Configuration object produced by predict.py.

        detection_dataset_config:

            Configuration object produced by detect.py.

        segmentation_dataset_config:

            Configuration object produced by segment.py.

        evaluation_dataset_config:

            Configuration object for the ground truth masks.

        crop_size (default = [252, 252]):

            ROI used by the scan node in gunpowder.

        p_salt_pepper (default = 0.01):

            Fraction of pixels that will have salt-pepper noise.

        num_infer_iterations (default = 16):

            Number of times the salt-peper noise is added to the raw image.
            This is used to infer the foreground and background in the raw image.

        bandwidth (default = None):

            Bandwidth used to perform mean-shift clustering on the predicted
            embeddings.

        threshold (default = None):

            Threshold to use for binary partitioning into foreground and background
            pixel regions. If None, this is figured out automatically by performing
            Otsu Thresholding on the last channel of the predicted embeddings.

        min_size (default = None):

            Ignore objects which are smaller than `min_size` number of pixels.

        device (default = 'cuda:0'):

            The device to infer on.
            Set to 'cpu' to infer without GPU.

        clustering (default = 'meanshift'):

            How to cluster the embeddings?
            Can be one of 'meanshift' or 'greedy'.

        use_seeds (default = False):

            If set to True, the local optima of the distance map from the
            predicted object centers is used.
            Else, seeds are determined by sklearn.cluster.MeanShift.

        num_bandwidths (default = 1):

            Number of bandwidths to obtain segmentations for.

        reduction_probability (default = 0.1):

            If set to less than 1.0, this fraction of available pixels are used
            to determine the clusters (fitting stage) while performing
            meanshift clustering.
            Once clusters are available, they are used to predict the cluster assignment
            of the remaining pixels (prediction stage).

        min_size (default = None):

            Objects below `min_size` pixels will be removed.

        post_processing (default= 'morphological'):

            Can be one of 'morphological' or 'intensity' operations.
            If 'morphological', the individual detections grow and shrink by
            'grow_distance' and 'shrink_distance' number of pixels.

            If 'intensity', each detection is partitioned based on a binary intensity
            threshold calculated automatically from the raw image data.
            By default, the channel `0` in the raw image is used for
            intensity thresholding.

        grow_distance (default = 3):

            Only used if post_processing (see above) is equal to 'morphological'.

        shrink_distance (default = 6):

            Only used if post_processing (see above) is equal to
            'morphological'.

    """

    dataset_config: DatasetConfig = attrs.field(
        default=None, converter=to_config(DatasetConfig)
    )

    prediction_dataset_config: DatasetConfig = attrs.field(
        default=None, converter=to_config(DatasetConfig)
    )

    detection_dataset_config: DatasetConfig = attrs.field(
        default=None, converter=to_config(DatasetConfig)
    )

    segmentation_dataset_config: DatasetConfig = attrs.field(
        default=None, converter=to_config(DatasetConfig)
    )

    evaluation_dataset_config: DatasetConfig = attrs.field(
        default=None, converter=to_config(DatasetConfig)
    )
    device: str = attrs.field(default="cuda:0", validator=instance_of(str))
    crop_size: List = attrs.field(default=[252, 252], validator=instance_of(List))
    p_salt_pepper = attrs.field(default=0.01, validator=instance_of(float))
    num_infer_iterations = attrs.field(default=16, validator=instance_of(int))
    threshold = attrs.field(
        default=None, validator=attrs.validators.optional(instance_of(float))
    )
    clustering = attrs.field(
        default="meanshift", validator=in_(["meanshift", "greedy"])
    )
    use_seeds = attrs.field(default=False, validator=instance_of(bool))
    bandwidth = attrs.field(
        default=None, validator=attrs.validators.optional(instance_of(float))
    )
    num_bandwidths = attrs.field(default=1, validator=instance_of(int))
    reduction_probability = attrs.field(default=0.1, validator=instance_of(float))
    min_size = attrs.field(
        default=None, validator=attrs.validators.optional(instance_of(int))
    )
    post_processing = attrs.field(default="cell", validator=in_(["cell", "nucleus"]))
    grow_distance = attrs.field(default=3, validator=instance_of(int))
    shrink_distance = attrs.field(default=6, validator=instance_of(int))
