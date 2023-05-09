import torch
from inferno.io.transform import Transform
import torch
from embeddingutils.affinities import embedding_to_affinities, label_equal_similarity_with_mask_le, label_equal_similarity_with_mask_max_le


class Segmentation2AffinitiesWithPadding(Transform):

    def __init__(self, offsets,
                 segmentation_to_binary=True,
                 invert_binary_segmentation=True,
                 ignore_label=-1,
                 retain_segmentation=True,
                 **super_kwargs):
        self.offsets = offsets
        self.ignore_label = ignore_label
        self.retain_segmentation = retain_segmentation
        self.segmentation_to_binary = segmentation_to_binary
        self.invert_binary_segmentation = invert_binary_segmentation

        super().__init__(**super_kwargs)

    def tensor_function(self, tensor):

        ttensor = torch.from_numpy(tensor[None])

        def am(e1, e2, dim=0):
            return label_equal_similarity_with_mask_le(e1, e2, ignore_label_le=self.ignore_label)

        out = embedding_to_affinities(ttensor,
                                      offsets=self.offsets,
                                      affinity_measure=am,
                                      pad_val=self.ignore_label)

        if self.segmentation_to_binary:
            if self.invert_binary_segmentation:
                binary_seg = (ttensor <= 0).float()
            else:
                binary_seg = (ttensor > 0).float()

            if self.ignore_label is not None:
                binary_seg[ttensor == self.ignore_label] = -1

            out = torch.cat((binary_seg.float(), out))

        if self.retain_segmentation:
            out = torch.cat((ttensor.float(), out))

        return out.numpy().astype(tensor.dtype)


class MaskToIgnoreLabel(Transform):
    """Applies a mask where the target tensor == ignore_label."""

    def __init__(self, ignore_label=-1, **super_kwargs):
        super().__init__(**super_kwargs)
        self.ignore_label = ignore_label

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        # validate target and extract segmentation from the target
        full_mask_variable = (target != self.ignore_label).type_as(prediction)
        full_mask_variable.requires_grad = False

        # Mask prediction with master mask
        masked_prediction = prediction * full_mask_variable
        return masked_prediction, target * full_mask_variable
