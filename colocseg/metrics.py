from scipy.stats import hmean
import numpy as np

def segmentation_metric(gt_label, res_label, overlap_threshold=0.5, match_iou=0.5, return_matches=False):

    seg = 0.
    n_matches = 0
    counter = 0
    imgCounter = 0
    assert(overlap_threshold >= 0.5)
    matches = {}

    compare_dtype = np.dtype([('res', res_label.dtype), ('gt', gt_label.dtype)])

    sum_iou = 0
    sum_gt_objects = 0
    sum_pred_objects = 0

    for t in range(len(res_label)):

        label_tuples = np.empty(res_label[t].shape, dtype=compare_dtype)
        label_tuples['res'] = res_label[t]
        label_tuples['gt'] = gt_label[t]

        both_foreground = np.logical_and(label_tuples['res'] > 0, label_tuples['gt'] > 0)
        index_pairs, intersections = np.unique(label_tuples[both_foreground], return_counts=True)
        gt_indexes, gt_size = np.unique(label_tuples['gt'][label_tuples['gt'] > 0], return_counts=True)
        sum_gt_objects += len(gt_indexes)
        sum_pred_objects += len(np.unique(np.unique(label_tuples['res'][label_tuples['res'] > 0])))

        if return_matches:
            for gt_idx in gt_indexes:
                matches[(t, gt_idx)] = (0., 0)

        for (res_idx, gt_idx), intersection in zip(index_pairs, intersections):
            gt_size = (label_tuples['gt'] == gt_idx).sum()
            res_size = (label_tuples['res'] == res_idx).sum()
            overlap = intersection / gt_size
            if overlap > overlap_threshold:
                iou = intersection / (gt_size + res_size - intersection)
                sum_iou += iou
                if return_matches:
                    matches[(t, gt_idx)] = (iou, res_idx)
                if iou > match_iou:
                    n_matches += 1

    scores = {}
    
    scores["sum_gt_objects"] = sum_gt_objects
    scores["sum_pred_objects"] = sum_pred_objects
    
    if sum_gt_objects == 0:
        scores["seg"] = 0
        recall = 0
    else:
        recall = n_matches / sum_gt_objects
        scores["seg"] = sum_iou / sum_gt_objects
    
    if sum_pred_objects == 0:
        precision = 0    
    else:
        precision = n_matches / sum_pred_objects
        
    scores["recall"] = recall
    scores["precision"] = precision
    scores["f1"] = hmean([recall, precision])
    if return_matches:
        scores["matched_iou"] = [float(v[0]) for v in matches.values()]
        scores["matched_res_idx"] = [int(v[1]) for v in matches.values()]
        scores["matched_t"] = [int(k[0]) for k in matches.keys()]
        scores["matched_gt_idx"] = [int(k[1]) for k in matches.keys()]

    return scores
