import numpy as np


def iou(box, boxes):
    """Computes the intersection over union for a given axis
    aligned bounding box with several others.
    # Arguments
        box: Bounding box, numpy array of shape (4).
            (x1, y1, x2, y2)
        boxes: Reference bounding boxes, numpy array of
            shape (num_boxes, 4).
    # Return
        iou: Intersection over union,
            numpy array of shape (num_boxes).
    """
    # compute intersection
    inter_upleft = np.maximum(boxes[:, :2], box[:2])
    inter_botright = np.minimum(boxes[:, 2:4], box[2:])
    inter_wh = inter_botright - inter_upleft
    inter_wh = np.maximum(inter_wh, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    # compute union
    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_pred + area_gt - inter
    # compute iou
    iou = inter / union
    return iou


def non_maximum_suppression_slow(boxes, confs, iou_threshold, top_k):
    """Does None-Maximum Suppresion on detection results.

    Intuitive but slow as hell!!!

    # Agruments
        boxes: Array of bounding boxes (boxes, xmin + ymin + xmax + ymax).
        confs: Array of corresponding confidenc values.
        iou_threshold: Intersection over union threshold used for comparing
            overlapping boxes.
        top_k: Maximum number of returned indices.

    # Return
        List of remaining indices.
    """
    idxs = np.argsort(-confs)
    selected = []
    for idx in idxs:
        if np.any(iou(boxes[idx], boxes[selected]) >= iou_threshold):
            continue
        selected.append(idx)
        if len(selected) >= top_k:
            break
    return selected


def non_maximum_suppression(boxes, confs, overlap_threshold, top_k):
    """Does None-Maximum Suppresion on detection results.

    # Agruments
        boxes: Array of bounding boxes (boxes, xmin + ymin + xmax + ymax).
        confs: Array of corresponding confidenc values.
        overlap_threshold:
        top_k: Maximum number of returned indices.

    # Return
        List of remaining indices.

    # References
        - Girshick, R. B. and Felzenszwalb, P. F. and McAllester, D.
          [Discriminatively Trained Deformable Part Models, Release 5](http://people.cs.uchicago.edu/~rbg/latent-release5/)
    """
    eps = 1e-15

    boxes = np.asarray(boxes, dtype='float32')

    pick = []
    x1, y1, x2, y2 = boxes.T

    idxs = np.argsort(confs)
    area = (x2 - x1) * (y2 - y1)

    while len(idxs) > 0:
        i = idxs[-1]

        pick.append(i)
        if len(pick) >= top_k:
            break

        idxs = idxs[:-1]

        xx1 = np.maximum(x1[i], x1[idxs])
        yy1 = np.maximum(y1[i], y1[idxs])
        xx2 = np.minimum(x2[i], x2[idxs])
        yy2 = np.minimum(y2[i], y2[idxs])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        I = w * h

        overlap = I / (area[idxs] + eps)
        # as in Girshick et. al.

        # U = area[idxs] + area[i] - I
        # overlap = I / (U + eps)

        idxs = idxs[overlap <= overlap_threshold]

    return pick
