# coding=utf-8

import cv2
import numpy as np


def mask_iou(mask1, mask2):
    """
    :param mask1:  l
    :param mask2:  l
    :return: iou
    """
    mask1 =mask1.reshape([-1])
    mask2=mask2.reshape([-1])
    t = np.array(mask1 > 0.5)
    p = mask2 > 0.
    intersection = np.logical_and(t, p)
    union = np.logical_or(t, p)
    iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)

    ap = dict()
    thresholds = np.arange(0.5, 1, 0.05)
    s = []
    for thresh in thresholds:
        ap[thresh] = float(iou > thresh)
    return iou,ap


def mask_processing(mask,info_img):
    h, w, nh, nw, dx, dy,_=info_img
    mask=mask[dy:dy + nh, dx:dx + nw,None]
    mask=cv2.resize(mask,(int(w),int(h)))
    return mask