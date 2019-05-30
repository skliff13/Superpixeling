import os
import json
import imageio
import numpy as np
import pandas as pd
from skimage import io, img_as_float
from skimage.segmentation import slic
from skimage.morphology import binary_dilation, binary_erosion, binary_closing, binary_opening
from scipy.ndimage import sobel


def super_precalc2d(im, super_sz, super_reg):
    feature_names = ('intensityMean', 'intensitySD', 'intensityEntropy', 'insideGradient', 'borderGradient',
                     'compactness', 'squareness', 'positiveSpectrum', 'negativeSpectrum')

    im = im.astype(float)
    sp_map = extract_superpixels(im, super_sz, super_reg)

    sp_features = calc_features(im, sp_map, feature_names)
    features_table = pd.DataFrame(sp_features, columns=['SP_id'] + list(feature_names))

    pairs = sp_pairs(sp_map)
    pairs = pd.DataFrame(pairs, columns=('ID1', 'ID2'))

    gm = grad2d(sp_map)
    sp_borders = (gm > 0).astype(np.uint8) * 255

    return features_table, pairs, sp_map, sp_borders


def sp_pairs(sp_map):
    global_border = sp_map * 0
    global_border[0, :] = 0
    global_border[-1, :] = 0
    global_border[:, 0] = 0
    global_border[:, -1] = 0

    sp_ids = set(sp_map.flatten())
    sp_ids = list(sp_ids)

    se1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(bool)

    pairs = []
    for sp_id in sp_ids:
        s = sp_map == sp_id

        s = _and_not(binary_dilation(s, se1), s)

        s = np.logical_and(s, sp_map > sp_id)

        sp2s = list(set(sp_map[s].flatten()))

        for sp2 in sp2s:
            pairs.append([sp_id, sp2])

    return np.array(pairs)


def calc_features(im, sp_map, feature_names):
    sp_ids = set(sp_map.flatten())
    sp_ids = list(sp_ids)

    gm = grad2d(im)

    se1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(bool)
    se2 = np.pad(se1, 1, mode='constant')
    se2 = binary_dilation(se2, se1)

    tbl = np.zeros((len(sp_ids), 1 + len(feature_names)))

    for i, sp_id in enumerate(sp_ids):
        selected_sp = sp_map == sp_id

        sp_cropped, ii, jj = get_cropping(selected_sp, bounding=2)

        inner = binary_erosion(sp_cropped, se1)
        inner[0, :] = 0
        inner[-1, :] = 0
        inner[:, 0] = 0
        inner[:, -1] = 0

        im_cropped = im[ii[0]:ii[1], jj[0]:jj[1]]
        gm_cropped = gm[ii[0]:ii[1], jj[0]:jj[1]]
        bbox, _, _ = get_cropping(selected_sp, bounding=0)

        if np.sum(sp_cropped) > 16:
            feature_values = calc_sp_features(sp_cropped, inner, im_cropped, gm_cropped, bbox, se1, se2)

            tbl[i, :] = np.array([float(sp_id)] + list(feature_values))

    tbl = tbl[tbl[:, 0] > 0, :]

    return tbl


def calc_sp_features(sp, inner, im, gm, bbox, se1, se2):
    four_pi = 12.5664
    area = np.sum(sp)
    perimeter = np.sum(_and_not(sp, inner))
    compactness = four_pi * area / perimeter**2

    if np.sum(inner) < 4:
        inner = sp

    mean_inner = np.mean(im[inner])
    std_inner = np.std(im[inner])
    grad_inner = np.mean(gm[inner])

    border = _and_not(sp, inner)
    if np.sum(border) > 0:
        grad_border = np.mean(gm[border])
    else:
        grad_border = grad_inner

    num_bins = 16
    bin_edges = np.arange(0, 1.001,  1 / num_bins)
    h, _ = np.histogram(im[inner].flatten(), bin_edges)
    h = h[h > 0] / np.sum(h)
    entropy = - np.sum(h * np.log(h)) / np.log(num_bins)

    squareness = np.sum(bbox) / np.size(bbox)

    q = _and_not(binary_opening(sp, se1), binary_opening(sp, se2))
    positive = np.sum(q) / perimeter
    q = _and_not(binary_closing(sp, se1), binary_closing(sp, se2))
    negative = np.sum(q) / perimeter

    return mean_inner, std_inner, entropy, grad_inner, grad_border, compactness, squareness, positive, negative


def _and_not(a, b):
    return np.logical_and(a, np.logical_not(b))


def grad2d(im):
    gx = sobel(im, axis=0, mode='constant')
    gy = sobel(im, axis=1, mode='constant')
    return np.hypot(gx, gy)


def get_cropping(bw, proj_thres=0, bounding=0):
    b = bounding

    proj0 = np.sum(bw, axis=1).flatten() > proj_thres
    proj1 = np.sum(bw, axis=0).flatten() > proj_thres

    pos0 = np.argwhere(proj0).flatten()
    pos1 = np.argwhere(proj1).flatten()

    ii = (max(0, pos0[0] - b), min(pos0[-1] + b + 1, bw.shape[0]))
    jj = (max(0, pos1[0] - b), min(pos1[-1] + b + 1, bw.shape[1]))

    cropped = bw[ii[0]:ii[1], jj[0]:jj[1]]

    return cropped, ii, jj


def extract_superpixels(im, sz, reg):
    num = int(0.5 + im.shape[0] / sz) * int(0.5 + im.shape[1] / sz)
    sp_map = slic(im, n_segments=num, compactness=reg, enforce_connectivity=True)
    return sp_map.astype(np.uint16) + 1
