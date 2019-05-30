import os
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt


def colorize_heat_map(im, hm, max_heat=3):
    hm_gray = 0.5 + 0.5 * hm / max_heat
    hm_gray[hm_gray > 1] = 1
    hm_gray[hm_gray < 0] = 0

    cmap = plt.get_cmap('jet')
    rgb1 = cmap(hm_gray)

    ch = im[..., np.newaxis]
    rgb0 = np.concatenate((ch, ch, ch), axis=2)

    overlay = rgb0 * 0.7 + 0.4 * rgb1[..., :3]
    overlay[overlay > 1] = 1
    overlay[overlay < 0] = 0

    overlay = (overlay * 255).astype(np.uint8)
    hm_gray = (hm_gray * 255).astype(np.uint8)

    return hm_gray, overlay


def build_heat_map(dictionary_size, im, sp_map, pairs, scores, sp_classes):
    hm = im * 0
    for pair in pairs:
        c1 = sp_classes[str(pair[0])]
        c2 = sp_classes[str(pair[1])]

        desc_element = (c2 - 1) * dictionary_size + c1 - 1
        score = scores[desc_element]
        if score > 0:
            selection = np.logical_or(sp_map == pair[0], sp_map == pair[1])
            hm[selection] += score

    return hm


def calc_scores(data_dir, dictionary_size, sp_str):
    path = os.path.join(data_dir, '_cls.txt')
    y = pd.read_csv(path, header=None).get_values()

    spcms_path = os.path.join(data_dir, 'SPdata', 'descs_spcm_%s_%i.txt' % (sp_str, dictionary_size))
    x = pd.read_csv(spcms_path, header=None).get_values()

    scores = np.zeros((x.shape[1], ))
    for j in range(x.shape[1]):
        col = x[:, j:j + 1]
        if np.std(col) > 0:
            r = pearsonr(col, y)[0][0]
            scores[j] = np.sign(r) * r ** 2

    return scores
