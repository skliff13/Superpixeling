import numpy as np
from scipy.spatial.distance import cdist


def super_describe(features, pairs, dictionary):
    num_bins = 16
    ranges = [(0, 1), (0, 0.2), (0, 1), (0, 1), (0, 1.5), (0, 1.5), (0, 1), (0, 1.8), (0, 0.4)]

    num_classes = dictionary.shape[0] - 1
    sp_classes = calc_sp_classes(features, dictionary)
    chist = calc_sp_classes_hist(sp_classes, num_classes)

    fhist = cals_sp_features_hist(features, ranges, num_bins)

    spcm = calc_sp_cooccurrence(sp_classes, pairs, num_classes)

    return fhist, chist, spcm, sp_classes


def calc_sp_cooccurrence(sp_classes, pairs, num_classes):
    spcm = np.zeros((num_classes, num_classes))

    for pair in pairs:
        c1 = sp_classes[pair[0]]
        c2 = sp_classes[pair[1]]

        spcm[c1 - 1, c2 - 1] += 1

    spcm += np.transpose(spcm)

    return spcm


def cals_sp_features_hist(features, ranges, num_bins):
    bin_edges = np.arange(0, 1.001, 1 / num_bins)

    hists = []
    for j, rng in enumerate(ranges):
        feature_values = features[:, 1 + j]
        feature_values = (feature_values - rng[0]) / (rng[1] - rng[0])

        hist = np.histogram(feature_values, bin_edges)[0]
        hists.append(hist)

    fhist = np.concatenate(tuple(hists))
    return fhist


def calc_sp_classes(features, dictionary):
    features_std = dictionary[0, :]
    centroids = dictionary[1:, :]

    sp_classes = {}
    for sp_features in features:
        sp_id = sp_features[0]
        sp_features = sp_features[1:] / features_std

        dists = cdist(sp_features[np.newaxis, ...], centroids).flatten()
        sp_classes[int(sp_id)] = int(1 + np.argwhere(dists == dists.min())[0][0])

    return sp_classes


def calc_sp_classes_hist(sp_classes, num_classes):
    chist = np.zeros((num_classes, ))
    for sp_class in sp_classes.values():
        chist[sp_class - 1] += 1

    return chist
