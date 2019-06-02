import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist


def evaluate_sp_descriptors(data_dir, config_path):
    print('Collecting SP descriptors for ' + data_dir)

    path = os.path.join(data_dir, '_idx.txt')

    case_ids = list(pd.read_csv(path, header=None)[0].get_values())
    y = list(pd.read_csv(path, header=None)[1].get_values())
    with open(config_path, 'r') as f:
        config = json.load(f)

    super_params = config['SuperParameters']
    dictionary_sizes = config['DictionarySizes']

    aucs = []
    for _ in dictionary_sizes:
        aucs.append([0] * len(super_params))

    for isp, sp_str in enumerate(super_params):
        for id, dictionary_size in enumerate(dictionary_sizes):
            spcms_path = os.path.join(data_dir, 'SPdata', 'descs_spcm_%s_%i.txt' % (sp_str, dictionary_size))
            descs = pd.read_csv(spcms_path, header=None).get_values()

            auc = evaluate_loocv_auc(descs, y)
            auc = round(auc, 4)

            print('descs_spcm_%s_%i.txt: AUC = %f' % (sp_str, dictionary_size, auc))
            aucs[id][isp] = auc

    d = {'sp_params': super_params}
    columns = ['sp_params']
    for id, dictionary_size in enumerate(dictionary_sizes):
        column = 'dict%i' % dictionary_size
        d[column] = aucs[id]
        columns.append(column)

    out_path = os.path.join(data_dir, 'SPdata', 'descs_spcm_AUCs.txt')
    print('Writing results to ' + out_path)
    pd.DataFrame(d, columns=columns).to_csv(out_path, index=None)


def evaluate_loocv_auc(descs, y, k_neighbours=10):
    y = np.array(y)
    pred = np.zeros((y.shape[0], ), dtype=float)

    ii = np.arange(0, descs.shape[0])
    for i in range(descs.shape[0]):
        x_train = descs[ii != i]
        y_train = y[ii != i]

        dists = cdist(descs[ii == i], x_train, metric='cityblock')
        dy = list(zip(list(dists.flatten()), list(y_train)))
        dy.sort()

        score = 0
        for j in range(k_neighbours):
            score += dy[j][1] / k_neighbours

        pred[i] = score

    auc = roc_auc_score(y.flatten(), pred.flatten())
    print(auc)
    return auc


def main():
    data_dir = 'data/ct2d'
    config_path = 'config0.json'

    evaluate_sp_descriptors(data_dir, config_path)


if __name__ == '__main__':
    main()
