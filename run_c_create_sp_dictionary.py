import os
import json
import pandas as pd
import numpy as np
from numpy.matlib import repmat
from random import sample
from sklearn.cluster import k_means


def create_sp_dictionaries(data_dir, config_path, max_samples=10000):
    print('Creating SP dictionaries for ' + data_dir)

    with open(config_path, 'r') as f:
        config = json.load(f)

    super_params = config['SuperParameters']
    dictionary_sizes = config['DictionarySizes']

    for sp_str in super_params:
        print('SuperParameters:', sp_str)

        features_path = os.path.join(data_dir, 'SPdata', 'sp_features_%s.txt' % sp_str)
        df_features = pd.read_csv(features_path)
        features = df_features.get_values()

        features_std = np.std(features, axis=0)
        features_std[features_std == 0] = 1

        features /= repmat(features_std, features.shape[0], 1)

        for dsz in dictionary_sizes:
            create_for_dictionary_size(data_dir, df_features, dsz, features, features_std, max_samples, sp_str)

    print('Done\n')


def create_for_dictionary_size(data_dir, df_features, dsz, features, features_std, max_samples, sp_str):
    if features.shape[0] > max_samples:
        rows = sample(range(features.shape[0]), max_samples)
        features = features[np.array(rows), :]

    output = k_means(features, dsz, init='random', max_iter=1000)
    centroids = output[0]

    to_save = np.concatenate((features_std[np.newaxis, ...], centroids), axis=0)
    out_path = os.path.join(data_dir, 'SPdata', 'sp_dict_%s_%i.txt' % (sp_str, dsz))
    print('Saving SP dictionary to ' + out_path)
    df = pd.DataFrame(to_save, columns=list(df_features.columns.values))
    df.to_csv(out_path, index=None)


def main():
    data_dir = 'data/ct2d'
    config_path = 'config0.json'

    create_sp_dictionaries(data_dir, config_path)


if __name__ == '__main__':
    main()
