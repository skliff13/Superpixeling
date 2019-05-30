import os
import json
import pandas as pd
import numpy as np


def collect_sp_features(data_dir, config_path):
    print('Collecting SP features in ' + data_dir)

    path = os.path.join(data_dir, '_idx.txt')
    case_ids = list(pd.read_csv(path, header=None)[0].get_values())

    with open(config_path, 'r') as f:
        config = json.load(f)

    super_params = config['SuperParameters']

    for sp_str in super_params:
        print('SuperParameters:', sp_str)

        collect_features_for_setup(case_ids, data_dir, sp_str)

    print('Done\n')


def collect_features_for_setup(case_ids, data_dir, sp_str):
    tbl = None
    columns = None
    for i, case_id in enumerate(case_ids):
        if i % 50 == 0:
            print('%02i / %02i' % (i + 1, len(case_ids)))

        case_dir = os.path.join(data_dir, 'SPdata', case_id[:-4])
        fts_path = os.path.join(case_dir, 'sp_features_%s.txt' % sp_str)
        features_table = pd.read_csv(fts_path)

        data = features_table.get_values()[:, 1:]
        if tbl is None:
            tbl = data
            columns = list(features_table.columns.values)[1:]
        else:
            tbl = np.concatenate((tbl, data))

    out_path = os.path.join(data_dir, 'SPdata', 'sp_features_%s.txt' % sp_str)
    print('Writing SP features table to ' + out_path)
    df = pd.DataFrame(tbl, columns=columns)
    df.to_csv(out_path, index=None)


def main():
    data_dir = 'data/ct2d'
    config_path = 'config0.json'

    collect_sp_features(data_dir, config_path)


if __name__ == '__main__':
    main()
