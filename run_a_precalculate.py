import os
import json
import imageio
import pandas as pd
from skimage import io, img_as_float

from super_precalc import super_precalc2d


def precalculate_dataset(data_dir, config_path):
    print('Precalculating data in ' + data_dir)

    path = os.path.join(data_dir, '_idx.txt')
    case_ids = list(pd.read_csv(path, header=None)[0].get_values())

    with open(config_path, 'r') as f:
        config = json.load(f)
    super_params = config['SuperParameters']

    for i, case_id in enumerate(case_ids):
        print('%02i / %02i : %s' % (i + 1, len(case_ids), case_id))

        out_dir = os.path.join(data_dir, 'SPdata', case_id[:-4])
        os.makedirs(out_dir, exist_ok=True)

        im = io.imread(os.path.join(data_dir, case_id))
        im = img_as_float(im)

        for sp_str in super_params:
            precalculate_for_setup(im, out_dir, sp_str)

    print('Done\n')


def precalculate_for_setup(im, out_dir, sp_str):
    super_sz = int(sp_str.split('_')[0])
    super_reg = float(sp_str.split('_')[1])

    out_map_path = os.path.join(out_dir, 'sp_map_%s.png' % sp_str)
    out_fts_path = os.path.join(out_dir, 'sp_features_%s.txt' % sp_str)
    out_pairs_path = os.path.join(out_dir, 'sp_pairs_%s.txt' % sp_str)
    out_brd_path = os.path.join(out_dir, 'sp_borders_%s.png' % sp_str)

    if not os.path.isfile(out_brd_path):
        features, pairs, sp_map, sp_borders = super_precalc2d(im, super_sz, super_reg)

        imageio.imwrite(out_map_path, sp_map)
        features.to_csv(out_fts_path, index=None)
        pairs.to_csv(out_pairs_path, index=None)
        imageio.imwrite(out_brd_path, sp_borders)


def main():
    data_dir = 'data/ct2d'
    config_path = 'config0.json'

    precalculate_dataset(data_dir, config_path)


if __name__ == '__main__':
    main()
