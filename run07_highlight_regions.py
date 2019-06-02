import os
import warnings
import json
import imageio
import pandas as pd
from skimage import io, img_as_float

from highlighting import calc_scores, build_heat_map, colorize_heat_map


def build_heat_maps(data_dir, config_path):
    print('Building heat maps for ' + data_dir)

    warnings.filterwarnings('ignore')

    with open(config_path, 'r') as f:
        config = json.load(f)

    super_params = config['SuperParameters']
    dictionary_sizes = config['DictionarySizes']

    for sp_str in super_params:
        for dictionary_size in dictionary_sizes:
            build_heat_maps_for_combination(data_dir, sp_str, dictionary_size)


def build_heat_maps_for_combination(data_dir, sp_str, dictionary_size):
    print('%s_%i' % (sp_str, dictionary_size))

    path = os.path.join(data_dir, '_idx.txt')
    case_ids = list(pd.read_csv(path, header=None)[0].get_values())

    scores = calc_scores(data_dir, dictionary_size, sp_str)

    overlays_dir = os.path.join(data_dir, 'overlays', '%s_%i' % (sp_str, dictionary_size))
    os.makedirs(overlays_dir, exist_ok=True)

    for i, case_id in enumerate(case_ids):
        if i % 50 == 0:
            print('%02i / %02i' % (i + 1, len(case_ids)))

        im = io.imread(os.path.join(data_dir, case_id))
        im = img_as_float(im)
        case_dir = os.path.join(data_dir, 'SPdata', case_id[:-4])

        sp_map, pairs, sp_classes = load_case_data(case_dir, dictionary_size, sp_str)

        hm = build_heat_map(dictionary_size, im, sp_map, pairs, scores, sp_classes)

        hm_gray, overlay = colorize_heat_map(im, hm)

        out_heatmap_path = os.path.join(case_dir, '%s_heatmap_%s_%i.png' % (case_id[:-4], sp_str, dictionary_size))
        out_overlay_path = os.path.join(overlays_dir, '%s_overlay_%s_%i.png' % (case_id[:-4], sp_str, dictionary_size))
        io.imsave(out_heatmap_path, hm_gray)
        io.imsave(out_overlay_path, overlay)

    print('Done\n')


def load_case_data(case_dir, dictionary_size, sp_str):
    map_path = os.path.join(case_dir, 'sp_map_%s.png' % sp_str)
    pairs_path = os.path.join(case_dir, 'sp_pairs_%s.txt' % sp_str)
    classes_path = os.path.join(case_dir, 'sp_classes_%s_%i.json' % (sp_str, dictionary_size))

    sp_map = imageio.imread(map_path)
    pairs = pd.read_csv(pairs_path).get_values()
    with open(classes_path, 'r') as f:
        sp_classes = json.load(f)

    return sp_map, pairs, sp_classes


def main():
    data_dir = 'data/ct2d'
    config_path = 'config0.json'

    build_heat_maps(data_dir, config_path)


if __name__ == '__main__':
    main()
