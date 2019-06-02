import os
import json
import pandas as pd

from super_describe import super_describe


def calc_sp_descriptors(data_dir, config_path):
    print('Calculating SP descriptors for ' + data_dir)

    path = os.path.join(data_dir, '_idx.txt')
    case_ids = list(pd.read_csv(path, header=None)[0].get_values())

    with open(config_path, 'r') as f:
        config = json.load(f)

    super_params = config['SuperParameters']
    dictionary_sizes = config['DictionarySizes']

    combinations = []
    for sp_str in super_params:
        for dictionary_size in dictionary_sizes:
            combinations.append((sp_str, dictionary_size))

    for combination in combinations:
        sp_str, dictionary_size = combination
        super_sz = int(sp_str.split('_')[0])
        super_reg = float(sp_str.split('_')[1])
        print('Sz=%i, Reg=%f, N=%i' % (super_sz, super_reg, dictionary_size))

        dict_path = os.path.join(data_dir, 'SPdata', 'sp_dict_%s_%i.txt' % (sp_str, dictionary_size))
        dictionary = pd.read_csv(dict_path).get_values()

        for i, case_id in enumerate(case_ids):
            if i % 20 == 0:
                print('%02i / %02i' % (i + 1, len(case_ids)))

            calc_for_case(case_id, data_dir, dictionary, dictionary_size, sp_str)

    print('Done\n')


def calc_for_case(case_id, data_dir, dictionary, dictionary_size, sp_str):
    case_dir = os.path.join(data_dir, 'SPdata', case_id[:-4])
    fts_path = os.path.join(case_dir, 'sp_features_%s.txt' % sp_str)
    pairs_path = os.path.join(case_dir, 'sp_pairs_%s.txt' % sp_str)

    features = pd.read_csv(fts_path).get_values()
    pairs = pd.read_csv(pairs_path).get_values()

    out_fhist_path = os.path.join(case_dir, 'desc_fhist_%s.txt' % sp_str)
    out_chist_path = os.path.join(case_dir, 'desc_chist_%s_%i.txt' % (sp_str, dictionary_size))
    out_spcm_path = os.path.join(case_dir, 'desc_spcm_%s_%i.txt' % (sp_str, dictionary_size))
    out_classes_path = os.path.join(case_dir, 'sp_classes_%s_%i.json' % (sp_str, dictionary_size))

    if not os.path.isfile(out_spcm_path):
        fhist, chist, spcm, sp_classes = super_describe(features, pairs, dictionary)

        pd.DataFrame(fhist).to_csv(out_fhist_path, index=None, header=None)
        pd.DataFrame(chist).to_csv(out_chist_path, index=None, header=None)
        pd.DataFrame(spcm).to_csv(out_spcm_path, index=None, header=None)
        with open(out_classes_path, 'w') as f:
            json.dump(sp_classes, f, indent=2)


def main():
    data_dir = 'data/ct2d'
    config_path = 'config0.json'

    calc_sp_descriptors(data_dir, config_path)


if __name__ == '__main__':
    main()
