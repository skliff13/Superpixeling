import os
import json
import numpy as np
import pandas as pd


def collect_sp_descriptors(data_dir, config_path):
    print('Collecting SP descriptors for ' + data_dir)

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

        chists, fhists, spcms = collect_for_combination(case_ids, data_dir, dictionary_size, sp_str)

        out_fhists_path = os.path.join(data_dir, 'SPdata', 'descs_fhist_%s.txt' % sp_str)
        out_chists_path = os.path.join(data_dir, 'SPdata', 'descs_chist_%s_%i.txt' % (sp_str, dictionary_size))
        out_spcms_path = os.path.join(data_dir, 'SPdata', 'descs_spcm_%s_%i.txt' % (sp_str, dictionary_size))

        for x, out_path in zip([fhists, chists, spcms], [out_fhists_path, out_chists_path, out_spcms_path]):
            print('Saving SP descs to ' + out_path)
            pd.DataFrame(x).to_csv(out_path, header=None, index=None)

    print('Done\n')


def collect_for_combination(case_ids, data_dir, dictionary_size, sp_str):
    fhists = None
    chists = None
    spcms = None
    for i, case_id in enumerate(case_ids):
        if i % 50 == 0:
            print('%02i / %02i' % (i + 1, len(case_ids)))

        case_dir = os.path.join(data_dir, 'SPdata', case_id[:-4])
        fhist_path = os.path.join(case_dir, 'desc_fhist_%s.txt' % sp_str)
        chist_path = os.path.join(case_dir, 'desc_chist_%s_%i.txt' % (sp_str, dictionary_size))
        spcm_path = os.path.join(case_dir, 'desc_spcm_%s_%i.txt' % (sp_str, dictionary_size))

        fhist = pd.read_csv(fhist_path, header=None).get_values().flatten()[np.newaxis, ...]
        chist = pd.read_csv(chist_path, header=None).get_values().flatten()[np.newaxis, ...]
        spcm = pd.read_csv(spcm_path, header=None).get_values().flatten()[np.newaxis, ...]

        if fhists is None:
            fhists = fhist
            chists = chist
            spcms = spcm
        else:
            fhists = np.concatenate((fhists, fhist), axis=0)
            chists = np.concatenate((chists, chist), axis=0)
            spcms = np.concatenate((spcms, spcm), axis=0)
    return chists, fhists, spcms


def main():
    data_dir = 'data/ct2d'
    config_path = 'config0.json'

    collect_sp_descriptors(config_path, data_dir)


if __name__ == '__main__':
    main()
