from run_a_precalculate import precalculate_dataset
from run_b_collect_sp_features import collect_sp_features
from run_c_create_sp_dictionary import create_sp_dictionaries
from run_d_calc_sp_descriptors import calc_sp_descriptors
from run_e_collect_descriptors import collect_sp_descriptors
from run_f_highlight_regions import build_heat_maps


def main():
    data_dir = 'data/ct2d'
    config_path = 'config0.json'

    sp_str = '8_0.3'
    dictionary_size = 16

    precalculate_dataset(data_dir, config_path)
    collect_sp_features(data_dir, config_path)
    create_sp_dictionaries(data_dir, config_path)
    calc_sp_descriptors(data_dir, config_path)
    collect_sp_descriptors(data_dir, config_path)
    build_heat_maps(data_dir, sp_str, dictionary_size)


if __name__ == '__main__':
    main()
