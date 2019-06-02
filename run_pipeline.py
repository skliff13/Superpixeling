from run01_precalculate import precalculate_dataset
from run02_collect_sp_features import collect_sp_features
from run03_create_sp_dictionary import create_sp_dictionaries
from run04_calc_sp_descriptors import calc_sp_descriptors
from run05_collect_descriptors import collect_sp_descriptors
from run06_evaluate import evaluate_sp_descriptors
from run07_highlight_regions import build_heat_maps


def main():
    data_dir = 'data/ovary'
    config_path = 'config0.json'

    precalculate_dataset(data_dir, config_path)
    collect_sp_features(data_dir, config_path)
    create_sp_dictionaries(data_dir, config_path)
    calc_sp_descriptors(data_dir, config_path)
    collect_sp_descriptors(data_dir, config_path)
    evaluate_sp_descriptors(data_dir, config_path)
    build_heat_maps(data_dir, config_path)


if __name__ == '__main__':
    main()
