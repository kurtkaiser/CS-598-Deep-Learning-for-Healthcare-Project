from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import argparse
import os


def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def process_table(table_name, table, stay_list, folder_path):
    table = table.loc[stay_list].copy()
    table.to_csv('{}/{}.csv'.format(folder_path, table_name))
    return

def shuffle_stays(stay_list, seed=9):
    return shuffle(stay_list, random_state=seed)

def split_train_test(path, is_test=True, seed=9, cleanup=True, MIMIC=False):

    time_str = 'preprocessed_timeseries.csv'
    diag_str = 'preprocessed_diagnoses.csv'
    labels_str = 'preprocessed_labels.csv'
    flat_str = 'preprocessed_flat.csv'

    labels = pd.read_csv(path + labels_str)
    labels.set_index('patient', inplace=True)
    patients = labels.uniquepid.unique()

    train, test = train_test_split(patients, test_size=0.15, random_state=seed)
    train, val = train_test_split(train, test_size=0.15/0.85, random_state=seed)

    # print('Loading data for splitting...')
    if is_test:
        timeseries = pd.read_csv(path + time_str, nrows=999999)
    else:
        timeseries = pd.read_csv(path + time_str)
    timeseries.set_index('patient', inplace=True)
    if not MIMIC:
        diagnoses = pd.read_csv(path + diag_str)
        diagnoses.set_index('patient', inplace=True)
    flat_features = pd.read_csv(path + flat_str)
    flat_features.set_index('patient', inplace=True)
    # print(flat_features)
    # delete the source files, as they won't be needed anymore
    if is_test is False and cleanup:
        print('Removing the unsorted data...')
        os.remove(path + time_str)
        if not MIMIC:
            os.remove(path + diag_str)
        os.remove(path + labels_str)
        os.remove(path + flat_str)

    for partition_name, partition in zip(['train', 'val', 'test'], [train, val, test]):
        print('Preparing {} data...'.format(partition_name))
        stays = labels.loc[labels['uniquepid'].isin(partition)].index
        folder_path = create_folder(path, partition_name)
        with open(folder_path + '/stays.txt', 'w') as f:
            for stay in stays:
                f.write("%s\n" % stay)
        stays = shuffle_stays(stays, seed=9)
        if not MIMIC:
            for table_name, table in zip(['labels', 'flat', 'diagnoses', 'timeseries'],
                                         [labels, flat_features, diagnoses, timeseries]):
                process_table(table_name, table, stays, folder_path)
        else:
            for table_name, table in zip(['labels', 'flat', 'timeseries'],
                                         [labels, flat_features, timeseries]):
                process_table(table_name, table, stays, folder_path)

    return

if __name__=='__main__':
    from eICU_preprocessing.run_all_preprocessing import eICU_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleanup', action='store_true')
    args = parser.parse_args()
    split_train_test(eICU_path, is_test=False, cleanup=args.cleanup)