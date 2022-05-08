from eICU_preprocessing.timeseries import timeseries_main
from eICU_preprocessing.diagnoses import diagnoses_main
from eICU_preprocessing.flat_and_labels import flat_and_labels_main
from eICU_preprocessing.split_train_test import split_train_test
import json
import os


with open('paths.json', 'r') as f:
    # print(r)
    eICU_path = json.load(f)["eICU_path"]

if __name__=='__main__':
    print('Removing the stays.txt file if it exists...')
    try:
        os.remove(eICU_path + 'stays.txt')
    except FileNotFoundError:
        pass
    timeseries_main(eICU_path, test=False)
    # print(eICU_path)
    diagnoses_main(eICU_path, 0.01)
    flat_and_labels_main(eICU_path)
    split_train_test(eICU_path, is_test=False)