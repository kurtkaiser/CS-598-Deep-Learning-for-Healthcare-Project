import pandas as pd
import numpy as np
from itertools import islice
import os
import json

def reconfigure_timeseries(timeseries, offset_column, feature_column=None, test=False):
    if test:
        timeseries = timeseries.iloc[300000:5000000]
    timeseries.set_index(['patientunitstayid', pd.to_timedelta(timeseries[offset_column], unit='T')], inplace=True)
    timeseries.drop(columns=offset_column, inplace=True)
    if feature_column is not None:
        timeseries = timeseries.pivot_table(columns=feature_column, index=timeseries.index)
    # convert index to multi-index with both patients and timedelta stamp
    timeseries.index = pd.MultiIndex.from_tuples(timeseries.index, names=['patient', 'time'])
    return timeseries

def resample_and_mask(timeseries, eICU_path, header, mask_decay=True, decay_rate=4/3, test=False,verbose=False, length_limit=24*14):
    if test:
        mask_decay = False
        verbose = True
    if verbose:
        print('Resampling to 1 hour intervals...')
    # take the mean of any duplicate index entries for unstacking
    timeseries = timeseries.groupby(level=[0, 1]).mean()

    timeseries.reset_index(level=1, inplace=True)
    timeseries.time = timeseries.time.dt.ceil(freq='H')
    timeseries.set_index('time', append=True, inplace=True)
    timeseries.reset_index(level=0, inplace=True)
    resampled = timeseries.groupby('patient').resample('H', closed='right', label='right').mean().drop(columns='patient')
    del (timeseries)

    def apply_mask_decay(mask_bool):
        mask = mask_bool.astype(int)
        mask.replace({0: np.nan}, inplace=True)  # so that forward fill works
        inv_mask_bool = ~mask_bool
        count_non_measurements = inv_mask_bool.cumsum() - \
                                 inv_mask_bool.cumsum().where(mask_bool).ffill().fillna(0)
        decay_mask = mask.ffill().fillna(0) / (count_non_measurements * decay_rate).replace(0, 1)
        return decay_mask

    # store which values had to be imputed
    if mask_decay:
        if verbose:
            print('Calculating mask decay features...')
        mask_bool = resampled.notnull()
        mask = mask_bool.groupby('patient').transform(apply_mask_decay)
        del (mask_bool)
    else:
        if verbose:
            print('Calculating binary mask features...')
        mask = resampled.notnull()
        mask = mask.astype(int)

    if verbose:
        print('Filling missing data forwards...')
    # carry forward missing values (note they will still be 0 in the nulls table)
    resampled = resampled.fillna(method='ffill')

    mask1 = mask.index.levels[1]
    mask = mask.rename(index=dict(zip(mask1, mask1.days*24 + mask1.seconds//3600)))
    re1 = resampled.index.levels[1]
    resampled = resampled.rename(index=dict(zip(re1, re1.days*24 + re1.seconds//3600)))

    # clip to length_limit
    if length_limit is not None:
        within_length_limit = resampled.index.get_level_values(1) < length_limit
        resampled = resampled.loc[within_length_limit]
        mask = mask.loc[within_length_limit]

    if verbose:
        print('Filling in remaining values with zeros...')
    resampled.fillna(0, inplace=True)

    # rename the columns in pandas for the mask so it doesn't complain
    mask.columns = [str(col) + '_mask' for col in mask.columns]

    # merge the mask with the features
    final = pd.concat([resampled, mask], axis=1)
    final.reset_index(level=1, inplace=True)
    final = final.loc[final.time > 0]

    if verbose:
        print('Saving progress...')
    # save to csv
    if test is False:
        final.to_csv(eICU_path + 'preprocessed_timeseries.csv', mode='a', header=header)
    return

def gen_patient_chunk(patients, size=1000):
    it = iter(patients)
    chunk = list(islice(it, size))
    while chunk:
        yield chunk
        chunk = list(islice(it, size))

def gen_timeseries_file(eICU_path, test=False):

    print('Loading data from timeseries files...')
    if not test:
        # print('not conditional switch testing')
        timeseries_lab = pd.read_csv(eICU_path + 'timeserieslab.csv')
        timeseries_resp = pd.read_csv(eICU_path + 'timeseriesresp.csv')
        timeseries_nurse = pd.read_csv(eICU_path + 'timeseriesnurse.csv')
        timeseries_periodic = pd.read_csv(eICU_path + 'timeseriesperiodic.csv')
        timeseries_aperiodic = pd.read_csv(eICU_path + 'timeseriesaperiodic.csv')
    else:
        timeseries_lab = pd.read_csv(eICU_path + 'timeserieslab.csv', nrows=500000)
        timeseries_resp = pd.read_csv(eICU_path + 'timeseriesresp.csv', nrows=500000)
        timeseries_nurse = pd.read_csv(eICU_path + 'timeseriesnurse.csv', nrows=500000)
        timeseries_periodic = pd.read_csv(eICU_path + 'timeseriesperiodic.csv', nrows=500000)
        timeseries_aperiodic = pd.read_csv(eICU_path + 'timeseriesaperiodic.csv', nrows=500000)


    print('Reconfiguring lab timeseries...')
    timeseries_lab = reconfigure_timeseries(timeseries_lab, offset_column='labresultoffset',feature_column='labname',test=test)
    timeseries_lab.columns = timeseries_lab.columns.droplevel()

    print('Reconfiguring respiratory timeseries...')
    timeseries_resp = timeseries_resp.replace('%', '', regex=True)
    timeseries_resp['respchartvalue'] = pd.to_numeric(timeseries_resp['respchartvalue'], errors='coerce')
    timeseries_resp = timeseries_resp.loc[timeseries_resp['respchartvalue'].notnull()]
    timeseries_resp = reconfigure_timeseries(timeseries_resp, offset_column='respchartoffset', feature_column='respchartvaluelabel', test=test)
    timeseries_resp.columns = timeseries_resp.columns.droplevel()

    print('Reconfiguring nurse timeseries...')
    # remove non numeric data
    timeseries_nurse['nursingchartvalue'] = pd.to_numeric(timeseries_nurse['nursingchartvalue'], errors='coerce')
    timeseries_nurse = timeseries_nurse.loc[timeseries_nurse['nursingchartvalue'].notnull()]
    timeseries_nurse = reconfigure_timeseries(timeseries_nurse, offset_column='nursingchartoffset', feature_column='nursingchartcelltypevallabel', test=test)
    timeseries_nurse.columns = timeseries_nurse.columns.droplevel()

    print('Reconfiguring aperiodic timeseries...')
    timeseries_aperiodic = reconfigure_timeseries(timeseries_aperiodic, offset_column='observationoffset', test=test)

    print('Reconfiguring periodic timeseries...')
    timeseries_periodic = reconfigure_timeseries(timeseries_periodic, offset_column='observationoffset',test=test)

    patients = timeseries_periodic.index.unique(level=0)
    size = 4000
    i = 4000
    gen_chunks = gen_patient_chunk(patients, size=4000)
    header = True  # for the first chunk include the header in the csv file

    print('Starting main processing loop...')

    for patient_chunk in gen_chunks:

        res = timeseries_lab.loc[patient_chunk].append(timeseries_resp.loc[patient_chunk], sort=False)
        res = res.append(timeseries_nurse.loc[patient_chunk], sort=False)
        res = res.append(timeseries_periodic.loc[patient_chunk], sort=False)
        res = res.append(timeseries_aperiodic.loc[patient_chunk], sort=True)

        if i == size:
            # all if not all are not normally distributed
            quantiles = res.quantile([0.05, 0.95])
            maxs = quantiles.loc[0.95]
            mins = quantiles.loc[0.05]

        res = 2 * (res - mins) / (maxs - mins) - 1

        # we then need to make sure that ridiculous outliers are clipped to something sensible
        res.clip(lower=-4, upper=4, inplace=True)  # room for +- 3 on each side, as variables are scaled roughly between 0 and 1
        d_rate = 4/3
        #print(d_rate)
        resample_and_mask(res, eICU_path, header, mask_decay=True, decay_rate=d_rate, test=test, verbose=False)
        header = False
        print('Processed ' + str(i) + ' patients...')
        i += size

    return

def add_time_of_day(processed_timeseries, flat_features):

    print('Adding time of day features...')
    processed_timeseries = processed_timeseries.join(flat_features[['hour']], how='inner', on='patient')
    temp = processed_timeseries['time'] + processed_timeseries['hour']
    # print(temp)
    processed_timeseries['hour'] = temp
    scaler = np.linspace(0, 1, 24)  # make sure it's still scaled well
    # print(scaler)
    processed_timeseries['hour'] = processed_timeseries['hour'].apply(lambda x: scaler[x%24 - 24])
    return processed_timeseries

def further_processing(eICU_path, test=False):

    time_str = 'preprocessed_timeseries.csv'
    if test:
        processed_timeseries = pd.read_csv(eICU_path + time_str, nrows=999999)
    else:
        processed_timeseries = pd.read_csv(eICU_path + time_str)
    processed_timeseries.rename(columns={'Unnamed: 1': 'time'}, inplace=True)
    processed_timeseries.set_index('patient', inplace=True)
    flat_features = pd.read_csv(eICU_path + 'flat_features.csv')
    flat_features.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    processed_timeseries.sort_values(['patient', 'time'], inplace=True)
    flat_features.set_index('patient', inplace=True)

    processed_timeseries = add_time_of_day(processed_timeseries, flat_features)

    if test is False:
        print('Saving finalised preprocessed timeseries...')
        # this will replace old one that was updated earlier in the script
        processed_timeseries.to_csv(eICU_path + time_str)

    return

def timeseries_main(eICU_path, test=False):
    # make sure the preprocessed_timeseries.csv file is not there because the first section of this script appends to it
    if test is False:
        print('Removing the preprocessed_timeseries.csv file if it exists...')
        try:
            os.remove(eICU_path + 'preprocessed_timeseries.csv')
        except FileNotFoundError:
            pass
    gen_timeseries_file(eICU_path, test)
    further_processing(eICU_path, test)
    return

if __name__=='__main__':
    with open('paths.json', 'r') as f:
        eICU_path = json.load(f)["eICU_path"]
    test = False
    timeseries_main(eICU_path, test)