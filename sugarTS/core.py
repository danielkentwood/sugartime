import csv
import datetime

import pandas as pd
import numpy as np


def get_datetime(df, var_name='EventDateTime', targ_name='dt'):
    """
    Parse the datetime strings coming from the clarity and tandem exports.
    Replace the column of strings with a column of datetime objects

    Inputs:
    * df: dataframe
    * var_name: the name of the variable containing the datetime string
    * targ_name: the new name for the column of transformed datetime objects
    Outputs:
    * df: dataframe with replaced datetime column
    """
    df[var_name] = df[var_name].apply(lambda x: x[:19])
    df[targ_name] = df.apply(lambda row: datetime.datetime.strptime
                             (row[var_name].replace
                              ('T', ' '), '%Y-%m-%d %H:%M:%S'), axis=1)
    df = df.drop([var_name], axis=1)
    return df


def load_tandem(tandem_fn):
    '''
    Open the raw tandem file exported from the website.

    Inputs:
    * tandem_fn: a file path (str) to the tandem file.
    Outputs:
    * tandem_dict: a dict containing 5 different measurements from the
    tandem pump device.
    '''
    # grab the data from the raw tandem file
    tandem_dict = {1: [], 2: [], 3: [], 4: [], 5: []}
    with open(tandem_fn) as csv_file:
        tandem_wb = csv.reader(csv_file, delimiter=',')
        for row in tandem_wb:
            if not row:
                continue
            else:
                if 't:slim X2 Insulin Pump OUS' in row[0] and 'EGV' in row[2]:
                    tandem_dict[1].append(row[:6])
                if 't:slim X2 Insulin Pump OUS' in row[0] and 'BG' in row[2]:
                    tandem_dict[2].append(row)
                if 'IOB' in row[0]:
                    tandem_dict[3].append(row[:4])
                if 'Basal' in row[0]:
                    tandem_dict[4].append(row[:3])
                if 'Bolus' in row[0]:
                    tandem_dict[5].append(row[:41])
    return tandem_dict


def shape_tandem(tandem_dict):
    '''
    Put the data into dataframes.
    Drop unnecessary measurements.

    Inputs:
    * tandem_dict: a dict containing the tandem data.
    Outputs:
    * egv: glucose values
    * iob: inuslin-on-board
    * br: basal rate of insulin
    * bolus: size of insulin bolus
    '''
    # put the data into dataframes
    egv = pd.DataFrame(tandem_dict[1], columns=['DeviceType', 'SerialNumber',
                                                'Description', 'EventDateTime',
                                                'bg', 'VOID'])
    iob = pd.DataFrame(tandem_dict[3], columns=['Type', 'EventID',
                                                'EventDateTime', 'iob'])
    br = pd.DataFrame(tandem_dict[4], columns=['Type', 'EventDateTime', 'br'])
    bolus = pd.DataFrame(tandem_dict[5],
                         columns=['Type', 'BolusDescription',
                                  'bg_bolus', 'iob_bolus',
                                  'BolusRequestID',
                                  'BolusCompletionID',
                                  'CompletionDateTime',
                                  'InsulinDelivered',
                                  'FoodDelivered',
                                  'CorrectionDelivered',
                                  'CompletionStatusID',
                                  'CompletionStatusDesc',
                                  'BolusIsComplete',
                                  'BolexCompletionID',
                                  'BolexSize',
                                  'BolexStartDateTime',
                                  'BolexCompletionDateTime',
                                  'BolexInsulinDelivered',
                                  'BolexIOB',
                                  'BolexCompletionStatusID',
                                  'BolexCompletionStatusDesc',
                                  'ExtendedBolusIsComplete',
                                  'EventDateTime',
                                  'RequestDateTime',
                                  'BolusType',
                                  'BolusRequestOptions',
                                  'StandardPercent',
                                  'Duration', 'CarbSize',
                                  'UserOverride', 'TargetBG',
                                  'CorrectionFactor',
                                  'FoodBolusSize',
                                  'CorrectionBolusSize',
                                  'ActualTotalBolusRequested',
                                  'IsQuickBolus',
                                  'EventHistoryReportEventDesc',
                                  'EventHistoryReportDetails',
                                  'NoteID',
                                  'IndexID', 'Note'])
    # shaping the continuous glucose monitor data
    egv = egv.drop(['DeviceType', 'SerialNumber', 'Description', 'VOID'],
                   axis=1)
    egv = get_datetime(egv)
    # shaping the basal rate data
    br = br.drop(['Type'], axis=1)
    br = get_datetime(br)
    br = br.replace('', np.nan)
    br.drop_duplicates(subset=None, keep='first', inplace=True)
    br = br.dropna()
    br.reset_index(drop=True, inplace=True)
    # shaping the insulin-on-board data
    iob = iob.drop(['Type', 'EventID'], axis=1)
    iob = get_datetime(iob)
    iob = iob.replace('', np.nan)
    # shaping the bolus data
    bolus = get_datetime(bolus)
    bolus = bolus[['InsulinDelivered', 'dt']]
    bolus.rename(columns={'InsulinDelivered': 'bolus_units'}, inplace=True)

    return egv, br, iob, bolus


def load_clarity(clarity_fn):
    '''
    Open the raw clarity file exported from the website.

    Inputs:
    * clarity_fn: a file path (str) to the clarity file.
    Outputs:
    * clarity: dataframe
    '''
    # read and shape clarity data
    clarity = pd.read_csv(clarity_fn)
    return clarity


def shape_clarity(clarity):
    '''
    Put the clarity data into dataframes.
    Drop unnecessary measurements.

    Inputs:
    * clarity: a dataframe containing raw clarity data.
    Outputs:
    * clarity: a clean dataframe with carb value measurements
    '''
    # shape clarity data
    clarity = clarity.drop(['Index'], axis=1)
    clarity = clarity[10:].reset_index(drop=True)
    clarity = clarity[['Carb Value (grams)',
                       'Timestamp (YYYY-MM-DDThh:mm:ss)']]
    clarity = get_datetime(clarity, var_name='Timestamp (YYYY-MM-DDThh:mm:ss)',
                           targ_name='dt')
    clarity.rename(columns={'Carb Value (grams)': 'carb_grams'}, inplace=True)
    return clarity


def combine_tandem_clarity(clarity, egv, br, iob, bolus):
    '''
    Combine the tandem and clarity data into a single pandas dataframe.
    Deal with empty values.

    Inputs:
    * clarity: dataframe with carb data from clarity device
    * egv: dataframe with blood glucose data from tandem device
    * br: dataframe with basal rate data from tandem device
    * iob: dataframe with insulin on board data from tandem device
    * bolus: dataframe with bolus size data from tandem device
    Outputs:
    * data: combined dataframe
    '''
    # find the timestamp to stop at
    ts_lim = np.min([clarity.dt.max(), egv.dt.max()])
    # concatenate the interesting columns: blood glucose (egv), basal rate,
    # insulin-on-board, bolus amount, and carb amount.
    d = pd.concat([egv, br, iob, bolus,
                   clarity[['dt', 'carb_grams']]]).sort_values('dt')
    # reset index
    d.reset_index(drop=True, inplace=True)
    # replace empty values with NaN
    d = d.replace('', np.nan)
    # put the datetime column into the index
    d.set_index('dt', inplace=True)
    d = d.loc[:ts_lim]
    # convert numeric values to float and resample
    data = d[['bg', 'br', 'iob']].astype(float).resample('5T').mean()
    data['carb_grams'] = pd.DataFrame(d['carb_grams'].astype(float).
                                      resample('5T').sum())
    data['bolus_units'] = pd.DataFrame(d['bolus_units'].astype(float).
                                       resample('5T').sum())
    # fill blood glucose and insulin-on-board with linearly interpolated values
    data[['bg', 'iob']] = data[['bg',
                                'iob']].interpolate(method='linear',
                                                    limit_direction='forward',
                                                    axis=0)
    # fill basal rate NaNs with forward fill (use the most recent valid value)
    # NOTE: This may be the wrong approach. It seems unlikely that there are
    # stretches of an hour or more where the basal rate is at 0. Look into
    # this further to see if interpolation of some sort is more appropriate.
    data['br'] = data['br'].fillna(method='ffill')
    # fill the NaNs in carb_grams with zeros
    data['carb_grams'] = data['carb_grams'].replace(np.nan, 0)
    # remove rows with leftover NaNs
    data = data.drop(data.index[data.isnull().any(1)])
    return data


def load_and_clean_example_data():
    '''
    Load the raw example data from Clarity and Tandem.
    Then clean it.
    Outputs:
    * data: a dataframe containing:
        - bg: blood glucose
        - iob: insulin on board
        - br: basal rate of insulin
        - carb_grams: number of carbs in grams
        - bolus_units: amount of insulin in bolus
    '''
    # define paths
    raw_dir = './examples/raw/'
    clarity_fn = raw_dir+'CLARITY.csv'
    tandem_fn = raw_dir+'TANDEM.csv'
    # load and shape tandem
    td_dict = load_tandem(tandem_fn)
    egv, br, iob, bolus = shape_tandem(td_dict)
    # load and shape clarity
    clarity = load_clarity(clarity_fn)
    clarity = shape_clarity(clarity)
    # combine the two datasets
    data = combine_tandem_clarity(clarity, egv, br, iob, bolus)
    return data


def load_and_clean_data(clarity_fn, tandem_fn):
    '''
    Load the raw data from Clarity and Tandem.
    Then clean it.
    Inputs:
    * clarity_fn: full path for raw Clarity data
    * tandem_fn: full path for raw Tandem data
    Outputs:
    * data: a dataframe containing:
        - bg: blood glucose
        - iob: insulin on board
        - br: basal rate of insulin
        - carb_grams: number of carbs in grams
        - bolus_units: amount of insulin in bolus
    '''
    # load and shape tandem
    td_dict = load_tandem(tandem_fn)
    egv, br, iob, bolus = shape_tandem(td_dict)
    # load and shape clarity
    clarity = load_clarity(clarity_fn)
    clarity = shape_clarity(clarity)
    # combine the two datasets
    data = combine_tandem_clarity(clarity, egv, br, iob, bolus)
    return data


def split_data(data):
    '''
    Split the data into training and testing sets.
    Input:
    * data: (dataframe) full dataset
    Outputs:
    * Xtrain: (dataframe)
    * ytrain: (series)
    * Xtest: (dataframe)
    * ytest: (series)
    '''
    split = int(len(data)*.75)
    ytrain = data['bg'].iloc[:split]
    Xtrain = data.loc[:, ['carb_grams', 'bolus_units']].iloc[:split, :]
    ytest = data['bg'].iloc[split:]
    Xtest = data.loc[:, ['carb_grams', 'bolus_units']].iloc[split:, :]
    return Xtrain, ytrain, Xtest, ytest
