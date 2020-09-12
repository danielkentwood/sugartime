import csv
import datetime
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import sugarTS.constants as constants


def get_datetime(df, var_name="EventDateTime", targ_name="dt"):
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
    df[targ_name] = df.apply(
        lambda row: datetime.datetime.strptime(
            row[var_name].replace("T", " "), "%Y-%m-%d %H:%M:%S"
        ),
        axis=1,
    )

    # df[targ_name] = df.apply(
    #     lambda row: datetime.datetime.strptime(
    #         row[var_name],
    #         "%Y-%m-%dT%H:%M:%S"
    #     ),
    #     axis=1,
    # )

    df = df.drop([var_name], axis=1)
    return df


def load_tandem(tandem_filename):
    """
    Open the raw tandem file exported from the website.

    Inputs:
    * tandem_filename: a file path (str) to the tandem file.
    Outputs:
    * tandem_dict: a dict containing 5 different measurements from the
    tandem pump device.
    """

    # grab the data from the raw tandem file
    tandem_dict = defaultdict(list)
    with open(tandem_filename) as csv_file:
        tandem_wb = csv.reader(csv_file, delimiter=",")
        for row in tandem_wb:
            if row:
                if "t:slim X2 Insulin Pump OUS" in row[0] and "EGV" in row[2]:
                    tandem_dict["estimated_glu"].append(row[:6])
                if "t:slim X2 Insulin Pump OUS" in row[0] and "BG" in row[2]:
                    tandem_dict["blood_glucose"].append(row)
                if "IOB" in row[0]:
                    tandem_dict["insulin_on_board"].append(row[:4])
                if "Basal" in row[0]:
                    tandem_dict["basal_rate"].append(row[:3])
                if "Bolus" in row[0]:
                    tandem_dict["bolus"].append(row[:41])
    return tandem_dict


def shape_tandem(tandem_dict):
    """
    Put the tandem data into dataframes.
    Drop unnecessary columns.

    Inputs:
    * tandem_dict: a dict containing the tandem data.
    Outputs:
    * estimated_glu: glucose values from continuous glucose monitor
    * insulin_on_board: amount of insulin-on-board
    * basal_rate: basal rate of background insulin administration
    * bolus: size of insulin bolus
    """

    # put the data into dataframes
    estimated_glu = pd.DataFrame(
        tandem_dict["estimated_glu"],
        columns=constants.tandem_estimated_glu_columns,
    )
    insulin_on_board = pd.DataFrame(
        tandem_dict["insulin_on_board"], columns=constants.tandem_iob_columns
    )
    basal_rate = pd.DataFrame(
        tandem_dict["basal_rate"], columns=constants.tandem_basal_rate_columns
    )
    bolus = pd.DataFrame(
        tandem_dict["bolus"],
        columns=constants.tandem_bolus_columns,
    )

    # shaping the continuous glucose monitor data
    estimated_glu = estimated_glu.drop(
        ["DeviceType", "SerialNumber", "Description", "VOID"], axis=1
    )
    estimated_glu = get_datetime(estimated_glu)
    estimated_glu.rename(
        columns={"bg": "estimated_glu"}, inplace=True)

    # shaping the basal rate data
    basal_rate = basal_rate.drop(["Type"], axis=1)
    basal_rate = get_datetime(basal_rate)
    basal_rate = basal_rate.replace("", np.nan)
    basal_rate.drop_duplicates(subset=None, keep="first", inplace=True)
    basal_rate = basal_rate.dropna()
    basal_rate.reset_index(drop=True, inplace=True)
    basal_rate.rename(
        columns={"br": "basal_rate"}, inplace=True)

    # shaping the insulin-on-board data
    insulin_on_board = insulin_on_board.drop(["Type", "EventID"], axis=1)
    insulin_on_board = get_datetime(insulin_on_board)
    insulin_on_board = insulin_on_board.replace("", np.nan)
    insulin_on_board.rename(
        columns={"iob": "insulin_on_board"}, inplace=True)

    # shaping the bolus data
    bolus = get_datetime(bolus)
    bolus = bolus[["InsulinDelivered", "dt"]]
    bolus.rename(
        columns={"InsulinDelivered": "bolus"}, inplace=True)

    return estimated_glu, basal_rate, insulin_on_board, bolus


def load_clarity(clarity_filename):
    """
    Open the raw clarity file exported from the website.

    Inputs:
    * clarity_filename: a file path (str) to the clarity file.

    Outputs:
    * clarity: dataframe
    """

    # read and shape clarity data
    return pd.read_csv(clarity_filename)


def shape_clarity(clarity):
    """
    Put the clarity data into dataframes.
    Drop unnecessary measurements.

    Inputs:
    * clarity: a dataframe containing raw clarity data.

    Outputs:
    * clarity: a clean dataframe with carb value measurements
    """

    # shape clarity data
    clarity = clarity.drop(["Index"], axis=1)
    clarity = clarity[10:].reset_index(drop=True)
    clarity = clarity[["Carb Value (grams)",
                       "Timestamp (YYYY-MM-DDThh:mm:ss)"]]
    clarity = get_datetime(
        clarity, var_name="Timestamp (YYYY-MM-DDThh:mm:ss)", targ_name="dt"
    )
    clarity.rename(columns={"Carb Value (grams)": "carb_grams"}, inplace=True)

    return clarity


def combine_tandem_clarity(
    clarity, estimated_glu,
    basal_rate, insulin_on_board, bolus
):
    """
    Combine the tandem and clarity data into a single pandas dataframe.
    Deal with empty values.

    Inputs:
    * clarity: dataframe with carb data from clarity device
    * estimated_glu: dataframe with blood glucose data from tandem device
    * basal_rate: dataframe with basal rate data from tandem device
    * insulin_on_board: dataframe with insulin on board data from tandem device
    * bolus: dataframe with bolus size data from tandem device

    Outputs:
    * data: combined dataframe
    """

    # find the timestamp to stop at
    timestamp_stop = np.min([clarity.dt.max(), estimated_glu.dt.max()])

    # concatenate the interesting columns:
    # estimated_glu, basal rate, insulin-on-board,
    # bolus amount, and carb amount.
    d = pd.concat(
        [
            estimated_glu,
            basal_rate,
            insulin_on_board,
            bolus,
            clarity[["dt", "carb_grams"]],
        ]
    ).sort_values("dt")

    # reset index
    d.reset_index(drop=True, inplace=True)
    d = d.replace("", np.nan)
    d.set_index("dt", inplace=True)
    d = d.loc[:timestamp_stop]

    # convert numeric values to float and resample
    data = (
        d[["estimated_glu",
           "basal_rate",
           "insulin_on_board"]].astype(float).resample("5T").mean()
    )
    data["carb_grams"] = pd.DataFrame(
        d["carb_grams"].astype(float).resample("5T").sum()
    )
    data["bolus"] = pd.DataFrame(
        d["bolus"].astype(float).resample("5T").sum()
    )
    # fill blood glucose and insulin-on-board with linearly interpolated values
    data[["estimated_glu", "insulin_on_board"]] = data[
        ["estimated_glu", "insulin_on_board"]].interpolate(
        method="linear", limit_direction="forward", axis=0
    )

    # fill basal rate NaNs with forward fill (use the most recent valid value)
    # NOTE: This may be the wrong approach. It seems unlikely that there are
    # stretches of an hour or more where the basal rate is at 0. Look into
    # this further to see if interpolation of some sort is more appropriate.
    data["basal_rate"] = data["basal_rate"].fillna(method="ffill")
    data["carb_grams"] = data["carb_grams"].replace(np.nan, 0)
    data = data.drop(data.index[data.isnull().any(1)])

    return data


def load_and_clean_example_data():
    """
    Load the raw example data from Clarity and Tandem.
    Then clean it.

    Outputs:
    * data: a dataframe containing:
        - estimated_glu: blood glucose
        - insulin_on_board: insulin on board
        - basal_rate: basal rate of insulin
        - carb_grams: number of carbs in grams
        - bolus: amount of insulin in bolus
    """

    # define paths
    raw_dir = Path("./examples/raw/")
    clarity_filename = raw_dir/"CLARITY.csv"
    tandem_filename = raw_dir/"TANDEM.csv"

    # load and clean example data
    data = load_and_clean_data(clarity_filename, tandem_filename)

    return data


def load_and_clean_data(clarity_filename, tandem_filename):
    """
    Load the raw data from Clarity and Tandem.
    Then clean it.

    Inputs:
    * clarity_filename: full path for raw Clarity data
    * tandem_filename: full path for raw Tandem data

    Outputs:
    * data: a dataframe containing:
        - estimated_glu: blood glucose
        - insulin_on_board: insulin on board
        - basal_rate: basal rate of insulin
        - carb_grams: number of carbs in grams
        - bolus: amount of insulin in bolus
    """

    # load and shape tandem and clarity data
    estimated_glu, basal_rate, insulin_on_board, bolus = shape_tandem(
        load_tandem(tandem_filename)
    )
    clarity = shape_clarity(load_clarity(clarity_filename))

    # combine the two datasets
    data = combine_tandem_clarity(
        clarity, estimated_glu, basal_rate, insulin_on_board, bolus
    )
    return data


def split_data(data):
    """
    Split the data into training and testing sets.

    Input:
    * data: (dataframe) full dataset

    Outputs:
    * Xtrain: (dataframe)
    * ytrain: (series)
    * Xtest: (dataframe)
    * ytest: (series)
    """

    # Do a 75/25 split for train and test sets.
    # No random sampling because it is a time series.
    split = int(len(data) * 0.75)
    ytrain = data["estimated_glu"].iloc[:split]
    Xtrain = data.loc[:, ["carb_grams", "bolus"]].iloc[:split, :]
    ytest = data["estimated_glu"].iloc[split:]
    Xtest = data.loc[:, ["carb_grams", "bolus"]].iloc[split:, :]

    return Xtrain, ytrain, Xtest, ytest
