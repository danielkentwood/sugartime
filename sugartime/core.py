import csv
import datetime
from collections import defaultdict
from pathlib import Path
import os

import pandas as pd
import numpy as np
import sugartime.constants as constants
from fireTS import utils
import plotly.graph_objects as go


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
            row[var_name],
            "%Y-%m-%dT%H:%M:%S"),
        axis=1,
    )
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
    estimated_glu.rename(columns={"bg": "estimated_glu"}, inplace=True)

    # shaping the basal rate data
    basal_rate = basal_rate.drop(["Type"], axis=1)
    basal_rate = get_datetime(basal_rate)
    basal_rate = basal_rate.replace("", np.nan)
    basal_rate.drop_duplicates(subset=None, keep="first", inplace=True)
    basal_rate = basal_rate.dropna()
    basal_rate.reset_index(drop=True, inplace=True)
    basal_rate.rename(columns={"br": "basal_rate"}, inplace=True)

    # shaping the insulin-on-board data
    insulin_on_board = insulin_on_board.drop(["Type", "EventID"], axis=1)
    insulin_on_board = get_datetime(insulin_on_board)
    insulin_on_board = insulin_on_board.replace("", np.nan)
    insulin_on_board.rename(columns={"iob": "insulin_on_board"}, inplace=True)

    # shaping the bolus data
    bolus = get_datetime(bolus)
    bolus = bolus[["InsulinDelivered", "dt"]]
    bolus.rename(columns={"InsulinDelivered": "bolus"}, inplace=True)

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
        clarity,
        estimated_glu,
        basal_rate,
        insulin_on_board,
        bolus):
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
        d[["estimated_glu", "basal_rate", "insulin_on_board"]]
        .astype(float)
        .resample("5T")
        .mean()
    )
    data["carb_grams"] = pd.DataFrame(
        d["carb_grams"].astype(float).resample("5T").sum()
    )
    data["bolus"] = pd.DataFrame(d["bolus"].astype(float).resample("5T").sum())
    # fill blood glucose and insulin-on-board with linearly interpolated values
    data[["estimated_glu", "insulin_on_board"]] = data[
        ["estimated_glu", "insulin_on_board"]
    ].interpolate(method="linear", limit_direction="forward", axis=0)

    # fill basal rate NaNs with forward fill (use the most recent valid value)
    # NOTE: This may be the wrong approach. It seems unlikely that there are
    # stretches of an hour or more where the basal rate is at 0. Look into
    # this further to see if interpolation of some sort is more appropriate.
    data["basal_rate"] = data["basal_rate"].fillna(method="ffill")
    data["carb_grams"] = data["carb_grams"].replace(np.nan, 0)
    data = data.drop(data.index[data.isnull().any(1)])

    return data


def feature_engineering(data):
    data["all_insulin"] = data.bolus + data.basal_rate
    data.drop(
        ["basal_rate", "insulin_on_board", "bolus"],
        axis=1,
        inplace=True)
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
    raw_dir = Path("./data/raw/")
    clarity_filename = raw_dir / "CLARITY.csv"
    tandem_filename = raw_dir / "TANDEM.csv"

    # load and clean example data
    data = load_and_clean_data(clarity_filename, tandem_filename)

    return data


def load_and_clean_synthetic_data():
    """
    Load the synthetic data.
    Then clean it.

    Outputs:
    * data: a dataframe containing:
        - estimated_glu: blood glucose
        - carb_grams: number of carbs in grams
        - all_insulin: amount of insulin in bolus and basal rate
    """

    data = pd.read_csv(os.path.join(".",
                                    "data",
                                    "example",
                                    "adolescent#001.csv"))

    data.rename(
        columns={
            "CGM": "estimated_glu",
            "CHO": "carb_grams",
            "insulin": "all_insulin",
            "Time": "dt",
        },
        inplace=True,
    )
    data = data.set_index("dt")
    data.index = pd.to_datetime(data.index)
    data = data.apply(lambda x: x.fillna(x.median()), axis=0)

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


def split_data(data, target_name, feature_names, split=[0.75]):
    """
    Split the data into training and testing (and maybe validation) sets.

    Input:
    * data: (dataframe) full dataset
    * target_name (str) name of target column
    * feature_names (list) names of feature columns
    """

    num_rows = len(data)
    # No random sampling because it is a time series.
    ytrain = data[target_name].iloc[: int(num_rows * split[0])]
    Xtrain = data[feature_names].iloc[: int(num_rows * split[0]), :]
    ytest = data[target_name].iloc[int(num_rows * split[-1]):]
    Xtest = data[feature_names].iloc[int(num_rows * split[-1]):, :]

    if len(split) == 2:
        yval = data[target_name].iloc[
            int(num_rows * split[0]): int(num_rows * split[1])
        ]
        Xval = data[feature_names].iloc[
            int(num_rows * split[0]): int(num_rows * split[1]), :
        ]
        return Xtrain, ytrain, Xval, yval, Xtest, ytest
    else:
        return Xtrain, ytrain, Xtest, ytest


def plot_forecast(obs, pred, return_flag=False):
    """
    Plots a forecast along with the observed data.
    Takes the following:
    * obs (dataframe): the target variable
    * pred (dataframe): model output
    * return_flag (logical): allows the output of the model to be either the
    figure object or a plot of the figure.
    """
    obs = obs[-50:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=obs.index, y=obs, mode="lines", name="y"))
    fig.add_trace(
        go.Scatter(x=pred.index, y=pred.ypred, mode="lines", name="y_forecast")
    )
    fig.update_xaxes(
        title_text="Time",
        title_font=dict(size=18),
        showline=True,
        linewidth=2,
        linecolor="black",
    )
    fig.update_yaxes(
        title_text="Blood Glucose",
        title_font=dict(size=18),
        showline=True,
        linewidth=2,
        linecolor="black",
    )
    if return_flag:
        return fig
    else:
        fig.show()


def plot_optimal_boundaries(patient, fig):
    """
    Plots the optimal blood glucose boundaries (default at 80 and 140).
    Takes the following:
    * patient (obj): the patient's data
    * fig (obj): the figure that you want to add the optimal boundaries to
    """
    minx = []
    maxx = []
    for i in fig.select_traces():
        cminx = min(i["x"])
        cmaxx = max(i["x"])
        if not minx:
            minx = cminx
        if not maxx:
            maxx = cmaxx
        if cminx < minx:
            minx = cminx
        if cmaxx > maxx:
            maxx = cmaxx
    y_ind = pd.date_range(start=minx, end=maxx, freq="5T")
    fig.add_trace(
        go.Scatter(
            x=y_ind,
            y=np.ones(len(y_ind)) * patient.target_range[0],
            mode="lines",
            name="lower bound",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_ind,
            y=np.ones(len(y_ind)) * patient.target_range[1],
            mode="lines",
            name="upper bound",
        )
    )
    return fig


def horizon_transform(X, y, horizon=12):
    """
    For each step in the horizon, adds a new column and shifts the
    target variable one step. This allows the model to be trained
    on arbitrary intervals into the future.
    Takes the following:
    * X (array): design matrix (already lag transformed)
    * y (array): target variable
    Returns:
    * X (array): design matrix transformed based on number of shifts
    * y (array): target variable transformed based on number of shifts
    """
    y = y.reshape(-1, 1)
    y_horizon = np.empty((len(y), horizon))
    y_horizon.fill(np.nan)
    for i in range(horizon):
        y_horizon[: -(i + 1), i] = y[(i + 1):].T
    return X[:-horizon], y_horizon[:-horizon, :]


def add_lags(X, y, auto_order, exog_order, exog_delay):
    """
    Adds lags based the orders and delays of the endogenous and exogenous
    variables.
    Takes the following:
    * X (dataframe): design matrix
    * y (dataframe): target variable
    * auto_order (int): the autoregressive order of the model
    * exog_order (list): the order of the exogenous variables
    * exog_delay (list): delay of the exogenous variables
    Returns:
    * features (array): transformed design matrix
    * target (array): transformed target variable
    """
    m = utils.MetaLagFeatureProcessor(
        X.values, y.values, auto_order, exog_order, exog_delay
    )
    features = m.generate_lag_features()
    target = utils.shift(y, -1)
    all_data = np.concatenate([target.reshape(-1, 1), features], axis=1)
    mask = np.isnan(all_data).any(axis=1)
    features, target = features[~mask], target[~mask]

    return features, target
