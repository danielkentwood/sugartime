import datetime as dtm
import itertools

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.base import clone

import sugartime.core as core


class Patient:
    """
    Object containing data for an
    individual patient.
    """

    def __init__(self):
        self.carbs_per_insulin = 8 # this modulates the grid search in self.find_optimal_bolus
        self.target_range = (80, 140)

    def load_example_data(self):
        """
        Loads the raw example patient dataset.
        """
        data = core.load_and_clean_example_data()
        self.data = core.feature_engineering(data)

    def load_synthetic_data(self):
        """
        Loads a synthetic dataset created with the simglucose package
        (https://github.com/jxx123/simglucose).
        """
        self.data = core.load_and_clean_synthetic_data()

    def load_device_data(self, clarity_filename, tandem_filename):
        """
        Loads and cleans a novel data set from a clarity continuous
        glucose monitor and a tandem glucose pump.
        """
        self.data = core.load_and_clean_data(
            clarity_filename,
            tandem_filename,
        )

    def load_data(self, X, y):
        """
        Loads a novel data set.
        Should be an N x 3 numpy array, where N is the # of
        observations.

        Output:
        Pandas dataframe with a datetime index and the following columns:
        * estimated_glu
        * carb_grams
        * all_insulin
        """
        # create datetime index for dataframe
        t = dtm.datetime(2020, 11, 3, 12, 5)
        ind = pd.date_range(
            start=t,
            end=t + dtm.timedelta(minutes=5 * (len(y)-1)),
            freq='5T')

        # make dataframe
        df = pd.DataFrame(
            np.concatenate((y, X), axis=1),
            columns=['estimated_glu','carb_grams','all_insulin'],
            index=ind
        )
        self.data = df

    def split_data(self, target_name, feature_names, split=[0.75]):
        """
        Split the data into training and testing (and maybe validation) sets.
        Saves the splits to the patient object.
        Inputs:
        * target_name (str): name of target variable
        * feature_names (str): name of feature variables
        * split (list): one- or two-element list. First element marks the end
        of the training set. Second element marks the of the validation set.
        Testing set is whatever is left.
        """
        if len(split) > 1:
            Xtrain, ytrain, Xval, yval, Xtest, ytest = core.split_data(
                self.data,
                target_name=target_name,
                feature_names=feature_names,
                split=split,
            )
            self.Xval = Xval
            self.yval = yval
        else:
            Xtrain, ytrain, Xtest, ytest = core.split_data(
                self.data,
                target_name=target_name,
                feature_names=feature_names,
                split=split,
            )
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest


class MultiOutModel:
    """
    A multioutput time series model that consists of multiple individual
    scikit-learn regression estimators. Each estimator is trained on a
    different time shift (t+N) of the target variable. Each estimator
    can be different (e.g., random forest for t+1, support vector machine
    for t+2, etc.), and can have unique hyperparameters for both the base
    estimator and the time series model design.
    Takes the following params:
    * patient (obj): a unique patient dataset
    * horizon (int): how far into the future the model will attempt to
    forecast.
    """
    def __init__(self, patient, horizon):
        self.horizon = horizon
        self.steps = list(range(horizon))
        self.patient = patient

    def add_step(self, step, multi_out_step):
        """
        Assigns a new regression estimator to a designated time step in the
        full multioutput model.
        Takes the following:
        * step (int): the step in the multioutput model that is being defined
        * estimator (obj): a MultiOutStep object containing the estimator for
        the current output time step.
        """
        self.steps[step] = multi_out_step
    
    def fit(
        self,
        X,
        y,
        estimator,
        auto_order,
        exog_order,
        exog_delay
        ):
        """
        Fits a multioutput model.

        Takes the following:
        * X (dataframe): design matrix for training data
        * y (dataframe): target variable for training data
        * estimators (dict): dict containing all of the estimators over which
            to perform grid search. Key is the estimator name (str), while
            value is the estimator itself.
        * auto_order (int): autoregressive order of endogenous variable
        * exog_order (list): order of exogenous variables
        * exog_delay (list): delay of exogenous variables

        Stores the fitted model(s) in the MultiOutModel object.
        """
        # impose lags from design params
        horizon = self.horizon
        features, target = core.add_lags(
            X, y, auto_order, exog_order, exog_delay)
        features, target = core.horizon_transform(
            features, target, horizon=horizon)

        # loop through all the time steps
        for i in range(horizon):
            for est in estimator.keys():
                cmdl = MultiOutStep(
                    clone(estimator[est]),
                    est,
                    auto_order,
                    exog_order,
                    exog_delay,
                    horizon,
                    i)
                cmdl.estimator = cmdl.fit(features, target)
                self.add_step(i, cmdl)

    def grid_search(self, X, y, Xval, yval, estimators, design_params):
        """
        Performs a brute force grid search in order to find the optimal
        time and amount of insulin that will maintain a forecasted
        time series as close as possible to a blood glucose level of
        110 mg/dL, given information about future carbohydrate consumption.

        Takes the following:
        * X (dataframe): design matrix for training data
        * y (dataframe): target variable for training data
        * Xval (dataframe): design matrix for validation data
        * yval (dataframe): target variable for validation data
        * estimators (dict): dict containing all of the estimators over which
            to perform grid search. Keys are the estimator names (str), while
            values are the estimators themselves.
        * design_params (list): list of tuples of all of the
            permutations of desired design params 
        
        Stores the best fitting model(s) in the MultiOutModel object.
        """
        # model = MultiOutModel(horizon)
        horizon = self.horizon

        # loop through all the time steps
        for i in range(horizon):
            best_r2 = False

            # loop through all the design params
            for idp, (ao, eo, ed) in enumerate(design_params):
                # create design matrix and target matrix
                features, target = core.add_lags(X, y, ao, eo, ed)
                features, target = core.horizon_transform(
                    features, target, horizon=horizon
                )

                # loop through all the models and perform hyperparameter search
                for est in estimators.keys():
                    cmdl = MultiOutStep(
                        clone(estimators[est]), est, ao, eo, ed, horizon, i
                    )
                    cmdl.estimator = cmdl.fit(features, target)
                    r2 = cmdl.model_performance(Xval, yval)
                    cmdl.r2 = r2[0]
                    # keep the model with the highest r_squared
                    if not best_r2:
                        self.add_step(i, cmdl)
                        best_r2 = r2
                    elif r2 > best_r2:
                        self.add_step(i, cmdl)
                        best_r2 = r2
            # print out the best model for each step
            print(
                "Best model for step {} is {}({},{},{}): r2 = {}".format(
                    i,
                    self.steps[i].name,
                    self.steps[i].auto_order,
                    self.steps[i].exog_order,
                    self.steps[i].exog_delay,
                    self.steps[i].r2,
                )
            )

    def multioutput_forecast(self, X, y):
        """
        Performs a multioutput forecast.
        Assumes the forecast should start at the end of the supplied
        X and y data.
        Takes the following:
        * X (dataframe): design matrix
        * y (dataframe): target variable
        Returns:
        * numpy array with the forecasted data.
        """
        ypred = []
        for step in self.steps:
            features, target = core.add_lags(
                X, y, step.auto_order, step.exog_order, step.exog_delay
            )
            ypred.append(step.predict(features[-1, :].reshape(-1, 1).T)[0])
        return ypred

    def dynamic_forecast(self, X, y, start_time, inserts):
        """
        Performs dynamic forecasting using future
        exogenous variable information (provided by the inserts parameter).
        Uses only the t+1 model from the overall multioutput model.
        Takes the following:
        * X (dataframe): design matrix
        * y (dataframe): target variable
        * start_time (datetime): start time of the forecast
        * inserts (dict): contains dict of dicts. First level key is the
                          name of the exogenous variable, with another dict
                          as the value. This second level dict has the
                          insertion datetime as the key, and the amount
                          of the insertion (i.e., grams of carbs, or amount
                          of insulin bolus)
        Returns:
        * dataframe with the forecasted values
        """
        # set some datetime variables
        insert_times = [j for k1 in inserts.keys() for j in inserts[k1].keys()]
        max_time = max(insert_times) + dtm.timedelta(minutes=5 * self.horizon)
        cur_time = start_time - dtm.timedelta(minutes=5)
        keep_dt = pd.date_range(start=start_time, end=max_time, freq="5T")
        # add inserts
        future_x = self.add_inserts_to_x(X, start_time, inserts)
        base_y = y.loc[future_x.index[0]: cur_time]

        # step through the forecast horizon while dynamically performing
        # inference on past inferences.
        for i, dt in enumerate(keep_dt):
            t = dt - dtm.timedelta(minutes=5)
            yhat = self.multioutput_forecast(future_x.loc[:t], base_y.loc[:t])
            # here, we only append the t+1 step of the multioutput forecast
            # to the history of the target variable for the next forecast step
            base_y = pd.concat(
                [base_y, pd.Series(yhat[0], index=[dt])],
                axis=0)
        return pd.DataFrame(
            base_y.loc[keep_dt],
            columns=["ypred"],
            index=keep_dt)

    def hybrid_forecast(self, X, y, start_time, inserts):
        """
        Performs staggered multioutput forecasting using future
        exogenous variable information (provided by the inserts parameter).
        Takes the following:
        * X (dataframe): design matrix
        * y (dataframe): target variable
        * start_time (datetime): start time of the forecast
        * inserts (dict): contains dict of dicts. First level key is the
                          name of the exogenous variable, with another dict
                          as the value. This second level dict has the
                          insertion datetime as the key, and the amount
                          of the insertion (i.e., grams of carbs, or amount
                          of insulin bolus)
        Returns:
        * dataframe with the forecasted values
        """
        cur_time = start_time - dtm.timedelta(minutes=5)
        future_x = self.add_inserts_to_x(X, start_time, inserts)
        base_y = y.loc[future_x.index[0]: cur_time]

        # make a list of the times where inserted events are happening
        insert_times = [j for k1 in inserts.keys() for j in inserts[k1].keys()]
        # here, we add the current time and the final time of the inference
        # horizon -- the hybrid_forecast method uses these
        insert_times = insert_times + list(
            set([cur_time,
                 max(insert_times) + dtm.timedelta(minutes=5 * self.horizon)
                 ]
                )
        )
        insert_times.sort()

        # make sure none of the forecasts you ask for are larger
        # than the horizon of the model
        time_diffs = np.diff(insert_times) / dtm.timedelta(minutes=5)
        while np.any(time_diffs > self.horizon):
            for i in np.where(time_diffs > self.horizon)[0]:
                insert_times.append(insert_times[i] + dtm.timedelta(minutes=5))
            insert_times.sort()
            time_diffs = np.diff(insert_times) / dtm.timedelta(minutes=5)

        # perform the forecast(s)
        ypred = []
        for i in range(len(insert_times) - 1):
            ind = pd.date_range(
                start=insert_times[i] + dtm.timedelta(minutes=5),
                end=insert_times[i + 1],
                freq="5T",
            )
            keep = len(ind)
            yhat = self.multioutput_forecast(
                future_x.loc[: insert_times[i]], base_y.loc[: insert_times[i]]
            )
            ypred = ypred + yhat[:keep]
            base_y = pd.concat(
                [base_y, pd.Series(yhat[:keep], index=ind)],
                axis=0)

        # add a datetime index and return a dataframe
        ind = pd.date_range(
            start=start_time,
            end=start_time + dtm.timedelta(minutes=5 * (len(ypred) - 1)),
            freq="5T",
        )
        return pd.DataFrame(ypred, columns=["ypred"], index=ind)

    def add_inserts_to_x(self, exog_df, start_time, inserts={}):
        """
        Insert arbitrary values into the exogenous time series
        Takes the following params:
        * exog_df (dataframe): pandas dataframe containing the exogenous
                               variables, prior to adding lags.
        * start_time (datetime): specifies the start time of the exogenous
                                 data segment
        * inserts (dict): contains dict of dicts. First level key is the
                          name of the exogenous variable, with another dict
                          as the value. This second level dict has the
                          insertion datetime as the key, and the amount
                          of the insertion (i.e., grams of carbs, or amount
                          of insulin bolus)
        Returns:
        * a dataframe containing a subset of the exogenous data with
        inserted entries.
        """
        # determine how far back in time we need to look in dataset
        # (given the order and the delay parameters and their
        # respective lags)
        back_size = np.sum(
            [
                self.horizon,
                (
                    np.sum(
                        np.amax(
                            [
                                [max(step.exog_order), max(step.exog_delay)]
                                for step in self.steps
                            ],
                            axis=0,
                        )
                    )
                ),
            ]
        )
        # create a list of the times where exog variables have been inserted
        insert_times = list(
            set([j for k1 in inserts.keys() for j in inserts[k1].keys()])
        )
        # datetime stuff
        cur_time = start_time - dtm.timedelta(minutes=5)
        ind = pd.date_range(
            start=start_time,
            end=max(insert_times) + dtm.timedelta(minutes=5 * self.horizon),
            freq="5T",
        )

        # insert any desired future actions
        base = np.empty((len(ind), exog_df.shape[1]))
        base[:] = np.nan
        if inserts:
            for col in inserts.keys():
                i = exog_df.columns.to_list().index(col)
                for date in inserts[col].keys():
                    ii = ind.to_list().index(date)
                    base[ii, i] = inserts[col][date]
        # append the future data to the relevant past data
        future_df = pd.DataFrame(base, index=ind, columns=exog_df.columns)
        df = pd.concat([exog_df.loc[:cur_time], future_df])
        # assume that non-inserted values are the median value of the feature
        df = df.apply(lambda x: x.fillna(x.median()), axis=0)
        # keep the relevant data
        df = df.loc[(cur_time - dtm.timedelta(minutes=5 * (int(back_size)))):]
        return df

    def find_optimal_bolus(self, carbs, carb_t):
        """
        Use a brute force grid search to find the optimal
        timing and amount of insulin, given an amount and
        timing of carbs for a meal.

        Inputs:
        * carbs: amount of carbs in upcoming meal
        * carb_t: datetime object describing the timing of
                  the upcoming meal

        Output:
        * Optimal: dict containing:
                    * amount of time in range
                    * optimal bolus amount
                    * optimal bolus time
                    * grid from grid search
        """
        # define range of bolus times and amounts to try.
        start_time = self.patient.Xtest.index[-1] + dtm.timedelta(minutes=5)
        carb_targ = int(carbs / self.patient.carbs_per_insulin)
        bolus_amounts = list(range(0, carb_targ + 12, 2))
        bolus_times = pd.date_range(start=start_time, end=carb_t, freq="5T")

        # create grid for brute force grid search
        grid = list(itertools.product(bolus_amounts, bolus_times))

        # loop through permutations and check forecast against target_range
        in_range = []
        for b, b_t in grid:
            inserts = {"carb_grams": {carb_t: carbs}, "all_insulin": {b_t: b}}
            # currently using dynamic_forecast instead of hybrid forecast
            ypred = self.dynamic_forecast(
                self.patient.Xtest, self.patient.ytest, start_time, inserts
            )

            # check performance
            # 110 is the optimal glucose value
            error = sum([(x - 110) ** 2 for x in ypred.ypred])
            in_range.append(error)

        # get the feature pair that maximizes time in target range
        min_ind = [i for i, x in enumerate(in_range) if x == min(in_range)]
        optimal_pair = grid[min_ind[0]]

        # save the optimal ypred
        inserts = {
            "carb_grams": {carb_t: carbs},
            "all_insulin": {optimal_pair[1]: optimal_pair[0]},
        }
        ypred = self.dynamic_forecast(
            self.patient.Xtest, self.patient.ytest, start_time, inserts
        )

        # handle potential errors
        if len(optimal_pair) == 0:
            raise ValueError(
                "Oops! Current bolus options do not result in "
                "forecasted blood glucose in target range."
                "Try new bolus options."
            )
        # define output variable
        optimal = {
            "error": str(min(in_range)),
            "bolus amount": optimal_pair[0],
            "bolus time": optimal_pair[1],
            "grid": grid,
            "grid_error": in_range,
            "ypred": ypred,
        }
        return optimal


class MultiOutStep:
    """
    Class for a single output step in a multioutput model.
    Takes the following parameters:
    * estimator (obj): the sklearn base estimator
    * estimator_title (str): alias for the estimator (mostly for plotting)
    * auto_order (int): the autoregressive order of the model
    * exog_order (list): the order of the exogenous variables
    * exog_delay (list): delay of the exogenous variables
    * horizon (int): total number of steps in the full multioutput model
    * step (int): the current output time step of this individual model
    """
    def __init__(
        self,
        estimator,
        estimator_title,
        auto_order,
        exog_order,
        exog_delay,
        horizon,
        step,
        ):
        self.estimator = estimator
        self.name = estimator_title
        self.auto_order = auto_order
        self.exog_order = exog_order
        self.exog_delay = exog_delay
        self.horizon = horizon
        self.output_step = step

    def fit(self, X, y):
        """
        Wrapper for the fit function of the base estimator.
        """
        return self.estimator.fit(X, y[:, self.output_step])

    def predict(self, X):
        """
        Wrapper for the predict function of the base estimator.
        """
        return self.estimator.predict(X)

    def model_performance(self, X, y):
        """
        Compute r^2 for the model using a subset of the data.
        """
        # add lags in design matrix
        features, target = core.add_lags(
            X, y, self.auto_order, self.exog_order, self.exog_delay
        )
        # add all target lags given the horizon
        features, target = core.horizon_transform(
            features, target, horizon=self.horizon
        )
        # predict values
        ypred = self.predict(features)
        # get r2 score
        r2 = r2_score(
            target[:, self.output_step],
            ypred,
            multioutput="raw_values")
        return r2
