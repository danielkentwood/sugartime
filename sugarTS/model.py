from datetime import timedelta
import itertools

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from sugarTS.core import load_and_clean_example_data
from sugarTS.core import split_data
from fireTS.models import NARX


class Patient():
    '''
    Object containing data, timeseries model, and forecasting for an
    individual patient.
    '''

    def __init__(self):
        self.carbs_per_insulin = 5
        self.data = load_and_clean_example_data()
        self.target_range = (80, 140)

    def find_optimal_bolus(self, carbs, carb_t):
        '''
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
        '''
        carb_targ = int(carbs/self.carbs_per_insulin)
        bolus_amounts = list(range(1, carb_targ+20, 1))
        bolus_times = pd.date_range(start=self.Xtest.index[-1] +
                                    timedelta(minutes=5),
                                    end=carb_t, freq='5T')
        # create grid for brute force grid search
        grid = [(r[0], r[1]) for r in itertools.product(bolus_amounts,
                                                        bolus_times)]
        # loop through permutations and check forecast against target_range
        in_range = list()
        for b, b_t in grid:
            X_future = self.build_x_future(carbs, carb_t, b, b_t)
            ypred = self.forecast(X_future)
            # check performance
            over = list(map(lambda x: x - self.target_range[1],
                        [x for x in ypred if x > self.target_range[1]]))
            under = list(map(lambda x: self.target_range[0] - x,
                         [x for x in ypred if x < self.target_range[0]]))
            # get the amount of error
            error = sum([sum(over), sum(under)])
            in_range.append(error)
        # get the feature pair that maximizes time in target range
        min_in_range = min(in_range)
        min_ind = [i for i, x in enumerate(in_range) if x == min_in_range]
        optimal_pair = grid[min_ind[0]]
        # handle potential errors
        if len(optimal_pair) == 0:
            raise ValueError('Oops! Current bolus options do not result in ' +
                             'forecasted blood glucose in target range.' +
                             'Try new bolus options')
            return -1
        optimal = {'error': str(min_in_range),
                   'bolus amount': optimal_pair[0],
                   'bolus time': optimal_pair[1],
                   'grid': grid}
        return optimal

    def fit_model(self):
        '''
        Fit a time-series model to the patient's data.
        Currently only uses the fireTS model
        '''
        # split data into train and test sets
        Xtrain, ytrain, Xtest, ytest = split_data(self.data)
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest
        # initialize model object
        narx_mdl = NARX(LinearRegression(),
                        auto_order=30,
                        exog_order=[60, 60],
                        exog_delay=[0, 0])
        # fit model
        narx_mdl.fit(Xtrain, ytrain)
        self.model = narx_mdl

    def build_x_future(self, carbs, carb_t, bolus, bolus_t):
        '''
        Build the future X matrix. This contains the future carbs and the
        future bolus.
        Inputs:
        * carbs: (int) number of future carbs to be eaten
        * carb_t: (datetime object) when the future carbs will be eaten
        * bolus: (int) units of insulin to administer
        * bolus_t: (datetime object) when the bolus will be given
        Output:
        * X_future: (dataframe) matrix containing future carbs and bolus.
        '''
        # make tests to ensure that carb_t and bolus_t are datetime objects
        # make a test to ensure that bolus_t is between start time and carb_t
        t_0 = self.Xtest.index[-1]+timedelta(minutes=5)
        # assume the end of the forecast is 1 hour after carb time
        t_end = carb_t+timedelta(minutes=5*12)
        ind = pd.date_range(start=t_0, end=t_end, freq='5T')
        X_future = pd.DataFrame(np.zeros([len(ind), 2]),
                                columns={'carb_grams', 'bolus_units'})
        X_future.index = ind
        X_future.loc[carb_t, 'carb_grams'] = carbs
        X_future.loc[bolus_t, 'bolus_units'] = bolus
        return X_future

    def forecast(self, X_future):
        '''
        Forecast the future blood glucose by supplying information about
        future food and future insulin.
        Input:
        * X_future: (dataframe) matrix containing future carbs and bolus.
        Outpu:
        * yforecast_narx: (array-like) vector containing forecasted
        blood glucose values.
        '''
        steps = X_future.shape[0]+1
        yforecast_narx = self.model.forecast(self.Xtest,
                                             self.ytest,
                                             step=steps,
                                             X_future=X_future)
        return yforecast_narx
