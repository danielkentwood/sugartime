from datetime import timedelta
import itertools

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from sugar.core import load_and_clean_example_data
from sugar.core import split_data
from fireTS.models import NARX


class Patient():
    '''
    Object containing data, timeseries model, and forecasting for an
    individual patient.
    '''

    def __init__(self):
        self.carbs_per_insulin = 5
        self.data = load_and_clean_example_data()
        self.target_range = (70, 100)

    def find_optimal_bolus(self, carbs, carb_t):
        '''

        '''
        carb_targ = carbs/self.carbs_per_insulin
        bolus_amounts = list(range(5, carb_targ+20, 5))
        bolus_times = pd.date_range(start=self.Xtest.index[-1] +
                                    timedelta(minutes=5),
                                    end=carb_t, freq='5T')
        grid = [(r[0], r[1]) for r in itertools.product(bolus_amounts,
                                                        bolus_times)]
        in_range = list()
        for b, b_t in grid:
            X_future = self.build_x_future(self, carbs, carb_t, b, b_t)
            ypred = self.forecast(self, X_future)
            in_range.append(np.logical_and(ypred > self.target_range[0],
                                           ypred < self.target_range[1]).sum())
        max_mask = [x in max(in_range) for x in in_range]
        optimal_pair = [i for i, j in zip(grid, max_mask) if j]
        if len(optimal_pair) == 0:
            raise ValueError('Oops! Current bolus options do not result in ' +
                             'forecasted blood glucose in target range.' +
                             'Try new bolus options')
            return -1
        if len(optimal_pair) > 1:
            print('Multiple bolus options result in optimal forecast. ' +
                  'Selecting first option.')
            optimal_pair = optimal_pair[0]
        return optimal_pair

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

        '''
        # make tests to ensure that carb_t and bolus_t are datetime objects
        # make a test to ensure that bolus_t is between start time and carb_t
        t_0 = self.Xtest.index[-1]+timedelta(minutes=5)
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

        '''
        steps = X_future.shape[0]+1
        yforecast_narx = self.model.forecast(self.Xtest,
                                             self.ytest,
                                             step=steps,
                                             X_future=X_future)
        return yforecast_narx
