import datetime
from datetime import timedelta
import pickle

import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from sugartime import core


@st.cache
def load_saved_model():
    """
    Load object containing a saved model and the patient data.
    """
    with open("./models/vanilla_SVR_patient01a.pickle", "rb") as f:
        return pickle.load(f)


def plot_test_set(patient, start_time):
    """
    Plots the blood glucose time series for the entire test set, along
    with a marker showing where the user has selected with a slider;
    this determines the start time of a forecast plot, located below
    this plot.
    Takes the following:
    * patient (obj): the patient's data
    * start_time (datetime): the time selected from the slider widget
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=patient.ytest.index,
            y=patient.ytest,
            mode="lines",
            name="y")
    )
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                # x-reference is assigned to the x-values
                xref="x",
                # y-reference is assigned to the plot paper [0,1]
                yref="paper",
                x0=start_time,
                y0=0,
                x1=start_time,
                y1=1,
                opacity=0.9,
                layer="above",
                line_width=4,
            )
        ]
    )
    fig.update_xaxes(
        title_text="Date/Time",
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
    fig.update_layout(
        title={
            "text": "Glucose values from test set",
            "y": 0.88,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        }
    )
    return fig


def write():
    st.markdown(
        """
        # SugarTime
        ### Model Performance
        This page lets you visualize how the model performs on data that
        it hasn't seen yet.
        """
    )
    with st.beta_expander("CLICK HERE to expand discussion"):
        st.markdown(
            """
            The dataset is split into two sets: a training set and a
            testing set. The model has been trained on the training set, and
            we can use the model to perform inference on data from the
            testing set here.

            The time series model is auto-regressive with exogenous variables
            (ARX). The base algorithm used in such a model can be any
            regression algorithm; here I currently use a support vector
            machine.

            The full model actually consists of several models, each
            individually
            fit to a different lag of the target variable. In other words,
            there
            is one model fit to the glucose data at time *t+1*, another fit to
            the
            glucose data at time *t+2*, another at *t+3*, etc.,
            all the way up to the
            selected horizon of the model (which defaults to 12 steps of 5
            minutes
            each, i.e., one hour). Each model represents the best performing
            model
            after optimizing the time-series design hyperparameters (e.g.,
            order of
            the *endogenous* or *target* variable, order of the *exogenous*
            variables, and/or delay of the exogenous variables) at that time
            step.

            Note that this model has essentially learned to revert to the mean.
            Since there is considerable autocorrelation in data from continuous
            glucose monitors, inference becomes less acurrate as the inference
            step gets farther away from the current time *t*.
            Here, instead of relying on the exogenous variables (i.e.,
            carbohydrates and insulin),
            the model does a better job by increasingly bringing the predicted
            value back to the mean, which for this patient is a blood glucose
            level of approximately 100 mg/dL.
            This is obviously not what we want the model to learn. But I have
            yet
            to find an estimator/algorithm that doesn't converge on this
            strategy
            to some extent, which suggests that these two exogenous variables
            are simply not predictive enough to account for significant
            variance beyond the autoregressive component of this model.
            """
        )
    st.markdown(
        """
        *Instructions:*
        Use the slider to select a time within the test set. The model
        will use the data up to that point to generate a forecast for
        the next hour.

        ***
        """
    )
    st.markdown("# Select date/time to show forecast.")

    # load patient data and fit model
    vm = load_saved_model()
    patient = vm.patient

    # make datetime selection slider
    x_index = patient.Xtest.index
    start_time = st.slider(
        "Move the slider to select the forecast date/time",
        min_value=x_index[40].to_pydatetime(),
        max_value=x_index[-40].to_pydatetime(),
        value=x_index[45].to_pydatetime(),
        step=timedelta(minutes=60),
        format="MM/DD/YY - hh:mm",
    )

    # plot glucose values for the test set
    fig = plot_test_set(patient, start_time)
    st.plotly_chart(fig)

    # plot performance of model
    st.markdown("# Show forecast vs actual")
    start_time_index = (x_index == pd.Timestamp(start_time)).argmax()
    nsteps = vm.horizon
    ypred = vm.multioutput_forecast(
        patient.Xtest[:start_time_index], patient.ytest[:start_time_index]
    )
    idx = pd.date_range(
        start=start_time,
        end=start_time + timedelta(minutes=5 * (len(ypred) - 1)),
        freq="5T",
    )
    ypred = pd.DataFrame(ypred, columns=["ypred"], index=idx)
    fig = core.plot_forecast(
        patient.ytest[(start_time_index - 40): (start_time_index + nsteps)],
        ypred,
        return_flag=True,
    )
    start_time_text = datetime.datetime.strftime(start_time, "%m/%d/%m %H:%M")
    fig.update_layout(
        title={
            "text": "start time: " + start_time_text,
            "y": 0.88,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        }
    )
    st.plotly_chart(fig)
