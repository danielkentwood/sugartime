import datetime
from datetime import timedelta

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from sugarTS import model
from sugarTS import core


@st.cache
def get_example_patient():
    # initiate patient object
    patient = model.Patient()
    # patient.load_example_data()
    patient.load_synthetic_data()
    patient.split_data(
        target_name=[],
        feature_names=["estimated_glu", "carb_grams", "all_insulin"],
        split=[0.75],
    )
    return patient


@st.cache
def fit_model(patient):
    vm = model.VARModel(patient)
    vm.fit_model(maxlags=40, ic="aic")
    return vm


def plot_test_set(patient, start_time):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=patient.Xtest.index,
            y=patient.Xtest["estimated_glu"],
            mode="lines",
            name="y",
        )
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
        # Sugar
        ### Model Performance
        The dataset is split into two sets: a training set and a
        testing set. The model is trained on the testing set, and
        we can use the model to perform inference on data from the
        testing set here.

        *Instructions:*
        Use the slider to select a time within the test set. The model
        will use the data up to that point to generate a forecast for
        the next 1.5 hours.

        ***
        """
    )
    st.markdown("# Select date/time to show forecast.")

    # initiate patient object
    patient = get_example_patient()

    # make datetime selection slider
    x_index = patient.Xtest.index
    start_time = st.slider(
        "When should the forecast start?",
        min_value=x_index[40].to_pydatetime(),
        max_value=x_index[-40].to_pydatetime(),
        value=x_index[45].to_pydatetime(),
        step=timedelta(minutes=60),
        format="MM/DD/YY - hh:mm",
    )

    # plot glucose values for the test set
    fig = plot_test_set(patient, start_time)
    st.plotly_chart(fig)

    # initialize and fit model
    vm = fit_model(patient)

    # plot performance of model
    st.markdown("# Show forecast vs actual")
    start_time_index = (x_index == pd.Timestamp(start_time)).argmax()
    nsteps = 12
    future_x = patient.Xtest[["carb_grams", "all_insulin"]][
        start_time_index : (start_time_index + nsteps)
    ]
    ypred = vm.dynamic_forecast(patient.Xtest[:start_time_index], future_x)
    fig = core.plot_forecast(
        patient.Xtest["estimated_glu"][
            (start_time_index - 40) : (start_time_index + nsteps)
        ],
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
