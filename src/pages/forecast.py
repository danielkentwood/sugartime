from datetime import timedelta
import datetime

import streamlit as st

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


def write():
    st.markdown(
        """
        # Sugar
        ### Forecasting
        placeholder text

        *Instructions:*
        placeholder text

        ***
        """
    )

    # load patient data and fit model
    patient = get_example_patient()
    vm = fit_model(patient)

    # get meal information from user
    st.sidebar.markdown("***")
    current_time = patient.Xtest.index[-1]
    st.sidebar.markdown("Current time is {}".format(current_time))
    meal_time = st.sidebar.time_input(
        "When is your next meal?", current_time + timedelta(minutes=30)
    )
    carb_time = datetime.datetime.combine(current_time.date(), meal_time)
    carbs = st.sidebar.number_input("How many carbs in your next meal?")

    # calculate the optimal bolus
    optimal = vm.find_optimal_bolus(patient.Xtest, carbs, carb_time)

    # plot the model forecast for the optimal bolus
    fig = core.plot_forecast(
        patient.Xtest["estimated_glu"], optimal["ypred"], return_flag=True
    )
    fig = core.plot_optimal_boundaries(patient, fig)
    st.plotly_chart(fig)

    # display decision support
    st.markdown(
        "## Give yourself a bolus of **{} units** at **{}**.".format(
            optimal["bolus amount"], optimal["bolus time"].time()
        )
    )
