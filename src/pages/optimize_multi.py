from datetime import timedelta
import pickle

import streamlit as st

from sugartime import core


@st.cache
def load_saved_model():
    """
    Load object containing a saved model and the patient data.
    """
    with open("./models/vanilla_SVR_patient01a.pickle", "rb") as f:
        return pickle.load(f)


def write():
    st.markdown(
        """
        # SugarTime
        ### Optimizing meals and insulin
        Here you can plan out your meal and ask the model to provide
        a recommendation for insulin (i.e., when and how much) that is
        optimized to keep your future blood glucose as close as possible
        to a level of 110 mg/dL.
        """
    )
    with st.beta_expander("CLICK HERE to expand discussion"):
        st.markdown(
            """
        This is the main point of the app. I wanted to create something
        that would help someone I love (who has diabetes) to make more
        informed decisions about their meals and compensatory insulin.
        And while this current iteration of the model doesn't provide
        a useful decision support due to not learning the true impulse
        response functions of carbohydrates and insulin, I think it is
        still interesting to play with optimizing this model *as if*
        it was capable of providing such support.

        I plan to continue exploring new models and data sources in
        order to find a model that can accurately learn the true
        effects of insulin and carbohydrates on the future blood
        glucose of a patient. In the meantime, I hope you enjoy
        playing with this app, and I hope it sparks new ideas about
        how to do this better than I have done it in this current
        iteration.
            """
        )
    st.markdown(
        """
        *Instructions:*
        Use the UI on the sidebar to input the timing and the amount of
        carbohydrates eaten in the near future.
        The app will then calculate the optimal timing and amount of
        insulin to inject, based on the model's understanding of how
        carbohydrates and insulin affect future blood glucose.

        ***
        """
    )

    # load patient data and fit model
    vm = load_saved_model()
    patient = vm.patient

    # set some time variables
    current_time = patient.Xtest.index[-1]
    time_list = list(range(5, 61, 5))

    # get meal information from user
    st.sidebar.markdown("***")
    st.sidebar.markdown("# Optimization Options")
    st.sidebar.markdown("***")
    st.sidebar.markdown("## Carbohydrates")
    st.sidebar.markdown("Current time is {}".format(current_time))
    meal_t = st.sidebar.select_slider(
        label="How many minutes until your next meal?", options=time_list
    )
    carb_time = current_time + timedelta(minutes=meal_t)
    carbs = st.sidebar.number_input("How many carbs in your next meal?")

    # calculate the optimal bolus
    optimal = vm.find_optimal_bolus(carbs, carb_time)

    # plot the model forecast for the optimal bolus
    fig = core.plot_forecast(patient.ytest, optimal["ypred"], return_flag=True)
    fig = core.plot_optimal_boundaries(patient, fig)
    st.plotly_chart(fig)

    # display decision support
    st.markdown(
        "## Give yourself a bolus of **{} units** at **{}**.".format(
            optimal["bolus amount"], optimal["bolus time"].time()
        )
    )
