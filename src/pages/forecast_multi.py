from datetime import timedelta
import pickle

import streamlit as st

from sugarTS import core


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
        # Sugar
        ### Forecasting
        What has the model learned about the exogenous variables? Use
        this page to explore this question.
        """
    )
    with st.beta_expander("CLICK HERE to expand discussion"):
        st.markdown(
            """
            NOTE: while the Model Performance page showed the performance
            of a multioutput model with a horizon of 1 hour (12 time steps),
            this page uses a dynamic forecasting approach (due to the
            insertion of future exogenous data that must be considered
            in the forecast). This approach iteratively forecasts just one
            step (5 minutes) ahead and then uses the predicted blood glucose
            value to forecast the next step, etc.

            You can see that this particular model has learned that
            whether you eat carbohydrates or
            inject insulin, you see an increase in blood glucose
            followed by a decrease.

            This is inaccurate, of course;
            in reality, insulin brings blood glucose down.
            So why did the model fail to learn correctly?
            I haven't dug into this particular question too deeply yet,
            but my first hypothesis would be this: unless
            a patient is *very good* at balancing their blood glucose without
            insulin, insulin will necessarily be administered quite frequently
            on a reactive basis. In other words, there is often an increase in
            blood sugar before the patient uses insulin to bring it down. And
            since insulin takes about 15-30 minutes to start bringing blood
            glucose down, the model will often see insulin ⟶ increasing blood
            sugar ⟶ decreasing blood sugar. In this sense, it isn't surprising
            that this is what the model thinks insulin is doing.

            Similarly, why did the
            model mistakenly learn that carbs will naturally drive blood sugar
            upward (which is true) and then downward (which is false)?
            It is possible
            that this patient had enough scenarios where insulin and carbs
            were given in close proximity that the model was unable to
            temporally disentangle their effects. Consequently, it attributed
            their combined effects to both of them. That's my theory, anyway.

            One possible way to test this theory would be to train a model
            only on
            streteches of time where just insulin (and not carbs) is present.
            If
            this resulted in the model learning the true effect of insulin, you
            could then further train that model on stretches of time where only
            carbs are present. You could then test the model on data where both
            insulin and carbs are present.
            """
        )
    st.markdown(
        """
        *Instructions:*
        Use the UI on the sidebar to input the timing and the amount of
        carbohydrates eaten and/or insulin injected in the near future.
        Then observe the model's predicted effect of these interventions
        on future blood glucose.

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
    st.sidebar.markdown("# Forecasting Options")
    st.sidebar.markdown("***")
    st.sidebar.markdown("## Carbohydrates")
    meal_t = st.sidebar.select_slider(
        label="How many minutes until your next meal?", options=time_list
    )
    carb_time = current_time + timedelta(minutes=meal_t)
    carbs = st.sidebar.number_input("How many carbs in your next meal?")

    # get bolus information from user
    st.sidebar.markdown("***")
    st.sidebar.markdown("## Insulin")
    bolus_t = st.sidebar.select_slider(
        label="How many minutes until your next insulin bolus?",
        options=time_list
    )
    bolus_time = current_time + timedelta(minutes=bolus_t)
    units = st.sidebar.number_input("How many units of insulin in this bolus?")

    # do the forecast
    start_time = current_time + timedelta(minutes=5)
    inserts = {
        "carb_grams": {carb_time: carbs},
        "all_insulin": {bolus_time: units}}
    ypred = vm.dynamic_forecast(
        vm.patient.Xtest,
        vm.patient.ytest,
        start_time,
        inserts)

    # plot the model forecast for the optimal bolus
    fig = core.plot_forecast(patient.ytest, ypred, return_flag=True)
    fig = core.plot_optimal_boundaries(patient, fig)
    st.plotly_chart(fig)

    # display info
    st.markdown(
        """
        ### **Predicted glucose with:**
        ### {} units of insulin at {}
        ### and
        ### {} grams of carbs at {}.
        """.format(
            units, bolus_time.time(), carbs, carb_time.time()
        )
    )
