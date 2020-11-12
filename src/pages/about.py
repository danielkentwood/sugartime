import streamlit as st


def write():
    st.markdown(
        """
        # SugarTime
        ### About

        This is a demo of a decision-support application designed to
        assist diabetics in balancing their glucose and insulin needs.

        Here's what it needs to work:
        * Data from a continuous glucose monitor
        * Amounts and times of carbohydrate intake from meals, snacks, etc.
        * Amounts and times of insulin inputs (including corrective boluses
        and basal rates)

        Some quick terminology. The time series model that we use in this app
        treats the glucose time series as the *endogenous* variable. In our
        case, we also have two variables (i.e., carbohydrates and insulin) that
        influence the endogenous variable, but are not influenced by it (at
        least not directly). We call these *exogenous* variables. I'll use
        these terms later on.

        ### Real-time vs. static data
        Ideally, this app would stream real-time updates of data from a
        patient's device(s). Unfortunately, FDA regulations have, until very
        recently, discouraged manufacturers of continuous glucose monitors from
        providing real-time access to a patient's glucose levels on any
        platform other than the device itself. Patients and their doctors
        were only able to access up data with a three-hour lag. As of now, the
        manufacturers
        still haven't updated their software to make real-time data available
        (hopefully soon!). The only workarounds that I am aware of require
        [third-party hardware] (http://www.nightscout.info/).

        Due to these constraints, the app is only able to simulate real-time
        forecasting with static exemplar data. Once the
        barriers to real-time streaming are down, the app can be modified
        to accept automated API requests to update patient data in real time.

        ### *A final caveat:*

        I am asking a model trained
        on impoverished, 5-minute resolution data to forecast typically 60-90
        minutes into the future.
        It does a pretty good job, but as you will see,
        it is not ready to be used for medical decision support.
        Why? Because blood glucose is influenced by so much more than
        just exogenous carbohydrates and insulin.
        It is massively influenced by sleep/wake cycle,
        exercise, medication, other hormonal factors, the type of food (e.g.,
        fat content, complex vs. simple carbs), etc. As wearable devices get
        better at tracking these sources of glucose variability, models trained
        on such data should begin to approach the kind of reliability needed
        before consumers use them to make medical decisions.

        ***

        ## CODE

        Here is the github repo for this app:
        https://github.com/danielkentwood/sugar


        """
    )
