import streamlit as st
from sugarTS.model import Patient
from datetime import timedelta
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import datetime


def main():
    # initiate patient object
    patient = Patient()

    # create sidebar
    st.sidebar.title("Sugar")
    st.sidebar.text("Helping Type-1 diabetics stay in range.")

    # get user input
    data_select = st.sidebar.selectbox(
        "Choose data source:",
        ["Let's use example data.", "I'll supply my own data."]
    )
    if data_select in "Let's use example data.":
        patient.load_example_data()
        patient.fit_model()
        current_time = patient.ytest.index[-1]
    # else:
    #     tandem_file = st.sidebar.file_uploader(
    #         "Upload Tandem export",
    #         type="csv")
    #     clarity_file = st.sidebar.file_uploader(
    #         "Upload Clarity export",
    #         type="csv")

    # get meal information from user
    st.sidebar.text('Current time is {}'.format(current_time))
    meal_time = st.sidebar.time_input(
        "When is your next meal?",
        current_time+timedelta(minutes=30)
    )
    carb_time = datetime.datetime.combine(current_time.date(), meal_time)
    carbs = st.sidebar.number_input("How many carbs in your next meal?")

    # calculate the optimal bolus
    optimal = patient.find_optimal_bolus(carbs, carb_time)
    # print('Current time is {}'.format(current_time))
    # print('Give yourself a bolus of ' + str(optimal['bolus amount']) +
    #     ' units at ' + str(optimal['bolus time']))

    # plot the model forecast for the optimal bolus
    bolus = optimal['bolus amount']
    bolus_t = optimal['bolus time']
    x_future = patient.build_x_future(carbs, carb_time, bolus, bolus_t)
    y_ind = pd.date_range(
        start=x_future.index[0],
        end=x_future.index[-1]+timedelta(minutes=5),
        freq='5T')
    ypred = pd.Series(patient.forecast(x_future), index=y_ind)

    st.markdown(
        '## Give yourself a bolus of **{} units** at **{}**.'.format(
            optimal['bolus amount'], optimal['bolus time'].time()
        )
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_ind,
            y=ypred,
            mode='lines',
            name='prediction')
        )
    fig.add_trace(
        go.Scatter(
            x=y_ind,
            y=np.ones(len(y_ind))*patient.target_range[0],
            mode='lines',
            name='lower bound')
        )
    fig.add_trace(
        go.Scatter(
            x=y_ind,
            y=np.ones(len(y_ind))*patient.target_range[1],
            mode='lines',
            name='upper bound')
        )
    fig.update_xaxes(
        title_text='Time',
        title_font=dict(size=18),
        showline=True,
        linewidth=2,
        linecolor='black')
    fig.update_yaxes(
        title_text='Predicted blood glucose',
        title_font=dict(size=18),
        showline=True,
        linewidth=2,
        linecolor='black')
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
