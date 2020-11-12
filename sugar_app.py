import streamlit as st

import src.pages.about
import src.pages.performance_multi
import src.pages.forecast_multi
import src.pages.optimize_multi

PAGES = {
    "About": src.pages.about,
    "Model Performance": src.pages.performance_multi,
    "Forecast": src.pages.forecast_multi,
    "Insulin optimization": src.pages.optimize_multi,
}


def main():

    # create sidebar
    st.sidebar.markdown("# SugarTime")
    st.sidebar.markdown(
        """
        Helping Type-1 diabetics stay in range.
        ***
        """
    )

    # navigation radio buttons
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    with st.spinner(f"Loading {selection} ..."):
        page.write()


if __name__ == "__main__":
    main()
