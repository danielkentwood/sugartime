FROM fedora

RUN dnf install -y python3-pip

RUN pip install tensorflow keras numpy pandas plotly scikit-learn firets
RUN pip install streamlit

RUN mkdir /app
COPY . /app

WORKDIR /app

ENTRYPOINT ["streamlit", "run", "--server.address", "0.0.0.0", "sugar_app.py"]
