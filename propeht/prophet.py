from fbprophet import Prophet
import streamlit as st
import pandas as pd

# 데이터 불러오기
@st.cache
def load_data(file):
    df = pd.read_csv(file)
    return df

# Prophet 모델 학습하기
def train_prophet_model(df):
    m = Prophet()
    m.fit(df)
    return m

# 예측 결과 시각화하기
def plot_prophet_forecast(m, forecast):
    fig = m.plot(forecast)
    st.write(fig)

def main():
    st.title("Prophet 라이브러리를 활용한 데이터 예측")

    # 데이터 불러오기
    data = load_data("data.csv")

    # Prophet 모델 학습하기
    model = train_prophet_model(data)

    # 예측하기
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # 예측 결과 시각화하기
    plot_prophet_forecast(model, forecast)