import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import plotly.graph_objects as go
from arch import arch_model
import streamlit as st
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error, r2_score

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.rcParams['axes.unicode_minus'] = False

class VolatilityPredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.lstm_model = None
        self.garch_results = None
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        # 데이터 로드 및 전처리
        self.df = pd.read_excel(self.file_path, skiprows=3, usecols=[0,1])
        self.df.columns = ['Date', 'Rate']
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        self.df = self.df.dropna(subset=['Date'])
        self.df = self.df.sort_values('Date', ascending=True).reset_index(drop=True)
        
        # 변동성 계산
        self.df['Daily_Change'] = self.df['Rate'].diff() * 100
        self.df['Volatility'] = self.df['Daily_Change'].rolling(window=20).std()
        self.df = self.df.dropna()
        
    def prepare_lstm_data(self, seq_length=20):
        # LSTM 데이터 준비
        scaled_data = self.scaler.fit_transform(self.df['Volatility'].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:(i + seq_length)])
            y.append(scaled_data[i + seq_length])
            
        return np.array(X), np.array(y)
    
    def train_lstm(self, X, y, train_size=0.9):
        # LSTM 모델 학습
        train_idx = int(len(X) * train_size)
        X_train, X_test = X[:train_idx], X[train_idx:]
        y_train, y_test = y[:train_idx], y[train_idx:]
        
        self.lstm_model = Sequential([
            LSTM(50, activation='relu', input_shape=(X.shape[1], 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mse')
        self.lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, 
                          validation_split=0.1, verbose=0)
        
        return X_train, X_test, y_train, y_test
    
    def train_garch(self):
        # GARCH 모델 학습
        model = arch_model(self.df['Rate'], vol="Garch", p=1, q=1, dist="normal")
        self.garch_results = model.fit(disp='off')
        return self.garch_results
    
    def create_streamlit_app(self):
        st.set_page_config(page_title="금리 변동성 예측 모델", page_icon="📈", layout="wide")
        st.title("국고채 금리 변동성 예측 대시보드")
        
        # 데이터 준비
        self.load_data()
        X, y = self.prepare_lstm_data()
        X_train, X_test, y_train, y_test = self.train_lstm(X, y)
        self.train_garch()
        
        # 탭 생성
        tab1, tab2, tab3 = st.tabs(["LSTM 모델", "GARCH 모델", "모델 비교"])
        
        with tab1:
            self.show_lstm_results(X, y, X_train, X_test)
            
        with tab2:
            self.show_garch_results()
            
        with tab3:
            self.show_model_comparison()
    
    def show_lstm_results(self, X, y, X_train, X_test):
        st.header("LSTM 기반 변동성 예측")
        
        # LSTM 예측
        predictions = self.lstm_model.predict(X)
        predictions = self.scaler.inverse_transform(predictions)
        actual = self.scaler.inverse_transform(y)
        
        # 성능 지표 계산
        train_size = len(X_train)
        metrics = self.calculate_metrics(actual, predictions, train_size)
        
        # 결과 표시
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("모델 성능 지표")
            st.dataframe(pd.DataFrame(metrics))
            
        # LSTM 그래프
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index[20:], y=actual.flatten(),
                               name="실제 변동성", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.df.index[20:], y=predictions.flatten(),
                               name="LSTM 예측", line=dict(color='red')))
        
        fig.update_layout(title="LSTM 변동성 예측 결과",
                         xaxis_title="시간",
                         yaxis_title="변동성 (bp)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_garch_results(self):
        st.header("GARCH(1,1) 기반 변동성 예측")
        
        # GARCH 결과
        conditional_vol = self.garch_results.conditional_volatility * 100
        forecast = self.garch_results.forecast(horizon=5)
        forecast_vol = np.sqrt(forecast.variance.iloc[-1]) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("GARCH 모델 파라미터")
            # 파라미터 데이터프레임 수정
            params = self.garch_results.params
            pvalues = self.garch_results.pvalues
            param_names = params.index.tolist()  # 실제 파라미터 이름 사용
            
            param_df = pd.DataFrame({
                '파라미터': param_names,
                '추정값': params.values,
                'P-value': pvalues.values
            })
            st.dataframe(param_df)
        
        with col2:
            st.subheader("5일 예측 변동성")
            forecast_df = pd.DataFrame({
                '예측일': [f'D+{i+1}' for i in range(5)],
                '예측 변동성(bp)': forecast_vol
            })
            st.dataframe(forecast_df)
        
        # GARCH 그래프
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Rate'],
                               name="실제 금리", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.df.index, y=conditional_vol,
                               name="GARCH 변동성", line=dict(color='red')))
        
        fig.update_layout(title="GARCH 조건부 변동성",
                         xaxis_title="시간",
                         yaxis_title="변동성 (bp)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_model_comparison(self):
        st.header("LSTM vs GARCH 모델 비교")
        
        comparison_df = pd.DataFrame({
            '구분': ['LSTM', 'GARCH'],
            '특징': [
                '비선형적 패턴 포착, 장기 의존성 학습 가능',
                '변동성 군집화 현상 포착, 해석 용이'
            ],
            '장점': [
                '복잡한 패턴 학습 가능, 높은 예측 정확도',
                '계산 효율적, 통계적 해석 가능'
            ],
            '단점': [
                '많은 데이터 필요, 블랙박스 특성',
                '선형적 관계만 포착, 복잡한 패턴 학습 제한'
            ]
        })
        
        st.dataframe(comparison_df)
        
        # 두 모델의 예측 결과 비교 그래프
        lstm_pred = self.lstm_model.predict(X)
        lstm_pred = self.scaler.inverse_transform(lstm_pred)
        garch_vol = self.garch_results.conditional_volatility * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index[20:], y=self.df['Volatility'][20:],
                               name="실제 변동성", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.df.index[20:], y=lstm_pred.flatten(),
                               name="LSTM 예측", line=dict(color='red')))
        fig.add_trace(go.Scatter(x=self.df.index, y=garch_vol,
                               name="GARCH 예측", line=dict(color='green')))
        
        fig.update_layout(title="모델 비교: LSTM vs GARCH",
                         xaxis_title="시간",
                         yaxis_title="변동성 (bp)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def calculate_metrics(actual, pred, train_size):
        train_rmse = np.sqrt(np.mean((actual[:train_size] - pred[:train_size]) ** 2))
        test_rmse = np.sqrt(np.mean((actual[train_size:] - pred[train_size:]) ** 2))
        train_mae = mean_absolute_error(actual[:train_size], pred[:train_size])
        test_mae = mean_absolute_error(actual[train_size:], pred[train_size:])
        train_r2 = r2_score(actual[:train_size], pred[:train_size])
        test_r2 = r2_score(actual[train_size:], pred[train_size:])
        
        return {
            '구분': ['학습', '테스트'],
            'RMSE': [train_rmse, test_rmse],
            'MAE': [train_mae, test_mae],
            'R²': [train_r2, test_r2]
        }

if __name__ == "__main__":
    file_path = "C:/Users/infomax/Documents/10yKTB.xlsx"
    predictor = VolatilityPredictor(file_path)
    predictor.create_streamlit_app()
