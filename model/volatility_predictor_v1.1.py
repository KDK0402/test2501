import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib.dates as mdates
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import plotly.graph_objects as go

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_excel('C:/Users/infomax/Documents/10yKTB.xlsx', skiprows=3, usecols=[0,1])
df.columns = ['Date', 'Rate']

# 날짜 형식 변환 및 유효하지 않은 날짜 제거
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # 유효하지 않은 날짜는 NaT로 변환
df = df.dropna(subset=['Date'])  # NaT 값이 있는 행 제거

# 데이터를 날짜 기준으로 오름차순 정렬 (과거 -> 현재)
df = df.sort_values('Date', ascending=True).reset_index(drop=True)

# 데이터 확인 출력
print("\n=== 데이터 확인 ===")
print(f"데이터 시작일: {df['Date'].min().strftime('%Y-%m-%d')}")
print(f"데이터 종료일: {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"총 데이터 수: {len(df)}")

# 일간 변화율 계산 (basis point 단위로 변환: 0.01 = 1bp)
df['Daily_Change'] = df['Rate'].diff() * 100  # bp 단위로 변환

# 변동성 계산 (20일 롤링 윈도우, bp 단위)
df['Volatility'] = df['Daily_Change'].rolling(window=20).std()  # 단순 표준편차만 사용
df = df.dropna()

print("\n=== 변동성 통계 ===")
print(f"평균 일별 변동성(bp): {df['Volatility'].mean():.2f}")
print(f"최대 일별 변동성(bp): {df['Volatility'].max():.2f}")
print(f"최소 일별 변동성(bp): {df['Volatility'].min():.2f}")

# 데이터 스케일링
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Volatility'].values.reshape(-1, 1))

# 시퀀스 데이터 생성
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 20
X, y = create_sequences(scaled_data, seq_length)

# 학습/테스트 데이터 분할 (90:10)
train_size = int(len(X) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 날짜 데이터 준비 (datetime 객체로 유지)
dates = df['Date'].values[seq_length:]
train_dates = dates[:train_size]
test_dates = dates[train_size:]

# 학습/테스트 기간 출력
print("\n=== 데이터 기간 정보 ===")
print(f"전체 데이터 기간: {pd.Timestamp(dates[0]).strftime('%Y-%m-%d')} ~ {pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}")
print(f"학습 데이터 기간: {pd.Timestamp(dates[0]).strftime('%Y-%m-%d')} ~ {pd.Timestamp(dates[train_size-1]).strftime('%Y-%m-%d')}")
print(f"테스트 데이터 기간: {pd.Timestamp(dates[train_size]).strftime('%Y-%m-%d')} ~ {pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}")

# LSTM 모델 구축
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 모델 학습
history = model.fit(X_train, y_train, 
                   epochs=50, 
                   batch_size=32, 
                   validation_split=0.1,
                   verbose=1)

# 전체 데이터에 대한 예측 수행
all_predictions = []
for i in range(len(X)):
    pred = model.predict(X[i:i+1], verbose=0)
    all_predictions.append(pred[0][0])
all_predictions = np.array(all_predictions)

# 예측값 역변환
all_predictions = scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
y_actual = scaler.inverse_transform(y).flatten()

# RMSE 계산
train_rmse = np.sqrt(np.mean((y_actual[:train_size] - all_predictions[:train_size]) ** 2))
test_rmse = np.sqrt(np.mean((y_actual[train_size:] - all_predictions[train_size:]) ** 2))
total_rmse = np.sqrt(np.mean((y_actual - all_predictions) ** 2))

# MAE (Mean Absolute Error) 계산
train_mae = mean_absolute_error(y_actual[:train_size], all_predictions[:train_size])
test_mae = mean_absolute_error(y_actual[train_size:], all_predictions[train_size:])

# R² score 계산
train_r2 = r2_score(y_actual[:train_size], all_predictions[:train_size])
test_r2 = r2_score(y_actual[train_size:], all_predictions[train_size:])

# 결과 테이블 확장
results_table = [
    ['구분', 'RMSE', 'MAE', 'R²'],
    ['학습 데이터', f'{train_rmse:.4f}', f'{train_mae:.4f}', f'{train_r2:.4f}'],
    ['테스트 데이터', f'{test_rmse:.4f}', f'{test_mae:.4f}', f'{test_r2:.4f}'],
    ['전체 데이터', f'{total_rmse:.4f}', f'{mean_absolute_error(y_actual, all_predictions):.4f}', 
     f'{r2_score(y_actual, all_predictions):.4f}']
]
print("\n=== 모델 성능 평가 ===")
print(tabulate(results_table, headers='firstrow', tablefmt='grid'))

# 메인 그래프: 실제 vs 예측 변동성 + 고변동성 구간
plt.figure(figsize=(15, 6))
plt.plot(dates, y_actual, label='실제 일별 변동성', alpha=0.7)
plt.plot(dates, all_predictions, label='예측 일별 변동성', alpha=0.7)
plt.axvline(x=dates[train_size], color='r', linestyle='--', 
            label=f'학습/테스트 데이터 구분\n({pd.Timestamp(dates[train_size]).strftime("%Y-%m-%d")})')

# 변동성 임계값 설정 및 고변동성 구간 표시
threshold = np.percentile(y_actual, 75)  # 상위 25% 기준
plt.axhline(y=threshold, color='g', linestyle='--', 
            label=f'고변동성 임계값 ({threshold:.2f}bp, 상위 25% 수준)')

# 고변동성 예측 구간 하이라이트
high_vol_mask = all_predictions >= threshold
plt.fill_between(dates, 0, all_predictions, where=high_vol_mask, 
                alpha=0.3, color='red', label='고변동성 예측 구간')

# 마지막 데이터 포인트에 수치 표시
last_date = dates[-1]
last_actual = y_actual[-1]
last_pred = all_predictions[-1]
plt.annotate(f'실제: {last_actual:.2f}bp',
            xy=(last_date, last_actual),
            xytext=(10, 10),
            textcoords='offset points',
            ha='left',
            va='bottom')
plt.annotate(f'예측: {last_pred:.2f}bp',
            xy=(last_date, last_pred),
            xytext=(10, -10),
            textcoords='offset points',
            ha='left',
            va='top')

plt.title('국고채 금리 일별 변동성 예측 결과')
plt.xlabel('날짜')
plt.ylabel('일별 변동성 (bp)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# 변동성 국면 예측 성과 통계
total_high_vol_days = np.sum(y_actual >= threshold)
correct_predictions = np.sum((y_actual >= threshold) & (all_predictions >= threshold))
missed_predictions = np.sum((y_actual >= threshold) & (all_predictions < threshold))
false_alarms = np.sum((y_actual < threshold) & (all_predictions >= threshold))

print("\n=== 변동성 국면 예측 성과 ===")
print(f"전체 고변동성 일수: {total_high_vol_days}일")
print(f"고변동성 예측 성공: {correct_predictions}일 ({correct_predictions/total_high_vol_days*100:.1f}%)")
print(f"고변동성 예측 실패: {missed_predictions}일 ({missed_predictions/total_high_vol_days*100:.1f}%)")
print(f"오탐(False Alarm): {false_alarms}일")

# 향후 고변동성 예측 구간 출력
future_high_vol = dates[high_vol_mask][-10:]  # 최근 10개의 고변동성 예측 구간
print("\n=== 최근 고변동성 예측 구간 ===")
for date in future_high_vol:
    print(f"{pd.Timestamp(date).strftime('%Y-%m-%d')}: 예측 변동성 {all_predictions[dates == date][0]:.2f}bp")

# Streamlit 웹앱 부분 추가
import streamlit as st
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(
    page_title="금리 변동성 예측 모델",
    page_icon="📈",
    layout="wide"
)

# 제목 및 설명
st.title("국고채 금리 변동성 예측 대시보드")
st.markdown("LSTM 모델을 활용한 금리 변동성 예측 결과를 보여줍니다.")

# 메인 대시보드 구성
col1, col2 = st.columns(2)

with col1:
    st.subheader("모델 성능 지표")
    # 결과 테이블을 데이터프레임으로 변환하여 표시
    results_df = pd.DataFrame(results_table[1:], columns=results_table[0])
    st.dataframe(results_df)

with col2:
    st.subheader("변동성 국면 예측 성과")
    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.metric("전체 고변동성 일수", f"{total_high_vol_days}일")
        st.metric("예측 성공률", f"{correct_predictions/total_high_vol_days*100:.1f}%")
    with metrics_col2:
        st.metric("예측 실패", f"{missed_predictions}일")
        st.metric("오탐(False Alarm)", f"{false_alarms}일")

# Plotly를 사용한 인터랙티브 그래프
st.subheader("변동성 예측 결과")
fig = go.Figure()

# 실제 변동성과 예측 변동성 라인
fig.add_trace(go.Scatter(
    x=dates, 
    y=y_actual,
    name="실제 일별 변동성",
    line=dict(color='blue', width=1)
))

fig.add_trace(go.Scatter(
    x=dates, 
    y=all_predictions,
    name="예측 일별 변동성",
    line=dict(color='red', width=1)
))

# 고변동성 임계값 라인 추가
fig.add_trace(go.Scatter(
    x=dates,
    y=[threshold] * len(dates),
    name=f'고변동성 임계값 ({threshold:.2f}bp, 상위 25% 수준)',
    line=dict(color='green', width=1, dash='dash')
))

# 고변동성 구간 표시 (실제 변동성이 임계값을 넘는 구간)
high_vol_mask = y_actual >= threshold
high_vol_dates = []
for i in range(len(dates)-1):
    if high_vol_mask[i]:
        fig.add_vrect(
            x0=dates[i],
            x1=dates[i+1],
            fillcolor="rgba(255, 0, 0, 0.2)",
            layer="below",
            line_width=0,
            name="고변동성 구간",
            showlegend=False
        )

fig.update_layout(
    title="국고채 금리 일별 변동성 예측 결과",
    xaxis_title="날짜",
    yaxis_title="일별 변동성 (bp)",
    hovermode='x unified',
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# 최근 5일의 실제/예측 변동성 정보를 표로 표시 (역순)
st.subheader("최근 5일 변동성 정보")
recent_dates = dates[-5:][::-1]  # 역순으로 정렬
recent_actual = y_actual[-5:][::-1]
recent_pred = all_predictions[-5:][::-1]

# 데이터 준비
data = []
for date, act, pred in zip(recent_dates, recent_actual, recent_pred):
    date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
    data.extend([
        [date_str, '실제', f'{act:.2f}'],
        [date_str, '예측', f'{pred:.2f}']
    ])

recent_vol_df = pd.DataFrame(data, columns=['날짜', '구분', '변동성(bp)'])
st.dataframe(recent_vol_df)

# 최근 고변동성 예측 구간
st.subheader("최근 고변동성 예측 구간")
recent_predictions = pd.DataFrame({
    '날짜': [pd.Timestamp(date).strftime('%Y-%m-%d') for date in future_high_vol],
    '예측 변동성(bp)': [all_predictions[dates == date][0] for date in future_high_vol]
})
st.dataframe(recent_predictions)

# 실행 방법을 일반 문자열로 표시
print("""
실행 방법:
1. 터미널/명령 프롬프트를 열고
2. 파일이 있는 디렉토리로 이동:
   cd C:/Users/infomax/Desktop/Cursor/model
3. 다음 명령어 실행:
   streamlit run volatility_predictor_v1.1.py
4. 커밋 후 푸쉬
""")