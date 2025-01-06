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
