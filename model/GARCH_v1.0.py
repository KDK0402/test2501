### 1. 시계열 모델 - GARCH(1,1) ###

# 필요한 라이브러리 불러오기
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# 데이터 불러오기 및 전처리
file_path = r"C:\Users\infomax\Documents\10yKTB.xlsx"
data = pd.read_excel(file_path, header=2, usecols=[0, 1], names=["Date", "Yield"])
data["Date"] = pd.to_datetime(data["Date"])  # 날짜 형식 변환
data.set_index("Date", inplace=True)        # 날짜를 인덱스로 설정

# GARCH(1,1) 모델 적용
model = arch_model(data["Yield"], vol="Garch", p=1, q=1, dist="normal")
results = model.fit()

# 조건부 변동성 추출
data["Volatility"] = results.conditional_volatility

# 5일 뒤 변동성 예측
forecast_5d = results.forecast(start=0, horizon=5)
vol_forecast_5d = forecast_5d.variance.iloc[:, 0]  # 예측된 변동성값 (5일)

# 5일 뒤 변동성 예측값 출력
print("5일 뒤 변동성 예측:")
print(vol_forecast_5d)

# 변동성 예측 시각화
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Yield"], label="10-Year Bond Yield")
plt.plot(data.index, data["Volatility"], label="GARCH Volatility (Conditional)", color="red")
plt.plot(data.index[-len(vol_forecast_5d):], vol_forecast_5d, label="5-Day Forecast Volatility", color="blue", linestyle="--")
plt.title("5-Day GARCH Volatility Forecast")
plt.xlabel("Date")
plt.ylabel("Yield / Volatility")
plt.legend()
plt.grid()
plt.show()

# 결과 요약 출력
print(results.summary())