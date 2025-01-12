# https://jocoding.net/gptbitcoin-bithumb

#4. 환경설정

# .env (해당 파일을 만들어서, 여기에다가 넣어두는거)
# BITHUMB_ACCESS_KEY="your-api-key"
# BITHUMB_SECRET_KEY="your-api-key"
# OPENAI_API_KEY="---"
# GEMINI_API_KEY="----"
# 참고로, Google AI Studio AI키는 "----" <- 공짜라서 오히려 더 좋을수도...

# requirements.txt (해당 파일을 만들어서, 여기에다가 넣어두는거)
# python-dotenv
# openai
# python-bithumb

# test.py
import os
from dotenv import load_dotenv
load_dotenv ()

print(os.getenv("BITHUMB_ACCESS_KEY"))
print(os.getenv("BITHUMB_SECRET_KEY"))
print(os.getenv("OPENAI_API_KEY"))

# 이상의 3개 파일 만들어서 넣고 터미널에서 3개의 파일 실행해보기
# python --version
# pip install -r requirements.txt
# python test.py

# 이제 제품 만들기
# https://github.com/youtube-jocoding/python-bithumb
# https://chatgpt.com/g/g-6764a2fc67988191a4382c8511d509d0-python-bithumb-gaideu
# https://www.deepl.com/en/translator
# OpenAI Playground: https://platform.openai.com/playground/

### mvp.py ###
import os
from dotenv import load_dotenv
load_dotenv()
import python_bithumb

def ai_trading():
    # 1. 빗썸 차트 데이터 가져오기 (30일 일봉)
    df = python_bithumb.get_ohlcv("KRW-BTC", interval="day", count=30)

    # 2. AI에게 데이터 제공하고 판단 받기
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert in Bitcoin investing. Tell me whether to buy, sell, or hold at the moment based on the chart data provided. response in json format.\n\nResponse Example:\n{\"decision\": \"buy\", \"reason\": \"some technical reason\"}\n{\"decision\": \"sell\", \"reason\": \"some technical reason\"}\n{\"decision\": \"hold\", \"reason\": \"some technical reason\"}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": df.to_json()
                    }
                ]
            }
        ],
        response_format={
            "type": "json_object"
        }
    )
    result = response.choices[0].message.content

    # 3. AI의 판단에 따라 실제로 자동매매 진행하기
    import json
    result = json.loads(result)
    access = os.getenv("BITHUMB_ACCESS_KEY")
    secret = os.getenv("BITHUMB_SECRET_KEY")
    bithumb = python_bithumb.Bithumb(access, secret)

    my_krw = bithumb.get_balance("KRW")
    my_btc = bithumb.get_balance("BTC")

    print("### AI Decision: ", result["decision"].upper(), "###")
    print(f"### Reason: {result['reason']} ###")
    
    if result["decision"] == "buy":
        if my_krw > 5000:
            print("### Buy Order Executed ###")
            bithumb.buy_market_order("KRW-BTC", my_krw*0.997)
        else:
            print("### Buy Order Failed: Insufficient KRW (less than 5000 KRW) ###")

    elif result["decision"] == "sell":
        current_price = python_bithumb.get_current_price("KRW-BTC")
        if my_btc * current_price > 5000:
            print("### Sell Order Executed ###")
            bithumb.sell_market_order("KRW-BTC", my_btc)
        else:
            print("### Sell Order Failed: Insufficient BTC (less than 5000 KRW worth) ###")

    elif result["decision"] == "hold":
        print("### Hold Position ###")

import time
while True: #while true는 반복문인데, 무한히 반복
    time.sleep(10) #10초간 쉬었다가 실행 등
    ai_trading()

# 터미널에 이제 파일 실행하면 계속 돌아가지. ./mvp.py 이런식으로 실행

## 현재까지 작성한 코드를 붙여넣고, "openai api가 아닌 gemini api를 활용하고 싶어. 코드를 변경해줘" 이렇식으로 하면 돼...ㅋㅋ

### Gemini API 이용한걸로(앞에는 Open AI API 이용) ###
import os
from dotenv import load_dotenv
load_dotenv()
import python_bithumb
import json
from openai import OpenAI
import time
import re

def ai_trading():
    # 1. 빗썸 차트 데이터 가져오기 (30일 일봉)
    df = python_bithumb.get_ohlcv("KRW-BTC", interval="day", count=30)

    # 2. Gemini API를 통해 데이터 제공하고 판단 요청
    API_KEY = os.getenv("GEMINI_API_KEY")
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in Bitcoin investing. "
                    "Tell me whether to buy, sell, or hold at the moment based on the chart data provided. "
                    "Response in JSON format.\n\n"
                    "Response Example:\n"
                    "{\"decision\": \"buy\", \"reason\": \"some technical reason\"}\n"
                    "{\"decision\": \"sell\", \"reason\": \"some technical reason\"}\n"
                    "{\"decision\": \"hold\", \"reason\": \"some technical reason\"}"
                )
            },
            {
                "role": "user",
                "content": df.to_json()
            }
        ]
    )

    # JSON 데이터만 추출
    raw_result = response.choices[0].message.content
    json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
    else:
        raise ValueError("JSON response could not be extracted.")

    # 3. AI의 판단에 따라 실제로 자동매매 진행하기
    access = os.getenv("BITHUMB_ACCESS_KEY")
    secret = os.getenv("BITHUMB_SECRET_KEY")
    bithumb = python_bithumb.Bithumb(access, secret)

    my_krw = bithumb.get_balance("KRW")
    my_btc = bithumb.get_balance("BTC")

    print("### AI Decision: ", result["decision"].upper(), "###")
    print(f"### Reason: {result['reason']} ###")

    if result["decision"] == "buy":
        if my_krw > 5000:
            print("### Buy Order Executed ###")
            bithumb.buy_market_order("KRW-BTC", my_krw*0.997)
        else:
            print("### Buy Order Failed: Insufficient KRW (less than 5000 KRW) ###")

    elif result["decision"] == "sell":
        current_price = python_bithumb.get_current_price("KRW-BTC")
        if my_btc * current_price > 5000:
            print("### Sell Order Executed ###")
            bithumb.sell_market_order("KRW-BTC", my_btc*0.997)
        else:
            print("### Sell Order Failed: Insufficient BTC (less than 5000 KRW worth) ###")

    elif result["decision"] == "hold":
        print("### Hold Position ###")

while True:
    time.sleep(10)
    ai_trading()

### 

