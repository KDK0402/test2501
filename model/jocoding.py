# https://jocoding.net/gptbitcoin-bithumb

#4. 환경설정

# .env (해당 파일을 만들어서, 여기에다가 넣어두는거)
# BITHUMB_ACCESS_KEY="your-api-key"
# BITHUMB_SECRET_KEY="your-api-key"
# OPENAI_API_KEY="---"
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

# mvp.py
import os
from dotenv import load_dotenv
load_dotenv()
import python_bithumb

# 1. 빗썸 차트 데이터 가져오기 (30일 일봉)
df = python_bithumb.get_ohlcv("KRW-BTC", interval="day", count=30)
print(df)

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
print(response.choices[0].message.content)

## 현재까지 작성한 코드를 붙여넣고, "openai api가 아닌 gemini api를 활용하고 싶어. 코드를 변경해줘" 이렇식으로 하면 돼...ㅋㅋ


