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

