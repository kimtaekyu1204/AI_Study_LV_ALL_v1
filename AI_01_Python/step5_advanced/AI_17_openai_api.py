# # OpenAI API 사용

# ★사전에 OpenAI API 키 받아오기★

# OpenAI API 를 사용하려면
# OPENAI_API_KEY 라는 이름의 환경변수로 OpenAI API Key 값이 설정되어 있어야 한다

# 원래 Api key 같이 '민감한 정보'는 코드에 넣는게 아니라
# OS 의 환경변수에 작성해서 읽어와야 한다

# Linux 의 경우 => export 명령
# ★ Colab 에선 환경변수 설정 불가  permission denied..

# $ export OPENAI_API_KEY=sk-...0K7L
# $ echo $OPENAI_API_KEY
# sk-...0K7L

!export OPENAI_API_KEY=sk-xxxxxxxxx

!echo $OPENAI_API_KEY
# >>> 출력:
# 

# # dotenv 사용하여 파일로부터 환경변수 세팅

!pip install python-dotenv
# >>> 출력:
# Requirement already satisfied: python-dotenv in /usr/local/lib/python3.11/dist-packages (1.0.1)

import os
from dotenv import load_dotenv

ENV_BASE_PATH = r'/content/drive/MyDrive/KoreaIT (코리아it)/250107 자연어처리/ENV'

load_dotenv(dotenv_path=os.path.join(ENV_BASE_PATH, 'NLPenv.txt'))
# >>> 출력:
# True

# python 에서 환경변수 지정하거나 읽기
# os.environ[name] = value 형식으로 지정은 가능
import os

api_key = os.getenv("OPENAI_API_KEY")  # 없으면 None 리턴

# api_key

# - request 만들기
# 
# https://platform.openai.com/docs/api-reference/making-requests
# 
# 
# - Chat request body
# 
# https://platform.openai.com/docs/api-reference/chat/create

# # requests 모듈 사용

import requests

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f'Bearer {api_key}'},
    json = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": "겨울에 피는 꽃은 어떤 것들이 있지?"
            }
          ]
    }
)

response

# API key 가 틀리면 401
# billing quota 설정 안해 놨으면 429 에러 뜬다.
# >>> 출력:
# <Response [200]>

import pprint
result = response.json()
pprint.pprint(result)
# >>> 출력:
# {'choices': [{'finish_reason': 'stop',
#               'index': 0,
#               'logprobs': None,
#               'message': {'content': '겨울에 피는 꽃은 다양하게 있습니다. 몇 가지 예시로는 겨울에 피는 '
#                                      '카네이션, 코스모스, 캘랜듈라, 캐롯, 튤립, 프리지아, 코랄벨 붙양, '
#                                      '로마셔리안, 겨울벽쉬아, 겨울자귀 등이 있습니다. 이 꽃들은 겨울철에도 '
#                                      '피어 축제 분위기를 연출해줍니다.',
#                           'refusal': None,
#                           'role': 'assistant'}}],
#  'created': 1738931937,
#  'id': 'chatcmpl-AyHvFeyt35sDWwnRTfWHjXbnRVDQJ',
#  'model': 'gpt-3.5-turbo-0125',
#  'object': 'chat.completion',
#  'service_tier': 'default',
#  'system_fingerprint': None,

result['choices'][0]['message']['content']
# >>> 출력:
# '겨울에 피는 꽃은 다양하게 있습니다. 몇 가지 예시로는 겨울에 피는 카네이션, 코스모스, 캘랜듈라, 캐롯, 튤립, 프리지아, 코랄벨 붙양, 로마셔리안, 겨울벽쉬아, 겨울자귀 등이 있습니다. 이 꽃들은 겨울철에도 피어 축제 분위기를 연출해줍니다.'

# # openai 모듈 사용

import openai

from openai import OpenAI

client = OpenAI(
    # api_key=api_key
)

chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "여름에 피는 꽃은 어떤 것들이 있지"}]
)

chat_completion
# >>> 출력:
# ChatCompletion(id='chatcmpl-AyI1IIIBqXcRrxhjRpu6i962zH7tw', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='여름에 피는 꽃으로는 해바라기, 백합, 난초, 데이지, 해밀, 베고니아, 코스모스, 물레나물, 라벤더, 튤립, 장미, 수국 등이 있습니다. 이 외에도 다양한 꽃들이 있지만, 여름에 아름다운 꽃들이 피어나는 것을 보며 여름의 아름다움을 느낄 수 있습니다.', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1738932312, model='gpt-3.5-turbo-0125', object='chat.completion', service_tier='default', system_fingerprint=None, usage=CompletionUsage(completion_tokens=144, prompt_tokens=28, total_tokens=172, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))

chat_completion.choices[0].message.content
# >>> 출력:
# '여름에 피는 꽃으로는 해바라기, 백합, 난초, 데이지, 해밀, 베고니아, 코스모스, 물레나물, 라벤더, 튤립, 장미, 수국 등이 있습니다. 이 외에도 다양한 꽃들이 있지만, 여름에 아름다운 꽃들이 피어나는 것을 보며 여름의 아름다움을 느낄 수 있습니다.'

def chat(query):
  chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": query}]
  )
  print(chat_completion.choices[0].message.content)

chat("대한민국의 겨울 날씨는 어때?")
# >>> 출력:
# 한반도에서 겨울은 주로 12월부터 2월까지 지속됩니다. 겨울이 추워서 한낮에도 0도 이하로 떨어지는 날씨가 많고, 특히 북부지방이나 산악지역은 더욱 한파가 심할 수 있습니다. 눈이 내리는 경우도 많고, 바람이 강하게 불면 체감 온도가 더욱 추워질 수 있습니다. 따뜻한 옷을 입고 방한을 잘 해야 합니다.

chat("내 이름은 김봉숙 이야")
# >>> 출력:
# 만나서 반가워요, 김봉숙 씨! 어떤 도움이 필요하신가요?

chat("내 이름이 뭐지?")
# >>> 출력:
# 죄송하지만, 저는 사용자의 실제 이름이나 개인 정보를 알 수 없습니다. 어떤 일에 대해 이야기해 주시면 도와드릴 수 있을지도 모릅니다!
