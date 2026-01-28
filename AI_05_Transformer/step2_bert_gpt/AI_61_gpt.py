# # GPT (Generative Pre-Trained Trasformer)

# ## 1. ChatGPT의 역사
# 
# ![](https://wikidocs.net/images/page/184363/gpt0.PNG)
# 
# - BERT가 트랜스포머의 '인코더'로 설계된 모델이라면,
# - GPT는 트랜스포머의 '디코더'로 설계된 모델.
# 
# - Open AI는 2019년에 GPT-1을 공개한 이후로,
# - 2019년 GPT-2, 2020년 GPT-3, 2022년 ChatGPT(GPT 3.5), 2023년에는 GPT-4, 2024년에는 GPT-4o를 공개하며 GPT 시리즈를 발전시켜 왔습니다.
# 
# 
# 
# | &nbsp;       | GPT-1     | GPT-2 | GPT-3  | GPT-3.5 | GPT-4 Models |
# |--------------|-----------|-------|--------|---------|--------------|
# | 파라미터 개수      | 1억 1700만개 | 15억   | 1,750억 | ?       | 1조 8천억(추정)   |
# | 디코더의 층       | 12        | 48    | 96     | ?       | ?            |
# | 처리 가능한 토큰 개수 | 512       | 1024  | 2048   | ?       | 128,000      |
# | 은닉층의 크기      | 768       | 1600  | 12288  | ?       | ?            |

# ## 2. 거대 언어 모델(Large Language Model)
# 
# - 언어 모델(Language Model)은 인공 지능 분야에서 컴퓨터가 사람의 언어를 이해하고 생성할 수 있도록 하는 기술입니다.
# 
# - 이 모델은 대규모의 텍스트 데이터를 통해, 이전 단어들로부터 다음 단어를 예측하는 방식으로 작동합니다.
# 
#   - 예를 들어, "아침에 일어나 창문을 열었더니"라는 문장이 주어지면, 언어 모델은 가장 그럴 듯한 다음 단어들을 나열하여 "바람이 상쾌하게 불어왔다"와 같은 이어지는 문장을 예측하는 모델입니다.
# 
# - 언어 모델의 대표적인 예로는 OpenAI의 GPT(Generative Pre-trained Transformer) 시리즈가 있습니다.
# 
# - 많은 입문자들이 착각하는 내용은 언어 모델과 GPT는 동격의 개념이 아니라는 것입니다.
# 
# - 기본적으로 언어 모델은 이전 단어들로부터 다음 단어를 예측하는 생성 모델이고, GPT는 수많은 언어 모델 중 하나입니다.
#   - 예를 들어서 구글의 Gemini, 앤트로픽의 Claude, 네이버의 CLOVA X와 같은 모델들도 언어 모델 중 하나일 것입니다.
#   
# - 이 중 딥 러닝 언어 모델로서 **파라미터 개수가 충분히 큰 언어 모델**을 우리는 **거대 언어 모델(Large Language Model)**이라고 부릅니다.

# ## 3. GPT의 아키텍처
# 
# ![](https://wikidocs.net/images/page/184363/gpt2.png)
# 
# - 트랜스포머: (인코더-디코더) 구조
# - BERT : 인코더-디코더 아키텍처에서 인코더만 분리한 모델
# 
# - **GPT(디코더 Only)** : 기본적으로 트랜스포머 디코더로만 구성
# 
#   - GPT는 기본적으로 BERT와는 달리 '이전 단어'들로부터 '다음 단어'를 예측하는 모델 이기 때문에,
# 
#   - 다음 단어를 지속적으로 생성할 수 있어 기본적으로 '글쓰기'가 가능한 '생성 모델'입니다.
# 
#   - 트랜스포머 디코더 아키텍처를 가지며 다음 단어를 생성하는 이러한 구조는 현재 대부분의 거대 언어 모델이 채택하고 있는 구조이기도 합니다.

# **[↓그림: GPT-2 아키텍처]**
# 
# ![](https://wikidocs.net/images/page/184363/gpt3.PNG)
# 
# - 위의 GPT-2 아키텍처를 도식화한 그림은 트랜스포머 디코더 층이 이전에 배웠던 초기 트랜스포머 디코더 층과 크게 다르지 않음을 보여줍니다.
# 
# - 이전에 배웠던 트랜스포머에서의 디코더는 인코더가 같이 존재하는 아키텍처였기 때문에 인코더-디코더 어텐션이 존재했는데 여기서는 해당 부분이 제거되었습니다.

# **[GPT-1, GPT-2, GPT-3 아키텍쳐 비교]**
# 
# ![](https://wikidocs.net/images/page/184363/gpt.PNG)
# 
# - 사실 GPT-1, GPT-2, GPT-3는 아키텍처 상으로는 큰 차이를 보이지 않습니다.
# 
# 1. GPT-1
#   - 예컨대, GPT-1은 초기 트랜스포머 디코더에서 인코더-디코더 어텐션이 제거된 아키텍처이고,
# 
# 1. GPT-2
#   - GPT-1에서는 층 정규화(Layer Normalization)가 서브층(Sub layer) 다음에 위치한다면, GPT-2는 서브층(Sub layer)의 입력으로 이동되었습니다.
# 
#   - 그 외에는 초기화 방법이나, 단어 집합의 크기, 토큰 길이의 확장(GPT-1이 512였으나 GPT-2에서 1024로 증가) 정도의 차이를 가집니다.
# 
# 1. GPT-3
#   - 이는 GPT-2에서 GPT-3로 모델이 다시 한 번 변경되었을 때에도 마찬가지입니다. OpenAI는 GPT-3에서도 특별히 새로운 모델, 아키텍처, 알고리즘을 제안하지 않습니다.
#   - 그저 모델이 거대 언어 모델(Large Language Model)이 되면서 새로운 능력들이 발현되기 시작했다는 점이 뚜렷하게 볼 수 있는 차이점입니다.

# # GPT-2 를 이용한 문장 생성

# ## KoGPT-2 로 문장 생성

import numpy as np
import random
import tensorflow as tf
from transformers import AutoTokenizer  # 특정모델에 알맞은 tokenizer 를 자동으로 불러오는 유틸리티 클래스
from transformers import TFGPT2LMHeadModel  # GPT-2 텍스트 생성 모델. TensorFlow기반

model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')
# >>> 출력:
# /usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
# The secret `HF_TOKEN` does not exist in your Colab secrets.
# To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
# You will be able to reuse this secret in all of your notebooks.
# Please note that authentication is recommended but still optional to access public models or datasets.
#   warnings.warn(
# Some weights of the PyTorch model were not used when initializing the TF 2.0 model TFGPT2LMHeadModel: ['transformer.h.6.attn.masked_bias', 'transformer.h.3.attn.masked_bias', 'transformer.h.10.attn.masked_bias', 'transformer.h.7.attn.masked_bias', 'transformer.h.2.attn.masked_bias', 'lm_head.weight', 'transformer.h.5.attn.masked_bias', 'transformer.h.4.attn.masked_bias', 'transformer.h.9.attn.masked_bias', 'transformer.h.11.attn.masked_bias', 'transformer.h.0.attn.masked_bias', 'transformer.h.1.attn.masked_bias', 'transformer.h.8.attn.masked_bias']
# - This IS expected if you are initializing TFGPT2LMHeadModel from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing TFGPT2LMHeadModel from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).
# All the weights of TFGPT2LMHeadModel were initialized from the PyTorch model.
# If your task is similar to the task the model of the checkpoint was trained on, you can already use TFGPT2LMHeadModel for predictions without further training.

sent = '근육이 커지기 위해서는'

input_ids = tokenizer.encode(sent)
input_ids = tf.convert_to_tensor([input_ids])
print(input_ids)
# >>> 출력:
# tf.Tensor([[33245 10114 12748 11357]], shape=(1, 4), dtype=int32)

output = model.generate(input_ids, max_length=128, repetition_penalty=2.0, use_cache=True)

output_ids = output.numpy().tolist()[0]
print(output_ids)
# >>> 출력:
# [33245, 10114, 12748, 11357, 23879, 39306, 9684, 7884, 10211, 15177, 26421, 387, 17339, 7889, 9908, 15768, 6903, 15386, 8146, 12923, 9228, 18651, 42600, 9564, 17764, 9033, 9199, 14441, 7335, 8704, 12557, 32030, 9510, 18595, 9025, 10571, 25741, 10599, 13229, 9508, 7965, 8425, 33102, 9122, 21240, 9801, 32106, 13579, 12442, 13235, 19430, 8022, 12972, 9566, 11178, 9554, 24873, 7198, 9391, 12486, 8711, 9346, 7071, 36736, 9693, 12006, 9038, 10279, 36122, 9960, 8405, 10826, 18988, 25998, 9292, 7671, 9465, 7489, 9277, 10137, 9677, 9248, 9912, 12834, 11488, 13417, 7407, 8428, 8137, 9430, 14222, 11356, 10061, 9885, 19265, 9377, 20305, 7991, 9178, 9648, 9133, 10021, 10138, 30315, 21833, 9362, 9301, 9685, 11584, 9447, 42129, 10124, 7532, 17932, 47123, 37544, 9355, 15632, 9124, 10536, 13530, 12204, 9184, 36152, 9673, 9788, 9029, 11764]

tokenizer.decode(output_ids)
# >>> 출력:
# '근육이 커지기 위해서는 무엇보다 규칙적인 생활습관이 중요하다.\n특히, 아침식사는 단백질과 비타민이 풍부한 과일과 채소를 많이 섭취하는 것이 좋다.\n또한 하루 30분 이상 충분한 수면을 취하는 것도 도움이 된다.\n아침 식사를 거르지 않고 규칙적으로 운동을 하면 혈액순환에 도움을 줄 뿐만 아니라 신진대사를 촉진해 체내 노폐물을 배출하고 혈압을 낮춰준다.\n운동은 하루에 10분 정도만 하는 게 좋으며 운동 후에는 반드시 스트레칭을 통해 근육량을 늘리고 유연성을 높여야 한다.\n운동 후 바로 잠자리에 드는 것은 피해야 하며 특히 아침에 일어나면 몸이 피곤해지기 때문에 무리하게 움직이면 오히려 역효과가 날 수도 있다.\n운동을'

# ## Top 5 뽑기

# "근육이 커지기 위해서는" 다음에 생성될 토큰 top5
output = model(input_ids)
top5 = tf.math.top_k(output.logits[0, -1], k=5)

tokenizer.convert_ids_to_tokens(top5.indices.numpy())
# >>> 출력:
# ['▁무엇보다', '▁우선', '▁반드시', '▁피부', '▁무엇보다도']

# ## Top5 + 램덤

sent = '근육이 커지기 위해서는'
input_ids = tokenizer.encode(sent)

while len(input_ids) < 50:
  output = model(np.array([input_ids]))
  # Top5 단어들 추출
  top5 = tf.math.top_k(output.logits[0, -1], k=5)
  # Top5 단어들 중 랜덤으로 다음 단어 선택.
  token_id = random.choice(top5.indices.numpy())
  input_ids.append(token_id)

tokenizer.decode(input_ids)
# >>> 출력:
# '근육이 커지기 위해서는 피부 보형물과 같은 수술에 비해 더 중요한 수술 방법이 된다.\n그런데, 이러한 수술은 여러 번 시술에 비해 회복기간도 오래 걸리는 단점이 있다. 만일 이러한 시술들이 실패하여 다시 재 시술로 이어지는 경우가 많다면, 그'

# # GPT-2 를 이용한 한국어 챗봇

model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
# >>> 출력:
# Some weights of the PyTorch model were not used when initializing the TF 2.0 model TFGPT2LMHeadModel: ['transformer.h.6.attn.masked_bias', 'transformer.h.3.attn.masked_bias', 'transformer.h.10.attn.masked_bias', 'transformer.h.7.attn.masked_bias', 'transformer.h.2.attn.masked_bias', 'lm_head.weight', 'transformer.h.5.attn.masked_bias', 'transformer.h.4.attn.masked_bias', 'transformer.h.9.attn.masked_bias', 'transformer.h.11.attn.masked_bias', 'transformer.h.0.attn.masked_bias', 'transformer.h.1.attn.masked_bias', 'transformer.h.8.attn.masked_bias']
# - This IS expected if you are initializing TFGPT2LMHeadModel from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing TFGPT2LMHeadModel from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).
# All the weights of TFGPT2LMHeadModel were initialized from the PyTorch model.
# If your task is similar to the task the model of the checkpoint was trained on, you can already use TFGPT2LMHeadModel for predictions without further training.

print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)
print('-' * 10)
print(tokenizer.decode(1))
print(tokenizer.decode(2))
print(tokenizer.decode(3))
print(tokenizer.decode(4))
# >>> 출력:
# 1
# 1
# 3
# ----------
# </s>
# <usr>
# <pad>
# <sys>

# ## 데이터 준비

import pandas as pd
import tqdm
import urllib.request

urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
train_data = pd.read_csv('ChatBotData.csv')
print('챗봇 데이터의 개수 :', len(train_data))
# >>> 출력:
# 챗봇 데이터의 개수 : 11823

# ## 데이터 전처리

def get_chat_data():
  for question, answer in zip(train_data.Q.to_list(), train_data.A.to_list()):
    bos_token = [tokenizer.bos_token_id]
    eos_token = [tokenizer.eos_token_id]
    sent = tokenizer.encode('<usr>' + question + '<sys>' + answer)
    yield bos_token + sent + eos_token

batch_size = 32
dataset = tf.data.Dataset.from_generator(get_chat_data, output_types=tf.int32)

dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=(None,),
                               padding_values=tokenizer.pad_token_id)

# 첫번째 배치

for batch in dataset:
  print(batch)
  break
# >>> 출력:
# tf.Tensor(
# [[    1     2  9349  7888   739  7318   376     4 12557  6824  9108  9028
#    7098 25856     1     3     3     3     3     3     3     3     3     3
#       3     3     3     3     3     3]
#  [    1     2  9020  8263  7497 10192 11615  8210  8006     4 12422  8711
#    9535  7483 12521     1     3     3     3     3     3     3     3     3
#       3     3     3     3     3     3]
#  [    1     2  9085  7597   395  8149 10624  7397 24224 13358  7182     4
#   12079  8135 16899  9677  8234   389     1     3     3     3     3     3
#       3     3     3     3     3     3]
#  [    1     2  9085  7597   395  8149  9465 10624  7397 24224 13358  7182
#       4 12079  8135 16899  9677  8234   389     1     3     3     3     3
#       3     3     3     3     3     3]
#  [    1     2  9943   422   418  9327  8702  7098     4  9847 16912 18328
#    8671  7415  8263  8234   389     1     3     3     3     3     3     3

# 첫번째 배치 중 첫번째 샘플 출력
print(tokenizer.decode(batch[0]))
# >>> 출력:
# </s><usr> 12시 땡!<sys> 하루가 또 가네요.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>

# ## 챗봇 학습하기

adam = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)

# 학습횟수 step 계산
steps = len(train_data) // batch_size + 1

EPOCHS = 3

# tf.GradientTape
#   https://www.tensorflow.org/api_docs/python/tf/GradientTape
#   Record operations for automatic differentiation.  (자동미분)

#   텐서플로는 자동 미분(주어진 입력 변수에 대한 연산의 그래디언트(gradient)를 계산하는 것)을
#   위한 tf.GradientTape API를 제공합니다.
#   tf.GradientTape는 컨텍스트(context) 안에서 실행된 모든 연산을 테이프(tape)에 "기록"합니다.
#   그 다음 텐서플로는 후진 방식 자동 미분(reverse mode differentiation)을 사용해
#   테이프에 "기록된" 연산의 그래디언트를 계산합니다.

# tape.gradient(target, sources)를 통해,
#     target: 미분 대상 (예: 손실 함수)
#     sources: 미분할 변수들

for epoch in range(EPOCHS):
  epoch_loss = 0

  for batch in tqdm.tqdm_notebook(dataset, total=steps):  # steps
    with tf.GradientTape() as tape:  # 실행된 모든 연산을 tape 에 '기록'
      result = model(batch, labels=batch)
      loss = result[0]
      batch_loss = tf.reduce_mean(loss)

    grads = tape.gradient(batch_loss, model.trainable_variables) # 기울기 계산 (미분)
    adam.apply_gradients(zip(grads, model.trainable_variables)) # 학습 진행

    epoch_loss += batch_loss / steps


  print('[Epoch: {:>4}] loss = {:>.9}'.format(epoch + 1, epoch_loss))
# >>> 출력:
# <ipython-input-18-e256cfc4ed75>:4: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0
# Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`
#   for batch in tqdm.tqdm_notebook(dataset, total=steps):  # steps
# WARNING:tensorflow:5 out of the last 5 calls to <function _BaseOptimizer._update_step_xla at 0x79fe6ea4dd00> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
# WARNING:tensorflow:6 out of the last 6 calls to <function _BaseOptimizer._update_step_xla at 0x79fe6ea4dd00> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
# [Epoch:    1] loss = 2.12707853
# [Epoch:    2] loss = 1.69818687
# [Epoch:    3] loss = 1.37541652

# ## 챗봇 실행

text = '오늘도 좋은 하루!'

sent = '<usr>' + text + '<sys>'

input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent)
input_ids = tf.convert_to_tensor([input_ids])

print('정수 인코딩 후 :', input_ids)
print('정수 인코딩 복원 :', tokenizer.decode(input_ids[0]))
# >>> 출력:
# 정수 인코딩 후 : tf.Tensor([[    1     2 10070  7235 10586 12557   376     4]], shape=(1, 8), dtype=int32)
# 정수 인코딩 복원 : </s><usr> 오늘도 좋은 하루!<sys>

output = model.generate(input_ids, max_length=50, early_stopping=True,
               eos_token_id=tokenizer.eos_token_id)

decoded_sentence = tokenizer.decode(output[0].numpy().tolist())
print(decoded_sentence)

# </s><usr> 오늘도 좋은 하루!<sys> 오늘도 좋은 하루네요.</s>
# >>> 출력:
# The following generation flags are not valid and may be ignored: ['early_stopping']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
# </s><usr> 오늘도 좋은 하루!<sys> 오늘도 좋은 하루네요.</s>

print(decoded_sentence.split('<sys> ')[1].replace('</s>', ''))
# >>> 출력:
# 오늘도 좋은 하루네요.

def return_answer_by_chatbot(user_text):
  sent = '<usr>' + user_text + '<sys>'
  input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent)
  input_ids = tf.convert_to_tensor([input_ids])
  output = model.generate(input_ids, max_length=50, do_sample=True, top_k=20)
  sentence = tokenizer.decode(output[0].numpy().tolist())
  chatbot_response = sentence.split('<sys> ')[1].replace('</s>', '')
  return chatbot_response

return_answer_by_chatbot("안녕! 반가워~")
# >>> 출력:
# '잘 지내고 있는 걸 수도 있어요.'

return_answer_by_chatbot("오늘 날씨가 참 좋네")
# >>> 출력:
# '봄의 전령 같아요.'

return_answer_by_chatbot("내 이야기를 들어줄래?")
# >>> 출력:
# '내 이야기는 다 들어줘야죠.'

return_answer_by_chatbot("영화 미션임파서블 재밌어?")
# >>> 출력:
# '영화는미션이 모두 해결될수는 없죠.'

return_answer_by_chatbot("너 인공지능 잘해?")
# >>> 출력:
# '그게 문제가 될 때가 있습니다.'
