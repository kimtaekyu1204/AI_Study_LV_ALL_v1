# !pip install transformers

# # 1. 데이터 로드 및 정제

import pandas as pd
import numpy as np
import urllib.request
import os
from tqdm import tqdm
import tensorflow as tf

from transformers import BertTokenizer, TFBertModel

# 네이버 영화 리뷰 데이터 학습을 위해 훈련 데이터와 테스트 데이터를 다운로드합니다.
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
                           filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
                           filename="ratings_test.txt")
# >>> 출력:
# ('ratings_test.txt', <http.client.HTTPMessage at 0x7a5639802290>)

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

print('훈련용', train_data.shape)
# >>> 출력:
# 훈련용 (150000, 3)

print('테스트용', test_data.shape)
# >>> 출력:
# 테스트용 (50000, 3)

train_data.head()
# >>> 출력:
#          id                                           document  label
# 0   9976970                                아 더빙.. 진짜 짜증나네요 목소리      0
# 1   3819312                  흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나      1
# 2  10265843                                  너무재밓었다그래서보는것을추천한다      0
# 3   9045019                      교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정      0
# 4   6483659  사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 ...      1

# 중복데이터와 결측값은 제거
train_data.drop_duplicates(subset=['document'], inplace=True)  # document 컬럼에서 중복된 내용이 있다면 제거.
train_data.dropna(how='any', inplace=True)   #  Null(NaN) 값이 존재하는 행 제거

train_data.shape
# >>> 출력:
# (146182, 3)

test_data.drop_duplicates(subset=['document'], inplace=True)  # document 컬럼에서 중복된 내용이 있다면 제거.
test_data.dropna(how='any', inplace=True)   #  Null(NaN) 값이 존재하는 행 제거

test_data.shape
# >>> 출력:
# (49157, 3)

# # 2. BERT 의 입력

# 'klue/bert-base' : huggingface 의 model hub 에 등록된, 한국어에 최적화된 토크나이저 설정과 vocab 파일까지 다운로드
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
# >>> 출력:
# /usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
# The secret `HF_TOKEN` does not exist in your Colab secrets.
# To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
# You will be able to reuse this secret in all of your notebooks.
# Please note that authentication is recommended but still optional to access public models or datasets.
#   warnings.warn(

# ## 정수 인코딩

tokenizer.tokenize("보는 내내 그대로 들어맞는 예측. 카리스마 없는 악역")
# >>> 출력:
# ['보', '##는', '내내', '그대로', '들어맞', '##는', '예측', '.', '카리스마', '없', '##는', '악역']

tokenizer.encode("보는 내내 그대로 들어맞는 예측. 카리스마 없는 악역")
# >>> 출력:
# [2, 1160, 2259, 6404, 4311, 20657, 2259, 5501, 18, 13132, 1415, 2259, 23713, 3]

tokenizer.decode(tokenizer.encode("보는 내내 그대로 들어맞는 예측. 카리스마 없는 악역"))
# >>> 출력:
# '[CLS] 보는 내내 그대로 들어맞는 예측. 카리스마 없는 악역 [SEP]'

tokenizer.cls_token, tokenizer.cls_token_id
# >>> 출력:
# ('[CLS]', 2)

tokenizer.sep_token, tokenizer.sep_token_id
# >>> 출력:
# ('[SEP]', 3)

tokenizer.pad_token, tokenizer.pad_token_id
# >>> 출력:
# ('[PAD]', 0)

# encode() : 정수인코딩 + 패딩 동시에 가능
#   max_length= : 최대 인코딩 길이
#   padding='max_length' : 최대길이까지 패딩
max_seq_len = 128

encoded_result = tokenizer.encode("전율을 일으키는 영화. 다시 보고 싶은 영화.",
                                  padding='max_length', max_length=max_seq_len)

print(encoded_result)
print('길이:', len(encoded_result))
# >>> 출력:
# [2, 1537, 2534, 2069, 6572, 2259, 3771, 18, 3690, 4530, 1335, 2073, 3771, 18, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 길이: 128

# BERT 의 입력
#  1. 정수인코딩
#  2. 세그먼트 인코딩  (문장구분)
#  3. 어텐션 마스크 (단어토큰, 패딩토큰 구분)

# ## 세그먼트 인코딩

# 우리는 한 종류의 문장만 있으므로 입력의 길이만큼 0 의 시퀀스 준비
print([0] * max_seq_len)
# >>> 출력:
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# ## 어텐션 마스크

valid_num = len(tokenizer.encode("전율을 일으키는 영화. 다시 보고 싶은 영화."))

print(valid_num * [1] + (max_seq_len - valid_num) * [0])
# >>> 출력:
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 입력된 전체 데이터에 대해서 위 과정을 수행하는 함수

def convert_examples_to_features(examples, labels, max_seq_len, tokenizer):
  input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

  for example, label in tqdm(zip(examples, labels), total=len(examples)):
    # 정수인코딩
    input_id = tokenizer.encode(example, padding='max_length', max_length=max_seq_len, truncation=True)
    # 어텐션 마스크
    padding_count = input_id.count(tokenizer.pad_token_id)
    attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count

    # 세그먼트 인코딩
    token_type_id = [0] * max_seq_len
    assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
    assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
    assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)

    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    token_type_ids.append(token_type_id)
    data_labels.append(label)


  # 최종리턴은 array
  input_ids = np.array(input_ids, dtype=int)
  attention_masks = np.array(attention_masks, dtype=int)  # 어텐션 마스크
  token_type_ids = np.array(token_type_ids, dtype=int)  # 세그먼트 인코딩

  data_labels = np.asarray(data_labels, dtype=np.int32)

  return (input_ids, attention_masks, token_type_ids), data_labels

# 훈련데이터 에 대해
train_X, train_y = convert_examples_to_features(train_data['document'], train_data['label'],
                                                max_seq_len=max_seq_len, tokenizer=tokenizer)
# >>> 출력:
# 100%|██████████| 146182/146182 [00:51<00:00, 2858.94it/s]

# 테스트데이터 에 대해
test_X, test_y = convert_examples_to_features(test_data['document'], test_data['label'],
                                                max_seq_len=max_seq_len, tokenizer=tokenizer)
# >>> 출력:
# 100%|██████████| 49157/49157 [00:17<00:00, 2783.96it/s]

# 첫번째 샘플 확인
input_id = train_X[0][0]
attention_mask = train_X[1][0]
token_type_id = train_X[2][0]
label = train_y[0]

print('정수 인코딩:', input_id)
print('어텐션 마스크:', attention_mask)
print('세그먼트 인코딩:', token_type_id)
print('각 인코딩 의 길이:', len(input_id))
print('정수 인코딩 복원:', tokenizer.decode(input_id))
print('레이블 :',label)
# >>> 출력:
# 정수 인코딩: [   2 1376  831 2604   18   18 4229 9801 2075 2203 2182 4243    3    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0]
# 어텐션 마스크: [1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
# 세그먼트 인코딩: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

# # BERT 의 출력

# 한국어 BERT klue/bert-base 모델 불러오기

model = TFBertModel.from_pretrained("klue/bert-base", from_pt=True)
# >>> 출력:
# Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
# WARNING:huggingface_hub.file_download:Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
# Some weights of the PyTorch model were not used when initializing the TF 2.0 model TFBertModel: ['cls.predictions.transform.LayerNorm.weight', 'bert.embeddings.position_ids', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.predictions.decoder.bias', 'cls.predictions.bias']
# - This IS expected if you are initializing TFBertModel from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing TFBertModel from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).
# All the weights of TFBertModel were initialized from the PyTorch model.
# If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertModel for predictions without further training.

# BERT 의 출력 => outputs.

input_ids_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
attention_masks_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
token_type_ids_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)

outputs = model([input_ids_layer, attention_masks_layer, token_type_ids_layer])

# outputs 에는 '두개의 출력' 존재한다.  각각 인덱스 0, 1

# 문장 길이만큼의 출력.
print(outputs[0])
# >>> 출력:
# KerasTensor(type_spec=TensorSpec(shape=(None, 128, 768), dtype=tf.float32, name=None), name='tf_bert_model/bert/encoder/layer_._11/output/LayerNorm/batchnorm/add_1:0', description="created by layer 'tf_bert_model'")

# CLS 토큰 위치 출력
print(outputs[1])
# >>> 출력:
# KerasTensor(type_spec=TensorSpec(shape=(None, 768), dtype=tf.float32, name=None), name='tf_bert_model/bert/pooler/dense/Tanh:0', description="created by layer 'tf_bert_model'")

# outputs[0] 은 (batch size, 128, 768)
#     문장의 길이 개수만큼의 출력.  Many-to-Many 태스크의 경우  outputs[0] 을 사용

# outputs[1] 은 (batch size, 768)
#     [CLS] 토큰 위치의 출력. Many-to_One 태스크의 경우 outputs[1] 을 사용.
#     지금과 같은 영화리뷰 분류 문제는 이에 해당

# # BERT 를 이용한 Many-to-One 모델 만들기

class TFBertForSequenceClassification(tf.keras.Model):
  def __init__(self, model_name):
    super(TFBertForSequenceClassification, self).__init__()
    self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
    self.classifier = tf.keras.layers.Dense(1, # 이진분류
                          kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), # 가중치 초기화 (평균0, 표준편차 0.02)
                          activation='sigmoid', name='classifier')

  def call(self, inputs):
    input_ids, attention_mask, token_type_ids = inputs
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    cls_token = outputs[1]  # Many-to-One
    prediction = self.classifier(cls_token)

    return prediction

model = TFBertForSequenceClassification('klue/bert-base')
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
# >>> 출력:
# Some weights of the PyTorch model were not used when initializing the TF 2.0 model TFBertModel: ['cls.predictions.transform.LayerNorm.weight', 'bert.embeddings.position_ids', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.predictions.decoder.bias', 'cls.predictions.bias']
# - This IS expected if you are initializing TFBertModel from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing TFBertModel from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).
# All the weights of TFBertModel were initialized from the PyTorch model.
# If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertModel for predictions without further training.

model.fit(train_X, train_y, epochs=2, batch_size=64, validation_split=0.2)
# >>> 출력:
# Epoch 1/2
# 1828/1828 [==============================] - 2993s 2s/step - loss: 0.2812 - accuracy: 0.8809 - val_loss: 0.2544 - val_accuracy: 0.9002
# Epoch 2/2
# 1828/1828 [==============================] - 2951s 2s/step - loss: 0.1882 - accuracy: 0.9259 - val_loss: 0.2359 - val_accuracy: 0.9055
# <tf_keras.src.callbacks.History at 0x7a5639a6db90>

results = model.evaluate(test_X, test_y, batch_size=1024)
print('test loss, test acc:', results)
# >>> 출력:
# 49/49 [==============================] - 406s 8s/step - loss: 0.2466 - accuracy: 0.9014
# test loss, test acc: [0.24658207595348358, 0.9013772010803223]

# # 예측

def sentiment_predict(new_sentence):
  input_id = tokenizer.encode(new_sentence, padding='max_length', max_length=max_seq_len, truncation=True)

  padding_count = input_id.count(tokenizer.pad_token_id)
  attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
  token_type_id = [0] * max_seq_len

  input_ids = np.array([input_id])
  attention_masks = np.array([attention_mask])
  token_type_ids = np.array([token_type_id])

  encoded_input = [input_ids, attention_masks, token_type_ids]
  score = model.predict(encoded_input)[0][0]

  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))

sentiment_predict('보던거라 계속보고있는데 전개도 느리고 주인공인 은희는 한두컷 나오면서 소극적인모습에 ')
# >>> 출력:
# 1/1 [==============================] - 3s 3s/step
# 98.02% 확률로 부정 리뷰입니다.

sentiment_predict("스토리는 확실히 실망이였지만 배우들 연기력이 대박이였다 특히 이제훈 연기 정말 ... 이 배우들로 이렇게밖에 만들지 못한 영화는 아쉽지만 배우들 연기력과 사운드는 정말 빛났던 영화. 기대하고 극장에서 보면 많이 실망했겠지만 평점보고 기대없이 집에서 편하게 보면 괜찮아요. 이제훈님 연기력은 최고인 것 같습니다")
# >>> 출력:
# 1/1 [==============================] - 0s 55ms/step
# 99.85% 확률로 긍정 리뷰입니다.

sentiment_predict("남친이 이 영화를 보고 헤어지자고한 영화. 자유롭게 살고 싶다고 한다. 내가 무슨 나비를 잡은 덫마냥 나에겐 다시 보고싶지 않은 영화.")
# >>> 출력:
# 1/1 [==============================] - 0s 55ms/step
# 87.58% 확률로 부정 리뷰입니다.

sentiment_predict("이 영화 존잼입니다 대박")
# >>> 출력:
# 1/1 [==============================] - 0s 54ms/step
# 98.41% 확률로 긍정 리뷰입니다.
