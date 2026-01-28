"""
의도분류모델

개요:
    이 모듈은 AI 자연어처리 수업의 딥러닝/NLP 파트에서 다루는 의도분류모델에 관한
    모든 코드와 설명을 정리한 파일함.

    원본 Jupyter Notebook: 50 의도분류모델.ipynb

============================================
사용 데이터셋: 챗봇 의도 분류 데이터 (한국어)
============================================
- 데이터셋명: Chatbot Intent Classification Dataset
- 출처: 한국어 챗봇 학습용 데이터셋 (공개 데이터)
- 설명: 한국어 질문-답변 쌍과 감정/의도 라벨 데이터
- 파일명: chatbot_data.csv
- 구성: 약 11,000+ 문장 쌍

컬럼 정보:
  * Q: 질문 문장 (입력)
  * A: 답변 문장 (참고용)
  * label: 의도/감정 라벨
    - 0: 일상다반사 (neutral)
    - 1: 이별/부정 (negative)
    - 2: 사랑/긍정 (positive)

df.info() 결과:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 11823 entries, 0 to 11822
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   Q       11823 non-null  object
 1   A       11823 non-null  object
 2   label   11823 non-null  int64
dtypes: int64(1), object(2)

df.head() 결과:
┌─────────────────────────────────────────────┬─────────────────────────────────────────────┬───────┐
│ Q                                           │ A                                           │ label │
├─────────────────────────────────────────────┼─────────────────────────────────────────────┼───────┤
│ 12시 땡!                                    │ 하루가 또 가네요.                           │ 0     │
│ 1지망 학교 떨어졌어                         │ 위로해 드릴께요.                            │ 1     │
│ 3박4일 놀러가고 싶다                        │ 여행은 언제나 좋죠.                         │ 0     │
│ 3박4일 놀러갔다왔어                         │ 잘 놀다 오셨어요?                           │ 0     │
│ PPL 심하네                                  │ 눈살이 찌푸려지죠.                          │ 1     │
└─────────────────────────────────────────────┴─────────────────────────────────────────────┴───────┘
============================================
"""

# # 문장분류를 위한 CNN모델
# '문장'을 '감정 클래스별'로 분류하는CNN 모델 구현
# - CNN 은 이미지 분류 외에도 자연어 분류에도 좋은 성능을 냅니다 (단 **임베딩 품질**이 좋아야 함.)
# - 이미지 이든, 자연어이든  수치(벡터) 로 표현 가능한 대상이면, 특징을 뽑아내도록 CNN 모델 학습인 가능한겁니다
# 필요한 모듈 임포트
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing

# 실행마다 동일한 결과를 얻기 위해 keras 에 랜덤시드 사용
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# CNN 모델 생성용 ↓
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate


# # 데이터 준비
base_path = r'/content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)'
data = pd.read_csv(os.path.join(base_path, 'chatbot_data', 'chatbot_data.csv'))

data
# - 데이터셋 구조
#     - Q (질문),  
#     - A (답변)
#     - label (감정)
#         - **0**: 일상다반사
#         - **1** : 이별(부정)
#         - **2** : 사랑(긍정)
data.shape
data.info()
data['label'].unique()
data['label'].value_counts()
features = data['Q'].tolist()
labels = data['label'].tolist()
features[0]
# # 단어 시퀀스 만들기
#
# ## text_to_word_sequence()
#
# **tf.keras.preprocessing.text.text_to_word_sequence()**
#
# 단어 시퀀스를 만든다.  단어 시퀀스란 단어 토큰들의 순차적 리스트
#
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/text_to_word_sequence
#
# ```python
# tf.keras.preprocessing.text.text_to_word_sequence(
#     input_text,
#     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
#     lower=True,
#     split=' '
# )
# ```
#
# 리턴값: A list of words (or tokens).
preprocessing.text.text_to_word_sequence(features[0])
# 단어 시퀀스 들의 벡터
corpus = [
    preprocessing.text.text_to_word_sequence(text) for text in features
]

corpus
# ## Tokenizer
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
tokenizer.word_index
# ## texts_to_sequence()
sequences = tokenizer.texts_to_sequences(corpus)
sequences
corpus[0], sequences[0]
corpus[8], sequences[8]
# ## .word_index 단어사전
word_index = tokenizer.word_index
len(word_index)
# 다른 문장으로 테스트
tokenizer.texts_to_sequences([
    ['여기', '어때'],
    ['정말', '좋아'],
    ['온달', '장군'],
])
# ## pad_sequences()
max([len(words) for words in corpus])
# 가장 긴 문장의 토큰 길이가 15 <- 이 길이로 패딩 진행.
MAX_SEQ_LEN = 15
padded_seqs = preprocessing.sequence.pad_sequences(
    sequences,
    maxlen=MAX_SEQ_LEN,
    padding='post'
)

padded_seqs
print(corpus[0])
print(sequences[0])
print(padded_seqs[0])
# # Dataset 객체 만들기
# 위에 패딩 처리된 시퀀스 (padded_seqs) 의 벡터 리스트와 감정(label) 리스트 전체를
# TF의 Dataset 객체로 만든다
# 그리고 데이터를 랜덤으로 섞고, train, validation, test 용
# 데이터 셋을 7:2:1 비율로 나눠 데이터셋을 각각 분리합니다
# 그리고 batch 로 묶겠습니다
padded_seqs.shape
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
# Dataset 의 첫번째 값 (data, target)

iter(ds).get_next()
# 랜덤 섞기
ds = ds.shuffle(len(features), seed=42)
# train: val: test = 7:2:1

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

len(ds), train_size, val_size, test_size
train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).batch(20)
# ↑ fit(), evaluate(), predict() 의 입력에 사용될수 있다.
len(train_ds), len(val_ds), len(test_ds)

# (414, 119, 60) <-- 데이터의 개수가 아니라 batch 의 개수!
# # 모델 생성
# - 문장 감정 클래스 분류 모델
# 하이퍼 파라미터 설정
dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(word_index) + 1  # 전체 단어수
# ## Embedding 레이어
# **tf.keras.layers.Embedding**
#
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
#
# ```python
# tf.keras.layers.Embedding(
#     input_dim,
#     output_dim,
#     embeddings_initializer='uniform',
#     embeddings_regularizer=None,
#     embeddings_constraint=None,
#     mask_zero=False,
#     weights=None,
#     lora_rank=None,
#     **kwargs
# )
# ```
# CNN 모델 정의

# ▶ 1. 단어 임베딩 영역
input_layer = Input(shape=(MAX_SEQ_LEN,))

embedding_layer = Embedding(
    VOCAB_SIZE,
    EMB_SIZE,
)(input_layer)

dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

# ▶ 2. 특징 추출 영역 ( feature extraction)
#    합성곱 필터와 연산을 통해 문장의 '특징정보(feature map)을 추출' ...  => flatten
#    Conv1D 를 이용해 크기 3, 4, 5 인 합성곱 필터를 128개씩 사용한 합성곱 계층을 3개 생성
#    합성곱 연산 과정 : 필터 크기에 맞게 입력 데이터 위를 슬라이딩 하게 되는데 이는 3, 4, 5-gram 언어 모델의 개념과 비슷
#    입베딩 벡터를 합성곱으로 받아 GlobalMaxPool1D () 를 이용해 최대 max pooling 연산 수행
#    각각 '병렬'로 진행

conv1 = Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu')(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(filters=128, kernel_size=4, padding='valid', activation='relu')(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(filters=128, kernel_size=5, padding='valid', activation='relu')(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

concat = concatenate([pool1, pool2, pool3])

# ▶ 3. 완전연결계층 (Full connected layer)

hidden = Dense(128, activation='relu')(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)

predictions = Dense(3, activation='softmax')(dropout_hidden)

model = Model(inputs=input_layer, outputs=predictions)

model.summary()





tf.keras.utils.plot_model(model, show_shapes=True)
# # 모델 컴파일 & 학습
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH)
# # 성능평가
loss, accuracy = model.evaluate(test_ds)

print('loss', loss)
print('accuracy', accuracy)
# # 저장하기
# 학습된 모델도 저장하고
# 학습에 사용한 사전정보도 저장해야 한다.
base_path
model_path = os.path.join(base_path, 'out', 'cnn_model.keras')
tokenizer_path = os.path.join(base_path, 'out', 'cnn-model-tokenizer.json')
# 모델 저장
model.save(model_path)
# 사전저장
# tf.keras.preprocessing.text.Tokenizer를 저장하고 다시 불러오는 방법은
# json 또는 pickle 형식으로 할 수 있습니다.

# 📌 주의: Pickle 방식은 Python 버전, TensorFlow 버전에 따라 호환성 문제가 생길 수 있습니다.
# 그래서 json 방식이 더 권장되는 편이다, 특히! 모델 배포 시!
tokenizer_json = tokenizer.to_json()
with open(tokenizer_path, "w", encoding='utf-8') as f:
  f.write(tokenizer_json)
# # 감정분류 모델 사용하기
# ## 모델, 사전 불러오기
model = None
tokenizer = None
# 모델 불러오기
model = keras.models.load_model(model_path)
model.summary()
# 사전 불러오기
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
# JSON 파일 읽기
with open(tokenizer_path, 'r', encoding='utf-8') as f:
  tokenizer_json = f.read()

# Tokenizer 복원
tokenizer = tokenizer_from_json(tokenizer_json)

tokenizer.word_index
# ## 평가
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
ds = ds.shuffle(len(features), seed=42)
test_ds = ds.take(2000).batch(20)
loss, accuracy = model.evaluate(test_ds)

print(loss, accuracy)
# ## 예측
i = 10212
print(corpus[i])
print(padded_seqs[i])
print(labels[i])
# i번째 데이터 감정 예측
picks = [i]

predict = model.predict(padded_seqs[picks])
predict_class = tf.math.argmax(predict, axis=1)

print('감정 예측 점수: ', predict)
print('감정 예측 클래스: ', predict_class.numpy())


# ## 실제 데이터 예측
# 문장 =>
#     Tokenizer 로 인덱싱 =>
#          padding 추가 =>  모델입력
classes = ['일상', '부정적', '긍정적']
MAX_SEQ_LEN
def pred(text):
  corpus = [preprocessing.text.text_to_word_sequence(text)]
  sequences = tokenizer.texts_to_sequences(corpus)
  padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

  predict = model.predict(padded_seqs)
  predict_class = tf.math.argmax(predict, axis=1)

  print(text, '->', classes[predict_class.numpy()[0]])
pred("내가 좋아하는 사람이 날 좋아했으면 좋겠어.")
sentences = [
    "가스불 켜놓고 나온거 같아",
    "어떻게 해야 여자친구가 제 진심을 알아줄까",
    "업무 스트레스 넘 심해",
    "남친이랑 뭐하고 놀지",
    "내가 좋아하는 사람이 날 좋아했으면 좋겠어.",
    "원본 데이터가 이상해요",
    "내가 최대한 잘해줘도 불만이 많아",
]
for sentence in sentences:
  pred(sentence)
