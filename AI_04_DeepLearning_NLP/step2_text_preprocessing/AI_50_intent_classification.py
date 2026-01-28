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
# >>> 출력:
#                              Q                         A  label
# 0                       12시 땡!                하루가 또 가네요.      0
# 1                  1지망 학교 떨어졌어                 위로해 드립니다.      0
# 2                 3박4일 놀러가고 싶다               여행은 언제나 좋죠.      0
# 3              3박4일 정도 놀러가고 싶다               여행은 언제나 좋죠.      0
# 4                      PPL 심하네                눈살이 찌푸려지죠.      0
# ...                        ...                       ...    ...
# 11818           훔쳐보는 것도 눈치 보임.        티가 나니까 눈치가 보이는 거죠!      2
# 11819           훔쳐보는 것도 눈치 보임.             훔쳐보는 거 티나나봐요.      2
# 11820              흑기사 해주는 짝남.                    설렜겠어요.      2
# 11821  힘든 연애 좋은 연애라는게 무슨 차이일까?  잘 헤어질 수 있는 사이 여부인 거 같아요.      2
# 11822               힘들어서 결혼할까봐        도피성 결혼은 하지 않길 바라요.      2
# 
# [11823 rows x 3 columns]

# - 데이터셋 구조
#     - Q (질문),  
#     - A (답변)
#     - label (감정)
#         - **0**: 일상다반사
#         - **1** : 이별(부정)
#         - **2** : 사랑(긍정)

data.shape
# >>> 출력:
# (11823, 3)

data.info()
# >>> 출력:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 11823 entries, 0 to 11822
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype 
# ---  ------  --------------  ----- 
#  0   Q       11823 non-null  object
#  1   A       11823 non-null  object
#  2   label   11823 non-null  int64 
# dtypes: int64(1), object(2)
# memory usage: 277.2+ KB

data['label'].unique()
# >>> 출력:
# array([0, 1, 2])

data['label'].value_counts()
# >>> 출력:
# label
# 0    5290
# 1    3570
# 2    2963
# Name: count, dtype: int64

features = data['Q'].tolist()
labels = data['label'].tolist()

features[0]
# >>> 출력:
# '12시 땡!'

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
# >>> 출력:
# ['12시', '땡']

# 단어 시퀀스 들의 벡터
corpus = [
    preprocessing.text.text_to_word_sequence(text) for text in features
]

corpus
# >>> 출력:
# [['12시', '땡'],
#  ['1지망', '학교', '떨어졌어'],
#  ['3박4일', '놀러가고', '싶다'],
#  ['3박4일', '정도', '놀러가고', '싶다'],
#  ['ppl', '심하네'],
#  ['sd카드', '망가졌어'],
#  ['sd카드', '안돼'],
#  ['sns', '맞팔', '왜', '안하지ㅠㅠ'],
#  ['sns', '시간낭비인', '거', '아는데', '매일', '하는', '중'],
#  ['sns', '시간낭비인데', '자꾸', '보게됨'],
#  ['sns보면', '나만', '빼고', '다', '행복해보여'],
#  ['가끔', '궁금해'],
#  ['가끔', '뭐하는지', '궁금해'],
#  ['가끔은', '혼자인게', '좋다'],
#  ['가난한', '자의', '설움'],

# ## Tokenizer

tokenizer = preprocessing.text.Tokenizer()

tokenizer.fit_on_texts(corpus)

tokenizer.word_index
# >>> 출력:
# {'너무': 1,
#  '좋아하는': 2,
#  '거': 3,
#  '싶어': 4,
#  '같아': 5,
#  '안': 6,
#  '나': 7,
#  '좀': 8,
#  '사람': 9,
#  '내가': 10,
#  '싶다': 11,
#  '어떻게': 12,
#  '썸': 13,
#  '왜': 14,
#  '내': 15,

# ## texts_to_sequence()

sequences = tokenizer.texts_to_sequences(corpus)
sequences
# >>> 출력:
# [[4646, 4647],
#  [4648, 343, 448],
#  [2580, 803, 11],
#  [2580, 804, 803, 11],
#  [4649, 2581],
#  [2582, 4650],
#  [2582, 64],
#  [805, 4651, 14, 4652],
#  [805, 4653, 3, 502, 238, 45, 106],
#  [805, 4654, 23, 4655],
#  [4656, 52, 1128, 28, 1373],
#  [693, 266],
#  [693, 2583, 266],
#  [2584, 4657, 324],
#  [4658, 4659, 4660],

corpus[0], sequences[0]
# >>> 출력:
# (['12시', '땡'], [4646, 4647])

corpus[8], sequences[8]
# >>> 출력:
# (['sns', '시간낭비인', '거', '아는데', '매일', '하는', '중'],
#  [805, 4653, 3, 502, 238, 45, 106])

# ## .word_index 단어사전

word_index = tokenizer.word_index

len(word_index)
# >>> 출력:
# 13398

# 다른 문장으로 테스트
tokenizer.texts_to_sequences([
    ['여기', '어때'],
    ['정말', '좋아'],
    ['온달', '장군'],
])
# >>> 출력:
# [[2147, 108], [38, 42], []]

# ## pad_sequences()

max([len(words) for words in corpus])
# >>> 출력:
# 15

# 가장 긴 문장의 토큰 길이가 15 <- 이 길이로 패딩 진행.
MAX_SEQ_LEN = 15

padded_seqs = preprocessing.sequence.pad_sequences(
    sequences,
    maxlen=MAX_SEQ_LEN,
    padding='post'
)

padded_seqs
# >>> 출력:
# array([[ 4646,  4647,     0, ...,     0,     0,     0],
#        [ 4648,   343,   448, ...,     0,     0,     0],
#        [ 2580,   803,    11, ...,     0,     0,     0],
#        ...,
#        [13395,  2517,    89, ...,     0,     0,     0],
#        [  147,    46,    91, ...,     0,     0,     0],
#        [  555, 13398,     0, ...,     0,     0,     0]], dtype=int32)

print(corpus[0])
print(sequences[0])
print(padded_seqs[0])
# >>> 출력:
# ['12시', '땡']
# [4646, 4647]
# [4646 4647    0    0    0    0    0    0    0    0    0    0    0    0
#     0]

# # Dataset 객체 만들기

# 위에 패딩 처리된 시퀀스 (padded_seqs) 의 벡터 리스트와 감정(label) 리스트 전체를
# TF의 Dataset 객체로 만든다
# 그리고 데이터를 랜덤으로 섞고, train, validation, test 용
# 데이터 셋을 7:2:1 비율로 나눠 데이터셋을 각각 분리합니다
# 그리고 batch 로 묶겠습니다

padded_seqs.shape
# >>> 출력:
# (11823, 15)

ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))

# Dataset 의 첫번째 값 (data, target)

iter(ds).get_next()
# >>> 출력:
# (<tf.Tensor: shape=(15,), dtype=int32, numpy=
#  array([4646, 4647,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0], dtype=int32)>,
#  <tf.Tensor: shape=(), dtype=int32, numpy=0>)

# 랜덤 섞기
ds = ds.shuffle(len(features), seed=42)

# train: val: test = 7:2:1

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

len(ds), train_size, val_size, test_size
# >>> 출력:
# (11823, 8276, 2364, 1182)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).batch(20)

# ↑ fit(), evaluate(), predict() 의 입력에 사용될수 있다.

len(train_ds), len(val_ds), len(test_ds)

# (414, 119, 60) <-- 데이터의 개수가 아니라 batch 의 개수!
# >>> 출력:
# (414, 119, 60)

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
# >>> 출력:
# <IPython.core.display.Image object>

# # 모델 컴파일 & 학습

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCH)
# >>> 출력:
# Epoch 1/5
# [1m414/414[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m17s[0m 34ms/step - accuracy: 0.5042 - loss: 0.9776 - val_accuracy: 0.8228 - val_loss: 0.5026
# Epoch 2/5
# [1m414/414[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 31ms/step - accuracy: 0.8071 - loss: 0.5163 - val_accuracy: 0.9234 - val_loss: 0.2583
# Epoch 3/5
# [1m414/414[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 31ms/step - accuracy: 0.9047 - loss: 0.2913 - val_accuracy: 0.9585 - val_loss: 0.1368
# Epoch 4/5
# [1m414/414[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 29ms/step - accuracy: 0.9488 - loss: 0.1623 - val_accuracy: 0.9759 - val_loss: 0.0887
# Epoch 5/5
# [1m414/414[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 27ms/step - accuracy: 0.9661 - loss: 0.1141 - val_accuracy: 0.9805 - val_loss: 0.0659
# <keras.src.callbacks.history.History at 0x793d3c505c50>

# # 성능평가

loss, accuracy = model.evaluate(test_ds)

print('loss', loss)
print('accuracy', accuracy)
# >>> 출력:
# [1m60/60[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.9738 - loss: 0.0715
# loss 0.05566059425473213
# accuracy 0.9830938577651978

# # 저장하기

# 학습된 모델도 저장하고
# 학습에 사용한 사전정보도 저장해야 한다.

base_path
# >>> 출력:
# '/content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)'

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
# >>> 출력:
# {'너무': 1,
#  '좋아하는': 2,
#  '거': 3,
#  '싶어': 4,
#  '같아': 5,
#  '안': 6,
#  '나': 7,
#  '좀': 8,
#  '사람': 9,
#  '내가': 10,
#  '싶다': 11,
#  '어떻게': 12,
#  '썸': 13,
#  '왜': 14,
#  '내': 15,

# ## 평가

ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
ds = ds.shuffle(len(features), seed=42)
test_ds = ds.take(2000).batch(20)

loss, accuracy = model.evaluate(test_ds)

print(loss, accuracy)
# >>> 출력:
# [1m100/100[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9762 - loss: 0.0809
# 0.07436700165271759 0.9779999852180481

# ## 예측

i = 10212
print(corpus[i])
print(padded_seqs[i])
print(labels[i])
# >>> 출력:
# ['썸', '타는', '여자가', '남사친', '만나러', '간다는데', '뭐라', '해']
# [   13    61   127  4320  1333 12162   856    31     0     0     0     0
#      0     0     0]
# 2

# i번째 데이터 감정 예측
picks = [i]

predict = model.predict(padded_seqs[picks])
predict_class = tf.math.argmax(predict, axis=1)

print('감정 예측 점수: ', predict)
print('감정 예측 클래스: ', predict_class.numpy())
# >>> 출력:
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 226ms/step
# 감정 예측 점수:  [[6.9041369e-07 3.7825400e-06 9.9999547e-01]]
# 감정 예측 클래스:  [2]

# ## 실제 데이터 예측

# 문장 =>
#     Tokenizer 로 인덱싱 =>
#          padding 추가 =>  모델입력

classes = ['일상', '부정적', '긍정적']

MAX_SEQ_LEN
# >>> 출력:
# 15

def pred(text):
  corpus = [preprocessing.text.text_to_word_sequence(text)]
  sequences = tokenizer.texts_to_sequences(corpus)
  padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

  predict = model.predict(padded_seqs)
  predict_class = tf.math.argmax(predict, axis=1)

  print(text, '->', classes[predict_class.numpy()[0]])

pred("내가 좋아하는 사람이 날 좋아했으면 좋겠어.")
# >>> 출력:
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
# 내가 좋아하는 사람이 날 좋아했으면 좋겠어. -> 긍정적

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
# >>> 출력:
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 91ms/step
# 가스불 켜놓고 나온거 같아 -> 일상
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step
# 어떻게 해야 여자친구가 제 진심을 알아줄까 -> 부정적
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 76ms/step
# 업무 스트레스 넘 심해 -> 일상
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 55ms/step
# 남친이랑 뭐하고 놀지 -> 긍정적
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
# 내가 좋아하는 사람이 날 좋아했으면 좋겠어. -> 긍정적
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
# 원본 데이터가 이상해요 -> 부정적
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
# 내가 최대한 잘해줘도 불만이 많아 -> 긍정적
