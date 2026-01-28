"""
순환신경망으로 IMDB 리뷰 분류

개요:
    이 모듈은 AI 자연어처리 수업의 딥러닝/NLP 파트에서 다루는 순환신경망으로 IMDB 리뷰 분류에 관한
    모든 코드와 설명을 정리한 파일함.
    
    원본 Jupyter Notebook: 52 순환신경망으로 IMDB 리뷰 분류.ipynb
"""

# # 순환신경망으로 IMDB 리뷰 분류
# 데이터셋을 두가지 방법으로 변형하여 신경망 입력

# 1. one-hot encoding

# 2. embedding (word emedding)
# # IMDB 리뷰 데이터셋
# - imdb.com 에서 수집한 리뷰를 감상평에 따라 '긍정' 과 '부정' 으로 분류해 놓은 데이터 셋
# - 총 50,000개의 샘플
# - 훈련:테스트 => 25,000개:25,000개
#
# # 토큰(Token)
# 왜?  텍스트 -> 순자데이터로 변환해야 하기 때문!
# He follows the cat.  He loves the cat
#  ↓    ↓      ↓   ↓     ↓   ↓     ↓  ↓
#  10   11   12   13    10  14    12  13

# 하나의 샘플은 여러개의 토큰으로 이루어져 있고, '1개의 토큰'이 '하나의 타임 스텝'에 해당된다.

'''
토큰에 할당되는 정수중 특별한 용도로 예약되어 있는 경우도 있다
ex)
0 - 패딩
1 - 문장의 시작
2 - 어휘 사전에 없는 토큰 (OOV)

'어휘사전' : 훈련세트에서 '고유한 단어'를 뽑아 만든 목록
'''
None
# # 기본 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras

import random
def set_seed(seed = 42):
  tf.keras.utils.set_random_seed(seed)
  tf.config.experimental.enable_op_determinism()

set_seed(42)
base_path = r'/content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)/out'
# # IMDB 데이터 준비
# keras 에선 이미 정수로 인덱싱된 데이터가 포함되어 있다
from tensorflow.keras.datasets import imdb
num_words = 10000   # 단어사전의 크기

(train_input, train_target), (test_input, test_target) = \
      imdb.load_data(num_words=num_words)
# ## Data 확인
print(train_input.shape, test_input.shape)
train_input.dtype

# dtype('O')  <- 데이터 원소가 Python object
# 첫번째 샘플 확인
print(train_input[0])  # <- 각 샘플이 list 다!
#문장의 각 토큰이 이미 정수 인덱싱화 한 결과
# 리뷰 텍스트의 list 길이가 제각각입니다.
# 따라서 고정크기의 2차원 배열에 담기 보다는
# 리뷰마다 별도의 파이썬 리스트로 담아서 메모리를 효율적으로 사용하는 겁니다

# num_words=10000 으로 지정 => 어휘사전 10000개의 단어만 인덱싱됨
#  OOV 는 전부 2로 인덱싱됨.


# 첫번째 샘플 리뷰의 단어개수
len(train_input[0])
len(train_input[1])
# ## Target 확인
train_target[:20]
# 이진 분류 문제.
# 긍정리뷰(1)
# 부정리뷰(0)
# # IMDB 데이터 확인
# 다음과 같은 순서로 데이터 확인은 가능하다

# 1. 단어인덱스 가져오기
#    단어 인덱스를 (단어 -> 인덱스)에서 (인덱스 -> 단어)로 변환
#    인덱스가 3부터 시작하므로 인덱스 오프셋을 적용
# 2. 리뷰를 실제 단어로 변환해서 출력하기

# ★ 주의!
#   직전의 토큰화 하여 정수 인덱싱된 입력데이터(문장)는 단어가 인덱스 3부터 시작
#   지금의 단어인덱스는 0-base 인덱스 사용
# ## 단어인덱스 (단어사전)
word_index = imdb.get_word_index()
print(len(word_index))  # dict, 88584개
print(word_index)
word_index['woods'] # 특정단어 -> 인덱스
# {단어: 인덱스} 에서 {인덱서: 단어} 로 변환
reverse_word_index = {value: key  for key, value in  word_index.items()}

print(reverse_word_index)
reverse_word_index[1408]
# reverse_word_index[0]  # 단어사전에 인덱스 0 없다

reverse_word_index[1]  # 가장 많이 등장한 단어.
reverse_word_index[88584]
# reverse_word_index[88585]  # keyerror!  word index 는 1 ~ 88584
# IMDB 데이터셋의 문장 인덱스는 3부터 시작하므로 인덱스 오프셋을 적용
# IMDB 데이터셋에서 인덱스 오프셋이 3인 이유는 다음과 같은 특별한 토큰들을 예약해 두기 위해서입니다:
# 0: 패딩 토큰 (padding token)
# 1: 시작 토큰 (start token)
# 2: 알 수 없는 단어 (unknown token)

# 정수 인코딩된 문장을 단어로 구성된 문장으로 디코딩
def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
i = 1
decoded_review = decode_review(train_input[i])
print(train_input[i])
print(decoded_review)
print(train_target[1])
# ## 어휘 축소 (수업환경)
num_words = 500
(train_input, train_target), (test_input, test_target) =\
  imdb.load_data(num_words=num_words)
# # Validation data 세트 분리
from sklearn.model_selection import train_test_split

# 주의! 한번만 실행!
train_input, val_input, train_target, val_target = \
  train_test_split(train_input, train_target, test_size=0.2, random_state=42)

train_input.shape, val_input.shape  # ((20000,), (5000,))
# # 패딩
# 적절한 길이의 padding 찾기
# 리뷰 평규 길이,  가장 긴 길이, 가장 짧은 길이 확인
# 문장의 길이들
lengths = np.array([len(x) for x in train_input])
print(np.mean(lengths), np.max(lengths), np.min(lengths), np.median(lengths))
# ↑ [관찰]
# 평균 단어 개수가 239개.  중간값 178 인것으로 보아 이 리뷰 데이터는 한쪽에 치우진 분포일듯 하다

# 히스토그램 으로 표현.
plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()
# [관찰]
# 역시 한쪽으로 치우졌다.  대부분의 리뷰 길이는 300 미만이다.
# 평균이 중간값보다 높은 이유는 오른쪽 끝에 아주 큰 데이터가 있기 때문이다..
# 어떤 리뷰는 1,000개의 단어가 넘어간다!
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 리뷰들의 길이를 100에 맞추어 패딩 해보자.
# (100개의 단어보다 작은 리뷰가 대부분)
# padding 전
len(train_input[0]), len(train_input[5])
max_len = 100
train_seq = pad_sequences(
    sequences=train_input,
    maxlen=max_len  # 100개의 토큰만 사용.
       # 100개의 단어보다 길면 잘래내고 (truncating=)
       # 100개의 단어보다 작으면 padding= 적용
)
train_seq.shape
# padding 후
len(train_seq[0]), len(train_seq[5])
train_seq[0]
# 원래 샘플의 앞/뒤부분 확인

print(train_input[0][:100])
print(train_input[0][-100:])  # <-- 앞부분이 잘린것을 확인할수 있다 truncating=pre
# 왜 앞부분을 자를까?
# 왜 앞부분에 패딩을 넣을까?

# 일반적으로 '뒷부분의 정보가 더 유용' 하리라 기대하기 때문.
# 영화리뷰데이터... 리뷰의 끝에 뭔가 결정적인 소감(결론) 을 말할 가능성이 높다!

# 순환신경망. 입력시퀀스.  먼저 입력된 데이터에 대한 기억은 나중에 입력된 데이터보다 더 희미해지기 때문.
# validation 세트도 길이 max_len 으로 맞추어 보자
val_seq = pad_sequences(val_input, maxlen=max_len)
# # 순환 신경망 만들기 SimpleRNN
#
# **keras.layers.SimpleRNN** 사용
#
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
#
# ```python
# tf.keras.layers.SimpleRNN(
#     units,
#     activation='tanh',
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     recurrent_initializer='orthogonal',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     recurrent_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     recurrent_constraint=None,
#     bias_constraint=None,
#     dropout=0.0,
#     recurrent_dropout=0.0,
#     return_sequences=False,
#     return_state=False,
#     go_backwards=False,
#     stateful=False,
#     unroll=False,
#     seed=None,
#     **kwargs
# )
# ```
#
set_seed(42)
model = keras.Sequential()

model.add(keras.layers.Input(shape=(100, 500)))  # 입력 차원 (100, 500) < 샘플의 시퀀스 길이 100,  그런데 뒤의 500은 뭘까?
model.add(keras.layers.SimpleRNN(units=8))  # 순환층 뉴런개수, 이 개수만큼 출력이 된다.
                                           # 기본 활성화 함수 activation='tanh'

model.add(keras.layers.Dense(1, activation='sigmoid'))


# # one-hot encoding
# 단어 토큰값(정숫값) 은 산술 연산과는 관련 없는 데이터다..
# 즉, 수치형데이터가 아니라 분류형 이다.
# 이를 고유하게 표현하는 방법으로 one-hot encoding 사용

#  정수값의 크기 속성을 없애고 각 정수를 고유하게 표현하기 위해
#   => one-hot encoding 사용
# 첫 샘플 seq
train_seq[0]
# 하나의 토큰을 0 과 1의 배열로 표현
# 그 배열에는 한개만 '1' 이고 나머지는 '0'
# 배열의 크기를 500.  <- imdb.load__data() 에서 500개의 단어만 사용

# one-hot encoding 전
train_seq.shape
train_oh = keras.utils.to_categorical(train_seq)
# one-hot encoding 후
train_oh.shape  # 토큰 하나가 500개의(0, 1)로 표현된다.
train_oh[0][0]  # 첫번째 토큰 10 이 one-hot encoding 변환된 모습.
np.sum(train_oh[0][0])
# 그래서!  Input shape 는 (100, 500) 이다
#          (sequence 길이,  단어표현길이)
# model.add(keras.layers.Input(shape=(100, 500))) 인 것이다!
# validation 세트도 one-hot encoding
val_oh = keras.utils.to_categorical(val_seq)

val_oh.shape  # (5000, 100, 500)
model.summary()
"""
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ simple_rnn (SimpleRNN)          │ (None, 8)              │         4,072 │

     # SimpleRNN 에 전달할 샘플의 크기는 (100, 500) 이지만
     # 이 순환층은 마지막 타임스텝의 은닉상태만 출력합니다
     # 이 때문에 출력크기가 순환층의 뉴런 개수와 동일한 8임을 확인할수 있다

    #  parameter 개수
    #  1) 입력 parameter
            입력토큰은 500차원의 one-hot 인코딩 배열, 순환층뉴런 8개와 fully connected
            500 inputs * 8 units = 4000 개의 입력 weights

    #  2) 은닉상태 parameter
            8 units (은닉상태크기) * 8 units (뉴런의 개수) = 64개의 weights

    #  3) bias
            +8 units 개의 weight

    # 1) + 2) + 3) => 4,072 개의 weights



│ dense (Dense)                   │ (None, 1)              │             9 │

    # (입력 8개 + bias 1) x 출력 1개 => 9개의 weights

 Total params: 4,081 (15.94 KB)
 Trainable params: 4,081 (15.94 KB)
 Non-trainable params: 0 (0.00 B)

"""
None
# # 순환 신경망 학습하기
# ## compile
set_seed(42)

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)

model.compile(optimizer=rmsprop,
              loss='binary_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    os.path.join(base_path, 'best-simplernn-model.keras'),
    save_best_only=True
)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# ## 모델 학습
history1 = model.fit(train_oh, train_target,
                     epochs=100, batch_size=32,
                     validation_data=(val_oh, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb],
                     )
# ## training loss, validation loss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
# # one-hot encoding 의 문제점
# - 입력데이터의 용량이 너무 커진다.
train_seq.shape, train_seq.dtype
train_seq.nbytes  # 용량 확인 (byte)

# 20000 x 100 x 4 (int32) => 8000000
# 반면 one-hot encoding 은?
train_oh.shape,  train_oh.dtype
train_oh.nbytes

# 20000 x 100 x 500 x 8byte => 8,000,000,000 (헉! 약 8G...)
# 결국... 훈련데이터가 커질수록 문제가 된다!

# 그래더! word embedding 이 필요하다!!
# # Word Embedding (단어 임베딩) 사용
# 순환신경망에서 텍스트를 처리할때 자주 사용하는 방법! 단어 임베딩!
# 각 '단어'를 '고정된 크기(개수)의 실수 벡터'로 바꾸어 줌.
# 가령 'cat' 이라는 단어 인베딩 벡터
# [0.2, 0.1, 1.3, 0.8, 0.2 0.4, 1.1, 0.9, 0.2, 0.1]

# 10개의 실수 벡터로 단어 임베딩

# 이런 단어 임베딩으로 만들어진 벡터는 One-Hot encoding 된 벡터보다 훨씬 의미있는 값으로 채워져 있기에
# 자연어처리 에서 더 좋은 성능을 내는 경우가 많습니다.

# ↓물론 keras 에도 단어 임베딩 벡터를 만드는 layer 준비되어 있다
# ### Embedding
# **tf.keras.layers.Embedding**
#
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding?hl=en
#
# ```python
# tf.keras.layers.Embedding(
#     input_dim,   # 어휘 사전의 크기
#     output_dim,  # 임베딩 벡터의 크기
#     embeddings_initializer='uniform',
#     embeddings_regularizer=None,
#     activity_regularizer=None,
#     embeddings_constraint=None,
#     mask_zero=False,
#     input_length=None,
#     **kwargs
# )
# ```
#
# - 이 레이어도 모델에 추가되어 **학습되는 레이어** 다 (즉 parameter 가 있다!)
# - 처음에는 모든 벡터가 랜덤으로 초기화 되지만 훈련을 통해 데이터에서 점점 좋은 단어 임베딩으로 학습되어진다
# 단어 임베딩 의 장점은 걍 '정수 데이터'를 받는 다는 거다
#  즉, One-Hot 인코딩으로 변경된 train_oh 배열이 아니라 train_seq 를 바로 사용할수 있다
#  따라서 !  메모리를 훨씬 효율적으로 사용할수 있다!!

# 앞서 One-Hot Encoding 은
# 샘플차원 하나를 500차원으로 늘렸기 때문에  (100, ) 의 샘플이 (100, 500) 으로 커짐

# 단어 임베딩도 (100, ) 크기의 샘플을, (예를들면->) (100, 20) 과 같이 2차원 배열로 늘립니다.
# 하지만 One-hot 인코딩과 달리 훨씬 작은 크기로도 단어를 잘 표현할수 있다
num_words # 단어사전의 크기
set_seed(42)

model2 = keras.Sequential()
model2.add(keras.layers.Input(shape=(100,)))

# Embedding 레이어를 SimpleRNN 앞에 추가
model2.add(keras.layers.Embedding(
    input_dim=num_words, # 어휘 사전의 크기
                       # 앞서 IMDB 리뷰 데이터셋에서 500개의 단어만 사용하도록 설정되었었다
                        # imdb.load_data(num_words=500)
    output_dim = 16,    # 임베딩 벡터의 크기.  One-hot 인코딩보다 훨~씬 작은 크기의 벡터 사용
))

model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

model2.summary()

"""
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ embedding (Embedding)           │ (None, 100, 16)        │         8,000 │
                    # 입력: (100,) -> 출력 (100, 16)
                    # paramter  개수
                # Embedding 클래스는 '500가지의 각 토큰'을 '크기가 16개인 벡터'로 변경하기 때문에
                #   500 x 16 = 8000 개의 모델 파라미터 가짐

│ simple_rnn_1 (SimpleRNN)        │ (None, 8)              │           200 │
          # 입력 16 x 8 units => 128
          # 은닉 8 x 8 => 64
          # bias         8


│ dense_1 (Dense)                 │ (None, 1)              │             9 │
           #  (입력 8 + bias 1) x 1

 Total params: 8,209 (32.07 KB)
 Trainable params: 8,209 (32.07 KB)
 Non-trainable params: 0 (0.00 B)
"""
None
# ## 모델 생성및 학습
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop,
               loss='binary_crossentropy',
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint(
      os.path.join(base_path, 'best-embedding-model.keras'),
      save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)
history2 = model2.fit(train_seq, train_target,
                      epochs=100, batch_size=64,
                      validation_data=(val_seq, val_target),
                      callbacks=[checkpoint_cb, early_stopping_cb])
# ↑ 출력결과를 보면 One-hot 인코딩일때와 비슷한 성능

# 무엇보다도 학습속도가 향상됨
# 그런데 전체 parameter 개수로 보면.. 음 ..4,081 -> 8,209   (두배나 늘었는데.??)

# RNN '순환층' 의 weight 가 학습속도에 크게 영향을 주는것이다!
# '순환층'의 weight 개수는 훨~~씬 작고, 훈련용 데이터 세트의 크기도 훨씬 줄어듬.
# SimpleRNN (4072 -> 200)
# train loss, validation loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
# ※ 현재... 그닥 성능은 좋지 않다

# num_words = 500 보다 크게 해보면..  10000개 정도 넘겨보자

# padding 의 max_len=100 보다 크게 해보면..

# 좀더 좋은 성능 기대해볼수 있다.  (학습시간 매우~ 길어짐.)
# # 평가
early_stopping_cb.best_epoch  # best_epoch 는 0-base 다
# 검증 세트 확인
loss, accuracy = model2.evaluate(val_seq, val_target, batch_size=64)
print('Val Loss:', loss)
print('Val Accuracy:', accuracy)
# 테스트
test_input.shape, test_target.shape
test_seq = pad_sequences(test_input, maxlen=max_len)

test_seq.shape
# 테스트 세트 확인
loss, accuracy = model2.evaluate(test_seq, test_target, batch_size=64)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
# # 예측하기
print(len(word_index))  # '전체' 사전크기

print(max_len) # 패딩 크기

sample_review = 'The best documentary I have watched in a very long time. This is definitely a must see for everyone. This family and their love and support for each other is truly amazing.'
sample_review
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
# 소문자 변환뒤 토큰화
tokens = word_tokenize(sample_review.lower())
print(tokens)
# 리뷰를 인덱스로 변환
sample_review_index = [word_index.get(word, 0) for word in tokens]

print(sample_review_index)
sample_review_index = [(idx if idx <= max_len else 0) for idx in sample_review_index]

print(tokens)
print(sample_review_index)
# 패딩
sample_review_index_padded = pad_sequences([sample_review_index], maxlen=max_len)

sample_review_index_padded
sample_review_index_padded.shape
# 예측
prediction = model2.predict(sample_review_index_padded)
prediction[0]
# 사전정보 word_index 꼭 필요★

def predict_review(review):
    # 리뷰를 인덱스로 변환합니다.
    sample_review_index = [word_index.get(word, 0) for word in word_tokenize(review.lower())]
    sample_review_index = [(idx if idx <= max_len else 0) for idx in sample_review_index]

    # 패딩
    sample_review_index_padded = pad_sequences([sample_review_index], maxlen=max_len)
    # 예측
    prediction = model2.predict(sample_review_index_padded)
    print(f'Prediction: {prediction[0][0]}')
predict_review("movie is good")
predict_review("movie is bad")
predict_review("This is a very one sided documentary about a woman who is sentenced to a 15 year mandatory prison sentence for dealing drugs. The documentary is made by her family and they want you to believe she doesn't deserve her sentence. However if you read the court reports you will see that she was a drug dealer, she lied to the police, and she was found guily after a trial. The tragic part of this documentary is that the woman left behind a husband and three little girls who will forever be damaged by not having their mother around. Yes you will feel sorry for the children because none of this was their fault. However their drug dealing mom got the sentence she deserved and now she has to spend the rest of her life making it up to these kids. The big lesson of this documentary is if you committ a crime in the US you will get locked up!")
