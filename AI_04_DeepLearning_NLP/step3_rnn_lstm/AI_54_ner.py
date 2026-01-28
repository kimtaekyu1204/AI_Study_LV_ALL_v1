# # 개체명 인식(NER)
# NER: Named Entity Recognition
# 
# 
# **개체명 인식(NER)** : 문장 내에 포한된 어떤 단어가 인물, 장소, 날짜 등을 의미하는 단어인지 인식하는 것
# 
# **개체명 인식기** : 딥러닝 모델이나 확률모델 등을 이용해서 문장에서 개체명을 인식하는 프로그램

# ex)
#  날짜(date) 와 지역(location) 에 대해 개체인식 할수 있는 모델
#  챗봇은 다음과 같이 문장을 분류 합니다

# 1. 입력문장 : "내일 부산 날씨 알려줘"
# 2. 문장 '의도' 분류 : 날씨요청
# 3. 개체명 인식 결과 :
#    "내일" - 날짜
#    "부산" - 지역

# # BIO 표기법
# Beginning, Inside, Outside
# 
# - 각 토큰마다 태그를 붙이기 위해 사용
# - **B** (Beginning) : 개체명이 시작되는 단어에 'B-개체명' 으로 태깅됨.
# - **I** (Inside) : 'B-개체명' 과 연결되는 단어일때 'I-개체명' 으로 태깅됨.
# - **O** (Outside) : 개체명 이외의 모든 토큰에 태깅됨
# 
# **BIO 표기 예**
# > "오늘부터 샤닐 길동은 삼성 전자에 근무합니다"
# 
# | 토큰 | BIO 태그 |
# |----------|----------|
# | 오늘    | B-Date   |
# | 부터    | O   |
# | 샤닐    | B-Person   |
# | 길동    | I-Person   |
# | 은    | O   |
# | 삼성    | B-Company   |
# | 전자    | I-Company   |
# | 에    | O   |
# | 근무    | O   |
# | 합니다    | O   |
# 
# 
# - 개체명 인식 모델을 학습하기 위해서는 **토큰별로 BIO 태그가 달린 데이터셋** 이 필요!
#     - 영어권에서는 관련된 유명한 데이터셋들이 풍부
#     - 한글의 경우는 BIO 태그 데이터셋을 구하기 힘들다.
#     - 국립국어원 언어정보 나눔터에서 개체명 인식모델을 위해 말뭉치를 공개
#         - https://github.com/machinereading/KoreanNERCorpus
#         - 위에서 ./original/train.txt 파일 다운로드

# ; 으로 시작하는 문장 : 원본 문장
# $ 로 시작하는 문장 : 해당문장에서 NER 처리된 결과
# 다음라인부터는:  토큰번호, 단어토큰, 품사태그, BIO태그
'''
; 한편, AFC챔피언스리그 E조에 속한 포항 역시 대회 8강 진출이 불투명하다 .
$한편, AFC챔피언스리그 <E조:OG>에 속한 포항 역시 대회 8강 진출이 불투명하다 .
1	한편	NNG	O
1	,	SP	O
2	AFC	SL	O
2	챔피언스	NNG	O
2	리그	NNG	O
3	E	SL	B_OG
3	조	NNG	I
3	에	JKB	O
4	속하	VV	O
4	ㄴ	ETM	O
5	포항	NNP	O
6	역시	MAJ	O
7	대회	NNG	O
8	8강	NNG	O
9	진출	NNG	O
9	이	JKS	O
10	불투명	NNG	O
10	하	VV	O
10	다	EC	O
11	.	SF	O
                       <- 다음 문장 전에 줄바꿈 있다.
'''
None

# # 기본 import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras

import random
def set_seed(seed = 42):
  tf.keras.utils.set_random_seed(seed)
  tf.config.experimental.enable_op_determinism()

set_seed(42)

from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split

base_path = r'/content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)/chatbot_data'

# # 말뭉치 데이터 읽어오기

# 학습파일 불러오기

# 1	한편	NNG	O
# 1	,	SP	O
# 2	AFC	SL	O
# ...

# ↓↓↓ 아래와 같이 변경하는 함수를 정의할거다

# [[('1', '한편', 'NNG', 'O'),
#   ('1', ',', 'SP', 'O'),
#   ('2', 'AFC', 'SL', 'O'),
# ...

def read_file(file_name):
  sents = []

  with open(file_name, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, l in enumerate(lines):  # 각 line 을 읽으면서
      if l[0] == ';' and lines[idx + 1][0] == '$':  # 문장시작
        this_sents = []  # sents 에 넣을 문장
      elif l[0] == '$' and lines[idx - 1][0] == ';':
        continue
      elif l[0] == '\n':
        sents.append(this_sents)
      else:
        this_sents.append(tuple(l.split()))  # tuple 의 리스트.

  return sents

corpus = read_file(os.path.join(base_path, 'train.txt'))

corpus[:3]

# 각 문장마다 [(토큰번호, 단어토큰, 품사태그, BIO태그),.....]
# >>> 출력:
# [[('1', '한편', 'NNG', 'O'),
#   ('1', ',', 'SP', 'O'),
#   ('2', 'AFC', 'SL', 'O'),
#   ('2', '챔피언스', 'NNG', 'O'),
#   ('2', '리그', 'NNG', 'O'),
#   ('3', 'E', 'SL', 'B_OG'),
#   ('3', '조', 'NNG', 'I'),
#   ('3', '에', 'JKB', 'O'),
#   ('4', '속하', 'VV', 'O'),
#   ('4', 'ㄴ', 'ETM', 'O'),
#   ('5', '포항', 'NNP', 'O'),
#   ('6', '역시', 'MAJ', 'O'),
#   ('7', '대회', 'NNG', 'O'),
#   ('8', '8강', 'NNG', 'O'),
#   ('9', '진출', 'NNG', 'O'),

# # 학습용 데이터 셋 생성

# 위 말뭉치 데이터에서 '단어' 와 'BIO태그' 만 불러와서 학습용 데이터셋 구성

sentences, tags = [], []

for t in corpus:  # 말뭉치에서 문장 하나씩 -> t (List)
  sentence, bio_tag = [], []
  for w in t:  # 문장에서 단어 토큰 하나씩 -> w (Tuple)
    sentence.append(w[1])   # 단어
    bio_tag.append(w[3])    # BIO 태그

  sentences.append(sentence)
  tags.append(bio_tag)

len(sentences), len(tags)  # 문장의 개수
# >>> 출력:
# (3555, 3555)

# 첫번째 문장 (단어 토큰들)
print(sentences[0])
# >>> 출력:
# ['한편', ',', 'AFC', '챔피언스', '리그', 'E', '조', '에', '속하', 'ㄴ', '포항', '역시', '대회', '8강', '진출', '이', '불투명', '하', '다', '.']

# 첫번째 문장의 bio tag 들
print(tags[0])
# >>> 출력:
# ['O', 'O', 'O', 'O', 'O', 'B_OG', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

print("샘플 크기 : \n", len(sentences))
print("0번째 샘플 문장 시퀀스 : \n", sentences[0])
print("0번째 샘플 bio 태그 : \n", tags[0])
print("샘플 문장 시퀀스 최대 길이 :", max(len(l) for l in sentences))
print("샘플 문장 시퀀스 평균 길이 :", (sum(map(len, sentences))/len(sentences)))
# >>> 출력:
# 샘플 크기 : 
#  3555
# 0번째 샘플 문장 시퀀스 : 
#  ['한편', ',', 'AFC', '챔피언스', '리그', 'E', '조', '에', '속하', 'ㄴ', '포항', '역시', '대회', '8강', '진출', '이', '불투명', '하', '다', '.']
# 0번째 샘플 bio 태그 : 
#  ['O', 'O', 'O', 'O', 'O', 'B_OG', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
# 샘플 문장 시퀀스 최대 길이 : 168
# 샘플 문장 시퀀스 평균 길이 : 34.03909985935302

# # Tokenizer

sent_tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')  # 첫번째 인덱스 'OOV'
sent_tokenizer.fit_on_texts(sentences)

tag_tokenizer = preprocessing.text.Tokenizer(lower=False)
tag_tokenizer.fit_on_texts(tags)

sent_tokenizer.word_index # 사전
# >>> 출력:
# {'OOV': 1,
#  '하': 2,
#  '.': 3,
#  '이': 4,
#  '을': 5,
#  '는': 6,
#  'ㄴ': 7,
#  '다': 8,
#  '의': 9,
#  '에': 10,
#  ',': 11,
#  '를': 12,
#  '은': 13,
#  '았': 14,
#  '고': 15,

tag_tokenizer.word_index
# >>> 출력:
# {'O': 1, 'I': 2, 'B_OG': 3, 'B_PS': 4, 'B_DT': 5, 'B_LC': 6, 'B_TI': 7}

# ## 사전의 크기

vocab_size = len(sent_tokenizer.word_index) + 1
tag_size = len(tag_tokenizer.word_index) + 1

print("BIO 태그 사전 크기 :", tag_size)
print("단어 사전 크기 :", vocab_size)
# >>> 출력:
# BIO 태그 사전 크기 : 8
# 단어 사전 크기 : 13834

tag_tokenizer.index_word

# {1: 'O', 2: 'I', 3: 'B_OG', 4: 'B_PS', 5: 'B_DT', 6: 'B_LC', 7: 'B_TI'}

#  1: 'O',
#  2: 'I',
#  3: 'B_OG',  <- 조직
#  4: 'B_PS',  <- 인물
#  5: 'B_DT',  <- 날짜
#  6: 'B_LC',  <- 지역
#  7: 'B_TI',  <- 시간
#  0: 'PAD'    <-- 이따가 패딩을 위해 추가해 예정
# >>> 출력:
# {1: 'O', 2: 'I', 3: 'B_OG', 4: 'B_PS', 5: 'B_DT', 6: 'B_LC', 7: 'B_TI'}

# ## 문장의 정수 시퀀스

x_train = sent_tokenizer.texts_to_sequences(sentences)
y_train = tag_tokenizer.texts_to_sequences(tags)

# 0번째 문장
print(sentences[0])
print(x_train[0])
print(tags[0])
print(y_train[0])
# >>> 출력:
# ['한편', ',', 'AFC', '챔피언스', '리그', 'E', '조', '에', '속하', 'ㄴ', '포항', '역시', '대회', '8강', '진출', '이', '불투명', '하', '다', '.']
# [183, 11, 4276, 884, 162, 931, 402, 10, 2608, 7, 1516, 608, 145, 1361, 414, 4, 6347, 2, 8, 3]
# ['O', 'O', 'O', 'O', 'O', 'B_OG', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
# [1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

#
index_to_word = sent_tokenizer.index_word  # 시퀀스 인덱스 -> 단어 변환
index_to_ner = tag_tokenizer.index_word    # 시퀀스 인덱스 -> NER (BIO 태그) 변환

index_to_ner  # BIO 태그에 '없는것' 누락되어 있다.
# >>> 출력:
# {1: 'O', 2: 'I', 3: 'B_OG', 4: 'B_PS', 5: 'B_DT', 6: 'B_LC', 7: 'B_TI'}

# 그래서 추가
index_to_ner[0] = 'PAD'

index_to_ner
# >>> 출력:
# {1: 'O',
#  2: 'I',
#  3: 'B_OG',
#  4: 'B_PS',
#  5: 'B_DT',
#  6: 'B_LC',
#  7: 'B_TI',
#  0: 'PAD'}

# # 패딩

max_len = 40    # 위에서 확인한 시퀀스 길이 평균보다 좀 크게 패딩.

x_padded = preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=max_len)
y_padded = preprocessing.sequence.pad_sequences(y_train, padding='post', maxlen=max_len)


x_padded.shape, y_padded.shape
# >>> 출력:
# ((3555, 40), (3555, 40))

x_padded[0]
# >>> 출력:
# array([ 183,   11, 4276,  884,  162,  931,  402,   10, 2608,    7, 1516,
#         608,  145, 1361,  414,    4, 6347,    2,    8,    3,    0,    0,
#           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#           0,    0,    0,    0,    0,    0,    0], dtype=int32)

# # train / test 분리

# train: test = 8:2 분리
# 한번만 실행!
x_train, x_test, y_train, y_test = \
  train_test_split(x_padded, y_padded, test_size=.2, random_state=42)

x_train.shape, x_test.shape
# ((2844, 40), (711, 40))
# >>> 출력:
# ((2844, 40), (711, 40))

y_train.shape, y_test.shape
# ((2844, 40), (711, 40))
# >>> 출력:
# ((2844, 40), (711, 40))

tag_size
# >>> 출력:
# 8

# # 출력 데이터 원-핫 인코딩

# 한번만 실행!
y_train = tf.keras.utils.to_categorical(y_train, num_classes=tag_size)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=tag_size)

print(y_train.shape, y_test.shape)

# (2844, 40, 8) (711, 40, 8)
# >>> 출력:
# (2844, 40, 8) (711, 40, 8)

y_train[0]
# >>> 출력:
# array([[0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 1., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 1., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0.],

print("학습 샘플 시퀀스 형상 : ", x_train.shape)
print("학습 샘플 레이블 형상 : ", y_train.shape)
print("테스트 샘플 시퀀스 형상 : ", x_test.shape)
print("테스트 샘플 레이블 형상 : ", y_test.shape)
# >>> 출력:
# 학습 샘플 시퀀스 형상 :  (2844, 40)
# 학습 샘플 레이블 형상 :  (2844, 40, 8)
# 테스트 샘플 시퀀스 형상 :  (711, 40)
# 테스트 샘플 레이블 형상 :  (711, 40, 8)

# # 모델

vocab_size, max_len
# >>> 출력:
# (13834, 40)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=30))
model.add(Bidirectional(LSTM(200, return_sequences=True, dropout=0.5)))
model.add(TimeDistributed(Dense(tag_size, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])

# # 학습

model.fit(x_train, y_train, batch_size=128, epochs=10)
print("평가결과: ", model.evaluate(x_test, y_test))
# >>> 출력:
# Epoch 1/10
# [1m23/23[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 26ms/step - accuracy: 0.7073 - loss: 1.0902
# Epoch 2/10
# [1m23/23[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - accuracy: 0.9077 - loss: 0.3371
# Epoch 3/10
# [1m23/23[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - accuracy: 0.9298 - loss: 0.2154
# Epoch 4/10
# [1m23/23[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - accuracy: 0.9444 - loss: 0.1628
# Epoch 5/10
# [1m23/23[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - accuracy: 0.9534 - loss: 0.1340
# Epoch 6/10
# [1m23/23[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - accuracy: 0.9622 - loss: 0.1132
# Epoch 7/10
# [1m23/23[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - accuracy: 0.9698 - loss: 0.0923
# Epoch 8/10

# 학습평가를 보면 93% 로 매우 높게 나옴.  그러나! 학습 데이터 성분을 살펴보면 문제가 있다
# BIO 태그의 경우 실제 의미있는 태그(B-I) 보다는 의미없는 O 태그가 대부분을 차지하고 있어서,
# 우리가 원하는 성능과 무관하게 높은 점수로 계산이 나온다
# 예를들어 10개의 예측 결과중 실제로 정확하게 예측된 B-I 는 한개도 없지만
# O태그는 정답과 비교했을때 9개가 동일했다 해도 accuracy 는 90% 인것이다

# 따라서 개체명 인식에 사용되는 성능평가는 'F1 스코어를' 사용해야 한다 (클래스별 분석)

# # F1 score

# 시퀀스를 NER 태그로 변환
def sequences_to_tag(sequences):  # <- sequences: 예측한 NER
  result = []

  for sequence in sequences:
    temp = []

    for pred in sequence:
      pred_index = np.argmax(pred)
      temp.append(index_to_ner[pred_index].replace("PAD", "O"))

    result.append(temp)

  return result

# test 세트로 NER 예측
y_predicted = model.predict(x_test) # (711,40) => model => (711, 40, 8)

pred_tags = sequences_to_tag(y_predicted)  # 예측된 NER
test_tags = sequences_to_tag(y_test)  # 실제 NER
# >>> 출력:
# [1m23/23[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step

# 결과 비교
print(pred_tags[0])
print(test_tags[0])
# >>> 출력:
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B_LC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B_OG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

# F1 스코어 계산을 위해 사용

# 그런데. 일반적인 숫자값들이 아니라,. 문자열 시퀀스인데, 어케 계산

# 사전에 pip install seqeval  설치
#    sequence label 들의 evaluation 모듈
#    https://github.com/chakki-works/seqeval

!pip install seqeval
# >>> 출력:
# Collecting seqeval
#   Downloading seqeval-1.2.2.tar.gz (43 kB)
# [?25l     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/43.6 kB[0m [31m?[0m eta [36m-:--:--[0m[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m43.6/43.6 kB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m
# [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
# Requirement already satisfied: numpy>=1.14.0 in /usr/local/lib/python3.11/dist-packages (from seqeval) (2.0.2)
# Requirement already satisfied: scikit-learn>=0.21.3 in /usr/local/lib/python3.11/dist-packages (from seqeval) (1.6.1)
# Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn>=0.21.3->seqeval) (1.15.2)
# Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn>=0.21.3->seqeval) (1.4.2)
# Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn>=0.21.3->seqeval) (3.6.0)
# Building wheels for collected packages: seqeval
#   Building wheel for seqeval (setup.py) ... [?25l[?25hdone
#   Created wheel for seqeval: filename=seqeval-1.2.2-py3-none-any.whl size=16162 sha256=bfcd80bf533961ef3f95786426a381d1eda79261815a54d16538ee8e6436f028
#   Stored in directory: /root/.cache/pip/wheels/bc/92/f0/243288f899c2eacdfa8c5f9aede4c71a9bad0ee26a01dc5ead
# Successfully built seqeval
# Installing collected packages: seqeval

from seqeval.metrics import f1_score, classification_report

print(classification_report(test_tags, pred_tags))
# >>> 출력:
# /usr/local/lib/python3.11/dist-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: B_OG seems not to be NE tag.
#   warnings.warn('{} seems not to be NE tag.'.format(chunk))
# /usr/local/lib/python3.11/dist-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: B_LC seems not to be NE tag.
#   warnings.warn('{} seems not to be NE tag.'.format(chunk))
# /usr/local/lib/python3.11/dist-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: B_PS seems not to be NE tag.
#   warnings.warn('{} seems not to be NE tag.'.format(chunk))
# /usr/local/lib/python3.11/dist-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: B_DT seems not to be NE tag.
#   warnings.warn('{} seems not to be NE tag.'.format(chunk))
# /usr/local/lib/python3.11/dist-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: B_TI seems not to be NE tag.
#   warnings.warn('{} seems not to be NE tag.'.format(chunk))
#               precision    recall  f1-score   support
# 
#            _       0.50      0.54      0.52       611
#          _DT       0.83      0.82      0.82       319
#          _LC       0.74      0.56      0.64       321
#          _OG       0.62      0.49      0.55       471
#          _PS       0.78      0.33      0.46       378
#          _TI       1.00      0.34      0.51        68
# 
#    micro avg       0.65      0.53      0.58      2168
#    macro avg       0.75      0.51      0.58      2168
# weighted avg       0.68      0.53      0.58      2168

print("F1-score: {:.1f}".format(f1_score(test_tags, pred_tags)))
# >>> 출력:
# F1-score: 0.6
# /usr/local/lib/python3.11/dist-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: B_OG seems not to be NE tag.
#   warnings.warn('{} seems not to be NE tag.'.format(chunk))
# /usr/local/lib/python3.11/dist-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: B_LC seems not to be NE tag.
#   warnings.warn('{} seems not to be NE tag.'.format(chunk))
# /usr/local/lib/python3.11/dist-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: B_PS seems not to be NE tag.
#   warnings.warn('{} seems not to be NE tag.'.format(chunk))
# /usr/local/lib/python3.11/dist-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: B_DT seems not to be NE tag.
#   warnings.warn('{} seems not to be NE tag.'.format(chunk))
# /usr/local/lib/python3.11/dist-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: B_TI seems not to be NE tag.
#   warnings.warn('{} seems not to be NE tag.'.format(chunk))

# # 문장에 대한 NER 예측

def pred_ner(sentence):
  word_to_index = sent_tokenizer.word_index
  new_sentence = sentence.split()

  new_x = []
  for w in new_sentence:
    new_x.append(word_to_index.get(w, 1))

  print('문장의 시퀀스 : ', new_x)

  new_padded_seqs = preprocessing.sequence.pad_sequences([new_x], padding='post', value=0, maxlen=max_len)

  # NER 예측
  p = model.predict(np.array([new_padded_seqs[0]]))
  p = np.argmax(p, axis=-1)  # 예측된 NER 인덱스 값 추출

  print("{:10} {:5}".format("단어", "예측된 NER"))
  print("-" * 50)

  for w, pred in zip(new_sentence, p[0]):
    print("{:10} {:5}".format(w, index_to_ner[pred]))

pred_ner('삼성전자 출시 스마트폰 오늘 애플 도전장 내밀다')
# >>> 출력:
# 문장의 시퀀스 :  [531, 307, 1476, 286, 1507, 6766, 1]
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 222ms/step
# 단어         예측된 NER
# --------------------------------------------------
# 삼성전자       B_OG 
# 출시         O    
# 스마트폰       O    
# 오늘         B_DT 
# 애플         B_OG 
# 도전장        I    
# 내밀다        I

pred_ner('오늘 10시에 잠실야구장에서 LG와 두산 맞대결한다')
# >>> 출력:
# 문장의 시퀀스 :  [286, 1, 1, 1, 430, 1]
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
# 단어         예측된 NER
# --------------------------------------------------
# 오늘         B_DT 
# 10시에       O    
# 잠실야구장에서    I    
# LG와        I    
# 두산         B_OG 
# 맞대결한다      I
