"""
개체명인식(NER)

개요:
    이 모듈은 AI 자연어처리 수업의 딥러닝/NLP 파트에서 다루는 개체명인식(NER)에 관한
    모든 코드와 설명을 정리한 파일함.
    
    원본 Jupyter Notebook: 54 개체명인식(NER).ipynb
"""

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
# 첫번째 문장 (단어 토큰들)
print(sentences[0])
# 첫번째 문장의 bio tag 들
print(tags[0])
print("샘플 크기 : \n", len(sentences))
print("0번째 샘플 문장 시퀀스 : \n", sentences[0])
print("0번째 샘플 bio 태그 : \n", tags[0])
print("샘플 문장 시퀀스 최대 길이 :", max(len(l) for l in sentences))
print("샘플 문장 시퀀스 평균 길이 :", (sum(map(len, sentences))/len(sentences)))
# # Tokenizer
sent_tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')  # 첫번째 인덱스 'OOV'
sent_tokenizer.fit_on_texts(sentences)

tag_tokenizer = preprocessing.text.Tokenizer(lower=False)
tag_tokenizer.fit_on_texts(tags)

sent_tokenizer.word_index # 사전
tag_tokenizer.word_index
# ## 사전의 크기
vocab_size = len(sent_tokenizer.word_index) + 1
tag_size = len(tag_tokenizer.word_index) + 1

print("BIO 태그 사전 크기 :", tag_size)
print("단어 사전 크기 :", vocab_size)
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
# ## 문장의 정수 시퀀스
x_train = sent_tokenizer.texts_to_sequences(sentences)
y_train = tag_tokenizer.texts_to_sequences(tags)

# 0번째 문장
print(sentences[0])
print(x_train[0])
print(tags[0])
print(y_train[0])
#
index_to_word = sent_tokenizer.index_word  # 시퀀스 인덱스 -> 단어 변환
index_to_ner = tag_tokenizer.index_word    # 시퀀스 인덱스 -> NER (BIO 태그) 변환
index_to_ner  # BIO 태그에 '없는것' 누락되어 있다.
# 그래서 추가
index_to_ner[0] = 'PAD'

index_to_ner
# # 패딩
max_len = 40    # 위에서 확인한 시퀀스 길이 평균보다 좀 크게 패딩.

x_padded = preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=max_len)
y_padded = preprocessing.sequence.pad_sequences(y_train, padding='post', maxlen=max_len)


x_padded.shape, y_padded.shape
x_padded[0]
# # train / test 분리
# train: test = 8:2 분리
# 한번만 실행!
x_train, x_test, y_train, y_test = \
  train_test_split(x_padded, y_padded, test_size=.2, random_state=42)
x_train.shape, x_test.shape
# ((2844, 40), (711, 40))
y_train.shape, y_test.shape
# ((2844, 40), (711, 40))
tag_size
# # 출력 데이터 원-핫 인코딩
# 한번만 실행!
y_train = tf.keras.utils.to_categorical(y_train, num_classes=tag_size)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=tag_size)

print(y_train.shape, y_test.shape)

# (2844, 40, 8) (711, 40, 8)
y_train[0]
print("학습 샘플 시퀀스 형상 : ", x_train.shape)
print("학습 샘플 레이블 형상 : ", y_train.shape)
print("테스트 샘플 시퀀스 형상 : ", x_test.shape)
print("테스트 샘플 레이블 형상 : ", y_test.shape)
# # 모델
vocab_size, max_len
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
# 결과 비교
print(pred_tags[0])
print(test_tags[0])
# F1 스코어 계산을 위해 사용

# 그런데. 일반적인 숫자값들이 아니라,. 문자열 시퀀스인데, 어케 계산

# 사전에 pip install seqeval  설치
#    sequence label 들의 evaluation 모듈
#    https://github.com/chakki-works/seqeval
!pip install seqeval
from seqeval.metrics import f1_score, classification_report
print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1f}".format(f1_score(test_tags, pred_tags)))
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
pred_ner('오늘 10시에 잠실야구장에서 LG와 두산 맞대결한다')
