# # Seq2Seq 모델을 활용한 챗봇 만들기

# ![](https://wikidocs.net/images/page/24996/%EC%9D%B8%EC%BD%94%EB%8D%94%EB%94%94%EC%BD%94%EB%8D%94%EB%AA%A8%EB%8D%B8.PNG)

# '묻고' & '답하는' 형태를 구성.

# Q "나이는 어떻게 되나요?" <- 질문을 인코더 입력
# A "저는 18살입니다" <-  디코더 출력

# '학습' 시 질문에 대한 압축된 정보가 context vector 안에 함축적으로 담겨 decoder 에 전달될거다.

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

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

base_path = r'/content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)'

out_path = os.path.join(base_path, 'out')

# # 데이터 셋 준비

"""
Korpora 라는 자연어 처리 데이터셋.  꽤 괜찮다.
GitHub : https://github.com/ko-nlp/Korpora <- 다양한 데이터셋
공식 : https://pypi.org/project/Korpora/  <- 사용법, 예제등.
"""
None

!pip install Korpora
# >>> 출력:
# Requirement already satisfied: Korpora in /usr/local/lib/python3.11/dist-packages (0.2.0)
# Requirement already satisfied: dataclasses>=0.6 in /usr/local/lib/python3.11/dist-packages (from Korpora) (0.6)
# Requirement already satisfied: numpy>=1.18.0 in /usr/local/lib/python3.11/dist-packages (from Korpora) (2.0.2)
# Requirement already satisfied: tqdm>=4.46.0 in /usr/local/lib/python3.11/dist-packages (from Korpora) (4.67.1)
# Requirement already satisfied: requests>=2.20.0 in /usr/local/lib/python3.11/dist-packages (from Korpora) (2.32.3)
# Requirement already satisfied: xlrd>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from Korpora) (2.0.1)
# Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests>=2.20.0->Korpora) (3.4.2)
# Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests>=2.20.0->Korpora) (3.10)
# Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests>=2.20.0->Korpora) (2.4.0)
# Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests>=2.20.0->Korpora) (2025.4.26)

# - 이 중 챗봇용 데이터셋인 `KoreanChatbotKorpus`를 다운로드 받습니다.
# - `KoreanChatbotKorpus` 데이터셋을 활용하여 챗봇 모델을 학습합니다.
# - text, pair로 구성되어 있습니다.
# - 질의는 **text**, 답변은 **pair**입니다.

# from Korpora import KoreanChatbotKorpus

# corpus = KoreanChatbotKorpus()

# ## 방법b 직접 다운로드

corpus = pd.read_csv('https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv')
corpus.head()
# >>> 출력:
#                  Q            A  label
# 0           12시 땡!   하루가 또 가네요.      0
# 1      1지망 학교 떨어졌어    위로해 드립니다.      0
# 2     3박4일 놀러가고 싶다  여행은 언제나 좋죠.      0
# 3  3박4일 정도 놀러가고 싶다  여행은 언제나 좋죠.      0
# 4          PPL 심하네   눈살이 찌푸려지죠.      0

corpus.shape
# >>> 출력:
# (11823, 3)

corpus = corpus.head(6000)  # 학습 한계상 6000개의 데이터만

# # 데이터 전처리

texts = corpus['Q'].tolist() # 질문들
pairs = corpus['A'].tolist() # 답변들

# ## 특수 문자는 제거
# 한글과 숫자를 제외한 특수문자 제거

import re

def clean_sentence(sentence):
  # 한글, 숫자, 띄어쓰기를 제외한 모든 문자를 제거합니다.
  sentence = re.sub(r'[^0-9ㄱ-ㅎㅏ-ㅣ가-힣 ]', r'', sentence)
  return sentence

clean_sentence("12시 땡^^!???")
# >>> 출력:
# '12시 땡'

clean_sentence('abcd가나다라^^#$#%$12시 좋아~요')
# >>> 출력:
# '가나다라12시 좋아요'

# ## 형태소 분석 Konlpy

!pip install konlpy
# >>> 출력:
# Requirement already satisfied: konlpy in /usr/local/lib/python3.11/dist-packages (0.6.0)
# Requirement already satisfied: JPype1>=0.7.0 in /usr/local/lib/python3.11/dist-packages (from konlpy) (1.5.2)
# Requirement already satisfied: lxml>=4.1.0 in /usr/local/lib/python3.11/dist-packages (from konlpy) (5.4.0)
# Requirement already satisfied: numpy>=1.6 in /usr/local/lib/python3.11/dist-packages (from konlpy) (2.0.2)
# Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from JPype1>=0.7.0->konlpy) (24.2)

from konlpy.tag import Okt

okt = Okt()

# 형태소 변환에 활용할 함수
def process_morph(sentence):
  return ' '.join(okt.morphs(sentence))

process_morph('한글은 홀소리와 닿소리 모두 소리틀을 본떠 만든 음소문자[1]로 한글 맞춤법에서는 닿소리 14개와 홀소리 10개, 모두 24개를 표준으로 삼는다. "나랏말이 중국과 달라" 문제를 느낀 조선의 세종대왕이 한국어는 물론 이웃나라 말까지 나타내도록 1443년 창제하여 1446년 반포하였다. ')
# >>> 출력:
# '한글 은 홀소리 와 닿소리 모두 소 리틀 을 본떠 만든 음소문자 [ 1 ] 로 한글 맞춤법 에서는 닿소리 14 개 와 홀소리 10 개 , 모두 24 개 를 표준 으로 삼는다 . " 나랏말 이 중국 과 달라 " 문제 를 느낀 조선 의 세종대왕 이 한국어 는 물론 이웃 나라 말 까지 나타내도록 1443년 창제 하여 1446년 반포 하였다 .'

# # 데이터셋 구성

# **Seq2Seq** 모델이 학습하기 위한 데이터셋을 구성할 때, 다음과 같이 **3가지 데이터셋**을 구성합니다.
# 
# - `question`: encoder input 데이터셋 (질의 전체)
# - `answer_input`: decoder input 데이터셋 (답변의 시작). **START 토큰**을 문장 처음에 추가 합니다.
# - `answer_output`: decoder output 데이터셋 (답변의 끝). **END 토큰**을 문장 마지막에 추가 합니다.

# ## 문장을 입력받아 3가지 데이터 셋 구성하는 함수

def clean_and_morph(sentence, is_question=True):
  # 한글 문장 전처리
  # 특수 문자 제거
  sentence = clean_sentence(sentence)
  # 형태소 변환
  sentence = process_morph(sentence)

  if is_question:
    return sentence  # Question 인 경우 바로 리턴.
  else:
    # Answer 인 경우
    # '<START> ' 토큰은 decoder input 에,  ' <END>' 토큰은 decoder output 에 추가
    return ('<START> ' + sentence, sentence + ' <END>')

clean_and_morph("12시 땡!")
# >>> 출력:
# '12시 땡'

clean_and_morph("12시 땡!", False)
# >>> 출력:
# ('<START> 12시 땡', '12시 땡 <END>')

# ## `text` 와 `pair` 에 대한 데이터셋 구성

# text 와 pair 에 대한 데이터 셋 구성 함수.

# 매개변수
#  texts=  질의
#  pairs=  답변

def preprocess(texts, pairs):
    questions = []  # encoder 의 입력
    answer_in = []  # decoder 의 입력
    answer_out = [] # decoder 의 출력 (타겟)

    # 질의에 대한 전처리
    for text in texts:
      question = clean_and_morph(text, is_question=True)
      questions.append(question)

    # 답변에 대한 전처리
    for pair in pairs:
      in_, out_ = clean_and_morph(pair, is_question=False)
      answer_in.append(in_)
      answer_out.append(out_)


    return questions, answer_in, answer_out

questions, answer_in, answer_out = preprocess(texts, pairs)

questions[:5]
# >>> 출력:
# ['12시 땡', '1 지망 학교 떨어졌어', '3 박 4일 놀러 가고 싶다', '3 박 4일 정도 놀러 가고 싶다', '심하네']

answer_in[:5]
# >>> 출력:
# ['<START> 하루 가 또 가네요',
#  '<START> 위로 해 드립니다',
#  '<START> 여행 은 언제나 좋죠',
#  '<START> 여행 은 언제나 좋죠',
#  '<START> 눈살 이 찌푸려지죠']

answer_out[:5]
# >>> 출력:
# ['하루 가 또 가네요 <END>',
#  '위로 해 드립니다 <END>',
#  '여행 은 언제나 좋죠 <END>',
#  '여행 은 언제나 좋죠 <END>',
#  '눈살 이 찌푸려지죠 <END>']

# ## 단어사전 구축을 위해 모든 데이터 셋 문장을 합치기

all_sentences = questions + answer_in + answer_out

len(all_sentences)
# >>> 출력:
# 18000

# 모든 문장을 공백으로 join 한뒤 다시 공백으로 쪼개기
a = (' '.join(questions) + ' '.join(answer_in) + ' '.join(answer_out)).split()

len(set(a))
# >>> 출력:
# 7338

# # 토큰화 (Tokenizer)

# ## 토큰의 option을 정의
# 
# - filter는 '(빈 문자열)'로 지정합니다.
# - lower는 False로 지정합니다.
# - oov_token은 '&lt;OOV&gt;'로 지정합니다.

tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')

# ## fit_on_text() 단어사전 구축

tokenizer.fit_on_texts(all_sentences)

tokenizer.word_index
# >>> 출력:
# {'<OOV>': 1,
#  '<START>': 2,
#  '<END>': 3,
#  '이': 4,
#  '거': 5,
#  '을': 6,
#  '예요': 7,
#  '가': 8,
#  '해보세요': 9,
#  '도': 10,
#  '에': 11,
#  '요': 12,
#  '잘': 13,
#  '를': 14,
#  '보세요': 15,

# ## 단어의 개수

len(tokenizer.word_index)
# >>> 출력:
# 7338

VOCAB_SIZE = len(tokenizer.word_index) + 1   # <- +1 은 padding 용  padding  문자 인덱스는 0
VOCAB_SIZE  # -> 나중에 one-hot encoding 시 필요
# >>> 출력:
# 7339

# ## texts_to_sequences()

# 문장을 정수 인덱스 시퀀스로 변환
question_sequence = tokenizer.texts_to_sequences(questions)
answer_in_sequence = tokenizer.texts_to_sequences(answer_in)
answer_out_sequence = tokenizer.texts_to_sequences(answer_out)

print(questions[0])
print(question_sequence[0])
# >>> 출력:
# 12시 땡
# [3958, 5168]

print(answer_in[0])
print(answer_in_sequence[0])
# >>> 출력:
# <START> 하루 가 또 가네요
# [2, 298, 8, 118, 2513]

# ## pad_sequences()

MAX_LENGTH = 30
TRUNCATING = 'post'
PADDING = 'post'

question_padded = pad_sequences(question_sequence, maxlen=MAX_LENGTH, truncating=TRUNCATING, padding=PADDING)
answer_in_padded = pad_sequences(answer_in_sequence, maxlen=MAX_LENGTH, truncating=TRUNCATING, padding=PADDING)
answer_out_padded = pad_sequences(answer_out_sequence, maxlen=MAX_LENGTH, truncating=TRUNCATING, padding=PADDING)

print(question_sequence[0])
print(question_padded[0])
# >>> 출력:
# [3958, 5168]
# [3958 5168    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0]

question_padded.shape
# >>> 출력:
# (6000, 30)

# # to_categorical() 단어별 원핫인코딩

from tensorflow.keras.utils import to_categorical

# to_categorical 사용
answer_in_one_hot = keras.utils.to_categorical(answer_in_padded, num_classes=VOCAB_SIZE)
answer_out_one_hot = keras.utils.to_categorical(answer_out_padded, num_classes=VOCAB_SIZE)

answer_in_one_hot[0].shape, answer_out_one_hot[0].shape
# >>> 출력:
# ((30, 7339), (30, 7339))

# # Tokenizer.index_word[idx]
# index -> word 변환

def convert_index_to_text(indexs, end_token):

  words = []


  # 모든 index 에 대해
  for index in indexs:
    if index == end_token: break  # 끝 단어에는 예측 중지

    # 사전에 존재하는 단어의 경우 단어를 추가
    if index > 0 and tokenizer.index_word[index] is not None:
      words.append(tokenizer.index_word[index])


  return ' '.join(words)

# # 모델 생성

# 1. 인코더 모델 만들고
# 2. 디코더 모델 만들고
# 3. 인코더와 디코더 가 연결된 seq2seq 모델을 만들기

from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

# ## 학습용 인코더(Encoder)

class Encoder(tf.keras.Model):
  def __init__(self, units, vocab_size, embedding_dim):
    super(Encoder, self).__init__()

    self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    self.dropout = Dropout(0.2)
    self.lstm = LSTM(units, return_state=True, dropout=0.2)


  def call(self, inputs):
    x = self.embedding(inputs)
    x = self.dropout(x)
    x, hidden_state, cell_state = self.lstm(x)

    # 'context 벡터' 리턴
    return [hidden_state, cell_state]

# ## 학습용 디코더 (Decoder)

class Decoder(tf.keras.Model):
  def __init__(self, units, vocab_size, embedding_dim):
    super(Decoder, self).__init__()

    self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    self.dropout = Dropout(0.2)
    self.lstm = LSTM(units, return_state=True, return_sequences=True, dropout=0.2)
    self.dense = Dense(vocab_size, activation='softmax')

  def call(self, inputs, initial_state):
    x = self.embedding(inputs)
    x = self.dropout(x)
    x, hidden_state, cell_state = self.lstm(x, initial_state=initial_state)
    x = self.dense(x)

    return x, hidden_state, cell_state  # 디코더에 리턴값에 x 포함!

# ## 하이퍼 파라미터

BATCH_SIZE = 16
EMBEDDING_DIM = 100
TIME_STEPS = MAX_LENGTH

START_TOKEN = tokenizer.word_index['<START>']  # <sos> 토큰
END_TOKEN = tokenizer.word_index['<END>']

UNITS = 128  # LSTM units

VOCAB_SIZE = len(tokenizer.word_index) + 1  # 사전의 크기
DATA_LENGTH = len(questions)  # 몇개의 문장 데이터

SAMPLE_SIZE = 3   # 학습 도중에 몇개만 샘플링해서 예측 결과값 관찰.
NUM_EPOCHS = 20   # fit() 의 epochs= 값이 아니라 전체 epoch 를 반복할 횟수 지정

# ## Seq2Seq 모델
# 
# ![](https://wikidocs.net/images/page/24996/%EC%9D%B8%EC%BD%94%EB%8D%94%EB%94%94%EC%BD%94%EB%8D%94%EB%AA%A8%EB%8D%B8.PNG)

class Seq2Seq(tf.keras.Model):
  def __init__(self, units=UNITS, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM
               , time_steps=TIME_STEPS
               , start_token=START_TOKEN, end_token=END_TOKEN):
    super(Seq2Seq, self).__init__()

    self.start_token = start_token
    self.end_token = end_token
    self.time_steps = time_steps

    self.encoder = Encoder(units, vocab_size, embedding_dim)
    self.decoder = Decoder(units, vocab_size, embedding_dim)

  # inputs= :
  #    학습모드: '인코더의 입력' 과 '디코더의 입력'
  #    예측모드: 인코더의 입력
  # training= : '학습상태' 일때와 '예측(추론)' 할때의 동작구분.
  def call(self, inputs, training=True):

    # '학습상태'
    if training:
      encoder_inputs, decoder_inputs = inputs # '인코더의 입력' 과 '디코더의 입력'
      context_vector = self.encoder(encoder_inputs)
      decoder_outputs, _, _ = self.decoder(inputs=decoder_inputs, initial_state=context_vector)
      return decoder_outputs

    # '예측(추론) 동작'
    else:
      # 첫번째 timestep 의 입력값 두개 준비 : context_vector 와 <sos>
      context_vector = self.encoder(inputs)
      target_seq = tf.constant([[self.start_token]], dtype=tf.float32)

      # ↓ results 는 배열로 만들어
      # 단어 하나하나를 예측하여 배열에 담아줄거다.
      # ※ 주의 call() 안에서는 numpy 배열 사용하면 안된다.
      #    내부적으로 graph 가 형성되어야 하기 때문에 TF 의 TensorArray
      # size=self.time_steps : 한 문장의 최대 길이는 어짜피 MAX_LENGTH 이니
      #    미리 그에 준하는 배열을 준비해둠.
      #    밑에서 results.write() 를 통해 하나하나 추가할거임.
      results = tf.TensorArray(dtype=tf.int32, size=self.time_steps)

      # decoder 에 차례대로 토큰 넣고, 결과 내고, 그 결과를 다음 타임스텝에 넣고... 반복
      for i in tf.range(self.time_steps):
        decoder_output, decoder_hidden, decoder_cell =\
           self.decoder(target_seq, initial_state=context_vector)

        #  decoder_output 은 softmax 결과이기 때문에 argmax() 로 index 변환.
        decoder_output = tf.cast(tf.argmax(decoder_output, axis=-1), dtype=tf.int32)
        # 그 index 를 다시 벡터변환
        decoder_output = tf.reshape(decoder_output, shape=(1, 1))
        # results 배열에 하나하나 쌓아 나아감
        results = results.write(i, decoder_output)

        if decoder_output == self.end_token:  # <eos> 이면 종료
          break

        # 다음 타입스텝에 전달할 입력토큰
        target_seq = decoder_output
        # 다음 타입스텝에 전달할 context 벡터 <- decoder 가 현재 타임스텝에서 출력한 은닉상태와 셀상태로 만들어줌.
        context_vector = [decoder_hidden, decoder_cell]

      return tf.reshape(results.stack(), shape=(1, self.time_steps))

# # 학습

# ## checkpoint callback

checkpoint_path = os.path.join(out_path, 'seq2seq_chatbot_checkpoint.keras')
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_best_only=True,
                             monitor='loss',
                             verbose=1
                            )

# ## 모델 생성 & compile

seq2seq = Seq2Seq(UNITS, VOCAB_SIZE, EMBEDDING_DIM, TIME_STEPS, START_TOKEN, END_TOKEN)

seq2seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ## 예측 함수 준비
# - 학습도중에도 일부 샘플을 예측한 결과를 출력하기 위한 함수

def make_prediction(model, question_inputs):
    # training=False : '예측'모드로 동작. inputs= 에는 인코더 입력값만 전달
    results = model(inputs=question_inputs, training=False)  # call() 에 전달되는 값
    # results 의 인덱스를 문장으로 변환하기 위해 1차원으로 변환
    results = np.asarray(results).reshape(-1)
    return results  # 나중에 이 리턴값을 문장으로 변형할거다.

# ## fit()

for epoch in range(NUM_EPOCHS):
  print(f'🧡processing epoch: {epoch * 10 + 1}...')
  seq2seq.fit([question_padded, answer_in_padded],  # 입력값
              answer_out_one_hot,  # 타겟값
              epochs=10,
              batch_size=BATCH_SIZE,
              callbacks=[checkpoint]
              )

  # 램덤 샘플 번호 추출
  samples = np.random.randint(DATA_LENGTH, size=SAMPLE_SIZE)

  # 질문과 예측값 비교 출력
  for idx in samples:
    question_inputs = question_padded[idx]
    results = make_prediction(seq2seq, np.expand_dims(question_inputs, 0))

    # 변환된 인덱스를 문장으로 변환
    results = convert_index_to_text(results, END_TOKEN)

    print(f'Q: {questions[idx]}')
    print(f'A: {results}')
    print('\n')
# >>> 출력:
# 🧡processing epoch: 1...
# Epoch 1/10
# [1m373/375[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 19ms/step - accuracy: 0.7997 - loss: 2.9013
# Epoch 1: loss improved from inf to 1.68170, saving model to /content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)/out/seq2seq_chatbot_checkpoint.keras
# [1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 19ms/step - accuracy: 0.7999 - loss: 2.8915
# Epoch 2/10
# [1m372/375[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 18ms/step - accuracy: 0.8377 - loss: 1.1282
# Epoch 2: loss improved from 1.68170 to 1.11419, saving model to /content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)/out/seq2seq_chatbot_checkpoint.keras
# [1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 19ms/step - accuracy: 0.8377 - loss: 1.1280
# Epoch 3/10
# [1m373/375[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 18ms/step - accuracy: 0.8402 - loss: 1.0587
# Epoch 3: loss improved from 1.11419 to 1.04949, saving model to /content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)/out/seq2seq_chatbot_checkpoint.keras
# [1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 19ms/step - accuracy: 0.8402 - loss: 1.0586
# Epoch 4/10
# [1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - accuracy: 0.8443 - loss: 1.0097

# # 예측

# ### `make_question` 함수 정의:
# 
# 자연어 '입력'을 받아서 '데이터 전처리' 파이프라인 수행 후 전처리 진행하는 함수를 정의합니다.
# 
# 1. 텍스트 정제
# 1. 형태소 변환
# 1. 학습에 사용된 Tokenizer 로 정수 인덱싱
# 1. 패딩

def make_question(sentence):
  sentence = clean_and_morph(sentence)
  question_sequence = tokenizer.texts_to_sequences([sentence]) # 사전 사용하여 정수 인덱싱
  question_padded = pad_sequences(question_sequence, maxlen=MAX_LENGTH, truncating=TRUNCATING, padding='post')
  return question_padded

make_question('3박4일 놀러가고 싶다')
# >>> 출력:
# array([[ 794, 3684, 3960,  400,  169,   73,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0]], dtype=int32)

make_question('커피 마시고 싶다')
# >>> 출력:
# array([[ 216, 1769,   73,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0]], dtype=int32)

# ## 예측결과를 자연어로 변환하는 함수 정의

def run_chatbot(question):
  question_inputs = make_question(question)

  results = make_prediction(seq2seq, question_inputs)

  results = convert_index_to_text(results, END_TOKEN)

  return results

run_chatbot("안녕하세요")
# >>> 출력:
# '안녕하세요'

# # 챗봇 테스트

while True:
  user_input = input('<< 말을 걸어보세요!\n')
  if user_input == 'q': break
  print('>> 챗봇 응답: {}'.format(run_chatbot(user_input)))
# >>> 출력:
# << 말을 걸어보세요!
# 안녕하세요
# >> 챗봇 응답: 안녕하세요
# << 말을 걸어보세요!
# 오늘은 기분이 좋습니다
# >> 챗봇 응답: 사람 이 크죠
# << 말을 걸어보세요!
# 집에 가고 싶습니다
# >> 챗봇 응답: 하다 보면 늘어요
# << 말을 걸어보세요!
# 오늘은 어디를 갈까요?
# >> 챗봇 응답: 눈 을 깜빡 거려 보세요
# << 말을 걸어보세요!
# q
