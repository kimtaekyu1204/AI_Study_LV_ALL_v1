# # ■ 자연어 처리 (NLP)
# Natural Language Processing 이란
# 
# **자연어 처리(NLP, Natural Language Processing)**는 컴퓨터가 인간의 언어를 이해하고, 해석하고, 생성하는 기술을 말합니다. NLP는 언어학과 컴퓨터 과학의 융합 분야로, 언어를 처리하고 분석하는 다양한 작업을 포함합니다. 그 목적은 사람과 컴퓨터 간의 상호작용을 원활하게 만들어 주는 것입니다.
# 
# 자연어 처리의 주요 작업은 다음과 같습니다:
# 
# 1. **형태소 분석 (Morphological Analysis)**: 문장에서 단어를 나누고, 각 단어의 형태를 분석합니다. 예를 들어, 한국어에서 "사랑한다"라는 단어는 "사랑"과 "한다"로 나누어집니다.
# 
# 1. **구문 분석 (Syntactic Parsing)**: 문장의 구조를 분석하고, 문법적 관계를 파악합니다. 예를 들어, "나는 사과를 먹는다"라는 문장에서 주어(나), 목적어(사과), 동사(먹는다) 관계를 식별합니다.
# 
# 1. **의미 분석 (Semantic Analysis)**: 문장이 전달하려는 의미를 추론합니다. 예를 들어, "배가 고프다"는 단순히 '배'라는 단어가 아니라 '배'가 '허기짐'을 의미하는지를 이해합니다.
# 
# 1. **감정 분석 (Sentiment Analysis)**: 텍스트에 포함된 감정을 분석하여 긍정적, 부정적, 중립적 감정을 분류합니다. 예를 들어, "이 영화는 정말 재미있다"는 긍정적인 감정을, "이 영화는 지루하다"는 부정적인 감정을 나타냅니다.
# 
# 1. **기계 번역 (Machine Translation)**: 한 언어에서 다른 언어로 번역하는 작업입니다. 예를 들어, 영어에서 한국어로 번역하는 시스템을 개발하는 것입니다.
# 
# 1. **질의 응답 (Question Answering)**: 사용자가 입력한 질문에 대해 적절한 답을 제공하는 시스템입니다. 예를 들어, "파리의 수도는 어디인가요?"라는 질문에 대해 "파리는 프랑스의 수도입니다."라는 답을 제공하는 것입니다.
# 
# 1. **텍스트 생성 (Text Generation)**: 주어진 주제나 조건에 맞춰 텍스트를 자동으로 생성하는 작업입니다. 예를 들어, 사용자가 제공한 주제에 대해 자동으로 기사를 작성하는 것이 이에 해당합니다.
# 
# 자연어 처리는 다양한 분야에서 활용됩니다. 예를 들어, 검색 엔진, 음성 인식 시스템, 챗봇, 자동 번역 시스템 등에서 사용되며, 최근에는 대규모 언어 모델(예: GPT) 등의 발전으로 자연어 처리 기술이 더욱 진화하고 있습니다.

# # ■ 코퍼스 (corpus, 말뭉치)
# 
# 자연어 분야의 연구를 위해 '특정한 목적'을 가지고 언어의 표본을 추출한 집합.
# 
# 이를 사용하여 텍스트 데이터 분석및 저연어처리 (NLP) 에서 모델을 학습시키게 되는 데이터로 활용된다.
# 
# ![](https://mblogthumb-phinf.pstatic.net/MjAyMjA1MDRfMTk4/MDAxNjUxNjI4NjQ4OTIx.ewwjMz43c_bhhXPLSv022-JqPZlMZmmjBh_P2_vSTfEg.y7dAYWl4kPg4Pm-30UIjKZoyiPriL6I0A9E8mjU83xQg.JPEG.mcstkorea/3._%EB%A7%90%EB%AD%89%EC%B9%98_%EC%A0%95%EC%9D%98.jpg?type=w800)
# 
# 
# 참조 사이트
# 국립국어원 말뭉치 검색
# https://kcorpus.korean.go.kr/
# ![](https://img.sbs.co.kr/newsnet/test/sdf/20210927/1422164704/28394_1617005927.jpg)

# # ■ 설치 패키지

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import os
import re

# !pip install gensim
# import gensim
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# konlpy : 한국여형태소분석기 (+품사태깅)
!pip install konlpy
import konlpy
# >>> 출력:
# [nltk_data] Downloading package punkt to /root/nltk_data...
# [nltk_data]   Unzipping tokenizers/punkt.zip.
# [nltk_data] Downloading package averaged_perceptron_tagger to
# [nltk_data]     /root/nltk_data...
# [nltk_data]   Unzipping taggers/averaged_perceptron_tagger.zip.
# Collecting konlpy
#   Downloading konlpy-0.6.0-py2.py3-none-any.whl.metadata (1.9 kB)
# Collecting JPype1>=0.7.0 (from konlpy)
#   Downloading jpype1-1.5.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)
# Requirement already satisfied: lxml>=4.1.0 in /usr/local/lib/python3.11/dist-packages (from konlpy) (5.3.2)
# Requirement already satisfied: numpy>=1.6 in /usr/local/lib/python3.11/dist-packages (from konlpy) (2.0.2)
# Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from JPype1>=0.7.0->konlpy) (24.2)
# Downloading konlpy-0.6.0-py2.py3-none-any.whl (19.4 MB)
# [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m19.4/19.4 MB[0m [31m59.3 MB/s[0m eta [36m0:00:00[0m
# [?25hDownloading jpype1-1.5.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (494 kB)
# [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m494.1/494.1 kB[0m [31m35.3 MB/s[0m eta [36m0:00:00[0m
# [?25hInstalling collected packages: JPype1, konlpy
# Successfully installed JPype1-1.5.2 konlpy-0.6.0

# # ■ 텍스트 전처리 (Text Preprocessing)
# 
# **텍스트 전처리**란?
# 주어진 문제를 해결하기 위해, 목적에 맞게 텍스트 데이터에 사전에 처리하는 작업입니다.
# 
# 요리에 비유하자면, 아무리 조리방법과 장비가 좋다하더라도, 재료가 나쁘면, 혹은 재료가 제대로 손질되어 있지 않으면 좋은 요리가 나오기 힘듭니다.
# 
# 텍스트 데이터 또한 적절한 전처리를 하지 않으면 앞으로 배울 자연어 처리 기법들이 제대로 동작하지 않습니다.

# 코퍼스 데이터는 사용하고자 하는 용도에 맞게 다양한 전처리 작업을 거치게 된다.
# 
# 예를 들면.
# - 토큰화(tokenization)
# - 정제(cleaning)
# - 정규화(normalization)

# # 1. 토큰화 (Tokenization)
# 
# - 주어진 코퍼스(corpus)에서 토큰(token)이라 불리는 단위로 나누는 작업을 토큰화(tokenization)라고 합니다
# 
# - 토큰의 단위는 상황에 따라 다르나, 보통 의미가 있는 단위로 토큰을 정의합니다.

# ## 단어 토큰화(Word Tokenization)
# 토큰의 기준을 단어(word)로 하는 경우, 단어 토큰화(word tokenization)라고 합니다. 다만, 여기서 단어(word)는 단어 단위 외에도 단어구, 의미를 갖는 문자열로도 간주되기도 합니다.
# 예를 들어보겠습니다. 아래의 입력으로부터 구두점(punctuation)과 같은 문자는 제외시키는 간단한 단어 토큰화 작업을 해봅시다. 구두점이란 마침표(.), 컴마(,), 물음표(?), 세미콜론(;), 느낌표(!) 등과 같은 기호를 말합니다.
# 
# 
# 입력: **Time is an illusion. Lunchtime double so!**
# 
# 
# 이러한 입력으로부터 구두점을 제외시킨 토큰화 작업의 결과는 다음과 같습니다.
# 
# 
# 출력 : "Time", "is", "an", "illustion", "Lunchtime", "double", "so"

# ## 다양한 토크나이저

sentence =  "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
sentence
# >>> 출력:
# "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."

"""
Don't 를 어떻게 토큰화

Don t
Dont

Do n't

John's
John s
...
"""
None

nltk.download('punkt_tab')
# >>> 출력:
# [nltk_data] Downloading package punkt_tab to /root/nltk_data...
# [nltk_data]   Unzipping tokenizers/punkt_tab.zip.
# True

from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer

print(sentence)
print(word_tokenize(sentence))
# >>> 출력:
# Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop.
# ['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']

print(WordPunctTokenizer().tokenize(sentence))
# >>> 출력:
# ['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']

# keras 의 text_to_word_seqeunce

from tensorflow.keras.preprocessing.text import text_to_word_sequence

print(text_to_word_sequence(sentence))
# >>> 출력:
# ["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']

# ## 토큰화에서 고려사항

# ### 구두점과 특수문자
# - 무조건 제거하는게 능사는 아니다

# 예]
#   단어 자체에 구두점 : m.p.h 나 Ph.D 나 AT&T
#   $, / 같은 특수문자 : $45.55 나 01/02/06
#   숫자 사이의 컴마: 123,456,789

# ### 띄어쓰기
# 줄임말과 단어 내의 띄어쓰기가 있는 경우

# 예]
#  what're는 what are의 줄임말, we're는 we are의 줄임말

#  "New York" 이나 "rock 'n' roll" 는 띄어쓰기가 있지만 한 단어로 토큰화 할  필요도 있다.

# ### 표준 토큰화 예제

# Penn Treebank Tokenization의 규칙

# 규칙 1. 하이픈으로 구성된 단어는 하나로 유지한다.
# 규칙 2. doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리해준다.

text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
text
# >>> 출력:
# "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."

from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
print(tokenizer.tokenize(text))
# >>> 출력:
# ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']

# ## 문장 토큰화(Sentence Tokenization)
# - 토큰의 단위가 문장(sentence) 일 경우.

#  !, ? 는 문장구분을 위한 명확한 구분자 (boundary) 역할을 하지만
# 마침표(.) 는 그렇지 않다

# EX1)
# IP 192.168.56.31 서버에 들어가서 로그 파일 저장해서 aaa@gmail.com로 결과 좀 보내줘. 그 후 점심 먹으러 가자.

# EX2)
# Since I'm actively looking for Ph.D. students, I get the same question a dozen times every year.

# ### NLTK sent_tokenize()

from nltk.tokenize import sent_tokenize

text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
text
# >>> 출력:
# 'His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near.'

sent_tokenize(text)
# >>> 출력:
# ['His barber kept his word.',
#  'But keeping such a huge secret to himself was driving him crazy.',
#  'Finally, the barber went up a mountain and almost to the edge of a cliff.',
#  'He dug a hole in the midst of some reeds.',
#  'He looked about, to make sure no one was near.']

text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
text
# >>> 출력:
# 'I am actively looking for Ph.D. students. and you are a Ph.D student.'

sent_tokenize(text)
# >>> 출력:
# ['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']

# ### kss
# 
# 한국어에 대한 문장 토큰화 도구 또한 존재합니다. 한국어의 경우에는 박상길님이 개발한 KSS(Korean Sentence Splitter)를 추천합니다. 다음과 같이 KSS를 설치합니다.

!pip install kss
# >>> 출력:
# Collecting kss
#   Downloading kss-6.0.4.tar.gz (1.1 MB)
# [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m9.5 MB/s[0m eta [36m0:00:00[0m
# [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
# Collecting emoji==1.2.0 (from kss)
#   Downloading emoji-1.2.0-py3-none-any.whl.metadata (4.3 kB)
# Collecting pecab (from kss)
#   Downloading pecab-1.0.8.tar.gz (26.4 MB)
# [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m26.4/26.4 MB[0m [31m46.9 MB/s[0m eta [36m0:00:00[0m
# [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
# Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from kss) (3.4.2)
# Collecting jamo (from kss)
#   Downloading jamo-0.4.1-py3-none-any.whl.metadata (2.3 kB)
# Collecting hangul-jamo (from kss)
#   Downloading hangul_jamo-1.0.1-py3-none-any.whl.metadata (899 bytes)

import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
text
# >>> 출력:
# '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'

kss.split_sentences(text)
# >>> 출력:
# WARNING:root:Oh! You have mecab in your environment. Kss will take this as a backend! :D
# ['딥 러닝 자연어 처리가 재미있기는 합니다.', '그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다.', '이제 해보면 알걸요?']

# ## 한국어 토큰화는 (특히) 어렵다

# 교착어: 조사, 어미 등을 붙여서 말을 만드는 언어

# ### 한국어는 교착어이기 때문에
# 
# 교착어의 의 특성
# 1. 조사
# 1. 띄어쓰기 단위
# 1. 형태소 (morpheme)

# 영어 '그' he/him 정도 뿐일텐데
#  한국어는  '그가', '그에게', '그를',  '그와',  '그는' ... 다양한 조사가 붙는다.

# ★ 대부분의 한국어 NLP 에선 조사는 분리해줄 필요가 있다.

# 한국어 토큰화 --> '형태소' 란 개념 반드시 이해
# 형태소 : 뜻을 가진 가장 작은 말의 단위

# 문장 : 에디가 책을 읽었다

# ▶이 문장을 띄어쓰기 단위 토큰화를 수행한다면
#   ['에디가', '책을', '읽었다']

# ▶ 형태소 단위로 분해하면
#   자립 형태소 : 에디, 책
#   의존 형태소 : -가, -을, 읽-, -었, -다

# ### 한국어의 띄어쓰기는 영어보다 잘 지켜지지 않기 때문

# EX1) 제가이렇게띄어쓰기를전혀하지않고글을썼다고하더라도글을이해할수있습니다.

# EX2) Tobeornottobethatisthequestion

# ## 품사 태깅 (Part-Of-Speech tagging)

# 'fly'  명사 - 파리
# 'fly'  동사 - 날다

# '못'   명사 - 못 박는다.
# '못'   부사 - 못한다.

# ## NLTK 와 KoNLPy 를 이용한 토큰화

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nltk.download('averaged_perceptron_tagger_eng')
# >>> 출력:
# [nltk_data] Downloading package averaged_perceptron_tagger_eng to
# [nltk_data]     /root/nltk_data...
# [nltk_data]   Unzipping taggers/averaged_perceptron_tagger_eng.zip.
# True

text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
text
# >>> 출력:
# 'I am actively looking for Ph.D. students. and you are a Ph.D. student.'

tokenized_sentence = word_tokenize(text)
tokenized_sentence
# >>> 출력:
# ['I',
#  'am',
#  'actively',
#  'looking',
#  'for',
#  'Ph.D.',
#  'students',
#  '.',
#  'and',
#  'you',
#  'are',
#  'a',
#  'Ph.D.',
#  'student',
#  '.']

print('단어토큰화:', tokenized_sentence)
print('품사태깅:', pos_tag(tokenized_sentence))
# >>> 출력:
# 단어토큰화: ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
# 품사태깅: [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]

# KoNLPy  코엔엘파이 http://konlpy.org/ko/latest/

# 코엔엘파이를 통해서 사용할 수 있는 형태소 분석기로 Okt(Open Korea Text), 메캅(Mecab),
# 코모란(Komoran), 한나눔(Hannanum), 꼬꼬마(Kkma)가 있습니다.

from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

text = "열심히 코딩한 당신, 연휴에는 여행을 가봐요"

print("Okt")
print('형태소분석: ', okt.morphs(text))
print('품사태깅: ', okt.pos(text))
print('명사추출: ', okt.nouns(text))
# >>> 출력:
# Okt
# 형태소분석:  ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
# 품사태깅:  [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
# 명사추출:  ['코딩', '당신', '연휴', '여행']

print("Kkma")
print('형태소분석: ', kkma.morphs(text))
print('품사태깅: ', kkma.pos(text))
print('명사추출: ', kkma.nouns(text))
# >>> 출력:
# Kkma
# 형태소분석:  ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']
# 품사태깅:  [('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
# 명사추출:  ['코딩', '당신', '연휴', '여행']

# # 정제(Cleaning), 정규화(Normalization)
# 
# - **정제(cleaning)** : 갖고 있는 코퍼스로부터 노이즈 데이터를 제거한다.
# - **정규화(normalization)** : 표현 방법이 다른 단어들을 통합시켜서 같은 단어로 만들어준다.

# (일반적으로)  정제 및 정규화는 '토큰화 작업전 (직후)' 에 수행하곤 한다

# ## 규칙에 기반한 표기가 다른 단어들은 -> 통합

# ex)  USA 와 US,  uh-huh 와 uhhuh <= 형태는 다르지만 여전히 같은 의미

# 표기가 다른 단어들을 통합하는 방법중
#   - 어간추출 (steemming)
#   - 표제어추출(lemmatization)

# ## 대,소문자 통합
# - 영어권 언어의 정규화 방법중 하나

# 대소문자를 통합해야 하는 경우
# ex) Ferrari 로 검색했을때와 ferrari 로 검색했을때 동일한 결과를 얻을수 있게 해야 한다

# 대소문자가 구분시켜야 하는 경우
# ex) US 와 us 는 다른 의미의 단어.
# ex) 회사 이름(General Motors)나, 사람 이름(Bush) 등은 대문자로 유지

# ## 불용어(Stop word) 제거
# 불필요한 단어 제거
# 
# ex)
# 1. 등장빈도가 적은 단어
# 1. 길이가 짧은 단어

text = "I was wondering if anyone out there could enlighten me on this car."

# 길이가 1~2 인 단어들은 정규표현식을 이용하여 삭제
shortword = re.compile(r'\W*\b\w{1,2}\b')
shortword.sub('', text)
# >>> 출력:
# ' was wondering anyone out there could enlighten this car.'

# # 어간 추출(Stemming) and 표제어 추출(Lemmatization)

# 이 두 작업이 갖고 있는 의미는. 눈으로 봤을 때는 서로 다른 단어들이지만,
# 하나의 단어로 일반화시킬 수 있다면 하나의 단어로 일반화시켜서
# 문서 내의 단어 수(종류)를 줄이겠다는 거다

# ## 표제어 추출(Lemmatization)

# 표제어(Lemma) : '기본 사전형 단어'
# 표제어 추출은 단어들이 다른 형태를 가지더라도, 그 뿌리 단어를 찾아가서 단어의 개수를 줄일 수 있는지 판단
# ex) am, are, is는 서로 다른 스펠링이지만 그 뿌리 단어는 be

# 형태소: '의미를 가진 가장 작은 단위'
# 형태소의 종류: 어간(stem)과 접사(affix)

# 1) 어간(stem)
# : 단어의 의미를 담고 있는 단어의 핵심 부분.

# 2) 접사(affix)
# : 단어에 추가적인 의미를 주는 부분.

# cats  :  cat(어간) -s (접사)
# fox : 더이상 분리 안됨.

nltk.download('wordnet')
nltk.download('punkt')
# >>> 출력:
# [nltk_data] Downloading package wordnet to /root/nltk_data...
# [nltk_data] Downloading package punkt to /root/nltk_data...
# [nltk_data]   Package punkt is already up-to-date!
# True

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
words
# >>> 출력:
# ['policy',
#  'doing',
#  'organization',
#  'have',
#  'going',
#  'love',
#  'lives',
#  'fly',
#  'dies',
#  'watched',
#  'has',
#  'starting']

print(words)
print([lemmatizer.lemmatize(word) for word in words])
# >>> 출력:
# ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
# ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']

# 위의 결과에서는 dy나 ha와 같이 의미를 알 수 없는 적절하지 못한 단어를 출력하고 있습니다.
# 이는 표제어 추출기(lemmatizer)가 본래 단어의 '품사 정보'를 알아야만 정확한 결과를 얻을 수 있기 때문입니다.

print(lemmatizer.lemmatize('dies', 'v'))
# >>> 출력:
# die

print(lemmatizer.lemmatize('watched', 'v'))
# >>> 출력:
# watch

print(lemmatizer.lemmatize('has', 'v'))
# >>> 출력:
# have

# ## 어간 추출 (Stemming)

# 어간(Stem)을 추출하는 작업을 어간 추출(stemming)이라고 합니다.
# 어간 추출은 형태학적 분석을 단순화한 버전이라고 볼 수도 있고,
# 정해진 규칙만 보고 단어의 어미를 자르는 어림짐작의 작업이라고 볼 수도 있다.

sentence = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
sentence
# >>> 출력:
# "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
tokenized_sentence = word_tokenize(sentence)

print(tokenized_sentence)
print([stemmer.stem(word) for word in tokenized_sentence])
# >>> 출력:
# ['This', 'was', 'not', 'the', 'map', 'we', 'found', 'in', 'Billy', 'Bones', "'s", 'chest', ',', 'but', 'an', 'accurate', 'copy', ',', 'complete', 'in', 'all', 'things', '--', 'names', 'and', 'heights', 'and', 'soundings', '--', 'with', 'the', 'single', 'exception', 'of', 'the', 'red', 'crosses', 'and', 'the', 'written', 'notes', '.']
# ['thi', 'wa', 'not', 'the', 'map', 'we', 'found', 'in', 'billi', 'bone', "'s", 'chest', ',', 'but', 'an', 'accur', 'copi', ',', 'complet', 'in', 'all', 'thing', '--', 'name', 'and', 'height', 'and', 'sound', '--', 'with', 'the', 'singl', 'except', 'of', 'the', 'red', 'cross', 'and', 'the', 'written', 'note', '.']

words = ['formalize', 'allowance', 'electricical']

print(words)
print([stemmer.stem(word) for word in words])
# >>> 출력:
# ['formalize', 'allowance', 'electricical']
# ['formal', 'allow', 'electric']

from nltk.stem import LancasterStemmer
lancaster_stemmer = LancasterStemmer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
words
# >>> 출력:
# ['policy',
#  'doing',
#  'organization',
#  'have',
#  'going',
#  'love',
#  'lives',
#  'fly',
#  'dies',
#  'watched',
#  'has',
#  'starting']

print(words)
print([stemmer.stem(w) for w in words])
print([lancaster_stemmer.stem(w) for w in words])
# >>> 출력:
# ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
# ['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']
# ['policy', 'doing', 'org', 'hav', 'going', 'lov', 'liv', 'fly', 'die', 'watch', 'has', 'start']

# ## 한국어에서의 어간 추출

# - 한국어는 5언 9품사 구조
# 
# |언|품사|
# |------|---|
# |체언|명사, 대명사, 수사|
# |수식언|관형사, 부사|
# |관계언|조사|
# |독립언|감탄사|
# |**용언**|**동사, 형용사**|
# 
# - 용언에 해당되는 '동사'와 '형용사'는 어간(stem)과 어미(ending)의 결합으로 구성

# 간혹 어간의 모양이 바뀌는 경우
#   긋다, 긋고, 그어서, 그어라

# 규칙활용
#  잡다 => '잡'/어간  + '다'/어미

# 불규칙활용

# '듣다' => '듣/들-',
# '돕다' => '돕/도우-'
# '곱다' => '곱/고우-'

# # 불용어 (stopword)

nltk.download('stopwords')
# >>> 출력:
# [nltk_data] Downloading package stopwords to /root/nltk_data...
# [nltk_data]   Unzipping corpora/stopwords.zip.
# True

from nltk.corpus import stopwords

# NLTK 에서 불용어 확인하기
stop_words_list = stopwords.words('english')
print(len(stop_words_list))
print(stop_words_list[:10])
# >>> 출력:
# 198
# ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an']

# ## 불용어 제거하기 (NLTK)

example = "Family is not an important thing. It's everything."

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example)

result = []
for word in word_tokens:
  if word not in stop_words:
    result.append(word)

print(word_tokens)
print(result)
# >>> 출력:
# ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
# ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']

# ## 한국어에서 불용어 제거하기

# 방법1: 간단하게 '조사' '접속사' 제거하는 방법
# 방법2: '사용자 지정 불용어 사전' 제공하여 제거하는 방법

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "를 아무렇게나 구 우려 고 안 돼 같은 게 구울 때 는"

stop_words = set(stop_words.split(' '))
stop_words
# >>> 출력:
# {'같은', '게', '고', '구', '구울', '는', '돼', '때', '를', '아무렇게나', '안', '우려'}

word_tokens = okt.morphs(example)
print(word_tokens)
# >>> 출력:
# ['고기', '를', '아무렇게나', '구', '우려', '고', '하면', '안', '돼', '.', '고기', '라고', '다', '같은', '게', '아니거든', '.', '예컨대', '삼겹살', '을', '구울', '때', '는', '중요한', '게', '있지', '.']

result = [word for word in word_tokens if word not in stop_words]

print('제거 전', word_tokens)
print('제거 후', result)
# >>> 출력:
# 제거 전 ['고기', '를', '아무렇게나', '구', '우려', '고', '하면', '안', '돼', '.', '고기', '라고', '다', '같은', '게', '아니거든', '.', '예컨대', '삼겹살', '을', '구울', '때', '는', '중요한', '게', '있지', '.']
# 제거 후 ['고기', '하면', '.', '고기', '라고', '다', '아니거든', '.', '예컨대', '삼겹살', '을', '중요한', '있지', '.']

# https://www.ranks.nl/stopwords/korean
#  불용어 사전의 절대적인 기준은 없다.
#  다루는 task 나 분석목적에 따라 운용.

# # ■ 정수 인코딩 (Integer Encoding)
# 
# 텍스트 -> 숫자
# - 각 단어를 '고유한 정수' 에 매핑 시키는 전처리 작업

# 단어 인덱스 부여.

# 인덱스 부여시 일반적으로 단어의 '등장 빈도수' 기준으로 부여함.

# NLTK 로 정수 인코딩 하는 방법

# ## dictionary 사용하기

raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
raw_text
# >>> 출력:
# 'A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain.'

# 문장 토큰화
sentences = sent_tokenize(raw_text)
sentences
# >>> 출력:
# ['A barber is a person.',
#  'a barber is good person.',
#  'a barber is huge person.',
#  'he Knew A Secret!',
#  'The Secret He Kept is huge secret.',
#  'Huge secret.',
#  'His barber kept his word.',
#  'a barber kept his word.',
#  'His barber kept his secret.',
#  'But keeping and keeping such a huge secret to himself was driving the barber crazy.',
#  'the barber went up a huge mountain.']

# 위 결과에 단어토큰화 수행
vocab = {}
preprocessed_sentences = []
stop_words = set(stopwords.words('english'))

for sentence in sentences:
  tokenized_sentence = word_tokenize(sentence)
  result = []

  for word in tokenized_sentence:
    word = word.lower()  # 모든 단어를 소문자화 하여 단어의 개수 줄임.
    if word not in stop_words:  # 토큰화된 단어 가 불용어인 경우 제거
      if len(word) > 2:    # 단어 길이 2 이하인 경우 제거
        result.append(word)

        # 등장빈도수 체크
        if word not in vocab:
          vocab[word] = 0
        vocab[word] += 1

  preprocessed_sentences.append(result)

preprocessed_sentences
# >>> 출력:
# [['barber', 'person'],
#  ['barber', 'good', 'person'],
#  ['barber', 'huge', 'person'],
#  ['knew', 'secret'],
#  ['secret', 'kept', 'huge', 'secret'],
#  ['huge', 'secret'],
#  ['barber', 'kept', 'word'],
#  ['barber', 'kept', 'word'],
#  ['barber', 'kept', 'secret'],
#  ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
#  ['barber', 'went', 'huge', 'mountain']]

# 단어 집합 (사전)
vocab  # 단어: 빈도수
# >>> 출력:
# {'barber': 8,
#  'person': 3,
#  'good': 1,
#  'huge': 5,
#  'knew': 1,
#  'secret': 6,
#  'kept': 4,
#  'word': 2,
#  'keeping': 2,
#  'driving': 1,
#  'crazy': 1,
#  'went': 1,
#  'mountain': 1}

vocab['barber']
# >>> 출력:
# 8

# 빈도수가 높은 순서대로 정렬
vocab_sorted = sorted(vocab.items(), key=lambda x:x[1], reverse=True)

vocab_sorted
# >>> 출력:
# [('barber', 8),
#  ('secret', 6),
#  ('huge', 5),
#  ('kept', 4),
#  ('person', 3),
#  ('word', 2),
#  ('keeping', 2),
#  ('good', 1),
#  ('knew', 1),
#  ('driving', 1),
#  ('crazy', 1),
#  ('went', 1),
#  ('mountain', 1)]

# 정수값 부여  1 부터 시작.
word_to_index = {}

i = 0
for (word, frequency) in vocab_sorted:
  if frequency > 1:  # 빈도수가 작은 단어 제외
    i += 1
    word_to_index[word] = i

word_to_index
# >>> 출력:
# {'barber': 1,
#  'secret': 2,
#  'huge': 3,
#  'kept': 4,
#  'person': 5,
#  'word': 6,
#  'keeping': 7}

# 만약 빈도수 상위 5개 단어만 사용한다면?

vocab_size = 5
words_frequency = [word for word, index in word_to_index.items() if index >= vocab_size + 1]

for w in words_frequency:
  del word_to_index[w]  # 해당 단어에 대한 인덱스 정보를 삭제

word_to_index
# >>> 출력:
# {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}

# ### OOV (Out-Of-Vocabulary)
# 사전에 없는 (인덱싱 안된) 단어토큰

# OOV 를 word_to_index  에 새롭게 추가
# 단어집합(사전) 에 없는 단어는 'OOV' 로 인덱싱 할거다.

word_to_index['OOV'] = len(word_to_index) + 1

word_to_index
# >>> 출력:
# {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'OOV': 6}

# word_to_index 를 사용하여 sentences 의 모든 단어들을 정수 인코딩 하자!

preprocessed_sentences
# >>> 출력:
# [['barber', 'person'],
#  ['barber', 'good', 'person'],
#  ['barber', 'huge', 'person'],
#  ['knew', 'secret'],
#  ['secret', 'kept', 'huge', 'secret'],
#  ['huge', 'secret'],
#  ['barber', 'kept', 'word'],
#  ['barber', 'kept', 'word'],
#  ['barber', 'kept', 'secret'],
#  ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
#  ['barber', 'went', 'huge', 'mountain']]

encoded_sentences = []

for sentence in preprocessed_sentences:
  encoded_sentence = []
  for word in sentence:
    try:
      encoded_sentence.append(word_to_index[word])
    except KeyError:
      encoded_sentence.append(word_to_index['OOV'])

  encoded_sentences.append(encoded_sentence)

encoded_sentences
# >>> 출력:
# [[1, 5],
#  [1, 6, 5],
#  [1, 3, 5],
#  [6, 2],
#  [2, 4, 3, 2],
#  [3, 2],
#  [1, 4, 6],
#  [1, 4, 6],
#  [1, 4, 2],
#  [6, 6, 3, 2, 6, 1, 6],
#  [1, 6, 3, 6]]

#  원본 문장과 인코딩 된 문장을 같이 출력해보자

for e, s in zip(encoded_sentences, sentences):
  print(f'{str(e):25} {s}')
# >>> 출력:
# [1, 5]                    A barber is a person.
# [1, 6, 5]                 a barber is good person.
# [1, 3, 5]                 a barber is huge person.
# [6, 2]                    he Knew A Secret!
# [2, 4, 3, 2]              The Secret He Kept is huge secret.
# [3, 2]                    Huge secret.
# [1, 4, 6]                 His barber kept his word.
# [1, 4, 6]                 a barber kept his word.
# [1, 4, 2]                 His barber kept his secret.
# [6, 6, 3, 2, 6, 1, 6]     But keeping and keeping such a huge secret to himself was driving the barber crazy.
# [1, 6, 3, 6]              the barber went up a huge mountain.

# ## keras 의 텍스트 전처리

# ### Tokenizer
# 
# **tf.keras.preprocessing.text.Tokenizer**
# - fit_on_texts()
# - word_index
# - word_counts
# - texts_to_sequences()
# - num_words =
# - oov_token =
# 
# ---
# 
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
# 
# ```python
# tf.keras.preprocessing.text.Tokenizer(
#     num_words=None,
#     filters=&#x27;!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
#     lower=True,
#     split=&#x27; ',
#     char_level=False,
#     oov_token=None,
#     analyzer=None,
#     **kwargs
# )
# ```

from tensorflow.keras.preprocessing.text import Tokenizer

preprocessed_sentences
# >>> 출력:
# [['barber', 'person'],
#  ['barber', 'good', 'person'],
#  ['barber', 'huge', 'person'],
#  ['knew', 'secret'],
#  ['secret', 'kept', 'huge', 'secret'],
#  ['huge', 'secret'],
#  ['barber', 'kept', 'word'],
#  ['barber', 'kept', 'word'],
#  ['barber', 'kept', 'secret'],
#  ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
#  ['barber', 'went', 'huge', 'mountain']]

# #### fit_on_texts()

tokenizer = Tokenizer()

# fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성.
tokenizer.fit_on_texts(preprocessed_sentences)

# #### .word_index  .word_counts

tokenizer.word_index
# >>> 출력:
# {'barber': 1,
#  'secret': 2,
#  'huge': 3,
#  'kept': 4,
#  'person': 5,
#  'word': 6,
#  'keeping': 7,
#  'good': 8,
#  'knew': 9,
#  'driving': 10,
#  'crazy': 11,
#  'went': 12,
#  'mountain': 13}

tokenizer.word_counts
# >>> 출력:
# OrderedDict([('barber', 8),
#              ('person', 3),
#              ('good', 1),
#              ('huge', 5),
#              ('knew', 1),
#              ('secret', 6),
#              ('kept', 4),
#              ('word', 2),
#              ('keeping', 2),
#              ('driving', 1),
#              ('crazy', 1),
#              ('went', 1),
#              ('mountain', 1)])

# #### .texts_to_sequences()

# texts_to_sequences()는 입력으로 들어온 코퍼스에 대해서 각 단어를 이미 정해진 인덱스로 변환합니다.

print(preprocessed_sentences)
print(tokenizer.texts_to_sequences(preprocessed_sentences))
# >>> 출력:
# [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
# [[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]

# Tokenizer (num_words=)
#   빈도수가 높은 상위 몇개의 단어만 사용 가능.

vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 1)
tokenizer.fit_on_texts(preprocessed_sentences)

# num_words 는 숫자를 0부터 카운트
# 만약 숫자 5를 설정하면 0 ~ 4번 단어 보존. => 1번단어 ~ 4번단어만 남는다.

# 그래서 1번 ~ 5번 단어를 사용하고 싶다면 5 + 1 값을 넣어준겁니다.

# keras Tokenizer 는 0번을 자연어처리에서 필요한 패딩(padding) 에 사용함.

tokenizer.word_index
# >>> 출력:
# {'barber': 1,
#  'secret': 2,
#  'huge': 3,
#  'kept': 4,
#  'person': 5,
#  'word': 6,
#  'keeping': 7,
#  'good': 8,
#  'knew': 9,
#  'driving': 10,
#  'crazy': 11,
#  'went': 12,
#  'mountain': 13}

# ?? 상위 5개만 사용한다며?  여전히 모든 단어가 출력?

tokenizer.word_counts
# >>> 출력:
# OrderedDict([('barber', 8),
#              ('person', 3),
#              ('good', 1),
#              ('huge', 5),
#              ('knew', 1),
#              ('secret', 6),
#              ('kept', 4),
#              ('word', 2),
#              ('keeping', 2),
#              ('driving', 1),
#              ('crazy', 1),
#              ('went', 1),
#              ('mountain', 1)])

# num_words= 가 적용되는 것은 texts_to_sequences() 를 사용할때 적용됨!

print(preprocessed_sentences)

print(tokenizer.texts_to_sequences(preprocessed_sentences))
# >>> 출력:
# [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
# [[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]

# 상위 5개 단어만 (1번~5번) 보존되고 나머지 단어들은 제거된 것과 같은 결과.

# 만약 word_index 와 word counts 에서도
# 지정된 num_words 만큼의 단어만 남기고 싶다면! ↓↓

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)

vocab_size
# >>> 출력:
# 5

words_frequency = [
  word
  for word, index in tokenizer.word_index.items()
  if index >= vocab_size + 1  # 인덱스 5 보다 큰 단어 제거
]

for word in words_frequency:
  del tokenizer.word_index[word]
  del tokenizer.word_counts[word]

print(tokenizer.word_index)
print(tokenizer.word_counts)
# >>> 출력:
# {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
# OrderedDict([('barber', 8), ('person', 3), ('huge', 5), ('secret', 6), ('kept', 4)])

print(preprocessed_sentences)
print(tokenizer.texts_to_sequences(preprocessed_sentences))
# >>> 출력:
# [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
# [[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]

# #### oov_token=
# - 단어집합에 없는 단어들 (out of vocabulary)
# - keras 는 'OOV' 인덱스를 1 로 설정 (기본)

# 숫자 0 (패딩) 과 OOV 를 고려해서 단어 집합의 크기는 +2
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 2, oov_token = 'OOV')
tokenizer.fit_on_texts(preprocessed_sentences)

tokenizer.word_index['OOV']  # OOV 인덱스는 1 로 설정된다.
# >>> 출력:
# 1

print(preprocessed_sentences)
print(tokenizer.texts_to_sequences(preprocessed_sentences))
# >>> 출력:
# [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
# [[2, 6], [2, 1, 6], [2, 4, 6], [1, 3], [3, 5, 4, 3], [4, 3], [2, 5, 1], [2, 5, 1], [2, 5, 3], [1, 1, 4, 3, 1, 2, 1], [2, 1, 4, 1]]

# 빈도수 상위 5개가 2번~6번 까지 인덱스 형성.

tokenizer.word_index
# >>> 출력:
# {'OOV': 1,
#  'barber': 2,
#  'secret': 3,
#  'huge': 4,
#  'kept': 5,
#  'person': 6,
#  'word': 7,
#  'keeping': 8,
#  'good': 9,
#  'knew': 10,
#  'driving': 11,
#  'crazy': 12,
#  'went': 13,
#  'mountain': 14}

# # ■ 패딩 (padding)

# ![](https://miro.medium.com/v2/resize:fit:1400/1*CPLhZoVSTCWgAxe2LKXoOA.png)

# 자연어 처리를 하다보면 각 문장(또는 문서)은 서로 길이가 다를 수 있습니다.
# 그런데 기계는 길이가 전부 동일한 문서들에 대해서는 하나의 행렬로 보고,
# 한꺼번에 묶어서 처리할 수 있습니다.

# 다시 말해 병렬 연산을 위해서 여러 문장의 길이를 임의로 동일하게 맞춰주는 작업이 필요!.

# ## numpy 로 패딩하기

preprocessed_sentences
# >>> 출력:
# [['barber', 'person'],
#  ['barber', 'good', 'person'],
#  ['barber', 'huge', 'person'],
#  ['knew', 'secret'],
#  ['secret', 'kept', 'huge', 'secret'],
#  ['huge', 'secret'],
#  ['barber', 'kept', 'word'],
#  ['barber', 'kept', 'word'],
#  ['barber', 'kept', 'secret'],
#  ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
#  ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)

encoded
# >>> 출력:
# [[1, 5],
#  [1, 8, 5],
#  [1, 3, 5],
#  [9, 2],
#  [2, 4, 3, 2],
#  [3, 2],
#  [1, 4, 6],
#  [1, 4, 6],
#  [1, 4, 2],
#  [7, 7, 3, 2, 10, 1, 11],
#  [1, 12, 3, 13]]

# 위 문장중에서 가장 길ㅇ기가 긴 문장의 길이로 계산
max_len = max([len(item) for item in encoded])

max_len
# >>> 출력:
# 7

# 모든 문장의 길이를 max_len 으로 맞추어봅니다
# 패딩문자는 'PAD' 라 하고, 정수 0 으로 정의
# 길이가 7보다 짧은 문장에는 숫자 0 으로 채워짐.

for sentence in encoded:
  while len(sentence) < max_len:
    sentence.append(0)  # 0 'PAD' 를 뒤에 추가

padded_arr = np.array(encoded)
padded_arr
# >>> 출력:
# array([[ 1,  5,  0,  0,  0,  0,  0],
#        [ 1,  8,  5,  0,  0,  0,  0],
#        [ 1,  3,  5,  0,  0,  0,  0],
#        [ 9,  2,  0,  0,  0,  0,  0],
#        [ 2,  4,  3,  2,  0,  0,  0],
#        [ 3,  2,  0,  0,  0,  0,  0],
#        [ 1,  4,  6,  0,  0,  0,  0],
#        [ 1,  4,  6,  0,  0,  0,  0],
#        [ 1,  4,  2,  0,  0,  0,  0],
#        [ 7,  7,  3,  2, 10,  1, 11],
#        [ 1, 12,  3, 13,  0,  0,  0]])

# ## keras 로 패딩하기 pad_sequences()

from keras.utils import pad_sequences

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences
# ```python
# tf.keras.utils.pad_sequences(
#     sequences,
#     maxlen=None,
#     dtype='int32',
#     padding='pre',
#     truncating='pre',
#     value=0.0
# )
# ```

# encoded 값을 패딩 이전 값으로 되돌리기

encoded = tokenizer.texts_to_sequences(preprocessed_sentences)

encoded
# >>> 출력:
# [[1, 5],
#  [1, 8, 5],
#  [1, 3, 5],
#  [9, 2],
#  [2, 4, 3, 2],
#  [3, 2],
#  [1, 4, 6],
#  [1, 4, 6],
#  [1, 4, 2],
#  [7, 7, 3, 2, 10, 1, 11],
#  [1, 12, 3, 13]]

padded = pad_sequences(encoded)
padded
# >>> 출력:
# array([[ 0,  0,  0,  0,  0,  1,  5],
#        [ 0,  0,  0,  0,  1,  8,  5],
#        [ 0,  0,  0,  0,  1,  3,  5],
#        [ 0,  0,  0,  0,  0,  9,  2],
#        [ 0,  0,  0,  2,  4,  3,  2],
#        [ 0,  0,  0,  0,  0,  3,  2],
#        [ 0,  0,  0,  0,  1,  4,  6],
#        [ 0,  0,  0,  0,  1,  4,  6],
#        [ 0,  0,  0,  0,  1,  4,  2],
#        [ 7,  7,  3,  2, 10,  1, 11],
#        [ 0,  0,  0,  1, 12,  3, 13]], dtype=int32)

# ↑ 'PAD' 0 를 문장의 '앞' 에 붙인다!

# padding='pre' (기본값)  문장 시퀀스의 '앞' 에 패딩값 0 으로 채움
# padding='post'  문장 시퀀스의 '뒤' 에 패딩값을 채움

padded = pad_sequences(encoded, padding='post')
padded
# >>> 출력:
# array([[ 1,  5,  0,  0,  0,  0,  0],
#        [ 1,  8,  5,  0,  0,  0,  0],
#        [ 1,  3,  5,  0,  0,  0,  0],
#        [ 9,  2,  0,  0,  0,  0,  0],
#        [ 2,  4,  3,  2,  0,  0,  0],
#        [ 3,  2,  0,  0,  0,  0,  0],
#        [ 1,  4,  6,  0,  0,  0,  0],
#        [ 1,  4,  6,  0,  0,  0,  0],
#        [ 1,  4,  2,  0,  0,  0,  0],
#        [ 7,  7,  3,  2, 10,  1, 11],
#        [ 1, 12,  3, 13,  0,  0,  0]], dtype=int32)

(padded_arr == padded).all()  # 전부 참이면 True
# >>> 출력:
# np.True_

# ### maxlen=
# 문장 시퀀스의 길이 지정

padded = pad_sequences(encoded, padding='post', maxlen=5)
padded
# >>> 출력:
# array([[ 1,  5,  0,  0,  0],
#        [ 1,  8,  5,  0,  0],
#        [ 1,  3,  5,  0,  0],
#        [ 9,  2,  0,  0,  0],
#        [ 2,  4,  3,  2,  0],
#        [ 3,  2,  0,  0,  0],
#        [ 1,  4,  6,  0,  0],
#        [ 1,  4,  6,  0,  0],
#        [ 1,  4,  2,  0,  0],
#        [ 3,  2, 10,  1, 11],
#        [ 1, 12,  3, 13,  0]], dtype=int32)

# 5보다 긴 문장들은 5의 크기로 줄어들면서 삭제된다.

# 뒤에서 두번째 문장을 모면
# [ 7,  7,  3,  2, 10,  1, 11]
#  [ 3,  2, 10,  1, 11]  <==  앞쪽이 삭제됨.

#  truncating='pre' (기본값)  데이터 손실 발생시 '앞' 의 단어가 삭제
#  truncating='post'  데이터 손실 발생시 '뒤' 의 단어가 삭제

# ### truncating=

padded = pad_sequences(encoded, padding='post', maxlen=5, truncating='post')
padded
# >>> 출력:
# array([[ 1,  5,  0,  0,  0],
#        [ 1,  8,  5,  0,  0],
#        [ 1,  3,  5,  0,  0],
#        [ 9,  2,  0,  0,  0],
#        [ 2,  4,  3,  2,  0],
#        [ 3,  2,  0,  0,  0],
#        [ 1,  4,  6,  0,  0],
#        [ 1,  4,  6,  0,  0],
#        [ 1,  4,  2,  0,  0],
#        [ 7,  7,  3,  2, 10],
#        [ 1, 12,  3, 13,  0]], dtype=int32)

# [ 7,  7,  3,  2, 10,  1, 11]  <- 원본 문장
#  [ 3,  2, 10,  1, 11]  <==  truncating='pre'
#  [ 7,  7,  3,  2, 10]   <-- truncating='post'

# ### value=
# 패딩 문자 값 지정

# 패딩값은 일반적으로는 '0' 값을 사용함.

# 다른 숫자로도 패딩할수 있다.
# ex) 단어집합(사전)의 크기에 +1 한 숫자를 패딩문자로 사용한다면..

last_value = len(tokenizer.word_index) + 1

last_value
# >>> 출력:
# 14

padded = pad_sequences(encoded, padding='post', value=last_value)
padded
# >>> 출력:
# array([[ 1,  5, 14, 14, 14, 14, 14],
#        [ 1,  8,  5, 14, 14, 14, 14],
#        [ 1,  3,  5, 14, 14, 14, 14],
#        [ 9,  2, 14, 14, 14, 14, 14],
#        [ 2,  4,  3,  2, 14, 14, 14],
#        [ 3,  2, 14, 14, 14, 14, 14],
#        [ 1,  4,  6, 14, 14, 14, 14],
#        [ 1,  4,  6, 14, 14, 14, 14],
#        [ 1,  4,  2, 14, 14, 14, 14],
#        [ 7,  7,  3,  2, 10,  1, 11],
#        [ 1, 12,  3, 13, 14, 14, 14]], dtype=int32)

# # ■ 원-핫 인코딩 (One-Hot Encoding)

# ## 단어집합 (vocabulary)

# 단어집합 (고유한 단어들) -> 정수인코딩 --> '벡터' 로 다루고 싶다면?

# ## 원-핫 인코딩 작성 (수동 )

# 단어집합의 크기를 벡터의 '차원' 으로 하고
# 표현하고 싶은 단어의 인덱스에 '1' 값을 부여, 다른 인덱스에는 '0' 을 부여하여 단어의 벡터 표현
# 이렇게 표현된 벡터를 원-핫 벡터 (one-hot vector)

# 토큰화 수행
tokens = okt.morphs('나는 자연어 처리를 배운다')
tokens
# >>> 출력:
# ['나', '는', '자연어', '처리', '를', '배운다']

# 고유한 정수 부여
word_to_index = {word: index for index, word in enumerate(tokens)}

word_to_index
# >>> 출력:
# {'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}

# 토큰을 입력받아 원-핫벡터 만드는 함수
def one_hot_encoding(word, word_to_index):
  one_hot_vector = [0] * (len(word_to_index)) #단어집합의 크기를 벡터의 '차원' 으로 하고
  index = word_to_index[word]  # 표현하고 싶은 단어의 인덱스에 '1' 값을 부여
  one_hot_vector[index] = 1
  return one_hot_vector

one_hot_encoding('자연어', word_to_index)
# >>> 출력:
# [0, 0, 1, 0, 0, 0]

# ## keras 사용 : to_categorical()

from keras.utils import to_categorical

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
# ```python
# tf.keras.utils.to_categorical(
#     x, num_classes=None
# )
# ```

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"
# 위와 같은 문장이 있다고 했을 때, 케라스 토크나이저를 이용한 정수 인코딩은 다음과 같습니다.
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
print('단어 집합 :',tokenizer.word_index)
# >>> 출력:
# 단어 집합 : {'갈래': 1, '점심': 2, '햄버거': 3, '나랑': 4, '먹으러': 5, '메뉴는': 6, '최고야': 7}

sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"

encoded = tokenizer.texts_to_sequences([sub_text])[0]
encoded
# >>> 출력:
# [2, 5, 1, 6, 3, 7]

# 위 정수 인코딩 결과를 원-핫 인코딩 해보자

one_hot = to_categorical(encoded)

one_hot
# >>> 출력:
# array([[0., 0., 1., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 0., 1., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 1.]])

# ##  원-핫 인코딩(One-Hot Encoding)의 한계
# - 저장공간의 문제
# - 단어의 유사도 표현 못함

# 단어의 유사성 알수 있다면..
#  웹 검색창에 '오사카 숙소' 검색.
#   '숙소' 와 유사한 결과들을 낼수 있다  ex) 호텔, 게스트 하우스, 료칸...

# 단어의 유사도 문제 해결 -> 단어를 다차원 공간에 벡터화 기법 두가지
# 1. 카운트 기반의 벡터화 : ex) LSA(잠재 의미 분석), HAL
# 2. 예측 기반으로 벡터화 : NNLM, RNNLM, Word2Vec, FastText
#  카운트 기반과 예측 기반 두 가지 방법을 모두 사용 : ex) GloVe
