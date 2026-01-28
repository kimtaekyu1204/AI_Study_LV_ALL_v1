"""
Word2Vec

개요:
    이 모듈은 AI 자연어처리 수업의 딥러닝/NLP 파트에서 다루는 Word2Vec에 관한
    모든 코드와 설명을 정리한 파일함.

    원본 Jupyter Notebook: 49 Word2Vec.ipynb

============================================
사용 데이터셋: 위키피디아 한국어 덤프 & 사전학습 모델
============================================
1. BEXX0003.txt (한국어 텍스트 코퍼스)
   - 출처: 한국어 위키피디아 덤프 또는 뉴스 코퍼스
   - 설명: Word2Vec 학습을 위한 한국어 텍스트 데이터
   - 형식: UTF-16 인코딩 HTML 형식 텍스트

2. wiki.model (사전학습된 Word2Vec 모델)
   - 출처: 한국어 위키피디아 기반 사전학습 모델
   - 설명: Gensim Word2Vec 포맷의 사전학습된 단어 임베딩
   - 파일 구성:
     * wiki.model: 모델 메타데이터
     * wiki.model.syn1neg.npy: Negative Sampling 가중치 (136MB)
     * wiki.model.wv.syn0.npy: 단어 벡터 행렬 (136MB)
   - 벡터 차원: 일반적으로 100~300차원

데이터 구조 예시 (학습 후):
┌──────────┬────────────────────────────────────────────────────┐
│ word     │ vector (100-dim)                                   │
├──────────┼────────────────────────────────────────────────────┤
│ 한국     │ [0.123, -0.456, 0.789, ..., 0.012]                 │
│ 서울     │ [0.234, -0.567, 0.890, ..., 0.023]                 │
│ 대한민국 │ [0.345, -0.678, 0.901, ..., 0.034]                 │
└──────────┴────────────────────────────────────────────────────┘
============================================
"""

# # Word2Vec
#
# 단어의 '의미' 나 '연관성' 을 벡터로 표현
# 단어의 '의미'를 '벡터'로 표현하면,
# 연관된 단어를 추출하거나, 단어와 단어의 유사도를 확인할수 있다.
# 또한 의미를 선형계산할수 있어서 "왕자 - 남성 + 여성 => 공주" 와 같은 계산을 할수 있다
# ![](https://miro.medium.com/max/1400/1*2r1yj0zPAuaSGZeQfG6Wtw.png)
# Word2Vec : 문장 내부의 단어를 벡터로 변환하는 도구
# 단어의 연결을 기반으로 단어의 연관성을 벡터로 만들어줌
# 즉, 단어를 벡터로 표현해줌.

# 단어를 벡터로 사용하면 , 단어의 '유사도' 를 쉽게 확인할수 있다.
base_path = r'/content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)'
!pip install konlpy
!pip install --upgrade --force-reinstall numpy
!pip install --upgrade --force-reinstall scipy
!pip install --upgrade --force-reinstall gensim
import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from gensim.models import word2vec
import os
fp = codecs.open(os.path.join(base_path, 'word_vec', 'BEXX0003.txt'), 'r', encoding='utf-16')
soup = BeautifulSoup(fp, 'html.parser')
body = soup.select_one('body > text')
text = body.getText()

text
# 어미, 조사, 구두점 제거
#  - 명사, 동사, 형용사 만 학습시키기 위함
#  - 동사, 형용사는 '기본형' 으로만 학습
okt = Okt()
results = []
lines = text.split('\n')
for line in lines:
  # 형태소 분석 (품사태깅)
  # 단어의 기본형 사용
  malist = okt.pos(line, norm=True, stem=True)
  r = []
  for word in malist:
    # 어미/조사/구두점 제외
    if not word[1] in ['Josa', 'Eomi', 'Punctuation']:
      r.append(word[0])

  r1 = (" ".join(r)).strip()
  results.append(r1)
  print(r1)

wakati_file = os.path.join('toji.wakati')
with open(wakati_file, 'w', encoding='utf-8') as fp:
  fp.write('\n'.join(results))
# 모델에 넣기 위한 LineSentence 객체 생성
data = word2vec.LineSentence(wakati_file)
data
# 워드벡터 모델 생성
model = word2vec.Word2Vec(data, vector_size=200, window=10, hs=1, min_count=2, sg=1)
    # vector_size : 문장의 벡터의 차원
    # window: 한 문장 내에서의 최대 거리값.
    # hs=1 : hierarchical softmax 학습
    # min_count : 발생빈도가 이보다 낮으면 무시
    # sg : 학습 알고리즘 선택 1 이면 -> skip-gram 사용

# 학습 모델 저장
model.save('toji.model')
# 저장한 모델 불러오기
model = word2vec.Word2Vec.load('toji.model')
# 학습된 말뭉치 개수
model.corpus_count
# 말뭉치내 전체 단어개수
model.corpus_total_words
# # 유사도 (similarity)
model.wv.most_similar(positive=['땅'])
model.wv.most_similar(positive=['집'])
# # 위키피디아 학습 모델 사용
# 위키피디아 (한국어판) 데이터 다운로드

#  https://dumps.wikimedia.org/kowiki/latest

model = word2vec.Word2Vec.load(os.path.join(base_path, 'word_vec', 'wiki.model'))
model.wv.most_similar(positive=['Python', '파이썬'])
# 아빠 - 남성 + 여성 =>
model.wv.most_similar(positive=['아빠', '여성'], negative=['남성'])
# 왕자 - 남성 + 여성
model.wv.most_similar(positive=['왕자', '여성'], negative=['남성'])
# 한국에서 서울에 해당하는 곳이 일본에서는 어디일까요?
model.wv.most_similar(positive=['서울', '일본'], negative=['한국'])
model.wv.most_similar(positive=['서울', '중국'], negative=['한국'])
model.wv.most_similar(positive=['서울', '맛집'])
model.wv['고양이']
model.wv['고양이'].shape
