# 🧠 AI_04_DeepLearning_NLP - 딥러닝/자연어처리

> Keras, PyTorch와 자연어처리 기법 정리

---

## 📚 목차

| 파일 | 주제 | 핵심 키워드 |
|:----:|------|-------------|
| [AI_44](./step1_keras_pytorch/AI_44_keras_models.py) | Keras 모델 | `Sequential`, `Functional API`, `Model` 클래스 |
| [AI_45](./step1_keras_pytorch/AI_45_tensor_comparison.py) | Tensor 비교 | `tf.Tensor`, `KerasTensor`, 차이점 |
| [AI_46](./step1_keras_pytorch/AI_46_tf_dataset.py) | tf.data | `Dataset`, `batch()`, `prefetch()` |
| [AI_47](./step1_keras_pytorch/AI_47_pytorch.py) | PyTorch | `torch.Tensor`, `nn.Module`, `autograd` |
| [AI_48](./step2_text_preprocessing/AI_48_text_preprocessing.py) | 텍스트 전처리 | 토큰화, 불용어, 어간 추출 |
| [AI_49](./step2_text_preprocessing/AI_49_word2vec.py) | Word2Vec | `CBOW`, `Skip-gram`, 단어 임베딩 |
| [AI_50](./step2_text_preprocessing/AI_50_intent_classification.py) | 의도 분류 | 텍스트 분류, 챗봇 의도 파악 |
| [AI_52](./step3_rnn_lstm/AI_52_rnn_imdb.py) | RNN | `SimpleRNN`, IMDB 감정 분류 |
| [AI_53](./step3_rnn_lstm/AI_53_lstm_gru.py) | LSTM/GRU | `LSTM`, `GRU`, `Bidirectional` |
| [AI_54](./step3_rnn_lstm/AI_54_ner.py) | NER | 개체명 인식, 시퀀스 라벨링 |
| [AI_55](./step4_seq2seq/AI_55_seq2seq.py) | Seq2Seq | 인코더-디코더, Teacher Forcing |
| [AI_56](./step4_seq2seq/AI_56_chatbot.py) | 챗봇 | Seq2Seq 챗봇, 응답 생성 |

---

## 1️⃣ Keras 모델 구축

### Sequential API
간단한 순차 모델에 사용한다.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(100,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
```

### Functional API
복잡한 모델 (다중 입력/출력)에 사용한다.

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(100,))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
```

### Model 클래스 상속
완전한 커스터마이징이 필요할 때 사용한다.

```python
class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
```

---

## 2️⃣ PyTorch 기초

### Tensor 생성
```python
import torch

# 텐서 생성
x = torch.tensor([1, 2, 3])
zeros = torch.zeros(3, 3)
ones = torch.ones(2, 4)
rand = torch.randn(3, 3)  # 정규분포

# GPU 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
```

### 모델 정의
```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = Net()
```

### 학습 루프
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
```

---

## 3️⃣ 텍스트 전처리

### 토큰화
```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

print(tokenizer.word_index)  # {'the': 1, 'a': 2, ...}
```

### 패딩
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(sequences, maxlen=100, padding='post')
```

### 한국어 전처리
```python
from konlpy.tag import Okt

okt = Okt()
tokens = okt.morphs("자연어 처리를 공부합니다")
# ['자연어', '처리', '를', '공부', '합니다']

nouns = okt.nouns("자연어 처리를 공부합니다")
# ['자연어', '처리', '공부']
```

---

## 4️⃣ Word2Vec

단어를 벡터로 표현하는 임베딩 기법이다.

### CBOW vs Skip-gram
| 모델 | 설명 |
|:----:|------|
| CBOW | 주변 단어 → 중심 단어 예측 |
| Skip-gram | 중심 단어 → 주변 단어 예측 |

```python
from gensim.models import Word2Vec

# 학습
sentences = [["자연어", "처리", "공부"], ["딥러닝", "모델", "학습"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# 유사 단어
model.wv.most_similar("자연어")

# 단어 벡터
vector = model.wv["자연어"]  # (100,)
```

---

## 5️⃣ RNN / LSTM / GRU

순차 데이터 처리를 위한 신경망이다.

### SimpleRNN
```python
from tensorflow.keras.layers import SimpleRNN, Embedding

model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    SimpleRNN(64),
    Dense(1, activation='sigmoid')
])
```

### LSTM (Long Short-Term Memory)
장기 의존성 문제를 해결한다.

```python
from tensorflow.keras.layers import LSTM

model = Sequential([
    Embedding(vocab_size, 128),
    LSTM(64, return_sequences=True),  # many-to-many
    LSTM(32),                          # many-to-one
    Dense(1, activation='sigmoid')
])
```

### Bidirectional LSTM
양방향으로 시퀀스를 처리한다.

```python
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Embedding(vocab_size, 128),
    Bidirectional(LSTM(64)),
    Dense(1, activation='sigmoid')
])
```

---

## 6️⃣ Seq2Seq (Sequence-to-Sequence)

시퀀스 입력 → 시퀀스 출력 모델이다.

### 구조
```
입력 시퀀스 → [인코더] → Context Vector → [디코더] → 출력 시퀀스
```

### 인코더
```python
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(vocab_size, 256)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]
```

### 디코더
```python
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(vocab_size, 256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
```

---

## 📖 학습 순서

```
step1 (Keras/PyTorch) → step2 (텍스트 전처리) → step3 (RNN/LSTM) → step4 (Seq2Seq)
```
