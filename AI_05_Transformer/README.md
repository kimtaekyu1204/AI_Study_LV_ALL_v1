# 🚀 AI_05_Transformer - Transformer 아키텍처

> Attention 메커니즘과 Transformer 기반 모델 (BERT, GPT) 정리

---

## 📚 목차

| 파일 | 주제 | 핵심 키워드 |
|:----:|------|-------------|
| [AI_57](./step1_attention/AI_57_attention.py) | Attention | `Self-Attention`, `Multi-Head`, `Scaled Dot-Product` |
| [AI_58](./step2_bert_gpt/AI_58_transformer.py) | Transformer | `Encoder-Decoder`, `Positional Encoding`, `FFN` |
| [AI_59](./step2_bert_gpt/AI_59_bert.py) | BERT | `MLM`, `NSP`, `Fine-tuning`, `[CLS]`, `[SEP]` |
| [AI_60](./step2_bert_gpt/AI_60_kobert.py) | KoBERT | 한국어 BERT, `SentencePiece`, 감정 분류 |
| [AI_61](./step2_bert_gpt/AI_61_gpt.py) | GPT | `Causal LM`, 자기회귀, 텍스트 생성 |

---

## 1️⃣ Attention 메커니즘

### Scaled Dot-Product Attention
Query, Key, Value를 사용한 어텐션 계산이다.

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Scaled Dot-Product Attention
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    d_k = K.shape[-1]

    # 1. QK^T 계산
    scores = np.matmul(Q, K.T)

    # 2. Scale (sqrt(d_k)로 나누기)
    scaled_scores = scores / np.sqrt(d_k)

    # 3. Softmax
    attention_weights = softmax(scaled_scores)

    # 4. Value와 곱
    output = np.matmul(attention_weights, V)

    return output, attention_weights
```

### Self-Attention
입력 시퀀스 내에서 각 위치가 다른 모든 위치를 참조한다.

```python
# Self-Attention: Q = K = V = 입력
# 문장 내 단어 간 관계 학습

# "The cat sat on the mat"
# "cat"이 "sat"과 "mat"에 높은 attention
```

### Multi-Head Attention
여러 개의 어텐션을 병렬로 수행한다.

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 각 head를 위한 가중치
        self.W_q = Dense(d_model)
        self.W_k = Dense(d_model)
        self.W_v = Dense(d_model)
        self.W_o = Dense(d_model)

    def split_heads(self, x, batch_size):
        """(batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)"""
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
```

---

## 2️⃣ Transformer 아키텍처

### 전체 구조
```
입력 → [Encoder] → Context → [Decoder] → 출력
         ↓                      ↓
    Self-Attention         Masked Self-Attention
         +                       +
    Feed Forward           Cross-Attention
                                 +
                           Feed Forward
```

### Positional Encoding
순서 정보를 임베딩에 추가한다.

```python
import numpy as np

def positional_encoding(max_len, d_model):
    """
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)  # 짝수 인덱스
    pe[:, 1::2] = np.cos(position * div_term)  # 홀수 인덱스

    return pe

# 예: max_len=100, d_model=512
pe = positional_encoding(100, 512)
```

### Encoder Layer
```python
class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

    def call(self, x):
        # Self-Attention + Residual + LayerNorm
        attn_output = self.mha(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed Forward + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x
```

---

## 3️⃣ BERT (Bidirectional Encoder Representations)

### 특징
| 항목 | 설명 |
|:----:|------|
| 방향 | 양방향 (Bidirectional) |
| 구조 | Transformer Encoder만 사용 |
| 사전학습 | MLM + NSP |
| 용도 | 분류, NER, QA 등 |

### Masked Language Model (MLM)
```python
# 입력: "The [MASK] sat on the mat"
# 예측: [MASK] → "cat"

# 마스킹 전략 (15% 토큰)
# - 80%: [MASK]로 대체
# - 10%: 랜덤 토큰으로 대체
# - 10%: 그대로 유지
```

### Next Sentence Prediction (NSP)
```python
# 입력: [CLS] 문장A [SEP] 문장B [SEP]
# 출력: IsNext / NotNext

# 예시
# IsNext: "나는 밥을 먹었다" + "그리고 커피를 마셨다"
# NotNext: "나는 밥을 먹었다" + "날씨가 좋다"
```

### BERT 사용 예시
```python
from transformers import BertTokenizer, BertModel

# 토크나이저 및 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 토큰화
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors='pt')

# 임베딩 추출
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state  # (1, seq_len, 768)
cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰
```

---

## 4️⃣ KoBERT (한국어 BERT)

### 특징
```python
# SKTBrain에서 개발한 한국어 BERT
# SentencePiece 토크나이저 사용
# 한국어 위키피디아로 사전학습
```

### 감정 분류 예시
```python
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

# KoBERT 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BertModel.from_pretrained('skt/kobert-base-v1')

# 한국어 텍스트 처리
text = "오늘 날씨가 정말 좋아서 기분이 좋습니다"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

# [CLS] 토큰으로 분류
cls_output = outputs.last_hidden_state[:, 0, :]
```

---

## 5️⃣ GPT (Generative Pre-trained Transformer)

### 특징
| 항목 | 설명 |
|:----:|------|
| 방향 | 단방향 (Left-to-Right) |
| 구조 | Transformer Decoder만 사용 |
| 사전학습 | Causal Language Modeling |
| 용도 | 텍스트 생성, 대화 |

### BERT vs GPT
| 구분 | BERT | GPT |
|:----:|:----:|:----:|
| Attention | 양방향 | 단방향 (Causal) |
| 학습 방식 | MLM + NSP | Next Token Prediction |
| 주요 용도 | 이해 (분류, NER) | 생성 (텍스트, 대화) |

### Causal Language Modeling
```python
# 입력: "I love"
# 예측: "machine" (다음 토큰)

# Masked Self-Attention
# 현재 위치에서 미래 토큰을 볼 수 없음
# [1, 0, 0]
# [1, 1, 0]
# [1, 1, 1]
```

### GPT 사용 예시
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 텍스트 생성
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors='pt')

outputs = model.generate(
    inputs['input_ids'],
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True
)

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)
```

---

## 📖 학습 순서

```
step1 (Attention) → step2 (Transformer/BERT/GPT)
```

### 핵심 개념 흐름
```
Attention → Self-Attention → Multi-Head → Transformer → BERT/GPT
```
