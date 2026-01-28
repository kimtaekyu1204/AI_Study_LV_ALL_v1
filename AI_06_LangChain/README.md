# 🔗 AI_06_LangChain - LangChain 프레임워크

> LLM 애플리케이션 개발을 위한 LangChain 프레임워크 정리

---

## 📚 목차

| 파일 | 주제 | 핵심 키워드 |
|:----:|------|-------------|
| [AI_62](./AI_62_hello_langchain.py) | LangChain 기초 | `Chain`, `Agent`, `Memory`, `Tools` |
| [AI_63](./AI_63_model_io.py) | Model IO | `PromptTemplate`, `OutputParser`, `LCEL` |
| [AI_64](./AI_64_rag.py) | RAG | `Retriever`, `VectorStore`, `Embedding` |
| [AI_65](./AI_65_huggingface.py) | HuggingFace | `Transformers`, `Pipeline`, `Local Model` |
| [AI_66](./AI_66_ollama.py) | Ollama | 로컬 LLM, `Modelfile`, 커스터마이징 |

---

## 1️⃣ LangChain 기초

### LangChain이란?
LLM을 활용한 애플리케이션 개발 프레임워크이다.

### 주요 구성 요소
| 구성 요소 | 설명 |
|:--------:|------|
| Chain | 여러 작업을 연결한 워크플로우 |
| Agent | LLM이 도구를 선택하고 실행 |
| Memory | 대화 문맥 유지 |
| Tools | 외부 API, 검색 등 기능 |

### 기본 사용
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 모델 초기화
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 2. 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트입니다."),
    ("user", "{input}")
])

# 3. 체인 연결 (LCEL)
chain = prompt | llm

# 4. 실행
response = chain.invoke({"input": "안녕하세요!"})
print(response.content)
```

---

## 2️⃣ Model IO

### PromptTemplate
변수를 포함한 프롬프트 템플릿이다.

```python
from langchain_core.prompts import PromptTemplate

# 기본 템플릿
template = PromptTemplate.from_template(
    "{country}의 수도는 어디인가요?"
)

# 변수 주입
prompt = template.format(country="한국")
# 결과: "한국의 수도는 어디인가요?"
```

### ChatPromptTemplate
대화형 프롬프트 템플릿이다.

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role} 전문가입니다."),
    ("user", "{question}")
])

prompt = template.format_messages(
    role="Python",
    question="리스트 컴프리헨션이 뭔가요?"
)
```

### OutputParser
LLM 출력을 구조화된 형식으로 변환한다.

```python
from langchain_core.output_parsers import JsonOutputParser

# JSON 파서
parser = JsonOutputParser()

# 체인에 연결
chain = prompt | llm | parser

# 결과: dict 형태로 반환
```

### LCEL (LangChain Expression Language)
파이프 연산자로 체인을 구성한다.

```python
# LCEL 문법
chain = prompt | llm | parser

# 동등한 표현
# chain = RunnableSequence(prompt, llm, parser)

# 실행
result = chain.invoke({"input": "질문"})
```

---

## 3️⃣ RAG (Retrieval-Augmented Generation)

### RAG란?
외부 문서를 검색하여 LLM 응답을 향상시키는 기법이다.

```
질문 → [검색] → 관련 문서 → [LLM + 문서] → 답변
```

### RAG 파이프라인 (6단계)

```python
# 1. 문서 로딩
from langchain_community.document_loaders import TextLoader
loader = TextLoader("data.txt")
documents = loader.load()

# 2. 문서 분할
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

# 3. 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# 4. 벡터 저장소
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. 검색기
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. QA 체인
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)
```

### 문서 처리 전략
| 전략 | 설명 |
|:----:|------|
| Stuff | 모든 문서를 한 번에 전달 |
| Refine | 순차적으로 답변 개선 |
| Map Reduce | 병렬 처리 후 결합 |
| Map Re-Rank | 각 문서 점수화 후 선택 |

---

## 4️⃣ HuggingFace 연동

### Hugging Face란?
오픈소스 ML 모델 허브이다.

### LangChain 연동
```python
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 모델 로드
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 파이프라인 생성
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100
)

# LangChain 래퍼
llm = HuggingFacePipeline(pipeline=pipe)

# 사용
response = llm.invoke("Hello, how are you?")
```

### 주요 모델 종류
| 모델 유형 | 용도 | 예시 |
|:--------:|------|------|
| 텍스트 생성 | 문장 완성 | GPT-2, Llama |
| 질의응답 | QA | BERT, RoBERTa |
| 요약 | 문서 요약 | BART, T5 |
| 번역 | 언어 번역 | mBART, NLLB |
| 감정 분석 | 분류 | DistilBERT |

---

## 5️⃣ Ollama (로컬 LLM)

### Ollama란?
로컬에서 LLM을 실행하는 도구이다.

### 장점
| 장점 | 설명 |
|:----:|------|
| 프라이버시 | 데이터가 로컬에 유지 |
| 비용 | API 호출 비용 없음 |
| 오프라인 | 인터넷 없이 사용 가능 |

### 설치 및 사용
```bash
# Ollama 설치
curl -fsSL https://ollama.ai/install.sh | sh

# 모델 다운로드
ollama pull llama2
ollama pull mistral

# 모델 실행
ollama run llama2
```

### LangChain 연동
```python
from langchain_community.llms import Ollama

# 모델 초기화
llm = Ollama(model="llama2")

# 사용
response = llm.invoke("Python이 뭔가요?")
print(response)
```

### Modelfile 커스터마이징
```dockerfile
# Modelfile
FROM llama2

# 시스템 프롬프트 설정
SYSTEM """
당신은 한국어 전문가입니다.
항상 친절하고 정확한 답변을 합니다.
"""

# 파라미터 설정
PARAMETER temperature 0.7
PARAMETER num_predict 1000
```

```bash
# 커스텀 모델 생성
ollama create korean-expert -f Modelfile
```

---

## 📖 학습 순서

```
AI_62 (기초) → AI_63 (Model IO) → AI_64 (RAG) → AI_65 (HuggingFace) → AI_66 (Ollama)
```

### 핵심 개념 흐름
```
LangChain 기초 → 프롬프트/파서 → RAG 파이프라인 → 모델 통합
```

---

## 🔧 필수 설치

```bash
# 기본
pip install langchain langchain-openai

# HuggingFace
pip install transformers torch langchain-huggingface

# Ollama
pip install langchain-community

# 벡터 저장소
pip install faiss-cpu chromadb
```
