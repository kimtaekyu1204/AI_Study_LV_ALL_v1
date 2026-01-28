# # AI_65_huggingface

# ================================================================================
# AI_65_huggingface.py
# LangChain과 Hugging Face 통합 완전 정리
# ================================================================================
# 
# Hugging Face란?
# Hugging Face는 머신러닝 모델, 데이터셋, 그리고 커뮤니티가 모여있는
# 거대한 오픈소스 플랫폼입니다.
# 
# 특징:
# - 수만 개의 사전 학습된 모델 제공
# - 다양한 작업 지원 (NLP, 비전, 음성 등)
# - 커뮤니티 중심의 모델 공유
# - 무료 사용 가능 (모델 제공)
# - API 기반 서비스도 제공
# 
# 주요 링크:
# - 웹사이트: https://huggingface.co/
# - 모델 허브: https://huggingface.co/models
# - 커뮤니티: https://huggingface.co/
# 
# ================================================================================
# Hugging Face 회원가입 및 설정
# ================================================================================
# 
# 1단계: 회원가입
# -----------
# 1. https://huggingface.co/ 방문
# 2. "Sign Up" 클릭
# 3. 이메일 입력 (또는 GitHub/Google 로그인)
# 4. 비밀번호 설정
# 5. 사용 목적 설명 (선택사항)
# 
# 중요: 회원가입 후 이메일 확인 필수!
# 받은 편지함에서 Hugging Face 확인 이메일 클릭
# 
# 2단계: API 토큰 생성
# -----------------
# 1. 프로필 아이콘 클릭 → Settings
# 2. 왼쪽 메뉴에서 "Access Tokens" 클릭
# 3. "New token" 클릭
# 4. Token Type: "Write" 선택 (권장)
# 5. "Generate a token" 클릭
# 6. 생성된 토큰 복사
# 
# 주의 사항:
# - 토큰은 절대 외부에 노출 금지!
# - 깃허브나 공개 저장소에 올리지 말 것
# - .env 파일에 저장 추천
# 
# 토큰 저장 예제:
#     # .env 파일
#     HUGGINGFACE_API_KEY="hf_xxxxxxxxxxxxxxxxxxxxx"
# 
# 3단계: Python 환경에서 토큰 설정
# ----------------------------
# 코드:
#     import os
#     from dotenv import load_dotenv
# 
#     load_dotenv()
#     hf_token = os.getenv("HUGGINGFACE_API_KEY")
# 
#     # 또는 직접 설정
#     os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxxx"
# 
# ================================================================================
# Hugging Face 모델 종류
# ================================================================================
# 
# 1. 텍스트 생성 모델 (Text Generation)
#    - GPT 계열: GPT-2, DistilGPT-2
#    - T5: 다양한 NLP 작업
#    - BLOOM: 다국어 모델
# 
# 2. 질의응답 모델 (Question Answering)
#    - BERT 기반: 높은 정확도
#    - DistilBERT: 경량 모델
# 
# 3. 요약 모델 (Summarization)
#    - T5, BART: 텍스트 요약 전문
#    - PegaSus: 뉴스 요약
# 
# 4. 번역 모델 (Translation)
#    - MarianMT: 다언어 번역
#    - M2M-100: 100개 언어 지원
# 
# 5. 감정 분석 모델 (Sentiment Analysis)
#    - BERT: 감정 분류
#    - RoBERTa: 더 높은 성능
# 
# 6. 코드 생성 모델 (Code Generation)
#    - CodeBERT
#    - GraphCodeBERT
#    - Codex
# 
# ================================================================================
# LangChain에서 Hugging Face 모델 사용
# ================================================================================
# 
# 방법 1: Hugging Face Inference API (온라인)
# -------------------------------------------
# 인터넷 연결 필수, 별도 설치 불필요
# 
# 장점:
# - 설치 간단
# - 최신 모델 자동 업데이트
# - 클라우드 기반 (로컬 리소스 절약)
# 
# 단점:
# - 인터넷 필수
# - API 호출 비용 가능
# - 외부 의존성
# 
# 코드:
#     from langchain.llms import HuggingFaceHub
#     from langchain.prompts import PromptTemplate
# 
#     # 토큰이 환경변수에 설정되어야 함
#     llm = HuggingFaceHub(
#         repo_id="gpt2",  # 모델 ID
#         model_kwargs={
#             "temperature": 0.7,
#             "max_new_tokens": 200
#         }
#     )
# 
#     prompt = PromptTemplate(
#         input_variables=["topic"],
#         template="Write a poem about {topic}"
#     )
# 
#     chain = prompt | llm
#     result = chain.invoke({"topic": "Python"})
#     print(result)
# 
# 방법 2: Hugging Face Local Pipeline (로컬)
# ------------------------------------------
# 모델을 로컬 PC에 다운로드하여 실행
# 
# 장점:
# - 인터넷 불필요 (한 번 다운로드 후)
# - 개인정보 보호
# - 무료 사용
# 
# 단점:
# - 많은 저장공간 필요
# - GPU 권장 (CPU는 느림)
# - 초기 다운로드 시간 필요
# 
# 설치:
#     pip install transformers torch
# 
# 코드:
#     from langchain.llms import HuggingFacePipeline
#     from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# 
#     # 모델과 토크나이저 로드
#     model_id = "gpt2"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id)
# 
#     # Hugging Face 파이프라인 생성
#     hf_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=200,
#         temperature=0.7
#     )
# 
#     # LangChain 래퍼
#     llm = HuggingFacePipeline(pipeline=hf_pipeline)
# 
#     # 사용
#     result = llm("Tell me about artificial intelligence")
#     print(result)
# 
# 방법 3: Hugging Face Transformers 직접 사용
# -------------------------------------------
# Transformers 라이브러리를 직접 사용
# 
# 장점:
# - 완전한 제어
# - 최대 커스터마이제이션
# - 다양한 모델 지원
# 
# 코드:
#     from transformers import pipeline
# 
#     # 텍스트 생성
#     generator = pipeline("text-generation", model="gpt2")
#     result = generator("Python is a great")
#     print(result[0]["generated_text"])
# 
#     # 질의응답
#     qa_model = pipeline("question-answering",
#                         model="deepset/roberta-base-squad2")
#     result = qa_model(
#         question="What is Python?",
#         context="Python is a programming language."
#     )
#     print(result)
# 
#     # 요약
#     summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#     result = summarizer(
#         "Long text here...",
#         max_length=150,
#         min_length=50
#     )
#     print(result[0]["summary_text"])
# 
# ================================================================================
# 실용적인 사용 예제
# ================================================================================
# 
# 예제 1: 로컬 모델로 텍스트 생성
# ----------------------------
#     from langchain.llms import HuggingFacePipeline
#     from langchain.prompts import PromptTemplate
#     from langchain.chains import LLMChain
#     from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# 
#     # 모델 로드
#     model_id = "gpt2"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id)
# 
#     # 파이프라인 생성
#     text_gen = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=100
#     )
# 
#     llm = HuggingFacePipeline(pipeline=text_gen)
# 
#     # 프롬프트 템플릿
#     prompt = PromptTemplate(
#         input_variables=["topic"],
#         template="Tell me interesting facts about {topic}:"
#     )
# 
#     # 체인 구성
#     chain = LLMChain(llm=llm, prompt=prompt)
# 
#     # 실행
#     result = chain.run(topic="Machine Learning")
#     print(result)
# 
# 예제 2: 질의응답 시스템
# ---------------------
#     from transformers import pipeline
# 
#     qa_pipeline = pipeline(
#         "question-answering",
#         model="deepset/roberta-base-squad2"
#     )
# 
#     context =

LangChain은 대규모 언어 모델을 다루기 위한 프레임워크입니다.
    이를 통해 복잡한 애플리케이션을 쉽게 개발할 수 있습니다.

# """
# 
#     questions = [
#         "LangChain이란?",
#         "LangChain의 용도는?"
#     ]
# 
#     for q in questions:
#         answer = qa_pipeline(question=q, context=context)
#         print(f"Q: {q}")
#         print(f"A: {answer['answer']}")
#         print(f"신뢰도: {answer['score']:.2%}")
# 
# 예제 3: 텍스트 요약
# ----------------
#     from transformers import pipeline
# 
#     summarizer = pipeline(
#         "summarization",
#         model="facebook/bart-large-cnn"
#     )
# 
#     long_text =

인공지능은 현재 사회의 많은 분야에 적용되고 있습니다.
    의료, 교육, 금융, 제조업 등 다양한 산업에서 AI 기술이
    사용되고 있습니다. 특히 자연어처리 기술은 챗봇, 기계번역,
    감정분석 등 다양한 애플리케이션에서 핵심 역할을 합니다.

# """
# 
#     summary = summarizer(long_text, max_length=50, min_length=25)
#     print("요약:")
#     print(summary[0]["summary_text"])
# 
# ================================================================================
# 성능 최적화 팁
# ================================================================================
# 
# 1. 모델 양자화 (Quantization)
#    메모리 사용량 감소, 속도 향상
#    코드:
#       model = AutoModelForCausalLM.from_pretrained(
#           model_id,
#           load_in_8bit=True  # 8비트 양자화
#       )
# 
# 2. 모델 정리 (Pruning)
#    불필요한 가중치 제거
#    코드:
#       from transformers import AutoModel
#       model = AutoModel.from_pretrained(model_id)
#       # 정리 수행
# 
# 3. GPU 활용
#    CPU 대비 훨씬 빠른 처리
#    코드:
#       model = AutoModelForCausalLM.from_pretrained(model_id)
#       model = model.to("cuda")
# 
# 4. 배치 처리
#    여러 샘플을 함께 처리
#    코드:
#       results = pipeline(
#           texts,  # 리스트
#           batch_size=32
#       )
# 
# 5. 캐싱
#    반복되는 요청에 대해 결과 캐싱
#    코드:
#       from functools import lru_cache
# 
#       @lru_cache(maxsize=128)
#       def process_text(text):
#           return pipeline(text)
# 
# ================================================================================
# Hugging Face 모델 선택 가이드
# ================================================================================
# 
# 작업별 추천 모델:
# 
# 텍스트 생성:
# - GPT-2: 가볍고 빠름
# - DistilGPT-2: 더 경량
# - T5-base: 다양한 작업 지원
# 
# 질의응답:
# - bert-base-cased-squad: 높은 정확도
# - distilbert-base-cased-distilled-squad: 경량
# 
# 요약:
# - facebook/bart-large-cnn: 뉴스 요약 최적
# - google/pegasus-cnn_dailymail: 뉴스 요약
# 
# 번역:
# - Helsinki-NLP/opus-mt-en-ko: 영→한 번역
# - Helsinki-NLP/opus-mt-ko-en: 한→영 번역
# 
# 감정분석:
# - distilbert-base-uncased-finetuned-sst-2-english
# - roberta-base-openai-detector
# 
# ================================================================================
# 보안 및 주의사항
# ================================================================================
# 
# 1. 토큰 보안
#    - .env 파일에 저장
#    - .gitignore에 .env 추가
#    - 절대 코드에 하드코딩 금지
# 
# 2. 모델 검증
#    - 신뢰할 수 있는 사용자의 모델만 사용
#    - 모델의 라이선스 확인
# 
# 3. 프라이버시
#    - 민감한 데이터는 로컬 모델 사용
#    - API 사용시 데이터 전송 주의
# 
# 4. 의존성 관리
#    - requirements.txt에 버전 명시
#    - 호환성 테스트
# 
# ================================================================================
# 완전한 Hugging Face 통합 예제
# ================================================================================
# 
# import os
# from dotenv import load_dotenv
# from langchain.llms import HuggingFaceHub
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# 
# load_dotenv()
# 
# # API 기반 (온라인)
# hf_llm = HuggingFaceHub(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.1",
#     model_kwargs={
#         "temperature": 0.7,
#         "max_new_tokens": 300
#     }
# )
# 
# # 프롬프트 생성
# prompt = ChatPromptTemplate.from_template(
#     "당신은 Python 전문가입니다. "
#     "다음 주제에 대해 설명해주세요: {topic}"
# )
# 
# # 체인 구성
# chain = prompt | hf_llm | StrOutputParser()
# 
# # 실행
# result = chain.invoke({"topic": "LangChain의 주요 기능"})
# print(result)
# 
# ================================================================================

if __name__ == "__main__":
    print(__doc__)
