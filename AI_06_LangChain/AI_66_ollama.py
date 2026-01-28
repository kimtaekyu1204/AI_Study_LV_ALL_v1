# # AI_66_ollama

# ================================================================================
# AI_66_ollama.py
# Ollama: 로컬 환경에서 LLM 실행하는 완전 정리
# ================================================================================
# 
# Ollama란?
# Ollama는 로컬 컴퓨터에서 대형 언어 모델(LLM)을 실행할 수 있도록
# 도와주는 오픈소스 도구입니다.
# 
# 핵심 개념:
# - 인터넷 없이 AI 모델을 개인 PC에서 실행
# - 개인정보 보호와 보안성 우선
# - 설치와 사용이 매우 간단
# - 다양한 오픈소스 LLM 지원
# 
# 공식 웹사이트: https://ollama.com/
# 
# ================================================================================
# Ollama의 주요 특징
# ================================================================================
# 
# 1. 로컬 실행 지원 (Privacy First)
#    - 인터넷 연결 불필요
#    - 데이터가 외부로 나가지 않음
#    - 개인정보 보호 최우선
#    - 금융/의료/법률 등 민감한 분야에 적합
#    - 기술: MacOS, Linux, Windows (WSL2)
# 
# 2. 다양한 AI 모델 지원
#    - Llama: Meta의 대규모 언어 모델
#    - Mistral: 고성능 경량 모델
#    - Gemma: Google의 경량 모델
#    - Code Llama: 코드 생성 특화
#    - Phi: Microsoft의 경량 모델
#    - Neural Chat: 대화형 최적화
# 
# 3. 간단한 설치 및 사용
#    - 원클릭 설치
#    - 터미널에서 간단한 명령어로 모델 실행
#    - 별도 설정 불필요
# 
# 4. 모델 커스터마이제이션
#    - Modelfile을 통한 모델 수정
#    - 시스템 프롬프트 커스터마이즈
#    - 파라미터 튜닝 가능
# 
# 5. 빠른 응답 속도
#    - 로컬 실행으로 네트워크 지연 없음
#    - GPU 활용으로 빠른 처리
#    - 자체 최적화 기술 적용
# 
# ================================================================================
# Ollama vs 기타 LLM 솔루션
# ================================================================================
# 
# Ollama (로컬)
# - 장점: 프라이버시, 무료, 오프라인 사용
# - 단점: 로컬 리소스 필요, 성능 제한
# - 용도: 개발, 프로토타입, 민감 데이터
# 
# OpenAI API (클라우드)
# - 장점: 최고 성능, 최신 모델, 신뢰성
# - 단점: 비용, 인터넷 필수, 데이터 전송
# - 용도: 프로덕션, 고성능 필요
# 
# Hugging Face (로컬/클라우드)
# - 장점: 다양한 모델, 유연성
# - 단점: 설정 복잡, 의존성 많음
# - 용도: 커스터마이제이션 필요시
# 
# GPT4All (로컬)
# - 장점: 가볍고 빠름, 설치 간단
# - 단점: 모델 선택 제한, 성능 제한
# - 용도: 초보자, 가벼운 작업
# 
# 최종 선택: 용도와 환경에 따라 선택
# - 프라이버시 중요 → Ollama
# - 최고 성능 필요 → OpenAI
# - 유연성 필요 → Hugging Face
# 
# ================================================================================
# 설치 및 환경 설정
# ================================================================================
# 
# Step 1: Ollama 다운로드 및 설치
# ------------------------------
# 1. https://ollama.com/ 방문
# 2. 운영체제 선택:
#    - macOS (Intel/Apple Silicon)
#    - Linux (Ubuntu, Debian 등)
#    - Windows (WSL2 필요)
# 
# 3. 설치 파일 다운로드
#    - 윈도우: ollama-windows-amd64.exe (약 1GB)
#    - macOS: ollama-darwin-amd64.zip
#    - Linux: curl -fsSL https://ollama.ai/install.sh | sh
# 
# 4. 설치 실행
#    - Windows/Mac: 다운로드 파일 실행
#    - Linux: 위 curl 명령어 실행
# 
# 5. 설치 확인
#    터미널/커맨드에서:
#    ollama --version
#    출력: ollama version 0.x.x (또는 유사)
# 
# Step 2: 모델 다운로드 (초기 한 번만)
# ---------------------------------
# 첫 번째 사용시 모델을 다운로드해야 합니다.
# (용량: 모델에 따라 2GB ~ 50GB)
# 
# 기본 모델들 (추천):
# 1. Mistral (7B)
#    - 크기: 5GB
#    - 속도: 빠름 (권장)
#    - 성능: 좋음
# 
# 2. Llama 2 (7B)
#    - 크기: 4GB
#    - 속도: 빠름
#    - 성능: 우수
# 
# 3. Neural Chat (7B)
#    - 크기: 4GB
#    - 특징: 대화 최적화
#    - 성능: 대화형 최우수
# 
# 다운로드 예제:
#    ollama pull mistral
#    ollama pull llama2
#    ollama pull neural-chat
# 
# Step 3: 모델 실행
# ----------------
# 설치된 모델 실행:
#    ollama run mistral
# 
# 대화식 인터페이스 시작:
#    >>> 이 프롬프트에 질문 입력
#    >>> exit  (또는 Ctrl+D) 로 종료
# 
# ================================================================================
# LangChain과 Ollama 통합
# ================================================================================
# 
# 설치:
#    pip install langchain
#    pip install ollama
# 
# 기본 사용 코드:
# 
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.llms import Ollama
# 
# # Ollama 초기화 (로컬호스트:11434)
# llm = Ollama(
#     model="mistral",                           # 모델명
#     callback_manager=CallbackManager([         # 스트리밍 출력
#         StreamingStdOutCallbackHandler()
#     ]),
#     temperature=0.7,                           # 창의성 (0~1)
#     num_predict=256                            # 최대 토큰 수
# )
# 
# # 간단한 사용
# response = llm("Tell me about artificial intelligence")
# print(response)
# 
# # 프롬프트 템플릿과 함께 사용
# from langchain.prompts import PromptTemplate
# 
# prompt = PromptTemplate(
#     input_variables=["topic"],
#     template="Explain {topic} in simple terms for a beginner."
# )
# 
# chain = prompt | llm
# 
# result = chain.invoke({"topic": "Machine Learning"})
# print(result)
# 
# ================================================================================
# Ollama 모델 커스터마이제이션 (Modelfile)
# ================================================================================
# 
# Modelfile이란?
# LLM의 파라미터와 동작을 정의하는 파일입니다.
# 기존 모델을 수정하여 새로운 모델을 만들 수 있습니다.
# 
# 기본 구조:
# FROM <base_model>      # 기본 모델 지정
# SYSTEM <prompt>        # 시스템 프롬프트
# PARAMETER <key> <val>  # 파라미터 설정
# 
# 예제 1: 한국어 전문가 모델
# -----------------------
# Modelfile 생성:
#    FROM mistral
# 
#    SYSTEM \"\"\"
#    당신은 한국어 AI 어시스턴트입니다.
#    한국 문화, 역사, 언어에 깊은 이해를 가지고 있습니다.
#    모든 답변을 한국어로 상세하게 제공하세요.
#    \"\"\"
# 
#    PARAMETER temperature 0.5
#    PARAMETER num_predict 512
# 
# 모델 생성:
#    ollama create korean-expert -f Modelfile
# 
# 모델 실행:
#    ollama run korean-expert
# 
# 예제 2: 데이터 분석 전문가 모델
# -----------------------------
# Modelfile:
#    FROM mistral
# 
#    SYSTEM \"\"\"
#    당신은 데이터 분석 전문가입니다.
#    다음과 같은 역할을 수행합니다:
#    - 데이터 분석 질문 답변
#    - Python/SQL 코드 작성
#    - 통계 분석 지원
#    - 데이터 시각화 제안
#    \"\"\"
# 
#    PARAMETER temperature 0.3
#    PARAMETER num_predict 1024
# 
# 생성:
#    ollama create data-expert -f Modelfile
# 
# 예제 3: 창의적 글쓰기 모델
# ------------------------
# Modelfile:
#    FROM llama2
# 
#    SYSTEM \"\"\"
#    당신은 창의적인 글쓰기 전문가입니다.
#    다음을 수행합니다:
#    - 소설, 시, 이야기 창작
#    - 캐릭터 개발
#    - 대사 작성
#    \"\"\"
# 
#    PARAMETER temperature 0.8
#    PARAMETER num_predict 2048
# 
# ================================================================================
# 파라미터 튜닝 가이드
# ================================================================================
# 
# temperature (0.0 ~ 1.0)
# - 0.0: 가장 결정적 (항상 같은 답변)
# - 0.5: 균형잡힌 (추천값)
# - 1.0: 가장 창의적 (무작위에 가까움)
# - 용도:
#   * 0.0~0.3: 팩트 기반 (Q&A, 분석)
#   * 0.5~0.7: 균형잡힌 대화
#   * 0.8~1.0: 창의적 작업 (글쓰기, 브레인스토밍)
# 
# num_predict (최대 토큰 수)
# - 작을수록: 빠르고 짧은 답변
# - 클수록: 자세하지만 느림
# - 추천: 256~1024
# 
# repeat_penalty (반복 페널티)
# - 높을수록: 반복 감소
# - 추천: 1.1
# 
# top_k (상위 K개 토큰 선택)
# - 작을수록: 다양성 감소
# - 클수록: 다양성 증가
# - 추천: 40
# 
# top_p (누적 확률)
# - 낮을수록: 일관성 높음
# - 높을수록: 다양성 높음
# - 추천: 0.9
# 
# ================================================================================
# Ollama 실전 예제
# ================================================================================
# 
# 예제 1: 간단한 Q&A 애플리케이션
# -----------------------------
# from langchain.llms import Ollama
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# 
# llm = Ollama(model="mistral", temperature=0.3)
# 
# # 질문 템플릿
# prompt = ChatPromptTemplate.from_template(
#     "다음 질문에 정확하게 답변하세요: {question}"
# )
# 
# # 체인
# chain = prompt | llm | StrOutputParser()
# 
# # 사용
# questions = [
#     "Python이란 무엇인가?",
#     "머신러닝의 주요 알고리즘은?",
#     "LangChain의 장점은?"
# ]
# 
# for q in questions:
#     print(f"Q: {q}")
#     answer = chain.invoke({"question": q})
#     print(f"A: {answer}\\n")
# 
# 예제 2: 대화형 챗봇
# -----------------
# from langchain.llms import Ollama
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain
# 
# llm = Ollama(model="neural-chat")
# 
# memory = ConversationBufferMemory()
# 
# conversation = ConversationChain(
#     llm=llm,
#     memory=memory,
#     verbose=True
# )
# 
# # 대화 진행
# print("챗봇과 대화하세요 (quit로 종료):")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "quit":
#         break
# 
#     response = conversation.predict(input=user_input)
#     print(f"Bot: {response}")
# 
# 예제 3: 문서 요약
# ---------------
# from langchain.llms import Ollama
# from langchain.prompts import PromptTemplate
# 
# llm = Ollama(model="mistral")
# 
# prompt = PromptTemplate(
#     template="다음 텍스트를 3줄로 요약하세요:\\n\\n{text}",
#     input_variables=["text"]
# )
# 
# chain = prompt | llm
# 
# # 긴 텍스트
# long_text =

LangChain은 대규모 언어 모델을 더 효율적이고 유연하게 활용할 수 있도록
설계된 프레임워크입니다. 다양한 LLM 모델과 외부 데이터를 쉽게 통합하여
복잡한 애플리케이션을 개발할 수 있습니다. RAG, 메모리, 에이전트 등
강력한 기능을 제공합니다.

# summary = chain.invoke({"text": long_text})
# print(summary)
# 
# ================================================================================
# 성능 최적화 팁
# ================================================================================
# 
# 1. GPU 활용
#    - NVIDIA GPU: CUDA 지원
#    - AMD GPU: ROCm 지원
#    - Metal (Mac): 자동 지원
#    설정: 자동 인식되므로 별도 설정 불필요
# 
# 2. 모델 선택
#    - 7B 모델: 빠르고 가벼움 (추천)
#    - 13B 모델: 더 나은 성능
#    - 70B 모델: 최고 성능 (고사양 필요)
# 
# 3. 메모리 관리
#    - 사용하지 않는 모델은 언로드
#    - 여러 모델 동시 실행 피하기
#    - RAM 확인: 최소 8GB (권장 16GB+)
# 
# 4. 배치 처리
#    - 여러 요청을 한 번에 처리
#    - API 호출 오버헤드 감소
# 
# 5. 캐싱
#    - 같은 입력에 대한 반복 호출 최소화
#    - 결과 저장 및 재사용
# 
# ================================================================================
# 문제 해결 및 FAQ
# ================================================================================
# 
# Q1: 모델 다운로드가 너무 느림
# A: 네트워크 연결 확인, 저장공간 충분한지 확인
# 
# Q2: 메모리 부족 오류
# A: 더 작은 모델 사용 (7B), RAM 증설, 배경 프로세스 종료
# 
# Q3: GPU 미인식
# A: 드라이버 업데이트, CUDA/ROCm 설치 확인
# 
# Q4: Windows에서 작동 안함
# A: WSL2 필요, WSL2 설치 후 Linux 버전으로 설치
# 
# Q5: 응답이 이상함
# A: temperature 조정, 다른 모델 시도, 프롬프트 개선
# 
# ================================================================================
# 보안 및 주의사항
# ================================================================================
# 
# 1. 로컬 데이터 보호
#    - 모든 데이터가 로컬에만 저장
#    - 외부 전송 없음 (기본값)
#    - 민감한 정보 처리에 적합
# 
# 2. 모델 검증
#    - 신뢰할 수 있는 소스에서만 모델 다운로드
#    - ollama.com에서 권장 모델 사용
# 
# 3. 리소스 관리
#    - 필요한 모델만 설치
#    - 디스크 공간 충분히 확보
#    - CPU/GPU 부하 모니터링
# 
# 4. API 보안
#    - Ollama API를 외부 노출 금지
#    - localhost:11434만 사용 (기본)
# 
# ================================================================================
# Ollama 전체 워크플로우
# ================================================================================
# 
# 1. Ollama 설치
#    → https://ollama.com/ 다운로드 및 실행
# 
# 2. 모델 다운로드
#    → ollama pull mistral
# 
# 3. 모델 테스트
#    → ollama run mistral
# 
# 4. LangChain 설정
#    → from langchain.llms import Ollama
# 
# 5. 애플리케이션 개발
#    → RAG, 챗봇, Q&A 시스템 등 구축
# 
# 6. 배포
#    → API로 서빙, 또는 스탠드얼론 애플리케이션
# 
# ================================================================================

if __name__ == "__main__":
    print(__doc__)

# # 추가 정보

print("\\n" + "="*80)
    print("Ollama 시작하기")
    print("="*80 + "\\n")
    print("1. https://ollama.com에서 Ollama 다운로드")
    print("2. 설치 후 터미널에서: ollama pull mistral")
    print("3. 모델 테스트: ollama run mistral")
    print("4. LangChain 통합하여 애플리케이션 개발")
    print("\\n더 알아보기:")
    print("- 공식 모델: ollama.com/library")
    print("- API 문서: ollama.com/api")
    print("- GitHub: github.com/ollama/ollama")
