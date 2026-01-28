# # AI_63_model_io

# ================================================================================
# AI_63_model_io.py
# LangChain Model IO (모델 입출력) 모듈 상세 정리
# ================================================================================
# 
# Model IO란?
# Model IO는 LangChain에서 LLM과의 상호작용을 관리하는 핵심 모듈입니다.
# 사용자의 입력을 적절한 형식의 프롬프트로 변환하고, LLM의 출력을 파싱하는
# 일련의 과정을 담당합니다.
# 
# 주요 구성:
# - Language Models (언어 모델)
# - Chat Models (채팅 모델)
# - Prompt Templates (프롬프트 템플릿)
# - Output Parsers (출력 파서)
# 
# ================================================================================
# Language Models vs Chat Models
# ================================================================================
# 
# Language Models (LM)
# - 텍스트 입력을 받아 텍스트 출력 생성
# - 과거 모델들의 표준 방식
# - 프롬프트: 문자열 (str)
# - 응답: 텍스트 (str)
# 
# Chat Models (CM)
# - 메시지 목록을 입력으로 받음
# - 역할(role) 기반 구조: system, user, assistant
# - 대화형 AI에 최적화
# - 프롬프트: 메시지 목록 (List[BaseMessage])
# - 응답: 메시지 (BaseMessage)
# 
# 현대적 추세: Chat Models를 주로 사용
# 
# 코드 예제:
# 
# # Language Model 사용
# from langchain.llms import OpenAI
# llm = OpenAI(temperature=0.7)
# response = llm.invoke("What is AI?")
# 
# # Chat Model 사용 (권장)
# from langchain_openai import ChatOpenAI
# chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
# response = chat.invoke("What is AI?")
# 
# ================================================================================
# 프롬프트 템플릿 (Prompt Templates)
# ================================================================================
# 
# 프롬프트 템플릿이란?
# 사용자 입력에 따라 동적으로 프롬프트를 생성하는 템플릿입니다.
# 하드코딩된 프롬프트 대신, 변수를 포함한 템플릿을 정의하여 재사용성을 높입니다.
# 
# 목적:
# - 프롬프트 재사용성 향상
# - 동적 입력에 따른 유연한 프롬프트 생성
# - 프롬프트 엔지니어링의 표준화
# - 복잡한 다단계 프롬프트 관리
# 
# 1. PromptTemplate (문자열 기반)
#    기본적인 프롬프트 템플릿
#    코드:
#       from langchain.prompts import PromptTemplate
# 
#       # 간단한 템플릿
#       template = "Write a poem about {topic} in {style} style."
#       prompt = PromptTemplate(
#           input_variables=["topic", "style"],
#           template=template
#       )
# 
#       # 프롬프트 생성
#       formatted = prompt.format(topic="AI", style="modern")
#       print(formatted)
#       # 출력: Write a poem about AI in modern style.
# 
# 2. ChatPromptTemplate (메시지 기반)
#    Chat Models를 위한 프롬프트 템플릿
#    여러 역할(role)의 메시지를 조합
#    코드:
#       from langchain.prompts import ChatPromptTemplate
# 
#       # Chat 모델용 템플릿
#       template = ChatPromptTemplate.from_messages([
#           ("system", "You are a helpful {role}."),
#           ("user", "{input}")
#       ])
# 
#       # 동적 프롬프트 생성
#       messages = template.format_messages(
#           role="data analyst",
#           input="What are the trends?"
#       )
# 
# 3. 프롬프트 템플릿 팩토리 (Factory Methods)
#    편리한 방법으로 템플릿 생성
#    코드:
#       # from_template 사용 (권장)
#       template = ChatPromptTemplate.from_template(
#           "Tell me a {length} story about {topic}"
#       )
# 
#       # from_messages 사용 (복잡한 구조)
#       template = ChatPromptTemplate.from_messages([
#           ("system", "You are a helpful assistant"),
#           ("user", "{input}"),
#           ("assistant", "{previous_response}"),
#           ("user", "{follow_up}")
#       ])
# 
# ================================================================================
# 출력 파서 (Output Parsers)
# ================================================================================
# 
# 출력 파서란?
# LLM의 출력(문자열)을 구조화된 형식으로 변환합니다.
# LLM이 생성한 텍스트를 Python 객체(dict, list, custom objects)로 파싱합니다.
# 
# 목적:
# - LLM 출력을 프로그래밍 가능한 형식으로 변환
# - 예측 불가능한 텍스트 출력을 구조화
# - 다운스트림 작업(후속 처리)을 위한 데이터 준비
# 
# 주요 파서 타입:
# 
# 1. StrOutputParser
#    LLM 출력을 문자열로 반환 (기본값)
#    코드:
#       from langchain_core.output_parsers import StrOutputParser
# 
#       chain = prompt | llm | StrOutputParser()
#       result = chain.invoke({"topic": "Python"})
#       # result는 문자열
# 
# 2. JsonOutputParser
#    JSON 형식의 출력을 파싱
#    코드:
#       from langchain.output_parsers import JsonOutputParser
#       from pydantic import BaseModel
# 
#       class PersonInfo(BaseModel):
#           name: str
#           age: int
#           hobbies: list[str]
# 
#       parser = JsonOutputParser(pydantic_object=PersonInfo)
# 
#       # 프롬프트에 파싱 지시사항 추가
#       prompt = PromptTemplate(
#           template="{format_instructions}\n{query}",
#           input_variables=["query"],
#           partial_variables={
#               "format_instructions": parser.get_format_instructions()
#           }
#       )
# 
#       chain = prompt | llm | parser
#       result = chain.invoke({"query": "Tell me about John, 25 years old..."})
#       # result는 PersonInfo 객체
# 
# 3. PydanticOutputParser
#    Pydantic 모델로 파싱 (구조화된 데이터)
#    코드:
#       from langchain.output_parsers import PydanticOutputParser
#       from pydantic import BaseModel, Field
# 
#       class ReviewSchema(BaseModel):
#           rating: int = Field(ge=1, le=5)
#           summary: str
#           pros: list[str]
#           cons: list[str]
# 
#       parser = PydanticOutputParser(pydantic_object=ReviewSchema)
# 
#       prompt = PromptTemplate(
#           template="{format_instructions}\nReview: {review_text}",
#           input_variables=["review_text"],
#           partial_variables={
#               "format_instructions": parser.get_format_instructions()
#           }
#       )
# 
#       chain = prompt | llm | parser
#       result = chain.invoke({"review_text": "Great product..."})
#       # result.rating, result.summary 등으로 접근
# 
# 4. CommaSeparatedListOutputParser
#    쉼표로 구분된 리스트로 파싱
#    코드:
#       from langchain.output_parsers import CommaSeparatedListOutputParser
# 
#       parser = CommaSeparatedListOutputParser()
#       prompt = PromptTemplate(
#           template="{format_instructions}\n{query}",
#           input_variables=["query"],
#           partial_variables={
#               "format_instructions": parser.get_format_instructions()
#           }
#       )
# 
#       chain = prompt | llm | parser
#       result = chain.invoke({"query": "List 5 programming languages"})
#       # result는 ['Python', 'JavaScript', 'Java', 'C++', 'Go']
# 
# ================================================================================
# 프롬프트 엔지니어링 팁
# ================================================================================
# 
# 1. 명확한 지시사항
#    애매한 프롬프트 보다 구체적이고 명확한 프롬프트 작성
#    코드:
#       # 나쁜 예
#       prompt = "What is Python?"
# 
#       # 좋은 예
#       prompt = PromptTemplate(
#           template="Explain Python programming language in {detail_level} detail. "
#                    "Include examples and use cases.",
#           input_variables=["detail_level"]
#       )
# 
# 2. 역할 정의 (Role Definition)
#    LLM의 역할을 명시적으로 정의
#    코드:
#       template = ChatPromptTemplate.from_messages([
#           ("system", "You are an expert Python programmer with 10+ years experience. "
#                      "Provide clear, practical explanations."),
#           ("user", "{question}")
#       ])
# 
# 3. 상황 설정 (Context Setting)
#    배경 정보와 예제 제공
#    코드:
#       template =

{context}

      Answer the question: {question}

      Example response format:
      [Your answer here]

# """
# 
# 4. 체인-오브-생각 (Chain-of-Thought)
#    단계별로 생각하도록 유도
#    코드:
#       prompt = PromptTemplate(
#           template="Let's solve this step by step:\n"
#                    "1. First, analyze: {question}\n"
#                    "2. Then, calculate: [steps]\n"
#                    "3. Finally, provide the answer:",
#           input_variables=["question"]
#       )
# 
# ================================================================================
# Model IO 워크플로우 (LCEL - LangChain Expression Language)
# ================================================================================
# 
# LCEL은 LangChain의 파이프라인 문법입니다.
# | 연산자로 단계들을 연결합니다.
# 
# 기본 구조:
#     prompt | llm | output_parser
# 
# 예제 1: 간단한 Q&A 체인
#     from langchain_openai import ChatOpenAI
#     from langchain.prompts import ChatPromptTemplate
#     from langchain_core.output_parsers import StrOutputParser
# 
#     llm = ChatOpenAI(model="gpt-3.5-turbo")
#     prompt = ChatPromptTemplate.from_template(
#         "Answer this question: {question}"
#     )
#     output_parser = StrOutputParser()
# 
#     chain = prompt | llm | output_parser
#     result = chain.invoke({"question": "What is machine learning?"})
#     print(result)
# 
# 예제 2: 구조화된 데이터 파싱
#     from pydantic import BaseModel, Field
# 
#     class QuestionAnswer(BaseModel):
#         question: str
#         answer: str
#         difficulty: str = Field(description="easy, medium, or hard")
# 
#     parser = PydanticOutputParser(pydantic_object=QuestionAnswer)
# 
#     prompt = ChatPromptTemplate.from_template(
#         "{format_instructions}\n{input}"
#     ).partial(format_instructions=parser.get_format_instructions())
# 
#     chain = prompt | llm | parser
#     result = chain.invoke({"input": "Q: What is AI?"})
#     print(f"Q: {result.question}")
#     print(f"A: {result.answer}")
#     print(f"Level: {result.difficulty}")
# 
# ================================================================================
# 주요 개념 정리
# ================================================================================
# 
# - Prompt: LLM에 전달하는 지시사항 및 입력
# - Template: 변수를 포함한 프롬프트 템플릿
# - Variables: 프롬프트에서 동적으로 채워지는 값
# - Parser: LLM 출력을 구조화된 형식으로 변환
# - Chain: 여러 구성요소를 파이프라인으로 연결
# - LCEL: LangChain Expression Language (| 연산자 사용)
# 
# ================================================================================
# 최적 사용 사례
# ================================================================================
# 
# 1. 다양한 질문에 동일한 프롬프트 구조 재사용
# 2. LLM 출력을 데이터베이스에 저장하기 전에 검증
# 3. 복잡한 응답을 구조화된 데이터로 변환
# 4. 여러 LLM의 출력 형식 통일
# 5. 프롬프트 버전 관리 및 A/B 테스트
# 
# ================================================================================

# 실제 사용 예제

if __name__ == "__main__":
    print(__doc__)

# # 간단한 예제 실행 코드 (API 키 필요)

print("\n" + "="*80)
    print("Model IO 실행 예제 (주석 처리 - 실제 사용시 활성화)")
    print("="*80 + "\n")

# # 실행 코드 (주석 처리)

# """
#     import os
#     from dotenv import load_dotenv
#     from langchain_openai import ChatOpenAI
#     from langchain.prompts import ChatPromptTemplate
#     from langchain_core.output_parsers import StrOutputParser
# 
#     # 환경 변수 로드
#     load_dotenv()
# 
#     # LLM 초기화
#     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
# 
#     # 프롬프트 템플릿 생성
#     prompt = ChatPromptTemplate.from_template(
#         "너는 {role}이다. 사용자의 질문에 답변해주세요: {question}"
#     )
# 
#     # 출력 파서
#     parser = StrOutputParser()
# 
#     # 체인 구성
#     chain = prompt | llm | parser
# 
#     # 실행
#     result = chain.invoke({
#         "role": "AI 교육 전문가",
#         "question": "LangChain의 Model IO는 무엇인가?"
#     })
# 
#     print("답변:")
#     print(result)

print("\n예제 코드:")
    print("- ChatPromptTemplate로 동적 프롬프트 생성")
    print("- StrOutputParser로 텍스트 출력")
    print("- | 연산자로 파이프라인 구성")
