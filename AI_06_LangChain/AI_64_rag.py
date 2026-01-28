# # AI_64_rag

# ================================================================================
# AI_64_rag.py
# RAG (Retrieval-Augmented Generation) 검색 증강 생성 완전 정리
# ================================================================================
# 
# RAG란?
# RAG (Retrieval-Augmented Generation)는 검색 증강 생성 기법으로,
# 외부 지식 기반(문서, 데이터베이스, FAQ 등)을 활용하여
# LLM이 더 정확하고 맥락에 맞는 응답을 생성하는 기술입니다.
# 
# 핵심 아이디어:
# - LLM의 사전 학습된 지식만으로는 부족
# - 실시간 정보나 특정 도메인 데이터 필요
# - 검색 + 생성을 결합하여 신뢰성 높은 답변 제공
# 
# ================================================================================
# RAG의 작동 원리 (2단계)
# ================================================================================
# 
# 1단계: Retrieval (검색/정보 검색)
# ---------------------------------------
# 목표: 사용자 질문과 관련된 정보를 외부 지식 기반에서 검색
# 
# 과정:
# 1. 사용자의 질문을 벡터로 변환 (임베딩)
# 2. 벡터 데이터베이스에서 관련 문서 검색
# 3. 유사도가 높은 문서 상위 K개 추출
# 4. 검색된 문서를 컨텍스트로 준비
# 
# 코드 예제:
#     from langchain.vectorstores import FAISS
#     from langchain.embeddings import OpenAIEmbeddings
# 
#     # 임베딩 모델 초기화
#     embeddings = OpenAIEmbeddings()
# 
#     # 벡터 데이터베이스에서 검색
#     query = "LangChain에서 벡터 데이터베이스의 역할은?"
#     relevant_docs = vector_store.similarity_search(query, k=3)
# 
#     for doc in relevant_docs:
#         print(doc.page_content)
# 
# 2단계: Augmented Generation (생성 증강)
# ----------------------------------------
# 목표: 검색된 정보를 기반으로 정확한 답변 생성
# 
# 과정:
# 1. 검색된 문서를 LLM 프롬프트에 컨텍스트로 추가
# 2. LLM이 검색된 정보와 자신의 지식을 조합
# 3. 정확하고 신뢰할 수 있는 답변 생성
# 
# 코드 예제:
#     from langchain.prompts import PromptTemplate
#     from langchain_openai import ChatOpenAI
#     from langchain.chains import RetrievalQA
# 
#     llm = ChatOpenAI(model="gpt-3.5-turbo")
# 
#     # 검색 + 생성 체인
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",  # 검색된 모든 문서를 컨텍스트에 포함
#         retriever=vector_store.as_retriever()
#     )
# 
#     # 실행
#     answer = qa_chain.run("LangChain에서 벡터 데이터베이스의 역할은?")
#     print(answer)
# 
# ================================================================================
# RAG의 주요 장점
# ================================================================================
# 
# 1. 정확성 향상
#    - 모델이 기억 기반이 아닌 검색된 최신 정보 사용
#    - 할루시네이션(거짓 정보 생성) 감소
#    - 신뢰할 수 있는 답변 제공
# 
# 2. 도메인 특화
#    - 특정 도메인에 최적화된 데이터 검색 가능
#    - 회사 고유 문서나 규정 기반 답변
#    - 전문 분야의 정확한 정보 활용
# 
# 3. 대규모 데이터 활용
#    - 사전 학습된 모델의 제한된 지식을 극복
#    - 수백만 개의 문서에서 동적으로 정보 검색
#    - 실시간 데이터 업데이트 반영 가능
# 
# 4. 비용 효율성
#    - 모든 데이터를 학습할 필요 없음
#    - 필요한 정보만 선택적으로 검색
#    - 모델 파인튜닝 비용 절감
# 
# 5. 투명성과 신뢰성
#    - 답변의 근거가 되는 문서 제시 가능
#    - 사용자가 출처 확인 가능
#    - 설명 가능한 AI (XAI) 구현
# 
# ================================================================================
# LangChain에서 RAG 구현 - 전체 파이프라인
# ================================================================================
# 
# Step 1: 문서 로딩 (Document Loading)
# ------------------------------------
# 다양한 형식의 문서를 로드합니다.
# 
# 코드:
#     from langchain.document_loaders import PyPDFLoader
#     from langchain.document_loaders import TextLoader
#     from langchain.document_loaders import WebBaseLoader
# 
#     # PDF 로드
#     pdf_loader = PyPDFLoader("document.pdf")
#     pdf_docs = pdf_loader.load()
# 
#     # 텍스트 파일 로드
#     text_loader = TextLoader("notes.txt")
#     text_docs = text_loader.load()
# 
#     # 웹 페이지 로드
#     web_loader = WebBaseLoader("https://example.com/page")
#     web_docs = web_loader.load()
# 
#     # 모든 문서 합치기
#     all_docs = pdf_docs + text_docs + web_docs
# 
# Step 2: 문서 분할 (Document Splitting)
# --------------------------------------
# 긴 문서를 작은 청크로 분할합니다.
# 
# 이유:
# - LLM의 토큰 제한 준수
# - 검색 정확도 향상
# - 효율적인 처리
# 
# 코드:
#     from langchain.text_splitters import RecursiveCharacterTextSplitter
# 
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,      # 청크 크기
#         chunk_overlap=200     # 청크 간 겹침
#     )
# 
#     split_docs = text_splitter.split_documents(all_docs)
# 
# Step 3: 임베딩 (Embedding)
# -------------------------
# 문서 텍스트를 벡터(수치 표현)로 변환합니다.
# 
# 코드:
#     from langchain.embeddings import OpenAIEmbeddings
#     # 또는 Hugging Face 사용
#     from langchain.embeddings import HuggingFaceEmbeddings
# 
#     # OpenAI 임베딩
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# 
#     # Hugging Face 임베딩
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
#     )
# 
# Step 4: 벡터 데이터베이스 생성 (Vector Store)
# --------------------------------------------
# 임베딩된 문서를 벡터 데이터베이스에 저장합니다.
# 
# 지원되는 벡터 DB:
# - FAISS (Facebook AI Similarity Search) - 로컬, 무료
# - Pinecone - 클라우드 기반
# - Weaviate - 오픈소스
# - Chroma - 경량
# - Milvus - 대규모 데이터
# 
# 코드:
#     from langchain.vectorstores import FAISS
# 
#     # 벡터 데이터베이스 생성 및 저장
#     vector_store = FAISS.from_documents(
#         documents=split_docs,
#         embedding=embeddings
#     )
# 
#     # 저장
#     vector_store.save_local("faiss_index")
# 
#     # 로드
#     vector_store = FAISS.load_local(
#         "faiss_index",
#         embeddings
#     )
# 
# Step 5: 검색기 생성 (Retriever)
# ------------------------------
# 벡터 데이터베이스로부터 정보를 검색하는 인터페이스
# 
# 코드:
#     retriever = vector_store.as_retriever(
#         search_type="similarity",        # 유사도 검색
#         search_kwargs={"k": 3}           # 상위 3개 문서 반환
#     )
# 
#     # 검색 테스트
#     relevant_docs = retriever.get_relevant_documents(
#         "LangChain의 주요 기능은?"
#     )
# 
# Step 6: QA 체인 생성 (Question-Answering Chain)
# ----------------------------------------------
# 검색된 문서를 기반으로 답변을 생성합니다.
# 
# 코드:
#     from langchain.chains import RetrievalQA
#     from langchain_openai import ChatOpenAI
# 
#     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# 
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",           # 검색된 모든 문서 포함
#         retriever=retriever,
#         return_source_documents=True  # 출처 문서 반환
#     )
# 
# ================================================================================
# 문서 처리 전략 (Chain Types)
# ================================================================================
# 
# LangChain에서 검색된 문서를 처리하는 다양한 방식:
# 
# 1. Stuff
#    - 검색된 모든 문서를 프롬프트에 포함
#    - 간단하지만 토큰 제한이 있을 수 있음
#    - 문서가 적을 때 권장
# 
# 2. Refine
#    - 문서를 순회하며 답변을 점진적으로 개선
#    - 첫 번째 문서로 답변 생성
#    - 이후 각 문서로 답변 개선
#    - 토큰 효율적, 시간이 많이 걸림
# 
# 3. Map Reduce
#    - 각 문서에 대해 독립적으로 처리
#    - 결과를 종합하여 최종 답변 생성
#    - 병렬 처리 가능, 일관성 문제 가능
# 
# 4. Map Re-Rank
#    - 각 문서에 대해 LLM이 점수 매김
#    - 점수 기반으로 재정렬
#    - 가장 관련성 높은 문서 선택
#    - 정확도 높지만 비용 증가
# 
# 사용 예제:
#     # Stuff 방식 (권장)
#     qa_stuff = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever
#     )
# 
#     # Refine 방식
#     qa_refine = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="refine",
#         retriever=retriever
#     )
# 
# ================================================================================
# RAG 성능 최적화 팁
# ================================================================================
# 
# 1. 청크 크기 조정
#    - 너무 작으면: 문맥 부족
#    - 너무 크면: 검색 성능 저하
#    - 추천: 500-1500 토큰
# 
# 2. 검색 문서 개수 최적화
#    - k 파라미터 조정 (보통 3-5개)
#    - 너무 많으면 관련 없는 정보 포함
# 
# 3. 임베딩 모델 선택
#    - 도메인 특화 모델 고려
#    - 다국어 지원 필요시 다국어 모델 선택
# 
# 4. 프롬프트 엔지니어링
#    - 검색된 정보 활용 명시
#    - 답변 형식 지정
#    - 따라야 할 규칙 명시
# 
# 5. 검색 전 전처리
#    - 쿼리 정규화
#    - 철저한 문서 정제
# 
# ================================================================================
# RAG 실제 활용 사례
# ================================================================================
# 
# 1. 고객 지원 챗봇
#    - FAQ 문서 기반 자동 답변
#    - 제품 설명서에서 정보 검색
#    - 고객 만족도 향상
# 
# 2. 기술 문서 검색
#    - API 문서, 개발 가이드 검색
#    - 개발자의 검색 시간 단축
#    - 정확한 기술 정보 제공
# 
# 3. 법률 및 규정 자문
#    - 법률 문서, 판례 검색
#    - 규정 변경 반영
#    - 정확한 법적 정보 제공
# 
# 4. 의료 정보 시스템
#    - 환자 기록 검색
#    - 의료 논문, 진료 가이드라인 활용
#    - 정확한 진단 보조
# 
# 5. 재무 분석
#    - 기업 보고서, 뉴스 검색
#    - 시장 데이터 활용
#    - 신뢰할 수 있는 재무 분석
# 
# ================================================================================
# 주요 개념 정리
# ================================================================================
# 
# - Retrieval: 질문과 관련된 정보를 외부 데이터에서 검색
# - Augmentation: 검색된 정보를 프롬프트에 추가하여 생성 강화
# - Generation: LLM이 증강된 정보를 기반으로 답변 생성
# - Embedding: 텍스트를 벡터로 변환
# - Vector Store: 벡터들을 저장하는 데이터베이스
# - Retriever: 벡터 스토어에서 정보를 검색하는 인터페이스
# - Chunk: 문서를 분할한 작은 단위
# 
# ================================================================================
# 완전한 RAG 파이프라인 예제 코드
# ================================================================================
# 
# import os
# from dotenv import load_dotenv
# from langchain.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain.text_splitters import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain_openai import ChatOpenAI
# 
# load_dotenv()
# 
# # 1. 문서 로드
# loader = DirectoryLoader(
#     "./documents",
#     glob="*.pdf",
#     loader_cls=PyPDFLoader
# )
# documents = loader.load()
# 
# # 2. 문서 분할
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )
# split_docs = splitter.split_documents(documents)
# 
# # 3. 임베딩
# embeddings = OpenAIEmbeddings()
# 
# # 4. 벡터 데이터베이스
# vector_store = FAISS.from_documents(split_docs, embeddings)
# 
# # 5. 검색기
# retriever = vector_store.as_retriever(
#     search_kwargs={"k": 3}
# )
# 
# # 6. LLM 및 QA 체인
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True
# )
# 
# # 질문 실행
# query = "LangChain의 주요 기능은 무엇인가?"
# result = qa_chain({"query": query})
# 
# print("답변:", result["result"])
# print("\\n출처 문서:")
# for doc in result["source_documents"]:
#     print(f"- {doc.metadata['source']}")
# 
# ================================================================================

if __name__ == "__main__":
    print(__doc__)
