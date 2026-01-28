# NLP Step 4: 문자열, 파일, 정규표현식, CSV 처리

## 작업 개요
AI 자연어처리 수업의 Step 4 (Python 기초 - 문자열 및 파일 처리) 자료를 정리한 프로젝트입니다.
ipynb 파일의 핵심 개념과 코드를 추출하여 체계적인 .py 파일로 재정리했습니다.

---

## 생성된 파일 목록

### 1. NLP_12_string_functions.py
**주제**: 문자열 함수(메소드) 학습

#### 포함된 내용:
- `upper()`, `lower()` - 대소문자 변환
- `strip()` - 공백 제거 (lstrip, rstrip)
- `replace()` - 문자열 치환 및 메소드 체이닝
- `split()`, `join()` - 문자열 분할과 결합
- `index()`, `find()` - 문자열 위치 찾기
- `count()` - 부분 문자열 개수 세기
- `startswith()`, `endswith()` - 시작/종료 문자열 확인
- `ord()`, `chr()` - 문자와 코드값 변환
- `isalpha()`, `isdigit()`, `isalnum()` - 문자 타입 확인

#### 실전 예제:
- 등장 알파벳 개수 세기 (딕셔너리 컴프리헨션)
- 가장 긴 단어 찾기
- 각 단어의 첫글자 대문자로 만들기
- 주민등록번호 마스킹

**파일 크기**: 14 KB
**코드 라인수**: ~450 줄

---

### 2. NLP_13_file_handling.py
**주제**: 파일 입출력 (File I/O) 처리

#### 포함된 내용:
- 파일 열기 모드 (r, w, a) 및 개념
- `open()`, `close()` 함수
- `write()` - 파일에 쓰기
- `readline()` - 한 줄 읽기
- `readlines()` - 모든 줄을 리스트로 읽기
- `read()` - 전체 파일 읽기
- `with` 구문 - 자동 파일 닫기 (Context Manager)
- 경로 관련 함수 (`os.path.join()`, `os.path.exists()`, `os.mkdir()`)
- 인코딩 (utf-8, cp949, euc-kr) 처리
- `errors` 매개변수를 사용한 인코딩 오류 처리

#### 실전 예제:
- 디렉토리 생성 및 파일 쓰기
- 파일 읽기 및 처리
- 다양한 인코딩으로 파일 저장
- 안전한 파일 처리 (예외 처리 포함)

**파일 크기**: 14 KB
**코드 라인수**: ~450 줄

---

### 3. NLP_14_regex.py
**주제**: 정규표현식 (Regular Expression)

#### 포함된 내용:
- 정규표현식 기본 개념 및 특징
- 메타 문자 (^, $, ., *, +, ?, [], {}, (), |, \)
- 특수 표현 (\s, \S, \w, \W, \d, \D)
- `re.match()` - 처음부터 패턴 매칭
- `re.search()` - 어디든 패턴 찾기
- `re.findall()` - 모든 패턴 찾기
- `re.sub()` - 패턴 기반 치환
- `re.split()` - 패턴 기반 분할
- Match 객체 메소드 (group, start, end, span)
- 문자 클래스 및 반복 표현

#### 실전 예제:
- 개인정보 마스킹 (주민등록번호, 전화번호, 이메일)
- 가격 정보 추출
- 이메일 형식 검증
- URL 형식 검증
- 전화번호 형식 검증
- 날짜 형식 (YYYY-MM-DD) 검증
- 그룹화를 이용한 부분 추출
- 패턴 미리 컴파일 (성능 최적화)

**파일 크기**: 16 KB
**코드 라인수**: ~550 줄

---

### 4. NLP_15_csv_file.py
**주제**: CSV 파일 처리

#### 포함된 내용:
- CSV 파일 개념 및 특징
- TSV, PSV 파일 형식
- CSV 파일 읽기 (텍스트 파일로, csv 모듈로)
- 타이타닉 데이터셋 설명
- `sort()` vs `sorted()` 함수 비교
- 딕셔너리 정렬 (items(), lambda)
- 객실 등급별 생존률 계산
- CSV 파일 쓰기
- csv 모듈을 사용한 전문적 처리

#### 실전 예제:
- 객실 등급별 생존률 분석
- CSV 데이터 필터링 (생존한 승객)
- 연령대별 생존률 계산
- 필터링된 데이터를 새 CSV 파일로 저장
- 람다 함수를 활용한 복잡한 정렬

**파일 크기**: 15 KB
**코드 라인수**: ~500 줄

---

### 5. 원본 ipynb 파일들
각 주제별로 원본 Jupyter Notebook 파일도 함께 제공됩니다:

- `[원본] 12_string_functions.ipynb` (27 KB)
- `[원본] 13_file_handling.ipynb` (16 KB)
- `[원본] 14_regex.ipynb` (15 KB)
- `[원본] 15_csv_file.ipynb` (94 KB)

---

## 파일 특징

### 구성 방식
1. **개념 설명**: 각 주제의 정의, 문법, 특징을 한국어 주석으로 상세히 기술
2. **메타 정보 블록**: 각 섹션마다 들여쓰기된 문자열 주석으로 구조화된 정보 제공
3. **실행 가능한 코드**: 각 예제는 함수로 정의되어 독립적으로 실행 가능
4. **체계적인 구성**: 기초부터 심화까지 단계적 학습 가능

### 학습 순서 권장
1. **NLP_12_string_functions.py** → 기본 문자열 처리 방법
2. **NLP_13_file_handling.py** → 파일 입출력 기초
3. **NLP_14_regex.py** → 고급 패턴 매칭
4. **NLP_15_csv_file.py** → 실제 데이터 처리 (통합)

---

## 핵심 학습 포인트

### 문자열 처리
- 메소드 체이닝 활용
- 문자열 불변성 이해
- 효율적인 문자열 조작

### 파일 처리
- `with` 구문의 중요성
- 인코딩 문제 해결
- 대용량 파일 효율적 처리

### 정규표현식
- 패턴 매칭의 강력함
- 데이터 검증 및 추출
- 문자열 전처리

### 데이터 분석
- CSV 파일 처리
- 데이터 분류 및 통계
- 정렬 및 필터링

---

## 사용 예시

### Python 스크립트로 실행
```python
# NLP_12_string_functions.py 실행
python3 NLP_12_string_functions.py

# 특정 함수만 실행
from NLP_12_string_functions import longest_word
result = longest_word("I am a Student")
print(result)  # Student
```

### Jupyter Notebook에서 사용
```python
# 셀에서 import
import sys
sys.path.append('/path/to/step4_string_file')
from NLP_15_csv_file import calculate_survival_rate_by_class

# 함수 실행
result = calculate_survival_rate_by_class()
```

---

## 파일 통계

| 파일명 | 크기 | 주요 함수 수 | 예제 개수 |
|--------|------|------------|---------|
| NLP_12_string_functions.py | 14 KB | 8+ | 15+ |
| NLP_13_file_handling.py | 14 KB | 6+ | 13+ |
| NLP_14_regex.py | 16 KB | 7+ | 15+ |
| NLP_15_csv_file.py | 15 KB | 8+ | 10+ |
| **총합** | **59 KB** | **29+** | **53+** |

---

## 추가 자료

### 참고 웹사이트
- 정규표현식 학습: https://regexr.com/
- 정규표현식 시각화: https://regexper.com/
- 정규표현식 단계별 학습: https://regexone.com/

### Python 공식 문서
- csv 모듈: https://docs.python.org/3/library/csv.html
- re 모듈: https://docs.python.org/3/library/re.html
- os.path 모듈: https://docs.python.org/3/library/os.path.html

---

## 작업 완료 정보

**작업 날짜**: 2026-01-28
**생성 위치**: `/sessions/zen-busy-wozniak/mnt/[AI자연어]-20250710T115642Z-1-001/AI자연어_정리완료/contents/NLP_01_PythonBasics/step4_string_file/`

**파일 형식**:
- `.py` 파일: Python 스크립트 (실행 가능)
- `.ipynb` 파일: Jupyter Notebook (원본)
- `.md` 파일: 마크다운 문서 (이 파일)

---

## 마모

모든 파일은 다음 문체로 작성되었습니다:
- ~함 / ~한다 (설명적, 교육적)
- 한국어 주석 (이해도 향상)
- 실전 예제 중심 (실무 적용성)

---

**끝**
