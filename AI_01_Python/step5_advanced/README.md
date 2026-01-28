# NLP_01_PythonBasics - Step 5: 고급 Python 주제

## 프로젝트 개요
AI 자연어처리 수업의 Python 기초 학습 자료 중 **Step 5 (고급 주제)**에 대한 정리 자료입니다.
원본 Jupyter Notebook 파일의 핵심 개념과 코드를 Python 파일로 정리하여 빠른 학습과 참고가 가능하도록 구성했습니다.

---

## 생성된 파일 목록

### 1. NLP_16_class.py (클래스)
**학습 포인트:** 객체지향 프로그래밍의 핵심 - 클래스 개념

주요 내용:
- 클래스 정의 및 인스턴스 생성
- 생성자(`__init__`) 메서드
- 인스턴스 변수(속성)와 메서드
- 상속과 다형성
- 캡슐화 개념

**핵심 문법:**
```python
class ClassName:
    def __init__(self, param):
        self.attribute = param

    def method(self):
        return self.attribute
```

---

### 2. NLP_17_openai_api.py (OpenAI API)
**학습 포인트:** OpenAI API를 활용한 AI 모델 상호작용

주요 내용:
- API 키 환경변수 설정
- OpenAI 클라이언트 초기화
- Chat Completion API 호출
- JSON 응답 파싱
- API 에러 처리 (401, 429 등)

**핵심 문법:**
```python
import os
from dotenv import load_dotenv

api_key = os.getenv("OPENAI_API_KEY")
# API 요청: requests.post()로 HTTP 호출
```

---

### 3. NLP_18_module_package.py (모듈 & 패키지)
**학습 포인트:** Python 코드 재사용 - 모듈과 패키지

주요 내용:
- 모듈의 정의와 생성
- import 문법 (기본, from, alias)
- 패키지 구조
- `__init__.py` 파일의 역할
- `__name__` 변수와 실행 방식 구분

**핵심 문법:**
```python
# 모듈 import
import module_name
from module_name import function_name
from package.subpackage import module_name

# 모듈 실행 체크
if __name__ == '__main__':
    # 모듈이 직접 실행될 때만 수행
    pass
```

---

### 4. NLP_19_os_shutil_glob.py (파일 시스템 조작)
**학습 포인트:** 파일 및 디렉토리 조작 - os, shutil, glob 모듈

주요 내용:
- os 모듈로 파일/디렉토리 조작
- 절대경로와 상대경로
- 경로 결합 (`os.path.join()`)
- shutil로 파일 복사 및 이동
- glob으로 패턴 매칭 파일 검색

**주요 함수:**
- `os.getcwd()`: 현재 디렉토리
- `os.listdir()`: 파일 목록 조회
- `os.path.exists()`: 경로 존재 확인
- `shutil.copy()`: 파일 복사
- `glob.glob()`: 패턴 매칭 검색

---

### 5. NLP_20_exception.py (예외 처리)
**학습 포인트:** 프로그램 안정성 - 예외 처리 메커니즘

주요 내용:
- try-except 기본 구조
- 예외 타입별 처리 (SyntaxError, ValueError, FileNotFoundError 등)
- else와 finally 블록
- raise로 예외 명시적 발생
- 사용자정의 예외 클래스

**핵심 문법:**
```python
try:
    # 예외 발생 가능 코드
    risky_operation()
except ValueError as e:
    # 특정 예외 처리
    print(f"ValueError: {e}")
except Exception as e:
    # 일반 예외 처리
    print(f"Error: {e}")
else:
    # 예외 없을 때만 실행
    print("Success!")
finally:
    # 항상 실행 (정리 코드)
    cleanup()
```

---

### 6. NLP_21_iterable.py (반복가능 객체)
**학습 포인트:** 효율적 순회 - Iterable과 Iterator

주요 내용:
- Iterable 객체 (list, tuple, dict, str 등)
- Iterator와 반복자 프로토콜
- Generator 함수와 yield
- 제너레이터 표현식
- enumerate, zip, map, filter 함수

**핵심 문법:**
```python
# Generator 정의
def my_generator():
    yield 1
    yield 2
    yield 3

# 리스트 컴프리헨션
result = [x * 2 for x in range(10)]

# 제너레이터 표현식 (메모리 효율적)
result = (x * 2 for x in range(10))

# 함수 조합
result = list(map(int, ['1', '2', '3']))
result = list(filter(lambda x: x % 2 == 0, range(10)))
```

---

### 7. NLP_22_input_multiple.py (여러 값 입력)
**학습 포인트:** 사용자 입력 - 여러 값 효율적으로 받기

주요 내용:
- input() 함수와 split()
- 여러 변수에 한 번에 할당 (unpacking)
- 타입 변환 (int, float 등)
- map() 함수로 일괄 변환
- 입력 검증과 예외처리

**핵심 문법:**
```python
# 단일 값 입력
name = input("이름: ")

# 여러 값 입력 (공백 구분)
a, b, c = input("3개 입력: ").split()

# 여러 값 입력 + 타입 변환
a, b = map(int, input("두 숫자: ").split())

# 리스트로 입력
values = list(map(float, input("점수들: ").split()))

# 예외처리
try:
    num = int(input("숫자: "))
except ValueError:
    print("숫자를 입력하세요")
```

---

## 파일 구조

```
step5_advanced/
├── NLP_16_class.py                      (6.0 KB)
├── NLP_17_openai_api.py                 (3.4 KB)
├── NLP_18_module_package.py             (3.4 KB)
├── NLP_19_os_shutil_glob.py             (3.7 KB)
├── NLP_20_exception.py                  (5.0 KB)
├── NLP_21_iterable.py                   (5.0 KB)
├── NLP_22_input_multiple.py             (4.8 KB)
├── 원본_16 클래스 (class).ipynb          (93 KB)
├── 원본_17 OpanAI API.ipynb             (15 KB)
├── 원본_18 모듈 & 패키지.ipynb          (413 KB)
├── 원본_19 os shutil glob.ipynb         (11 KB)
├── 원본_20 예외.ipynb                    (30 KB)
├── 원본_21 iterable.ipynb               (18 KB)
├── 원본_22 input() 여러값 입력.ipynb      (9 KB)
└── README.md                             (이 파일)
```

---

## 사용 가이드

### 학습 방법
1. **개요 학습**: 각 .py 파일의 상단 주석 읽기 (핵심 개념)
2. **코드 분석**: 예제 코드 검토 및 이해
3. **심화 학습**: 원본 ipynb 파일 실행 및 추가 학습
4. **실습**: 배운 내용을 활용하여 프로젝트에 적용

### 원본 파일 활용
- 각 주제별 `원본_*.ipynb` 파일에 더 상세한 설명과 실행 가능한 코드 포함
- Jupyter Notebook에서 열어 인터랙티브하게 학습 가능
- 각 셀(cell)을 순차적으로 실행하며 실시간 학습 가능

---

## 학습 순서 권장

1. **NLP_16_class.py**: 객체지향 기초 (클래스 이해)
2. **NLP_17_openai_api.py**: 실제 API 활용 (프로젝트에 필수)
3. **NLP_18_module_package.py**: 코드 조직화 (프로젝트 구조)
4. **NLP_19_os_shutil_glob.py**: 파일 처리 (데이터 관리)
5. **NLP_20_exception.py**: 에러 처리 (코드 안정성)
6. **NLP_21_iterable.py**: 효율적 순회 (성능 최적화)
7. **NLP_22_input_multiple.py**: 사용자 입력 (프로젝트 기능)

---

## 주요 학습 포인트 정리

| 주제 | 핵심 키워드 | 활용도 |
|------|----------|--------|
| 클래스 | OOP, self, 상속, 다형성 | 매우 높음 |
| OpenAI API | HTTP 요청, JSON, 환경변수 | 매우 높음 |
| 모듈/패키지 | import, __name__, 재사용성 | 높음 |
| 파일 시스템 | 경로 처리, 파일 I/O | 높음 |
| 예외처리 | try-except, 안정성 | 높음 |
| Iterable | Generator, yield, 메모리효율 | 중간 |
| 입력받기 | split, unpacking, 타입변환 | 중간 |

---

## 추가 자료

- Python 공식 문서: https://docs.python.org/
- OpenAI API 가이드: https://platform.openai.com/docs/
- Real Python 튜토리얼: https://realpython.com/

---

## 생성 정보

- **생성 일시**: 2025-01-28
- **생성 도구**: Python, Jupyter Notebook 파일 분석 자동화
- **총 생성 파일**: 7개 .py + 7개 원본 ipynb
- **총 용량**: ~40 MB (원본 포함)

---

## 주의사항

1. 이 정리 자료는 **학습용** 참고 자료입니다
2. 실행 가능한 완전한 코드는 아니며, **핵심 개념만 추출**하여 정리했습니다
3. 실제 실행과 학습은 **원본 ipynb 파일**을 Jupyter에서 실행하기를 권장합니다
4. 각 파일의 상단 주석에 상세한 설명이 있으니 참고하세요

---

## 피드백 및 개선

더 나은 학습 자료를 위해 다음 내용의 개선을 고려해보세요:
- 더 많은 실제 예제 코드 추가
- 각 주제별 연습 문제 추가
- 주제 간 연관성 표시
- 일반적인 실수와 해결 방법 추가

---

*Happy Learning! Python을 활용한 AI 자연어처리 학습을 응원합니다.*
