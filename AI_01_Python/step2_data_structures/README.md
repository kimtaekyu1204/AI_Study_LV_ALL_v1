# 파이썬 데이터 구조 정리 (Step 2: Data Structures)

## 개요
AI 자연어 처리 기초 수업의 **Step 2: 데이터 구조** 부분을 정리한 자료입니다.
파이썬에서 여러 개의 데이터를 담는 4가지 주요 집합 데이터 타입을 다룹니다.

---

## 생성된 파일 목록

### 1. NLP_05_list.py
**리스트 (List) 완전 정리**

#### 주요 내용:
- 리스트의 정의와 특징
- 리스트 생성과 초기화
- 인덱싱 (0-based indexing, 음수 인덱싱)
- 슬라이싱 (start:end:step)
- 다차원 리스트
- 리스트 수정, 추가, 삭제
- 리스트 정렬과 뒤집기
- 멤버십 확인 (in 연산자)
- 문자열과 리스트의 관계

#### 특징:
- 순서 있음 (indexing, slicing 가능)
- 중복 허용
- Mutable (생성 후 수정 가능)
- 대괄호 [] 사용

#### 주요 메서드:
```python
animals = ['dog', 'cat', 'bird']
animals[0]          # 첫 번째 원소
animals[-1]         # 마지막 원소
animals[0:2]        # 슬라이싱
animals.append()    # 추가
animals.sort()      # 정렬
animals.reverse()   # 뒤집기
len(animals)        # 길이
```

---

### 2. NLP_06_tuple.py
**튜플 (Tuple) 완전 정리**

#### 주요 내용:
- 튜플의 정의와 특징
- 튜플 생성 (괄호 사용, 생략)
- 원소 1개 튜플 만드는 법
- 튜플 인덱싱과 슬라이싱
- 불변성 (immutable)
- 튜플 연산 (+ 연결, * 반복)
- 튜플 언팩킹
- count() 메서드

#### 특징:
- 순서 있음 (indexing, slicing 가능)
- 중복 허용
- Immutable (생성 후 수정 불가능)
- 괄호 () 또는 콤마 , 사용

#### 주요 메서드:
```python
animals = ('dog', 'cat', 'bird')
animals[0]          # 첫 번째 원소
animals[-1]         # 마지막 원소
animals[0:2]        # 슬라이싱
animals.count(x)    # 원소 개수 세기
w, h = (100, 200)   # 언팩킹
```

#### 주의사항:
```python
animals = ("dog")       # str이 아니라 tuple 만들기!
animals = ("dog",)      # 원소 1개는 반드시 콤마 필요
```

---

### 3. NLP_07_set.py
**집합 (Set) 완전 정리**

#### 주요 내용:
- 집합의 정의와 특징
- 집합 생성 (중괄호, set() 함수)
- 중복 자동 제거
- 빈 집합 만드는 법
- 집합 추가, 제거 (add, remove)
- 멤버십 확인
- 집합 연산 (합집합, 교집합, 차집합, 대칭차)
- 리스트에서 중복 제거

#### 특징:
- 순서 없음 (indexing 불가능)
- 중복 제거 (같은 원소는 한 번만 저장)
- Mutable (추가, 삭제 가능)
- 중괄호 {} 또는 set() 함수 사용

#### 주요 메서드와 연산:
```python
animals = {"dog", "cat", "bird"}
animals.add("fish")     # 추가
animals.remove("dog")   # 제거
'cat' in animals        # 멤버십 확인

set1 = {1, 2, 3}
set2 = {3, 4, 5}
set1 | set2             # 합집합: {1, 2, 3, 4, 5}
set1 & set2             # 교집합: {3}
set1 - set2             # 차집합: {1, 2}
set1 ^ set2             # 대칭차: {1, 2, 4, 5}
```

#### 주의사항:
```python
a = {}      # dict이다! (set이 아님)
a = set()   # 빈 집합 만들기
```

---

### 4. NLP_08_dict.py
**딕셔너리 (Dictionary) 완전 정리**

#### 주요 내용:
- 딕셔너리의 정의와 특징
- Key-Value 쌍으로 데이터 저장
- 딕셔너리 생성과 초기화
- 값 접근 ([], get())
- 값 수정, 추가, 삭제
- update() 메서드
- 복잡한 딕셔너리 (중첩)
- 딕셔너리 순회 (keys, values, items)
- 멤버십 확인
- 컴프리헨션

#### 특징:
- Key-Value 쌍으로 구성
- 순서 없음 (Python 3.7+ insertion order 유지)
- Key는 중복 불가능
- Mutable (추가, 수정, 삭제 가능)
- 중괄호 {} 사용

#### 주요 메서드:
```python
student = {"name": "John", "age": 25}

# 값 접근
student["name"]         # "John" (없으면 KeyError)
student.get("name")     # "John" (없으면 None)
student.get("age", 20)  # 없으면 기본값 반환

# 값 수정/추가
student["age"] = 26
student["email"] = "john@mail.com"

# 삭제
del student["email"]

# 순회
student.keys()          # 모든 key
student.values()        # 모든 value
student.items()         # (key, value) 쌍

# 업데이트
student.update({"name": "Jane"})
```

#### 주의사항:
```python
# Key로 사용 가능 (hashable):
# - int, str, float, bool, None, tuple

# Key로 사용 불가능 (unhashable):
# - list, set, dict
```

---

## 데이터 구조 비교표

| 특성 | List | Tuple | Set | Dict |
|------|------|-------|-----|------|
| 순서 | ✓ | ✓ | ✗ | ✗ (3.7+) |
| 중복 | ✓ | ✓ | ✗ | ✗ (key) |
| 변경 가능 | ✓ | ✗ | ✓ | ✓ |
| Indexing | ✓ | ✓ | ✗ | name 기반 |
| 속도 | 중간 | 빠름 | 빠름 | 매우 빠름 |
| 표현 | [] | () | {} | {} |

---

## 각 데이터 구조 사용 시기

### List (리스트)
- 순서가 중요할 때
- 같은 데이터 타입의 컬렉션
- 추가, 삭제, 수정이 빈번할 때
- Indexing이 필요할 때

### Tuple (튜플)
- 데이터가 변경되지 말아야 할 때
- Dictionary의 key로 사용할 때
- 함수에서 여러 값을 반환할 때
- 성능이 중요할 때

### Set (집합)
- 중복을 제거해야 할 때
- 원소의 존재 여부만 중요할 때
- 수학적 연산 (합, 교, 차 등)이 필요할 때
- 빠른 검색이 필요할 때

### Dictionary (딕셔너리)
- 데이터를 이름으로 접근하고 싶을 때
- 복잡한 데이터 구조를 표현할 때
- JSON 같은 형식의 데이터를 다룰 때
- 설정, 매핑 데이터 저장할 때

---

## 포함된 원본 파일

생성된 .py 파일과 함께 다음 원본 Jupyter 노트북 파일도 포함되어 있습니다:

1. **05 list.ipynb** - 원본 리스트 강의 노트
2. **06 tuple.ipynb** - 원본 튜플 강의 노트
3. **07 set.ipynb** - 원본 집합 강의 노트
4. **08 dict.ipynb** - 원본 딕셔너리 강의 노트

---

## 학습 순서 권장

1. **NLP_05_list.py** - 가장 기본적이고 자주 사용됨
2. **NLP_06_tuple.py** - 리스트와의 차이점 이해
3. **NLP_07_set.py** - 특화된 용도 학습
4. **NLP_08_dict.py** - 가장 강력하고 유용함

---

## 핵심 요점 정리

### 기억해야 할 사항

1. **0-based Indexing**: 파이썬은 첫 번째 원소가 0부터 시작합니다.
   ```python
   lst = [10, 20, 30]
   lst[0]  # 10
   ```

2. **Slicing**: start:end:step (end는 포함되지 않음)
   ```python
   lst = [0, 1, 2, 3, 4, 5]
   lst[1:4]    # [1, 2, 3]
   lst[::2]    # [0, 2, 4]
   lst[::-1]   # [5, 4, 3, 2, 1, 0]
   ```

3. **Mutable vs Immutable**:
   - Mutable (리스트, 집합, 딕셔너리): 수정 가능
   - Immutable (튜플, 문자열): 수정 불가능

4. **Dictionary 접근**:
   ```python
   d = {"name": "John"}
   d["name"]       # "John" - KeyError 가능
   d.get("name")   # "John" - 안전함
   ```

5. **집합의 빈 상태**:
   ```python
   a = {}      # dict (빈 딕셔너리)
   a = set()   # set (빈 집합)
   ```

---

## 추가 학습 자료

각 Python 파일은 다음을 포함하고 있습니다:

- 정의와 특징 설명
- 생성 방법
- 기본 연산과 메서드
- 실제 사용 예제
- 주의사항과 함정 (pitfalls)
- 다른 데이터 구조와의 비교

Python 파일을 직접 실행하거나, IDE에서 열어서
코드를 직접 입력하고 실행하면서 학습하길 권장합니다.

---

**작성일**: 2025년 1월 28일
**과정**: AI 자연어처리 기초 (AI#1)
**주제**: 파이썬 데이터 구조 완전 정리
