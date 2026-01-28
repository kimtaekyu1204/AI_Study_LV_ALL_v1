# 📘 AI_01_Python - Python 기초

> Python 프로그래밍의 기본 문법과 핵심 개념 정리

---

## 📚 목차

| 파일 | 주제 | 핵심 키워드 |
|:----:|------|-------------|
| [AI_01](./step1_intro_variables/AI_01_hello_python.py) | Hello Python | `print()`, 기본 출력 |
| [AI_02](./step1_intro_variables/AI_02_markdown_latex.py) | 마크다운/LaTeX | Jupyter Notebook 문서화 |
| [AI_03](./step1_intro_variables/AI_03_types_variables.py) | 타입과 변수 | `int`, `float`, `str`, `bool`, `type()` |
| [AI_04](./step1_intro_variables/AI_04_output_input.py) | 출력과 입력 | `print()`, `input()`, f-string |
| [AI_05](./step2_data_structures/AI_05_list.py) | 리스트 | `list`, `append()`, `pop()`, 슬라이싱 |
| [AI_06](./step2_data_structures/AI_06_tuple.py) | 튜플 | `tuple`, 불변성, 언패킹 |
| [AI_07](./step2_data_structures/AI_07_set.py) | 집합 | `set`, 합집합, 교집합, 차집합 |
| [AI_08](./step2_data_structures/AI_08_dict.py) | 딕셔너리 | `dict`, `keys()`, `values()`, `items()` |
| [AI_09](./step3_control_flow/AI_09_conditional.py) | 조건문 | `if`, `elif`, `else` |
| [AI_10](./step3_control_flow/AI_10_loop.py) | 반복문 | `for`, `while`, `break`, `continue` |
| [AI_11](./step3_control_flow/AI_11_function.py) | 함수 | `def`, `return`, `*args`, `**kwargs` |
| [AI_12](./step4_string_file/AI_12_string_functions.py) | 문자열 함수 | `split()`, `join()`, `strip()`, `replace()` |
| [AI_13](./step4_string_file/AI_13_file_handling.py) | 파일 처리 | `open()`, `read()`, `write()`, `with` |
| [AI_14](./step4_string_file/AI_14_regex.py) | 정규표현식 | `re`, `match()`, `search()`, `findall()` |
| [AI_15](./step4_string_file/AI_15_csv_file.py) | CSV 파일 | `csv`, `pandas.read_csv()` |
| [AI_16](./step5_advanced/AI_16_class.py) | 클래스 | `class`, `__init__`, 상속, `super()` |
| [AI_17](./step5_advanced/AI_17_openai_api.py) | OpenAI API | API 호출, 챗봇 구현 |
| [AI_18](./step5_advanced/AI_18_module_package.py) | 모듈/패키지 | `import`, `from`, `__name__` |
| [AI_19](./step5_advanced/AI_19_os_shutil_glob.py) | 파일 시스템 | `os`, `shutil`, `glob` |
| [AI_20](./step5_advanced/AI_20_exception.py) | 예외 처리 | `try`, `except`, `finally`, `raise` |
| [AI_21](./step5_advanced/AI_21_iterable.py) | Iterable | `iter()`, `next()`, 제너레이터 |
| [AI_22](./step5_advanced/AI_22_input_multiple.py) | 다중 입력 | `map()`, `split()` |

---

## 1️⃣ 데이터 타입

Python의 기본 데이터 타입이다.

| 타입 | 설명 | 예시 |
|:----:|------|------|
| `int` | 정수 | `10`, `-5`, `0` |
| `float` | 실수 | `3.14`, `-0.5` |
| `str` | 문자열 | `"hello"`, `'world'` |
| `bool` | 불리언 | `True`, `False` |
| `None` | 없음 | `None` |

```python
x = 10          # int
y = 3.14        # float
s = "hello"     # str
b = True        # bool
n = None        # NoneType

print(type(x))  # <class 'int'>
```

---

## 2️⃣ 자료구조

### 리스트 (List)
순서가 있고, 변경 가능한 자료구조이다.

```python
fruits = ["apple", "banana", "cherry"]
fruits.append("orange")   # 추가
fruits.pop()              # 마지막 요소 제거
print(fruits[0])          # apple (인덱싱)
print(fruits[1:3])        # ['banana', 'cherry'] (슬라이싱)
```

### 딕셔너리 (Dictionary)
키-값 쌍으로 데이터를 저장한다.

```python
person = {"name": "Kim", "age": 25}
print(person["name"])     # Kim
person["city"] = "Seoul"  # 추가
print(person.keys())      # dict_keys(['name', 'age', 'city'])
```

### 튜플 (Tuple)
순서가 있고, 변경 불가능한 자료구조이다.

```python
coords = (10, 20)
x, y = coords  # 언패킹
print(x, y)    # 10 20
```

### 집합 (Set)
중복이 없고, 순서가 없는 자료구조이다.

```python
a = {1, 2, 3}
b = {2, 3, 4}
print(a | b)   # {1, 2, 3, 4} 합집합
print(a & b)   # {2, 3} 교집합
print(a - b)   # {1} 차집합
```

---

## 3️⃣ 제어문

### 조건문
```python
score = 85

if score >= 90:
    print("A")
elif score >= 80:
    print("B")
else:
    print("C")
# 출력: B
```

### 반복문
```python
# for문
for i in range(5):
    print(i, end=" ")  # 0 1 2 3 4

# while문
count = 0
while count < 3:
    print(count)
    count += 1
```

### 리스트 컴프리헨션
```python
squares = [x**2 for x in range(5)]
print(squares)  # [0, 1, 4, 9, 16]

evens = [x for x in range(10) if x % 2 == 0]
print(evens)    # [0, 2, 4, 6, 8]
```

---

## 4️⃣ 함수

### 기본 함수
```python
def greet(name, msg="Hello"):
    """인사 함수"""
    return f"{msg}, {name}!"

print(greet("Kim"))           # Hello, Kim!
print(greet("Lee", "Hi"))     # Hi, Lee!
```

### 가변 인자
```python
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4, 5))  # 15

def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Kim", age=25)
# name: Kim
# age: 25
```

### 람다 함수
```python
square = lambda x: x ** 2
print(square(5))  # 25

nums = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, nums))
print(squared)  # [1, 4, 9, 16, 25]
```

---

## 5️⃣ 클래스

### 클래스 정의
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"저는 {self.name}이고, {self.age}살입니다."

p = Person("Kim", 25)
print(p.introduce())  # 저는 Kim이고, 25살입니다.
```

### 상속
```python
class Student(Person):
    def __init__(self, name, age, grade):
        super().__init__(name, age)
        self.grade = grade

    def introduce(self):
        return f"{super().introduce()} {self.grade}학년입니다."

s = Student("Lee", 20, 3)
print(s.introduce())  # 저는 Lee이고, 20살입니다. 3학년입니다.
```

---

## 6️⃣ 예외 처리

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"에러 발생: {e}")
except Exception as e:
    print(f"기타 에러: {e}")
finally:
    print("실행 완료")

# 출력:
# 에러 발생: division by zero
# 실행 완료
```

---

## 7️⃣ 파일 처리

```python
# 파일 쓰기
with open("test.txt", "w") as f:
    f.write("Hello, World!")

# 파일 읽기
with open("test.txt", "r") as f:
    content = f.read()
    print(content)  # Hello, World!
```

---

## 📖 학습 순서

```
step1 (변수/타입) → step2 (자료구조) → step3 (제어문) → step4 (문자열/파일) → step5 (고급)
```
