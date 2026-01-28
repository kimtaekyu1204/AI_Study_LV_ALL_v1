# # 주석 (Comment)

# 안녕하세요

# 주석 (comment)
# 프로그램의 실행과는 관계없이
# 코드에 메모등을 남길때 사용

# 1. 한줄 주석:  line comment)
#    '#' 으로 시작
#    '#' 우측부터의 한 줄은 '주석'으로 처리

# 2. 블럭주석 block comment
"""
  이 안의 내용들은
  모두 주석처리
  됩니다.
"""
None

# # 숫자(정수, 실수) 데이터

# 숫자 (정수, 실수) 데이터
# 소숫점이 없으면 '정수' (int : integer)
# 소숫점이 있으면 '실수' (float)
# 컴퓨터 프로그래밍에서는 실수를 floating point number 라 함

10   # 정수 int
# >>> 출력:
# 10

3.14  # 실수 float
# >>> 출력:
# 3.14

10.0  # float
# >>> 출력:
# 10.0

10.   # float
# >>> 출력:
# 10.0

0.1
# >>> 출력:
# 0.1

.1
# >>> 출력:
# 0.1

10 - 2 * 4
# >>> 출력:
# 2

(10 - 2) * 4
# >>> 출력:
# 32

print((10 - 2) * 4)
# >>> 출력:
# 32

# 정수와 정수끼리의 연산결과는 정수(int)
# 실수와의 연산결과는 실수(float)

4 * 5
# >>> 출력:
# 20

4 * 5.0
# >>> 출력:
# 20.0

4 / 2 # ★ 나눗셈 연산 결과는 언제나 실수 !
# >>> 출력:
# 2.0

# 나눗셈 후 소수점 이하를 버리는 연산자 :  //
# 정수 끼리 딱 떨어지는 결과에 대해서는 정수로 결과가 나옴
# 나눗셈 후 소숫점 이하 버리는 연산자


4 // 2
# >>> 출력:
# 2

3.4 / 1.1
# >>> 출력:
# 3.0909090909090904

3.4 // 1.1
# >>> 출력:
# 3.0

4 // 2.0
# >>> 출력:
# 2.0

5 / 0
# >>> 출력:
# ZeroDivisionError: division by zero

# 나머지 연산자 %

13 % 3
# >>> 출력:
# 1

12.5 % 4.1  # 12.5 = 4.1 * 3 + 0.2
# >>> 출력:
# 0.20000000000000107

# ★컴퓨터의 실수 계산결과는 항상 오차가 있다!

# 제곱연산자 **

2 ** 4
# >>> 출력:
# 16

-3 ** 3
# >>> 출력:
# -27

2 ** -1
# >>> 출력:
# 0.5

2 ** (1/2)
# >>> 출력:
# 1.4142135623730951

# # 문자열 str
# string

#----------------------------------
# 문자열 타입 : str (string)
# 문자열 리터럴 만드는 방법
# 1. 쌍따옴표 (double quotation)
# 2. 홀따옴표 (single quotation)
# 3. 쌍따옴표3개
# 4. 홀따옴표3개
# 그밖에
#   f-string
#   r-string
#   ..///

"Python is fun"
# >>> 출력:
# 'Python is fun'

print("Python is fun")
print('Python is fun')
print("""Python is fun""")
print('''Python is fun''')
# >>> 출력:
# Python is fun
# Python is fun
# Python is fun
# Python is fun

# She's gone
print("She's gone")
# print('She's gone')
# >>> 출력:
# She's gone

# He says "It's OK!"

print('''He says "It's OK!"''')
# >>> 출력:
# He says "It's OK!"

# print("Python
# is fun")

print("Python \
is fun")
# >>> 출력:
# Python is fun

print("Python "
      + "is fun")
# >>> 출력:
# Python is fun

# ''' ~ ''' 와 """ ~ """ 는 줄바꿈도 인식
print('''Python
is fun''')
# >>> 출력:
# Python 
# is fun

# 문자열 연산 +
"Hello" + 'World'
# >>> 출력:
# 'HelloWorld'

'Hello' + ' ' + 'Python'
# >>> 출력:
# 'Hello Python'

# 문자열 연산 *

"-" * 20
# >>> 출력:
# '--------------------'

"파이썬" * 3
# >>> 출력:
# '파이썬파이썬파이썬'

# len() : 문자열 안의 문자 개수
# length

len('Python')
# >>> 출력:
# 6

len("안녕하세요 Python")  # 공백도 하나의 문자다!
# >>> 출력:
# 12

# # bool 데이터
# boolean

# bool 타입
# True (참) 혹은 False (거짓) 값만 갖는 타입

True
# >>> 출력:
# True

False
# >>> 출력:
# False

20 > 10  # '비교연산' 의 결과는 bool 타입
# >>> 출력:
# True

# 비교연산자(들)
print(20 > 10)
print(20 < 10)
print(20 >= 10)
print(20 >= 20)
print(20 > 20)
print(20 <= 30)
print(20 == 20)
print(20 != 20)
# >>> 출력:
# True
# False
# True
# True
# False
# True
# True
# False

10 / 2 == 5.0
# >>> 출력:
# True

5.0 == 5
# >>> 출력:
# True

10 / 2 == 5
# >>> 출력:
# True

2.1 + 0.2 == 2.3   # 실수연산결과는 정확하지 않다!  절대 == 비교연산 하지 말자!
# >>> 출력:
# False

2.1 + 0.2
# >>> 출력:
# 2.3000000000000003

# # None 타입
# 아무런 값도 없는 데이터 타입

None

print(None)
# >>> 출력:
# None

# # type() 함수

type(10)
# >>> 출력:
# int

type(10.0)
# >>> 출력:
# float

type("10.0")
# >>> 출력:
# str

type(True)
# >>> 출력:
# bool

type(None)
# >>> 출력:
# NoneType

print(type(10))
# >>> 출력:
# <class 'int'>

# # Indentation
# 들여쓰기

# 파이썬은 들여쓰기 '열' 맞추어야 한다
# print(100)
#  print(200)
# print(300)
# >>> 출력:
# 100
# 200
# 300

# # 변수 (Variable)

# 변수(variable) 은 데이터를 담아두는 공간
# 이름을 정해서 담아둔다.  이 이름을 변수이름 (variable name) 이라한다
# 변수의 데이터는 언제든지 변경할수 있다. (변할수 있다!)

# 변수 사용법
#  변수명 = 값

# 이와 같이 = 연산자를 사용하여 변수에 값을 저장하는 것을 대입(assign 한다 라고 하며
# = 을 대입연산자 (assignment operator) 라고 한다

# ★★프로그래머는 변수에 어떠한 '타입'의 어떠한 '값'이 담겨 있는지 놓치면 안된다!   타입! 값!

# 변수명은 대소문자 구분한다

a = 10  # a 라는 변수에 정수 int 값 10을 대입.

print(a)  # a 라는 변수의 값을 읽어서 출력
# >>> 출력:
# 10

type(a)
# >>> 출력:
# int

b = 5

print(b)
# >>> 출력:
# 5

a + b
# >>> 출력:
# 15

print(100, 200, 300)
# >>> 출력:
# 100 200 300

print(a, b, a * b, a - b)
# >>> 출력:
# 10 5 50 5

print(c) # 정의되지 않은 변수는 사용 불가! NameError`
# >>> 출력:
# NameError: name 'c' is not defined

print(A)  # a 와 A 는 다르다!
# >>> 출력:
# NameError: name 'A' is not defined

# ## 형(type) 변환 함수

# 형변환 함수
# int(), float(), str(), bool() ....

age = 10

print("제 나이는", age, "살 입니다")
# >>> 출력:
# 제 나이는 10 살 입니다

print("제 나이는" + age + "살 입니다") # str 과 숫자간 + 연산 안됨!
# >>> 출력:
# TypeError: can only concatenate str (not "int") to str

print("제 나이는" + str(age) + "살 입니다")
# >>> 출력:
# 제 나이는10살 입니다

age
# >>> 출력:
# 10

str(age)
# >>> 출력:
# '10'

num = "100"
# print(num + 2)
print(int(num) + 2)
# >>> 출력:
# 102

int(3.14)  # 실수 -> 정수 변환, 소숫점 이하 제거.
# >>> 출력:
# 3

bool(0)  # 0 --> False
# >>> 출력:
# False

bool(1)  # 0 외의 숫자 --> True
# >>> 출력:
# True

int(False)
# >>> 출력:
# 0

int(True)
# >>> 출력:
# 1

# ## 변수명 규칙

#변수명규칙
# 알파벳, 숫자, _  등 사용 가능
# 숫자로는 시작할수 없다
# 변수명에 띄어쓰기 불가
# 특수문자 불가
# 파이썬의 예약어(reserved word)는 변수명으로 사용불가
#      and,  as,  assert,  break,  class,  continue,  def,  del,  elif,  else,  except,  is,
#      finally,  for, from,  global,  if,  import,  in,  lambda,  nonlocal,
#      not,  or,  pass,  raise,  return,  try,  while,  with, yield

lapTime = 10.2
abc2018 = 10
myName = "john"
my_name = "JOhn"
_value23_ = 3.14

# 불가능
# 55num = 3
# @!#$!@# = 33
# $abc = 10
# if = 100

# ## del() : 변수제거
# delete

name = "홍길동"

print(name, type(name))
# >>> 출력:
# 홍길동 <class 'str'>

del(name)   # 변수 제거

print(name, type(name))
# >>> 출력:
# NameError: name 'name' is not defined

# ## 여러 변수 한번에 선언하기

a = 10
b = 20
c = 30
print(a, b, c)
# >>> 출력:
# 10 20 30

a = 10; b = 20; c = 30
print(a, b, c)
# >>> 출력:
# 10 20 30

#그러나 파이썬에서는 여러 변수를 선언할때 아래와 같은 표현을 더 많이 사용한다
# 파이썬 스러운 (pythonic) 한 방법

a, b, c = 100, 200, 300

print(a, b, c)
# >>> 출력:
# 100 200 300

# ## 변수의 값 증감

a = 10
print(a)
a = a + 1
print(a)
a = a * 2
print(a)
# >>> 출력:
# 10
# 11
# 22

# ### 복합대입연산자

a += 10   # a = a + 10
print(a)
# >>> 출력:
# 32

# +=, -=, *=, /= , //=, %= ....

b
# >>> 출력:
# 200

b /= 2
print(b)
# >>> 출력:
# 100.0

# ## 변수 값 바꾸기 (swap)

a, b = 11, 33
print('a=', a, ', b=', b)
temp = a
a = b
b = temp
print('a=', a, ', b=', b)
# >>> 출력:
# a= 11 , b= 33
# a= 33 , b= 11

a, b = 100, 300
print('a=', a, ', b=', b)
a, b = b, a
print('a=', a, ', b=', b)
# >>> 출력:
# a= 100 , b= 300
# a= 300 , b= 100
