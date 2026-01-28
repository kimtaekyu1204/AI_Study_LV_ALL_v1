# 제어문 (Control)
#     프로그램의 기본 흐름 변경
#   1. 조건문 Conditional
#       - if

#   2. 순환문 Loop
#       - for, while

# # 조건문 (Conditional)

# 조건문 (if ~ elif ~ else)


# if 조건식:
#    참일때 수행 구문 블록  (블록이란 문장들의 집합)


# if 조건식:
#    참일때 수행 구문 블록
# else:
#    거짓일때 수행 구문 블록


# if 조건식1:
#    조건식1 참일때 수행 구문 블록
# elif 조건식2:
#    조건식2 참일때 수행 구문 블록
# elif 조건식3:
#    조건식3 참일때 수행 구문 블록
# else:
#    어떤 조건식도 거짓일때 수행 구문 블록


# 각 블록은 반드시 동일한 인덴트로 작성되어야 한다

# 조건식에는 비교연산자와 논리연산자 등을 잘 활용해야 한다
# 비교연산자 <, >, ==, !=, >=, <=
# 논리연산자 and, or, not

a = 0

if a > 0:
  print('변수 a 값은')
  print(a, '는 양수입니다.')

print('종료')
# >>> 출력:
# 종료

a = 9

if a % 2 == 0:
  print('짝수')
  print('입니다')
else:
  print('홀수')
  print('가 확실합니다')

print('종료')
# >>> 출력:
# 홀수
# 가 확실합니다
# 종료

score = 95

if score >= 90:
  print('A')
elif score >= 80:
  print('B')
elif score >= 70:
  print('C')
elif score >= 60:
  print('D')
else:
  print('F')

print('종료')
# >>> 출력:
# A
# 종료

# ## 중첩(nested) 조건문

age = 13
gender = 'F'   # 'M', 'F'

# 20살 이상 남자 -> MAN
#      미만 남자 -> BOY
# 20살 이상 여자 -> WOMAN
#      미만 여자 -> GIRL

if age >= 20:
  if gender == 'M':
    print('MAN')
  elif gender == 'F':
    print('WOMAN')

else:
  if gender == 'M':
    print('BOY')
  elif gender == 'F':
    print('GIRL')

print('종료')
# >>> 출력:
# GIRL
# 종료

# # 비교연산자, 논리연산자

# 조건식에 사용가능한 비교연산자, 논리연산자
# 비교연산자 : >, >=, <, <=, !=, ==
# 논리연산자 : and, or, not, ^ (xor)
# 위 연산자들의 결과값은 항상 True / False
#      and : 둘다 '참' 일때 '참'
#      or : 둘중 하나만 '참' 이면 '참'
#      not : 참 -> 거짓, 거짓 -> 참
#      ^ : eXclusive OR (XOR, 배타적 논리합)
#          같으면 거짓, 다르면 참

d = 10

result = d >= 10  # T
result = d >= 10 and d < 100
result = d >= 10 and d > 100
result = d >= 10 or d > 100

result = (d >= 10) ^ (d > 100)

result = (d < 10) ^ (d > 100)

result = not (d < 10)

if result:
  print(result, '참 입니다')
else:
  print(result, '거짓 입니다')
# >>> 출력:
# True 참 입니다

# score 가 0 <= ~ <= 100 범위인가?

score = -10

if 0 <= score and score <= 100:
  print(score, '는 유효한 점수입니다')
else:
  print(score, '는 유효하지 않은 점수입니다')
# >>> 출력:
# -10 는 유효하지 않은 점수입니다

# # 비교연산자는 임의로 chaining 된다.

score = 70

if 0 <= score <= 100:   # 범위값 표현이 가능하다?
  # 0 <= score and score <= 100
  print(score, '는 유효한 점수입니다')
else:
  print(score, '는 유효하지 않은 점수입니다')
# >>> 출력:
# 70 는 유효한 점수입니다

False == True == False

# False == True and True == False
# >>> 출력:
# False

# # 조건식의 '참' '거짓' 판정

# 조건식의 참 / 거짓 판정

# 조건문,  순환문 등에 사용되는  '조건식' 은 참, 거짓이 판정되어야 하는데
# 파이썬에서는 bool 타입 외에도 조건식에서 참, 거짓 판정이 된다.

#           │     참     │   거짓
# ───────────────────
# bool 타입 :     True         False
# 숫자 타입 :  0 아닌 숫자      0
# str  타입 :      "abc"        ""   빈문자열
# list 타입 :    [1, 2, 3]      []
# tuple 타입 :   (1, 2, 3)      ()
# dict 타입 :    {"name":"john"}    {}

# None 타입 :   무조건 거짓

result = 0.001  # T
result = -100   # T
result = 0       # F
result = 0.0     # F
result = "abc"   # T
result = ""     #F
result = ' '    #T
result = [1, 2, 3]  # T
result = []  #F
result = [0] #T
result = (0) #F  숫자 0
result = (0,) # T  tuple
result = {}  # F
result = {'name':'john'} #T
result = None #F

if result:
  print('참', type(result), result)
else:
  print('거짓', type(result), result)
# >>> 출력:
# 거짓 <class 'NoneType'> None

# # SCE : Short-circuit evaluation
# 논리연산자 **and**, **or** 의 결과
# 
# aka. Lazy evaluation

#논리 연산자 and, or 표현식과의 관계
#참, 거짓 판정에 이어 논리연산자의 결과는
# expression 값이 된다.
# 이를 short-circuit evalutaiton (SCE) 혹은
# lazy evalutation 이라 한다

# or
# 왼쪽이 참인 경우 '왼쪽' 수행결과값 리턴
# 왼쪽이 거짓인 경우 '오른쪽' 수행결과값 리턴

# and
# 왼쪽이 참인 경우 '오른쪽' 수행결과값 리턴
# 왼쪽이 거짓인 경우 '왼쪽' 수행결과값 리턴

result = True or False
result = False and True

result = 0 or 100

result = 120 or 0

result = "Hello" and 555
result = [] and "Python"


if result:
  print('참', type(result), result)
else:
  print('거짓', type(result), result)
# >>> 출력:
# 거짓 <class 'list'> []

n = 10

if n % 5 == 0:
  print(n, '5의 배수')
else:
  print(n, '5의 배수 아님')
# >>> 출력:
# 10 5의 배수

n = 10
(n % 5 == 0) and print(n, '5의 배수')
# >>> 출력:
# 10 5의 배수

n = 9
(n % 5 == 0) or print(n, '5의 배수 아님')
# >>> 출력:
# 9 5의 배수 아님

print('AAA') and print('BBB')
# >>> 출력:
# AAA

r = print('AAA')
# >>> 출력:
# AAA

r, type(r)
# >>> 출력:
# (None, NoneType)

print('AAA') or print('BBB')
# >>> 출력:
# AAA
# BBB

# dict.update : 있으면 업데이트하고, 없으면 추가하고  , None 리턴
# dict.get    : 있으면 가져오고, 없으면 None 리턴 (default 값이 있으면 default 값 리턴)

# list.append  .... 원본 변경; return None
# dict.update ... .원본 변경; return None

data = [10, 20, 30]
data.append(100) or data
# >>> 출력:
# [10, 20, 30, 100]

data = {'name': 'John'}
data.update({'age': 20}) or data
# >>> 출력:
# {'name': 'John', 'age': 20}

# # pass
# 아무 문장도 없는 블럭 명시

a = 10

if a > 0:
  # 아무것도 수행하지 않는 블럭에는 반드시 pass 명시
  pass
else:
  print('0보다 작습니다')

# # 한 문장 뿐인 블럭은...

if 10 in (10, 20, 30):
  print('있습니다')
else:
  print('없습니다')
# >>> 출력:
# 있습니다

if 10 in (10, 20, 30): print('있습니다')
else: print('없습니다')
# >>> 출력:
# 있습니다

# # Conditional Expressions
# AKA : 삼항연산자.
# 
# 구문 `(참일때의 값) if (조건식) else (거짓일때의 값)`
# 
# https://docs.python.org/3/reference/expressions.html#conditional-expressions

n = -1

"양수" if n > 0 else "음수"
# >>> 출력:
# '음수'

# 두 숫자간의 차 (difference)

a = 54
b = 72

diff = (a - b) if a > b else (b - a)

diff
# >>> 출력:
# 18

# # 연습문제1

"""
년도를 입력받아 윤년(leap year)인지
평년(common year)인지 판단하는 프로그램을 작성하시오.

400으로 나누어떨어지면 윤년이다.
    또는
4로 나누어떨어지고 100으로 나누어떨어지지 않으면 윤년이다.

나머지는 모두 평년이다.


입력예]
2004

출력]
leap year

입력예]
2100

출력예]
common year

"""
# 4의 배수임에도 평년인 경우
# 1900, 2100, 2200, 2300, 2500....
None

year = int(input())

if year % 400 == 0 or year % 4 == 0 and not year % 100 == 0:
    print("leap year")
else:
    print("common year")
# >>> 출력:
# 2025
# common year
