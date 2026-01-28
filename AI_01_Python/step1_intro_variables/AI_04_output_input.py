# # 기본출력함수 print()

# 파이썬의 가장 기본적인 출력 함수 print()

# print( 출력내용1, 출력내용2, 출력내용3, ...)
#  위와 같이 print() 함수의 parameter 로 출력할 내용들을 기입하면 된다

# print() 함수는 출력한후 자동적으로 "줄바꿈" 됩니다 (디폴트)

print(10, 20, 30)
print()
print('hello')
print()
print(3.14)
# >>> 출력:
# 10 20 30
# 
# hello
# 
# 3.14

# ## end=
# 출력이 끝난뒤 마지막에 출력하는 문자열

print(10)
print('hello')
print(3.14)
# >>> 출력:
# 10
# hello
# 3.14

print(10, end='##')
print('hello', end='**')
print(3.14, end='--')
# >>> 출력:
# 10##hello**3.14--

print(10, end='')
print('hello', end='')
print(3.14, end='')
# >>> 출력:
# 10hello3.14

# ## sep=
# separator.  출력할 값들간 구분문자열

print(10, 20, 30)
# >>> 출력:
# 10 20 30

print(10, 20, 30, sep='***')
# >>> 출력:
# 10***20***30

# # 이스케이프 문자 (escape character)

# '문자열' 내에서 특수한 문자 출력할때 사용
# \와 조합하여 출력

# 많이 사용되는 이스케이프 코드
#    \n : 줄바꿈
#    \t : 탭
#    \\ : 역슬래시
#    \' : 홀따옴표
#    \" : 쌍따옴표

print("Hello Python")
print("Hello \nPython")
print("He\tllo \tPython")
print("Hello\" \\Python")
# >>> 출력:
# Hello Python
# Hello 
# Python
# He	llo 	Python
# Hello" \Python

# ! 윈도우의 file separator 문자는 '\' 다
path = "C:\Windows\System\011_Python" # 이 경로를 출력하면 escaping 된다.
print(path)
# >>> 출력:
# C:\Windows\System	_Python

path = "C:\\Windows\\System\\011_Python"
print(path)
# >>> 출력:
# C:\Windows\System\011_Python

# # Raw String  (r-string)
# 문자열 내부의 모~든 특수기능 제거한 문자열

path = r"C:\Windows\System\011_Python"
print(path)
# >>> 출력:
# C:\Windows\System\011_Python

# # 문자열 포맷팅 (formatting)

# ## 방법1  % 연산자 사용

# 서식지정자(format specifier) 를 포함한 문자열과,
# 각 '서식문자'에 대응하는 데이터들을 연결하여 문자열 완성

# 구문)
#    "서식문자포함문자열" % ( 데이터 튜플 )

a = "Hello %s"

a
# >>> 출력:
# 'Hello %s'

"Hello %s" % ("파이썬")
# >>> 출력:
# 'Hello 파이썬'

a % ('Java')
# >>> 출력:
# 'Hello Java'

a % ('Swift')
# >>> 출력:
# 'Hello Swift'

"주소: %s  우편번호: %d" % ('서울', 12345)
# >>> 출력:
# '주소: 서울  우편번호: 12345'

"주소: %s  우편번호: %d" % ('강원도', 98765)
# >>> 출력:
# '주소: 강원도  우편번호: 98765'

# format specifier (서식 지정자)
#  https://docs.python.org/2/library/string.html

# %d   십진수 정수로 출력
# %f   실수로 출력
# %s   문자열
# %%   %  문자 자체

"%d %f" % (10, 10)
# >>> 출력:
# '10 10.000000'

import math
PI = math.pi
PI
# >>> 출력:
# 3.141592653589793

"%d %f %.1f %.3f" % (PI, PI, PI, PI)
# >>> 출력:
# '3 3.141593 3.1 3.142'

# ## 방법2  format() 함수 사용

"My name is {}, I'm {} years old".format("박수진", 10)
# >>> 출력:
# "My name is 박수진, I'm 10 years old"

"My name is {name}, I'm {age} years old".format(name = "박수진", age = 10)
# >>> 출력:
# "My name is 박수진, I'm 10 years old"

"My name is {name}, I'm {age} years old".format(age = 10, name = "박수진")
# >>> 출력:
# "My name is 박수진, I'm 10 years old"

print("PI = {}".format(PI))
print("PI = {:.3f}".format(PI))
# >>> 출력:
# PI = 3.141592653589793
# PI = 3.142

# ## 방법3 f-string 사용

lang = 'Python'
author = 'Guido van Rossum'

'Language: {}, Author: {}'.format(lang, author)
# >>> 출력:
# 'Language: Python, Author: Guido van Rossum'

f'Language: {lang}, Author: {author}'
# >>> 출력:
# 'Language: Python, Author: Guido van Rossum'

f'원주율은 {PI} 입니다'
# >>> 출력:
# '원주율은 3.141592653589793 입니다'

f'원주율은 {PI:.2f} 입니다'
# >>> 출력:
# '원주율은 3.14 입니다'

# # 기본입력함수 input()

# 파이썬 기본입력 함수 input()
# 기본적으로 input() 함수는 키보드로 부터 입력받아
# 문자열(str) 으로 리턴합니다

a = input()
# >>> 출력:
# 안녕하세요

a
# >>> 출력:
# '안녕하세요'

type(a)
# >>> 출력:
# str

a + 10
# >>> 출력:
# TypeError: can only concatenate str (not "int") to str

a = input()
# >>> 출력:
# 100

a
# >>> 출력:
# '100'

int(a) + 10
# >>> 출력:
# 110

# input(prompt)
input("키를 입력하세요 cm:")
# >>> 출력:
# 키를 입력하세요 cm:192.34
# '192.34'

# ## 연습: 단위변환

# 1야드(yd)는 91.44cm이고 1인치(in)는 2.54cm이다.
# 처음에는 야드를 입력받고, 두번째는 인치를 입력받아
# 각각 cm로 변환하여 다음 형식에 맞추어 소수 둘째자리까지 출력하시오.​

# [실행예]
# yard 입력: 23.45
# inch 입력: 41.273
# 23.45 yard 는 2144.27cm
# 41.27 inch 는 104.83cm

yard = float(input("yard 입력: "))
inch = float(input("inch 입력: "))
print(f'{yard:.2f} yard 는 {yard * 91.44:.2f}cm')
print(f'{inch:.2f} inch 는 {inch * 2.54:.2f}cm')
# print("%.2f yard 는 %.2fcm" % (yard, yard * 91.44))
# print("%.2f inch 는 %.2fcm" % (inch, inch * 2.54))ㄴ
# >>> 출력:
# yard 입력: 23.45
# inch 입력: 41.273
# 23.45 yard 는 2144.27cm
# 41.27 inch 는 104.83cm
