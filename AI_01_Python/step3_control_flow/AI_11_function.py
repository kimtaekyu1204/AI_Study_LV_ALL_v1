# # 함수(Function)

print("안녕하세요")
print("제 이름은 최한빈 입니다")

print("안녕하세요")
print("제 이름은 이승원 입니다")

print("안녕하세요")
print("제 이름은 김기현 입니다")

print("안녕하세요")
print("제 이름은 김유진 입니다")
# >>> 출력:
# 안녕하세요
# 제 이름은 최한빈 입니다
# 안녕하세요
# 제 이름은 이승원 입니다
# 안녕하세요
# 제 이름은 김기현 입니다
# 안녕하세요
# 제 이름은 김유진 입니다

# 반복되는 코드들이 보인다!

# ## 함수를 왜 사용하나?

# 프로그래밍에서 동일한(혹은 거의 비슷한) 내용의 코드가 반복될때가 있다.
# 바로 이러한 코드 낭비를 없애기 위해
# 반복되는 코드를 묶어서 하나의 함수로 정의해 놓고 사용하는 것이다

# 즉, 반복되는 부분이 있을 경우 "반복적으로 사용되는 가치 있는 부분"을
# 한 뭉치로 묶어서
#  1. 어떤 입력값을 주었을 때,
#  2. 어떠한 일을 수행하고,
#  3. 어떤 결과값을 돌려준다 라는식의
# 함수로 작성하는 것이 현명하다.

# # 함수의 정의, 호출

# 함수 만들기 (함수 정의 : Function Definition)

# def 키워드로 작성

# def 함수이름( 매개변수 들):
#     함수 본체 <수행할 문장1>
#     함수 본체 <수행할 문장2>
#     ...

# 함수의 동작 정의
#   입력(매개변수) -> 본체 수행 -> 결과(리턴)

# 함수이름은 변수 이름 작성과 거의 동일한 규칙

#----------------------------------------------------------
# 함수 호출 (함수 사용 )  Function call, invoke
# 함수호출시 넘겨지는 인자(parameter) 값들은
# 함수의 매개변수(argument)들이 받습니다.
# 매개변수는 0개, 한개, 여러개 있을수 있을수 있습니다

# 함수에는 리턴값이 있다.
# 리턴값은 함수를 호출(call) 한쪽에 돌려준다
# return 키워드로 리턴값을
# 함수 본체 수행중 return 을 만나면 함수를 종료 하게 됩니다
# 어떠한 타입도 리턴할수 있다.
# 리턴값은 없을수도 있다. None 리턴

# 함수 정의
def sayAnthem():
  print('동해물과 백두산이')
  print("마르고 닳도록")
  print("하느님이 보우하사")
  print("우리나라 만세")

# 함수 호출
sayAnthem()
print('💚' * 10)
sayAnthem()
print('💙' * 10)
# >>> 출력:
# 동해물과 백두산이
# 마르고 닳도록
# 하느님이 보우하사
# 우리나라 만세
# 💚💚💚💚💚💚💚💚💚💚
# 동해물과 백두산이
# 마르고 닳도록
# 하느님이 보우하사
# 우리나라 만세
# 💙💙💙💙💙💙💙💙💙💙

type(sayAnthem)
# >>> 출력:
# function

# # 매개변수 (parameter, argument)

def sayName(name):
  print('안녕하세요')
  print(f'제 이름은 {name} 입니다')

sayName('도라에몽')  # 함수 호출시 명시한 값이 매개변수에 전달.
# >>> 출력:
# 안녕하세요
# 제 이름은 도라에몽 입니다

sayName('노진구')
# >>> 출력:
# 안녕하세요
# 제 이름은 노진구 입니다

sayName()
# >>> 출력:
# TypeError: sayName() missing 1 required positional argument: 'name'

sayName('퉁퉁이', 34)
# >>> 출력:
# TypeError: sayName() takes 1 positional argument but 2 were given

# ## positional argument

def sayHello1(name, age):
  sayName(name)
  print(f'제 나이는 {age} 입니다')

sayHello1('캡틴', 132)
# >>> 출력:
# 안녕하세요
# 제 이름은 캡틴 입니다
# 제 나이는 132 입니다

sayHello1(100, '토르')
# >>> 출력:
# 안녕하세요
# 제 이름은 100 입니다
# 제 나이는 토르 입니다

# ## keyword argument

sayHello1(name='토끼', age=50)
# >>> 출력:
# 안녕하세요
# 제 이름은 토끼 입니다
# 제 나이는 50 입니다

sayHello1(age=50, name='토끼')
# >>> 출력:
# 안녕하세요
# 제 이름은 토끼 입니다
# 제 나이는 50 입니다

sayHello1(age=50, name='토끼', address='서울')
# >>> 출력:
# TypeError: sayHello1() got an unexpected keyword argument 'address'

sayHello1(age = 50)
# >>> 출력:
# TypeError: sayHello1() missing 1 required positional argument: 'name'

# # return

# return [값]
# 호출된 함수 실행중 return 을 만나게 되면
#  -- 함수 종료
#  -- 호출한 쪽으로 '값 하나' 을 돌려준다

def codeEveryday():
    print("파이썬 열공중")
    print("Life is short")
    print("You need Python")
    return # None

codeEveryday()
# >>> 출력:
# 파이썬 열공중
# Life is short
# You need Python

def codeEveryday():
    print("파이썬 열공중")
    print("Life is short")
    return
    print("You need Python")

codeEveryday()
# >>> 출력:
# 파이썬 열공중
# Life is short

# 함수 정의 예
# 입력: 두개의 수를 입력 받아서
# 수행: 덧셈을 수행한뒤
# 리턴: 덧셈 결과를 리턴

def add(a, b):   # 매개변수: 함수의 입력
  result = a + b
  return result  # 리턴값: 함수의 출력

value1 = add(10, 20)
print(value1)
# >>> 출력:
# 30

add(10, add(1, 2))
# >>> 출력:
# 13

#       add(1, 2)
#          ↓
#
# add(10,  3   )

print('out =', add(150, 40))
# >>> 출력:
# out = 190

def aaa():
  a = 10
  b = 20

aaa()

# # 리턴값이 여러개 ?

# 결론적으로 말하자면 함수의 리턴 값은 오직 '하나' 다
# 그런데 파이썬에선 여러개의 값도 리턴이 가능했다!  어떻게 가능?  tuple 로 리턴하면 가능하다.
# 기술적으로는 tuple 하나 만 리턴했지만, tuple 안에 값이 여러개 있기 때문에
# 여러개의 값을 리턴한것과 같은 효과를 보는것이다

def sum_and_mul(a, b):
  return a + b, a * b  # tuple 로 리턴!

sum_and_mul(3, 4)
# >>> 출력:
# (7, 12)

(aaa, bbb) = sum_and_mul(10, 7)

aaa
# >>> 출력:
# 17

bbb
# >>> 출력:
# 70

# # 파이썬의 내장함수 (Built-in Functions)
# 별도의 import 없이 바로 사용가능한 함수들

# 파이썬의 내장함수( built-in functions)
# 별도의 import 없이 바로 사용 가능한 함수들.

# 레퍼런스 :  https://docs.python.org/3/library/functions.html

# abs()	dict()	help()	min()	setattr()
# all()	dir()	hex()	next()	slice()
# any()	divmod()	id()	object()	sorted()
# ascii()	enumerate()	input()	oct()	staticmethod()
# bin()	eval()	int()	open()	str()
# bool()	exec()	isinstance()	ord()	sum()
# bytearray()	filter()	issubclass()	pow()	super()
# bytes()	float()	iter()	print()	tuple()
# callable()	format()	len()	property()	type()
# chr()	frozenset()	list()	range()	vars()
# classmethod()	getattr()	locals()	repr()	zip()
# compile()	globals()	map()	reversed()	__import__()
# complex()	hasattr()	max()	round()
# delattr()	hash()	memoryview()	set()

abs(-111)
# >>> 출력:
# 111

data = [10, 20, 30, 40]

min(data)
# >>> 출력:
# 10

max(data)
# >>> 출력:
# 40

sum(data)
# >>> 출력:
# 100

# 파이썬에선 함수도 데이터타입이다!!

type(sayName)
# >>> 출력:
# function

ppp = sayName

ppp('홍길동')
# >>> 출력:
# 안녕하세요
# 제 이름은 홍길동 입니다

# # 파이썬의 표준 외장함수
# Python Standard Library

# 파이썬 설치시 '기본적'으로 설치된 라이브러리로부터 사용 가능한 함수들

# 라이브러리(library) 란 파이썬 프로그래밍에 사용가능한 프로그램을을 모아놓은 것.

# https://docs.python.org/3/library/index.html    <-- Python Standard Library

import random  # random 모듈 사용

random.random()  # 0.0 <=  < 1.0  사이의 난수값 리턴 [0.0, 1.0)
# >>> 출력:
# 0.9094634906720631

for _ in range(10):
  print(random.random())
# >>> 출력:
# 0.09483350501831034
# 0.6408634221699938
# 0.6289148013387
# 0.3034108725508894
# 0.9354677237878486
# 0.26130499587915046
# 0.42096120495324285
# 0.37191154880548993
# 0.45882372284302886
# 0.7086236068809306

# random.seed(임의의 숫자K)  <-  랜덤 결과값 고정
random.seed(43)
for _ in range(10):
  print(random.random())
# >>> 출력:
# 0.038551839337380045
# 0.6962243226370528
# 0.14393322139536102
# 0.46253225482908755
# 0.671646764117767
# 0.7929512716552943
# 0.45318922846621235
# 0.4982722297980512
# 0.01915710802434778
# 0.4323888389514402

for _ in range(10):
  print(random.randint(1, 10))  # [1, 10] 정수 난수값
# >>> 출력:
# 5
# 5
# 2
# 9
# 10
# 2
# 10
# 1
# 2
# 10

data = ['dog', 'cat', 'bird', 'fish']
random.shuffle(data)
data
# >>> 출력:
# ['dog', 'fish', 'cat', 'bird']

# shuffle 사용해서 로또번호 추출   1 ~ 45 숫자중 랜덤 6개

data = [i + 1 for i in range(45)]
random.shuffle(data)
print(data[:6])
# >>> 출력:
# [24, 39, 28, 9, 14, 19]

import math

math.pi
# >>> 출력:
# 3.141592653589793

math.e
# >>> 출력:
# 2.718281828459045

math.radians(180)
# >>> 출력:
# 3.141592653589793

# ceil(n) 주어진 숫자보다 큰 정수중에 가장 작은 정수
# floor(n) 주어진 숫자보다 작은 정수중에 가장 큰 정수
# trunc(n) 소숫점 이하 제거

math.ceil(3.14)
# >>> 출력:
# 4

math.floor(3.14)
# >>> 출력:
# 3

math.trunc(3.14)
# >>> 출력:
# 3

import sys

sys.version
# >>> 출력:
# '3.11.11 (main, Dec  4 2024, 08:55:07) [GCC 11.4.0]'

sys.maxsize
# >>> 출력:
# 9223372036854775807

sys.path
# >>> 출력:
# ['/content',
#  '/env/python',
#  '/usr/lib/python311.zip',
#  '/usr/lib/python3.11',
#  '/usr/lib/python3.11/lib-dynload',
#  '',
#  '/usr/local/lib/python3.11/dist-packages',
#  '/usr/lib/python3/dist-packages',
#  '/usr/local/lib/python3.11/dist-packages/IPython/extensions',
#  '/usr/local/lib/python3.11/dist-packages/setuptools/_vendor',
#  '/root/.ipython']

# # 실수로 내장함수를 덮어쓰기 한 경우 복원하기

sum([10, 20])
# >>> 출력:
# 30

sum = 10 + 20

sum([10, 20])
# >>> 출력:
# TypeError: 'int' object is not callable

sum = __builtins__.sum

sum([10, 20])
# >>> 출력:
# 30

# # Default Arguments(디폴트 매개변수)

# 함수정의시 매개변수에 디폴트 값을 지정해 주면,  호출시 생략가능하다
# 생략된 상태에서 호출되면 디폴트 값으로 작동된다

# 디폴트 매개변수 작성 구문
#    def 함수명(매개변수명=디폴트값)

def say_myself(name, age, gender='M'):
    print(f"제 이름은 {name} 입니다")
    print(f'나이는 {age} 입니다')

    (gender == 'M') and print('남자입니다')
    (gender == 'F') and print('여자입니다')
    print()

say_myself('고길동', 43, 'M')
say_myself('김유진', 25)
say_myself('이송미', 32, 'F')
# >>> 출력:
# 제 이름은 고길동 입니다
# 나이는 43 입니다
# 남자입니다
# 
# 제 이름은 김유진 입니다
# 나이는 25 입니다
# 남자입니다
# 
# 제 이름은 이송미 입니다
# 나이는 32 입니다
# 여자입니다

print(10, 20, 30, sep=',')  # sep=' '   <- 디폴트값.
# >>> 출력:
# 10,20,30

# # 가변매개변수 *args
# ### argument packing
# 입력값이 몇개가 올른지 모른다, 임의의 개수 인자 받기

# 가변매개변수(various arguments) 함수 구문

# def 함수이름(*매개변수):
#     <수행할 문장>
#     ...

# 함수 호출시 전달된 복수개의 매개변수는 하나의 tuple 의 형태로 묶여서(packing) 다루어진다
# 가변 매개변수도 어떠한 이름으로도 가능하긴 하나  관례적으로 args 를 많이 사용한다

def var_args(*args):
  print(type(args), ":", len(args), ":",  args)

var_args(10)
var_args(10, 20, 30)
# >>> 출력:
# <class 'tuple'> : 1 : (10,)
# <class 'tuple'> : 3 : (10, 20, 30)

print(10)
print(10, 20, 30)
# >>> 출력:
# 10
# 10 20 30

# # 키워드 매개변수 **kwargs
# keyword arguments

# kwargs : keyword arguments 약자

# def 함수이름(**매개변수):
#     <수행할 문장>
#     ...


# 함수호출시 함수의 인수로 key = value 형태로 주어지면
# 함수에선 kwargs 가 dict 형태로 packing 하여 받아옴

def func(**kwargs):
  print(type(kwargs), ":", len(kwargs), ":", kwargs)

# func('John')
func(name = 'John')
func(name = 'John', age = 32, address = '서울')
# >>> 출력:
# <class 'dict'> : 1 : {'name': 'John'}
# <class 'dict'> : 3 : {'name': 'John', 'age': 32, 'address': '서울'}

# *args, **kwargs 동시 사용.

def func2(*args, **kwargs):
  print(args, "|", kwargs)

func2(10)
func2(10, 20, 30)
func2(10, 20, 30, name = 'John', age = 32, address = '서울')
# >>> 출력:
# (10,) | {}
# (10, 20, 30) | {}
# (10, 20, 30) | {'name': 'John', 'age': 32, 'address': '서울'}

# # Argument Unpacking
# 함수 **호출** 할때 *

def print_val(kor, eng, math):
    print(kor, eng, math)
    print('총점=', kor + eng + math)

print_val(10, 20, 30)
# >>> 출력:
# 10 20 30
# 총점= 60

score = [11, 22, 33]
# print_val(score)

print_val(score[0], score[1], score[2])
# >>> 출력:
# 11 22 33
# 총점= 66

print_val(*score)  # argument unpacking 발생!
# >>> 출력:
# 11 22 33
# 총점= 66

print_val(*(100, 200, 300))
# >>> 출력:
# 100 200 300
# 총점= 600

student = {"name": "Sam", "email": "sam@mail.com"}

def print_dict(name, email):
    print(name, email)

print_dict(student['name'], student['email'])
# >>> 출력:
# Sam sam@mail.com

print_dict(**student)   # dict 를 ** 로 unpacking
# >>> 출력:
# Sam sam@mail.com

# print 와 똑같이 동작하는 함수 log() 를 정의

def log(*args, **kwargs):  # <- argument packing
  print(*args, **kwargs)   # <- argument unpacking

log(10, 20, 30, end = '---', sep = ':')
# >>> 출력:
# 10:20:30---

# # 람다 (lambda)
# '이름이 없는(익명)' inline 함수 정의
# 
# 구문: `lambda [parameters]: expression`

def square(x):
  return x ** 2

square(2)
# >>> 출력:
# 4

square(3)
# >>> 출력:
# 9

def absolute(x):
  return x * (1 if x >= 0 else -1)

absolute(-45)
# >>> 출력:
# 45

# 가령 다음과 같은 기능을 하는 함수들를 만드려고 한다

#      (입력)        (리턴)
# 1. [1, 2, 3] => [1, 4, 9]        <-- 제곱을 하는 함수

# 2. [-1, 2, -3] => [1, 2, 3]      <--  절대값을 구하는 함수


# 즉, 집합데이터형을 입력 받아서 '각 원소'들에게 '무언가 적용' 한 결과를 만드는 함수!!

def apply_func(numbers, func):   # 함수를 매개변수로 받는 함수 정의
  return [func(number) for number in numbers]

apply_func([3, -2, 7], square)
# >>> 출력:
# [9, 4, 49]

apply_func([3, -2, 7], absolute)
# >>> 출력:
# [3, 2, 7]

lambda x: x ** 2
# >>> 출력:
# <function __main__.<lambda>(x)>

sqr = lambda x: x ** 2

sqr(10)
# >>> 출력:
# 100

(lambda x: x ** 2)(100)
# >>> 출력:
# 10000

(lambda : print('hello'))()
# >>> 출력:
# hello

(lambda x, y: x + y)(10, 20)
# >>> 출력:
# 30

apply_func([3, -2, 7], lambda x: x ** 2)
# >>> 출력:
# [9, 4, 49]

apply_func([3, -2, 7], lambda x: x * (1 if x >= 0 else -1))
# >>> 출력:
# [3, 2, 7]

# map(), filter(), reduce() 등에서 lambda 활용 많이 함.

# map: N -> N  (N개의 입력을 받아 '어떠한 동작을' 적용하여 N개의 결괏값을 생성)
# filter: N -> N' (N' <= N)
# reduce: N -> 1

# # map() 함수 : N개 => N개

# '집합데이터' 에 ~~한 동작/연산을 수행/적용하는 함수' 를 세트로 주어
# 일괄 처리 할수 있다
# 파이썬에는 이와 같은 일을 처리하는 함수가 있다.
# 바로 map() 이다

# 구문
#  map(함수, iterable 데이터)     #<-- 이때 data 는 iterable 해야 한다

#  map() 결과 리턴값은 'map객체' 이고 이 또한 iterable 하다!

m = map(square, [1, 2, 3])  # map객체 리턴
m
# >>> 출력:
# <map at 0x7eceee3324a0>

list(m)
# >>> 출력:
# [1, 4, 9]

list(map(lambda x: x ** 2, [10, 20, 30]))
# >>> 출력:
# [100, 400, 900]

# 도전
# 주어진 리스트의 멤버들을 제곱해서 음수값 내기
# lambda 와 map 사용

#[3, 2, -4]  --->  [-9, -4, -16]


list(map(lambda x: -(x ** 2), [3, 2, -4]))
# >>> 출력:
# [-9, -4, -16]

# # filter() : (N' <= N)

# 구문 : filter(function, iterable)

# filter에 인자로 사용되는 function은 처리되는 각각의 요소에 대해 참/거짓 값을 반환합니다.
# '참'을 반환하면 그 요소는 남게 되고, '거짓'을 반환하면 그 요소는 제거 됩니다

# filter() 결과는 filter 객체  (이또한 iterable 하다)

def multiple3(x):
  return x % 3  == 0

multiple3(8)
# >>> 출력:
# False

multiple3(9)
# >>> 출력:
# True

list(filter(multiple3, [3, 7, 9]))
# >>> 출력:
# [3, 9]

# 입력: [3,    7,    9]
#       ↓      ↓    ↓
# 적용:   multiple3()
#       ↓      ↓    ↓
#       T      F    T
#       ↓           ↓
# 결과: [3,         9]

list(filter(lambda x: x % 3 == 0, [3, 7, 9]))
# >>> 출력:
# [3, 9]

dataset = [1, -2, 3, -4, 5]

# 양수인 숫자들을 제곱한 결과  => [1, 9, 25]

list(filter(lambda x: x > 0, dataset))
# >>> 출력:
# [1, 3, 5]

list(map(lambda x: x ** 2, filter(lambda x: x > 0, dataset)))
# >>> 출력:
# [1, 9, 25]

list(map(
    lambda x: x ** 2,
    filter(lambda x: x > 0, dataset)))
# >>> 출력:
# [1, 9, 25]

[x ** 2 for x in dataset if x > 0]
# >>> 출력:
# [1, 9, 25]

# # global variable, global scope
# 함수 바깥에서 선언한 변수.  
# 
# 스크립트 전체에서 접근할 수 있는 변수를 전역 변수(global variable)라고 부릅니다.
# 
# 특히 전역 변수에 접근할 수 있는 범위를 전역 범위(global scope)라고 합니다

x = 10  # 전역변수

def foo():
  print('foo() x =', x)  # x 는 전역변수 x : <- '읽기 동작'

foo()

print(x)
# >>> 출력:
# foo() x = 10
# 10

# # local variable, local scope

def goo():
  y = 10  # goo() 내에서 선언한 지역변수
  print('goo() y =', y)  # y 는 goo() 함수 안에서만 사용 가능. 리턴하면 소멸됨.

goo()
print(y)  # 리턴후에 y 값 없다!
# >>> 출력:
# goo() y = 10
# NameError: name 'y' is not defined

z = 10  # 전역변수

def foo():
  z = 20  # z 는 foo() 의 지역변수!  '쓰기동작'
  print('foo() z =', z)

foo()

print(z)
# >>> 출력:
# foo() z = 20
# 10

z = 10  # 전역변수

def foo():
  global z  # z 는 전역변수 z 를 의미
  z = 20
  print('foo() z =', z)

foo()

print(z)
# >>> 출력:
# foo() z = 20
# 20
