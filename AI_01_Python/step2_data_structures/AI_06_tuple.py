# # tuple

# 여러개의 데이터를 담는 집합 데이터 타입들..
#1. list   :  순서있다.  중복허용,  mutable
#2. tuple  :  순서있다,  중복허용,  immutable
#3. set : 순서없다,  중복허용안함
#4. dict : key, value 쌍으로 구성, 순서없다.

# tuple 튜플 데이터 타입
# 콤마 ,  로 구분된 집합 데이터 타입

animals = ('dog', 'cat', 'dog', 'bird', 'shark')

animals
# >>> 출력:
# ('dog', 'cat', 'dog', 'bird', 'shark')

type(animals)
# >>> 출력:
# tuple

student = 'kim', 'park', 'lee'

student
# >>> 출력:
# ('kim', 'park', 'lee')

animals = ("dog")

animals
# >>> 출력:
# 'dog'

type(animals)
# >>> 출력:
# str

# 주의! 원소 '하나'짜리 tuple을 만드려면..
# 반드시 원소뒤에 콤마하나 붙여주세요
animals = ("dog",)

animals
# >>> 출력:
# ('dog',)

animals = "cat",

animals
# >>> 출력:
# ('cat',)

# # index, slice 가능

fruits = ('apple', 'banana', 'apple', 'kiwi')

fruits[0]
# >>> 출력:
# 'apple'

fruits[-3]
# >>> 출력:
# 'banana'

fruits[1:3]
# >>> 출력:
# ('banana', 'apple')

# tuple 은 immutable 하다
fruits[0] = 'mango' # 에러!
# >>> 출력:
# TypeError: 'tuple' object does not support item assignment

# # 기본연산

animals + animals
# >>> 출력:
# ('cat', 'cat')

animals = animals + animals

animals
# >>> 출력:
# ('cat', 'cat')

fruits * 3
# >>> 출력:
# ('apple',
#  'banana',
#  'apple',
#  'kiwi',
#  'apple',
#  'banana',
#  'apple',
#  'kiwi',
#  'apple',
#  'banana',
#  'apple',
#  'kiwi')

# tuple 은 언제 사용하나?
# 1. 주로 변경되지 말아야 할 데이터들
# 2. 복수의 값 '전달' 목적으로.

10, 20, 30
# >>> 출력:
# (10, 20, 30)

"a", "b", 'c'
# >>> 출력:
# ('a', 'b', 'c')

# # Assignment Unpacking
# tuple 은 iterable 객체다
# 
# 대입연산자와 iterable 을 사용하여 여러 개의 변수에 대입 가능  (이를 Assignment unpacking 이라 함)

# iterable 객체
# str, list, tuple, set, dict ... <- iterable 객체다

rec = [100, 200]  # [w, h]

width = rec[0]
height = rec[1]

print(width, height)
# >>> 출력:
# 100 200

w, h = rec

w, h
# >>> 출력:
# (100, 200)

w, h = (111, 222)

print(w, h)
# >>> 출력:
# 111 222

a, b, c = 10, 20, 30

a, b, c = "XYZ"

print(a, b, c)
# >>> 출력:
# X Y Z

# # count()

myList = [10, 20, 30, 10, 10, 10, 20, 20]

myList.count(10)
# >>> 출력:
# 4

myTuple = 10, 20, 30, 10, 10, 10, 20, 20

myTuple.count(10)
# >>> 출력:
# 4

myStr = "abbbbccbcaaabbbccbababababccbcbcb"

myStr.count("b")
# >>> 출력:
# 16
