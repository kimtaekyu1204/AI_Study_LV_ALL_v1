# 제어문 (Control)
#     프로그램의 기본 흐름 변경
#   1. 조건문 Conditional
#       - if

#   2. 순환문 Loop, Iteration
#       - for, while

# # 순환문 (Loop)

# 특정 코드(블럭)를 일정 회수 반복수행하는 경우 사용하는 구문
# 순환문(loop) 혹은 반복문(iteration) 이라고 한다

# 파이썬에는 while, for 구문이 순환문을 수행합니다

# # while 순환문

# while 순환문 구조,  조건식이 '참' 인 동안 수행문장을 반복수행

# while <조건식>:
#     <수행할 문장1>
#     <수행할 문장2>
#     <수행할 문장3>
#     ...
# else:
#      순환문 빠져나오기 전에 수행

num = 0
while num < 10:
  print(num)
  num += 1

print('종료', num)
# >>> 출력:
# 0
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 종료 10

# 순환문에서 중요한 것은
# 1. 몇번 순환을 하는가?
# 2. 순환하는 동안 변수값의 변화 범위는?
# 3. 순환문 종료후 변수값은?

# 위 순환문의 경우
# 1. 총 10번 순환을 했고
# 2. 순환하는 동안 num 변수값은 0 부터 9 까지 변화
# 3. 순환문 종료후 num 값은 10

# 20 ~ 10 출력
num = 20
while num >= 10:
  print(num, end=' ')
  num -= 2

print('종료', num)
# >>> 출력:
# 20 18 16 14 12 10 종료 8

# 구구단 2단을 출력해보자, while  사용
"""
2 x 1 = 2
2 x 2 = 4
2 x 3 = 6
2 x 4 = 8
2 x 5 = 10
2 x 6 = 12
2 x 7 = 14
2 x 8 = 16
2 x 9 = 18
"""
i = 1
while i <= 9:
  print(f'2 x {i} = {2 * i}')
  i += 1
# >>> 출력:
# 2 x 1 = 2
# 2 x 2 = 4
# 2 x 3 = 6
# 2 x 4 = 8
# 2 x 5 = 10
# 2 x 6 = 12
# 2 x 7 = 14
# 2 x 8 = 16
# 2 x 9 = 18

# # 중첩 순환문 (nested-loop)

# 중첩 순환문 (nested loop)
# 조건문 안에 조건문 블럭이 들어갈수 있듯이
# 순환문 안에도 얼마든지 순환문이 포함될수 있다..
# 조건문과 순환문 의 어떠한 조합도 가능하다

# 구구단 출력 : 2단 ~ 9단
# 중첩된 for 문

'''
2 x 1 = 2
2 x 2 = 4
...
2 x 9 = 18
3 x 1 = 3
...
4 x 1 = 4
…
...
9 x 9 = 81
'''

dan = 2

while dan <= 9: # 2단 ~ 9단

  mul = 1
  while mul <= 9:   # x1 ~ x9
    print(dan, 'x', mul, '=', dan * mul)
    mul += 1

  dan += 1
# >>> 출력:
# 2 x 1 = 2
# 2 x 2 = 4
# 2 x 3 = 6
# 2 x 4 = 8
# 2 x 5 = 10
# 2 x 6 = 12
# 2 x 7 = 14
# 2 x 8 = 16
# 2 x 9 = 18
# 3 x 1 = 3
# 3 x 2 = 6
# 3 x 3 = 9
# 3 x 4 = 12
# 3 x 5 = 15
# 3 x 6 = 18

# # break

# break
# 순환문 (while, for) 안 에서 순환문을 강제로 종료시키는 키워드

# break 는 감싸고 있는 가장 가까운 순환문을 종료합니다

n = 1
while n <= 100:
  n += 1

  if n == 10:
    break
  print(n)



print('종료', n)
# >>> 출력:
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 종료 10

# 연습
# 구구단 출력시
# 2단은 x 2 까지 출력
# 3단은 x 3 까지 출력
"""
2 단
2 x 1 = 2
2 x 2 = 4

3 단
3 x 1 = 3
3 x 2 = 6
3 x 3 = 9
...
"""

dan = 2

while dan <= 9: # 2단 ~ 9단

  mul = 1
  while mul <= 9:   # x1 ~ x9
    print(dan, 'x', mul, '=', dan * mul)
    if dan == mul: break
    mul += 1

  dan += 1
# >>> 출력:
# 2 x 1 = 2
# 2 x 2 = 4
# 3 x 1 = 3
# 3 x 2 = 6
# 3 x 3 = 9
# 4 x 1 = 4
# 4 x 2 = 8
# 4 x 3 = 12
# 4 x 4 = 16
# 5 x 1 = 5
# 5 x 2 = 10
# 5 x 3 = 15
# 5 x 4 = 20
# 5 x 5 = 25
# 6 x 1 = 6

# # continue

# continue
# 순환문 처음으로 돌아가기
# 순환문은 종료하지 않되, 특정 조건만 skip 하는 경우 사용

num = 1
while num <= 10:
  num += 1
  if num == 6: break
  print(num)
# >>> 출력:
# 2
# 3
# 4
# 5

num = 1
while num <= 10:
  num += 1
  if num == 6: continue
  print(num)
# >>> 출력:
# 2
# 3
# 4
# 5
# 7
# 8
# 9
# 10
# 11

# 정수를 계속해서 입력받다가, 0 이 입력되면
# 그때까지의 정수의 '합' 과 '평균' 을 출력

"""
[입력예]
10
20
30
0

[출력예]
합: 60
평균: 20.0

"""
total = 0 # 합
cnt = 0 # 개수

while True:  # 무한루프
  n = int(input())
  if n == 0: break   # 0 입력하면 무한루프 종료
  total += n  # 합계 누적 합산
  cnt += 1   # 개수 누적

print('합:', total)
print('평균:%.1f' % (total / cnt))
# >>> 출력:
# 112
# -23
# 56
# 789
# 0
# 합: 934
# 평균:233.5

# # for 순환문

# for 순환문
# 기존의 다른 프로그래밍 언어의 for 문에 비해 파이썬은 매우 직관적이고 사용하기 편리

# 구문]
# for 변수 in iterable객체:
#     수행할 문장1
#     수행할 문장2
#     ....
# else:
#      순환문 빠져나오기 전에 수행

# iterable한 객체들 => range(숫자), str, list, set, tuple, dict ...

range(3)   # 0, 1, 2 의 숫자가 담긴 iterable 객체 생성
# >>> 출력:
# range(0, 3)

range(5, 8)  # 5, 6, 7 의 ...  ...
# >>> 출력:
# range(5, 8)

for i in range(3):
  print('hello python', i)
# >>> 출력:
# hello python 0
# hello python 1
# hello python 2

for i in range(5, 9):
  print('hello python', i)
# >>> 출력:
# hello python 5
# hello python 6
# hello python 7
# hello python 8

for i in range(4, 15, 3):   # 세번째 매개변수는 step= 값
  print('hello python', i)
# >>> 출력:
# hello python 4
# hello python 7
# hello python 10
# hello python 13

for i in range(10, 0, -2):
  print('hello python', i)
# >>> 출력:
# hello python 10
# hello python 8
# hello python 6
# hello python 4
# hello python 2

# for i in 100:
#   print('hello python', i)
# >>> 출력:
# TypeError: 'int' object is not iterable

# list 로 for 순환
animals = ['dog', 'cat', 'bird', 'puppy', 'kitty']
len(animals)
# >>> 출력:
# 5

for i in animals:
  print('I love', i)
# >>> 출력:
# I love dog
# I love cat
# I love bird
# I love puppy
# I love kitty

for ch in "Hello":
  print(ch)
# >>> 출력:
# H
# e
# l
# l
# o

data = [11, 22, 33, 44]

# [22, 44, 66, 88]

result = []
for d in data:
  result.append(d * 2)

print(result)
# >>> 출력:
# [22, 44, 66, 88]

data = [11, 22, 33, 44]

# [22, 44]

# 도전!

result = []
for d in data:
  if d % 2 == 0:
    result.append(d)
print(result)
# >>> 출력:
# [22, 44]

result = []
for d in data:
  (d % 2) or result.append(d)
print(result)
# >>> 출력:
# [22, 44]

# dict 는 for문에서  key  값이 추출됨.
myDict = {'name':'hong', 'age':24, 'grade':4}

for key in myDict:
  print(key)
# >>> 출력:
# name
# age
# grade

myDict.keys()
# >>> 출력:
# dict_keys(['name', 'age', 'grade'])

for k in myDict.keys():
  print(k)
# >>> 출력:
# name
# age
# grade

myDict.values()
# >>> 출력:
# dict_values(['hong', 24, 4])

for v in myDict.values():
  print(v)
# >>> 출력:
# hong
# 24
# 4

myDict.items()  # (k, v) 들의 iterable 객체
# >>> 출력:
# dict_items([('name', 'hong'), ('age', 24), ('grade', 4)])

for item in myDict.items():
  print(type(item), item)
# >>> 출력:
# <class 'tuple'> ('name', 'hong')
# <class 'tuple'> ('age', 24)
# <class 'tuple'> ('grade', 4)

for item in myDict.items():
  print(item[0], ':', item[1])
# >>> 출력:
# name : hong
# age : 24
# grade : 4

k, v = [10, 20]

print(k, v)
# >>> 출력:
# 10 20

for k, v in myDict.items():
  print(k, ":", v)
# >>> 출력:
# name : hong
# age : 24
# grade : 4

# # enumerate(iterable)
# (index, data) 쌍의 iterable 객체 리턴

animals
# >>> 출력:
# ['dog', 'cat', 'bird', 'puppy', 'kitty']

for animal in animals:
  print(animal)
# >>> 출력:
# dog
# cat
# bird
# puppy
# kitty

for x in enumerate(animals):
  print(x)
# >>> 출력:
# (0, 'dog')
# (1, 'cat')
# (2, 'bird')
# (3, 'puppy')
# (4, 'kitty')

for idx, animal in enumerate(animals):
  print(idx, animal)
# >>> 출력:
# 0 dog
# 1 cat
# 2 bird
# 3 puppy
# 4 kitty

# # zip()
# - zip(iterable, iterable, ...)
# - 매개변수 iterable 데이터들의 각 아이템들을 묶은 tuple들의 iterable 객체 생성

zip([10, 20, 30], ['a','b','c'])
# >>> 출력:
# <zip at 0x7ab42ef23c00>

list(zip([10, 20, 30], ['a','b','c']))
# >>> 출력:
# [(10, 'a'), (20, 'b'), (30, 'c')]

list(zip([10, 20, 30], ['a','b','c','d']))
# >>> 출력:
# [(10, 'a'), (20, 'b'), (30, 'c')]

for x in zip([10, 20, 30], ['a','b','c']):
  print(x)
# >>> 출력:
# (10, 'a')
# (20, 'b')
# (30, 'c')

for num, ch in zip([10, 20, 30], ['a','b','c']):
  print(num, ch)
# >>> 출력:
# 10 a
# 20 b
# 30 c

for x in zip([10, 20, 30], ['a','b','c'], "XYZ"):
  print(x)
# >>> 출력:
# (10, 'a', 'X')
# (20, 'b', 'Y')
# (30, 'c', 'Z')

# # break, continue 와 for

for anim in animals:
  print(anim)
  if len(anim) >= 4: break
# >>> 출력:
# dog
# cat
# bird

# # Comprehension

# List Comprehension  -->  list 안에 포함된 for문  --> list 생성
# Dict Comprehension  -->  dict 안에 포함된 for문 --> dict 생성
# Set Comprehension -->  set 안에 포함된 for문 --> set 생성

# ## List Comprehension

# List Comprehension 구문
# [표현식 for 항목 in 반복가능객체 (if 조건)]

# 주어진 리스트의 원소 데이터 들을 * 3 하여 새로운 리스트 작성하기
# ex) [1, 2, 3, 4] ==> [3, 6, 9, 12]

# 일단 지금까지 배운 방법으로 한다면

a = [1, 2, 3, 4]
result = []
for num in a:
  result.append(num * 3)

print(result)
# >>> 출력:
# [3, 6, 9, 12]

[num * 3 for num in a]
# >>> 출력:
# [3, 6, 9, 12]

animals
# >>> 출력:
# ['dog', 'cat', 'bird', 'puppy', 'kitty']

[len(animal) for animal in animals]
# >>> 출력:
# [3, 3, 4, 5, 5]

[i for i in range(10)]
# >>> 출력:
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

[i for i in range(10) if i % 2 == 0]
# >>> 출력:
# [0, 2, 4, 6, 8]

['hello' for i in range(10) if i % 2 == 0]
# >>> 출력:
# ['hello', 'hello', 'hello', 'hello', 'hello']

#  짝수에만 3을 곱하여 담고 싶다면 다음과 같이 "if 조건"을 사용할 수 있다
# [1, 2, 3, 4] --> [6, 12]

a = [1, 2, 3, 4]

# 실습!

[a * 3 for a in a if a % 2 == 0]
# >>> 출력:
# [6, 12]

# 아래와 같이 짝수에만 적용되게 하려면? 홀수는 그대로.
# [1, 2, 3, 4]
#  ↓  ↓  ↓  ↓
# [1, 6, 3, 12]   <- 2, 4 에만 x3 적용

a = [1, 2, 3, 4]

[(i * 3 if i % 2 == 0 else i) for i in a ]
# >>> 출력:
# [1, 6, 3, 12]

# ## Set Comprehension

[num % 3 for num in range(10)]
# >>> 출력:
# [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]

{num % 3 for num in range(10)}
# >>> 출력:
# {0, 1, 2}

# ## Dict Comprehension

{num:num % 3 for num in range(10)}
# >>> 출력:
# {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2, 6: 0, 7: 1, 8: 2, 9: 0}

fruits = {"apple": "red", "banana": "yellow", "peach": "pink"}

# {'apple': 'red', 'peach': 'pink'}

{k:v for k, v in fruits.items() if k != 'banana'}
# >>> 출력:
# {'apple': 'red', 'peach': 'pink'}

{
    k:v
    for k, v in fruits.items()
    if k != 'banana'
}
# >>> 출력:
# {'apple': 'red', 'peach': 'pink'}

{k:v for k, v in fruits.items() if v != 'red'}
# >>> 출력:
# {'banana': 'yellow', 'peach': 'pink'}

# # 도전과제

# 1부터 100 까지의 숫자 출력.  한줄에 10개씩 출력
# 숫자 중에 3, 6, 9 가 있으면 숫자대신 '*' 출력
# 단! 문자열 함수 쓰지 말기

"""
    1    2    *    4    5    *    7    8    *   10
   11   12    *   14   15    *   17   18    *   20
   21   22    *   24   25    *   27   28    *    *
    *    *    *    *    *    *    *    *    *   40
   41   42    *   44   45    *   47   48    *   50
   51   52    *   54   55    *   57   58    *    *
    *    *    *    *    *    *    *    *    *   70
   71   72    *   74   75    *   77   78    *   80
   81   82    *   84   85    *   87   88    *    *
    *    *    *    *    *    *    *    *    *  100
"""
None

n = 100
i = 1
while i <= n:  # 1 ~ 100까지
  digit1 = i % 10  #  1의 자리
  digit2 = i // 10 # 10의 자리

  # 3, 6, 9 여부에 따른 출력
  if digit1 == 3 or digit1 == 6 or digit1 == 9 \
    or digit2 == 3 or digit2 == 6 or digit2 == 9:
    print(f'{"*":>5}', end='')
  else:
    print(f'{i:5}', end='')

  i % 10 or print()  # 10개 출력후 줄바꿈.
  i += 1
# >>> 출력:
#     1    2    *    4    5    *    7    8    *   10
#    11   12    *   14   15    *   17   18    *   20
#    21   22    *   24   25    *   27   28    *    *
#     *    *    *    *    *    *    *    *    *   40
#    41   42    *   44   45    *   47   48    *   50
#    51   52    *   54   55    *   57   58    *    *
#     *    *    *    *    *    *    *    *    *   70
#    71   72    *   74   75    *   77   78    *   80
#    81   82    *   84   85    *   87   88    *    *
#     *    *    *    *    *    *    *    *    *  100
