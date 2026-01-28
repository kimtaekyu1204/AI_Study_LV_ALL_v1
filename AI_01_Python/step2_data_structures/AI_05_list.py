# # list

# 여러개의 데이터를 담는 집합 데이터 타입들..
#1. list   :  순서있다.  중복허용,  mutable
#2. tuple  :  순서있다,  중복허용,  immutable
#3. set : 순서없다,  중복허용안함
#4. dict : key, value 쌍으로 구성, 순서없다.

"""
list

# list 는   [   ]  으로 만든다
# 데이터(원소) 들은 , 콤마로 구분한다
# 각 데이터(원소) 들은 어떠한 타입도 가능하다
"""
None

animals = ['dog', 'cat', 'bird', 'fish']

animals
# >>> 출력:
# ['dog', 'cat', 'bird', 'fish']

type(animals)
# >>> 출력:
# list

animals = [
    'dog',
    'cat',
    'bird',
    'fish',  # 마지막 원소 뒤에 콤마 붙여도 OK
    ]

animals
# >>> 출력:
# ['dog', 'cat', 'bird', 'fish']

# # 인덱스 (index)
# 'list 는 **순서** 가 있다'

animals[0] # index  첫번째 원소는 0 부터 시작한다 (0-base index)
# >>> 출력:
# 'dog'

animals[1]
# >>> 출력:
# 'cat'

type(animals[0])  # 원소의 타입
# >>> 출력:
# str

animals[4] # index 범위 벗어나면 IndexError 발생
# >>> 출력:
# IndexError: list index out of range

animals
# >>> 출력:
# ['dog', 'cat', 'bird', 'fish']

animals[-1]  # 음수 인덱싱 지원!
# >>> 출력:
# 'fish'

# # 원소 변경 가능
# list 는 **mutable** 하다

animals[2] = 'mouse'

animals
# >>> 출력:
# ['dog', 'cat', 'mouse', 'fish']

len('hello')
# >>> 출력:
# 5

len(animals)  # 리스트 원소 개수
# >>> 출력:
# 4

# # 데이터 중복 가능

fruits = ['apple', 'banana', "apple", '''kiwi''']

fruits
# >>> 출력:
# ['apple', 'banana', 'apple', 'kiwi']

# 비어있는 리스트
data = []

data
# >>> 출력:
# []

len(data)
# >>> 출력:
# 0

# # 슬라이싱(slicing)
# 일부분 추출.  
# 
# 범위연산자 [:] 사용
# 
# step 값 사용가능

colors = ['red', 'orange', 'blue', 'green', 'white', 'black']

colors
# >>> 출력:
# ['red', 'orange', 'blue', 'green', 'white', 'black']

colors[0:3]  # 0 부터 3 전까지
# >>> 출력:
# ['red', 'orange', 'blue']

colors[3]
# >>> 출력:
# 'green'

colors[1:4]
# >>> 출력:
# ['orange', 'blue', 'green']

print(colors)

colors[:2]   # 처음부터 2 전까지
# >>> 출력:
# ['red', 'orange', 'blue', 'green', 'white', 'black']
# ['red', 'orange']

colors[2:]   # 2부터 끝까지
# >>> 출력:
# ['blue', 'green', 'white', 'black']

colors[:]  # 전체
# >>> 출력:
# ['red', 'orange', 'blue', 'green', 'white', 'black']

# step 값
myList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(myList[1:7])
# >>> 출력:
# [1, 2, 3, 4, 5, 6]

myList[1:7:2]  # step 2  건너띄는 값
# >>> 출력:
# [1, 3, 5]

myList[3::3]
# >>> 출력:
# [3, 6, 9]

myList[::2]
# >>> 출력:
# [0, 2, 4, 6, 8, 10]

myList
# >>> 출력:
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

myList[5:0]
# >>> 출력:
# []

myList[5:0:-1]  # 5 부터 0전까지 -1 step
# >>> 출력:
# [5, 4, 3, 2, 1]

myList[::-1]  # 역순, 뒤집기
# >>> 출력:
# [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

# # append()

# append() :  데이터 (원소) 추가 , 리스트 뒤에
#  ※ list 는 mutable 하기 때문에 가능

animals
# >>> 출력:
# ['dog', 'cat', 'mouse', 'fish']

animals.append('cow')  # append() 의 리턴값은 None

animals
# >>> 출력:
# ['dog', 'cat', 'mouse', 'fish', 'cow']

# ※ None 을 리턴하는 데이터 조작함수는 원본을 변경하는 함수인 경우가 많다.

animals[1:3]
# >>> 출력:
# ['cat', 'mouse']

animals
# >>> 출력:
# ['dog', 'cat', 'mouse', 'fish', 'cow']

# # del() 원소 삭제

del(animals[1])

animals
# >>> 출력:
# ['dog', 'mouse', 'fish', 'cow']

# # list 연산

print(animals)
print(colors)
# >>> 출력:
# ['dog', 'mouse', 'fish', 'cow']
# ['red', 'orange', 'blue', 'green', 'white', 'black']

animals + colors  # list + list => list
# >>> 출력:
# ['dog',
#  'mouse',
#  'fish',
#  'cow',
#  'red',
#  'orange',
#  'blue',
#  'green',
#  'white',
#  'black']

animals * 2
# >>> 출력:
# ['dog', 'mouse', 'fish', 'cow', 'dog', 'mouse', 'fish', 'cow']

# # 1차원 리스트
# 인덱스를 하나만 사용하는 리스트

colors[0]
# >>> 출력:
# 'red'

# 리스트의 원소는 '어떠한 타입' 도 가능 .  심지어 '리스트'도 원소로 가질수 있다!

mylist = [100, 200, 'John', 0.29, False, None]

mylist
# >>> 출력:
# [100, 200, 'John', 0.29, False, None]

# # 다차원 리스트
# 리스트의 원소가 리스트
# 
# 인덱스를 여러개 사용

data = [
    [1, 2, 3],
    [10, 20, 30],
    [100, 200, 300],
]

data
# >>> 출력:
# [[1, 2, 3], [10, 20, 30], [100, 200, 300]]

len(data)
# >>> 출력:
# 3

data[0]
# >>> 출력:
# [1, 2, 3]

data[0][0]  # 2차원 리스트는 [][] 인덱스 2개 사용하여 원소에 접근.
# >>> 출력:
# 1

# 200
data[2][1]
# >>> 출력:
# 200

data
# >>> 출력:
# [[1, 2, 3], [10, 20, 30], [100, 200, 300]]

data[0]
# >>> 출력:
# [1, 2, 3]

data[0][0]
# >>> 출력:
# 1

# data --> 2차원 list
# data[0] --> 1차원 list
# data[0][0] --> 0차원 ,  scalar 값

# ※ 인덱싱을 한다는 것은 차원 -1 축소 발생

# 2차원 리스트의 원소는 1차원 리스트
# 3 ..           ..   2 ..
# 99 ..          ..   98 ..
# n 차원 리스트의 원소는 n-1 차원 리스트

# 다차원 배열에서 아래과 같은 slicing 가능
data = [
    [1, 2, 3, 4, 5],
    [10, 20, 30, 40, 50],
    [11, 22, 33, 44, 55]
]

data
# >>> 출력:
# [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [11, 22, 33, 44, 55]]

# data 는 3 x 5  2차원 배열이라고도 함.
#        3행 5열.

data[1][:3]
# >>> 출력:
# [10, 20, 30]

data[0:2]    # 슬라이싱 결과 차원변화 없다
# >>> 출력:
# [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]

# # .sort() 함수
# 리스트 정렬

animals
# >>> 출력:
# ['dog', 'mouse', 'fish', 'cow']

animals.sort()

animals
# >>> 출력:
# ['cow', 'dog', 'fish', 'mouse']

animals.sort(reverse=True) # 내림차순 정렬

animals
# >>> 출력:
# ['mouse', 'fish', 'dog', 'cow']

animals.reverse()  # reverse() 뒤집기

animals
# >>> 출력:
# ['cow', 'dog', 'fish', 'mouse']

# # in 연산자

"dog" in animals
# >>> 출력:
# True

"shark" in animals
# >>> 출력:
# False

# # str 의 index, slice, in

str2 = "hello python"
str2
# >>> 출력:
# 'hello python'

str2[0]
# >>> 출력:
# 'h'

str2[1]
# >>> 출력:
# 'e'

str2[-4]
# >>> 출력:
# 't'

str2[:4]
# >>> 출력:
# 'hell'

print(str2)
str2[::2]
# >>> 출력:
# hello python
# 'hlopto'

str2[::-1]
# >>> 출력:
# 'nohtyp olleh'

"p" in str2
# >>> 출력:
# True

"P" in str2
# >>> 출력:
# False

str2
# >>> 출력:
# 'hello python'

str2[0] = 'Y'  # str 은 immutable(불변) 하다!
# >>> 출력:
# TypeError: 'str' object does not support item assignment

# # list 와 str

# str.join(list):  list -> 하나의  str

animals
# >>> 출력:
# ['cow', 'dog', 'fish', 'mouse']

"--".join(animals)
# >>> 출력:
# 'cow--dog--fish--mouse'

":".join(["23", "45", "12"])
# >>> 출력:
# '23:45:12'

# split() 에 매개변수 없이 사용하면 공백 기준으로 문자열 쪼개짐
# 공백 : 띄어쓰기, 탭, 줄바꿈...

myStr = "Hello Python 2024"

myStr.split()
# >>> 출력:
# ['Hello', 'Python', '2024']

myStr = "2025-01-16"

myStr.split("-")
# >>> 출력:
# ['2025', '01', '16']

# ※ 위와 같이 특정 문자열을 기준으로 동작할때
#  그러한 역할을 하는 문자열을 delimiter 라고 한다.

# # 형변환 list()

myStr
# >>> 출력:
# '2025-01-16'

list(myStr)
# >>> 출력:
# ['2', '0', '2', '5', '-', '0', '1', '-', '1', '6']

myStr = "animals"

# "a-n-i-m-a-l-s" 만들어보기

"-".join(myStr)
# >>> 출력:
# 'a-n-i-m-a-l-s'

"-".join(list(myStr))
# >>> 출력:
# 'a-n-i-m-a-l-s'
