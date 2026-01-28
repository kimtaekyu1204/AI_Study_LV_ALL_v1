# # iterable

animals = ['dog', 'cat', 'bird', 'puppy', 'kitty']
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

student = {
    'name':'홍반장',
    'email':'hong@mail.com',
    'address':'강남역'
}

for key in student:
  print(key )
# >>> 출력:
# name
# email
# address

# 현재 언어들 특히 high-level language 일수록
# for animal in animals <-- 이런 형태의 반복문이 가능.

# 파이썬에서 이렇게 가능한 이유
# 1. iteration  ( loop action )   animals
#     가령 animals 는 5번의 iteration 이 발생하는 것이다

# 2. iterable 정의 ★
#     특정 객체가 'iterable 하다'는 것은
#     그 객체에 .__iter__() 함수가 정의되어 있어야 하고
#     __iter__() 함수의 의 결과로 'iterator 객체'가 리턴되어야 함.

#     iterable 하다는 것은 곧 for 문 돌릴수 있다는 거다
#         range(), list, str, dict, tuple, set  모두 iterable 한거다

# 3. iterator 정의 ★
#         __next__() 함수 정의되어 있고
#         __next__() 는 호출될때마다 iterable 객체의 원소들을 하나씩 리턴
#         모든원소들이 리턴된뒤 __next__() 를 호출하면 StopIteration Exception 발생

animals
# >>> 출력:
# ['dog', 'cat', 'bird', 'puppy', 'kitty']

animals.__iter__
# >>> 출력:
# <method-wrapper '__iter__' of list object at 0x7d834f990980>

itr = animals.__iter__()  # __iter__() 의 리턴값은 iterator 객체
itr
# >>> 출력:
# <list_iterator at 0x7d835c604c10>

itr.__next__
# >>> 출력:
# <method-wrapper '__next__' of list_iterator object at 0x7d835c604c10>

itr.__next__()
# >>> 출력:
# 'dog'

itr.__next__()
# >>> 출력:
# 'cat'

itr.__next__()
# >>> 출력:
# 'bird'

itr.__next__()
# >>> 출력:
# 'puppy'

itr.__next__()
# >>> 출력:
# 'kitty'

itr.__next__()  #  모든원소들이 리턴된뒤 __next__() 를 호출하면 StopIteration Exception 발생
# >>> 출력:
# StopIteration: 

# for animal in animals:  를 수행하면  내부적으로 다음과 같은 순으로 동작하는 것이다
#     animals => (__iter__) => animals iterator => (__next__)

# iterable 객체에서 첫번째 데이터 꺼내기
animals.__iter__().__next__()
# >>> 출력:
# 'dog'

student
# >>> 출력:
# {'name': '홍반장', 'email': 'hong@mail.com', 'address': '강남역'}

student.__iter__().__next__()
# >>> 출력:
# 'name'

student.items()  # items() 결과도 iterable 객체다.
# >>> 출력:
# dict_items([('name', '홍반장'), ('email', 'hong@mail.com'), ('address', '강남역')])

itr = student.items().__iter__()

itr.__next__()
# >>> 출력:
# StopIteration: 

# # iterable 여부 확인하기

from collections.abc import Iterable

var = [1, 3, 5, 7]   # True
var = 100  # False
var = (10, 20, 20) # True
var = 'hello' # True
isinstance(var, Iterable)
# >>> 출력:
# (True, False)

# list(iterable)

list("hello")
# >>> 출력:
# ['h', 'e', 'l', 'l', 'o']

list({'name': 'hong', 'age': 20})
# >>> 출력:
# ['name', 'age']
