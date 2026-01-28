# # set

# 여러개의 데이터를 담는 집합 데이터 타입들..
#1. list   :  순서있다.  중복허용,  mutable
#2. tuple  :  순서있다,  중복허용,  immutable
#3. set : 순서없다,  중복허용안함
#4. dict : key, value 쌍으로 구성, 순서없다.

# set
# 만드는 방법
# 1.set(iterable)  함수로 만들수도 있고
# 2. {  } 으로 만들수도 있다.

animals = {"dog", "cat", "dog", "bird"}

animals  # 중복허용 안함, 순서 유지 안됨.
# >>> 출력:
# {'bird', 'cat', 'dog'}

type(animals)
# >>> 출력:
# set

len(animals)
# >>> 출력:
# 3

animals[0]
# >>> 출력:
# TypeError: 'set' object is not subscriptable

animals = ["dog", "dog", "cat", "bird"]
# 위 list 에서 중복된 데이터를 제거하고
# 싶다면?

list(set(animals))
# >>> 출력:
# ['bird', 'cat', 'dog']

# 빈 set 만들기

a = {}  # 이건 dict 다!

type(a)
# >>> 출력:
# dict

a = set()  # 빈 set 만들기

a
# >>> 출력:
# set()

type(a)
# >>> 출력:
# set

len(a)
# >>> 출력:
# 0

# # add(), remove()

a.add("dog")

a
# >>> 출력:
# {'dog'}

a.add('cat')

a
# >>> 출력:
# {'cat', 'dog'}

a.remove('dog')

a
# >>> 출력:
# {'cat'}

a.remove('dog')
# >>> 출력:
# KeyError: 'dog'

# in 연산자
'bird' in a
# >>> 출력:
# False

'cat' in a
# >>> 출력:
# True
