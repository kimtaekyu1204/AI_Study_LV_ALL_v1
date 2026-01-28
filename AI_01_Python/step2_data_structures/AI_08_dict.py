# # dict
# 딕셔너리

# 여러개의 데이터를 담는 집합 데이터 타입들..
#1. list   :  순서있다.  중복허용,  mutable
#2. tuple  :  순서있다,  중복허용,  immutable
#3. set : 순서없다,  중복허용안함
#4. dict : key, value 쌍으로 구성, 순서없다.

# dict : 딕셔너리

# dictionary 데이터 타입은  key-value 쌍으로 저장되는 데이터 집합이다.

# {Key1:Value1, Key2:Value2, Key3:Value3 ...}
# "이름" = "홍길동", "생일" = "몇 월 몇 일"   과 같은 자료형 담음

# 기존의 list, tuple 등은 value 중심
# 그러나 이 또한 알고 보면 key-value 쌍으로 구성

# 순서는 고정이 안된다
# key 는 중복 안된다.

student = {"name":"손오공", "email":"sonson@mail.com"}

type(student)
# >>> 출력:
# dict

student
# >>> 출력:
# {'name': '손오공', 'email': 'sonson@mail.com'}

len(student)   # key-value 쌍의 개수
# >>> 출력:
# 2

# # value 읽기

student["name"]   # 방법1
# >>> 출력:
# '손오공'

student.get("name")  # 방법2
# >>> 출력:
# '손오공'

print(student)
# >>> 출력:
# {'name': '손오공', 'email': 'sonson@mail.com'}

student['age']  # KeyError: 'age'
# >>> 출력:
# KeyError: 'age'

student.get('age')  # key 가 없으면 None

# get()  을 사용하면 예외적인 상황에서도
# 동작 가능하게 처리 가능
# student.get(key, default) : key 값이 없으면 default 값으로 리턴

student.get('age', 20)
# >>> 출력:
# 20

# # 추가, 수정, 삭제

# 수정
student['name'] = '홍길동'

student
# >>> 출력:
# {'name': '홍길동', 'email': 'sonson@mail.com'}

# 추가
student['age'] = 34 # 기존에 없던 key 값에 대입하면 새로운 k-v 쌍 추가됨.

student
# >>> 출력:
# {'name': '홍길동', 'email': 'sonson@mail.com', 'age': 34}

# 삭제
del(student['email'])

student
# >>> 출력:
# {'name': '홍길동', 'age': 34}

# # dict.update()
# 해당 key값 업데이트<br>
# 해당 key 가 없으면 새로이 추가

student.update({'name': '이순신'})  # None 리턴!!

student
# >>> 출력:
# {'name': '이순신', 'age': 34}

student.update({'address': '한양', 'gender': '남'})

student
# >>> 출력:
# {'name': '이순신', 'age': 13, 'gender': '남', 'address': '한양'}

# # key, value 의 타입
# - **value** 는 어떠한 타입이라도 가능!
# - **key** 는 hash 가능한 타입만 가능!   (ex: int, float, str, bool, tuple..)

dict1 = {
    1: 'haha',
    2: 'hehe',
    2: 'hehe',  # key 는 중복 불가!
    # [10]: 3.14,   # list, set, dict 등은 dict 의 key로 사용불가
    "two": {
        3.14: "pi",
        "pi": 3.14,
    },
    False: [10, 20, 30],
    (1, 2): 'nice',    # tuple 이 key 로 사용될수 있다!  --> 나중에 Numpy, Pandas 등에서 활용됨.
}

dict1
# >>> 출력:
# {1: 'haha',
#  2: 'hehe',
#  'two': {3.14: 'pi', 'pi': 3.14},
#  False: [10, 20, 30],
#  (1, 2): 'nice'}

dict1[1]
# >>> 출력:
# 'haha'

print(type(dict1))
print(type(dict1[1]))
# >>> 출력:
# <class 'dict'>
# <class 'str'>

dict1['two']
# >>> 출력:
# {3.14: 'pi', 'pi': 3.14}

dict1['two'][3.14]
# >>> 출력:
# 'pi'

dict1[False]
# >>> 출력:
# [10, 20, 30]

dict1[False][1]
# >>> 출력:
# 20

dict1[(1, 2)]
# >>> 출력:
# 'nice'

dict1[1, 2]
# >>> 출력:
# 'nice'

# # dict.keys(),  dict.values(), dict.items()

student = {
        "name":"최현진",
        "email":"choi@mail.com",
        "age": 23,
        "addr" : "서울"
        }

student
# >>> 출력:
# {'name': '최현진', 'email': 'choi@mail.com', 'age': 23, 'addr': '서울'}

student.keys()  # key 들로만 구성된 iterable 객체 리턴함.
# >>> 출력:
# dict_keys(['name', 'email', 'age', 'addr'])

student.values()  # value 들로만 구성된 iterable 객체 리턴함.
# >>> 출력:
# dict_values(['최현진', 'choi@mail.com', 23, '서울'])

student.items()  # (key, value) ... 들로 구성된 iterable 객체 리턴함.
# >>> 출력:
# dict_items([('name', '최현진'), ('email', 'choi@mail.com'), ('age', 23), ('addr', '서울')])

# # in 연산자
# key 가 존재하는지 여부

student
# >>> 출력:
# {'name': '최현진', 'email': 'choi@mail.com', 'age': 23, 'addr': '서울'}

'name' in student
# >>> 출력:
# True

'서울' in student
# >>> 출력:
# False

'서울' in student.values()
# >>> 출력:
# True

# # 비어있는(empty) 데이터

a = []
print(a)
print(len(a), type(a))
# >>> 출력:
# []
# 0 <class 'list'>

a = ""
print(a)
print(len(a), type(a))
# >>> 출력:
# 
# 0 <class 'str'>

a = ()
print(a)
print(len(a), type(a))
# >>> 출력:
# ()
# 0 <class 'tuple'>

a = {}
print(a)
print(len(a), type(a))
# >>> 출력:
# {}
# 0 <class 'dict'>

a = set()
print(a)
print(len(a), type(a))
# >>> 출력:
# set()
# 0 <class 'set'>
