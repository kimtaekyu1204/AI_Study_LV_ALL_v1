# # CSV 파일

# CSV 파일 : Comma Seperated Value  (콤마 , 로 구분된 데이터 포맷)
# TSV 파일 : Tab Seperated Value  (Tab \t 로 구분된 데이터 포맷)

# titanic.csv 파일

# 컬럼 의미
# survival - Survival (0 = No; 1 = Yes)
# Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# name - Name
# sex - Sex
# age - Age
# sibsp - Number of Siblings/Spouses Aboard
# parch - Number of Parents/Children Aboard
# ticket - Ticket Number
# fare - Passenger Fare
# cabin - Cabin
# embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# boat - Lifeboat (if survived)
# body - Body number (if did not survive and body was recovered)

import os

base_path = r'/content/drive/MyDrive/KoreaIT (코리아it)/250107 자연어처리/[AI자연어]/dataset(AI2501)'

file_path = os.path.join(base_path, 'titanic.csv')

file_path
# >>> 출력:
# '/content/drive/MyDrive/KoreaIT (코리아it)/250107 자연어처리/[AI자연어]/dataset(AI2501)/titanic.csv'

with open(file_path, 'r') as f:
  data = f.readlines()  # [line str]

data
# >>> 출력:
# ['PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked\n',
#  '1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S\n',
#  '2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C\n',
#  '3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S\n',
#  '4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S\n',
#  '5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S\n',
#  '6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q\n',
#  '7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S\n',
#  '8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S\n',
#  '9,1,3,"Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",female,27,0,2,347742,11.1333,,S\n',
#  '10,1,2,"Nasser, Mrs. Nicholas (Adele Achem)",female,14,1,0,237736,30.0708,,C\n',
#  '11,1,3,"Sandstrom, Miss. Marguerite Rut",female,4,1,1,PP 9549,16.7,G6,S\n',
#  '12,1,1,"Bonnell, Miss. Elizabeth",female,58,0,0,113783,26.55,C103,S\n',
#  '13,0,3,"Saundercock, Mr. William Henry",male,20,0,0,A/5. 2151,8.05,,S\n',
#  '14,0,3,"Andersson, Mr. Anders Johan",male,39,1,5,347082,31.275,,S\n',

data[1]
# >>> 출력:
# '1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S\n'

# 객실 등급별 생존률 구하기

"""
3등급] 총 491 명, 생존 119 명, 생존률 24.2%
1등급] 총 216 명, 생존 136 명, 생존률 63.0%
2등급] 총 184 명, 생존 87 명, 생존률 47.3%
"""

# ※ pandas 등 데이터 라이브러리는 사용하지 마세요
# >>> 출력:
# '\n3등급] 총 491 명, 생존 119 명, 생존률 24.2%\n1등급] 총 216 명, 생존 136 명, 생존률 63.0%\n2등급] 총 184 명, 생존 87 명, 생존률 47.3%\n'

passenger = data[1].split(',')
passenger
# >>> 출력:
# ['1',
#  '0',
#  '3',
#  '"Braund',
#  ' Mr. Owen Harris"',
#  'male',
#  '22',
#  '1',
#  '0',
#  'A/5 21171',
#  '7.25',
#  '',
#  'S\n']

passenger[1]  # 1번승객의 사망여부
# >>> 출력:
# '0'

passenger[2]  # 1번 승객의 객실등급
# >>> 출력:
# '3'

# 단계
# line 별로 읽어들이기  (첫 line 제거)

# line 을 , (콤마) 로 쪼개기 (split())

# 생존여부, 객실등급 데이터 추출 ([1],[2]번째 데이타)

# 객실등급별 집계 ( dict 결과{'1': [...], '2' : [...]})

# 출력

result = {}

for line in data[1:]:
  passenger = line.split(',')
  survived = int(passenger[1])  # "1", "0"
  pclass = int(passenger[2])   # "1" "2" "3"

  if result.get(pclass):
    result[pclass].append(survived)
  else:
    result[pclass] = [survived]


# result
#  {
#     1: [0, 0, 0, 1, 1, 1, ......  1, 0, 1],
#     2: [1, 1, 0, 1, ......  1, 0, 1],
#     3: [1, 1, 1, 1, .... 0, 0]
#  }

for pclass in result:
  total = len(result[pclass])
  survived = result[pclass].count(1)
  print("%d등급] 총 %d 명, 생존 %d 명, 생존률 %.1f%%" % (int(pclass), total, survived, (survived / total)))
# >>> 출력:
# 3등급] 총 491 명, 생존 119 명, 생존률 0.2%
# 1등급] 총 216 명, 생존 136 명, 생존률 0.6%
# 2등급] 총 184 명, 생존 87 명, 생존률 0.5%

# # sort() 와 sorted()

data = [10, 4, 5, 6, -3, 10, 12]

data
# >>> 출력:
# [10, 4, 5, 6, -3, 10, 12]

data.sort()

data
# >>> 출력:
# [-3, 4, 5, 6, 10, 10, 12]

data.sort(reverse=True)

data
# >>> 출력:
# [12, 10, 10, 6, 5, 4, -3]

myDict = {"b":5, "c":3, "a":2}

myDict
# >>> 출력:
# {'b': 5, 'c': 3, 'a': 2}

# myDict.sort()
# >>> 출력:
# AttributeError: 'dict' object has no attribute 'sort'

# ## sorted() 함수
# ```python
# sorted(iterable, /, *, key=None, reverse=False)
# ```
# https://docs.python.org/ko/3/library/functions.html#sorted
# 
# - iterable 의 항목들을 정렬하여 **'리스트'를 리턴**
# - key= 에 정렬 조건을 함수로 지정해줄수 있다.
#   - 매개변수 1개, 리턴값 1개 형태의 함수

data
# >>> 출력:
# [12, 10, 10, 6, 5, 4, -3]

sorted(data)  # 원본 변화없다.
# >>> 출력:
# [-3, 4, 5, 6, 10, 10, 12]

data
# >>> 출력:
# [12, 10, 10, 6, 5, 4, -3]

myDict
# >>> 출력:
# {'b': 5, 'c': 3, 'a': 2}

sorted(myDict)  # key 를 오름차순 정렬한 결과
# >>> 출력:
# ['a', 'b', 'c']

# ## 활용) dict 정렬하여 출력하기
# 기본적으로 dict 는 '순서' 가 없다
# - key 로 정렬하여 출력
# - value 로 정렬하여 출력

dict_items = myDict.items()
dict_items
# >>> 출력:
# dict_items([('b', 5), ('c', 3), ('a', 2)])

sorted(dict_items)  # key 오름차순
# >>> 출력:
# [('a', 2), ('b', 5), ('c', 3)]

sorted(dict_items, reverse=True)  # key 내림차순
# >>> 출력:
# [('c', 3), ('b', 5), ('a', 2)]

sorted(dict_items, key = lambda x: x[1])  # dict 의 value 오름차순 정렬된 리스트
# >>> 출력:
# [('a', 2), ('c', 3), ('b', 5)]

sorted(dict_items, key = lambda x: x[1], reverse=True)  # value 내림차순
# >>> 출력:
# [('b', 5), ('c', 3), ('a', 2)]

# ## 객실 순서대로 출력

result.keys()
# >>> 출력:
# dict_keys([3, 1, 2])

for pclass, survives in sorted(result.items()):
    total = len(survives)
    survived = int(survives.count(1))
    print("%d등급] 총 %d 명, 생존 %d 명, 생존률 %.1f%%" % (int(pclass), total, survived, (survived / total) * 100 ))
# >>> 출력:
# 1등급] 총 216 명, 생존 136 명, 생존률 63.0%
# 2등급] 총 184 명, 생존 87 명, 생존률 47.3%
# 3등급] 총 491 명, 생존 119 명, 생존률 24.2%
