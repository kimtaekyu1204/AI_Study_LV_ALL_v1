# # 문자열 함수(메소드)들

# # upper(), lower()

# upper() : 문자열을 전부 대문자로 만드는 함수
#  lower() : 문자열을 전부 소문자로 만드는 함수

str1 = "Apple"
str1
# >>> 출력:
# 'Apple'

str1.upper()
# >>> 출력:
# 'APPLE'

str1
# >>> 출력:
# 'Apple'

str1.lower()
# >>> 출력:
# 'apple'

'Car' == 'car'
# >>> 출력:
# False

'Car'.upper()  == 'car'.upper()
# >>> 출력:
# True

# # strip()
# 문자을의 좌우 공백 제거

a = "    Hello, World!    "
print('[%s]' % a)
# >>> 출력:
# [    Hello, World!    ]

print('[%s]' % a.strip())
# >>> 출력:
# [Hello, World!]

# # replace()

# replace(a, b)  문자열 치환
# 주어진 문자열에서 a 를 찾아 b 로 치환

a.upper().strip().replace('H', 'J')
# >>> 출력:
# 'JELLO, WORLD!'

# 대부분의 문자열 메소드(함수)는 메소드 체이닝 (method chaining 됨) 가능2

# 도전
# "  21,300원 "  --> 정수 21300 으로 바꾸기
a = "  21,300원 "

int(a.strip().replace(',','').replace('원',''))
# >>> 출력:
# 21300

int(a.strip().replace(',', '')[:-1])
# >>> 출력:
# 21300

# # split(), join()

data = "사과,바나나,파인애플,포도,복숭아"

data.split(',')
# >>> 출력:
# ['사과', '바나나', '파인애플', '포도', '복숭아']

"강아지 고양이 도마뱀".split() # 매개변수 없으면 디폴트로 '공백 기준' 으로 쪼갬.
# >>> 출력:
# ['강아지', '고양이', '도마뱀']

"    강아지    고양이    도마뱀     ".split()
# >>> 출력:
# ['강아지', '고양이', '도마뱀']

"    강아지    고양이    도마뱀     ".split(' ')
# >>> 출력:
# ['',
#  '',
#  '',
#  '',
#  '강아지',
#  '',
#  '',
#  '',
#  '고양이',
#  '',
#  '',
#  '',
#  '도마뱀',
#  '',
#  '',

data = ["A", "B", "C", "D", "E"]

",".join(data)
# >>> 출력:
# 'A,B,C,D,E'

# # index(), find()

str3 = "hello python"

str3.index('lo')   # 주어진 문자열을 '찾으면' 원본문자열 내의 index 리턴 (0 이상의 값)
# >>> 출력:
# 3

str3.index('o')
# >>> 출력:
# 4

str3.index('l')
# >>> 출력:
# 2

# str3.index('xx')  # 발견하지 못하면 에러 ValueError: substring not found
# >>> 출력:
# ValueError: substring not found

str3.find('lo')
# >>> 출력:
# 3

str3.find('xx')  # 발견하지 못하면 -1 리턴
# >>> 출력:
# -1

# # count()

# 문자열 내에서 특정 문자열 패턴의 발견 개수 리턴

str4 = "aabababbabbaababb"

str4.count("a")
# >>> 출력:
# 8

str4.count("ba")
# >>> 출력:
# 5

# # startswith(), endswith()

# 문자열이 특정 문자열로 시작/종료 하는지 여부
str5 = "https://www.nytimes.com/2000/01/01/news/visions-identity-a-generation-s-anthem-smells-like-teen-pressure.html"

str5.startswith('http')
# >>> 출력:
# True

str5.endswith('.html')
# >>> 출력:
# True

str5.endswith('.com')
# >>> 출력:
# False

# # ord(), chr()

# ord() : 문자의 코드값
# chr() : 코드의 문자값

ord('a')
# >>> 출력:
# 97

ord('b')
# >>> 출력:
# 98

# 알파벳 개수
ord('z') - ord('a') + 1
# >>> 출력:
# 26

ord('A')
# >>> 출력:
# 65

# 한글 개수
ord('힣') - ord('가') + 1
# >>> 출력:
# 11172

# # 등장 개수 세기
# 계수하기

# 등장하는 알파벳의 개수 -> dict 결과 만들기
# Dict Comprehension 사용
# 대소문자 구분 하지 않기, 1개 이상 등장하는 알파벳만
# hint : 알파벳 리스트 , lower, upper, count

word = "Alice in wonderland"

# {'a': 2,
#  'c': 1,
#  'd': 2,
#  'e': 2,
#  'i': 2,
#  'l': 2,
#  'n': 3,
#  'o': 1,
#  'r': 1,
#  'w': 1}

data = word.lower()

result = {}

for ch in data:  # 한 글자씩 순환
  if 'a' <= ch <= 'z':  #알파벳만
    if not result.get(ch):  # 첫 등장이면
      result[ch] = 1
    else:  # 이미 이전에 한번이라도 등장했다면
      result[ch] += 1

  # print(ch, result) # 확인용

print(result)
# >>> 출력:
# {'a': 2, 'l': 2, 'i': 2, 'c': 1, 'e': 2, 'n': 3, 'w': 1, 'o': 1, 'd': 2, 'r': 1}

result = {}

for ch in word.lower():
  if ch.isalpha():
    result[ch] = result.get(ch, 0) + 1

print(result)
# >>> 출력:
# {'a': 2, 'l': 2, 'i': 2, 'c': 1, 'e': 2, 'n': 3, 'w': 1, 'o': 1, 'd': 2, 'r': 1}

w = word.lower()
result = {ch:w.count(ch) for ch in w if ch.isalpha()}
result
# >>> 출력:
# {'a': 2,
#  'l': 2,
#  'i': 2,
#  'c': 1,
#  'e': 2,
#  'n': 3,
#  'w': 1,
#  'o': 1,
#  'd': 2,
#  'r': 1}

data_list = list(result.items())

data_list.sort(key=lambda entry: entry[0])

data_list
# >>> 출력:
# [('a', 2),
#  ('c', 1),
#  ('d', 2),
#  ('e', 2),
#  ('i', 2),
#  ('l', 2),
#  ('n', 3),
#  ('o', 1),
#  ('r', 1),
#  ('w', 1)]

# ---
# # [도전과제]

# *******************************
# 가장 긴 단어 찾기
#

data = [
    "I am a Student",        # ->  Student
    "That elephant is big",  # -> elephant
    "She loves cat very much", # -> loves
]

def longestWord(sentence):   # "I am a Student"
    max_word = "";  # 가장 길이가 긴 단어
    for word in sentence.strip().split():   # "I", "am", "a", "Student"
      if len(word) > len(max_word):
        max_word = word

    return max_word;



print(list(map(longestWord, data)))

# ↓ 결과 ['Student', 'elephant', 'loves']
# >>> 출력:
# ['Student', 'elephant', 'loves']

# *******************************
# 문장에서 각 단어 첫글자만 대문자 만들기
#

data = [
    "i am a PROGRAMMER",     # -> I Am A Programmer
    "THAT ELEPHANT IS BIG",  # -> That Elephant Is Big
]

def letterCapitalize(sentence):
    result = []

    for word in sentence.strip().lower().split():   # "i" "am" "a" "programmer"
      result.append(word[0].upper() + word[1:])  #

    return " ".join(result)

print(list(map(letterCapitalize, data)))

# ↓ 결과 ['I Am A Programmer', 'That Elephant Is Big']
# >>> 출력:
# ['I Am A Programmer', 'That Elephant Is Big']
