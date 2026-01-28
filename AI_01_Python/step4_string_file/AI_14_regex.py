# # 정규표현식

# 정규표현식 regular expression

# 문자열 검색, 치환  등의 동작에 있어서
# 단순한 '문자열 비교' 를 하는 것이 아니라
# 특정 '패턴'과 비교하고자 할때 이를 단 몇줄의 코드로 구현 가능!
# 주어진 문자열에서 패턴을 찾아내는 것을 '패턴 매칭(pattern matching)' 이라 함

# 사용자가 입력한 문자열 패턴 유효성 체크 등에 많이 사용
#  		ex) 주민등록번호, URL, email, 비밀번호,
#  			날짜포맷(yyyy-mm-dd)
#  			전화번호(010-xxxx-xxxx) ...

# 장점: 코딩량 저감, 거의 대부분의 언어에서 공용으로 사용.
# 단점: 처음에 배우기 어렵고, 코드 가독성 떨어뜨림.

# 정규표현식 연습 사이트 추천
# : https://regexr.com/    (정규식 , 문자열 매칭 연습)
# : https://regexone.com/  ( step by step 으로 연습 하기 좋음)
# : https://regexper.com/  (특징: 시각화, 정규식을 이미지로 다운가능)
# : https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html  (오라클 공식)

#  정규표현식 설명
#  ^ 문자열 시작
#  $ 문자열 종료
#  . 임의의 문자 [단 ‘'는 넣을 수 없습니다.]
#  * 앞 문자가 0개 이상의 개수가 존재할 수 있습니다.
#  + 앞 문자가 1개 이상의 개수가 존재할 수 있습니다.
#  ? 앞 문자가 없거나 하나 있을 수 있습니다.
#  [] 문자의 집합이나 범위를 표현합니다. -기호를 통해 범위를 나타낼 수 있습니다. ^가 존재하면 not을 나타냅니다.
#  {} 횟수 또는 범위를 나타냅니다.
#  () 괄호안의 문자를 하나의 문자로 인식합니다. (그룹)
#  | 패턴을 OR 연산을 수행할 때 사용합니다.
#  \s 공백 문자
#  \S 공백 문자가 아닌 나머지 문자
#  \w 알파벳이나 숫자
#  \W 알파벳이나 숫자를 제외한 문자
#  \d [0-9] 숫자
#  \D 숫자를 제외한 모든 문자

import re  # 정규표현식 기본 모듈

# 패턴 문자열
pat = r'My....'  # My1234, Myabcd, Myaaaaaaaaaa, MMMMMMyyyyyyyyy (OK)  My10  (X)

# match(pattern, string) : 전체매칭 => 리턴값은 Match 객체

m = re.match(pat, 'My1234')
m
# >>> 출력:
# <re.Match object; span=(0, 6), match='My1234'>

m = re.match(pat, 'Myabcd')
m
# >>> 출력:
# <re.Match object; span=(0, 6), match='Myabcd'>

# Match 객체의 메소드들

# 어떤 문자열이 매치되었는가?
# 매치된 문자열의 인덱스는 어디서부터 어디까지인가?

# group()	매치된 문자열을 리턴한다.
# start()	매치된 문자열의 시작 위치를 리턴한다.
# end()	매치된 문자열의 끝 위치를 리턴한다.
# span()	매치된 문자열의 (시작, 끝) 에 해당되는 튜플을 리턴한다.

m.start()
# >>> 출력:
# 0

m.end()
# >>> 출력:
# 6

m.group()
# >>> 출력:
# 'Myabcd'

m.span()
# >>> 출력:
# (0, 6)

m = re.findall(pat, 'asdfMy 23_asdkfMy  asdfasdf Mysdafk')
m
# >>> 출력:
# ['My 23_', 'My  as', 'Mysdaf']

re.findall('[a-z]', 'a100b300def500')
# >>> 출력:
# ['a', 'b', 'd', 'e', 'f']

re.findall('[a-z]+', 'a100b300def500')
# >>> 출력:
# ['a', 'b', 'def']

re.findall(r'\w', 'life is too short')
# >>> 출력:
# ['l', 'i', 'f', 'e', 'i', 's', 't', 'o', 'o', 's', 'h', 'o', 'r', 't']

re.findall(r'\w+', 'life is too short')
# >>> 출력:
# ['life', 'is', 'too', 'short']

re.findall(r'\w{2}', 'life is too short')
# >>> 출력:
# ['li', 'fe', 'is', 'to', 'sh', 'or']

a = "123456-1234567"

re.sub(r'[0-9]', "*", a)
# >>> 출력:
# '******-*******'

re.sub(r'[0-9]{7}', "*******", a)
# >>> 출력:
# '123456-*******'

re.sub(r'\d{7}', "*******", a)
# >>> 출력:
# '123456-*******'

b = "정가 14,500원"

# 도전
# 정규표현식 사용하여 int값 14500 추출하기
# re.sub() 등을 활용해보세요.

int(re.sub(r'\D', '', b))
# >>> 출력:
# 14500

int(''.join(re.findall('[0-9]',b)))
# >>> 출력:
# 14500
