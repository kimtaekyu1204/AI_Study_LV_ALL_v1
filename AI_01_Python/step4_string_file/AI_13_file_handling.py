# # 파일 다루기

# ## 파일 생성

# 파일을 다루기 위해선 open(), close() 함수로 감싸여진다

# 파일객체 = open(파일 이름, 파일 열기 모드)
#   ... 파일객체사용 (읽기 or 쓰기)
#   ... 파일객체사용 (읽기 or 쓰기)
#   ... 파일객체사용 (읽기 or 쓰기)
# 파일객체.close()

# 파일 열기 모드
# r  읽기모드 - 파일을 읽기만 할 때 사용
# w  쓰기모드 - 파일에 내용을 쓸 때 사용.  해당 파일이 없으면 새로 생성.   해당 파일이 있었으면 삭제하고 새로 생성 ★
# a  추가모드 - 파일의 마지막에 새로운 내용을 추가 시킬 때 사용.  해당 파일이 없으면 새로 생성

f = open("새파일.txt", "w")

f.close()

# ## 파일에 쓰기

f = open("새파일.txt", "w")

for i in range(1, 11):
  f.write(f'Line {i}\n')

f.close()

!cat 새파일.txt
# >>> 출력:
# Line 1
# Line 2
# Line 3
# Line 4
# Line 5
# Line 6
# Line 7
# Line 8
# Line 9
# Line 10

# ## 읽기

f = open("새파일.txt", "r") # 읽기모드

line = f.readline()  # 한 줄 읽기
print(line)

line = f.readline()  # 한 줄 읽기
print(line)

f.close()
# >>> 출력:
# Line 1
# 
# Line 2

f = open("새파일.txt", "r") # 읽기모드

while True:
  line = f.readline()  # 한 줄 읽기
  if not line: break # readline() 은 더 이상 읽을 라인이 없는 경우 None 리턴
  print(line, end='')

f.close()
# >>> 출력:
# Line 1
# Line 2
# Line 3
# Line 4
# Line 5
# Line 6
# Line 7
# Line 8
# Line 9
# Line 10



f = open("새파일.txt", "r") # 읽기모드

data = f.read()  # 파일 전체 읽기
print(data)

f.close()
# >>> 출력:
# Line 1
# Line 2
# Line 3
# Line 4
# Line 5
# Line 6
# Line 7
# Line 8
# Line 9
# Line 10

# ## 추가(append)

f = open("새파일.txt", "a")  # append mode

for i in range(11, 21):
  f.write(f'{i} Line appended')

f.close()

!cat 새파일.txt
# >>> 출력:
# Line 1
# Line 2
# Line 3
# Line 4
# Line 5
# Line 6
# Line 7
# Line 8
# Line 9
# Line 10
# 11 Line appended12 Line appended13 Line appended14 Line appended15 Line appended16 Line appended17 Line appended18 Line appended19 Line appended20 Line appended

# # with 구문

# 매번 close() 를 해주는게 불편하고, 까먹기도 쉽다
# with 구문을 사용하면 자동적으로 close 해준다

with open("새파일.txt", "r") as f: # 블럭 형태로 작성해야 한다.  블럭이 끝나면 자동적으로 f 를 close() 해준다
  data = f.read()
  print(data)
# >>> 출력:
# Line 1
# Line 2
# Line 3
# Line 4
# Line 5
# Line 6
# Line 7
# Line 8
# Line 9
# Line 10
# 11 Line appended12 Line appended13 Line appended14 Line appended15 Line appended16 Line appended17 Line appended18 Line appended19 Line appended20 Line appended

base_path = r'/content/drive/MyDrive/KoreaIT (코리아it)/250107 자연어처리/[AI자연어]'

base_path
# >>> 출력:
# '/content/drive/MyDrive/KoreaIT (코리아it)/250107 자연어처리/[AI자연어]'

import os

target_dir = os.path.join(base_path, 'out')
target_dir
# >>> 출력:
# '/content/drive/MyDrive/KoreaIT (코리아it)/250107 자연어처리/[AI자연어]/out'

os.path.join(base_path, 'aaa', 'bbb', 'c1/c2/c3', 'ddd')
# >>> 출력:
# '/content/drive/MyDrive/KoreaIT (코리아it)/250107 자연어처리/[AI자연어]/aaa/bbb/c1/c2/c3/ddd'

os.path.exists(target_dir)
# >>> 출력:
# False

if not os.path.exists(target_dir):
  os.mkdir(target_dir)  # 디렉토리 생성
  print('디렉토리 생성')
else:
  print('디렉토리가 이미 존재')
# >>> 출력:
# 디렉토리가 이미 존재

target_file = os.path.join(target_dir, 'test.txt')
target_file
# >>> 출력:
# '/content/drive/MyDrive/KoreaIT (코리아it)/250107 자연어처리/[AI자연어]/out/test.txt'

with open(target_file, 'w') as f:
  for i in range(1, 11):
    f.write(f'{i} 번째 줄입니다\n')

!cat "/content/drive/MyDrive/KoreaIT (코리아it)/250107 자연어처리/[AI자연어]/out/test.txt"
# >>> 출력:
# 1 번째 줄입니다
# 2 번째 줄입니다
# 3 번째 줄입니다
# 4 번째 줄입니다
# 5 번째 줄입니다
# 6 번째 줄입니다
# 7 번째 줄입니다
# 8 번째 줄입니다
# 9 번째 줄입니다
# 10 번째 줄입니다

# # encoding

# Windows 는 시스템 인코딩이 기본 2byte (cp949, ANSI, euc-kr)이다
# 특별히 encoding 옵션을  지정하지 않으면 파일은 2byte 로 인코딩되어 있다
# ※ 반면 MAC, Linux 등에선 기본인코딩이 3byte (utf8) 이다

with open(os.path.join(target_dir, 'test2.txt'), 'w', encoding='cp949') as f:
  for i in range(1, 11):
    f.write(f'{i} 번째 줄입니다\n')

with open(os.path.join(target_dir, 'test2.txt'), 'r', errors='ignore') as f:
  print(f.read())
# >>> 출력:
# 1 ° Դϴ
# 2 ° Դϴ
# 3 ° Դϴ
# 4 ° Դϴ
# 5 ° Դϴ
# 6 ° Դϴ
# 7 ° Դϴ
# 8 ° Դϴ
# 9 ° Դϴ
# 10 ° Դϴ
