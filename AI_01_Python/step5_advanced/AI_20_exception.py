# # 예외 (Exception) 혹은 (Error)

#  아래와 같은 경우를 생각해보자.

def func_sum(data):
    result = 0
    for i in data:
        result = result + i
    return result#

func_sum([10, 20, 30])
# >>> 출력:
# 60

func_sum({"name": "John", "age": 20})
# >>> 출력:
# TypeError: unsupported operand type(s) for +: 'int' and 'str'

open("없는 파일", "r")
# >>> 출력:
# FileNotFoundError: [Errno 2] No such file or directory: '없는 파일'

a = 10
b = 0
print(a + b)
print(a / b)
# >>> 출력:
# 10
# ZeroDivisionError: division by zero

# # 예외(오류) 처리: try ~ except

# 물론 사전에 if 등의 조건식을 활용해서 오류 를 사전에 예방은 가능하나

a = 10
b = 0
if b != 0:
    print(a / b)
else:
    print('0 으로 나눌수 없습니다')

# 수~~~많은 예외상황을 매번 if 조건문으로 처리하면
# 코드가 매우 난잡해진다. --> 유지보수 힘들어짐.

# 그래서, 예외처리 관련하여
# '코드수행블럭'  와 '예외처리블럭' 구분하여 예외를 다루는게 바람직.

# 파이썬 에서는
# try ~ except 문으로 오류를 다룹니다

# 구문]
# try:
#     코드수행블럭
#     ...
# except [발생 오류[as 오류 메시지 변수]]:
#     예외처리블럭
#     ...

#  try 블록 수행 중 오류가 발생하면 except 블록이 수행된다.
# 하지만 try블록에서 오류가 발생하지 않는다면 except 블록은 수행되지 않는다.

# ## 1.try~except

num = 0

try:
  result = 10 / num  # try 블럭 안에서 예외 발생하면 except: 로 넘어감.
  print('1 결과', result)
except:
  print('2.오류발생')

print('3 프로그램 종료')
# >>> 출력:
# 2.오류발생
# 3 프로그램 종료

# ## 2.try: ~ except 오류:

num = 0

try:
  result = 10 / num
  print('1 결과', result)
# except ZeroDivisionError:  # except 에는 처리하고자 하는 에러 명시 가능
except TypeError:  # 처리(handling) 되지 않은 Error -> 프로그램 종료.
  print('2.오류발생')

print('3 프로그램 종료')
# >>> 출력:
# ZeroDivisionError: division by zero

num = 0

try:
  result = 10 / num
  print('1 결과', result)
except TypeError:
  print('2-1. TypeError 오류발생')
except ZeroDivisionError:  # except: 는 여러개 명시 가능!
  print('2-2. ZeroDivisionError 오류발생')

print('3 프로그램 종료')
# >>> 출력:
# 2-2. ZeroDivisionError 오류발생
# 3 프로그램 종료

# ## 3.try~except 에러 as 변수:

num = 0

try:
  result = 10 / num
  print('1 결과', result)
except TypeError:
  print('2-1. TypeError 오류발생')
except ZeroDivisionError as e:
  print('2-2. ZeroDivisionError 오류발생')
  print(e.args)
  print(e)

print('3 프로그램 종료')
# >>> 출력:
# 2-2. ZeroDivisionError 오류발생
# ('division by zero',)
# division by zero
# 3 프로그램 종료

# ## 4. finally
# 예외가 발생했든 아니든 반드시 수행해야 하는 코드들은 finally  에 작성

f = open('foo.txt', 'w')
a = 10
a = a + bbbbbbb
print('1. 파일 처리')
f.close()  # 꼭 반드시 실행되어야 하는 코드인데 실행이 안된다!!!?
# >>> 출력:
# NameError: name 'bbbbbbb' is not defined

try:
  f = open('foo.txt', 'w')
  a = 10
  a = a + bbbbbbb
  print('1. 파일 처리')

# except:
#   print('2. 오류발생')

finally:
  print('3. finally')
  f.close()

print('4. 프로그램 종료')
# >>> 출력:
# 3. finally
# NameError: name 'bbbbbbb' is not defined

try:
  f = open('foo.txt', 'w')
  a = 10
  a = a + bbbbbbb
  print('1. 파일 처리')

except NameError as e:
  print('2. 오류발생', e.args)

finally:
  print('3. finally')
  f.close()

print('4. 프로그램 종료')
# >>> 출력:
# 2. 오류발생 ("name 'bbbbbbb' is not defined",)
# 3. finally
# 4. 프로그램 종료

# ## except 오류, 오류...

try:
  print('1 try: 시작')
  a = ['alpha', 'beta']
  print(a[0])
  b = 4 / 0
  print('2 try: 종료')
except IndexError as e:
  print("3-1 IndexError 에러 발생", e.args)
except ZeroDivisionError as e:
  print('3-2 ZeroDivisionError 에러 발생', e.args)

print('4 프로그램 종료')
# >>> 출력:
# 1 try: 시작
# alpha
# 3-2 ZeroDivisionError 에러 발생 ('division by zero',)
# 4 프로그램 종료

try:
  print('1 try: 시작')
  a = ['alpha', 'beta']
  print(a[100])
  b = 4 / 0
  print('2 try: 종료')
except (IndexError, ZeroDivisionError) as e:
  print("3 Error 에러 발생", e.args, e)

print('4 프로그램 종료')
# >>> 출력:
# 1 try: 시작
# 3 Error 에러 발생 ('list index out of range',) list index out of range
# 4 프로그램 종료

# # 예외발생: raise
# 강제로 오류 발생시키기

#  Bird라는 클래스를 상속받는 자식 클래스는
# 반드시 fly라는 함수를 구현하도록 만들고 싶은 경우(강제로 그렇게 하고 싶은 경우)가 있을 수 있다. 다음 예를 보자
# '구현(implement)' : 부모클래스의 메소드를 오버라이딩해서 재정의하는것.

class Bird:

  def fly(self):
    print('fly() 호출')
    raise NotImplementedError
    print('fly() 종료')

bird1 = Bird()
bird1.fly()
# >>> 출력:
# fly() 호출
# NotImplementedError: 

class Eagle(Bird):
  pass

eagle = Eagle()
eagle.fly()
# >>> 출력:
# fly() 호출
# NotImplementedError: 

# Bird 구현
class Eagle(Bird):
  # 오버라이딩
  def fly(self):
    print('Eagle.fly() 호출. 독수리는 높이 빠르게 난다.')

bird2 = Eagle()
bird2.fly()
# >>> 출력:
# Eagle.fly() 호출. 독수리는 높이 빠르게 난다.

# Bird 구현
class Penguin(Bird):
  # 오버라이딩
  def fly(self):
    print('Penquin.fly() 호출. 펭귄은 날지 못합니다.')

bird3 = Penguin()
bird3.fly()
# >>> 출력:
# Penquin.fly() 호출. 펭귄은 날지 못합니다.
