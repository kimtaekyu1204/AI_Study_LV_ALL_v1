# # 클래스 (class)

# 지금까지의 프로그래밍은 '함수 중심'

# ex) 아래와 같이 goto() 함수 안에 필요한 인자를 넘겨준다

# goto("이길자", "집(논현동 188)")
# goto("김해성", "집(서초동 333)")
# 함수 중심이라는 것은 곧 '동작' 중심이라는 거다.
# 그러나 이는 '실세계'를 표현하는데는 한계가 있다.


# 사람이 인지하는 세상은 '객체 중심' 이기 때문이다

# '객체 중심' 프로그래밍은 다음과 같다.

# 예시]
# Person 클래스(class) 안에 다음의 '틀' 을 정의
#     - 이름 : 이름 정보 담는 '속성'
#     - 집주소 : 주소 정보 담는 '속성'
#     - goHome() : 집으로 가는 '동작' 수행

#     '속성' 은 변수로 선언
#     '동작' 은 함수로 선언 -> '메소드(method)' 라 한다

# 사람객체(object)들 생성
# lee = Person("이길자", "논현동 188") <- 이름, 집주소 등에 대한 정보가 객체에 담김
# kim = Person("김해성", "서초동 333") <- 이름, 집수소 등에 대한 정보가 객체에 담김

# lee.goHome()
# kim.goHome()

# 프로그래밍의 중심이 '함수' 중심이 아니라, 각각의 객체 중심( lee, kim )..


# 객체는 클래스로 정의를 하고
# 클래스를 통하여 객체를 생성한다   생성된 객체를 인스턴스(instance) 라 한다

# 흔히 클래스(Class ) 를 '붕어빵 틀'이라 하고
# 객체(object) 를 Class 로 찍어낸 '붕어빵'에 비유하곤 한다


# 파이썬에선 다음 순서대로 클래스를 사용한 객체중심 프로그래밍을 한다
#     ① 클래스 정의, class 키워드 사용
#     ② 객체 생성 (생성자 사용),생성된 객체를 '인스턴스(instance)' 라 한다
#     ③ 객체 사용 (객체의 변수, 메소드)

# 클래스 이름은
# 변수나 함수 이름 정의하는 규칙과 거의 동일하나
# 관례적으로 첫글자는 대문자로 한다

# 클래스 '정의' - 붕어빵 틀
class Cookie:
  pass

# 객체 생성 - 붕어빵 찍어냄
a = Cookie() # Cookie 객체 생성. 인스턴스(instance) 라 함.
b = Cookie()
c = Cookie()

a
# >>> 출력:
# <__main__.Cookie at 0x7ef2cf38add0>

type(a)  # 클래스를 정의한다 -> type 을 정의하는 것이다.
# >>> 출력:
# __main__.Cookie

a == b  # False.  타입만 같을 뿐.  다른 인스턴스다!
# >>> 출력:
# False

b
# >>> 출력:
# <__main__.Cookie at 0x7ef2cf38b7d0>

# 클래스를 통해 만들어진 객체를 인스턴스(instance)라고도 한다
# 객체와 인스턴스는 다소 혼용되어 표현되기도 한다.
# "a 는 Cookie 의 객체" 다
# "a 는 Cookie 의 인스턴스" 다

#----------------------------------------------------------------

# 객체의 '속성' --> 객체변수로 정의
# 객체의 '동작' --> 객체함수(메소드)로 정의

# 클래스 가 가져야 하는 변수를 '객체변수(instance variable)',
# 혹은 속성값 (attribute value) 혹은 멤버변수(member variable) 혹은 필드(field) 라고도 함.
# 어떠한 메소드에서도 self. 키워드를 사용하여 정의 가능하고
# 인스턴스를 통해 생성도 가능.

# 객체변수 사용구문   . 사용
#   객체.객체변수

# 생성자 (Constructor)  __init__()
# 객체가 생성될때 '자동'으로 호출되는 특별한 메소드
# 주로 객체변수를 정의하고 초기화 하는 역할 수행

# 클래스 안에서 정의된 함수를 메소드(method) 라 함
# 메소드 호출 구문    . 사용
#    객체.메소드()
# 메소드가 호출될때는 해당 '객체' 에 대해서 동작하게 됨

# 모든 메소드는 기본적으로 첫번째 매개변수가 self 이다.

# # 객체변수, 생성자

# 사각형 이란 객체에 대해 설계해보자.
# 사각형 클래스를 정의해보자.

# 사각형의 속성은? -->  가로(width),  세로(height)
# 사각형의 동작은? --> ??

# Rectangle 객체 정의 (클래스 정의,  붕어빵틀 제작)

class Rectangle:
  # 생성자 (constructor)
  def __init__(self):
    print("Rectangle 객체 생성")
    # 객체변수들 (속성)
    self.width = 0
    self.height = 0

r1 = Rectangle()
r2 = Rectangle()
# >>> 출력:
# Rectangle 객체 생성
# Rectangle 객체 생성

r1.width
# >>> 출력:
# 0

r1.height
# >>> 출력:
# 0

r2.width, r2.height
# >>> 출력:
# (0, 0)

r1.width, r1.height = 100, 200
r2.width, r2.height = 50, 30

print(r1.width, r1.height)
print(r2.width, r2.height)
# >>> 출력:
# 100 200
# 50 30

# width, height 객체변수는 각 인스턴스 별로 '각각' 가지고 있다  (그래서, 이를 인스턴스 변수라고도 한다)

# r1.area
# AttributeError: 'Rectangle' object has no attribute 'area'

r1.area = 19

print(r1.area)
# >>> 출력:
# 19

# r2.area

# # 메소드

#------------------------------------------
# 사각형의 동작? ----> 객체 메소드(함수)로 정의
#   - 사각형의 넓이를 구하기
#   - 사각형의 둘레를 구하기

class Rectangle:
  def __init__(self):
    print("Rectangle 객체 생성")
    # 객체변수들 (속성)
    self.width = 0
    self.height = 0

  # 사각형의 넓이 구하기
  def getArea(self):
    area = self.width * self.height
    return area

  # 사각혁의 둘레 구하기
  def getPerimeter(self):
    return (self.width + self.height) * 2

r1 = Rectangle()
r2 = Rectangle()

r1.height, r1.width = 10, 20
# >>> 출력:
# Rectangle 객체 생성
# Rectangle 객체 생성

r1.getArea()
# >>> 출력:
# 200

r1.getPerimeter()
# >>> 출력:
# 60

r2.getArea()
# >>> 출력:
# 0

# ## 첫번째 매개변수 self

r1.getArea()
# >>> 출력:
# 200

Rectangle.getArea(r1)  # r1 이 self 매개변수로 전달되는 것이다!
# >>> 출력:
# 200

"Hello".upper()
# >>> 출력:
# 'HELLO'

str.upper("Hello")
# >>> 출력:
# 'HELLO'

# # 생성자 매개변수

# r3 = Rectangle(100, 200)  # => Rectangle.__init__(r3, 100, 200)

class Rectangle:
  def __init__(self, width=1, height=1):
    print(f"Rectangle({width}, {height})객체 생성")
    # 객체변수들 (속성)
    self.width = width
    self.height = height

  # 사각형의 넓이 구하기
  def getArea(self):
    area = self.width * self.height
    return area

  # 사각혁의 둘레 구하기
  def getPerimeter(self):
    return (self.width + self.height) * 2

  # 매개변수 있는 메소드
  def setSize(self, w, h):
    self.width = w
    self.height = h

r1 = Rectangle()
# >>> 출력:
# Rectangle(1, 1)객체 생성

r1 = Rectangle(30, 20)
# >>> 출력:
# Rectangle(30, 20)객체 생성

r1.getArea()
# >>> 출력:
# 600

r1.setSize(100, 500)

r1.getArea()
# >>> 출력:
# 50000

# # is 연산자

# r1 이 Rectangle 타입의 객체인지 확인
type(r1) is Rectangle
# >>> 출력:
# True

# # id()
# - 인스턴스 고유 id 값 (aka. 주소값)

id(r1)
# >>> 출력:
# 139581317673808

r2 = Rectangle()
id(r2)
# >>> 출력:
# Rectangle(1, 1)객체 생성
# 139581317764944

# # dict__ 속성

# 인스턴스의 __dict__ 속성

r1.__dict__
# >>> 출력:
# {'width': 100, 'height': 500}

r2.__dict__
# >>> 출력:
# {'width': 1, 'height': 1}

r1.color = 'red'

r1.__dict__
# >>> 출력:
# {'width': 100, 'height': 500, 'color': 'red'}

# 클래스의 __dict__ 속성

Rectangle.__dict__
# >>> 출력:
# mappingproxy({'__module__': '__main__',
#               '__init__': <function __main__.Rectangle.__init__(self, width=1, height=1)>,
#               'getArea': <function __main__.Rectangle.getArea(self)>,
#               'getPerimeter': <function __main__.Rectangle.getPerimeter(self)>,
#               'setSize': <function __main__.Rectangle.setSize(self, w, h)>,
#               '__dict__': <attribute '__dict__' of 'Rectangle' objects>,
#               '__weakref__': <attribute '__weakref__' of 'Rectangle' objects>,
#               '__doc__': None})

# # 클래스 메소드, 인스턴스 메소드
# class method, instance method

class Foo:

  # 클래스 메소드 (class method)
  # self 가 안붙은 메소드, '클래스 이름' 으로 사용
  def func1():
    print('function1')

  # 인스턴스 메소드 (instance method)
  # self 가 붙은 메소드 ,  '인스턴스 생성' 하여 사용
  def func2(self):
    print('function2')

f1 = Foo()  # Foo 타입 인스턴스 생성
# f1.func1()  # 에러
f1.func2()
# >>> 출력:
# function2

Foo.func1()  # 클래스 메소드 호출
# >>> 출력:
# function1

# # 클래스 변수, 인스턴스 변수
# class variable, instance variable

# self.변수이름  : <-- 인스턴스 변수
#       인스턴스 변수는 '인스턴스 마다' 생성        --> 인스턴스 네임스페이스에 생성
# 클래스 내부에서 선언된변수 <-- 클래스 변수  (self 가 안 붙음)
#        사용하려면 클래스이름.변수이름  으로 사용해야 함
#        클래스 변수는 '클래스에 딱 하나' 생성  --> 클래스 네임스페이스에 생성

class Account:

  num_accounts = 0  # 클래스 변수

  def __init__(self, name):
    print(f'Account({name}) 생성')
    self.name = name  # 인스턴스 변수.
    Account.num_accounts += 1

  # 소멸자 (destructor) : 인스턴스가 소멸될때 자동으로 호출
  def __del__(self):
    print(f'Account({self.name}) 소멸')
    Account.num_accounts -= 1

Account.num_accounts
# >>> 출력:
# 0

acc1 = Account('회사')
# >>> 출력:
# Account(회사) 생성

Account.num_accounts
# >>> 출력:
# 1

acc2 = Account('개인')
# >>> 출력:
# Account(개인) 생성

Account.num_accounts
# >>> 출력:
# 2

acc1.name, acc2.name
# >>> 출력:
# ('회사', '개인')

acc1.num_accounts, acc2.num_accounts  # 인스턴스로도 클래스변수 접근은 가능하나 비추!
# >>> 출력:
# (2, 2)

id(acc1), id(acc2)
# >>> 출력:
# (139581317837520, 139581317839248)

acc1.__dict__
# >>> 출력:
# {'name': '회사'}

acc2.__dict__
# >>> 출력:
# {'name': '개인'}

Account.__dict__
# >>> 출력:
# mappingproxy({'__module__': '__main__',
#               'num_accounts': 2,
#               '__init__': <function __main__.Account.__init__(self, name)>,
#               '__del__': <function __main__.Account.__del__(self)>,
#               '__dict__': <attribute '__dict__' of 'Account' objects>,
#               '__weakref__': <attribute '__weakref__' of 'Account' objects>,
#               '__doc__': None})

del(acc2)
# >>> 출력:
# Account(개인) 소멸

Account.num_accounts
# >>> 출력:
# 1

# # 연습] PartTimer 클래스

# '매 직원(PartTimer)'에 공통적으로 적용되는 자료
# - 시급
# - 전체 직원수

# 각 직원별 객체 생성시 직원별로 '별칭'과 '근무지' '급여총액' 초기화 (속성)
#   '근무지' 생략시 '113동' 으로 지정
#   직원별로 '급여총액'  0으로 초기화

# 직원의 급여 계산하기(동작)
#    '몇시간 근무',  '+상여금'  에 따른 직원급여 계산
#   '상여금' 은 지정안하면 0 으로 처리


# 예]
# park = PartTimer('라이언')   // park 은 ‘라이언’ 이라는 닉네임의 직원으로 등록

# park 이 4시간 일한 급여 총액은?  → 34400
# park 이 2시간 일한 급여 총액은? → 17200
# park 이 2시간 일한 급여 + 상여급 2000 총액은? → 19200

class PartTimer:
  # 클래스변수
  hour_rate = 10030  # 시급
  total_part_timers = 0 # 전체 직원수

  # 생성자 (nicknam, workplace)
  #   인스턴스 변수들 선언 (nickname, workplace, total_wage)
  def __init__(self, nickname, workplace = '113동'):
      PartTimer.total_part_timers += 1   # 클래스 인스턴스 는 '클래스이름.이름' 의 형식으로 사용.
      self.nickname = nickname
      self.workplace = workplace
      self.total_wage = 0

  # 급여계산
  def calculate_wage(self, hours, bonus=0):
      self.total_wage = PartTimer.hour_rate * hours + bonus
      return self.total_wage

park = PartTimer('라이언')
lee = PartTimer('네오', '127-1동')

park.total_wage

# 0
# >>> 출력:
# 0

lee.total_wage

# 0
# >>> 출력:
# 0

park.calculate_wage(4)

# 시급 x 4
# >>> 출력:
# 40120

PartTimer.total_part_timers
# 2
# >>> 출력:
# 2

lee.calculate_wage(3, 15000)

# 시급 x 3 + 15000
# >>> 출력:
# 45090

# # 클래스의 상속 (inheritance)

# 기존의 만들어진 클래스를 상속받아 새로운 클래스 정의 가능

# 상속받아 만들어진 클래스는 기존의 클래스의 메소드, 객체변수 를 그대로 가지고 있다.
# 상속받은뒤, 새로운 객체변수, 메소드 추가 할수 있다.
# 상속받은뒤, 상속받은 메소드 재정의 가능 (오버라이딩)

# 상속의 장점:
#    기존의 코드를 다시 재작성 할 필요 없이. 새로이 변경 추가 되는 코드에만 집중할수 있기 때문에 생산성 향상

# 기존의 클래스 상속하여 새로운 클래스 정의하는 구문
#    class 새클래스명(기존의 클래스명)
#    class 새클래스명(기존클래스1, 기존클래스2, ...)  <-- 다중상속 허용

# 기존의 클래스를 '부모클래스(parent class)' 라고 하고  (혹은 super class,  base class ...)
# 상속받은 클래스를 '자식클래스(child class) 라고 한다  (혹은 sub class, derived class ...)

# ## 상속은 왜 필요한가?

class BasicTV:
  def __init__(self):
    self.power = False  # 전원
    self.channel = 0    # 채널
    self.volume = 0     # 볼륨

  def display_info(self):
    print("전원:", self.power)
    print("채널:", self.channel)
    print("볼륨:", self.volume)

tv1 = BasicTV()

tv1.power = True
tv1.channel = 10
tv1.volume = 5

tv1.display_info()
# >>> 출력:
# 전원: True
# 채널: 10
# 볼륨: 5

class SmartTV:
  def __init__(self):
    self.power = False  # 전원
    self.channel = 0    # 채널
    self.volume = 0     # 볼륨
    self.IP = "192.168.0.1"  # IP  (추가된 속성)

  def display_info(self):
    print("전원:", self.power)
    print("채널:", self.channel)
    print("볼륨:", self.volume)
    print("IP:", self.IP)

tv2 = SmartTV()

tv2.power = True
tv2.channel = 10
tv2.volume = 5
tv2.IP = "192.168.0.2"

tv2.display_info()
# >>> 출력:
# 전원: True
# 채널: 10
# 볼륨: 5
# IP: 192.168.0.2

# 상속을 사용하여 SmartTV 정의

# BasicTV
#   └─ SmartTV

class SmartTV(BasicTV):

  def __init__(self):
    super().__init__()   # 부모클래스의 생성자 호출
    self.IP = "192.168.0.1"   # 자식클래스에서 추가된 속성

  def display_info(self):
    super().display_info()  # 부모클래스의 메소드 호출
    print("IP:", self.IP)

tv3 = SmartTV()

tv3.__dict__
# >>> 출력:
# {'power': False, 'channel': 0, 'volume': 0, 'IP': '192.168.0.1'}

tv3.display_info()
# >>> 출력:
# 전원: False
# 채널: 0
# 볼륨: 0
# IP: 192.168.0.1

# # 연습] Circle, Sphere

import math

# 2차원 Circle 정의
class Circle:
  def __init__(self, radius=0):
    self.radius = radius

  def getArea(self):  # 면적
    return math.pi * self.radius ** 2

  def getPerimeter(self): # 둘레구하기
    return 2 * math.pi * self.radius

c1 = Circle(10)
print(c1.getArea())
print(c1.getPerimeter())
# >>> 출력:
# 314.1592653589793
# 62.83185307179586

# Circle 을 상속받아 구(Sphere) 객체 생성

class Sphere(Circle):
  # Sphere 클래스에서 추가되는 메소드
  def getVolume(self):
      return (4 / 3) * math.pi * self.radius ** 3

  def getArea(self):  # 메소드 오버라이딩
      return 4 * math.pi * self.radius ** 2

s1 = Sphere(10)
print('s1의 면적', s1.getArea())     #  4x PI x r ** 2
print('s1의 부피', s1.getVolume())   #   4/3 x PI x r ** 3
# s1의 면적 1256.6370614359173
# s1의 부피 4188.790204786391
# >>> 출력:
# s1의 면적 1256.6370614359173
# s1의 부피 4188.790204786391

c1.__dict__
# >>> 출력:
# {'radius': 10}

s1.__dict__
# >>> 출력:
# {'radius': 10}

Circle.__dict__
# >>> 출력:
# mappingproxy({'__module__': '__main__',
#               '__init__': <function __main__.Circle.__init__(self, radius=0)>,
#               'getArea': <function __main__.Circle.getArea(self)>,
#               'getPerimeter': <function __main__.Circle.getPerimeter(self)>,
#               '__dict__': <attribute '__dict__' of 'Circle' objects>,
#               '__weakref__': <attribute '__weakref__' of 'Circle' objects>,
#               '__doc__': None,
#               '__annotations__': {}})

Sphere.__dict__
# >>> 출력:
# mappingproxy({'__module__': '__main__',
#               'getVolume': <function __main__.Sphere.getVolume(self)>,
#               'getArea': <function __main__.Sphere.getArea(self)>,
#               '__doc__': None,
#               '__annotations__': {}})

s1.getArea()
# >>> 출력:
# 1256.6370614359173

# 부모쪽의 메소드 소출하려면?
Circle.getArea(s1)
# >>> 출력:
# 314.1592653589793

# # 상속과 생성자

class Vehicle:
  def __init__(self, speed):
    print('Vehicle 생성 speed=', speed)
    self.speed = speed

vehicle1 = Vehicle(100)
# >>> 출력:
# Vehicle 생성 speed= 100

class Car(Vehicle):
  pass

# car1 = Car()
# TypeError: Vehicle.__init__() missing 1 required positional argument: 'speed'
# >>> 출력:
# TypeError: Vehicle.__init__() missing 1 required positional argument: 'speed'

Vehicle.__dict__
# >>> 출력:
# mappingproxy({'__module__': '__main__',
#               '__init__': <function __main__.Vehicle.__init__(self, speed)>,
#               '__dict__': <attribute '__dict__' of 'Vehicle' objects>,
#               '__weakref__': <attribute '__weakref__' of 'Vehicle' objects>,
#               '__doc__': None,
#               '__annotations__': {}})

Car.__dict__
# >>> 출력:
# mappingproxy({'__module__': '__main__',
#               '__doc__': None,
#               '__annotations__': {}})

car1 = Car(10)
# >>> 출력:
# Vehicle 생성 speed= 10

class Vehicle:
  def __init__(self, speed = 0):
    print('Vehicle 생성 speed=', speed)
    self.speed = speed

class Car(Vehicle):
  pass

car1 = Car()
# >>> 출력:
# Vehicle 생성 speed= 0

class Vehicle:
  def __init__(self, speed = 0):
    print('Vehicle 생성 speed=', speed)
    self.speed = speed

class Car(Vehicle):
  def __init__(self, oil = 0):
    print('Car 생성 oil=', oil)
    self.oil = oil

car1 = Car()  # 부모쪽의 생성자가 호출되는게 아니다!?
# >>> 출력:
# Car 생성 oil= 0

car1.oil
# >>> 출력:
# 0

# car1.speed   # 그래서! 부모쪽의 인스턴스 변수도 생기지 않았다!
# AttributeError: 'Car' object has no attribute 'speed'
# >>> 출력:
# AttributeError: 'Car' object has no attribute 'speed'

# 상속받은 자식쪽에서 생성자를 제공하는 경우 (오버라이딩)
# 부모쪽의 생성자를 명시적으로 호출해줄 필요가 있다.

class Vehicle:
  def __init__(self, speed = 0):
    print('Vehicle 생성 speed=', speed)
    self.speed = speed

class Car(Vehicle):
  def __init__(self, oil = 0):
    super().__init__(20)  # 명시적으로 부모의 생성자 호출
    print('Car 생성 oil=', oil)
    self.oil = oil

car1 = Car(10)
# >>> 출력:
# Vehicle 생성 speed= 20
# Car 생성 oil= 10

car1.speed
# >>> 출력:
# 20

class Vehicle:
  def __init__(self, speed = 0):
    print('Vehicle 생성 speed=', speed)
    self.speed = speed

class Car(Vehicle):
  def __init__(self, speed=0, oil = 0):
    super().__init__(speed)
    print('Car 생성 oil=', oil)
    self.oil = oil

car1 = Car(5, 9)
# >>> 출력:
# Vehicle 생성 speed= 5
# Car 생성 oil= 9

class HybridCar(Car):
  def __init__(self, electricity=0, speed=0, oil=0):
    super().__init__(speed, oil)
    print(f'HybridCar 생성 electricity:{electricity},speed:{speed},oil:{oil}')
    self.electricity = electricity

hybrid1 = HybridCar(10000, 46, 700)
# >>> 출력:
# Vehicle 생성 speed= 46
# Car 생성 oil= 700
# HybridCar 생성 electricity:10000,speed:46,oil:700

# 언제 '상속' 관계로 만들것인가?

# IS-A 관계 -> 상속관계로 만든다
# HAS-A 관계 -> 클래스의 속성으로 만든다.

# Vehicle
#   └─ Car
#       └─ HybridCar

# Vehicle IS-A Car  ? X
# Car IS-A Vehicle ? ○

# 반면에
# Car, Tire

# Car IS-A Tire ?  X
# Tire IS-A Car ?  X
#  따라서 위 둘은 상속관계로 만들면 안된다.

# Car HAS-A Tire ? O
# Tire HAS-A Car ? X

#  따라서 위 경우는 객체의 변수(필드)로 만들어 준다

class Tire:
  pass

class Car:
  def __init__(self):
    self.tire1 = Tire()
    self.tire2 = Tire()
    self.tire3 = Tire()
    self.tire4 = Tire()

# # ■ Magic Method
# 
# special method 라고도 한다
# 
# https://docs.python.org/3/reference/datamodel.html#specialnames

# 20 + "20"
# >>> 출력:
# TypeError: unsupported operand type(s) for +: 'int' and 'str'

# Magic Method 매직 메소드란?
# 클래스안에 정의할 수 있는 스페셜 메소드이며 클래스를 int, str, list등의 파이썬의 빌트인 타입(built-in type)과 같은 작동을 하게 해준다.
# +, -, >, < 등의 오퍼레이터에 대해서 각각의 데이터 타입에 맞는 메소드로 오버로딩하여 백그라운드에서 연산을 한다.
# __init__이나 __str__과 같이 메소드 이름 앞뒤에 더블 언더스코어("__")를 붙인다.

# special method 라고도 한다 https://docs.python.org/3/reference/datamodel.html#specialnames

"""

연산자     메소드                     설명
─────────────────────────
         <binary operator>

 +      __add__(self, other)         덧셈
 *      __mul__(self, other)         곱셈
 -      __sub__(self, other)         뺄셈
 /      __truediv__(self, other)     나눗셈
 //     __floordiv__(self, other)
 %       __mod__(self, other)             나머지
 **      __pow__(self, other[, modulo])
 >>      __lshift__(self, other)
 <<      __rshift__(self, other)
 &       __and__(self, other)
 ^      __xor__(self, other)
 |      __or__(self, other)


             <Extended operator>
+=         __iadd__(self, other)
-=         __isub__(self, other)
*=         __imul__(self, other)
/=         __idiv__(self, other)
//=        __ifloordiv__(self, other)
%=         __imod__(self, other)
**=        __ipow__(self, other)
<<=        __ilshift__(self, other)
>>=        __irshift__(self, other)
&=         __iand__(self, other)
^=         __ixor__(self, other)
|=         __ior__(self, other)

        <unary operators>
-
+
abs()
~
complex()
int()
long()
float()
oct()
hex()



 <       __lt__(self, other)         작다(미만)
 <=      __le__(self, other)         작거나 같다(이하)
 ==      __eq__(self, other)         같다
 !=      __ne__(self, other)         같지 않다
 >      __gt__(self, other)          크다(초과)
 >=     __ge__(self, other)          크거나 같다(이상)
 [index]   __getitem__(self, index)   인덱스 연산자
 in       __contains__(self, value)   멤버 확인
 len     __len__(self)                요소 길이
 str      __str__(self)                문자열 표현


         __init__
         __del__
         __new__

         __repr__(self)              representative form
 """
None

s1 = "Hello"
s2 = "Python"

s1 + s2, s1.__add__(s2)
# >>> 출력:
# ('HelloPython', 'HelloPython')

s1 * 2, s1.__mul__(2)
# >>> 출력:
# ('HelloHello', 'HelloHello')

"PYTHON" > "Python", "PYTHON".__gt__("Python")
# >>> 출력:
# (False, False)

s1[0], s1.__getitem__(0)
# >>> 출력:
# ('H', 'H')

'e' in s1, s1.__contains__('e')
# >>> 출력:
# (True, True)

# ## $__repr__, __str__$ 차이

# __repr__(),  __str__()
# 공통점 : 객체를 문자열 리턴
# 차이점
#    `__repr__` : out 값
#    `__str__` : str(), print() 등에서 문자열(str) 변환시 호출됨,
#                 오버라이딩 안되어 있으면 __repr__ 의 값을 리턴한다

class Student:
  def __init__(self, name, grade):
    self.name = name
    self.grade = grade

s1 = Student('김수진', 3)
s1  # __repr__()
# >>> 출력:
# <__main__.Student at 0x7ef2a7bd7890>

print(s1)   # __str__()
# >>> 출력:
# <__main__.Student object at 0x7ef2a7bd7890>

class Student:
  def __init__(self, name, grade):
    self.name = name
    self.grade = grade

  def __repr__(self):
    return f'이름 {self.name}, 학년 {self.grade}'

s2 = Student('박수진', 4)
s2
# >>> 출력:
# 이름 박수진, 학년 4

print(s2)  # __str__ 결과값을 출력.  없으면 __repr__ 결과값 출력
# >>> 출력:
# 이름 박수진, 학년 4

class Student:
  def __init__(self, name, grade):
    self.name = name
    self.grade = grade

  def __repr__(self):
    return f'이름 {self.name}, 학년 {self.grade}'

  def __str__(self):
    return f'{self.name}:{self.grade}'

s3 = Student('노진구', 4)
s3  # __repr__
# >>> 출력:
# 이름 노진구, 학년 4

print(s3)  # __str__
# >>> 출력:
# 노진구:4

# "hello" + s3 # str + Student  에러

"hello" + str(s3)   # __str__() 가 str() 형변환 함수에서 호출된다.
# >>> 출력:
# 'hello노진구:4'

f'hello {s3}'  # f-string 안에서도 __str__ 이 사용됨.
# >>> 출력:
# 'hello 노진구:4'

# ## 객체간의 연산
# Magic method 를 통해 가능!

class Number:
    def __init__(self, number):
        self.number = number

    def __repr__(self):
        return str(self.number)

f1 = Number(10)
f2 = Number(20)

print(f1, f2)
# >>> 출력:
# 10 20

# f1 + f2   # Number + Number  =>  f1.__add__(f2)
# TypeError: unsupported operand type(s) for +: 'Number' and 'Number'
# >>> 출력:
# TypeError: unsupported operand type(s) for +: 'Number' and 'Number'

class Number:
    def __init__(self, number):
        self.number = number

    def __repr__(self):
        return str(self.number)

    def __add__(self, other):
      return self.number + other.number

f1 = Number(10)
f2 = Number(20)

f1 + f2
# >>> 출력:
# 30

f1.__add__(f2)  # self <- f1,  other <- f2
# >>> 출력:
# 30

f3 = Number(30)
f1 + f2 + f3
# >>> 출력:
# TypeError: unsupported operand type(s) for +: 'int' and 'Number'

class Number:
    def __init__(self, number):
        self.number = number

    def __repr__(self):
        return str(self.number)

    def __add__(self, other):
      return Number(self.number + other.number)

f1 = Number(10)
f2 = Number(20)
f3 = Number(30)
print(f1 + f2 + f3)
# >>> 출력:
# 60

# ## $__call__()$ 메소드

class Foo:

  def hello(self):
    print('hello Foo')

f1 = Foo()
f1.hello()
# >>> 출력:
# hello Foo

Foo().hello()
# >>> 출력:
# hello Foo

# Foo()()
# TypeError: 'Foo' object is not callable
# >>> 출력:
# TypeError: 'Foo' object is not callable

class Foo:

  def hello(self):
    print('hello Foo')

  # __call__ : 객체를 호출하는 magic method
  def __call__(self, postfix='Python'):
    print(f'hello {postfix}')

Foo()()
# >>> 출력:
# hello Python

Foo()('Java')
# >>> 출력:
# hello Java

class Counter:
  def __init__(self):
    self.count = 0

  def __call__(self, inc=1):
    self.count += inc
    return self.count

c = Counter()

c()
# >>> 출력:
# 1

c()
# >>> 출력:
# 2

c(100)
# >>> 출력:
# 102

a = 100
