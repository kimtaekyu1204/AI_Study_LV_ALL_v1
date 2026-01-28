a, b, c = input("3개 입력하세요")
# >>> 출력:
# 3개 입력하세요10 20 30
# ValueError: too many values to unpack (expected 3)

# # 여러값 입력받기  input().split()

a, b, c = input("3개 입력하세요").split()
print(f'a={a}, b={b}, c={c}')
# >>> 출력:
# 3개 입력하세요10 20 30
# a=10, b=20, c=30

hour, minute, second = input("시:분:초 형식입력").split(":")
print(f'hour={hour} minute={minute} second={second}')
# >>> 출력:
# 시:분:초 형식입력23:34:18
# hour=23 minute=34 second=18

hour = int(hour)
minute = int(minute)
second = int(second)

hour * 3600 + minute * 60 + second
# >>> 출력:
# 84858

# ## 입력한 str(들)을 한번에 변환하기 map() 사용

a, b, c = input("3개의 정수 입력: ").split()
print(a, b, c)
print(a + b + c)
# >>> 출력:
# 3개의 정수 입력: 10 20 30
# 10 20 30
# 102030

a, b, c = map(int, input("3개의 정수 입력: ").split())
print(a, b, c)
print(a + b + c)
# >>> 출력:
# 3개의 정수 입력: 10 20 30
# 10 20 30
# 60

"""
https://jungol.co.kr/problem/133?cursor=eyJwcm9ibGVtc2V0IjoyLCJmaWVsZCI6NiwiaWR4IjoyMH0=
133 : 반복제어문2 - 형성평가4
문제
100 이하의 자연수 n을 입력받고 n개의 정수를 입력받아 평균을 출력하는 프로그램을 작성하시오.
(평균은 반올림하여 소수 둘째자리까지 출력하도록 한다.)


입력 예]
3
99 65 30

출력 예]
64.67

"""
None

n = int(input())

total = sum(map(int, input().split()))

print("{:.2f}".format(total / n))
# >>> 출력:
# 3
# 99 65 30
# 64.67

a
# >>> 출력:
# '10 20 30'
