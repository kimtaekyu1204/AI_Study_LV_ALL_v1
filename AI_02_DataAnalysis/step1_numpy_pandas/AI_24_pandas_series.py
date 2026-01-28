# # Pandas
# panel data and analysis<br>
# 공식: https://pandas.pydata.org<br>
# ![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fzy1Or%2FbtqM39h0uLB%2FfHnntePsHiwAIFStdqGsLk%2Fimg.png)
# 
# <img src="http://image.yes24.com/Goods/73268296/800x0" width="200px">

# Pandas : 데이터 분석 라이브러리
# 행(row), 열(column) 형태의 정형 데이터 객체를 다룸
#  2차원 데이터
#  엑셀의 시트, RDB 의 테이블과 같은 형태

# 대표적인 객체는
# DataFrame 과 Series

# # Series
#   - pandas의 기본 객체 중 하나
#   - numpy의 ndarray를 기반으로 **인덱싱 기능**이 추가된 **1차원 배열**을 나타냄 (index x value 의 형태)
#   - index를 지정하지 않을 시, 기본적으로 ndarray와 같이 0-based 인덱스 생성, 지정할 경우 명시적으로 지정된 index를 사용
#   - **같은 타입**의 0개 이상의 데이터를 가질 수 있음

import numpy as np
import pandas as pd

# # Series 생성

s1 = pd.Series([10, 20, 30])
s1
# >>> 출력:
# 0    10
# 1    20
# 2    30
# dtype: int64

# 왼쪽이 index, 오른쪽이 value (값),  dtype : 데이터 타입
# 데이터로만 생성했을때,  index 는 0-base 인덱스로 생성됨

s1.values
# >>> 출력:
# array([10, 20, 30])

s1.index
# >>> 출력:
# RangeIndex(start=0, stop=3, step=1)

s1.dtype
# >>> 출력:
# dtype('int64')

s3 = pd.Series(np.arange(200))
s3
# >>> 출력:
# 0        0
# 1        1
# 2        2
# 3        3
# 4        4
#       ... 
# 195    195
# 196    196
# 197    197
# 198    198
# 199    199
# Length: 200, dtype: int64

# indexing
s3[4]
# >>> 출력:
# 4

# ## data=, index= 명시하여 Series 생성

s4 = pd.Series(data=[1, 2, 3], index=[100, 200, 300])
s4
# >>> 출력:
# 100    1
# 200    2
# 300    3
# dtype: int64

# s4[0]  # keyError
s4[100]
# >>> 출력:
# 1

s4.index
# >>> 출력:
# Index([100, 200, 300], dtype='int64')

# index 는 굳이 정수일 필요 없다.
s5 = pd.Series([1, 2, 3], ['John', 'Mark', "Charlie"])
s5
# >>> 출력:
# John       1
# Mark       2
# Charlie    3
# dtype: int64

s5['John']
# >>> 출력:
# 1

# index 중복 허용된다 ?!?!?!?!?
s6 = pd.Series([1, 2, 3], ['John', 'John', "Charlie"])
s6
# >>> 출력:
# John       1
# John       2
# Charlie    3
# dtype: int64

s6['John']   # 결과 Series
# >>> 출력:
# John    1
# John    2
# dtype: int64

# ## dict 로 Series 생성

sdata = {'김철수': 35000, '이상현': 67000, '최종수': 12000, '박진관': 4000}
sdata
# >>> 출력:
# {'김철수': 35000, '이상현': 67000, '최종수': 12000, '박진관': 4000}

s6 = pd.Series(sdata)
s6
# >>> 출력:
# 김철수    35000
# 이상현    67000
# 최종수    12000
# 박진관     4000
# dtype: int64

s7 = pd.Series(data = np.arange(5), index=np.arange(100, 105), dtype=np.int16)
s7
# >>> 출력:
# 100    0
# 101    1
# 102    2
# 103    3
# 104    4
# dtype: int16

s7.index
# >>> 출력:
# Index([100, 101, 102, 103, 104], dtype='int64')

s7.index.values
# >>> 출력:
# array([100, 101, 102, 103, 104])

s7[102], s7[104]
# >>> 출력:
# (2, 4)

s7[[102, 104]]
# >>> 출력:
# 102    2
# 104    4
# dtype: int16

s7[100]  # scalar value
# >>> 출력:
# 0

s7[[100]] # Series
# >>> 출력:
# 100    0
# dtype: int16

s7[[104, 101, 104, 104, 102]]
# >>> 출력:
# 104    4
# 101    1
# 104    4
# 104    4
# 102    2
# dtype: int16

s7
# >>> 출력:
# 100    0
# 101    1
# 102    2
# 103    3
# 104    4
# dtype: int16

s7[104] = 70  # value 변경 가능.
s7
# >>> 출력:
# 100     0
# 101     1
# 102     2
# 103     3
# 104    70
# dtype: int16

# s7[105]

s7[105] = 90  # 기존에 없던 인덱스에 값 대입 --> 새로운 값 추가 !
s7
# >>> 출력:
# 100     0
# 101     1
# 102     2
# 103     3
# 104    70
# 105    90
# dtype: int16

s7[[101, 102]] = 10, 20

s7
# >>> 출력:
# <ipython-input-31-32815ed5ee2c>:1: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '(10, 20)' has dtype incompatible with int16, please explicitly cast to a compatible dtype first.
#   s7[[101, 102]] = 10, 20
# 100     0
# 101    10
# 102    20
# 103     3
# 104    70
# 105    90
# dtype: int64

s7.index
# >>> 출력:
# Index([100, 101, 102, 103, 104, 105], dtype='int64')

# 기존의 index 사용하여 새로운 Series 생성 가능
s8 = pd.Series(np.arange(6), s7.index)
s8
# >>> 출력:
# 100    0
# 101    1
# 102    2
# 103    3
# 104    4
# 105    5
# dtype: int64

# # Series name, index name, index 변경

s6
# >>> 출력:
# 김철수    35000
# 이상현    67000
# 최종수    12000
# 박진관     4000
# dtype: int64

s6.name = '급여'

s6
# >>> 출력:
# 김철수    35000
# 이상현    67000
# 최종수    12000
# 박진관     4000
# Name: 급여, dtype: int64

s6.index.name = '이름'

s6
# >>> 출력:
# 이름
# 김철수    35000
# 이상현    67000
# 최종수    12000
# 박진관     4000
# Name: 급여, dtype: int64

s6.index
# >>> 출력:
# Index(['김철수', '이상현', '최종수', '박진관'], dtype='object', name='이름')

s6.index = ['Kim', "Lee", "Choi", "Park"]

s6
# >>> 출력:
# Kim     35000
# Lee     67000
# Choi    12000
# Park     4000
# Name: 급여, dtype: int64

# # **Series ndim, shape, size, len, unique, count, value_counts 함수**
#  - size : 개수 반환
#  - shape : 튜플형태로 shape반환
#  - unique: 유일한 값만 ndarray로 반환
#  - count : NaN을 제외한 개수를 반환
#  - mean: NaN을 제외한 평균
#  - value_counts: NaN을 제외하고 각 값들의 빈도를 반환

s1 = pd.Series([1, 1, 2, 1, 2, 2, 2, 1, 1, 3, 3, 4, 5, 5, 7, np.NaN])
s1

# NaN (Not a Number : 숫자가 들어와야 하는데 숫자가 아닌 값)
# 실무 데이터를 다루다 보면 NaN 을 엄청나게 자주 만나게 된다.
#    원본데이터의 '누락' 이라든지, DBMS 상의 null 값이라든지 등등. 여러가지 이슈로 NaN 값 발생
#    그래서 데이터 분석시 NaN 값을 제거 하든지 혹은 치환 하든지.. 해서 처리할줄 알아야 한다
# >>> 출력:
# 0     1.0
# 1     1.0
# 2     2.0
# 3     1.0
# 4     2.0
# 5     2.0
# 6     2.0
# 7     1.0
# 8     1.0
# 9     3.0
# 10    3.0
# 11    4.0
# 12    5.0
# 13    5.0
# 14    7.0

s1.ndim
# >>> 출력:
# 1

s1.shape
# >>> 출력:
# (16,)

s1.size
# >>> 출력:
# 16

len(s1)
# >>> 출력:
# 16

s1.unique()
# >>> 출력:
# array([ 1.,  2.,  3.,  4.,  5.,  7., nan])

s1.count()  # 데이터가 있는것만 count 한다!  NaN 은 제외된다!
# >>> 출력:
# 15

a1 = np.array([2, 2, 2, 2, np.NaN])
a1.mean()  # nan!  numpy 의 경우 원소중에 NaN 있으면 연산 불가 (결과 NaN!)
# >>> 출력:
# nan

pd.Series(a1).mean()  # NaN 은 제외하고(무시하고) 평균값 계산!
# >>> 출력:
# 2.0

s1.value_counts()
# >>> 출력:
# 1.0    5
# 2.0    4
# 3.0    2
# 5.0    2
# 4.0    1
# 7.0    1
# Name: count, dtype: int64

s1.head()  # 상위 5개
# >>> 출력:
# 0    1.0
# 1    1.0
# 2    2.0
# 3    1.0
# 4    2.0
# dtype: float64

s1.head(3)
# >>> 출력:
# 0    1.0
# 1    1.0
# 2    2.0
# dtype: float64

s1.tail()  # 하위 5개
# >>> 출력:
# 11    4.0
# 12    5.0
# 13    5.0
# 14    7.0
# 15    NaN
# dtype: float64

# # Series 연산
# - index 기준으로 연산

# numpy 와 마찬가지로
# Series 도 Series 끼리 연산 가능.
#  기억할 포인트는 '같은 인덱스' 기준으로 연산한다는 겁니다
#  이 특징은 DataFrame 에서도 동일하게 적용됩니다.

s1 = pd.Series([1, 2, 3, 4], ['a', 'b', 'c', 'd'])
s2 = pd.Series([6, 3, 2, 1], ['d', 'c', 'b', 'a'])  # 인덱스 순서 다르다

s1
# >>> 출력:
# a    1
# b    2
# c    3
# d    4
# dtype: int64

s2
# >>> 출력:
# d    6
# c    3
# b    2
# a    1
# dtype: int64

s1 + s2
# >>> 출력:
# a     2
# b     4
# c     6
# d    10
# dtype: int64

# # 산술연산
#  - Series의 경우에도 **스칼라와의 연산** 각 원소별로 스칼라와의 연산이 적용
#  - Series와의 연산은 각 인덱스에 맞는 값끼리 연산이 적용
#    - 이때, 인덱스의 pair가 맞지 않으면, 결과는 NaN

s1 ** 2
# >>> 출력:
# a     1
# b     4
# c     9
# d    16
# dtype: int64

s1 ** s2
# >>> 출력:
# a       1
# b       4
# c      27
# d    4096
# dtype: int64

s1['k'] = 7
s2['e'] = 9

s1
# >>> 출력:
# a    1
# b    2
# c    3
# d    4
# k    7
# dtype: int64

s2
# >>> 출력:
# d    6
# c    3
# b    2
# a    1
# e    9
# dtype: int64

s1 + s2
# >>> 출력:
# a     2.0
# b     4.0
# c     6.0
# d    10.0
# e     NaN
# k     NaN
# dtype: float64

#  # **Boolean selection**
#   - boolean Series가 []와 함께 사용되면 True 값에 해당하는 값만 새로 반환되는 Series객체에 포함됨
#   - 다중조건의 경우, &(and), |(or)를 사용하여 연결 가능

s = pd.Series(np.arange(10), np.arange(10) + 1)
s
# >>> 출력:
# 1     0
# 2     1
# 3     2
# 4     3
# 5     4
# 6     5
# 7     6
# 8     7
# 9     8
# 10    9
# dtype: int64

s > 5   # 결과는 Boolean Series
# >>> 출력:
# 1     False
# 2     False
# 3     False
# 4     False
# 5     False
# 6     False
# 7      True
# 8      True
# 9      True
# 10     True
# dtype: bool

s[s > 5]
# >>> 출력:
# 7     6
# 8     7
# 9     8
# 10    9
# dtype: int64

s[s % 2 == 0]
# >>> 출력:
# 1    0
# 3    2
# 5    4
# 7    6
# 9    8
# dtype: int64

# index 를 사용한 필터링도 가능!

s.index
# >>> 출력:
# Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='int64')

s.index > 5  # boolean mask
# >>> 출력:
# array([False, False, False, False, False,  True,  True,  True,  True,
#         True])

s[s.index > 5]
# >>> 출력:
# 6     5
# 7     6
# 8     7
# 9     8
# 10    9
# dtype: int64

# # Series 값 삭제: drop()

s = pd.Series(np.arange(100, 105), ['a', 'b', 'c', 'd', 'e'])
s
# >>> 출력:
# a    100
# b    101
# c    102
# d    103
# e    104
# dtype: int64

s['a'] = 200
s
# >>> 출력:
# a    200
# b    101
# c    102
# d    103
# e    104
# dtype: int64

s['k'] = 300
s
# >>> 출력:
# a    200
# b    101
# c    102
# d    103
# e    104
# k    300
# dtype: int64

s.drop('k') #  원본 변화 없다!
# >>> 출력:
# a    200
# b    101
# c    102
# d    103
# e    104
# dtype: int64

s
# >>> 출력:
# a    200
# b    101
# c    102
# d    103
# e    104
# k    300
# dtype: int64

# ## inplace=

# ★중요★
# drop() 함수는 drop 한'결과' 를 새로 만들어서 리턴한거고
# 원본 데이터 s 는 '변경'하지 않습니다

# numpy , pandas, 그밖의 수많은 데이터 관련 모듈들 에서의
# 대부분의 객체 함수들은 호출한 원본 객체를 '변경'하진 않고
# 원본의 '사본'을 만든뒤 '사본' 에 연산수행하여 그 '사본' 을 리턴하도록 동작 한다
# (함수마다 조금씩 다르긴 하다)

# 그러나!
# inplace=True 파라미터를 주면 '사본' 을 만들지 않고 '원본'을 변화시키도록 동작한다

s.drop('k', inplace=True)   # None 리턴  (원본 변화 발생!)

s
# >>> 출력:
# a    200
# b    101
# c    102
# d    103
# e    104
# dtype: int64

# # **Slicing**
#  - 리스트, ndarray와 동일하게 적용
#  - index 가 문자열인 경우 slicing 시 마지막도 포함 (ndarray 와 차이점!)

s1 = pd.Series(np.arange(100, 105))
s1
# >>> 출력:
# 0    100
# 1    101
# 2    102
# 3    103
# 4    104
# dtype: int64

s1[1:3]
# >>> 출력:
# 1    101
# 2    102
# dtype: int64

# 인덱스가 '문자' 인 경우.
s2 = pd.Series(np.arange(100, 105), ['a','b','c','d','e'])
s2
# >>> 출력:
# a    100
# b    101
# c    102
# d    103
# e    104
# dtype: int64

s2[1]
# >>> 출력:
# <ipython-input-82-8755b941c10d>:1: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
#   s2[1]
# 101

s2[1:3]
# >>> 출력:
# b    101
# c    102
# dtype: int64

s2['b':'d']  # index 가 문자인 경우, slicing 할때 마지막도 '포함'
# >>> 출력:
# b    101
# c    102
# d    103
# dtype: int64

# # Multi level index

s3 = pd.Series([1, 2, 3, 4, 5, 6], ['a', 'a', 'b', 'b', 'c', 'c'])
s3
# >>> 출력:
# a    1
# a    2
# b    3
# b    4
# c    5
# c    6
# dtype: int64

s3.index
# >>> 출력:
# Index(['a', 'a', 'b', 'b', 'c', 'c'], dtype='object')

# ↑ 단일 level index 다!

# multi-level index 를 가진 Series
s4 = pd.Series(
    np.arange(100, 106),
    index = [
        ['a', 'a', 'b', 'b', 'c', 'c'],
        [ 1 ,  2 ,  1 ,  2,   1,   2 ],
      ]
)

s4
# >>> 출력:
# a  1    100
#    2    101
# b  1    102
#    2    103
# c  1    104
#    2    105
# dtype: int64

s4.index
# >>> 출력:
# MultiIndex([('a', 1),
#             ('a', 2),
#             ('b', 1),
#             ('b', 2),
#             ('c', 1),
#             ('c', 2)],
#            )

# tuple 인덱싱으로 value 접근
s4['a', 1]
# >>> 출력:
# 100

s4['a']
# >>> 출력:
# 1    100
# 2    101
# dtype: int64

# s4: 2-level index
# s4['a'] : 1-level index

s4['a'][1]
# >>> 출력:
# 100

s4['a':'b']
# >>> 출력:
# a  1    100
#    2    101
# b  1    102
#    2    103
# dtype: int64

s4[['a', 'c']]
# >>> 출력:
# a  1    100
#    2    101
# c  1    104
#    2    105
# dtype: int64

# ## xs() 를 사용한 인덱스
# cross-section

s4
# >>> 출력:
# a  1    100
#    2    101
# b  1    102
#    2    103
# c  1    104
#    2    105
# dtype: int64

s4.xs('a')   # level 0 인덱싱 <= s4['a'] 와 결과 동일
# >>> 출력:
# 1    100
# 2    101
# dtype: int64

s4.xs(('a', 1))   # s4['a', 1] 과 동일
# >>> 출력:
# 100

s4
# >>> 출력:
# a  1    100
#    2    101
# b  1    102
#    2    103
# c  1    104
#    2    105
# dtype: int64

# s4.xs(2)  # 에러  level 0 에는 2 라는 index 가 없다.
s4.xs(2, level=1)   # level 1 인덱싱
# >>> 출력:
# a    101
# b    103
# c    105
# dtype: int64

s4.xs(2, level=-1)
# >>> 출력:
# a    101
# b    103
# c    105
# dtype: int64

s4.xs((2, 'b'), level=(1, 0))
# >>> 출력:
# b  2    103
# dtype: int64

# ## Sereis 에서 level 별 계산

data = pd.Series(
    [100, 200, 300, 400, 500, 600],
    index=[
        ["a", "a", "a", "b", "b", "c"],  # level 0 index
        [1,   2,   3,   1,   2,   1],    # level 1 index
    ]
)
data
# >>> 출력:
# a  1    100
#    2    200
#    3    300
# b  1    400
#    2    500
# c  1    600
# dtype: int64

data.sum()
# >>> 출력:
# 2100

data.mean()
# >>> 출력:
# 350.0

data.groupby(level = 0).sum()
# >>> 출력:
# a    600
# b    900
# c    600
# dtype: int64
