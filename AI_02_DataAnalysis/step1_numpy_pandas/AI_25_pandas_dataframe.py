# # DataFrame
#   - Series가 1차원이라면 DataFrame은 2차원으로 확대된 버젼
#   - Excel spreadsheet이라고 생각하면 이해하기 쉬움
#   - 2차원이기 때문에 인덱스가 row, column로 구성됨
#    - row는 각 개별 데이터를, column은 개별 속성을 의미 ('feature' 라고도 함)
#   - Data Analysis, Machine Learning에서 data 변형을 위해 가장 많이 사용

import numpy as np
import pandas as pd
import os

base_path = r'/content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)'

filepath = os.path.join(base_path, 'titanic.csv')
filepath
# >>> 출력:
# '/content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)/titanic.csv'

pd.read_csv(filepath)
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 0              1         0       3   
# 1              2         1       1   
# 2              3         1       3   
# 3              4         1       1   
# 4              5         0       3   
# ..           ...       ...     ...   
# 886          887         0       2   
# 887          888         1       1   
# 888          889         0       3   
# 889          890         1       1   
# 890          891         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 0                              Braund, Mr. Owen Harris    male  22.0      1   

df = None
def load_titanic():
  return pd.read_csv(filepath)

df = load_titanic()
df
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 0              1         0       3   
# 1              2         1       1   
# 2              3         1       3   
# 3              4         1       1   
# 4              5         0       3   
# ..           ...       ...     ...   
# 886          887         0       2   
# 887          888         1       1   
# 888          889         0       3   
# 889          890         1       1   
# 890          891         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 0                              Braund, Mr. Owen Harris    male  22.0      1   

# 컬럼 의미
# survival - Survival (0 = No; 1 = Yes)
# class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
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

type(df)
# >>> 출력:
# pandas.core.frame.DataFrame

df.head()
# >>> 출력:
#    PassengerId  Survived  Pclass  \
# 0            1         0       3   
# 1            2         1       1   
# 2            3         1       3   
# 3            4         1       1   
# 4            5         0       3   
# 
#                                                 Name     Sex   Age  SibSp  \
# 0                            Braund, Mr. Owen Harris    male  22.0      1   
# 1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 2                             Heikkinen, Miss. Laina  female  26.0      0   
# 3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 4                           Allen, Mr. William Henry    male  35.0      0   
# 
#    Parch            Ticket     Fare Cabin Embarked  

df.tail()
# >>> 출력:
#      PassengerId  Survived  Pclass                                      Name  \
# 886          887         0       2                     Montvila, Rev. Juozas   
# 887          888         1       1              Graham, Miss. Margaret Edith   
# 888          889         0       3  Johnston, Miss. Catherine Helen "Carrie"   
# 889          890         1       1                     Behr, Mr. Karl Howell   
# 890          891         0       3                       Dooley, Mr. Patrick   
# 
#         Sex   Age  SibSp  Parch      Ticket   Fare Cabin Embarked  
# 886    male  27.0      0      0      211536  13.00   NaN        S  
# 887  female  19.0      0      0      112053  30.00   B42        S  
# 888  female   NaN      1      2  W./C. 6607  23.45   NaN        S  
# 889    male  26.0      0      0      111369  30.00  C148        C  
# 890    male  32.0      0      0      370376   7.75   NaN        Q

# ## DataFrame 데이터 파악하기
#  - shape 속성 (row, column)
#  - describe 함수 - 숫자형 데이터의 통계치 계산  (descriptive statistics : 기술통계량)
#  - info 함수 - 데이터 타입, 각 아이템의 개수 등 출력

df.shape
# >>> 출력:
# (891, 12)

df.describe()
# 숫자형 데이터에 대해 기술통계량 정보
# count, mean, std : 개수, 평균, 표준편차
# max, min : 최대, 최소
# 25%, 50%, 75%  : 하위 4분위 하위 2분위...
# >>> 출력:
#        PassengerId    Survived      Pclass         Age       SibSp  \
# count   891.000000  891.000000  891.000000  714.000000  891.000000   
# mean    446.000000    0.383838    2.308642   29.699118    0.523008   
# std     257.353842    0.486592    0.836071   14.526497    1.102743   
# min       1.000000    0.000000    1.000000    0.420000    0.000000   
# 25%     223.500000    0.000000    2.000000   20.125000    0.000000   
# 50%     446.000000    0.000000    3.000000   28.000000    0.000000   
# 75%     668.500000    1.000000    3.000000   38.000000    1.000000   
# max     891.000000    1.000000    3.000000   80.000000    8.000000   
# 
#             Parch        Fare  
# count  891.000000  891.000000  
# mean     0.381594   32.204208  
# std      0.806057   49.693429  
# min      0.000000    0.000000  

df.info()
# 각 컬럼별 타입, 비어있지 않은 값의 개수
# >>> 출력:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   PassengerId  891 non-null    int64  
#  1   Survived     891 non-null    int64  
#  2   Pclass       891 non-null    int64  
#  3   Name         891 non-null    object 
#  4   Sex          891 non-null    object 
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64  
#  7   Parch        891 non-null    int64  
#  8   Ticket       891 non-null    object 
#  9   Fare         891 non-null    float64

df
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 0              1         0       3   
# 1              2         1       1   
# 2              3         1       3   
# 3              4         1       1   
# 4              5         0       3   
# ..           ...       ...     ...   
# 886          887         0       2   
# 887          888         1       1   
# 888          889         0       3   
# 889          890         1       1   
# 890          891         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 0                              Braund, Mr. Owen Harris    male  22.0      1   

# ## 인덱스(index)
#  - index 속성
#  - 각 아이템을 특정할 수 있는 고유의 값을 저장
#  - 복잡한 데이터의 경우, 멀티 인덱스로 표현 가능

df.index
# >>> 출력:
# RangeIndex(start=0, stop=891, step=1)

df.index.values
# >>> 출력:
# array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
#         13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
#         26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
#         39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
#         52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
#         65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
#         78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
#         91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
#        104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
#        117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
#        130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
#        143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
#        156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
#        169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
#        182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,

# ## 컬럼(column)
#  - columns 속성
#  - 각각의 특성(feature)을 나타냄
#  - 복잡한 데이터의 경우, 멀티 컬럼으로 표현 가능

df.columns
# >>> 출력:
# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object')

df.columns.values
# >>> 출력:
# array(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], dtype=object)

# # DataFrame 생성하기

df = pd.DataFrame([1, 2, 3])

df
# >>> 출력:
#    0
# 0  1
# 1  2
# 2  3

df = pd.DataFrame([
      'dog',  # 첫번째 행
      'cat',  # 두번째 행
      'bird',
    ])

df
# >>> 출력:
#       0
# 0   dog
# 1   cat
# 2  bird

df = pd.DataFrame([
      [1, 2, 3],  # 첫번째 행
      [4, 5, 6],  # 두번째 행
    ])

df
# >>> 출력:
#    0  1  2
# 0  1  2  3
# 1  4  5  6

# ## shape, ndim, size, len()

len(df)
# >>> 출력:
# 2

df.shape
# >>> 출력:
# (2, 3)

df.size
# >>> 출력:
# 6

df.ndim
# >>> 출력:
# 2

# df.dtype

df.dtypes
# >>> 출력:
# 0    int64
# 1    int64
# 2    int64
# dtype: object

# ## column, index 변경

df2 = pd.DataFrame([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
], columns=['a', 'b', 'c', 'd'])

df2
# >>> 출력:
#    a   b   c   d
# 0  1   2   3   4
# 1  5   6   7   8
# 2  9  10  11  12

df2.columns
# >>> 출력:
# Index(['a', 'b', 'c', 'd'], dtype='object')

# 컬럼 변경
df2.columns = ['국어', '영어', '수학', '과학']
df2
# >>> 출력:
#    국어  영어  수학  과학
# 0   1   2   3   4
# 1   5   6   7   8
# 2   9  10  11  12

df2.index
# >>> 출력:
# RangeIndex(start=0, stop=3, step=1)

# index 변경
df2.index = ['고길동', '둘리', '마이클']
df2
# >>> 출력:
#      국어  영어  수학  과학
# 고길동   1   2   3   4
# 둘리    5   6   7   8
# 마이클   9  10  11  12

# ## dict 로 DataFrame 만들기

data = {'a': 100, 'b': 200, 'c': 300}

pd.DataFrame(data, index = ['x'])
# >>> 출력:
#      a    b    c
# x  100  200  300

pd.DataFrame(data, index = ['x', 'y', 'z'])
# >>> 출력:
#      a    b    c
# x  100  200  300
# y  100  200  300
# z  100  200  300

data = {'a': [100], 'b': [200], 'c': [300]}  # value 가 1차원 데이터
pd.DataFrame(data)  # index= 없어도 생성됨!
# >>> 출력:
#      a    b    c
# 0  100  200  300

data = {'a': [100, 200, 300], 'b': ['cat', 'dog', 'bird'], 'c': [40, 50, 60]}
pd.DataFrame(data)
# >>> 출력:
#      a     b   c
# 0  100   cat  40
# 1  200   dog  50
# 2  300  bird  60

# ## Series로 부터 DataFrame 생성하기
#  - 각 Series의 인덱스 → column
#  -결국 DataFrame = Series x Series x ...  (Series 가 쌓인 형태가 DataFrame)

a = pd.Series([100, 200, 300], ['a', 'b', 'c'])
b = pd.Series([101, 202, 303], ['a', 'b', 'c'])
c = pd.Series([110, 220, 330], ['a', 'b', 'c'])

a
# >>> 출력:
# a    100
# b    200
# c    300
# dtype: int64

pd.DataFrame([a, b, c])
# >>> 출력:
#      a    b    c
# 0  100  200  300
# 1  101  202  303
# 2  110  220  330

d = pd.Series([400, 420, 430], ['a', 'b', 'd'])

pd.DataFrame([a, b, c, d])
# >>> 출력:
#        a      b      c      d
# 0  100.0  200.0  300.0    NaN
# 1  101.0  202.0  303.0    NaN
# 2  110.0  220.0  330.0    NaN
# 3  400.0  420.0    NaN  430.0

# ## '특정 컬럼' 의 이름 변경 rename()

df2
# >>> 출력:
#      국어  영어  수학  과학
# 고길동   1   2   3   4
# 둘리    5   6   7   8
# 마이클   9  10  11  12

df2.rename(columns={'국어': 'kor', '과학': 'Sci'})
# >>> 출력:
#      kor  영어  수학  Sci
# 고길동    1   2   3    4
# 둘리     5   6   7    8
# 마이클    9  10  11   12

df2
# >>> 출력:
#      국어  영어  수학  과학
# 고길동   1   2   3   4
# 둘리    5   6   7   8
# 마이클   9  10  11  12

# ## set_index(), reset_index()

df2
# >>> 출력:
#      국어  영어  수학  과학
# 고길동   1   2   3   4
# 둘리    5   6   7   8
# 마이클   9  10  11  12

df2.reset_index()
# 기존의 index 가 column 레벨로 올라오고
# 새로운 index 가 붙는다.
# >>> 출력:
#   index  국어  영어  수학  과학
# 0   고길동   1   2   3   4
# 1    둘리   5   6   7   8
# 2   마이클   9  10  11  12

df2.reset_index(drop=True)  # 기존의 index 는 제거됨.
# >>> 출력:
#    국어  영어  수학  과학
# 0   1   2   3   4
# 1   5   6   7   8
# 2   9  10  11  12

df2
# >>> 출력:
#      국어  영어  수학  과학
# 고길동   1   2   3   4
# 둘리    5   6   7   8
# 마이클   9  10  11  12

df2.set_index('국어')  # '국어' 컬럼이 index 로 내려가고,  기존의 인덱스를 대체한다.
# >>> 출력:
#     영어  수학  과학
# 국어            
# 1    2   3   4
# 5    6   7   8
# 9   10  11  12

df2
# >>> 출력:
#      국어  영어  수학  과학
# 고길동   1   2   3   4
# 둘리    5   6   7   8
# 마이클   9  10  11  12

# 퀴즈: df2 의 index 와 '국어' 컬럼을 바꾸려면?
# 	    index	  영어	수학	과학
# 국어
#  1	  고길동	  2	  3	    4
#  5	  둘리	    6	  7	    8
#  9    마이콜	 10	  11	  12

df2.reset_index().set_index('국어')
# >>> 출력:
#    index  영어  수학  과학
# 국어                  
# 1    고길동   2   3   4
# 5     둘리   6   7   8
# 9    마이클  10  11  12

# ## inplace=True
# 원본 변경

df2.set_index('국어', inplace=True)

df2
# >>> 출력:
#     영어  수학  과학
# 국어            
# 1    2   3   4
# 5    6   7   8
# 9   10  11  12

# # Multi-level index, Multi-level column

pd.DataFrame({'k': [10]})
# >>> 출력:
#     k
# 0  10

pd.DataFrame({('k0', 'k1'): [10]})
# >>> 출력:
#    k0
#    k1
# 0  10

pd.DataFrame({('k0', 'k1'): {'a': 10, 'b': 40}})
# >>> 출력:
#    k0
#    k1
# a  10
# b  40

pd.DataFrame({('k0', 'k1'): {('a1', 'a2'): 10, ('b1', 'b2'): 40}})
# >>> 출력:
#        k0
#        k1
# a1 a2  10
# b1 b2  40

pd.DataFrame({
    ('k', 'k1') : [10, 20, 30, 31],
    ('k', 'k2') : [40, 50, 60, 61],
    ('j', 'j1') : [70, 80, 90, 91],
    ('j', 'j2') : [100, 110, 120, 121],
}, index=[['서울', '서울', '경기', '경기'], ['평일', '휴일', '평일', '휴일']])
# >>> 출력:
#         k       j     
#        k1  k2  j1   j2
# 서울 평일  10  40  70  100
#    휴일  20  50  80  110
# 경기 평일  30  60  90  120
#    휴일  31  61  91  121

# # column 선택하기
#   - 기본적으로 [ ]는 column을 추출
#   - 컬럼 인덱스일 경우 인덱스의 리스트 사용 가능
#     - 리스트를 전달할 경우 결과는 Dataframe
#     - 하나의 컬럼명을 전달할 경우 결과는 Series

df = load_titanic()
df.head()
# >>> 출력:
#    PassengerId  Survived  Pclass  \
# 0            1         0       3   
# 1            2         1       1   
# 2            3         1       3   
# 3            4         1       1   
# 4            5         0       3   
# 
#                                                 Name     Sex   Age  SibSp  \
# 0                            Braund, Mr. Owen Harris    male  22.0      1   
# 1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 2                             Heikkinen, Miss. Laina  female  26.0      0   
# 3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 4                           Allen, Mr. William Henry    male  35.0      0   
# 
#    Parch            Ticket     Fare Cabin Embarked  

# ## 단일 컬럼 선택

# df[0]  # 에러!  index 0 를 선택 하는게 아니다.

df['Survived']  # Series
# >>> 출력:
# 0      0
# 1      1
# 2      1
# 3      1
# 4      0
#       ..
# 886    0
# 887    1
# 888    0
# 889    1
# 890    0
# Name: Survived, Length: 891, dtype: int64

df.Survived
# >>> 출력:
# 0      0
# 1      1
# 2      1
# 3      1
# 4      0
#       ..
# 886    0
# 887    1
# 888    0
# 889    1
# 890    0
# Name: Survived, Length: 891, dtype: int64

df.Age
# >>> 출력:
# 0      22.0
# 1      38.0
# 2      26.0
# 3      35.0
# 4      35.0
#        ... 
# 886    27.0
# 887    19.0
# 888     NaN
# 889    26.0
# 890    32.0
# Name: Age, Length: 891, dtype: float64

df.Age
# >>> 출력:
# 0      22.0
# 1      38.0
# 2      26.0
# 3      35.0
# 4      35.0
#        ... 
# 886    27.0
# 887    19.0
# 888     NaN
# 889    26.0
# 890    32.0
# Name: Age, Length: 891, dtype: float64

# ## 복수의 컬럼 선택하기

df['Survived']  # 결과는 Seriese 다
# >>> 출력:
# 0      0
# 1      1
# 2      1
# 3      1
# 4      0
#       ..
# 886    0
# 887    1
# 888    0
# 889    1
# 890    0
# Name: Survived, Length: 891, dtype: int64

df[['Survived']]   # 결과는 DataFrame 이다!
# >>> 출력:
#      Survived
# 0           0
# 1           1
# 2           1
# 3           1
# 4           0
# ..        ...
# 886         0
# 887         1
# 888         0
# 889         1
# 890         0
# 
# [891 rows x 1 columns]

df[['Survived', 'Age', 'Name']]
# >>> 출력:
#      Survived   Age                                               Name
# 0           0  22.0                            Braund, Mr. Owen Harris
# 1           1  38.0  Cumings, Mrs. John Bradley (Florence Briggs Th...
# 2           1  26.0                             Heikkinen, Miss. Laina
# 3           1  35.0       Futrelle, Mrs. Jacques Heath (Lily May Peel)
# 4           0  35.0                           Allen, Mr. William Henry
# ..        ...   ...                                                ...
# 886         0  27.0                              Montvila, Rev. Juozas
# 887         1  19.0                       Graham, Miss. Margaret Edith
# 888         0   NaN           Johnston, Miss. Catherine Helen "Carrie"
# 889         1  26.0                              Behr, Mr. Karl Howell
# 890         0  32.0                                Dooley, Mr. Patrick
# 
# [891 rows x 3 columns]

df[['Survived', 'Age', 'Name', 'Survived']]
# >>> 출력:
#      Survived   Age                                               Name  \
# 0           0  22.0                            Braund, Mr. Owen Harris   
# 1           1  38.0  Cumings, Mrs. John Bradley (Florence Briggs Th...   
# 2           1  26.0                             Heikkinen, Miss. Laina   
# 3           1  35.0       Futrelle, Mrs. Jacques Heath (Lily May Peel)   
# 4           0  35.0                           Allen, Mr. William Henry   
# ..        ...   ...                                                ...   
# 886         0  27.0                              Montvila, Rev. Juozas   
# 887         1  19.0                       Graham, Miss. Margaret Edith   
# 888         0   NaN           Johnston, Miss. Catherine Helen "Carrie"   
# 889         1  26.0                              Behr, Mr. Karl Howell   
# 890         0  32.0                                Dooley, Mr. Patrick   
# 
#      Survived  
# 0           0  

# # row 선택하기

# ## DataFrame slicing
#   - dataframe의 경우 기본적으로 [] 연산는 **column 선택**에 사용  `(인덱싱 용이 아니다!!)`
#   - 하지만, slicing은 row 레벨로 지원

df.head(15)
# >>> 출력:
#     PassengerId  Survived  Pclass  \
# 0             1         0       3   
# 1             2         1       1   
# 2             3         1       3   
# 3             4         1       1   
# 4             5         0       3   
# 5             6         0       3   
# 6             7         0       1   
# 7             8         0       3   
# 8             9         1       3   
# 9            10         1       2   
# 10           11         1       3   
# 11           12         1       1   
# 12           13         0       3   
# 13           14         0       3   

df[:10]  # 10개의 row 선택
# >>> 출력:
#    PassengerId  Survived  Pclass  \
# 0            1         0       3   
# 1            2         1       1   
# 2            3         1       3   
# 3            4         1       1   
# 4            5         0       3   
# 5            6         0       3   
# 6            7         0       1   
# 7            8         0       3   
# 8            9         1       3   
# 9           10         1       2   
# 
#                                                 Name     Sex   Age  SibSp  \
# 0                            Braund, Mr. Owen Harris    male  22.0      1   
# 1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   

df[7:10]
# >>> 출력:
#    PassengerId  Survived  Pclass  \
# 7            8         0       3   
# 8            9         1       3   
# 9           10         1       2   
# 
#                                                 Name     Sex   Age  SibSp  \
# 7                     Palsson, Master. Gosta Leonard    male   2.0      3   
# 8  Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)  female  27.0      0   
# 9                Nasser, Mrs. Nicholas (Adele Achem)  female  14.0      1   
# 
#    Parch  Ticket     Fare Cabin Embarked  
# 7      1  349909  21.0750   NaN        S  
# 8      2  347742  11.1333   NaN        S  
# 9      0  237736  30.0708   NaN        C

df[::10]
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 0              1         0       3   
# 10            11         1       3   
# 20            21         0       2   
# 30            31         0       1   
# 40            41         0       3   
# ..           ...       ...     ...   
# 850          851         0       3   
# 860          861         0       3   
# 870          871         0       3   
# 880          881         1       2   
# 890          891         0       3   
# 
#                                                Name     Sex   Age  SibSp  \
# 0                           Braund, Mr. Owen Harris    male  22.0      1   

# ## loc, iloc
#   - Series의 경우 []로 row 선택이 가능하나, **DataFrame의 경우는 기본적으로 column을 선택하도록 설계**
#   - **.loc[], .iloc[]**로 row 선택 가능
#     - loc - 인덱스 자체를 사용
#     - iloc - 0 based index로 사용
#     - 위 두가지는 ,를 사용하여 column 선택도 가능

df.head()
# >>> 출력:
#    PassengerId  Survived  Pclass  \
# 0            1         0       3   
# 1            2         1       1   
# 2            3         1       3   
# 3            4         1       1   
# 4            5         0       3   
# 
#                                                 Name     Sex   Age  SibSp  \
# 0                            Braund, Mr. Owen Harris    male  22.0      1   
# 1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 2                             Heikkinen, Miss. Laina  female  26.0      0   
# 3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 4                           Allen, Mr. William Henry    male  35.0      0   
# 
#    Parch            Ticket     Fare Cabin Embarked  

# index 변경
df.index += 100

df.head()
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 100            1         0       3   
# 101            2         1       1   
# 102            3         1       3   
# 103            4         1       1   
# 104            5         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   
# 101  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 102                             Heikkinen, Miss. Laina  female  26.0      0   
# 103       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 104                           Allen, Mr. William Henry    male  35.0      0   
# 
#      Parch            Ticket     Fare Cabin Embarked  

df
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 100            1         0       3   
# 101            2         1       1   
# 102            3         1       3   
# 103            4         1       1   
# 104            5         0       3   
# ..           ...       ...     ...   
# 986          887         0       2   
# 987          888         1       1   
# 988          889         0       3   
# 989          890         1       1   
# 990          891         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   

df.loc[986]  # Series
# >>> 출력:
# PassengerId                      887
# Survived                           0
# Pclass                             2
# Name           Montvila, Rev. Juozas
# Sex                             male
# Age                             27.0
# SibSp                              0
# Parch                              0
# Ticket                        211536
# Fare                            13.0
# Cabin                            NaN
# Embarked                           S
# Name: 986, dtype: object

df.loc[[986]] # DataFrame
# >>> 출력:
#      PassengerId  Survived  Pclass                   Name   Sex   Age  SibSp  \
# 986          887         0       2  Montvila, Rev. Juozas  male  27.0      0   
# 
#      Parch  Ticket  Fare Cabin Embarked  
# 986      0  211536  13.0   NaN        S

df.loc[[986, 100, 110, 990]]
# >>> 출력:
#      PassengerId  Survived  Pclass                             Name     Sex  \
# 986          887         0       2            Montvila, Rev. Juozas    male   
# 100            1         0       3          Braund, Mr. Owen Harris    male   
# 110           11         1       3  Sandstrom, Miss. Marguerite Rut  female   
# 990          891         0       3              Dooley, Mr. Patrick    male   
# 
#       Age  SibSp  Parch     Ticket   Fare Cabin Embarked  
# 986  27.0      0      0     211536  13.00   NaN        S  
# 100  22.0      1      0  A/5 21171   7.25   NaN        S  
# 110   4.0      1      1    PP 9549  16.70    G6        S  
# 990  32.0      0      0     370376   7.75   NaN        Q

df.loc[np.arange(100, 105)]
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 100            1         0       3   
# 101            2         1       1   
# 102            3         1       3   
# 103            4         1       1   
# 104            5         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   
# 101  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 102                             Heikkinen, Miss. Laina  female  26.0      0   
# 103       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 104                           Allen, Mr. William Henry    male  35.0      0   
# 
#      Parch            Ticket     Fare Cabin Embarked  

df[100:105]  # slicing 과 loc[] 차이!
# >>> 출력:
#      PassengerId  Survived  Pclass                              Name     Sex  \
# 200          101         0       3           Petranec, Miss. Matilda  female   
# 201          102         0       3  Petroff, Mr. Pastcho ("Pentcho")    male   
# 202          103         0       1         White, Mr. Richard Frasar    male   
# 203          104         0       3        Johansson, Mr. Gustaf Joel    male   
# 204          105         0       3    Gustafsson, Mr. Anders Vilhelm    male   
# 
#       Age  SibSp  Parch   Ticket     Fare Cabin Embarked  
# 200  28.0      0      0   349245   7.8958   NaN        S  
# 201   NaN      0      0   349215   7.8958   NaN        S  
# 202  21.0      0      1    35281  77.2875   D26        S  
# 203  33.0      0      0     7540   8.6542   NaN        S  
# 204  37.0      2      0  3101276   7.9250   NaN        S

# iloc
df.iloc[0]  # 0번 index 가 아니라!  0번째 row.  Series
# >>> 출력:
# PassengerId                          1
# Survived                             0
# Pclass                               3
# Name           Braund, Mr. Owen Harris
# Sex                               male
# Age                               22.0
# SibSp                                1
# Parch                                0
# Ticket                       A/5 21171
# Fare                              7.25
# Cabin                              NaN
# Embarked                             S
# Name: 100, dtype: object

df.iloc[[0]]  # DataFrame
# >>> 출력:
#      PassengerId  Survived  Pclass                     Name   Sex   Age  \
# 100            1         0       3  Braund, Mr. Owen Harris  male  22.0   
# 
#      SibSp  Parch     Ticket  Fare Cabin Embarked  
# 100      1      0  A/5 21171  7.25   NaN        S

df.iloc[[0, 100, 200, 2]]
# >>> 출력:
#      PassengerId  Survived  Pclass                            Name     Sex  \
# 100            1         0       3         Braund, Mr. Owen Harris    male   
# 200          101         0       3         Petranec, Miss. Matilda  female   
# 300          201         0       3  Vande Walle, Mr. Nestor Cyriel    male   
# 102            3         1       3          Heikkinen, Miss. Laina  female   
# 
#       Age  SibSp  Parch            Ticket    Fare Cabin Embarked  
# 100  22.0      1      0         A/5 21171  7.2500   NaN        S  
# 200  28.0      0      0            349245  7.8958   NaN        S  
# 300  28.0      0      0            345770  9.5000   NaN        S  
# 102  26.0      0      0  STON/O2. 3101282  7.9250   NaN        S

df.head(3)
df[:3]
df.iloc[:3]
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 100            1         0       3   
# 101            2         1       1   
# 102            3         1       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   
# 101  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 102                             Heikkinen, Miss. Laina  female  26.0      0   
# 
#      Parch            Ticket     Fare Cabin Embarked  
# 100      0         A/5 21171   7.2500   NaN        S  
# 101      0          PC 17599  71.2833   C85        C  
# 102      0  STON/O2. 3101282   7.9250   NaN        S

# ## row, column 동시에 선택하기
# loc[], iloc[] 속성을 이용할때  , (콤마)를 사용하여 row, column 둘다 명시 가능

df.tail()
# >>> 출력:
#      PassengerId  Survived  Pclass                                      Name  \
# 986          887         0       2                     Montvila, Rev. Juozas   
# 987          888         1       1              Graham, Miss. Margaret Edith   
# 988          889         0       3  Johnston, Miss. Catherine Helen "Carrie"   
# 989          890         1       1                     Behr, Mr. Karl Howell   
# 990          891         0       3                       Dooley, Mr. Patrick   
# 
#         Sex   Age  SibSp  Parch      Ticket   Fare Cabin Embarked  
# 986    male  27.0      0      0      211536  13.00   NaN        S  
# 987  female  19.0      0      0      112053  30.00   B42        S  
# 988  female   NaN      1      2  W./C. 6607  23.45   NaN        S  
# 989    male  26.0      0      0      111369  30.00  C148        C  
# 990    male  32.0      0      0      370376   7.75   NaN        Q

df.loc[986, 'Survived']  # loc[row, column]  <- loc[axis0, axis1]
# >>> 출력:
# 0

df.iloc[-5, -3]  # iloc[row, column]
# >>> 출력:
# 13.0

df.head()
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 100            1         0       3   
# 101            2         1       1   
# 102            3         1       3   
# 103            4         1       1   
# 104            5         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   
# 101  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 102                             Heikkinen, Miss. Laina  female  26.0      0   
# 103       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 104                           Allen, Mr. William Henry    male  35.0      0   
# 
#      Parch            Ticket     Fare Cabin Embarked  

df.loc[102, 'Name']
# >>> 출력:
# 'Heikkinen, Miss. Laina'

df.iloc[2, 3]
# >>> 출력:
# 'Heikkinen, Miss. Laina'

df['Name'][102]
# >>> 출력:
# 'Heikkinen, Miss. Laina'

df.loc[102]['Name']
# >>> 출력:
# 'Heikkinen, Miss. Laina'

df.loc[[986, 100, 110, 990], ['Survived', 'Name', 'Sex', 'Age']]
# >>> 출력:
#      Survived                             Name     Sex   Age
# 986         0            Montvila, Rev. Juozas    male  27.0
# 100         0          Braund, Mr. Owen Harris    male  22.0
# 110         1  Sandstrom, Miss. Marguerite Rut  female   4.0
# 990         0              Dooley, Mr. Patrick    male  32.0

df.loc[[986, 100, 110, 990]][['Survived', 'Name', 'Sex', 'Age']]
# >>> 출력:
#      Survived                             Name     Sex   Age
# 986         0            Montvila, Rev. Juozas    male  27.0
# 100         0          Braund, Mr. Owen Harris    male  22.0
# 110         1  Sandstrom, Miss. Marguerite Rut  female   4.0
# 990         0              Dooley, Mr. Patrick    male  32.0

df[['Survived', 'Name', 'Sex', 'Age']].loc[[986, 100, 110, 990]]
# >>> 출력:
#      Survived                             Name     Sex   Age
# 986         0            Montvila, Rev. Juozas    male  27.0
# 100         0          Braund, Mr. Owen Harris    male  22.0
# 110         1  Sandstrom, Miss. Marguerite Rut  female   4.0
# 990         0              Dooley, Mr. Patrick    male  32.0

# # boolean selection으로 row 선택하기
#  - numpy에서와 동일한 방식으로 해당 조건에 맞는 row만 선택

# 나이 30대 이면서 1등객식에 탑승한 승객들 선택

df.head()
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 100            1         0       3   
# 101            2         1       1   
# 102            3         1       3   
# 103            4         1       1   
# 104            5         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   
# 101  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 102                             Heikkinen, Miss. Laina  female  26.0      0   
# 103       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 104                           Allen, Mr. William Henry    male  35.0      0   
# 
#      Parch            Ticket     Fare Cabin Embarked  

df.Pclass
# >>> 출력:
# 100    3
# 101    1
# 102    3
# 103    1
# 104    3
#       ..
# 986    2
# 987    1
# 988    3
# 989    1
# 990    3
# Name: Pclass, Length: 891, dtype: int64

df.Pclass.unique()
# >>> 출력:
# array([3, 1, 2])

df.Pclass.value_counts()
# >>> 출력:
# Pclass
# 3    491
# 1    216
# 2    184
# Name: count, dtype: int64

pclass_mask = df['Pclass'] == 1
pclass_mask
# >>> 출력:
# 100    False
# 101     True
# 102    False
# 103     True
# 104    False
#        ...  
# 986    False
# 987     True
# 988    False
# 989     True
# 990    False
# Name: Pclass, Length: 891, dtype: bool

df[pclass_mask]  # boolean selection!
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 101            2         1       1   
# 103            4         1       1   
# 106            7         0       1   
# 111           12         1       1   
# 123           24         1       1   
# ..           ...       ...     ...   
# 971          872         1       1   
# 972          873         0       1   
# 979          880         1       1   
# 987          888         1       1   
# 989          890         1       1   
# 
#                                                   Name     Sex   Age  SibSp  \
# 101  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   

# 30대 승객에 대한 boolean mask
age_mask = (df.Age >= 30) & (df.Age < 40)
age_mask
# >>> 출력:
# 100    False
# 101     True
# 102    False
# 103     True
# 104     True
#        ...  
# 986    False
# 987    False
# 988    False
# 989    False
# 990     True
# Name: Age, Length: 891, dtype: bool

df[age_mask]
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 101            2         1       1   
# 103            4         1       1   
# 104            5         0       3   
# 113           14         0       3   
# 118           19         0       3   
# ..           ...       ...     ...   
# 967          868         0       1   
# 972          873         0       1   
# 981          882         0       3   
# 985          886         0       3   
# 990          891         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 101  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   

df[pclass_mask & age_mask]
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 101            2         1       1   
# 103            4         1       1   
# 161           62         1       1   
# 237          138         0       1   
# 315          216         1       1   
# 318          219         1       1   
# 324          225         1       1   
# 330          231         1       1   
# 348          249         1       1   
# 357          258         1       1   
# 358          259         1       1   
# 369          270         1       1   
# 373          274         0       1   
# 409          310         1       1   

df[pclass_mask & age_mask][['Pclass', 'Age']]
# >>> 출력:
#      Pclass   Age
# 101       1  38.0
# 103       1  35.0
# 161       1  38.0
# 237       1  37.0
# 315       1  31.0
# 318       1  32.0
# 324       1  38.0
# 330       1  35.0
# 348       1  37.0
# 357       1  30.0
# 358       1  35.0
# 369       1  35.0
# 373       1  37.0
# 409       1  30.0

# ### 도전] 3등급 객실의 생존자 수는?
# 
# > 결과 119명

pc3sur_mask = (df.Survived == 1) & (df.Pclass == 3)
pc3sur_mask
len(df[pc3sur_mask])
# >>> 출력:
# 119

Survived_mask = df.Survived == 1
df[Survived_mask & (df.Pclass == 3)]['PassengerId'].count()
# >>> 출력:
# 119

len(df[(df['Pclass'] == 3) & (df['Survived'] == 1)])
# >>> 출력:
# 119

# ### 도전] 남성과 여성의 생존률은 ?
# 
# > [결과 예시]<br>
# > 남성생존률 0.18890814558058924<br>
# > 여성생존률 0.7420382165605095

msur_mask = (df.Survived == 1) & (df.Sex == 'male')
fsur_mask = (df.Survived == 1) & (df.Sex == 'female')
msur_ratio = len(df[msur_mask])/len(df[df.Sex=='male'])
fsur_ratio = len(df[fsur_mask])/len(df[df.Sex=='female'])
print('남성생존률', msur_ratio)
print('여성생존률', fsur_ratio)
# >>> 출력:
# 남성생존률 0.18890814558058924
# 여성생존률 0.7420382165605095

male_mask = df.Sex == 'male'
male_mask
female_mask = df.Sex == 'female'
female_mask
survive_mask = df.Survived == 1
survive_mask
print('남성생존률 {}'.format(len(df[male_mask & survive_mask]) / len(df[male_mask])))
print('여성생존률 {}'.format(len(df[female_mask & survive_mask]) / len(df[female_mask])))
# >>> 출력:
# 남성생존률 0.18890814558058924
# 여성생존률 0.7420382165605095

male_mask = df.Sex == 'male'
female_mask = df.Sex == 'female'
Survived_mask = df.Survived == 1
df[Survived_mask & male_mask]['PassengerId'].count()/df[male_mask]['PassengerId'].count()
df[Survived_mask & female_mask]['PassengerId'].count()/df[female_mask]['PassengerId'].count()
# >>> 출력:
# 0.7420382165605095

male_survive_rate = len(df[(df["Sex"]=="male")&(df["Survived"]==1)])/len(df[df["Sex"]=="male"])
female_survive_rate = len(df[(df["Sex"]=="female")&(df["Survived"]==1)])/len(df[df["Sex"]=="female"])
print(f"남성생존률 {male_survive_rate}")
print(f"여성생존률 {female_survive_rate}")
# >>> 출력:
# 남성생존률 0.18890814558058924
# 여성생존률 0.7420382165605095

male_survived_mask = (df.Sex == 'male') & (df.Survived == 1)
male_mask = df.Sex == 'male'
male_sur_rate = len(df[male_survived_mask])/len(df[male_mask])
male_sur_rate
# >>> 출력:
# 0.18890814558058924

male_survived = df[df["Sex"] == "male"]["Survived"].mean()
print("남성 생존률", male_survived)
female_survived = df[df["Sex"] == "female"]["Survived"].mean()
print("여성 생존률", female_survived)
# >>> 출력:
# 남성 생존률 0.18890814558058924
# 여성 생존률 0.7420382165605095

# # row, column 추가, 삭제

# ## column 추가하기
# - [] 사용하여 추가하기
# - insert() 사용하여 원하는 위치에 추가

df['Name']
# >>> 출력:
# 100                              Braund, Mr. Owen Harris
# 101    Cumings, Mrs. John Bradley (Florence Briggs Th...
# 102                               Heikkinen, Miss. Laina
# 103         Futrelle, Mrs. Jacques Heath (Lily May Peel)
# 104                             Allen, Mr. William Henry
#                              ...                        
# 986                                Montvila, Rev. Juozas
# 987                         Graham, Miss. Margaret Edith
# 988             Johnston, Miss. Catherine Helen "Carrie"
# 989                                Behr, Mr. Karl Howell
# 990                                  Dooley, Mr. Patrick
# Name: Name, Length: 891, dtype: object

# df['Age_Double']  # KeyError

df['Age'] * 2
# >>> 출력:
# 100    44.0
# 101    76.0
# 102    52.0
# 103    70.0
# 104    70.0
#        ... 
# 986    54.0
# 987    38.0
# 988     NaN
# 989    52.0
# 990    64.0
# Name: Age, Length: 891, dtype: float64

# 위 결과를 df 의 새로운 컬럼으로 추가
df['Age_Double'] = df['Age'] * 2

df
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 100            1         0       3   
# 101            2         1       1   
# 102            3         1       3   
# 103            4         1       1   
# 104            5         0       3   
# ..           ...       ...     ...   
# 986          887         0       2   
# 987          888         1       1   
# 988          889         0       3   
# 989          890         1       1   
# 990          891         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   

df[['Age', 'Age_Double']].head()
# >>> 출력:
#       Age  Age_Double
# 100  22.0        44.0
# 101  38.0        76.0
# 102  26.0        52.0
# 103  35.0        70.0
# 104  35.0        70.0

# **파생변수**
# 
# 기존에 존재하는 속성(변수, 컬럼) 으로부터 새로운 속성(컬럼) 을 만들어 낸것.

df['Age_tripple'] = df['Age_Double'] + df['Age']

df
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 100            1         0       3   
# 101            2         1       1   
# 102            3         1       3   
# 103            4         1       1   
# 104            5         0       3   
# ..           ...       ...     ...   
# 986          887         0       2   
# 987          888         1       1   
# 988          889         0       3   
# 989          890         1       1   
# 990          891         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   

df[['Age', 'Age_Double', 'Age_tripple']].head()
# >>> 출력:
#       Age  Age_Double  Age_tripple
# 100  22.0        44.0         66.0
# 101  38.0        76.0        114.0
# 102  26.0        52.0         78.0
# 103  35.0        70.0        105.0
# 104  35.0        70.0        105.0

# ### insert()
# 원하는 위치에 컬럼 추가

df.head()
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 100            1         0       3   
# 101            2         1       1   
# 102            3         1       3   
# 103            4         1       1   
# 104            5         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   
# 101  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 102                             Heikkinen, Miss. Laina  female  26.0      0   
# 103       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 104                           Allen, Mr. William Henry    male  35.0      0   
# 
#      Parch            Ticket     Fare Cabin Embarked  Age_Double  Age_tripple  

df['Fare'] / 10
# >>> 출력:
# 100    0.72500
# 101    7.12833
# 102    0.79250
# 103    5.31000
# 104    0.80500
#         ...   
# 986    1.30000
# 987    3.00000
# 988    2.34500
# 989    3.00000
# 990    0.77500
# Name: Fare, Length: 891, dtype: float64

df.insert(3, 'Fare10', df['Fare'] / 10)  # 컬럼 인덱스 3 위치에 'Fare10' 이라는 새로운 컬럼 삽입

df.head()
# >>> 출력:
#      PassengerId  Survived  Pclass   Fare10  \
# 100            1         0       3  0.72500   
# 101            2         1       1  7.12833   
# 102            3         1       3  0.79250   
# 103            4         1       1  5.31000   
# 104            5         0       3  0.80500   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   
# 101  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 102                             Heikkinen, Miss. Laina  female  26.0      0   
# 103       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 104                           Allen, Mr. William Henry    male  35.0      0   
# 
#      Parch            Ticket     Fare Cabin Embarked  Age_Double  Age_tripple  

# ## column 삭제하기
#  - drop 함수 사용하여 삭제
#    - axis= 사용
#    - 리스트를 사용하여 멀티플 삭제 가능

# DataFrame 은 2차원 데이터,  axis= 값은 0, 1 두가지가 존재 한다
# axis=0 : row level  (default)
# axis=1 : column level

# df.drop?

df.head()
# >>> 출력:
#      PassengerId  Survived  Pclass   Fare10  \
# 100            1         0       3  0.72500   
# 101            2         1       1  7.12833   
# 102            3         1       3  0.79250   
# 103            4         1       1  5.31000   
# 104            5         0       3  0.80500   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   
# 101  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 102                             Heikkinen, Miss. Laina  female  26.0      0   
# 103       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 104                           Allen, Mr. William Henry    male  35.0      0   
# 
#      Parch            Ticket     Fare Cabin Embarked  Age_Double  Age_tripple  

# df.drop('Age_tripple')   # 에러  axis=0 가 디폴트라서 index 값이 'Age_tripple' 인것을 삭제하려 한다

df.drop('Age_tripple', axis=1)  # 열 제거, 원본 변화 없다
# >>> 출력:
#      PassengerId  Survived  Pclass   Fare10  \
# 100            1         0       3  0.72500   
# 101            2         1       1  7.12833   
# 102            3         1       3  0.79250   
# 103            4         1       1  5.31000   
# 104            5         0       3  0.80500   
# ..           ...       ...     ...      ...   
# 986          887         0       2  1.30000   
# 987          888         1       1  3.00000   
# 988          889         0       3  2.34500   
# 989          890         1       1  3.00000   
# 990          891         0       3  0.77500   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   

# 여러 컬럼 삭제

df.drop(['Age_tripple', 'Age_Double'], axis=1)
# >>> 출력:
#      PassengerId  Survived  Pclass   Fare10  \
# 100            1         0       3  0.72500   
# 101            2         1       1  7.12833   
# 102            3         1       3  0.79250   
# 103            4         1       1  5.31000   
# 104            5         0       3  0.80500   
# ..           ...       ...     ...      ...   
# 986          887         0       2  1.30000   
# 987          888         1       1  3.00000   
# 988          889         0       3  2.34500   
# 989          890         1       1  3.00000   
# 990          891         0       3  0.77500   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   

# 원본변화
df.drop(['Age_tripple', 'Age_Double'], axis=1, inplace=True)

df.head()
# >>> 출력:
#      PassengerId  Survived  Pclass   Fare10  \
# 100            1         0       3  0.72500   
# 101            2         1       1  7.12833   
# 102            3         1       3  0.79250   
# 103            4         1       1  5.31000   
# 104            5         0       3  0.80500   
# 
#                                                   Name     Sex   Age  SibSp  \
# 100                            Braund, Mr. Owen Harris    male  22.0      1   
# 101  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 102                             Heikkinen, Miss. Laina  female  26.0      0   
# 103       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 104                           Allen, Mr. William Henry    male  35.0      0   
# 
#      Parch            Ticket     Fare Cabin Embarked  

# ## 행 (row) 추가 삭제
# - 추가 loc[]
# - 삭제 drop()

df2 = pd.DataFrame([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
], columns=['a','b','c','d'], index = ['김길동', '최진수', '허수민'])
df2
# >>> 출력:
#      a   b   c   d
# 김길동  1   2   3   4
# 최진수  5   6   7   8
# 허수민  9  10  11  12

# 새로운 row 추가
df2.loc['김갑수'] = pd.Series([100, 200, 300, 400], index=['a', 'b', 'c', 'd'])

df2
# >>> 출력:
#        a    b    c    d
# 김길동    1    2    3    4
# 최진수    5    6    7    8
# 허수민    9   10   11   12
# 김갑수  100  200  300  400

df2.loc['김철수'] = [101, 202, 303, 404]

df2
# >>> 출력:
#        a    b    c    d
# 김길동    1    2    3    4
# 최진수    5    6    7    8
# 허수민    9   10   11   12
# 김갑수  100  200  300  400
# 김철수  101  202  303  404

# 행 삭제
# drop() 사용,   axis=0 (디폴트)
df2.drop(['최진수', '허수민'])
# >>> 출력:
#        a    b    c    d
# 김길동    1    2    3    4
# 김갑수  100  200  300  400
# 김철수  101  202  303  404

# 행 (값 변경)
df2.iloc[1] = [111, 222, 333, 444]

df2
# >>> 출력:
#        a    b    c    d
# 김길동    1    2    3    4
# 최진수  111  222  333  444
# 허수민    9   10   11   12
# 김갑수  100  200  300  400
# 김철수  101  202  303  404

# ## 행 삽입

# df2.insert?  # <-- insert() 는 columns 만 삽입하는 용도

# 방법: concat + iloc[:] 사용

df2.iloc[:2]
# >>> 출력:
#        a    b    c    d
# 김길동    1    2    3    4
# 최진수  111  222  333  444

df2.iloc[2:]
# >>> 출력:
#        a    b    c    d
# 허수민    9   10   11   12
# 김갑수  100  200  300  400
# 김철수  101  202  303  404

pd.concat([
    df2.iloc[:2],
    pd.DataFrame({'a':10, 'b':20, 'c':30, 'd':40}, index=['정태경']),
    df2.iloc[2:],
])
# >>> 출력:
#        a    b    c    d
# 김길동    1    2    3    4
# 최진수  111  222  333  444
# 정태경   10   20   30   40
# 허수민    9   10   11   12
# 김갑수  100  200  300  400
# 김철수  101  202  303  404

# 방법2 : transpose + insert()

df_t = df2.T

df_t.insert(2, '정태경', [10, 20, 30, 40])

df_t.T
# >>> 출력:
#        a    b    c    d
# 김길동    1    2    3    4
# 최진수  111  222  333  444
# 정태경   10   20   30   40
# 허수민    9   10   11   12
# 김갑수  100  200  300  400
# 김철수  101  202  303  404

# # Correlation : 상관관계

# ※ 컬럼, 속성, 변수, 특징(feature) <- 다 똑같은 말.

# 상관관계 (correlation)
# 특정한 '값' 과의 상관관계,  두 변수간의 흐름이 얼마나 비슷한 가를 나타내는 척도
#                           두 변수간의 흐름이 어떤 선형적(linear) 한 관계를 나타내는 척도

# 가령
#   a 값이 증가할때 b 값도 증가하나?(혹은 감소하나?).
#  증감의 성향은 어떠한가?  <-- 데이터의 분포, 추세를 보는데 중요

# -1 ~ +1 사이로 표현한다
# +1 에 가까울수록 a, b  둘이 비슷하게 증감 한다.
# -1 에 가까울수록 a, b 증감은 반대로 진행 한다.
# 절대값이 클수록 관계가 크다 라고 볼수 있다.

# 그러나
# ★ +1 에 가깝다고 해서 이것이 '인과관계' 를 나타내는 것은 아니다. (일수도, 아닐수도 있는것.)

# ## 변수(column) 사이의 상관계수(correlation)
#  - corr() 함수를 통해 상관계수 연산 (-1, 1 사이의 결과)
#    - 연속성(숫자형)데이터에 대해서만 연산
#    - 인과관계를 의미하진 않음

df = load_titanic()

df.corr(numeric_only=True)
# >>> 출력:
#              PassengerId  Survived    Pclass       Age     SibSp     Parch  \
# PassengerId     1.000000 -0.005007 -0.035144  0.036847 -0.057527 -0.001652   
# Survived       -0.005007  1.000000 -0.338481 -0.077221 -0.035322  0.081629   
# Pclass         -0.035144 -0.338481  1.000000 -0.369226  0.083081  0.018443   
# Age             0.036847 -0.077221 -0.369226  1.000000 -0.308247 -0.189119   
# SibSp          -0.057527 -0.035322  0.083081 -0.308247  1.000000  0.414838   
# Parch          -0.001652  0.081629  0.018443 -0.189119  0.414838  1.000000   
# Fare            0.012658  0.257307 -0.549500  0.096067  0.159651  0.216225   
# 
#                  Fare  
# PassengerId  0.012658  
# Survived     0.257307  
# Pclass      -0.549500  
# Age          0.096067  
# SibSp        0.159651  

# 절대값이 높다 -> 상관관계가 높다.
# 어떤 컬럼들이 상관관계가 높나?

# Fare - Pclass
# Survived - Pclass
# Survived - Fare
# Age - Pclass
# parch-SibSp
# SibSp-Age

import matplotlib.pyplot as plt

plt.matshow(df.corr(numeric_only=True))
plt.colorbar()
plt.show()

# # NaN 처리

# NaN : Not a Number  <-- '결측값(missing value)' 이라고도 한다
#    (NA: Not Available 라고도 표기하며  np.NaN, None 등이 해당한다)
# 여러가지 이유에 의해 데이터 안에 결측값이 존재할수 있다. (누락, 소실 등...)

# 값이 0 이라는 뜻이 아니다!  값이 '없다'는 뜻이다.

# 데이터 분석 이전에 반드시 NaN 값 존재 여부도 확인해야 한다

# 데이터 분석시 '전처리' 단계에서 NaN 값에 대한 적절한 전처리를 해주어야.
# 정확한 분석과 인공지능 학습을 기대할수 있다.

# 일반적으로 두가지 방법 사용하여 NaN 처리
#  1. 결측값은 무시하든지
#  2. 다른값으로 채우든지

# ## NaN 값 확인
#  - describe, info함수를 통하여 개수 확인
#  - isna함수를 통해 boolean 타입으로 확인

df.info()
# ↓ Age, Cabin, Embarked 는 결측치가 있슴을 확인할수 있다.
# >>> 출력:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   PassengerId  891 non-null    int64  
#  1   Survived     891 non-null    int64  
#  2   Pclass       891 non-null    int64  
#  3   Name         891 non-null    object 
#  4   Sex          891 non-null    object 
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64  
#  7   Parch        891 non-null    int64  
#  8   Ticket       891 non-null    object 
#  9   Fare         891 non-null    float64

df.isna()

# 결측치는 True 로 표시된다.
# >>> 출력:
#      PassengerId  Survived  Pclass   Name    Sex    Age  SibSp  Parch  Ticket  \
# 0          False     False   False  False  False  False  False  False   False   
# 1          False     False   False  False  False  False  False  False   False   
# 2          False     False   False  False  False  False  False  False   False   
# 3          False     False   False  False  False  False  False  False   False   
# 4          False     False   False  False  False  False  False  False   False   
# ..           ...       ...     ...    ...    ...    ...    ...    ...     ...   
# 886        False     False   False  False  False  False  False  False   False   
# 887        False     False   False  False  False  False  False  False   False   
# 888        False     False   False  False  False   True  False  False   False   
# 889        False     False   False  False  False  False  False  False   False   
# 890        False     False   False  False  False  False  False  False   False   
# 
#       Fare  Cabin  Embarked  
# 0    False   True     False  

# 특정 컬럼에 대해서 확인
df['Age'].isna()
# >>> 출력:
# 0      False
# 1      False
# 2      False
# 3      False
# 4      False
#        ...  
# 886    False
# 887    False
# 888     True
# 889    False
# 890    False
# Name: Age, Length: 891, dtype: bool

# 결측된 데이터만 확인 <- boolean selection  활용
df[df['Age'].isna()]
# >>> 출력:
#      PassengerId  Survived  Pclass                                      Name  \
# 5              6         0       3                          Moran, Mr. James   
# 17            18         1       2              Williams, Mr. Charles Eugene   
# 19            20         1       3                   Masselmani, Mrs. Fatima   
# 26            27         0       3                   Emir, Mr. Farred Chehab   
# 28            29         1       3             O'Dwyer, Miss. Ellen "Nellie"   
# ..           ...       ...     ...                                       ...   
# 859          860         0       3                          Razi, Mr. Raihed   
# 863          864         0       3         Sage, Miss. Dorothy Edith "Dolly"   
# 868          869         0       3               van Melkebeke, Mr. Philemon   
# 878          879         0       3                        Laleff, Mr. Kristo   
# 888          889         0       3  Johnston, Miss. Catherine Helen "Carrie"   
# 
#         Sex  Age  SibSp  Parch      Ticket     Fare Cabin Embarked  
# 5      male  NaN      0      0      330877   8.4583   NaN        Q  

# ## NaN 삭제
#  - 데이터에서 삭제
#    - dropna 함수

df.shape
# >>> 출력:
# (891, 12)

df.dropna()
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 1              2         1       1   
# 3              4         1       1   
# 6              7         0       1   
# 10            11         1       3   
# 11            12         1       1   
# ..           ...       ...     ...   
# 871          872         1       1   
# 872          873         0       1   
# 879          880         1       1   
# 887          888         1       1   
# 889          890         1       1   
# 
#                                                   Name     Sex   Age  SibSp  \
# 1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   

# ↑ 183개의 row 가 나온다 (891개에서 확 줄었다)
# index 나 PassengerId 를 보면 중간중간에 빠진거 확인


# dropna() 기본적으로 row 기준(axis=0)으로 동작한다
# 각 row 에서, 하나라도 NaN 값이 있으면 해당 row 를 지워버린다

# 특정 컬럼에 대해서만 dropna() 하기
df.dropna(subset=['Age'])  # Age 컬럼이 NaN 인 경우만 row 가 drop 됨 (axis=0)
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 0              1         0       3   
# 1              2         1       1   
# 2              3         1       3   
# 3              4         1       1   
# 4              5         0       3   
# ..           ...       ...     ...   
# 885          886         0       3   
# 886          887         0       2   
# 887          888         1       1   
# 889          890         1       1   
# 890          891         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 0                              Braund, Mr. Owen Harris    male  22.0      1   

df.dropna(subset=['Age', 'Cabin'])
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 1              2         1       1   
# 3              4         1       1   
# 6              7         0       1   
# 10            11         1       3   
# 11            12         1       1   
# ..           ...       ...     ...   
# 871          872         1       1   
# 872          873         0       1   
# 879          880         1       1   
# 887          888         1       1   
# 889          890         1       1   
# 
#                                                   Name     Sex   Age  SibSp  \
# 1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   

# NaN 이 있는 '컬럼' 삭제하기
df.dropna(axis=1)

# Age, Cabin, Embarked 컬럼 삭제!
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 0              1         0       3   
# 1              2         1       1   
# 2              3         1       3   
# 3              4         1       1   
# 4              5         0       3   
# ..           ...       ...     ...   
# 886          887         0       2   
# 887          888         1       1   
# 888          889         0       3   
# 889          890         1       1   
# 890          891         0       3   
# 
#                                                   Name     Sex  SibSp  Parch  \
# 0                              Braund, Mr. Owen Harris    male      1      0   

# ## NaN 값 대체하기
#  - fillna()
#  - ex)평균으로 대체하기
#  - ex) 생존자/사망자 별 평균으로 대체하기

# Age 평균값
df['Age'].mean()  # NaN 이 배제된 연산결과다!  주의!
# >>> 출력:
# 29.69911764705882

df['Age'].count()
# >>> 출력:
# 714

# 위 mean() 값은 891명에 대한 평균값이 아니라 714 명에 대한 평균값이다
df['Age'].sum() / df['Age'].count()
# >>> 출력:
# 29.69911764705882

# NaN 을 평균값으로 대체
df['Age'].fillna(df['Age'].mean())
# >>> 출력:
# 0      22.000000
# 1      38.000000
# 2      26.000000
# 3      35.000000
# 4      35.000000
#          ...    
# 886    27.000000
# 887    19.000000
# 888    29.699118
# 889    26.000000
# 890    32.000000
# Name: Age, Length: 891, dtype: float64

df.tail()
# >>> 출력:
#      PassengerId  Survived  Pclass                                      Name  \
# 886          887         0       2                     Montvila, Rev. Juozas   
# 887          888         1       1              Graham, Miss. Margaret Edith   
# 888          889         0       3  Johnston, Miss. Catherine Helen "Carrie"   
# 889          890         1       1                     Behr, Mr. Karl Howell   
# 890          891         0       3                       Dooley, Mr. Patrick   
# 
#         Sex   Age  SibSp  Parch      Ticket   Fare Cabin Embarked  
# 886    male  27.0      0      0      211536  13.00   NaN        S  
# 887  female  19.0      0      0      112053  30.00   B42        S  
# 888  female   NaN      1      2  W./C. 6607  23.45   NaN        S  
# 889    male  26.0      0      0      111369  30.00  C148        C  
# 890    male  32.0      0      0      370376   7.75   NaN        Q

# # 수치형(numerical) & 범주형(categorical) 데이터의 이해
#  - dtype 에 대한 이야기가 아니다!

df.info()

# 숫자형은 int64.. float64..
# 문자형은 object...
# >>> 출력:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   PassengerId  891 non-null    int64  
#  1   Survived     891 non-null    int64  
#  2   Pclass       891 non-null    int64  
#  3   Name         891 non-null    object 
#  4   Sex          891 non-null    object 
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64  
#  7   Parch        891 non-null    int64  
#  8   Ticket       891 non-null    object 
#  9   Fare         891 non-null    float64

# ## 수치형(Numerical Type) 데이터
#  - 연속성을 띄는 숫자로 이루어진 데이터
#    - 예) Age, Fare 등
#  - 대소비교, 산술연산 (평균, 합계, 중간값) 의미있슴

# Age 를 생각해보자
#  대소비교 가능한가 : OK
#  평균값 (연산) 의미 있나 : ex) 탑승객의 평균 나이.  OK
#  나이 40 = 나이 20 x 2
#  연속성을 띄고 있나? : OK

# 따라서 '수치형' 으로 다루면 된다.

# Pclass 은 어떠한가 . 일단 이것도 dtype int64 타입이긴 한데.  그러면 수치형인가?

# 이들의 대소 관계 비교가 의미 있나?    1등급 < 3등급  ???
# 평균값은 의미 있나?   승객들의 객실등급 평균은 2.308641975308642  입니다.
#     3등급 객실 = 1등급 객실 x 3   ????  <-- 의미 없는 연산

# 연속성도 없다.  1.1등급? 1.7등급?

# 이는 단순히 1등급, 2등급, 3등급 이란 '범주(category)' 를 나타내는 (구별하기 위한) 데이터 이다.

# ## 범주형(Categorical Type) 데이터
#  - 연속적이지 않은 값(대부분의 경우 숫자를 제외한 나머지 값)을 갖는 데이터를 의미
#    - 예) Name, Sex, Ticket, Cabin, Embarked
#  - 어떤 경우, 데이터타입(dtype)이 숫자형 타입이라 할지라도 개념적으로 범주형으로 처리해야할 경우가 있음
#    - 예) Pclass, Survived   
#  - 대소비교, 산술연산(평균, 합계, 중간값...) 의미없슴

df['Pclass'].mean()
# >>> 출력:
# 2.308641975308642

# 대한민국 우편번호는 '숫자' 로 이루어져 있다
# 이는 '수치형?'   '범주형?'

# 역삼동 06240
# 제주시 애월읍 63131

# ↑ 범주형 데이터다!

# ## 숫자형 데이터를 범주형으로 변환
#  - Pclass 변수 변환
#  - astype 사용하여 간단히 타입만 변환

df.info()
# >>> 출력:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   PassengerId  891 non-null    int64  
#  1   Survived     891 non-null    int64  
#  2   Pclass       891 non-null    int64  
#  3   Name         891 non-null    object 
#  4   Sex          891 non-null    object 
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64  
#  7   Parch        891 non-null    int64  
#  8   Ticket       891 non-null    object 
#  9   Fare         891 non-null    float64

df['Pclass'] = df['Pclass'].astype(str)

df.info()
# >>> 출력:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   PassengerId  891 non-null    int64  
#  1   Survived     891 non-null    int64  
#  2   Pclass       891 non-null    object 
#  3   Name         891 non-null    object 
#  4   Sex          891 non-null    object 
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64  
#  7   Parch        891 non-null    int64  
#  8   Ticket       891 non-null    object 
#  9   Fare         891 non-null    float64

# ## Age 변수 변환
# - apply(func) 함수 활용

# 10대, 20대, 30대 ...   <---  수치형?  범주형(분류형)?

# 가령
# Age 값이 너무 세분화(?) 되어 있어 보인다.?
# 그래서 이렇게 해볼수 있습니다.

# 10대, 20대, 30대..   <-- 이런식으로 '분류(Categorical)' 하는 식으로 변환해볼수도 있을겁니다.

# 22.0 23.0 ,.. ==> '20대' 로 분류
# 30.0 31.0 32.0 ... ==> '30대' 로 분류

# 위와 같이 '숫자형' 데이터를 다시 '분류형Category' 으로 만들어 보고 싶을수도 있습니다.

# ### apply(func) 함수
# - 변환로직 func 을 적용

# 23 -> 20
# 43 -> 40
def age_categorize(age):
  return int(age / 10) * 10

age_categorize(23)
# >>> 출력:
# 20

age_categorize(43)
# >>> 출력:
# 40

# 특정 컬럼에 apply(함수)

# df['Age'].apply(age_categorize)

# ValueError: cannot convert float NaN to integer
# 그러나 Age 컬럼에 NaN 값이 있어서 에러 난다@!

import math

def age_categorize(age):
  if np.isnan(age):   # math.isnan()
    return -1   # Age 값이 없으면 -1 처리

  return int(age / 10) * 10

age_span = df['Age'].apply(age_categorize)
age_span
# >>> 출력:
# 0      20
# 1      30
# 2      20
# 3      30
# 4      30
#        ..
# 886    20
# 887    10
# 888    -1
# 889    20
# 890    30
# Name: Age, Length: 891, dtype: int64

age_span.unique()  # 분류형의 경우 확인해보자
# >>> 출력:
# array([20, 30, -1, 50,  0, 10, 40, 60, 70, 80])

age_span.value_counts()
# >>> 출력:
# Age
#  20    220
# -1     177
#  30    167
#  10    102
#  40     89
#  0      62
#  50     48
#  60     19
#  70      6
#  80      1
# Name: count, dtype: int64

# 위 범주형 데이터를 df 에 추가 (파생변수 추가)
df['Age_span'] = age_span
df
# >>> 출력:
#      PassengerId  Survived Pclass  \
# 0              1         0      3   
# 1              2         1      1   
# 2              3         1      3   
# 3              4         1      1   
# 4              5         0      3   
# ..           ...       ...    ...   
# 886          887         0      2   
# 887          888         1      1   
# 888          889         0      3   
# 889          890         1      1   
# 890          891         0      3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 0                              Braund, Mr. Owen Harris    male  22.0      1   

df[['Age', 'Age_span']]
# >>> 출력:
#       Age  Age_span
# 0    22.0        20
# 1    38.0        30
# 2    26.0        20
# 3    35.0        30
# 4    35.0        30
# ..    ...       ...
# 886  27.0        20
# 887  19.0        10
# 888   NaN        -1
# 889  26.0        20
# 890  32.0        30
# 
# [891 rows x 2 columns]

# ## One-hot encoding
# 범주형(categorical) 데이터 전처리

# "Red", "Green", "Blue"

# 범주형 데이터는 연산이 불가하다 (혹은 의미 없다)
# 그래서 '처리' 가 가능하도록 데이터를 바꿔주어야 할 필요가 있다.  (????)

# 범주형 테이터 컬럼한개를 '범주의 개수' 만큼 늘려서
# 범주에 해당하는 값에만 '1' 을 주고
# 나머지에는 '0' 을 주어 처리하는 방법
# --> 이를 One-hot encoding 이라 합니다  <-- 결국 '1' 은 한개만 등장한다

# 가령
# Color 란 컬럼에  "Red", "Green", "Blue"  라는 '세가지' 종류의 범주값만 있다면

# '세 개'의 '0, 1 로 구성된 값'으로 변환시키는 겁니다 ('1' 은 한개만 등장)
#   Red ->   [1, 0, 0]
#   Green -> [0, 1, 0]
#   Blue ->  [0, 0, 1]

#  - 범주형 데이터는 분석단계에서 계산이 어렵기 때문에 숫자형으로 변경이 필요함
#  - 범주형 데이터의 각 범주(category)를 column레벨로 변경
#  - 해당 범주에 해당하면 1, 아니면 0으로 채우는 인코딩 기법
#  - pandas.get_dummies 함수 사용
#    - drop_first : 첫번째 카테고리 값은 사용하지 않음

df['Pclass'].unique()
# >>> 출력:
# array(['3', '1', '2'], dtype=object)

pd.get_dummies(df)
# >>> 출력:
#      PassengerId  Survived   Age  SibSp  Parch     Fare  Age_span  Pclass_1  \
# 0              1         0  22.0      1      0   7.2500        20     False   
# 1              2         1  38.0      1      0  71.2833        30      True   
# 2              3         1  26.0      0      0   7.9250        20     False   
# 3              4         1  35.0      1      0  53.1000        30      True   
# 4              5         0  35.0      0      0   8.0500        30     False   
# ..           ...       ...   ...    ...    ...      ...       ...       ...   
# 886          887         0  27.0      0      0  13.0000        20     False   
# 887          888         1  19.0      0      0  30.0000        10      True   
# 888          889         0   NaN      1      2  23.4500        -1     False   
# 889          890         1  26.0      0      0  30.0000        20      True   
# 890          891         0  32.0      0      0   7.7500        30     False   
# 
#      Pclass_2  Pclass_3  ...  Cabin_F G73  Cabin_F2  Cabin_F33  Cabin_F38  \
# 0       False      True  ...        False     False      False      False   

# columns=  파라미터로
# one-hot encoding 컬럼으로 만들어질 컬럼들을 지정해줄수 있다.

pd.get_dummies(df, columns=['Pclass', 'Sex', 'Embarked'])
# >>> 출력:
#      PassengerId  Survived                                               Name  \
# 0              1         0                            Braund, Mr. Owen Harris   
# 1              2         1  Cumings, Mrs. John Bradley (Florence Briggs Th...   
# 2              3         1                             Heikkinen, Miss. Laina   
# 3              4         1       Futrelle, Mrs. Jacques Heath (Lily May Peel)   
# 4              5         0                           Allen, Mr. William Henry   
# ..           ...       ...                                                ...   
# 886          887         0                              Montvila, Rev. Juozas   
# 887          888         1                       Graham, Miss. Margaret Edith   
# 888          889         0           Johnston, Miss. Catherine Helen "Carrie"   
# 889          890         1                              Behr, Mr. Karl Howell   
# 890          891         0                                Dooley, Mr. Patrick   
# 
#       Age  SibSp  Parch            Ticket     Fare Cabin  Age_span  Pclass_1  \
# 0    22.0      1      0         A/5 21171   7.2500   NaN        20     False   

pd.get_dummies(df, columns=['Pclass', 'Sex', 'Embarked'], dtype=int)
# >>> 출력:
#      PassengerId  Survived                                               Name  \
# 0              1         0                            Braund, Mr. Owen Harris   
# 1              2         1  Cumings, Mrs. John Bradley (Florence Briggs Th...   
# 2              3         1                             Heikkinen, Miss. Laina   
# 3              4         1       Futrelle, Mrs. Jacques Heath (Lily May Peel)   
# 4              5         0                           Allen, Mr. William Henry   
# ..           ...       ...                                                ...   
# 886          887         0                              Montvila, Rev. Juozas   
# 887          888         1                       Graham, Miss. Margaret Edith   
# 888          889         0           Johnston, Miss. Catherine Helen "Carrie"   
# 889          890         1                              Behr, Mr. Karl Howell   
# 890          891         0                                Dooley, Mr. Patrick   
# 
#       Age  SibSp  Parch            Ticket     Fare Cabin  Age_span  Pclass_1  \
# 0    22.0      1      0         A/5 21171   7.2500   NaN        20         0   

pd.get_dummies(df, columns=['Pclass', 'Sex', 'Embarked'], dtype=int, drop_first=True)
# >>> 출력:
#      PassengerId  Survived                                               Name  \
# 0              1         0                            Braund, Mr. Owen Harris   
# 1              2         1  Cumings, Mrs. John Bradley (Florence Briggs Th...   
# 2              3         1                             Heikkinen, Miss. Laina   
# 3              4         1       Futrelle, Mrs. Jacques Heath (Lily May Peel)   
# 4              5         0                           Allen, Mr. William Henry   
# ..           ...       ...                                                ...   
# 886          887         0                              Montvila, Rev. Juozas   
# 887          888         1                       Graham, Miss. Margaret Edith   
# 888          889         0           Johnston, Miss. Catherine Helen "Carrie"   
# 889          890         1                              Behr, Mr. Karl Howell   
# 890          891         0                                Dooley, Mr. Patrick   
# 
#       Age  SibSp  Parch            Ticket     Fare Cabin  Age_span  Pclass_2  \
# 0    22.0      1      0         A/5 21171   7.2500   NaN        20         0   

# 이렇게 one-hot encoding 함으로,
# 범주형 데이터도 '연산'이 가능한 형태로 바뀌었다.
# 도대체 '연산'이 무엇이기에?

# 머신러닝등을 하다보면 범주형 데이터에 대한 예측값이 정확히 [1, 0, 0] 이런식으로 나오지 않는다
# 수~~~ 많은 '연산'을 통해  [0.98232424,  0.0003455, 0.000455] <-- 이런식으로 나온다.

# [0.98232424,  0.0003455, 0.000455] 은 다음 세개중 무엇에 가장 가까운가?
#    [1, 0, 0]
#    [0, 1, 0]
#    [0, 0, 1]
# 이 판단 과정도 산술 '연산'인 것이다.

# 이레 바로 '연산' 이 가능한 형태라는 의미다.

# # groupby
#   + 데이터를 특정 컬럼으로 묶어서 그룹화
#   + 아래의 세 단계를 적용하여 데이터를 그룹화(groupping) (SQL의 group by 와 개념적으로는 동일, 사용법은 유사)
#     - 데이터 분할 (split)
#     - operation 적용 (apply)
#     - 데이터 병합 (combine)

df = load_titanic()
df.head()
# >>> 출력:
#    PassengerId  Survived  Pclass  \
# 0            1         0       3   
# 1            2         1       1   
# 2            3         1       3   
# 3            4         1       1   
# 4            5         0       3   
# 
#                                                 Name     Sex   Age  SibSp  \
# 0                            Braund, Mr. Owen Harris    male  22.0      1   
# 1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 2                             Heikkinen, Miss. Laina  female  26.0      0   
# 3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 4                           Allen, Mr. William Henry    male  35.0      0   
# 
#    Parch            Ticket     Fare Cabin Embarked  

# ## groupby(),  groups 속성
#  - 각 그룹과 그룹에 속한 index를 dict 형태로 표현

# groupby(컬럼 혹은 컬럼의 리스트)

grouped_pclass = df.groupby('Pclass')
grouped_pclass  # DataFrameGroupBy 객체 (이하 'groupby 객체')
# >>> 출력:
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7de5251ad510>

# groupby 객체의 속성
#   - groups -> dict 형태
#   - key -> 컬럼의 각 값
#   - value -> 인덱스 값들의 list

grouped_pclass.groups  # Pclass 의 값은 1, 2, 3 세가지 밖에 없으니 그룹의 개수는 3개다.
# groups 의 key 값은 'Plcass' 의 값들.
#          value 는 index 들의 list
# >>> 출력:
# {1: [1, 3, 6, 11, 23, 27, 30, 31, 34, 35, 52, 54, 55, 61, 62, 64, 83, 88, 92, 96, 97, 102, 110, 118, 124, 136, 137, 139, 151, 155, 166, 168, 170, 174, 177, 185, 187, 194, 195, 209, 215, 218, 224, 230, 245, 248, 252, 256, 257, 258, 262, 263, 268, 269, 270, 273, 275, 284, 290, 291, 295, 297, 298, 299, 305, 306, 307, 309, 310, 311, 318, 319, 325, 329, 331, 332, 334, 336, 337, 339, 341, 351, 356, 366, 369, 370, 373, 375, 377, 380, 383, 390, 393, 412, 430, 434, 435, 438, 445, 447, ...], 2: [9, 15, 17, 20, 21, 33, 41, 43, 53, 56, 58, 66, 70, 72, 78, 84, 98, 99, 117, 120, 122, 123, 133, 134, 135, 144, 145, 148, 149, 150, 161, 178, 181, 183, 190, 191, 193, 199, 211, 213, 217, 219, 221, 226, 228, 232, 234, 236, 237, 238, 239, 242, 247, 249, 259, 265, 272, 277, 288, 292, 303, 308, 312, 314, 316, 317, 322, 323, 327, 340, 342, 343, 344, 345, 346, 357, 361, 385, 387, 389, 397, 398, 399, 405, 407, 413, 416, 417, 418, 426, 427, 432, 437, 439, 440, 443, 446, 450, 458, 463, ...], 3: [0, 2, 4, 5, 7, 8, 10, 12, 13, 14, 16, 18, 19, 22, 24, 25, 26, 28, 29, 32, 36, 37, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51, 57, 59, 60, 63, 65, 67, 68, 69, 71, 73, 74, 75, 76, 77, 79, 80, 81, 82, 85, 86, 87, 89, 90, 91, 93, 94, 95, 100, 101, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 119, 121, 125, 126, 127, 128, 129, 130, 131, 132, 138, 140, 141, 142, 143, 146, 147, 152, 153, 154, 156, 157, 158, 159, ...]}

# size()
#   각 group 별로 담겨 있는 데이터 개수 확인

grouped_pclass.size()
# >>> 출력:
# Pclass
# 1    216
# 2    184
# 3    491
# dtype: int64

# 성별(Sex) 로 grouping
grouped_sex = df.groupby('Sex')
grouped_sex
# >>> 출력:
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7de5251ed550>

grouped_sex.groups
# >>> 출력:
# {'female': [1, 2, 3, 8, 9, 10, 11, 14, 15, 18, 19, 22, 24, 25, 28, 31, 32, 38, 39, 40, 41, 43, 44, 47, 49, 52, 53, 56, 58, 61, 66, 68, 71, 79, 82, 84, 85, 88, 98, 100, 106, 109, 111, 113, 114, 119, 123, 128, 132, 133, 136, 140, 141, 142, 147, 151, 156, 161, 166, 167, 172, 177, 180, 184, 186, 190, 192, 194, 195, 198, 199, 205, 208, 211, 215, 216, 218, 229, 230, 233, 235, 237, 240, 241, 246, 247, 251, 254, 255, 256, 257, 258, 259, 264, 268, 269, 272, 274, 275, 276, ...], 'male': [0, 4, 5, 6, 7, 12, 13, 16, 17, 20, 21, 23, 26, 27, 29, 30, 33, 34, 35, 36, 37, 42, 45, 46, 48, 50, 51, 54, 55, 57, 59, 60, 62, 63, 64, 65, 67, 69, 70, 72, 73, 74, 75, 76, 77, 78, 80, 81, 83, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 101, 102, 103, 104, 105, 107, 108, 110, 112, 115, 116, 117, 118, 120, 121, 122, 124, 125, 126, 127, 129, 130, 131, 134, 135, 137, 138, 139, 143, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, ...]}

grouped_sex.size()
# >>> 출력:
# Sex
# female    314
# male      577
# dtype: int64

# ## get_group()
# - 그룹별 DataFrame 리턴

grouped_pclass.get_group(1)  # Pclass 가 1 인 그룹의 DataFrame 리턴
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 1              2         1       1   
# 3              4         1       1   
# 6              7         0       1   
# 11            12         1       1   
# 23            24         1       1   
# ..           ...       ...     ...   
# 871          872         1       1   
# 872          873         0       1   
# 879          880         1       1   
# 887          888         1       1   
# 889          890         1       1   
# 
#                                                   Name     Sex   Age  SibSp  \
# 1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   

grouped_sex.get_group('female').head()
# >>> 출력:
#    PassengerId  Survived  Pclass  \
# 1            2         1       1   
# 2            3         1       3   
# 3            4         1       1   
# 8            9         1       3   
# 9           10         1       2   
# 
#                                                 Name     Sex   Age  SibSp  \
# 1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 2                             Heikkinen, Miss. Laina  female  26.0      0   
# 3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 8  Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)  female  27.0      0   
# 9                Nasser, Mrs. Nicholas (Adele Achem)  female  14.0      1   
# 
#    Parch            Ticket     Fare Cabin Embarked  

# # groupby 객체의 기초 연산 메소드
#  - 그룹 데이터에 적용 가능한 통계 함수(NaN은 제외하여 연산)
#  - count - 데이터 개수
#  - sum   - 데이터의 합
#  - mean, std, var - 평균, 표준편차, 분산
#  - min, max - 최소, 최대값

grouped_pclass.count()
# >>> 출력:
#         PassengerId  Survived  Name  Sex  Age  SibSp  Parch  Ticket  Fare  \
# Pclass                                                                      
# 1               216       216   216  216  186    216    216     216   216   
# 2               184       184   184  184  173    184    184     184   184   
# 3               491       491   491  491  355    491    491     491   491   
# 
#         Cabin  Embarked  
# Pclass                   
# 1         176       214  
# 2          16       184  
# 3          12       491

# groupby() 로 쪼갬 <-- split
# group 별로 count()연산 <-- apply
# DataFrame 으로 합하여 리턴 <-- combine    (원본과는 다른 새로운 DataFrame)

grouped_pclass.sum(numeric_only=True)
# >>> 출력:
#         PassengerId  Survived      Age  SibSp  Parch        Fare
# Pclass                                                          
# 1             99705       136  7111.42     90     77  18177.4125
# 2             82056        87  5168.83     74     70   3801.8417
# 3            215625       119  8924.92    302    193   6714.6951

grouped_pclass[['PassengerId', 'Survived', 'Age']].mean(numeric_only=True)
# >>> 출력:
#         PassengerId  Survived        Age
# Pclass                                  
# 1        461.597222  0.629630  38.233441
# 2        445.956522  0.472826  29.877630
# 3        439.154786  0.242363  25.140620

# 객실 등급별 승객 나이 평균
grouped_pclass[['Age']].mean()
# >>> 출력:
#               Age
# Pclass           
# 1       38.233441
# 2       29.877630
# 3       25.140620

grouped_pclass.mean(numeric_only=True)['Age']
# >>> 출력:
# Pclass
# 1    38.233441
# 2    29.877630
# 3    25.140620
# Name: Age, dtype: float64

# 각 객실 등급별 생존률 구하기
grouped_pclass[['Survived']].mean()
# >>> 출력:
#         Survived
# Pclass          
# 1       0.629630
# 2       0.472826
# 3       0.242363

# ## 복수 columns로 groupping 하기
#  - groupby에 column 리스트를 전달
#  - 통계함수를 적용한 결과는 multiindex를 갖는 dataframe

# 'Pclass' 와 'Sex' 에 따른 생존률 구하기.

grouped_multi = df.groupby(['Pclass', 'Sex'])
grouped_multi
# >>> 출력:
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7de5250f3e50>

grouped_multi.groups
# >>> 출력:
# {(1, 'female'): [1, 3, 11, 31, 52, 61, 88, 136, 151, 166, 177, 194, 195, 215, 218, 230, 256, 257, 258, 268, 269, 275, 290, 291, 297, 299, 306, 307, 309, 310, 311, 318, 319, 325, 329, 334, 337, 341, 356, 366, 369, 375, 380, 383, 393, 412, 435, 457, 486, 496, 498, 504, 513, 520, 523, 537, 539, 540, 556, 558, 571, 577, 581, 585, 591, 609, 627, 641, 669, 689, 700, 708, 710, 716, 730, 742, 759, 763, 765, 779, 781, 796, 809, 820, 829, 835, 842, 849, 853, 856, 862, 871, 879, 887], (1, 'male'): [6, 23, 27, 30, 34, 35, 54, 55, 62, 64, 83, 92, 96, 97, 102, 110, 118, 124, 137, 139, 155, 168, 170, 174, 185, 187, 209, 224, 245, 248, 252, 262, 263, 270, 273, 284, 295, 298, 305, 331, 332, 336, 339, 351, 370, 373, 377, 390, 430, 434, 438, 445, 447, 449, 452, 453, 456, 460, 462, 467, 475, 484, 487, 492, 493, 505, 507, 512, 515, 527, 536, 544, 545, 550, 555, 557, 572, 583, 587, 599, 602, 604, 607, 621, 625, 630, 632, 633, 645, 647, 659, 660, 662, 671, 679, 681, 690, 694, 698, 701, ...], (2, 'female'): [9, 15, 41, 43, 53, 56, 58, 66, 84, 98, 123, 133, 161, 190, 199, 211, 237, 247, 259, 272, 303, 312, 316, 322, 323, 327, 345, 346, 357, 387, 389, 399, 416, 417, 426, 427, 432, 437, 440, 443, 446, 458, 472, 473, 506, 516, 518, 526, 530, 535, 546, 576, 580, 596, 600, 608, 615, 618, 635, 651, 670, 706, 717, 720, 726, 747, 750, 754, 772, 774, 801, 854, 865, 866, 874, 880], (2, 'male'): [17, 20, 21, 33, 70, 72, 78, 99, 117, 120, 122, 134, 135, 144, 145, 148, 149, 150, 178, 181, 183, 191, 193, 213, 217, 219, 221, 226, 228, 232, 234, 236, 238, 239, 242, 249, 265, 277, 288, 292, 308, 314, 317, 340, 342, 343, 344, 361, 385, 397, 398, 405, 407, 413, 418, 439, 450, 463, 466, 476, 481, 529, 543, 547, 549, 551, 562, 570, 582, 586, 594, 619, 626, 637, 655, 658, 665, 666, 672, 673, 674, 684, 685, 695, 705, 714, 722, 723, 728, 732, 733, 734, 755, 757, 791, 795, 800, 808, 812, 817, ...], (3, 'female'): [2, 8, 10, 14, 18, 19, 22, 24, 25, 28, 32, 38, 39, 40, 44, 47, 49, 68, 71, 79, 82, 85, 100, 106, 109, 111, 113, 114, 119, 128, 132, 140, 141, 142, 147, 156, 167, 172, 180, 184, 186, 192, 198, 205, 208, 216, 229, 233, 235, 240, 241, 246, 251, 254, 255, 264, 274, 276, 279, 289, 293, 300, 315, 328, 330, 347, 358, 359, 362, 367, 368, 374, 376, 381, 394, 396, 402, 404, 409, 415, 419, 423, 431, 436, 448, 469, 474, 479, 483, 485, 501, 502, 503, 533, 534, 541, 542, 554, 559, 564, ...], (3, 'male'): [0, 4, 5, 7, 12, 13, 16, 26, 29, 36, 37, 42, 45, 46, 48, 50, 51, 57, 59, 60, 63, 65, 67, 69, 73, 74, 75, 76, 77, 80, 81, 86, 87, 89, 90, 91, 93, 94, 95, 101, 103, 104, 105, 107, 108, 112, 115, 116, 121, 125, 126, 127, 129, 130, 131, 138, 143, 146, 152, 153, 154, 157, 158, 159, 160, 162, 163, 164, 165, 169, 171, 173, 175, 176, 179, 182, 188, 189, 196, 197, 200, 201, 202, 203, 204, 206, 207, 210, 212, 214, 220, 222, 223, 225, 227, 231, 243, 244, 250, 253, ...]}

grouped_multi.size()
# >>> 출력:
# Pclass  Sex   
# 1       female     94
#         male      122
# 2       female     76
#         male      108
# 3       female    144
#         male      347
# dtype: int64

grouped_multi[['Survived', 'Age']].mean()
# >>> 출력:
#                Survived        Age
# Pclass Sex                        
# 1      female  0.968085  34.611765
#        male    0.368852  41.281386
# 2      female  0.921053  28.722973
#        male    0.157407  30.740707
# 3      female  0.500000  21.750000
#        male    0.135447  26.507589

# # index를 이용한 groupby()
#  - index가 있는 경우, groupby 함수에 level 사용 가능
#    - level은 index의 depth를 의미하며, 가장 왼쪽부터 0부터 증가
#    
# * **set_index** 함수
#  - column 데이터를 index 레벨로 변경
# * **reset_index** 함수
#  - 인덱스 초기화

df.head()  # 기본적으로 0-base 인덱스
# >>> 출력:
#    PassengerId  Survived  Pclass  \
# 0            1         0       3   
# 1            2         1       1   
# 2            3         1       3   
# 3            4         1       1   
# 4            5         0       3   
# 
#                                                 Name     Sex   Age  SibSp  \
# 0                            Braund, Mr. Owen Harris    male  22.0      1   
# 1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 2                             Heikkinen, Miss. Laina  female  26.0      0   
# 3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 4                           Allen, Mr. William Henry    male  35.0      0   
# 
#    Parch            Ticket     Fare Cabin Embarked  

df.set_index('Pclass')
# >>> 출력:
#         PassengerId  Survived  \
# Pclass                          
# 3                 1         0   
# 1                 2         1   
# 3                 3         1   
# 1                 4         1   
# 3                 5         0   
# ...             ...       ...   
# 2               887         0   
# 1               888         1   
# 3               889         0   
# 1               890         1   
# 3               891         0   
# 
#                                                      Name     Sex   Age  \

df.set_index(['Pclass', 'Sex'])
# >>> 출력:
#                PassengerId  Survived  \
# Pclass Sex                             
# 3      male              1         0   
# 1      female            2         1   
# 3      female            3         1   
# 1      female            4         1   
# 3      male              5         0   
# ...                    ...       ...   
# 2      male            887         0   
# 1      female          888         1   
# 3      female          889         0   
# 1      male            890         1   
# 3      male            891         0   
# 
#                                                             Name   Age  SibSp  \

df.set_index(['Pclass', 'Sex']).reset_index()
# >>> 출력:
#      Pclass     Sex  PassengerId  Survived  \
# 0         3    male            1         0   
# 1         1  female            2         1   
# 2         3  female            3         1   
# 3         1  female            4         1   
# 4         3    male            5         0   
# ..      ...     ...          ...       ...   
# 886       2    male          887         0   
# 887       1  female          888         1   
# 888       3  female          889         0   
# 889       1    male          890         1   
# 890       3    male          891         0   
# 
#                                                   Name   Age  SibSp  Parch  \
# 0                              Braund, Mr. Owen Harris  22.0      1      0   

df.set_index('Embarked').groupby(level=0).size()

# level=0 인덱스를 기준으로 쪼갠다.
#  지금은 Embarked 가 index 이기 때문에 'C', 'S', 'Q' 으로 쪼개진다.
# >>> 출력:
# Embarked
# C    168
# Q     77
# S    644
# dtype: int64

# # groupby(함수)

# groupby(함수) =>
#   함수의 매개변수는 index 다
#   함수가 리턴하는 값이 grouping 기준이 된다!

# ## 나이대별 생존률 구하기

df.set_index('Age')
# >>> 출력:
#       PassengerId  Survived  Pclass  \
# Age                                   
# 22.0            1         0       3   
# 38.0            2         1       1   
# 26.0            3         1       3   
# 35.0            4         1       1   
# 35.0            5         0       3   
# ...           ...       ...     ...   
# 27.0          887         0       2   
# 19.0          888         1       1   
# NaN           889         0       3   
# 26.0          890         1       1   
# 32.0          891         0       3   
# 
#                                                    Name     Sex  SibSp  Parch  \

# groupby() 에 전달할 함수

# 전단되는 매개변수는 index
age_categorize(34)  # 리턴값이 곧 grouping 기준이 된다.
# >>> 출력:
# 30

df.set_index('Age').groupby(age_categorize)['Survived'].mean()
# >>> 출력:
# Age
# -1     0.293785
#  0     0.612903
#  10    0.401961
#  20    0.350000
#  30    0.437126
#  40    0.382022
#  50    0.416667
#  60    0.315789
#  70    0.000000
#  80    1.000000
# Name: Survived, dtype: float64

# group(by=??)  by= 매개변수에는 무엇이 전달될수 있나?
# - groupby(컬럼, 혹은 컬럼 리스트)
# - groupby(level)
# - groupby(함수)
# - groupby(Series 혹은 dict)

# ## 도전: 이름(Name) 시작 알파벳으로 승객수 집계

# 도전) 이름(Name) 시작 알파벳별로 승객수 집계
"""
A    51
B    72
C    69
D    43
...
...
U     1
V    15
W    33
Y     7
Z     3
"""
None

# 변희언
def get_first_alphabet(name):
    return name[0].upper()
df.set_index('Name').groupby(get_first_alphabet)['PassengerId'].count()
# >>> 출력:
# Name
# A    51
# B    72
# C    69
# D    43
# E    12
# F    31
# G    41
# H    69
# I     6
# J    30
# K    28
# L    48
# M    74
# N    29

# 김성제
df.set_index('Name').groupby(lambda x : x.upper()[0]).size()
# >>> 출력:
# Name
# A    51
# B    72
# C    69
# D    43
# E    12
# F    31
# G    41
# H    69
# I     6
# J    30
# K    28
# L    48
# M    74
# N    29

# # aggregate(집계) 함수 사용하기
#  - groupby 결과에 '집계함수'를 적용하여 그룹별 데이터 확인 가능

# aggregate 의 매개변수로 주어지는 함수는
#     입력 : n ->  리턴 :1개 값     이와같은 집계함수 형태이어야 한다

df.groupby(['Pclass', 'Sex'])['PassengerId'].sum()
# >>> 출력:
# Pclass  Sex   
# 1       female     44106
#         male       55599
# 2       female     33676
#         male       48380
# 3       female     57561
#         male      158064
# Name: PassengerId, dtype: int64

df.groupby(['Pclass', 'Sex'])['PassengerId'].aggregate(np.sum)
# >>> 출력:
# <ipython-input-230-67dc3c01be08>:1: FutureWarning: The provided callable <function sum at 0x7de572165080> is currently using SeriesGroupBy.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
#   df.groupby(['Pclass', 'Sex'])['PassengerId'].aggregate(np.sum)
# Pclass  Sex   
# 1       female     44106
#         male       55599
# 2       female     33676
#         male       48380
# 3       female     57561
#         male      158064
# Name: PassengerId, dtype: int64

np.sum([10, 20, 30])
# >>> 출력:
# 60

df.groupby(['Pclass', 'Sex'])['PassengerId'].aggregate([np.sum, np.mean, np.max])
# >>> 출력:
# <ipython-input-232-4792d91451e3>:1: FutureWarning: The provided callable <function sum at 0x7de572165080> is currently using SeriesGroupBy.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
#   df.groupby(['Pclass', 'Sex'])['PassengerId'].aggregate([np.sum, np.mean, np.max])
# <ipython-input-232-4792d91451e3>:1: FutureWarning: The provided callable <function mean at 0x7de572166160> is currently using SeriesGroupBy.mean. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "mean" instead.
#   df.groupby(['Pclass', 'Sex'])['PassengerId'].aggregate([np.sum, np.mean, np.max])
# <ipython-input-232-4792d91451e3>:1: FutureWarning: The provided callable <function max at 0x7de572165760> is currently using SeriesGroupBy.max. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "max" instead.
#   df.groupby(['Pclass', 'Sex'])['PassengerId'].aggregate([np.sum, np.mean, np.max])
#                   sum        mean  max
# Pclass Sex                            
# 1      female   44106  469.212766  888
#        male     55599  455.729508  890
# 2      female   33676  443.105263  881
#        male     48380  447.962963  887
# 3      female   57561  399.729167  889
#        male    158064  455.515850  891

df.groupby(['Pclass', 'Sex'])['PassengerId'].aggregate(['sum', 'mean', 'max'])
# >>> 출력:
#                   sum        mean  max
# Pclass Sex                            
# 1      female   44106  469.212766  888
#        male     55599  455.729508  890
# 2      female   33676  443.105263  881
#        male     48380  447.962963  887
# 3      female   57561  399.729167  889
#        male    158064  455.515850  891

# ## 도전: 각 객실 등급별 탑승승객인원, 생존자수, 생존률?

# ![image.png](data:image/png;base64,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)

# 이은솔
df.groupby(['Pclass'])['Survived'].aggregate(['count','sum','mean'])\
  .rename(columns={'count': '승객수', 'sum': '생존자수', 'mean': '생존률'})
# >>> 출력:
#         승객수  생존자수       생존률
# Pclass                     
# 1       216   136  0.629630
# 2       184    87  0.472826
# 3       491   119  0.242363

# # sort_values(), sort_index()
# 값 기준 정렬, index 기준 정렬

# by=  정렬기준
#  컬럼면, 컬럼들의 리스트
df.sort_values(by='Pclass')
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 445          446         1       1   
# 310          311         1       1   
# 309          310         1       1   
# 307          308         1       1   
# 306          307         1       1   
# ..           ...       ...     ...   
# 379          380         0       3   
# 381          382         1       3   
# 382          383         0       3   
# 371          372         0       3   
# 890          891         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 445                          Dodge, Master. Washington    male   4.0      0   

df.sort_values(by='Pclass', ascending=False)  # 내림차순
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 0              1         0       3   
# 511          512         0       3   
# 500          501         0       3   
# 501          502         0       3   
# 502          503         0       3   
# ..           ...       ...     ...   
# 102          103         0       1   
# 710          711         1       1   
# 711          712         0       1   
# 712          713         1       1   
# 445          446         1       1   
# 
#                                                  Name     Sex   Age  SibSp  \
# 0                             Braund, Mr. Owen Harris    male  22.0      1   

df.sort_values(by=['Pclass', 'Age'])   # 우선 Pclass 오름차순, 그리고 Age 오름차순
# >>> 출력:
#      PassengerId  Survived  Pclass                                      Name  \
# 305          306         1       1            Allison, Master. Hudson Trevor   
# 297          298         0       1              Allison, Miss. Helen Loraine   
# 445          446         1       1                 Dodge, Master. Washington   
# 802          803         1       1       Carter, Master. William Thornton II   
# 435          436         1       1                 Carter, Miss. Lucile Polk   
# ..           ...       ...     ...                                       ...   
# 859          860         0       3                          Razi, Mr. Raihed   
# 863          864         0       3         Sage, Miss. Dorothy Edith "Dolly"   
# 868          869         0       3               van Melkebeke, Mr. Philemon   
# 878          879         0       3                        Laleff, Mr. Kristo   
# 888          889         0       3  Johnston, Miss. Catherine Helen "Carrie"   
# 
#         Sex    Age  SibSp  Parch      Ticket      Fare    Cabin Embarked  
# 305    male   0.92      1      2      113781  151.5500  C22 C26        S  

df.sort_values(by=['Pclass', 'Age'], ascending=[True, False])
# >>> 출력:
#      PassengerId  Survived  Pclass                                      Name  \
# 630          631         1       1      Barkworth, Mr. Algernon Henry Wilson   
# 96            97         0       1                 Goldschmidt, Mr. George B   
# 493          494         0       1                   Artagaveytia, Mr. Ramon   
# 745          746         0       1              Crosby, Capt. Edward Gifford   
# 54            55         0       1            Ostby, Mr. Engelhart Cornelius   
# ..           ...       ...     ...                                       ...   
# 859          860         0       3                          Razi, Mr. Raihed   
# 863          864         0       3         Sage, Miss. Dorothy Edith "Dolly"   
# 868          869         0       3               van Melkebeke, Mr. Philemon   
# 878          879         0       3                        Laleff, Mr. Kristo   
# 888          889         0       3  Johnston, Miss. Catherine Helen "Carrie"   
# 
#         Sex   Age  SibSp  Parch      Ticket     Fare Cabin Embarked  
# 630    male  80.0      0      0       27042  30.0000   A23        S  

# axis=
# df.sort_values(by=3, axis=1)   # 에러. 인덱스 3 row 는 비교 불가능한 값들인지라.

df[['Pclass', 'Fare', 'Age']].sort_values(by=3, axis=1)
# >>> 출력:
#      Pclass   Age     Fare
# 0         3  22.0   7.2500
# 1         1  38.0  71.2833
# 2         3  26.0   7.9250
# 3         1  35.0  53.1000
# 4         3  35.0   8.0500
# ..      ...   ...      ...
# 886       2  27.0  13.0000
# 887       1  19.0  30.0000
# 888       3   NaN  23.4500
# 889       1  26.0  30.0000
# 890       3  32.0   7.7500
# 
# [891 rows x 3 columns]

df.sort_index(ascending=False)  # index 기준 내림차순
# >>> 출력:
#      PassengerId  Survived  Pclass  \
# 890          891         0       3   
# 889          890         1       1   
# 888          889         0       3   
# 887          888         1       1   
# 886          887         0       2   
# ..           ...       ...     ...   
# 4              5         0       3   
# 3              4         1       1   
# 2              3         1       3   
# 1              2         1       1   
# 0              1         0       3   
# 
#                                                   Name     Sex   Age  SibSp  \
# 890                                Dooley, Mr. Patrick    male  32.0      0   

df.sort_index(axis=1)
# >>> 출력:
#       Age Cabin Embarked     Fare  \
# 0    22.0   NaN        S   7.2500   
# 1    38.0   C85        C  71.2833   
# 2    26.0   NaN        S   7.9250   
# 3    35.0  C123        S  53.1000   
# 4    35.0   NaN        S   8.0500   
# ..    ...   ...      ...      ...   
# 886  27.0   NaN        S  13.0000   
# 887  19.0   B42        S  30.0000   
# 888   NaN   NaN        S  23.4500   
# 889  26.0  C148        C  30.0000   
# 890  32.0   NaN        Q   7.7500   
# 
#                                                   Name  Parch  PassengerId  \
# 0                              Braund, Mr. Owen Harris      0            1   

# # pivot, pd.pivot_table 함수

df = pd.DataFrame({
    '지역': ['서울', '서울', '서울', '경기', '경기', '부산', '서울', '서울', '부산', '경기', '경기', '경기'],
    '요일': ['월요일', '화요일', '수요일', '월요일', '화요일', '월요일', '목요일', '금요일', '화요일', '수요일', '목요일', '금요일'],
    '강수량': [100, 80, 1000, 200, 200, 100, 50, 100, 200, 100, 50, 100],
    '강수확률': [80, 70, 90, 10, 20, 30, 50, 90, 20, 80, 50, 10]
                  })

df
# >>> 출력:
#     지역   요일   강수량  강수확률
# 0   서울  월요일   100    80
# 1   서울  화요일    80    70
# 2   서울  수요일  1000    90
# 3   경기  월요일   200    10
# 4   경기  화요일   200    20
# 5   부산  월요일   100    30
# 6   서울  목요일    50    50
# 7   서울  금요일   100    90
# 8   부산  화요일   200    20
# 9   경기  수요일   100    80
# 10  경기  목요일    50    50
# 11  경기  금요일   100    10

df.T
# >>> 출력:
#        0    1     2    3    4    5    6    7    8    9    10   11
# 지역     서울   서울    서울   경기   경기   부산   서울   서울   부산   경기   경기   경기
# 요일    월요일  화요일   수요일  월요일  화요일  월요일  목요일  금요일  화요일  수요일  목요일  금요일
# 강수량   100   80  1000  200  200  100   50  100  200  100   50  100
# 강수확률   80   70    90   10   20   30   50   90   20   80   50   10

# ## pivot()
#  - dataframe의 형태를 변경
#  - 인덱스, 컬럼, 데이터로 사용할 컬럼을 명시

#  pivot(index=None, columns=None, values=None)

# index 로 지정할 컬럼
# column 으로 지정할 컬럼
# values 나머지 채울 값들.

# 이제 지역별로(index)  요일별(column) 강수량/강수확률(value)을 보고싶다

df
# >>> 출력:
#     지역   요일   강수량  강수확률
# 0   서울  월요일   100    80
# 1   서울  화요일    80    70
# 2   서울  수요일  1000    90
# 3   경기  월요일   200    10
# 4   경기  화요일   200    20
# 5   부산  월요일   100    30
# 6   서울  목요일    50    50
# 7   서울  금요일   100    90
# 8   부산  화요일   200    20
# 9   경기  수요일   100    80
# 10  경기  목요일    50    50
# 11  경기  금요일   100    10

df.pivot(index='지역', columns='요일')
# >>> 출력:
#       강수량                              강수확률                        
# 요일    금요일   목요일     수요일    월요일    화요일   금요일   목요일   수요일   월요일   화요일
# 지역                                                                 
# 경기  100.0  50.0   100.0  200.0  200.0  10.0  50.0  80.0  10.0  20.0
# 부산    NaN   NaN     NaN  100.0  200.0   NaN   NaN   NaN  30.0  20.0
# 서울  100.0  50.0  1000.0  100.0   80.0  90.0  50.0  90.0  80.0  70.0

df.loc[len(df)] = pd.Series(['경기', '금요일', 111, 11], index=['지역', '요일', '강수량', '강수확률'])
df
# >>> 출력:
#     지역   요일   강수량  강수확률
# 0   서울  월요일   100    80
# 1   서울  화요일    80    70
# 2   서울  수요일  1000    90
# 3   경기  월요일   200    10
# 4   경기  화요일   200    20
# 5   부산  월요일   100    30
# 6   서울  목요일    50    50
# 7   서울  금요일   100    90
# 8   부산  화요일   200    20
# 9   경기  수요일   100    80
# 10  경기  목요일    50    50
# 11  경기  금요일   100    10
# 12  경기  금요일   111    11

# index= 와 columns=  지정한 컬럼에 중복된 데이터가 있으면 에러!
df.pivot(index='지역', columns='요일')
# >>> 출력:
# ValueError: Index contains duplicate entries, cannot reshape

df.drop([len(df) - 1], inplace=True)

df
# >>> 출력:
#     지역   요일   강수량  강수확률
# 0   서울  월요일   100    80
# 1   서울  화요일    80    70
# 2   서울  수요일  1000    90
# 3   경기  월요일   200    10
# 4   경기  화요일   200    20
# 5   부산  월요일   100    30
# 6   서울  목요일    50    50
# 7   서울  금요일   100    90
# 8   부산  화요일   200    20
# 9   경기  수요일   100    80
# 10  경기  목요일    50    50
# 11  경기  금요일   100    10

df.pivot(index='요일', columns='지역')
# >>> 출력:
#        강수량                 강수확률            
# 지역      경기     부산      서울    경기    부산    서울
# 요일                                         
# 금요일  100.0    NaN   100.0  10.0   NaN  90.0
# 목요일   50.0    NaN    50.0  50.0   NaN  50.0
# 수요일  100.0    NaN  1000.0  80.0   NaN  90.0
# 월요일  200.0  100.0   100.0  10.0  30.0  80.0
# 화요일  200.0  200.0    80.0  20.0  20.0  70.0

df.pivot(index='요일', columns='지역', values='강수량')
# >>> 출력:
# 지역      경기     부산      서울
# 요일                       
# 금요일  100.0    NaN   100.0
# 목요일   50.0    NaN    50.0
# 수요일  100.0    NaN  1000.0
# 월요일  200.0  100.0   100.0
# 화요일  200.0  200.0    80.0

df.pivot(index='요일', columns='지역')['강수량']
# >>> 출력:
# 지역      경기     부산      서울
# 요일                       
# 금요일  100.0    NaN   100.0
# 목요일   50.0    NaN    50.0
# 수요일  100.0    NaN  1000.0
# 월요일  200.0  100.0   100.0
# 화요일  200.0  200.0    80.0

# ## pd.pivot_table()
#  - 기능적으로 pivot과 동일
#  - pivot과의 차이점
#    - 중복되는 모호한 값이 있을 경우, aggregation 함수 사용하여 값을 채움

pd.pivot_table(df, index='지역', columns='요일')
# >>> 출력:
#       강수량                              강수확률                        
# 요일    금요일   목요일     수요일    월요일    화요일   금요일   목요일   수요일   월요일   화요일
# 지역                                                                 
# 경기  100.0  50.0   100.0  200.0  200.0  10.0  50.0  80.0  10.0  20.0
# 부산    NaN   NaN     NaN  100.0  200.0   NaN   NaN   NaN  30.0  20.0
# 서울  100.0  50.0  1000.0  100.0   80.0  90.0  50.0  90.0  80.0  70.0

# 중복데이터를 만들어 본다.  서울-월요일
df2 = pd.DataFrame({
    '지역': ['서울', '서울', '서울', '경기', '경기', '부산', '서울', '서울', '부산', '경기', '경기', '경기'],
    '요일': ['월요일', '월요일', '수요일', '월요일', '화요일', '월요일', '목요일', '금요일', '화요일', '수요일', '목요일', '금요일'],
    '강수량': [100, 80, 1000, 200, 200, 100, 50, 100, 200, 100, 50, 100],
    '강수확률': [80, 70, 90, 10, 20, 30, 50, 90, 20, 80, 50, 10]
                  })

df2  # 서울 - 월요일 <-- 2개 있다.
# >>> 출력:
#     지역   요일   강수량  강수확률
# 0   서울  월요일   100    80
# 1   서울  월요일    80    70
# 2   서울  수요일  1000    90
# 3   경기  월요일   200    10
# 4   경기  화요일   200    20
# 5   부산  월요일   100    30
# 6   서울  목요일    50    50
# 7   서울  금요일   100    90
# 8   부산  화요일   200    20
# 9   경기  수요일   100    80
# 10  경기  목요일    50    50
# 11  경기  금요일   100    10

# df2.pivot(index='지역', columns='요일')  # 에러
# >>> 출력:
# ValueError: Index contains duplicate entries, cannot reshape

# pd.pivot_table?

# pivot() 은 중복된 엔트리가 있으면 오류발생
# pivot_table() 은 중복된 엔트리는 aggfunc= 를 사용하여 새로운 값 대체

# 서울, 월요일의 '평균값' 으로 채워진다
pd.pivot_table(df2, index='지역', columns='요일', aggfunc='mean')
# >>> 출력:
#       강수량                              강수확률                        
# 요일    금요일   목요일     수요일    월요일    화요일   금요일   목요일   수요일   월요일   화요일
# 지역                                                                 
# 경기  100.0  50.0   100.0  200.0  200.0  10.0  50.0  80.0  10.0  20.0
# 부산    NaN   NaN     NaN  100.0  200.0   NaN   NaN   NaN  30.0  20.0
# 서울  100.0  50.0  1000.0   90.0    NaN  90.0  50.0  90.0  75.0   NaN

pd.pivot_table(df2, index='지역', columns='요일', aggfunc='max')
# >>> 출력:
#       강수량                              강수확률                        
# 요일    금요일   목요일     수요일    월요일    화요일   금요일   목요일   수요일   월요일   화요일
# 지역                                                                 
# 경기  100.0  50.0   100.0  200.0  200.0  10.0  50.0  80.0  10.0  20.0
# 부산    NaN   NaN     NaN  100.0  200.0   NaN   NaN   NaN  30.0  20.0
# 서울  100.0  50.0  1000.0  100.0    NaN  90.0  50.0  90.0  80.0   NaN

# #  stack & unstack
#  - stack : 컬럼 레벨에서 인덱스 레벨(row 레벨)로 dataframe 변경
#   - 즉, 데이터를 쌓아올리는 개념으로 이해하면 쉬움
#  - unstack : 인덱스 레벨에서 컬럼 레벨로 dataframe 변경
#   - stack의 반대 operation
# 
#  - 둘은 역의 관계에 있음

df
# >>> 출력:
#     지역   요일   강수량  강수확률
# 0   서울  월요일   100    80
# 1   서울  화요일    80    70
# 2   서울  수요일  1000    90
# 3   경기  월요일   200    10
# 4   경기  화요일   200    20
# 5   부산  월요일   100    30
# 6   서울  목요일    50    50
# 7   서울  금요일   100    90
# 8   부산  화요일   200    20
# 9   경기  수요일   100    80
# 10  경기  목요일    50    50
# 11  경기  금요일   100    10

new_df = df.set_index(['지역', '요일'])
new_df
# >>> 출력:
#          강수량  강수확률
# 지역 요일             
# 서울 월요일   100    80
#    화요일    80    70
#    수요일  1000    90
# 경기 월요일   200    10
#    화요일   200    20
# 부산 월요일   100    30
# 서울 목요일    50    50
#    금요일   100    90
# 부산 화요일   200    20
# 경기 수요일   100    80
#    목요일    50    50
#    금요일   100    10

# unstack() : index(row) -> column

# unstack(level=-1, fill_value=None)
#   level : 인덱스의 레벨
#            new_df 의 경우 지역 의 인덱스 레벨이 0 이다
#                 -1 은 오른쪽 끝 인덱스 '요일' 이 컬럼이다.

new_df.unstack() # level=-1 (디폴트) 였던 '요일' index 가 '컬럼' 으로 올라간다
# >>> 출력:
#       강수량                              강수확률                        
# 요일    금요일   목요일     수요일    월요일    화요일   금요일   목요일   수요일   월요일   화요일
# 지역                                                                 
# 경기  100.0  50.0   100.0  200.0  200.0  10.0  50.0  80.0  10.0  20.0
# 부산    NaN   NaN     NaN  100.0  200.0   NaN   NaN   NaN  30.0  20.0
# 서울  100.0  50.0  1000.0  100.0   80.0  90.0  50.0  90.0  80.0  70.0

new_df.unstack(0)  # level=0 인덱스가 컬럼으로 올라감.
# >>> 출력:
#        강수량                 강수확률            
# 지역      경기     부산      서울    경기    부산    서울
# 요일                                         
# 금요일  100.0    NaN   100.0  10.0   NaN  90.0
# 목요일   50.0    NaN    50.0  50.0   NaN  50.0
# 수요일  100.0    NaN  1000.0  80.0   NaN  90.0
# 월요일  200.0  100.0   100.0  10.0  30.0  80.0
# 화요일  200.0  200.0    80.0  20.0  20.0  70.0

# stack() : column -> index(row)
# stack(level=-1, dropna=True)
#         level : 컬럼의 레벨

df2 = new_df.unstack(0)
df2
# >>> 출력:
#        강수량                 강수확률            
# 지역      경기     부산      서울    경기    부산    서울
# 요일                                         
# 금요일  100.0    NaN   100.0  10.0   NaN  90.0
# 목요일   50.0    NaN    50.0  50.0   NaN  50.0
# 수요일  100.0    NaN  1000.0  80.0   NaN  90.0
# 월요일  200.0  100.0   100.0  10.0  30.0  80.0
# 화요일  200.0  200.0    80.0  20.0  20.0  70.0

df2.stack(0)  # level=0 의 컬럼 이 index 로 내려옴
# >>> 출력:
# <ipython-input-274-a23c5fc34b9b>:1: FutureWarning: The previous implementation of stack is deprecated and will be removed in a future version of pandas. See the What's New notes for pandas 2.1.0 for details. Specify future_stack=True to adopt the new implementation and silence this warning.
#   df2.stack(0)  # level=0 의 컬럼 이 index 로 내려옴
# 지역           경기     부산      서울
# 요일                            
# 금요일 강수량   100.0    NaN   100.0
#     강수확률   10.0    NaN    90.0
# 목요일 강수량    50.0    NaN    50.0
#     강수확률   50.0    NaN    50.0
# 수요일 강수량   100.0    NaN  1000.0
#     강수확률   80.0    NaN    90.0
# 월요일 강수량   200.0  100.0   100.0
#     강수확률   10.0   30.0    80.0
# 화요일 강수량   200.0  200.0    80.0
#     강수확률   20.0   20.0    70.0

df2.stack(1)
# >>> 출력:
# <ipython-input-275-41ea66a78fc9>:1: FutureWarning: The previous implementation of stack is deprecated and will be removed in a future version of pandas. See the What's New notes for pandas 2.1.0 for details. Specify future_stack=True to adopt the new implementation and silence this warning.
#   df2.stack(1)
#            강수량  강수확률
# 요일  지역              
# 금요일 경기   100.0  10.0
#     서울   100.0  90.0
# 목요일 경기    50.0  50.0
#     서울    50.0  50.0
# 수요일 경기   100.0  80.0
#     서울  1000.0  90.0
# 월요일 경기   200.0  10.0
#     부산   100.0  30.0
#     서울   100.0  80.0
# 화요일 경기   200.0  20.0
#     부산   200.0  20.0
#     서울    80.0  70.0

df2.stack().stack()   # 모든 컬럼이 다 내리면?  결과는 -> Series
# >>> 출력:
# <ipython-input-278-5bee9182d0f6>:1: FutureWarning: The previous implementation of stack is deprecated and will be removed in a future version of pandas. See the What's New notes for pandas 2.1.0 for details. Specify future_stack=True to adopt the new implementation and silence this warning.
#   df2.stack().stack()
# 요일   지역      
# 금요일  경기  강수량      100.0
#          강수확률      10.0
#      서울  강수량      100.0
#          강수확률      90.0
# 목요일  경기  강수량       50.0
#          강수확률      50.0
#      서울  강수량       50.0
#          강수확률      50.0
# 수요일  경기  강수량      100.0
#          강수확률      80.0
#      서울  강수량     1000.0
#          강수확률      90.0
# 월요일  경기  강수량      200.0
#          강수확률      10.0

new_df
# >>> 출력:
#          강수량  강수확률
# 지역 요일             
# 서울 월요일   100    80
#    화요일    80    70
#    수요일  1000    90
# 경기 월요일   200    10
#    화요일   200    20
# 부산 월요일   100    30
# 서울 목요일    50    50
#    금요일   100    90
# 부산 화요일   200    20
# 경기 수요일   100    80
#    목요일    50    50
#    금요일   100    10

new_df.unstack(0).stack()
# >>> 출력:
# <ipython-input-281-224b45726769>:1: FutureWarning: The previous implementation of stack is deprecated and will be removed in a future version of pandas. See the What's New notes for pandas 2.1.0 for details. Specify future_stack=True to adopt the new implementation and silence this warning.
#   new_df.unstack(0).stack()
#            강수량  강수확률
# 요일  지역              
# 금요일 경기   100.0  10.0
#     서울   100.0  90.0
# 목요일 경기    50.0  50.0
#     서울    50.0  50.0
# 수요일 경기   100.0  80.0
#     서울  1000.0  90.0
# 월요일 경기   200.0  10.0
#     부산   100.0  30.0
#     서울   100.0  80.0
# 화요일 경기   200.0  20.0
#     부산   200.0  20.0
#     서울    80.0  70.0

# # pd.concat()
# DataFrame 병합
# 
#  - pandas.concat 함수
#  - 축(axis)을 따라 dataframe을 병합 가능
#    - axis = 0 → 행단위 병합 (디폴트)
#    - axis = 1 → 열단위 병합

df1 = pd.DataFrame({'key1' : np.arange(10), 'value1' : np.random.randn(10)})
df2 = pd.DataFrame({'key1' : np.arange(10), 'value1' : np.random.randn(10)})

df1
# >>> 출력:
#    key1    value1
# 0     0 -1.524831
# 1     1  1.326351
# 2     2 -0.236311
# 3     3  0.366659
# 4     4 -1.813540
# 5     5 -0.397491
# 6     6  0.475407
# 7     7 -0.007848
# 8     8  0.695512
# 9     9 -0.619076

df2
# >>> 출력:
#    key1    value1
# 0     0  0.193192
# 1     1 -0.461970
# 2     2 -1.933439
# 3     3  0.481198
# 4     4  0.165976
# 5     5 -1.583018
# 6     6  0.922799
# 7     7 -0.724760
# 8     8  0.979539
# 9     9 -0.472387

pd.concat([df1, df2])   # 행 방향으로 병합 (axis=0)
# >>> 출력:
#    key1    value1
# 0     0 -1.524831
# 1     1  1.326351
# 2     2 -0.236311
# 3     3  0.366659
# 4     4 -1.813540
# 5     5 -0.397491
# 6     6  0.475407
# 7     7 -0.007848
# 8     8  0.695512
# 9     9 -0.619076
# 0     0  0.193192
# 1     1 -0.461970
# 2     2 -1.933439
# 3     3  0.481198

pd.concat([df1, df2], axis=1)  # 열방향 병합
# >>> 출력:
#    key1    value1  key1    value1
# 0     0 -1.524831     0  0.193192
# 1     1  1.326351     1 -0.461970
# 2     2 -0.236311     2 -1.933439
# 3     3  0.366659     3  0.481198
# 4     4 -1.813540     4  0.165976
# 5     5 -0.397491     5 -1.583018
# 6     6  0.475407     6  0.922799
# 7     7 -0.007848     7 -0.724760
# 8     8  0.695512     8  0.979539
# 9     9 -0.619076     9 -0.472387

# column / index 가 다른 경우는?

df3 = pd.DataFrame({'key2' : np.arange(10), 'value2' : np.random.randn(10)})
df3  # 컬럼명이 df1, df2 와는 다르다
# >>> 출력:
#    key2    value2
# 0     0  2.768038
# 1     1 -0.243857
# 2     2  1.534253
# 3     3 -1.841641
# 4     4 -0.937764
# 5     5  0.340998
# 6     6 -0.489722
# 7     7  2.439315
# 8     8  0.606435
# 9     9 -0.416153

pd.concat([df1, df3])

# 병합시 서로에게 없는 컬럼값은 NaN 으로 채워짐
# >>> 출력:
#    key1    value1  key2    value2
# 0   0.0 -1.524831   NaN       NaN
# 1   1.0  1.326351   NaN       NaN
# 2   2.0 -0.236311   NaN       NaN
# 3   3.0  0.366659   NaN       NaN
# 4   4.0 -1.813540   NaN       NaN
# 5   5.0 -0.397491   NaN       NaN
# 6   6.0  0.475407   NaN       NaN
# 7   7.0 -0.007848   NaN       NaN
# 8   8.0  0.695512   NaN       NaN
# 9   9.0 -0.619076   NaN       NaN
# 0   NaN       NaN   0.0  2.768038
# 1   NaN       NaN   1.0 -0.243857
# 2   NaN       NaN   2.0  1.534253
# 3   NaN       NaN   3.0 -1.841641

pd.concat([df1, df3], axis=1)
# >>> 출력:
#    key1    value1  key2    value2
# 0     0 -1.524831     0  2.768038
# 1     1  1.326351     1 -0.243857
# 2     2 -0.236311     2  1.534253
# 3     3  0.366659     3 -1.841641
# 4     4 -1.813540     4 -0.937764
# 5     5 -0.397491     5  0.340998
# 6     6  0.475407     6 -0.489722
# 7     7 -0.007848     7  2.439315
# 8     8  0.695512     8  0.606435
# 9     9 -0.619076     9 -0.416153

# 서로 index 가 다르면?
df2.index += 10
df2
# >>> 출력:
#     key1    value1
# 10     0  0.193192
# 11     1 -0.461970
# 12     2 -1.933439
# 13     3  0.481198
# 14     4  0.165976
# 15     5 -1.583018
# 16     6  0.922799
# 17     7 -0.724760
# 18     8  0.979539
# 19     9 -0.472387

pd.concat([df1, df2], axis=1)
# >>> 출력:
#     key1    value1  key1    value1
# 0    0.0 -1.524831   NaN       NaN
# 1    1.0  1.326351   NaN       NaN
# 2    2.0 -0.236311   NaN       NaN
# 3    3.0  0.366659   NaN       NaN
# 4    4.0 -1.813540   NaN       NaN
# 5    5.0 -0.397491   NaN       NaN
# 6    6.0  0.475407   NaN       NaN
# 7    7.0 -0.007848   NaN       NaN
# 8    8.0  0.695512   NaN       NaN
# 9    9.0 -0.619076   NaN       NaN
# 10   NaN       NaN   0.0  0.193192
# 11   NaN       NaN   1.0 -0.461970
# 12   NaN       NaN   2.0 -1.933439
# 13   NaN       NaN   3.0  0.481198

# # pd.merge()
# 데이터프레임 병합과 조인

# ## dataframe merge
#  - SQL의 join처럼 특정한 column을 기준으로 병합
#    - join 방식: how 파라미터를 통해 명시
#      - inner: 기본값, 일치하는 값이 있는 경우
#      - left: left outer join
#      - right: right outer join
#      - outer: full outer join
#      
#  - pandas.merge 함수가 사용됨

customer = pd.DataFrame({'customer_id' : np.arange(6),
                    'name' : ['철수'"", '영희', '길동', '영수', '수민', '동건'],
                    '나이' : [40, 20, 21, 30, 31, 18]})

customer
# >>> 출력:
#    customer_id name  나이
# 0            0   철수  40
# 1            1   영희  20
# 2            2   길동  21
# 3            3   영수  30
# 4            4   수민  31
# 5            5   동건  18

orders = pd.DataFrame({'customer_id' : [1, 1, 2, 2, 2, 3, 3, 1, 4, 9],
                    'item' : ['치약', '칫솔', '이어폰', '헤드셋', '수건', '생수', '수건', '치약', '생수', '케이스'],
                    'quantity' : [1, 2, 1, 1, 3, 2, 2, 3, 2, 1]})
orders
# >>> 출력:
#    customer_id item  quantity
# 0            1   치약         1
# 1            1   칫솔         2
# 2            2  이어폰         1
# 3            2  헤드셋         1
# 4            2   수건         3
# 5            3   생수         2
# 6            3   수건         2
# 7            1   치약         3
# 8            4   생수         2
# 9            9  케이스         1

# 고객 x 구매내역
pd.merge(customer, orders, on='customer_id')
# >>> 출력:
#    customer_id name  나이 item  quantity
# 0            1   영희  20   치약         1
# 1            1   영희  20   칫솔         2
# 2            1   영희  20   치약         3
# 3            2   길동  21  이어폰         1
# 4            2   길동  21  헤드셋         1
# 5            2   길동  21   수건         3
# 6            3   영수  30   생수         2
# 7            3   영수  30   수건         2
# 8            4   수민  31   생수         2

# customer_id 컬럼을 기준으로 merge 됨.
# customer_id 컬럼은 한번만 등장 (마치 Natural join)


# customer 테이블에 있었던 '동건' '철수' 은 안보인다.
# orders 테이블에  9번 customer 가 주문한 내용도 빠졌다.
# 왜?  merge 조건에서 빠졌다.
# 기본적으로 inner join 이기에 조건에 맞지 않으면 빠지는 거다  (how="inner") (디폴트)

# left outer join
pd.merge(customer, orders, on='customer_id', how='left')

# 위 경우 customer 가 left, orders 가 right! (왼쪽, 오른쪽 중요)
# 철수, 동건이 보인다 . 이들은 구매내역이 orders 에 없으므로 item, quantity 는 NaN
# >>> 출력:
#     customer_id name  나이 item  quantity
# 0             0   철수  40  NaN       NaN
# 1             1   영희  20   치약       1.0
# 2             1   영희  20   칫솔       2.0
# 3             1   영희  20   치약       3.0
# 4             2   길동  21  이어폰       1.0
# 5             2   길동  21  헤드셋       1.0
# 6             2   길동  21   수건       3.0
# 7             3   영수  30   생수       2.0
# 8             3   영수  30   수건       2.0
# 9             4   수민  31   생수       2.0
# 10            5   동건  18  NaN       NaN

# right outer join
pd.merge(customer, orders, on='customer_id', how='right')
# >>> 출력:
#    customer_id name    나이 item  quantity
# 0            1   영희  20.0   치약         1
# 1            1   영희  20.0   칫솔         2
# 2            2   길동  21.0  이어폰         1
# 3            2   길동  21.0  헤드셋         1
# 4            2   길동  21.0   수건         3
# 5            3   영수  30.0   생수         2
# 6            3   영수  30.0   수건         2
# 7            1   영희  20.0   치약         3
# 8            4   수민  31.0   생수         2
# 9            9  NaN   NaN  케이스         1

# full outer join
pd.merge(customer, orders, on='customer_id', how='outer')  # outer = left + right
# >>> 출력:
#     customer_id name    나이 item  quantity
# 0             0   철수  40.0  NaN       NaN
# 1             1   영희  20.0   치약       1.0
# 2             1   영희  20.0   칫솔       2.0
# 3             1   영희  20.0   치약       3.0
# 4             2   길동  21.0  이어폰       1.0
# 5             2   길동  21.0  헤드셋       1.0
# 6             2   길동  21.0   수건       3.0
# 7             3   영수  30.0   생수       2.0
# 8             3   영수  30.0   수건       2.0
# 9             4   수민  31.0   생수       2.0
# 10            5   동건  18.0  NaN       NaN
# 11            9  NaN   NaN  케이스       1.0

# ## index 기준으로 merge 하기

# pd.merge(
#     left,
#     right,
#     how: str = 'inner',
#     on=None,
#     left_on=None,
#     right_on=None,
#     left_index: bool = False,   <-- merge 기준이 index 인지 여부
#     right_index: bool = False,  <-- merge 기준이 index 인지 여부
#     sort: bool = False,
#     suffixes=('_x', '_y'),
#     copy: bool = True,
#     indicator: bool = False,
#     validate=None,
# ) -> 'DataFrame'

cust1 = customer.set_index('customer_id')
cust1
# >>> 출력:
#             name  나이
# customer_id         
# 0             철수  40
# 1             영희  20
# 2             길동  21
# 3             영수  30
# 4             수민  31
# 5             동건  18

order1 = orders.set_index('customer_id')
order1
# >>> 출력:
#             item  quantity
# customer_id               
# 1             치약         1
# 1             칫솔         2
# 2            이어폰         1
# 2            헤드셋         1
# 2             수건         3
# 3             생수         2
# 3             수건         2
# 1             치약         3
# 4             생수         2
# 9            케이스         1

# 'index 이름'이 같으면 굳이 on=  없이도 index기준으로 merge가능
pd.merge(cust1, order1, left_index=True, right_index=True)
# >>> 출력:
#             name  나이 item  quantity
# customer_id                        
# 1             영희  20   치약         1
# 1             영희  20   칫솔         2
# 1             영희  20   치약         3
# 2             길동  21  이어폰         1
# 2             길동  21  헤드셋         1
# 2             길동  21   수건         3
# 3             영수  30   생수         2
# 3             영수  30   수건         2
# 4             수민  31   생수         2

# # join() 함수
#  - 내부적으로 pd.merge 함수 사용
#  - 기본적으로 index를 사용하여 left join (디폴트)

cust1
# >>> 출력:
#             name  나이
# customer_id         
# 0             철수  40
# 1             영희  20
# 2             길동  21
# 3             영수  30
# 4             수민  31
# 5             동건  18

order1
# >>> 출력:
#             item  quantity
# customer_id               
# 1             치약         1
# 1             칫솔         2
# 2            이어폰         1
# 2            헤드셋         1
# 2             수건         3
# 3             생수         2
# 3             수건         2
# 1             치약         3
# 4             생수         2
# 9            케이스         1

cust1.join(order1)

# index 로 병합
# ↓ 철수, 동건이가 나왔다! --> left outer join 됨!
# >>> 출력:
#             name  나이 item  quantity
# customer_id                        
# 0             철수  40  NaN       NaN
# 1             영희  20   치약       1.0
# 1             영희  20   칫솔       2.0
# 1             영희  20   치약       3.0
# 2             길동  21  이어폰       1.0
# 2             길동  21  헤드셋       1.0
# 2             길동  21   수건       3.0
# 3             영수  30   생수       2.0
# 3             영수  30   수건       2.0
# 4             수민  31   생수       2.0
# 5             동건  18  NaN       NaN

cust1.join(order1, how='inner')
# >>> 출력:
#             name  나이 item  quantity
# customer_id                        
# 1             영희  20   치약         1
# 1             영희  20   칫솔         2
# 1             영희  20   치약         3
# 2             길동  21  이어폰         1
# 2             길동  21  헤드셋         1
# 2             길동  21   수건         3
# 3             영수  30   생수         2
# 3             영수  30   수건         2
# 4             수민  31   생수         2

# ## 실습1] 가장 많이 팔린 아이템은?
# 
# > 결과=> 수건 5

# 김태규
customer = pd.DataFrame({'customer_id' : np.arange(6),
                    'name' : ['철수'"", '영희', '길동', '영수', '수민', '동건'],
                    '나이' : [40, 20, 21, 30, 31, 18]})
customer
orders = pd.DataFrame({'customer_id' : [1, 1, 2, 2, 2, 3, 3, 1, 4, 9],
                    'item' : ['치약', '칫솔', '이어폰', '헤드셋', '수건', '생수', '수건', '치약', '생수', '케이스'],
                    'quantity' : [1, 2, 1, 1, 3, 2, 2, 3, 2, 1]})
cust1 = customer.set_index('customer_id')
order1 = orders.set_index('customer_id')

oth = pd.merge(cust1, order1, left_index=True, right_index=True)
order2 = oth.set_index('item')
order2.groupby(level=0)['quantity'].sum().sort_values(ascending=False)[:1]
# >>> 출력:
# item
# 수건    5
# Name: quantity, dtype: int64

# 변희언
# pd.pivot_table(order1, index='item', aggfunc='sum').max()
# >>> 출력:
# quantity    5
# dtype: int64

# 김성제
pd.pivot_table(orders, 'quantity','item',aggfunc=sum).sort_values(by='quantity').iloc[-1]
# >>> 출력:
# <ipython-input-322-737a7a40fc93>:2: FutureWarning: The provided callable <built-in function sum> is currently using DataFrameGroupBy.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
#   pd.pivot_table(orders, 'quantity','item',aggfunc=sum).sort_values(by='quantity').iloc[-1]
# quantity    5
# Name: 수건, dtype: int64

# 이은솔
total_order = order1.groupby('item')['quantity'].sum()
# total_order
total_order[total_order == total_order.max()]
# >>> 출력:
# item
# 수건    5
# Name: quantity, dtype: int64

# ## 실습2] 영희가 가장 많이 구매한 아이템은?
# 
# > 결과 => 치약 4

# 김정호
new = pd.merge(customer,orders,on = 'customer_id',how = 'outer')
new.groupby(['name','item'])['quantity'].sum()['영희'].sort_values(ascending = False)[:1]
# >>> 출력:
# item
# 치약    4.0
# Name: quantity, dtype: float64
