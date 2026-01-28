# # ✅롤(League Of Legend) 데이터 분석

# # 데이터 셋 준비

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ## pandas 로 csv 파일 읽기

base_path = r'/content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)'

file_name = r'high_diamond_ranked_10min.csv'

file_path = os.path.join(base_path, file_name)
df = pd.read_csv(file_path)
df
# >>> 출력:
#           gameId  blueWins  blueWardsPlaced  blueWardsDestroyed  \
# 0     4519157822         0               28                   2   
# 1     4523371949         0               12                   1   
# 2     4521474530         0               15                   0   
# 3     4524384067         0               43                   1   
# 4     4436033771         0               75                   4   
# ...          ...       ...              ...                 ...   
# 9874  4527873286         1               17                   2   
# 9875  4527797466         1               54                   0   
# 9876  4527713716         0               23                   1   
# 9877  4527628313         0               14                   4   
# 9878  4523772935         1               18                   0   
# 
#       blueFirstBlood  blueKills  blueDeaths  blueAssists  blueEliteMonsters  \
# 0                  1          9           6           11                  0   

# ## 데이터 소개
# - 이번 주제는 League of Legends Diamond Ranked Games (10 min) 데이터셋을 사용합니다.
# 
#     
# - 다음 1개의 csv 파일을 사용합니다.
#     - high_diamond_ranked_10min.csv
#     
#     
# - 각 파일의 컬럼은 아래와 같습니다.
#     - gameId: 게임 판의 고유 ID
#     - **blueWins**: 블루팀의 승리 여부 (0: 패배, 1: 승리)  <-- ⭐타겟값
#     - xxxWardsPlaced: xxx팀에서 설치한 와드의 수
#     - xxxWardsDestroyed: xxx팀에서 파괴한 와드의 수
#     - xxxFirstBlood: xxx팀의 첫번째 킬 달성 여부
#     - xxxKills: xxx팀의 킬 수
#     - xxxDeaths: xxx팀의 죽음 수
#     - xxxAssists: xxx팀의 어시스트 수
#     - xxxEliteMonsters: xxx팀이 죽인 엘리트 몬스터 수
#     - xxxDragons: xxx팀이 죽인 용의 수
#     - xxxHeralds: xxx팀이 죽인 전령의 수
#     - xxxTowersDestroyed: xxx팀이 파괴한 탑의 수
#     - xxxTotalGold: xxx팀의 전체 획득 골드
#     - xxxAvgLevel: xxx팀의 평균 레벨
#     - xxxTotalExperience: xxx팀의 총 경험치 획득량
#     - xxxTotalMinionsKilled: xxx팀의 총 미니언 킬 수
#     - xxxTotalJungleMinionsKilled: xxx팀의 총 정글 미니언 킬 수
#     - xxxGoldDiff: xxx팀과 다른 팀 간의 골드 획득량 차이
#     - xxxExperienceDiff: xxx팀과 다른 팀과의 경험치 획득량 차이
#     - xxxCSPerMin: xxx팀의 분당 CS 스코어
#     - xxxGoldPerMin: xxx팀의 분당 골드 획득량
#       
#         
# - 데이터 출처: https://www.kaggle.com/bobbyscience/league-of-legends-diamond-ranked-games-10-min

# # EDA, 기초 통계 분석

df.head()
# >>> 출력:
#        gameId  blueWins  blueWardsPlaced  blueWardsDestroyed  blueFirstBlood  \
# 0  4519157822         0               28                   2               1   
# 1  4523371949         0               12                   1               0   
# 2  4521474530         0               15                   0               0   
# 3  4524384067         0               43                   1               0   
# 4  4436033771         0               75                   4               0   
# 
#    blueKills  blueDeaths  blueAssists  blueEliteMonsters  blueDragons  ...  \
# 0          9           6           11                  0            0  ...   
# 1          5           5            5                  0            0  ...   
# 2          7          11            4                  1            1  ...   
# 3          4           5            5                  1            0  ...   
# 4          6           6            6                  0            0  ...   
# 
#    redTowersDestroyed  redTotalGold  redAvgLevel  redTotalExperience  \

df.columns
# >>> 출력:
# Index(['gameId', 'blueWins', 'blueWardsPlaced', 'blueWardsDestroyed',
#        'blueFirstBlood', 'blueKills', 'blueDeaths', 'blueAssists',
#        'blueEliteMonsters', 'blueDragons', 'blueHeralds',
#        'blueTowersDestroyed', 'blueTotalGold', 'blueAvgLevel',
#        'blueTotalExperience', 'blueTotalMinionsKilled',
#        'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
#        'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
#        'redFirstBlood', 'redKills', 'redDeaths', 'redAssists',
#        'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed',
#        'redTotalGold', 'redAvgLevel', 'redTotalExperience',
#        'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redGoldDiff',
#        'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin'],
#       dtype='object')

df[['blueGoldDiff', 'redGoldDiff']]
# >>> 출력:
#       blueGoldDiff  redGoldDiff
# 0              643         -643
# 1            -2908         2908
# 2            -1172         1172
# 3            -1321         1321
# 4            -1004         1004
# ...            ...          ...
# 9874          2519        -2519
# 9875           782         -782
# 9876         -2416         2416
# 9877          -839          839
# 9878           927         -927
# 
# [9879 rows x 2 columns]

df[['blueAvgLevel']]
# >>> 출력:
#       blueAvgLevel
# 0              6.6
# 1              6.6
# 2              6.4
# 3              7.0
# 4              7.0
# ...            ...
# 9874           7.2
# 9875           7.2
# 9876           7.0
# 9877           6.6
# 9878           7.0
# 
# [9879 rows x 1 columns]

df[['blueAvgLevel']].max()
# >>> 출력:
# blueAvgLevel    8.0
# dtype: float64

# bludGoldDiff :
#     아마 regGoldDiff 와 부호가 다른 관계에 있을듯.  즉 중복된 정보일 가능성이 크다.
#     이 경우 한가지만 사용해도 될것이다.
#     multi-core linearity 를 가지는 값들은 drop 해서 제거할 필요가 있어 보인다.

df.info()
# >>> 출력:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 9879 entries, 0 to 9878
# Data columns (total 40 columns):
#  #   Column                        Non-Null Count  Dtype  
# ---  ------                        --------------  -----  
#  0   gameId                        9879 non-null   int64  
#  1   blueWins                      9879 non-null   int64  
#  2   blueWardsPlaced               9879 non-null   int64  
#  3   blueWardsDestroyed            9879 non-null   int64  
#  4   blueFirstBlood                9879 non-null   int64  
#  5   blueKills                     9879 non-null   int64  
#  6   blueDeaths                    9879 non-null   int64  
#  7   blueAssists                   9879 non-null   int64  
#  8   blueEliteMonsters             9879 non-null   int64  
#  9   blueDragons                   9879 non-null   int64  

df.describe()
# >>> 출력:
#              gameId     blueWins  blueWardsPlaced  blueWardsDestroyed  \
# count  9.879000e+03  9879.000000      9879.000000         9879.000000   
# mean   4.500084e+09     0.499038        22.288288            2.824881   
# std    2.757328e+07     0.500024        18.019177            2.174998   
# min    4.295358e+09     0.000000         5.000000            0.000000   
# 25%    4.483301e+09     0.000000        14.000000            1.000000   
# 50%    4.510920e+09     0.000000        16.000000            3.000000   
# 75%    4.521733e+09     1.000000        20.000000            4.000000   
# max    4.527991e+09     1.000000       250.000000           27.000000   
# 
#        blueFirstBlood    blueKills   blueDeaths  blueAssists  \
# count     9879.000000  9879.000000  9879.000000  9879.000000   
# mean         0.504808     6.183925     6.137666     6.645106   
# std          0.500002     3.011028     2.933818     4.064520   
# min          0.000000     0.000000     0.000000     0.000000   

df.describe()[['blueDragons', 'redDragons']]
# >>> 출력:
#        blueDragons   redDragons
# count  9879.000000  9879.000000
# mean      0.361980     0.413098
# std       0.480597     0.492415
# min       0.000000     0.000000
# 25%       0.000000     0.000000
# 50%       0.000000     0.000000
# 75%       1.000000     1.000000
# max       1.000000     1.000000

# ## 각 컬럼의 corr 을 히트맵으로 시각화

df.corr()
# >>> 출력:
#                                 gameId  blueWins  blueWardsPlaced  \
# gameId                        1.000000  0.000985         0.005361   
# blueWins                      0.000985  1.000000         0.000087   
# blueWardsPlaced               0.005361  0.000087         1.000000   
# blueWardsDestroyed           -0.012057  0.044247         0.034447   
# blueFirstBlood               -0.011577  0.201769         0.003228   
# blueKills                    -0.038993  0.337358         0.018138   
# blueDeaths                   -0.013160 -0.339297        -0.002612   
# blueAssists                  -0.023329  0.276685         0.033217   
# blueEliteMonsters             0.016599  0.221944         0.019892   
# blueDragons                   0.008962  0.213768         0.017676   
# blueHeralds                   0.015551  0.092385         0.010104   
# blueTowersDestroyed          -0.007467  0.115566         0.009462   
# blueTotalGold                -0.033754  0.417213         0.019725   
# blueAvgLevel                 -0.040956  0.357820         0.034349   

sns.heatmap(df.corr())
# >>> 출력:
# <Axes: >

fig = plt.figure(figsize=(10, 10))
sns.heatmap(df.corr())
# >>> 출력:
# <Axes: >

fig = plt.figure(figsize=(4, 10))
sns.heatmap(df.corr()[['blueWins']], annot=True)
# >>> 출력:
# <Axes: >

df.corr()
# >>> 출력:
#                                 gameId  blueWins  blueWardsPlaced  \
# gameId                        1.000000  0.000985         0.005361   
# blueWins                      0.000985  1.000000         0.000087   
# blueWardsPlaced               0.005361  0.000087         1.000000   
# blueWardsDestroyed           -0.012057  0.044247         0.034447   
# blueFirstBlood               -0.011577  0.201769         0.003228   
# blueKills                    -0.038993  0.337358         0.018138   
# blueDeaths                   -0.013160 -0.339297        -0.002612   
# blueAssists                  -0.023329  0.276685         0.033217   
# blueEliteMonsters             0.016599  0.221944         0.019892   
# blueDragons                   0.008962  0.213768         0.017676   
# blueHeralds                   0.015551  0.092385         0.010104   
# blueTowersDestroyed          -0.007467  0.115566         0.009462   
# blueTotalGold                -0.033754  0.417213         0.019725   
# blueAvgLevel                 -0.040956  0.357820         0.034349   

# ### multi-colinearity 문제

"""
↑ [관찰]
 + 혹은 - 로 나타나는 관계가 있다

   blueKills 와 redKills
   blueDeaths 와 redDeaths


즉, 정확히 반대로 나타나는 데이터들
이 상태에서 LinearRegression 이나 LogisticRegression 을 하게 되면
multi-collinearity 가 발생할수 있다.

즉 blueKills 와 redKills 과 음/양 완전히 반대인 상태에서 학습이 양쪽 방향으로 진행되면서
원하는 대로의 학습이 진행되지 않는 문제들이 발생할수 있다.

이런 경우 multi-colinearity 를 배제하여 훈련시킬 필요가 있다.

"""
None

# # 각 컬럼의 승리여부 관계 분석

# ## 수치형 데이터 분석

df.columns
# >>> 출력:
# Index(['gameId', 'blueWins', 'blueWardsPlaced', 'blueWardsDestroyed',
#        'blueFirstBlood', 'blueKills', 'blueDeaths', 'blueAssists',
#        'blueEliteMonsters', 'blueDragons', 'blueHeralds',
#        'blueTowersDestroyed', 'blueTotalGold', 'blueAvgLevel',
#        'blueTotalExperience', 'blueTotalMinionsKilled',
#        'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
#        'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
#        'redFirstBlood', 'redKills', 'redDeaths', 'redAssists',
#        'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed',
#        'redTotalGold', 'redAvgLevel', 'redTotalExperience',
#        'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redGoldDiff',
#        'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin'],
#       dtype='object')

sns.histplot(data=df, x='blueGoldDiff', hue='blueWins', palette='RdBu', kde=True)
# >>> 출력:
# <Axes: xlabel='blueGoldDiff', ylabel='Count'>

sns.histplot(data=df, x='blueKills', hue='blueWins', palette='RdBu', kde=True, bins=8)
# >>> 출력:
# <Axes: xlabel='blueKills', ylabel='Count'>

sns.jointplot(data=df, x='blueKills', y='blueGoldDiff', hue='blueWins')
# >>> 출력:
# <seaborn.axisgrid.JointGrid at 0x7fb048351950>

sns.jointplot(data=df, x='blueExperienceDiff', y='blueGoldDiff', hue='blueWins')
# >>> 출력:
# <seaborn.axisgrid.JointGrid at 0x7fb0480a0950>

sns.jointplot(data=df, x='blueGoldDiff', y='blueGoldDiff', hue='blueWins')
# >>> 출력:
# <seaborn.axisgrid.JointGrid at 0x7fb040c35450>

# ## 분류형 데이터 분석

df.blueDragons.value_counts()
# >>> 출력:
# blueDragons
# 0    6303
# 1    3576
# Name: count, dtype: int64

sns.countplot(data=df, x='blueDragons', hue='blueWins', palette='RdBu')
# >>> 출력:
# <Axes: xlabel='blueDragons', ylabel='count'>

sns.countplot(data=df, x='redDragons', hue='blueWins', palette='RdBu')
# >>> 출력:
# <Axes: xlabel='redDragons', ylabel='count'>

sns.countplot(data=df, x='blueFirstBlood', hue='blueWins', palette='RdBu')
# >>> 출력:
# <Axes: xlabel='blueFirstBlood', ylabel='count'>

# # 모델 학습을 위한 데이터 전처리

# ## 학습에 불필요한 컬럼을 제거 .
# - **df** 에서 다음의 불필요한 컬럼들을 제거합니다
# >'gameId', 'redFirstBlood', 'redKills', 'redDeaths','redTotalGold', 'redTotalExperience', 'redGoldDiff','redExperienceDiff'
# 
# - 컬럼이 제거된 df 의 df.head() 출력 하기
# - 컬럼이 제거된 df 의 column 개수 출력 하기

df.columns
# >>> 출력:
# Index(['gameId', 'blueWins', 'blueWardsPlaced', 'blueWardsDestroyed',
#        'blueFirstBlood', 'blueKills', 'blueDeaths', 'blueAssists',
#        'blueEliteMonsters', 'blueDragons', 'blueHeralds',
#        'blueTowersDestroyed', 'blueTotalGold', 'blueAvgLevel',
#        'blueTotalExperience', 'blueTotalMinionsKilled',
#        'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
#        'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
#        'redFirstBlood', 'redKills', 'redDeaths', 'redAssists',
#        'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed',
#        'redTotalGold', 'redAvgLevel', 'redTotalExperience',
#        'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redGoldDiff',
#        'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin'],
#       dtype='object')

df.drop(['gameId', 'redFirstBlood', 'redKills', 'redDeaths',
         'redTotalGold', 'redTotalExperience',
         'redGoldDiff','redExperienceDiff'], axis=1, inplace=True)

df.columns
# >>> 출력:
# Index(['blueWins', 'blueWardsPlaced', 'blueWardsDestroyed', 'blueFirstBlood',
#        'blueKills', 'blueDeaths', 'blueAssists', 'blueEliteMonsters',
#        'blueDragons', 'blueHeralds', 'blueTowersDestroyed', 'blueTotalGold',
#        'blueAvgLevel', 'blueTotalExperience', 'blueTotalMinionsKilled',
#        'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
#        'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
#        'redAssists', 'redEliteMonsters', 'redDragons', 'redHeralds',
#        'redTowersDestroyed', 'redAvgLevel', 'redTotalMinionsKilled',
#        'redTotalJungleMinionsKilled', 'redCSPerMin', 'redGoldPerMin'],
#       dtype='object')

df.head()
# >>> 출력:
#    blueWins  blueWardsPlaced  blueWardsDestroyed  blueFirstBlood  blueKills  \
# 0         0               28                   2               1          9   
# 1         0               12                   1               0          5   
# 2         0               15                   0               0          7   
# 3         0               43                   1               0          4   
# 4         0               75                   4               0          6   
# 
#    blueDeaths  blueAssists  blueEliteMonsters  blueDragons  blueHeralds  ...  \
# 0           6           11                  0            0            0  ...   
# 1           5            5                  0            0            0  ...   
# 2          11            4                  1            1            0  ...   
# 3           5            5                  1            0            1  ...   
# 4           6            6                  0            0            0  ...   
# 
#    redAssists  redEliteMonsters  redDragons  redHeralds  redTowersDestroyed  \

len(df.columns)
# >>> 출력:
# 32

# ## 수치형 데이터 표준화하기
# - df에서 수치형 데이터만 모아서 **X_num** 변수에 담기 (DataFrame)
# - df에서 분류형 데이터만 모아서 **X_cat** 변수에 담기 (DataFrame)
#     - ※ dtype 이 숫자타입이더라도 값의 종류가 2개 이하이면 분류형으로 담으세요
# 
# 
# - **StandardScaler** 를 사용하여 **X_num** 의 표준화를 진행한뒤 결과를 DataFrame 으로 바꾸어 변수 **X_scaled** 에 저장.   스케일링을 진행한 Scaler 객체는 **scaler** 변수에 담기
# 
# 
# - 데이터 **X** <= X_scaled 와 X_cat 을 합한 데이터 담기
# - 타겟값 **y** <= 'blueWins'  담기
# 
# 
# - X.head() 출력
# - X.describe() 출력

from sklearn.preprocessing import StandardScaler

df.columns
# >>> 출력:
# Index(['blueWins', 'blueWardsPlaced', 'blueWardsDestroyed', 'blueFirstBlood',
#        'blueKills', 'blueDeaths', 'blueAssists', 'blueEliteMonsters',
#        'blueDragons', 'blueHeralds', 'blueTowersDestroyed', 'blueTotalGold',
#        'blueAvgLevel', 'blueTotalExperience', 'blueTotalMinionsKilled',
#        'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
#        'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
#        'redAssists', 'redEliteMonsters', 'redDragons', 'redHeralds',
#        'redTowersDestroyed', 'redAvgLevel', 'redTotalMinionsKilled',
#        'redTotalJungleMinionsKilled', 'redCSPerMin', 'redGoldPerMin'],
#       dtype='object')

X_num = df[['blueWardsPlaced', 'blueWardsDestroyed',
       'blueKills', 'blueDeaths', 'blueAssists', 'blueEliteMonsters',
       'blueTowersDestroyed', 'blueTotalGold',
       'blueAvgLevel', 'blueTotalExperience', 'blueTotalMinionsKilled',
       'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
       'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
       'redAssists', 'redEliteMonsters',
       'redTowersDestroyed', 'redAvgLevel', 'redTotalMinionsKilled',
       'redTotalJungleMinionsKilled', 'redCSPerMin', 'redGoldPerMin']]

X_cat = df[['blueFirstBlood', 'blueDragons', 'blueHeralds', 'redDragons', 'redHeralds']]

scaler = StandardScaler()
scaler.fit(X_num)
X_scaled = scaler.transform(X_num)
X_scaled = pd.DataFrame(X_scaled, index=X_num.index, columns=X_num.columns)

X = pd.concat([X_scaled, X_cat], axis=1)
y = df['blueWins']

X
# >>> 출력:
#       blueWardsPlaced  blueWardsDestroyed  blueKills  blueDeaths  blueAssists  \
# 0            0.316996           -0.379275   0.935301   -0.046926     1.071495   
# 1           -0.570992           -0.839069  -0.393216   -0.387796    -0.404768   
# 2           -0.404494           -1.298863   0.271042    1.657424    -0.650812   
# 3            1.149484           -0.839069  -0.725346   -0.387796    -0.404768   
# 4            2.925460            0.540312  -0.061087   -0.046926    -0.158724   
# ...               ...                 ...        ...         ...          ...   
# 9874        -0.293496           -0.379275   0.271042   -0.728666    -0.404768   
# 9875         1.759976           -1.298863  -0.061087   -0.728666     0.333364   
# 9876         0.039499           -0.839069  -0.061087    0.293944    -0.404768   
# 9877        -0.459994            0.540312  -1.389604   -1.069536    -0.896856   
# 9878        -0.237997           -1.298863  -0.061087   -0.046926    -0.404768   
# 
#       blueEliteMonsters  blueTowersDestroyed  blueTotalGold  blueAvgLevel  \
# 0             -0.879231            -0.210439       0.460179     -1.035635   

X.describe()
# >>> 출력:
#        blueWardsPlaced  blueWardsDestroyed     blueKills    blueDeaths  \
# count     9.879000e+03        9.879000e+03  9.879000e+03  9.879000e+03   
# mean     -2.876982e-17        5.034719e-18  1.125619e-16 -1.179563e-16   
# std       1.000051e+00        1.000051e+00  1.000051e+00  1.000051e+00   
# min      -9.594869e-01       -1.298863e+00 -2.053863e+00 -2.092146e+00   
# 25%      -4.599937e-01       -8.390689e-01 -7.253456e-01 -7.286663e-01   
# 50%      -3.489952e-01        8.051859e-02 -6.108705e-02 -4.692613e-02   
# 75%      -1.269983e-01        5.403123e-01  6.031716e-01  6.348140e-01   
# max       1.263783e+01        1.111557e+01  5.252982e+00  5.406995e+00   
# 
#         blueAssists  blueEliteMonsters  blueTowersDestroyed  blueTotalGold  \
# count  9.879000e+03       9.879000e+03         9.879000e+03   9.879000e+03   
# mean  -1.111234e-16       3.308530e-17        -2.733133e-17  -2.366318e-16   
# std    1.000051e+00       1.000051e+00         1.000051e+00   1.000051e+00   
# min   -1.634988e+00      -8.792310e-01        -2.104390e-01  -3.760305e+00   

# ## 3. 최적 파라미터 찾기
# 
# - 위에서 만든 X, y 를 LogisticRegression 모델에 학습시키기 위한 최적의 파라미터를 찾는다
# - 다음 2개의 파라미터 값의 grid 를 다음과 같이 주고 최적값을 찾아낸다
#     - 'max_iter' : [100, 200, 300]
#     - 'C' : [0.5, 1.0, 1.5, 2.0]
#     
#     
# - GridSearchCV 사용: 8분할(8-fold)하여 Cross-validation 수행
# 
# 
# - 결과의 최적점수 출력하기
# - 결과의 최적 파라미터 (dict)를 **best_params** 변수에 담고 출력하기
# - 결과의 cv_results_ 값을 DataFrame 으로 출력하기

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_iter' : [100, 200, 300],
    'C' : [0.5, 1.0, 1.5, 2.0]
}

gs = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, cv=8)
result = gs.fit(X, y)

print('최적점수:', result.best_score_)
print('최적파라미터:', result.best_params_)
print(gs.best_estimator_)
# >>> 출력:
# 최적점수: 0.7311460377036594
# 최적파라미터: {'C': 2.0, 'max_iter': 100}
# LogisticRegression(C=2.0)

pd.DataFrame(result.cv_results_)
# >>> 출력:
#     mean_fit_time  std_fit_time  mean_score_time  std_score_time  param_C  \
# 0        0.095491      0.029469         0.007532        0.003437      0.5   
# 1        0.091342      0.023909         0.006034        0.002881      0.5   
# 2        0.141353      0.079577         0.013021        0.006616      0.5   
# 3        0.101390      0.041047         0.006886        0.004499      1.0   
# 4        0.054280      0.007782         0.004229        0.001063      1.0   
# 5        0.060587      0.014430         0.003697        0.000079      1.0   
# 6        0.053770      0.007941         0.003668        0.000231      1.5   
# 7        0.049930      0.003946         0.004045        0.001248      1.5   
# 8        0.048193      0.001803         0.003592        0.000136      1.5   
# 9        0.048915      0.001103         0.003648        0.000149      2.0   
# 10       0.053596      0.006850         0.003852        0.000667      2.0   
# 11       0.049368      0.001452         0.003714        0.000135      2.0   
# 
#     param_max_iter                       params  split0_test_score  \

# ## 4. 학습데이터와 테스트데이터 분리하기
# - 학습용(train) : 테스트용(test) = 7:3 비율로 분리하기
# - 분리된 각 데이터와 레이블은 아래 변수명으로 담습니다.
#     - 학습용 데이터 **X_train**
#     - 테스트용 데이터 **X_test**
#     - 학습용 레이블 **y_train**
#     - 테스트용 레이블 **y_test**
#     
#     
# - shuffle은 랜덤으로 진행되되 고정된 랜덤형태로 분리될수 있도록 한다
# - 분리된 '학습용 데이터개수' 와 '테스트용 데이터 개수' 출력하기

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=42)

len(X_train), len(X_test)
# >>> 출력:
# (6915, 2964)

# ## 5. Classification 모델 생성/학습하기
# - LogisticRegression 모델로 학습합니다
# - 위 에서 찾은 최적 파라미터 적용하기
# - 모델 변수명 **model_lr**

result.best_params_
# >>> 출력:
# {'C': 2.0, 'max_iter': 100}

model_lr = LogisticRegression(**result.best_params_)
model_lr.fit(X_train, y_train)
# >>> 출력:
# LogisticRegression(C=2.0)

# ## 6. 교차검증 해보기
# - 원 변수 X, y 에 대한 교차검증 (cross validation) 하기
# - 4-fold(분할) cross validation 진행하기
# 
# 
# - 교차검증의 정확도 score 출력
# 
# - 교차검증의 정확도 score 의 평균과 표준편차 출력

from sklearn.model_selection import cross_val_score, cross_validate

scores = cross_val_score(model_lr, X, y, cv=4)

scores
# >>> 출력:
# array([0.7340081 , 0.7242915 , 0.73198381, 0.73106521])

print('교차검증 정확도', np.mean(scores), '+/-', np.std(scores))
# >>> 출력:
# 교차검증 정확도 0.7303371523490472 +/- 0.0036492227147931224

# ## 7-1. 모델 학습 결과 평가하기
# - 테스트 데이터에 대한 예측값을 내어보고
# - 예측값에 대한  classification_report 출력하기

from sklearn.metrics import classification_report

pred = model_lr.predict(X_test)
print(classification_report(y_test, pred))
# >>> 출력:
#               precision    recall  f1-score   support
# 
#            0       0.73      0.74      0.73      1480
#            1       0.73      0.72      0.73      1484
# 
#     accuracy                           0.73      2964
#    macro avg       0.73      0.73      0.73      2964
# weighted avg       0.73      0.73      0.73      2964

# ## 7-2 ROC curve 그리기

from sklearn.metrics import RocCurveDisplay

fig = plt.figure()
ax = fig.gca()
RocCurveDisplay.from_estimator(model_lr, X_test, y_test, ax=ax)
# >>> 출력:
# <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x7fb036a6b550>

# ## 7-3 중요한 feature 고르기
# - XGBClassifier 를 사용하여 승패에 가장 중요한 가장 큰 영향을 끼친 feature 하나를 찾아내세요

from xgboost import XGBClassifier

model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
# >>> 출력:
# XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=None, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=None, max_leaves=None,
#               min_child_weight=None, missing=nan, monotone_constraints=None,
#               multi_strategy=None, n_estimators=None, n_jobs=None,
#               num_parallel_tree=None, random_state=None, ...)

model_xgb.feature_importances_
# >>> 출력:
# array([0.02510191, 0.0246698 , 0.02823661, 0.02869373, 0.02876465,
#        0.03396531, 0.02521414, 0.03058143, 0.02946877, 0.03030336,
#        0.02770249, 0.02854351, 0.22107595, 0.04199795, 0.        ,
#        0.        , 0.02810658, 0.02955643, 0.02711246, 0.03769168,
#        0.00919579, 0.02810738, 0.02811058, 0.02513596, 0.        ,
#        0.02782114, 0.02732326, 0.03913873, 0.02778825, 0.03374451,
#        0.0268477 ], dtype=float32)

plt.bar(X.columns, model_xgb.feature_importances_)
plt.xticks(rotation=90)
plt.show()
