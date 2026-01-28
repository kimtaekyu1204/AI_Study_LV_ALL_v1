# # 탐색적 데이터 분석(EDA) - 심장병 데이터

# **EDA(Exploratory Data Analysis, 탐색적 데이터 분석)**
# 
# - EDA(Exploratory Data Analysis, 탐색적 데이터 분석)는 벨연구소의 수학자 ‘존 튜키’가 개발한 데이터분석 과정에 대한 개념
# - 데이터를 분석하고 결과를 내는 과정에 있어서 지속적으로 해당 데이터에 대한 ‘탐색과 이해’를 기본으로 가져야 한다는 것을 의미
# - 다양한 측면으로 데이터의 탐색적 분석을 시도해보아야 한다
#   - 데이터의 각 column들과 row의 의미및 연관성에 대한 이해
#   - 결측치 처리 및 데이터필터링
#   - 시각화

# # 의료 데이터 셋

# ### 의료 데이터의 수집
# - 의료데이터는 어떻게 '수집' 이 되느냐 부터가 굉장히 중요
# - 우리나라는 2020.1월에 새로운 법이 통과!  => 데이터 3법
#     - 그중에 가장 핵심은 개인정보에 대한 것
#     - 의료정보는 굉장히 민감한 개인정보들이 많기 때문에, 수집이 되더라도,  사람들에게 전파되기 어려운 법적인 특성을 있었습니다.    - '가명정보' 를 사용할수 있게 허용됨.
#     - '실명정보' 에서 개인식별 정보를 비식별화 (de-identification) 과정을 거쳐서 만들어진 데이터, 쉽게 말해서 가명정보화 하여 일반적으로 사용할수 있는 데이터화 하여, 이후에 관련분야 민간 연구자들에게도 전달될수 있도록 하여 제공하도록 함 --> 데이터3법을 통해 의료데이터에 생긴 핵심적인 변화.
#     - 2020년동안 많은 분들이 '바이오 데이터 분석가' 라든지 '4차 산업 혁명' 에 이어 '바이오 데이터' 관련 된 것들이 뜨기 시작함.  (법적인 배경이 있었던 것이다)
#     - 교육현장, 민간 사업체 등에서 공공데이터로 제공이 됨.
#     - 데이터를 생산해서 제공하는 의료기관이라든지, 연구기관들에서 더 인센티브를 주는 방식으로 하여 데이터 생산을 더더욱 독력하는 중.
#     - 그리하여 의료데이터 수집이 가속회 되고 있고, 그에 따라 의료데이터를 다룰 인력들에 대한 수요가 점점 늘어나고 있다.
#         - 기본적으로 가공해서 업로딩 하는 데이터 엔지니어
#         - 데이터분석, 모델링 활용하는 사람들

# ### 의료 데이터 분석의 현재
# 
# - 우리나라는 이제 걸음마 단계.  (그동안 법적 제약)
# - 그 이전에는 주로 연구 되었던 것들이 의료 영상 같은 것들. MRI, CT (3D 스캔)
#     - 병 판명, 등에 사용
# - 아직 분석의 결과는 이렇다 할만한게 없다.
# - 의료 데이터를 적극 사용해왔던 국가들
#     - 핀란드 : 환자들에게 전자포털 제공, 자신의 데이터를 제공할지 안할지 관리 가능케 함. 인프라 잘 구성됨.
#     - 덴마크 : 국가에서 포털 구축, 진료기록 99% 에게 환자 주치의에게 전달됨. 과거 기록들을 통해 더 좋은 진료서비스, 개인 병력, 과거 어떤 치료 진료 내역등이 공유됨.
#     - 영국 : 전자 의무 기록 활용
#     - 중국 : 의료 융합 추진, 병원 공실률 down 시켜 의료 인프라 효율적 활용.  특정 병원에 사람이 쏠리지 않도록 함.  국가적으로 컨트롤 함.
#     - 미국 : 주로 연구및 연구 발표 활발. 민간단체에서 임상데이터 수집 가능.  연구가능한데이터 많이 취득함.  의료 빅데이터 관련해서만 논문이 100건 이상 나오는 중.
#     - 대한민국 : 우리는 이제 걸음마 시작단계, 할일이 많다.  다른 관점에서 보면, 의료 데이터 다루는 것이 국내에선 수요가 계속 늘어날 것이다!  그래서 이 분야 진입은 전망이 좋다.

# # 1.데이터셋 준비

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ## pandas 로 csv 파일 읽기

base_path = r'/content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)'

file_name = r'heart_failure_clinical_records_dataset.csv'

file_path = os.path.join(base_path, file_name)
df = pd.read_csv(file_path)
df
# >>> 출력:
#       age  anaemia  creatinine_phosphokinase  diabetes  ejection_fraction  \
# 0    75.0        0                       582         0                 20   
# 1    55.0        0                      7861         0                 38   
# 2    65.0        0                       146         0                 20   
# 3    50.0        1                       111         0                 20   
# 4    65.0        1                       160         1                 20   
# ..    ...      ...                       ...       ...                ...   
# 294  62.0        0                        61         1                 38   
# 295  55.0        0                      1820         0                 38   
# 296  45.0        0                      2060         1                 60   
# 297  45.0        0                      2413         0                 38   
# 298  50.0        0                       196         0                 45   
# 
#      high_blood_pressure  platelets  serum_creatinine  serum_sodium  sex  \
# 0                      1  265000.00               1.9           130    1   

# ## 데이터 소개
# 1. 이번 주제는 Heart Failure Prediction 데이터셋을 사용합니다.
#     
# 1. 다음 1개의 csv 파일을 사용합니다.
#     heart_failure_clinical_records_dataset.csv
#     
# 1. 각 파일의 컬럼은 아래와 같습니다.
#     - age: 환자의 나이
#     - anaemia: 환자의 빈혈증 여부 (0: 정상, 1: 빈혈)
#     - creatinine_phosphokinase: 크레아틴키나제 검사 결과
#     - diabetes: 당뇨병 여부 (0: 정상, 1: 당뇨)
#     - ejection_fraction: 박출계수 (%)
#     - high_blood_pressure: 고혈압 여부 (0: 정상, 1: 고혈압)
#     - platelets: 혈소판 수 (kiloplatelets/mL)
#     - serum_creatinine: 혈중 크레아틴 레벨 (mg/dL)
#     - serum_sodium: 혈중 나트륨 레벨 (mEq/L)
#     - sex: 성별 (0: 여성, 1: 남성)
#     - smoking: 흡연 여부 (0: 비흡연, 1: 흡연)
#     - time: 관찰 기간 (일)
#     - DEATH_EVENT: 사망 여부 (0: 생존, 1: 사망)
#     
#     
#     
# 1. 데이터 URL: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

# # EDA 및 데이터 기초 통계 분석

# ## 각 컬럼 분석하기

df.head()
# >>> 출력:
#     age  anaemia  creatinine_phosphokinase  diabetes  ejection_fraction  \
# 0  75.0        0                       582         0                 20   
# 1  55.0        0                      7861         0                 38   
# 2  65.0        0                       146         0                 20   
# 3  50.0        1                       111         0                 20   
# 4  65.0        1                       160         1                 20   
# 
#    high_blood_pressure  platelets  serum_creatinine  serum_sodium  sex  \
# 0                    1  265000.00               1.9           130    1   
# 1                    0  263358.03               1.1           136    1   
# 2                    0  162000.00               1.3           129    1   
# 3                    0  210000.00               1.9           137    1   
# 4                    0  327000.00               2.7           116    0   
# 
#    smoking  time  DEATH_EVENT  

# head(), tail()
# info()
# describe()
# shape
#...
# 각 컬럼들이 어떠한 값들, 어떠한 범위를 갖고 있는지 확인

df.info()
# >>> 출력:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 299 entries, 0 to 298
# Data columns (total 13 columns):
#  #   Column                    Non-Null Count  Dtype  
# ---  ------                    --------------  -----  
#  0   age                       299 non-null    float64
#  1   anaemia                   299 non-null    int64  
#  2   creatinine_phosphokinase  299 non-null    int64  
#  3   diabetes                  299 non-null    int64  
#  4   ejection_fraction         299 non-null    int64  
#  5   high_blood_pressure       299 non-null    int64  
#  6   platelets                 299 non-null    float64
#  7   serum_creatinine          299 non-null    float64
#  8   serum_sodium              299 non-null    int64  
#  9   sex                       299 non-null    int64  

"""
info() 를 통해 확인할것 : 결측치 여부 (missing value), dType,

총 299개의 ROW가 있고 인덱스 0 ~ 298
Non-null 이 '전부' 299  .  결측치는 없슴을 확인했다
매우 클린한 데이터 셋이다.

총 3개의 float 와 10개의 int 타입으로 dtype 이 설정되었다.
그러나,

비록 dtype 은 int 타입이나, '분류형'으로 보아야 할 데이터들이 있는거 같다.
ex) DEATH_EVENT, smoking, sex, anaemia, diabetes, high_blood_pressure..

"""
None

"""
age : 환자의 나이 (숫자형) <- 정수값일줄 알았는데 float 타입으로 읽혀짐. 확인해볼 필요 있슴.
anaemia : 빈혈증 여부  0, 1 값만 가지고 있슴 확인 (분류형) (0: 정상, 1: 빈혈)
creatinine_phosphokinase : 크레아틴키나제 검사 결과 (숫자형)
diabetes : 당뇨병 여부 0, 1 값만 가지고 있슴 확인 (분류형)  (0: 정상, 1: 당뇨병)
ejection_fraction : 박출계수 (%) 심박이 이루어질때 나오는 피의 비율  0 ~ 100 (숫자형)
high_blood_pressure : 고혈압 여부. 0, 1 값만 가지고 있슴 확인 (분류형) (0: 정상, 1: 고혈압)
platelets : 혈소판 수치 (kiloplatelets/mL) 단위.  혈액 ml 당  (숫자형)
serum_creatinine: 혈중 크레아틴 레벨 (mg/dL) (숫자형)
serum_sodium: 혈중 나트륨 레벨 (mEq/L)
sex: 성별  0, 1 값만 가지고 있슴 확인 (분류형) (0: 여성, 1: 남성)
smoking: 흡연 여부 (분류형) (0: 비흡연, 1: 흡연)
time: 관찰기간 (일) (숫자형)
DEATH_EVENT: 사망 여부 (분류형) (0: 생존, 1: 사망)

"""
None

# 분류형인 경우 확인해볼 필요도 있다.
df.anaemia.unique()
# >>> 출력:
# array([0, 1])

df.anaemia.value_counts()
# >>> 출력:
# anaemia
# 0    170
# 1    129
# Name: count, dtype: int64

# 기술통계량
df.describe()
# >>> 출력:
#               age     anaemia  creatinine_phosphokinase    diabetes  \
# count  299.000000  299.000000                299.000000  299.000000   
# mean    60.833893    0.431438                581.839465    0.418060   
# std     11.894809    0.496107                970.287881    0.494067   
# min     40.000000    0.000000                 23.000000    0.000000   
# 25%     51.000000    0.000000                116.500000    0.000000   
# 50%     60.000000    0.000000                250.000000    0.000000   
# 75%     70.000000    1.000000                582.000000    1.000000   
# max     95.000000    1.000000               7861.000000    1.000000   
# 
#        ejection_fraction  high_blood_pressure      platelets  \
# count         299.000000           299.000000     299.000000   
# mean           38.083612             0.351171  263358.029264   
# std            11.834841             0.478136   97804.236869   
# min            14.000000             0.000000   25100.000000   

# describe() 를 통해 보아야 하는 것
# 1. balanced vs. imbalanced data 여부 확인
# 2. 이상치 (outlier) 여부

# anameia :    0 ~ 1 의 범위인데, 평균이 약 0.43 (0.5에 가까운 값)
# diabetes, high_blood_pressure, sex, smoking, DEATH_EVENT
#  이 정도면 심한 imbalanced data 라 볼수 없어요

# 0 ~ 1 사잇값들의 평균이기에 정확히 0.5 이면 1:1 인거구요.
# 0 혹은 1 로 몰려 있을수록 평균이 한쪽으로 치우치겠죠.

# 지금은 그렇게 심하지 않습니다.
# DEATH_EVENT 가 가장 심한데.. 0.32107
#  약 1/3 이 사망, 2/3 이 생존한 분들의 데이터다.

# ![](https://miro.medium.com/max/450/1*zsyN08VVrgHbAEdvv27Pyw.png)
# #### 이상치 (outlier)
# 이상치 (Outlier)란 관측된 데이터의 범위에서 많이 벗어나 아주 작은 값이나 아주 큰 값
# ![](https://t1.daumcdn.net/cfile/tistory/9951C8475C518F180B)

df.describe()
# >>> 출력:
#               age     anaemia  creatinine_phosphokinase    diabetes  \
# count  299.000000  299.000000                299.000000  299.000000   
# mean    60.833893    0.431438                581.839465    0.418060   
# std     11.894809    0.496107                970.287881    0.494067   
# min     40.000000    0.000000                 23.000000    0.000000   
# 25%     51.000000    0.000000                116.500000    0.000000   
# 50%     60.000000    0.000000                250.000000    0.000000   
# 75%     70.000000    1.000000                582.000000    1.000000   
# max     95.000000    1.000000               7861.000000    1.000000   
# 
#        ejection_fraction  high_blood_pressure      platelets  \
# count         299.000000           299.000000     299.000000   
# mean           38.083612             0.351171  263358.029264   
# std            11.834841             0.478136   97804.236869   
# min            14.000000             0.000000   25100.000000   

"""
숫자로만 보면 알기 어려운 것들이 있다.
그래도 확인해볼수 있는 것들은
max, min 값이 과도하게 크거나 작은 것들

예를 들어 creatinine_phosphokinase 의 경우
min 값이 23 d이고 중앙값이 250 인데... max  값이 7861 이다!?!?
앞의 것들 min ~ 75% 의 증가세에 비하면 max 값이 상당히 큰 값이다. 상당한 outlier  에 해당한다 볼수 있다.
 상위 몇개 데이터는 배제해볼 필요성도 있겠구나.. 라고  생각해볼수 잇습니다.

ejection_fraction : 크게 문제 없어 보인다
platelets(혈소판 수치) : 크게 문제 없어 보인다  . 증가세가 일정한 느낌
serum_sodium : 크게 문제 없어 보인다.  그러나 값의 범위가 좁은 느김
                   113 ~ 148 범위. 평균이 136  인데... std 가 4.4. 정도밖에 안된다.
"""
None

# #### 이상치와 박스플롯 (box plot)
# ![](https://wikidocs.net/images/page/33920/Rplot19.png)
# ![](https://mblogthumb-phinf.pstatic.net/MjAxOTAzMDZfMjUz/MDAxNTUxODgzOTE4Mjgy.O4NTMQ3OpXjpJ6yoxWwXxyA_yzD6Hk2WlmVhXxVeieQg.04WzOuL6S3Lf5Bv5lrWDb3F9XEZPUxDpszfYw5yIMyMg.PNG.pmw9440/7.2_%EC%9D%B4%EC%83%81%EC%B9%98.png?type=w800)
# 
# 이상치 탐색을 위해 **박스플롯**으로 시각화하곤 합니다. 박스플롯은 다음과 같은 원리로 그려집니다.
# 
# 
# 박스플롯은 분위수를 기준으로 그려집니다.<br>
# 상자 안에 그려져 있는 직선은 중위수(Median)을 나타냅니다.<br>
# 상자의 밑변은 1분위수를 나타내며, 윗변은 3분위수를 나타냅니다.<br>
# 상자를 중심으로 위 아래에, 직선이 있는 것을 볼 수 있습니다. <br>
# 
# 이 직선은 울타리라고 부릅니다.<br>
# 상자로부터 아래 직선의 계산식 : Q1 - 1.5 * (Q3 - Q1)<br>
# 상자로부터 위 직선의 계산식 : Q3 + 1.5 *(Q3 - Q1)<br>
# 이 울타리를 벗어난 값들을 Outlier라고 부릅니다.<br>
# 
# 
# Outlier는 기본적으로 통계추정에 있어서 방해가 되고는 합니다.<br>
# 통계분석은 전부 귀납법인데, 이상치같은 특수 케이스가 규칙을 만드는데 방해가 되기 때문입니다.<br>
# 
# #### Outlier의 처리방법
# 
# 1. 제거를 하는 방법  (단점: 데이터가 버려진다)
# 
# 2. 데이터 변형을 통해 Outlier문제를 줄여줍니다.
# 
#     - 통계추정에세는 정규분포를 맞추어 주는 것이 매우 중요합니다. 보통 Outlier로 인해 한 쪽으로 치우친 분포는 log 변환을 통해 정규성을 맞추어주고는 합니다.

# # 수치형 데이터 EDA 하기

# ## 나이(age) 와 사망여부(DEATH_EVNET) 의 관계 분석

sns.histplot(x = 'age', data = df)  # age 분포
# >>> 출력:
# <Axes: xlabel='age', ylabel='Count'>

sns.histplot(x = 'age', data = df, hue='DEATH_EVENT')
# >>> 출력:
# <Axes: xlabel='age', ylabel='Count'>

# kde : kernel density estimate : 좀 더 부드러운 곡선으로 분포 추이 표시 (kde 플롯)
sns.histplot(x = 'age', data = df, hue='DEATH_EVENT', kde=True)
# >>> 출력:
# <Axes: xlabel='age', ylabel='Count'>

# ## creatinine_phosphokinase 와 DEATH_EVENT 의 관계분석

df.columns
# >>> 출력:
# Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
#        'ejection_fraction', 'high_blood_pressure', 'platelets',
#        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
#        'DEATH_EVENT'],
#       dtype='object')

sns.histplot(x = 'creatinine_phosphokinase', data=df)
# >>> 출력:
# <Axes: xlabel='creatinine_phosphokinase', ylabel='Count'>

# outlier 가 많다.
#  이를 배제하고 들여다 봅니다.  3000 이상의 값은 떼어내 보자.

sns.histplot(x = 'creatinine_phosphokinase', data = df.loc[df['creatinine_phosphokinase'] < 3000])
# >>> 출력:
# <Axes: xlabel='creatinine_phosphokinase', ylabel='Count'>

# x= 없이 아래와 같이 사용 가능
sns.histplot(data = df.loc[df['creatinine_phosphokinase'] < 3000, 'creatinine_phosphokinase'])
# >>> 출력:
# <Axes: xlabel='creatinine_phosphokinase', ylabel='Count'>

# 이렇게 보아도 히스토그램에서 정보를 얻기가 쉽지 않다 <- 통계적인 특징 이 잘 드러나지 않는다.

# ## ejection_fraction(박출계수) 와 DEATH_EVENT 의 관계분석

df.columns
# >>> 출력:
# Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
#        'ejection_fraction', 'high_blood_pressure', 'platelets',
#        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
#        'DEATH_EVENT'],
#       dtype='object')

sns.histplot(x = 'ejection_fraction', data=df)
# >>> 출력:
# <Axes: xlabel='ejection_fraction', ylabel='Count'>

sns.histplot(x = 'ejection_fraction', data=df, bins=10)
# >>> 출력:
# <Axes: xlabel='ejection_fraction', ylabel='Count'>

sns.histplot(x = 'ejection_fraction', data=df, bins=13)
# >>> 출력:
# <Axes: xlabel='ejection_fraction', ylabel='Count'>

sns.histplot(x = 'ejection_fraction', data=df, bins=13, hue='DEATH_EVENT', kde=True)
# >>> 출력:
# <Axes: xlabel='ejection_fraction', ylabel='Count'>

# ## platelets(혈소판) 과 DEATH_EVENT 의 관계 분석

df.columns
# >>> 출력:
# Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
#        'ejection_fraction', 'high_blood_pressure', 'platelets',
#        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
#        'DEATH_EVENT'],
#       dtype='object')

sns.histplot(x = 'platelets', data = df)
# >>> 출력:
# <Axes: xlabel='platelets', ylabel='Count'>

sns.histplot(x = 'platelets', data = df, hue='DEATH_EVENT', kde=True)
# >>> 출력:
# <Axes: xlabel='platelets', ylabel='Count'>

"""
혈소판 데이터는 DEATH_EVENT 를 가르는데 도움이 될거 같지 않다 (사망자/생존자 모두 골고루 분포 되어 있다.)
=> 별 상관성이 없다는 뜻.
"""
None

# ## 'platelets' 과 'creatinine_phosphokinase' 그리고 DEATH_EVENT 와의 관계성 시각화, 분석

df.columns
# >>> 출력:
# Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
#        'ejection_fraction', 'high_blood_pressure', 'platelets',
#        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
#        'DEATH_EVENT'],
#       dtype='object')

sns.jointplot(data=df, x='platelets', y='creatinine_phosphokinase', hue='DEATH_EVENT')
# >>> 출력:
# <seaborn.axisgrid.JointGrid at 0x7f0051b29150>

sns.jointplot(data=df, x='platelets', y='creatinine_phosphokinase', hue='DEATH_EVENT', alpha=0.3)
# >>> 출력:
# <seaborn.axisgrid.JointGrid at 0x7f004b5da710>

# alpha 를 주면 점을 투명하게 찍어주기 때문에
# 진해지는 정도로 겹쳐있는 것을 확인할수 있습니다.
"""
밑에 많이 뭉쳐 있어서 판단하기 어렵습니다.
뭉쳐 있는 모양 -> 어쨌든 DEATH_EVENT 를  가르는데는 도움이 되지 않습니다.
"""
None

# # 범주형 데이터 EDA 하기
# (분류형 데이터)

# seaborn의 Boxplot 계열(boxplot(), violinplot(), swarmplot())을 사용
# Hint) hue 키워드를 사용하여 범주 세분화 가능


# 범주형 데이터의 경우
# 위 히스토그램 데이터를 볼수 있는 것과 달리
# boxplot 을 통해 범주별로 따로 통계를 내야 어느정도 확인을 할수 있다.

# ## ejection_fraction(박출계수) 와 DEATH_EVENT 의 관계분석

df.columns
# >>> 출력:
# Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
#        'ejection_fraction', 'high_blood_pressure', 'platelets',
#        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
#        'DEATH_EVENT'],
#       dtype='object')

sns.boxplot(data=df, x='DEATH_EVENT', y='ejection_fraction', hue='DEATH_EVENT')
# >>> 출력:
# <Axes: xlabel='DEATH_EVENT', ylabel='ejection_fraction'>

# DEATH_EVENT !  생존(0)이냐? 사망(1)이냐? 에 따른 ejection_fration 을 boxplot 을 본겁니다.

# - 일단 평균에서 어느정도 차이가 난다
# - 전체적인 값의 범위의 차이도 확인 가능

# boxplot 의 최대 장점은 outlier 를 바깥으로 빼줍니다.  두개(범주)의 통계수치를 간단하게 비교할수 있다는 점입니다.

# 히스토그램으로 보았을때는 위의 ↓ 과 비슷한 급의 정보다.
#     sns.histplot(x='ejection_fraction', data=df, bins=13, hue="DEATH_EVENT", kde=True)
# 히스토그램은 '겹쳐져' 보이기 때문에 많은 정보를 '한번에' 파악하기 어려웠다면
# boxplot 을 이용할때는 일목요연하게 비교를 할수가 있다.

# boxplot은 '경영층' 에서 많이 사용하는 통계수치 입니다.
# 여러분이 경영층과 이야기 할때는 이와 같은 boxplot 을 이용해서 프레젠테이션 하면, 경영층에서
# 더 쉽게 받아들일수 있는 중요한 그래프라고 볼수 있습니다.

sns.histplot(x='ejection_fraction', data=df, bins=13, hue="DEATH_EVENT", kde=True)
# >>> 출력:
# <Axes: xlabel='ejection_fraction', ylabel='Count'>

# ## smoking(흡연여부) 와 ejection_fraction 의 관계 분석

sns.boxplot(data=df, x = 'smoking', y='ejection_fraction', hue='smoking')
# >>> 출력:
# <Axes: xlabel='smoking', ylabel='ejection_fraction'>

sns.violinplot(data=df, x='DEATH_EVENT', y='ejection_fraction', hue='DEATH_EVENT')
# >>> 출력:
# <Axes: xlabel='DEATH_EVENT', ylabel='ejection_fraction'>

# boxplot 에 표시되었었던 내용이 안쪽으로 표시됩니다.
# - 평균이 다르다.
# - 기본적인 '갑의 범위' 의 차이
# - 좌우로 퍼진 모습은 히스토그램이라 보시면 됨

# 히스토그램 + boxplot 을 하나로 묶어서 표현하는게 violin plot .
# outlier  까지 표현됨.  (좌측 0 의 끝 상단.. )

# 좀더 다양+많은 데이터를 '한번에' 시각화를 할수 있습니다.

# 참고로 violin plot 은 아주 많이 사용되는 plot 방식은 아닙니다.

# 일반적으로 경영층에서 활용하는 plot 이 아니라는 뜻입니다.

# '분석' 단계에서 자세히 데이터를 들여다 볼때는 violinplot을 활용해도
# '보고' 할때는 차라리 boxplot 이 좋습니다.

# ## smoking, ejection_fraction, DEATH_EVENT 의 관계분석

df.columns
# >>> 출력:
# Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
#        'ejection_fraction', 'high_blood_pressure', 'platelets',
#        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
#        'DEATH_EVENT'],
#       dtype='object')

sns.violinplot(data=df, x='DEATH_EVENT', y='ejection_fraction', hue='smoking')
# >>> 출력:
# <Axes: xlabel='DEATH_EVENT', ylabel='ejection_fraction'>

# smoking 여부 (0, 1) x DEATH_EVNET (0, 1) => 총 4가지로 표현됨.

sns.swarmplot(data=df, x='DEATH_EVENT', y='platelets', hue='DEATH_EVENT')
# >>> 출력:
# <Axes: xlabel='DEATH_EVENT', ylabel='platelets'>

sns.swarmplot(data=df, x='DEATH_EVENT', y='platelets', hue='smoking')
# >>> 출력:
# <Axes: xlabel='DEATH_EVENT', ylabel='platelets'>

# ## countplot 을 사용하여 범주형 통계 확인

sns.countplot(data=df, x='DEATH_EVENT', hue='DEATH_EVENT')
# >>> 출력:
# <Axes: xlabel='DEATH_EVENT', ylabel='count'>

df.describe()
# >>> 출력:
#               age     anaemia  creatinine_phosphokinase    diabetes  \
# count  299.000000  299.000000                299.000000  299.000000   
# mean    60.833893    0.431438                581.839465    0.418060   
# std     11.894809    0.496107                970.287881    0.494067   
# min     40.000000    0.000000                 23.000000    0.000000   
# 25%     51.000000    0.000000                116.500000    0.000000   
# 50%     60.000000    0.000000                250.000000    0.000000   
# 75%     70.000000    1.000000                582.000000    1.000000   
# max     95.000000    1.000000               7861.000000    1.000000   
# 
#        ejection_fraction  high_blood_pressure      platelets  \
# count         299.000000           299.000000     299.000000   
# mean           38.083612             0.351171  263358.029264   
# std            11.834841             0.478136   97804.236869   
# min            14.000000             0.000000   25100.000000   

# ## sex 와 DEATH_EVENT

sns.countplot(data=df, x='sex', hue='DEATH_EVENT')
# >>> 출력:
# <Axes: xlabel='sex', ylabel='count'>

# # 새로운 파생변수를 생성하여 분석하기

df['age']
# >>> 출력:
# 0      75.0
# 1      55.0
# 2      65.0
# 3      50.0
# 4      65.0
#        ... 
# 294    62.0
# 295    55.0
# 296    45.0
# 297    45.0
# 298    50.0
# Name: age, Length: 299, dtype: float64

# ## 나이대 (age_span) 컬럼 추가
# 기존의 컬럼을 사용하여 새로운 컬럼 데이터 만들고 이를 분석에 활용할수 있다.

df['age_span'] = (df['age'] / 10).astype(int) * 10

df
# >>> 출력:
#       age  anaemia  creatinine_phosphokinase  diabetes  ejection_fraction  \
# 0    75.0        0                       582         0                 20   
# 1    55.0        0                      7861         0                 38   
# 2    65.0        0                       146         0                 20   
# 3    50.0        1                       111         0                 20   
# 4    65.0        1                       160         1                 20   
# ..    ...      ...                       ...       ...                ...   
# 294  62.0        0                        61         1                 38   
# 295  55.0        0                      1820         0                 38   
# 296  45.0        0                      2060         1                 60   
# 297  45.0        0                      2413         0                 38   
# 298  50.0        0                       196         0                 45   
# 
#      high_blood_pressure  platelets  serum_creatinine  serum_sodium  sex  \
# 0                      1  265000.00               1.9           130    1   

df.groupby('age_span').size()
# >>> 출력:
# age_span
# 40    47
# 50    82
# 60    93
# 70    52
# 80    19
# 90     6
# dtype: int64

df.groupby('age_span').mean()
# >>> 출력:
#                 age   anaemia  creatinine_phosphokinase  diabetes  \
# age_span                                                            
# 40        44.212766  0.297872                802.148936  0.446809   
# 50        53.329268  0.451220                620.658537  0.402439   
# 60        62.949828  0.505376                499.215054  0.526882   
# 70        72.307692  0.365385                379.250000  0.288462   
# 80        82.631579  0.368421                932.526316  0.263158   
# 90        92.333333  0.833333                251.500000  0.333333   
# 
#           ejection_fraction  high_blood_pressure      platelets  \
# age_span                                                          
# 40                37.191489             0.297872  285931.705319   
# 50                37.353659             0.317073  265001.807073   
# 60                38.473118             0.311828  248990.840645   
# 70                38.538462             0.557692  267036.349038   

df.groupby('age_span').count()
# >>> 출력:
#           age  anaemia  creatinine_phosphokinase  diabetes  ejection_fraction  \
# age_span                                                                        
# 40         47       47                        47        47                 47   
# 50         82       82                        82        82                 82   
# 60         93       93                        93        93                 93   
# 70         52       52                        52        52                 52   
# 80         19       19                        19        19                 19   
# 90          6        6                         6         6                  6   
# 
#           high_blood_pressure  platelets  serum_creatinine  serum_sodium  sex  \
# age_span                                                                        
# 40                         47         47                47            47   47   
# 50                         82         82                82            82   82   
# 60                         93         93                93            93   93   
# 70                         52         52                52            52   52   

# 나이대별 사망자 수 계산
ab = df.groupby('age_span').sum()['DEATH_EVENT']
ab
# >>> 출력:
# age_span
# 40    11
# 50    20
# 60    27
# 70    20
# 80    13
# 90     5
# Name: DEATH_EVENT, dtype: int64

plt.bar(ab.index, ab)
# >>> 출력:
# <BarContainer object of 6 artists>

# 내림차순, 오름차순
df2 = pd.DataFrame(ab)
df2
# >>> 출력:
#           DEATH_EVENT
# age_span             
# 40                 11
# 50                 20
# 60                 27
# 70                 20
# 80                 13
# 90                  5

df2.sort_values(by='DEATH_EVENT')
# >>> 출력:
#           DEATH_EVENT
# age_span             
# 90                  5
# 40                 11
# 80                 13
# 50                 20
# 70                 20
# 60                 27

df2.sort_values(by='DEATH_EVENT', ascending=False)
# >>> 출력:
#           DEATH_EVENT
# age_span             
# 60                 27
# 50                 20
# 70                 20
# 80                 13
# 40                 11
# 90                  5

df2.sort_values(by='DEATH_EVENT', ascending=False).plot(kind='bar')
# >>> 출력:
# <Axes: xlabel='age_span'>

# pairplot 은 모든 수치형데이터에 대한 조합을 jointplot 으로 한번에 보여줌.

df.columns
# >>> 출력:
# Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
#        'ejection_fraction', 'high_blood_pressure', 'platelets',
#        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
#        'DEATH_EVENT', 'age_span'],
#       dtype='object')

df[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'DEATH_EVENT']]
# >>> 출력:
#       age  creatinine_phosphokinase  ejection_fraction  DEATH_EVENT
# 0    75.0                       582                 20            1
# 1    55.0                      7861                 38            1
# 2    65.0                       146                 20            1
# 3    50.0                       111                 20            1
# 4    65.0                       160                 20            1
# ..    ...                       ...                ...          ...
# 294  62.0                        61                 38            0
# 295  55.0                      1820                 38            0
# 296  45.0                      2060                 60            0
# 297  45.0                      2413                 38            0
# 298  50.0                       196                 45            0
# 
# [299 rows x 4 columns]

sns.pairplot(df[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'DEATH_EVENT']], hue='DEATH_EVENT')
# >>> 출력:
# <seaborn.axisgrid.PairGrid at 0x7f004afe46d0>

# # 모델 학습을 위한 데이터 전처리

# ## StandardScaler 를 이용한 데이터 전처리

from sklearn.preprocessing import StandardScaler

df.columns
# >>> 출력:
# Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
#        'ejection_fraction', 'high_blood_pressure', 'platelets',
#        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
#        'DEATH_EVENT', 'age_span'],
#       dtype='object')

# 수치형 입력데이터, 범주형 입력데이터, 출력데이터 구분

X_num = df[['age', 'creatinine_phosphokinase',
       'ejection_fraction', 'platelets',
       'serum_creatinine', 'serum_sodium']]

X_cat = df[['anaemia', 'diabetes','high_blood_pressure',
       'sex', 'smoking']]

y = df['DEATH_EVENT']

X_num.head()  # 수치형 데이터
# >>> 출력:
#     age  creatinine_phosphokinase  ejection_fraction  platelets  \
# 0  75.0                       582                 20  265000.00   
# 1  55.0                      7861                 38  263358.03   
# 2  65.0                       146                 20  162000.00   
# 3  50.0                       111                 20  210000.00   
# 4  65.0                       160                 20  327000.00   
# 
#    serum_creatinine  serum_sodium  
# 0               1.9           130  
# 1               1.1           136  
# 2               1.3           129  
# 3               1.9           137  
# 4               2.7           116

X_cat.head() # 범주형 데이터
# >>> 출력:
#    anaemia  diabetes  high_blood_pressure  sex  smoking
# 0        0         0                    1    1        0
# 1        0         0                    0    1        0
# 2        0         0                    0    1        1
# 3        1         0                    0    1        0
# 4        1         1                    0    0        0

y.head()
# >>> 출력:
# 0    1
# 1    1
# 2    1
# 3    1
# 4    1
# Name: DEATH_EVENT, dtype: int64

# 수치형 입력데이터를 전처리하고 입력데이터 통합하기
scaler = StandardScaler()
scaler.fit(X_num)
X_scaled = scaler.transform(X_num)
X_scaled
# >>> 출력:
# array([[ 1.19294523e+00,  1.65728387e-04, -1.53055953e+00,
#          1.68164843e-02,  4.90056987e-01, -1.50403612e+00],
#        [-4.91279276e-01,  7.51463953e+00, -7.07675018e-03,
#          7.53566018e-09, -2.84552352e-01, -1.41976151e-01],
#        [ 3.50832977e-01, -4.49938761e-01, -1.53055953e+00,
#         -1.03807313e+00, -9.09000174e-02, -1.73104612e+00],
#        ...,
#        [-1.33339153e+00,  1.52597865e+00,  1.85495776e+00,
#          4.90208200e+00, -5.75030855e-01,  3.12043840e-01],
#        [-1.33339153e+00,  1.89039811e+00, -7.07675018e-03,
#         -1.26338936e+00,  5.92615005e-03,  7.66063830e-01],
#        [-9.12335403e-01, -3.98321274e-01,  5.85388775e-01,
#          1.34823057e+00,  1.99578485e-01, -1.41976151e-01]])

X_scaled = pd.DataFrame(data=X_scaled, index=X_num.index, columns=X_num.columns)
X_scaled.head()
# >>> 출력:
#         age  creatinine_phosphokinase  ejection_fraction     platelets  \
# 0  1.192945                  0.000166          -1.530560  1.681648e-02   
# 1 -0.491279                  7.514640          -0.007077  7.535660e-09   
# 2  0.350833                 -0.449939          -1.530560 -1.038073e+00   
# 3 -0.912335                 -0.486071          -1.530560 -5.464741e-01   
# 4  0.350833                 -0.435486          -1.530560  6.517986e-01   
# 
#    serum_creatinine  serum_sodium  
# 0          0.490057     -1.504036  
# 1         -0.284552     -0.141976  
# 2         -0.090900     -1.731046  
# 3          0.490057      0.085034  
# 4          1.264666     -4.682176

X_scaled.describe()
# >>> 출력:
#                 age  creatinine_phosphokinase  ejection_fraction  \
# count  2.990000e+02                299.000000       2.990000e+02   
# mean   5.703353e-16                  0.000000      -3.267546e-17   
# std    1.001676e+00                  1.001676       1.001676e+00   
# min   -1.754448e+00                 -0.576918      -2.038387e+00   
# 25%   -8.281242e-01                 -0.480393      -6.841802e-01   
# 50%   -7.022315e-02                 -0.342574      -7.076750e-03   
# 75%    7.718891e-01                  0.000166       5.853888e-01   
# max    2.877170e+00                  7.514640       3.547716e+00   
# 
#           platelets  serum_creatinine  serum_sodium  
# count  2.990000e+02      2.990000e+02  2.990000e+02  
# mean   7.723291e-17      1.425838e-16 -8.673849e-16  
# std    1.001676e+00      1.001676e+00  1.001676e+00  
# min   -2.440155e+00     -8.655094e-01 -5.363206e+00  

# scale 된 숫자형 과 범주형을 합친다
X = pd.concat([X_scaled, X_cat], axis=1)
X.head()
# >>> 출력:
#         age  creatinine_phosphokinase  ejection_fraction     platelets  \
# 0  1.192945                  0.000166          -1.530560  1.681648e-02   
# 1 -0.491279                  7.514640          -0.007077  7.535660e-09   
# 2  0.350833                 -0.449939          -1.530560 -1.038073e+00   
# 3 -0.912335                 -0.486071          -1.530560 -5.464741e-01   
# 4  0.350833                 -0.435486          -1.530560  6.517986e-01   
# 
#    serum_creatinine  serum_sodium  anaemia  diabetes  high_blood_pressure  \
# 0          0.490057     -1.504036        0         0                    1   
# 1         -0.284552     -0.141976        0         0                    0   
# 2         -0.090900     -1.731046        0         0                    0   
# 3          0.490057      0.085034        1         0                    0   
# 4          1.264666     -4.682176        1         1                    0   
# 
#    sex  smoking  

# ## 학습데이터와 테스트 데이터 분리

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
  train_test_split(X, y, test_size=0.3, random_state=1)

print(X_train.shape, X_test.shape)
# >>> 출력:
# (209, 11) (90, 11)

# # 모델 학습하기

# ## LogisticRegression 모델

from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
# >>> 출력:
# LogisticRegression()

model_lr = LogisticRegression(max_iter=1000, verbose=1)
model_lr.fit(X_train, y_train)
# >>> 출력:
# LogisticRegression(max_iter=1000, verbose=1)

# ## 모델 학습 결과 평가하기

from sklearn.metrics import classification_report

pred = model_lr.predict(X_test)
print(classification_report(y_test, pred))
# >>> 출력:
#               precision    recall  f1-score   support
# 
#            0       0.78      0.92      0.84        64
#            1       0.64      0.35      0.45        26
# 
#     accuracy                           0.76        90
#    macro avg       0.71      0.63      0.65        90
# weighted avg       0.74      0.76      0.73        90

"""
              precision    recall  f1-score   support

           0       0.78      0.92      0.84        64
           1       0.64      0.35      0.45        26

    accuracy                           0.76        90 <-- 종합 accuracy 0.76 총 90개 테스트데이터 평가한것중 76% 정확도
   macro avg       0.71      0.63      0.65        90
weighted avg       0.74      0.76      0.73        90
"""
None

# ## XGBoost 모델

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

# ## 모델 학습 결과 평가

pred = model_xgb.predict(X_test)
print(classification_report(y_test, pred))
# >>> 출력:
#               precision    recall  f1-score   support
# 
#            0       0.81      0.86      0.83        64
#            1       0.59      0.50      0.54        26
# 
#     accuracy                           0.76        90
#    macro avg       0.70      0.68      0.69        90
# weighted avg       0.75      0.76      0.75        90

"""
              precision    recall  f1-score   support

           0       0.81      0.86      0.83        64
           1       0.59      0.50      0.54        26

    accuracy                           0.76        90 <-- 비슷한 결과
   macro avg       0.70      0.68      0.69        90
weighted avg       0.75      0.76      0.75        90
"""
None

# ## 특성 중요도 확인

model_xgb.feature_importances_
# >>> 출력:
# array([0.11038695, 0.07547454, 0.13747513, 0.07426631, 0.16953382,
#        0.09146173, 0.06754536, 0.08684321, 0.05830629, 0.09005816,
#        0.03864855], dtype=float32)

model_xgb.feature_importances_.sum()
# >>> 출력:
# np.float32(1.0000001)

plt.plot(model_xgb.feature_importances_)
# >>> 출력:
# [<matplotlib.lines.Line2D at 0x7f0042451410>]

plt.bar(X.columns, model_xgb.feature_importances_)
plt.xticks(rotation=90)
plt.show()

sns.jointplot(x='ejection_fraction', y='serum_creatinine', data=df, hue='DEATH_EVENT')
# >>> 출력:
# <seaborn.axisgrid.JointGrid at 0x7f002d54cc50>

# # 모델 학습 결과 심화 분석

# ## Precision-Recall 커브
#   - Recall 을 유지하면서 Precision 을 얼마나 끌어올릴수 있나?

from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve

# LogisticRegression 모델의 Precision-Recall 커브
PrecisionRecallDisplay.from_estimator(model_lr, X_test, y_test)
# >>> 출력:
# <sklearn.metrics._plot.precision_recall_curve.PrecisionRecallDisplay at 0x7f002d311550>

# Precision 이 1부터 시작 (좌상단)  Recall 을 증가시킬때 (우측방향) 어떻게 요동치는지 볼수 있다.
# AP 값이 높을수록 Recall 을 증가시키면서 Precision 이 잘 유지된다느 뜻 (1에 가까울수록 좋다.)

# XGB 와 같이 출력
PrecisionRecallDisplay.from_estimator(model_lr, X_test, y_test)
PrecisionRecallDisplay.from_estimator(model_xgb, X_test, y_test)
# >>> 출력:
# <sklearn.metrics._plot.precision_recall_curve.PrecisionRecallDisplay at 0x7f00420eb990>

# 두 모델의 Precision-Recal 커브를 한번에 겹쳐서 그리기.
fig = plt.figure()
ax = fig.gca()  # get current axesplot
PrecisionRecallDisplay.from_estimator(model_lr, X_test, y_test, ax=ax)
PrecisionRecallDisplay.from_estimator(model_xgb, X_test, y_test, ax=ax)
# >>> 출력:
# <sklearn.metrics._plot.precision_recall_curve.PrecisionRecallDisplay at 0x7f00426cdfd0>

# ## ROC 커브 확인하기

from sklearn.metrics import RocCurveDisplay, roc_curve

fig = plt.figure()
ax = fig.gca()
RocCurveDisplay.from_estimator(model_lr, X_test, y_test, ax=ax)
RocCurveDisplay.from_estimator(model_xgb, X_test, y_test, ax=ax)
# >>> 출력:
# <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x7f002d14c410>
