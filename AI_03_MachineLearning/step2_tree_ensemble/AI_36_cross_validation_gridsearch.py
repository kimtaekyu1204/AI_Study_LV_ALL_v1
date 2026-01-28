# # 교차검증과 그리드 서치

# 하이퍼파라미터
# max_depth=  3 외에 다른 값들을 주어가며 '테스트세트'로 평가해보면
#   가장좋은 max_depth 값을 찾아낼것 같다...

# 문제점! 테스트세트 에만 잘 맞는 모델이 만들어질 가능성이 커진다

# ★ 그래서 '테스트세트' 는 마지막에 딱 한번만 사용하는게 좋다!

# max_depth= 와 같은 하이퍼파라미터 튜닝은 어떻게 하면 좋을까?

# # 검증세트 (validation set)

# ![](https://velog.velcdn.com/images/jiazzang/post/2b28c228-5b4c-41ee-82c3-4fdf48b9fda7/image.PNG)

# 검증 데이터 세트를 사용하여 하이퍼 파라미터 튜닝한 모델을 평가한다.

# ※ 보통은 전체 데이터의 20~30% 를 테스트세트와 검증세트로 떼어놓는다. (이는 문제 사안에 다르다)

# # 홀드 아웃(Hold-out) 검증
# 
# - step1 훈련세트로 → 모델을 훈련
# - step2 검증세트로 모델을 평가 → 하이퍼 파라미터를 바꿔가며 최적의 하이퍼 파라미터를 가진 가장 좋은 모델을 고름
# - step3 훈련세트와 검증세트를 합쳐 전체 훈련 데이터에서 모델을 다시 훈련.
# - step4 테스트세트 -> 모델을 최종적으로 평가. 최종점수! ←  실전에서의 성능기대 수치.
# 
# ![](https://t1.daumcdn.net/cfile/tistory/994042405E24E8081C)

# # 데이터 준비

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

base_path = r'/content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)'

file_path = os.path.join(base_path, 'wine.csv')
wine_df = pd.read_csv(file_path)
wine_df
# >>> 출력:
#       alcohol  sugar    pH  class
# 0         9.4    1.9  3.51    0.0
# 1         9.8    2.6  3.20    0.0
# 2         9.8    2.3  3.26    0.0
# 3         9.8    1.9  3.16    0.0
# 4         9.4    1.9  3.51    0.0
# ...       ...    ...   ...    ...
# 6492     11.2    1.6  3.27    1.0
# 6493      9.6    8.0  3.15    1.0
# 6494      9.4    1.2  2.99    1.0
# 6495     12.8    1.1  3.34    1.0
# 6496     11.8    0.8  3.26    1.0
# 
# [6497 rows x 4 columns]

data = wine_df[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine_df['class'].to_numpy()

# # 검증 세트 나누기

# train / test 세트 나누기
from sklearn.model_selection import train_test_split

# train: test = 8:2 로 나누기
train_input, test_input, train_target, test_target = \
  train_test_split(data, target, test_size=0.2, random_state=42)

# train_input, train_target 을 다시 '훈련세트' 와 '검증세트' 로 나눈다.

sub_input, val_input, sub_target, val_target = \
  train_test_split(train_input, train_target, test_size=0.2, random_state=42)

# 훈련세트: sub_input, sub_target
# 검증세트: val_input, val_target

print(train_input.shape, sub_input.shape, val_input.shape, test_input.shape)
# >>> 출력:
# (5197, 3) (4157, 3) (1040, 3) (1300, 3)

# 훈련세트 와 검증세트를 사용해 모델 만들고 평가하기

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)

print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))
# >>> 출력:
# 0.9971133028626413
# 0.864423076923077

# 딱 봐도 overfit 이다
# 하이퍼파라미터를 바꾸어 가면서 '더 좋은 모델' 을 찾아보자.

# 그 전에 검증세트 관련해 '교차검증' 을 알아보자.

# # 교차 검증 (Cross Validation)

# 문제점]
# 원래 훈련데이터를 또 쪼개었다..

# 문제1. 훈련시켜야 할 데이터 양이 줄었다
# 문제2. 훈련세트를 확보하기 위해 검증세트를 줄이면 검증점수가 들쑥날쑥해진다.

# 이 문제를 해결하기 위해 '교차검증 (cross validation)'을 사용해
#   -> 안정적인 점수도 얻고
#   -> 훈련에 더 많은 데이터 사용 가능.

# **교차 검증** 은 검증세트를 떼어 내어 평가하는 과정을 여러번 '반복'합니다.
# 그 다음 이 '결과' 를 평균하여 최종 검증 점수를 얻습니다.
# 
# k 등분으로 떼어내어 '반복' 하여 교차검증 하는 것을 **'k-fold cross validation (k폴드 교차검증)'** 이라 한다
# 
# ![](https://wikidocs.net/images/page/223699/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2023-11-30_152347.png)

# ## cross_validate()

from sklearn.model_selection import cross_validate

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
# 
# ```python
# sklearn.model_selection.cross_validate(
#   estimator,  # 모델
#   X,          # 입력데이타
#   y=None,     # 타겟 데이타
#   *,
#   groups=None,
#   scoring=None,
#   cv=None,   # fold 개수, 기본은 5
#   n_jobs=None,  # cpu 코어 사용 개수
#   verbose=0,
#   fit_params=None,
#   params=None,
#   pre_dispatch='2*n_jobs',
#   return_train_score=False, # 검증 점수뿐만 아니라 훈련세트에 대한 점수도 같이 반환
#   return_estimator=False,
#   return_indices=False,
#   error_score=nan)
# ```

# 검증세트 분리하기 '전' 의 '훈련세트 전체' 를 넘겨준다

scores = cross_validate(dt, train_input, train_target)

scores  # dict
# >>> 출력:
# {'fit_time': array([0.03005338, 0.02838182, 0.10211444, 0.02275157, 0.06066895]),
#  'score_time': array([0.01319528, 0.00469637, 0.00207949, 0.00508595, 0.00381374]),
#  'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}

# 기본적으로 5-fold 교차검증 수행 (cv= : 폴드수)
# fit_time, score_time : 각각 모델을 '훈련'하는 시간과 '검증'하는 시간
# test_score: 모델의 각 폴드별 검증 점수

# 교차검증 최종점수는 5개의 검증점수의 평균

print(np.mean(scores['test_score']))
# >>> 출력:
# 0.855300214703487

# 주의!
#  cross_validate() 는 훈련세트를 섞어 폴드를 나누지 않는다.
#  ※ train_test_split() 은 전체 데이터 섞은후 훈련세트 준비했었다.

# 교차검증시 훈련세트 섞으려면 splitter(분할기) 를 지정해주어야 한다.
#     cross_validate() 는 기본적으로
#     - 회귀모델 은 'KFold 분할기' 사용
#     - 분류모델 은 StratifiedKFold 사용.

from sklearn.model_selection import StratifiedKFold

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
# 
# ```python
# class sklearn.model_selection.StratifiedKFold(
#   n_splits=5,   # fold 개수
#   *,
#   shuffle=False,   # 섞을지 여부
#   random_state=None)
# ```

# 앞서 수행한 교차검증은 다음 코드와 동일하다.
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
# >>> 출력:
# 0.855300214703487

# 훈련세트를 '섞은후' + '10-fold' 교차검증을 수행하려면.
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
# >>> 출력:
# 0.8574181117533719

scores
# >>> 출력:
# {'fit_time': array([0.01975799, 0.01386189, 0.01777291, 0.0145843 , 0.01314902,
#         0.01498222, 0.01300693, 0.01365876, 0.01398492, 0.01607108]),
#  'score_time': array([0.00211501, 0.00668502, 0.0051465 , 0.00158095, 0.00177503,
#         0.00159454, 0.00184941, 0.0018816 , 0.00168729, 0.00395155]),
#  'test_score': array([0.83461538, 0.87884615, 0.85384615, 0.85384615, 0.84615385,
#         0.87307692, 0.85961538, 0.85549133, 0.85163776, 0.86705202])}

# # 하이퍼 파라미너 튜닝 GridSearchCV
# 교차검증과 최적의 하이퍼파라미터 찾기

"""
■ 모델파라미터: 머신러닝 모델이 학습하는 파라미터

■ 하이퍼파라미터: 모델이 학습할 수 없어서 '사용자'가 '학습전에 지정'해야만 하는 파라미터
  => 사이킷런에서는 이런 하이퍼파라미터는 모두 클래스나 메소드의 매개변수로 표현된다
"""
None

"""
최적의 하이퍼 파라미터를 찾기 위해 하이퍼파라미터 튜닝하는 작업은 어떻게 진행하나?
step1 : 라이브러리가 제공하는 기본값을 그대로 사용해 모델 훈련.
step2 : 검증세트의 점수나 교차 검증을 통해서 매개변수를 '조금씩' 바꿔본다.
       모델마다 적게는 1~2개, 많게는 5~6개의 매개변수가 제공되는데...
       이 매개변수를 바꿔가면서 모델을 훈련하고 교차 검증 수행해야 한다.
"""
None

# 위 방식의 문제점!
#  가령, DecisionTree 모델에서
#  일단 최적의 max_depth= 값을 찾았다고 치자.
#   그 max_depth 값을 '고정'하고
#  또 다른 하이퍼파라미터인 min_samples_split= 을 바꿔가면서 최적의 값을 찾았다.

# 이런식으로 최적값을 찾아가는게 맞을까?  -> NO!!!!

# max_depth= 의 최적값은 min_samples_split= 값이 바뀌면 '함께' 바뀐다.
# 즉, 이 두 매개변수를 '동시' 에 바꿔가며 최적의 값을 찾아내야만 한다!

# 그런데 매개변수가 많아지면 복잡해진다...

# => 이를 편리하게 수행해주는게 '그리드 서치 (GridSearch)' 다.

# GridSearchCV
# cross validation 을 여러번 반복해서, 여기에서 최적의 하이퍼 파라미터 를 찾음

# GridSearchCV 는 '하이퍼 파라미터 탐색' + '교차검증' 을 한번에 수행

from sklearn.model_selection import GridSearchCV

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# 
# ```python
# class sklearn.model_selection.GridSearchCV(
#   estimator,  # 탐색대상 모델
#   param_grid, # 탐색 파라미터와 값들, dict
#   *,
#   scoring=None,
#   n_jobs=None,  # 병렬실행에 사용할 CPU 코어수
#   refit=True,
#   cv=None,  # fold 개수, 5-fold 수행
#   verbose=0,
#   pre_dispatch='2*n_jobs',
#   error_score=nan,
#   return_train_score=False)
# ```

# ## 한개의 파라미터 탐색

# 결정트리 모델에서 min_impurity_decrease=  의 최적값을 찾아보기.

# estimator(모델) 가 갖고 있는 하이퍼 파라미터들 확인
DecisionTreeClassifier().get_params()
# >>> 출력:
# {'ccp_alpha': 0.0,
#  'class_weight': None,
#  'criterion': 'gini',
#  'max_depth': None,
#  'max_features': None,
#  'max_leaf_nodes': None,
#  'min_impurity_decrease': 0.0,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'min_weight_fraction_leaf': 0.0,
#  'monotonic_cst': None,
#  'random_state': None,
#  'splitter': 'best'}

# '탐색할 매개변수' 와 '탐색할 값(들)의 리스트' 를 dict 로 준비.
min_impurity_decrease = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]

params = {'min_impurity_decrease': min_impurity_decrease}

# dict 를 아래와 같이 생성 가능
# params = dict(min_impurity_decrease = min_impurity_decrease)

# GridSearchCV 클래스에 '탐색대상모델' 과 params 변수를 전달하여 그리드 서치 객체 생성
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, cv=5)

# 모델 훈련하듯이 fit() 호출

# cv=5 : 5-fold
#  min_impurity_decrease 값마다 5-fold 수행

# => 5-fold x 5 = 25개의 모델을 훈련한다!

# 병렬작업을 위해 n_jobs=-1 지정 (CPU 코어 최대 활용.)

# 참고] CPU 코어개수 확인
import multiprocessing
multiprocessing.cpu_count()
# >>> 출력:
# 2

# 그리드 서치 수행
gs.fit(train_input, train_target)   # 수행결과는 gs 객체에 담겨 있다. (리턴값도 gs 다)
# >>> 출력:
# GridSearchCV(cv=5, estimator=DecisionTreeClassifier(random_state=42), n_jobs=-1,
#              param_grid={'min_impurity_decrease': [0.0001, 0.0002, 0.0003,
#                                                    0.0004, 0.0005]})

print("최적점수:", gs.best_score_)
print("최적파라미터:", gs.best_params_)
# >>> 출력:
# 최적점수: 0.8681929740134745
# 최적파라미터: {'min_impurity_decrease': 0.0001}

"""
교차 검증에서
최적의 하이퍼파라미터를 찾으면 전체 훈련 세트로 모델을 다시 만들어야 한다고 했었습니다.

아주 편리하게도 사이킷런의 그리드 서치는 훈련이 끝나면 25개의 모델중에서
검증 점수가 가장 높은 모델의 매개변수 조합으로 전체 훈련 세트에서 자동으로 다시 모델을 훈련합니다.
이 모델은 gs 객체의 best_estimator_ 속성에 저장되어 있습니다.
이 모델을 일반 결정트리 처럼 똑같이 사용할 수 있습니다
"""
None

dt = gs.best_estimator_  # 최적의 파라미터로 학습된 모델

print(dt.score(train_input, train_target))
# >>> 출력:
# 0.9615162593804117

# 각 매개변수에서 수행한 교차 검증 결과 => cv_results_ 속성
gs.cv_results_
# >>> 출력:
# {'mean_fit_time': array([0.01468701, 0.00896039, 0.01170421, 0.01218061, 0.01376419]),
#  'std_fit_time': array([0.00751019, 0.00068206, 0.0036363 , 0.00546144, 0.00149283]),
#  'mean_score_time': array([0.00196562, 0.00169106, 0.00171261, 0.00274887, 0.00175304]),
#  'std_score_time': array([2.54346817e-04, 1.42114945e-04, 1.17891119e-04, 1.68613084e-03,
#         8.19795827e-05]),
#  'param_min_impurity_decrease': masked_array(data=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
#               mask=[False, False, False, False, False],
#         fill_value=1e+20),
#  'params': [{'min_impurity_decrease': 0.0001},
#   {'min_impurity_decrease': 0.0002},
#   {'min_impurity_decrease': 0.0003},
#   {'min_impurity_decrease': 0.0004},
#   {'min_impurity_decrease': 0.0005}],
#  'split0_test_score': array([0.86923077, 0.87115385, 0.86923077, 0.86923077, 0.86538462]),
#  'split1_test_score': array([0.86826923, 0.86346154, 0.85961538, 0.86346154, 0.86923077]),

pd.DataFrame(gs.cv_results_)
# >>> 출력:
#    mean_fit_time  std_fit_time  mean_score_time  std_score_time  \
# 0       0.014687      0.007510         0.001966        0.000254   
# 1       0.008960      0.000682         0.001691        0.000142   
# 2       0.011704      0.003636         0.001713        0.000118   
# 3       0.012181      0.005461         0.002749        0.001686   
# 4       0.013764      0.001493         0.001753        0.000082   
# 
#    param_min_impurity_decrease                             params  \
# 0                       0.0001  {'min_impurity_decrease': 0.0001}   
# 1                       0.0002  {'min_impurity_decrease': 0.0002}   
# 2                       0.0003  {'min_impurity_decrease': 0.0003}   
# 3                       0.0004  {'min_impurity_decrease': 0.0004}   
# 4                       0.0005  {'min_impurity_decrease': 0.0005}   
# 
#    split0_test_score  split1_test_score  split2_test_score  split3_test_score  \

# 5 row : 5개의 min_impurity_decrease= 탐색값에 CV 수행

# 컬럼 split0_test_score ~ split4_test_score <- 5번의 CV 스코어
# 컬럼 mean_test_score  <- 5번의 CV 점수의 평균값 .  이중의 최댓값이 best_score_

# 각 매개변수에서 수행한 교차검증의 평균점수는
gs.cv_results_['mean_test_score']
# >>> 출력:
# array([0.86819297, 0.86453617, 0.86492226, 0.86780891, 0.86761605])

best_index = np.argmax(gs.cv_results_['mean_test_score'])  # 가장 큰 값의 인덱스 추출.
print(best_index)
print(gs.cv_results_['params'][best_index])
# >>> 출력:
# 0
# {'min_impurity_decrease': 0.0001}

# ## Grid search 과정 정리
# 1. 탐색할 매개변수 및 탐색할 값들 지정
# 1. 훈련세트에서 그리드 서치 수행.
#   - 결과: 최상의 평균점수가 나오는 매개변수 조합을 찾아낸다.
#   - 이 조합은 그리드 서치 객체에 저장되어 있다
# 1. 그리드 서치는 최상의 매개변수에서 (교차검증에 사용한 훈련 세트가 아니라) 전체 훈련세트를 사용해 최종 모델을 훈련한다.  이 모델도 그리드 서치 객체에 저장된다.

# ## 복잡한 매개변수 조합 탐색

"""
결정 트리에서
  min_impurity_decrease= 는 노드를 분할하기 위한 불순도 감소 최소량을 지정.
  max_depth= 로 트리의 깊이를 제한
  min_samples_split= 으로 노드를 나누기 위한 최소 샘플 수 지정

"""
None

# 탐색할 하이퍼 파라미터(들) 을 dict 로 준비
params = {
    'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),  # 9개의 값
    'max_depth': range(5, 20, 1),   # 15개의 값
    'min_samples_split': range(2, 100, 10),  # 10개의 값
}

params
# >>> 출력:
# {'min_impurity_decrease': array([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,
#         0.0009]),
#  'max_depth': range(5, 20),
#  'min_samples_split': range(2, 100, 10)}

#  위 매개변수로 수행할 교차검증 횟수 : 9 x 15 x 10 = 1,350개

#  기본 5-fold 교차검증을 수행하면 만들어 지는 모델의 수는 1,350 x 5 = 6,750개 !!

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
# >>> 출력:
# GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), n_jobs=-1,
#              param_grid={'max_depth': range(5, 20),
#                          'min_impurity_decrease': array([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,
#        0.0009]),
#                          'min_samples_split': range(2, 100, 10)})

# 최상의 파라미터 조합은?
gs.best_params_
# >>> 출력:
# {'max_depth': 14,
#  'min_impurity_decrease': np.float64(0.0004),
#  'min_samples_split': 12}

# 최상의 교차검증 점수?
print(gs.best_score_)
print(np.max(gs.cv_results_['mean_test_score']))
# >>> 출력:
# 0.8683865773302731
# 0.8683865773302731

df = pd.DataFrame(gs.cv_results_)
df
# >>> 출력:
#       mean_fit_time  std_fit_time  mean_score_time  std_score_time  \
# 0          0.032338      0.030893         0.001875        0.000155   
# 1          0.010603      0.004428         0.001763        0.000200   
# 2          0.015064      0.003531         0.003350        0.002479   
# 3          0.011098      0.001468         0.004841        0.003697   
# 4          0.013170      0.002244         0.003395        0.002479   
# ...             ...           ...              ...             ...   
# 1345       0.007561      0.000737         0.001532        0.000110   
# 1346       0.011827      0.006425         0.001668        0.000160   
# 1347       0.007222      0.000330         0.001590        0.000137   
# 1348       0.010741      0.003476         0.001753        0.000189   
# 1349       0.009868      0.003277         0.001706        0.000231   
# 
#       param_max_depth  param_min_impurity_decrease  param_min_samples_split  \
# 0                   5                       0.0001                        2   

# 매개변수의 탐색범위를 보다 더 넓히거나 혹은 좁히거나 하는 방법으로 탐색할수 있을까?

# # 램덤서치  RandomizedSearchCV

# 매개변수 값이 '수치' 일때는 갑의 범위나 간격을 미리 정하기 어려울수도 있다.
# 또한 너무 많은 매개변수 조건은 그리드 서치 시간이 오래 걸리게 할수 있다.

# 이럴때 '랜덤 서치' 사용!
# 랜덤서치에는 값의 목록을 전달하는게 아니라, 매개변수 샘플링 하는 '확률 분포 객체' 를 전달

# scipy 에서 확률분포 클래스
from scipy.stats import uniform, randint

# uniform(), randint() 는 주어진 범위에서 고르게 램덤 값을 뽑습니다.
# => 균등분포 에서 랜덤 샘플링

# uniform() : 실수값 램덤 샘플링
# randint() : 정수값 램덤 샘플링

rgen = randint(0, 10)   # [0, 10) 사이 범위를 갖는 randint 객체 생성.
print(type(rgen))
rgen.rvs(10)  # 10개의 숫자를 랜덤 샘플링
# >>> 출력:
# <class 'scipy.stats._distn_infrastructure.rv_discrete_frozen'>
# array([0, 2, 4, 9, 6, 5, 9, 8, 5, 0])

np.unique(rgen.rvs(1000), return_counts=True)
# >>> 출력:
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
#  array([103,  93,  81,  95, 114, 104, 107,  94, 105, 104]))

ugen = uniform(0, 1)  # [0.0, 1.0) 사이에서 10개의 실수 랜덤 샘플링
ugen.rvs(10)
# >>> 출력:
# array([0.98070615, 0.74291065, 0.63210902, 0.99149939, 0.34311387,
#        0.36982013, 0.86960923, 0.68918884, 0.10434343, 0.26791039])

# 탐색할 매개변수 준비
# 여기서는 min_samples_leaf=  를 탐색대상에 추가해 봅니다.
#      ↑ 이는 리프노드가 되기 위한 최소 샘플의 개수 입니다.
#       어떤 노드가 분할하여 만들어질 자식 노드의 샘플수가 이 값보다 작을 경우 분할하지 않습니다

params = {
    'min_impurity_decrease': uniform(0.0001, 0.001),
    'max_depth': randint(20, 50),
    'min_samples_split': randint(2, 25),
    'min_samples_leaf': randint(1, 25),
}

params
# >>> 출력:
# {'min_impurity_decrease': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x7982aeaaefd0>,
#  'max_depth': <scipy.stats._distn_infrastructure.rv_discrete_frozen at 0x7982aed77e90>,
#  'min_samples_split': <scipy.stats._distn_infrastructure.rv_discrete_frozen at 0x7982aeaafe90>,
#  'min_samples_leaf': <scipy.stats._distn_infrastructure.rv_discrete_frozen at 0x7982aeaae550>}

# 샘플링 횟수는 사이킷런의 랜덤 서치 클래스인 RandomizedSearchCV 의
# n_iter= 매개변수에서 지정

from sklearn.model_selection import RandomizedSearchCV

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
# 
# ```python
# class sklearn.model_selection.RandomizedSearchCV(
#   estimator,
#   param_distributions,  # 확률분포 객체를 가진 params
#   *,
#   n_iter=10, # 샘플링 횟수
#   scoring=None,
#   n_jobs=None,
#   refit=True,
#   cv=None,
#   verbose=0,
#   pre_dispatch='2*n_jobs', random_state=None,
#   error_score=nan, r
#   eturn_train_score=False)
# ```

gs = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    params,
    n_iter=100,  # 샘플링 횟수
    n_jobs=-1,
    random_state=42,
    cv=5
)

gs.fit(train_input, train_target)
# >>> 출력:
# RandomizedSearchCV(cv=5, estimator=DecisionTreeClassifier(random_state=42),
#                    n_iter=100, n_jobs=-1,
#                    param_distributions={'max_depth': <scipy.stats._distn_infrastructure.rv_discrete_frozen object at 0x7982aed77e90>,
#                                         'min_impurity_decrease': <scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x7982aeaaefd0>,
#                                         'min_samples_leaf': <scipy.stats._distn_infrastructure.rv_discrete_frozen object at 0x7982aeaae550>,
#                                         'min_samples_split': <scipy.stats._distn_infrastructure.rv_discrete_frozen object at 0x7982aeaafe90>},
#                    random_state=42)

gs.best_params_
# >>> 출력:
# {'max_depth': 39,
#  'min_impurity_decrease': np.float64(0.00034102546602601173),
#  'min_samples_leaf': 7,
#  'min_samples_split': 13}

print(gs.best_score_)
# >>> 출력:
# 0.8695428296438884

# 최적의 모델은 이미 전체 훈련세트(train_input, train_target)으로 훈련되어 있다.
dt = gs.best_estimator_

# 위 모델을 최종 모델로 결정하고 테스트세트의 성능 확인
print(dt.score(test_input, test_target))
# >>> 출력:
# 0.86

# AutoML : 사람의 개입없이 하이퍼 파라미터 튜닝을 자동으로 수행하느 기술.
