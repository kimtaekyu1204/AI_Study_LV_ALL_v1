# # 정형 데이터 와 비정형 데이터
# 
# 1. 정형데이터(Structured Data)
#     - 주로 행 과 열로 구성된 형태의 데이터
#     - ex) CSV, 데이터베이스, 엑셀 파일 ..
#     
#     
# 1. 비정형 데이터 (Unstructured Data)
#     - 정형데이터 형태로 표현하기 어려운 데이터들
#     - ex) 책, 글 텍스트, 사진, 음악

# # 앙상블 학습이란
# (ensemble learning)
# 
# 더 좋은 예측 결과를 만들기 위해 '여러 개의 모델'을 훈련하는 머신러닝 알고리즘

# 정형데이타를 다룰때 가장 뛰어난 성과를 내는 알고리즘 => '앙상블 학습(ensemble learning)'

# 앙상블 학습 이란 '여러개의 모델'을 훈련하여 더 좋은 예측 결과를 내는 머신러닝 알고리즘
# ※ 대부분 '결정트리' 기반으로 만들어진다..

# 반면, 비정형 데이터는 어떤 알고리즘 사용?  --> '신경망 알고리즘'
# 비정형 데이터는 '규칙성' 을 찾기 어려워서 전통적인 머신러닝 알고리즘으로는 모델만들기 어렵다.

# # 랜덤 포레스트 Random Forest

# Decision Tree : 나무
# Random Forest : decision tree 를 랜덤하게 만들어 '숲' 을 만든다.
#                각 decision tree 의 예측을 사용해 '최종예측'을 만든다

# ![](https://blog.kakaocdn.net/dn/AdyUy/btqFWYWEPgW/ROyxKdQFvQDTItuPM3fKB1/img.png)
# ![](https://blog.kakaocdn.net/dn/C3YZf/btroCvP78MF/ewyyGhKks8JdcqKywK2JS0/img.png)

# ## 부트스트랩 샘플 (bootstrap sample)

# 램덤포레스트는 각 트리를 훈련하기 위핸 데이터를 아래와 같이 랜덤하게 만든다.

# 입력한 훈련데이터에서 '랜덤하게 샘플을 추출'하여 훈련데이터를 만드는데.
# 이때 한 샘플이 '중복' 되어 추출될 수도 있습니다.
# 이런식으로 만들어진 샘플을 부트스트랩 샘플 이라 한다.

# 기본적으로 부트스트램 샘플은 '훈련세트와 같은 크기' 로 만든다.
# ex) 1000개가 들어있는 가방에서 1000개의 샘플을 뽑기 (중복허용)

# ## RandomForestClassifier

# 트리에선 각 노드를 분할할때 '전체특성' 중에서 '일부특성' 을 '무작위' 로 고른다음 '최선의 분할' 을 찾습니다.

# RandomForestClassifier 는 기본적으로 '전체특성의 개수의 제곱근만큼' 의 특성을 선택함.
#    ex) 전체 4개의 특성이 있다면 -> 노드마다 2개를 램덤하게 선택하여 사용.

# RandomForestRegressor 는 '전체특성을 사용'

# 사이킷럿의 랜덤포레스트는 기본적으로 100개의 결정트리를 위와 같은 방식으로 훈련함.

# 최종 예측은!
#  '분류' 일때는 각 트리의 클래스별 확률을 평균하여 가장 높은 확률을 가진 클래스를 예측으로 삼습니다
#  '회귀' 일때는 단순히 각 트리의 예측을 평균합니다.

"""
랜덤 포레스트는 랜덤하게 선택한 샘플과 특성을 사용하기 때문에
훈련세트에 과대적합되는 것을 막아주고
검증세트와 테스트세트에서 안정적인 성능을 얻을 수 있습니다.
종종 기본 매개변수 설정 만으로도 아주 좋은 결과를 냅니다.
"""
None

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

# train / test 세트 나누기
from sklearn.model_selection import train_test_split

# train: test = 8:2 로 나누기
train_input, test_input, train_target, test_target = \
  train_test_split(data, target, test_size=0.2, random_state=42)

# cross_validate()
#  return_train_score=  를 True 로 지정하면, 검증 점수뿐 아니라 훈련세트에 대한 점수로 같이 리턴함.
#                               => 훈련세트와 검증세트 점수를 비교하여 overfit 파악 용이.

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# 
# ```python
# class sklearn.ensemble.RandomForestClassifier(
#   n_estimators=100,
#   *,
#   criterion='gini',
#   max_depth=None,
#   min_samples_split=2,
#   min_samples_leaf=1,
#   min_weight_fraction_leaf=0.0,
#   max_features='sqrt',
#   max_leaf_nodes=None,
#   min_impurity_decrease=0.0,
#   bootstrap=True,
#   oob_score=False, # out of bag 샘플을 사용한 평가점수
#   n_jobs=None,  # 학습에 사용하는 cpu 코어개수
#   random_state=None, # 랜덤 시드
#   verbose=0,
#   warm_start=False,
#   class_weight=None,
#   ccp_alpha=0.0,
#   max_samples=None,
#   monotonic_cst=None)
# ````
# 
# ↑ 기본적으로 DecisionTreeClassifier 의 거의 대부분의 매개변수들을 가지고 있다

rf = RandomForestClassifier(n_jobs=-1, random_state=42)

scores = cross_validate(rf, train_input, train_target,
                        return_train_score=True, # 훈련데이터에 대한 점수도 리턴
                        n_jobs=-1)

scores
# >>> 출력:
# {'fit_time': array([2.06796288, 1.90983081, 2.15232301, 2.30740261, 1.61568284]),
#  'score_time': array([0.16433001, 0.13888812, 0.21824956, 0.15233898, 0.08719587]),
#  'test_score': array([0.88461538, 0.88942308, 0.90279115, 0.88931665, 0.88642926]),
#  'train_score': array([0.9971133 , 0.99663219, 0.9978355 , 0.9973545 , 0.9978355 ])}

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# >>> 출력:
# 0.9973541965122431 0.8905151032797809

# ↑ 훈련세트에 오버핏!

# ## 특성 중요도 feature_importances_

# 랜덤포레스트의 '특성중요도' 는 각 결정트리의 특성중요도를 취합한 거다

rf.fit(train_input, train_target)
print(rf.feature_importances_)
# >>> 출력:
# [0.23167441 0.50039841 0.26792718]

#    알코올       당도       pH
# [0.12345626, 0.86862934, 0.0079144 ]  <- DecisionTree

# [0.23167441 0.50039841 0.26792718]   <- RandomForest

# 당도의 중요도는 감소하고, 알코올과 ph 중요도가 상승했다.
# 이유는!  랜덤포레스트라 '특성의 일부' 를 '랜덤' 하게 선택하여 결정트리를 훈련함!
#   => 그 결과 하나의 특성에 과도하게 집중하지 않고, 더 많은 특성이 훈련에 기여할 기회를 얻게 됨.
#   => overfit 을 줄이고 일반화 성능을 높일수 있을것으로 기대.

# ## OOB (out of bag) 샘플
# 부트스트랩 샘플에 포함되지 않고 남은 샘플

# RandomForestClassifier 는 OOB 샘플을 활용하여 자체적으로 모델을 평가하는 기능이 있다

#  oob_score= 를 True 로 지정해야 한다

rf = RandomForestClassifier(
    oob_score=True,  # OOB 활용!
    n_jobs=-1,
    random_state=42
)

rf.fit(train_input, train_target)
print(rf.oob_score_)
# >>> 출력:
# 0.8934000384837406

# # 엑스트라 트리
# Extra Tree

# ## ExtraTreesClassifier

# 엑스트라트리 가

# 랜덤포레스트와 비슷한점
#    - 기본적으로 100개의 결정트리 훈련
#    - 결정트리가 제공하는 대부분의 매개변수 지원
#    - 일부특성을 램덤하게 선택하여 노드 분할

# 랜덤포레스트와의 차이점!
#    - 부트스프램 샘플 사용하지 않는다!
#    - 결정트리 만들때 '전체세트' 사용.
#    - 노드를 분할할때 가장 좋은 분할을 찾는게 아니라 '무작위 분할'함!
#       - 사실 DecisionTreeClassifier 에서 splitter='random' 으로 지정한 결정트리가 바로 이거다.

# 하나의 결정트리였다면.. 무작위 분할은 성능이 낮아지지만
# 많은 트리를 앙상블 하면 overfit 억제하고 검증점수 높이는 효과 기대.

from sklearn.ensemble import ExtraTreesClassifier

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
# ```python
# class sklearn.ensemble.ExtraTreesClassifier(
#   n_estimators=100,
#   *,
#   criterion='gini',
#   max_depth=None,
#   min_samples_split=2,
#   min_samples_leaf=1,
#   min_weight_fraction_leaf=0.0,
#   max_features='sqrt',
#   max_leaf_nodes=None,
#   min_impurity_decrease=0.0,
#   bootstrap=False,
#   oob_score=False,
#   n_jobs=None,
#   random_state=None,
#   verbose=0,
#   warm_start=False,
#   class_weight=None,
#   ccp_alpha=0.0,
#   max_samples=None,
#   monotonic_cst=None)
# ```

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target,
                        return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# >>> 출력:
# 0.9974503966084433 0.8887848893166506

# 일반적으론,
# 엑스트라트라가 무작위성이 좀 더 크기 때문에 랜덤포레스트보다는 더 많은 결정트리 훈련해야 한다.
# 램덤하게 노드 분할하기때문에 계산속도는 상대적으로 빠르다

# 특성중요도
et.fit(train_input, train_target)
print(et.feature_importances_)
# >>> 출력:
# [0.20183568 0.52242907 0.27573525]

#    알코올       당도       pH
# [0.12345626, 0.86862934, 0.0079144 ]  <- DecisionTree

# [0.23167441 0.50039841 0.26792718]   <- RandomForest

# [0.20183568 0.52242907 0.27573525]   <- ExtraTrees

# # 그레디언트 부스팅
# Gradient Boosting

# ## GradientBoostingClassifier

# GradientBoostingClassifier 는 얕은깊이의 결정트리 사용
#    깊이3 결정트리 x 100개 사용
#    깊이가 얕기에 오버핏 억제에 유리, 높은 일반화 성능 기대.

# '경사하강법' 을 사용하여 트리를 앙상블에 추가함.

# 그레디언트는 무엇이? 낮을 곳을 찾아 이동하게 되나?
#  '그레디언트 부스팅' 은 결정트리를 '계속 추가'하면서 가장 낮은 곳을 찾아 이동.
#   손실함수의 낮은곳으로 '천천히 조금씩' 이동하듯이
#   그레디언트 부스팅도 '깊이가 얕은 트리를 사용'하여  조금씩 낮은곳으로 이동.

from sklearn.ensemble import GradientBoostingClassifier

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
# 
# ```python
# class sklearn.ensemble.GradientBoostingClassifier(
#   *,
#   loss='log_loss',   # 손실함수
#   learning_rate=0.1, # 학습률  
#   n_estimators=100,  # 트리의 개수
#   subsample=1.0,
#   criterion='friedman_mse',
#   min_samples_split=2,
#   min_samples_leaf=1,
#   min_weight_fraction_leaf=0.0,
#   max_depth=3,
#   min_impurity_decrease=0.0,
#   init=None,
#   random_state=None,
#   max_features=None,
#   verbose=0,
#   max_leaf_nodes=None,
#   warm_start=False,
#   validation_fraction=0.1,
#   n_iter_no_change=None,
#   tol=0.0001,
#   ccp_alpha=0.0)
# ```

gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target,
                        return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# >>> 출력:
# 0.8881086892152563 0.8720430147331015

#  ↑ overfit 이 많이 억제되고 있다.
#  그레디언트 부스팅은 결정트리의 개수를 늘려도 overfit 에 강하다.

# 학습률 learning_rate 와 트리 개수 증가.

gb = GradientBoostingClassifier(
      n_estimators=500,   # 트리의 개수를 늘려보자
      learning_rate=0.2,   # 학습률 증가 (기본값 0.1)
      random_state=42
)
scores = cross_validate(gb, train_input, train_target,
                        return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# >>> 출력:
# 0.9464595437171814 0.8780082549788999

# 결정트리의 개수를 x5배나 늘렸지만, 그에 비해선 overfit 을 잘 억제하고 있다.

# 특성중요도
gb.fit(train_input, train_target)
print(gb.feature_importances_)
# >>> 출력:
# [0.15887763 0.6799705  0.16115187]

#    알코올       당도       pH
# [0.12345626, 0.86862934, 0.0079144 ]  <- DecisionTree

# [0.23167441 0.50039841 0.26792718]   <- RandomForest

# [0.20183568 0.52242907 0.27573525]   <- ExtraTrees

# [0.15887763 0.6799705  0.16115187]  <- GradientBoosting

# ## subsample= 파라미터

# subsample: 트리훈련에 사용할 훈련세트의 비율을 정함.
#       기본값 1.0 => '전체훈련세트' 사용
#       1.0보다 작으면 => '훈련세트의 일부' 사용
#              => 확률적 경사하강법, 미니배치경사하강법 과 비슷하게
#                 경사 하강법 단계마다 일부 샘플을 랜덤하게 선택하여 진행.

# 일반적으로 그레디언트 부스팅이 랜덤포레스트보다 성능은 좀더 높지만, 훈련속도가 느리다.
#  이유는!  n_jobs 매개변수가 없다.  (병렬 프로세싱 안된다...)
#  순서대로 트리를 추가해가면서 훈련하기 때문이다.

# # 히스토그램 기반 그레디언트 부스팅
# Histogram-based Gradient Boosting

# ## HistGradientBoostingClassifier

# 입력데이터의 특성을 '구간' 으로 나눈다.  (기본 256개의 구간)
# => 노드를 분할할때 최적의 분할을 매우 빠르게 찾을수 있다.

# 256개의 구간중에서 하나는 떼어놓는다 => 누락된 값을 위해서 사용하기 위함.

# 기본 매개변수에서 안정적인 성능
# 트리의 개수를 지정하는데 n_estimators= 대신에  max_iter= 사용.
# 부스팅 반복횟수를 지정하는 max_iter 사용.  <- 성능을 높일때 이 매개변수를 조정해보세요.

from sklearn.ensemble import HistGradientBoostingClassifier

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
# 
# ```python
# class sklearn.ensemble.HistGradientBoostingClassifier(
#   loss='log_loss',
#   *,
#   learning_rate=0.1,
#   max_iter=100, # 부스팅 반복 횟수
#   max_leaf_nodes=31,
#   max_depth=None,
#   min_samples_leaf=20,
#   l2_regularization=0.0,
#   max_features=1.0,
#   max_bins=255,
#   categorical_features='from_dtype',
#   monotonic_cst=None,
#   interaction_cst=None,
#   warm_start=False,
#   early_stopping='auto',
#   scoring='loss',
#   validation_fraction=0.1,
#   n_iter_no_change=10,
#   tol=1e-07,
#   verbose=0,
#   random_state=None,
#   class_weight=None)
# ```

hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# >>> 출력:
# 0.9321723946453317 0.8801241948619236

# 0.8881086892152563 0.8720430147331015  <- Gradient Boosting
# 0.9321723946453317 0.8801241948619236  <- HistGradientBoosting

# ## 특성중요도 permuration_importance()

# 특성중요도 : permutation_importance() 사용
#  특성을 하나씩 램덤하게 섞어서 모델의 성능이 변화하는지 관찰한 후 어떤 특성이 중요한지 계산
#  n_repeats=  : 랜덤하게 섞을 횟수 지정 (디폴트 5)

from sklearn.inspection import permutation_importance

# https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
# 
# ```python
# sklearn.inspection.permutation_importance(
#   estimator,
#   X,
#   y,
#   *,
#   scoring=None,
#   n_repeats=5, # 랜덤하게 섞을 횟수
#   n_jobs=None,
#   random_state=None,
#   sample_weight=None,
#   max_samples=1.0)
# ```

# '훈련세트' 에서 특성 중요도 계산
hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10,
                                random_state=42, n_jobs=-1)

print(result.importances_mean)
# >>> 출력:
# [0.08876275 0.23438522 0.08027708]

#  알코올        당도       pH

# [0.12345626, 0.86862934, 0.0079144 ]  <- DecisionTree

# [0.23167441 0.50039841 0.26792718]   <- RandomForest

# [0.20183568 0.52242907 0.27573525]   <- ExtraTrees

# [0.15887763 0.6799705  0.16115187]  <- GradientBoosting

# [0.08876275 0.23438522 0.08027708]  <= Hist Gradient Boosting

# '테스트 세트' 에서 특성 중요도 계산
result = permutation_importance(hgb, test_input, test_target, n_repeats=10,
                                random_state=42, n_jobs=-1)

print(result.importances_mean)
# >>> 출력:
# [0.05969231 0.20238462 0.049     ]

# [0.05969231 0.20238462 0.049     ]

# 테스트 세트 점수
hgb.score(test_input, test_target)
# >>> 출력:
# 0.8723076923076923

# ## XGBoost

# XGBoost 는 다양한 부스팅 알고리즘 지원
# tree_method=  를 'hist' 로 지정하면
# 히스토그램 기반 그레디언트 부스팅 을 사용할수 있다

from xgboost import XGBClassifier

xgb = XGBClassifier(tree_method='hist', random_state=42)

scores = cross_validate(xgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# >>> 출력:
# 0.9558403027491312 0.8782000074035686

# ## LightGBM

# XGBoost보다 속도가 빠르고 메모리 효율성이 높음

from lightgbm import LGBMClassifier

lgb = LGBMClassifier(random_state=42)

scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# >>> 출력:
# 0.935828414851749 0.8801251203079884

# # 앙상블 학습 (정리)
# 
# 앙상블 학습은 **'정형 데이터'**에서 가장 뛰어난 성능을 내는 머신러닝 알고리즘 중 하나입니다
# 
# ↓대표적인 앙상블 학습들
# 
# 1. 사이킷럿
#     1. 랜덤포레스트 : 부트스트랩 샘플 사용. 대표 앙상블 학습알고리즘
#     1. 엑스트라 트리 : 결정트리의 노드를 랜덤하게 분할함
#     1. 그레디언트 부스팅 : 이전 트리의 손실을 보완하는 식으로 얕은 결정 트리를 연속적으로 추가함
#     1. 히스토그램 기반 그레디언트 부스팅 : 훈련 데이터를 256개 정수 구간으로 나누어 빠르고 높은 성능을 냄
#     
# 1. 그 밖의 라이브러리
#     1. XGBoost
#     1. LightGBM
