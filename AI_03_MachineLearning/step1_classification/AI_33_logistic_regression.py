# # logistic regression

# NLP_33_LogisticRegression [분류]
# 
# 로지스틱 회귀(Logistic Regression)는 분류 문제를 해결하는 대표적인 선형 모델이다.
# 확률 기반의 이진 분류 및 다중 분류 문제를 해결한다.
# 
# 정의:
# - 로지스틱 함수를 사용하여 입력값을 0과 1 사이의 확률로 변환한다.
# - 선형 결합에 시그모이드 함수를 적용하여 클래스 확률을 예측한다.
# - 최대 우도 추정(Maximum Likelihood Estimation)을 통해 가중치를 학습한다.
# 
# 특징:
# - 빠른 학습과 예측 속도
# - 확률 기반 예측으로 신뢰도 제공
# - 과적합 위험이 적음
# - 특성 간 상호작용을 직접 고려하지 않음
# 
# 용도:
# - 이진 분류 문제
# - 다중 클래스 분류 문제
# - 확률 기반 예측이 필요한 경우

# # ==================== 기본 라이브러리 임포트 ====================
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# # ==================== 1. 데이터 로딩 및 전처리 ====================
# 
# def load_fish_data():

# 생선 데이터셋 로딩 함수
#     - 7가지 생선 종류를 분류하는 다중 클래스 분류 문제
#     - 특성: 길이, 높이, 두께, 대각선, 무게

# fish = pd.read_csv('https://bit.ly/fish_csv_data')
#     return fish
# 
# def preprocess_data(fish):

# 데이터 전처리 함수
#     - 특성과 타겟 분리
#     - 훈련/테스트 분할
#     - 표준화 처리

# # 특성과 타겟 분리
#     X = fish[['Length', 'Height', 'Width', 'Diagonal', 'Height', 'Width']].values
#     y = fish['Species'].values
# 
#     # 훈련/테스트 분할 (80:20)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
# 
#     # 특성 표준화 (로지스틱 회귀는 스케일링이 중요함)
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
# 
#     return X_train, X_test, y_train, y_test, scaler
# 
# # ==================== 2. 로지스틱 회귀 모델 학습 ====================
# 
# def train_logistic_regression(X_train, y_train):

# 로지스틱 회귀 모델 학습 함수
# 
#     Parameters:
#     - max_iter: 최적화 알고리즘의 최대 반복 횟수
#     - C: 규제 강도의 역수 (작을수록 강한 규제)
#     - random_state: 재현성을 위한 시드값

# lr = LogisticRegression(
#         max_iter=1000,
#         C=1.0,
#         random_state=42
#     )
# 
#     # 모델 학습
#     lr.fit(X_train, y_train)
# 
#     return lr
# 
# # ==================== 3. 모델 평가 ====================
# 
# def evaluate_model(model, X_train, X_test, y_train, y_test):

# 모델 성능 평가 함수
#     - 훈련 정확도와 테스트 정확도 비교
#     - 혼동 행렬 계산
#     - 분류 리포트 출력

# # 예측값 계산
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)
# 
#     # 정확도 계산
#     train_acc = accuracy_score(y_train, y_train_pred)
#     test_acc = accuracy_score(y_test, y_test_pred)
# 
#     print(f"훈련 정확도: {train_acc:.4f}")
#     print(f"테스트 정확도: {test_acc:.4f}")
# 
#     # 혼동 행렬
#     cm = confusion_matrix(y_test, y_test_pred)
#     print("\n혼동 행렬:")
#     print(cm)
# 
#     # 분류 리포트
#     print("\n분류 리포트:")
#     print(classification_report(y_test, y_test_pred))
# 
#     return train_acc, test_acc, cm
# 
# # ==================== 4. 확률 예측 ====================
# 
# def predict_probabilities(model, X_new):

# 새로운 샘플에 대한 클래스 확률 예측 함수
# 
#     로지스틱 회귀는 predict_proba()를 통해 각 클래스에 대한 확률을 반환한다.
#     - 반환값: (샘플 수, 클래스 수) 형태의 확률 배열
#     - 각 행의 합은 1.0 (확률의 합)

# # 확률 예측
#     proba = model.predict_proba(X_new)
# 
#     # 클래스명
#     classes = model.classes_
# 
#     # 결과를 DataFrame으로 변환 (가독성 향상)
#     result_df = pd.DataFrame(proba, columns=classes)
# 
#     return result_df
# 
# # ==================== 5. 시각화 ====================
# 
# def visualize_confusion_matrix(cm, classes):

# 혼동 행렬 시각화 함수

# plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=classes, yticklabels=classes)
#     plt.title('Confusion Matrix - Logistic Regression')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.show()
# 
# # ==================== 6. 모델 특성값 확인 ====================
# 
# def print_model_coefficients(model, feature_names, classes):

# 로지스틱 회귀 모델의 계수(가중치) 확인 함수
# 
#     각 특성이 클래스 예측에 미치는 영향을 분석한다.
#     - 양수: 해당 클래스 확률 증가
#     - 음수: 해당 클래스 확률 감소

# print("=== 모델 계수(가중치) ===")
# 
#     coef_df = pd.DataFrame(
#         model.coef_.T,
#         columns=classes,
#         index=feature_names
#     )
# 
#     print(coef_df)
# 
#     # 절편(intercept)
#     print("\n절편(Intercept):")
#     print(pd.Series(model.intercept_, index=classes))
# 
# # ==================== 7. 예시 코드 ====================
# 
# if __name__ == "__main__":
#     # 데이터 로딩
#     print("데이터 로딩 중...")
#     fish = load_fish_data()
#     print(f"데이터셋 크기: {fish.shape}")
# 
#     # 데이터 전처리
#     print("\n데이터 전처리 중...")
#     X_train, X_test, y_train, y_test, scaler = preprocess_data(fish)
# 
#     # 모델 학습
#     print("\n모델 학습 중...")
#     lr_model = train_logistic_regression(X_train, y_train)
# 
#     # 모델 평가
#     print("\n모델 평가:")
#     train_acc, test_acc, cm = evaluate_model(lr_model, X_train, X_test, y_train, y_test)
# 
#     # 새로운 샘플에 대한 확률 예측
#     print("\n새로운 샘플 확률 예측:")
#     sample = X_test[:5]
#     proba = predict_probabilities(lr_model, sample)
#     print(proba)
# 
#     # 모델 계수 출력
#     print("\n모델 계수:")
#     feature_names = ['Length', 'Height', 'Width', 'Diagonal', 'Height', 'Width']
#     classes = lr_model.classes_
#     print_model_coefficients(lr_model, feature_names, classes)
# 
# # ==================== 핵심 요점 ====================

# 1. 로지스틱 회귀는 선형 모델이므로 빠른 학습이 가능함
# 2. 특성 표준화(StandardScaler)가 중요함 - 계수 해석을 용이하게 함
# 3. predict_proba()로 확률 기반 예측을 제공함
# 4. 다중 클래스 분류는 One-vs-Rest 방식으로 처리됨
# 5. 특성과 타겟 간의 선형 관계를 가정함
