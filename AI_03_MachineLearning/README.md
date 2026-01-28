# 🔮 AI_03_MachineLearning - 머신러닝

> 머신러닝 알고리즘과 신경망 기초 정리

---

## 📚 목차

| 파일 | 주제 | 핵심 키워드 |
|:----:|------|-------------|
| [AI_33](./step1_classification/AI_33_logistic_regression.py) | 로지스틱 회귀 | `LogisticRegression`, 이진 분류, Sigmoid |
| [AI_34](./step1_classification/AI_34_sgd.py) | SGD | `SGDClassifier`, 확률적 경사 하강법 |
| [AI_35](./step2_tree_ensemble/AI_35_decision_tree.py) | 결정 트리 | `DecisionTreeClassifier`, 불순도, 가지치기 |
| [AI_36](./step2_tree_ensemble/AI_36_cross_validation_gridsearch.py) | 교차검증 | `cross_val_score`, `GridSearchCV` |
| [AI_37](./step2_tree_ensemble/AI_37_tree_ensemble.py) | 앙상블 | `RandomForest`, `GradientBoosting`, `XGBoost` |
| [AI_38](./step3_unsupervised/AI_38_clustering.py) | 군집화 | `KMeans`, `DBSCAN`, 엘보우 기법 |
| [AI_39](./step3_unsupervised/AI_39_pca.py) | PCA | 주성분 분석, 차원 축소, `explained_variance_` |
| [AI_40](./step4_neural_network/AI_40_dl_terminology.py) | 딥러닝 용어 | 에폭, 배치, 손실함수, 활성화 함수 |
| [AI_41](./step4_neural_network/AI_41_neural_network.py) | 인공신경망 | `Dense`, 퍼셉트론, MLP |
| [AI_42](./step4_neural_network/AI_42_neural_network_training.py) | 신경망 훈련 | `Dropout`, `BatchNormalization`, 과적합 방지 |
| [AI_43](./step4_neural_network/AI_43_cnn.py) | CNN | `Conv2D`, `MaxPooling2D`, 합성곱 |

---

## 1️⃣ 분류 (Classification)

### 로지스틱 회귀
이진 분류를 위한 선형 모델이다.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)  # 확률값

print(f"정확도: {model.score(X_test, y_test)}")
```

### SGD (확률적 경사 하강법)
대용량 데이터에 효율적인 학습 방법이다.

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='log_loss', max_iter=1000)
model.fit(X_train, y_train)
```

---

## 2️⃣ 트리 기반 모델

### 결정 트리
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 특성 중요도
print(model.feature_importances_)
```

### Random Forest (랜덤 포레스트)
여러 결정 트리의 앙상블이다.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,  # 트리 개수
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)
```

### Gradient Boosting
순차적으로 약한 학습기를 결합한다.

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
model.fit(X_train, y_train)
```

---

## 3️⃣ 교차 검증 & 하이퍼파라미터 튜닝

### 교차 검증 (Cross Validation)
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"평균 정확도: {scores.mean():.4f}")
print(f"표준편차: {scores.std():.4f}")
```

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 점수: {grid_search.best_score_:.4f}")
```

---

## 4️⃣ 비지도 학습

### K-Means 군집화
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

print(f"클러스터 중심: {kmeans.cluster_centers_}")
print(f"관성: {kmeans.inertia_}")  # 엘보우 기법용
```

### PCA (주성분 분석)
차원 축소 기법이다.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"설명된 분산 비율: {pca.explained_variance_ratio_}")
print(f"누적 분산: {sum(pca.explained_variance_ratio_):.4f}")
```

---

## 5️⃣ 인공신경망 (Neural Network)

### 기본 구조
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')  # 다중 분류
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 학습
```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=5)]
)
```

---

## 6️⃣ CNN (합성곱 신경망)

이미지 처리에 특화된 신경망이다.

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

### CNN 구성 요소
| 레이어 | 설명 |
|:------:|------|
| `Conv2D` | 합성곱 연산, 특징 추출 |
| `MaxPooling2D` | 다운샘플링, 연산량 감소 |
| `Flatten` | 2D → 1D 변환 |
| `Dense` | 완전연결층 |

---

## 📖 학습 순서

```
step1 (분류) → step2 (트리/앙상블) → step3 (비지도학습) → step4 (신경망)
```
