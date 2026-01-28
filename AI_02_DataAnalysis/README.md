# 📊 AI_02_DataAnalysis - 데이터 분석

> NumPy, Pandas, Matplotlib을 활용한 데이터 분석 기초 정리

---

## 📚 목차

| 파일 | 주제 | 핵심 키워드 |
|:----:|------|-------------|
| [AI_23](./step1_numpy_pandas/AI_23_numpy.py) | NumPy | `ndarray`, `shape`, `reshape()`, 브로드캐스팅 |
| [AI_24](./step1_numpy_pandas/AI_24_pandas_series.py) | Pandas Series | `Series`, 인덱싱, `value_counts()` |
| [AI_25](./step1_numpy_pandas/AI_25_pandas_dataframe.py) | Pandas DataFrame | `DataFrame`, `loc`, `iloc`, `groupby()` |
| [AI_26](./step2_visualization_eda/AI_26_matplotlib.py) | Matplotlib | `plot()`, `bar()`, `scatter()`, `hist()` |
| [AI_27](./step2_visualization_eda/AI_27_seaborn.py) | Seaborn | `heatmap()`, `pairplot()`, `countplot()` |
| [AI_28](./step2_visualization_eda/AI_28_eda.py) | EDA | 탐색적 데이터 분석, 결측치, 이상치 |
| [AI_30](./step3_ml_basics/AI_30_scikit_learn.py) | Scikit-learn | `fit()`, `predict()`, `train_test_split()` |
| [AI_31](./step3_ml_basics/AI_31_ml_basics.py) | ML 기초 | 지도학습, 비지도학습, 과적합 |
| [AI_32](./step3_ml_basics/AI_32_regression.py) | 회귀 분석 | `LinearRegression`, R², MSE |

---

## 1️⃣ NumPy

수치 연산을 위한 핵심 라이브러리이다.

### 배열 생성
```python
import numpy as np

# 배열 생성
arr = np.array([1, 2, 3, 4, 5])
print(arr.shape)    # (5,)
print(arr.dtype)    # int64

# 특수 배열
zeros = np.zeros((3, 3))    # 0으로 채운 배열
ones = np.ones((2, 4))      # 1로 채운 배열
arange = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
```

### 배열 연산
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)   # [5, 7, 9]
print(a * b)   # [4, 10, 18]
print(a @ b)   # 32 (내적)

# 통계 함수
print(np.mean(a))  # 2.0
print(np.std(a))   # 0.816
print(np.sum(a))   # 6
```

### Reshape
```python
arr = np.arange(12)
reshaped = arr.reshape(3, 4)
print(reshaped)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
```

---

## 2️⃣ Pandas

데이터 분석을 위한 핵심 라이브러리이다.

### Series
1차원 데이터 구조이다.

```python
import pandas as pd

s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s['a'])        # 10
print(s.values)      # [10, 20, 30]
print(s.index)       # Index(['a', 'b', 'c'])
```

### DataFrame
2차원 테이블 구조이다.

```python
df = pd.DataFrame({
    'name': ['Kim', 'Lee', 'Park'],
    'age': [25, 30, 28],
    'city': ['Seoul', 'Busan', 'Seoul']
})

print(df.head())       # 상위 5행
print(df.info())       # 컬럼 정보
print(df.describe())   # 통계 요약
```

### 데이터 조회
```python
# 컬럼 선택
df['name']
df[['name', 'age']]

# 행 선택
df.loc[0]              # 인덱스로 선택
df.iloc[0:2]           # 위치로 선택

# 조건 필터링
df[df['age'] > 25]
df[df['city'] == 'Seoul']
```

### 그룹화와 집계
```python
df.groupby('city')['age'].mean()
# city
# Busan    30.0
# Seoul    26.5

df.groupby('city').agg({
    'age': ['mean', 'max'],
    'name': 'count'
})
```

---

## 3️⃣ Matplotlib

데이터 시각화 라이브러리이다.

### 기본 플롯
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.xlabel('X축')
plt.ylabel('Y축')
plt.title('선 그래프')
plt.show()
```

### 다양한 그래프
```python
# 막대 그래프
plt.bar(['A', 'B', 'C'], [10, 20, 15])

# 산점도
plt.scatter(x, y)

# 히스토그램
data = np.random.randn(1000)
plt.hist(data, bins=30)

# 파이 차트
plt.pie([30, 40, 30], labels=['A', 'B', 'C'])
```

### 서브플롯
```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(x, y)
axes[0].set_title('Line')

axes[1].bar(['A', 'B'], [10, 20])
axes[1].set_title('Bar')

plt.tight_layout()
plt.show()
```

---

## 4️⃣ Seaborn

고급 시각화 라이브러리이다.

```python
import seaborn as sns

# 히트맵 (상관관계)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# 카운트 플롯
sns.countplot(data=df, x='city')

# 박스 플롯
sns.boxplot(data=df, x='city', y='age')

# 페어플롯
sns.pairplot(df)
```

---

## 5️⃣ EDA (탐색적 데이터 분석)

### 기본 분석
```python
# 데이터 확인
df.shape          # (행, 열)
df.columns        # 컬럼명
df.dtypes         # 데이터 타입
df.isnull().sum() # 결측치 개수

# 기술 통계
df.describe()
df['column'].value_counts()
```

### 결측치 처리
```python
# 결측치 확인
df.isnull().sum()

# 결측치 제거
df.dropna()

# 결측치 대체
df.fillna(0)
df.fillna(df.mean())
```

---

## 6️⃣ Scikit-learn 기초

### 기본 워크플로우
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 3. 예측
y_pred = model.predict(X_test)

# 4. 평가
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")
```

---

## 📖 학습 순서

```
step1 (NumPy/Pandas) → step2 (시각화/EDA) → step3 (ML 기초)
```
