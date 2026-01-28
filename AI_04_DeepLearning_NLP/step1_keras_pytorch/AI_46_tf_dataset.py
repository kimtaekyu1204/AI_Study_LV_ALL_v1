# # ■ tf.data.Dataset
# 
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# 
# 
# 데이터 입력 pipeline 객체
# 1. 입력데이터로부터 source dataset 생성
# 1. dataset 에 대한 transformation 수행 (preprocess!)
# 1. dataset 의 각 element 에 대해 일련의 작업 수행
# 1. model.fit() 의 매개변수로 Dataset 전달
# 
# 참조 : TF 입력파이프라인 빌드 https://www.tensorflow.org/guide/data

import tensorflow as tf
import numpy
from tensorflow.data import Dataset  # tf.data.Dataset

def set_seed(seed = 42):
  tf.keras.utils.set_random_seed(seed)
  tf.config.experimental.enable_op_determinism()

set_seed(42)

# ## from_tensor_slices() : Dataset 생성
# tf.data.Dataset.from_tensor_slices
# 
# 주어진 tensors 로 Dataset 생성
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices
# 
# ```python
# @staticmethod
# from_tensor_slices(
#     tensors, name=None
# ) -> 'DatasetV2'
# ```

# Slicing a 2D tensor produces 1D tensor elements.
# 2D tensor 를 slice 하여 1D tensor element 생성

ds = Dataset.from_tensor_slices([[1, 2], [3, 4]])
ds
# >>> 출력:
# <_TensorSliceDataset element_spec=TensorSpec(shape=(2,), dtype=tf.int32, name=None)>

# as_numpy_iterator() : numpy iterable 추출

list(ds.as_numpy_iterator())
# >>> 출력:
# [array([1, 2], dtype=int32), array([3, 4], dtype=int32)]

# Slicing a tuple of 1D tensors produces tuple elements containing
# scalar tensors.

ds = Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))
list(ds.as_numpy_iterator())
# >>> 출력:
# [(np.int32(1), np.int32(3), np.int32(5)),
#  (np.int32(2), np.int32(4), np.int32(6))]

# Dictionary structure is also preserved.

ds = Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})
list(ds.as_numpy_iterator())
# >>> 출력:
# [{'a': np.int32(1), 'b': np.int32(3)}, {'a': np.int32(2), 'b': np.int32(4)}]

# Two tensors can be combined into one Dataset object.
features = tf.constant([[1, 3], [2, 1], [3, 3]]) # ==> 3x2 tensor
labels = tf.constant(['A', 'B', 'A']) # ==> 3x1 tensor

print(features)
print(labels)

ds = Dataset.from_tensor_slices((features, labels))
list(ds.as_numpy_iterator())
# >>> 출력:
# tf.Tensor(
# [[1 3]
#  [2 1]
#  [3 3]], shape=(3, 2), dtype=int32)
# tf.Tensor([b'A' b'B' b'A'], shape=(3,), dtype=string)
# [(array([1, 3], dtype=int32), b'A'),
#  (array([2, 1], dtype=int32), b'B'),
#  (array([3, 3], dtype=int32), b'A')]

# ## zip() : Dataset들 묶기
# tf.data.Dataset.zip()
# 
# 주어진 dataset(들)을 묶어서 Dataset 생성
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#zip
# 
# ```python
# @staticmethod
# zip(
#     *args, datasets=None, name=None
# ) -> 'DatasetV2'
# ```

# Both the features and the labels tensors can be converted
# to a Dataset object separately and combined after.
features_dataset = Dataset.from_tensor_slices(features)
labels_dataset = Dataset.from_tensor_slices(labels)

ds = Dataset.zip((features_dataset, labels_dataset))
list(ds.as_numpy_iterator())
# >>> 출력:
# [(array([1, 3], dtype=int32), b'A'),
#  (array([2, 1], dtype=int32), b'B'),
#  (array([3, 3], dtype=int32), b'A')]

# A batched feature and label set can be converted to a Dataset
# in similar fashion.

# 배치 feature  와 label 세트도 위와 비슷한 방식으로 Dataset 으로 변환 가능
batched_features = tf.constant([[[1, 3], [2, 3]],
                                [[2, 1], [1, 2]],
                                [[3, 3], [3, 2]]], shape=(3, 2, 2))
batched_labels = tf.constant([['A', 'A'],
                              ['B', 'B'],
                              ['A', 'B']], shape=(3, 2, 1))

ds = Dataset.from_tensor_slices((batched_features, batched_labels))

for element in ds.as_numpy_iterator():
  print(element)
# >>> 출력:
# (array([[1, 3],
#        [2, 3]], dtype=int32), array([[b'A'],
#        [b'A']], dtype=object))
# (array([[2, 1],
#        [1, 2]], dtype=int32), array([[b'B'],
#        [b'B']], dtype=object))
# (array([[3, 3],
#        [3, 2]], dtype=int32), array([[b'A'],
#        [b'B']], dtype=object))

# ## take() : 초반 몇개 추출
# ```python
# take(
#     count, name=None
# )
# ```
# - Dataset 의 초반 count  개수만큼 추출한 Dataset 리턴
# - https://www.tensorflow.org/api_docs/python/tf/data/Dataset#take

ds = Dataset.range(10)

ds = ds.take(3)
list(ds.as_numpy_iterator())
# >>> 출력:
# [np.int64(0), np.int64(1), np.int64(2)]

# ## batch() : batch묶음
# - Dataset 에서 batch_size 만큼씩 batch 로 묶음
# - https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch    
# ```python
# batch(
#     batch_size,
#     drop_remainder=False,   # batch_size 만큼 묶이지 않는 데이터를 버릴지 여부
#     num_parallel_calls=None,
#     deterministic=None,
#     name=None
# )
# ```

ds = Dataset.range(8)

ds = ds.batch(batch_size=3)
list(ds.as_numpy_iterator())
# >>> 출력:
# [array([0, 1, 2]), array([3, 4, 5]), array([6, 7])]

ds = Dataset.range(8)

ds = ds.batch(3, drop_remainder=True)
list(ds.as_numpy_iterator())
# >>> 출력:
# [array([0, 1, 2]), array([3, 4, 5])]

# ## skip() : 일정요소 skip한 결과
# 
# - Dataset 에서 count 만큼의 element 를 skip 한 Dataset 리턴
# - https://www.tensorflow.org/api_docs/python/tf/data/Dataset#skip
# 
# ```python
# skip(
#     count, name=None
# )
# ```

ds = Dataset.range(10)
list(ds.as_numpy_iterator())
# >>> 출력:
# [np.int64(0),
#  np.int64(1),
#  np.int64(2),
#  np.int64(3),
#  np.int64(4),
#  np.int64(5),
#  np.int64(6),
#  np.int64(7),
#  np.int64(8),
#  np.int64(9)]

ds = Dataset.range(10)

ds = ds.skip(7)
list(ds.as_numpy_iterator())
# >>> 출력:
# [np.int64(7), np.int64(8), np.int64(9)]

# 예측해보자
ds = Dataset.range(10)

list(ds.skip(4).take(3).as_numpy_iterator())
# >>> 출력:
# [np.int64(4), np.int64(5), np.int64(6)]

# dataset 은 내부적으로 iterating 한 정보를 갖고 있다.
# 현재 위치정보가 있다는 뜻이다!

# ## window() : window dataset
# ```python
# window(
#     size, shift=None, stride=1, drop_remainder=False, name=None
# )
# ```
# 
# - https://www.tensorflow.org/api_docs/python/tf/data/Dataset#window
# - Returns a dataset of "windows".

# 잠깐 동작을 보자면
ds = Dataset.range(7).window(3)

for window in ds:
  print(window)
# >>> 출력:
# <_VariantDataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)>
# <_VariantDataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)>
# <_VariantDataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)>

ds = Dataset.range(7).window(3)

for window in ds:
  print([item.numpy() for item in window])
# >>> 출력:
# [np.int64(0), np.int64(1), np.int64(2)]
# [np.int64(3), np.int64(4), np.int64(5)]
# [np.int64(6)]

# shift=
# drop_remainder=
ds = Dataset.range(7).window(3, shift=1, drop_remainder=True)

for window in ds:
  print([item.numpy() for item in window])
# >>> 출력:
# [np.int64(0), np.int64(1), np.int64(2)]
# [np.int64(1), np.int64(2), np.int64(3)]
# [np.int64(2), np.int64(3), np.int64(4)]
# [np.int64(3), np.int64(4), np.int64(5)]
# [np.int64(4), np.int64(5), np.int64(6)]

# stride=
ds = Dataset.range(7).window(3, shift=1, stride=2, drop_remainder=True)

for window in ds:
  print([item.numpy() for item in window])
# >>> 출력:
# [np.int64(0), np.int64(2), np.int64(4)]
# [np.int64(1), np.int64(3), np.int64(5)]
# [np.int64(2), np.int64(4), np.int64(6)]

# ## map(): map_func 적용 결과
# - dataset 에 map_func 적용
# - https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map

ds = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]

ds = ds.map(lambda x: x + 1)
list(ds.as_numpy_iterator())
# >>> 출력:
# [np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6)]

# ## flat_map() : map_func 적용+flatten
# 
# - dataset 에 map_func 적용하고 flatten 한 결과 리턴
# - https://www.tensorflow.org/api_docs/python/tf/data/Dataset#flat_map
# 
# ```python
# flat_map(
#     map_func, name=None
# )
# ```

ds = Dataset.from_tensor_slices(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

ds = ds.flat_map(Dataset.from_tensor_slices)
list(ds.as_numpy_iterator())
# >>> 출력:
# [np.int32(1),
#  np.int32(2),
#  np.int32(3),
#  np.int32(4),
#  np.int32(5),
#  np.int32(6),
#  np.int32(7),
#  np.int32(8),
#  np.int32(9)]

# ## repeat() : 반복
# - https://www.tensorflow.org/api_docs/python/tf/data/Dataset#repeat
# ```python
# repeat(
#     count=None,
#     name=None
# )
# ```

ds = Dataset.from_tensor_slices([1, 2, 3])

ds = ds.repeat(3)
list(ds.as_numpy_iterator())
# >>> 출력:
# [np.int32(1),
#  np.int32(2),
#  np.int32(3),
#  np.int32(1),
#  np.int32(2),
#  np.int32(3),
#  np.int32(1),
#  np.int32(2),
#  np.int32(3)]

# ## shuffle()
# 
# - https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
# ```python
# shuffle(
#     buffer_size,
#     seed=None,
#     reshuffle_each_iteration=None, # bool, (디폴트 True)
#     name=None
# )
# ```
# 
# - buffer_size : 전체 shuffling 이 부담스러운 경우. buffer_size 만큼 샘플링 하여 랜덤 수행하고, 그리고 다음 buffser_size 만큼 샘플링 + 랜덤..  식으로 진행

ds = Dataset.range(3)

ds = ds.shuffle(3, reshuffle_each_iteration=True)  # 매 iteration 마다 shuffle 재수행
ds = ds.repeat(2)
list(ds.as_numpy_iterator())
# >>> 출력:
# [np.int64(0), np.int64(2), np.int64(1), np.int64(2), np.int64(0), np.int64(1)]

ds = Dataset.range(3)

ds = ds.shuffle(3, reshuffle_each_iteration=False)
ds = ds.repeat(2)
list(ds.as_numpy_iterator())
# >>> 출력:
# [np.int64(2), np.int64(0), np.int64(1), np.int64(2), np.int64(0), np.int64(1)]

# # ■ CNN + Dataset 활용

# 1. 데이터 로딩
(train_images, train_labels), (test_images, test_labels) = \
  tf.keras.datasets.fashion_mnist.load_data()
# >>> 출력:
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
# [1m29515/29515[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
# [1m26421880/26421880[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
# [1m5148/5148[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1us/step
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
# [1m4422102/4422102[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step

# 2. Dataset 객체 생성 (전처리 포함)
# ※ tf.data.AUTOTUNE는 차후 설명

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0 # 정규화
    image = tf.expand_dims(image, axis=-1) # 채널 차원 추가 (28x28 → 28x28x1)
    return image, label

batch_size = 64

# ↓ train_ds 에 input 과 target 둘다 포함되어 있다.
train_ds = Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# train/validation 분리
validation_split = 0.2
num_samples = len(train_images)
val_size = int(num_samples * validation_split)

# 현재 Dataset 은 최초 로딩 당시 '순차적' 이기 때문에 shuffle 한뒤 분리해야 한다.
train_ds = train_ds.shuffle(buffer_size=num_samples, seed=42)

val_ds = train_ds.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# test_ds
test_ds = Dataset.from_tensor_slices((test_images, test_labels))
test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ## tf.data.AUTOTUNE ?

# tf.data.AUTOTUNE 란?

# AUTOTUNE은 TensorFlow가 시스템 리소스(CPU, 메모리 등)를 기반으로
# '최적의 병렬 처리 수'를 자동으로 결정하도록 도와주는 자동 튜닝 상수입니다.

# ex()
# map(..., num_parallel_calls=tf.data.AUTOTUNE)
#   map() 함수에서 preprocess 같은 작업을 백그라운드에서 여러 쓰레드로 병렬 처리하는데,
#   몇 개의 쓰레드를 쓸지는 TensorFlow가 자동으로 결정합니다.

# prefetch(tf.data.AUTOTUNE)
#   모델이 훈련 중일 때, 다음 배치를 미리 로드해두는 작업도 병렬로 수행되며,
#   몇 개를 prefetch할지 역시 자동 결정.


# CPU가 놀고 있는 동안 데이터를 미리 처리해서
# GPU가 데이터 기다리지 않고 바로 학습에 집중할 수 있게 돼요.
# 시스템마다 최적 병렬 수가 다르기 때문에 AUTOTUNE을 쓰면
# 코드가 이식성도 좋아지고 성능도 최적화된다.

# 3. 모델 정의
set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 학습
# fit(Dataset, alidation_data=Dataset)
model.fit(train_ds, epochs=5, validation_data=val_ds)

# 6. 평가
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ 테스트 정확도: {test_acc:.4f}")
# >>> 출력:
# Epoch 1/5
# [1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m15s[0m 12ms/step - accuracy: 0.7410 - loss: 0.7386 - val_accuracy: 0.8783 - val_loss: 0.3433
# Epoch 2/5
# [1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 12ms/step - accuracy: 0.8729 - loss: 0.3490 - val_accuracy: 0.8910 - val_loss: 0.3053
# Epoch 3/5
# [1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m17s[0m 8ms/step - accuracy: 0.8898 - loss: 0.2973 - val_accuracy: 0.9057 - val_loss: 0.2621
# Epoch 4/5
# [1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 9ms/step - accuracy: 0.9066 - loss: 0.2575 - val_accuracy: 0.9078 - val_loss: 0.2518
# Epoch 5/5
# [1m750/750[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 9ms/step - accuracy: 0.9142 - loss: 0.2355 - val_accuracy: 0.9222 - val_loss: 0.2125
# [1m157/157[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9031 - loss: 0.2788
# 
# ✅ 테스트 정확도: 0.9027

# # `tf.data.Dataset` 사용하는 이점
#  **사용 편의성과 성능 최적화 측면에서 차이**
# 
# ---
# 
# ### ✅ `tf.data.Dataset`을 사용하는 방식의 **장점**
# 
# | 항목 | 설명 |
# |------|------|
# | 🔄 **메모리 효율** | 데이터를 **메모리에 한꺼번에 로드하지 않고**, 필요한 만큼만 처리 가능 (특히 대규모 데이터셋에 유리) |
# | ⚙️ **성능 최적화** | `prefetch`, `shuffle`, `cache`, `map` 등 파이프라인 구성으로 **GPU/CPU 리소스를 최대한 활용** |
# | 📦 **입력 파이프라인 통합** | 전처리, augmentation, 배치, 병렬 처리 등을 모두 **파이프라인 내에서 처리** 가능 — 코드 구조 일관성 증가 |
# | 🔧 **유연성** | `take`, `skip`, `interleave`, `repeat`, `filter` 등 다양한 연산을 통해 복잡한 데이터 흐름을 직접 설계 가능 |
# | 🌍 **대용량 TFRecord, streaming 데이터** 지원 | `train_test_split`은 NumPy에 한정되어 있지만, `tf.data`는 다양한 원천 지원 (TFRecord, CSV, 이미지 폴더 등) |
# | 🎯 **훈련 중 최적 성능** | `AUTOTUNE`과 함께 사용 시 **훈련 속도 대폭 향상** — GPU가 데이터를 기다리지 않도록 만듦 |
# 
# ---
# 
# ### ❌ `tf.data.Dataset` 방식의 단점 (vs `train_test_split`)
# 
# | 단점 | 설명 |
# |------|------|
# | 🧠 **학습 곡선** | 처음 접하면 `map`, `prefetch`, `AUTOTUNE`, `batch`, `repeat` 등이 복잡해 보일 수 있음 |
# | 🧪 **간단한 실험에는 과함** | 작은 데이터셋/단순 실험에는 오히려 `train_test_split` + `model.fit()`이 더 간편하고 직관적 |
# | 🧮 **NumPy-style indexing 불가** | Dataset은 직접 인덱싱이 어려워 샘플 단위 디버깅이 조금 불편할 수 있음 |
# 
# ---
# 
# ### ✅ `train_test_split` 방식의 장점
# 
# | 항목 | 설명 |
# |------|------|
# | 🔥 빠른 실험 | NumPy 기반으로 손쉽게 split → 빠른 프로토타이핑 가능 |
# | 👶 직관적 | 익숙한 방식 (특히 Scikit-learn 사용자에게) |
# | 🧪 분석 편함 | 배열 기반이므로 바로 시각화/분석에 활용 가능 |
# 
# ---
# 
# ### 🎯 요약
# 
# | 상황 | 추천 방식 |
# |------|-----------|
# | 소규모 실험, 시각화, 빠른 prototyping | ✅ `train_test_split` |
# | 대규모 데이터, 실제 학습 환경 최적화 | ✅ `tf.data.Dataset` |
# 
# ---
# 
# 원하는 용도나 현재 작업 중인 프로젝트 성격에 따라 어떤 방식을 쓸지 달라지겠죠! 혹시 지금 진행 중인 프로젝트나 작업 방향 알려주시면, 어떤 방식이 더 잘 맞을지 구체적으로 도와드릴게요 😄
