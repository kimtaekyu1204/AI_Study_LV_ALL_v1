# # PyTorch Basic

import numpy as np
import torch

torch.__version__
# >>> 출력:
# '2.6.0+cu124'

# # torch.Tensor
# 
# - torch 의 다차원 배열 객체
# 
# https://pytorch.org/docs/stable/torch.html#tensors

np.arange(9)
# >>> 출력:
# array([0, 1, 2, 3, 4, 5, 6, 7, 8])

tensor = torch.arange(9)
tensor
# >>> 출력:
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])

type(tensor)
# >>> 출력:
# torch.Tensor

tensor.shape
# >>> 출력:
# torch.Size([9])

tensor.numpy()
# >>> 출력:
# array([0, 1, 2, 3, 4, 5, 6, 7, 8])

tensor.reshape(3, 3)
# >>> 출력:
# tensor([[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]])

randoms = torch.rand((3, 3))
randoms
# >>> 출력:
# tensor([[0.6267, 0.0305, 0.1989],
#         [0.2702, 0.7547, 0.8177],
#         [0.9071, 0.8266, 0.6088]])

randoms.dtype
# >>> 출력:
# torch.float32

randoms.size()  # numpy 와는 다름!
# >>> 출력:
# torch.Size([3, 3])

torch.zeros((3, 3))
# >>> 출력:
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])

torch.zeros_like(randoms) # randoms 와 shape 이 동일한 0 으로 구성된 tensor
# >>> 출력:
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])

torch.ones((3, 3))
# >>> 출력:
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])

# # Operations

tensor
# >>> 출력:
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])

tensor * 3
# >>> 출력:
# tensor([ 0,  3,  6,  9, 12, 15, 18, 21, 24])

tensor = tensor.reshape(3, 3)
tensor
# >>> 출력:
# tensor([[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]])

tensor + tensor
# >>> 출력:
# tensor([[ 0,  2,  4],
#         [ 6,  8, 10],
#         [12, 14, 16]])

tensor + 10
# >>> 출력:
# tensor([[10, 11, 12],
#         [13, 14, 15],
#         [16, 17, 18]])

torch.add(tensor, 10)
# >>> 출력:
# tensor([[10, 11, 12],
#         [13, 14, 15],
#         [16, 17, 18]])

# # Tensor Views
# 
# https://pytorch.org/docs/stable/tensor_view.html#tensor-views
# 
# - 기존 tensor 의 데이터는 공유하지만, shape 이 다른 view 제공
# - view() 는 reshape() 거의 같다..
# - view() 는 사실, 훨씬 오래전부터 있어왔다.   (사실 reshape() 를 더 추천한다.)
# - ※ 단, 공식 예제에선 View 를 사용함.

range_nums = torch.arange(9).reshape(3, 3)
range_nums
# >>> 출력:
# tensor([[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]])

range_nums.view(1, 9)
# >>> 출력:
# tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]])

range_nums.reshape(1, 9)
# >>> 출력:
# tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]])

range_nums.view(-1)
# >>> 출력:
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])

# # slice & index

nums = torch.arange(9).reshape(3, 3)
nums
# >>> 출력:
# tensor([[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]])

nums[1]
# >>> 출력:
# tensor([3, 4, 5])

nums[1, 1]
# >>> 출력:
# tensor(4)

nums[1:]
# >>> 출력:
# tensor([[3, 4, 5],
#         [6, 7, 8]])

nums[1:, 1:]
# >>> 출력:
# tensor([[4, 5],
#         [7, 8]])

# # Compile

arr = np.array([1, 1, 1])
arr
# >>> 출력:
# array([1, 1, 1])

arr_torch = torch.from_numpy(arr)
arr_torch
# >>> 출력:
# tensor([1, 1, 1])

arr_torch.dtype  # int64
# >>> 출력:
# torch.int64

arr_torch.float()
# >>> 출력:
# tensor([1., 1., 1.])

# GPU detect 하기

torch.cuda.is_available()
# >>> 출력:
# True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
# >>> 출력:
# 'cuda'

# Tensor 를 특정 device 에 compile
# to(device)

arr_torch.to(device)
# >>> 출력:
# tensor([1, 1, 1], device='cuda:0')

arr_torch.device
# >>> 출력:
# device(type='cpu')

# ### PyTorch 와 Tensorflow 에서 compile 차이점(요약)
# 
# | **특징**              | **PyTorch `torch.compile`**                                   | **TensorFlow `model.compile`**                      |
# |-----------------------|-------------------------------------------------------------|---------------------------------------------------|
# | **목적**              | 실행 속도 최적화                                              | 학습 설정 (손실 함수, 옵티마이저 등 지정)             |
# | **언제 사용?**        | 학습 및 추론 속도를 개선하려고 할 때                              | 모델을 학습하기 전에 필요한 설정을 정의할 때          |
# | **비유**              | 효율적으로 요리하는 셰프가 되도록 준비                          | 어떤 재료와 조리법으로 요리할지 결정하는 단계         |
# | **주요 입력**         | 모델 객체                                                   | 옵티마이저, 손실 함수, 평가지표 등                   |
# | **결과**              | 더 빠른 실행 가능                                              | 학습 가능한 상태의 모델 생성                        |
# 
# ---
# 
# ### 요약
# - **PyTorch `torch.compile`**: 속도 최적화를 위한 기술.
# - **TensorFlow `model.compile`**: 학습을 시작하기 전에 필요한 설정.
# 
# 둘 다 이름은 비슷하지만 역할은 완전히 다릅니다!

# # AutoGrad
# '기울기' 를 주어 학습이 되게 하는 것.

x = torch.ones(2, 2, requires_grad=True)
x
# >>> 출력:
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)

x.grad  # 학습기울기,  처음에는 없다 None

y = x + 2
y
# >>> 출력:
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)

y.grad_fn
# >>> 출력:
# <AddBackward0 at 0x7a1b2d10ec20>

z = y * y * 3
z
# >>> 출력:
# tensor([[27., 27.],
#         [27., 27.]], grad_fn=<MulBackward0>)

out = z.mean()
out
# >>> 출력:
# tensor(27., grad_fn=<MeanBackward0>)

out.backward()  # back-propagation(역전파) 수행! -> 하면 x 의 기울기를 구할수 있다.

x.grad   # x 의 학습 기울기가 계산되어 있다!!!
# >>> 출력:
# tensor([[4.5000, 4.5000],
#         [4.5000, 4.5000]])

x.requires_grad
# >>> 출력:
# True

(x ** 2).requires_grad
# >>> 출력:
# True

# Tensorflow : Define and Run 방식
# PyTorch : Define By Run 방식

# PyTorch 에선 train mode 와 test mode 가 있다.
# '학습모드(train mode)' 에서는 기울기 구하는게 가능하지만
# '테스트모드(test mode)' 에서는 이기능을 꺼야 겠죠
#    -> no_grad() 사용

with torch.no_grad():
  print((x ** 2).requires_grad)  # no_grad() 안에서는 False, 기울기를 구하지 않게 된다.
  #
# >>> 출력:
# False

# no_grad() 이면  기울기를 구하지 않게 됩니다.
# 따라서 batch, normalization dropout 들이 작동 안함
# 작동 속도는 train mode 일때보다 test mode 가 더 빠르겠죠.

# # PyTorch Data Preprocess

import matplotlib.pyplot as plt

from torchvision import datasets, transforms

# ## DataLoader()
# PyTorch 는 DataLoader() 사용하여 model 에 데이터 입력

batch_size = 32
test_batch_size = 32

save_dir = r'.'

# train loader
train_loader = torch.utils.data.DataLoader(
    dataset = datasets.FashionMNIST(
        save_dir,  # 저장할 디렉토리.   {save_dir}/FashionMNIST 에 다운로드
        train=True, # 학습용
        download=True,  # 없으면 다운로드.
        transform=transforms.Compose([  # 데이터 로딩시 필요현 변환(transform) (들)을 나열.
            transforms.ToTensor(),   # 데이터 다운 받은뒤 Tensor 로 변환.
            transforms.Normalize(mean=(0.5,), std=(0.5,)) # 평균값 0.5, std 0.5 로 스케일링 변환
        ]),
    ),
    batch_size = batch_size,  # 배치 사이즈
    shuffle=True,
)
# >>> 출력:
# 100%|██████████| 26.4M/26.4M [00:02<00:00, 11.0MB/s]
# 100%|██████████| 29.5k/29.5k [00:00<00:00, 176kB/s]
# 100%|██████████| 4.42M/4.42M [00:01<00:00, 3.28MB/s]
# 100%|██████████| 5.15k/5.15k [00:00<00:00, 9.76MB/s]

# test loader
test_loader = torch.utils.data.DataLoader(
    dataset = datasets.FashionMNIST(
        save_dir,  # 저장할 디렉토리.   {save_dir}/FashionMNIST 에 다운로드
        train=False, # 학습모드로 사용하는 데이터가 아니다!
        transform=transforms.Compose([  # 데이터 로딩시 필요현 변환(transform) (들)을 나열.
            transforms.ToTensor(),   # 데이터 다운 받은뒤 Tensor 로 변환.
            transforms.Normalize(mean=(0.5,), std=(0.5,)) # 평균값 0.5, std 0.5 로 스케일링 변환
        ]),
    ),
    batch_size = batch_size,  # 배치 사이즈
    shuffle=True,
)

# ## 데이터 확인

type(train_loader)
# DataLoader 객체 <-- iterable 하다
# batch 단위로 데이터 iteration.
# >>> 출력:
# torch.utils.data.dataloader.DataLoader

images, labels = next(iter(train_loader))  # 첫번째 batch

images.shape, labels.shape
# >>> 출력:
# (torch.Size([32, 1, 28, 28]), torch.Size([32]))

# TF (batch, height, width, channel)

# PyTorch( batch, channel, height, width)

# ↑ 뭔가 다르죠?
# 
# 
# TensorFlow 에선 [32, 28, 28. 1]   즉 [batch size, height, width, channel] 이었습니다
# 
# 
# PyTorch는 TensorFlow와 다르게 [Batch Size, Channel, Height, Width] 임을 명심해야함
# 
# 만약 gray 가 아니라 rgb 였으면 [32, 3, 28, 28] 이 되는 겁니다

# 첫번째 이미지
images[0].shape
# >>> 출력:
# torch.Size([1, 28, 28])

# 시각화를 위해서 axis 0 를 없애기
torch.squeeze(images[0]).shape
# >>> 출력:
# torch.Size([28, 28])

torch_image = torch.squeeze(images[0])
torch_image.shape
# >>> 출력:
# torch.Size([28, 28])

image = torch_image.numpy()
image.shape
# >>> 출력:
# (28, 28)

# 첫번째 label
label = labels[0].numpy()
label  # Dataloader 에서 shuffle 했기 때문에 random 인 상태.
# >>> 출력:
# array(1)

datasets.FashionMNIST.classes  # 레이블 명 확인
# >>> 출력:
# ['T-shirt/top',
#  'Trouser',
#  'Pullover',
#  'Dress',
#  'Coat',
#  'Sandal',
#  'Shirt',
#  'Sneaker',
#  'Bag',
#  'Ankle boot']

np.min(image), np.max(image)
# >>> 출력:
# (np.float32(-1.0), np.float32(1.0))

plt.title(label)
plt.imshow(image, cmap='gray_r')
plt.show()

# # PyTorch 의 Layer

# train loader
train_loader = torch.utils.data.DataLoader(
    dataset = datasets.FashionMNIST(
        save_dir,  # 저장할 디렉토리.   {save_dir}/FashionMNIST 에 다운로드
        train=True, # 학습용
        download=True,  # 없으면 다운로드.
        transform=transforms.Compose([  # 데이터 로딩시 필요현 변환(transform) (들)을 나열.
            transforms.ToTensor(),   # 데이터 다운 받은뒤 Tensor 로 변환.
            # 현재는 모델학습이 목적이 아니라서 스케일링 생략.
        ]),
    ),
    batch_size = 1,  # 이번 예제 목적은 이미지 '하나하나' 를 레이어에 넣어 '확인'
    # shuffle=True,
)

# 첫번째 배치
image, label = next(iter(train_loader))

image.shape
# >>> 출력:
# torch.Size([1, 1, 28, 28])

plt.imshow(image[0, 0, :, :], 'gray')
plt.title(label[0])
plt.show()

# PyTorch 에선 레이어를 쌓기 위해 다음과 같은 레이어 필요
import torch.nn as nn   # nn 레이어와
import torch.nn.functional as F  # F 레이어

# nn : 파라미터 있는 레이어
# F : 파라미터 없는 연산을 위한 레이어

# ## nn.Conv2D()
# - in_channels: 받게 될 channel의 갯수
# - out_channels: 보내고 싶은 channel의 갯수  
# - kernel_size: 만들고 싶은 kernel(weights)의 사이즈
# 
# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d

nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
# >>> 출력:
# Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))

# ↑ TensorFlow 에서 봤을때와 좀 다르죠.

#  TensorFlow 에선 out channel 만 나왔습니다.  (얼마나 내보낼 것인지, filters=값)
#  PyTorch 는 앞에서 몇개를 받을지도 지정해주어야 하는 것입니다. (in_channels)
#  위 이미지의 채널은 1 이었습니다'  >> torch.Size([1, 1, 28, 28])
#  그리고 이것을 20개로 내보냅니다 (out_channels)

#  kernel_size = 5   TensorFlow 때는 3x3 이었는데
# PyTorch 공식예제 에선 5x5 로 되어 있어서 5로 주었습니다.

# stride = 1

layer_conv1 = nn.Conv2d(1, 20, 5, 1).to(torch.device(device))
layer_conv1
# >>> 출력:
# Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))

# weight 꺼내보기
weight = layer_conv1.weight
weight.shape
# >>> 출력:
# torch.Size([20, 1, 5, 5])

# torch.Size([20, 1, 5, 5])
#  20개 filters x channel 1개 x height 5 x width 5

weight
# >>> 출력:
# Parameter containing:
# tensor([[[[-0.1561,  0.1409, -0.1376, -0.0040,  0.1922],
#           [-0.0417, -0.1668, -0.1897,  0.0496,  0.0649],
#           [ 0.1960, -0.1077, -0.0262,  0.1291,  0.1690],
#           [-0.1313,  0.0737,  0.0343, -0.0584, -0.0935],
#           [-0.1760, -0.1891, -0.0174,  0.0921, -0.1370]]],
# 
# 
#         [[[ 0.0969, -0.0854, -0.1651, -0.1116, -0.0750],
#           [ 0.0634,  0.1248, -0.1473, -0.1462,  0.1355],
#           [-0.1290, -0.0123, -0.1680, -0.1795,  0.0497],
#           [ 0.0590, -0.0003,  0.1031, -0.0518,  0.1540],
#           [ 0.1275, -0.1382,  0.1517, -0.0875,  0.0050]]],
# 
# 

# weight.numpy()  # 에러

# - 여기서 weight는 '학습 가능한 상태'이기 때문에 바로 numpy로 뽑아낼 수 없음
#     - '학습 가능?' 유연한, 말랑말랑한 tensor..
# - detach() method는 그래프에서 잠깐 빼서 gradient에 영향을 받지 않게 함

weight = weight.detach().cpu().numpy()
weight
# >>> 출력:
# array([[[[-0.15612161,  0.14093299, -0.13756226, -0.00399542,
#            0.19217455],
#          [-0.04169893, -0.16681533, -0.18965359,  0.04964447,
#            0.06486001],
#          [ 0.19600634, -0.1077437 , -0.02619166,  0.12913917,
#            0.16895664],
#          [-0.13130474,  0.07373822,  0.03433037, -0.05844705,
#           -0.09345265],
#          [-0.17596996, -0.18913977, -0.01738875,  0.09213748,
#           -0.13703366]]],
# 
# 
#        [[[ 0.09687934, -0.0854281 , -0.16507328, -0.11161575,
#           -0.07502554],
#          [ 0.06338181,  0.12481642, -0.14731972, -0.14624174,

# 20개의 filter 중 첫번째 시각화
plt.imshow(weight[0, 0, :, :], 'jet')
plt.colorbar()
plt.show()

# output 시각화

# image 를 layer 에 넣어본다
# GPU 모델을 만든 경우 입력데이타도 GPU 데이터 이어야 합니다
output_data = layer_conv1(image if device == 'cpu' else image.cuda())

output_data.shape
# >>> 출력:
# torch.Size([1, 20, 24, 24])

output_data
# >>> 출력:
# tensor([[[[ 0.0569,  0.0569,  0.0569,  ...,  0.0570,  0.0531,  0.0501],
#           [ 0.0569,  0.0569,  0.0569,  ...,  0.0509,  0.0579,  0.0586],
#           [ 0.0569,  0.0569,  0.0569,  ..., -0.1123, -0.0082, -0.0056],
#           ...,
#           [ 0.2422, -0.0594, -0.0923,  ...,  0.1393,  0.1588, -0.0958],
#           [ 0.2190,  0.1920,  0.0980,  ...,  0.0140, -0.0386, -0.1793],
#           [ 0.1902,  0.2002,  0.1769,  ..., -0.1275, -0.1292, -0.2103]],
# 
#          [[ 0.1501,  0.1501,  0.1501,  ...,  0.1506,  0.1517,  0.1497],
#           [ 0.1501,  0.1501,  0.1501,  ...,  0.1509,  0.1475,  0.1549],
#           [ 0.1501,  0.1501,  0.1501,  ...,  0.1627,  0.1725,  0.1733],
#           ...,
#           [-0.3483, -0.2294, -0.2457,  ..., -0.2510, -0.3575, -0.4623],
#           [-0.1963, -0.2311, -0.2451,  ..., -0.2968, -0.2836, -0.2882],
#           [-0.0073, -0.0776, -0.1923,  ..., -0.1127, -0.0793,  0.0019]],

output_data.data  # .data 만 추출.
# >>> 출력:
# tensor([[[[ 0.0569,  0.0569,  0.0569,  ...,  0.0570,  0.0531,  0.0501],
#           [ 0.0569,  0.0569,  0.0569,  ...,  0.0509,  0.0579,  0.0586],
#           [ 0.0569,  0.0569,  0.0569,  ..., -0.1123, -0.0082, -0.0056],
#           ...,
#           [ 0.2422, -0.0594, -0.0923,  ...,  0.1393,  0.1588, -0.0958],
#           [ 0.2190,  0.1920,  0.0980,  ...,  0.0140, -0.0386, -0.1793],
#           [ 0.1902,  0.2002,  0.1769,  ..., -0.1275, -0.1292, -0.2103]],
# 
#          [[ 0.1501,  0.1501,  0.1501,  ...,  0.1506,  0.1517,  0.1497],
#           [ 0.1501,  0.1501,  0.1501,  ...,  0.1509,  0.1475,  0.1549],
#           [ 0.1501,  0.1501,  0.1501,  ...,  0.1627,  0.1725,  0.1733],
#           ...,
#           [-0.3483, -0.2294, -0.2457,  ..., -0.2510, -0.3575, -0.4623],
#           [-0.1963, -0.2311, -0.2451,  ..., -0.2968, -0.2836, -0.2882],
#           [-0.0073, -0.0776, -0.1923,  ..., -0.1127, -0.0793,  0.0019]],

output = output_data.data.cpu().numpy()
output
# >>> 출력:
# array([[[[ 0.05688229,  0.05688229,  0.05688229, ...,  0.0570148 ,
#            0.05309859,  0.05013842],
#          [ 0.05688229,  0.05688229,  0.05688229, ...,  0.05087246,
#            0.05790441,  0.05859342],
#          [ 0.05688229,  0.05688229,  0.05688229, ..., -0.11226177,
#           -0.00818891, -0.00563464],
#          ...,
#          [ 0.242227  , -0.05939331, -0.09225681, ...,  0.13928102,
#            0.15878586, -0.09582344],
#          [ 0.21899104,  0.1919928 ,  0.09802069, ...,  0.0139825 ,
#           -0.03859902, -0.17930235],
#          [ 0.19022985,  0.20016317,  0.17687538, ..., -0.12749195,
#           -0.12922938, -0.210338  ]],
# 
#         [[ 0.15008421,  0.15008421,  0.15008421, ...,  0.15055844,

# 시각화를 위해 image 에서 numpy 배열 추출
image_arr = image.cpu().numpy()
image_arr.shape
# >>> 출력:
# (1, 1, 28, 28)

plt.figure(figsize=(15, 30))

plt.subplot(131)
plt.title('input')
plt.imshow(np.squeeze(image_arr), 'gray_r')

plt.subplot(132)
plt.title('weight')
plt.imshow(weight[0, 0, :, :], 'jet')

plt.subplot(133)
plt.title('output')
plt.imshow(output[0, 0, :, :], 'gray_r')


plt.show()

# ## F.max_pool2d()

pool = F.max_pool2d(image, kernel_size=2, stride=2)
pool.shape #  (1, 1, 14, 14) <- (1, 1, 28, 28)
# >>> 출력:
# torch.Size([1, 1, 14, 14])

pool_arr = pool.numpy()
pool_arr
# >>> 출력:
# array([[[[0.        , 0.        , 0.        , 0.        , 0.        ,
#           0.        , 0.        , 0.        , 0.        , 0.        ,
#           0.        , 0.        , 0.        , 0.        ],
#          [0.        , 0.        , 0.        , 0.        , 0.        ,
#           0.        , 0.00392157, 0.05098039, 0.28627452, 0.00392157,
#           0.01568628, 0.        , 0.00392157, 0.00392157],
#          [0.        , 0.        , 0.        , 0.        , 0.        ,
#           0.        , 0.02352941, 0.8       , 0.6901961 , 0.5647059 ,
#           0.09019608, 0.01176471, 0.04705882, 0.03921569],
#          [0.        , 0.        , 0.        , 0.        , 0.        ,
#           0.00392157, 0.27058825, 0.9254902 , 0.85490197, 0.84705883,
#           0.6313726 , 0.57254905, 0.5529412 , 0.6745098 ],
#          [0.        , 0.        , 0.        , 0.        , 0.00392157,
#           0.00392157, 0.78431374, 0.9098039 , 0.9137255 , 0.92156863,
#           0.8784314 , 0.8784314 , 0.9607843 , 0.8980392 ],

plt.figure(figsize=(10, 15))

plt.subplot(121)
plt.title('input')
plt.imshow(np.squeeze(image_arr), 'gray_r')

plt.subplot(122)
plt.title('output')
plt.imshow(np.squeeze(pool_arr), 'gray_r')


plt.show()

# ## nn.Linear()
# - nn.Linear는 2d가 아닌 1d만 들어가기 때문에 .view() 1D로 펼쳐줘야함
# - TensorFlow 처럼 Flatten 이라는 레이어가 따로 있는게 아니라서  
#     - 우선, reshape 나 view 를 사용해서 펼쳐 준뒤, Linear() 에 넣어야 합니다

image.shape
# >>> 출력:
# torch.Size([1, 1, 28, 28])

flatten = image.view(-1, 28 * 28)
flatten.shape
# >>> 출력:
# torch.Size([1, 784])

lin = nn.Linear(in_features=784, out_features=10)(flatten)
lin.shape
# >>> 출력:
# torch.Size([1, 10])

lin
# >>> 출력:
# tensor([[-0.3003,  0.0697, -0.2856, -0.1324, -0.1366,  0.5068,  0.2015,  0.1441,
#           0.0905, -0.2063]], grad_fn=<AddmmBackward0>)

plt.imshow(lin.detach().numpy(), 'jet')
plt.show()

# ## F.softmax()

with torch.no_grad():
  flatten = image.view(1, 28 * 28)
  lin = nn.Linear(784, 10)(flatten)  # (1, 10)
  softmax = F.softmax(lin, dim=1)

softmax
# >>> 출력:
# tensor([[0.1171, 0.0917, 0.0931, 0.0984, 0.1129, 0.0977, 0.0979, 0.0880, 0.0778,
#          0.1253]])

np.sum(softmax.numpy())
# >>> 출력:
# np.float32(1.0)

# # Layer 쌓기

class Net(nn.Module):  # nn.Module 을 상속받아 모델의 layer 쌓기

  # 생성자 에는 '학습(train)' 이 가능한 것, 즉 weight 가 들어있는 레이어 정의
  # 위 예제코드에선 Conv 와 Linear 가 그런 레이어었댜.
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 20, 5, 1)
    self.conv2 = nn.Conv2d(20, 50, 5, 1)

    self.fc1 = nn.Linear(4 * 4 * 50, 500)
    self.fc2 = nn.Linear(500, 10)  # 최종 출력 10개 클래스

  def forward(self, x):   # x : 입력
    # 1. Feature extraction
    x = F.relu(self.conv1(x))  # Convolution 결과는 활성화 함수 relu 를 거쳐 출력.
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)

    # conv1 입력 (1, 28, 28) => conv1 => (20, 24, 24) => maxpool => (20, 12, 12)
    # conv2 입력 (20, 12, 12) => conv2 => (50, 8, 8) => maxpool => (50, 4, 4)
    # 위 결과를 펼쳐서 Linear 의 입력으로 전달

    # 2. Classification
    x = x.view(-1, 4 * 4 * 50)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)

    # 3. softmax
    return F.softmax(x, dim=1)

model = Net()  # 모델 생성!

result = model.forward(image)

result
# >>> 출력:
# tensor([[0.1039, 0.0966, 0.1035, 0.0963, 0.1018, 0.1021, 0.0955, 0.0988, 0.1007,
#          0.1007]], grad_fn=<SoftmaxBackward0>)

model.conv1
# >>> 출력:
# Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))

model.conv1(image)
# >>> 출력:
# tensor([[[[-0.0473, -0.0473, -0.0473,  ..., -0.0467, -0.0463, -0.0506],
#           [-0.0473, -0.0473, -0.0473,  ..., -0.0555, -0.0575, -0.0447],
#           [-0.0473, -0.0473, -0.0473,  ..., -0.1220, -0.0524, -0.0083],
#           ...,
#           [-0.4148, -0.3663, -0.3370,  ..., -0.3333, -0.3152, -0.1994],
#           [-0.1547, -0.2400, -0.2933,  ..., -0.1318,  0.0007,  0.0461],
#           [ 0.0418, -0.1265, -0.1469,  ...,  0.0120,  0.0426, -0.0575]],
# 
#          [[ 0.1148,  0.1148,  0.1148,  ...,  0.1136,  0.1123,  0.1134],
#           [ 0.1148,  0.1148,  0.1148,  ...,  0.1141,  0.1152,  0.1117],
#           [ 0.1148,  0.1148,  0.1148,  ...,  0.1259,  0.1113,  0.0733],
#           ...,
#           [ 0.1650,  0.0270, -0.1417,  ..., -0.0826, -0.0009,  0.1712],
#           [ 0.1921,  0.2242,  0.2078,  ...,  0.0319,  0.0523,  0.1824],
#           [ 0.0804,  0.1455,  0.2034,  ...,  0.2161,  0.2890,  0.3358]],

# # Optimization

seed = 1   # shuffle 시 동일하게 섞기 => DataLoader

batch_size = 64
test_batch_size = 64

no_cuda = False

use_cuda = not no_cuda and torch.cuda.is_available()  # GPU 사용 여부
use_cuda
# >>> 출력:
# True

device
# >>> 출력:
# 'cuda'

save_dir
# >>> 출력:
# '.'

# ## Preprocess

torch.manual_seed(seed)  # 랜덤 seed

train_loader = torch.utils.data.DataLoader(
    dataset = datasets.FashionMNIST(
        save_dir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ]),
    ),
    batch_size = batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset = datasets.FashionMNIST(
        save_dir,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ]),
    ),
    batch_size = test_batch_size,
    shuffle=True,
)

# ## Optimization
# - Model 과 Optimization 설정

model = Net().to(device)  # model 생성하여 device 에 컴파일 해준다

# 학습 parameter 들 확인 가능
params = list(model.parameters())
params
# >>> 출력:
# [Parameter containing:
#  tensor([[[[ 0.1031, -0.0883, -0.0388,  0.0939, -0.1883],
#            [ 0.1199, -0.0411,  0.1017,  0.0278, -0.0245],
#            [ 0.0555,  0.0099,  0.0730, -0.0779, -0.0146],
#            [-0.0180,  0.0290, -0.0008,  0.1748,  0.0622],
#            [-0.0745, -0.1208, -0.0335, -0.0863, -0.0641]]],
#  
#  
#          [[[ 0.0096,  0.1192,  0.1087, -0.1955,  0.1240],
#            [ 0.0559,  0.1897,  0.1320, -0.1822, -0.1902],
#            [-0.0965,  0.1756, -0.0333,  0.0856, -0.0929],
#            [ 0.1962, -0.0846,  0.1500,  0.0024, -0.1054],
#            [ 0.1028, -0.1062,  0.0588, -0.0578, -0.0219]]],
#  
#  

len(params)  # 8개의 Tensor 객체

# 학습 가능한 레이어 4개 + 4 (각 레이어마다 bias)
# >>> 출력:
# 8

for param in params:
  print(param.size())
# >>> 출력:
# torch.Size([20, 1, 5, 5])
# torch.Size([20])
# torch.Size([50, 20, 5, 5])
# torch.Size([50])
# torch.Size([500, 800])
# torch.Size([500])
# torch.Size([10, 500])
# torch.Size([10])

# torch.Size([20, 1, 5, 5]) <-- conv1 의 weight size
# torch.Size([20])          <-- conv1 의 bias (out channel 의 개수만큼!)
# torch.Size([50, 20, 5, 5])<-- conv2 의 weight size
# torch.Size([50])          <-- conv2 의 bias (out channel 의 개수만큼!)
# torch.Size([500, 800])    <-- fc1 의 weight size (out, in)
# torch.Size([500])         <-- fc1 의 bias  (out channel 의 개수만큼!)
# torch.Size([10, 500])     <-- fc2 의 weight size (out, in)
# torch.Size([10])          <-- fc2 의 bias  (out channel 의 개수만큼!)

# Optimizer 설정
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
optimizer
# >>> 출력:
# SGD (
# Parameter Group 0
#     dampening: 0
#     differentiable: False
#     foreach: None
#     fused: None
#     lr: 0.001
#     maximize: False
#     momentum: 0.5
#     nesterov: False
#     weight_decay: 0
# )

# ## before training
# - Model 이 train 할수 있도록 train mode 변환

model.train()  # 학습하기전에 train mode로 전환
# >>> 출력:
# Net(
#   (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (conv2): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
#   (fc1): Linear(in_features=800, out_features=500, bias=True)
#   (fc2): Linear(in_features=500, out_features=10, bias=True)
# )

# train mode (학습) ↔ evaluation mode (테스트, 예측)
# train mode 했다가 evalutaion mode 했다가   다시 train mode 로 돌아올때에도 .train() 을 해주어야 합니다

# 모델에 입력하기 위한 첫 batch 추출
data, target = next(iter(train_loader))  # 64개의 데이터

data.shape, target.shape
# >>> 출력:
# (torch.Size([64, 1, 28, 28]), torch.Size([64]))

# 추출한 batch 데이터도 device 에 compile
data, target = data.to(device), target.to(device)

# gradients 를 clear
# 학습하기 전에 Optimizer 를 clear해주어야 합니다.
# zero_grad() : gradient 를 clear 해서 새로운 최적값 을 찾기 위한 준비

optimizer.zero_grad()

# 준비한 데이터를 model 에 input 으로 입력하여 output 획득

output = model(data)  # 이건 '예측' 하는 동작 아니다 (현재 train mode!)
                    # 이 output 으로 loss 값 계산해야 한다!

output.shape
# >>> 출력:
# torch.Size([64, 10])

# Loss function
# Negative Log-Likelihood Loss(Nll) 라는 loss function 사용
loss = F.nll_loss(output, target)
loss
# >>> 출력:
# tensor(-0.1017, device='cuda:0', grad_fn=<NllLossBackward0>)

# back propagation 을 통해 gradients 를 계산
loss.backward()  # 기울기 계산!

# 기울기 계산후 Optimizer 에 업데이트 해주어야 한다.
# => paramete update
#    step(): 순방향 -> loss -> 역전파 -> 업데이트

optimizer.step()

# 이상이 '학습' 의 "1 스텝"입니다
# - train 모드 변환
# - 데이터 넣어주고
# - 기울기 clear
# - model 에 데이터 넣고
# - loss 계산하고
# - back propagation 하여 gradient 계산하고
# - parameter 업데이트

# ## Training
# 위의 optimization 과정을 반복하여 학습

# hyper param 설정
epochs = 10

log_interval = 100  # 학습진행 중간중간에 로그를 확인하기 위해 몇 step 마다 로그 출력할지 결정

for epoch in range(1, epochs + 1):
  # 1. train 모드 전환
  model.train()

  # 2. batch 단위로 모델에 입력하고 학습 진행
  for batch_idx, (data, target) in enumerate(train_loader):
    # 3. 데이터를 device 에 compile
    data, target = data.to(device), target.to(device)
    # 4. 기울기 clear
    optimizer.zero_grad()

    #5. model 에 데이터 입력
    output = model(data)

    #6. loss 계산
    loss = F.nll_loss(output, target)

    #7. back propagation 하여 gradient 계산
    loss.backward()

    #8. parameter 업데이트
    optimizer.step()

    # 중간 중간에 로그 확인
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch,
          batch_idx * len(data),
          len(train_loader.dataset),
          100 * batch_idx / len(train_loader),
          loss.item()  # loss 값
      ))
# >>> 출력:
# Train Epoch: 1 [0/60000 (0%)]	Loss: -0.101359
# Train Epoch: 1 [6400/60000 (11%)]	Loss: -0.100655
# Train Epoch: 1 [12800/60000 (21%)]	Loss: -0.102867
# Train Epoch: 1 [19200/60000 (32%)]	Loss: -0.102166
# Train Epoch: 1 [25600/60000 (43%)]	Loss: -0.107604
# Train Epoch: 1 [32000/60000 (53%)]	Loss: -0.104661
# Train Epoch: 1 [38400/60000 (64%)]	Loss: -0.106451
# Train Epoch: 1 [44800/60000 (75%)]	Loss: -0.110184
# Train Epoch: 1 [51200/60000 (85%)]	Loss: -0.111417
# Train Epoch: 1 [57600/60000 (96%)]	Loss: -0.114701
# Train Epoch: 2 [0/60000 (0%)]	Loss: -0.112632
# Train Epoch: 2 [6400/60000 (11%)]	Loss: -0.112813
# Train Epoch: 2 [12800/60000 (21%)]	Loss: -0.112155
# Train Epoch: 2 [19200/60000 (32%)]	Loss: -0.130607
# Train Epoch: 2 [25600/60000 (43%)]	Loss: -0.119766

# # Evaluation
# 
# - 앞에서 model.train() 모드로 변한 것처럼 평가 할 때는 model.eval()로 설정
#     - Batch Normalization이나 Drop Out 같은 Layer들을 잠금

model.eval()  # evaluation 모드 전환
# >>> 출력:
# Net(
#   (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (conv2): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
#   (fc1): Linear(in_features=800, out_features=500, bias=True)
#   (fc2): Linear(in_features=500, out_features=10, bias=True)
# )

# batch 하나를 evaluate
correct = 0
test_loss = 0  # 손실값 계산 (누적)

with torch.no_grad():
    # evaluation 에선  batch normalization 이나 drop out 같은 것들을 잠금
    # no_grad() 는 back propagation 이나 gradient 등의 계산을 꺼서 memory usage 를 줄이고 속도 높임
    # 테스트 모드일때는 이를 꺼주는 것이 좋다!

    # 테스트데이터에서 batch 하나 꺼내기
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    output = model(data)

    # loss 계산.  no_grad라서 weight 업데이트 용이 아닐, 순수하게 손실값 계산용.
    test_loss += F.nll_loss(output, target, reduction='sum').item()

    pred = output.argmax(dim=1, keepdim=True)
            #  제일 강한 인덱스를 구함.  즉 컴이 계산한 정답 예측값 꺼냄.
            # keepdim=True : output 과 pred 의 dimension 유지

    correct = pred.eq(target.view_as(pred)).sum().item()  # pred 와 target 이 얼마나 같은가 판정
        # pred.eq(target.view_as(pred))  <-- 여기는 True / False 값 나옴
        #  .sum() 그것들을 다 더함.
        # .item() 값을 꺼내 담음

test_loss
# >>> 출력:
# -43.739219665527344

correct   # 한개 의 batch (64장의 이미지) 중 맞춘 개수 .
# >>> 출력:
# 44

# evaluation (완성)

model.eval()

test_loss = 0
correct = 0

with torch.no_grad():
  for data, target in test_loader:  # test 의 모든 batch 돌려본다
    data, target = data.to(device), target.to(device)
    output = model(data)
    test_loss += F.nll_loss(output, target, reduction='sum').item()
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)  # loss 의 평균값

print('Test set : Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss,
    correct,
    len(test_loader.dataset),
    100.0 * correct / len(test_loader.dataset)
))
# >>> 출력:
# Test set : Average Loss: -0.5989, Accuracy: 6182/10000 (62%)

# # Training + Evaluation

for epoch in range(1, epochs + 1):
  # Train Mode
  model.train()

  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch,
          batch_idx * len(data),
          len(train_loader.dataset),
          100 * batch_idx / len(train_loader),
          loss.item()  # loss 값
      ))

  # Eval Mode
  model.eval()

  test_loss = 0
  correct = 0

  with torch.no_grad():
    for data, target in test_loader:  # test 의 모든 batch 돌려본다
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)  # loss 의 평균값

  print('Test set : Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss,
      correct,
      len(test_loader.dataset),
      100.0 * correct / len(test_loader.dataset)
  ))
# >>> 출력:
# Train Epoch: 1 [0/60000 (0%)]	Loss: -0.610362
# Train Epoch: 1 [6400/60000 (11%)]	Loss: -0.618503
# Train Epoch: 1 [12800/60000 (21%)]	Loss: -0.516445
# Train Epoch: 1 [19200/60000 (32%)]	Loss: -0.496100
# Train Epoch: 1 [25600/60000 (43%)]	Loss: -0.556156
# Train Epoch: 1 [32000/60000 (53%)]	Loss: -0.611409
# Train Epoch: 1 [38400/60000 (64%)]	Loss: -0.673765
# Train Epoch: 1 [44800/60000 (75%)]	Loss: -0.664772
# Train Epoch: 1 [51200/60000 (85%)]	Loss: -0.626428
# Train Epoch: 1 [57600/60000 (96%)]	Loss: -0.701297
# Test set : Average Loss: -0.6075, Accuracy: 6242/10000 (62%)
# 
# Train Epoch: 2 [0/60000 (0%)]	Loss: -0.631033
# Train Epoch: 2 [6400/60000 (11%)]	Loss: -0.644511
# Train Epoch: 2 [12800/60000 (21%)]	Loss: -0.532661

# # 학습모델 저장하기

import os
save_path = os.path.join(save_dir, 'FashionMNIST.pth')
torch.save(model.state_dict(), save_path)

# # 저장한 모델 불러오기

model = None

model = Net().to(device)
model.load_state_dict(torch.load(save_path))
# >>> 출력:
# <All keys matched successfully>

# # 실제 데이터 적용 (예측하기)

base_path = r'/content/drive/MyDrive/DATA_SET/mnist_fashion'

import glob
img_paths = glob.glob(os.path.join(base_path, 'img_*.*'))
img_paths
# >>> 출력:
# ['/content/drive/MyDrive/DATA_SET/mnist_fashion/img_shirt.jpg',
#  '/content/drive/MyDrive/DATA_SET/mnist_fashion/img_ankle_boots.jpg',
#  '/content/drive/MyDrive/DATA_SET/mnist_fashion/img_coat.jpg',
#  '/content/drive/MyDrive/DATA_SET/mnist_fashion/img_sandal.png',
#  '/content/drive/MyDrive/DATA_SET/mnist_fashion/img_pants.png']

from PIL import Image
import PIL.ImageOps as ops

def predict(file_path):
  img = Image.open(file_path)
  mono8img = img.convert('L')
  invImg = ops.invert(mono8img)
  resizeImg = invImg.resize((28, 28))
  # torch 에서의 입력 shape 로 변환
  data_arr = np.array(resizeImg).reshape(1, 1, 28, 28)
  tensor = torch.Tensor(data_arr)  # Tensor 로 변환
  transformed = transforms.Normalize((0.1307,), (0.3081,))(tensor)

  output = model(transformed.to(device))
  pred = output.argmax(dim=1, keepdim=True)

  return pred.item()

datasets.FashionMNIST.classes
# >>> 출력:
# ['T-shirt/top',
#  'Trouser',
#  'Pullover',
#  'Dress',
#  'Coat',
#  'Sandal',
#  'Shirt',
#  'Sneaker',
#  'Bag',
#  'Ankle boot']

for img_path in img_paths:
  print(img_path, '->', datasets.FashionMNIST.classes[predict(img_path)])
# >>> 출력:
# /content/drive/MyDrive/DATA_SET/mnist_fashion/img_shirt.jpg -> T-shirt/top
# /content/drive/MyDrive/DATA_SET/mnist_fashion/img_ankle_boots.jpg -> Ankle boot
# /content/drive/MyDrive/DATA_SET/mnist_fashion/img_coat.jpg -> Coat
# /content/drive/MyDrive/DATA_SET/mnist_fashion/img_sandal.png -> Bag
# /content/drive/MyDrive/DATA_SET/mnist_fashion/img_pants.png -> Trouser

for img_path in img_paths:
  fig, ax = plt.subplots(1, 1, figsize=(4, 4))
  ax.axis('off')
  ax.set_title(datasets.FashionMNIST.classes[predict(img_path)])
  arr = plt.imread(img_path)
  plt.imshow(arr)
