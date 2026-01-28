# # Matplotlib
# https://matplotlib.org/ <br>
# 
# <img src="https://matplotlib.org/_static/logo_light.svg" width="300px"/>

# 파이썬의 자료(DataFrame, Series 등..)을  차트(chart) 나 플롯(plot) 으로
# 시각화(visualization) 하는 모듈

# Matplotlib는 다음과 같은 정형화된 차트나 플롯 이외에도
# 저수준 api를 사용한 다양한 시각화 기능을 제공한다.

# 라인 플롯(line plot)
# 스캐터 플롯(scatter plot)
# 컨투어 플롯(contour plot)
# 서피스 플롯(surface plot)
# 바 차트(bar chart)
# 히스토그램(histogram)
# 박스 플롯(box plot)

# ※ "맷플롯립" 라이브러리 로 읽힌다.

# ## matplotlib 시각화 예제 갤러리
# https://matplotlib.org/stable/gallery/index.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # Line plot
# 가장 간단한 플롯,  선을 그리는 라인 플롯<br>
# 데이터가 시간, 순서 등에 따라 어떻게 변화하는지 보여주기 위한 용도
# 
# http://Matplotlib.org/api/pyplot_api.html#Matplotlib.pyplot.plot

arr = np.array([10, 30, 14, 20])

plt.plot(arr)
# ↓ index 는 x 축으로, value 는 y 축으로 표현.
plt.show()

# 2차원 array 의 경우는?
arr = np.array([
    [10, 30, 14, 20],
    [20, 50, 33, 10],
    [30, 20, 5, 0]
])

plt.plot(arr)
plt.show()

# 두개의 array 객체로 각각 x, y
x = np.arange(0, 1, 0.01)
y = x ** 2

plt.plot(x, y)
plt.show()

# Series 로 line plot 그리기
s = pd.Series(np.random.randn(10).cumsum(), index = np.arange(0, 100, 10))
s
# >>> 출력:
# 0    -0.884078
# 10   -0.536506
# 20   -0.066269
# 30   -1.361944
# 40   -0.063105
# 50    1.956844
# 60    2.247172
# 70    2.442583
# 80    2.415467
# 90    3.385371
# dtype: float64

plt.plot(s)
plt.show()
# Series 는 index 가 x축, value 가 y축

# pandas 객체는 자체적으로 matplotlib 기반의 시각화 메소드 제공.
s.plot()
# >>> 출력:
# <Axes: >

df = pd.DataFrame(np.random.randn(10, 4).cumsum(axis=0),
                  columns=["A", "B", "C", "D"],
                  index=np.arange(0, 100, 10))
df
# >>> 출력:
#            A         B         C         D
# 0   1.605119 -0.467877  0.438291  0.033206
# 10  2.247478 -0.380529 -0.822106  1.863770
# 20  3.678304 -0.579278 -1.923295  0.720546
# 30  4.698307 -2.030703 -1.953137 -0.298433
# 40  3.157678 -2.950716 -0.619969 -1.018193
# 50  2.197038 -4.041473 -0.187288 -2.171526
# 60  1.588618 -2.409122 -0.741068 -2.897000
# 70  1.338457 -1.714647  0.052704 -2.429337
# 80  1.227469 -0.781119  0.624010 -3.240366
# 90  0.869277 -2.439334 -0.520305 -4.532890

df.plot()
# >>> 출력:
# <Axes: >

df.B.plot()
# >>> 출력:
# <Axes: >

df[['B', 'A']].plot()
# >>> 출력:
# <Axes: >

plt.plot(df[['B', 'A']])
plt.show()

# # Bar plot
# 막대 그래프

s2 = pd.Series(np.random.rand(16), index=list("abcdefghijklmnop"))
s2
# >>> 출력:
# a    0.207986
# b    0.481928
# c    0.254688
# d    0.056672
# e    0.175384
# f    0.144991
# g    0.632544
# h    0.819883
# i    0.445533
# j    0.219110
# k    0.592426
# l    0.593582
# m    0.670294
# n    0.788217
# o    0.343283

s2.plot()
# >>> 출력:
# <Axes: >

s2.plot(kind='bar')
# >>> 출력:
# <Axes: >

s2.plot(kind='barh')
plt.show()

df2 = pd.DataFrame(np.random.rand(6, 4),
                   index=["one", "two", "three", "four", "five", "six"],
                   columns=pd.Index(["A", "B", "C", "D"], name="Genus"))
df2
# >>> 출력:
# Genus         A         B         C         D
# one    0.524546  0.789696  0.930573  0.262097
# two    0.808312  0.792505  0.555478  0.818729
# three  0.422547  0.884922  0.658968  0.491201
# four   0.334027  0.821503  0.696076  0.075884
# five   0.399970  0.030357  0.795531  0.271647
# six    0.710739  0.992371  0.003696  0.465074

df2.plot(kind='bar')
plt.show()

df2.plot(kind='barh')
plt.show()

df2.plot(kind='barh', stacked=True)
plt.show()

# # Histogram
# 도수분포표의 하나, 가로축이 계급, 세로축이 도수

# histgram 에는 index 필요 없다.
s3 = pd.Series(np.random.normal(0, 1, size=200))
s3
# >>> 출력:
# 0      0.235560
# 1      1.835420
# 2     -2.493620
# 3     -1.084796
# 4     -0.831947
#          ...   
# 195   -1.225980
# 196   -0.153463
# 197    0.014770
# 198   -0.335851
# 199   -0.624434
# Length: 200, dtype: float64

s3.hist()
plt.show()

# 가로축이 value (구간)
# 세로축이 분포 (수량, 개수)

plt.hist(s3)
plt.show()

s3.hist(bins=50)  # bins= 구간
plt.show()

# # Scatter plot 그리기
# 산점도 : 산점도의 경우에는 서로 다른 두 개의 독립변수에 대해 두 변수가 어떤 관계가 있는지 살펴보기 위해 사용된다.

x1 = np.random.normal(1, 1, size=(100, 1))
x2 = np.random.normal(-2, 4, size=(100, 1))

X = np.concatenate((x1, x2), axis=1)
X
# >>> 출력:
# array([[ -0.93078772,  -1.4768403 ],
#        [  1.00444172,   2.88808679],
#        [  1.7749542 ,  -7.05561235],
#        [ -0.08034262,   3.02998948],
#        [  0.40552153,  -5.08513684],
#        [  0.30539587,  -2.02964027],
#        [  0.59601538,  -2.3762814 ],
#        [  2.03951732,  -2.56249457],
#        [  1.39829744,  -6.70422687],
#        [  1.18033919,  -5.33273007],
#        [  0.62347875, -11.03219435],
#        [  0.82407657,  -4.99426057],
#        [ -0.04082407,  -0.09809033],
#        [  1.6591446 ,   2.93549103],
#        [  1.28579709,   0.98429461],

df3 = pd.DataFrame(X, columns=["x1", "x2"])
df3
# >>> 출력:
#           x1        x2
# 0  -0.930788 -1.476840
# 1   1.004442  2.888087
# 2   1.774954 -7.055612
# 3  -0.080343  3.029989
# 4   0.405522 -5.085137
# ..       ...       ...
# 95  2.100005  0.559738
# 96  1.169434  3.141972
# 97  2.615198  2.605676
# 98  2.101768 -0.080153
# 99  1.127979 -1.419327
# 
# [100 rows x 2 columns]

plt.scatter(df3.x1, df3.x2)
# >>> 출력:
# <matplotlib.collections.PathCollection at 0x7df535204a50>

# # Matplotlib 에서의 한글문제

s4 = pd.Series(np.random.rand(10), index=list("가나다라마바사아자차"))
s4
# >>> 출력:
# 가    0.318772
# 나    0.361297
# 다    0.705907
# 라    0.431470
# 마    0.297947
# 바    0.730778
# 사    0.527012
# 아    0.838256
# 자    0.991154
# 차    0.014444
# dtype: float64

s4.plot()
plt.plot()
# >>> 출력:
# []
# /usr/local/lib/python3.11/dist-packages/IPython/core/events.py:89: UserWarning: Glyph 44032 (\N{HANGUL SYLLABLE GA}) missing from font(s) DejaVu Sans.
#   func(*args, **kwargs)
# /usr/local/lib/python3.11/dist-packages/IPython/core/events.py:89: UserWarning: Glyph 45796 (\N{HANGUL SYLLABLE DA}) missing from font(s) DejaVu Sans.
#   func(*args, **kwargs)
# /usr/local/lib/python3.11/dist-packages/IPython/core/events.py:89: UserWarning: Glyph 47560 (\N{HANGUL SYLLABLE MA}) missing from font(s) DejaVu Sans.
#   func(*args, **kwargs)
# /usr/local/lib/python3.11/dist-packages/IPython/core/events.py:89: UserWarning: Glyph 49324 (\N{HANGUL SYLLABLE SA}) missing from font(s) DejaVu Sans.
#   func(*args, **kwargs)
# /usr/local/lib/python3.11/dist-packages/IPython/core/events.py:89: UserWarning: Glyph 51088 (\N{HANGUL SYLLABLE JA}) missing from font(s) DejaVu Sans.
#   func(*args, **kwargs)
# /usr/local/lib/python3.11/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 44032 (\N{HANGUL SYLLABLE GA}) missing from font(s) DejaVu Sans.
#   fig.canvas.print_figure(bytes_io, **kw)
# /usr/local/lib/python3.11/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 45796 (\N{HANGUL SYLLABLE DA}) missing from font(s) DejaVu Sans.
#   fig.canvas.print_figure(bytes_io, **kw)
# /usr/local/lib/python3.11/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 47560 (\N{HANGUL SYLLABLE MA}) missing from font(s) DejaVu Sans.

# Colab 한글 글꼴 설치 fonts-nanum

!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
# >>> 출력:
# Reading package lists... Done
# Building dependency tree... Done
# Reading state information... Done
# fonts-nanum is already the newest version (20200506-1).
# 0 upgraded, 0 newly installed, 0 to remove and 23 not upgraded.
# /usr/share/fonts: caching, new cache contents: 0 fonts, 1 dirs
# /usr/share/fonts/truetype: caching, new cache contents: 0 fonts, 3 dirs
# /usr/share/fonts/truetype/humor-sans: caching, new cache contents: 1 fonts, 0 dirs
# /usr/share/fonts/truetype/liberation: caching, new cache contents: 16 fonts, 0 dirs
# /usr/share/fonts/truetype/nanum: caching, new cache contents: 12 fonts, 0 dirs
# /usr/local/share/fonts: caching, new cache contents: 0 fonts, 0 dirs
# /root/.local/share/fonts: skipping, no such directory
# /root/.fonts: skipping, no such directory
# /usr/share/fonts/truetype: skipping, looped directory detected
# /usr/share/fonts/truetype/humor-sans: skipping, looped directory detected

plt.rc('font', family='NanumBarunGothic')

# Colab 에선 '세션 다시시작 및 모두 실행' 을 해주어야 적용된다.

s4.plot()
plt.show()

# # figure, subplot
# - figure(그림)
# - subplot(그림 안의 공간)
# 
# ![a](https://matplotlib.org/stable/_images/anatomy.png)

fig = plt.figure()   # Figure 객체 셍성
fig
# >>> 출력:
# <Figure size 640x480 with 0 Axes>

# subplot(그림 공간) 추가
ax1 = fig.add_subplot(2, 2, 1)   # 2 x 2 그림공간 생성하고, 그중에서 1번째 (첫번째)
ax1
# AxesSubplot 객체  (이하 subplot.)
# >>> 출력:
# <Axes: >

ax4 = fig.add_subplot(2, 2, 4)

fig
# >>> 출력:
# <Figure size 640x480 with 2 Axes>

# 오늘날(?) add_subplot() 보다는
# plt.subplots(row, col) 추천

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.set_title('1st graph')
ax4.set_title('2nd graph')
fig
# >>> 출력:
# <Figure size 640x480 with 4 Axes>

ax1.hist(np.random.randn(100), bins=20)
ax2.plot(np.random.randn(50).cumsum())
ax3.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
fig
# >>> 출력:
# /usr/local/lib/python3.11/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   fig.canvas.print_figure(bytes_io, **kw)
# <Figure size 640x480 with 4 Axes>

# # plot 을 만들었던 방법들 (정리)

# 1. Series 나 DataFrame 에서 plot() 호출하여 직접 작성
s = pd.Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
s.plot()
# >>> 출력:
# <Axes: >
# /usr/local/lib/python3.11/dist-packages/IPython/core/events.py:89: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   func(*args, **kwargs)
# /usr/local/lib/python3.11/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   fig.canvas.print_figure(bytes_io, **kw)

# 2. pyplot 의 메소드 호출 ex: plot()
plt.plot(s)
# >>> 출력:
# [<matplotlib.lines.Line2D at 0x7df51447be50>]
# /usr/local/lib/python3.11/dist-packages/IPython/core/events.py:89: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   func(*args, **kwargs)
# /usr/local/lib/python3.11/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   fig.canvas.print_figure(bytes_io, **kw)

# 3. figure 생성후에 subplot 에 그리기
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(s)
# >>> 출력:
# [<matplotlib.lines.Line2D at 0x7df5143bfed0>]
# /usr/local/lib/python3.11/dist-packages/IPython/core/events.py:89: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   func(*args, **kwargs)
# /usr/local/lib/python3.11/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   fig.canvas.print_figure(bytes_io, **kw)

fig, ax1 = plt.subplots(1, 1)
ax1.plot(s)
# >>> 출력:
# [<matplotlib.lines.Line2D at 0x7df513d71f10>]
# /usr/local/lib/python3.11/dist-packages/IPython/core/events.py:89: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   func(*args, **kwargs)
# /usr/local/lib/python3.11/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   fig.canvas.print_figure(bytes_io, **kw)

# 1. plt module(matplotlib.pyplot) API
#       `plt.plot(...)`, `plt.title(...)`    <-- 거의 대부분 이 함수만 사용할거다.
#       장점: 간결. 간편
#       단점: 복잡한 그래프를 그리는데 한계 있다.

# 2. Figure, Subplot 을 이용해서 직접 그래프를 그리는 방법
#       그래프 는 여러 파트로 구성되어 있는데
#          전체 그림 --> Figure
#          Figure 안의 세부 그림 --> Subplot
#       코드는 좀더 복잡하지만..  (개인적으로 자주 애용)


# 3. Pandas ( pd.Series, pd.DataFrame )
#      Series 와 DataFrame 에선 Figure, Subplot 을 다루고 생성하는 함수들 제공
#
#     Series.________, DataFrame._____________
#     => Figure, Subplot
#
#     그러나, 결국 커스터 마이징을 하려면 2 번 방법을 알아야 함

# # plot 꾸미기

plt.plot(np.random.randn(50), color='g', marker='o', linestyle='--')
plt.show()
# >>> 출력:
# /usr/local/lib/python3.11/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   fig.canvas.print_figure(bytes_io, **kw)

# plot 꾸미기 옵션

# color
# 값 색상
# "b" blue
# "g" green
# "r" red
# "c" cyan
# "m" magenta
# "y" yellow
# "k" black
# "w" white

# marker
# 값 마킹
# "." point
# "," pixel
# "o" circle
# "v" triangle_down
# "^" triangle_up
# "<" triangle_left
# ">" triangle_right
# "8" octagon
# "s" square
# "p" pentagon
# "*" star
# "h" hexagon
# "+" plus
# "x" x
# "D" diamond

# line style
# 값 라인 스타일
# "-" solid line
# "--" dashed line
# "-." dash-dotted line
# ":" dotted line
# "None" draw nothing

plt.plot(np.random.randn(30), 'k.-')
# >>> 출력:
# [<matplotlib.lines.Line2D at 0x7df51381e3d0>]
# /usr/local/lib/python3.11/dist-packages/IPython/core/events.py:89: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   func(*args, **kwargs)
# /usr/local/lib/python3.11/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   fig.canvas.print_figure(bytes_io, **kw)

data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data
# >>> 출력:
# a    0.595662
# b    0.132186
# c    0.091117
# d    0.703321
# e    0.945734
# f    0.113483
# g    0.394101
# h    0.481682
# i    0.799435
# j    0.804642
# k    0.948992
# l    0.122461
# m    0.568687
# n    0.548842
# o    0.796404

data.plot(kind='bar', color='k', alpha=0.5)
# >>> 출력:
# <Axes: >

data.plot(kind='barh', color='g', alpha=0.3)
# >>> 출력:
# <Axes: >

fig, axes = plt.subplots(2, 1)

data.plot(kind='bar', color='k', alpha=0.7, ax=axes[0])  # ax=  <- 그림을 그릴 subplot 지정
data.plot(kind='barh', color='g', alpha=0.3, ax=axes[1])
# >>> 출력:
# <Axes: >

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

for i in range(3):
  ax.plot(np.random.randn(1000).cumsum(), label=f'{i} value')

ax.set_xticks([0, 250, 500, 750, 1000])
ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30)

ax.set_title('random walk plot')

ax.set_xlabel('Stages')
ax.set_ylabel('Value')

ax.grid(True)

ax.legend()  # 범례
# >>> 출력:
# <matplotlib.legend.Legend at 0x7df513a3a750>
# /usr/local/lib/python3.11/dist-packages/IPython/core/events.py:89: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   func(*args, **kwargs)
# /usr/local/lib/python3.11/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from font(s) NanumBarunGothic.
#   fig.canvas.print_figure(bytes_io, **kw)

# # figure -> 파일로 저장하기

import os

base_path = r'/content/drive/MyDrive/KoreaIT (코리아it)/250107 💚자연어처리/[AI자연어]/dataset(AI2501)'

if not os.path.exists(os.path.join(base_path, 'out')):
  os.makedirs(os.path.join(base_path, 'out'))

x = [1, 2, 3]
y = [10, 20, 30]

plt.plot(x, y, color='g', linestyle='-', marker='v')
plt.title('3rd Green Graph')

plt.savefig(os.path.join(base_path, 'out', 'figimage1.png'))
plt.savefig(os.path.join(base_path, 'out', 'figimage2.png'), dpi=200)  # 해상도
# 출판물은 300dpi 이상 선호

# 벡터 포맷 저장
#   확대, 축소 등의 상황에서도 이미지가 깨지지 않고 선명하게 보임.
plt.savefig(os.path.join(base_path, 'out', 'figimage3.svg')) # svg 벡터이미지 포맷
plt.savefig(os.path.join(base_path, 'out', 'figimage4.pdf'))  # pdf 도 벡터 포맷

# # 이미지 시각화

from IPython.display import Image

img_path = os.path.join(base_path, 'cat.jpg')
Image(img_path)
# >>> 출력:
# <IPython.core.display.Image object>

arr = plt.imread(img_path)  # 이미지 -> array
arr
# >>> 출력:
# array([[[  5,   3,   4],
#         [  5,   3,   4],
#         [  5,   3,   4],
#         ...,
#         [  2,   2,   2],
#         [  2,   2,   2],
#         [  2,   2,   2]],
# 
#        [[  5,   3,   4],
#         [  5,   3,   4],
#         [  5,   3,   4],
#         ...,
#         [  2,   2,   2],
#         [  2,   2,   2],
#         [  2,   2,   2]],

# 이미지 shape
# (height, width, color channel)

# color channel 은 (r, g, b) 값  각각 0 ~ 255

arr.shape
# >>> 출력:
# (1200, 1600, 3)

# 첫번째 row
arr[0]
# >>> 출력:
# array([[5, 3, 4],
#        [5, 3, 4],
#        [5, 3, 4],
#        ...,
#        [2, 2, 2],
#        [2, 2, 2],
#        [2, 2, 2]], dtype=uint8)

arr[0, 5]  # row:0, col: 5 의 pixel rgb
# >>> 출력:
# array([5, 3, 4], dtype=uint8)

arr[0, 5, 1]  # row:0, col: 5 의 green 값
# >>> 출력:
# 3

plt.imshow(arr)
# >>> 출력:
# <matplotlib.image.AxesImage at 0x7df513618850>

# # 이미지 slicing

plt.imshow(arr[170:800])
# >>> 출력:
# <matplotlib.image.AxesImage at 0x7df511e94990>

plt.imshow(arr[170:800, 230:850])
# >>> 출력:
# <matplotlib.image.AxesImage at 0x7df5116cc550>

# # 이미지 flip

plt.imshow(arr)
# >>> 출력:
# <matplotlib.image.AxesImage at 0x7df5114b38d0>

plt.imshow(arr[:, ::-1, :])
# >>> 출력:
# <matplotlib.image.AxesImage at 0x7df512faff10>

plt.imshow(arr[::-1, ::-1, :])
# >>> 출력:
# <matplotlib.image.AxesImage at 0x7df513622fd0>

# # gray scale 변경

arr.shape
# >>> 출력:
# (1200, 1600, 3)

# red 만 추출
r = arr[:, :, 0]  # 2차원 데이터
r.shape
plt.imshow(r)
# >>> 출력:
# <matplotlib.image.AxesImage at 0x7df512398710>

#     # image_arr.shape == (1200, 1600, 3)
#     r = image_arr[:, :, 0]    # r.shape = (1200, 1600)
#     g = image_arr[:, :, 1]
#     b = image_arr[:, :, 2]

# r, g, b  '3개의 색값'을 사용하여 '한개의 색' 로 변경하는 공식 예
#         r * 0.299 + g * 0.587 + b * 0.114   (green 값 강조)


# 즉
# (1200 x 1600 x 3) ...→ ... ( 1200 x 1600 ) 으로 변화시키면 된다.

# 이는 다음과 같은 행렬 곱을 하면 된다.
# (1200 x 1600 x 3) x ( 3 x 1 ) => 1200 x 1600

def color_to_grayscale(image_arr):
  return np.dot(
      image_arr,  # (h x w x 3)
      np.array([0.299, 0.587, 0.114])   # (3 x 1)
  )

gray = color_to_grayscale(arr)
gray.shape
# >>> 출력:
# (1200, 1600)

gray[0, 0]
# >>> 출력:
# 3.7119999999999997

plt.imshow(gray)
# >>> 출력:
# <matplotlib.image.AxesImage at 0x7df512436bd0>

plt.imshow(gray, cmap='gray')
# >>> 출력:
# <matplotlib.image.AxesImage at 0x7df511822e90>

plt.imshow(gray, cmap='hot')
# >>> 출력:
# <matplotlib.image.AxesImage at 0x7df511f40710>

plt.imshow(gray, cmap='nipy_spectral')
# >>> 출력:
# <matplotlib.image.AxesImage at 0x7df511551050>

plt.imshow(gray)
plt.colorbar()
# >>> 출력:
# <matplotlib.colorbar.Colorbar at 0x7df511553c90>

# # Interpolation

from PIL import Image

img = Image.open(img_path)

type(img)
# >>> 출력:
# PIL.JpegImagePlugin.JpegImageFile

img = Image.open(img_path)
img.thumbnail((64, 64))
imgplot = plt.imshow(img)

img = Image.open(img_path)
img.thumbnail((64, 64))
imgplot = plt.imshow(img, interpolation='nearest')

img = Image.open(img_path)
img.thumbnail((64, 64))
imgplot = plt.imshow(img, interpolation='bicubic')

# # Seaborn 패키지
# ![](https://seaborn.pydata.org/_static/logo-wide-lightbg.svg)
# 
# Seaborn은 matplotlib 패키지를 기반으로 하여 보다 편하게 통계를 시각화할 수 있는 도구입니다. 일반적으로 데이터 사이언스에서 사용하는 대부분의 그래프를 지원합니다.
# 
# 공식: https://seaborn.pydata.org/
# 
# 갤러리: https://seaborn.pydata.org/examples/index.html
