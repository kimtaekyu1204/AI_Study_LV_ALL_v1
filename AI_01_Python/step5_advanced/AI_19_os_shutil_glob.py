# # os 모듈: 디렉토리, 파일 관리

import os

savedir = "tmpdir1"

if not os.path.exists(savedir):
  os.mkdir(savedir)
  print(savedir, '폴더 생성')

else:
  print(savedir, '이미 존재')
# >>> 출력:
# tmpdir1 이미 존재

savedir = r"tmpdir2/aaa/bbb/ccc"   # 경로는 raw-string 사용!

if not os.path.exists(savedir):
  os.makedirs(savedir)
  print(savedir, '폴더 생성')

else:
  print(savedir, '이미 존재')
# >>> 출력:
# tmpdir2/aaa/bbb/ccc 폴더 생성

os.listdir('./sample_data')
# >>> 출력:
# ['anscombe.json',
#  'README.md',
#  'mnist_train_small.csv',
#  'mnist_test.csv',
#  'california_housing_train.csv',
#  'california_housing_test.csv']

os.getcwd()  # 현재 경로 (current working directory)
# >>> 출력:
# '/content'

# 경로 문자열 생성
os.path.join(os.getcwd(), 'aaa', 'bbb/ccc')
# >>> 출력:
# '/content/aaa/bbb/ccc'

# # glob 모듈

import glob

glob.glob('./sample_data/*')
# >>> 출력:
# ['./sample_data/anscombe.json',
#  './sample_data/README.md',
#  './sample_data/mnist_train_small.csv',
#  './sample_data/mnist_test.csv',
#  './sample_data/california_housing_train.csv',
#  './sample_data/california_housing_test.csv']

glob.glob('./sample_data/*.csv')
# >>> 출력:
# ['./sample_data/mnist_train_small.csv',
#  './sample_data/mnist_test.csv',
#  './sample_data/california_housing_train.csv',
#  './sample_data/california_housing_test.csv']

# # shutil 모듈
# shell util
# 
# 파일및 디렉토리 작업

import shutil

shutil.copy(r'./sample_data/README.md', 'My.md')
# >>> 출력:
# 'My.md'

shutil.copytree('tmpdir2', 'tmpdir3')
# >>> 출력:
# 'tmpdir3'

shutil.rmtree('tmpdir2')
