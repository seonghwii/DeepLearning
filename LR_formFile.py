import torch
import torch.nn as nn #nn.Linear 라이브러리를 사용하기 위해 import

# F.mse(mean squared error) <--linear regression, 다양한 LOSS Function 존재
# Classification problem에서 사용하는 loss function: Cross-Entropy (집합체 : torch.nn.functional에 존재한다.)

import torch.nn.functional as F
import torch.optim as optim #SGD, Adam, etc. 최적화 라이브러리
from file_read import read_file
import numpy as np

#test = 5개, train = 23개

house_info, cnt = read_file("sell_house.txt")

A = np.array(house_info)

x_train = A[:-5, 1:-1]
x_train = x_train.reshape(23, 11)
y_train = A[:-5, -1:] #결과치

x_test = A[-5:, 1:-1]
x_test = x_test.reshape(5, 11)
y_test = A[-5:, -1:] #결과치












