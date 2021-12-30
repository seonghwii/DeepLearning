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

A = torch.FloatTensor(house_info)

x_train = A[:-5, 1:-1]
y_train = A[:-5, -1:] #결과치

x_test = A[-5:, 1:-1]
y_test = A[-5:, -1:] #결과치

model = nn.Linear(11, 1)

optimizer = optim.SGD(model.parameters(), lr=0.000002)

nb_epochs = 7500
for epoch in range(nb_epochs+1):
    pred = model(x_train)
    cost = F.mse_loss(pred, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch: {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')

print(list(model.parameters()))

new_var = torch.FloatTensor(x_test)
pred_y = model(new_var)
f = open("result.txt", "w")
# f.write(f'{str(pred_y)}\n')
f.close()







